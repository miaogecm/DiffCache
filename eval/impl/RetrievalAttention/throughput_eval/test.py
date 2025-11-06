import os
import sys
import json
import torch
import argparse
import random
import numpy as np
from termcolor import colored
from transformers import AutoTokenizer
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--context_len", type=int, default=-1, help="Input context length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",                                                 \
                        choices=["Full_Flash_Attn", "RetroInfer"], help="Attention method")
    parser.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",                  \
                        choices=["gradientai/Llama-3-8B-Instruct-Gradient-1048k", "Qwen/Qwen2.5-7B-Instruct",               \
                        "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"], help="huggingface model name")
    parser.add_argument("--task_name", type=str, default="multivalue", choices=["NIAH", "fwe", "vt", "qa1"],                \
                        help="Test task name")
    args = parser.parse_args()
    
    return args


def load_model(model_name, max_len, dtype, device):
    if 'Llama' in model_name:
        llm = LlamaModel(model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    elif 'Qwen' in model_name:
        llm = QwenModel(model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return llm


def generate_config(model_name, context_len, attn_type):
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1]+'.json'
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE, "r") as f:
        original_config = json.load(f)
    
    n_clusters = max(int(context_len/16), 1)
    n_segments = max(int(context_len/8000), 1)
    # compute the nearest multiple of (n_segments*32)
    lower = (n_clusters // (n_segments*32)) * (n_segments*32)
    upper = lower + (n_segments*32)
    n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
    nprobe = int(n_clusters*0.018)

    if attn_type == 'RetroInfer':
        original_config[attn_type]['n_centroids'] = n_clusters
        original_config[attn_type]['n_segment'] = n_segments
        original_config[attn_type]['nprobe'] = nprobe
        original_config[attn_type]['cache_cluster_num'] = nprobe*3
        original_config[attn_type]['max_compute_cluster_num'] = int(n_clusters/4)
    
    if attn_type != "Full_Flash_Attn":
        print(original_config[attn_type])
    
    return original_config


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_seed(2025)

    model_name = args.model_name
    batch_size = args.batch_size
    attn_type = args.attn_type
    dtype = torch.float16 if args.dtype=='fp16' else torch.bfloat16
    device = args.device
    task_name = args.task_name

    if task_name == "NIAH":
        TEST_DIR = os.path.join(PROJECT_ROOT, "throughput_eval")
        TEST_FILE = os.path.join(TEST_DIR, f"test_data/NIAH_{args.context_len}.json")
        data = json.load(open(TEST_FILE))[0]
        prompt = data['input']
        groundtruth = data['answer']
        attn_config = generate_config(model_name, args.context_len, attn_type)
    else:
        TEST_DIR = os.path.join(PROJECT_ROOT, "throughput_eval")
        TEST_FILE = os.path.join(TEST_DIR, f"test_data/{task_name}.json")
        data = json.load(open(TEST_FILE))
        prompt = data['input']
        groundtruth = data['outputs']
        attn_config = generate_config(model_name, 120000, attn_type)
    
    prompts = [prompt for _ in range(batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    input_len = input_ids.shape[1]
    gen_len = 100
    max_len = input_len + gen_len
    print(colored(f"Input length: {input_len}", 'yellow'))

    llm = load_model(model_name, max_len, dtype, device)
    out = llm.generate(attention_type=attn_type,
        inputs_ids = input_ids.to(llm.layers[0].device),
        attention_masks = attention_masks.to(llm.layers[0].device),
        max_new_length=gen_len, attn_config=attn_config)
    
    result = tokenizer.batch_decode(out, skip_special_tokens=True)
    print(groundtruth)
    print(result)
    