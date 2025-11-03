import os
import sys
import json
import math
import torch
import argparse
import random
import numpy as np
from termcolor import colored
from transformers import AutoTokenizer
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
from models import LlamaModel, QwenModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gen_len", type=int, default=100, help="Generation length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",                  \
                        choices=["gradientai/Llama-3-8B-Instruct-Gradient-1048k", "Qwen/Qwen2.5-7B-Instruct",               \
                        "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"], help="huggingface model name")
    parser.add_argument("--data_path", type=str, default="", help="Input json file path")
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


def generate_config(model_name, context_len):
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1]+'.json'
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE, "r") as f:
        original_config = json.load(f)
    return original_config


if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)

    model_name = args.model_name
    batch_size = args.batch_size
    dtype = torch.float16 if args.dtype=='fp16' else torch.bfloat16
    device = args.device
    data_path = args.data_path

    # load input data
    if data_path == "":
        TEST_FILE = os.path.join(PROJECT_ROOT, "simple_test_data.json")
    else:
        TEST_FILE = os.path.join(PROJECT_ROOT, f"{data_path}")
    print(colored(f"Loading test data from {TEST_FILE}", 'yellow'))
    data = json.load(open(TEST_FILE))   # [{"input": str, "outputs": str}, ...]
    prompt = []
    groundtruth = []
    for dd in data:
        prompt.append(dd['input'])
        groundtruth.append(dd['outputs'])
    
    # copy to fit batch size
    copy_round = math.ceil(batch_size/len(prompt))
    prompts = []
    groundtruths = []
    for i in range(copy_round):
        prompts.extend(prompt)
        groundtruths.extend(groundtruth)
    prompts = prompts[:batch_size]
    groundtruths = groundtruths[:batch_size]

    # tokenize input data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    input_len = input_ids.shape[1]
    gen_len = args.gen_len
    max_len = input_len + gen_len
    print(colored(f"Input length: {input_len}", 'yellow'))

    if data_path == "":
        attn_config = generate_config(model_name, 122880)
    else:
        attn_config = generate_config(model_name, input_len)

    llm = load_model(model_name, max_len, dtype, device)
    out = llm.generate(
        inputs_ids=input_ids.to(llm.layers[0].device),
        attention_masks=attention_masks.to(llm.layers[0].device),
        max_new_length=gen_len, attn_config=attn_config
    )
    
    result = tokenizer.batch_decode(out, skip_special_tokens=True)
    for gt, res in zip(groundtruths, result):
        print(colored(f"Answer: {gt}", 'yellow'))
        print(f"{[res]}")
