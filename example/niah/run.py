#!/usr/bin/env python3

import os
import argparse
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffcache import LlamaModel, QwenModel
from codetiming import Timer
import atexit


def insert_needle(text: str, needle: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    pos = int(len(text) * ratio)
    return text[:pos] + "\n" + needle + "\n" + text[pos:]


def load_model(model_name, max_len, dtype):
    if 'Llama' in model_name:
        llm = LlamaModel(model_name,
            max_length=max_len,
            dtype=dtype)
    elif 'Qwen' in model_name:
        llm = QwenModel(model_name,
            max_length=max_len,
            dtype=dtype)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--book_path", type=str, default="../.data/war-and-peace-128k.txt")
    parser.add_argument("--needle_ratio", type=float, default=0.5, help="Insertion ratio in [0,1].")
    parser.add_argument("--max_gen_len", type=int, default=32)
    args = parser.parse_args()

    # Load and inject needle
    with open(args.book_path, "r", encoding="utf-8") as f:
        book_text = f.read()
    needle_text = "\n\nNEEDLE: The secret verification code is JUPITER-19.\n\n"
    book_with_needle = insert_needle(book_text, needle_text, args.needle_ratio)

    question_text = (
        "\n\nIn the document above there is a line that starts with \"NEEDLE:\".\n"
        "Find that line and output it exactly.\n\nAnswer:"
    )

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    messages = [
        {
            "role": "system",
            "content": "You are a careful assistant. When asked, you MUST copy exactly "
                       "the line that starts with 'NEEDLE:' from the given document.",
        },
        { "role": "user", "content": book_with_needle + question_text }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask
    input_len = input_ids.shape[1]
    max_len = input_len + args.max_gen_len
    print(f"Input length: {input_len}")
    
    # Load model
    llm = load_model(args.model_name, max_len, dtype=torch.bfloat16)
    out = llm.generate(
        inputs_ids=input_ids.cuda(),
        attention_masks=attention_masks.cuda(),
        max_new_length=args.max_gen_len
    )

    result = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    print("=== Generation Result ===")
    print(result)


def print_stats():
    print("Timers summary:")
    for name in Timer.timers.data.keys():
        total = Timer.timers.total(name)
        count = Timer.timers.count(name)
        mean = Timer.timers.mean(name)
        print(f"  {name}: total {total:.4f}s over {count} runs (mean {mean:.4f}s)")


if __name__ == "__main__":
    atexit.register(print_stats)
    main()
