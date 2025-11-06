#!/usr/bin/env python3
import sys
import torch
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Read a book from stdin and output truncated text to stdout.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=128000)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Read full text from stdin
    text = sys.stdin.read()

    # Tokenize with truncation to model's max length
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
    truncated_ids = inputs["input_ids"][0]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    # Output to stdout
    sys.stdout.write(truncated_text)

if __name__ == "__main__":
    main()
