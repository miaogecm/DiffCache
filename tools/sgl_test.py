#!/usr/bin/env python
import argparse
import torch
from transformers import AutoTokenizer
from sglang import Runtime, set_default_runtime, function, gen


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5 with sglang Runtime on a long prompt from input_ids.pt (offline, greedy decoding)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model name or local path used by sglang Runtime.",
    )
    parser.add_argument(
        "--input_ids_path",
        type=str,
        default="../.data/input_ids.pt",
        help="Path to input_ids.pt (tensor of token ids).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--output_text_path",
        type=str,
        default="../.data/sglang_offline_output.txt",
        help="Path to save the full output text (prompt + completion).",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size for sglang Runtime.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load input_ids
    print(f"Loading input_ids from {args.input_ids_path} ...")
    input_ids = torch.load(args.input_ids_path)
    if not torch.is_tensor(input_ids):
        raise ValueError("Loaded input_ids is not a tensor.")

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    elif input_ids.dim() != 2:
        raise ValueError(f"input_ids must have shape [batch, seq_len] or [seq_len], got {input_ids.shape}.")

    if input_ids.size(0) != 1:
        raise ValueError(f"This script expects batch size 1, got batch={input_ids.size(0)}.")

    print(f"input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
    token_ids = input_ids[0].tolist()

    # 2) Decode to text using HF tokenizer (must match the model)
    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Keep special tokens so that encode(decode(ids)) is as faithful as possible
    prompt_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    # 3) Create sglang Runtime (offline, in-process)
    print(f"Initializing sglang Runtime with model {args.model_path} ...")
    runtime = Runtime(
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        trust_remote_code=True,
        tp_size=args.tp_size,
    )
    set_default_runtime(runtime)

    eos_token_id = tokenizer.eos_token_id

    # 4) Define a sglang function that does greedy decoding (temperature=0, top_p=1)
    @function
    def long_prompt_chat(s, prompt: str, max_tokens: int):
        s += prompt
        s += gen(
            "answer",
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
        )

    print("Running sglang long_prompt_chat ...")
    out = long_prompt_chat.run(
        prompt=prompt_text,
        max_tokens=args.max_new_tokens,
    )

    completion = out["answer"]
    full_text = prompt_text + completion

    print("\n===== SGLANG OFFLINE OUTPUT (prompt + completion) =====\n")
    print(full_text)
    print("\n=======================================================\n")

    with open(args.output_text_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved full output to {args.output_text_path}")


if __name__ == "__main__":
    main()
