#!/usr/bin/env python
import argparse
import torch
from transformers import AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Dump prefill logits for given input_ids.pt")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--input_ids_path",
        type=str,
        default="../.data/input_ids.pt",
        help="Path to input_ids.pt (tensor of token ids)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../.data/logits.pt",
        help="Path to save logits tensor",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = get_dtype(args.dtype)

    print(f"Loading input_ids from {args.input_ids_path} ...")
    input_ids = torch.load(args.input_ids_path)
    if not torch.is_tensor(input_ids):
        raise ValueError("Loaded input_ids is not a tensor")

    # Ensure shape [batch, seq_len]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    elif input_ids.dim() != 2:
        raise ValueError(f"input_ids must have shape [batch, seq_len] or [seq_len], got {input_ids.shape}")

    print(f"input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")

    print(f"Loading model {args.model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
        # If you already baked rope_scaling into your checkpoint, you can remove this
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    )
    model.eval()

    # Move input_ids to the first device used by the model
    # If device_map="auto", lm_head is usually on the same device as the last block.
    first_param = next(model.parameters())
    input_ids = input_ids.to(first_param.device)

    with torch.no_grad():
        print("Running forward pass (prefill) ...")
        outputs = model(
            input_ids=input_ids,
            use_cache=False,  # pure prefill, no kv cache needed
        )
        logits = outputs.logits[:, -1, :].contiguous().clone()   # [batch, vocab_size]

    print(f"logits shape: {logits.shape}, dtype: {logits.dtype}")
    print(f"Saving logits to {args.output_path} ...")
    torch.save(logits, args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
