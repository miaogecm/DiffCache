#!/usr/bin/env python
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Greedy prefill+decode with HF model using input_ids.pt")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model name or local checkpoint path",
    )
    parser.add_argument(
        "--input_ids_path",
        type=str,
        default="../.data/input_ids.pt",
        help="Path to input_ids.pt (tensor of token ids)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to decode",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--output_text_path",
        type=str,
        default="../.data/output_text.txt",
        help="Path to save the decoded text",
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

    # Make everything deterministic on the Python side
    torch.manual_seed(0)

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

    if input_ids.size(0) != 1:
        raise ValueError(f"This script expects batch size 1, got batch={input_ids.size(0)}")

    print(f"input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")

    print(f"Loading tokenizer and model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    )
    model.eval()

    # Move input_ids to the first parameter device
    first_param = next(model.parameters())
    device_model = first_param.device
    input_ids = input_ids.to(device_model)

    eos_token_id = model.config.eos_token_id
    if eos_token_id is None:
        # Fallback: try tokenizer
        eos_token_id = tokenizer.eos_token_id

    print("Starting greedy prefill + decode ...")
    with torch.no_grad():
        # Prefill + first forward
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        past_key_values = outputs.past_key_values

        # Greedy: always take argmax
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
        generated = torch.cat([next_token], dim=-1)

        # Decode up to max_new_tokens
        for step in range(1, args.max_new_tokens):
            outputs = model(
                input_ids=generated[:, -1:],  # only last token
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if eos token is generated
            if eos_token_id is not None and (next_token == eos_token_id).all():
                print(f"EOS token generated at step {step}. Stopping decoding.")
                break

    # Decode full sequence including the prompt
    generated_ids = generated[0].cpu().tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print("\n===== DECODED TEXT =====\n")
    print(text)
    print("\n========================\n")

    # Save to file
    with open(args.output_text_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved decoded text to {args.output_text_path}")


if __name__ == "__main__":
    main()
