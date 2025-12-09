#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sample attention score distributions (q·k / sqrt(d)) from Qwen2.5-7B-Instruct
for a single layer and head, over multiple query positions in a long context.

For each sampled position `pos`, we compute scores between q(pos) and all keys
k(0..pos), sort them in descending order, and save them to a JSON file:

{
  "model_name": "...",
  "layer_idx": 14,
  "head_idx": 0,
  "seq_len": 12345,
  "positions": [
    {
      "pos": 1024,
      "num_keys": 1025,
      "scores_sorted": [ ... floats, descending ... ]
    },
    ...
  ]
}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--book-path",
        type=str,
        default="../example/.data/war-and-peace-16k.txt",
        help="Path to plain text book file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="attn_samples.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128000,
        help="Max token length to keep from the book",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=14,
        help="Layer index to hook",
    )
    parser.add_argument(
        "--head-idx",
        type=int,
        default=14,
        help="Head index to hook",
    )
    parser.add_argument(
        "--num-positions",
        type=int,
        default=64,
        help="Number of query positions to sample",
    )
    parser.add_argument(
        "--min-pos",
        type=int,
        default=64,
        help="Minimum position index to sample (avoid very early tokens)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for position sampling",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap book into a chat template prompt (system+user) for instruct models",
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    rope_scaling = {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        rope_scaling=rope_scaling,
        attn_implementation="flash_attention_2",
    )

    model.eval()
    return model, tokenizer


def prepare_inputs(tokenizer, book_path: str, max_tokens: int, use_chat_template: bool, device: torch.device):
    with open(book_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if use_chat_template:
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Summarize the following book: " + content,
            },
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            [text],
            return_tensors="pt",
            max_length=max_tokens,
            truncation=True,
        )
    else:
        encoded = tokenizer(
            [content],
            return_tensors="pt",
            max_length=max_tokens,
            truncation=True,
            add_special_tokens=False,
        )

    encoded = {k: v.to(device) for k, v in encoded.items()}
    return encoded


def sample_positions(seq_len: int, num_positions: int, min_pos: int, seed: int) -> List[int]:
    import random

    random.seed(seed)
    low = max(min_pos, 1)
    high = seq_len - 1
    if high <= low:
        raise ValueError(f"Sequence too short: seq_len={seq_len}, min_pos={min_pos}")

    num_positions = min(num_positions, high - low + 1)
    positions = random.sample(range(low, high + 1), num_positions)
    positions.sort()
    return positions


def make_qk_hook(store: Dict[str, torch.Tensor], head_idx: int):
    """
    Build a forward hook for Qwen2Attention that extracts q and k after RoPE and KV repetition
    for a specific attention head.

    The Qwen2Attention forward is called with keyword arguments including:
      - hidden_states: (batch, seq_len, hidden_size)
      - position_embeddings: (batch, seq_len, head_dim) or similar

    In this hook, we:
      1) project hidden_states to q and k
      2) apply RoPE
      3) repeat_kv to get per-head keys
      4) store q_rope[:, head_idx, :, :] and k_rope_full[:, head_idx, :, :]
    """

    def hook(module, args, kwargs, output):
        hidden_states = kwargs["hidden_states"]        # (batch, seq_len, hidden_size)
        cos, sin = kwargs["position_embeddings"]  # (batch, seq_len, head_dim) via rotary_emb

        batch_size, seq_len, hidden_size = hidden_states.shape
        head_dim = module.head_dim
        num_heads = module.config.num_attention_heads
        num_kv_heads = module.config.num_key_value_heads

        # Project to q and k
        q = module.q_proj(hidden_states)  # (batch, seq_len, hidden_size)
        k = module.k_proj(hidden_states)  # (batch, seq_len, hidden_size_kv)

        # Reshape to (batch, num_heads, seq_len, head_dim) / (batch, num_kv_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE using the provided position_embeddings via rotary_emb
        # Qwen2 uses rotary_emb(hidden_states, position_embeddings) under the hood;
        # here we mimic that behavior via apply_rotary_pos_emb.
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat k over groups to match num_heads (GQA)
        k_rope_full = repeat_kv(k_rope, module.num_key_value_groups)  # (batch, num_heads, seq_len, head_dim)

        # Store only the requested head, and move to CPU to free GPU memory
        store["q"] = q_rope[:, head_idx, :, :].detach().cpu()  # (batch, seq_len, head_dim)
        store["k"] = k_rope_full[:, head_idx, :, :].detach().cpu()  # (batch, seq_len, head_dim)

    return hook


def compute_scores_for_positions_from_qk(
    q: torch.Tensor,  # (1, seq_len, head_dim)
    k: torch.Tensor,  # (1, seq_len, head_dim)
    positions: List[int],
) -> Dict[int, List[float]]:
    """
    Given q and k for a single head (batch=1), compute q·k/sqrt(d) scores for
    query positions in `positions`, using only keys 0..pos (causal).

    Returns:
        dict[pos] = [scores_sorted_desc]
    """
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("Expected q and k to have shape (1, seq_len, head_dim).")

    bsz, seq_len_q, head_dim = q.shape
    bsz_k, seq_len_k, head_dim_k = k.shape

    if bsz != 1 or bsz_k != 1:
        raise ValueError("This script assumes batch_size = 1.")
    if seq_len_q != seq_len_k:
        raise ValueError("q and k must have the same seq_len.")
    if head_dim != head_dim_k:
        raise ValueError("q and k must have the same head_dim.")

    scale = head_dim ** -0.5
    scores_dict: Dict[int, List[float]] = {}

    # Move to a single device (CPU) for simplicity
    q = q[0]  # (seq_len, head_dim)
    k = k[0]  # (seq_len, head_dim)

    for pos in positions:
        if pos >= seq_len_q:
            raise ValueError(f"pos={pos} out of range for seq_len={seq_len_q}")

        q_vec = q[pos]           # (head_dim,)
        k_mat = k[: pos + 1, :]  # (pos+1, head_dim)

        scores = (k_mat @ q_vec) * scale  # (pos+1,)

        sorted_scores, _ = torch.sort(scores, descending=True)
        scores_dict[int(pos)] = sorted_scores.to(torch.float32).tolist()

    return scores_dict


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    print("Loaded model and tokenizer.")

    encoded = prepare_inputs(
        tokenizer=tokenizer,
        book_path=args.book_path,
        max_tokens=args.max_tokens,
        use_chat_template=args.use_chat_template,
        device=model.device,
    )

    input_ids = encoded["input_ids"]
    seq_len = int(input_ids.shape[1])
    print(f"Input tokens: {input_ids.shape}")

    # Choose layer/head
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    layer_idx = args.layer_idx
    head_idx = args.head_idx

    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"layer_idx={layer_idx} out of range [0, {num_layers - 1}]")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx={head_idx} out of range [0, {num_heads - 1}]")

    print(f"Hooking layer_idx = {layer_idx} / {num_layers}, head_idx = {head_idx} / {num_heads}")

    # Hook to capture q and k for this layer/head
    store: Dict[str, torch.Tensor] = {}
    attn_module = model.model.layers[layer_idx].self_attn
    handle = attn_module.register_forward_hook(
        make_qk_hook(store, head_idx),
        with_kwargs=True,
    )

    # Run forward to trigger the hook
    print("Running forward...")
    with torch.no_grad():
        _ = model(**encoded)

    handle.remove()

    if "q" not in store or "k" not in store:
        raise RuntimeError("Hook did not capture q and k; check hook wiring.")

    q = store["q"]  # (1, seq_len, head_dim) on CPU
    k = store["k"]  # (1, seq_len, head_dim) on CPU
    print(f"Captured q shape: {q.shape}, k shape: {k.shape}")

    # Sample positions
    positions = sample_positions(
        seq_len=seq_len,
        num_positions=args.num_positions,
        min_pos=args.min_pos,
        seed=args.seed,
    )
    print(f"Sampled {len(positions)} positions (sorted), first few: {positions[:10]}{' ...' if len(positions) > 10 else ''}")

    # Compute sorted q·k scores for each sampled position
    scores_dict = compute_scores_for_positions_from_qk(q, k, positions)

    # Build JSON output
    output = {
        "model_name": args.model_name,
        "book_path": str(Path(args.book_path).resolve()),
        "seq_len": seq_len,
        "layer_idx": layer_idx,
        "head_idx": head_idx,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "positions": [
            {
                "pos": pos,
                "num_keys": pos + 1,
                "scores_sorted": scores_dict[pos],
            }
            for pos in positions
        ],
    }

    out_path = Path(args.output_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"Saved attention score samples to {out_path}")


if __name__ == "__main__":
    main()