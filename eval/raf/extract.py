#!/usr/bin/env python3

# Extract Q and K vectors from NIAH workload

import os
import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Qwen2 helpers (apply_rotary_pos_emb, repeat_kv)
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


def insert_needle(text: str, needle: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    pos = int(len(text) * ratio)
    return text[:pos] + "\n" + needle + "\n" + text[pos:]


def build_messages(system_prompt: str, question_prefix: str, book_text_with_needle: str, answer_text: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_prefix + book_text_with_needle},
        {"role": "assistant", "content": answer_text}
    ]
    return messages


def lengths_for_spans(tokenizer, messages_full, messages_sys_only, messages_sys_user_only):
    # Tokenize three versions to compute exact token boundaries
    txt_sys = tokenizer.apply_chat_template(messages_sys_only, tokenize=False, add_generation_prompt=False)
    txt_sys_user = tokenizer.apply_chat_template(messages_sys_user_only, tokenize=False, add_generation_prompt=False)
    txt_full = tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)

    ids_sys = tokenizer(txt_sys, return_tensors="pt").input_ids[0]
    ids_sys_user = tokenizer(txt_sys_user, return_tensors="pt").input_ids[0]
    ids_full = tokenizer(txt_full, return_tensors="pt").input_ids[0]

    q_start = len(ids_sys)
    q_end = len(ids_sys_user)                 # question tokens are [q_start, q_end)
    a_start = len(ids_sys_user)
    a_end = len(ids_full)                     # answer tokens are [a_start, a_end)

    return txt_full, (q_start, q_end, a_start, a_end)


def make_qk_hook(store, head_idx):
    def hook(module, args, kwargs, output):
        hidden_states = kwargs['hidden_states']            # (batch, seq_len, hidden_size)
        cos, sin = kwargs['position_embeddings']           # (batch, seq_len, head_dim)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)

        q = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (batch, n_heads,    seq_len, head_dim)
        k = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (batch, n_kv_heads, seq_len, head_dim)

        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        k_rope_full = repeat_kv(k_rope, module.num_key_value_groups)         # (batch, n_heads,    seq_len, head_dim)

        store['q'] = q_rope[:, head_idx, :, :].detach().cpu()       # (batch, seq_len, head_dim)
        store['k'] = k_rope_full[:, head_idx, :, :].detach().cpu()  # (batch, seq_len, head_dim)

    return hook


def main():
    parser = argparse.ArgumentParser(description="Prefill teacher model, extract RoPE-augmented Q/K at a given layer/head, and dump question-K and answer-Q.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--book_path", type=str, default="book.txt")
    parser.add_argument("--needle_ratio", type=float, default=0.5, help="Insertion ratio in [0,1].")
    parser.add_argument("--needle_text", type=str, default="[[[NEEDLE: The secret verification code is JUPITER-19.]]]")
    parser.add_argument("--question_prefix", type=str, default="Return only the hidden verification code. Here is the book:\n")
    parser.add_argument("--answer_text", type=str, default="JUPITER-19")
    parser.add_argument("--layer_idx", type=int, default=14)
    parser.add_argument("--head_idx", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=128000)
    parser.add_argument("--q_path", type=str, default="q.bin")
    parser.add_argument("--k_path", type=str, default="k.bin")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True
    )
    model.eval()

    # Register hook
    store = {}
    attn_module = model.model.layers[args.layer_idx].self_attn
    attn_module.register_forward_hook(make_qk_hook(store, args.head_idx), with_kwargs=True)

    # Load and inject needle
    with open(args.book_path, "r", encoding="utf-8") as f:
        book_text = f.read()
    book_with_needle = insert_needle(book_text, args.needle_text, args.needle_ratio)

    # Build messages
    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    messages_full = build_messages(system_prompt, args.question_prefix, book_with_needle, args.answer_text)
    messages_sys_only = [{"role": "system", "content": system_prompt}]
    messages_sys_user_only = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.question_prefix + book_with_needle},
    ]
    # Compute token spans for question/answer
    full_text, (q_start, q_end, a_start, a_end) = lengths_for_spans(tokenizer, messages_full, messages_sys_only, messages_sys_user_only)

    # Tokenize full input for prefill (teacher forcing; no generation prompt)
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=args.max_length)
    # Track truncation impact on spans
    total_len = inputs["input_ids"].shape[1]
    q_start = min(q_start, total_len)
    q_end = min(q_end, total_len)
    a_start = min(a_start, total_len)
    a_end = min(a_end, total_len)

    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Prefill
    with torch.no_grad():
        _ = model(**inputs)

    if "q" not in store or "k" not in store:
        raise RuntimeError("Failed to capture Q/K from the specified layer/head. Check layer_idx/head_idx.")
    
    # Shapes: (bsz=1, seqlen, head_dim)
    q_all = store["q"]
    k_all = store["k"]

    if q_all.shape[0] != 1 or k_all.shape[0] != 1:
        raise RuntimeError(f"Unexpected batch size in captured tensors: q={q_all.shape}, k={k_all.shape}")
    
    # Slice by spans
    k_question = k_all[0, q_start:q_end, :]    # K for question portion
    q_answer = q_all[0, a_start:a_end, :]      # Q for answer portion

    # Save as float32 raw binary
    k_question.to(torch.float32).cpu().numpy().tofile(args.k_path)
    q_answer.to(torch.float32).cpu().numpy().tofile(args.q_path)

    # Optional: also print shapes and spans for sanity check
    print(f"Total tokens: {total_len}")
    print(f"Question span: [{q_start}, {q_end}) -> K shape {tuple(k_question.shape)}")
    print(f"Answer   span: [{a_start}, {a_end}) -> Q shape {tuple(q_answer.shape)}")
    print(f"Saved K(question) to {args.k_path}, Q(answer) to {args.q_path}")


if __name__ == "__main__":
    main()
