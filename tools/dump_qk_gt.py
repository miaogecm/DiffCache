#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump per-layer Q/K after RoPE from HF Qwen2.5 prefill"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--input_ids",
        type=str,
        default="../.data/input_ids.pt",
        help="Path to torch.save'd LongTensor with token ids",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/ubuntu/projects/DiffCache/.data/qkvdump",
        help="Output directory for dumped Q/K tensors",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on, e.g. cuda / cuda:0 / cpu",
    )
    parser.add_argument(
        "--rope_factor",
        type=float,
        default=4.0,
        help="Yarn rope_scaling factor",
    )
    parser.add_argument(
        "--original_max_pos",
        type=int,
        default=32768,
        help="Yarn original_max_position_embeddings",
    )
    return parser.parse_args()


def str_to_dtype(s: str) -> torch.dtype:
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load input_ids
    input_ids = torch.load(args.input_ids)
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("input_ids.pt must contain a torch.Tensor")

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    elif input_ids.dim() == 2:
        pass
    else:
        raise ValueError(f"Unsupported input_ids shape: {input_ids.shape}")

    bsz, seq_len = input_ids.shape
    print(f"Loaded input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}")

    device = torch.device(args.device)
    torch_dtype = str_to_dtype(args.dtype)

    # 2) Load HF model with Yarn rope_scaling
    print(f"Loading model {args.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=None,
        rope_scaling={
            "type": "yarn",
            "factor": args.rope_factor,
            "original_max_position_embeddings": args.original_max_pos,
        },
        attn_implementation="flash_attention_2"
    )
    model.to(device)
    model.eval()
    print("Model loaded.")

    # 3) Register hooks on each attention module
    handles = []

    def make_attn_hook(layer_idx: int):
        """
        Hook for Qwen2Attention.forward with with_kwargs=True.

        We recompute:
          - q_proj(hidden_states)
          - k_proj(hidden_states)
          - apply_rotary_pos_emb(q, k, cos, sin)
        and save Q/K after RoPE.
        """

        def hook(module: nn.Module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", None)
            if hidden_states is None:
                if len(args) >= 1:
                    hidden_states = args[0]
                else:
                    raise RuntimeError(
                        "Cannot find hidden_states in args or kwargs in attention hook"
                    )

            position_embeddings = kwargs.get("position_embeddings", None)
            if position_embeddings is None:
                raise RuntimeError(
                    "position_embeddings not found in kwargs; "
                    "this script expects a recent Qwen2 modeling version "
                    "where Qwen2Attention.forward gets (cos, sin) as position_embeddings"
                )

            cos, sin = position_embeddings
            bsz_local, seq_len_local, _ = hidden_states.shape

            head_dim = module.head_dim
            num_heads = module.q_proj.out_features // head_dim
            num_kv_heads = module.k_proj.out_features // head_dim

            # raw Q/K
            q_proj_out = module.q_proj(hidden_states)
            k_proj_out = module.k_proj(hidden_states)

            q = q_proj_out.view(bsz_local, seq_len_local, num_heads, head_dim).transpose(1, 2)
            k = k_proj_out.view(bsz_local, seq_len_local, num_kv_heads, head_dim).transpose(1, 2)
            # q, k: [bsz, num_heads/num_kv_heads, seq_len, head_dim]

            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

            q_rot = q_rot.transpose(1, 2).contiguous()  # [bsz, seq_len, num_heads, head_dim]
            k_rot = k_rot.transpose(1, 2).contiguous()  # [bsz, seq_len, num_kv_heads, head_dim]

            q_path = os.path.join(args_obj.out_dir, f"layer_{layer_idx:02d}_q.pt")
            k_path = os.path.join(args_obj.out_dir, f"layer_{layer_idx:02d}_k.pt")

            print(
                f"[layer {layer_idx:02d}] saving Q {tuple(q_rot.shape)} -> {q_path}; "
                f"K {tuple(k_rot.shape)} -> {k_path}"
            )

            torch.save(q_rot.to("cpu"), q_path)
            torch.save(k_rot.to("cpu"), k_path)

            # do not modify output
            del q_rot, k_rot, q_proj_out, k_proj_out

        # small trick: capture args in closure (for out_dir)
        args_obj = args  # just to keep a reference in closure is not needed, remove

        return hook

    # fix: we need args inside hook for out_dir; instead, bind it explicitly
    def make_attn_hook_with_cfg(layer_idx: int, cfg):
        def hook(module: nn.Module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", None)
            if hidden_states is None:
                if len(args) >= 1:
                    hidden_states = args[0]
                else:
                    raise RuntimeError(
                        "Cannot find hidden_states in args or kwargs in attention hook"
                    )

            position_embeddings = kwargs.get("position_embeddings", None)
            if position_embeddings is None:
                raise RuntimeError(
                    "position_embeddings not found in kwargs; "
                    "this script expects a recent Qwen2 modeling version "
                    "where Qwen2Attention.forward gets (cos, sin) as position_embeddings"
                )

            cos, sin = position_embeddings
            bsz_local, seq_len_local, _ = hidden_states.shape

            head_dim = module.head_dim
            num_heads = module.q_proj.out_features // head_dim
            num_kv_heads = module.k_proj.out_features // head_dim

            q_proj_out = module.q_proj(hidden_states)
            k_proj_out = module.k_proj(hidden_states)

            q = q_proj_out.view(bsz_local, seq_len_local, num_heads, head_dim).transpose(1, 2)
            k = k_proj_out.view(bsz_local, seq_len_local, num_kv_heads, head_dim).transpose(1, 2)

            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

            q_rot = q_rot.transpose(1, 2).contiguous()
            k_rot = k_rot.transpose(1, 2).contiguous()

            q_path = os.path.join(cfg.out_dir, f"layer_{layer_idx:02d}_q.pt")
            k_path = os.path.join(cfg.out_dir, f"layer_{layer_idx:02d}_k.pt")

            print(
                f"[layer {layer_idx:02d}] saving Q {tuple(q_rot.shape)} -> {q_path}; "
                f"K {tuple(k_rot.shape)} -> {k_path}"
            )

            torch.save(q_rot.to("cpu"), q_path)
            torch.save(k_rot.to("cpu"), k_path)

            del q_rot, k_rot, q_proj_out, k_proj_out

        return hook

    # register hooks
    for layer_idx, layer in enumerate(model.model.layers):
        attn_mod = layer.self_attn
        h = attn_mod.register_forward_hook(
            make_attn_hook_with_cfg(layer_idx, args),
            with_kwargs=True,
        )
        handles.append(h)

    # 4) Run a single full forward pass (prefill only, no cache)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        print(f"Running full forward: seq_len={seq_len}, bsz={bsz}")
        _ = model(input_ids=input_ids, use_cache=False)

    # 5) Cleanup hooks
    for h in handles:
        h.remove()

    print("Done. All per-layer Q/K (after RoPE) have been saved to:", args.out_dir)


if __name__ == "__main__":
    main()
