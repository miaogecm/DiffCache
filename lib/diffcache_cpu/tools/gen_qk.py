#!/usr/bin/env python3

import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


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
    model_name = 'Qwen/Qwen2.5-7B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # add hook
    store = {}
    head_idx = 0
    layer_idx = 14
    model.model.layers[layer_idx].self_attn.register_forward_hook(make_qk_hook(store, head_idx), with_kwargs=True)

    print('Loading book...')
    with open('2600-0.txt', 'r', encoding='utf-8') as f:
        content = f.read().strip()

    print('Tokenizing...')
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Summarize the following book: " + content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt", max_length=128000, truncation=True).to(model.device)
    print('Input tokens:', inputs['input_ids'].shape)

    print('Running prefilling...')
    with torch.no_grad():
        outputs = model(**inputs)
    print(store['q'].shape, store['k'].shape)

    print('Saving q and k to q.bin and k.bin as float32...')
    store['q'].to(torch.float32).numpy().tofile('q.bin')
    store['k'].to(torch.float32).numpy().tofile('k.bin')


if __name__ == "__main__":
    main()
