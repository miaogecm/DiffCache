from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
import torch
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                         dtype=torch.bfloat16,
                                         local_files_only=True).eval()
seq_len = 2048
inp = torch.randint(0, model.config.vocab_size, (1, seq_len))
layer = model.model.layers[23]                   # << your hook location
hidden = layer.input_layernorm(model.model.embed_tokens(inp))
attn = layer.self_attn
q = attn.q_proj(hidden).view(1, seq_len, model.config.num_attention_heads, attn.head_dim).transpose(1, 2)
k = attn.k_proj(hidden).view(1, seq_len, model.config.num_key_value_heads, attn.head_dim).transpose(1, 2)
cos, sin = model.model.rotary_emb(q, torch.arange(seq_len).unsqueeze(0))
_, k = apply_rotary_pos_emb(q, k, cos, sin)
norms = torch.linalg.vector_norm(k.transpose(1, 2), dim=-1).mean(dim=1)
print(norms)
