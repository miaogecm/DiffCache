import gc
import re
import os
import math
import json
import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config
from .base import BaseModel
from flash_attn import flash_attn_with_kvcache
from .kvcache.diffcache import DiffCache



class QwenLayer:
    """
    A class representing the Qwen layer.
    """

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device
    
    def init_layer(self, hf_qwen_layer):
        self.wq = hf_qwen_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_qwen_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_qwen_layer.self_attn.v_proj.weight.detach()
        self.bq = hf_qwen_layer.self_attn.q_proj.bias.detach()
        self.bk = hf_qwen_layer.self_attn.k_proj.bias.detach()
        self.bv = hf_qwen_layer.self_attn.v_proj.bias.detach()
        self.wqkv = torch.cat((self.wq, self.wk, self.wv), dim=0).to(self.device, non_blocking=True)
        self.bqkv = torch.cat((self.bq, self.bk, self.bv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_qwen_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)
        
        self.gate_proj = hf_qwen_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_qwen_layer.mlp.up_proj.weight.detach()
        self.gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=0).to(self.device, non_blocking=True)
        self.down_proj = hf_qwen_layer.mlp.down_proj.weight.detach().to(self.device, non_blocking=True)

        self.input_layernorm_weight = hf_qwen_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_qwen_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_qwen_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_qwen_layer.post_attention_layernorm.variance_epsilon

        del self.wq, self.wk, self.wv, self.bq, self.bk, self.bv, self.gate_proj, self.up_proj


class QwenModel(BaseModel):
    """
    A class representing the Qwen model.
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(model_name, max_length, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = Qwen2Config.from_pretrained(model_name)
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.base = self.config.rope_theta
        self.max_position_embeddings = self.config.max_position_embeddings
        self.yarn_factor = 4         # # qwen2.5 use yarn in context length larger than 32768
        self.vocab_size = self.config.vocab_size
        self.eos_tokens = [self.config.eos_token_id]

        self.init_model()


    def _set_cos_sin_cache(self):
        device = self.device_map
        device_type = "cuda" if "cuda" in str(device) else "cpu"

        t = torch.arange(self.max_length, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(torch.float32)

        with torch.autocast(device_type=device_type, enabled=False):        # force float32
            freqs = torch.outer(t, inv_freq)
            cos = freqs.cos() * float(self.attention_scaling)
            sin = freqs.sin() * float(self.attention_scaling)

        return cos, sin


    def init_model(self):
        hf_qwen = Qwen2ForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=self.dtype,
            rope_scaling={
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        )

        self.embed_tokens = hf_qwen.model.embed_tokens.weight.detach().to(self.device_map, non_blocking=True)
        self.lm_head = hf_qwen.lm_head.weight.detach().to(self.device_map, non_blocking=True)

        self.norm_weight = hf_qwen.model.norm.weight.detach().to(self.device_map, non_blocking=True)
        self.norm_variance_epsilon = hf_qwen.model.norm.variance_epsilon

        self.position_ids = torch.arange(0, self.max_length).to(self.device_map, non_blocking=True)
        self.inv_freq = hf_qwen.model.rotary_emb.inv_freq.detach().to(self.device_map, non_blocking=True)
        self.attention_scaling = hf_qwen.model.rotary_emb.attention_scaling
        self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
        self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

        self.layers = []
        for idx, hf_qwen_layer in enumerate(hf_qwen.model.layers):
            qwen_layer = QwenLayer(idx, device=self.device_map)
            qwen_layer.init_layer(hf_qwen_layer)
            self.layers.append(qwen_layer)
            hf_qwen.model.layers[idx] = None

        del self.inv_freq, self.cos_cache, self.sin_cache
        gc.collect()
        torch.cuda.empty_cache()

    
    def init_kv_cache(self, real_input_length, valid_start, attn_config=None):
        if attn_config is None:
            PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
            CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
            MODEL_NAME = self.model_name.split("/")[-1]+'.json'
            CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)

            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        else:
            config = attn_config

        print(f"prompt_len={real_input_length} gen_max_len={self.max_new_length}")
        
        # Init kv cache
        r_sq = self.calc_r_sq()
        self.kv_cache = DiffCache(
            num_layers = self.num_layers,
            batch_size = self.batch_size,
            max_length = self.max_new_length + real_input_length,
            num_kv_heads = self.num_key_value_heads,
            num_q_heads = self.num_heads,
            head_dim = self.head_dim,
            index_layer_prob = config["index_layer_prob"],
            max_index_layer_sz = config["max_index_layer_sz"],
            nsw_m = config["nsw_m"],
            nsw_ef_cons = config["nsw_ef_cons"],
            r_sq = r_sq,
            minibatch_size = config["minibatch_size"],
            num_seeds = config["num_seeds"],
            retrieval_budget = config["retrieval_budget"],
            batch_prefill = config["batch_prefill"],
            prefix_kvcache_len = config["prefix_kvcache_len"],
            suffix_kvcache_len = config["suffix_kvcache_len"]
        )
        print(f"KVCache init, r_sq={r_sq}")

    
    def calc_r_sq(self):
        r_sq = []
        dim = self.hidden_size
        kv_dim = self.num_key_value_heads * self.head_dim

        for layer in self.layers:
            norm_weight = layer.input_layernorm_weight.float()
            wk_all = layer.wqkv[self.hidden_size:self.hidden_size + kv_dim, :]
            wk_heads = wk_all.view(self.num_key_value_heads, self.head_dim, self.hidden_size)
            bk_all = layer.bqkv[self.hidden_size:self.hidden_size + kv_dim]
            bk_heads = bk_all.view(self.num_key_value_heads, self.head_dim)

            r_sq_layer = []

            for head_idx in range(self.num_key_value_heads):
                # same as llama
                head_weight = wk_heads[head_idx].float()
                scaled = head_weight * norm_weight
                gram = scaled @ scaled.transpose(0, 1)
                gram = 0.5 * (gram + gram.transpose(0, 1))
                gram_cpu = gram.cpu().double()
                sigma_sq = torch.linalg.eigvalsh(gram_cpu).amax().item()
                sigma_sq = max(sigma_sq, 0.0)

                # add bias term
                bias = bk_heads[head_idx].float()
                bias_norm = torch.linalg.vector_norm(bias).item()
                head_r = math.sqrt(dim * sigma_sq) + bias_norm
                r_sq_layer.append(head_r * head_r)

                del head_weight, scaled, gram, gram_cpu, bias

            r_sq.append(r_sq_layer)
            del norm_weight, wk_all, wk_heads, bk_all, bk_heads

        attention_scale = float(self.attention_scaling)
        if attention_scale != 1.0:
            scale_sq = attention_scale * attention_scale
            r_sq = [[val * scale_sq for val in r_sq_layer] for r_sq_layer in r_sq]    # RoPE Yarn scaling inflates ||k|| by attention_scale

        return r_sq

    
    def word_embedding(self, inputs_id):
        hidden_states = F.embedding(inputs_id, self.embed_tokens)
        return hidden_states

    
    def lm(self, hidden_states):
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits


    def wqkv(self, hidden_states, layer):
        qkv = F.linear(hidden_states, layer.wqkv, layer.bqkv)
        query_states, key_states, value_states = qkv.split([self.hidden_size, self.hidden_size//self.num_key_value_groups, self.hidden_size//self.num_key_value_groups], dim=-1)
        return query_states, key_states, value_states

    
    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        hidden_states = F.linear(hidden_states, layer.wo)
        return hidden_states

    
    def prefill_attention(self, query_states, key_states, value_states):
        return flash_attn_with_kvcache(
            q=query_states, 
            k_cache=key_states, 
            v_cache=value_states,
            causal=True
        )
    

    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        return self.kv_cache.compute(query_states, layer_idx)

    
    def mlp(self, hidden_states, layer):
        hidden_states = F.linear(hidden_states, layer.gate_up_proj)
        dim = hidden_states.shape[-1] // 2
        hidden_shape = (hidden_states.shape[:-1] + (dim,))
        out = torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        flashinfer.activation.silu_and_mul(hidden_states, out)
        hidden_states = F.linear(out, layer.down_proj)
        return hidden_states 

    
    def layernorm(self, hidden_states, epsilon, weight):
        bsz, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.reshape(bsz * seq_len, dim)
        hidden_states = flashinfer.rmsnorm(hidden_states, weight, epsilon)
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        return hidden_states


    def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
        bsz, _, hidden_dim = query_states.shape
        _, _, kv_dim = key_states.shape
        query_states = query_states.view(-1, hidden_dim)
        key_states = key_states.view(-1, kv_dim)
        positions = position_ids.reshape(-1).contiguous()
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(positions, query_states, key_states, self.head_dim, self.cos_sin_cache, True)
        query_states = query_states.view(bsz, -1, hidden_dim)
        key_states = key_states.view(bsz, -1, kv_dim)
        return query_states, key_states


    def position_embedd(self, query_states, key_states, cached_seq_len):
        bsz, seq_len, _ = key_states.shape

        position_ids = self.position_ids[cached_seq_len:cached_seq_len+seq_len].unsqueeze(0).repeat(bsz, 1)

        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

        return query_states, key_states
