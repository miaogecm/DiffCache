import math
from typing import List
from codetiming import Timer
import torch

from tqdm import tqdm
from .._cpu import DiffCacheCPU, init_metrics
from cuvs.neighbors import cagra
import numpy as np
import torch.utils.dlpack as dlpack
import os
import json


class DiffCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_length: int,
        num_kv_heads: int,
        num_q_heads: int,
        head_dim: int,
        retrieval_budget: int,
        nsw_m: int,
        nsw_ef_cons: int,
        kvbuf_prefix_len: int,
        kvbuf_suffix_len: int,
        kvbuf_suffix_maxlen: int,
        r_sq: List[List[float]]
    ) -> None:
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.retrieval_budget = retrieval_budget
        self.nsw_m = nsw_m
        self.nsw_ef_cons = nsw_ef_cons
        self.r_sq = r_sq
        self.data_layers = [None for _ in range(num_layers)]

        self.retrieval_region_len = 0

        # kvcache buffer (kvbuf) on GPU
        #    ┌────────────────────┬────────────┬───────────────┐    
        # ◄──┼  retrieval_budget  │ prefix_len │ suffix_maxlen ┼───►
        #    └────────────────────┴────────────┴───────────────┘    
        # Note that the permutation is (bsz, kh, seq_len, hd), not (bsz, seq_len, kh, hd)
        self.kvbuf_suffix_maxlen = kvbuf_suffix_maxlen
        self.kvbuf_prefix_len = kvbuf_prefix_len
        self.kvbuf_suffix_len = kvbuf_suffix_len
        kvbuf_len = retrieval_budget + kvbuf_prefix_len + kvbuf_suffix_maxlen
        self.kbuf = [torch.empty(
            (batch_size, num_kv_heads, kvbuf_len, head_dim),
            device='cuda',
            dtype=torch.bfloat16,
        ) for _ in range(num_layers)]
        self.vbuf = [torch.empty(
            (batch_size, num_kv_heads, kvbuf_len, head_dim),
            device='cuda',
            dtype=torch.bfloat16,
        ) for _ in range(num_layers)]

        # CPU kv buffer
        # Note that the permutation is (bsz, kh, seq_len, hd), not (bsz, seq_len, kh, hd)
        self.kbuf_cpu = [torch.zeros(
            (batch_size, num_kv_heads, self.retrieval_budget, head_dim),
            device='cpu',
            dtype=torch.bfloat16,
        ).pin_memory() for _ in range(num_layers)]
        self.vbuf_cpu = [torch.zeros(
            (batch_size, num_kv_heads, self.retrieval_budget, head_dim),
            device='cpu',
            dtype=torch.bfloat16,
        ).pin_memory() for _ in range(num_layers)]
        self.qbuf_cpu = torch.zeros((batch_size, num_kv_heads, head_dim), device='cpu', dtype=torch.bfloat16).pin_memory()
        # since numpy does not support bfloat16, we use uint16 to store bfloat16 data, and pass to data layer
        self.kbuf_cpu_np = [buf.view(dtype=torch.uint16).numpy() for buf in self.kbuf_cpu]
        self.vbuf_cpu_np = [buf.view(dtype=torch.uint16).numpy() for buf in self.vbuf_cpu]
        self.qbuf_cpu_np = self.qbuf_cpu.view(dtype=torch.uint16).numpy()

        # initialize data layer
        for layer_idx in range(num_layers):
            self.data_layers[layer_idx] = DiffCacheCPU(
                bsz=self.batch_size,
                q_head_num=self.num_q_heads,
                kv_head_num=self.num_kv_heads,
                head_dim=self.head_dim,
                max_ctx_len=self.max_length,
                m=self.nsw_m,
                ef_cons=self.nsw_ef_cons,
                r_sq=self.r_sq[layer_idx],
                name=f"l{layer_idx:03d}"
            )
        
        # initialize performance metrics
        init_metrics()

    def _prefill_update_index(
        self, 
        key_states: torch.Tensor,     # (bsz, seq_len, num_kv_heads, head_dim)
        value_states: torch.Tensor,   # (bsz, seq_len, num_kv_heads, head_dim)
        layer_idx
    ):
        keys = key_states.view(self.batch_size, key_states.size(1), key_states.size(2), key_states.size(3))
        values = value_states.view(self.batch_size, value_states.size(1), value_states.size(2), value_states.size(3))
        deg = 2 * self.nsw_m
        seq_len = key_states.size(1)

        torch.cuda.synchronize()
        with Timer(f"cagra_build", logger=None):
            # build data layer using CAGRA
            neighbours = np.empty((self.batch_size, self.num_kv_heads, seq_len, deg), dtype=np.uint32)

            for b in range(self.batch_size):
                for h in range(self.num_kv_heads):
                    k = keys[b, :, h, :].to(torch.float32)  # (seq_len, head_dim)

                    # lift dim
                    norm_sq = (k * k).sum(dim=1, keepdim=True)
                    extra = torch.sqrt(torch.clamp(self.r_sq[layer_idx][h] - norm_sq, min=0.0))
                    k = torch.cat([k, extra], dim=1)  # (seq_len, head_dim + 1)

                    # build index and extract graph
                    build_params = cagra.IndexParams(
                        metric="sqeuclidean", 
                        graph_degree=deg,
                        intermediate_graph_degree=4 * deg,
                    )
                    index = cagra.build(build_params, k)
                    neighbours[b, h, :, :] = index.graph.copy_to_host()

                    del index, k, norm_sq, extra
            torch.cuda.synchronize()

        # CPU prefill
        data_layer = self.data_layers[layer_idx]
        k_cpu = keys.permute(0, 2, 1, 3).contiguous().view(dtype=torch.uint16).cpu().numpy()        # (batch_size, num_kv_heads, seq_len, head_dim)
        v_cpu = values.permute(0, 2, 1, 3).contiguous().view(dtype=torch.uint16).cpu().numpy()      # (batch_size, num_kv_heads, seq_len, head_dim)
        neighbours_cpu = neighbours[:, :, :, :]                                     # (batch_size, num_kv_heads, seq_len, deg)
        data_layer.prefill(k_cpu, v_cpu, neighbours_cpu)

    def prefill_update_kv_cache(
        self, 
        key_states: torch.Tensor,     # (bsz, seq_len, num_kv_heads, head_dim)
        value_states: torch.Tensor,   # (bsz, seq_len, num_kv_heads, head_dim)
        layer_idx
    ):
        seq_len = key_states.size(1)

        # estimate r_sq based on prefill result
        r_sq = torch.einsum("bshd,bshd->bsh", key_states, key_states).max(dim=1).values.max(dim=0).values.tolist()
        self.r_sq[layer_idx] = r_sq
        self.data_layers[layer_idx].update_r_sq(r_sq)

        # (1) save prefix region to GPU
        prefix_len = min(self.kvbuf_prefix_len, seq_len)
        self.kbuf[layer_idx][:, :, self.retrieval_budget:self.retrieval_budget + prefix_len, :].copy_(key_states[:, :prefix_len, :, :].permute(0, 2, 1, 3))
        self.vbuf[layer_idx][:, :, self.retrieval_budget:self.retrieval_budget + prefix_len, :].copy_(value_states[:, :prefix_len, :, :].permute(0, 2, 1, 3))
        self.kvbuf_prefix_len = prefix_len

        # (2) save suffix region to GPU
        suffix_len = min(self.kvbuf_suffix_len, seq_len - prefix_len)
        self.kbuf[layer_idx][:, :, self.retrieval_budget + prefix_len:self.retrieval_budget + prefix_len + suffix_len, :].copy_(key_states[:, seq_len - suffix_len:seq_len, :, :].permute(0, 2, 1, 3))
        self.vbuf[layer_idx][:, :, self.retrieval_budget + prefix_len:self.retrieval_budget + prefix_len + suffix_len, :].copy_(value_states[:, seq_len - suffix_len:seq_len, :, :].permute(0, 2, 1, 3))
        self.kvbuf_suffix_len = suffix_len
        
        # (3) save retrieval region to ANNS index
        retrieval_key_states = key_states[:, prefix_len:seq_len - suffix_len, :, :]
        retrieval_value_states = value_states[:, prefix_len:seq_len - suffix_len, :, :]
        self._prefill_update_index(
            key_states=retrieval_key_states,
            value_states=retrieval_value_states,
            layer_idx=layer_idx
        )
        self.retrieval_region_len = retrieval_key_states.size(1)

        return key_states, value_states

    def get_kvcache(self, layer_idx):
        retrieval_len = min(self.retrieval_region_len, self.retrieval_budget)
        start = self.retrieval_budget - retrieval_len
        return self.kbuf[layer_idx][:, :, start:, :].permute(0, 2, 1, 3), self.vbuf[layer_idx][:, :, start:, :].permute(0, 2, 1, 3)
    
    def decode_pre_query_operation(self, queries: torch.Tensor):
        group_size = self.num_q_heads // self.num_kv_heads
        q_mean = queries.view(self.batch_size, self.num_kv_heads, group_size, self.head_dim).mean(dim=2)  # (bsz, num_kv_heads, head_dim)
        self.qbuf_cpu.copy_(q_mean, non_blocking=True)

    def decode_post_query_operation(self, layer_idx):
        retrieval_len = min(self.retrieval_region_len, self.retrieval_budget)
        if retrieval_len <= 0:
            return
        
        # D2H copy retrieved KV from CPU to GPU
        self.kbuf[layer_idx][:, :, self.retrieval_budget - retrieval_len:self.retrieval_budget, :].copy_(
            self.kbuf_cpu[layer_idx][:, :, :retrieval_len, :],
            non_blocking=True
        )
        self.vbuf[layer_idx][:, :, self.retrieval_budget - retrieval_len:self.retrieval_budget, :].copy_(
            self.vbuf_cpu[layer_idx][:, :, :retrieval_len, :],
            non_blocking=True
        )

    def decode_query_kvcache(self, layer_idx):
        retrieval_len = min(self.retrieval_region_len, self.retrieval_budget)
        if retrieval_len > 0:
            # wait for previous query copy
            torch.cuda.synchronize()
            with Timer("data_layer_query", logger=None):
                # CPU retrieved KV: (bsz, num_kv_heads, retrieval_len, head_dim)
                self.data_layers[layer_idx].query(self.kbuf_cpu_np[layer_idx][:, :, :retrieval_len, :], 
                                self.vbuf_cpu_np[layer_idx][:, :, :retrieval_len, :], 
                                self.qbuf_cpu_np, retrieval_len)
        assert self.kvbuf_suffix_len < self.kvbuf_suffix_maxlen
        if layer_idx == self.num_layers - 1:
            self.kvbuf_suffix_len += 1

    def save_prefill_cache(self, prefill_cache_path):
        print("Saving prefill cache...")

        path = os.path.join(prefill_cache_path, "metadata.txt")
        metadata = {
            "r_sq": self.r_sq,
            "kvbuf_prefix_len": self.kvbuf_prefix_len,
            "kvbuf_suffix_len": self.kvbuf_suffix_len,
            "retrieval_region_len": self.retrieval_region_len
        }
        with open(path, 'w') as f:
            json.dump(metadata, f)

        # save prefix cache
        for layer_idx in range(self.num_layers):
            layer_path = os.path.join(prefill_cache_path, f"p_l_{layer_idx:03d}")
            os.makedirs(layer_path, exist_ok=False)
            torch.save(self.kbuf[layer_idx][:, :, self.retrieval_budget:self.retrieval_budget + self.kvbuf_prefix_len, :], os.path.join(layer_path, "k.pt"))
            torch.save(self.vbuf[layer_idx][:, :, self.retrieval_budget:self.retrieval_budget + self.kvbuf_prefix_len, :], os.path.join(layer_path, "v.pt"))

        # save suffix cache
        for layer_idx in range(self.num_layers):
            layer_path = os.path.join(prefill_cache_path, f"s_l_{layer_idx:03d}")
            os.makedirs(layer_path, exist_ok=False)
            torch.save(self.kbuf[layer_idx][:, :, self.retrieval_budget + self.kvbuf_prefix_len:self.retrieval_budget + self.kvbuf_prefix_len + self.kvbuf_suffix_len, :], os.path.join(layer_path, "k.pt"))
            torch.save(self.vbuf[layer_idx][:, :, self.retrieval_budget + self.kvbuf_prefix_len:self.retrieval_budget + self.kvbuf_prefix_len + self.kvbuf_suffix_len, :], os.path.join(layer_path, "v.pt"))

        # data layer (for retrieval index)
        for layer_idx in tqdm(range(self.num_layers)):
            data_layer = self.data_layers[layer_idx]
            layer_path = os.path.join(prefill_cache_path, f"d_l_{layer_idx:03d}")
            os.makedirs(layer_path, exist_ok=False)
            data_layer.save(layer_path)
        
        print("Prefill cache saved.")

    def load_prefill_cache(self, inputs_ids, prefill_cache_path):
        print("Loading prefill cache...")

        ctx_len = inputs_ids.size(1)

        path = os.path.join(prefill_cache_path, "metadata.txt")
        with open(path, 'r') as f:
            metadata = json.load(f)
        self.r_sq = metadata["r_sq"]
        self.kvbuf_prefix_len = metadata["kvbuf_prefix_len"]
        self.kvbuf_suffix_len = metadata["kvbuf_suffix_len"]
        self.retrieval_region_len = metadata["retrieval_region_len"]
        for layer_idx in range(self.num_layers):
            self.data_layers[layer_idx].update_r_sq(self.r_sq[layer_idx])
            
        for layer_idx in range(self.num_layers):
            # prefix load
            prefix_layer_path = os.path.join(prefill_cache_path, f"p_l_{layer_idx:03d}")
            self.kbuf[layer_idx][:, :, self.retrieval_budget:self.retrieval_budget + self.kvbuf_prefix_len, :] = torch.load(os.path.join(prefix_layer_path, "k.pt"))
            self.vbuf[layer_idx][:, :, self.retrieval_budget:self.retrieval_budget + self.kvbuf_prefix_len, :] = torch.load(os.path.join(prefix_layer_path, "v.pt"))

            # suffix load
            suffix_layer_path = os.path.join(prefill_cache_path, f"s_l_{layer_idx:03d}")
            self.kbuf[layer_idx][:, :, self.retrieval_budget + self.kvbuf_prefix_len:self.retrieval_budget + self.kvbuf_prefix_len + self.kvbuf_suffix_len, :] = torch.load(os.path.join(suffix_layer_path, "k.pt"))
            self.vbuf[layer_idx][:, :, self.retrieval_budget + self.kvbuf_prefix_len:self.retrieval_budget + self.kvbuf_prefix_len + self.kvbuf_suffix_len, :] = torch.load(os.path.join(suffix_layer_path, "v.pt"))

        # data layer
        for layer_idx in tqdm(range(self.num_layers)):
            data_layer = self.data_layers[layer_idx]
            layer_path = os.path.join(prefill_cache_path, f"d_l_{layer_idx:03d}")
            data_layer.load(layer_path, ctx_len)

        print("Prefill cache loaded.")

    def seq_len(self, layer_idx):
        return self.kvbuf_prefix_len + self.kvbuf_suffix_len + self.retrieval_region_len

    def seq_len_gpu(self, layer_idx):
        retrieval_len = min(self.retrieval_region_len, self.retrieval_budget)
        return self.kvbuf_prefix_len + self.kvbuf_suffix_len + retrieval_len
