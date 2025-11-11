import math
from typing import List
import torch

from flash_attn import flash_attn_with_kvcache
from tqdm import tqdm
from .index_layer import IndexLayer, QueryHandle
from .._cpu import DiffCacheCPU
from cuvs.neighbors import cagra
import numpy as np
import torch.utils.dlpack as dlpack


class DiffCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_length: int,
        num_kv_heads: int,
        num_q_heads: int,
        head_dim: int,
        index_layer_prob: float,
        max_index_layer_sz: int,
        minibatch_size: int,
        num_seeds: int,
        retrieval_budget: int,
        group_query: bool,
        nsw_m: int,
        nsw_ef_cons: int,
        r_sq: List[float],
        cpu_thread_pool_size: int,
    ) -> None:
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.index_layer_prob = index_layer_prob
        self.max_index_layer_sz = max_index_layer_sz
        self.mbsize = minibatch_size
        self.num_seeds = num_seeds
        self.retrieval_budget = retrieval_budget
        self.group_query = group_query
        self.nsw_m = nsw_m
        self.nsw_ef_cons = nsw_ef_cons
        self.r_sq = r_sq
        self.cpu_thread_pool_size = cpu_thread_pool_size
        assert batch_size % minibatch_size == 0, "batch_size must be divisible by minibatch_size"
        self.num_mbs = batch_size // minibatch_size
        self.index_layers = [[None for _ in range(num_layers)] for _ in range(self.num_mbs)]
        self.data_layers = [[None for _ in range(num_layers)] for _ in range(self.num_mbs)]
        self.index_stream = torch.cuda.Stream()
        self.cached_seq_len = [0 for _ in range(num_layers)]

        # initialize two layers
        for mb in range(self.num_mbs):
            for layer_idx in range(num_layers):
                # index layer on GPU
                self.index_layers[mb][layer_idx] = IndexLayer(
                    self.mbsize,
                    self.max_index_layer_sz,
                    self.num_kv_heads,
                    self.head_dim,
                    self.r_sq[layer_idx],
                    self.group_query,
                )

                # data layer on CPU
                self.data_layers[mb][layer_idx] = DiffCacheCPU(
                    bsz=self.mbsize,
                    q_head_num=self.num_q_heads,
                    kv_head_num=self.num_kv_heads,
                    head_dim=self.head_dim,
                    max_ctx_len=self.max_length,
                    m=self.nsw_m,
                    ef_cons=self.nsw_ef_cons,
                    thread_pool_size=self.cpu_thread_pool_size,
                    r_sq=self.r_sq[layer_idx],
                    group_query=self.group_query
                )

    def cache_insert(
        self,
        k: torch.Tensor,    # (bsz, 1, num_kv_heads, dim)
        v: torch.Tensor,    # (bsz, 1, num_kv_heads, dim)
        node_id: int,
        layer_idx: int,
        mb_idx: int
    ):
        k = k.squeeze(1)   # (bsz, num_kv_heads, dim)
        v = v.squeeze(1)   # (bsz, num_kv_heads, dim)

        # 1. index layer lookup
        data_layer = self.data_layers[mb_idx][layer_idx]
        ep_ids, ep_dists = self.index_layers[mb_idx][layer_idx].query(k, ef=self.num_seeds).collect()     # index layer lookup

        # 2. insert into data layer
        # since numpy does not support bfloat16, we use uint16 to store bfloat16 data, and pass to data layer
        k_cpu = k.view(dtype=torch.uint16).cpu().numpy()
        v_cpu = v.view(dtype=torch.uint16).cpu().numpy()
        ep_ids_cpu = ep_ids.cpu().numpy()
        ep_dists_cpu = ep_dists.cpu().numpy()
        data_layer.insert(k_cpu, v_cpu, ep_dists_cpu, ep_ids_cpu)   # data layer insert

        # 2. insert into index layer by probability
        if torch.rand(1).item() < self.index_layer_prob:
            k_expand = k.unsqueeze(1)
            node_ids = torch.full(
                (self.mbsize, 1),
                fill_value=node_id,
                dtype=torch.int64,
                device=k.device,
            )
            self.index_layers[mb_idx][layer_idx].insert(k_expand, node_ids)


    def prefill_update_kv_cache(
        self, 
        key_states: torch.Tensor,     # (bsz, seq_len, num_kv_heads, head_dim)
        value_states: torch.Tensor,   # (bsz, seq_len, num_kv_heads, head_dim)
        layer_idx,
    ):
        keys = key_states.view(self.num_mbs, self.mbsize, key_states.size(1), key_states.size(2), key_states.size(3))
        values = value_states.view(self.num_mbs, self.mbsize, value_states.size(1), value_states.size(2), value_states.size(3))
        deg = 2 * self.nsw_m
        seq_len = key_states.size(1)
        
        # build data layer using CAGRA
        neighbours = np.empty((self.num_mbs, self.mbsize, self.num_kv_heads, seq_len, deg), dtype=np.uint32)

        for mb in range(self.num_mbs):
            for b in range(self.mbsize):
                for h in range(self.num_kv_heads):
                    k = keys[mb, b, :, h, :].to(torch.float32)  # (seq_len, head_dim)

                    # lift dim
                    norm_sq = (k * k).sum(dim=1, keepdim=True)
                    extra = torch.sqrt(torch.clamp(self.r_sq[layer_idx][h] - norm_sq, min=0.0))
                    k = torch.cat([k, extra], dim=1)  # (seq_len, head_dim + 1)

                    # build index and extract graph
                    build_params = cagra.IndexParams(metric="sqeuclidean", graph_degree=deg)
                    index = cagra.build(build_params, k)
                    neighbours[mb, b, h, :, :] = index.graph.copy_to_host()

                    del index, k, norm_sq, extra

        # CPU prefill
        for mb in range(self.num_mbs):
            data_layer = self.data_layers[mb][layer_idx]
            k_cpu = keys[mb].contiguous().view(dtype=torch.uint16).cpu().numpy()        # (mbsize, seq_len, num_kv_heads, head_dim)
            v_cpu = values[mb].contiguous().view(dtype=torch.uint16).cpu().numpy()      # (mbsize, seq_len, num_kv_heads, head_dim)
            neighbours_cpu = neighbours[mb, :, :, :, :]                                 # (mbsize, num_kv_heads, seq_len, deg)
            data_layer.prefill(k_cpu, v_cpu, neighbours_cpu)    # async

        # build index layer from sample
        # sample index_layer_prob fraction of keys
        sample_num = int(seq_len * self.index_layer_prob)
        for mb in range(self.num_mbs):
            sampled_indices = torch.randperm(seq_len)[:sample_num]
            k_sampled = keys[mb, :, sampled_indices, :, :]  # (mbsize, sample_num, num_kv_heads, head_dim)
            node_ids = sampled_indices.unsqueeze(0).repeat(self.mbsize, 1).to(k_sampled.device)  # (mbsize, sample_num)
            self.index_layers[mb][layer_idx].insert(k_sampled, node_ids)

        self.cached_seq_len[layer_idx] = seq_len

        return key_states, value_states
        
    def decode_update_kv_cache(self,
        key_states,     # (bs, length(=1), num_kv_heads, dim)
        value_states,   # (bs, length(=1), num_kv_heads, dim)
        layer_idx
    ):
        key = key_states.view(self.num_mbs, self.mbsize, key_states.size(1), key_states.size(2), key_states.size(3))
        value = value_states.view(self.num_mbs, self.mbsize, value_states.size(1), value_states.size(2), value_states.size(3))

        for mb in range(self.num_mbs):
            self.cache_insert(
                k=key[mb, :, :, :, :], 
                v=value[mb, :, :, :, :], 
                layer_idx=layer_idx,
                mb_idx=mb,
            )
        
        return key_states, value_states
    
    def compute(
        self, 
        queries,        # (bsz, num_q_heads, head_dim)
        layer_idx
    ):
        # 1. divide into minibatches
        query = queries.view(self.num_mbs, self.mbsize, self.num_q_heads, self.head_dim)

        # 2. issue GPU index layer queries in a stream
        query_handles = []
        with torch.cuda.stream(self.index_stream):
            for mb in range(self.num_mbs):
                q_mini = query[mb, :, :, :]   # (mbsize, num_q_heads, head_dim)
                handle = self.index_layers[mb][layer_idx].query(q_mini, ef=self.num_seeds)
                query_handles.append(handle)

        total_bsz = queries.size(0)
        device = queries.device
        k_all = torch.empty(
            (total_bsz, self.retrieval_budget, self.num_kv_heads, self.head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        v_all = torch.empty_like(k_all)

        # 3. query CPU data layer immediately after index layer mb finish (pipelining)
        for mb, handle in enumerate(query_handles):
            # wait for index layer query to finish
            ep_ids, ep_dists = handle.collect()

            # copy args to CPU
            ep_ids_cpu = ep_ids.cpu().numpy()
            ep_dists_cpu = ep_dists.cpu().numpy()
            # since numpy does not support bfloat16, we use uint16 to store bfloat16 data, and pass to data layer
            q_cpu = query[mb, :, :, :].contiguous().view(dtype=torch.uint16).cpu().numpy()

            # data layer query
            data_layer = self.data_layers[mb][layer_idx]
            # retrieved KV: (mbsize, retrieval_budget, num_kv_heads, head_dim)
            k_retrieved_cpu, v_retrieved_cpu = data_layer.query(q_cpu, self.retrieval_budget, ep_dists_cpu, ep_ids_cpu).collect()

            # copy retrieved KV to GPU
            k_retrieved = torch.from_numpy(k_retrieved_cpu).view(dtype=torch.bfloat16).to(device=device).view(self.mbsize, self.retrieval_budget, self.num_kv_heads, self.head_dim)
            v_retrieved = torch.from_numpy(v_retrieved_cpu).view(dtype=torch.bfloat16).to(device=device).view(self.mbsize, self.retrieval_budget, self.num_kv_heads, self.head_dim)

            start = mb * self.mbsize
            end = start + self.mbsize
            k_all[start:end].copy_(k_retrieved)
            v_all[start:end].copy_(v_retrieved)

        # 4. use flash attention with retrieved KV cache
        attn_out = flash_attn_with_kvcache(
            q=queries, 
            k_cache=k_all, 
            v_cache=v_all,
            causal=True
        )

        return attn_out
