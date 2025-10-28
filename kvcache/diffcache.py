import math
import torch

from flash_attn import flash_attn_with_kvcache
from index_layer import IndexLayer, QueryHandle
import diffcache_cpu


class DiffCache:
    def __init__(
        self,
        layer_num: int,
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
        r_sq: float,
        cpu_thread_pool_size: int,
        dtype: torch.dtype,
    ) -> None:
        self.layer_num = layer_num
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
        self.dtype = dtype
        assert batch_size % minibatch_size == 0, "batch_size must be divisible by minibatch_size"
        self.num_mbs = batch_size // minibatch_size
        self.index_layers = [[None] * layer_num] * self.num_mbs
        self.data_layers = [[None] * layer_num] * self.num_mbs
        self.index_stream = torch.cuda.Stream()

        # initialize two layers
        for mb in range(self.num_mbs):
            for layer_idx in range(layer_num):
                # index layer on GPU
                self.index_layers[mb][layer_idx] = IndexLayer(
                    self.batch_size,
                    self.max_index_layer_sz,
                    self.num_kv_heads,
                    self.head_dim,
                    self.group_query
                )

                # data layer on CPU
                self.data_layers[mb][layer_idx] = diffcache_cpu.DiffCacheCPU(
                    bsz=self.mbsize,
                    q_head_num=self.num_q_heads,
                    kv_head_num=self.num_kv_heads,
                    head_dim=self.head_dim,
                    max_ctx_len=self.max_length,
                    m=self.nsw_m,
                    ef_cons=self.nsw_ef_cons,
                    thread_pool_size=self.cpu_thread_pool_size,
                    num_seeds=self.num_seeds,
                    r_sq=self.r_sq,
                    group_query=self.group_query
                )

    def cache_insert(
        self,
        k: torch.Tensor,    # (bsz, 1, group_num, dim)
        v: torch.Tensor,    # (bsz, 1, group_num, dim)
        layer_idx: int,
        mb_idx: int
    ):
        # 1. index layer lookup
        data_layer = self.data_layers[mb_idx][layer_idx]
        ep_ids, ep_dists = self.index_layers[mb_idx][layer_idx].query().collect()     # index layer lookup

        # 2. insert into data layer
        k_cpu = k.view(torch.uint16).numpy()
        v_cpu = v.view(torch.uint16).numpy()
        ep_ids_cpu = ep_ids.view(torch.uint16).numpy()
        ep_dists_cpu = ep_dists.view(torch.uint16).numpy()
        data_layer.insert(k_cpu, v_cpu, ep_ids_cpu, ep_dists_cpu)   # data layer insert

        # 2. insert into index layer by probability
        if torch.rand(1).item() < self.index_layer_prob:
            self.index_layers[layer_idx].insert(k, v)

    def prefill_update_kv_cache(
        self, 
        key_states: torch.Tensor,     # (bsz, seq_len, num_kv_heads, head_dim)
        value_states: torch.Tensor,   # (bsz, seq_len, num_kv_heads, head_dim)
        layer_idx,
    ):
        # TODO: accelerate prefill with batch insert

        key = key_states.view(self.num_mbs, self.mbsize, key_states.size(1), key_states.size(2), key_states.size(3))
        value = value_states.view(self.num_mbs, self.mbsize, value_states.size(1), value_states.size(2), value_states.size(3))

        seq_len = key_states.size(1)
        for t in range(seq_len):
            for mb in range(self.num_mbs):
                self.cache_insert(
                    k=key[mb, :, t:t+1, :, :], 
                    v=value[mb, :, t:t+1, :, :], 
                    layer_idx=layer_idx,
                    mb_idx=mb,
                )
        
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

        # 3. query CPU data layer immediately after index layer mb finish (pipelining)
        for mb, handle in enumerate(query_handles):
            # wait for index layer query to finish
            ep_ids, ep_dists = handle.collect()

            # copy args to CPU
            ep_ids_cpu = ep_ids.view(torch.uint32).numpy()
            ep_dists_cpu = ep_dists.view(torch.float32).numpy()
            q_cpu = query[mb, :, :, :].contiguous().view(torch.uint16).numpy()

            # data layer query
            data_layer = self.data_layers[mb][layer_idx]
            # retrieved KV: (mbsize, retrieval_budget, num_kv_heads, head_dim)
            k_retrieved_cpu, v_retrieved_cpu = data_layer.query(q_cpu, self.retrieval_budget, ep_dists_cpu, ep_ids_cpu)

            # copy retrieved KV to GPU
            k_retrieved = torch.from_numpy(k_retrieved_cpu).cuda().view(self.mbsize, self.retrieval_budget, self.num_kv_heads, self.head_dim)
            v_retrieved = torch.from_numpy(v_retrieved_cpu).cuda().view(self.mbsize, self.retrieval_budget, self.num_kv_heads, self.head_dim)

            # k_all: (bsz, retrieval_budget, num_kv_heads, head_dim)
            # v_all: (bsz, retrieval_budget, num_kv_heads, head_dim)
            if mb == 0:
                k_all = k_retrieved
                v_all = v_retrieved
            else:
                k_all = torch.cat([k_all, k_retrieved], dim=0)
                v_all = torch.cat([v_all, v_retrieved], dim=0)

        # 4. use flash attention with retrieved KV cache
        attn_out = flash_attn_with_kvcache(
            q=query, 
            k_cache=k_all, 
            v_cache=v_all,
            causal=True
        )

        return attn_out
