import math
from typing import List
from codetiming import Timer
import torch

from flash_attn import flash_attn_with_kvcache
from tqdm import tqdm
from .index_layer import IndexLayer, QueryHandle
from .._cpu import DiffCacheCPU, init_metrics
from cuvs.neighbors import cagra
import numpy as np
import torch.utils.dlpack as dlpack
import os
import json


MAX_QUERY_HISTORY = 32


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
        nsw_m: int,
        nsw_ef_cons: int,
        prefix_kvcache_len: int,
        suffix_kvcache_len: int,
        r_sq: List[List[float]],
        batch_prefill: bool
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
        self.nsw_m = nsw_m
        self.nsw_ef_cons = nsw_ef_cons
        self.r_sq = r_sq
        assert batch_size % minibatch_size == 0, "batch_size must be divisible by minibatch_size"
        self.num_mbs = batch_size // minibatch_size
        self.index_layers = [[None for _ in range(num_layers)] for _ in range(self.num_mbs)]
        self.data_layers = [[None for _ in range(num_layers)] for _ in range(self.num_mbs)]
        self.index_stream = torch.cuda.Stream()
        self.num_decoded_tokens = [0 for _ in range(num_layers)]
        self.batch_prefill = batch_prefill
        self.queries = torch.empty(
            (num_layers, self.batch_size, self.num_q_heads, MAX_QUERY_HISTORY, self.head_dim),
            dtype=torch.bfloat16,
            device="cuda"
        )
        self.prefix_kvcache_len = prefix_kvcache_len
        self.suffix_kvcache_len = suffix_kvcache_len
        self.retrieval_region_len = [0 for _ in range(num_layers)]

        self.prefix_k_cache = [torch.empty((
            self.batch_size, self.prefix_kvcache_len, self.num_kv_heads, self.head_dim
        ), dtype=torch.bfloat16, device="cuda") for _ in range(num_layers)]
        self.prefix_v_cache = [torch.empty((
            self.batch_size, self.prefix_kvcache_len, self.num_kv_heads, self.head_dim
        ), dtype=torch.bfloat16, device="cuda") for _ in range(num_layers)]
        self.suffix_k_cache = [torch.empty((
            self.batch_size, self.suffix_kvcache_len, self.num_kv_heads, self.head_dim
        ), dtype=torch.bfloat16, device="cuda") for _ in range(num_layers)]
        self.suffix_v_cache = [torch.empty((
            self.batch_size, self.suffix_kvcache_len, self.num_kv_heads, self.head_dim
        ), dtype=torch.bfloat16, device="cuda") for _ in range(num_layers)]
        # ring cache
        self.suffix_cache_start = [0 for _ in range(num_layers)]

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
                    r_sq=self.r_sq[layer_idx],
                    name=f"m{mb:02d}l{layer_idx:03d}"
                )
        
        # initialize performance metrics
        init_metrics()

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
        ep_ids, ep_dists = self.index_layers[mb_idx][layer_idx].query(k, ef=self.num_seeds, kkdist=True).collect()     # index layer lookup

        # 2. insert into data layer
        # since numpy does not support bfloat16, we use uint16 to store bfloat16 data, and pass to data layer
        k_cpu = k.view(dtype=torch.uint16).cpu().numpy()
        v_cpu = v.view(dtype=torch.uint16).cpu().numpy()
        ep_ids_cpu = ep_ids.cpu().numpy()
        ep_dists_cpu = ep_dists.cpu().numpy()
        insert_handle = data_layer.insert(k_cpu, v_cpu, ep_dists_cpu, ep_ids_cpu)
        insert_handle.wait()   # ensure CPU index updated before proceeding

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
        layer_idx
    ):
        seq_len = key_states.size(1)

        # estimate r_sq based on prefill result
        r_sq = torch.einsum("bshd,bshd->bsh", key_states, key_states).max(dim=1).values.max(dim=0).values.tolist()
        self.r_sq[layer_idx] = r_sq
        for mb in range(self.num_mbs):
            self.index_layers[mb][layer_idx].set_r_sq(r_sq)
            self.data_layers[mb][layer_idx].update_r_sq(r_sq)

        # save prefix region to GPU
        prefix_len = min(self.prefix_kvcache_len, seq_len)
        self.prefix_k_cache[layer_idx][:, :prefix_len, :, :].copy_(key_states[:, :prefix_len, :, :])
        self.prefix_v_cache[layer_idx][:, :prefix_len, :, :].copy_(value_states[:, :prefix_len, :, :])
        # FIXME: prefix kvcache will never be updated after prefill phase
        self.prefix_kvcache_len = prefix_len

        # save suffix region to GPU
        suffix_len = min(self.suffix_kvcache_len, seq_len - prefix_len)
        if suffix_len > 0:
            self.suffix_k_cache[layer_idx][:, :suffix_len, :, :].copy_(key_states[:, -suffix_len:, :, :])
            self.suffix_v_cache[layer_idx][:, :suffix_len, :, :].copy_(value_states[:, -suffix_len:, :, :])
        # FIXME: suffix kvcache size will never be updated at the decoding phase
        self.suffix_kvcache_len = suffix_len
        
        # save retrieval region to GPU/CPU hybird ANNS index
        retrieval_key_states = key_states[:, prefix_len:seq_len - suffix_len, :, :]
        retrieval_value_states = value_states[:, prefix_len:seq_len - suffix_len, :, :]
        self.retrieval_region_len[layer_idx] = retrieval_key_states.size(1)
        if self.batch_prefill:
            self.batch_prefill_update_index(
                key_states=retrieval_key_states,
                value_states=retrieval_value_states,
                layer_idx=layer_idx
            )
        else:
            self.naive_prefill_update_index(
                key_states=retrieval_key_states,
                value_states=retrieval_value_states,
                layer_idx=layer_idx
            )

        return key_states, value_states


    def naive_prefill_update_index(
        self, 
        key_states: torch.Tensor,     # (bsz, seq_len, num_kv_heads, head_dim)
        value_states: torch.Tensor,   # (bsz, seq_len, num_kv_heads, head_dim)
        layer_idx
    ):
        keys = key_states.view(self.num_mbs, self.mbsize, key_states.size(1), key_states.size(2), key_states.size(3))
        values = value_states.view(self.num_mbs, self.mbsize, value_states.size(1), value_states.size(2), value_states.size(3))
        seq_len = key_states.size(1)

        for mb in range(self.num_mbs):
            for t in range(seq_len):
                k = keys[mb, :, t, :, :]    # (mbsize, num_kv_heads, head_dim)
                v = values[mb, :, t, :, :]  # (mbsize, num_kv_heads, head_dim)
                node_id = t
                
                # insert into cache
                self.cache_insert(
                    k=k.unsqueeze(1), 
                    v=v.unsqueeze(1), 
                    layer_idx=layer_idx,
                    mb_idx=mb,
                    node_id=node_id
                )


    def batch_prefill_update_index(
        self, 
        key_states: torch.Tensor,     # (bsz, seq_len, num_kv_heads, head_dim)
        value_states: torch.Tensor,   # (bsz, seq_len, num_kv_heads, head_dim)
        layer_idx
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
        prefill_handles = []
        for mb in range(self.num_mbs):
            data_layer = self.data_layers[mb][layer_idx]
            k_cpu = keys[mb].contiguous().view(dtype=torch.uint16).cpu().numpy()        # (mbsize, seq_len, num_kv_heads, head_dim)
            v_cpu = values[mb].contiguous().view(dtype=torch.uint16).cpu().numpy()      # (mbsize, seq_len, num_kv_heads, head_dim)
            neighbours_cpu = neighbours[mb, :, :, :, :]                                 # (mbsize, num_kv_heads, seq_len, deg)
            handle = data_layer.prefill(k_cpu, v_cpu, neighbours_cpu)    # async
            prefill_handles.append(handle)

        for handle in prefill_handles:
            handle.wait()

        # build index layer from sample
        # sample index_layer_prob fraction of keys
        sample_num = int(seq_len * self.index_layer_prob)
        for mb in range(self.num_mbs):
            sampled_indices = torch.randperm(seq_len)[:sample_num]
            k_sampled = keys[mb, :, sampled_indices, :, :]  # (mbsize, sample_num, num_kv_heads, head_dim)
            node_ids = sampled_indices.unsqueeze(0).repeat(self.mbsize, 1).to(k_sampled.device)  # (mbsize, sample_num)
            self.index_layers[mb][layer_idx].insert(k_sampled, node_ids)
        
    def decode_update_kv_cache(self,
        key_states,     # (bs, length(=1), num_kv_heads, dim)
        value_states,   # (bs, length(=1), num_kv_heads, dim)
        layer_idx
    ):
        assert key_states.size(1) == 1, "decode_update_kv_cache only supports length=1"

        if self.suffix_kvcache_len > 0:
            # move head pointer of the circular suffix cache
            insert_pos = self.suffix_cache_start[layer_idx]
            self.suffix_cache_start[layer_idx] = (insert_pos + 1) % self.suffix_kvcache_len

            # kick the origin KV at the insert position to index
            for mb in range(self.num_mbs):
                k_old = self.suffix_k_cache[layer_idx][mb * self.mbsize:(mb + 1) * self.mbsize, insert_pos, :, :]  # (mbsize, num_kv_heads, head_dim)
                v_old = self.suffix_v_cache[layer_idx][mb * self.mbsize:(mb + 1) * self.mbsize, insert_pos, :, :]  # (mbsize, num_kv_heads, head_dim)
                node_id = self.retrieval_region_len[layer_idx]
                # insert into cache
                self.cache_insert(
                    k=k_old.unsqueeze(1), 
                    v=v_old.unsqueeze(1), 
                    layer_idx=layer_idx,
                    mb_idx=mb,
                    node_id=node_id
                )
            self.retrieval_region_len[layer_idx] += 1

            # insert into suffix cache
            self.suffix_k_cache[layer_idx][:, insert_pos, :, :].copy_(key_states[:, 0, :, :])
            self.suffix_v_cache[layer_idx][:, insert_pos, :, :].copy_(value_states[:, 0, :, :])
        else:
            # directly insert into index layer
            for mb in range(self.num_mbs):
                k = key_states[mb * self.mbsize:(mb + 1) * self.mbsize, 0, :, :]  # (mbsize, num_kv_heads, head_dim)
                v = value_states[mb * self.mbsize:(mb + 1) * self.mbsize, 0, :, :]  # (mbsize, num_kv_heads, head_dim)
                node_id = self.retrieval_region_len[layer_idx]
                # insert into cache
                self.cache_insert(
                    k=k.unsqueeze(1), 
                    v=v.unsqueeze(1), 
                    layer_idx=layer_idx,
                    mb_idx=mb,
                    node_id=node_id
                )
            self.retrieval_region_len[layer_idx] += 1

        return key_states, value_states
    
    def compute(
        self,
        queries,        # (bsz, 1, num_q_heads, head_dim)
        layer_idx
    ):
        if self.num_decoded_tokens[layer_idx] < MAX_QUERY_HISTORY:
            self.queries[layer_idx, :, :, self.num_decoded_tokens[layer_idx], :].copy_(queries.view(self.batch_size, self.num_q_heads, self.head_dim))
        self.num_decoded_tokens[layer_idx] += 1

        # 1. divide into minibatches
        query = queries.view(self.num_mbs, self.mbsize, self.num_q_heads, self.head_dim)

        # 2. issue GPU index layer queries in a stream
        query_handles = []
        with torch.cuda.stream(self.index_stream):
            for mb in range(self.num_mbs):
                q_mini = query[mb, :, :, :]   # (mbsize, num_q_heads, head_dim)
                handle = self.index_layers[mb][layer_idx].query(q_mini, ef=self.num_seeds, kkdist=False)
                query_handles.append(handle)

        total_bsz = queries.size(0)
        device = queries.device
        retrieval_len = min(self.retrieval_region_len[layer_idx], self.retrieval_budget)
        total_cache_len = self.prefix_kvcache_len + retrieval_len + self.suffix_kvcache_len
        k_all = torch.empty(
            (total_bsz, total_cache_len, self.num_kv_heads, self.head_dim),
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
            if retrieval_len > 0:
                # retrieved KV: (mbsize, retrieval_len, num_kv_heads, head_dim)
                with Timer("data_layer_query", logger=None):
                    k_retrieved_cpu, v_retrieved_cpu = data_layer.query(q_cpu, retrieval_len, ep_dists_cpu, ep_ids_cpu).collect()

                # copy retrieved KV to GPU
                k_retrieved = torch.from_numpy(k_retrieved_cpu).view(dtype=torch.bfloat16).to(device=device).view(self.mbsize, retrieval_len, self.num_kv_heads, self.head_dim)
                v_retrieved = torch.from_numpy(v_retrieved_cpu).view(dtype=torch.bfloat16).to(device=device).view(self.mbsize, retrieval_len, self.num_kv_heads, self.head_dim)
            else:
                k_retrieved = torch.empty((self.mbsize, 0, self.num_kv_heads, self.head_dim), device=device, dtype=torch.bfloat16)
                v_retrieved = torch.empty_like(k_retrieved)

            # combine with prefix and suffix kv cache
            k_combined = torch.cat([
                self.prefix_k_cache[layer_idx][mb * self.mbsize:(mb + 1) * self.mbsize, :self.prefix_kvcache_len, :, :],
                k_retrieved,
                self.suffix_k_cache[layer_idx][mb * self.mbsize:(mb + 1) * self.mbsize, :self.suffix_kvcache_len, :, :]
            ], dim=1)   # (mbsize, total_kvcache_len, num_kv_heads, head_dim)
            v_combined = torch.cat([
                self.prefix_v_cache[layer_idx][mb * self.mbsize:(mb + 1) * self.mbsize, :self.prefix_kvcache_len, :, :],
                v_retrieved,
                self.suffix_v_cache[layer_idx][mb * self.mbsize:(mb + 1) * self.mbsize, :self.suffix_kvcache_len, :, :]
            ], dim=1)   # (mbsize, total_kvcache_len, num_kv_heads, head_dim)

            start = mb * self.mbsize
            end = start + self.mbsize
            k_all[start:end].copy_(k_combined)
            v_all[start:end].copy_(v_combined)

        # 4. use flash attention with retrieved KV cache
        attn_out = flash_attn_with_kvcache(
            q=queries, 
            k_cache=k_all, 
            v_cache=v_all,
            causal=True
        )

        return attn_out

    def save_prefill_cache(self, prefill_cache_path):
        print("Saving prefill cache...")

        path = os.path.join(prefill_cache_path, "metadata.txt")
        metadata = {
            "r_sq": self.r_sq,
        }
        with open(path, 'w') as f:
            json.dump(metadata, f)

        # save prefix cache
        for layer_idx in range(self.num_layers):
            layer_path = os.path.join(prefill_cache_path, f"p_l_{layer_idx:03d}")
            os.makedirs(layer_path, exist_ok=False)
            torch.save(self.prefix_k_cache[layer_idx][:, :self.prefix_kvcache_len, :, :], os.path.join(layer_path, "k.pt"))
            torch.save(self.prefix_v_cache[layer_idx][:, :self.prefix_kvcache_len, :, :], os.path.join(layer_path, "v.pt"))

        # save suffix cache
        for layer_idx in range(self.num_layers):
            layer_path = os.path.join(prefill_cache_path, f"s_l_{layer_idx:03d}")
            os.makedirs(layer_path, exist_ok=False)
            torch.save(self.suffix_k_cache[layer_idx][:, :self.suffix_kvcache_len, :, :], os.path.join(layer_path, "k.pt"))
            torch.save(self.suffix_v_cache[layer_idx][:, :self.suffix_kvcache_len, :, :], os.path.join(layer_path, "v.pt"))

        # data layer
        for mb in range(self.num_mbs):
            for layer_idx in range(self.num_layers):
                data_layer = self.data_layers[mb][layer_idx]
                layer_path = os.path.join(prefill_cache_path, f"d_mb_{mb:02d}_l_{layer_idx:03d}")
                os.makedirs(layer_path, exist_ok=False)
                data_layer.wait()   # wait for previous prefill done
                data_layer.save(layer_path)

        # index layer
        for mb in range(self.num_mbs):
            for layer_idx in range(self.num_layers):
                index_layer = self.index_layers[mb][layer_idx]
                layer_path = os.path.join(prefill_cache_path, f"i_mb_{mb:02d}_l_{layer_idx:03d}")
                os.makedirs(layer_path, exist_ok=False)
                index_layer.save(layer_path)

        # wait for data layer to be done
        for mb in range(self.num_mbs):
            for layer_idx in range(self.num_layers):
                data_layer = self.data_layers[mb][layer_idx]
                data_layer.wait()
        
        print("Prefill cache saved.")

    def load_prefill_cache(self, inputs_ids, prefill_cache_path):
        print("Loading prefill cache...")

        ctx_len = inputs_ids.size(1)

        path = os.path.join(prefill_cache_path, "metadata.txt")
        with open(path, 'r') as f:
            metadata = json.load(f)
        self.r_sq = metadata["r_sq"]
        for mb in range(self.num_mbs):
            for layer_idx in range(self.num_layers):
                self.index_layers[mb][layer_idx].set_r_sq(self.r_sq[layer_idx])
                self.data_layers[mb][layer_idx].update_r_sq(self.r_sq[layer_idx])
            
        for layer_idx in range(self.num_layers):
            prefix_layer_path = os.path.join(prefill_cache_path, f"p_l_{layer_idx:03d}")
            suffix_layer_path = os.path.join(prefill_cache_path, f"s_l_{layer_idx:03d}")
            self.prefix_k_cache[layer_idx] = torch.load(os.path.join(prefix_layer_path, "k.pt"))
            self.prefix_v_cache[layer_idx] = torch.load(os.path.join(prefix_layer_path, "v.pt"))
            self.suffix_k_cache[layer_idx] = torch.load(os.path.join(suffix_layer_path, "k.pt"))
            self.suffix_v_cache[layer_idx] = torch.load(os.path.join(suffix_layer_path, "v.pt"))
            self.retrieval_region_len[layer_idx] = ctx_len - self.prefix_k_cache[layer_idx].shape[1] - self.suffix_k_cache[layer_idx].shape[1]
        self.prefix_kvcache_len = self.prefix_k_cache[0].shape[1]
        self.suffix_kvcache_len = self.suffix_k_cache[0].shape[1]

        # data layer
        for mb in range(self.num_mbs):
            for layer_idx in range(self.num_layers):
                data_layer = self.data_layers[mb][layer_idx]
                layer_path = os.path.join(prefill_cache_path, f"d_mb_{mb:02d}_l_{layer_idx:03d}")
                data_layer.load(layer_path, ctx_len)

        # index layer
        for mb in range(self.num_mbs):
            for layer_idx in range(self.num_layers):
                index_layer = self.index_layers[mb][layer_idx]
                layer_path = os.path.join(prefill_cache_path, f"i_mb_{mb:02d}_l_{layer_idx:03d}")
                index_layer.load(layer_path)

        # wait for data layer to be done
        for mb in range(self.num_mbs):
            for layer_idx in range(self.num_layers):
                data_layer = self.data_layers[mb][layer_idx]
                data_layer.wait()

        print("Prefill cache loaded.")

    def save_keys(self, dir: str):
        print("Saving keys...")
        for mb in range(self.num_mbs):
            for layer_idx in tqdm(range(self.num_layers)):
                keys = self.data_layers[mb][layer_idx].get_keys()  # (bsz, seq_len, head_dim)
                np.save(os.path.join(dir, f"k_m{mb:02d}l{layer_idx:03d}.npy"), keys)

    def save_queries(self, dir: str):
        print("Saving queries...")
        queries = self.queries[:, :, :, :self.num_decoded_tokens[0], :].float().cpu().numpy()  # (bsz, num_q_heads, num_decoded_tokens, head_dim)
        np.save(os.path.join(dir, f"q.npy"), queries)

    def cached_seq_len(self, layer_idx):
        return self.prefix_kvcache_len + self.retrieval_region_len[layer_idx] + self.suffix_kvcache_len
