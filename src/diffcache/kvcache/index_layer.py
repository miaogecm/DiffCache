"""
DiffCache: Differential KVCache Indexing for Fast, Memory-efficient LLM Inference

Flat GPU-Index Layer
"""


from dataclasses import dataclass
from typing import List
import torch
import triton
import triton.language as tl


@dataclass
class QueryHandle:
    top_ids: torch.Tensor      # (bsz, num_kv_heads, ef), on GPU
    top_scores: torch.Tensor   # (bsz, num_kv_heads, ef), on GPU
    event: torch.cuda.Event    # signals completion of topk on the submit stream
    stream: torch.cuda.Stream  # the stream where ops were enqueued

    def ready(self) -> bool:
        return self.event.query()

    def collect(self):
        self.event.synchronize()
        return self.top_ids, self.top_scores


class IndexLayer:
    def __init__(self, bsz: int, max_num_seeds: int, num_kv_heads: int, head_dim: int, r_sq: List[float], group_query=True):
        self.bsz, self.max_num_seeds, self.num_kv_heads, self.head_dim = bsz, max_num_seeds, num_kv_heads, head_dim
        self.k = torch.empty((bsz, max_num_seeds, num_kv_heads, head_dim + 1), device='cuda', dtype=torch.bfloat16) # + L2 lifting dim
        self.node_ids = torch.empty((bsz, max_num_seeds), device='cuda', dtype=torch.int64)
        self.group_query = group_query
        self.num_seeds = 0
        self.r_sq = torch.tensor(r_sq, device='cuda', dtype=torch.float32)  # (num_kv_heads,)

    # insert into index layer
    # k: (bsz, count, num_kv_heads, head_dim)
    # node_ids: (bsz, count)
    def insert(self, k: torch.Tensor, node_ids: torch.Tensor):
        count = k.shape[1]
        assert (self.bsz, count, self.num_kv_heads, self.head_dim) == k.shape, "Key shape mismatch"
        assert (self.bsz, count) == node_ids.shape, "Node id shape mismatch"
        if self.num_seeds + count > self.max_num_seeds:
            raise RuntimeError("IndexLayer is full; cannot insert more keys/values")
        
        # L2 lifting
        k_l2_sq = torch.sum(k * k, dim=-1, keepdim=True)   # (bsz, count, num_kv_heads, 1)
        r_sq = self.r_sq.view(1, 1, self.num_kv_heads, 1)  # (1, 1, num_kv_heads, 1)
        k_lift = torch.sqrt(torch.clamp(r_sq - k_l2_sq, min=0.0))  # (bsz, count, num_kv_heads, 1)
        k = torch.cat([k, k_lift], dim=-1)                 # (bsz, count, num_kv_heads, head_dim + 1)

        self.k[:, self.num_seeds:self.num_seeds + count, :, :] = k
        self.node_ids[:, self.num_seeds:self.num_seeds + count] = node_ids
        self.num_seeds += count

    # query top-ef nearest neighbors
    # q: (bsz, num_query_heads, head_dim)
    # return: (bsz, num_kv_heads, ef) if group_query else (bsz, num_query_heads, ef) 
    #         (indices: uint32, scores: float32)
    def query(self, q: torch.Tensor, ef, stream=None, kkdist=False):
        if stream is None:
            stream = torch.cuda.current_stream()
        
        bsz, num_query_heads, head_dim = q.shape
        assert bsz == self.bsz, "Batch size of query must match index layer"
        assert head_dim == self.head_dim, "Head dimension of query must match index layer"
        assert num_query_heads % self.num_kv_heads == 0, "num_query_heads must be multiple of num_kv_heads"
        if not self.group_query or kkdist:
            assert num_query_heads == self.num_kv_heads, "When group_query is False or kkdist is True, num_query_heads must equal num_kv_heads"

        group_size = num_query_heads // self.num_kv_heads
        ef = min(int(ef), self.num_seeds)

        # num_query_heads -> num_kv_heads x group_size
        q_grouped = q.reshape(bsz, self.num_kv_heads, group_size, head_dim)

        # L2 lifting (q_grouped -> (bsz, num_kv_heads, group_size, head_dim + 1))
        if not kkdist: # add zero at last dim
            z = torch.zeros((bsz, self.num_kv_heads, group_size, 1), device=q.device, dtype=q.dtype)
            q_grouped = torch.cat([q_grouped, z], dim=-1)
        else:
            q_l2_sq = torch.sum(q_grouped * q_grouped, dim=-1, keepdim=True)   # (bsz, num_kv_heads, group_size, 1)
            r_sq = self.r_sq.view(1, self.num_kv_heads, 1, 1)                    # (1, num_kv_heads, 1, 1)
            q_lift = torch.sqrt(torch.clamp(r_sq - q_l2_sq, min=0.0))           # (bsz, num_kv_heads, group_size, 1)
            q_grouped = torch.cat([q_grouped, q_lift], dim=-1)                  # (bsz, num_kv_heads, group_size, head_dim + 1)

        # q:    (bsz, num_kv_heads, group_size, head_dim + 1)
        # k:    (bsz, num_seeds, num_kv_heads, head_dim + 1)
        with torch.cuda.stream(stream):
            r_sq = self.r_sq.view(1, 1, self.num_kv_heads, 1)  # (1, 1, num_kv_heads, 1)

            k = self.k[:, :self.num_seeds, :, :]  # (bsz, num_seeds, num_kv_heads, head_dim + 1)
            # Compute L2 distance via (q - k)^2
            sub = k.unsqueeze(3) - q_grouped.unsqueeze(1)  # (bsz, num_seeds, num_kv_heads, group_size, head_dim + 1)
            dists = torch.sum(sub * sub, dim=-1)           # (bsz, num_seeds, num_kv_heads, group_size)

            if self.group_query and not kkdist:
                sim = dists.min(dim=-1).values  # (bsz, num_seeds, num_kv_heads), groupwise min
            else:
                sim = dists.reshape(bsz, self.num_seeds, num_query_heads)  # (bsz, num_seeds, num_query_heads)

            # Top-k along seq dimension for each kv head
            top_scores, top_idx = torch.topk(sim, k=ef, dim=1, largest=False, sorted=True)  # (bsz, ef, num_*_heads)

            # Reorder to (bsz, num_*_heads, ef)
            top_idx = top_idx.transpose(1, 2).contiguous()
            top_scores = top_scores.transpose(1, 2).to(torch.float32).contiguous()

            if ef == 0:
                top_ids = self.node_ids.new_empty((bsz, top_idx.shape[1], 0), dtype=torch.int64)
            else:
                seed_node_ids = self.node_ids[:, :self.num_seeds]
                seed_node_ids = seed_node_ids.unsqueeze(1).expand(-1, top_idx.shape[1], -1)
                top_ids = torch.gather(seed_node_ids, 2, top_idx)
            top_ids = top_ids.to(torch.uint32).contiguous()

            evt = torch.cuda.Event(enable_timing=False)
            evt.record(stream)

        return QueryHandle(
            top_ids=top_ids,
            top_scores=top_scores,
            event=evt,
            stream=stream,
        )
