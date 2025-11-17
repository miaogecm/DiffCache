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
    def __init__(self, bsz: int, max_num_seeds: int, num_kv_heads: int, head_dim: int, r_sq: List[float]):
        self.bsz, self.max_num_seeds, self.num_kv_heads, self.head_dim = bsz, max_num_seeds, num_kv_heads, head_dim
        self.k = torch.empty((bsz, max_num_seeds, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
        self.node_ids = torch.empty((bsz, max_num_seeds), device='cuda', dtype=torch.int64)
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

        self.k[:, self.num_seeds:self.num_seeds + count, :, :] = k
        self.node_ids[:, self.num_seeds:self.num_seeds + count] = node_ids
        self.num_seeds += count

    # query top-ef nearest neighbors
    # q: (bsz, num_query_heads, head_dim) if not kkdist
    #.   (bsz, num_kv_heads, head_dim).   if kkdist
    # return: (bsz, num_kv_heads, ef)
    #         (indices: uint32, scores: float32)
    def query(self, q: torch.Tensor, ef, stream=None, kkdist=False):
        if stream is None:
            stream = torch.cuda.current_stream()
        
        bsz, num_heads, head_dim = q.shape
        assert bsz == self.bsz, "Batch size of query must match index layer"
        assert head_dim == self.head_dim, "Head dimension of query must match index layer"
        if kkdist:
            assert num_heads == self.num_kv_heads, "When kkdist is True, num_heads must equal num_kv_heads"
        else:
            assert num_heads % self.num_kv_heads == 0, "num_heads must be multiple of num_kv_heads"

        ef = min(int(ef), self.num_seeds)

        k = self.k[:, :self.num_seeds, :, :]  # (bsz, num_seeds, num_kv_heads, head_dim)
        if not kkdist:
            q = q.reshape(bsz, self.num_kv_heads, num_heads // self.num_kv_heads, head_dim).mean(dim=2)  # (bsz, num_kv_heads, head_dim)

        # q: (bsz, num_kv_heads, head_dim)
        # k: (bsz, num_seeds, num_kv_heads, head_dim)
        with torch.cuda.stream(stream):
            # The distance should be consistent with Index::distance function in nsw.rs
            dists = torch.einsum('bhd,bshd->bsh', q, k)  # (bsz, num_seeds, num_kv_heads)
            if kkdist:
                # use |q - k|^2 + (sqrt(r^2 - q^2) - sqrt(r^2 - k^2))^2
                # = 2r^2 - 2q.k - 2sqrt(r^2 - |q|^2)sqrt(r^2 - |k|^2)
                q_sq = torch.sum(q * q, dim=-1).unsqueeze(1)  # (bsz, 1, num_kv_heads)
                k_sq = torch.sum(k * k, dim=-1)               # (bsz, num_seeds, num_kv_heads)
                r_sq = self.r_sq.unsqueeze(0).unsqueeze(0)    # (1, 1, num_kv_heads)
                s1 = torch.sqrt(torch.clamp(r_sq - q_sq, min=0.0))
                s2 = torch.sqrt(torch.clamp(r_sq - k_sq, min=0.0))
                dists = r_sq - dists - s1 * s2                # (bsz, num_seeds, num_kv_heads)
            else:
                # FIXME: we assume that 65536.0 - q.k >= 0
                dists = 65536.0 - dists                       # (bsz, num_seeds, num_kv_heads)

            # Top-k along seq dimension for each kv head – we want the smallest distances
            top_dists, top_idx = torch.topk(dists, k=ef, dim=1, largest=False, sorted=True)  # (bsz, ef, num_kv_heads)

            # Reorder to (bsz, num_kv_heads, ef) and gather node IDs
            top_dists = top_dists.transpose(1, 2).to(torch.float32).contiguous()
            top_idx = top_idx.transpose(1, 2)
            seed_node_ids = self.node_ids[:, :self.num_seeds]
            seed_node_ids = seed_node_ids.unsqueeze(1).expand(-1, self.num_kv_heads, -1)  # (bsz, num_kv_heads, num_seeds)
            top_ids = torch.gather(seed_node_ids, 2, top_idx)
            top_ids = top_ids.to(torch.uint32).contiguous()

            evt = torch.cuda.Event(enable_timing=False)
            evt.record(stream)

        return QueryHandle(
            top_ids=top_ids,
            top_scores=top_dists,
            event=evt,
            stream=stream,
        )

    def save(self, prefix: str):
        torch.save(self.k[:, :self.num_seeds, :, :].cpu(), f"{prefix}_k.pt")
        torch.save(self.node_ids[:, :self.num_seeds].cpu(), f"{prefix}_node_ids.pt")

    def load(self, prefix: str):
        assert self.num_seeds == 0, "Can only load into empty IndexLayer"
        k = torch.load(f"{prefix}_k.pt").to(self.k.device)
        node_ids = torch.load(f"{prefix}_node_ids.pt").to(self.node_ids.device)
        count = k.shape[1]
        self.k[:, :count, :, :] = k
        self.node_ids[:, :count] = node_ids
        self.num_seeds = count
