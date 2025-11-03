"""
DiffCache: Differential KVCache Indexing for Fast, Memory-efficient LLM Inference

Flat GPU-Index Layer
"""


from dataclasses import dataclass
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
    def __init__(self, bsz: int, max_num_seeds: int, num_kv_heads: int, head_dim: int, r_sq: float, group_query=True):
        self.bsz, self.max_num_seeds, self.num_kv_heads, self.head_dim = bsz, max_num_seeds, num_kv_heads, head_dim
        self.k = torch.empty((bsz, max_num_seeds, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
        self.node_ids = torch.empty((bsz, max_num_seeds), device='cuda', dtype=torch.int64)
        self.group_query = group_query
        self.num_seeds = 0
        self.r_sq = r_sq

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
    # q: (bsz, num_query_heads, head_dim)
    # return: (bsz, num_kv_heads, ef) if group_query else (bsz, num_query_heads, ef) 
    #         (indices: uint32, scores: float32)
    def query(self, q: torch.Tensor, ef, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        
        bsz, num_query_heads, head_dim = q.shape
        assert bsz == self.bsz, "Batch size of query must match index layer"
        assert head_dim == self.head_dim, "Head dimension of query must match index layer"
        assert num_query_heads % self.num_kv_heads == 0, "num_query_heads must be multiple of num_kv_heads"

        group_size = num_query_heads // self.num_kv_heads
        ef = min(int(ef), self.num_seeds)

        # num_query_heads -> num_kv_heads x group_size
        q_grouped = q.reshape(bsz, self.num_kv_heads, group_size, head_dim)

        # q:    (bsz, num_kv_heads, group_size, head_dim)
        # k:    (bsz, num_seeds, num_kv_heads, head_dim)
        with torch.cuda.stream(stream):
            k = self.k[:, :self.num_seeds, :, :]  # (bsz, num_seeds, num_kv_heads, head_dim)
            dots = torch.einsum('b h g d, b s h d -> b s h g', q_grouped, k)                       # (bsz, num_seeds, num_kv_heads, group_size)
            dists = torch.sum(q_grouped * q_grouped, dim=-1).unsqueeze(1) + self.r_sq - 2 * dots   # (bsz, num_seeds, num_kv_heads, group_size)

            if self.group_query:
                sim = dists.min(dim=-1).values  # (bsz, num_seeds, num_kv_heads), groupwise max
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
