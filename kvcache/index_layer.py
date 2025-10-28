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
    top_idx: torch.Tensor      # (bsz, num_kv_heads, ef), on GPU
    top_scores: torch.Tensor   # (bsz, num_kv_heads, ef), on GPU
    event: torch.cuda.Event    # signals completion of topk on the submit stream
    stream: torch.cuda.Stream  # the stream where ops were enqueued

    def ready(self) -> bool:
        return self.event.query()

    def collect(self):
        self.event.synchronize()
        return self.top_idx, self.top_scores


class IndexLayer:
    def __init__(self, bsz: int, max_index_size: int, num_kv_heads: int, head_dim: int, group_query=True):
        self.bsz, self.max_index_size, self.num_kv_heads, self.head_dim = bsz, max_index_size, num_kv_heads, head_dim
        self.k = torch.empty((bsz, max_index_size, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
        self.v = torch.empty((bsz, max_index_size, num_kv_heads, head_dim), device='cuda', dtype=torch.bfloat16)
        self.group_query = group_query
        self.index_size = 0

    # insert into index layer
    # k: (bsz, count, num_kv_heads, head_dim)
    # v: (bsz, count, num_kv_heads, head_dim)
    def insert(self, k: torch.Tensor, v: torch.Tensor):
        count = k.shape[1]
        assert (self.bsz, count, self.num_kv_heads, self.head_dim) == k.shape, "Key shape mismatch"
        assert v.shape == k.shape, "Value shape mismatch"
        if self.index_size + count > self.max_index_size:
            raise RuntimeError("IndexLayer is full; cannot insert more keys/values")
        self.k[:, self.index_size:self.index_size + count, :, :] = k
        self.v[:, self.index_size:self.index_size + count, :, :] = v
        self.index_size += count

    # query top-ef nearest neighbors
    # q: (bsz, num_query_heads, head_dim)
    # return: (bsz, num_kv_heads, ef) if group_query else (bsz, num_query_heads, ef)
    def query(self, q: torch.Tensor, ef, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        
        bsz, num_query_heads, head_dim = q.shape
        assert bsz == self.bsz, "Batch size of query must match index layer"
        assert head_dim == self.head_dim, "Head dimension of query must match index layer"
        assert num_query_heads % self.num_kv_heads == 0, "num_query_heads must be multiple of num_kv_heads"

        group_size = num_query_heads // self.num_kv_heads
        ef = min(int(ef), self.seq_len)

        # num_query_heads -> num_kv_heads x group_size
        q_grouped = q.reshape(bsz, self.num_kv_heads, group_size, head_dim)

        # q:    (bsz, num_kv_heads, group_size, head_dim)
        # k:    (bsz, index_size, num_kv_heads, head_dim)
        with torch.cuda.stream(stream):
            dots = torch.einsum('b h g d, b s h d -> b s h g', q_grouped, self.k) # (bsz, index_size, num_kv_heads, group_size)

            if self.group_query:
                sim = dots.max(dim=-1).values  # (bsz, index_size, num_kv_heads), groupwise max
            else:
                sim = dots.reshape(bsz, self.index_size, num_query_heads)  # (bsz, index_size, num_query_heads)

            # Top-k along seq dimension for each kv head
            top_scores, top_idx = torch.topk(sim, k=ef, dim=1, largest=True, sorted=True)  # (bsz, ef, num_*_heads)

            # Reorder to (bsz, num_*_heads, ef)
            top_idx = top_idx.transpose(1, 2).contiguous()
            top_scores = top_scores.transpose(1, 2).contiguous()

            evt = torch.cuda.Event(enable_timing=False)
            evt.record(stream)

        return QueryHandle(
            top_idx=top_idx,
            top_scores=top_scores,
            event=evt,
            stream=stream,
        )
