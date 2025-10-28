"""
DiffCache: Differential KVCache Indexing for Fast, Memory-efficient LLM Inference

Flat GPU-Index Layer
"""


from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@triton.jit
def _groupmax_dot_kernel(
    Q, K, OUT,
    bsz, seq_len, num_kv_heads,
    stride_q_b, stride_q_h, stride_q_g, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_o_b, stride_o_s, stride_o_h,
    scale,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len

    max_sim = tl.full([BLOCK_S], -float("inf"), tl.float32)

    for g in range(GROUP):
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        for d0 in range(0, HEAD_DIM, BLOCK_D):
            d_off = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_off < HEAD_DIM

            q_ptrs = Q + pid_b*stride_q_b + pid_h*stride_q_h + g*stride_q_g + d_off*stride_q_d
            q_vec = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

            k_ptrs = K + pid_b*stride_k_b + s_off[:, None]*stride_k_s + pid_h*stride_k_h + d_off[None, :]*stride_k_d
            k_blk  = tl.load(k_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            acc += tl.sum(k_blk * q_vec[None, :], axis=1)

        acc = acc * scale
        max_sim = tl.maximum(max_sim, acc)

    out_ptrs = OUT + pid_b*stride_o_b + s_off*stride_o_s + pid_h*stride_o_h
    tl.store(out_ptrs, max_sim, mask=s_mask)


def _cdiv(a, b): return (a + b - 1) // b


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

    # TODO: optimize query with fused triton kernel
    def query_triton(
        self, q, ef: int, *,
        scale: float = 1.0,
        stream: torch.cuda.Stream | None = None,
        BLOCK_S: int = 128,
        BLOCK_D: int = 64,
        num_warps: int = 4,
        num_stages: int = 2,
    ) -> QueryHandle:
        assert q.device.type == 'cuda', "q must be on GPU"
        bsz, num_query_heads, head_dim = q.size()
        assert bsz == self.bsz, "Batch size of query must match index layer"
        assert head_dim == self.head_dim, "Head dimension of query must match index layer"
        assert num_query_heads % self.num_kv_heads == 0, "num_query_heads must be multiple of num_kv_heads"

        group = num_query_heads // self.num_kv_heads
        ef = min(int(ef), self.seq_len)

        # Grouped layout (contiguous blocks): (bsz, num_kv_heads, group, head_dim)
        q_grouped = q.reshape(bsz, self.num_kv_heads, group, head_dim).contiguous()

        # Output similarity: (bsz, seq_len, num_kv_heads), keep fp32 for numerical stability
        sim = torch.empty((bsz, self.seq_len, self.num_kv_heads), device=q.device, dtype=torch.float32)

        # Select the submission stream (default to current)
        if stream is None:
            stream = torch.cuda.current_stream(q.device)

        # Enqueue all work on the chosen stream
        with torch.cuda.stream(stream):
            # Launch fused kernel over (bsz, num_kv_heads, ceil_div(seq_len, BLOCK_S))
            grid = (bsz, self.num_kv_heads, _cdiv(self.seq_len, BLOCK_S))
            q_strides = q_grouped.stride()
            k_strides = self.k.stride()
            o_strides = sim.stride()

            _groupmax_dot_kernel[grid](
                q_grouped, self.k, sim,
                bsz, self.seq_len, self.num_kv_heads,
                q_strides[0], q_strides[1], q_strides[2], q_strides[3],
                k_strides[0], k_strides[1], k_strides[2], k_strides[3],
                o_strides[0], o_strides[1], o_strides[2],
                scale,
                GROUP=group,
                HEAD_DIM=self.head_dim,
                BLOCK_S=BLOCK_S,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            # topk along seq_len (still GPU, still async on `stream`)
            top_scores, top_idx = torch.topk(sim, k=ef, dim=1, largest=True, sorted=True)
            # Re-layout to (bsz, num_kv_heads, ef) and cast scores to q.dtype if you prefer
            top_idx    = top_idx.transpose(1, 2).contiguous()
            top_scores = top_scores.transpose(1, 2).contiguous().to(q.dtype)

            # Record completion event after all ops in this stream
            evt = torch.cuda.Event(enable_timing=False)
            evt.record(stream)

        # Return tensors + event; no synchronization here.
        return QueryHandle(
            top_idx=top_idx,
            top_scores=top_scores,
            event=evt,
            stream=stream,
        )
