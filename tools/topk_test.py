#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from typing import Optional, Tuple, Dict, Any
import os
import numpy as np
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AttentionInterface,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod

# GPU-side CAGRA + HNSW (cuVS)
try:
    from cuvs.neighbors import cagra as cuvs_cagra
    from cuvs.neighbors import hnsw as cuvs_hnsw
except ImportError:
    cuvs_cagra = None
    cuvs_hnsw = None


def set_attn_impl(config, impl: str):
    # Small helper to be robust across transformer versions.
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = impl
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = impl


def make_gqa_topk_attention(
    top_k: int,
    base_impl_name: str,
    prefix_keep_len: int = 0,
    suffix_keep_len: int = 0,
    use_hnsw: bool = False,
    hnsw_M: int = 16,
    hnsw_ef: int = 64,
    hnsw_index_dir: str = "",
):
    """
    Create a custom attention function that:
    - Falls back to the original implementation (sdpa / flash / flex / eager)
      when q_len > 1 (prefill) or top_k <= 0.
    - In decode (q_len == 1), splits KV positions into:
        [ prefix | retrieval_region | suffix ]
      * prefix_keep_len  tokens from the start are always kept
      * suffix_keep_len  tokens from the end are always kept
      * only the middle retrieval_region is subject to top-k selection
      * top_k applies only inside that middle region
    - Top-k selection can be:
      * exact (torch.topk) when use_hnsw=False
      * approximate via CAGRA (GPU build) + cuVS HNSW (CPU search)
        when use_hnsw=True.
        In the HNSW path:
          * K vectors are lifted to an L2 space of dimension (D+1)
            so that we can use L2 distance for inner-product similarity.
          * CAGRA builds the graph on GPU.
          * cuVS converts the CAGRA index into an HNSW index.
          * We search that index on CPU via cuVS' hnsw.search().
    - After selecting indices, it delegates the actual attention computation
      on the sparse KV to the base implementation.
    """

    base_attn_fn = ALL_ATTENTION_FUNCTIONS.get(base_impl_name, None)
    repeat_kv = qwen2_mod.repeat_kv

    # Per-module in-memory state: HNSW indices etc.
    # Keyed by id(module) to avoid storing references as dict keys.
    module_state: Dict[int, Dict[str, Any]] = {}

    # --------------------------- HNSW build via CAGRA ---------------------------

    def build_hnsw_for_module_and_head(
        state: Dict[str, Any],
        key: torch.Tensor,  # [B, KV, kv_len, D]
        retrieval_start: int,
        retrieval_end: int,
        head_dim: int,
        module_tag: str,
    ):
        """
        Build (or load) cuVS HNSW indices for each KV head on CPU for the
        retrieval region positions [retrieval_start, retrieval_end).

        Pipeline:
        - Take key_mid in R^D (CUDA tensor)
        - Lift to R^{D+1} for L2-IP equivalence, all in torch on GPU
        - Build CAGRA index on GPU using cuVS (dataset is a CUDA array interface object)
        - Convert to cuVS HNSW index
        - Optionally save to disk via cuVS hnsw.save()
        - Keep the HNSW index objects in memory for CPU search
        """

        if cuvs_cagra is None or cuvs_hnsw is None:
            raise RuntimeError(
                "HNSW path requires cuVS to be installed "
                "(e.g. `pip install libcuvs-cu12` or similar)."
            )

        batch_size, num_kv_heads, kv_len, _ = key.shape
        assert batch_size == 1, "This script assumes batch_size == 1."

        retrieval_len = max(0, retrieval_end - retrieval_start)
        dim_l2 = head_dim + 1

        # No retrieval region: nothing to build
        if retrieval_len <= 0:
            state["hnsw_indices"] = None
            state["hnsw_dim"] = dim_l2
            state["retrieval_start"] = retrieval_start
            state["retrieval_end"] = retrieval_end
            return

        indices_per_head = []

        # Optional on-disk cache for this module
        cache_dir = None
        meta_path = None
        use_cache = bool(hnsw_index_dir)
        if use_cache:
            cache_dir = os.path.join(hnsw_index_dir, module_tag)
            meta_path = os.path.join(cache_dir, "meta.npz")

        # 1) Try to load cached cuVS HNSW indices if meta + all files exist
        cache_loaded = False
        if use_cache and os.path.exists(meta_path):
            meta = np.load(meta_path)
            cached_rs = int(meta["retrieval_start"])
            cached_re = int(meta["retrieval_end"])
            cached_dim_l2 = int(meta["dim_l2"])
            cached_num_kv_heads = int(meta["num_kv_heads"])

            if (
                cached_rs == retrieval_start
                and cached_re == retrieval_end
                and cached_dim_l2 == dim_l2
                and cached_num_kv_heads == num_kv_heads
            ):
                for kv_id in range(num_kv_heads):
                    idx_path = os.path.join(cache_dir, f"head_{kv_id}.bin")
                    if not os.path.exists(idx_path):
                        indices_per_head = []
                        break
                    cagra_params = cuvs_hnsw.IndexParams()
                    hindex = cuvs_hnsw.load(cagra_params, idx_path, dim_l2, np.float32, "sqeuclidean")
                    indices_per_head.append({"index": hindex})
                if len(indices_per_head) == num_kv_heads:
                    cache_loaded = True

        if cache_loaded:
            print(f"[INFO] Loaded cached cuVS HNSW indices for module {module_tag} from {cache_dir}.")
            state["hnsw_indices"] = indices_per_head
            state["hnsw_dim"] = dim_l2
            state["retrieval_start"] = retrieval_start
            state["retrieval_end"] = retrieval_end
            return

        # 2) No usable cache: build one CAGRA+HNSW index per KV head on GPU,
        #    then keep cuVS HNSW index objects in memory.
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)

        # Temp directory for non-cached runs (to hold cuVS-written HNSW files)
        tmpdir = None
        if not use_cache:
            import tempfile

            tmpdir = state.get("_tmpdir", None)
            if tmpdir is None:
                tmpdir = tempfile.TemporaryDirectory()
                state["_tmpdir"] = tmpdir

        for kv_id in range(num_kv_heads):
            # key_head: [kv_len, D], assumed on CUDA already
            key_head = key[0, kv_id, :, :]  # [kv_len, D]
            key_mid = key_head[retrieval_start:retrieval_end, :]  # [retrieval_len, D]

            # Ensure float32 on device
            key_mid_f = key_mid.detach().to(key.device, torch.float32)  # [N, D]

            # Lift to L2 sphere: [x; sqrt(R^2 - ||x||^2)]
            norms2 = (key_mid_f * key_mid_f).sum(dim=1, keepdim=True)  # [N, 1], on device
            R2 = norms2.max()  # scalar tensor on device
            extra = (R2 - norms2).clamp_min(0.0).sqrt()  # [N, 1], on device

            data_l2 = torch.cat([key_mid_f, extra], dim=1).contiguous()  # [N, D+1], CUDA

            # Build CAGRA index on GPU using torch CUDA tensor
            cagra_params = cuvs_cagra.IndexParams(
                metric="sqeuclidean",
                graph_degree=hnsw_M,
                intermediate_graph_degree=4 * hnsw_M,
            )
            cagra_index = cuvs_cagra.build(cagra_params, data_l2)

            # Convert to cuVS HNSW index
            hparams = cuvs_hnsw.IndexParams()  # hierarchy default is fine here
            hindex = cuvs_hnsw.from_cagra(hparams, cagra_index)

            # Optionally save cuVS HNSW index
            if use_cache:
                idx_path = os.path.join(cache_dir, f"head_{kv_id}.bin")
            else:
                idx_path = os.path.join(tmpdir.name, f"{module_tag}_head_{kv_id}.bin")

            cuvs_hnsw.save(idx_path, hindex)

            # Keep the cuVS HNSW index object in memory for search
            indices_per_head.append({"index": hindex})
        
        print(f"[INFO] Built cuVS HNSW indices for module {module_tag} (heads: {num_kv_heads}, "
              f"retrieval positions: {retrieval_start}-{retrieval_end}, dim_l2: {dim_l2}).")

        # Write meta for cache
        if use_cache:
            np.savez(
                meta_path,
                retrieval_start=retrieval_start,
                retrieval_end=retrieval_end,
                dim_l2=dim_l2,
                num_kv_heads=num_kv_heads,
            )

        state["hnsw_indices"] = indices_per_head
        state["hnsw_dim"] = dim_l2
        state["retrieval_start"] = retrieval_start
        state["retrieval_end"] = retrieval_end

    # --------------------------- HNSW query (CPU, via cuVS) ---------------------------

    def hnsw_topk_indices(
        state: Dict[str, Any],
        query_grouped: torch.Tensor,  # [B, KV, groups, 1, D]
        top_k: int,
    ) -> Optional[torch.Tensor]:
        """
        Use per-head cuVS HNSW indices (built via CAGRA + cuVS HNSW) to get
        approximate top-k indices in the retrieval region for each KV head.

        Returns:
            retrieval_idx: [B, num_kv_heads, k_eff] on CUDA (same device as query),
            or None if no retrieval region.
        """
        batch_size, num_kv_heads, num_groups, _, head_dim = query_grouped.shape
        assert batch_size == 1, "This script assumes batch_size == 1."

        indices_per_head = state.get("hnsw_indices", None)
        if indices_per_head is None:
            # No retrieval region
            return None

        retrieval_start = state["retrieval_start"]
        retrieval_end = state["retrieval_end"]
        retrieval_len = max(0, retrieval_end - retrieval_start)
        if retrieval_len <= 0:
            return None

        k_eff = min(top_k, retrieval_len)
        if k_eff <= 0:
            return None

        # Average queries over groups: [B, KV, 1, D]
        q_mean = query_grouped.mean(dim=2)  # [1, KV, 1, D]

        all_idx = []
        search_params = cuvs_hnsw.SearchParams(ef=hnsw_ef, num_threads=0)

        for kv_id in range(num_kv_heads):
            q_vec = q_mean[0, kv_id, 0, :]  # [D]
            q_np = q_vec.detach().to("cpu", torch.float32).numpy()  # [D]

            # Lift query to same L2 space as keys: [q; 0]
            extra = np.array([0.0], dtype=np.float32)
            q_ext = np.concatenate([q_np, extra], axis=0).astype("float32")  # [D+1]
            q_ext = q_ext.reshape(1, -1)  # [1, D+1], CPU numpy

            info = indices_per_head[kv_id]
            index = info["index"]  # cuVS HNSW Index
            _, neighbors = cuvs_hnsw.search(
                search_params,
                index,
                q_ext,
                k_eff,
            )  # neighbors: [1, k_eff], CPU numpy, 0..retrieval_len-1

            labels = neighbors[0].astype(np.int64)  # [k_eff]
            # Map back to global KV positions:
            labels = labels + retrieval_start
            all_idx.append(torch.from_numpy(labels).long())

        # Shape: [KV, k_eff]
        retrieval_idx = torch.stack(all_idx, dim=0)  # [KV, k_eff]
        retrieval_idx = retrieval_idx.unsqueeze(0)   # [1, KV, k_eff]
        return retrieval_idx.to(query_grouped.device)

    # --------------------------- Main attention wrapper ---------------------------

    def gqa_topk_attention(
        module: torch.nn.Module,
        query: torch.Tensor,        # [B, num_heads, q_len, D]
        key: torch.Tensor,          # [B, num_kv_heads, kv_len, D]
        value: torch.Tensor,        # [B, num_kv_heads, kv_len, D]
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Use the base implementation for prefill, disabled top_k, or if base is missing.
        if query.size(2) != 1 or top_k <= 0 or base_attn_fn is None:
            return base_attn_fn(
                module,
                query,
                key,
                value,
                attention_mask,
                **kwargs,
            )

        batch_size, num_heads, q_len, head_dim = query.shape
        _, num_kv_heads, kv_len, _ = key.shape

        # Qwen2 GQA structure: num_heads = num_kv_heads * num_key_value_groups
        num_groups = getattr(module, "num_key_value_groups", None)
        if num_groups is None or num_kv_heads * num_groups != num_heads:
            # Not GQA or mismatch: just fall back.
            return base_attn_fn(
                module,
                query,
                key,
                value,
                attention_mask,
                **kwargs,
            )

        # Group queries by KV head: [B, H, 1, D] -> [B, KV, groups, 1, D]
        query_grouped = query.view(
            batch_size,
            num_kv_heads,
            num_groups,
            q_len,
            head_dim,
        )

        # Scaling for q·k: keep it consistent with base attention.
        scaling = kwargs.get(
            "scaling",
            getattr(module, "scaling", head_dim**-0.5),
        )

        # ---- Split KV positions into prefix / retrieval / suffix ----
        # prefix part: [0, prefix_len)
        # suffix part: [kv_len - suffix_len, kv_len)
        # retrieval region: [prefix_len, kv_len - suffix_len)
        p_len = max(0, min(prefix_keep_len, kv_len))
        s_len = max(0, min(suffix_keep_len, kv_len - p_len))

        retrieval_start = p_len
        retrieval_end = kv_len - s_len
        retrieval_len = max(0, retrieval_end - retrieval_start)

        # Build prefix indices: [B, KV, p_len]
        prefix_idx = None
        if p_len > 0:
            base_prefix = torch.arange(p_len, device=key.device, dtype=torch.long)  # [p_len]
            prefix_idx = base_prefix.view(1, 1, p_len).expand(batch_size, num_kv_heads, p_len)

        # Build suffix indices: [B, KV, s_len]
        suffix_idx = None
        if s_len > 0:
            start = kv_len - s_len
            base_suffix = torch.arange(start, kv_len, device=key.device, dtype=torch.long)  # [s_len]
            suffix_idx = base_suffix.view(1, 1, s_len).expand(batch_size, num_kv_heads, s_len)

        # ---- Top-k over the retrieval region only ----
        retrieval_idx = None
        if retrieval_len > 0 and top_k > 0:
            k_eff = min(top_k, retrieval_len)

            if use_hnsw:
                # Build or load per-module HNSW indices if not already present
                mid = id(module)
                state = module_state.get(mid)
                if state is None:
                    state = {}
                    module_state[mid] = state

                if "hnsw_indices" not in state:
                    module_tag = getattr(module, "_hnsw_tag", f"mod_{mid}")
                    build_hnsw_for_module_and_head(
                        state=state,
                        key=key,
                        retrieval_start=retrieval_start,
                        retrieval_end=retrieval_end,
                        head_dim=head_dim,
                        module_tag=module_tag,
                    )

                retrieval_idx = hnsw_topk_indices(
                    state=state,
                    query_grouped=query_grouped,
                    top_k=top_k,
                )
            else:
                # Exact top-k via dot-product scores
                # Compute q·k per group:
                # q: [B, KV, groups, 1, D]
                # k: [B, KV, 1, kv_len, D]
                q_expanded = query_grouped
                k_expanded = key.unsqueeze(2)  # [B, KV, 1, kv_len, D]

                # scores_raw: [B, KV, groups, kv_len]
                scores_raw = (q_expanded * k_expanded).sum(-1) * scaling
                # Mean over groups → [B, KV, kv_len]
                scores = scores_raw.mean(dim=2)

                # scores_retrieval: [B, KV, retrieval_len]
                scores_retrieval = scores[:, :, retrieval_start:retrieval_end]
                _, top_local = torch.topk(scores_retrieval, k=k_eff, dim=-1)  # [B, KV, k_eff]
                retrieval_idx = top_local + retrieval_start  # shift back to global positions

        # Concatenate prefix / retrieval / suffix indices: [B, KV, total_kept]
        idx_pieces = []
        if prefix_idx is not None:
            idx_pieces.append(prefix_idx)
        if retrieval_idx is not None:
            idx_pieces.append(retrieval_idx)
        if suffix_idx is not None:
            idx_pieces.append(suffix_idx)

        if not idx_pieces:
            # Degenerate case: nothing kept, fall back to base attention on full KV
            return base_attn_fn(
                module,
                query,
                key,
                value,
                attention_mask,
                **kwargs,
            )

        all_indices = torch.cat(idx_pieces, dim=-1)  # [B, KV, total_kept]

        # Gather sparse K/V: [B, KV, total_kept, D]
        idx_expanded = all_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        key_sparse = torch.gather(key, dim=2, index=idx_expanded)
        value_sparse = torch.gather(value, dim=2, index=idx_expanded)

        # Repeat KV to full heads: [B, H, total_kept, D]
        key_sparse_rep = repeat_kv(key_sparse, num_groups)
        value_sparse_rep = repeat_kv(value_sparse, num_groups)

        # For simplicity we ignore attention_mask in sparse mode in this script:
        # - In the needle-in-a-haystack setup there is no padding.
        # - All keys are past tokens in decode (no causal violation).
        sparse_mask = None

        # Delegate the *actual attention* on the reduced KV to the base function.
        return base_attn_fn(
            module,
            query,
            key_sparse_rep,
            value_sparse_rep,
            sparse_mask,
            **kwargs,
        )

    return gqa_topk_attention


def compute_cache_bytes(cache) -> int:
    """
    Try to estimate KV cache memory usage in bytes.

    Handles:
    - New-style Cache / DynamicCache with .key_cache / .value_cache
    - Objects that support .to_legacy_cache()
    - Plain tuples/lists of (k, v) tensors
    """
    import torch

    if cache is None:
        return 0

    total = 0

    # Case 1: new-style Cache / DynamicCache
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        for k, v in zip(key_cache, value_cache):
            if isinstance(k, torch.Tensor):
                total += k.numel() * k.element_size()
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
        return total

    # Case 2: something that can be converted to legacy cache
    if hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()
            return compute_cache_bytes(legacy)
        except Exception:
            pass

    # Case 3: plain tuple/list of layers, each layer is (k, v) or similar
    if isinstance(cache, (list, tuple)):
        for layer in cache:
            if isinstance(layer, (list, tuple)):
                for t in layer:
                    if isinstance(t, torch.Tensor):
                        total += t.numel() * t.element_size()
        return total

    # Fallback: unknown structure
    return total


def format_bytes(num_bytes: int) -> str:
    mib = num_bytes / (1024 ** 2)
    gib = num_bytes / (1024 ** 3)
    return f"{num_bytes} bytes (~{mib:.2f} MiB, ~{gib:.2f} GiB)"


def build_needle_prompt_tokens(
    tokenizer,
    book_path: str,
    needle_text: str,
    needle_ratio: float,
    max_context_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Build a needle-in-a-haystack input at token level, then wrap it
    into a Qwen2.5 chat template as a single user message.
    """

    with open(book_path, "r", encoding="utf-8") as f:
        book_text = f.read()

    # 1) Token-level construction: book + needle + question
    book_ids = tokenizer(book_text, add_special_tokens=False).input_ids

    needle_sentence = f"\n\nNEEDLE: {needle_text}\n\n"
    needle_ids = tokenizer(needle_sentence, add_special_tokens=False).input_ids

    question_text = (
        "\n\nIn the document above there is a line that starts with \"NEEDLE:\".\n"
        "Find that line and output it exactly.\n\nAnswer:"
    )
    question_ids = tokenizer(question_text, add_special_tokens=False).input_ids

    n_book = len(book_ids)
    insert_pos = int(n_book * needle_ratio)
    insert_pos = max(0, min(n_book, insert_pos))

    full_ids = (
        book_ids[:insert_pos]
        + needle_ids
        + book_ids[insert_pos:]
        + question_ids
    )

    if len(full_ids) > max_context_tokens:
        overflow = len(full_ids) - max_context_tokens

        trim_front = min(overflow, insert_pos)
        book_front_trimmed = book_ids[trim_front:insert_pos]

        remaining_overflow = overflow - trim_front
        if remaining_overflow > 0:
            book_back_trimmed = book_ids[insert_pos : n_book - remaining_overflow]
        else:
            book_back_trimmed = book_ids[insert_pos:]

        full_ids = book_front_trimmed + needle_ids + book_back_trimmed + question_ids

    # 2) Decode these tokens back to text as the user content
    user_content = tokenizer.decode(full_ids, skip_special_tokens=True)

    # 3) Wrap into chat template: system + user
    messages = [
        {
            "role": "system",
            "content": "You are a careful assistant. When asked, you MUST copy exactly "
                       "the line that starts with 'NEEDLE:' from the given document.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # add assistant prefix
    )

    encoded = tokenizer(chat_text, return_tensors="pt")
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask

    return input_ids, attention_mask, needle_sentence.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-7B-Instruct GQA top-k sparse attention test (needle-in-a-haystack, AttentionInterface)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path (Hugging Face format).",
    )
    parser.add_argument(
        "--book-path",
        type=str,
        default="../.data/war-and-peace-128k.txt",
        help="Path to the book text file (UTF-8).",
    )
    parser.add_argument(
        "--needle-text",
        type=str,
        default="This is the hidden needle sentence used for testing.",
        help="Needle text to be inserted into the book.",
    )
    parser.add_argument(
        "--needle-ratio",
        type=float,
        default=0.5,
        help="Insertion position as a ratio of the book token length (0.0 - 1.0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Number of tokens per KV head to keep in decode (top-k).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate during decode.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=128_000,
        help="Maximum context length (tokens) for the prompt.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--prefix-keep-len",
        type=int,
        default=128,
        help="Number of prefix KV positions to always keep for each head.",
    )
    parser.add_argument(
        "--suffix-keep-len",
        type=int,
        default=128,
        help="Number of suffix KV positions to always keep for each head.",
    )
    parser.add_argument(
        "--use-hnsw",
        action="store_true",
        default=True,
        help="Use CAGRA+HNSW (cuVS) on CPU for top-k selection in the retrieval region.",
    )
    parser.add_argument(
        "--hnsw-M",
        type=int,
        default=32,
        help="Logical HNSW degree / CAGRA graph_degree.",
    )
    parser.add_argument(
        "--hnsw-index-dir",
        type=str,
        default="../.data/hnsw_cache",
        help="Directory to cache/load cuVS HNSW indices. "
             "If empty, HNSW indices are kept in memory only.",
    )

    args = parser.parse_args()

    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    if args.use_hnsw:
        if cuvs_cagra is None or cuvs_hnsw is None:
            raise RuntimeError(
                "You enabled --use-hnsw but cuVS is not available. "
                "Install it, e.g. `pip install libcuvs-cu12` (or matching CUDA version)."
            )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[INFO] Loading tokenizer from {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"[INFO] Loading model from {args.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        },
    ).to(device)

    model.config.use_cache = True

    # Tag each attention module with a stable HNSW id for file naming
    attn_modules = []
    for m in model.modules():
        # Qwen2 self-attention modules have num_key_value_groups + q_proj
        if hasattr(m, "num_key_value_groups") and hasattr(m, "q_proj"):
            attn_modules.append(m)
    for idx, m in enumerate(attn_modules):
        setattr(m, "_hnsw_tag", f"attn_{idx}")
    print(f"[INFO] Tagged {len(attn_modules)} attention modules for HNSW.")

    # Figure out which backend we should treat as "base" (for prefill etc.)
    base_impl = getattr(model.config, "attn_implementation", None)
    if base_impl is None:
        base_impl = getattr(model.config, "_attn_implementation", None)
    if base_impl is None:
        # Reasonable default
        base_impl = "sdpa"

    print(f"[INFO] Base attention implementation detected: {base_impl}")

    print(f"[INFO] Building needle-in-a-haystack input from {args.book_path} ...")
    input_ids, attention_mask, needle_line = build_needle_prompt_tokens(
        tokenizer=tokenizer,
        book_path=args.book_path,
        needle_text=args.needle_text,
        needle_ratio=args.needle_ratio,
        max_context_tokens=args.max_context_tokens,
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    print(f"[INFO] Prompt token length: {input_ids.shape[1]}")

    # ------------------ Prefill (dense, base attention implementation) ------------------
    print("[INFO] Running prefill (full KV cache, dense attention) ...")
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    past_key_values = prefill_outputs.past_key_values

    cache_bytes = compute_cache_bytes(past_key_values)
    print("[INFO] Full KV cache size after prefill:")
    print(f"       {format_bytes(cache_bytes)}")

    # ------------------ Register custom GQA top-k attention and switch for decode --------
    custom_name = "gqa_topk_sparse"
    print(
        f"[INFO] Registering custom attention '{custom_name}' with "
        f"top_k={args.top_k}, use_hnsw={args.use_hnsw} ..."
    )

    custom_attn_fn = make_gqa_topk_attention(
        top_k=args.top_k,
        base_impl_name=base_impl,
        prefix_keep_len=args.prefix_keep_len,
        suffix_keep_len=args.suffix_keep_len,
        use_hnsw=args.use_hnsw,
        hnsw_M=args.hnsw_M,
        hnsw_index_dir=args.hnsw_index_dir,
        hnsw_ef=args.top_k,
    )
    AttentionInterface.register(custom_name, custom_attn_fn)

    print(f"[INFO] Switching attention implementation to '{custom_name}' for decode ...")
    set_attn_impl(model.config, custom_name)

    # ------------------ Decode loop (sparse top-k attention) ------------------
    print("[INFO] Running decode with GQA-aware sparse attention ...")

    cur_input_ids = input_ids[:, -1:]
    generated_tokens = []

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(model.generation_config, "eos_token_id", None)

    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(args.max_new_tokens)):
            outputs = model(
                input_ids=cur_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            cur_input_ids = next_token

            if eos_token_id is not None and next_token[0, 0].item() == eos_token_id:
                break

    if generated_tokens:
        generated_ids = torch.cat(generated_tokens, dim=1)
    else:
        generated_ids = torch.empty((1, 0), dtype=torch.long, device=device)

    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True
    )

    print("\n===== Generated answer (new tokens only) =====")
    print(generated_text)
    print("=============================================\n")

    print("[INFO] Ground-truth needle line:")
    print(needle_line)


if __name__ == "__main__":
    main()