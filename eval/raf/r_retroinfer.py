#!/usr/bin/env python3
import argparse
import os
import math
import pickle
import numpy as np
import torch

try:
    from sklearn.cluster import KMeans
except ImportError as e:
    raise ImportError(
        "scikit-learn is required. Install via `pip install scikit-learn`."
    ) from e

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def infer_len(path: str, head_dim: int, dtype: str) -> int:
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype]
    size = os.path.getsize(path)
    if size % dtype_bytes != 0:
        raise ValueError(f"File size of {path} is not aligned with dtype {dtype}.")
    n_elem = size // dtype_bytes
    if n_elem % head_dim != 0:
        raise ValueError(
            f"File size of {path} incompatible with head_dim={head_dim}: "
            f"n_elem={n_elem}."
        )
    return n_elem // head_dim


def load_bin(path: str, length: int, head_dim: int, dtype: str) -> np.ndarray:
    np_dtype = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": np.float16,
    }[dtype]
    arr = np.fromfile(path, dtype=np_dtype)
    if arr.size != length * head_dim:
        raise ValueError(
            f"Unexpected size for {path}: got {arr.size}, "
            f"expected {length * head_dim}."
        )
    arr = arr.reshape(length, head_dim)
    return arr.astype(np.float32, copy=False)


def build_segment_kmeans_index(
    k: np.ndarray,
    num_segments: int,
    clusters_per_segment: int,
    random_state: int = 0,
    cache_path: str = "",
    refresh_cache: bool = False,
):
    """
    Returns:
      segments: list of dicts with:
        - start: int
        - end: int
        - centers: torch.FloatTensor [C, D]
        - cluster_to_indices: list of 1D np.ndarray of absolute key indices
    """
    n_keys, dim = k.shape
    num_segments = max(1, min(num_segments, n_keys))
    seg_size = math.ceil(n_keys / num_segments)
    segments = []

    cache_path = cache_path or ""
    if cache_path and not refresh_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            cached_meta = payload.get("meta", {})
            cached_segments = payload.get("segments", [])
            if (
                cached_meta.get("n_keys") == n_keys
                and cached_meta.get("dim") == dim
                and cached_meta.get("num_segments") == num_segments
                and cached_meta.get("clusters_per_segment") == clusters_per_segment
                and cached_meta.get("random_state") == random_state
            ):
                print(f"Loading segmented k-means index from cache: {cache_path}")
                for seg in cached_segments:
                    centers_np = np.asarray(seg["centers"], dtype=np.float32)
                    cluster_to_indices = [
                        np.asarray(arr, dtype=np.int64)
                        for arr in seg["cluster_to_indices"]
                    ]
                    segments.append(
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "centers": torch.from_numpy(centers_np),
                            "cluster_to_indices": cluster_to_indices,
                        }
                    )
                return segments
            else:
                print(f"# Cache mismatch detected for {cache_path}; rebuilding index.")
        except Exception as exc:
            print(f"# Warning: failed to load cache {cache_path}: {exc}. Recomputing.")

    raw_segments = []

    print(f"Building segmented k-means index: {num_segments} segments, "
          f"{clusters_per_segment} clusters/segment")

    for s in tqdm(range(num_segments), desc="Segment k-means"):
        start = s * seg_size
        end = min(n_keys, (s + 1) * seg_size)
        if end <= start:
            break

        seg_keys = k[start:end]
        c = min(clusters_per_segment, seg_keys.shape[0])
        if c <= 0:
            continue

        km = KMeans(
            n_clusters=c,
            n_init=10,
            random_state=random_state,
        )
        labels = km.fit_predict(seg_keys)
        centers = km.cluster_centers_.astype(np.float32)  # [c, dim]

        cluster_to_indices = []
        for cid in range(c):
            local_idx = np.where(labels == cid)[0]
            if local_idx.size == 0:
                cluster_to_indices.append(np.empty(0, dtype=np.int64))
            else:
                cluster_to_indices.append(local_idx.astype(np.int64) + start)

        seg = {
            "start": start,
            "end": end,
            "centers": centers,
            "cluster_to_indices": cluster_to_indices,
        }
        raw_segments.append(seg)

    if cache_path:
        cache_payload = {
            "meta": {
                "n_keys": n_keys,
                "dim": dim,
                "num_segments": num_segments,
                "clusters_per_segment": clusters_per_segment,
                "random_state": random_state,
            },
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "centers": seg["centers"],
                    "cluster_to_indices": seg["cluster_to_indices"],
                }
                for seg in raw_segments
            ],
        }
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved segmented k-means index cache to {cache_path}")
        except Exception as exc:
            print(f"# Warning: failed to write cache {cache_path}: {exc}")

    for seg in raw_segments:
        segments.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "centers": torch.from_numpy(seg["centers"]),
                "cluster_to_indices": seg["cluster_to_indices"],
            }
        )

    return segments


def select_keys_for_query(q_vec: torch.Tensor,
                          segments,
                          top_m_clusters: int,
                          scale: float,
                          device: torch.device):
    """
    q_vec: [D] on device
    segments: list of dicts from build_segment_kmeans_index
    Returns:
      selected_indices: 1D np.ndarray of unique key indices
    """
    idx_list = []
    for seg in segments:
        centers = seg["centers"].to(device=device)
        # [C]
        scores = torch.matmul(centers, q_vec) * scale
        m = min(top_m_clusters, centers.size(0))
        if m <= 0:
            continue
        _, top_idx = torch.topk(scores, k=m, dim=0)
        top_idx = top_idx.cpu().numpy()
        for cid in top_idx:
            cand = seg["cluster_to_indices"][int(cid)]
            if cand.size > 0:
                idx_list.append(cand)

    if not idx_list:
        return np.empty(0, dtype=np.int64)

    all_idx = np.concatenate(idx_list, axis=0)
    # unique and sorted for determinism
    return np.unique(all_idx)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "RetroInfer-style segmented k-means retrieval simulation on q/k.bin "
            "for a single head. Outputs per-query read key count, total reads, "
            "and per-query attention mass over selected keys."
        )
    )
    parser.add_argument("--q_path", type=str, default="data/q.bin")
    parser.add_argument("--k_path", type=str, default="data/k.bin")
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--q_len", type=int, default=-1)
    parser.add_argument("--k_len", type=int, default=-1)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--num_segments", type=int, default=16)
    parser.add_argument("--clusters_per_segment", type=int, default=512)
    parser.add_argument("--top_m_clusters", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--segments_cache", type=str, default="data/segment_cache.pkl",
                        help="Optional path to load/save segmented k-means index.")
    parser.add_argument("--refresh_segments_cache", action="store_true",
                        help="Rebuild the segmented k-means index even if cache exists.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Infer lengths if not provided
    q_len = args.q_len if args.q_len > 0 else infer_len(args.q_path, args.head_dim, args.dtype)
    k_len = args.k_len if args.k_len > 0 else infer_len(args.k_path, args.head_dim, args.dtype)

    print(f"Loading Q from {args.q_path} (len={q_len}, dim={args.head_dim})")
    print(f"Loading K from {args.k_path} (len={k_len}, dim={args.head_dim})")

    q_np = load_bin(args.q_path, q_len, args.head_dim, args.dtype)
    k_np = load_bin(args.k_path, k_len, args.head_dim, args.dtype)

    # Build segmented k-means index on CPU
    segments = build_segment_kmeans_index(
        k_np,
        num_segments=args.num_segments,
        clusters_per_segment=args.clusters_per_segment,
        random_state=args.random_state,
        cache_path=args.segments_cache,
        refresh_cache=args.refresh_segments_cache,
    )

    # Move Q and K to device
    q = torch.from_numpy(q_np).to(device=device, dtype=torch.float32)  # [Q, D]
    k = torch.from_numpy(k_np).to(device=device, dtype=torch.float32)  # [K, D]

    scale = 1.0 / math.sqrt(args.head_dim)

    per_query_reads = []
    per_query_mass = []

    print("Running RetroInfer-style retrieval and attention evaluation...")
    for qi in tqdm(range(q_len), desc="Queries"):
        q_vec = q[qi]  # [D]

        # Select candidate keys via segmented k-means
        selected = select_keys_for_query(
            q_vec=q_vec,
            segments=segments,
            top_m_clusters=args.top_m_clusters,
            scale=scale,
            device=device,
        )
        read_count = int(selected.size)

        # Compute dense attention distribution for this query
        # scores: [K]
        scores = torch.matmul(k, q_vec) * scale
        scores = scores - scores.max()
        probs = torch.softmax(scores, dim=0)

        # Attention mass over selected keys
        if read_count > 0:
            idx_t = torch.from_numpy(selected).to(device=device, dtype=torch.long)
            mass = float(probs[idx_t].sum().item())
        else:
            mass = 0.0

        per_query_reads.append(read_count)
        per_query_mass.append(mass)

    per_query_reads = np.array(per_query_reads, dtype=np.int64)
    per_query_mass = np.array(per_query_mass, dtype=np.float32)

    total_reads = int(per_query_reads.sum())

    # Output
    print("# query_index,keys_read,attn_mass")
    for i in range(q_len):
        print(f"{i},{per_query_reads[i]},{per_query_mass[i]:.6f}")

    print("# summary")
    print(f"num_queries={q_len}")
    print(f"total_keys={k_len}")
    print(f"total_keys_read={total_reads}")
    print(f"avg_keys_read_per_query={per_query_reads.mean():.3f}")
    print(f"avg_attn_mass={per_query_mass.mean():.6f}")
    print(f"median_attn_mass={np.median(per_query_mass):.6f}")

if __name__ == "__main__":
    main()
