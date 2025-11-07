#!/usr/bin/env python3
import argparse
import os
import math
import pickle
import hashlib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def infer_len(path: str, head_dim: int) -> int:
    size = os.path.getsize(path)
    if size % (4 * head_dim) != 0:
        raise ValueError(
            f"File size of {path} ({size} bytes) is not divisible by 4 * head_dim={4*head_dim}."
        )
    return size // (4 * head_dim)


def load_bin(path: str, length: int, head_dim: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != length * head_dim:
        raise ValueError(
            f"Size mismatch for {path}: got {arr.size} elements, "
            f"expected {length * head_dim}."
        )
    return arr.reshape(length, head_dim)


def train_pq(keys: np.ndarray, n_subvectors: int, ksub: int, seed: int = 42):
    """
    Train product quantization codebooks on keys.

    keys: [K, D]
    n_subvectors: number of subspaces M
    ksub: centroids per subspace
    Returns:
        codebooks: list of np.ndarray with shape [ksub, d_sub]
    """
    K, D = keys.shape
    if D % n_subvectors != 0:
        raise ValueError(
            f"D={D} is not divisible by n_subvectors={n_subvectors}. "
            f"Use a compatible head_dim or adjust n_subvectors."
        )
    d_sub = D // n_subvectors
    codebooks = []
    for m in tqdm(range(n_subvectors)):
        sub = keys[:, m * d_sub : (m + 1) * d_sub]
        # Standard k-means (L2); simple and good enough for our "bare PQ" variant.
        km = MiniBatchKMeans(
            n_clusters=ksub,
            n_init=8,
            random_state=seed + m,
            batch_size=8192,
            verbose=0,
        )
        km.fit(sub)
        codebooks.append(km.cluster_centers_.astype(np.float32))
    return codebooks


def _hash_array(arr: np.ndarray) -> str:
    arr_contig = np.ascontiguousarray(arr)
    return hashlib.sha256(memoryview(arr_contig.view(np.uint8))).hexdigest()


def _hash_codebooks(codebooks) -> str:
    hasher = hashlib.sha256()
    for cb in codebooks:
        arr = np.ascontiguousarray(cb)
        hasher.update(memoryview(arr.view(np.uint8)))
    return hasher.hexdigest()


def encode_pq(
    keys: np.ndarray,
    codebooks,
    n_subvectors: int,
    cache_path: str = "",
    refresh_cache: bool = False,
    keys_hash=None,
    codebook_hash=None,
):
    """
    Encode keys into PQ codes.

    keys: [K, D]
    codebooks: list of [ksub, d_sub]
    Returns:
        codes: np.ndarray[int32] of shape [K, n_subvectors]
    """
    K, D = keys.shape
    d_sub = D // n_subvectors

    cache_path = cache_path or ""
    ksub = codebooks[0].shape[0] if codebooks else 0
    if keys_hash is None:
        keys_hash = _hash_array(keys)
    if codebook_hash is None:
        codebook_hash = _hash_codebooks(codebooks)

    if cache_path and not refresh_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            meta = payload.get("meta", {})
            codes = payload.get("codes", None)
            if (
                isinstance(codes, np.ndarray)
                and codes.shape == (K, n_subvectors)
                and codes.dtype == np.int32
                and meta.get("n_keys") == K
                and meta.get("dim") == D
                and meta.get("n_subvectors") == n_subvectors
                and meta.get("ksub") == ksub
                and meta.get("codebook_sha256") == codebook_hash
                and meta.get("keys_sha256") == keys_hash
            ):
                print(f"# Loading PQ codes from cache: {cache_path}")
                return codes
            else:
                print(f"# Cache mismatch detected at {cache_path}; re-encoding.")
        except Exception as exc:
            print(f"# Warning: failed to load PQ code cache {cache_path}: {exc}. Re-encoding.")

    print("# Encoding keys to PQ codes...")
    codes = np.empty((K, n_subvectors), dtype=np.int32)
    for m in range(n_subvectors):
        sub = keys[:, m * d_sub : (m + 1) * d_sub]          # [K, d_sub]
        cb = codebooks[m]                                   # [ksub, d_sub]
        # Compute L2 distances to all centroids; choose nearest.
        # (Bare PQ; no fancy IP-optimized training.)
        # dist^2 = ||x||^2 + ||c||^2 - 2 x·c; we can just brute-force here.
        # [K, ksub]
        # To keep code simple and robust, we do direct broadcasting.
        diff = sub[:, None, :] - cb[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        codes[:, m] = np.argmin(dist2, axis=1)

    if cache_path:
        payload = {
            "meta": {
                "n_keys": K,
                "dim": D,
                "n_subvectors": n_subvectors,
                "ksub": ksub,
                "codebook_sha256": codebook_hash,
                "keys_sha256": keys_hash,
            },
            "codes": codes,
        }
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"# Saved PQ codes to cache: {cache_path}")
        except Exception as exc:
            print(f"# Warning: failed to save PQ code cache {cache_path}: {exc}")

    return codes


def approx_scores_ip(q: np.ndarray, codebooks, codes, n_subvectors: int):
    """
    Compute approximate inner-product scores between a single query q and all keys,
    using PQ codes and codebooks.

    q: [D]
    codebooks: list of [ksub, d_sub]
    codes: [K, n_subvectors]
    Returns:
        scores: [K] approximate q·k
    """
    K, M = codes.shape
    D = q.shape[0]
    d_sub = D // M

    # Precompute LUT: for each subspace m and centroid j, q_sub · c[m][j]
    # lut[m, j]
    lut = []
    for m in range(M):
        q_sub = q[m * d_sub : (m + 1) * d_sub]            # [d_sub]
        cb = codebooks[m]                                 # [ksub, d_sub]
        # [ksub]
        lut_m = cb @ q_sub.astype(np.float32)
        lut.append(lut_m)
    lut = np.stack(lut, axis=0)                           # [M, ksub]

    # For each key i: score_i = sum_m lut[m, codes[i,m]]
    # Vectorized gather
    # lut[range(M)[:,None], codes.T] -> [M,K]
    idx_m = np.arange(M, dtype=np.int64)[:, None]
    scores = lut[idx_m, codes.T].sum(axis=0)              # [K]
    return scores.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Simplified PQCache-style retrieval on single-head Q/K using bare Product Quantization.\n"
            "Steps:\n"
            "  1) Train PQ codebooks on K.\n"
            "  2) Encode K to PQ codes.\n"
            "  3) For each Q, use PQ approx inner product to select top-M keys.\n"
            "  4) Compute dense softmax attention on original Q/K and report coverage."
        )
    )
    parser.add_argument("--q_path", type=str, default="data/q.bin",
                        help="Path to Q binary (float32, shape [Q, D]).")
    parser.add_argument("--k_path", type=str, default="data/k.bin",
                        help="Path to K binary (float32, shape [K, D]).")
    parser.add_argument("--head_dim", type=int, default=128,
                        help="Head dimension D.")
    parser.add_argument("--q_len", type=int, default=-1,
                        help="Optional Q length; if <0, infer from file size.")
    parser.add_argument("--k_len", type=int, default=-1,
                        help="Optional K length; if <0, infer from file size.")
    parser.add_argument("--n_subvectors", type=int, default=8,
                        help="Number of PQ subspaces M; D must be divisible by this.")
    parser.add_argument("--ksub", type=int, default=256,
                        help="Number of centroids per subspace.")
    parser.add_argument("--top_m", type=int, default=3500,
                        help="Number of keys to retrieve per query.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for dense attention computation.")
    parser.add_argument("--summary_only", action="store_true",
                        help="If set, print only summary statistics.")
    parser.add_argument("--save_csv", type=str, default="",
                        help="Optional path to save per-query stats as CSV.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for PQ k-means.")
    parser.add_argument("--pq_codebooks_cache", type=str, default="data/codebooks.pkl",
                        help="Optional path to cache PQ codebooks (pickle).")
    parser.add_argument("--refresh_pq_codebooks_cache", action="store_true",
                        help="Ignore existing PQ codebooks cache and retrain.")
    parser.add_argument("--pq_codes_cache", type=str, default="data/codes.pkl",
                        help="Optional path to cache PQ codes (pickle).")
    parser.add_argument("--refresh_pq_codes_cache", action="store_true",
                        help="Ignore existing PQ codes cache and recompute.")
    args = parser.parse_args()

    if args.top_m <= 0:
        raise ValueError("top_m must be > 0.")

    # Infer lengths
    q_len = args.q_len if args.q_len > 0 else infer_len(args.q_path, args.head_dim)
    k_len = args.k_len if args.k_len > 0 else infer_len(args.k_path, args.head_dim)

    # Load raw Q/K
    q_raw = load_bin(args.q_path, q_len, args.head_dim).astype(np.float32, copy=False)
    k_raw = load_bin(args.k_path, k_len, args.head_dim).astype(np.float32, copy=False)

    D = k_raw.shape[1]
    if D % args.n_subvectors != 0:
        raise ValueError(
            f"D={D} is not divisible by n_subvectors={args.n_subvectors}. "
            f"Adjust head_dim or n_subvectors."
        )
    d_sub = D // args.n_subvectors
    keys_hash = _hash_array(k_raw)

    codebooks = None
    codebook_hash = None
    codebooks_cache_path = args.pq_codebooks_cache or ""
    if (codebooks_cache_path
            and not args.refresh_pq_codebooks_cache
            and os.path.exists(codebooks_cache_path)):
        try:
            with open(codebooks_cache_path, "rb") as f:
                payload = pickle.load(f)
            meta = payload.get("meta", {})
            stored = payload.get("codebooks", [])
            if (
                meta.get("n_keys") == k_len
                and meta.get("dim") == D
                and meta.get("n_subvectors") == args.n_subvectors
                and meta.get("ksub") == args.ksub
                and meta.get("seed") == args.seed
                and meta.get("keys_sha256") == keys_hash
                and isinstance(stored, list)
                and len(stored) == args.n_subvectors
            ):
                loaded = []
                valid = True
                for cb in stored:
                    arr = np.asarray(cb, dtype=np.float32)
                    if arr.shape != (args.ksub, d_sub):
                        valid = False
                        break
                    loaded.append(arr)
                if valid:
                    codebooks = loaded
                    codebook_hash = meta.get("codebook_sha256")
                    if not codebook_hash:
                        codebook_hash = _hash_codebooks(codebooks)
                    print(f"# Loading PQ codebooks from cache: {codebooks_cache_path}")
                else:
                    print(f"# Codebook cache shape mismatch at {codebooks_cache_path}; retraining.")
            else:
                print(f"# Codebook cache mismatch at {codebooks_cache_path}; retraining.")
        except Exception as exc:
            print(f"# Warning: failed to load PQ codebooks cache {codebooks_cache_path}: {exc}. Retraining.")

    if codebooks is None:
        print(f"# Training PQ on {k_len} keys, D={D}, "
              f"M={args.n_subvectors}, ksub={args.ksub} ...")
        codebooks = train_pq(k_raw, args.n_subvectors, args.ksub, seed=args.seed)
        codebook_hash = _hash_codebooks(codebooks)
        if codebooks_cache_path:
            payload = {
                "meta": {
                    "n_keys": k_len,
                    "dim": D,
                    "n_subvectors": args.n_subvectors,
                    "ksub": args.ksub,
                    "seed": args.seed,
                    "keys_sha256": keys_hash,
                    "codebook_sha256": codebook_hash,
                },
                "codebooks": codebooks,
            }
            try:
                with open(codebooks_cache_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"# Saved PQ codebooks to cache: {codebooks_cache_path}")
            except Exception as exc:
                print(f"# Warning: failed to save PQ codebooks cache {codebooks_cache_path}: {exc}")

    if codebook_hash is None:
        codebook_hash = _hash_codebooks(codebooks)

    codes = encode_pq(
        k_raw,
        codebooks,
        args.n_subvectors,
        cache_path=args.pq_codes_cache,
        refresh_cache=args.refresh_pq_codes_cache,
        keys_hash=keys_hash,
        codebook_hash=codebook_hash,
    )   # [K, M]

    # Device for dense attention
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    q_raw_t = torch.from_numpy(q_raw).to(device)
    k_raw_t = torch.from_numpy(k_raw).to(device)
    scale = 1.0 / math.sqrt(args.head_dim)

    selected = np.zeros(q_len, dtype=np.int64)
    cov = np.zeros(q_len, dtype=np.float32)

    print("# Running PQCache-style retrieval per query...")
    with torch.no_grad():
        for i in range(q_len):
            # Approximate scores via PQ
            scores_approx = approx_scores_ip(
                q=q_raw[i],
                codebooks=codebooks,
                codes=codes,
                n_subvectors=args.n_subvectors,
            )  # [K]

            # Choose top_m by approximate score
            if args.top_m >= k_len:
                top_ids = np.arange(k_len, dtype=np.int64)
            else:
                top_ids = np.argpartition(-scores_approx, args.top_m - 1)[: args.top_m]
            m = int(top_ids.size)

            # Dense attention on original Q/K
            qi = q_raw_t[i : i + 1]                         # [1, D]
            logits = (qi @ k_raw_t.t() * scale).squeeze(0)  # [K]
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=-1)

            coverage = probs[top_ids].sum().item() if m > 0 else 0.0

            selected[i] = m
            cov[i] = coverage

    # Summary
    avg_sel = float(selected.mean()) if q_len > 0 else 0.0
    avg_cov = float(cov.mean()) if q_len > 0 else 0.0
    min_cov = float(cov.min()) if q_len > 0 else 0.0
    max_cov = float(cov.max()) if q_len > 0 else 0.0

    if not args.summary_only:
        print("# query_index,selected_keys,attn_coverage")
        for i in range(q_len):
            print(f"{i},{int(selected[i])},{cov[i]:.6f}")

    print("# Summary")
    print(f"total_selected_keys={int(np.sum(selected))} over {q_len} queries")
    print(f"queries={q_len} keys={k_len} top_m={args.top_m} "
          f"M={args.n_subvectors} ksub={args.ksub}")
    print(f"avg_selected_keys={avg_sel:.3f}")
    print(f"avg_attn_coverage={avg_cov:.6f}")
    print(f"min_attn_coverage={min_cov:.6f} max_attn_coverage={max_cov:.6f}")

    if args.save_csv:
        with open(args.save_csv, "w") as f:
            f.write("query_index,selected_keys,attn_coverage\n")
            for i in range(q_len):
                f.write(f"{i},{int(selected[i])},{cov[i]:.6f}\n")


if __name__ == "__main__":
    main()
