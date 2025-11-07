#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np
import torch
import hnswlib


def infer_len(path: str, head_dim: int) -> int:
    size = os.path.getsize(path)
    if size % (4 * head_dim) != 0:
        raise ValueError(f"File size of {path} is not divisible by 4 * head_dim.")
    return size // (4 * head_dim)


def load_bin(path: str, length: int, head_dim: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != length * head_dim:
        raise ValueError(
            f"Size mismatch for {path}: got {arr.size}, expected {length * head_dim}"
        )
    return arr.reshape(length, head_dim)


def lift_to_l2(keys: np.ndarray) -> np.ndarray:
    """
    Lift keys from R^d to R^(d+1) so that maximum inner product search
    can be reduced to L2 nearest neighbor search.

    k'_i = [k_i, sqrt(R^2 - ||k_i||^2)]
    where R^2 = max_j ||k_j||^2
    """
    norms2 = np.sum(keys * keys, axis=1)
    R2 = float(np.max(norms2))
    # Numerical guard: clip small negatives caused by float error
    extra = np.sqrt(np.maximum(R2 - norms2, 0.0)).astype(np.float32)
    lifted = np.concatenate([keys.astype(np.float32), extra[:, None]], axis=1)
    return lifted


def lift_query_to_l2(queries: np.ndarray, new_dim: int) -> np.ndarray:
    """
    q' = [q, 0], pad one extra dim with zeros.
    """
    Q, d = queries.shape
    if new_dim != d + 1:
        raise ValueError("new_dim must be original_dim + 1 for lifting.")
    lifted = np.zeros((Q, new_dim), dtype=np.float32)
    lifted[:, :d] = queries.astype(np.float32)
    return lifted


def main():
    parser = argparse.ArgumentParser(
        description=(
            "HNSW with inner-product-to-L2 lifting on single-head Q/K.\n"
            "For each query, retrieve top-M keys via lifted L2 HNSW and report:\n"
            "- selected_keys: number of retrieved keys (M)\n"
            "- attn_coverage: dense softmax attention mass on these keys."
        )
    )
    parser.add_argument("--q_path", type=str, default="data/q.bin",
                        help="Path to Q binary (float32, shape [Q, D]).")
    parser.add_argument("--k_path", type=str, default="data/k.bin",
                        help="Path to K binary (float32, shape [K, D]).")
    parser.add_argument("--head_dim", type=int, default=128,
                        help="Head dimension D.")
    parser.add_argument("--q_len", type=int, default=-1,
                        help="Optional explicit Q length; if <0, infer from file size.")
    parser.add_argument("--k_len", type=int, default=-1,
                        help="Optional explicit K length; if <0, infer from file size.")
    parser.add_argument("--top_m", type=int, default=45,
                        help="Number of keys to retrieve per query.")
    parser.add_argument("--M", type=int, default=32,
                        help="HNSW M parameter (graph degree).")
    parser.add_argument("--ef_construction", type=int, default=200,
                        help="HNSW ef_construction parameter.")
    parser.add_argument("--ef_search", type=int, default=45,
                        help="HNSW ef_search parameter.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for dense attention computation.")
    parser.add_argument("--summary_only", action="store_true",
                        help="If set, print only summary statistics.")
    parser.add_argument("--save_csv", type=str, default="",
                        help="Optional path to save per-query stats as CSV.")
    args = parser.parse_args()

    # Infer lengths
    q_len = args.q_len if args.q_len > 0 else infer_len(args.q_path, args.head_dim)
    k_len = args.k_len if args.k_len > 0 else infer_len(args.k_path, args.head_dim)

    # Load raw Q/K (used for true attention)
    q_raw = load_bin(args.q_path, q_len, args.head_dim).astype(np.float32, copy=False)
    k_raw = load_bin(args.k_path, k_len, args.head_dim).astype(np.float32, copy=False)

    # Lifting: build index on lifted keys for L2 search
    k_lifted = lift_to_l2(k_raw)          # [K, D+1]
    new_dim = k_lifted.shape[1]
    q_lifted = lift_query_to_l2(q_raw, new_dim)  # [Q, D+1]

    # Build HNSW index (L2)
    print(f"# Building HNSW index for {k_len} keys, dim={new_dim}, space=l2...")
    index = hnswlib.Index(space="l2", dim=new_dim)
    index.init_index(
        max_elements=k_len,
        ef_construction=args.ef_construction,
        M=args.M
    )
    index.add_items(k_lifted, np.arange(k_len))
    index.set_ef(args.ef_search)

    # Device choice for dense attention
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    q_raw_t = torch.from_numpy(q_raw).to(device)
    k_raw_t = torch.from_numpy(k_raw).to(device)
    scale = 1.0 / math.sqrt(args.head_dim)

    mer = np.zeros(q_len, dtype=np.int64)
    cov = np.zeros(q_len, dtype=np.float32)

    print("# Querying lifted HNSW and computing attention coverage...")
    with torch.no_grad():
        for i in range(q_len):
            # HNSW search in lifted space
            labels, _ = index.knn_query(q_lifted[i], k=args.top_m)
            sel = labels[0]
            p = int(sel.size)

            if p == 0:
                mer[i] = 0
                cov[i] = 0.0
                continue

            # Dense attention on original Q/K
            qi = q_raw_t[i : i + 1]                      # [1, D]
            logits = (qi @ k_raw_t.t() * scale).squeeze(0)  # [K]
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=-1)

            coverage = probs[sel].sum().item()
            mer[i] = p
            cov[i] = coverage

    # Summary
    avg_mer = float(mer.mean()) if q_len > 0 else 0.0
    avg_cov = float(cov.mean()) if q_len > 0 else 0.0
    min_mer = int(mer.min()) if q_len > 0 else 0
    max_mer = int(mer.max()) if q_len > 0 else 0

    if not args.summary_only:
        print("# query_index,selected_keys,attn_coverage")
        for i in range(q_len):
            print(f"{i},{int(mer[i])},{cov[i]:.6f}")

    print("# Summary")
    print(f"total_selected_keys={int(np.sum(mer))} over {q_len} queries")
    print(f"queries={q_len} keys={k_len} top_m={args.top_m} "
          f"M={args.M} ef_construction={args.ef_construction} ef_search={args.ef_search}")
    print(f"avg_selected_keys={avg_mer:.3f}")
    print(f"avg_attn_coverage={avg_cov:.6f}")
    print(f"min_selected_keys={min_mer} max_selected_keys={max_mer}")

    if args.save_csv:
        with open(args.save_csv, "w") as f:
            f.write("query_index,selected_keys,attn_coverage\n")
            for i in range(q_len):
                f.write(f"{i},{int(mer[i])},{cov[i]:.6f}\n")


if __name__ == "__main__":
    main()