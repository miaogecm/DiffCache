#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np
import torch

def infer_len_from_file(path: str, head_dim: int, dtype: str) -> int:
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype]
    n_elems = os.path.getsize(path) // dtype_bytes
    if n_elems % head_dim != 0:
        raise ValueError(
            f"File size is not divisible by head_dim: {path}, "
            f"elements={n_elems}, head_dim={head_dim}"
        )
    return n_elems // head_dim

def load_bin(path: str, length: int, head_dim: int, dtype: str) -> np.ndarray:
    np_dtype = {"float32": np.float32, "float16": np.float16, "bfloat16": np.float16}[dtype]
    arr = np.fromfile(path, dtype=np_dtype)
    if arr.size != length * head_dim:
        raise ValueError(
            f"Size mismatch for {path}: got {arr.size}, expected {length*head_dim}"
        )
    arr = arr.reshape(length, head_dim)
    return arr

def mer_for_queries(q: torch.Tensor, k: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    q: [Q, D]
    k: [K, D]
    returns: MER per query, int64 tensor of shape [Q]
    """
    Q, D = q.shape
    K = k.shape[0]
    scale = 1.0 / math.sqrt(D)

    # [Q, K]
    scores = torch.matmul(q, k.transpose(0, 1)) * scale

    # softmax with stability
    scores = scores - scores.max(dim=1, keepdim=True).values
    probs = torch.softmax(scores, dim=1)  # [Q, K]

    # sort descending per row
    probs_sorted, _ = torch.sort(probs, dim=1, descending=True)  # [Q, K]
    cumsum = torch.cumsum(probs_sorted, dim=1)  # [Q, K]

    # first index where cumsum >= threshold
    mask = (cumsum >= threshold).to(torch.int64)
    # argmin over 0/1 mask is tricky; use where to get first index
    # Convert rows to positions by finding the first True
    # Add a sentinel column of zeros to avoid all-False edge cases (threshold<=0)
    first_idx = torch.argmax(mask, dim=1)  # returns 0 if first element is True, else first True position
    # If threshold==0, MER should be 0; clamp to at least 1 if threshold>0
    mer = first_idx + 1  # count keys, 1-based
    return mer

def main():
    parser = argparse.ArgumentParser(
        description="Compute MER (Minimum Effective Read) from Q/K binaries at a given attention coverage threshold."
    )
    parser.add_argument("--q_path", type=str, default="data/q.bin", help="Path to Q binary (float32 by default).")
    parser.add_argument("--k_path", type=str, default="data/k.bin", help="Path to K binary (float32 by default).")
    parser.add_argument("--threshold", type=float, default=0.8, help="Target cumulative attention coverage in (0,1].")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension D used to reshape binaries.")
    parser.add_argument("--q_len", type=int, default=-1, help="Optional explicit Q length; if <0, infer from file size.")
    parser.add_argument("--k_len", type=int, default=-1, help="Optional explicit K length; if <0, infer from file size.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Storage dtype of binaries.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Computation device.")
    parser.add_argument("--summary_only", action="store_true", help="If set, print only summary statistics.")
    parser.add_argument("--save_csv", type=str, default="", help="Optional path to save per-query MER as CSV.")
    args = parser.parse_args()

    if not (0.0 < args.threshold <= 1.0):
        raise ValueError("threshold must be in (0, 1].")

    # Infer sequence lengths if not provided
    q_len = args.q_len if args.q_len > 0 else infer_len_from_file(args.q_path, args.head_dim, args.dtype)
    k_len = args.k_len if args.k_len > 0 else infer_len_from_file(args.k_path, args.head_dim, args.dtype)

    # Load binaries (assume stored as raw, row-major)
    q_np = load_bin(args.q_path, q_len, args.head_dim, args.dtype).astype(np.float32, copy=False)
    k_np = load_bin(args.k_path, k_len, args.head_dim, args.dtype).astype(np.float32, copy=False)

    # Pick device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    q = torch.from_numpy(q_np).to(device=device, dtype=torch.float32)  # [Q, D]
    k = torch.from_numpy(k_np).to(device=device, dtype=torch.float32)  # [K, D]

    with torch.no_grad():
        mer = mer_for_queries(q, k, args.threshold)  # [Q]
        mer_cpu = mer.cpu().numpy()

    # Summary
    mean = float(np.mean(mer_cpu)) if mer_cpu.size > 0 else 0.0
    median = float(np.median(mer_cpu)) if mer_cpu.size > 0 else 0.0
    p90 = float(np.percentile(mer_cpu, 90)) if mer_cpu.size > 0 else 0.0
    maxv = int(np.max(mer_cpu)) if mer_cpu.size > 0 else 0

    if not args.summary_only:
        print(f"# MER per query (threshold={args.threshold}):")
        print(" ".join(str(int(x)) for x in mer_cpu))

    print("# Summary")
    print(f"total_mer={int(np.sum(mer_cpu))} over {mer_cpu.size} queries")
    print(f"queries={mer_cpu.size} threshold={args.threshold} head_dim={args.head_dim} k_len={k_len}")
    print(f"mean={mean:.3f} median={median:.3f} p90={p90:.3f} max={maxv}")

    if args.save_csv:
        with open(args.save_csv, "w") as f:
            f.write("query_index,MER\n")
            for i, v in enumerate(mer_cpu):
                f.write(f"{i},{int(v)}\n")

if __name__ == "__main__":
    main()
