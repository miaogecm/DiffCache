import os
import sys
import json
import math
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

def parse_attn_args(parser: argparse.ArgumentParser):
    parser.add_argument("--budget_ratio", type=float, default=0.018, help="ratio of budget")
    parser.add_argument("--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters for RetriveInfer")

    return parser


def generate_config(
    model_name, 
    context_len, 
    attn_type,
    budget_ratio=0.018,
    estimate_ratio=0.25,
    # default retrieve infer configs
    n_segments=None,
):
    aprox_cluster_size = 16

    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1]+'.json'
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE, "r") as f:
        original_config = json.load(f)
    
    if n_segments is None:
        n_segments = max(1, context_len // 8192)
    
    n_clusters = math.ceil(context_len/aprox_cluster_size)

    if attn_type == 'RetroInfer':
        # compute the nearest multiple of (n_segments*32)
        lower = (n_clusters // (n_segments*32)) * (n_segments*32)
        upper = lower + (n_segments*32)
        n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
    
    nprobe = max(1, int(n_clusters*budget_ratio))
    print(f"context_len: {context_len}, n_clusters: {n_clusters}, nprobe: {nprobe}, n_segments: {n_segments}")

    if attn_type == 'RetroInfer':
        original_config[attn_type]['n_centroids'] = n_clusters
        original_config[attn_type]['n_segment'] = n_segments
        original_config[attn_type]['nprobe'] = nprobe
        original_config[attn_type]['cache_cluster_num'] = int(nprobe*3)
        original_config[attn_type]['max_compute_cluster_num'] = int(n_clusters*estimate_ratio) + nprobe

        print(original_config[attn_type])
    
    return original_config