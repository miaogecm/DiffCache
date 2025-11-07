# needle_bench_lmcache_vllm.py
# Minimal needle-in-a-haystack benchmark with vLLM + LMCache CPU KV offload.

import os
import time
import argparse

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME


def parse_args():
    p = argparse.ArgumentParser(
        description="Needle-in-a-haystack benchmark with vLLM + LMCache."
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                   help="HF model name.")
    p.add_argument("--book-path", type=str, default="data/book.txt",
                   help="Path to the book text file.")
    p.add_argument("--needle-percent", type=float, default=0.5,
                   help="Needle position as fraction of book length (0.0-1.0).")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Max decode tokens.")
    p.add_argument("--gpu-mem-util", type=float, default=0.85,
                   help="vLLM gpu_memory_utilization.")
    return p.parse_args()


def setup_lmcache():
    # Minimal LMCache CPU offload config; tune for your machine.
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "16.0"  # GB, adjust as needed


def build_prompt(book_text: str, needle_percent: float) -> str:
    needle = "NEEDLE-PHRASE-42-UNICORN"
    n = max(0, min(len(book_text), int(len(book_text) * needle_percent)))
    prompt_text = book_text[:n] + f"\n\nThe secret phrase is: {needle}\n\n" + book_text[n:]

    question = (
        "\n\nYou just read a long text above.\n"
        "Question: What is the exact secret phrase mentioned in the text?\n"
        "Answer (just the phrase):"
    )
    return prompt_text + question, needle


def main():
    args = parse_args()
    setup_lmcache()

    with open(args.book_path, "r", encoding="utf-8") as f:
        book_text = f.read()

    prompt, needle = build_prompt(book_text, args.needle_percent)

    kv_cfg = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm = LLM(
        model=args.model,
        kv_transfer_config=kv_cfg,
        max_model_len=128000,              # assume long book; adjust to your context length
        gpu_memory_utilization=args.gpu_mem_util,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    # Run once; for a proper benchmark you can loop.
    start = time.time()
    outputs = llm.generate([prompt], sampling)
    end = time.time()

    out = outputs[0].outputs[0]
    text = out.text
    decode_tokens = len(out.token_ids)          # vLLM output tokens count
    elapsed = end - start
    decode_throughput = decode_tokens / elapsed if elapsed > 0 else 0.0

    hit = needle in text

    print("=== Needle-in-a-Haystack Result ===")
    print(f"Needle position (percent of book): {args.needle_percent:.3f}")
    print(f"Needle found in answer: {hit}")
    print(f"Model answer: {text.strip()}")
    print("=== Decode Throughput (approx) ===")
    print(f"Decode tokens: {decode_tokens}")
    print(f"Elapsed time: {elapsed:.4f} s")
    print(f"Decode throughput: {decode_throughput:.2f} tok/s")

    # Clean up LMCache backend
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    main()
