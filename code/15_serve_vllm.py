#!/usr/bin/env python
"""
15_serve_vllm.py — serve a fine-tuned model and benchmark it honestly.

Serving choices
---------------
  vLLM          highest throughput on GPU; PagedAttention, continuous batching,
                hot-swappable LoRA adapters, OpenAI-compatible API. The default.
  SGLang        comparable, often faster on structured/constrained decoding.
  TGI           HuggingFace's server; good HF ecosystem integration.
  llama.cpp     CPU / Apple Silicon / edge. The only option without a GPU.
  Ollama        llama.cpp with a friendly wrapper; great for local dev.
  transformers  fine for a demo, ~10-50x slower than vLLM under concurrency.

The metric that matters
-----------------------
Throughput (tokens/sec) at your target concurrency, and time-to-first-token (TTFT).
Not "tokens/sec on a single request with batch size 1" — that number is meaningless
and is what most blog posts report.

Two distinct regimes:
  * **Prefill-bound** (long prompts, short answers): dominated by prompt length. This
    is where RAG systems live. TTFT is what your users feel.
  * **Decode-bound** (short prompts, long answers): dominated by memory bandwidth.
    Tokens/sec is what your users feel.

Run it
------
    python 15_serve_vllm.py --model ./out/merged --serve
    python 15_serve_vllm.py --model ./out/merged --benchmark --concurrency 1 8 32
    python 15_serve_vllm.py --model Qwen/Qwen2.5-7B --adapter ./out/sft-lora --serve
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common.memory import inference_gb, kv_cache_gb      # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="Serve a LoRA adapter without merging")
    p.add_argument("--serve", action="store_true", help="Start an OpenAI-compatible server")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--quantization", default=None,
                   choices=[None, "awq", "gptq", "fp8", "squeezellm"])
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--quant-bits", type=float, default=2.0,
                   help="Weight bits per param, for the VRAM estimate (2=fp16, 1=int8, 0.5=int4)")
    p.add_argument("--n-requests", type=int, default=48)
    a = p.parse_args()

    # ----------------------------------------------------------------------------------
    # VRAM budget BEFORE you launch — the most common vLLM failure is an OOM at startup
    # because max_model_len was set larger than the KV cache can fit.
    # ----------------------------------------------------------------------------------
    size = next((m for m in ["0.5B", "1B", "1.5B", "2B", "3B", "7B", "8B", "13B",
                             "14B", "32B", "70B"] if m.lower() in a.model.lower()), "7B")
    print(f"\n  ── VRAM budget ({size}, max_model_len={a.max_model_len}) ──")
    weights = inference_gb(size, 1, 1, a.quant_bits, framework_overhead_gb=0)
    for conc in (1, 8, 32, 128):
        kv = kv_cache_gb(size, a.max_model_len, conc, a.quant_bits)
        total = weights + kv + 1.0        # ~1GB CUDA context + workspace
        print(f"    concurrency {conc:>4}:  weights {weights:>5.1f} GB + "
              f"KV {kv:>6.2f} GB = {total:>5.1f} GB")
    print("\n  If the largest row exceeds your GPU, lower --max-model-len or add")
    print("  --quantization. vLLM pre-allocates the KV cache for max_model_len x")
    print("  max_concurrency, so an over-large max_model_len fails at startup, not at")
    print("  request time — which is at least a fast failure.\n")

    if a.serve:
        _serve(a)
    elif a.benchmark:
        _benchmark(a)
    else:
        print("  Pass --serve or --benchmark.\n")


def _serve(a) -> None:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", a.model,
        "--port", str(a.port),
        "--max-model-len", str(a.max_model_len),
        "--gpu-memory-utilization", str(a.gpu_mem_util),
    ]
    if a.quantization:
        cmd += ["--quantization", a.quantization]
    if a.adapter:
        # Serving the adapter unmerged lets you hot-swap between adapters over one base,
        # which is far more memory-efficient than running N merged models.
        cmd += ["--enable-lora", "--lora-modules", f"finetuned={a.adapter}"]

    print("  launching vLLM:")
    print("   ", " ".join(cmd))
    print(f"\n  OpenAI-compatible endpoint:  http://localhost:{a.port}/v1")
    print(f"  model name to pass:          {'finetuned' if a.adapter else a.model}\n")

    import subprocess
    subprocess.run(cmd)


def _benchmark(a) -> None:
    """
    Benchmark via the HTTP API, measuring what users actually feel:
      * TTFT  — time to first token (perceived responsiveness)
      * TPOT  — time per output token (perceived speed once it starts)
      * throughput — total tokens/sec across all concurrent requests
    """
    import concurrent.futures as cf

    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")

    url = f"http://localhost:{a.port}/v1/chat/completions"
    model_name = "finetuned" if a.adapter else a.model

    prompts = [
        "Explain what a learning rate schedule is, in two paragraphs.",
        "Write a Python function that merges two sorted lists.",
        "Summarise the causes of the 2008 financial crisis.",
        "What are the trade-offs of 4-bit quantization?",
        "Draft a short project update email for a delayed migration.",
        "Explain precision vs recall with an example.",
    ]

    def one(prompt: str, max_tokens: int = 128) -> dict:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        }
        t0 = time.perf_counter()
        ttft = None
        chunks = 0
        try:
            with requests.post(url, json=payload, stream=True, timeout=180) as r:
                for line in r.iter_lines():
                    if not line or not line.startswith(b"data: "):
                        continue
                    body = line[6:]
                    if body == b"[DONE]":
                        break
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
        except Exception as e:                              # noqa: BLE001
            return {"error": str(e)}
        total = time.perf_counter() - t0
        return {"ttft": ttft or total, "total": total, "tokens": chunks}

    print(f"\n  ── benchmarking {url} ──")
    for conc in a.concurrency:
        with cf.ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(one, prompts[i % len(prompts)]) for i in range(a.n_requests)]
            results = [f.result() for f in futs]

        ok = [r for r in results if "error" not in r]
        if not ok:
            print(f"  concurrency {conc}: all requests failed — is the server running?")
            print(f"    first error: {results[0].get('error')}")
            continue

        ttfts = sorted(r["ttft"] for r in ok)
        total_tokens = sum(r["tokens"] for r in ok)
        wall = max(r["total"] for r in ok)

        print(f"\n  concurrency {conc}   ({len(ok)}/{len(results)} ok)")
        print(f"    TTFT p50 / p95     {statistics.median(ttfts)*1000:>7.0f} ms / "
              f"{ttfts[int(len(ttfts)*0.95)]*1000:>7.0f} ms")
        print(f"    output tokens      {total_tokens:,}")
        print(f"    wall clock         {wall:>7.2f} s")
        print(f"    THROUGHPUT         {total_tokens/wall:>7.1f} tok/s   ← the number that matters")
        if conc == a.concurrency[0]:
            print("    (compare against your SLO: TTFT < 500ms feels instant; "
                  "> 2s feels slow)")

    print("\n  ── interpreting results ──")
    print("    Throughput should RISE with concurrency (continuous batching). If it is")
    print("    flat, you are compute-bound, not batching-bound.")
    print("    TTFT rising steeply with concurrency means the KV cache is saturated —")
    print("    lower --max-model-len or reduce concurrency.")
    print("    If a request failed with a context-length error, the prompt+max_tokens")
    print("    exceeded --max-model-len. vLLM rejects rather than truncates.")


if __name__ == "__main__":
    main()
