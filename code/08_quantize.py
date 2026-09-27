#!/usr/bin/env python
"""
08_quantize.py — Post-training quantization: GPTQ, AWQ, bitsandbytes, GGUF.

Choose by DEPLOYMENT TARGET, not by benchmark table:

  Target                     Method          Why
  -------------------------  --------------  ----------------------------------------
  Quick local test           bitsandbytes    one flag, no calibration, works anywhere
  GPU serving (vLLM/TGI)     AWQ or GPTQ     fast kernels, best throughput
  Best quality at 4-bit      AWQ             protects activation-salient channels
  CPU / Apple / edge         GGUF (llama.cpp) the only real option; runs on anything
  Max throughput (NVIDIA)    TensorRT-LLM    fastest, but the most painful to set up
  Keep training afterwards   QLoRA (NF4)     see 01_sft_lora.py — most formats are frozen

The rule that surprises people: **a quantized model usually cannot be fine-tuned.**
GPTQ/AWQ/GGUF are inference formats; the weights are frozen integers. The exception is
bitsandbytes NF4, which QLoRA trains adapters on top of.

Run it
------
    python 08_quantize.py --help
    python 08_quantize.py --model Qwen/Qwen2.5-7B-Instruct --method awq --out ./out/awq
    python 08_quantize.py --model ./out/merged --method gptq --bits 4 --group-size 128
    python 08_quantize.py --model ./out/merged --method gguf --quant-type Q4_K_M
    python 08_quantize.py --model Qwen/Qwen2.5-7B-Instruct --method bnb --bits 4 --eval-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common.memory import inference_gb                      # noqa: E402

# A small, general calibration set. Calibration is what makes GPTQ/AWQ work: the
# algorithm needs to see the *distribution of activations* your model actually
# produces, so it can decide which weights matter.
#
# The trap: if you calibrate on text that looks nothing like your production inputs,
# quality on your real traffic degrades even though perplexity on wikitext is fine.
# ALWAYS calibrate on 128-512 samples of YOUR OWN data.
DEFAULT_CALIB = [
    "Explain the difference between supervised fine-tuning and continued pretraining.",
    "Write a Python function that reverses a linked list iteratively.",
    "Summarise the main causes of the 2008 financial crisis in three paragraphs.",
    "What are the trade-offs between LoRA rank 8 and rank 64?",
    "Translate the following into formal English: 'gonna be late, sry'.",
    "Given a table of quarterly revenue, describe how you would compute YoY growth.",
    "List five symptoms of vitamin B12 deficiency and their mechanisms.",
    "Write a SQL query to find the second-highest salary per department.",
    "Explain backpropagation through time to a software engineer.",
    "Draft a polite email declining a meeting invitation.",
    "What is the difference between precision and recall? When does each matter more?",
    "Describe how a transformer's attention mechanism scales with sequence length.",
    "Explain quantisation-aware training and the straight-through estimator.",
    "What legal risks arise from training on scraped web data?",
    "Write unit tests for a function that parses ISO-8601 timestamps.",
    "Compare REST and gRPC for an internal microservice architecture.",
]


def build_calibration(n: int, dataset: str | None, text_field: str = "text") -> list[str]:
    """Return calibration samples. Prefer your own data; fall back to the built-in set."""
    if dataset:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset, split="train")
            col = text_field if text_field in ds.column_names else ds.column_names[0]
            samples = [str(r[col]) for r in ds.select(range(min(n, len(ds))))]
            print(f"  calibration        {len(samples)} samples from {dataset}:{col}")
            print("  ⚠  Make sure this corpus resembles your PRODUCTION inputs, not "
                  "just generic web text.")
            return samples
        except Exception as e:                              # noqa: BLE001
            print(f"  ⚠  could not load {dataset} ({e}); using the built-in set")

    reps = (n // len(DEFAULT_CALIB)) + 1
    samples = (DEFAULT_CALIB * reps)[:n]
    print(f"  calibration        {len(samples)} built-in samples")
    print("  ⚠  These are generic English prompts. For production, pass "
          "--calib-dataset with data that matches your traffic.")
    return samples


def _vram_table(model: str) -> None:
    print(f"\n  ── VRAM by precision ({model}) ──")
    for label, bpp in [("fp16 / bf16", 2.0), ("8-bit", 1.0), ("4-bit", 0.5), ("2-bit", 0.25)]:
        gb = inference_gb(model, seq_len=4096, batch=1, weight_bytes_per_param=bpp)
        print(f"    {label:<14} {gb:>6.2f} GB   (weights + 4k KV cache + ~1GB overhead)")
    print("    Note: weights shrink linearly with bits; the KV cache does not.\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--method", required=False,
                   choices=["gptq", "awq", "bnb", "gguf"],
                   help="quantization method. HQQ is deliberately NOT offered: it was "
                        "previously listed but fell through to the AWQ branch, silently "
                        "writing an AWQ checkpoint into a directory named hqq-4bit.")
    p.add_argument("--out", default="./out/quantized")
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128,
                   help="GPTQ/AWQ group size. Smaller = better quality, more overhead. "
                        "128 is standard; 32 is better at 3-bit.")
    p.add_argument("--calib-samples", type=int, default=256,
                   help="128 minimum, 256-512 preferred. More is better but slower.")
    p.add_argument("--calib-dataset", default=None,
                   help="HF dataset to calibrate on. STRONGLY recommended.")
    p.add_argument("--quant-type", default="Q4_K_M",
                   choices=["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_M",
                            "Q6_K", "Q8_0", "F16"],
                   help="GGUF k-quant type. Q4_K_M is the quality/size sweet spot.")
    p.add_argument("--desc-act", action="store_true",
                   help="GPTQ act_order: quantize columns in order of decreasing "
                        "activation importance. Better quality, ~10%% slower.")
    p.add_argument("--eval-only", action="store_true",
                   help="Only report the VRAM table for the model; do not quantize.")
    a = p.parse_args()

    model_size = _sniff_size(a.model)
    _vram_table(model_size)

    if a.eval_only:
        return

    if not a.method:
        sys.exit("--method is required (gptq | awq | bnb | gguf).")

    # bitsandbytes only offers 4-bit and 8-bit. Without this guard, --bits 3 would
    # be interpolated into `load_in_3bit=True` and fail deep inside transformers.
    if a.method == "bnb" and a.bits not in (4, 8):
        sys.exit(f"--method bnb supports --bits 4 or 8, not {a.bits}.\n"
                 "  For 2/3-bit, use --method gguf (Q2_K/Q3_K) or --method gptq.")

    if a.method == "gguf":
        _run_gguf(a)
    elif a.method == "bnb":
        _run_bnb(a)
    else:
        _run_gptq_awq(a)


# --------------------------------------------------------------------------------------
def _sniff_size(model_id: str) -> str:
    """Pick the closest preset size from a model id, e.g. 'Qwen2.5-32B-Instruct' -> 32B.

    A plain substring test is WRONG here and silently so: "2b" is a substring of "32b",
    and "2B" appears earlier in the list, so `meta-llama/Llama-2-32b` sniffs as 2B and
    the VRAM table is printed for a model 16x smaller than the one requested. Match on
    a token boundary instead, and try the longest labels first so "1.5B" cannot lose to
    "1B".
    """
    import re
    sizes = ["0.5B", "1B", "1.5B", "2B", "3B", "7B", "8B", "13B", "14B", "32B", "70B"]
    haystack = model_id.lower()
    for label in sorted(sizes, key=len, reverse=True):
        # (?<![0-9.]) stops "2b" matching inside "32b" or "1.5b"; (?![0-9]) stops
        # "1b" matching the "1b" of a hypothetical "1b5".
        if re.search(rf"(?<![0-9.]){re.escape(label.lower())}(?![0-9])", haystack):
            return label
    return "7B"


# --------------------------------------------------------------------------------------
def _run_gptq_awq(a) -> None:
    """
    GPTQ and AWQ both:
      1. load the model in fp16
      2. run calibration samples through it to observe activations
      3. solve a layer-wise optimization to choose quantization parameters
      4. save a checkpoint in that format

    This needs the whole model in fp16 RAM/VRAM, so a 7B needs ~16GB free, and a 70B
    needs ~150GB. For very large models use `--method gguf` on CPU, or shard.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    calib = build_calibration(a.calib_samples, a.calib_dataset)
    out = Path(a.out) / f"{a.method}-{a.bits}bit"
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    print(f"  loading {a.model} in fp16 (this is the memory peak)...")
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    if a.method == "gptq":
        # GPTQ: second-order (Hessian-based) layer-wise quantization with error
        # compensation — the error introduced in one column is pushed into the
        # not-yet-quantized columns, so it does not accumulate.
        from optimum.gptq import GPTQQuantizer
        q = GPTQQuantizer(bits=a.bits, dataset=calib, group_size=a.group_size,
                          desc_act=a.desc_act, damp_percent=0.01, block_name_to_quantize=None)
        model = q.quantize_model(model, tokenizer=tok)
        q.save(model, tok, str(out))
    else:
        # AWQ: activation-aware. It observes activation magnitudes, identifies the
        # ~1% of weight channels that matter most, and protects them by scaling
        # rather than by keeping them in higher precision. Usually beats GPTQ on
        # instruction-tuned and multimodal models.
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_pretrained(a.model, **{"low_cpu_mem_usage": True})
        # calib_data MUST be passed. AutoAWQ falls back to its own generic English
        # corpus when it is omitted, so the "calibrate on YOUR OWN data" warning this
        # script prints would be a lie and the artifact would be tuned on the wrong
        # distribution — the exact benchmark-fine, production-bad failure the header
        # warns about. Note AutoAWQ was archived in 2024; for new work prefer
        # llm-compressor. Verify the calib_data argument against your pinned version.
        model.quantize(tok, quant_config={
            "zero_point": True, "q_group_size": a.group_size,
            "w_bit": a.bits, "version": "GEMM",
        }, calib_data=calib)
        model.save_quantized(str(out))
        tok.save_pretrained(str(out))

    print(f"\n  ✅ {a.method.upper()} saved to {out}")
    _post_quant_checks(a.model, str(out), calib[:16], a.method)


def _run_bnb(a) -> None:
    """bitsandbytes: no calibration, no conversion, quantize at load time."""
    print("  bitsandbytes quantizes on load — there is nothing to save.")
    print("  Use it directly:")
    print(f"""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    bnb = BitsAndBytesConfig(
        load_in_{a.bits}bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained("{a.model}", quantization_config=bnb)
""")
    print("  Trade-off: zero setup, but slower kernels than AWQ/GPTQ for serving, and")
    print("  it does not reduce disk size in a portable format.")


def _run_gguf(a) -> None:
    """
    GGUF via llama.cpp. This is the path to CPU, Apple Silicon, Ollama and LM Studio.

    The k-quant naming, decoded:
      Q4_K_M  = 4-bit, K-quant (block-wise with a learned scale/min per super-block),
                M = medium — a mixed scheme that keeps some tensors at higher precision.
      _S / _M / _L = small / medium / large: how much of the model is kept at higher
                precision. M is the usual default; L is closest to fp16.
      Q8_0    = essentially lossless; use when size does not matter.
      Q2_K    = do not use unless you are truly desperate; it is a visible cliff.
    """
    out = Path(a.out) / f"gguf-{a.quant_type}"
    out.mkdir(parents=True, exist_ok=True)
    f16 = out / "model-f16.gguf"

    print(f"  step 1/2  convert HF → GGUF (f16) ...")
    cmd1 = [sys.executable, "-m", "llama_cpp.convert_hf_to_gguf",
            a.model, "--outfile", str(f16), "--outtype", "f16"]
    if subprocess.run(cmd1).returncode != 0:
        sys.exit("Conversion failed. Install llama.cpp:\n"
                 "  pip install llama-cpp-python\n"
                 "  or clone https://github.com/ggerganov/llama.cpp and build it")

    print(f"  step 2/2  quantize → {a.quant_type} ...")
    # The importance matrix (imatrix) is optional but measurably improves low-bit
    # quality: it weights the quantization error by how much each activation matters,
    # using statistics from a calibration corpus.
    cmd2 = [sys.executable, "-m", "llama_cpp.llama_quant",
            str(f16), str(out / f"model-{a.quant_type}.gguf"), a.quant_type]
    proc = subprocess.run(cmd2, capture_output=True, text=True)
    if proc.returncode != 0:
        # Do NOT delete the f16 intermediate and do NOT claim success. The previous
        # version ran this with check=False, unlinked the f16, and then printed
        # "GGUF saved" unconditionally — so a failed quantize left an empty output
        # directory, no error, and a success message. The f16 is the expensive part;
        # keep it so the retry is cheap.
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        sys.exit(
            f"Quantization to {a.quant_type} failed (exit {proc.returncode}).\n"
            f"  The f16 intermediate is kept at {f16} — retry from there.\n"
            "  If the module is missing, llama-cpp-python may not ship a runnable\n"
            "  `llama_cpp.llama_quant`; use the llama.cpp binary instead:\n"
            "    llama-quantize model-f16.gguf "
            f"model-{a.quant_type}.gguf {a.quant_type}\n"
            + (f"\n  --- tool output ---\n{tail}" if tail else "")
        )

    f16.unlink(missing_ok=True)
    print(f"\n  ✅ GGUF saved to {out}")
    print("\n  Run it:")
    print(f"    ollama create mymodel -f - <<< 'FROM {out}/model-{a.quant_type}.gguf'")
    print(f"    # or:  llama-cli -m {out}/model-{a.quant_type}.gguf -p \"Hello\"")
    print("\n  ⚠  Known GGUF gotcha: the chat template lives in GGUF metadata. If it is")
    print("     missing or wrong, the model will produce degraded output with no error.")
    print("     Always test with the same prompts you used pre-quantization.")


def _post_quant_checks(orig_model: str, quant_path: str, prompts: list[str], method: str) -> None:
    """
    The comparison that matters: generate with the fp16 model and the quantized model
    on the SAME prompts and compare. Perplexity is a weak signal — a model can hold
    perplexity while losing instruction-following, JSON compliance and long-context
    recall. Always compare generations, not just loss.
    """
    print("\n  ── mandatory post-quantization validation ──")
    print("  Perplexity will look fine even when quality has dropped. Check these instead:")
    print("    1. INSTRUCTION FOLLOWING — do the same 50 prompts, compare outputs side by side")
    print("    2. FORMAT COMPLIANCE   — does it still emit valid JSON / your schema?")
    print("    3. LONG CONTEXT        — run a needle-in-a-haystack test at your max context")
    print("    4. CODE / MATH         — these degrade FIRST and most visibly")
    print("    5. KL divergence       — measure it against the fp16 model on held-out text:")
    print("""
    # KL between fp16 and quantized next-token distributions
    import torch, torch.nn.functional as F
    def kl_divergence(p_logits, q_logits):
        p = F.log_softmax(p_logits.float(), dim=-1)
        q = F.log_softmax(q_logits.float(), dim=-1)
        return F.kl_div(q, p, log_target=True, reduction="batchmean").item()
""")
    print(f"  Quantized model at: {quant_path}")
    print(f"  Compare against:    {orig_model}")
    print("\n  If KL > ~0.1 mean on held-out text, or format compliance drops >2%,")
    print("  go up one bit-width or reduce --group-size to 32.")


if __name__ == "__main__":
    main()
