# CH-11 — Quantization II: GPTQ, AWQ, GGUF Cheat Sheet

**One-line purpose:** pick the right quantized *format* for the machine that will run it, choose the calibration set that decides the quality, and know which artifact you are actually holding.
**Use when:** you have a bf16 model and a deployment target — a GPU behind vLLM/TGI, a CPU or Mac, an edge box, or a QLoRA training run — and you must decide what to produce.
**Do NOT use when:** you need the *theory* (scales, zero-points, `MSE = Δ²/12`, PTQ vs QAT, per-channel vs per-tensor, the outlier problem) — that is **CH-10**. This card is the practitioner's half: which format, which tool, which flag, what breaks.
**Companion:** `code/08_quantize.py` implements every branch of §3.

> **The one sentence that matters.** The format is chosen by **where the model runs**, not by which benchmark table looked best — and the calibration set, not the algorithm, is what decides whether the result is good.

> **The second sentence.** A QLoRA adapter and a GPTQ checkpoint are not two views of one model. They are different artifacts at different stages of a chain, and most production incidents are someone serving the wrong one.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **GGUF for CPU/Mac/edge. GPTQ & AWQ for GPU serving. NF4 for *training*.** | These are not interchangeable. Each is built for one runtime. |
| 2 | **bitsandbytes NF4 is a QLoRA format, not a serving format.** | No Marlin-class kernel — it is slow at inference; its job is to make a base model trainable. |
| 3 | **AWQ is W4A16.** "Activation-aware" describes the *statistic used*, not the deployed precision. | The activation-side scale is folded into the preceding op; the activation path is exact fp16. |
| 4 | **You cannot serve a QLoRA adapter as-is.** Merge (→ bf16, lossless) **then** quantise the merged model. | Two quantisations in the chain; the second one is the one you ship. |
| 5 | **The calibration set is the highest-leverage knob in the whole pipeline.** | 200 in-domain samples beats 10,000 generic ones. It is also the #1 cause of "benchmark fine, production bad". |
| 6 | **`group_size=128` is the default; 64 or 32 buy accuracy with real bytes.** | 128→32 costs **+0.75 bpw** (~0.66 GB on a 7B). "Smaller is better" is true for quality and false for everything else. |
| 7 | **`desc_act=True` is ~10% slower inference and some kernels refuse it.** | It reorders columns by Hessian diagonal; the permutation breaks the kernel's contiguous layout. |
| 8 | **4-bit does not use 4-bit tensor cores.** The kernel dequantises to FP16 and runs FP16 maths. | You saved **bandwidth**, not FLOPs. Expect ~2–3× at batch 1, not 4×. |
| 9 | **Perplexity is not a gate.** A model holds perplexity while losing JSON, maths and long-context recall. | Gate on mean KL, p95 KL and top-1 agreement against your own fp16 baseline. |
| 10 | **The KV cache does not shrink when you quantise weights.** | At 32k × batch 8 an 8B's cache is 8× its int4 weights. Quantise the KV cache separately (FP8, sm_89+). |

---

## 2. Core Formulas

Everything here is the *practitioner's* arithmetic — the size of the file you produce, whether the run fits, and whether the result is acceptable. The quantiser itself (`q = round(x/s) + z`, `Δ`, `MSE = Δ²/12`) is **CH-10 §2**.

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Weight size** | `GB = params × bpw / 8 / 2^30` | bpw = **effective** bits per weight | 7B, 4.5 bpw → `7e9 × 4.5/8 = 3.94 GB` |
| **Effective bpw (symmetric)** | `b + 16/g` | `g` = group size | 4-bit, `g=128` → `4 + 0.125 = 4.125` |
| **Effective bpw (asymmetric)** | `b + 32/g` | scale **and** zero-point | 4-bit, `g=128` → `4.25` |
| **Group metadata cost** | `16/g` bpw per table | — | 128→32 = `+0.75 bpw` = **+0.66 GB** at 7B |
| **`Q4_K_M` real bpw** | ~4.85, not 4.00 | k-quant block metadata is real bytes | 70B "4-bit" GGUF ≈ **42 GB**, not 35 GB |
| **KV cache** | `2 · L · h_kv · d_head · seq · batch · bytes` | one K + one V per layer | 8B GQA (32/8/128): **128 KB/token** fp16 |
| **GPTQ objective** | `argmin_Ŵ ‖WX − ŴX‖²_F` | layer **output** error, not weight error | — |
| **GPTQ Hessian** | `H = XᵀX` (`d_in × d_in`, once per layer) | second moment of calibration activations | 4096² fp32 = 67 MB |
| **GPTQ sensitivity** | `Δ_err ≈ (w_j − ŵ_j)² · H_jj` | per *column*. The square is load-bearing | `(−0.5)² × 56 = 14` |
| **GPTQ compensation** | `w_{k>j} -= δ_j · (H⁻¹)_{jk} / (H⁻¹)_{jj}` | `δ_j = w_j − ŵ_j` | pushes error into un-quantised columns |
| **AWQ scale** | `s_j = a_j^α / w_j^(1−α)`, `α ≈ 0.5` | `a_j = max|X_:,j|`, `w_j = max|W_:,j|` | applied as `W·diag(s)`, `X·diag(s)⁻¹` |
| **AWQ identity** | `W·X = (W·diag(s))·(diag(s)⁻¹·X)` | exact in fp16; folded back before export | zero inference overhead |
| **NF4 + double quant** | `4 + 8/64 + 32/16384 = 4.127 bpw` | block-64 absmax, itself block-256 int8 | 7B base = **3.61 GB** |
| **Hadamard range shrink** | `‖x‖∞ : ‖x‖₂ → ‖x‖₂/√n` | rotation spreads the outlier | `n=4096` → up to **64× smaller** range |
| **Roofline ridge point** | `intensity = 2·batch / bytes_per_weight` | H100 ≈ 295 FLOP/byte | fp16 memory-bound while `batch < 295`; int4 while `batch < 74` |
| **Quantise-run peak** | `params × 2 B + H buffers (~2–4 GB)` | the fp16 model must be resident | 7B → ~16–20 GB; 70B → ~150 GB |
| **Decode speed (batch 1)** | `∝ 1 / bytes_per_param` | memory-bound | fp16→int4 = ¼ the bytes → **~2–3×**, not 4× |
| **Aggregate throughput** | `∝ KV room → batch` | the freeing of weight memory | ~3–4× through concurrency |
| **Gate metrics** | `mean KL`, `p95 KL`, `top1 agreement` | vs **your own** fp16 baseline | ≤0.10 / ≤1.00 / ≥0.95 |

---

## 3. Decision Tree — choose the format from the deployment target

```
Q0. Will you TRAIN on top of this model?
├─ YES, and the bf16 base does NOT fit in VRAM
│     → QLoRA: bitsandbytes NF4 + double quant + paged AdamW + LoRA.
│       This is a TRAINING format. You do not serve it. Go to Q1 for the serving artifact.
├─ YES, and the bf16 base DOES fit
│     → plain LoRA on bf16. Do not quantise the base at all. (CH-13 §4)
└─ NO, inference only → Q1

Q1. Where does the model RUN?          ← this is the decisive question
│
├─ CPU / Apple silicon / edge / air-gapped / a single portable file
│     → GGUF + llama.cpp / Ollama / LM Studio
│       start Q4_K_M → Q5_K_M if RAM allows → Q6_K / Q8_0 if RAM is free
│       below 4-bit: build an imatrix FIRST (see Q3)
│       ⚠ verify tokenizer.chat_template survived the conversion
│
├─ GPU behind vLLM / TGI / SGLang  → Q2
│
├─ NVIDIA + you own the build + max throughput
│     → TensorRT-LLM engine (FP8 / INT8 / NVFP4). Most painful, fastest.
│
└─ Apple-native GPU with no GGUF step
      → MLX quantised (different weight layout from AWQ/GPTQ; re-quantise, never convert)

Q2. GPU serving — what is the binding constraint?
├─ CAPACITY (the model does not fit at bf16)
│     ├─ sm_80+ (A100/A10/3090/L4/L40S/4090/H100)
│     │     → W4A16: AWQ if quality-first, GPTQ if you need the widest ecosystem
│     │       kernel: Marlin (`group_size=128, sym=True` for the fast path)
│     └─ sm_75 (T4) → W4A16 with a Triton/ExLlama kernel (Marlin needs sm_80+)
│
├─ COMPUTE at high batch (batch ≥ 32) — weight-only gains collapse here
│     ├─ sm_89+ (L4/L40S/4090/H100) → FP8 W8A8 via llm-compressor  ← real compute win
│     ├─ sm_80  (A100)              → INT8 W8A8 + SmoothQuant
│     └─ sm_70  (V100)              → W8A8 only. No FP8 anywhere on this card.
│
└─ Nothing fits even at 4-bit
      → the answer is not a lower bit-width. Shard (TP), shrink the model
        (distillation — CH-08), or quantise the KV cache (Q4).

Q3. Is context length > 8k?
├─ NO  → stop here.
└─ YES → the KV cache is now a SEPARATE budget from the weights:
          ├─ sm_89+ → `--kv-cache-dtype fp8 --calculate-kv-scales`   (2× cache reduction)
          └─ sm_80  → NOT SUPPORTED. Cap `--max-model-len`, cap `--max-num-seqs`,
                       or add tensor parallelism instead.
          ⚠ NEVER drop K below FP8 without a needle test at depth 0.5L and 0.9L.
            K errors flip the attention argmax — a discontinuous failure.

Q4. Is the workload REASONING or CODE (tight accuracy tolerance)?
├─ Tolerant (chat, summarisation, short-doc RAG) → W4A16, group_size 128. Done.
└─ Tight (maths, code, multi-step tools, long-context retrieval)
      escalate IN THIS ORDER, stopping when the §8 gate passes:
        1. group_size 128 → 64                         (+0.25 bpw)
        2. AWQ → GPTQ with desc_act=True               (~10% slower inference)
        3. domain-matched calibration + imatrix         (512 samples, your traffic)
        4. rotation (QuaRot / SpinQuant)                (if you aspire to W4A4)
        5. W4A16 → W8A8 FP8                             (halves the saving, restores quality)
        6. QAT at the target bit-width                  (only if you OWN the model)
        7. accept W8A16 / bf16 and buy a bigger GPU

STOP CONDITIONS
  X1  You cannot state the gate threshold BEFORE quantising → you are shipping on vibes.
  X2  The served artifact was not re-evaluated after conversion/serialisation → you
      tested a different artifact than you shipped.
  X3  Calibration overlaps eval → your numbers are inflated. Hash both, assert disjoint.
  X4  You cannot reproduce the artifact from (config, calib hash, library version, GPU arch).
  X5  You are about to quantise KV below FP8 without a needle test.
```

---

## 4. Hyperparameter Quick Reference

### 4.1 GPTQ — the knobs that change your outcome

Fields below are the `QuantizeConfig` (gptqmodel) surface described in **CS-11 §7.1**. Verify names against your pinned version — this library's schema changes between minor releases.

| Param | Default | Sweep | Effect of getting it wrong |
|---|---|---|---|
| `bits` | 4 | 3, 4, 8 | 3 needs `group_size=32`; 2 is a cliff on most models; above 8 there is no benefit |
| `group_size` | 128 | −1, 32, 64, 128 | Quality vs metadata. **`-1` means per-channel, NOT per-tensor** — it is what the fastest Marlin path wants |
| `desc_act` | `True` (gptqmodel) / `False` (optimum) | True/False | True = better accuracy, ~10% slower inference, rejected by some kernel builds |
| `damp_percent` | 0.05 (gptqmodel) / 0.01 (optimum, `08_quantize.py`) | 0.01–0.2 | Too low → singular `H`, NaN or one destroyed layer. Too high → degrades toward plain RTN |
| `damp_auto_increment` | 0.01 | rare | The mechanism that saves a marginal layer; leave it |
| `sym` | `True` | True/False | False adds a zero-point: `+16/g` bpw ≈ **+110 MB on a 7B** and breaks the fast Marlin path |
| `true_sequential` | `True` | rare | Disabling is a speed hack with a quality cost |
| `lm_head` | `False` | keep False | The output projection sets the logit scale; quantising it costs disproportionate quality for ~1% memory |
| `mse` | 0 | ≥0 | MSE-minimising rounding; buys accuracy at quantisation-time cost |
| `quant_method` | GPTQ | GPTQ / GPTQv2 / EoRA / QQQ | GPTQv2 = better accuracy, more VRAM; QQQ = speed |
| `format` | GPTQ | GPTQ / Marlin / ExLlamaV2 | Match your serving kernel, or you pay a dequant penalty at load |
| `rotation` | None | RQ / QRQ | Only for W4A4-class work (§9, rotation row) |
| `mock_quantization` | False | — | Measure the predicted size before paying the GPU-hours |

### 4.2 AWQ — the knobs

| Knob | Typical | Effect |
|---|---|---|
| `w_bit` | 4 | Weight bits. The artifact is **W4A16** regardless |
| `q_group_size` | 128 | Same trade as GPTQ's `group_size` |
| `zero_point` | `True` | Asymmetric. AWQ's scale search relies on it |
| `version` | `"GEMM"` / `"GEMV"` | `GEMM` for batched serving; `GEMV` for single-stream low latency. Wrong choice = correct output at ~half throughput |
| `max_calib_samples` | **128–512** | **The highest-impact knob.** More domain-matched samples = better scales |
| `max_calib_seq_len` | match deployment (512–4096) | Calibrating at 128 tokens for a 32k deployment is a category error |
| `n_parallel_calib_samples` | 8–16 on a real GPU | Throughput only. `1` is the OOM-avoidance setting |

> **Correction:** the source video's AWQ practical uses **100 hand-written generic sentences ×10** as calibration, and its OOM fallback retries with *fewer* samples (**CS-11 §6.4**). The right move on OOM is to keep 128–512 samples and **shorten `max_calib_seq_len`** — dropping to 20 generic samples is how you get a model that benchmarks fine and answers badly. AWQ's own config here (`zero_point`, `q_group_size=128`, `w_bit=4`, `version="GEMM"`) is correct and matches **CS-11 §7.2**.

### 4.3 GGUF — decode the k-quant name, then pick

`Q< bits >_K_< S|M|L >` = block-wise quant, `K` = k-quant (nested scales per super-block), suffix = how much of the model is kept at higher precision. `_S` < `_M` < `_L`.

Allowed values in `code/08_quantize.py --quant-type`: `Q2_K, Q3_K_S, Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16` (default `Q4_K_M`). The full quality-vs-size ladder is **CH-10 §9** — the practitioner's rule is: **start `Q4_K_M`; below 4-bit build an imatrix first; `Q2_K` is a visible cliff, not a compression option.**

### 4.4 The calibration set — the real hyperparameter

| Property | Target | What it decides | Failure if wrong |
|---|---|---|---|
| **Domain** | Your production traffic, not WikiText/C4 | The activation ranges the scales are fit to | Benchmarks flat, production worse. The #1 incident in CS-11 §15 |
| **Size** | 128 minimum; **256–512 preferred** | Quality of `H` / `max|X_j|` estimates | Under-estimated max → clipping at inference |
| **Sequence length** | Deployment range (512–4096; 2048 is a good default) | Activation statistics depend on seq len | A 128-token calibration for a 32k deployment is a category error |
| **Disjointness** | Hash both sets, assert no overlap | Whether your gate measures quality or memorisation | A quantised model that *beats* fp16 — treat as a bug |
| **Shape** | Must contain the modalities you serve | VLMs need image-text pairs; text-only never exercises the projector | Field-level exact match 91.4% → 78.9% (CS-11 §15.3) |
| **Provenance** | A calibration corpus is *training data* | Licence/data-governance surface | PII in a customer corpus is a data-processing event |

`code/08_quantize.py` builds a 16-item generic English fallback (`DEFAULT_CALIB`) and prints a warning on every run that it is generic. It repeats that list to reach `--calib-samples`. **That fallback is a smoke test, not a calibration set** — the script says so in its own header.

### 4.5 bitsandbytes NF4 — for completeness, since it is *not* a serving format

`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`, `bnb_4bit_use_double_quant=True`. All four knobs, the `fp4` alternative and the block-size knob are tabulated in **CH-10 §4** — do not duplicate the reading. The practitioner's point: **NF4 has no artifact.** It quantises at load time, saves nothing on disk in a portable format, and has no Marlin-class kernel. It exists to make a base model small enough to attach adapters to.

### 4.6 The two knobs that are in no quantiser: batch size and context

| Deployment | Dominant memory | First lever | Second lever |
|---|---|---|---|
| ≤ 4k context | weights | W4A16 | — |
| 8k–16k | weights ≈ KV | W4A16 | FP8 KV |
| 32k | **KV** | FP8 KV | W4A16, cap `--max-model-len` |
| 128k+ | **KV by 4–32×** | FP8 KV + `--calculate-kv-scales` | tensor parallel, reduce batch |

| Peak batch | What weight-only quantisation buys |
|---|---|
| 1–8 (interactive) | The full win. Deeply memory-bound; ~4× fewer bytes → ~2–3× tokens/sec |
| ≥ 32–74 (throughput) | **Collapses toward 1×.** The kernel is now compute-bound on FP16 tensor cores — it still issues FP16 maths after dequantising |
| ≥ 74 | Weight quantisation alone can be **slower** than fp16. Add activation quantisation (FP8 W8A8 on sm_89+, INT8 on sm_80) |

---

## 5. Copy-Paste Code Snippets

### 5.1 GPTQ — the maintained path (gptqmodel)

```python
# pip install -v gptqmodel --no-build-isolation ; pip install "protobuf<6.30"
# NOTE: AutoGPTQ (auto-gptq, 0.7.1, Aug 2024, pinned to Python 3.11) is DEPRECATED.
#       Anything you find online saying `pip install auto-gptq` is at least two years stale.
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset

model_id   = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

# Calibration: YOUR domain. Never generic web text for a specialised model.
calibration = load_dataset(
    "allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train"
).select(range(1024))["text"]

quant_config = QuantizeConfig(
    bits=4,             # 2/3/4/8
    group_size=128,     # 128 = Marlin-friendly; 64 for a quality bump at +0.25 bpw
    desc_act=True,      # activation-order columns; better accuracy, ~10% slower
    damp_percent=0.05,  # Hessian damping; raise to 0.1-0.2 if a layer fails
    sym=True,           # symmetric → no zero-point → 0.25 bpw cheaper, Marlin fast path
    lm_head=False,      # never quantise the output projection
)
model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration, batch_size=500)   # batch_size = VRAM, not quality
model.save(quant_path)
```

### 5.2 AWQ — the maintained path (llm-compressor)

```python
# pip install llm-compressor
# NOTE: AutoAWQ was archived (11 May 2024) and is superseded by vLLM's llm-compressor.
from llmcompressor.modifiers.quantization import AWQModifier
from llmcompressor.transformers import oneshot
from llmcompressor import CompressionRecipe

recipe = CompressionRecipe(
    AWQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])
)
oneshot(
    model="meta-llama/Llama-3.1-8B-Instruct",
    recipe=recipe,
    dataset=calibration_dataset,          # 128-512 in-domain samples
    num_calibration_samples=512,
    max_seq_length=2048,
)
# `scheme="W4A16"` is the API telling you, in writing, that AWQ does not
# quantise activations. If you need that, you want W8A8 — a different family.
```

### 5.3 GGUF — convert, imatrix, quantise

```bash
# 1. HF -> GGUF F16 container (no quantisation yet). You cannot re-quantise from a k-quant.
python -m llama_cpp.convert_hf_to_gguf ./Llama-3.1-8B-Instruct --outfile model-f16.gguf --outtype f16

# 2. Importance matrix from YOUR corpus — the single biggest free quality lever in GGUF.
#    llama.cpp computes per-tensor importance (Sum X_j^2 style) from the text you give it.
llama-imatrix -m model-f16.gguf -f domain_corpus.txt -o model.imatrix -ngl 99 -c 512 --chunks 200

# 3. Quantise, WITH the imatrix.
llama-quantize --imatrix model.imatrix model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

### 5.4 bitsandbytes NF4 — the QLoRA base (a training artifact, not a serving one)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map="auto")
# ... attach LoRA, train (CH-13 §5) ...
# The base weights never move. The adapter is the only thing that learned.
```

### 5.5 The chain of artifacts — train → merge → quantise → serve

```bash
# STEP 1   train:      bf16 base + LoRA adapter          -> out/sft-lora/   (adapter only)
python code/01_sft_lora.py --data code/data/sample_sft.jsonl --out out/sft-lora

# STEP 2   merge:      adapter + base -> FULL bf16 weights
#          This is the dequantisation step. It is lossless for a bf16 LoRA,
#          and it is where a QLoRA run's base is restored to bf16.
python code/09_merge_and_export.py --adapter out/sft-lora --out out/merged

# STEP 3   quantise:   the MERGED bf16 model -> the serving format
python code/08_quantize.py --model ./out/merged --method awq  --bits 4 --group-size 128 \
       --calib-dataset ./data/domain.jsonl --calib-samples 256 --out ./out/serve
python code/08_quantize.py --model ./out/merged --method gguf --quant-type Q4_K_M --out ./out/serve

# STEP 4   serve:      the QUANTISED artifact
python code/15_serve_vllm.py --model ./out/serve/awq-4bit
```

**Which artifact am I holding?**

| Path looks like | It is | Serves? | Trains? |
|---|---|---|---|
| `out/sft-lora/` (small, `adapter_model.safetensors`) | a LoRA adapter | No — needs a base | It *is* the training output |
| `out/merged/` (full size, bf16 `*.safetensors`) | a merged full-precision model | Yes (expensive, slow) | Yes — this is the right thing to quantise |
| `out/serve/awq-4bit/` (`quantization_config`, `qweight`) | a quantised serving artifact | **Yes** | **No** |
| `out/serve/gguf-Q4_K_M/` (one `.gguf` file) | a GGUF container | Yes, in llama.cpp/Ollama | **No** |

### 5.6 The gate — run this before you ship anything

```python
import math, torch, torch.nn.functional as F

@torch.no_grad()
def kl_gate(fp16_model, quant_model, tokenizer, texts, max_len=2048, device="cuda"):
    """Mean/p95 KL, top-1 agreement and perplexity delta vs YOUR fp16 baseline."""
    kls, agrees, nll_f, nll_q, ntok = [], 0, 0.0, 0.0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 8:
            continue
        lf = fp16_model(ids).logits[:, :-1].float()
        lq = quant_model(ids).logits[:, :-1].float()
        pf, pq = F.log_softmax(lf, -1), F.log_softmax(lq, -1)
        kls.append((pf.exp() * (pf - pq)).sum(-1).flatten().cpu())   # KL(P_fp || P_q)
        agrees += (lf.argmax(-1) == lq.argmax(-1)).sum().item()
        tgt = ids[:, 1:]
        nll_f += F.cross_entropy(lf.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        nll_q += F.cross_entropy(lq.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        ntok  += tgt.numel()
    kl = torch.cat(kls)
    return {"mean_kl": kl.mean().item(), "p95_kl": kl.quantile(0.95).item(),
            "max_kl": kl.max().item(), "top1_agree": agrees / ntok,
            "ppl_delta_pct": 100 * (math.exp(nll_q/ntok) / math.exp(nll_f/ntok) - 1)}

# GATES:  mean_kl <= 0.10   AND   p95_kl <= 1.00   AND   top1_agree >= 0.95
#         AND   ppl_delta_pct <= 1.0   AND   your task metric within 2 points
# ALWAYS read p95 and max, not just the mean. mean_kl = 0.03 with max_kl = 47 on
# 200 positions is a broken model with a good average.
```

---

## 6. CLI Commands

```bash
# ── this handbook's script — the REAL flags (read from code/08_quantize.py) ──────────
python code/08_quantize.py --model meta-llama/Llama-3.1-8B-Instruct --eval-only
python code/08_quantize.py --model ./out/merged --method awq --bits 4 --group-size 128 \
       --calib-dataset ./data/domain.jsonl --calib-samples 256 --out ./out
python code/08_quantize.py --model ./out/merged --method gptq --bits 4 --group-size 128 \
       --desc-act --out ./out
python code/08_quantize.py --model ./out/merged --method gguf --quant-type Q4_K_M
python code/08_quantize.py --model meta-llama/Llama-3.1-8B-Instruct --method bnb --bits 4
#   --method choices: gptq | awq | bnb | gguf      --bits: 2 | 3 | 4 | 8
#   (no `hqq` — it is deliberately absent. It was once listed but fell through to the
#    AWQ branch, silently writing an AWQ checkpoint into a directory named `hqq-4bit`.)
#   --quant-type choices: Q2_K Q3_K_S Q3_K_M Q4_K_S Q4_K_M Q5_K_M Q6_K Q8_0 F16
#   --desc-act is a STORE_TRUE flag (default off). --calib-samples default 256.
#   ⚠ the script's own VRAM table (--eval-only) is the fastest way to sanity-check a fit.

# ── GPTQModel (the maintained GPTQ CLI) ─────────────────────────────────────────────
gptqmodel --model meta-llama/Llama-3.3-70B-Instruct \
          --bits 4 --group_size 128 --sym true --desc_act true \
          --damp_percent 0.01 --calibration_samples 512 --calibration_seqlen 2048 \
          --calibration_dataset production_prompts.jsonl \
          --output_dir ./Llama-3.3-70B-gptq-4bit-g128

# ── llama.cpp / GGUF ────────────────────────────────────────────────────────────────
python -m llama_cpp.convert_hf_to_gguf ./model-hf --outfile m-f16.gguf --outtype f16
llama-imatrix  -m m-f16.gguf -f domain_corpus.txt -o m.imatrix -ngl 99 -c 512
llama-quantize --imatrix m.imatrix m-f16.gguf m-Q4_K_M.gguf Q4_K_M
llama-cli    -m m-Q4_K_M.gguf -p "What is quantization?" -n 128 -ngl 99
llama-server -m m-Q4_K_M.gguf --port 8080 -ngl 99 -c 8192 -np 4     # OpenAI-compatible
llama-bench  -m m-Q4_K_M.gguf -p 512 -n 128 -ngl 99                 # real t/s numbers
llama-gguf   m-Q4_K_M.gguf | head -40                               # dump metadata
#   -ngl / n_gpu_layers = partial offload: move n blocks to GPU, leave the rest on CPU.
#   This is how a 70B runs on a 24 GB card at a few tokens/second.

# ── Ollama ──────────────────────────────────────────────────────────────────────────
ollama create mymodel -f Modelfile      # Modelfile: FROM ./m-Q4_K_M.gguf
ollama show mymodel                     # ← check the chat template survived
ollama run  mymodel

# ── vLLM serving — the flags that decide throughput and fit ─────────────────────────
vllm serve ./Llama-3.3-70B-gptq-4bit-g128 \
  --quantization gptq_marlin \        # or awq_marlin | fp8 | bitsandbytes | gguf
  --dtype float16 \                   # COMPUTE dtype. NOT the weight precision.
  --kv-cache-dtype fp8 \              # sm_89+ only; the highest-leverage single flag
  --calculate-kv-scales \             # calibrated KV scales instead of a running max
  --max-model-len 8192 \              # set to the REAL need, not the model's max
  --max-num-seqs 8 \                  # concurrency cap; drives the KV budget
  --gpu-memory-utilization 0.92 \     # leave ~8-10% for fragmentation
  --enable-prefix-caching             # big win for shared system prompts
#   `--quantization` selects the weight-loading scheme; `--dtype` selects the compute
#   dtype for non-quantised paths. Conflating them is how people mis-diagnose quality.

# ── merge (step 2 of the chain) ─────────────────────────────────────────────────────
python code/09_merge_and_export.py --adapter out/sft-lora --out out/merged
```

---

## 7. VRAM / Cost Calculator

**The full budget — not just the weights.** The weights table (all sizes × all precisions) and the KV-per-token table are **CH-10 §7**; the point here is that they are *separate terms that do not move together*.

```
TOTAL = weights                       params x bpw / 8
      + KV cache                      2 x L x h_kv x d_head x seq x batch x bytes
      + activation peak               PREFILL is the peak, not decode
      + CUDA context + cuBLAS workspaces
      + framework overhead            PyTorch allocator, ~10-15% fragmentation
      + training state                grads + optimizer + master weights (if any)
```

### 7.1 Does it fit? — worked from the binding constraint

| Scenario | Weights | KV cache | + overhead | Total / budget | Verdict |
|---|---|---|---|---|---|
| 70B, bf16, 1×A100-80, 8k×4 | 140.0 GB | 10.7 GB | ~1.5 GB | 152.2 / 73.6 GB | **Impossible** |
| 70B, W4A16, fp8 KV, 8k×4 | 35.0 GB | 5.4 GB | ~1.5 GB | **42.4 / 73.6 GB** | Fits, 30 GB spare |
| 8B, W4A16, 4k×8, 1×A10G-24 | 4.1 GB | 1.1 GB | ~1.5 GB | 6.7 / 22.1 GB | Fits easily |
| 3B, W4A16, 8k×32, 1×A10G-24 | 1.5 GB | **18.9 GB** | ~1.5 GB | 21.9 / 22.1 GB | Fits *barely* |
| 235B MoE, FP8, TP=8, 32k×20 | 235/8 GB | 12.9 GB | — | 44.6 / 73.6 GB | Fits |

**Read row 4 carefully.** It is a KV-cache problem disguised as a weight problem: a 3B spends 18.9 GB of a 22.1 GB budget on the cache and 1.5 GB on weights. Quantising the weights harder would change nothing. The levers are `--max-num-seqs`, `--max-model-len`, and FP8 KV.

**Read rows 1 and 2 together.** That is the entire business case for 4-bit in one comparison: 2.07× over budget → 58% of budget.

### 7.2 Will the *quantisation run* fit? (the constraint people forget)

GPTQ and AWQ load the model in **fp16** and keep Hessian/statistic buffers, so the peak is roughly `params × 2 B + 2–4 GB`.

| Model | fp16 resident | Plus buffers | Minimum realistic GPU |
|---|---|---|---|
| 1B / 1.1B | ~2.2 GB | ~1 GB | any 4–8 GB card, or CPU + GGUF |
| 7B / 8B | ~14–16 GB | ~4 GB | 24 GB (a 16 GB card is tight) |
| 13B / 14B | ~26–28 GB | ~6 GB | 40 GB |
| 32B | ~64 GB | ~10 GB | 80 GB |
| 70B | ~140 GB | ~10 GB | 80 GB + 128 GB host RAM, or shard, or GGUF on CPU |

`code/08_quantize.py` says this in its own docstring: *"This needs the whole model in fp16 RAM/VRAM, so a 7B needs ~16 GB free, and a 70B needs ~150 GB. For very large models use `--method gguf` on CPU, or shard."*

### 7.3 Quantisation-run cost (one-time)

| Method | 7B time | Hardware | Notes |
|---|---|---|---|
| GPTQ `desc_act=True` | 25–60 min | 1× A100 40 GB | scales with calibration size |
| GPTQ `desc_act=False` | 10–25 min | 1× A100 | faster, slightly worse |
| AWQ | 10–30 min | 1× A100 | faster than GPTQ |
| bitsandbytes NF4 | **~0** | any | at load time; nothing is written |
| GGUF `Q4_K_M` + imatrix | 5–15 min | CPU or GPU | the imatrix pass dominates |
| FP8 W8A8 (llm-compressor) | 20–40 min | 1× H100 / L40S | needs the FP8 recipe |
| **QAT** | **days–weeks** | 8–64 GPUs | 1–10% of pretrain tokens |

**The asymmetry to internalise:** PTQ costs *minutes* and buys 4×. QAT costs *days* and buys maybe 1–1.5 further bits. Only run QAT if you own the model and ship millions of copies.

### 7.4 What quantisation changes, and what it does not

| Property | Effect of 4-bit weights | Why |
|---|---|---|
| Weight bytes read per token | **÷4** | the format is the point |
| Decode speed, batch 1 | ×2–3, **not ×4** | compute still runs on FP16 tensor cores |
| Aggregate throughput | ×3–4 | concurrency: 4× the KV room → 4× the batch |
| Single-request latency, batch 1 | modest | the memory-bound read shrinks, kernel overhead does not |
| Prefill / TTFT on long context | small | prefill is compute-bound |
| **KV-cache capacity** | **unchanged** | unless the KV cache is separately quantised |
| Accuracy | monotonically **non-improving** | quantisation only destroys information |

### 7.5 Unit economics

`$/1M output tokens = $/hr ÷ (tok/s × 3600) × 1e6`. The **ratio between rows is stable; the absolute dollar figure is not** — rental prices move every year. Recompute with your own contract price (see **CH-19 §7** for the managed-endpoint equivalent, where the per-token price is set for you).

| Deployment | Aggregate throughput | $/1M output tokens |
|---|---|---|
| bf16 70B, 2×A100-80 (TP=2) | ~110 tok/s at batch 2 | the reference |
| W4A16 70B, 1×A100-80 | ~190 tok/s at batch 4 | **3.4× cheaper** |
| W4A16 70B + FP8 KV, batch 8 | ~320 tok/s at batch 8 | another **1.7× cheaper** |

The last row is the one teams leave on the table: the weight quantisation gave 3.4×, and quantising the *cache* of an already-quantised model gave another 1.7×.

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | First fix |
|---|---|---|
| Perplexity fine, JSON / code / maths broken | Perplexity is blind to sequence-level structure and the long tail | p95 KL on your schema prompts; constrained decoding; exclude `lm_head` |
| Benchmark fine, production answers are worse | **Calibration corpus ≠ production traffic** | Re-quantise on 128–512 of *your* samples; add an imatrix (GGUF) |
| Output fluent but ignores instructions (GGUF) | Chat template missing from the GGUF metadata | Dump the metadata; compare against `apply_chat_template`; re-convert |
| Quantised model **beats** the fp16 baseline | Calibration leaked into the eval set | Hash both sets, assert disjoint. Treat the result as a bug |
| 4-bit is *slower* than fp16 at high batch | Weight-only kernel is now compute-bound on FP16 tensor cores | Add FP8/INT8 activation quantisation, or serve fp16 |
| 4-bit is slower than fp16 at batch 1 | Wrong kernel selected | Check the kernel name; `gptq_marlin` / `awq_marlin`; AWQ `GEMM`↔`GEMV`; fully on GPU |
| Quality worse than the model card claims | 7B recipes do not transfer to ≤3B models | `group_size` 128 → 32–64; consider 5–6 bit; re-validate |
| Short prompts perfect, retrieval at 32k wrong | **Attention argmax flips** — a K-quantisation failure | Needle test at depth 0.5L/0.9L; per-channel K, per-token V; stop at FP8 KV |
| Model never stops, always hits `max_new_tokens` | Missing generation config / wrong EOS id in the quantised dir | Copy `generation_config.json` + tokenizer into the artifact |
| `--max-model-len` at startup OOM despite weights loading | vLLM sized the KV pool for the model's native context | Cap `--max-model-len` to the real need; cap `--max-num-seqs` |
| VRAM still OOMs after quantising the weights | The KV cache did not shrink | FP8 KV (sm_89+); reduce context/batch; recompute §7 |
| Vision model lost grounding | `vision_tower` / `multi_modal_projector` were quantised, or calibration was text-only | Add them to `ignore`; calibrate with image-text pairs |
| MoE quality dropped, KL looked fine | Quantisation error flipped **expert routing** | Measure router agreement rate (≥98%), not just output KL |
| One layer quantises badly / NaN | Ill-conditioned `H` | `damp_percent` 0.01 → 0.05–0.2; more calibration tokens |
| Two quantisations give different models | Non-associative FP sums across kernels/arch | Pin library + torch + CUDA + GPU arch; hash the output; treat as a new artifact |
| A rebuild six months later is worse | Tooling defaults moved; calibration corpus went stale | Re-run the gate on every rebuild; re-calibrate quarterly or on drift |
| Adapter trained on an NF4 base lost quality after merge | Merged into a *quantised* base rather than bf16 | Merge into bf16, **then** quantise (§5.5) |
| Published checkpoint is worse than your own | It was calibrated on the uploader's data | A locally quantised model with domain-matched calibration usually wins at the same bit-width |
| Fine-tuning a quantised checkpoint does nothing | No gradient path into integer codes | QLoRA over NF4; never fine-tune a GPTQ/AWQ/GGUF artifact |

---

## 9. Comparison Matrix

### 9.1 The format table — what each one is *for*

| | **bitsandbytes NF4** | **GPTQ** | **AWQ** | **GGUF k-quant** | **SmoothQuant / FP8 W8A8** | **QAT** |
|---|---|---|---|---|---|---|
| What it is | load-time RTN round-to-nearest | layer-wise 2nd-order PTQ | activation-aware PTQ | container format + block quantiser | outlier migration / FP8 PTQ | training-time simulation |
| **What it is FOR** | **QLoRA training** | universal 4-bit GPU serving | best 4-bit GPU quality | **CPU / Mac / edge / one file** | **throughput at high batch** | the last 1–1.5 bits, if you own the model |
| Scheme | W4A16 / W8A16 | W4A16 | W4A16 | W4A16 (CPU/Metal) | W8A8 / W4A8 | W2…W8 / A8 |
| Artifact? | **No** — quantises at load | Yes, safetensors + `quantization_config` | Yes, safetensors + `quantization_config` | Yes, a single `.gguf` | Yes, safetensors | Yes, a full checkpoint |
| Algorithm | none (RTN) | `H = XᵀX`, error compensation | `s_j = a_j^α/w_j^(1−α)`, ~1% salient channels | block-wise, nested scales, optional imatrix | `s_j = a_j^α/w_j^(1−α)` (same shape, different target) | fake-quant + STE |
| Calibration | none | yes, 128–1024 | **yes, sensitive** | optional but high-value (imatrix) | yes | n/a (real training data) |
| Quality at 4-bit | good | strong | strong, often slightly better on chat | very good | near-lossless (at 8-bit) | best |
| Time (7B) | 0 | 25–60 min | 10–30 min | 5–15 min | 20–40 min | days |
| Fast kernel | none | Marlin / ExLlamaV2 | Marlin / GEMM | CPU AVX2/AMX, Metal, CUDA | FP8 tensor cores | n/a |
| Runs on | CUDA only | CUDA | CUDA | **CPU, Apple, CUDA, Vulkan** | sm_80+ (FP8 needs **sm_89+**) | any (once converted) |
| Serves in vLLM | `bitsandbytes` (slow) | `gptq` / `gptq_marlin` | `awq` / `awq_marlin` | `gguf` (limited) | `fp8` | as FP8/INT8 |
| Best at batch | small | 1–8 | 1–8 | any (CPU) | **32+** | any |
| **Fine-tunable after?** | **QLoRA only** | **No** | **No** | **No** | No | n/a |
| Memory win at 4-bit | ~3.5× | ~3.3× in practice, not 4× | ~3.3× | ~3.3× (`Q4_K_M` ≈ 4.85 bpw) | 2× | same as PTQ at the same bits |
| Throughput win @batch 1 | weak | ~2–3× (Marlin) | ~2–3× | excellent on CPU/Metal | modest | same as PTQ |
| Status 2026 | mature, unchanged | mature (**AutoGPTQ dead → gptqmodel**) | mature (**AutoAWQ archived → llm-compressor**) | mature, format still churning | rising; the sm_89+ default | niche; **download it, don't run it** |
| Failure signature | slow at inference | kernel-sensitive; calibration-sensitive | calibration-sensitive | chat template / tokenizer mismatch | static activation scales clip outliers | you paid days for 1 bit |

**The two rows that decide most real choices.** *What it is FOR* and *runs on*. GPTQ and AWQ are the same deployment point reached by different maths; GGUF is a different machine entirely; NF4 is a different **phase** (training, not serving).

**Why "4-bit" is not 4× smaller.** Only the linear weights are quantised — embeddings and `lm_head` are usually excluded, RMSNorm and biases stay fp16, and the metadata is not free (`group_size=128` costs 0.25 bpw for scales, another 0.25 for zero-points if asymmetric). `Q4_K_M` is ~4.85 bpw. A 70B goes 140 GB → ~42 GB, which is **3.3×**, not 4×. And the KV cache, often the larger term at long context, does not shrink at all.

### 9.2 Deployment target → format (the one-line answer)

| Target | Format | Produce it with | Kernel |
|---|---|---|---|
| vLLM on sm_80+ (A100/A10/3090) | AWQ or GPTQ W4A16 | llm-compressor / gptqmodel | Marlin (`group_size=128`, `sym=True`) |
| vLLM on sm_89+ / sm_90, high batch | FP8 W8A8 (+ FP8 KV) | llm-compressor FP8 recipe | FP8 tensor cores |
| NVIDIA, own the build, max throughput | TensorRT-LLM engine | `trtllm-build` per GPU/TP config | TRT plugins |
| CPU / Apple / edge / one portable file | GGUF `Q4_K_M` (+ imatrix) | `convert_hf_to_gguf` → `llama-imatrix` → `llama-quantize` | AVX2/AMX / Metal |
| Apple-native MLX app | MLX group-wise quant | `mlx_lm.convert -q` | Metal |
| QLoRA training | NF4 + double quant (+ paged AdamW) | `BitsAndBytesConfig` | dequant-on-the-fly |
| Publish to the Hub | GPTQ **and** AWQ safetensors, plus GGUF `Q4_K_M`/`Q5_K_M` | one run per target | — |

### 9.3 Engine → accepted formats

| Stack | Accepts | Quantise in-stack? | Watch out |
|---|---|---|---|
| **vLLM** | gptq, awq, fp8, marlin, gguf (limited), bitsandbytes, compressed-tensors | via llm-compressor | `--quantization` ≠ `--dtype`; pick `*_marlin` explicitly |
| **TensorRT-LLM** | its own engine, built per GPU/TP config | yes | the engine is per-config; a rebuild per target |
| **SGLang** | the vLLM families via its own loaders | via llm-compressor | RadixAttention prefix caching |
| **TGI** | awq, gptq, eetq, bitsandbytes, fp8 | no | FP8 on sm_80 will not work |
| **llama.cpp / Ollama / LM Studio** | GGUF quant types only | yes (`llama-quantize`) | safetensors needs conversion first |
| **MLX** | MLX-native quant (a *different* layout) | yes | a GGUF is not an MLX quant |
| **PEFT / QLoRA** | NF4 / FP4 via bitsandbytes | n/a | never take a GPTQ/AWQ artifact as a training base |

**Three rules fall out:** one artifact per engine; the `quantization_config` block travels *with* the weights and is part of the artifact; and **re-quantise rather than convert** — there is no supported AWQ→GGUF, GPTQ→GGUF or GPTQ→MLX path. Every re-target is a fresh run from the bf16 base.

### 9.4 Rotation methods — the 2025–26 frontier, in one table

| Method | Mechanism | Bit-width it unlocks | Cost |
|---|---|---|---|
| SmoothQuant | migrate the outlier from activations into weights | W8A8 | minutes; folds into weights, zero inference overhead |
| QuaRot | Hadamard rotation of weights/activations/KV | W4A4, KV4 | needs kernel support or weight folding; R4 (post-RoPE Q/K) is **not** fusable |
| SpinQuant | **learned** orthogonal rotations on a small calibration set | W4A4 | more setup; reported ~45% smaller 4-bit accuracy gap than QuaRot |
| LLM.int8() | FP16 escape hatch for the ~0.1% of dims with magnitude > 6.0 | W8A8 | free (bitsandbytes); outliers emerge only above ~6.7B params |
| SpQR / SqueezeLLM | FP16 sparse outliers + codebook for the dense part | 3–3.5 bit | custom kernels |
| BitNet b1.58 | ternary weights, trained from scratch | 1.58 bit | not a PTQ method — you cannot apply it post hoc |

> **Correction:** the source video's own AWQ practical asserts AWQ is "post training quantization only", which is right, but the surrounding framing implies AWQ quantises activations (**CS-11 §17.4**). It does not: the deployed artifact is **W4A16**. "Activation-aware" names the *statistic* (`mean|X_j|` per input channel) used to choose the scale — the activation-side factor `diag(s)⁻¹` is folded into the preceding LayerNorm/Linear and is exact fp16 at inference. If you need quantised activations you are choosing W8A8, a different family. The instructor corrects himself later; the correction is the examinable fact.

> **Correction:** at [3:02:31] the instructor says *"GGML is not performing the quantization… GGML cannot perform the quantization"*, and at [3:15:17] describes the GGUF practical as *"we are not going to use any explicit quantization method like GPTQ, AWQ, QAT"* (**CS-11 §17**, Appendix A). **`llama-quantize` is a quantization method** — post-training (no gradients), weight-only (activations stay fp16/fp32), block-wise (16/32/256-weight blocks with quantised scales) and mixed-precision (the `_S`/`_M`/`_L` suffix selects which tensors stay higher precision). It is a *different family* from GPTQ/AWQ — no Hessian, no activation-aware scaling, no Cholesky — but a `Q4_K_M` file is a ~4.85-bit-per-weight quantisation of the fp16 base, not the unquantised model.

> **Correction:** at [11:18] the instructor says *"IN4 is taking one byte… four bit right"* and at [12:33] that 1.58-bit is *"0.1 byte"* (**CS-11 §15.2**). **4-bit weights occupy 0.5 byte; 8-bit weights are the ones that occupy 1 byte; 1.58 bits is 0.1975 bytes.** Budget with `bytes = bits/8 × params` and then add metadata: a "4-bit" 70B GGUF is **~42 GB, not 35 GB**, because `Q4_K_M` is ~4.85 bpw in practice.

> **Beyond the video:** the video never quantises a KV cache, never shows a serving flag, and never measures whether the quantised model is worse. Those three gaps are the difference between a model that loads and a model you can ship — §7.4 and §8 exist because of them.

---

## 10. Numbers To Memorize

| Number | Meaning |
|---|---|
| `4 / 4.25 / 4.5 / 5.0` bpw | effective bits at `g = -1 / 128 / 64 / 32` (asymmetric) |
| `+0.75` bpw | the cost of `group_size` 128 → 32 (= 0.66 GB at 7B) |
| `4.127` bpw | NF4 + double quantization (`4 + 8/64 + 32/16384`) |
| `3.61 GB` | a 7B base at NF4 + double quant |
| `~4.85` bpw | the real figure for `Q4_K_M`, not 4.00 |
| `3.3×` not `4×` | the honest size reduction at 4-bit |
| `128` | `group_size` default; also `−1` = per-channel, **not** per-tensor |
| `0.05` / `0.01` | `damp_percent` default (gptqmodel / optimum) |
| `~10%` | the inference cost of `desc_act=True` |
| `~1%` | the fraction of channels AWQ protects |
| `6.0` | LLM.int8()'s magnitude threshold for the FP16 escape hatch |
| `0.1%` | the fraction of dimensions that threshold extracts |
| `128 KB` / `320 KB` | KV per token, fp16 (8B GQA / 70B GQA) |
| `2×` | what FP8 KV buys; requires **sm_89+** |
| `~295` FLOP/byte | the H100 ridge point; int4 goes compute-bound at batch ~74 |
| `~2–3×` / `~3–4×` | batch-1 decode / aggregate throughput, honestly |
| `128–512` | calibration samples. In-domain. Disjoint from eval |
| `2048` | a good `max_calib_seq_len`; match your deployment context |
| `0.10 / 1.00 / 0.95` | the gate: mean KL / p95 KL / top-1 agreement |
| `> 0.1` mean KL | `08_quantize.py`'s threshold to go up a bit-width or drop to `group_size=32` |
| `~0.2–0.6` Δppl | the normal 3-bit cost; `1–5+` at 2-bit; ~0.05 at 4-bit |
| `25–60 / 10–30 / 0 / 5–15 min` | GPTQ / AWQ / bnb / GGUF-7B quantisation time |
| `7` | the manifest fields: config, calib hash, library versions, GPU arch, base revision, weights hash, gate result |

---

## 11. Common Errors And Their Exact Messages

Library names and messages in this space change between minor versions. Where a string is version-sensitive it is marked **⚠ verify against your pinned version** — do not copy a fix from a blog post without checking the library it was written for.

| Message / symptom | Meaning | Fix |
|---|---|---|
| `ValueError: ExllamaV2 cannot be used with this model` | The ExLlamaV2 kernel cannot fuse this checkpoint (usually `desc_act=True`) | `disable_exllamav2=True` / `use_exllamav2=False`, or re-quantise with `desc_act=False` |
| `ImportError: cannot import name 'AutoAWQForCausalLM' from 'awq'` | The research `auto-awq` package is installed, not `autoawq` | `pip uninstall auto-awq && pip install autoawq` — or better, move to `llm-compressor` (AutoAWQ is archived) |
| `NameError: name 'AutoGPTQForCausalLM' is not defined` | `auto-gptq` missing (or deprecated and absent on this Python) | Prefer `pip install gptqmodel`; AutoGPTQ 0.7.1 is pinned to Python 3.11 |
| `KeyError: 'qweight'` / `'qzeros'` on load | A GPTQ checkpoint loaded with an AWQ loader, or vice versa | Use the matching library; check `config.json` → `quant_method` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` after quantising | `group_size` does not divide the layer's input dimension | Pick a group size that divides all your layer widths (128 usually does) |
| `AssertionError: padding token must be set` during AWQ/GPTQ calibration | The tokenizer has no pad token | `tok.pad_token = tok.eos_token` |
| `RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'` | You are running a 4-bit/GPTQ path on CPU | Move to CUDA, or use GGUF — the CPU path |
| `llama_quantize: failed to load model` | The input GGUF is not f16/f32 — k-quants cannot be re-quantised from a k-quant | Always convert to `--outtype f16` first, then quantise |
| `error: unknown argument '--outtype'` | A pre-2024 `convert.py` against a post-2024 model (or vice versa) | Use `convert_hf_to_gguf.py` / `llama_cpp.convert_hf_to_gguf` |
| `RuntimeError: Expected all tensors to be on the same device` | Calibration batch on CPU while the model is on GPU | `.to(model.device)` the batch |
| `torch.cuda.OutOfMemoryError` during `model.quantize(...)` | The fp16 model + Hessian buffers do not fit | Bigger GPU, shard, `low_cpu_mem_usage=True`, or the CPU/GGUF path (§7.2) |
| `KeyError: 'q_proj'` during a QLoRA/merge step | `target_modules` names do not match this architecture | Print `model.named_modules()` and use the real names |
| `ValueError: Tokenizer class LlamaTokenizer does not exist` during GGUF conversion | The source repo needs `trust_remote_code` / converter flags | Convert from the original repo, not from an already-converted one |
| `UserWarning: TypedStorage is deprecated` **with wrong output** | A stale `bitsandbytes`/`torch` combination | Pin `bitsandbytes` to the version matching your CUDA/torch build |
| Marlin refuses the checkpoint (kernel error at load) ⚠ | `group_size` not in `{128, -1}`, or asymmetric weights | Re-quantise with `group_size=128, sym=True` |
| The engine loads the model and then runs *slowly*, no error | It fell back to a generic dequant kernel instead of Marlin | Set `--quantization awq_marlin` / `gptq_marlin` explicitly; pin the engine version |
| Engine tries to dequantise a module that was never quantised ⚠ | `modules_to_not_convert` exists only in the Python recipe and never reached `config.json` | Write the ignore list into `config.json`; verify it there |
| Garbage output on tokens added *after* quantisation | `resize_token_embeddings(n)` rows do not line up with the saved tokenizer | Assert `model.get_input_embeddings().weight.shape[0] == len(tokenizer)` on the **loaded quantised** model |
| Fluent, well-formed, completely wrong answers (GGUF) | Chat template missing or wrong; **no error is raised** | `llama-gguf … \| head -40`; compare the rendered prompt against `apply_chat_template` |
| `ggml_metal_init: failed` / `no kernel image is available` | The runtime was not built for this GPU arch | Rebuild with the right `-DCMAKE_CUDA_ARCHITECTURES`, or use the matching wheel |
| Fine-tuning a quantised checkpoint does nothing or destroys it | No gradient path into integer codes | QLoRA over NF4; never fine-tune GPTQ/AWQ/GGUF |
| `--kv-cache-dtype fp8` errors or silently falls back | FP8 KV needs **sm_89+** | Check `torch.cuda.get_device_capability()` before planning capacity |

---

## 12. Copy-Paste Starter Config

The default decision for a 7B-class model going to GPU serving: **AWQ W4A16, `group_size=128`, domain-matched calibration, and a real gate.** Edit the two marked lines and run it.

```bash
#!/usr/bin/env bash
# quantize_and_gate.sh — produce and validate one serving artifact.
# Requires: pip install llm-compressor transformers datasets
set -euo pipefail

# ── CHANGE THESE TWO ────────────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-3.1-8B-Instruct"   # ← 1. your bf16 base. NOT a quantised one.
CALIB="./data/domain.jsonl"                # ← 2. YOUR traffic: 128-512 rows, {"text": ...}
# ── CHANGE NOTHING BELOW FOR RUN ONE ────────────────────────────────────────────────
OUT="./out/awq-4bit"
N_SAMPLES=256
SEQ_LEN=2048

python - <<'PY'
import json, os
from llmcompressor.modifiers.quantization import AWQModifier
from llmcompressor.transformers import oneshot
from llmcompressor import CompressionRecipe
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL, CALIB, OUT = os.environ["MODEL"], os.environ["CALIB"], os.environ["OUT"]
N, SEQ = int(os.environ["N_SAMPLES"]), int(os.environ["SEQ_LEN"])

texts = [json.loads(l)["text"] for l in open(CALIB, encoding="utf-8")][:N]
assert len(texts) >= 128, f"need >=128 calibration samples, got {len(texts)}"

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="auto")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# W4A16 written explicitly. `ignore` is the whole technique for VLMs and MoE:
# add "vision_tower", "multi_modal_projector" and the router gate for those families.
recipe = CompressionRecipe(
    AWQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])
)
oneshot(model=model, recipe=recipe, dataset=texts,
        num_calibration_samples=N, max_seq_length=SEQ)

model.save_pretrained(OUT, save_compressed=True)   # writes quantization_config
tok.save_pretrained(OUT)                           # ← copying the tokenizer prevents a
                                                   #   silent prompt-format regression
print(f"saved {OUT}")
PY

# ── the gate: mean KL, p95 KL, top-1 agreement vs fp16, on YOUR held-out data ────────
#   PASS  if  mean_kl <= 0.10  AND  p95_kl <= 1.00  AND  top1_agree >= 0.95
#   plus a TASK gate: JSON-validity rate, IFEval, GSM8K, needle@0.5L and @0.9L.
#   Perplexity alone is not a gate. Compare to YOUR fp16 baseline, never to a paper.
python code/08_quantize.py --model "$MODEL" --eval-only     # the VRAM sanity table
echo "next: run the gate BEFORE shipping; then load-test at PRODUCTION batch size"
```

**Choosing differently — one-line edits:**

| Target | Change |
|---|---|
| GPTQ instead of AWQ | `gptqmodel` with `bits=4, group_size=128, desc_act=True, sym=True, lm_head=False` |
| CPU / Mac / edge | `convert_hf_to_gguf --outtype f16` → `llama-imatrix` → `llama-quantize --imatrix … Q4_K_M` |
| High batch (≥32) on sm_89+ | FP8 W8A8 via the llm-compressor FP8 recipe, **plus** `--kv-cache-dtype fp8` |
| 3-bit (memory-tight) | `group_size=32` and GPTQ with `desc_act=True` |
| Long context (32k+) | keep the weights as-is and add `--kv-cache-dtype fp8 --calculate-kv-scales` **first** |
| Quick experiment only | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` — no artifact |
| You will fine-tune | QLoRA (NF4 + LoRA). Not this script. Then merge → **then** run this script |

### The five checks before you quantise

```bash
# 1. What will actually be quantised? (embeddings, lm_head, vision tower, router)
python -c "
from transformers import AutoConfig; c = AutoConfig.from_pretrained('MODEL')
print(c.architectures, c.num_hidden_layers, getattr(c,'num_key_value_heads',None),
      getattr(c,'tie_word_embeddings',False))"
#   tied embeddings → lm_head IS the embedding matrix; quantising it hits rare tokens

# 2. Will the quantisation RUN fit?  (fp16 resident + buffers — §7.2)
python code/08_quantize.py --model MODEL --eval-only

# 3. Is the calibration corpus disjoint from the eval corpus?
python -c "
import hashlib
h=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
print('calib', h('data/domain.jsonl'), 'eval', h('data/held_out.jsonl'))"
#   if these match, you are measuring memorisation

# 4. Does the tokenizer carry the chat template?  (the #1 GGUF failure)
python -c "
from transformers import AutoTokenizer as A
t=A.from_pretrained('MODEL'); print(repr(t.apply_chat_template(
  [{'role':'user','content':'hi'}], tokenize=False)))"

# 5. What is the KV cache at MY deployment context?  (§2, CH-10 §7)
python -c "
L,h,d,seq,b=32,8,128,32768,8
print(f'{2*L*h*d*seq*b*2/2**30:.1f} GiB fp16  |  {2*L*h*d*seq*b*1/2**30:.1f} GiB fp8')"
```

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| The theory: scales, zero-points, granularity, PTQ vs QAT, the outlier problem | **CH-10 §2–§4** |
| The full derivations and the notebook reproductions | **CS-10 — Quantization I: Fundamentals** |
| GPTQ's Hessian arithmetic, AWQ's correction, KV-cache quantization, the precision lattice (W8A8…W4A4), QLoRA's memory algebra, serving flags, the honest evaluation protocol | **CS-11 §4 and §12** — the source of truth for this card |
| Train the model you are about to quantise, and the correct masking | **CH-13 / CS-13 — Instruction Fine-Tuning** |
| Understand what QLoRA is doing to the base you are about to merge | **CS-11 §4.11**; **CH-13 §4.2** (CS-23 planned, not yet written) |
| The other compression axis — a smaller function rather than a smaller encoding | **CH-08 / CS-08 / CS-09 — Knowledge Distillation** |
| Cost the deployment, or use a managed endpoint where the per-token price is fixed | **CH-19 §7 / CS-19 — Vertex AI & Gemini** |
| Serve the artifact and tune the flags that matter | `code/15_serve_vllm.py` |
| The runnable script implementing every branch of §3 | `code/08_quantize.py` |
| Merge the adapter before you quantise | `code/09_merge_and_export.py` |
| Practice being interviewed on this | **IQ-11** (Part-2 question set) |
