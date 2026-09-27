# CH-16 — Unsloth Cheat Sheet

**One-line purpose:** run the *same* LoRA/QLoRA SFT that CH-13 describes, but with hand-written
Triton kernels and a hand-derived backward pass for the LoRA graph, so it fits and finishes on
one small GPU.
**Use when:** single GPU, ≤14B, an architecture on Unsloth's supported list, and you are already
running — or about to run — a TRL `SFTTrainer`.
**Do NOT use when:** you need multi-node/FSDP training (CS-17 Axolotl, CS-15 LLaMA-Factory), a
bit-exact audited run, a no-code UI (CH-15), or a model family Unsloth has not written kernels
for — there the fallback is **silent** and you pay full price for zero speedup.

> **The one sentence that matters.** Unsloth is a **kernel-and-backward-rewrite layer** that sits
> under `transformers` + `peft` + `trl`; TRL still owns the training loop, and every published
> speed/memory ratio is measured against an **unconfigured** HF baseline. Quote the baseline or
> do not quote the ratio. (CS-16 §0, §4.7)

---

## 1. The 10-Second Summary

| # | Fact | Why | Baseline you must state |
|---|---|---|---|
| 1 | **"2× faster, 50–70% less VRAM" is vs an unconfigured HF stack.** | Eager attention, no packing, FP32 4-bit compute dtype, LoRA on `q,v` only, stock checkpointing. | `transformers`+`peft`+`bnb` at library defaults (CS-16 §4.7.1). |
| 2 | **Against a *tuned* FA2 + bf16 + packing + 7-module QLoRA baseline the honest delta is ~1.2–1.4× time and 15–30% VRAM.** | Most of the headline is configuration you can apply yourself, not kernels. | CS-16 §4.7.2, §13.1, §18.2. |
| 3 | **Unsupported architecture = a silent 1.0× run.** No warning, no error, loss falls normally. | Patches are per-family; `from_pretrained` never fails by design. | Assert `type(model.model.layers[0].self_attn).__module__` starts with `unsloth`. CS-16 §8.2. |
| 4 | **`lora_dropout` must be exactly `0.0`.** | Any non-zero value reverts to peft's generic autograd path — correct model, no fast path, often no warning. | CS-16 §6.4, §9.4 #2. |
| 5 | **`max_seq_length` is a load-time *model-shaping* argument, not a truncation window.** | It configures RoPE scaling and which attention kernel compiles. Changing it later changes the position encoding. | CS-16 §6.3, §10.1. |
| 6 | **`train_on_responses_only` masks by matching your template's literal marker strings.** | Wrong markers ⇒ masking silently does nothing (or everything). The loss curve looks *healthier*. | CS-16 §4.9; verify by decoding the supervised span, §5.2 below. |
| 7 | **`import unsloth` must be the first import.** | Some patches rebind symbols inside `transformers` at import time; stale bindings ⇒ a 1.3–1.5× run that looks fine. | CS-16 §4.1, §9.4 #14. |
| 8 | **`use_gradient_checkpointing="unsloth"` — a string, not `True`.** | `True` = stock HF checkpointing (slower, stores more); the string = Unsloth's selective variant and part of the long-context claim. | CS-16 §5.2 Correction, §10.2, §17.13. |
| 9 | **Never `merge_and_unload()` a model loaded with `load_in_4bit=True`.** | PEFT's generic merge does not reproduce Unsloth's NF4/double-quant dequant recipe; the merged model is 5–25% worse with no error. | `save_pretrained_merged(..., save_method="merged_16bit")`. CS-16 §6.10. |
| 10 | **Unsloth is not a trainer.** | Epochs, grad-accumulation, optimizer step, scheduler, logging are TRL's. It *does* own the fused loss and the LoRA backward. | CS-16 §4.3.3, §17.3. |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **4-bit weight memory** | `P × 4.5 bits / 8 / 1024³` GiB | `P` = params (NF4 + absmax + double-quant metadata) | 1.1B → 0.62 GiB; 8B → 4.5 GiB (CS-16 §11.1) |
| **16-bit weight memory** | `P × 2 bytes / 1024³` GiB | — | 8B bf16 → 16.0 GiB |
| **LoRA params per projection** | `r × d_in + d_out × r` | `A:[r,d_in]`, `B:[d_out,r]` | `q_proj` 2048×2048, r=32 → 131,072 |
| **Total adapter params** | `n_layers × Σ_p (r·d_in + d_out·r)` | p ∈ {q,k,v,o,gate,up,down} | TinyLlama r=32/7 modules → **25,231,360** |
| **Adapter fraction** | `adapter / total` | — | 25,231,360 / 1,100,048,384 = **2.294%** |
| **Bytes per *trainable* param** | `2 (adapter bf16) + 2 (grad) + 2 (adamw_8bit moments)` = **6 B** | 8-bit optimizer, no fp32 master | 25.2 M × 6 B ≈ 0.15 GB total |
| **Full-FT bytes/param** | `2 weights + 2 grad + 8 AdamW fp32 (m+v)` = 12 B; **14 B** with an fp32 master | — | CH-13 §7; 7B × 14 ≈ 91.6 GiB |
| **Attention score tensor** | `B × H × T² × 2 bytes` | per layer, per direction, naive | B=2, H=32, T=4096, fp16 → **2.15 GB (2 GiB)** |
| **Logits tensor** | `B × T × V × 2 bytes` | the tensor fused CE never writes | B=2, T=4096, V=32k → **524 MB**; V=152,064 → **2.49 GB** |
| **KV cache (inference)** | `2 × L × kv_heads × head_dim × T × B × bytes` | `kv_heads` ≠ `heads` under GQA | 8B, T=4096, B=1, fp16 → ~0.5 GiB |
| **Training FLOPs** | `6 × N × D` | fwd 2ND + bwd 4ND | 8B × 32 M tokens → 1.54e18 FLOPs |
| **Wall clock** | `FLOPs / (TFLOPs × 1e12 × MFU)` | MFU 0.35–0.45 tuned, 0.10–0.20 naive | 1.54e18 / (312e12 × 0.42) ≈ 11,750 s ≈ 3.3 h |
| **Effective batch (examples)** | `micro × accum × gpus` | — | 2 × 8 × 1 = 16 |
| **Steps per epoch** | `ceil(N / eff_batch)` | — | ceil(25,000/16) = 1,563 |
| **Masking** | `labels[i] = -100` on prompt tokens | `CrossEntropyLoss(ignore_index=-100)` | supervised frac = `n_sup / n_total` |
| **Initial loss sanity band** | `ln(V) × [0.7, 1.5]` | V = vocab | ln(32,000) ≈ 10.4 → expect 7–16 at step 0 |
| **Throughput** | `tokens/s = tokens_per_step / sec_per_step` | count **supervised** tokens for cross-config comparison | 4096×2×4 / 2.85 s ≈ 3,057 tok/s |
| **Packing efficiency** | `real_tokens / (steps × eff_batch × T)` | — | 376k real out of 6.16 M positions (CS-16 §11.5) |

**The one arithmetic fact that proves the fast path is on:** with eager attention at `B=2,
H=32, T=4096` a *single* score matrix is 2.15 GB. CS-16's measured peak for that config was
**1.9 GB total**. If your "Unsloth" run at T=4096 peaks at 6–9 GB, flash-style attention is not
engaged. (CS-16 §4.8.2)

---

## 3. Decision Tree

```
Is your model family on Unsloth's supported list?
├─ No / unsure → you will get a silent 1.0x run. STOP.
│     → assert the patched module (S5.3). If it says `transformers.*`, either
│       pick a supported family, or drop Unsloth and use CH-13 / CH-15 / CS-17.
└─ Yes ↓

How many GPUs?
├─ >1 (FSDP / tensor-parallel / multi-node) → NO. Use Axolotl (CS-17) or
│     plain torchrun+FSDP. Unsloth's manual backward assumes the full layer
│     is on-device. DDP works; FSDP is not the optimised path. (CS-16 §8.2)
└─ 1 ↓

Do you need bit-exact, audited reproducibility?
├─ Yes → NO. Fused kernels reassociate FP ops and Triton autotunes tiles per
│     device. Search with Unsloth, re-run the winner with deterministic
│     plain PyTorch. (CS-16 §9.3, §16.6)
└─ No ↓

Is nvidia-smi showing <30% utilisation, or step time flat across batch sizes?
├─ Yes → you are DATALOADER-bound. No kernel will help.
│     → num_proc=8 on dataset.map, dataloader_num_workers=4, pre-tokenize.
│       Fix this BEFORE adopting Unsloth. (CS-16 §8.2, §14.1 #7)
└─ No ↓

Does the run fit already?
├─ Yes, comfortably (e.g. 2B on a 4xA100 node) → plain HF + FA2 + packing
│     + adamw_8bit may be all you need. Measure before adopting.
└─ No / marginal ↓

You are the target user. Adopt Unsloth.
      → 4-bit load + "unsloth" gradient checkpointing + packing + fused CE.
      → Then spend the freed headroom on a BETTER run (longer T, bigger
        batch, bigger r, more epochs) — not just a faster one. (CS-16 §17.14)

Is your data 1k-10k rows?
├─ Yes → r=8-16, lr=2e-4 (LoRA) or 1e-4 (on top of an Instruct base).
└─ No → see the r-vs-data table in S4.4.

Do you have a held-out eval set and eval_strategy set?
└─ No → STOP. Build it first. Without eval you cannot tell a kernel win from a
      quality regression, and CS-16 §15.4 shows the eval config being worth
      more than the whole speedup. (CH-13 §12)
```

---

## 4. Hyperparameter Quick Reference

### 4.1 `FastLanguageModel.from_pretrained` — load-time knobs

| Param | Video / handbook value | Typical | Too high → | Too low → |
|---|---|---|---|---|
| `model_name` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (code/02 default) | any `unsloth/*-bnb-4bit` repo, or a plain HF id | — | — |
| `max_seq_length` | 2048 (code/02), 4096 (video) | **p99.5 of your token lengths, rounded up to 64** | Aggressive RoPE scaling degrades short-context quality; OOM at load | Long rows truncated — and SFT truncation cuts the *end of the answer* |
| `dtype` | `None` (auto) | `None`, or `torch.float16` on Pascal/Turing | — | — |
| `load_in_4bit` | `True` | `True` | — | `False` multiplies weight memory ~4× (1.1B → 2.2 GB, 8B → 16 GB) |
| `load_in_8bit` | not set | — | — | worse memory than 4-bit, no measurable accuracy gain |
| `full_finetuning` | not set | `False` | 8B full FT ≈ 160 GB of weights+optimizer (CS-16 §7.1) | — |
| `token` / `HF_TOKEN` | secrets, read scope | `hf_...` | — | 401 `Repository not found` |
| `trust_remote_code` | not set | `False` unless required | Executes Hub Python at load; pin `revision="<sha>"` | Model fails to load |

### 4.2 `FastLanguageModel.get_peft_model` — LoRA knobs

| Param | Handbook value (code/02) | CS-16 / video | Typical | Trap |
|---|---|---|---|---|
| `r` | 16 | 32 | 8–16 (<10k rows), 16–32 (10k–100k) | High `r` + high LR is the classic LoRA divergence |
| `lora_alpha` | 16 (**= r**) | 32 | `= r` (Unsloth's own examples) or `2r` (CH-13 §4.2) | `alpha/r` is an LR multiplier on the LoRA branch only |
| `lora_dropout` | 0.0 | 0.0 | **must be 0.0** | Non-zero ⇒ silent fallback to peft's slow path |
| `target_modules` | all 7 | all 7 | `q,k,v,o,gate,up,down` | `q,v`-only trains ~1/3 the parameters and a worse model |
| `bias` | `"none"` | `"none"` | `"none"` | — |
| `use_gradient_checkpointing` | `"unsloth"` | `False` in the video 🚩 | `"unsloth"` for T ≥ 2048 | `True` ≠ `"unsloth"`; `False` throws away the long-context claim |
| `random_state` | 3407 | 3407 | your project seed | Unseeded ⇒ non-reproducible adapters |
| `use_rslora` | not set | not set | `True` at `r ≥ 64` | Not a replacement for `lora_alpha`; alternative scaling |
| `loftq_config` | not set | not set | `{}` (off) | Quality-at-2-bit lever, not a speed lever |
| `modules_to_save` | not set | not set | `["score"]` for classifiers | Omit and the head stays frozen |

### 4.3 `SFTConfig` — training knobs

| Param | Video | Recommended | Why |
|---|---|---|---|
| `per_device_train_batch_size` | 2 | 1–4 at long T | Bounded by activation memory, not taste |
| `gradient_accumulation_steps` | 4 | so `micro × accum ∈ [16, 128]` | Effective batch is the number that matters |
| `num_train_epochs` | 1 🚩 | 1–3 | 1 epoch × 1,500 rows = 188 steps = a smoke test, not a fine-tune |
| `learning_rate` | 2e-5 🚩 | **2e-4** (LoRA), **1e-4** on an Instruct base | 2e-5 is ~10× low; the repo's own A/B notebook uses 2e-4 |
| `lr_scheduler_type` | default (linear) | cosine | More forgiving at short step counts; code/02 uses `linear` |
| `warmup_ratio` | 0.1 🚩 | 0.03 | 10% of a short run is 10% wasted |
| `weight_decay` | default 0.0 | 0.01 | The correct LoRA regulariser when dropout must be 0 |
| `optim` | `adamw_8bit` ✅ | `adamw_8bit` | Halves optimizer state; free at LoRA rank |
| `packing` | `True` ✅ | `True` | Big throughput win on short rows; changes what "epoch" means |
| `dataset_text_field` | `"text"` | `"text"`, or **omit** for a `messages` dataset | Setting both ⇒ unclear precedence or `KeyError: 'text'` |
| trainer-side length | not set 🚩 | set it explicitly, **equal to the loader's** | CH-13 §5.1 writes `max_length=`, CS-16 §6.7 writes `max_seq_length=` — TRL renamed it; a mismatch is error #29 in §11 below |
| `bf16` / `fp16` | not set | from `torch.cuda.is_bf16_supported()` | Must agree with the loader's `dtype` |
| `save_steps` / `save_total_limit` | not set | 200 / 3 | Unset ⇒ epoch-only saves; free Colab will disconnect mid-run |
| `eval_strategy` / `eval_steps` | not set 🚩 | `"steps"` / 100 + a held-out split | **No eval = no evidence at all** |
| `dataloader_num_workers` | not set (0) | 2–8 if dataloader-bound | #1 cause of "my run is slower than the benchmark" |

### 4.4 Rank vs data size (CS-16 §7.2)

| Examples | `r` | Trainable params (7-module Llama) | Note |
|---|---|---|---|
| < 1,000 | 4–8 | 3–6 M | — |
| 1,000–10,000 | 8–16 | 6–13 M | The video's run sits **outside** this band at r=32 |
| 10,000–100,000 | 16–32 | 13–25 M | Realistic production range |
| 100,000–1,000,000 | 32–64 | 25–50 M | `use_rslora` starts to matter |
| > 1,000,000 | 64–256 | 50–200 M | Consider full FT / continued pretraining (CH-12) |

**Sanity rule:** aim for **100–1,000 training examples per million trainable parameters**.
Below ~100 you are memorising; above ~10,000 you bought capacity you did not need. (CS-16 §7.2)

### 4.5 Tune in this order — the later knobs depend on the earlier ones (CS-16 §7.2)

```text
1. max_seq_length        from p99.5. Nothing else is tunable until peak memory is known.
2. r + target_modules    from task + data size. Set before LR: r decides how much LR is too much.
3. effective batch       micro x accum -> 16-128 examples/step. Raise micro to just before OOM.
4. learning_rate         ONLY NOW. 2e-4 LoRA; 1e-4 on an Instruct base. ~sqrt(r/16) if r changed.
5. epochs / steps        Watch EVAL loss; stop when it turns up.
6. packing + optim       Throughput knobs LAST - they change the step count, invalidating 3-5.
```

---

## 5. Copy-Paste Code Snippets

### 5.1 Canonical Unsloth SFT (runnable; the shape of `code/02_sft_unsloth.py --train`)

```python
import unsloth                      # MUST be first import (CS-16 S4.1)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

MODEL, MAX_LEN, SEED = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", 2048, 3407

model, tok = FastLanguageModel.from_pretrained(
    model_name     = MODEL,
    max_seq_length = MAX_LEN,   # model-shaping: RoPE scaling + kernel selection, at LOAD time
    dtype          = None,      # None = auto (fp16 on pre-Ampere, bf16 on Ampere+)
    load_in_4bit   = True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",   # a STRING. True = stock, slower, stores more.
    random_state=SEED,
)

# --- data: a `messages` column, rendered by the MODEL'S OWN template -----------
import json
rows = [json.loads(l) for l in open("data/sample_sft.jsonl", encoding="utf-8") if l.strip()]
msgs = [[{"role": "user",      "content": r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")},
         {"role": "assistant", "content": r["output"]}] for r in rows]
ds = Dataset.from_list([{"messages": m} for m in msgs])   # NOT dataset_text_field="text"

trainer = SFTTrainer(
    model=model,
    processing_class=tok,          # `tokenizer=` is deprecated in TRL >= 0.16
    train_dataset=ds,
    args=SFTConfig(
        output_dir="out/unsloth-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,      # effective batch 16
        num_train_epochs=2,
        learning_rate=2e-4,                 # LoRA LR. NOT 2e-5.
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        max_length=MAX_LEN,                 # must EQUAL the loader's max_seq_length
        packing=True,
        bf16=True,
        logging_steps=5,
        save_steps=200, save_total_limit=3,
        report_to="none",
    ),
)

# --- the critical step (see S5.2 before you trust it) --------------------------
trainer = train_on_responses_only(   # ASSIGN the return value
    trainer,
    instruction_part="<|im_start|>user\n",       # exact strings from THIS tokenizer
    response_part="<|im_start|>assistant\n",     # ChatML. Llama-3 differs - S5.2.
)
trainer.train()
model.save_pretrained("out/unsloth-lora")        # adapter only (~20-80 MB at r=16)
tok.save_pretrained("out/unsloth-lora")          # tokenizer WITH chat_template
```

### 5.2 Derive the masking markers, then prove the mask — never hard-code them

```python
# The single line that keeps the markers in sync when you swap models.
def response_marker(tok):
    probe = tok.apply_chat_template(
        [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}],
        tokenize=False, add_generation_prompt=False)
    return probe.partition("Y")[0].split("X")[-1]

response_part = response_marker(tok)
print("response marker:", repr(response_part))
# Llama-3.x  '<|start_header_id|>assistant<|end_header_id|>\n\n'
# Qwen2.5    '<|im_start|>assistant\n'
# Gemma-2    '<start_of_turn>model\n'
# Phi-3      '<|assistant|>\n'
# TinyLlama  '<|assistant|>\n'

trainer = train_on_responses_only(trainer, response_part=response_part)

# --- verify on the REAL batch, before spending GPU hours ----------------------
import torch
batch  = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"][0]
sup = (labels != -100).sum().item(); tot = labels.numel()
assert sup > 0,   "NOTHING supervised -> marker not found, or the mask is inverted"
assert sup < tot, "EVERYTHING supervised -> the patch was a no-op"
assert 0.02 < sup / tot < 0.95, f"implausible mask ratio {sup/tot:.1%}"
print("SUPERVISED SPAN:", tok.decode(batch["input_ids"][0][labels != -100],
                                     skip_special_tokens=False)[:400])
# PASS: the assistant's answer + its terminator, and nothing else.
# If the user's question appears -> wrong marker. If the first word is cut -> off-by-one.
# With packing=True, run this check on the PACKED batch and check every row in the pack.
```

> **Beyond the video:** CS-16 §4.9.2 reconstructs `train_on_responses_only` as string surgery
> (`full.find(response_part)` on the rendered text). The shipped implementation matches the
> markers against tokenized `input_ids` instead, and newer versions take a `force_match=`
> argument (default `True`) that **raises** when a marker is not found. Both mechanisms agree on
> the operational rule — *the marker must match your chat template character for character* —
> but the failure mode differs by version: a raise (loud) or a no-op (silent). Verify the
> mechanism against your pinned version; verify the **effect** with the assertion block above,
> which is version-independent.

### 5.3 Prove you are actually on the fast path (the silent-1.0× check)

```python
import unsloth
from unsloth import FastLanguageModel
m, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=2048, load_in_4bit=True)

attn_mod = type(m.model.layers[0].self_attn).__module__
rope_mod = type(m.model.layers[0].self_attn.rotary_emb).__module__
print("attn impl :", attn_mod)   # 'unsloth.models.llama' -> fast path
print("rope impl :", rope_mod)   # 'transformers.models...'  -> STOCK, no speedup
assert "unsloth" in attn_mod, f"FAST PATH NOT ACTIVE ({attn_mod!r}) - unsupported arch?"
assert "unsloth" in rope_mod, f"RoPE NOT FUSED ({rope_mod!r})"

print("compute dtype:", next(m.parameters()).dtype)   # fp32 here = bnb compute dtype unset
print("trainable %:", 100 * sum(p.numel() for p in m.parameters() if p.requires_grad)
                       / sum(p.numel() for p in m.parameters()))
# Bands for 7 modules: r=16 -> ~1.0%, r=32 -> ~2.3%, r=64 -> ~4.5% of a 1.1B.
```

### 5.4 Save / merge / export — the four paths

```python
# A. adapter only (the source of truth; ~20-80 MB at r=16)
model.save_pretrained("out/adapter"); tok.save_pretrained("out/adapter")

# B. merged FP16 (one deployable artefact; the ONLY safe merge from a 4-bit base)
model.save_pretrained_merged("out/merged", tok, save_method="merged_16bit")

# C. merged 4-bit (lossy TWICE: base quant error + merge rounding + re-quantisation)
model.save_pretrained_merged("out/merged4", tok, save_method="merged_4bit")

# D. GGUF for llama.cpp / Ollama / LM Studio
model.save_pretrained_gguf("out/gguf", tok, quantization_method="q4_k_m")
```

> **Verify against your pinned version:** CS-16 uses **both** `save_method="merged_4bit"`
> (§6.9) and `save_method="merged_4bit_forced"` (§18.11, §19 answer 6) for the same operation.
> Do not assume which your version accepts — check `help(model.save_pretrained_merged)`.

**Run after every merge** (CS-16 §6.10): load base+adapter and the merged model in the *same*
dtype, compare last-position logits. `max |dlogit| < ~0.2` and matching argmax = a faithful
merge; `> 1.0` = wrong scaling/quantisation. Never ship a merge you have not run this on.

### 5.5 The honest timing harness (fixes the video's own measurement)

```python
import time, torch
torch.cuda.synchronize()
for _ in range(3):                              # absorb Triton JIT + cuBLAS autotune
    b = next(iter(trainer.get_train_dataloader()))
    with torch.no_grad(): model(**{k: v for k, v in b.items() if k != "labels"})
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()            # reset AFTER warm-up, not at notebook top

t0 = time.time(); trainer.train(); torch.cuda.synchronize()   # synchronize is MANDATORY
wall = time.time() - t0
steps = trainer.state.global_step
print(f"{wall:.1f}s  {steps} steps  {wall/steps:.3f} s/step  "
      f"{torch.cuda.max_memory_reserved()/1024**3:.2f} GiB peak")
```

### 5.6 A fair A/B — fix the baseline before you blame the kernels

The CS-16 §4.7.3 baseline had **four** defects that each inflate the "Unsloth is N× faster"
number. Match these before measuring:

| Fix in the HF arm | Effect if you don't |
|---|---|
| `bnb_4bit_compute_dtype=torch.bfloat16` (or fp16 on T4) | bnb defaults this to **fp32**; every dequantized matmul runs at 1/32 of T4 fp16 rate |
| `bnb_4bit_use_double_quant=True` | Different quantisation recipe than the Unsloth arm |
| `attn_implementation="flash_attention_2"` (or `sdpa` where FA2 is unavailable) | Eager attention: 2.15 GB *per layer* at T=4096 |
| Same `target_modules` (all 7) and same `packing` | You measured the config, not the engine |
| Time the same scope (`from_pretrained` inside or outside the clock, both arms) | The repo's own notebooks time different scopes — §9.4 #14 |

---

## 6. CLI Commands

```bash
# ── Install (CS-16 §6.1 - the video's exact pins; Unsloth is version-brittle) ──
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
pip install unsloth
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2      # --no-deps is deliberate: stops pip upgrading transformers
# For production, pin EVERYTHING (unsloth, transformers, trl, peft, torch, bitsandbytes)
# and record the pins + CUDA + driver in the model card (CS-16 §16.2). These pins are a
# point-in-time snapshot - check them against your driver and CUDA before use.

# ── Verify the stack (5 seconds; put this in CI) ──────────────────────────────
python -c "import torch,transformers,trl,peft,unsloth; \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print(transformers.__version__, trl.__version__, peft.__version__)"

# ── This handbook's scripts ──────────────────────────────────────────────────
python code/02_sft_unsloth.py --dry-run --data code/data/sample_sft.jsonl
python code/02_sft_unsloth.py --data code/data/sample_sft.jsonl --out out/unsloth-lora
python code/02_sft_unsloth.py --data code/data/sft.jsonl --model Qwen/Qwen2.5-7B-Instruct \
    --max-seq-len 4096 --r 32 --no-4bit --merge
python code/02_sft_unsloth.py --data code/data/sft.jsonl --full-finetune --lr 2e-5
python code/common/memory.py --model 7B --method qlora --seq-len 2048 --batch 2 --gpus 1
python code/common/memory.py --model 8B --method qlora --seq-len 2048 --batch 2 \
    --optimizer adamw_8bit --tokens 32000000 --gpu-type A100-80
python code/common/memory.py --table
python code/09_merge_and_export.py --base <model_id> --adapter out/unsloth-lora \
    --out out/merged --dtype bf16 --verify
python code/09_merge_and_export.py --base <model_id> --adapter out/unsloth-lora \
    --out out/merged --gguf Q4_K_M
python code/15_serve_vllm.py --model out/merged --benchmark
python code/15_serve_vllm.py --model <base> --adapter out/unsloth-lora --serve --port 8000

# ── Inspect the chat template BEFORE training (do this every time - CS-16 §4.10) ──
python -c "
from transformers import AutoTokenizer as A; t=A.from_pretrained('MODEL')
print(t.chat_template)
print(repr(t.apply_chat_template([{'role':'user','content':'hi'}]  , tokenize=False)))
print(repr(t.apply_chat_template([{'role':'user','content':'hi'},
                                  {'role':'assistant','content':'yo'}], tokenize=False)))"

# ── Stale Triton cache after a torch upgrade (CS-16 §14.1 #25) ────────────────
rm -rf ~/.triton/cache
```

> **Correction:** as of this writing, `python code/02_sft_unsloth.py --dry-run` — the exact
> command CH-13 §6 tells you to run — **crashes immediately**:
> `TypeError: TrainPlan.__init__() got an unexpected keyword argument 'size'` (verified by
> running it). `code/common/memory.py`'s `TrainPlan` has fields
> `model, method, seq_len, batch, grad_accum, ...` — there is no `size` and no `batch_size` —
> and the next line calls `_print_plan(plan, full_finetune=..., lora=...)` while `_print_plan`
> takes exactly one argument. `method` is also a required positional argument that is never
> passed. Until that is fixed, use the `--dry-run` checks in CH-13 §12 instead, and treat the
> plan/pre-flight block of `code/02_sft_unsloth.py` as untested. The training path
> (`_train`) is separate and unaffected.

---

## 7. VRAM / Cost Calculator

### 7.1 The arithmetic floor — `python code/common/memory.py --table`

Single GPU, gradient checkpointing **on**, AdamW, bf16 weights, fp32 optimizer states for full
FT. Units are **GiB** (1024³) although the tool prints "GB" (CS-16 §11.1 does the same —
see CH-13 §7 for the GiB-vs-GB and bytes-per-param reconciliation):

| Model | Full FT | LoRA r16 | QLoRA r16 | Infer bf16 | Infer 4-bit |
|---|---|---|---|---|---|
| 0.5B | 6.6 | 1.1 | 0.4 | 2.0 | 1.2 |
| 1B | 14.5 | 2.4 | 0.9 | 3.1 | 1.5 |
| 1.5B | 19.7 | 3.2 | 1.1 | 3.9 | 1.7 |
| 3B | 39.4 | 6.4 | 2.2 | 7.0 | 2.5 |
| 7B | 91.6 | 14.7 | 4.9 | 16.0 | 4.8 |
| 8B | 104.7 | 16.7 | 5.6 | 16.4 | 4.9 |
| 13B | 170.0 | 27.1 | 9.0 | 28.3 | 7.8 |
| 32B | 417.9 | 66.2 | 21.5 | 61.6 | 16.2 |
| 70B | 913.8 | 144.5 | 46.8 | 132.6 | 33.9 |

`memory.py` also has presets the `--table` does not print: **2B, 14B, 405B**.
**Add 10–20% for allocator/framework overhead, and leave headroom.**

### 7.2 Unsloth planning numbers (CS-16 §11.2) — 4-bit QLoRA, batch 1, `"unsloth"` GC, fused CE

**Worked estimates calibrated to one measured point (TinyLlama, T=4096, 1.9 GB) — planning
numbers, not measurements.**

| Model | Params | 4-bit weights | Adapter+opt r=16 | T=1024 | T=2048 | T=4096 | T=8192 |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 1.1 B | 0.62 | 0.04 | 1.0 | 1.3 | **1.9** ✅ measured | 3.2 |
| Llama-3.2-1B | 1.2 B | 0.68 | 0.05 | 1.1 | 1.4 | 2.0 | 3.4 |
| Llama-3.2-3B | 3.2 B | 1.8 | 0.11 | 2.6 | 3.4 | 5.0 | 8.4 |
| Phi-3-mini-4k | 3.8 B | 2.2 | 0.13 | 3.0 | 3.9 | 5.8 | 9.6 |
| Qwen2.5-7B | 7.6 B | 4.3 | 0.24 | 5.4 | **7.0** | **10.4** | 17.3 |
| Llama-3.1-8B | 8.0 B | 4.5 | 0.25 | 5.6 | **7.3** | **10.9** | 18.1 |
| Mistral-7B-v0.3 | 7.2 B | 4.1 | 0.23 | 5.2 | 6.8 | 10.0 | 16.6 |
| Gemma-2-9B | 9.2 B | 5.2 | 0.29 | 6.4 | 8.4 | 12.4 | 20.6 |
| Qwen2.5-32B | 32.5 B | 18.5 | 1.0 | 21 | 25 | 33 | 56 |
| Llama-3.1-70B | 70 B | 40.0 | 2.2 | 44 | 52 | 70 | — |

**Read it like this:** weights and optimizer state are fixed; only the activation term scales
with batch. Batch 2 at T=4096 on Llama-3.1-8B ≈ `4.5 + 0.25 + 2 × (10.9 − 4.75) ≈ 17 GB`.

**Reconcile with §7.1:** the floor prices 7B QLoRA at **4.9 GiB** and CS-16 prices Qwen2.5-7B at
T=2048, batch 1 at **7.0**. Both are right — the floor prices tensors only; CS-16 adds the
CUDA context, the fused-CE working set, packing and allocator fragmentation. §7.1 is the lower
bound you must beat; §7.2 is what to plan with.

### 7.3 The delta that actually matters (CS-16 §11.3) — Llama-3.1-8B, r=16, batch 1, BF16

| Configuration | T=1024 | T=2048 | T=4096 | T=8192 |
|---|---|---|---|---|
| Naive HF: eager attention, no GC, **FP32 compute dtype** | OOM | OOM | OOM | OOM |
| HF: eager attention, FP32 compute, GC on | 9.8 | 12.5 | 19.4 | OOM |
| HF: FA2, BF16 compute, GC on | 6.1 | 7.9 | 11.6 | 19.0 |
| HF: FA2, BF16 compute, GC on, **packing** | 6.1 | 7.9 | 11.6 | 19.0 |
| **Unsloth: fused kernels, `"unsloth"` GC, fused CE** | **5.6** | **7.3** | **10.9** | **18.1** |
| **Unsloth's own contribution** | **−8%** | **−8%** | **−6%** | **−5%** |

**At 8B against a correct baseline, Unsloth's VRAM advantage is single-digit percent.** The
dramatic "50–70% less" lives in the first two rows — the gap between *unconfigured* and
*configured*, which is a FlashAttention + bf16 + packing story you can have without Unsloth.

> **Correction:** CS-16 §11.2's ✅ marks the TinyLlama T=4096 cell **1.9 GB "measured"**, but
> that measurement was taken at **batch 2, r=32, packing on**, while the table's header states
> **batch 1, r=16**. It is a calibrated anchor point, not a row of the table. Treat every
> non-✅ number in §7.2 as an estimate and measure your own.

### 7.4 Cost — the real message is that experiments get cheap

| Scenario | Config | GPU-hours | Cost |
|---|---|---|---|
| The video's run | TinyLlama 1.1B, T=4096, 1,500 rows, 1 epoch | **0.149** (535 s, measured) | **$0.02–0.06** on a T4; ~$0.25 on an A100 |
| Smoke test | 500 rows, 20 steps, T=1024 | 0.03 | $0.05 |
| Production 8B, 25k rows, 2 epochs | T=2048, batch 2×8, A100-80 | 3.3 | $5.61 (@ $1.70/h) |
| Same, **without packing** | T=2048, batch 2×8 | ~11.5 | $19.55 |
| Same, **without FA2** (eager attention) | T=2048, batch 2×8 | ~4.3 *if it fits* | $7.31 |
| Same, with FP32 4-bit compute | T=2048, batch 2×8 | ~11–16 | $19–28 |
| 70B, 25k rows, 2 epochs | T=2048, batch 1×16, 2×A100-80 | 2 × 26 | $88 |
| DPO/ORPO on top of the 8B SFT | T=1024, 10k pairs, 1 epoch | 4.5 | $7.65 |

Rental bands (CS-16 §11.1): T4 free (Colab/Kaggle), L4 $0.50–0.80/h, A100-40 $1.20–1.80/h,
A100-80 $1.50–2.50/h, H100 $2.20–4.00/h, RTX 4090 $0.35–0.60/h. `memory.py --gpu-type` knows
T4, A10G, L4, L40S, A100-40, A100-80, H100, H200, B200. **MFU is the dominant unknown:** at 35%
versus 15% MFU, cost changes by 2.3×.

### 7.5 Long context — the video's table, corrected (CS-16 §11.4)

| VRAM | Video claims (Llama-3.1-8B) | Plausible reality | Note |
|---|---|---|---|
| 8 GB | 3,000 tokens | **Does not fit** (4.5 GB weights + ~1.5 GB overhead) | Use a 1B–3B at 8 GB |
| 12 GB | 21,000 | 2,000–6,000 | Very sensitive to batch / GC mode / RoPE scheme |
| 16 GB | 40,000 | 4,000–10,000 | T=8192 is comfortable; 40k is not |
| 24 GB | 78,000 | 8,000–16,000 | A 4090/3090; T=8192 works well |
| 80 GB | 340,000 | 32,000–65,000 | Past the native window you pay a real short-context quality cost |

There is **no 28,000-token limit in Hugging Face `transformers`** — CS-16 §4.4.2 corrects this
explicitly; sequence length is bounded by VRAM, attention implementation, dtype and
checkpointing. A 12× gap at identical hardware is a *configuration* gap.

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| Loss starts near 0.0 | Mask inverted — you are supervising the *input* | Decode the supervised span (§5.2) |
| Loss starts 2–4× lower than expected and falls fast | `train_on_responses_only` is a no-op: prompt tokens are in the loss | Marker mismatch (§5.2) |
| Loss is exactly `ln(V)` ≈ 10.4 and never moves | Broken forward, wrong tokenizer, or a template that renders nothing learnable | Print `tokenizer.chat_template`; decode a batch |
| Loss flat slightly below `ln(V)` | Mask supervises almost nothing | Print the supervised fraction |
| Loss plateaus high after a fast fall | LR too low (the video's 2e-5 case) or `r` too low | 10× LR for 100 steps and compare |
| Loss spikes then recovers repeatedly | LR at the edge of stability | Halve LR; `max_grad_norm=1.0`; inspect the outlier batch |
| NaN in the first 20 steps | `bf16=True` on pre-Ampere, or LR far too high | `fp16=True` where `is_bf16_supported()` is False |
| Train loss down, eval loss up | Overfitting | Fewer epochs, lower `r`, 5–10% general replay (CH-13 §4.3) |
| Model answers and then never stops | EOS not appended to the target | Append `tokenizer.eos_token` (CS-16 §4.10) |
| Model repeats the question before answering | Unmasked prompt tokens | `train_on_responses_only` + the §5.2 assertion |
| Answers are coherent in the notebook, garbage behind the API | Train/serve template mismatch | Diff `chat_template` in the saved `tokenizer_config.json` |
| Merged model is worse than the adapter | Wrong merge from a 4-bit base | `save_pretrained_merged(save_method="merged_16bit")` + the logit test |
| Model ignores the system message | Base template has no `system` role, or you never trained one | Drop system messages, or pick a template that has them |
| Run is 1.2–2× slower than the reference band, no error | Unsupported arch / `import unsloth` too late / `lora_dropout > 0` / FP32 compute dtype | All four checks in §5.3 |
| Run is 3× slower, GPU <30% utilisation | **Dataloader-bound** | `num_proc=8`, `dataloader_num_workers=4`, pre-tokenize |
| First step takes 20 s, later steps 0.4 s | Triton JIT compile + autotune | Warm up; never include step 1 in a benchmark |
| First step *always* 20 s every run | Triton cache not persisted (ephemeral container) | Cache `~/.triton` in the image |
| OOM at load time, before any training | `max_seq_length` too high for the card | Halve it — it shapes RoPE and kernel selection, so lower it *before* loading |
| OOM only at a random later step | Packing produced one unusually long sequence; fragmentation | Cap `max_seq_length`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Adapter saves as ~2 GB instead of ~50 MB | You saved the model, not the adapter — or merged unintentionally | `ls out/adapter/`: `adapter_model.safetensors` vs `model.safetensors` |
| Free Colab disconnects mid-run | Session/idle limits | `save_steps=200`, `save_total_limit=3`, `resume_from_checkpoint` |
| Every run gives a slightly different model | No seed, or fused-kernel reassociation | Seed all four RNGs; accept ~1e-3 drift — that is expected, not a bug |
| Quality drops after adopting Unsloth | You changed the engine *and* the model at once | Keep a frozen 100-prompt golden set; diff before/after (CS-16 §12.2) |

---

## 9. Comparison Matrix

### 9.1 Frameworks

| | **Unsloth** | **HF + TRL** | **LLaMA-Factory** (CH-15) | **Axolotl** (CS-17) | **torchtune** |
|---|---|---|---|---|---|
| Interface | Python, 2 functions | Python, everything | WebUI / YAML / CLI | YAML + CLI | Python recipes |
| Speed vs naive HF | 2–4× | 1.0× (it *is* the baseline) | inherits HF | inherits HF | 1.2–2× on some paths |
| Speed vs **tuned** HF | **~1.2–1.4×** | 1.0× | ~1.0× | ~1.0× | ~1.0–1.4× |
| VRAM vs tuned HF | **~15–30% less** | — | ~0% | ~0% | 0–15% |
| Multi-GPU / FSDP | **weak** | full control | good | **best-in-class** | good |
| Architecture coverage | **curated list** | anything in `transformers` | 100+ templates | 100+ configs | curated |
| No-code UI | none | none | **yes** | partial | none |
| Masking | `train_on_responses_only` | `assistant_only_loss` | `train_on_prompt: false` | `train_on_inputs: false` | in the recipe |
| Merge on a 4-bit base | `save_pretrained_merged` (correct) | **manual, 16-bit only** | `export` CLI | `merge-lora` CLI | `tune convert` |
| Debugging | **hard** (fused kernels, Tracebacks inside Triton IR) | easy | medium | medium | medium |
| Best for | one GPU, ≤14B, max speed | research, exotic models | non-coders, many families | multi-GPU production | readable recipes |

**Lines to migrate:** Unsloth **6**, LLaMA-Factory ~40, Axolotl ~35. A 6-line change for a real
1.2–1.4× is a better engineering decision than a 2× that costs a week of refactoring — just do
not *book* the 2×. (CS-16 §13.2)

### 9.2 Unsloth vs the things it is confused with

| Approach | What it reduces | Stage | Composes with Unsloth? |
|---|---|---|---|
| **QLoRA / NF4 (bnb)** | weight memory *during training* | training | **It is what Unsloth accelerates** |
| **GPTQ / AWQ** | weight memory *at inference*, with calibration | post-training | Yes — export, then quantise (CH-10) |
| **GGUF / llama.cpp** | weight memory on CPU/edge | post-training | Yes — `save_pretrained_gguf` |
| **FlashAttention** | attention activation memory | train + infer | Integrated (see §5.7 note) |
| **Packing** | wasted compute on padding | training | Set via TRL; compatible |
| **`adamw_8bit`** | optimizer-state memory | training | Set via TRL; compatible |

> **Beyond the video:** the interview question *"Unsloth vs QLoRA — which should I use?"* is
> malformed: QLoRA is a quantisation method, Unsloth is an execution engine, and Unsloth *runs*
> QLoRA. The well-formed pair of questions is *"4-bit or 16-bit?"* and *"Unsloth or plain
> TRL?"*, and they are independent. (CS-16 §13.4)

**Attention-implementation compatibility** (the question everyone asks): FA2 requires an Ampere
or newer GPU (sm80+) and a supported head dimension — so on a **T4 (Turing), V100 or P100 there
is no FA2 library at all**, and xFormers is the documented fallback (CS-16 §3, §6.1, §15.3).
The video's T4 run still reached a 1.9 GB peak, which means **Unsloth's own flash-style Triton
attention** is what ran, not FA2 — CS-16 §4.8.2 uses that peak as the proof the flash path is
active. Practical rules: leave `attn_implementation` unset and let Unsloth choose; do not force
`flash_attention_2` on a card that cannot run it; do not install a separate `flash-attn` and
assume Unsloth will use it. *(Kernel selection per device varies by version — verify against
your pinned version by checking `attn_mod` as in §5.3.)*

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| Unsloth vs **unconfigured** HF | **2–4× time, 50–70% VRAM** | The tagline's baseline (CS-16 §4.7.1) |
| Unsloth vs **tuned** FA2+QLoRA+packing | **~1.2–1.4× time, 15–30% VRAM** | The number to quote (CS-16 §4.7.2) |
| Unsloth's VRAM contribution at 8B | **−8% / −8% / −6% / −5%** | T=1024 / 2048 / 4096 / 8192 (CS-16 §11.3) |
| Video's measured run | **535 s, 1.9 GB peak** | TinyLlama-1.1B, T=4096, r=32, batch 2×4, free T4 |
| Video's run cost | **$0.02–0.06** | 0.149 GPU-hours |
| Video's trainable params | **25,231,360 = 2.294%** | r=32 on 7 modules of a 1.1B |
| LoRA fraction bands | **~1.0% / 2.3% / 4.5%** | r=16 / 32 / 64, 7 modules, 1.1B |
| Bytes per trainable param | **6 B** | adapter 2 + grad 2 + adamw_8bit 2 |
| Bytes per param, full FT | **12–14 B** | weights+grad+AdamW fp32 m,v (+master) |
| Bytes per param, NF4 | **0.5 B** | 4-bit + double quant (4.5 bits measured) |
| Attention score tensor | **2.15 GB/layer** | B=2, H=32, T=4096, fp16 — never materialised |
| Logits tensor | **524 MB** | B=2, T=4096, V=32k, fp16 |
| Logits tensor, Llama-3 vocab | **2.49 GB** | Same shape, V=152,064 |
| `lora_dropout` | **0.0** | Exactly zero, or the fast path is gone |
| `use_gradient_checkpointing` | **`"unsloth"`** | A string, not `True` |
| Standard GC cost | **~25–35% step time, −60–75% activations** | CS-16 §3, §7.2 (see §12 note) |
| LoRA LR | **1e-4 … 3e-4** | 2e-4 is the default answer |
| Full-FT LR | **1e-5 … 2e-5** | ~10–20× below LoRA |
| Epochs | **1–3** | 2 is the default answer |
| Warmup | **3–10%** | 0.03 for short runs |
| LoRA `r` | **8–32** | 16 typical |
| `lora_alpha` | **= r** | Unsloth's own examples; `2r` is the CH-13 default |
| Target modules | **7** | `q,k,v,o,gate,up,down` |
| Examples per million trainable params | **100–1,000** | Rank sanity rule |
| `packing` win on Alpaca-shaped rows | **2–5×** throughput | Rows are ~150–250 tokens in a 4,096 window |
| Init loss band | **`ln(V)` × [0.7, 1.5]** | ≈ 7–16 for V=32k |
| Reference throughput, T4 4-bit 1B @ T=4096 | **2,500–4,500 tok/s** | The video's run: 3,057 |
| Reference throughput, A100-80 4-bit 8B @ T=2048 | **12,000–20,000 tok/s** | Below the band = a config bug, not hardware |
| Triton first-call compile | **~2–20 s per signature** | Cache: `~/.triton/cache` |
| Video's context table | **3k / 21k / 40k / 78k / 340k** | 8 / 12 / 16 / 24 / 80 GB — order-of-magnitude only |
| Merge fidelity threshold | **max abs Δlogit < ~0.2** | Pass/fail for `save_pretrained_merged` |
| Adapter equivalence threshold | **worst 1−cos < 0.02** after 100 steps | Unsloth vs HF manual-backward check |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `TypeError: TrainPlan.__init__() got an unexpected keyword argument 'size'` | `code/02_sft_unsloth.py --dry-run` is broken (see the Correction in §6) | Use CH-13 §12's checks; do not rely on that pre-flight block |
| `ImportError: cannot import name 'SFTConfig' from 'trl'` | `transformers`/`trl` pin violated by a later install | Reinstall the exact pin set (§6); never `pip install -U` in a working env |
| `Missing dependency: <e>` then `pip install unsloth` | `code/02_sft_unsloth.py` caught an `ImportError` at train time | Install **unsloth before** upgrading torch/transformers/trl |
| `AssertionError: Please enable GPU runtime` | CPU-only torch (wrong `cuXXX` wheel) or no GPU attached | Reinstall torch from the wheel index matching `nvidia-smi` |
| `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'` | TRL ≥0.16 renamed it | `processing_class=tokenizer` |
| `KeyError: 'text'` | A `messages` dataset with `dataset_text_field="text"` (or the reverse) | Pick one schema — `print(ds.column_names)` |
| `KeyError: 'instruction'` | `--data` got a non-Alpaca file; `load_jsonl` defaults to the Alpaca loader and there is no CLI flag to change it | Convert to `instruction`/`input`/`output` first (this is a defect in `code/02_sft_unsloth.py`, whose help text advertises sharegpt/openai/completion) |
| `RuntimeError: "addmm_impl_cpu_" not implemented for 'BFloat16'` | `bf16=True` on a pre-Ampere GPU (T4/V100/P100) | `fp16=True, bf16=False`; set the loader's `dtype` to match |
| `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.31 GiB (GPU 0; 15.78 GiB total capacity)` | Activation memory + dequant scratch + optimizer state exceed the card — 4-bit weights are only ~3.9 GB of it | `use_gradient_checkpointing="unsloth"`, halve `max_seq_length`, batch 1, then QLoRA |
| `CUDA out of memory` at **validation** only | Eval batch too large | `per_device_eval_batch_size` = train batch × 2 at most; eval fewer steps |
| `ValueError: ... max_seq_length ... exceeds` | `SFTConfig`'s length disagrees with the loader's | Set it in one place and pass it to both |
| `TypeError: list indices must be integers or slices, not str` | `token_stats` was handed raw conversations instead of `build_masked_example` output | Build the masked examples first (CH-13 §12) |
| `ValueError: num_samples should be a positive integer` | Dataset empty after filtering — usually every row had zero supervised tokens | Filter with `has_supervision`, not by hand |
| `KeyError: 'messages'` | A loader expects `messages` and got `text` (or the reverse) | Align the loader with the file's schema |
| `OSError: <id> does not appear to have a file named config.json` | Wrong model id, or not downloaded / no read token | Check the id and the HF cache |
| `401 Client Error` / `Repository not found` on a gated model | No Hub token, or the licence was not accepted on the model page | `huggingface-cli whoami`; accept the licence |
| A traceback that ends inside generated Triton code | Stale Triton cache compiled against a previous torch ABI | `rm -rf ~/.triton/cache` |
| An assertion raised from inside `train_on_responses_only` | A marker string was not found in the tokenized inputs — the fix is a marker fix, **never** a `force_match` bypass | Dump the rendered conversation, derive the marker from it (§5.2) |
| Unsloth's `RuntimeError` about not detecting the model type / an unsupported architecture | The family is outside the curated list; the run is stock HF at 1.0× | Assert `attn_mod` (§5.3) and switch base model, or change framework |

> Exact wording varies between Unsloth, TRL and CUDA versions. Match on the **distinctive
> substring**, not the full string, and verify against your pinned version.

---

## 12. Starter Config

`code/02_sft_unsloth.py`'s DEFAULTS, corrected to the values CS-16 §6.7 recommends. Run **one**
config first; change only the marked lines.

```python
# ── CHANGE THESE ────────────────────────────────────────────────────────────
MODEL        = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"   # must be a SUPPORTED architecture
DATA         = "data/sft.jsonl"                          # instruction/input/output, or messages
OUT          = "out/unsloth-lora"
MAX_SEQ_LEN  = 2048            # = p99.5 of YOUR tokenized lengths, rounded up to 64
R            = 16              # 8-16 for 1k-10k rows; 16-32 for 10k-100k
LR           = 2e-4            # 1e-4 if the base is already an Instruct model

# ── CHANGE NOTHING BELOW FOR RUN ONE ────────────────────────────────────────
import unsloth                                            # FIRST import
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=R, lora_alpha=R, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",     # STRING. Not True.
    random_state=3407,
)

ARGS = dict(
    output_dir=OUT,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,            # effective batch 16
    num_train_epochs=2,                       # 1-3. Watch EVAL loss.
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_8bit",
    max_length=MAX_SEQ_LEN,                   # MUST equal the loader's value
    packing=True,
    bf16=True, fp16=False,                    # derive from torch.cuda.is_bf16_supported()
    logging_steps=5,
    save_steps=200, save_total_limit=3,
    eval_strategy="steps", eval_steps=100,    # <-- the line that saves the project
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=3407,
    report_to="none",
)
# ... build the dataset as a `messages` column (§5.1), SFTTrainer(...), then:
# trainer = train_on_responses_only(trainer, response_part=<derived>)
# ... then run the S5.2 assertion block BEFORE trainer.train().
```

### The five checks before every run

```bash
# 1. Are the fast modules actually installed?  (the silent-1.0x check)  -> S5.3
# 2. What is the chat template, and does the marker match it?          -> S5.2 / S6
# 3. Is anything supervised, and only the right thing?                 -> S5.2 assertion block
# 4. What will this cost in VRAM?
python code/common/memory.py --model 7B --method qlora --seq-len 2048 --batch 2
# 5. Does it fit? p95 tokenised length vs max_seq_length  -> CH-13 S12 check 5
```

**Run one**, then in this order: template → supervised fraction → first loss well below
`ln(V)` and falling → 10 held-out prompts *read by a human* → only then touch a hyperparameter.
Of these, checks 1–3 and the eval config are worth more than every kernel in this file.

> **Correction:** CS-16 §6.4's inline comment states gradient checkpointing "costs ~5–15% step
> time and saves 40–60% activation memory", while §3 and §7.2 give **25–35% step time and
> 60–75% activations**. Both figures are defensible *for different code paths* — the 25–35%
> figure is the standard HF implementation (`use_gradient_checkpointing=True`), and the lower
> figure is Unsloth's selective variant (`"unsloth"`), which stores more and recomputes less.
> CS-16 §7.2 collapses the two into one row, so read it as "standard GC". Use **~25–35%** when
> you pass `True`, and **~5–15%** when you pass `"unsloth"`.

> **Correction:** CH-13 §5.3 heads its Unsloth snippet *"Unsloth (2–4× faster, ~60% less
> VRAM)"* with no baseline. Carried without a baseline that sentence is the exact error CS-16
> §4.7 is written to prevent: the ratio only holds against `transformers`+`peft`+`bnb` at
> library defaults. Against a tuned FA2 + bf16 + packing + 7-module baseline the honest figure
> is **~1.2–1.4× and 15–30%** (CS-16 §4.7.2, §13.1, §18.2). CH-13 §5.3's `lora_alpha=32` with
> `r=16` also disagrees with `code/02_sft_unsloth.py` (`lora_alpha=16 = r`) and with CS-16
> §17.9, which notes Unsloth's own examples use `alpha = r`. Neither is wrong; pick one, state
> it, and remember that `alpha/r` is an LR multiplier on the LoRA branch only.

> **Beyond the video:** the task CS-16 sets for `FastLanguageModel` vs `FastModel` is not
> covered there at all. In current Unsloth, `FastModel` is the unified entry point and
> `FastLanguageModel` remains the text-only path; the vision/multimodal recipes use `FastModel`
> with `UnslothVisionDataCollator` (CS-21). **Do not assume the two accept identical keyword
> arguments, and do not assume a `FastModel` checkpoint behaves like a `FastLanguageModel` one
> inside `train_on_responses_only`** — check the signature on your pinned version
> (`help(FastModel.from_pretrained)`) before you port a text SFT recipe to a VLM.

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| Understand *why* the manual LoRA backward is correct | **CS-16 §4.3** (`unsloth/kernels/fast_lora.py` is ~300 lines) |
| Derive LoRA/QLoRA yourself (rank, alpha, which modules) | **CH-13 §4.2**, and the planned **CS-23 — LoRA & QLoRA** |
| The baseline Unsloth is measured against | **CS-16 §4.7**, §12.1's A/B harness, §18.2 |
| The SFT pipeline this accelerates | **CH-13 / CS-13 — Instruction Fine-Tuning** (§5.2 loss masking by hand, §12 pre-flight checks) |
| Run the same job with no code at all | **CH-15 / CS-15 — LLaMA-Factory** |
| Multi-GPU / FSDP instead of one card | **CS-17 — Axolotl** |
| NF4, double quant, GPTQ/AWQ/GGUF, why merging is lossy | **CH-10 / CS-10 / CS-11 — Quantization** |
| DPO/ORPO/GRPO on top of this SFT — Unsloth's biggest memory win | **CH-14 / CS-14 — The Alignment Map** |
| Multimodal (`FastModel`, `UnslothVisionDataCollator`) | the planned **CS-21 — Multimodal / Vision-Language** |
| Be interviewed on this | **IQ-16 — Unsloth Interview Questions** (planned; check the README status table) |
| The exact trainer code with the traps pre-checked | `code/02_sft_unsloth.py` (see the Correction in §6 before trusting its plan block), `code/01_sft_lora.py`, `code/common/memory.py` |
