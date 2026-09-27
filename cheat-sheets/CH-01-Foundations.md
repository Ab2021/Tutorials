# CH-01 — Foundations Cheat Sheet

**One-line purpose:** Every formula, constant, decision and error message the rest of the handbook assumes you already know.
**Use when:** Sizing a run, budgeting GPU-hours, choosing full-FT vs LoRA vs QLoRA, debugging a loss curve, revising the night before an interview.
**Do NOT use when:** You need the *reasoning* — that is CS-01. This card gives the answer, not the derivation.

> Pairs with **CS-01** (case study) and **IQ-01** (100 interview questions). Deep dives: **CS-05** (why the Transformer), **CS-10/11** (quantization), **CS-12** (continued pretraining), **CS-13** (SFT), **CS-13 §6.8** / **CS-11 §4.11** (LoRA/QLoRA), **CS-04** (FT vs RAG vs agents).

---

## 1. The 10-Second Summary

| # | Fact | Number |
|---|---|---|
| 1 | Pretraining and fine-tuning differ by **3–5 orders of magnitude** in compute | 7B pretrain ≈ 184k A100-h; 7B QLoRA ≈ 2–6 A100-h |
| 2 | Full fine-tuning needs **16 bytes per parameter** | 7B = 112 GB — LoRA exists because of this one number |
| 3 | Training FLOPs = **6ND**; inference = **2N per token** | 7B × 2T tokens = 8.4e19 FLOPs |
| 4 | Compute-optimal is **~20 tokens/parameter**; production overtrain **14–94x past it** | Llama-3-8B = 1,875 tok/param |
| 5 | **Base ≠ instruct** — different weights, template, behaviour | The most misunderstood thing in the field |
| 6 | The **chat template** is the #1 silent SFT failure | Loss falls smoothly; model is useless |
| 7 | Fine-tuning teaches **form, not facts** | Knowledge problem → RAG (CS-04) |
| 8 | Decode is **bandwidth-bound**, not compute-bound | 7B fp16 on A100 ≈ 7 ms/token, ~143 tok/s ceiling |
| 9 | Default recipe: **QLoRA, r=16, alpha=32, lr=1e-4, 1 epoch, all linear targets** | Works for ~80% of first projects |
| 10 | **Build the eval set and the prompt-only baseline before training** | Otherwise you cannot tell whether it worked |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| Training FLOPs | `C_train ≈ 6 · N · D` | N = params, D = tokens | 7B × 2e9 → `6×7e9×2e9 = 8.4e19` |
| Inference FLOPs | `C_infer ≈ 2 · N` per token | — | 7B = 14 GFLOP/token |
| Attention FLOPs | `12 · L · s² · h` per sequence | L layers, s len, h hidden | 7B at s=128k: `12×32×1.64e10×4096 = 2.6e15` |
| Full-FT VRAM | `N × 16 bytes` + activations | bf16 w(2)+g(2)+Adam m,v(8)+master(4) | 13B × 16 = **208 GB** |
| LoRA VRAM | `N × 2 bytes` + `P_lora × 16` + act. | P_lora = adapter params | 7B: 14 + 0.6 ≈ 15 GB |
| QLoRA VRAM | `N × 0.5 bytes` + `P_lora × 16` + act. | NF4 + double quant | 7B: 3.9 + 0.6 ≈ 4.5 GB |
| LoRA adapter params | `L × Σ r·(d_in + d_out)` over targets | r = rank | 7B, r=16, 7 targets ≈ 40M |
| LoRA merge | `W' = W + (α/r) · B·A` | α = lora_alpha | α = r → scale 1.0 |
| Cross-entropy (CLM) | `L = −(1/T) Σ log p(x_t \| x_<t)` | T = supervised tokens | 0.51, 1.05, 2.12, 1.22 → **1.23** |
| Perplexity | `ppl = exp(L)` | — | L = 1.23 → **3.41** |
| Uniform-loss baseline | `L = ln(V)` | V = vocab size | 32k → **10.37**; 128k → **11.76** |
| CE gradient | `∂L/∂z = softmax(z) − y` | z = logits | pred [.66,.24,.10], target 1 → `[−0.34, 0.76, −0.42]` |
| Global batch (tokens) | `micro × accum × n_gpus × seq_len` | — | 4×2×8×2048 = **131,072** tok/step |
| Steps per epoch | `ceil(dataset / (micro × accum × n_gpus))` | — | 10,000 / 64 = **157 steps** |
| AdamW update | `θ ← θ − η·(m̂/(√v̂+ε)) − η·λ·θ` | β₁=.9, β₂=.95, ε=1e-8 | `0.500 − 1e-4×(0.5/0.1) = 0.4995` |
| KV cache / sequence | `2 · L · n_kv_heads · d_head · bytes` | — | Llama-3-8B, 8k, GQA-8: **1.07 GB** |
| Arithmetic intensity | `FLOPs / bytes_moved` | — | decode batch 1 ≈ 1 FLOP/byte |
| Ridge point | `peak_FLOPs / bandwidth` | — | A100 **156**; H100 **295** |
| Decode time/token | `weight_bytes / bandwidth` | — | 14 GB / 2 TB/s = **7 ms** |
| Training time | `6ND / (n_gpus × peak × MFU)` | MFU 0.2–0.5 | 7B/2B tok, 35% MFU → 214 A100-h |
| Chinchilla | `D_opt ≈ 20 · N` | — | 7B → **140B tokens**; 3B → 60B |
| Cost | `GPU_hours × $/hr × 1.6` | 1.6 = real-world overhead | 214 × 1.6 × $1.79 = **$612** |
| Break-even vs API | `train_cost / (saving_per_req × req/mo)` | — | $3 / ($0.0175×900k) → **first month** |
| Tokens ≈ words | `1 token ≈ 4 chars ≈ 0.75 words` | — | 1,000 words ≈ 1,330 tokens |

---

## 3. Decision Tree

```
Is the problem KNOWLEDGE (facts, current data, citations)?
├─ Yes → RAG, not fine-tuning. Stop. (CS-04)
└─ No  → is it BEHAVIOUR (format, tone, schema, refusals, jargon)?
    Have you tried a well-prompted base model with 4–8 examples?
    ├─ No  → do that first. It is the baseline you must beat.
    └─ Yes → does a 2k+ token few-shot prompt dominate your bill?
        ├─ Yes → STRONG financial case: fine-tune to shrink the prompt.
        └─ No  → quality case only; proceed if the gap is measured.

METHOD
├─ < 10k examples, 7B–8B, one 24 GB GPU   → QLoRA    (r=16, a=32, lr=1e-4)
├─ 10k–100k examples, 40–80 GB available  → LoRA     (r=32, a=64, lr=2e-4)
├─ > 100k examples, diverse, budget exists→ full FT  (lr=2e-5, 16 B/param)
└─ New language / new tokenizer needed    → continued pretraining FIRST (CS-12)

BASE MODEL
├─ Classification / NER / embeddings  → encoder (BERT-base/large) — CS-07; embeddings: code/10_embedding_finetune.py
├─ Generative, English, 1 GPU         → Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct
├─ Non-English                        → check tokens/word FIRST; prefer Llama-3/Qwen2.5
└─ On-prem, tiny GPU                  → 0.5B–3B (Qwen2.5-3B, Phi-3.5-mini) — CS-20 (planned, not yet written)

CHECKPOINT: assistant/chat behaviour → INSTRUCT. Raw domain completion → BASE.
Unsure → instruct: fine-tuning a base by mistake wastes the run.

EPOCHS: <2k examples → 1–2 · 2k–100k → 1–3 · >100k → 1 (only if eval still improves)
```

---

## 4. Hyperparameter Quick Reference

| Param | Default | Sweep | Effect / gotcha |
|---|---|---|---|
| `learning_rate` | FT 2e-5 · LoRA 2e-4 · QLoRA 1e-4 | ×3 | 10x too high with full FT destroys the base — the #1 config error |
| `num_train_epochs` | 1 | 1–3 | >3 on small data = memorisation; stop on eval loss |
| `per_device_train_batch_size` | 2 | 1–8 | VRAM driver; raise only after activations are understood |
| `gradient_accumulation_steps` | 8 | 4–32 | Free in FLOPs, costly in wall clock; keeps global batch constant |
| `max_seq_length` | p99 of your data | 512–4096 | Set from data, never a round number; `s²` without flash-attn |
| `lora_rank` (`r`) | 16 | 8–64 | Capacity; >64 flattens; `r=128` on 500 examples overfits |
| `lora_alpha` | 32 | `2r` or `r` | Effective scale `alpha/r`; tune with `lr`, not independently |
| `lora_dropout` | 0.05 | 0.0–0.1 | 0.05 small data; 0.0 large |
| `target_modules` | all linear | — | `q,k,v,o_proj,gate,up,down_proj`; attention-only is measurably worse |
| `warmup_ratio` | 0.03 | 0.0–0.1 | Adam `v` unstable early; use `warmup_steps` for short runs |
| `lr_scheduler_type` | `cosine` | cosine/linear/constant | Constant is fine below 200 steps |
| `weight_decay` | 0.01 | 0.0–0.1 | Exclude bias and norm weights; 0.0 common for LoRA |
| `bf16` | `True` | — | Never `fp16` on Ampere+ except inference on V100/T4 |
| `gradient_checkpointing` | `True` | — | +25–33% time for 5–10x less activation memory |
| `optim` | `adamw_torch` / `paged_adamw_8bit` | — | Paged 8-bit halves optimiser memory on QLoRA |
| `eval_strategy` / `eval_steps` | `"steps"` / 50–100 | 20–200 | Needed for `load_best_model_at_end` to be useful |
| `load_best_model_at_end` | `True` | — | With `metric_for_best_model="eval_loss"` |
| `early_stopping_patience` | 2 | 1–3 | On `eval_loss`; costs nothing, often saves the run |
| `packing` | `False` | — | 2–3x throughput; corrupts SFT if boundaries are unmasked |
| `save_total_limit` | 2 | — | Stops checkpoints filling a 100 GB disk |
| `seed` / `data_seed` | 42 | — | Set both; `dataloader_num_workers=0` for repro runs |

---

## 5. Copy-Paste Code Snippets

### 5.1 Minimal working QLoRA SFT (complete)

```python
# pip install "transformers>=4.45" "peft>=0.13" "trl>=0.12" "bitsandbytes>=0.44" datasets accelerate
import torch, math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL = "meta-llama/Llama-3.2-1B-Instruct"        # swap for any instruct checkpoint

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token                  # Llama/Mistral ship no pad token
tok.padding_side = "right"                         # left only for batched generation

# --- PROVE the chat template BEFORE spending a GPU-hour (the whole game) ---
sample = [{"role": "user", "content": "Name three uses of LoRA."},
          {"role": "assistant", "content": "Style transfer, schema adherence, prompt compression."}]
rendered = tok.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
print(rendered)
assert tok.eos_token_id in tok(rendered).input_ids, "EOS missing -> never stops generating"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map="auto", attn_implementation="sdpa")
model.config.use_cache = False                     # required with gradient checkpointing
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])

ds = load_dataset("tatsu-lab/alpaca", split="train[:2000]")
ds = ds.map(lambda e: {"messages": [
    {"role": "user", "content": e["instruction"] + ("\n\n" + e["input"] if e["input"] else "")},
    {"role": "assistant", "content": e["output"]}]}, remove_columns=ds.column_names)

cfg = SFTConfig(output_dir="./out", num_train_epochs=1,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,   # global = 16 seqs
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    weight_decay=0.01, max_grad_norm=1.0, optim="paged_adamw_8bit",
    bf16=True, gradient_checkpointing=True, max_seq_length=1024, packing=False,
    eval_strategy="steps", eval_steps=50, save_steps=50, save_total_limit=2,
    logging_steps=10, seed=42, data_seed=42,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", report_to="none")

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     eval_dataset=ds.select(range(100)), processing_class=tok)
n = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
assert n > 0, "Nothing trainable -> loss stays pinned at ln(V)"
print(f"trainable={n:,}  ln(V)={math.log(model.config.vocab_size):.2f}")
trainer.train(); trainer.save_model("./adapter"); tok.save_pretrained("./adapter")
```

### 5.2 Common variations

```python
# bf16 LoRA (no quantization; 20-40% faster steps). lr=2e-4, optim="adamw_torch"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
# Full FT (only when LoRA is demonstrably the bottleneck). lr=2e-5, no LoraConfig,
#   no prepare_model_for_kbit_training, expect 16 bytes/param.
# Unsloth (2-4x faster, lowest VRAM; CS-16):
#   from unsloth import FastLanguageModel
#   model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=2048, load_in_4bit=True)
#   model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32,
#       use_gradient_checkpointing="unsloth", target_modules=[...])
# Encoder classification (CS-07): no template, no causal mask, lr=2e-5, warmup_ratio=0.1
# Merge for serving -- merge in bf16/fp32, NEVER into an NF4 base:
#   merged = PeftModel.from_pretrained(base, "./adapter").merge_and_unload()
```

### 5.3 The four preflight checks that prevent most wasted runs

```python
import math
def preflight(model, tok, dataset, max_seq_length=1024):
    ex = dataset[0]["messages"]
    s = tok.apply_chat_template(ex, tokenize=False, add_generation_prompt=False)
    print("1) TRAIN STRING:", repr(s[-120:]))
    assert tok.eos_token_id in tok(s).input_ids, "EOS missing -> model will not stop"
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"2) TRAINABLE={n:,}"); assert n > 0, "Nothing trainable"
    lens = [len(tok(tok.apply_chat_template(d["messages"], tokenize=False)).input_ids)
            for d in dataset.select(range(min(500, len(dataset))))]
    print(f"3) p50={sorted(lens)[len(lens)//2]} p99={sorted(lens)[int(.99*len(lens))-1]} "
          f"truncated={sum(l > max_seq_length for l in lens)/len(lens):.1%}")
    print(f"4) ln(V)={math.log(model.config.vocab_size):.2f} -- a flat loss here = no labels")
```

---

## 6. CLI Commands

```bash
# ---- environment -------------------------------------------------------
pip install "transformers>=4.45" "peft>=0.13" "trl>=0.12" "bitsandbytes>=0.44" datasets accelerate
pip install flash-attn --no-build-isolation        # Ampere+ only; needs CUDA toolkit
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
# bf16 False -> V100/T4: use fp16 + loss scaling, not bf16

# ---- LLaMA-Factory (no-code; CS-15) -----------------------------------
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory
pip install -e ".[torch,metrics]"
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli webui                                        # browser UI on :7860
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml

# ---- Axolotl (YAML at scale; CS-17) -----------------------------------
git clone https://github.com/axolotl-ai-cloud/axolotl && cd axolotl
pip install -e ".[flash-attn,deepspeed]"
accelerate launch -m axolotl.cli.train config.yml
accelerate launch -m axolotl.cli.inference config.yml --lora_model_dir=./out/lora

# ---- Unsloth (CS-16) / Ollama (CS-10) ---------------------------------
pip install unsloth
ollama pull llama3.2 && ollama run llama3.2 "Name three uses of LoRA."

# ---- vLLM: one base + many adapters (multi-tenant; CS-16) -------------
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-lora --max-loras 8 \
  --max-lora-rank 64 --lora-modules tenant_a=./adapter_a tenant_b=./adapter_b \
  --max-model-len 8192 --gpu-memory-utilization 0.90
# curl localhost:8000/v1/chat/completions -d '{"model":"tenant_a","messages":[...]}'

# ---- Cost sanity check: 7B on 2B tokens, 35% MFU, A100 ----------------
python -c "N,D=7e9,2e9; h=6*N*D/(312e12*0.35)/3600*1.6; print(f'{h:,.0f} A100-h -> \${h*1.79:,.0f}')"
# -> 342 A100-h -> $612

# ---- Reproducibility / OOM --------------------------------------------
export CUBLAS_WORKSPACE_CONFIG=:4096:8                     # for deterministic algorithms
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # fixes most "it fit yesterday" OOMs
```

---

## 7. VRAM / Cost Calculator

**Full fine-tuning VRAM (GB) — 16 bytes/param + activations:**

| Model | bf16 w | Grad | Adam m,v | Master | Subtotal | +Activ. | **Total** | Min GPUs (80 GB) |
|---|---|---|---|---|---|---|---|---|
| 1B | 2 | 2 | 8 | 4 | 16 | 1–2 | **17–18** | 1 × A100-40 |
| 3B | 6 | 6 | 24 | 12 | 48 | 3–5 | **51–53** | 1 × A100-80 |
| 7B | 14 | 14 | 56 | 28 | 112 | 8–16 | **120–128** | 2 × A100-80 |
| 8B | 16 | 16 | 64 | 32 | 128 | 9–18 | **137–146** | 2 × A100-80 |
| 13B | 26 | 26 | 104 | 52 | 208 | 12–25 | **220–233** | 4 × A100-80 |
| 70B | 140 | 140 | 560 | 280 | 1,120 | 60–120 | **1,180–1,240** | 16 × A100-80 (ZeRO-3) |

**PEFT VRAM (GB) — frozen base + adapters:**

| Model | Base bf16 | Base NF4 | Adapter+opt (r=16) | Activ. | **LoRA** | **QLoRA** | Fits on |
|---|---|---|---|---|---|---|---|
| 1B | 2 | 0.5 | 0.1 | 1.5 | **~4** | **~2.5** | T4 16 GB |
| 3B | 6 | 1.5 | 0.3 | 3.5 | **~11** | **~6** | RTX 3060 12 GB (QLoRA) |
| 7B | 14 | 3.9 | 0.6 | 7 | **~23** | **~13** | RTX 4090 24 GB / A100-40 |
| 8B | 16 | 4.5 | 0.7 | 8 | **~26** | **~15** | A100-40 / L40S 48 GB |
| 13B | 26 | 6.5 | 1.1 | 11 | **~40** | **~20** | A100-40 (LoRA) / 4090 (QLoRA) |
| 70B | 140 | 35 | 4 | 25 | **~175** | **~65** | 2 × A100-80 (QLoRA, FSDP) |

> Activations assume FlashAttention/SDPA **and** gradient checkpointing — without checkpointing multiply by ~5. Doubling `max_seq_length` roughly doubles activations at short context and quadruples them at long. Keep 10–15% headroom for allocator fragmentation.

**Quantized inference VRAM (weights only, GB):**

| Model | fp32 | fp16/bf16 | int8 | int4/NF4 | GGUF Q4_K_M |
|---|---|---|---|---|---|
| 1B | 4 | 2 | 1.1 | 0.7 | 0.8 |
| 7B | 28 | 14 | 7.5 | 3.9 | 4.4 |
| 13B | 52 | 26 | 14 | 7 | 7.9 |
| 70B | 280 | 140 | 75 | 39 | 42 |

**Rental pricing (2026 ballpark, USD/GPU-hour, on-demand):**

| Provider | T4 16 GB | A10G 24 GB | L40S 48 GB | A100-80 GB | H100-80 GB |
|---|---|---|---|---|---|
| RunPod | — | $0.79–1.19 | $1.19–1.79 | $1.64–2.49 | $2.79–3.99 |
| Lambda | — | — | — | $1.79–2.49 | $2.49–3.29 |
| AWS p4d/g5 | ~$0.53 | ~$1.01 | — | ~$4.10–5.12 | ~$6.88 |
| GCP a2/a3 | — | — | — | ~$3.67 | ~$3.00–11.06 |
| Colab Pro+ / Kaggle | — | — | — | — | free T4/P100, 12 h limits |

**Cost of one 7B / 15M-token / 1-epoch job:**

| Method | GPU | GPU-h | On-demand | Reserved | Notes |
|---|---|---|---|---|---|
| QLoRA r=16 | RTX 4090 24 GB | ~4 | ~$3 | — | The default first run |
| QLoRA r=16 | A100-80 GB | ~2.5 | ~$5 | ~$3 | Faster, underused card |
| LoRA bf16 r=32 | A100-80 GB | ~2 | ~$4 | ~$2.20 | Better quality than QLoRA |
| Full FT | 2–4 × A100-80 GB | 3–6 | ~$12 | ~$7 | 16 bytes/param; rarely justified |

> Multiply every GPU-hour figure by **1.5–2x** for a first attempt: restarts, failed configs, evaluation runs and dataloading are real. The second run costs a third of the first.

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | First check | Fix |
|---|---|---|---|
| Loss flat at `ln(V)` (10.37/11.76), never moves | No supervised labels / nothing trainable | `labels[labels != -100]`; `sum(p.requires_grad)` | Fix the mask span; add `target_modules` that match |
| Loss falls smoothly, model is useless | **Wrong chat template** | Render `apply_chat_template(..., tokenize=False)` and read it | Retrain with the model's own template; verify EOS present |
| Loss → 0.05, eval loss rises | Overfitting / leakage | MinHash train vs eval; epochs vs dataset size | 1 epoch; early stop on `eval_loss`; add data |
| NaN at step 0 | fp16 overflow, LR too high, empty/zero-masked row | `bf16=True`? ids < vocab? all-zero mask? | bf16; LR /10; filter empty sequences |
| NaN at step ~500 | LR too high, gradient spike | grad_norm before the NaN | `max_grad_norm=1.0`; halve LR; add warmup |
| Loss starts well below `ln(V)` | Prompt tokens not masked | Reconstruct the supervised span | Mask prompt to `-100`; `train_on_inputs=False` |
| Repeats the question, then answers | Training string lacks the assistant header or the EOS | Last 120 chars of the rendered train string | Rebuild data with the template + EOS; **not** `repetition_penalty` |
| Generation never stops | EOS not in serving stops, or truncated away | `tok.eos_token_id in ids`; serving stop strings | Add the end-of-turn token to stops; `max_seq_length` above p99 |
| Works in notebook, garbage in batch serving | Padding side / template skew | Batch 1 works, batch >1 fails → padding | `padding_side="left"`; share the tokenizer artifact |
| Worse at everything after FT | LR 10x too high, or base/instruct swapped | `optimizer.param_groups[0]["lr"]`; which checkpoint | FT → 2e-5, LoRA → 2e-4; match the checkpoint class |
| Improved benchmark, users unhappy | Distribution mismatch or proxy metric | Sample 200 real production inputs and evaluate | Rebuild eval from production; add a regression suite |
| OOM at a batch size that fit yesterday | Length shift, eager attention, fragmentation | p99 length in this shard; `attn_implementation` | `expandable_segments:True`; gradient checkpointing; micro-batch 1 |
| 0% exact match, 0.82 token F1 | Format mismatch, not a learning failure | The eval harness's normalisation | Normalise first — do **not** retrain |
| Cannot reproduce a run | Data order, kernels, versions, unpinned revision | `num_workers`, `seed`, `data_seed`, `revision` | Seed everything; `num_workers=0`; pin model+dataset commits |
| Merged QLoRA model worse than the adapter | Merged into an NF4 base | Merge dtype | Dequantize to bf16 → merge → re-quantize → re-evaluate |
| 30x slower than estimated | Checkpointing on, `packing=False`, `num_workers=0`, tiny batch | Achieved tok/s vs the estimate | Enable packing (masked); raise micro-batch; check MFU |
| OOM only on the eval step | Logits accumulating on GPU | `eval_accumulation_steps` | Set `eval_accumulation_steps=1`; `prediction_loss_only=True` |

---

## 9. Comparison Matrix

**Lifecycle stages:**

| Stage | Data | Supervision | Compute (7B) | Output |
|---|---|---|---|---|
| Tokenizer training | 1–100 GB text | none | minutes | `tokenizer.json` |
| **Pretraining** | 1–15 T tokens | self-supervised, 100% of tokens | ~184k A100-h | **base model** |
| Continued pretraining | 1–100 B domain tokens | self-supervised | 1k–20k A100-h | domain-adapted base |
| **SFT** | 1k–1M pairs | answers only (response masked in) | 2–6 A100-h | **instruct model** |
| Preference alignment | 10k–1M pairs | chosen/rejected | 5–50 A100-h | aligned model |
| Evaluation | 200–5,000 items | gold labels | minutes–hours | eval report |

**Pretraining objectives:**

| Objective | Architecture | Masking | Supervision | Generation? | Models |
|---|---|---|---|---|---|
| Causal LM | decoder-only | causal (left) | **100%** | Yes | GPT, Llama, Qwen, Gemma, Mistral |
| Masked LM | encoder-only | 15% random | ~15% | No | BERT/RoBERTa: classification, NER, embeddings |
| Span corruption | encoder-decoder | random spans | ~15–25% | Yes | T5/mT5/BART: translation, summarisation |

**Fine-tuning methods:**

| | Full FT | LoRA (bf16) | QLoRA (NF4) |
|---|---|---|---|
| Trainable params (7B) | 7 B (100%) | ~40 M (0.6%) | ~40 M (0.6%) |
| VRAM (7B) | ~120–128 GB | ~23 GB | ~13 GB |
| Quality vs full FT | baseline | 98–100% | 96–99% (gap grows past ~1M examples) |
| Step speed | 1.0x | 1.1–1.3x | 0.6–0.8x |
| Learning rate | 1e-5–5e-5 | 1e-4–3e-4 | 1e-4 |
| Serving | 1 model | merge or adapter | dequantize → merge → re-quantize |
| Use when | >100k examples, budget proven | default for quality | default for 1 GPU / tight VRAM |
| Catastrophic forgetting | high risk | low (base frozen) | low (base frozen) |

**Prompting vs RAG vs fine-tuning vs agents (CS-04):**

| | Prompting | RAG | Fine-tuning | Agents |
|---|---|---|---|---|
| Changes | input | input + retrieval | **weights** | control flow |
| Teaches | nothing | facts, citations | form, style, schema, refusals | multi-step behaviour |
| Data needed | 0 | a corpus | 1k–100k pairs | tools + traces |
| Freshness | model cutoff | instant | frozen at train time | instant (via tools) |
| Latency | baseline | +50–300 ms | **lowest** (short prompts) | highest |
| Best for | prototypes | knowledge, citations | format, tone, cost | workflows, actions |

---

## 10. Numbers To Memorize

| Number | Value |
|---|---|
| Full-FT memory per parameter | **16 bytes** |
| LoRA / QLoRA base per parameter | **2 bytes** (bf16) / **0.5** (NF4) |
| Training FLOPs | **6ND** · Inference **2N per token** |
| Chinchilla tokens/param | **~20** · Llama-3-8B at **1,875** (overtrained ~94x) |
| `ln(32000)` / `ln(128256)` | **10.37** / **11.76** |
| Tokens per English word | **~1.3** (1 token ≈ 4 chars) |
| A100-80 GB bf16 peak / bandwidth / ridge | **312 TFLOP/s** / 2.0 TB/s / **156** |
| H100 SXM bf16 peak / bandwidth / ridge | **989 TFLOP/s** / 3.35 TB/s / **295** |
| Realistic MFU | **20–50%** |
| Llama-2-7B pretraining | **~184,000 A100-hours** (2T tokens) |
| 7B QLoRA fine-tune | **2–6 GPU-hours, $3–15** |
| Default LoRA config | **r=16, alpha=32, dropout=0.05, lr=1e-4–2e-4** |
| Default SFT epochs / global batch | **1–3** / **~128k tokens/step** |
| Gradient checkpointing cost | **+25–33% time** for 5–10x memory |
| Adam moments | **m, v = 2 × fp32 = 8 bytes/param** (+4 master) |
| BERT MLM mask rate · LIMA | **15%** · **1,000 curated examples** |
| A100-80 GB on-demand | **$1.10–2.50 /hr** |
| KV cache, Llama-3-8B GQA-8 @ 8k fp16 | **1.07 GB / sequence** |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `ValueError: Asking to pad but the tokenizer does not have a padding token.` | Llama/Mistral-class tokenizers ship no pad token | `tok.pad_token = tok.eos_token`; set `pad_token_id` |
| `Some weights of the model checkpoint were not used ... ['lm_head.weight']` | Instruct checkpoint loaded into a base class (or vice versa) | Match `AutoModelForCausalLM` to the checkpoint; check `config.architectures` |
| `RuntimeError: CUDA out of memory. Tried to allocate X GiB` | Activations or optimiser state exceed the card | Gradient checkpointing, micro-batch 1, raise accum, `expandable_segments:True` |
| `torch.cuda.OutOfMemoryError ... 20.00 MiB is free` | Fragmentation, not a real shortage | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; restart; check for a stale job |
| `IndexError: index out of range in self` (embedding lookup) | Token ids exceed `vocab_size` — tokenizer/model mismatch | `assert ids.max() < model.config.vocab_size`; `resize_token_embeddings(len(tok))` |
| `UserWarning: The following columns are not used: ['labels']` | Collator dropped labels; the response is not supervised | `DataCollatorForSeq2Seq` or TRL's SFT collator with `train_on_inputs=False` |
| `loss: nan` from step 0 | fp16 overflow, LR too high, or an all-zero attention mask | `bf16=True`; LR /10; filter empty sequences |
| `RuntimeError: element 0 of tensors does not require grad` | Everything frozen — PEFT matched no target module | `target_modules` must match the architecture (`q_proj` for Llama, `query` for T5) |
| `ImportError: flash_attn_2_cuda ... undefined symbol` | flash-attn built against a different torch/CUDA | Reinstall `--no-build-isolation`, or use `attn_implementation="sdpa"` |
| `Expected all tensors to be on the same device` | Model sharded across devices while inputs sit on one | `device_map="auto"` + `prepare_model_for_kbit_training` |
| `RepositoryNotFoundError` on a gated repo | Llama/Gemma licence not accepted | Accept on the Hub; `huggingface-cli login` |
| `UnicodeDecodeError` loading a dataset | Non-UTF-8 bytes in a scraped corpus | `load_dataset(..., encoding="utf-8", errors="ignore")`; clean first |
| `bitsandbytes` CUDA setup failed | bnb vs CUDA mismatch, or CPU-only environment | `pip install -U bitsandbytes`; check `torch.cuda.is_available()` |
| `SFTTrainer.__init__() got an unexpected keyword argument 'dataset_text_field'` | TRL ≥0.12 moved args into `SFTConfig` | Move it into `SFTConfig(...)`; pass `processing_class=tok` |

---

## 12. Copy-Paste Starter Config

Change two strings (`MODEL`, `DATA_PATH`) and this trains. Save as `train_sft.py`.

```python
#!/usr/bin/env python
"""Minimal, correct QLoRA SFT run.  Usage: python train_sft.py"""
import json, math, torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, EarlyStoppingCallback)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ------------------------------- CONFIG ----------------------------------
MODEL     = "meta-llama/Llama-3.2-1B-Instruct"   # any instruct checkpoint
DATA_PATH = "data/sft.jsonl"                     # JSONL: one {"messages":[...]} per line
OUT_DIR   = "./out-lora"
SEED      = 42                                    # change per run; record it
# -------------------------------------------------------------------------
torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"                        # left for batched generation only

# ---- Data: split by SOURCE in real projects, not by row ------------------
ds = load_dataset("json", data_files=DATA_PATH, split="train").shuffle(seed=SEED)
n_val = max(50, int(0.1 * len(ds)))
val, train = ds.select(range(n_val)), ds.select(range(n_val, len(ds)))
print(f"train={len(train)} val={len(val)}")

# ---- Chat-template preflight: DO NOT SKIP --------------------------------
ex = train[0]["messages"]
train_str = tok.apply_chat_template(ex, tokenize=False, add_generation_prompt=False)
prod_str  = tok.apply_chat_template(ex[:-1], tokenize=False, add_generation_prompt=True)
assert tok.eos_token_id in tok(train_str).input_ids, "EOS missing -> never stops"
print("TRAIN:", repr(train_str[-140:]));  print("PROD :", repr(prod_str[-140:]))

# ---- Model ---------------------------------------------------------------
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    attn_implementation="sdpa")                   # "flash_attention_2" if installed
model.config.use_cache = False                    # required with gradient checkpointing
model = prepare_model_for_kbit_training(model)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])

args = SFTConfig(
    output_dir=OUT_DIR, num_train_epochs=1,
    per_device_train_batch_size=2, per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,             # global batch = 2*8 = 16 seqs x 1024 tok
    learning_rate=1e-4,                        # QLoRA. LoRA bf16: 2e-4. Full FT: 2e-5.
    lr_scheduler_type="cosine", warmup_ratio=0.03,
    weight_decay=0.01, max_grad_norm=1.0, optim="paged_adamw_8bit",
    bf16=True, fp16=False, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=1024,                       # set from your p99, not a round number
    packing=False,                             # True only with masked boundaries
    eval_strategy="steps", eval_steps=50,
    save_strategy="steps", save_steps=50, save_total_limit=2,
    logging_steps=10, logging_first_step=True, eval_accumulation_steps=1,
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    seed=SEED, data_seed=SEED, dataloader_num_workers=2, report_to="none")

trainer = SFTTrainer(model=model, args=args, train_dataset=train,
                     eval_dataset=val, processing_class=tok,
                     callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

n = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
assert n > 0, "Nothing trainable: check target_modules names"
print(f"trainable={n:,} ({100*n/sum(p.numel() for p in trainer.model.parameters()):.3f}%)"
      f"  ln(V)={math.log(model.config.vocab_size):.2f}")

trainer.train(); trainer.save_model(OUT_DIR); tok.save_pretrained(OUT_DIR)
json.dump({"base_model": MODEL, "trainable_params": n, "n_train": len(train),
           "n_val": len(val), "seed": SEED, "data_path": DATA_PATH,
           "config": {k: str(v) for k, v in args.to_dict().items()}},
          open(f"{OUT_DIR}/manifest.json", "w"), indent=2)
print(f"\nSaved to {OUT_DIR}. Next: merge (bf16) -> quantize -> RE-EVALUATE -> canary 5%.")
```

**`data/sft.jsonl`** — one JSON object per line; the assistant turn is the only supervised span.

```json
{"messages":[{"role":"system","content":"You extract structured data. Reply with JSON only."},{"role":"user","content":"Invoice 4471 from Acme Corp, dated 2026-03-02, total $12,480.00."},{"role":"assistant","content":"{\"invoice_id\":\"4471\",\"vendor\":\"Acme Corp\",\"date\":\"2026-03-02\",\"total\":12480.00,\"currency\":\"USD\"}"}]}
{"messages":[{"role":"system","content":"You extract structured data. Reply with JSON only."},{"role":"user","content":"Refund of 89.99 EUR issued to R. Mehta on 01/15/2026, ref RF-2201."},{"role":"assistant","content":"{\"invoice_id\":\"RF-2201\",\"vendor\":null,\"date\":\"2026-01-15\",\"total\":-89.99,\"currency\":\"EUR\"}"}]}
```

**Verify, then evaluate** — strict JSON match and continuous token F1, base vs adapter:

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
BASE, ADAPTER, DATA = "meta-llama/Llama-3.2-1B-Instruct", "./out-lora", "data/eval.jsonl"
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.pad_token or tok.eos_token
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]

def score(m, name):
    strict = cont = 0.0
    for r in rows:
        ms = r["messages"]
        ids = tok(tok.apply_chat_template(ms[:-1], tokenize=False, add_generation_prompt=True),
                  return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        pred = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        gold = ms[-1]["content"].strip()
        try:    strict += float(json.loads(pred) == json.loads(gold))
        except Exception: strict += 0.0
        a, b = set(pred.split()), set(gold.split())
        cont += (2 * len(a & b) / (len(a) + len(b))) if (a or b) else 0.0
    n = len(rows); print(f"{name:<8} strict={strict/n:.3f} tokenF1={cont/n:.3f} n={n}")
    return strict / n, cont / n

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
s0, f0 = score(base, "base")
s1, f1 = score(PeftModel.from_pretrained(base, ADAPTER), "adapter")
print(f"delta   strict={s1-s0:+.3f} tokenF1={f1-f0:+.3f}")
# RELEASE only if delta > 0 on BOTH and the regression suite is unchanged.
```

**Eight checks before you launch:**

| # | Check | Pass condition |
|---|---|---|
| 1 | Rendered chat string ends with the assistant header **and** EOS | `repr(train_str[-140:])` |
| 2 | EOS survives tokenization | `tok.eos_token_id in tok(train_str).input_ids` |
| 3 | Trainable params > 0 | Expect ~0.6% of total at r=16 |
| 4 | `max_seq_length` ≥ p99 of your data | Truncation < 2% |
| 5 | Split by source, no MinHash near-duplicates | Leakage check run |
| 6 | Prompt-only and base-model baselines recorded | Written down *before* training |
| 7 | LR matches the method (2e-5 / 2e-4 / 1e-4) | Config reviewed against §4 |
| 8 | VRAM has 10–15% headroom | `nvidia-smi` during warmup, not at idle |

---

## 13. What To Read Next

| If you need… | Go to |
|---|---|
| The reasoning behind every number here | **CS-01** — Foundations: Pretraining, Training & the LLM Lifecycle |
| Interview practice on this material | **IQ-01** — 100 questions (30/30/22/8/10 by level) |
| Why the Transformer unlocked fine-tuning | **CS-05** — RNN/LSTM → Attention |
| Transfer learning and the two ways to fine-tune | **CS-02** — Transfer Learning & Model Fine-Tuning |
| Choosing the right framework in 2025 | **CS-03** — The Framework Landscape |
| Deciding FT vs RAG vs agents properly | **CS-04** — Choosing the Architecture |
| The Hugging Face stack in depth | **CS-06** — Hugging Face Masterclass |
| Encoders: BERT for NER, sentiment, QA | **CS-07** — Fine-Tuning BERT |
| Quantization internals (PTQ, QAT, GPTQ, AWQ, GGUF) | **CS-10 / CS-11** |
| Domain-adaptive continued pretraining | **CS-12** |
| Instruction tuning (SFT) in depth | **CS-13** |
| RLHF, PPO, DPO, ORPO, GRPO | **CS-14 §4.6.1–4.6.10** |
| Faster / lower-VRAM training in practice | **CS-16** (Unsloth), **CS-17** (Axolotl), **CS-15** (LLaMA-Factory) |
| Small language models | **CS-09** (LLM → SLM); **CS-20** planned, not yet written |
| Embedding models and retrieval quality | `code/10_embedding_finetune.py`; **CS-22** planned, not yet written |
| LoRA and QLoRA from first principles | **CS-13 §6.8** / **CS-11 §4.11** |
| The whole pipeline end to end | **CS-28** — Capstone, planned, not yet written; **CS-13** + **CS-16/CS-17** are the written run-throughs |

**CS-20–CS-28 are planned, not yet written** (README status table). Where a row above names an unwritten module, the written nearest-equivalent is named beside it: **CS-20** → **CS-09**; **CS-21** (multimodal) → `code/12_multimodal_vlm.py`; **CS-22** → `code/10_embedding_finetune.py`; **CS-23** → **CS-13 §6.8** + **CS-11 §4.11**; **CS-24–CS-27** → **CS-14 §4.6.1–4.6.10**; **CS-28** → **CS-13** + **CS-16/CS-17**.
