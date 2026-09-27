# CH-13 — Instruction Fine-Tuning (SFT) Cheat Sheet

**One-line purpose:** turn a pretrained/base model into one that follows instructions, by
training it on (instruction → response) pairs with the loss applied **only to the response**.
**Use when:** you need the model to reliably do a *task* in a *format* — JSON output, a
house style, a domain tone, tool-call syntax, a classification rubric.
**Do NOT use when:** you only need it to *know* new facts (that is RAG or continued
pretraining — see CH-04 and CH-12), or you only need preference/quality tuning on top of an
already-instruction-tuned model (that is DPO/ORPO — see CH-14).

> **The one sentence that matters.** SFT teaches *behaviour*, not *knowledge*. If you are
> reaching for SFT to make the model know something, you are using the wrong tool.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **Mask the prompt.** Loss on the completion only. | Otherwise the model learns to generate your questions. |
| 2 | **The chat template at train time must equal the one at inference time.** | Mismatch is the #1 silent quality killer. No error is raised. |
| 3 | **Append EOS to every target.** | Without it the model never learns to stop; it rambles or runs to `max_new_tokens`. |
| 4 | **1k–10k high-quality examples beats 100k scraped ones.** | LIMA. Quality is the entire game below ~50k. |
| 5 | **LoRA LR ≈ 1e-4…2e-4. Full-FT LR ≈ 1e-5…2e-5.** | 10× apart. Using the LoRA LR for full FT destroys the model. |
| 6 | **1–3 epochs.** 3 is already a lot. | SFT overfits fast; epoch 4 usually *hurts* general ability. |
| 7 | **Training loss < ~0.5 on SFT means memorisation.** | Not success. Check held-out data. |
| 8 | **Narrow SFT causes catastrophic forgetting.** | Mix 5–10% general instruction data back in as replay. |
| 9 | **QLoRA r16 trains a 7B on ~5 GB.** | Full FT of the same 7B needs ~92 GB. |
| 10 | **Always run the validator before the GPU.** | Every failure in §8 is caught in seconds by a `--dry-run`. |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Masking** | `labels[i] = -100` for prompt tokens | −100 = `CrossEntropyLoss(ignore_index=)` | 512-token row, 400 prompt + 112 answer → 400 masked |
| **Supervised fraction** | `n_supervised / n_total` | — | 112/512 = 22%. Under 10% is a red flag. |
| **Effective batch** | `per_device × grad_accum × n_gpu` | — | 2 × 8 × 1 = 16 |
| **LR scaling (LoRA)** | `lr ∝ 1/√r` roughly; alpha = 2r | r = rank | r=16 → alpha=32 |
| **Training FLOPs** | `6 × N × D` | N = params, D = tokens | 7B, 10M tokens → 4.2e17 FLOPs |
| **Tokens from words** | `tokens ≈ words × 1.33` | English, BPE | 1,000 words ≈ 1,330 tokens |
| **Chars from tokens** | `tokens ≈ chars / 4` | English | 4,000 chars ≈ 1,000 tokens |
| **Steps per epoch** | `ceil(N / eff_batch)` | N = examples | 5,000 / 16 = 313 steps |
| **Epochs from steps** | `steps × eff_batch / N` | — | 300 × 16 / 5000 = 0.96 |
| **Warmup steps** | `total_steps × warmup_ratio` | — | 313 × 0.03 ≈ 10 steps |
| **Adapter size (LoRA)** | `r × (d_in + d_out)` per matrix | — | r=16 on 4096×4096 → 131k params |
| **LoRA param fraction** | `~r × n_target / (12 × d)` | d = hidden | r=16, 7 layers → ~0.5–2% |
| **Initial loss (random head)** | `ln(K)` | K = classes | 3 classes → 1.099 |
| **NEFTune** | `emb += uniform(-α, α) / √(L·d)` | L = seq len, d = dim | α=5 is the paper default |

---

## 3. Decision Tree

```
Do you have (instruction, response) pairs already?
├─ No  → generate them: code/data/make_instruction_data.py, or distil from a
│        stronger model (see CS-09). Do NOT hand-write 10,000 by hand.
└─ Yes ↓

Is the base model already instruction-tuned?
├─ Yes (Instruct/Chat variant) → SFT will be FAST and can DEGRADE it.
│     → Use a LOW LR (5e-5…1e-4 with LoRA), 1–2 epochs, and mix in replay data.
└─ No (base model) → you need more data and more epochs. 1e-4…2e-4 LoRA, 2–3 epochs.

Can you fit FULL fine-tuning in VRAM?
├─ Yes (see §7) and you have ≥50k examples → full FT is worth it.
│     LR 1e-5…2e-5, warmup 3–10%, 2–3 epochs.
└─ No / smaller data → LoRA (r=16, alpha=32, all linear modules).
      └─ Still OOM? → QLoRA (4-bit NF4 + double quant + paged AdamW).
            └─ Still OOM? → shorten max_seq_len, or gradient accumulation
                            with batch 1, or a smaller base model.

Is your task FORMAT-ONLY (fixed JSON schema, no reasoning)?
├─ Yes → consider STRUCTURED OUTPUTS / constrained decoding FIRST.
│        It is cheaper, deterministic, and cannot be forgotten.
│        Fine-tune only if the CONTENT also needs to change.
└─ No ↓

Is your task KNOWLEDGE (new facts, new documents)?
├─ Yes → RAG. SFT will not reliably install facts and will hallucinate
│        confidently around them. (CH-04 has the full decision.)
└─ No → you have a behaviour task. SFT is correct. Proceed.

Do you have a held-out eval set?
├─ No  → STOP. Build one first. You cannot tell success from
│        memorisation without it, and you will ship a worse model
│        while the training loss looks better.
└─ Yes → train.
```

---

## 4. Hyperparameter Quick Reference

### 4.1 The knobs that actually matter

| Param | LoRA default | QLoRA default | Full FT default | Sweep range | Effect of getting it wrong |
|---|---|---|---|---|---|
| `learning_rate` | 2e-4 | 2e-4 | 2e-5 | 5e-5 … 3e-4 (LoRA); 5e-6 … 5e-5 (full) | Too high → loss spikes, model degrades. Too low → nothing is learned, loss plateaus high. |
| `num_train_epochs` | 2 | 2 | 2 | 1 … 3 | 3+ overfits; 4+ forgets. |
| `per_device_train_batch_size` | 2 | 1 | 1 | 1 … 8 | Bigger is better for stability but is the main VRAM consumer. |
| `gradient_accumulation_steps` | 8 | 8 | 8 | 4 … 32 | Restores effective batch without VRAM. |
| `max_seq_len` | 2048 | 2048 | 2048 | 1024 … 8192 | Cost is ~quadratic in attention; truncation silently drops the answer. |
| `warmup_ratio` | 0.03 | 0.03 | 0.06 | 0.01 … 0.10 | No warmup → early large updates damage the base model. |
| `lr_scheduler_type` | cosine | cosine | cosine | cosine / linear | Linear is fine; constant is usually worse. |
| `weight_decay` | 0.0–0.01 | 0.01 | 0.01 | 0 … 0.1 | Minor. |
| `optim` | `adamw_8bit` | `paged_adamw_8bit` | `adamw_torch_fused` | — | 8-bit halves optimiser memory; paged survives spikes. |
| `gradient_checkpointing` | `True` | `True` | `True` | — | ~30% more time, ~50–60% less activation memory. Usually mandatory. |
| `packing` | `False` | `False` | `True` | — | See §10 gotcha #4 — packing without position-id resets causes cross-contamination. |
| `bf16` | `True` | `True` | `True` | — | Prefer `bf16` over `fp16`; fp16 needs loss scaling and can NaN. |
| `train_on_prompt` | `False` | `False` | `False` | — | Setting `True` teaches the model to write your prompts. |
| `neftune_noise_alpha` | 5 | 5 | 5 | 0 … 15 | Free quality win on SFT. `0` disables. |

### 4.2 LoRA-specific

| Param | Default | Range | Notes |
|---|---|---|---|
| `r` (rank) | 16 | 4 … 128 | Higher = more capacity + more overfit risk. 8–32 covers most work. |
| `lora_alpha` | 32 | `2×r` | Scale = `alpha / r`. Setting alpha = 2r keeps effective scale ~2. **Take `alpha` and `lr` together when you port a recipe** — `alpha=r` (scale 1.0) is equally common, and mixing the two conventions is a silent 2× LR change. See CS-13 §7. |
| `lora_dropout` | 0.05 | 0 … 0.1 | 0 is faster; 0.05 helps on small datasets. |
| `target_modules` | all linear | — | `q,k,v,o,gate,up,down`. Attention-only is cheaper but weaker. |
| `bias` | `"none"` | — | `"none"` standard; `"all"` rarely helps. |
| `use_rslora` | `False` | — | Rank-stabilised: scale = `alpha/√r`. Helps at high rank. |
| `use_dora` | `False` | — | Weight-decomposed LoRA. Better at very low rank, slower. |

### 4.3 Data sizing rules of thumb

| Goal | Examples needed | Evidence |
|---|---|---|
| Format / style adoption | 100 – 1,000 | LIMA: 1,000 examples was enough for strong instruction following |
| A narrow task | 500 – 5,000 | The classic Alpaca recipe used 52k but 2–5k curated does better |
| Broad behaviour change | 5,000 – 100,000 | Below 5k you trade general ability for narrow skill |
| A new language / domain tone | 10,000+ | Plus replay of general instruction data |
| Full capability shift (pretraining-like) | 1B+ tokens | This is continued pretraining, not SFT — see CH-12 |

---

## 5. Copy-Paste Code Snippets

### 5.1 Minimal working example — HF Trainer + PEFT (full, runnable)

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token          # Llama has no pad token by default

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto",
)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

ds = load_dataset("json", data_files="data/sample_sft.jsonl", split="train")

def to_text(ex):
    # apply_chat_template with tokenize=False returns the exact string the
    # tokenizer will see. This is what guarantees the train/serve template match.
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

ds = ds.map(to_text)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="out/sft-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,       # effective batch 16
        num_train_epochs=2,
        learning_rate=2e-4,                  # LoRA LR — NOT 2e-5
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=2048,
        optim="adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        neftune_noise_alpha=5,               # free quality win
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        # assistant_only_loss=True,          # TRL ≥0.12: masks the prompt for you
        # packing=True,                      # faster, but read §10 gotcha #4 first
    ),
    train_dataset=ds,
    peft_config=peft_cfg,
)
trainer.train()
trainer.save_model("out/sft-lora")
tok.save_pretrained("out/sft-lora")
```

### 5.2 Doing the loss masking yourself (no TRL)

```python
IGNORE_INDEX = -100

def build_masked_example(tokenizer, messages, max_len=2048):
    """Tokenize incrementally so the assistant span is found EXACTLY.

    Why incremental: tokenize(a + b) != tokenize(a) + tokenize(b) at the join
    point. BPE merges across the boundary, so one token can straddle prompt and
    answer. Tokenizing the prompt alone, then continuing the SAME token stream,
    avoids the whole class of bug.
    """
    input_ids, labels = [], []
    for i, msg in enumerate(messages):
        # add_generation_prompt MUST stay False. It appends the assistant's opening
        # header (e.g. "<|im_start|>assistant\n") to the END of whatever you render.
        # Setting it per-role appends a spurious header AFTER each assistant turn and
        # then supervises it — so the last supervised tokens become a header instead of
        # EOS, and the model learns to open a new assistant turn after every answer.
        rendered = tokenizer.apply_chat_template(
            messages[: i + 1],
            tokenize=False,
            add_generation_prompt=False,   # <- False here is what makes the prefix stable
        )
        # Re-tokenize the growing prefix, but only take the NEW tokens
        new_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        prev_len = len(input_ids)
        ids = new_ids[len(input_ids):] if len(new_ids) > prev_len else new_ids
        input_ids += ids
        if msg["role"] == "assistant":
            labels += ids                       # supervise the answer
        else:
            labels += [IGNORE_INDEX] * len(ids) # mask the prompt

    input_ids, labels = input_ids[:max_len], labels[:max_len]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
```

> If you take one thing from this file: **`labels` must be the same length as
> `input_ids`, with `-100` wherever you do not want the loss computed.** A shape
> mismatch or an all-`-100` row is the two ways this breaks.

### 5.3 Unsloth (2–4× faster, ~60% less VRAM)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=2048, load_in_4bit=True, dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",   # Unsloth's checkpointing is the speed win
    random_state=3407,
)
# ... then a normal SFTTrainer, with per_device_train_batch_size=2, optim="adamw_8bit"
```

### 5.4 LLaMA-Factory (no code at all)

```yaml
# train.yaml — then: llamafactory-cli train train.yaml
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all
dataset: my_data                # registered in data/dataset_info.json
template: llama3                # <-- MUST match the model, see §10 gotcha #1
cutoff_len: 2048
output_dir: out/lf-sft
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
```

---

## 6. CLI Commands

```bash
# ── LLaMA-Factory ────────────────────────────────────────────────────────────
llamafactory-cli train train.yaml                 # run the SFT
llamafactory-cli chat  train.yaml                 # interactive test of the ADAPTER
llamafactory-cli export train.yaml                # merge adapter into base weights
llamafactory-cli webui                            # the no-code UI

# ── TRL / HF ────────────────────────────────────────────────────────────────
python -m trl.scripts.sft --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name my_data --max_seq_length 2048 --num_train_epochs 2 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 --bf16 --use_peft --lora_r 16 --lora_alpha 32

# ── This handbook's scripts ─────────────────────────────────────────────────
python code/01_sft_lora.py --dry-run --data code/data/sample_sft.jsonl
python code/01_sft_lora.py --data code/data/sample_sft.jsonl --out out/sft-lora
python code/02_sft_unsloth.py --dry-run --data code/data/sample_sft.jsonl
python code/09_merge_and_export.py --adapter out/sft-lora --out out/merged

# ── Inspect the chat template BEFORE training (do this every time) ──────────
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
print(t.chat_template)
print(repr(t.apply_chat_template([{'role':'user','content':'hi'},
                                  {'role':'assistant','content':'hello'}],
                                 tokenize=False)))
"
```

---

## 7. VRAM / Cost Calculator

From `code/common/memory.py --table`. Single GPU, gradient checkpointing **on**, AdamW,
bf16 weights, fp32 optimiser states for full FT. GB:

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

**Add 10–20% for allocator and framework overhead, and leave headroom.**

> **Why you will see *different* numbers elsewhere — and both are right.** This table is the
> **arithmetic floor**: it prices the tensors and nothing else. Other sources quote higher
> figures for the same model and method. Three reasons, in order of size:
>
> 1. **Bytes per parameter.** Full FT here is `4 B` master weights `+ 2 B` gradients
>    `+ 8 B` AdamW (`m` + `v`) `= 14 B/param` → 7B × 14 = **91.6 GiB**. The common
>    back-of-envelope uses **16 B/param** (an extra `2 B` for a separate bf16 working copy)
>    → 7B × 16 = **112 GB**. Same recipe, +22%, purely a bookkeeping choice.
> 2. **GiB vs GB.** The table is binary (1024³). Vendor slide figures are usually decimal
>    (1000³). That is another **+7.4%** on every row — 91.6 GiB is 98.4 GB.
> 3. **QLoRA dequantises on the fly.** bitsandbytes stores the base in NF4 but upcasts each
>    linear layer to bf16 to run the matmul, so peak memory transiently holds NF4 + bf16 for
>    the layer in flight. It is *not* simply ¼ of bf16, which is why a 7B QLoRA run that the
>    floor prices at **4.9 GiB** is commonly observed at **10–12 GB** once the CUDA context
>    (~0.5–1 GB), the tokenizer/dataloader, longer `cutoff_len` activations and allocator
>    fragmentation are added.
>
> **How to use this:** treat the table as the *lower bound you must beat*, add 10–20% for a
> plan, and trust a real measurement over any table — including this one. If a number here
> and a number in the case study disagree by ~20–30%, check bytes-per-param and GiB-vs-GB
> before assuming one of them is wrong.

### Why full FT is ~20× QLoRA

| Term | Full FT | LoRA | QLoRA |
|---|---|---|---|
| Weights | 2 bytes × N | 2 bytes × N (frozen) | 0.5 bytes × N (NF4) |
| Gradients | 2 bytes × N | ~0 (adapter only) | ~0 |
| Optimiser states | **8 bytes × N** (fp32 m + v) | ~0 | ~0 (8-bit paged) |
| Adapter | — | `~0.5–2%` of N, ×2–4 bytes | same |
| **Dominant term** | **12N** | **2N + tiny** | **0.5N + tiny** |

The optimiser states are the whole story. AdamW keeps two fp32 moments per trainable
parameter — that is **8 bytes per parameter** before you have stored a single activation.

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| Loss flat near `ln(vocab)` ≈ 10.4, never falls | Everything masked, or LR = 0, or wrong labels | Print `has_supervision(ex)`; count non-`-100` labels. |
| Loss falls beautifully but the model is worse | **Training/serving chat-template mismatch** | Render with the same `apply_chat_template` you serve with. Test the adapter interactively. |
| Model never stops generating | EOS not in the targets | Append `eos_token` to each target; check the template includes it. |
| Model answers its own questions / writes the prompt | `train_on_prompt=True`, or masking is off by a turn | Mask every non-assistant turn. |
| Loss ≈ 0.05 by epoch 2 | Memorisation | Fewer epochs, more data, higher dropout, add eval set. |
| General ability collapsed after SFT | Catastrophic forgetting | Lower LR, 1–2 epochs, LoRA instead of full FT, mix 5–10% replay data. |
| Replies are fluent but generic / not in domain | Under-trained, or wrong target modules | More epochs or higher LR; ensure `gate/up/down` are in `target_modules`. |
| Output is truncated mid-sentence in training data | `max_seq_len` too small | Raise it, or filter rows exceeding it. Check p95 length first. |
| `RuntimeError: expected all tensors on same device` | Model sharded, inputs not | `device_map="auto"` plus let the Trainer place inputs; or use a single device. |
| Loss spikes to NaN around step ~50 | LR too high (esp. full FT with a LoRA LR) | Drop LR 5–10×; add warmup; switch fp16 → bf16. |
| OOM at batch size 1 | `max_seq_len` too large | Halve `max_seq_len`; enable gradient checkpointing; switch to QLoRA. |
| OOM only at eval time | Eval batch too large / no `torch.no_grad` | `per_device_eval_batch_size = train_batch × 2`; eval fewer steps. |
| Adapter loads but output is unchanged | Adapter not applied, or wrong base model | `model.load_adapter(...)` then `merge_and_unload()`; verify the base revision. |
| Quality varies wildly run to run | No seed, or LR at the edge of stability | Set `seed=42`; lower LR; more warmup. |
| Val loss rises while train loss falls at epoch 2 | Classic overfit | Stop at epoch 1–2; use `load_best_model_at_end`. |
| JSON output invalid ~5% of the time | Format learned statistically, not guaranteed | Constrained decoding / structured outputs, or a validator + retry. |

---

## 9. Comparison Matrix

### 9.1 Full FT vs LoRA vs QLoRA

| Dimension | Full FT | LoRA | QLoRA |
|---|---|---|---|
| Trainable params | 100% | 0.1–2% | 0.1–2% |
| 7B training VRAM | ~92 GB | ~15 GB | ~5 GB |
| Quality ceiling | Highest | Close (within noise on most tasks) | Slightly below LoRA |
| Catastrophic forgetting | Worst | Moderate | Moderate |
| Mergeable | n/a | ✅ lossless | ✅ (dequantise first) |
| Serving cost | 1 model | +0% (merged) | +0% (merged) |
| Multi-adapter serving | ❌ | ✅ (vLLM `--enable-lora`) | ✅ |
| Best when | ≥50k examples, big GPU, max quality | The default | GPU-poor |

### 9.2 SFT vs the alternatives

| Need | SFT | RAG | Continued pretraining | DPO/ORPO |
|---|---|---|---|---|
| Teach a format | ✅ best | ❌ | ❌ | ⚠️ |
| Teach a style/tone | ✅ best | ⚠️ | ⚠️ | ✅ |
| Install new facts | ❌ unreliable | ✅ best | ✅ | ❌ |
| Add a new language | ⚠️ | ❌ | ✅ best | ❌ |
| Fix "knows but won't do it" | ✅ | ❌ | ❌ | ✅ |
| Make it prefer A over B | ⚠️ | ❌ | ❌ | ✅ best |
| Data format | (prompt, response) | a corpus | raw text | (prompt, chosen, rejected) |
| Cost to run | Low | Very low | High | Low |

### 9.3 Where each framework fits

| Framework | Interface | Best for | Watch out |
|---|---|---|---|
| HF `Trainer` + PEFT | Python | Full control, research | You write the masking |
| TRL `SFTTrainer` | Python | Fastest correct default | `assistant_only_loss` needs a recent version |
| Unsloth | Python | Single GPU, 2–4× faster | Its kernels are patched per model family |
| LLaMA-Factory | YAML / WebUI | No code, 100+ models | `template:` must match the model |
| Axolotl | YAML | Reproducible multi-GPU | YAML schema churns |
| torchtune | Python recipes | Minimal, readable | Smaller model zoo |

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| `IGNORE_INDEX` | **−100** | `CrossEntropyLoss` default `ignore_index` |
| LoRA LR | **1e-4 … 2e-4** | 2e-4 is the common default |
| Full-FT LR | **1e-5 … 2e-5** | 10× below LoRA |
| Encoder (BERT) LR | **2e-5** | Not 2e-4 — see CH-07 |
| Epochs | **1–3** | 2 is the default answer |
| Warmup | **3–10%** of steps | |
| Effective batch | **32–128** | via gradient accumulation |
| LoRA `r` | **8–32** | 16 typical |
| `lora_alpha` | **2 × r** | 32 when r=16 |
| LoRA dropout | **0.05** | 0.0 for speed |
| NEFTune α | **5** | paper default |
| Tokens per word | **≈1.33** | English |
| Chars per token | **≈4** | English |
| Bytes/param, bf16 | **2** | weights |
| Bytes/param, fp32 | **4** | weights |
| Bytes/param, AdamW state | **8** | 2 × fp32 moments |
| Bytes/param, NF4 | **0.5** | 4-bit + double quant |
| Bytes/param, int8 | **1** | LLM.int8() |
| Training FLOPs | **6ND** | forward + backward |
| Attention cost | **O(L²)** | the reason long context is expensive |
| SFT overfit signal | **loss < 0.5** | memorisation, not success |
| Initial loss, random head | **ln(K)** | K classes |
| LIMA dataset size | **1,000** | quality beats quantity |
| QLoRA vs full FT VRAM | **~20×** | 4.9 vs 91.6 GB at 7B |
| Typical supervised fraction | **10–40%** | of tokens; <10% is a red flag |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `Cannot handle this data type: ...` | Dataset column type unsupported | `dataset.map(..., remove_columns=...)` or `remove_unused_columns=False` |
| `The following columns are unused: [...]` | Trainer dropped your text column | Set `remove_unused_columns=False` for a custom collator |
| `element 0 of tensors does not require grad` | Base model frozen *and* adapter not attached | Confirm `peft_config` was passed / `get_peft_model` called |
| `Expected a batch size of ...` / ragged tensors | Padding not applied | Use a collator with `padding=True, return_tensors="pt"` |
| `Token indices sequence length is longer than the specified maximum` | A row exceeds `max_seq_len` | Expected warning; confirm the *answer* is not the part being cut |
| `CUDA out of memory. Tried to allocate X GiB` | See §8 | Halve batch or seq len; QLoRA; gradient checkpointing |
| `torch.cuda.OutOfMemoryError` at *validation* | Eval batch too big | Lower `per_device_eval_batch_size` |
| `ValueError: num_samples should be a positive integer` | Dataset empty after filtering | Check the filter that produced it — usually a schema mismatch |
| `KeyError: 'messages'` | Wrong field name for your loader | Align the loader with the file: `instruction/output`, `prompt/completion`, `messages` |
| `OSError: ... does not appear to have a file named config.json` | Wrong model id / not downloaded | Check the id and your HF cache |
| `UserWarning: pad_token_id is not set` | Model has no pad token | `tok.pad_token = tok.eos_token` (and match it at inference) |
| `AssertionError: ... attention mask ...` with packing | Packed sequences not separated | Reset `position_ids` per example, or turn packing off |
| `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | Optimiser has no trainable params | Print `sum(p.requires_grad for p in model.parameters())` |
| `ImportError: cannot import name 'SFTConfig'` | TRL too old/new | `pip install -U trl`; `SFTConfig` replaced `TrainingArguments` in TRL 0.12 |

---

## 12. Copy-Paste Starter Config

`train.yaml` — a correct, conservative SFT starting point. Change **only** the marked lines
for your first run.

```yaml
# ── CHANGE THESE ────────────────────────────────────────────────────────────
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
dataset: my_data              # registered in data/dataset_info.json
output_dir: out/sft-lora
template: llama3              # MUST match the model family. Getting this wrong is
                              # the #1 silent failure — see §8.
# ── CHANGE NOTHING BELOW FOR RUN ONE ────────────────────────────────────────
stage: sft
do_train: true

finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 4

per_device_train_batch_size: 2
gradient_accumulation_steps: 8      # effective batch 16
learning_rate: 2.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
max_grad_norm: 1.0
bf16: true
gradient_checkpointing: true

logging_steps: 5
save_strategy: epoch
plot_loss: true
report_to: none

# ── VALIDATE WITH THESE BEFORE YOU TRUST THE RUN ────────────────────────────
val_size: 0.05
per_device_eval_batch_size: 2
eval_strategy: epoch
load_best_model_at_end: true
```

**Run one.** Then, in this order: render the chat template and eyeball it → confirm the
masked fraction is 10–40% → confirm the first loss is well below `ln(vocab)` and falling
smoothly → test the adapter interactively on 10 held-out prompts → *then* consider touching
a hyperparameter.

### The five checks before every run

> Checks 1 and 2 need `transformers` (and a real tokenizer) installed — they are for the
> machine you train on, not a bare laptop. Checks 3–5 are pure Python and run anywhere.
>
> Note the API shape: `load_jsonl` returns a list of **message lists**, not dicts. You must
> build examples with `build_masked_example` *before* `token_stats` can count supervised
> tokens. Passing the raw conversations to `token_stats` raises
> `TypeError: list indices must be integers or slices, not str` — that error means you
> skipped the build step.

```bash
# 1. Is the template what you think it is?
python -c "from transformers import AutoTokenizer as A; t=A.from_pretrained('MODEL'); \
print(repr(t.apply_chat_template([{'role':'user','content':'hi'}], tokenize=False)))"

# 2. How many rows have nothing to learn from?
python -c "
import sys; sys.path.insert(0,'code')
from common import data_utils as du
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('MODEL')
c = du.load_jsonl('code/data/sample_sft.jsonl')
ex = [du.build_masked_example(tok, m) for m in c]
s = du.token_stats(ex)
print(s['with_supervision'], '/', s['examples'], 'supervised',
      f\"frac={s['supervised_token_frac']:.1%}\")
"

# 3. What will this cost in VRAM?
python code/common/memory.py --table

# 4. What will this cost in time?
#    steps = ceil(N / eff_batch) * epochs;  multiply by seconds/step from your log

# 5. Does it fit? p95 length vs max_seq_len
python -c "
import json,statistics as st
L=[len(json.dumps(json.loads(l)))//4 for l in open('code/data/sample_sft.jsonl',encoding='utf-8')]
print('p50',st.median(L),'p95',sorted(L)[int(len(L)*.95)],'max',max(L))
"
```

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| Understand *why* SFT works, from first principles | **CS-13 — Instruction Fine-Tuning** (the full treatment) |
| Know when SFT is the wrong tool at all | **CH-04 / CS-04 — Fine-Tuning vs RAG vs Agents** |
| Train on preferences instead of demonstrations | **CH-14 / CS-14 — The Alignment Map** |
| Understand what LoRA is doing mathematically | **CS-13 §6.8**; **CS-11 §4.11** (CS-23 planned, not yet written) |
| Train without a GPU budget | **CS-16 — Unsloth**, **CS-15 — LLaMA-Factory** |
| Install facts rather than behaviour | **CH-12 / CS-12 — Domain-Adaptive Continued Pretraining** |
| Make a smaller model that behaves like a bigger one | **CH-09 / CS-09 — Distillation: LLM → SLM** |
| See the exact trainer code with the traps pre-checked | `code/01_sft_lora.py`, `code/02_sft_unsloth.py` |
| Practice being interviewed on this | **IQ-13 — Interview Questions: Instruction Fine-Tuning** |

