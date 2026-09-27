# CH-06 — The Hugging Face Masterclass Cheat Sheet

**One-line purpose:** the library surface you touch in *every* fine-tuning run —
`transformers`, `datasets`, `tokenizers`, `peft`, `trl`, `accelerate`, `evaluate`, and the Hub.
**Use when:** always. This is the substrate the other modules sit on.
**Do NOT use when:** never — but know when to drop below it. If you need a custom loss,
a custom collator, or control over the backward pass, you are writing PyTorch, and the
abstraction is now in your way.

> **The one sentence that matters.** Almost every "my fine-tune silently failed" story is one
> of four things: the wrong `AutoModel*` class, the wrong `padding_side`, a double-added
> special token, or `datasets.map` losing the column you needed. This card is mostly those
> four.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | `AutoModel` has **no LM head**. `AutoModelForCausalLM` does. | The #1 "why can't it generate" bug. |
| 2 | **`padding_side="left"` for generation, `"right"` for training.** | Right-padding + batched generation = garbage continuations. |
| 3 | Llama has **no pad token**. Set `pad_token = eos_token`. | Otherwise padding crashes or silently corrupts. |
| 4 | `tokenizer(text)` adds special tokens **by default**. | Double-BOS when you also call `add_special_tokens=True` on a template-rendered string. |
| 5 | `datasets.map(batched=True)` receives **lists**, returns **lists**. | Writing scalar code in a batched map is the classic crash. |
| 6 | `remove_columns` matters. | Unused columns reach the Trainer and break the collator. |
| 7 | `device_map="auto"` needs `accelerate` installed. | Otherwise you get a confusing error, not a fallback. |
| 8 | `safetensors` is the default and is **safer than pickle**. | `.bin` loading executes arbitrary code. Prefer `safetensors`. |
| 9 | `trust_remote_code=True` **runs arbitrary code**. | Only for repos you have read. |
| 10 | `num_proc` speeds up `map` but **pickles** your function. | Closures over unpicklable objects fail only when `num_proc > 1`. |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Model VRAM (bf16)** | `params × 2` bytes | — | 7B → 14 GB |
| **Full FT VRAM** | `params × (2 + 2 + 8)` ≈ `12 × params` | weights + grads + AdamW | 7B → ~84 GB (+activation) |
| **Shard count** | `ceil(size / max_shard_size)` | default 5 GB | 14 GB → 3 shards |
| **Tokens per word** | `× 1.33` | English | |
| **Dataset memory** | `rows × avg_bytes` | arrow-backed, on disk | 1M rows × 2 KB = 2 GB |
| **Batch tokens** | `batch × seq_len` | — | 8 × 2048 = 16,384 |
| **Tokenizer speedup** | fast (Rust) ≈ **10–100×** slow (Python) | `use_fast=True` | Default since v4 |

---

## 3. Decision Tree

```
What do you want to DO with a model?
├─ Generate text                    → AutoModelForCausalLM
├─ Classify / regress               → AutoModelForSequenceClassification
├─ Tag tokens (NER)                 → AutoModelForTokenClassification
├─ Embed (no head)                  → AutoModel  (+ mean/cls pooling)
├─ Sentence embeddings              → sentence-transformers
├─ Answer questions over context    → AutoModelForQuestionAnswering
├─ Vision-language                  → AutoModelForVision2Seq / the family's own class
└─ Just features, no head           → AutoModel  ← the "why can't it generate" trap

Is this BATCHED GENERATION?
├─ Yes → tokenizer.padding_side = "left"    ← or outputs are garbage
└─ No  → padding_side = "right" is fine (and is the default)

Does the tokenizer have a pad token?
├─ No (Llama, many decoder-only) → pad_token = eos_token
└─ Yes → leave it alone (Gemma, Qwen have real pad tokens)

Are you calling apply_chat_template?
├─ Yes → tokenize=False, then tokenize the STRING with add_special_tokens=False
│        (the template already inserts the special tokens)
└─ No  → tokenizer(text) with defaults is fine

Does your dataset.map need extra columns?
├─ Yes → remove_columns=[...] explicitly, or the Trainer will choke
└─ No  → set remove_unused_columns=False on TrainingArguments

Do you need speed on a big dataset?
├─ Yes → num_proc=4 and a MODULE-LEVEL function (picklable)
└─ No  → a plain map is fine and easier to debug
```

---

## 4. Hyperparameter Quick Reference

### 4.1 Tokenizer settings that change your results

| Setting | Default | Set it to | Why |
|---|---|---|---|
| `padding_side` | `"right"` | **`"left"` for generation** | Right-padding makes the model continue from pads |
| `pad_token` | varies | `= eos_token` if `None` | Llama et al. have none |
| `truncation_side` | `"right"` | depends | Truncating the *left* can drop the instruction |
| `model_max_length` | model-specific | don't exceed it | Beyond it, positions are undefined |
| `add_special_tokens` | `True` | **`False`** when the text came from a template | Double-BOS |
| `use_fast` | `True` | leave `True` | 10–100× faster, same output |
| `clean_up_tokenization_spaces` | `True` | `False` for round-trip fidelity | It "fixes" spaces and breaks exact decode |

### 4.2 `from_pretrained` arguments

| Arg | Typical | Effect |
|---|---|---|
| `torch_dtype` / `dtype` | `torch.bfloat16` | **Renamed to `dtype` in newer transformers.** Check your version. |
| `device_map` | `"auto"` | Needs `accelerate`. Shards across GPUs/CPU. |
| `load_in_4bit` | `True` | Needs `bitsandbytes`. QLoRA path. |
| `attn_implementation` | `"flash_attention_2"` / `"sdpa"` | `sdpa` is a safe default; FA2 needs a compatible GPU |
| `low_cpu_mem_usage` | `True` | The default in recent versions |
| `trust_remote_code` | `False` | **Security.** Only for repos you have read. |
| `use_safetensors` | `True` | Refuse pickle-format weights |
| `revision` | a branch/sha | **Pin it** for reproducibility |

### 4.3 `TrainingArguments` keys that bite

| Key | Set it to | Why |
|---|---|---|
| `remove_unused_columns` | `False` for a custom collator | Otherwise your extra columns are deleted |
| `report_to` | `"none"` with no tracker | Avoids W&B auth hangs |
| `bf16` | `True` on Ampere+ | Prefer over fp16 (no loss scaling, no NaN) |
| `gradient_checkpointing` | `True` | ~30% slower, ~50–60% less activation memory |
| `save_safetensors` | `True` | Default; keep it |
| `dataloader_num_workers` | `0` when debugging | Workers + CUDA can deadlock |
| `seed` | `42` | Reproducibility |
| `optim` | `"adamw_torch_fused"` | Faster on modern GPUs |

---

## 5. Copy-Paste Code Snippets

### 5.1 The correct load, for generation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL)
# 1. Llama has no pad token — give it one BEFORE anything else touches it.
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# 2. Batched generation REQUIRES left padding.
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,      # older transformers: torch_dtype=
    device_map="auto",          # requires: pip install accelerate
)
model.eval()

msgs = [[{"role": "user", "content": "In one sentence, what is LoRA?"}]]
prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
           for m in msgs]

enc = tok(prompts, return_tensors="pt", padding=True,
          add_special_tokens=False).to(model.device)   # template already added them
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=128, do_sample=False,
                         pad_token_id=tok.pad_token_id)
# Slice off the PROMPT — and note it is variable length under left padding.
gen = out[:, enc["input_ids"].shape[1]:]
print(tok.batch_decode(gen, skip_special_tokens=True)[0])
```

> **Why `add_special_tokens=False` here.** `apply_chat_template` has already inserted the
> model's real special tokens. Letting the tokenizer add them again produces
> `BOS BOS <|user|>...`, which the model was never trained on. It usually still generates
> *something*, which is why this bug survives to production.

### 5.2 The correct load, for training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"        # right padding for training

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)

args = TrainingArguments(
    output_dir="out/run",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    remove_unused_columns=False,   # keep if you use a custom collator
    report_to="none",
    seed=42,
)
```

### 5.3 `datasets` — the parts you actually use

```python
from datasets import load_dataset, Dataset, DatasetDict

# From the Hub
ds = load_dataset("tatsu-lab/alpaca", split="train")

# From local JSONL / CSV / Parquet
ds = load_dataset("json", data_files="data/sample_sft.jsonl", split="train")
ds = load_dataset("csv",  data_files=["a.csv", "b.csv"], split="train")

# From memory
ds = Dataset.from_list([{"text": "hi"}, {"text": "there"}])

# Inspect BEFORE you transform
print(ds)                 # features + row count + arrow types
print(ds[0])              # one row — do this every time
print(ds.features)        # the schema your map must respect

# map: scalar vs batched
def scalar_fn(ex):
    ex["n_chars"] = len(ex["text"])
    return ex

def batched_fn(batch):                      # batch["text"] is a LIST
    return {"n_chars": [len(t) for t in batch["text"]]}

ds = ds.map(batched_fn, batched=True, batch_size=1000,
            remove_columns=[c for c in ds.column_names if c != "n_chars"],
            num_proc=4,                     # needs a PICKLABLE (module-level) fn
            desc="counting chars")

# Split, filter, select
splits = ds.train_test_split(test_size=0.05, seed=42)
ds = ds.filter(lambda ex: len(ex["text"]) > 50)
small = ds.select(range(100));  small = ds.shuffle(seed=42).select(range(100))

# Streaming — for datasets too big for disk
stream = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
for ex in stream.take(5):
    print(ex["text"][:80])               # NOTE: no len(), no indexing, no shuffling

ds.push_to_hub("my-user/my-dataset", private=True)
```

> **`map` rewrites the whole dataset to disk.** HuggingFace caches by a hash of the function
> source plus the input fingerprint. Editing the function body invalidates the cache
> correctly, but editing a *global it closes over* does not — you will silently reuse stale
> results. Pass `load_from_cache_file=False` when in doubt.

### 5.4 Tokenization patterns, and the one that is wrong

```python
# ❌ WRONG for chat data: tokenizer adds BOS on top of the template's BOS
text = tok.apply_chat_template(msgs, tokenize=False)
ids = tok(text)["input_ids"]                       # double special tokens

# ✅ RIGHT
ids = tok(text, add_special_tokens=False)["input_ids"]

# ✅ Also right — let the template do the tokenizing
ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)

# Batch + padding (training shape)
enc = tok(texts, padding=True, truncation=True, max_length=2048,
          return_tensors="pt", add_special_tokens=False)
# enc: input_ids, attention_mask  — build labels from input_ids and mask the prompt

# Round-trip check — do this once, it catches most template bugs
decoded = tok.decode(ids, skip_special_tokens=False)
assert decoded == text, "tokenizer is not round-tripping the template"
```

### 5.5 PEFT — attach, train, merge

```python
from peft import LoraConfig, get_peft_model, PeftModel

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                 task_type="CAUSAL_LM", target_modules="all-linear")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()      # ~0.5-2% — check this, not the loss

# ... train ...

model.save_pretrained("out/adapter")     # small: adapter only
tok.save_pretrained("out/adapter")

# Merge into a standalone model (needs the BASE model again)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)  # CPU is fine
merged = PeftModel.from_pretrained(base, "out/adapter").merge_and_unload()
merged.save_pretrained("out/merged", safe_serialization=True)
```

> **Merge on CPU in bf16/fp32, not on a GPU in fp16.** Merging in fp16 loses precision in the
> `W + BA` sum, and the merged model can be measurably worse than the adapter it came from.
> CPU + fp32 is slow and correct.

### 5.6 The Hub

```python
from huggingface_hub import login, HfApi, create_repo

login()                                     # or: huggingface-cli login
model.push_to_hub("my-user/my-model", private=True)
tok.push_to_hub("my-user/my-model", private=True)   # DON'T forget the tokenizer

api = HfApi()
api.upload_folder(folder_path="out/merged", repo_id="my-user/my-model",
                  repo_type="model")
```

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir ./models/llama-3.2-1b --local-dir-use-symlinks False
huggingface-cli upload my-user/my-model out/merged .
export HF_HOME=/big-disk/hf        # move the cache off your boot drive
export HF_HUB_ENABLE_HF_TRANSFER=1 # much faster downloads (pip install hf_transfer)
```

---

## 6. CLI Commands

```bash
# ── Hub auth & transfer ─────────────────────────────────────────────────────
pip install -U huggingface_hub
huggingface-cli login                       # paste a token with WRITE scope
huggingface-cli whoami
huggingface-cli logout

# ── Download / upload ───────────────────────────────────────────────────────
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir ./models/llama-3.2-1b
huggingface-cli download tatsu-lab/alpaca --repo-type dataset --local-dir ./data/alpaca
huggingface-cli upload my-user/my-model out/merged .
huggingface-cli upload my-user/my-dataset data/ --repo-type dataset

# ── Repo management ─────────────────────────────────────────────────────────
huggingface-cli repo create my-model --private
huggingface-cli repo create my-dataset --repo-type dataset --private
huggingface-cli delete-cache                # reclaim disk from old downloads
huggingface-cli scan-cache                  # see what is actually cached

# ── Env vars worth setting ONCE ─────────────────────────────────────────────
export HF_HOME=/big-disk/hf                 # cache location (models are 100s of GB)
export HF_HUB_ENABLE_HF_TRANSFER=1          # faster downloads (pip install hf_transfer)
export HF_HUB_OFFLINE=1                     # fully offline / air-gapped runs
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false         # silences the fork warning; avoids deadlocks

# ── Environment diagnostic — paste this into any issue ──────────────────────
python -c "
import transformers, datasets, torch, peft, trl, accelerate, sys
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__, 'cuda', torch.version.cuda,
      'avail', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('datasets    ', datasets.__version__)
print('accelerate  ', accelerate.__version__)
print('peft        ', peft.__version__)
print('trl         ', trl.__version__)
if torch.cuda.is_available():
    print('gpu         ', torch.cuda.get_device_name(0))
    print('vram GB     ', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
"

# ── Inspect a tokenizer without loading a model ─────────────────────────────
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
print('pad  ', repr(t.pad_token), t.pad_token_id)
print('eos  ', repr(t.eos_token), t.eos_token_id)
print('bos  ', repr(t.bos_token), t.bos_token_id)
print('side ', t.padding_side, '| fast:', t.is_fast, '| vocab:', t.vocab_size)
"
```

---

## 7. VRAM / Cost Calculator

### 7.1 What each object costs

| Object | Memory | Note |
|---|---|---|
| Model weights (bf16) | `2 × params` | 7B → 14 GB |
| Model weights (fp32) | `4 × params` | 7B → 28 GB |
| Gradients (full FT, bf16) | `2 × params` | |
| AdamW optimiser state | `8 × params` | 2 × fp32 moments — **the dominant term** |
| LoRA adapter | `~0.5–2% × params × 2` | 7B r16 → ~40 MB |
| Activations | `batch × seq × layers × hidden × bytes` | The term gradient checkpointing attacks |
| KV cache (inference) | `2 × layers × heads × head_dim × seq × batch × bytes` | Grows linearly with context |
| Tokenizer | ~10–100 MB | Negligible |

### 7.2 Full FT vs LoRA vs QLoRA, by model

From `code/common/memory.py --table`, gradient checkpointing on, single GPU:

| Model | Full FT | LoRA r16 | QLoRA r16 | Infer bf16 | Infer 4-bit |
|---|---|---|---|---|---|
| 1B | 14.5 GB | 2.4 GB | 0.9 GB | 3.1 GB | 1.5 GB |
| 3B | 39.4 GB | 6.4 GB | 2.2 GB | 7.0 GB | 2.5 GB |
| 7B | 91.6 GB | 14.7 GB | 4.9 GB | 16.0 GB | 4.8 GB |
| 13B | 170.0 GB | 27.1 GB | 9.0 GB | 28.3 GB | 7.8 GB |
| 70B | 913.8 GB | 144.5 GB | 46.8 GB | 132.6 GB | 33.9 GB |

**Add 10–20% for allocator and framework overhead.**

### 7.3 Hub storage and transfer

| Item | Typical size | Note |
|---|---|---|
| 7B bf16 | ~14 GB | 3 shards at the 5 GB default |
| 7B 4-bit | ~4 GB | one shard |
| LoRA adapter r16 | ~40–160 MB | trivially small — this is the point of adapters |
| Tokenizer | ~2–11 MB | **Always upload it with the model** |
| 1M-row text dataset | ~1–5 GB | Arrow on disk |

> **Shard size** is controlled by `max_shard_size` (default `"5GB"`). Bigger shards load
> slightly faster; smaller shards are friendlier to slow connections and to partial re-use.

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| **Can't generate — `model.generate` missing** | Loaded with `AutoModel` (no head) | `AutoModelForCausalLM` |
| **Batched outputs are garbage / repeat the prompt** | `padding_side="right"` | `tok.padding_side = "left"` for generation |
| Output starts with a stray token or double BOS | Special tokens added twice | `add_special_tokens=False` on template output |
| `ValueError: Asking to pad but the tokenizer does not have a padding token` | Llama et al. have no pad token | `tok.pad_token = tok.eos_token` |
| Loss is ~0 or ~`ln(vocab)` and never moves | Everything masked, or labels not built | Count non-`-100` labels |
| `TypeError: object of type 'int' has no len()` in a map | Scalar code inside `batched=True` | Handle lists in batched maps |
| `map` works with `num_proc=1` but fails with `num_proc=4` | The function isn't picklable | Move it to module level |
| Stale results after editing the map function | Cache keyed on source + input only | `load_from_cache_file=False` |
| Trainer crashes on an unexpected column | Unused columns survived `map` | `remove_columns=[...]` |
| `ImportError: Using device_map requires Accelerate` | Missing package | `pip install accelerate` |
| `Expected all tensors to be on the same device` | Manual `.to()` fought `device_map` | Let `device_map` place; move inputs to `model.device` |
| `KeyError: 'labels'` at the Trainer | Collator didn't emit labels | Return `input_ids`/`attention_mask`/`labels` |
| `RuntimeError: The size of tensor a (2048) must match ...` | Ragged batch, no padding | Collator with `padding=True, return_tensors="pt"` |
| Model loads but output is nonsense | Merged in fp16, or adapter/base mismatch | Merge in bf16/fp32 on CPU; verify the base revision |
| `FutureWarning: torch_dtype is deprecated` | Renamed in newer transformers | Use `dtype=` (check your version) |
| Gated repo error on download | Licence not accepted | Accept it on the model page; the token must be logged in |
| `RepositoryNotFoundError` | Wrong id, or it's private and you're logged out | Check the id; `huggingface-cli whoami` |
| Push fails with 403 | Token lacks WRITE scope | Re-issue the token |
| Push succeeds but the model card is empty | No README.md | Write one; see CS-12 §16.7 for a template |
| Streaming dataset: `TypeError: 'IterableDataset' object is not subscriptable` | Streaming has no random access | Use `.take()` / non-streaming for indexing |
| Download fills the boot drive | `HF_HOME` unset | `export HF_HOME=/big-disk/hf` |

---

## 9. Comparison Matrix

### 9.1 The ecosystem map — what each library is for

| Library | Role | You touch it when |
|---|---|---|
| `transformers` | Models, tokenizers, `Trainer` | Always |
| `tokenizers` | Rust BPE/WordPiece backend | Rarely directly; it powers `use_fast=True` |
| `datasets` | Arrow-backed datasets, streaming, Hub | Always |
| `accelerate` | Device placement, mixed precision, distributed | Any multi-GPU or `device_map` run |
| `peft` | LoRA, QLoRA, DoRA, adapters | Every PEFT run |
| `trl` | `SFTTrainer`, `DPOTrainer`, `ORPOTrainer`, `GRPOTrainer` | Every post-training run |
| `bitsandbytes` | 8-bit and 4-bit quantisation | QLoRA, `load_in_4bit` |
| `evaluate` | Metrics (with Hub-hosted metric scripts) | Evaluation |
| `safetensors` | Safe weight serialisation | Implicitly, always |
| `huggingface_hub` | The Hub API | Push/pull, `login()` |
| `optimum` | Hardware backends (ONNX, OpenVINO, IPEX) | Deploying to non-NVIDIA hardware |
| `sentence-transformers` | Embedding models + contrastive losses | Embedding fine-tuning (CH/CS-22) |
| `timm` | Vision backbones | VLM work |
| `diffusers` | Image/video diffusion | Not for LLM fine-tuning |

### 9.2 High-level API vs lower-level: when to drop down

| Need | Stay high-level | Drop to PyTorch |
|---|---|---|
| Standard SFT | ✅ `Trainer` / `SFTTrainer` | |
| Custom loss | | ✅ your own `compute_loss` |
| Custom masking | | ✅ your own collator |
| Unusual metric | | ✅ `compute_metrics` |
| Research on the backward pass | | ✅ a hand-written loop |
| Multi-GPU full FT | ✅ `accelerate` / DeepSpeed | |
| Fused/custom kernels | | ✅ (see Unsloth, CS-16) |

### 9.3 Format trade-offs

| Format | Safe? | Size | Load speed | Use |
|---|---|---|---|---|
| `safetensors` | ✅ no code execution | same | fast | **Default. Keep it.** |
| `.bin` (pickle) | ❌ executes code on load | same | fast | Legacy; avoid |
| GGUF | ✅ | quantised | mmap, very fast | llama.cpp / CPU |
| ONNX | ✅ | varies | fast | Non-NVIDIA deployment |
| 4-bit bitsandbytes | ✅ | ~¼ | fast | Training-time quantisation |

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| Bytes/param, bf16 | **2** | weights |
| Bytes/param, fp32 | **4** | weights |
| Bytes/param, AdamW states | **8** | 2 × fp32 moments |
| Full FT total | **~12 bytes/param** + activations | Why full FT ≈ 20× QLoRA |
| Bytes/param, NF4 | **~0.5** | 4-bit + double quant |
| 7B bf16 weights | **~14 GB** | Inference floor |
| 7B 4-bit weights | **~4 GB** | |
| Default `max_shard_size` | **5 GB** | |
| Tokens per word | **≈1.33** | English |
| Chars per token | **≈4** | English |
| Fast tokenizer speedup | **10–100×** | Rust vs Python |
| Typical tokenizer size | **2–11 MB** | |
| LoRA adapter r16 (7B) | **~40–160 MB** | |
| `padding_side` for generation | **left** | |
| `padding_side` for training | **right** | |
| Default `learning_rate` | **5e-5** | Usually wrong — override it |
| Safe serialisation default | **True** | |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as pad_token (tokenizer.pad_token = tokenizer.eos_token e.g.) or add a new pad token via tokenizer.add_special_tokens({'pad_token': '[PAD]'})` | No pad token | `tok.pad_token = tok.eos_token` |
| `ImportError: Using device_map requires Accelerate: pip install accelerate` | Missing package | `pip install accelerate` |
| `ImportError: Using low_cpu_mem_usage=True requires Accelerate` | Same | Same |
| `OSError: Can't load tokenizer for '...'` | Not a tokenizer repo, or wrong id | Check the id; some repos are model-only |
| `OSError: ... does not appear to have a file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt or flax_model.msgpack` | No weights at that revision | Check `revision`; the repo may be gated |
| `ValueError: Unrecognized model in .../ . Should have a model_type key in its config.json` | Corrupt/absent config | Re-download; check the cache |
| `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` | `device_map` + manual `.to()` | Move inputs to `model.device`; don't `.to()` the model |
| `RuntimeError: The size of tensor a (N) must match the size of tensor b (M) at non-singleton dimension 1` | Ragged batch | Pad via the collator |
| `ValueError: The model did not return a loss from the inputs, only the following keys: ...` | No `labels` in the batch | Emit `labels` from the collator |
| `TypeError: 'int' object is not iterable` in a map | Scalar code in `batched=True` | Operate on lists |
| `TypeError: cannot pickle '_thread.lock' object` with `num_proc` | Function isn't picklable | Module-level function, no closures |
| `huggingface_hub.utils._errors.GatedRepoError: 401 Client Error ...` | Licence not accepted | Accept on the model page, then `login()` |
| `huggingface_hub.utils._errors.RepositoryNotFoundError` | Wrong id / private / logged out | Verify the id and auth |
| `huggingface_hub.utils._errors.HfHubHTTPError: 403 ...` | Token lacks WRITE | New token with write scope |
| `FutureWarning: \`torch_dtype\` is deprecated and will be removed. Use \`dtype\` instead.` | API rename | `dtype=` on newer transformers |
| `We couldn't connect to 'https://huggingface.co' to load this file` | Offline / network | Set `HF_HUB_OFFLINE=1` if you truly are offline, and pre-download |
| `Tokenizers will be parallelized... set TOKENIZERS_PARALLELISM` | Fork warning | `export TOKENIZERS_PARALLELISM=false` |

---

## 12. Copy-Paste Starter Config

### 12.1 A complete, correct end-to-end script

```python
"""minimal_hf_pipeline.py — load, tokenize, mask, train, merge, push. All correct."""
import sys, torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
IGNORE_INDEX = -100

# ── 1. Tokenizer ────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"          # RIGHT for training

# ── 2. Data ─────────────────────────────────────────────────────────────────
raw = load_dataset("json", data_files="data/sample_sft.jsonl", split="train")
print(raw)                          # inspect before transforming
print(raw[0])                       # and look at one real row

def to_text(ex):
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

ds = raw.map(to_text, remove_columns=raw.column_names)

def encode(batch):
    out = tok(batch["text"], truncation=True, max_length=2048,
              add_special_tokens=False)          # template already added them
    # Mask the prompt: find the LAST assistant header and supervise everything after.
    # Simple version here; code/common/data_utils.build_masked_example does it exactly.
    labels = [list(ids) for ids in out["input_ids"]]
    marker = tok("assistant", add_special_tokens=False)["input_ids"]
    for i, ids in enumerate(out["input_ids"]):
        cut = 0
        for j in range(len(ids) - len(marker) + 1):
            if ids[j:j + len(marker)] == marker:
                cut = j + len(marker)
        labels[i] = [IGNORE_INDEX] * cut + ids[cut:]
    out["labels"] = labels
    return out

ds = ds.map(encode, batched=True, remove_columns=["text"])

# Sanity: supervised fraction should be 10-40%. If it is 0 or 100, STOP.
sup = sum(1 for row in ds.select(range(min(50, len(ds)))) for l in row["labels"] if l != IGNORE_INDEX)
tot = sum(len(row["labels"]) for row in ds.select(range(min(50, len(ds)))))
print(f"supervised fraction: {sup/max(tot,1):.1%}")
assert 0.05 < sup / max(tot, 1) < 0.95, "masking is wrong — fix before training"

# ── 3. Model ────────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model = __import__("peft").get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM", target_modules="all-linear"))
model.print_trainable_parameters()

# ── 4. Train ────────────────────────────────────────────────────────────────
Trainer(model=model, train_dataset=ds, args=TrainingArguments(
    output_dir="out/run", per_device_train_batch_size=2,
    gradient_accumulation_steps=8, num_train_epochs=2, learning_rate=2e-4,
    lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
    gradient_checkpointing=True, save_strategy="epoch", logging_steps=5,
    report_to="none", seed=42,
)).train()

# ── 5. Save, merge, push ────────────────────────────────────────────────────
model.save_pretrained("out/adapter"); tok.save_pretrained("out/adapter")

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)  # CPU, fp32
PeftModel.from_pretrained(base, "out/adapter").merge_and_unload() \
    .save_pretrained("out/merged", safe_serialization=True)
tok.save_pretrained("out/merged")

# model.push_to_hub("my-user/my-model", private=True)
# tok.push_to_hub("my-user/my-model", private=True)   # don't forget this one
print("done")
```

### 12.2 The six checks before you trust a run

| # | Check | How | Pass |
|---|---|---|---|
| 1 | Tokenizer round-trips the template | `tok.decode(ids) == text` | Identical |
| 2 | `pad_token`, `eos_token`, `padding_side` | §6 diagnostic | pad set; side correct for the task |
| 3 | One real row, printed | `print(ds[0])` | Looks like what you expect |
| 4 | Supervised fraction | Script above | 5–40% |
| 5 | Trainable params | `model.print_trainable_parameters()` | 0.1–2% for LoRA |
| 6 | Version pins | §6 diagnostic | Recorded in the run's notes |

> Check 1 catches template bugs, check 4 catches masking bugs, and check 6 catches
> everything you will otherwise be unable to explain in three months.

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| The full masterclass treatment | **CS-06 — The Hugging Face Masterclass** |
| To actually run SFT | **CH-13 / CS-13 — Instruction Fine-Tuning** |
| To understand the Trainer alternatives | **CH-03 / CS-03 — Framework Landscape** |
| The PEFT deep dive | **CS-23 — LoRA & QLoRA** |
| Quantisation flags in `from_pretrained` | **CH-10 / CH-11 — Quantization** |
| To serve what you saved | `code/15_serve_vllm.py` |
| The repo's own correct masking implementation | `code/common/data_utils.py` → `build_masked_example` |
| Practice being interviewed on this | **IQ-06 — Interview Questions: The Hugging Face Masterclass** |

