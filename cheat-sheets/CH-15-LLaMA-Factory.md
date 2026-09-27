# CH-15 — LLaMA-Factory Cheat Sheet

**One-line purpose:** fine-tune 100+ model families (full FT, LoRA, QLoRA, and every
preference method) from a YAML file or a browser UI, with no Python written.
**Use when:** you want a reproducible, version-controllable training config; you need to
support many models behind one interface; or you are not a Python engineer.
**Do NOT use when:** you need a custom loss, a custom data collator, or an unusual training
loop. LLaMA-Factory is opinionated; fighting it costs more than writing 50 lines of TRL.

> **The one sentence that matters.** LLaMA-Factory's entire contract is the `template:`
> field. If it does not match your model's real chat template, everything runs, the loss
> falls, and the model is worse — silently.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **`template:` must match the model.** | The #1 silent failure. `llama3` ≠ `llama2` ≠ `chatml`. |
| 2 | **Datasets must be registered in `data/dataset_info.json`.** | The YAML refers to a *name*, not a path. |
| 3 | **`stage:` picks the whole recipe.** | `sft`, `dpo`, `orpo`, `ppo`, `rm`, `kto`, and the pretraining stage. Verify the pretraining spelling (`pt` vs `pretrain`) against your pinned version — see §3 note. |
| 4 | **`finetuning_type: lora` + `quantization_bit: 4` = QLoRA.** | Two lines, and the VRAM halves. |
| 5 | **`adapter_name_or_path` chains stages.** | DPO starts from your SFT adapter, not the base model. |
| 6 | **`llamafactory-cli export` merges the adapter.** | Until you export, you have a two-piece artifact. |
| 7 | **`mask_history: true` for multi-turn SFT.** | Without it you train on the user's own turns. |
| 8 | **`cutoff_len` silently truncates.** | Check p95 length; truncation often cuts the answer. |
| 9 | **`report_to: none`** if you have no tracker. | Otherwise it tries W&B and can hang on auth. |
| 10 | **Everything is a YAML file — commit it.** | This is the actual reason to use LLaMA-Factory. |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Effective batch** | `per_device × grad_accum × n_gpu` | — | 2 × 8 × 4 = 64 |
| **Steps per epoch** | `ceil(N / eff_batch)` | N examples | 10,000 / 64 = 157 |
| **Warmup steps** | `total_steps × warmup_ratio` | — | 157 × 0.03 ≈ 5 |
| **LoRA targets** | `lora_target: all` | — | All linear layers; safer than a list |
| **Quantised VRAM** | roughly `¼` of bf16 weights | — | `quantization_bit: 4` |
| **DeepSpeed ZeRO-3 shards** | `÷ n_gpu` for weights+optimiser | — | 4 GPUs → ~¼ per GPU |
| **`cutoff_len` in tokens** | `words × 1.33` | — | 1,500 words → 2,000 tokens |

---

## 3. Decision Tree

```
Which stage do you need?
├─ Install knowledge/domain text (no labels)  → stage: pretrain
├─ Teach (instruction, response)               → stage: sft        (CH-13)
├─ Train a reward model on comparisons         → stage: rm         (then stage: ppo)
├─ Optimise against preferences                → stage: dpo / orpo / kto (CH-14)
│    └─ and the LOSS is a separate knob: pref_loss: sigmoid | orpo | simpo | ipo | hinge
│       SimPO and IPO are NOT stages — pick stage: dpo and set pref_loss.
└─ Optimise against a verifiable reward        → stage: ppo (or use TRL GRPO)

Which finetuning_type?
├─ Big GPU + big dataset  → full
├─ Default                → lora
└─ Small GPU              → lora + quantization_bit: 4     (= QLoRA)

Which dataset FORMAT does your file use?
├─ {"instruction","input","output"}     → formatting: alpaca
├─ {"conversations":[{from,value},...]} → formatting: sharegpt
└─ neither → convert first. Do not guess; look at one line of the file.

Do you have chosen/rejected pairs?   → add "ranking": true to the registry entry
                                       (NOT a formatting value — see §5.1)

Is this a MULTI-TURN dataset?
├─ Yes → mask_history: true, and verify the masked fraction
└─ No  → mask_history: false

Are you continuing from a previous stage?
├─ Yes → adapter_name_or_path: out/previous-stage   (NOT model_name_or_path)
└─ No  → model_name_or_path: <base or instruct model>

Do you have >1 GPU?
├─ Yes → deepspeed: examples/deepspeed/ds_z3_config.json
│        (z3 for full FT; z2 is often enough and faster for LoRA)
└─ No  → omit the deepspeed key entirely
```

> **`stage` and `pref_loss` are two different axes, and conflating them is a documented
> source of confusion across this handbook's own files.** `stage` selects the *recipe* —
> what data is loaded, what the model head is, what the training loop optimises. `pref_loss`
> selects the *loss function* inside a preference stage. So ORPO and KTO are stages
> (`stage: orpo`, `stage: kto`), while **SimPO and IPO are not** — they are
> `stage: dpo` with `pref_loss: simpo` / `pref_loss: ipo`. Writing `stage: simpo` fails.
>
> The exact accepted spelling of the pretraining stage (`pt` in recent versions) and the full
> `pref_loss` list are **version-sensitive**. Do not trust a list from a blog post or from
> this card: run `llamafactory-cli train --help` against your pinned install, or read
> `Stage` / `pref_loss` in `llamafactory/extras/enums.py` in your checkout. The same caution
> applies to every error string in §11 — they are greppable in the source, which is the point.

---

## 4. Hyperparameter Quick Reference

### 4.1 The YAML keys that matter

| Key | Default | Typical | Effect of getting it wrong |
|---|---|---|---|
| `model_name_or_path` | — | a HF id or local path | Wrong id → download failure or a surprise base model |
| `template` | — | **must match the model family** | **The silent killer.** Wrong template ≈ a broken model. |
| `stage` | — | `sft` | Picks the recipe; wrong stage = wrong loss |
| `finetuning_type` | `lora` | `lora` / `full` / `freeze` | `full` on a small GPU OOMs |
| `dataset` | — | a key in `dataset_info.json` | Not a path. A wrong name fails loudly; a *right* name with a wrong format fails silently. |
| `cutoff_len` | 2048 | 1024–8192 | Truncation drops the answer |
| `output_dir` | — | `out/run-name` | — |
| `per_device_train_batch_size` | 8 | 1–8 | The main VRAM knob |
| `gradient_accumulation_steps` | 8 | 4–32 | Effective batch |
| `learning_rate` | 5e-5 | 2e-4 (LoRA) / 2e-5 (full) | LoRA-FT LR mix-up is common |
| `num_train_epochs` | 3.0 | 1–3 | Overfits past 3 |
| `lr_scheduler_type` | `cosine` | `cosine` | — |
| `warmup_ratio` | 0 | 0.03–0.1 | 0 is LLaMA-Factory's default and is worth overriding |
| `bf16` | *not fixed* | `true` | **Set it explicitly.** HF's `TrainingArguments.bf16` defaults to `False`; LLaMA-Factory derives precision from the device rather than promising a default. Prefer bf16 over fp16 (fp16 needs loss scaling and can silently NaN). |
| `gradient_checkpointing` | `true` | `true` | Required for most single-GPU runs |
| `mask_history` | `false` | `true` for multi-turn | Trains on the user's turns if left off |
| `packing` | `false` | `true` for pretrain | Cross-contamination risk on SFT — see §8 |
| `neat_packing` | `false` | `true` | Safer packing (resets attention per example) |
| `train_on_prompt` | `false` | `false` | `true` teaches the model to write prompts |
| `report_to` | `none` | `none` | Avoids W&B auth hangs |
| `val_size` | 0 | 0.05 | Set it — otherwise you cannot detect overfitting |
| `eval_strategy` | `no` | `epoch` | Must be set for `val_size` to do anything |
| `load_best_model_at_end` | `false` | `true` | Requires `eval_strategy` + `save_strategy` to match |

### 4.2 LoRA keys

| Key | Default | Typical | Notes |
|---|---|---|---|
| `lora_rank` | 8 | **16** | 8–32 |
| `lora_alpha` | 16 | **32** | `= 2 × rank` |
| `lora_dropout` | 0 | 0.05 | 0 is faster |
| `lora_target` | `all` | `all` | All linear layers |
| `use_rslora` | `false` | `false` | Rank-stabilised; helps at high rank |
| `use_dora` | `false` | `false` | Better at very low rank, slower |
| `loraplus_lr_ratio` | — | — | LoRA+ : higher LR on the B matrix |
| `create_new_adapter` | `false` | `false` | `true` adds a *second* adapter rather than reusing |

### 4.3 Quantisation keys

| Key | Value | Meaning |
|---|---|---|
| `quantization_bit` | `4` / `8` | bitsandbytes quantisation during **training** |
| `quantization_method` | `bnb` / `hqq` / `eetq` / `gptq` / `awq` | Backend |
| `quantization_type` | `nf4` / `fp4` | NF4 is standard for QLoRA |
| `double_quantization` | `true` | Quantise the quantisation constants — saves ~0.4 bits/param |

> `quantization_bit` is a **training-time** setting. For serving, export and quantise
> separately (CH-10/CH-11) — the two are not the same artifact.

---

## 5. Copy-Paste Code Snippets

### 5.1 Registering a dataset (`data/dataset_info.json`)

```json
{
  "my_sft": {
    "file_name": "my_sft.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations" },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "my_alpaca": {
    "file_name": "my_alpaca.jsonl",
    "formatting": "alpaca"
  },
  "my_pairs": {
    "file_name": "my_pairs.jsonl",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

| Key | Expected row shape |
|---|---|
| `"formatting": "alpaca"` | `{"instruction": "...", "input": "...", "output": "..."}` |
| `"formatting": "sharegpt"` | `{"conversations": [{"role":"user","content":"..."}, ...]}` |
| `"ranking": true` | Any of the above **plus** `chosen` / `rejected` |

> **`"ranking": true` is what makes it a preference dataset — not the columns.** Without it,
> a `stage: dpo` run parses your file as ordinary SFT data: it finds the `conversations`
> field, builds normal instruction examples, trains happily, and reports a falling loss. The
> `chosen` and `rejected` fields are simply ignored. Nothing errors. You get a model that was
> SFT'd a second time and is not preference-tuned at all — which is why "DPO didn't change
> anything" so often turns out to be a missing boolean in a JSON file.
>
> A useful sanity check: after registering, run `stage: dpo` for a few steps and confirm the
> logged loss is a *preference* loss (it starts near `ln 2 ≈ 0.693` for a sigmoid DPO at
> `beta=0.1`, and the reference-model term appears in the log). A plain cross-entropy curve
> starting high and falling smoothly on a `ranking`-less dataset is the tell.

> `columns` and `tags` exist so you can point at *your* key names without rewriting the file.
> Almost every "dataset registered but parses to zero rows" bug is a `tags` mismatch — and
> almost every "preference training did nothing" bug is a missing `"ranking": true`.

### 5.2 SFT with LoRA — `train.yaml`

```yaml
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
template: llama3
stage: sft
do_train: true

finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

dataset: my_sft
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 4

output_dir: out/lf-sft
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
bf16: true
gradient_checkpointing: true
logging_steps: 5
save_strategy: epoch
plot_loss: true
report_to: none

val_size: 0.05
per_device_eval_batch_size: 2
eval_strategy: epoch
load_best_model_at_end: true
```

### 5.3 QLoRA — the two-line change

```yaml
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
double_quantization: true
```

### 5.4 DPO starting from the SFT adapter

```yaml
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
adapter_name_or_path: out/lf-sft        # <-- the SFT run, NOT the base
template: llama3
stage: dpo
do_train: true
finetuning_type: lora
pref_beta: 0.1
pref_loss: sigmoid
dataset: my_pairs
cutoff_len: 1024
output_dir: out/lf-dpo
learning_rate: 5.0e-6                   # LOW
num_train_epochs: 1.0
bf16: true
gradient_checkpointing: true
report_to: none
```

### 5.5 Multi-GPU

```yaml
deepspeed: examples/deepspeed/ds_z3_config.json   # z3 for full FT
# deepspeed: examples/deepspeed/ds_z2_config.json # z2 is often enough for LoRA
ddp_timeout: 180000000

per_device_train_batch_size: 4
gradient_accumulation_steps: 4
```

```bash
# Multi-node / multi-GPU launcher
FORCE_TORCHRUN=1 llamafactory-cli train train.yaml
# or explicitly:
torchrun --nproc_per_node 4 $(python -c "import llamafactory; print(llamafactory.__path__[0])")/launcher.py train.yaml
```

### 5.6 Merging and exporting

```bash
# Merge the LoRA adapter into the base weights and write a standalone model
llamafactory-cli export export.yaml
```

```yaml
# export.yaml
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
adapter_name_or_path: out/lf-sft
template: llama3
finetuning_type: lora
export_dir: out/merged
export_size: 4                 # shard size in GB
export_device: cpu             # or 'auto' — cpu needs no GPU and is safest
export_legacy_format: false
# export_hub_model_id: my-user/my-model   # push straight to the Hub
```

> **If you trained with `quantization_bit: 4`, export dequantises.** The merged artifact is
> bf16/fp16. Quantise it again afterwards if you need a 4-bit *serving* artifact — that is a
> different step with a different tool (CH-10/CH-11), and skipping it is why some people's
> exported models are unexpectedly large.

### 5.7 Serving the result

```bash
# Interactive chat against the adapter (no merge needed)
llamafactory-cli chat inference.yaml

# OpenAI-compatible API server
API_PORT=8000 llamafactory-cli api inference.yaml
```

```yaml
# inference.yaml
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
adapter_name_or_path: out/lf-sft
template: llama3
infer_backend: vllm        # or 'huggingface'
vllm_enforce_eager: false
max_new_tokens: 512
top_p: 0.7
temperature: 0.95
repetition_penalty: 1.0
```

---

## 6. CLI Commands

```bash
# ── Install ─────────────────────────────────────────────────────────────────
pip install llamafactory                 # or: git clone && pip install -e .[torch,metrics]
pip install llamafactory[vllm,deepspeed,metrics]   # extras for serving / multi-GPU

# ── The five subcommands you will actually use ──────────────────────────────
llamafactory-cli train  train.yaml         # run any stage from a YAML
llamafactory-cli chat   inference.yaml     # interactive REPL against the adapter
llamafactory-cli api    inference.yaml     # OpenAI-compatible HTTP server
llamafactory-cli export export.yaml        # merge adapter → standalone model
llamafactory-cli webui                     # the no-code browser UI

# ── Environment / diagnostics ───────────────────────────────────────────────
llamafactory-cli env                       # versions of everything — paste this into issues
llamafactory-cli version

# ── Multi-GPU ───────────────────────────────────────────────────────────────
FORCE_TORCHRUN=1 llamafactory-cli train train.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train train.yaml

# ── Inspect what templates are available (run this FIRST for a new model) ───
python -c "
from llamafactory.extras.constants import TEMPLATES
print(sorted(TEMPLATES))
"

# ── Confirm the dataset actually registered and parsed ──────────────────────
python -c "
import json
d = json.load(open('data/dataset_info.json'))
print('registered:', list(d))
for k, v in d.items():
    print(k, '->', v.get('file_name'), v.get('formatting'))
"

# ── Look at one line of your data BEFORE registering it ─────────────────────
python -c "
import json
print(json.dumps(json.loads(open('data/my_sft.jsonl',encoding='utf-8').readline()),
                 indent=2)[:600])
"
```

> The template-inspection command is the single most valuable one in this file. Run it before
> every new model. `llama3` and `llama2` and `chatml` all exist and all "work" — and only one
> is right for your model.

---

## 7. VRAM / Cost Calculator

### 7.1 What each `finetuning_type` + `quantization_bit` costs

Single GPU, gradient checkpointing on, from `code/common/memory.py`:

| Model | `full` | `lora` | `lora` + `quantization_bit: 4` |
|---|---|---|---|
| 1B | 14.5 GB | 2.4 GB | 0.9 GB |
| 3B | 39.4 GB | 6.4 GB | 2.2 GB |
| 7B | 91.6 GB | 14.7 GB | 4.9 GB |
| 8B | 104.7 GB | 16.7 GB | 5.6 GB |
| 13B | 170.0 GB | 27.1 GB | 9.0 GB |
| 70B | 913.8 GB | 144.5 GB | 46.8 GB |

**Add 10–20% for allocator and framework overhead.**

### 7.2 Multi-GPU: what DeepSpeed buys you

| Config | Shards | Use for | Note |
|---|---|---|---|
| `ds_z2_config.json` | gradients + optimiser | **LoRA on multi-GPU** | Usually enough, and faster |
| `ds_z3_config.json` | + parameters | **Full FT on multi-GPU** | Needed when weights don't fit |
| `ds_z3_offload_config.json` | + offloads to CPU | Fitting a model that really shouldn't fit | Very slow |

> ZeRO-3 is not automatically better. It shards the *weights*, which means every forward pass
> must gather them — extra communication for no benefit when the weights already fit. For LoRA
> runs on 2–4 GPUs, **ZeRO-2 is usually the right answer.**

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| **Model is worse after training, loss looked fine** | **Wrong `template:`** | Render the template and read it. §6. |
| `ValueError: Undefined template: ...` | Name not in the registry | §6 diagnostic command lists valid names |
| `ValueError: Dataset ... not found` | Not registered in `dataset_info.json`, or a name typo | The YAML uses the *key*, not the filename |
| Registered but "0 samples" / no rows parsed | `formatting` or `tags` mismatch | Print one line of the file; align `tags` to your keys |
| Loss trains but the model ignores instructions | `mask_history: false` on multi-turn data, or wrong template | Set `mask_history: true`; verify the masked fraction |
| OOM at batch 1 | `cutoff_len` too large | Halve it; add `quantization_bit: 4`; confirm `gradient_checkpointing: true` |
| OOM only when `stage: full` | Full FT is ~20× LoRA | Use `finetuning_type: lora` |
| Training hangs at startup | `report_to` pointing at W&B with no auth | `report_to: none` |
| `deepspeed` import/config error on 1 GPU | `deepspeed:` key present with no multi-GPU setup | Remove the key for single-GPU runs |
| Adapter trains, `chat` shows no change | `adapter_name_or_path` missing from `inference.yaml`, or a different `template` | Both files must agree on model + template |
| Exported model is unexpectedly huge | You expected a 4-bit artifact | Export dequantises. Quantise again as a separate step (CH-10/CH-11). |
| DPO run degrades the model | Learning rate at the SFT value | DPO LR ≈ 5e-6, not 2e-4 (CH-14) |
| Resumed run restarts from scratch | `resume_from_checkpoint` not set | Point it at a checkpoint dir, or use `overwrite_output_dir: true` deliberately |
| Loss is exactly 0.0 | Empty dataset / everything masked | Check the row count actually loaded |
| Multi-GPU run slower than single | ZeRO-3 sharding for a LoRA run | Switch to `ds_z2_config.json` |
| `max_samples` quietly truncated your data | Default caps the set | Set `max_samples` explicitly to your row count |

> **The "worse after training" row is worth internalising.** It is the failure LLaMA-Factory is
> most often blamed for and it is almost always the `template` field. The reason it is silent:
> a wrong template still produces valid token sequences and a falling loss. The model just
> learned to continue a distribution that does not match how you will prompt it.

---

## 9. Comparison Matrix

| Dimension | **LLaMA-Factory** | Unsloth | Axolotl | TRL | torchtune |
|---|---|---|---|---|---|
| Interface | **YAML + WebUI** | Python | YAML | Python | Python recipes |
| No-code path | ✅ WebUI | ❌ | ❌ | ❌ | ❌ |
| Model coverage | **100+ families** | Patched per family | Many | Any HF | Curated |
| Speed vs baseline | 1× | **2–4× faster** | 1× | 1× | 1× |
| VRAM vs baseline | 1× | **~40–60% less** | 1× | 1× | 1× |
| Stages supported | pretrain, sft, rm, ppo, dpo, orpo, kto, simpo | sft, dpo, orpo, grpo | sft, dpo, orpo, kto | all | sft, dpo |
| Multi-GPU | DeepSpeed / FSDP | Limited | **DeepSpeed / FSDP** | Accelerate / DS | FSDP |
| Custom loss / collator | Hard | Medium | Medium | **Easy** | Hard |
| Reproducibility | **Best** (commit the YAML) | Script | Good (YAML) | Script | Recipe + config |
| Serving integration | vLLM / API server | vLLM | vLLM | — | — |
| Learning curve | **Lowest** | Low | Medium | Medium | High |

**Which to pick:**

| Situation | Pick |
|---|---|
| "I want it working today and I'm not a Python engineer" | **LLaMA-Factory WebUI** |
| "I have one 24 GB GPU and a 7B model" | **Unsloth** |
| "I need a reproducible multi-node full fine-tune" | **Axolotl** or LLaMA-Factory + ZeRO-3 |
| "I need a custom loss or a research modification" | **TRL** |
| "I want to read every line of the training loop" | **torchtune** |

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| `cutoff_len` default | **2048** | Raise for long-form data |
| `lora_rank` | **16** | 8–32 |
| `lora_alpha` | **32** | `= 2 × rank` |
| `lora_dropout` | **0.05** | |
| LR (LoRA) | **2e-4** | |
| LR (full) | **2e-5** | 10× lower |
| LR (DPO) | **5e-6** | ~40× lower still |
| `num_train_epochs` default | **3.0** | Lower to 2 for SFT |
| `warmup_ratio` default | **0** | **Override it** — 0.03 is better |
| `quantization_bit` | **4** | NF4 + double quant = QLoRA |
| VRAM cut from 4-bit | **~3×** | 14.7 → 4.9 GB at 7B LoRA |
| Full FT vs LoRA VRAM | **~6×** | At 7B: 91.6 vs 14.7 GB |
| ZeRO-3 shard factor | **÷ n_gpu** | Weights + optimiser |
| `val_size` default | **0** | Set 0.05 |
| Registered templates | **100+** | Check with the §6 command |
| `export_size` | **4 GB** | Shard size on export |

---

## 11. Common Errors And Their Exact Messages

Two of these are the *loud* failures you actually want — the template and dataset lookups are
the only places the pipeline refuses to guess. Grep the strings in your own checkout
(`grep -rn "does not exist" src/llamafactory/data/`) rather than trusting a card.

| Error message | Meaning | Fix |
|---|---|---|
| `ValueError: Template X does not exist.` | Template name not in `TEMPLATES` | §6 lists valid names; check spelling and family |
| `ValueError: Undefined dataset X in dataset_info.json.` | The YAML `dataset:` is not a registry key | Add the key; the YAML uses the **key**, never the filename or a Hub URL |
| `AssertionError: ... no valid data` | Formatting/tags matched no rows | Print one line; fix `formatting` / `tags` |
| `KeyError: 'conversations'` | `formatting: sharegpt` on an alpaca file | Change `formatting`, or add a `columns` mapping |
| `TypeError: expected string or bytes-like object` | A field is `null` or a nested dict | Clean the data; DPO fields must be plain strings |
| `torch.cuda.OutOfMemoryError: ... Tried to allocate X GiB` | See §8 | Halve `cutoff_len`; QLoRA; check checkpointing |
| `ImportError: Please install deepspeed` | `deepspeed:` key without the package | `pip install deepspeed`, or drop the key |
| `deepspeed` config not found | Relative path resolved from the wrong cwd | Use an absolute path or run from the repo root |
| `RuntimeError: Cannot re-initialize CUDA in forked subprocess` | DataLoader workers + CUDA | `preprocessing_num_workers: 0` or `1` |
| `wandb.errors.UsageError: api_key not configured` | `report_to` left at the W&B default | `report_to: none` |
| `ValueError: max_samples must be positive` | Set to 0 meaning "all" | Omit the key instead |
| `OSError: ... does not appear to have a file named config.json` | Bad `model_name_or_path` | Check the id / local path |
| `AssertionError: adapter_name_or_path ... not found` | Wrong path for the previous stage | Point at the SFT `output_dir` |
| `Token indices sequence length is longer than ...` | A row exceeds `cutoff_len` | Expected warning; check *what* got cut |
| `RuntimeError: NCCL error` / timeout on multi-GPU | Slow init on a large model | `ddp_timeout: 180000000` |

---

## 12. Copy-Paste Starter Config

A complete, correct single-GPU LoRA SFT. Change the four marked lines and nothing else.

```yaml
# ── CHANGE THESE FOUR ───────────────────────────────────────────────────────
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
template: llama3               # MUST match the model family — §8
dataset: my_sft                # a KEY in data/dataset_info.json, not a path
output_dir: out/lf-sft
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
gradient_accumulation_steps: 8
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

val_size: 0.05
per_device_eval_batch_size: 2
eval_strategy: epoch
load_best_model_at_end: true
```

### The pre-flight checklist

| # | Check | Command / how |
|---|---|---|
| 1 | Template name is real | `llamafactory-cli env` + the §6 template listing |
| 2 | Template renders the RIGHT thing | `llamafactory-cli chat` on the **base** model before training |
| 3 | Dataset registered AND parses | §6 registration diagnostic |
| 4 | Row shape matches `formatting` | Print one line of the JSONL |
| 5 | VRAM fits | `code/common/memory.py --table` |
| 6 | p95 length ≤ `cutoff_len` | Compute it; truncation usually cuts the answer |
| 7 | `val_size > 0` | Otherwise overfitting is invisible |

> Check 2 is the one that saves you a wasted run: chat with the **untrained** model first. If
> it already produces sensible, well-formatted answers, your template is right. If it rambles
> or refuses oddly, your template is wrong — and no amount of training will fix it.

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| The full treatment, including the notebook walkthrough | **CS-15 — LLaMA-Factory** |
| To understand what SFT is doing | **CH-13 / CS-13 — Instruction Fine-Tuning** |
| To run preference optimisation from a YAML | **CH-14 / CS-14 — The Alignment Map** |
| A faster single-GPU path | **CS-16 — Unsloth** |
| A YAML-driven multi-GPU path | **CS-17 — Axolotl** |
| To quantise the exported model for serving | **CH-10 / CH-11 — Quantization** |
| To serve it properly | `code/15_serve_vllm.py` |
| The Python equivalent, with the traps checked in code | `code/01_sft_lora.py` |
| Practice being interviewed on this | **IQ-15 — Interview Questions: LLaMA-Factory** |

