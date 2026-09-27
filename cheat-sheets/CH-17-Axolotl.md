# CH-17 — Axolotl Cheat Sheet

**One-line purpose:** define a whole training run — SFT, QLoRA, DPO/ORPO/KTO/GRPO, FSDP
multi-GPU — in one reviewable YAML file and launch it with `axolotl train config.yml`.
**Use when:** you need more than one GPU, an auditable record of how a model was trained,
or ≥10 experiment variations that must be diffable and re-runnable next quarter.
**Do NOT use when:** you have one GPU, one afternoon, and one run. Unsloth (CS-16) or a
30-line TRL script gets you there faster; Axolotl's config schema is a translation layer
between you and the tensors, and at N=1 that layer is pure cost (CS-17 §8.2).

> **The one sentence that matters.** Axolotl's failure profile is the **inverse** of
> LLaMA-Factory's: Axolotl validates the *config* loudly and the *data* quietly, while
> LLaMA-Factory validates the *data* at the one place it looks and the *template* not at
> all. Neither tool's schema can tell you your chat template is wrong — so
> `axolotl preprocess --debug` stays mandatory in both.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **The YAML is the artefact; the checkpoint is derived from it.** | The whole product is a diffable, reviewable, re-runnable record of an experiment (CS-17 §4.1.2). |
| 2 | **Config validation runs *before* the model loads.** `adapter: qlora` without `load_in_4bit: true` is a startup error. | Illegal *combinations* are caught for free — see §4.5. |
| 3 | **Unknown keys are *warned*, not failed, unless `strict: true`.** | `DictDefault` returns `None` for a typo'd key and the default silently applies. This is Axolotl's one quiet config failure. §4.4. |
| 4 | **`sequence_len`, not `max_seq_length`. `num_epochs`, not `num_train_epochs`.** | The `max_*` spellings are TRL/HF names. `sequence_len` is Axolotl's; its default is only **512**. |
| 5 | **The dataset `type:` decides the contract — mask, template, and fields.** | `alpaca` ≠ `chat_template` ≠ `completion`. A wrong `type:` produces a *low, plausible* loss (CS-17 §4.5.4). |
| 6 | **`sample_packing: true` needs a varlen attention backend.** `sdpa` and `eager` do not have one. | With a non-varlen backend, `sample_packing` is where cross-document attention becomes possible (CS-17 §4.6.4, issues #3453/#3608). |
| 7 | **`axolotl preprocess config.yml --debug` before every GPU-minute.** | 30 seconds of printed `input_ids` + labels prevents the most expensive failure class (CS-17 §6.5). |
| 8 | **Packing changes the step count, therefore the LR schedule.** | Same `warmup_steps` on a 3× packed run is a 3× longer warmup *as a fraction of the run* (CS-17 §4.6.6). |
| 9 | **`rl:` selects the preference method; `rl_beta` replaced `dpo_beta`.** | `rl: dpo|orpo|kto|simpo|grpo|gdpo|ebft`. IPO is `dpo` + `dpo_loss_type: [ipo]`, not its own `rl:`. |
| 10 | **`fsdp_version: 2` + `fsdp_config:`.** The bare `fsdp:` list is **rejected**. | FSDP1 is removed; `fsdp_version: 1` is a hard error (CS-17 §5.6). |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Effective batch (seq)** | `micro_batch_size × grad_accum × num_gpus` | — | 2 × 8 × 4 = 64 |
| **Effective batch (tokens)** | `eff_batch_seq × sequence_len` | — | 16 × 2048 = 32,768 |
| **Steps per epoch** | `rows_surviving ÷ (micro_bs × grad_accum)` | — | 8,840 ÷ 8 = **1,105** ✔ (CS-17 §1.4) |
| **Rows surviving tokenisation** | `steps × grad_accum` | — | **The single most useful diagnostic in the tool.** 1,105 × 8 = 8,840 rows from a 10k file ⇒ ~12% dropped. |
| **Tokens trained** | `total_steps × micro_bs × grad_accum × sequence_len` | — | 1,866 × 24,576 = 45.9 M |
| **Packing multiplier** | `sequence_len ÷ mean_row_tokens` | — | 2048 ÷ 340 ≈ 6× (why the claim is "2–6×", not "6×") |
| **Padding efficiency** | `mean ÷ batch_max` (unpacked), ≈95–99% (packed) | — | 300 ÷ 850 ≈ 35% → ~99% |
| **LoRA params, one matrix** | `r × (d_in + d_out)` | r = rank | 36 × 32 × (2048+2048) = 9.4 M for q,v |
| **LoRA update scale** | `alpha / r` | — | r=32, alpha=64 → scale 2 |
| **Adapter fraction** | `~1–3%` of base params | GQA-dependent | 7B r=32, 7 modules ≈ 75–160 M |
| **Full-FT bytes/param** | `2 (bf16 w) + 2 (grad) + 4 (fp32 master) + 8 (Adam m,v)` | — | `16 B/param` → 7B = **112 GB** |
| **Floor bytes/param (memory.py)** | `2 + 2 + 8 = 14` | no master copy | 7B = **91.6 GiB** — see §7 |
| **ZeRO-3 per-GPU** | `16P ÷ N` + activations | N = GPUs | 70B on 8 → 140 GB/GPU (still > 80 GB; needs offload) |
| **VRAM peak (adapter)** | `W_base + W_adapter + G_adapter + O_state + A_activations + F_frag` | — | 3B QLoRA, seq 1024, mbs 1 ⇒ ~4.4–6.6 GB |
| **`cu_seqlens`** | cumulative offsets `[0, l₁, l₁+l₂, …]` | — | The *only* thing preventing cross-document attention when packing |
| **Training FLOPs** | `6 × N × D` | N params, D tokens | 7B, 10M tokens → 4.2e17 |
| **Tokens from words** | `tokens ≈ words × 1.33` | English, BPE | 1,000 words ≈ 1,330 tokens |
| **GPU-hours** | `total_steps × sec_per_step ÷ 3600` | — | 1,866 × 5.8 s ≈ 3.0 GPU-h |

---

## 3. Decision Tree

```
WHICH INTERFACE?
├─ Production / CI / audit trail        → Path A: YAML + `axolotl train config.yml`
└─ Notebook spike / need a debugger     → Path B: DictDefault + load_cfg + train()
        └─ but write the YAML out anyway. A notebook run with no config file
           is an experiment you cannot re-run (CS-17 §4.2.2).

WHICH ADAPTER?
├─ Big GPU + big data + a measured gap over LoRA → full FT (omit `adapter:`)
├─ Default                                        → adapter: lora
└─ One GPU / tight VRAM                           → adapter: qlora + load_in_4bit: true
        └─ Rule: quantised base ⇒ frozen base ⇒ adapter-only. You cannot
           QLoRA today and full-fine-tune the same run tomorrow.
WHICH DATASET `type:`?  (look at ONE LINE of the file first)
├─ {"messages":[{role,content}]}        → type: chat_template       ← the modern default
├─ {"conversations":[{from,value}]}     → type: chat_template + field_messages: conversations
│                                          + message_property_mappings: {role: from, content: value}
├─ {"instruction","input","output"}     → type: alpaca (+ field_* if your names differ)
├─ {"text": "..."}                      → type: completion  (⚠ no role masking — see §8)
└─ {"input_ids","labels",...}           → type: (leave empty)  — you own the masking

WHICH CHAT TEMPLATE?
├─ The tokenizer has one  → chat_template: tokenizer_default     ← never type a name you can inherit
├─ It does not, and you know the family → chat_template: qwen3 | chatml | llama3
└─ Custom Jinja string    → chat_template: jinja + chat_template_jinja: "<|...|>"
        └─ Then verify eot_tokens are SINGLE tokens (§4.4 trap 4).
PACK THE SEQUENCES?
├─ rows ≪ sequence_len AND backend is FA2/FA3/FA4 → sample_packing: true
├─ backend is sdpa/eager/xformers                 → packing is NOT legal; leave it false
├─ rows are already near sequence_len             → packing buys ~1.0×; skip the risk
└─ preference training (rl: ...)                  → leave it false (CS-17 §15.4)
HOW MANY GPUs?
├─ 1        → nothing to configure. DDP is implicit.
├─ 2–8, LoRA/QLoRA → FSDP2 (`fsdp_version: 2` + `fsdp_config:`) — Axolotl's recommendation
├─ 2–8, existing ZeRO JSONs → deepspeed: deepspeed_configs/zero2.json (then 3 if OOM)
└─ weights don't fit at all → ZeRO-3 + offload, or FSDP2 with offload_params: true

HAVE YOU VALIDATED?
└─ No → STOP. `axolotl preprocess config.yml --debug --debug-num-examples 3` — §4.5.
```

> **The `type:` choice is the highest-risk key in the file, and it is not the one people
> check.** A wrong `chat_template` degrades quality; a wrong `type:` changes *what the
> model is trained on*. `type: completion` on a chat dataset trains on the serialised
> conversation as raw text with no role masking: loss looks great (~0.5) and the model
> answers questions by continuing them (CS-17 §4.5.2, §15.1).

---

## 4. Hyperparameter Quick Reference

### 4.1 The YAML keys that matter

| Key | Default | Typical | Effect of getting it wrong |
|---|---|---|---|
| `base_model` | — | HF id or local path | Wrong family ⇒ vocabulary **and** chat-template mismatch |
| `adapter` | omitted (= full FT) | `lora` / `qlora` / `loftq` | `qlora` without `load_in_4bit: true` → validation error |
| `load_in_4bit` | `false` | `true` for QLoRA | Quantises the *frozen base*; makes full FT impossible |
| `sequence_len` | **512** | p99.5 of your data | Rows longer than this are **dropped** by default, not truncated (`excess_length_strategy: drop`) |
| `micro_batch_size` | — | 1–8 | The main VRAM knob. Not the batch size. |
| `gradient_accumulation_steps` | — | 4–16 | Restores effective batch; costs **zero** VRAM |
| `learning_rate` | — | 1e-4…3e-4 (LoRA) / 1e-5…5e-5 (full) | The LoRA-FT LR mix-up is the classic way to destroy a model |
| `num_epochs` | **1.0** | 1–3 | 3+ overfits, 4+ forgets |
| `lr_scheduler` | `cosine` | `cosine` | `cosine` decays to ~0, which is what makes a short run settle |
| `warmup_steps` / `warmup_ratio` | — | 5–10% of steps | **Mutually exclusive.** Warmup is only meaningful as a fraction |
| `optimizer` | **`adamw_torch_fused`** | `paged_adamw_8bit` (QLoRA) | The default *changed*; write it down or you have a silent A/B confound |
| `max_grad_norm` | — | 1.0 | 0.1 (the video's demo value) is tight and slows long runs |
| `sample_packing` | off unless set | `true` when rows ≪ `sequence_len` | 2–6× throughput, or silent cross-document attention |
| `pad_to_sequence_len` | `true` when packing is on | state it explicitly | With packing off, you pay full compute for pad tokens |
| `train_on_inputs` | **`false`** | leave `false` | `true` teaches the model to write *your prompts* |
| `roles_to_train` | `["assistant"]` | `["assistant"]` | Dataset-level; the modern expression of prompt masking |
| `train_on_eos` | `turn` | `turn` | If the EOS is masked, the model never learns to stop |
| `val_set_size` | **0.0** | 0.05 | **No eval by default.** Mutually exclusive with `test_datasets:` |
| `eval_sample_packing` | inherits `sample_packing` | `false` | Packing changes the loss denominator — an unpacked eval stays comparable |
| `strict` | `false` | `true` in CI | Off ⇒ a typo'd key is a silent `None` |
| `excess_length_strategy` | `drop` | `drop` / `truncate` / `raise` | Default silently deletes data; use `raise` while developing |
| `dataset_prepared_path` | `last_run_prepared` | same | Change the template and you may reuse a **stale cache** |
| `seed` | Axolotl sets one | `42` | Unrecorded seed ⇒ the run is a distribution, not a point |
| `bf16` / `fp16` | `auto` | `bf16` (Ampere+), `fp16` (Turing/T4) | `bf16: true` on a T4 fails; `fp16` on Ampere can NaN |
| `attn_implementation` | derived | `flash_attention_2` | A stripped legacy boolean silently leaves you on SDPA |
| `gradient_checkpointing` | — | `true` | −60…70% activation memory for +25…30% step time |
| `deepspeed` | — | `deepspeed_configs/zero2.json` | Key present on 1 GPU ⇒ DeepSpeed import / `DummyOptim` errors |
| `fsdp_version` + `fsdp_config` | `2` | see §5.5 | The bare `fsdp:` list is rejected; `fsdp_version: 1` is a hard error |
| `strict`-safe extra | — | `gradient_checkpointing_kwargs: {use_reentrant: false}` | Copy the value your topology needs; it is not a tuning knob |

### 4.2 LoRA keys

| Key | Default | Typical | Notes |
|---|---|---|---|
| `lora_r` | — | **16** | 8–32 covers most work; the video uses 32 |
| `lora_alpha` | — | **2 × r** (32) | Only the ratio `alpha/r` matters — it is the adapter's loudness |
| `lora_dropout` | **0.0** | 0.05 on small data | **Incompatible with the fused LoRA kernels** (`lora_qkv_kernel` / `lora_o_kernel` / `lora_mlp_kernel`) |
| `lora_target_modules` | — | all 7: `q,k,v,o,gate,up,down_proj` | Attention-only underfits complex tasks |
| `lora_target_linear` | — | `true` to be model-agnostic | Removes the "`q_proj` vs `query`" class of bug; also hits layers you may not want |
| `lora_model_dir` | — | path to an adapter | Continue from an existing adapter |

> **`r` sets capacity; `alpha/r` sets loudness; `lora_dropout` regularises.** Change `r` and
> `alpha` *together* at a fixed ratio, then sweep LR. Doubling `r` while holding `alpha`
> halves the per-parameter update, and compensating with a higher LR changes the optimiser
> dynamics instead of the parameterisation (CS-17 §7.2, §17.6).

### 4.3 Dataset entry keys and the `type:` table

```yaml
datasets:
  - path: ./data/train.jsonl     # HF id, local file/dir, or cloud URI (fsspec)
    ds_type: json                # json | csv | parquet | arrow  — LOCAL data only
    type: chat_template          # the prompt strategy — NOT the same key as ds_type
    chat_template: tokenizer_default
    field_messages: messages
    message_property_mappings: {role: role, content: content}
    # revision: <sha>            # pin a Hub dataset
```

| `type:` | Row shape | Required keys | If you use it on the wrong data |
|---|---|---|---|
| `chat_template` | `{"messages":[{role,content}]}` | `chat_template:`, `field_messages`, `message_property_mappings` | On an alpaca file: no message list found → error or empty conversations |
| `alpaca` | `{"instruction","input","output"}` | `field_instruction`, `field_input`, `field_output`, optional `field_system` | On a `messages` file: empty/null fields, loss collapses toward zero |
| `sharegpt` | `{"conversations":[{from,value}]}` | `message_property_mappings` | **Deprecated** — migrate to `chat_template` + `field_messages: conversations` + `{role: from, content: value}` |
| `completion` | `{"text": "..."}` | optional `field:` to rename the column | On chat data: trains on the serialised conversation, **no masking** — loss ~0.5, model continues your questions |
| `pretrain` | `{"text": "..."}` (streaming) | `pretraining_dataset:`, `text_column`, and `max_steps` | Under `datasets:` it is a config error. Streams forever |
| `input_output` (template-free) | `{"segments":[{text,label}]}` | explicit per-segment `label` | You must hand-label every trained span |
| pre-tokenised | `{"input_ids","attention_mask","labels"}` | leave `type:` **empty** | Nothing checks your masking for you |
| `chat_template.default` | `{messages, chosen, rejected}` | `field_messages`, `field_chosen`, `field_rejected` | **This is the DPO type.** `type: dpo` / `type: preference` are not documented today |
| `chatml.ultra`, `chatml.intel`, `chatml.argilla`, `llama3.ultra`, … | format-specific DPO | none beyond the type | Each expects a precise layout (`chatml.intel` wants `{question, chosen, rejected}`) |
| `user_defined.default` | anything | `field_prompt`, `field_chosen`, `field_rejected` + `prompt_format`/`chosen_format`/`rejected_format` | The escape hatch when your preference data matches none of the canned shapes |

### 4.4 The eight YAML traps, precisely

| # | Trap | The right form | What goes wrong |
|---|---|---|---|
| 1 | **`chat_template:` is a *name*, not a template** | `chat_template: qwen3` (a named constant) | `chat_template: jinja` + `chat_template_jinja: "<|im_start|>..."` for an inline Jinja string. Mixing them up gives a rendering error or the wrong format |
| 2 | **`type:` ≠ `ds_type:`** | `type:` = prompt strategy; `ds_type:` = file format for local data | Setting `ds_type: chat_template` is a config error; setting `type: json` silently selects nothing useful |
| 3 | **Field renaming needs `field_*` + `message_property_mappings`** | `field_messages`, `field_instruction`, `field_output`, `field_chosen`, `field_rejected`; `message_property_mappings: {role:, content:}` | Renamed columns with no mapping ⇒ empty or misaligned targets |
| 4 | **`eot_tokens` must each be ONE tokenizer token** | `eot_tokens: ["<|im_end|>"]` | A multi-token terminator shifts the mask offsets; Axolotl warns that the tokenizer will split it |
| 5 | **`special_tokens:` resizes embeddings** | `special_tokens: {pad_token: "<|eot_id|>"}` | Growing is automatic; shrinking needs `shrink_embeddings: true`. Asymmetry ⇒ merge-time shape mismatches |
| 6 | **`sequence_len`, not `max_seq_length`** | `sequence_len: 2048` | With `strict: true` a `max_seq_length` key fails the config; without it, it is silently ignored and you get **512** |
| 7 | **`adapter: lora_llama` and friends** | `adapter: lora` | Older configs use model-family aliases. They are redundant today; if your version rejects them, drop the suffix. Verify against your pinned install, not a card |
| 8 | **`sample_packing` + `pad_to_sequence_len` travel together** | `pad_to_sequence_len` defaults to `true` **when packing is on** | Stating it explicitly is free; assuming it when packing is off wastes compute on pad tokens |

> **YAML itself, in one line each.** Indentation is syntax and **tabs are illegal**
> (`yaml.scanner.ScannerError: found character '\t' that cannot start any token` — the most
> common copy-paste failure from a web page). The space after `key:` is required;
> `key:value` parses as one scalar string. Unquoted `yes`/`no`/`on`/`off` are booleans.
> `learning_rate: 2e-4` is a float; `lora_alpha: "2e-4"` is a string and will fail type
> validation. Check syntax in 50 ms before you schedule the GPU:
> `python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" config.yml`.

### 4.5 Fail fast — the workflow win over LLaMA-Factory

Axolotl parses and validates the whole config through a pydantic schema **before it loads a
model**, so an illegal *combination* fails at startup rather than at step 900. What that
does and does not cover:

| Layer | Validated loudly by Axolotl? | Example |
|---|---|---|
| YAML syntax | At `load_cfg()` | A tab character, an unquoted `key:value` |
| Key *values* and *types* | **Yes** — pydantic schema | `micro_batch_size: "two"` |
| Illegal *combinations* | **Yes** | `adapter: qlora` without `load_in_4bit: true`; a legacy attention boolean **plus** `attn_implementation`; `fsdp_version: 1`; the bare `fsdp:` list |
| Unknown **keys** | **No** — warned and dropped | `training_type: sft` (not an Axolotl key); `max_seq_length` |
| Deprecated keys | **No** — `DeprecationWarning`, then **stripped** | `flash_attention: true` → you get the computed default (often SDPA), not FA2 |
| The dataset contract (`type:`, template, mask) | **No** — nothing can check this for you | `type: completion` on chat data; a wrong `chat_template` name |

The three-step loop that turns the bottom two rows into loud failures:

```bash
# 1. YAML syntax — 50 ms, no imports, no GPU.
python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" sft_pharma_v1.yaml

# 2. Schema + deprecations — loads the config, validates it, prints the resolved result.
#    Run ONCE with strict: true in the YAML. A typo now fails instead of becoming None.
axolotl preprocess sft_pharma_v1.yaml --debug --debug-num-examples 3

# 3. The dataset contract — the part no schema covers. Read the printed input_ids:
#    are the model's OWN turn markers present? is the user span -100? is the final
#    EOS/EOT position LABELLED (not -100)? does the answer end naturally?
```

Any "no" in step 3 is a config change, not a hyperparameter search — and it costs seconds
instead of the 40 minutes the case study's first attempt burned (CS-17 §15.1).

---

## 5. Copy-Paste Code Snippets

### 5.1 A full, working single-GPU YAML (QLoRA SFT)

```yaml
# sft_pharma_v1.yaml — run:  axolotl preprocess sft_pharma_v1.yaml --debug --debug-num-examples 3
#                           axolotl train sft_pharma_v1.yaml

# ── 1. Model identity + adapter method (quantised base ⇒ frozen base ⇒ adapter-only) ──
base_model: meta-llama/Llama-3.1-8B-Instruct
tokenizer_type: AutoTokenizer
adapter: qlora
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16        # Ampere+; float16 on a T4
lora_r: 32
lora_alpha: 64                          # 2 × rank
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# ── 2. Attention, memory, sequence, batching ─────────────────────────────────
attn_implementation: flash_attention_2  # varlen-capable; required for packing
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
embeddings_skip_upcast: true
sequence_len: 2048                      # p99.5 of YOUR token lengths, not the model's max
sample_packing: true
pad_to_sequence_len: true               # default when packing is on; stated for clarity
micro_batch_size: 2
gradient_accumulation_steps: 8          # 2 × 8 × 1 gpu × 2048 = 32,768 tokens/step

# ── 3. Optimisation + precision ──────────────────────────────────────────────
optimizer: paged_adamw_8bit
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.05                      # mutually exclusive with warmup_steps
num_epochs: 2
max_grad_norm: 1.0
bf16: true                              # `auto` is the default; be explicit
fp16: false
tf32: true

# ── 4. Data ──────────────────────────────────────────────────────────────────
datasets:
  - path: ./data/pharma_sft_train.jsonl
    ds_type: json
    type: chat_template
    chat_template: tokenizer_default     # never type a template name you can inherit
    field_messages: messages
    message_property_mappings: {role: role, content: content}
    roles_to_train: ["assistant"]
    train_on_eos: turn
val_set_size: 0.05
eval_steps: 50
eval_sample_packing: false               # keep eval unpacked so metrics compare across runs
special_tokens:                          # only if the tokenizer needs them
  pad_token: "<|eot_id|>"

# ── 5. Run management + observability ────────────────────────────────────────
output_dir: ./outputs/pharma-v1
dataset_prepared_path: ./last_run_prepared
save_steps: 200
saves_per_epoch: 2
logging_steps: 5
seed: 42
strict: true                             # fail on an unknown key. Always.
# hub_model_id: acme/pharma-qwen-v1      # omit to skip the automatic upload
wandb_project: pharma-sft
wandb_name: llama31-8b-qlora-r32-lr2e4-v1
wandb_watch: gradients
wandb_log_model: end
```

**Change only these for your own run:** `base_model`, `datasets[].path`, `sequence_len`
(measure it), `output_dir`. Everything else is a starting point you should *measure*.

### 5.2 LoRA instead of QLoRA — the three-line diff

```yaml
adapter: lora                 # was: qlora
load_in_4bit: false           # was: true    (the key can stay, but be explicit)
bf16: true                    # the base is now unquantised: 2 B/param, not 0.5
# delete: bnb_4bit_quant_type, bnb_4bit_compute_dtype
micro_batch_size: 1           # 7–8B does not fit at mbs 2 on a 24 GB card at 2048
optimizer: adamw_torch_fused  # paged_adamw_8bit is a QLoRA habit, not a requirement
learning_rate: 2e-4           # unchanged — LoRA and QLoRA share the LR band
```

### 5.3 QLoRA — what the fragment must contain

```yaml
# A "qlora.yaml" fragment is a diff, NOT a runnable config. Axolotl has no
# --base-config overlay: compose the file yourself and keep THAT as the artefact.
load_in_4bit: true
bnb_4bit_compute_dtype: float16   # T4; bfloat16 on Ampere+
bnb_4bit_quant_type: nf4
adapter: qlora
optimizer: paged_adamw_8bit
```

### 5.4 DPO on top of the SFT adapter

```yaml
# dpo_pharma_v1.yaml — base_model is the SFT result, NOT the original base model
base_model: ./outputs/pharma-v1
adapter: qlora
load_in_4bit: true
lora_r: 32
lora_alpha: 64
attn_implementation: flash_attention_2
gradient_checkpointing: true
sequence_len: 2048
sample_packing: false             # do not pack preference pairs
micro_batch_size: 1
gradient_accumulation_steps: 16

rl: dpo                           # dpo | orpo | kto | simpo | grpo | gdpo | ebft
rl_beta: 0.1                      # replaces the deprecated `dpo_beta`
dpo_loss_type: [sigmoid]          # [ipo] for IPO — IPO is a loss, not an `rl:` value
learning_rate: 5e-6               # 10–40× below the SFT LR
lr_scheduler: cosine
warmup_ratio: 0.1
num_epochs: 1
optimizer: paged_adamw_8bit
bf16: true
max_grad_norm: 1.0

datasets:
  - path: ./data/pharma_prefs.jsonl
    ds_type: json
    type: chat_template.default   # the current DPO type; NOT `type: preference`
    field_messages: messages
    field_chosen: chosen
    field_rejected: rejected
    message_property_mappings: {role: role, content: content}

val_set_size: 0.05
remove_unused_columns: false      # required by several RL trainers
output_dir: ./outputs/pharma-dpo-v1
```

> **DPO's reference distribution is the model you start from.** `base_model:` points at the
> SFT checkpoint; if it points at the raw base model, `π_ref` is a model that cannot follow
> instructions and the preference signal is meaningless. This is the single most common DPO
> setup error (CS-17 §6.3).

### 5.5 Multi-GPU with FSDP2 (the corrected dialect)

```yaml
fsdp_version: 2                         # default; `1` is a hard error now
fsdp_config:
  offload_params: false                 # true = CPU-offload when idle (smaller, slower)
  cpu_ram_efficient_loading: true       # rank-0 load + broadcast: saves host RAM
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer   # find it via _no_split_modules
  reshard_after_forward: true           # ≈ FSDP1 FULL_SHARD; safest, slowest
  state_dict_type: FULL_STATE_DICT      # SHARDED_STATE_DICT for very large models
  activation_checkpointing: true
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
```

```yaml
# Or DeepSpeed, with the profiles the CLI fetches for you:
deepspeed: deepspeed_configs/zero2.json   # start at 2; go to 3 only if you OOM
# or inline, for a self-contained artefact:
deepspeed:
  zero_optimization: {stage: 2}
  bf16: {enabled: true}
```

**The effective-batch trap in one line:** moving from 1 GPU to 4 multiplies your effective
batch by 4. Either divide `gradient_accumulation_steps` by the GPU count (holds tokens/step
constant — comparable runs) or scale the LR and record that you did. Never silently do
neither (CS-17 §6.2, §14.2 row 15).

---

## 6. CLI Commands

```bash
# ── Install (the instructor's own sequence; `--no-build-isolation` is required for flash-attn)
pip uninstall -y axolotl peft transformers accelerate datasets trl optimum flash-attn
pip install --no-build-isolation "axolotl[flash-attn]>=0.9.1"
pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@318b7e2"
# then RESTART the session — flash-attn and bitsandbytes link against the installed torch
# canonical repo is axolotl-ai-cloud/axolotl; OpenAccess-AI-Collective is stale

# ── Validate BEFORE you spend a GPU-second (§4.5) ───────────────────────────
python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" sft_pharma_v1.yaml
axolotl preprocess sft_pharma_v1.yaml --debug --debug-num-examples 3
rm -rf last_run_prepared/                 # never debug against a stale cache

# ── Train ──────────────────────────────────────────────────────────────────
axolotl train sft_pharma_v1.yaml                          # the main entry point
axolotl train sft_pharma_v1.yaml --learning-rate 1e-4     # any key can be overridden
axolotl train sft_pharma_v1.yaml --max-steps 20 --val-set-size 0   # a 20-step smoke test

# ── Distributed (everything after `--` goes to the launcher) ────────────────
axolotl train sft_pharma_v1.yaml --launcher torchrun -- --nproc_per_node=4 --nnodes=1
axolotl train sft_pharma_v1.yaml --launcher accelerate -- --config_file=accelerate.yaml --num_processes=8
accelerate launch -m axolotl.cli.train sft_pharma_v1.yaml  # the legacy form; still works

# ── Resume ─────────────────────────────────────────────────────────────────
axolotl train sft_pharma_v1.yaml --resume_from_checkpoint ./outputs/pharma-v1/checkpoint-400

# ── Inspect / fetch known-good configs and the ZeRO JSONs ────────────────────
axolotl fetch examples                      # --dest <folder> to choose where
axolotl fetch deepspeed_configs

# ── Inference, merge, export ───────────────────────────────────────────────
axolotl inference sft_pharma_v1.yaml --lora-model-dir ./outputs/pharma-v1 --gradio
axolotl merge-lora sft_pharma_v1.yaml --lora-model-dir ./outputs/pharma-v1   # → merged/
axolotl merge-lora sft_pharma_v1.yaml --lora-model-dir ./outputs/pharma-v1 --dequant  # bf16 out
axolotl merge-sharded-fsdp-weights <sharded_checkpoint_dir>   # FSDP shards are not servable
axolotl export sft_pharma_v1.yaml --quantize Q4_K_M           # GGUF for llama.cpp/Ollama

# ── Evaluate ───────────────────────────────────────────────────────────────
axolotl evaluate sft_pharma_v1.yaml --lora-model-dir ./outputs/pharma-v1
axolotl lm-eval  sft_pharma_v1.yaml --lora-model-dir ./outputs/pharma-v1  # needs lm_eval_tasks

# ── Environment hygiene the video's Colab cell does ────────────────────────
export AXOLOTL_DO_NOT_TRACK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```python
# The same job as an object (Path B) — note `load_cfg` is where validation happens
from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault
from axolotl.common.datasets import load_datasets
from axolotl.train import train

config = DictDefault(base_model="Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True,
                     adapter="qlora", sequence_len=1024,
                     datasets=[{"path": "winglian/pirate-ultrachat-10k",
                                "type": "chat_template", "split": "train",
                                "eot_tokens": ["<|im_end|>"]}])
cfg = load_cfg(config)                       # ← schema validation + defaulting
dataset_meta = load_datasets(cfg=cfg)        # ← input_ids / labels / attention_mask
cfg.max_steps = 25
model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

# The one-line assertion that has saved more GPU-hours than any sweep:
assert model.config._attn_implementation == "flash_attention_2", \
    f"expected FA2, got {model.config._attn_implementation}"
```

```bash
# ── Deprecation sweep: run before copying ANY config from a tutorial or a repo ──
grep -nE 'flash_attention|xformers_attention|sdp_attention|dpo_beta|training_type|^fsdp:|type: *preference' *.yaml
```

---

## 7. VRAM / Cost Calculator

### 7.1 The arithmetic floor, from `code/common/memory.py --table`

Single GPU, gradient checkpointing **on**, AdamW, bf16 weights, fp32 optimiser states. GB:

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

> **Why CS-17 §11.4 shows numbers ~1.7–2× higher than this table — and both are right.**
> §11.4 quotes *observed peaks* on real hardware; `memory.py` prices the *tensors and
> nothing else*. Three reasons for the gap, in order of size:
>
> 1. **Full FT: 16 vs 14 bytes/param.** §11.4 counts a separate fp32 master copy
>    (`2 + 2 + 4 + 8 = 16 B/param` → 7B = 112 GB). `memory.py` folds it into 14 B/param
>    (7B = 91.6 GiB). Same recipe, +22% purely from bookkeeping.
> 2. **QLoRA dequantises on the fly.** bitsandbytes stores the base in NF4 but upcasts each
>    layer to bf16 for the matmul, so peak memory transiently holds both. A 7B QLoRA run
>    priced at **4.9 GiB** is routinely observed at **8–12 GB** once the CUDA context
>    (~0.5–1 GB), the tokenizer/dataloader, longer `sequence_len` activations and allocator
>    fragmentation land on top. CS-17 §11.4's "~10 GB" for 7–8B QLoRA is that observation.
> 3. **GiB vs GB.** This table is binary (1024³); vendor and blog figures are usually decimal
>    (1000³) — another +7.4% on every row.
>
> **How to use it:** the table is the *lower bound you must beat*. Add 10–20% for a plan and
> trust a real measurement over any table — including this one. CH-13 §7 has the full
> accounting; CS-17 §4.4.1 has the six-term model.

### 7.2 The six-term model (why the panic estimate is always wrong)

```
VRAM_peak ≈  W_base          (frozen weights — quantised: P × 0.5 B)
           + W_adapter       (the A/B matrices — ~1–3% of P)
           + G_adapter       (their gradients)
           + O_state         (Adam moments for the adapter only)
           + A_activations   (≈ k × micro_bs × seq_len × d_model × layers × bytes)  ← dominant
           + F_frag          (CUDA context + workspaces + fragmentation: 5–15%)
```

The video's run: **Qwen2.5-3B QLoRA, `sequence_len: 1024`, `micro_batch_size: 1`, T4.**

| Component | GB |
|---|---|
| Base weights, NF4 (3.09e9 × 0.5 B + double-quant scales) | ~1.60 |
| Adapter + gradients (7 modules, r=32, ~75 M) | ~0.30 |
| Optimizer state (`paged_adamw_8bit`) | ~0.15–0.45 |
| Activations, 1024 × 2048 × 36, checkpointed | ~1.5–3.0 |
| CUDA context, workspaces, fragmentation | ~0.8–1.2 |
| **Total** | **≈ 4.4 – 6.6** |

> **Correction:** at [48:32]–[48:45] the instructor warns the free GPU has *"just 12 GB of
> VRAM"* and that the run *"maybe requires 24 GB of VRAM at minimum"*. **Both figures are
> wrong.** The free Colab GPU is a **Tesla T4 with 16 GB** of GDDR6 (~14.5–15 GiB usable
> after driver reservation), and the run needed **~5 GB** — it completed on the T4 with a
> 4-bit base, `micro_batch_size: 1`, `sequence_len: 1024` and gradient checkpointing. The
> "24 GB minimum" figure applies to an *unquantised* LoRA fine-tune of a 3B at a longer
> sequence length. The instructive part is *why* the estimate was 3× high: it was reasoned
> from the model's parameter count instead of from the six terms above (CS-17 §4.4.2).

### 7.3 Full FT and multi-GPU — the 16-bytes-per-parameter budget

```
2 B weights + 2 B gradients + 4 B fp32 master + 4 B Adam m + 4 B Adam v = 16 B/param
   →  7 B ≈ 112 GB   →   70 B ≈ 1,120 GB   →  neither fits one 80 GB card
```

| Strategy | Weights | Gradients | Optimiser state | Per-GPU, 7B, N=8 |
|---|---|---|---|---|
| DDP (nothing configured) | full (2P) | full (2P) | full (12P) | `16P` = **112 GB** → OOM |
| ZeRO-1 | full (2P) | full (2P) | `12P/N` | **38.5 GB** |
| ZeRO-2 | full (2P) | `2P/N` | `12P/N` | **26.3 GB** |
| ZeRO-3 | `2P/N` | `2P/N` | `12P/N` | **14 GB** |
| FSDP2 | per-layer shard | sharded | sharded | ≈ ZeRO-3, **~14 GB** |
| + activations + workspace | — | — | — | **+ 4–12 GB** with checkpointing |

**Pick the lowest stage that fits.** ZeRO-3 shards the *parameters*, so every forward pass
must re-gather them: more communication for no benefit when the weights already fit. For LoRA
on 2–4 GPUs, **ZeRO-2 is usually the right answer** (CH-15 §7.2 says the same thing).

### 7.4 Minimum viable hardware, and throughput

| Goal | Hardware |
|---|---|
| QLoRA ≤8B / 13–34B / 70B | 1 × 16 GB (T4, 4060 Ti) / 1 × 24 GB (3090/4090, L4, A10G) / 1 × 48–80 GB (A6000, A100) |
| LoRA 7–8B | 1 × 24–40 GB |
| Full FT 7–8B / 70B | 2–4 × A100-80GB (FSDP2 or ZeRO-3) / 8–16 × A100-80 or H100, ZeRO-3 **+ offload** — 8 GPUs alone is not enough (140 GB/GPU) |

| GPU | VRAM | ≈ $/hr | ≈ QLoRA 8B tok/s | Note |
|---|---|---|---|---|
| T4 | 16 GB | 0.15–0.35 | 500–700 | **~0.5k tok/s** — a 10 M-token epoch is ~5 h |
| RTX 4090 | 24 GB | 0.35–0.70 | 2,000–3,000 | Cheapest per token for QLoRA |
| L4 / A10G | 24 GB | 0.50–0.90 | 1,200–1,800 | Reliable single-GPU work |
| A100-40/80 | 40/80 GB | 1.20–2.50 | 3,500–5,000 | The default serious single GPU |
| H100-80 | 80 GB | 2.50–4.00 | 6,000–9,000 | Deadline-driven runs |

```text
The estimation recipe (CS-17 §11.1) — step 5 is the only one nobody measures:
1. rows_surviving = steps_per_epoch × micro_batch_size × gradient_accumulation_steps
2. total_steps    = steps_per_epoch × num_epochs      (or max_steps)
3. tokens_trained = total_steps × micro_bs × grad_accum × sequence_len
4. sec_per_step   = MEASURE it from a 20-step smoke run      ← do this
5. gpu_hours      = total_steps × sec_per_step / 3600
6. dollars        = gpu_hours × price_per_gpu_hour
```

> **Training is nearly free; data is the cost centre.** CS-17 §11.3 prices a 45.9 M-token
> 8B QLoRA run at **≈ $4.60**, and the 10,000 curated rows that fed it at **~$400,000** of
> expert time — four orders of magnitude apart. The engineering consequence is not "save GPU
> money"; it is "spend GPU money freely to *check the data*" (more ablations, more eval),
> because a re-run costs $5.

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| **Loss flat at ~ln(vocab)** (≈11.9 for Qwen's 152k vocab) and never moves | Every label is `-100` — a wrong `type:`, a wrong template, or a template that never marks an assistant turn | `axolotl preprocess --debug`; read the rendered text. See §11 row 1. |
| Loss falls but the model **echoes the prompt** | `train_on_inputs: true`, or the user turn rendered as trainable | `train_on_inputs: false`; `roles_to_train: ["assistant"]`; verify the mask |
| Model **never stops generating** | `train_on_eos` not training the EOS, or `eot_tokens` missing so `<\|im_end\|>` was masked | `train_on_eos: turn`; `eot_tokens: ["<\|im_end\|>"]`; confirm the final position is **labelled** |
| Loss looks fine, output is fluent but **wrong shape** (no JSON, wrong language, ignores instructions) | **Wrong `chat_template`** | `chat_template: tokenizer_default`; render and read one example (§4.4 trap 1) |
| Loss ~0.5 and the model **continues your question** | `type: completion` on chat data — no role masking at all | `type: chat_template` + `field_messages` |
| Loss **→ 0** / perplexity **→ 1** early | Packing with broken boundary metadata — the model is reading the next document as its own continuation | Stop. A/B test packed vs unpacked for 100 steps at fixed `max_steps` (§4.3 note below) |
| Loss **~1.5–2× the unpacked baseline** and flat | Attention crossing packed document boundaries | Same A/B test; disable packing for that architecture |
| `sample_packing: true` set and the backend is `sdpa`/`eager` | Packing is **not legal** without a varlen backend | `attn_implementation: flash_attention_2`, or `sample_packing: false` |
| You asked for FA2 and got SDPA | A deprecated boolean (`flash_attention: true`) was **stripped** with only a warning | `attn_implementation: flash_attention_2`; assert `model.config._attn_implementation` |
| **Steps far fewer than expected** | `excess_length_strategy: drop` (the default) removed over-long rows | `steps × grad_accum` vs the file's row count; raise `sequence_len`, or filter, or `raise` in dev |
| Answers truncated mid-sentence in the training data | `sequence_len` too small — and loss looks *better* with truncated answers in | Raise `sequence_len`, or filter rows to fit. Check p95 first |
| **`CUDA out of memory` at step 1** | `micro_batch_size × sequence_len` | Halve `micro_batch_size`; `gradient_checkpointing: true`; QLoRA; halve `sequence_len` |
| **OOM only at eval** | `eval_sample_packing` differs, or the eval batch is not scaled down | `eval_sample_packing: false`; smaller eval batch |
| **OOM after N steps**, not step 1 | Fragmentation with variable-length batches and packing off | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; enable packing; sort by length |
| Model is **worse than the base** but loss was fine | Template/train-serve mismatch, or a stale `dataset_prepared_path` cache | `rm -rf last_run_prepared/` after **any** template or `type:` change; re-run `preprocess --debug` |
| **Loss differs between 1 GPU and 4** | Effective batch changed by 4×; LR never rescaled | Divide `grad_accum` by the world size, or scale the LR (√ rule) — and record which |
| Multi-GPU **hangs** at startup | NCCL cannot see the GPUs, or `--nproc_per_node` ≠ the real GPU count | Match the count; in Docker add `--gpus all --ipc=host`; `NCCL_DEBUG=INFO` |
| `deepspeed` errors on a **single GPU** | The `deepspeed:` key with no distributed launcher (`mpi4py`, `DummyOptim`) | Remove the key for single-GPU runs |
| `exitcode: -9` on a multi-GPU run | **Host RAM** exhaustion, not VRAM (offload + `cpu_ram_efficient_loading` raise host pressure) | Reduce offload, or get a machine with more system RAM |
| Run **restarts from step 0** despite checkpoints | `--resume_from_checkpoint` unset, or `output_dir` changed | `axolotl train cfg.yml --resume_from_checkpoint ./out/checkpoint-500` |
| Adapter loads but **output is unchanged** | Base loaded without the adapter, or model not in eval, or an empty adapter | `axolotl inference cfg.yml --lora-model-dir ...`; confirm `adapter_config.json` exists |
| DPO run **degrades** the model | LR carried over from SFT (2e-4 instead of 5e-6), or `sample_packing: true` left on | §5.4's config; LR 5e-6, packing off |
| Adapter merge fails with a **shape mismatch** | `special_tokens` grew the embeddings on one side only | Grow both; merge with `axolotl merge-lora`, not a manual `PeftModel` merge |
| **`W&B shows no metrics`** | `wandb_project` set but not logged in, or `wandb_mode: disabled` left from a debug run | `wandb login`; fail the job if `wandb_mode != online` |
| `val_set_size: 0` (the default) and no eval loss ever appears | Nobody notices because the train loss is fine | `val_set_size: 0.05` + `eval_steps: 50` |

> **The packing A/B test is the only definitive check, and it belongs in your repo.** Run
> 50–100 steps twice, identical config except `sample_packing`, with `max_steps` fixed. Some
> gap is *expected* (packing changes the loss denominator), so calibrate the expected gap once
> on a known-good model and treat deviations from *that* as the alarm — not deviations from
> zero (CS-17 §4.6.4).

---

## 9. Comparison Matrix

### 9.1 Axolotl vs the alternatives

| Dimension | **Axolotl** | **LLaMA-Factory** | **Unsloth** | **torchtune** |
|---|---|---|---|---|
| Interface | YAML + Python API | YAML + **WebUI** + Python | Python (patch API) | YAML recipe + Python recipes |
| No-code path | ❌ | ✅ WebUI | ❌ | ❌ |
| Config-as-artefact | **Yes — the whole point** | Yes | Script | Recipe + config |
| Dataset contract | In the config (`type:`) | In a **separate registry** (`data/dataset_info.json`) | In code | In the recipe |
| Model coverage | Very broad (text + VLM + MoE) | **Broadest** — hundreds | Popular open families, added incrementally | Meta-family-first, growing |
| Speed, single GPU | Good | Good | **Best** (2–4× class) | Good |
| Speed, multi-GPU | **Excellent** (FSDP2, ZeRO 1–3, seq parallel, multi-node) | Good (DeepSpeed/FSDP) | Limited | Good (FSDP2, tensor parallel) |
| Memory efficiency | Very good (QLoRA, packing, Liger, cut-CE, offload) | Very good | **Best on one GPU** | Good |
| Config surface | ~200 keys, fast-moving | Large, WebUI-driven | Minimal (kwargs) | Small, typed, clean |
| Dataset formats | 20+ `type:` strategies | 20+ via the registry | A few + custom | Chat/instruct, less magic |
| Preference methods | SFT, pretrain, DPO, IPO, ORPO, KTO, SimPO, GDPO, GRPO, RM/PRM, EBFT | SFT, pretrain, DPO, ORPO, KTO, PPO, RM | SFT, DPO, GRPO, growing | SFT, DPO, PPO, GRPO |
| Multi-GPU | **DeepSpeed / FSDP2** | DeepSpeed / FSDP | Limited | FSDP2 |
| Custom loss / collator | Plugin or fork | Hard | Medium | Hard |
| Reproducibility | **Best** (YAML + resolved config + Docker) | Best for a UI-driven flow | Script | Recipe |
| Debuggability | Good (`preprocess --debug`) | Good (UI dataset preview) | Good (you hold the object) | **Best** — readable plain PyTorch |
| Learning curve | Medium (YAML + schema churn) | **Lowest** | **Lowest** | Medium–high |
| Failure profile | **Loud on the config, quiet on the data** | Loud on template/dataset *lookups*, quiet everywhere else | You write it, so you own it | You write it |
| Licence | Apache-2.0 | Apache-2.0 | Apache-2.0 (+ a commercial tier) | BSD-3 (Meta) |

> **Correction:** CH-15 §9.1 lists Axolotl's supported stages as *"sft, dpo, orpo, kto"*.
> That list is out of date. Current Axolotl also carries **pretraining, IPO, SimPO, GDPO,
> GRPO, EBFT and reward/process-reward modelling** (CS-17 §13.1). Read the two cards
> together and prefer CS-17's list; then verify against your pinned release, because this is
> exactly the surface that churns.

### 9.2 The same key, two names — Axolotl ↔ LLaMA-Factory

| Concept | **Axolotl** | **LLaMA-Factory** (CH-15 §4) |
|---|---|---|
| Base model | `base_model` | `model_name_or_path` |
| Method | `adapter: lora` / `qlora` / omitted | `finetuning_type: lora` / `full` + `quantization_bit: 4` |
| Sequence window | `sequence_len` | `cutoff_len` |
| Epochs | `num_epochs` | `num_train_epochs` |
| Batch | `micro_batch_size` | `per_device_train_batch_size` |
| LR schedule | `lr_scheduler` | `lr_scheduler_type` |
| Stage selection | *absence* = SFT; `rl:` for preference; `pretraining_dataset:` for CPT | `stage: sft|dpo|orpo|kto|ppo|rm|pretrain` |
| Preference β | `rl_beta` | `pref_beta` |
| Preference loss | `dpo_loss_type: [ipo]` | `pref_loss: simpo|ipo|…` (**not** a stage) |
| LoRA rank / scale | `lora_r` / `lora_alpha` | `lora_rank` / `lora_alpha` |
| LoRA targets | `lora_target_modules` list / `lora_target_linear: true` | `lora_target: all` |
| Attention backend | `attn_implementation: flash_attention_2` | `flash_attn: fa2` |
| Packing | `sample_packing` (+ `pad_to_sequence_len`) | `packing` (+ `neat_packing`) |
| Masking | `roles_to_train`, `train_on_inputs`, `train_on_eos` | `mask_history`, `train_on_prompt` |
| Dataset wiring | `datasets: [{path, type, field_*}]` — **inline** | `dataset: <key>` → `data/dataset_info.json` — **a registry** |
| Multi-GPU | `fsdp_version`/`fsdp_config`, `deepspeed` | `deepspeed: examples/deepspeed/ds_z3_config.json` |
| Merge | `axolotl merge-lora` | `llamafactory-cli export` |
| Validation before training | **Yes — pydantic schema at `load_cfg`** (§4.5) | Partial — template and dataset *lookups* are the only loud failures |

**Which to pick:**

| Situation | Pick |
|---|---|
| "I want it working today and I'm not a Python engineer" | **LLaMA-Factory WebUI** |
| "I have one 24 GB GPU and a 7B model" | **Unsloth** |
| "4–64 GPUs, and it must be reproducible and auditable" | **Axolotl** |
| "I want to read every line of the training loop" | **torchtune** |
| "I need a custom loss or a research modification" | **TRL** |
| "One T4, one afternoon, one run" | **Unsloth or TRL** — not Axolotl (CS-17 §8.1) |

> All four call the same `transformers` + `peft` + `trl` + `bitsandbytes` underneath, and a
> LoRA adapter trained by any of them loads in all of them. The durable skills — the chat
> template, the loss mask, the effective batch, the packing boundary — transfer unchanged.
> **Optimise for the artefact you can hand to a reviewer, not for the framework.**

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| `sequence_len` default | **512** | Raise it — most people assume 2048 |
| `num_epochs` default | **1.0** | 1–3 for SFT |
| `lora_r` | **16** | 8–32 |
| `lora_alpha` | **2 × r** = 32 | Only `alpha/r` matters |
| `lora_dropout` default | **0.0** | 0.05 on small data; illegal with the fused kernels |
| LR (LoRA/QLoRA) | **1e-4 … 3e-4** | 2e-4 is the common default |
| LR (full FT) | **1e-5 … 5e-5** | 10× lower |
| LR (DPO) | **5e-6** | 10–40× below SFT |
| Warmup | **5–10%** of steps | Use `warmup_ratio`, not `warmup_steps` |
| `val_set_size` default | **0.0** | **No eval by default.** Set 0.05 |
| `excess_length_strategy` default | **`drop`** | Silently deletes over-long rows |
| `pad_to_sequence_len` default | **`true` when packing is on** | — |
| `strict` default | **`false`** | Set `true` in CI |
| `optimizer` default | **`adamw_torch_fused`** | The default *changed* — write it down |
| Packing speedup | **2–6×** | = `sequence_len ÷ mean_row_tokens` |
| Padding efficiency, unpacked | **~35%** | → ~99% packed |
| `alpha / r` | the adapter's loudness | Not `alpha` alone |
| Bytes/param, full FT (floor) | **14 B** | 2 + 2 + 8 |
| Bytes/param, full FT (budget) | **16 B** | with an fp32 master copy |
| Bytes/param, NF4 | **0.5 B** | 4-bit + double quant |
| Full FT vs LoRA VRAM, 7B | **~6×** | 91.6 vs 14.7 GB floor |
| QLoRA vs full FT VRAM, 7B | **~20×** | 4.9 vs 91.6 GB floor |
| Healthy train loss (chat SFT) | **0.5 – 2.0** | < 0.4 ⇒ memorisation or empty targets |
| Healthy grad norm | **0.1 – 10** | > 100 ⇒ instability |
| Loss flat at `ln(V)` | **≈ 11.9** | Qwen's 152k vocab. **Not 2.3** — see §11 row 1 |
| Free Colab T4 VRAM | **16 GB** | ~14.5–15 GiB usable. **Not 12** |
| Video's QLoRA run, peak VRAM | **~5 GB** | **Not 24 GB** |
| T4 QLoRA throughput | **~0.5k tok/s** | A 10 M-token epoch ≈ 5 h |
| Adapter size, 7B r=32, 7 modules | **~160 MB** (2-byte dtype) | The base is a dependency, not an artefact |

---

## 11. Common Errors And Their Exact Messages

**Axolotl's loudness is asymmetric, and knowing which column you are in is the point.**

| Error message | Meaning | Fix |
|---|---|---|
| `yaml.scanner.ScannerError: found character '\t' that cannot start any token` | A literal tab in the YAML — usually pasted from a web page | Re-indent with spaces; validate with `yaml.safe_load` in 50 ms |
| A pydantic validation error naming `adapter` / `load_in_4bit` | **Loud, at `load_cfg`, before the model loads.** `adapter: qlora` requires `load_in_4bit: true` | Add `load_in_4bit: true`, or drop to `adapter: lora` |
| An error about `attn_implementation` and a legacy attention boolean | Setting both — Axolotl refuses to pick a winner | Delete `flash_attention:` / `xformers_attention:` / `sdp_attention:`; keep `attn_implementation:` |
| An error on `fsdp_version: 1`, or `fsdp:` given as a list | FSDP1 is removed | `fsdp_version: 2` + an `fsdp_config:` mapping (§5.5) |
| `KeyError: 'input_ids'` from `axolotl preprocess` | `pretraining_dataset:` or `skip_prepare_dataset: true` — those are prepared on demand and cannot be preprocessed offline | Do not run `preprocess` on those configs; train directly |
| `chat_template is tokenizer_default but tokenizer's chat_template is null` | You asked for the tokenizer's own template and the tokenizer has none | Supply the template in the **tokenizer config**, not the training YAML — the serving stack needs it too |
| `ValueError: prefetch_factor option could only be specified in multiprocessing. let num_workers > 0 to enable multiprocessing` | `dataloader_num_workers: 0` **combined with** an explicit `dataloader_prefetch_factor`. Zero workers is legal; the pair is not | Set `dataloader_num_workers: 2` (and keep `prefetch_factor: 2`), **or** keep 0 workers and delete `dataloader_prefetch_factor` entirely |
| `ValueError: num_samples should be a positive integer value, but got num_samples=0` | The dataset resolved to zero rows after filtering/tokenisation — or the same worker/prefetch misconfiguration above | Print the row count in the preprocess log; check the `type:` and `field_*` mappings actually matched |
| A `DeprecationWarning` about `flash_attention` / `xformers_attention` / `dpo_beta` / `fsdp_sharding_strategy` | The key was **stripped** and a default substituted | Treat a `DeprecationWarning` as a failing test, not a note (CS-17 §5.5.6) |
| `training_type: sft` accepted with no effect | **Not an Axolotl key.** Silently ignored because `DictDefault` returns `None` | Delete it. SFT is selected by the *absence* of `rl:` / `reward_model:` / `pretraining_dataset:`. Add `strict: true` so this fails instead |
| A `max_seq_length` / `num_train_epochs` key accepted with no effect | HF/TRL spellings, not Axolotl's | `sequence_len`, `num_epochs` |
| `mpi4py` import error, or `DummyOptim`, on **one** GPU | A `deepspeed:` config with no distributed launcher | Remove the key for single-GPU runs |
| `exitcode: -9` on a multi-GPU job | **Host RAM** exhaustion, not VRAM | Reduce offload / `cpu_ram_efficient_loading`; a ZeRO-3 + offload job can want 100 GB+ of system RAM |
| FSDP/ZeRO-3 checkpoint will not load for serving | Sharded output is not directly servable | `axolotl merge-sharded-fsdp-weights <dir>`; a DeepSpeed ZeRO-3 checkpoint recombines via `zero_to_fp32` |
| Adapter merge fails / shape mismatch | `special_tokens` grew the embeddings asymmetrically | Merge with `axolotl merge-lora`, not a manual `PeftModel` merge; `--dequant` if the base was 4-bit |
| **No error at all**, but the model is worse | The quiet failures: wrong `chat_template`, wrong `type:`, stale `dataset_prepared_path`, `train_on_inputs: true` | `axolotl preprocess --debug` (§4.5 step 3). §8 and §9.4 of CS-17 are entirely made of these |

> **Correction:** the widely repeated claim that *"Axolotl rejects unknown keys loudly"* is
> **half true, and the half that is false is the half that bites.** Axolotl validates key
> *values* and illegal *combinations* through a pydantic schema at `load_cfg()` — that is the
> loud part, and it is genuinely better than LLaMA-Factory, which has no equivalent gate. But
> an unrecognised *key* is not rejected: it is warned about and dropped, exactly like the
> typo it probably is, because `DictDefault` returns `None` for anything it does not find
> (CS-17 §3, §4.3.0, §9.2). The only way to make an unknown key loud is `strict: true`. So:
> **set `strict: true`, and still run `preprocess --debug`** — the schema covers the config,
> and nothing covers the data.

> **Correction:** CS-17 §14.2 row 1 describes a fully-masked run as *"Loss flat at ~2.3
> (`ln(V)`)"*. **`ln(V)` is not 2.3.** `2.303` is `ln(10)`. For Qwen2.5's 151,936-token
> vocabulary, `ln(V) ≈ 11.93`; for a 32k-vocabulary Llama/GPT-NeoX tokenizer, `≈ 10.4`.
> CH-13 §2 and §8 state the same constant as `ln(vocab) ≈ 10.4`, which is the correct order
> of magnitude. The `~2.31` figure that also appears in CS-17 §15.1 is a *converged* loss for
> a degenerate run, not an initial one — so the diagnostic to teach is **"flat, and nowhere
> near ln(V)"**, and the number to compare against is your tokenizer's `math.log(vocab_size)`.
> (Cite: CS-17 §14.2 row 1, §15.1 vs CH-13 §2.)

> **Correction:** at [52:20]–[52:49] the instructor says *"don't set zero over here… you
> don't need to set the value of this dataloader number of worker"*. **`num_workers: 0` is
> perfectly legal** — it means "load in the main process", and it is the *only* option on
> Windows and in some sandboxed notebooks. What is illegal is zero workers **plus** an
> explicit `dataloader_prefetch_factor` (the `ValueError` above): fix it with
> `dataloader_num_workers: 2` **or** by deleting the prefetch key — not by "never writing
> zero". His explanation of *why* prefetching needs parallel loading is correct (CS-17 §4.3.5).

---

## 12. Copy-Paste Starter Config

`starter_sft.yaml` — **§5.1, with four edits and nothing else.** Copy it, and change only
these keys for run one:

```yaml
# ── the four edits to §5.1 ──────────────────────────────────────────────────
base_model: meta-llama/Llama-3.2-1B-Instruct  # was Llama-3.1-8B-Instruct
output_dir: ./out/sft-qwen-smoke               # your own path
excess_length_strategy: raise                  # NOT `drop`: see the row count before
                                               # you silently lose 12% of your data
num_epochs: 1                                  # the default; 2 is the second run, not the first

# ── and the three edits to §5.1 that make it survive on a 16 GB card ───────
adapter: qlora
load_in_4bit: true
bnb_4bit_compute_dtype: float16                # Turing/T4: bf16 is unsupported
attn_implementation: xformers                  # T4 again: FA2 needs Ampere+
bf16: false
fp16: true
```

> **Deliberately absent: `deepspeed:`, `fsdp_version:`, `wandb_project:`.** Each is correct
> for a *specific* topology and an error for a single-GPU debug run — a `deepspeed:` key with
> no distributed launcher causes the `mpi4py` / `DummyOptim` failures in §11, and a W&B key in
> CI turns a training job into an authentication hang.

**Run one.** Then, in this order: read the `preprocess --debug` output → confirm the
assistant span is labelled and the user span is `-100` → confirm the final EOS/EOT position
is **labelled** → confirm `steps × grad_accum` ≈ your row count → *then* consider touching a
hyperparameter.

### The pre-flight checklist

| # | Check | Command / how |
|---|---|---|
| 1 | YAML parses | `python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" starter_sft.yaml` |
| 2 | Schema valid, no unknown keys | Run once with `strict: true`; delete the file if it complains |
| 3 | No deprecated keys | The `grep -nE 'flash_attention\|dpo_beta\|training_dype…'` sweep in §6 |
| 4 | `type:` matches the data's real shape | Print one line of the JSONL and look at it |
| 5 | Template renders the model's own markers | `tokenizer.apply_chat_template(...)` vs the `preprocess --debug` printout |
| 6 | EOS/EOT position is **labelled** | Same printout — last labelled token must be the terminator, not `-100` |
| 7 | `eot_tokens` are single tokens | `len(tok(t, add_special_tokens=False)["input_ids"]) == 1` for each |
| 8 | Rows survive tokenisation | `steps × grad_accum` vs `wc -l` on the JSONL (§2) |
| 9 | p95 token length ≤ `sequence_len` | Compute it; truncation usually cuts the answer |
| 10 | VRAM fits | `python code/common/memory.py --model 7B --method qlora --seq-len 2048` |
| 11 | `val_set_size > 0` | Otherwise overfitting is invisible |
| 12 | `sample_packing` is legal for the backend | `attn_implementation` is `flash_attention_2`/`_3`/`_4`, or packing is `false` |
| 13 | The cache is fresh | `rm -rf last_run_prepared/` after any template or `type:` change |
| 14 | 20 steps, then look | `axolotl train starter_sft.yaml --max-steps 20` and read `sec/step` and grad norm |

> Check 5 is the one that saves the wasted run. Render the **base** model's template before
> training: if it already produces sensible, well-formatted turns, your template is right. If
> it rambles or refuses oddly, your template is wrong — and no amount of training fixes it.

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| The full treatment, with the video walkthrough and the five annotated repo configs | **CS-17 — Axolotl** (§4.3 the annotated config, §4.5 the `type:` zoo, §4.6 packing) |
| The other YAML-first trainer, and the `dataset_info.json` registry model | **CH-15 / CS-15 — LLaMA-Factory** (§4 key reference, §9 head-to-head, §11 error table) |
| The fast single-GPU path for one-off runs | **CS-16 — Unsloth** |
| What SFT is actually doing — masking, templates, epochs | **CH-13 / CS-13 — Instruction Fine-Tuning** (§4.2 masking, §8 symptom table) |
| Preference tuning from a YAML — DPO/ORPO/KTO/GRPO and what β means | **CH-14 / CS-14 — The Alignment Map** |
| LoRA/QLoRA maths — `r`, `alpha/r`, NF4, double quantisation | **CS-23 — LoRA & QLoRA** |
| Attention backends (FA2/FA3/sdpa/flex) and why packing needs varlen; ZeRO vs FSDP2 | **AP-03 — Attention Backends**; **AP-05 — Distributed Training**; CS-17 §4.3.4, §4.8.4 |
| Quantising and deploying the merged model; versioning and rolling back the adapter | **CH-10 / CH-11 — Quantization**; CS-17 §16; `code/15_serve_vllm.py` |
| The Python equivalent of every snippet above, with the traps pre-checked | `code/01_sft_lora.py`, `code/04_dpo.py`, `code/09_merge_and_export.py` |
| Practice being interviewed on this | **IQ-17 — Interview Questions: Axolotl** |

> **The four things worth remembering when you close this card.** The config is the artefact.
> The loud failures are the cheap ones — `strict: true` and `preprocess --debug` convert the
> quiet ones into loud ones. `sample_packing` is the biggest throughput win and the only
> silent correctness bug in the file. And an adapter without its base revision and chat
> template is not a model, it is an unreproducible delta.
