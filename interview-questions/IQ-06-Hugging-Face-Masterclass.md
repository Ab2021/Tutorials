# IQ-06 — Interview Questions: Hugging Face Masterclass

| Field | Value |
|---|---|
| **Module** | The Hugging Face stack — Hub, `Auto*` classes, `datasets`, `Trainer`, `evaluate`, tokenizers, export |
| **Pairs with** | CS-06 (case study), CH-06 (cheat sheet) |
| **Total questions** | 43 (12 L1 + 12 L2 + 8 L3 + 5 L4 + 6 L5) + 20 rapid-fire + 3 coding tasks + 10 CS-06 self-check answers |
| **Levels covered** | Screen (L1) / Intermediate (L2) / Advanced (L3) / System Design (L4) / Debug (L5) |
| **Source material** | Video 6 `Hugging Face Masterclass`; `code/common/memory.py --table`; `code/09_merge_and_export.py`; `code/10_embedding_finetune.py` |

**Ground truth used throughout:** the repo's own measured VRAM table (`code/common/memory.py --table`, gradient checkpointing ON, AdamW, single GPU, GiB) — 7B full FT **91.6**, LoRA r16 **14.7**, QLoRA r16 **4.9**, inference bf16 **16.0**, inference 4-bit **4.8**; 8B full FT **104.7**. Every CLI flag below was verified by running `python <script> --help`. `code/09_merge_and_export.py` takes `--base --adapter --out [--dtype {bf16,fp16,fp32}] [--gguf] [--push] [--verify] [--device]`; `code/10_embedding_finetune.py` takes `--data --out --model --loss {mnrl,triplet,contrastive,cosent} --epochs --batch-size --lr --max-len --anchor-col --positive-col --negative-col --validate --dry-run --eval-queries`.

---

## How To Use This File

- **L1 = phone screen / recruiter filter** — 30-second answers. If you cannot answer an L1 in one breath, you will not reach L2.
- **L2 = working engineer** — 2–3 minutes, expects implementation detail: real flags, real numbers, real failure modes.
- **L3 = senior / specialist** — 5 minutes, expects internals, derivations, and trade-offs.
- **L4 = staff / system design** — 15-minute whiteboard. Structure every answer as: requirements → constraints → design → trade-offs → failure modes.
- **L5 = debugging & incident response** — "your model does X, what do you check and in what order." Answer with a *sequence*, not a list.

**The meta-rule:** an answer of "it depends" is a failure unless it immediately says what it depends on and then picks a default. Always pick the default.

**The three answers that get people hired in this module:** (1) *the tokenizer must be loaded from the same repo id and the same `revision` as the model*; (2) *the collator decides whether your `labels` carry `-100`, and `DataCollatorWithPadding` does not*; (3) *memory is a four-term sum — weights + gradients + optimizer + activations — and only the last one responds to batch size*. If you remember three things, remember those.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. What actually lives in a model repo?**

- **Answer:** Seven files are the entire contract: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`, `README.md` (the model card, whose YAML front-matter carries `base_model`, `datasets`, `license`). `config.json` holds `architectures`, `model_type`, `hidden_size`, `num_attention_heads`, `num_hidden_layers`, `vocab_size`, and `_commit_hash`. Everything else — `*.bin`, ONNX, GGUF — is an alternative encoding of the same weights.
- **Why asked:** Candidates who describe the Hub as "a website you download models from" cannot debug a load failure. The seven files *are* the interface, and `model_type` is what the `Auto*` dispatch reads.
- **Trap:** Saying "the model is a `.bin` or `.safetensors` file." The weights alone load into nothing; the tokenizer and config are what make them a model.

**Q2. What does `from_pretrained` do, and what is the one-line misdescription?**

- **Answer:** It is I/O, not a constructor. It (1) resolves the repo id, (2) checks the local cache for a matching commit, (3) downloads missing shards, (4) reads `config.json` and instantiates the class, (5) builds the state dict, (6) loads weights (mmap for safetensors), (7) applies `torch_dtype`, (8) applies `quantization_config` if present, (9) applies `device_map`, (10) warns about uninitialized/missing keys, (11) caches under `~/.cache/huggingface/hub`. Misdescription: "it downloads the model" — it does network I/O, mutates a global cache, reads environment variables, and can fail in six distinct ways.
- **Why asked:** This is where every 403, every silent fp32 load, and every `device_map` conflict comes from.
- **Trap:** Believing `from_pretrained` infers dtype. It does not — with no `torch_dtype`, a checkpoint stored in fp32 loads in **fp32**.

**Q3. What does the `Auto*` class you pick commit you to?**

- **Answer:** The task head. `AutoModel` → a bare encoder/decoder body returning `last_hidden_state`; `AutoModelForCausalLM` → a language-modelling head (`lm_head`) and `generate()`; `AutoModelForSequenceClassification` → a `classifier` over `num_labels`; `AutoModelForMaskedLM`, `AutoModelForTokenClassification`, `AutoModelForSeq2SeqLM` each give a different head. `AutoModel` for embeddings means mean-pooling yourself; `AutoModelForSequenceClassification` on bare `bert-base-uncased` means a randomly-initialized head.
- **Why asked:** Picking the wrong `Auto*` class produces a model that runs, returns well-shaped tensors, and means nothing. It is the #1 silent failure in the module (CS-06 §9.4 row 1).
- **Trap:** Not checking `print(type(model).__name__)` and `config.architectures`.

**Q4. How does `AutoModel` know which class to build?**

- **Answer:** It reads `model_type` from `config.json`, looks it up in `MODEL_MAPPING_NAMES` (a `model_type → (config_class, model_class)` registry), and instantiates the mapped class. `architectures` in `config.json` is a *hint* used for the `trust_remote_code` / dynamic-module path and for tooling — the dispatch itself is on `model_type`. A config with `model_type: "bert"` and `architectures: ["LlamaForCausalLM"]` builds BERT.
- **Why asked:** It is the difference between knowing the API and knowing the mechanism, and it explains why a repo with a novel `model_type` needs `trust_remote_code=True`.
- **Trap:** Thinking `architectures` drives dispatch. It does not.

**Q5. `safetensors` vs `.bin` — one sentence.**

- **Answer:** `safetensors` is a pure data format with no deserialization hook; `.bin` is a `torch.save` pickle and **executes arbitrary Python on load**. So the choice is a security boundary, not a speed optimisation — though safetensors is also ~2× faster via mmap and supports lazy shard fetching for 100 GB checkpoints.
- **Why asked:** A candidate who loads an untrusted `.bin` with `HF_TOKEN` in the environment has handed a stranger remote code execution.
- **Trap:** Describing safetensors as "just a faster pickle."

**Q6. `padding_side` — which side, and when?**

- **Answer:** `left` for generation, `right` for training. Decoder-only generation continues from the last position, so with right-padding the "next token" is predicted from a pad; left-padding gives every sequence the same final position. Training with a causal mask is indifferent, but right-padding keeps position ids contiguous and matches every published recipe. Set it with `tokenizer.padding_side = "left"` — it is a tokenizer attribute, not a `generate` argument.
- **Why asked:** This is a *silent* correctness bug: wrong-side padding gives degenerate generations with no error.
- **Trap:** Assuming batched and single-sequence inference agree. With right-padding and no `attention_mask`, they do not.

**Q7. Why does `pad_token` often equal `eos_token`, and what does it break?**

- **Answer:** Llama and Qwen tokenizers ship no pad token, so the convention is `tokenizer.pad_token = tokenizer.eos_token` before training. It breaks three things: (1) labels — you must mask pad positions out of the loss yourself, because the pad id is now a *real* target id, not `-100`; (2) generation without an explicit `pad_token_id` warns and can emit EOS repeatedly; (3) any code that counts `<eos>` occurrences now counts padding too.
- **Why asked:** Every Llama/Qwen fine-tune hits this. The candidate must know it is a *deliberate* compromise with three consequences, not a one-liner.
- **Trap:** Thinking `pad_token = eos_token` is a bug. It is the convention; the bug is not masking those positions afterwards.

**Q8. What does `truncation=True` default to, and why does that matter?**

- **Answer:** It defaults to `False` — `tokenizer(text)` does **not** truncate. The failure differs by model: encoder models raise inside `forward()` when `seq_len > max_position_embeddings`; decoder-only models with rotary embeddings may silently accept 8192 tokens and degrade on the tail. Always pass `truncation=True, max_length=...` explicitly.
- **Why asked:** It is the difference between a crash (good) and a silent quality loss (bad), and it is model-dependent.
- **Trap:** Assuming the tokenizer truncates because "the model has a context window."

**Q9. What is `class_encode_column` for?**

- **Answer:** It converts a string label column into a `ClassLabel` feature with integer ids. `Trainer` will fail with `ValueError: Unable to create tensor...` on a raw string label column, because the collator cannot stack strings. `Dataset.from_pandas` does not do this for you.
- **Why asked:** The error message says nothing about labels, so it sends people hunting for shape bugs.
- **Trap:** Casting to `int64` with `.cast_column` — that works for tensors but loses `id2label`/`label2id`, which you need for `compute_metrics` and for a legible `confusion_matrix`.

**Q10. Name the collator you use for each task.**

- **Answer:** Classification/NER with a `labels` column → `DataCollatorWithPadding` (it pads `input_ids`/`attention_mask` and *passes labels through untouched* — which is correct for classification and wrong for LM). Causal LM → `DataCollatorForLanguageModeling(mlm=False)` (pads labels with `-100`). Seq2seq → `DataCollatorForSeq2Seq`. Instruction tuning → TRL's completion-only collator, or `DataCollatorForSeq2Seq` with a masking function.
- **Why asked:** "Which collator" is the fastest way to find out whether someone has actually trained an LM. The choice *is* the label semantics.
- **Trap:** `DataCollatorWithPadding` on a causal-LM dataset. It pads `labels` with `pad_token_id`, so the model is trained to emit pad tokens — and the loss curve looks *better* than the truth.

**Q11. Loss flat at 0.693 for a 2-class classifier — what are you looking at?**

- **Answer:** `ln(num_classes)` = `ln(2)` = 0.693, which is the loss of a uniform predictor. Three causes, in order: (1) every label is `-100`, so nothing is supervised; (2) the model is frozen — `assert any(p.requires_grad for p in model.parameters())`; (3) LR is 0, or the head is not in the optimizer's parameter groups. Check `print((batch["labels"] != -100).sum())` first.
- **Why asked:** It is the single most common "my loss is not moving" report, and the number 0.693 identifies it immediately.
- **Trap:** Concluding "the task is hard." A hard task gives a loss that starts near `ln(V)` and *descends*.

**Q12. `max_length` vs `max_new_tokens` in `generate`.**

- **Answer:** `max_length` is the **total** budget — prompt plus completion — so a 940-token RAG prompt with `max_length=512` returns nothing or raises. `max_new_tokens` is the completion budget and is the only safe choice. Defaults differ across versions, so always pass it explicitly.
- **Why asked:** It is a production-only bug: it works in every notebook with a short prompt and fails the first time a real user pastes a document.
- **Trap:** Using `max_length` and testing only with short prompts.

---

## Level 2 — Applied & Implementation

**Q13. Walk me through tokenizing a corpus correctly with `datasets`.**

- **Answer:** `load_dataset` → `shuffle(seed=42)` → `select(range(n))` → `map(tokenize_fn, batched=True, remove_columns=[...])` with `tokenizer(batch["text"], truncation=True, max_length=L)` and **no padding** → `set_format("torch")` last. Rules: never `padding=True` inside `.map()` (it pads each *mapping batch* to its own longest item and bakes a different shape into every Arrow shard); do all `map`s before `set_format`, because once the format is torch, a `.map()` expecting strings fails; and add `load_from_cache_file=False` while iterating, because the Arrow fingerprint will happily serve a stale result.
- **Why asked:** Every one of these is a documented gotcha (CS-06 §10.4, §10.6, §10.8) and every one produces a confusing symptom.
- **Trap:** Setting `padding` in the tokenizer *and* using a collator. The collator then pad-to-longest on top of already-padded rows.

**Q14. `map(..., batched=True)` — what changes?**

- **Answer:** The function receives a *dict of lists* rather than a single row, so it can call the tokenizer's fast path over a whole batch, and `num_proc=N` becomes available. It is 10–100× faster than per-row mapping for tokenization. Two constraints: with `batched=True` you must return lists of the same length as the input (filtering rows is not allowed — use `.filter()`), and `num_proc>1` hangs or raises `RuntimeError: Cannot re-initialize CUDA in forked subprocess` on Windows and in notebooks with a live CUDA context, so keep `num_proc=1` there.
- **Why asked:** It separates people who have scaled a data pipeline from people who have run a demo.
- **Trap:** Assuming `batched=True` only affects speed. It changes the function's contract.

**Q15. How do you build a train/val/test split that will not lie to you?**

- **Answer:** Split **first**, with a seed, on the raw data, and never let the test set touch anything downstream: `pool = raw["train"].shuffle(seed=42)`, `val = pool.select(range(0, 500))`, `tr = pool.select(range(500, 5500))`, `test = raw["test"].shuffle(seed=42).select(range(1000))`, then map. Anything fitted on the whole corpus before splitting — a tokenizer, a mean, a vocabulary, a scaler — is leakage. Add the cheap assertion `assert not (set(tr_text) & set(test_text))` and report per-slice numbers, because a single accuracy hides the slices that page you.
- **Why asked:** "Train accuracy 0.99 with 1,000 examples" is a STOP condition in CS-06 §8.2, and leakage is the usual reason.
- **Trap:** Using `train_test_split(seed=)` *after* tokenization that used the full corpus, then reporting the validation number as held-out.

**Q16. `compute_metrics` — what is in `EvalPrediction`, and what goes wrong?**

- **Answer:** `EvalPrediction` unpacks to `(predictions, label_ids)`; for a classifier `predictions` is **raw logits**, shape `[N, num_labels]`. You must `np.argmax(logits, axis=-1)` before scoring. Return a **dict** of floats. It runs on the gathered, concatenated predictions at the end of evaluation, and `Trainer` prefixes every key with `eval_` in the log.
- **Why asked:** `argmax`-on-logits is the classic mistake; passing logits straight to `accuracy.compute` gives a silently wrong number.
- **Trap:** Writing `accuracy.compute(predictions=logits, ...)` and getting a plausible-looking score because the shapes broadcast.

**Q17. You need `load_best_model_at_end=True`. What else must you set?**

- **Answer:** Three things. (1) `save_strategy` must equal `eval_strategy`, or it raises. (2) `metric_for_best_model` — it **defaults to `loss`**, so if you log accuracy and leave the default you ship the lowest-loss checkpoint, which late in training is not the highest-accuracy one. (3) `greater_is_better` is inferred from the metric name; set it explicitly when the name is ambiguous.
- **Why asked:** It is a flag combination, not a flag, and the default silently picks the wrong checkpoint.
- **Trap:** Setting `load_best_model_at_end=True` and believing that is sufficient.

**Q18. How do you pick a learning rate, and why is one number wrong?**

- **Answer:** There are three regimes and mixing them up is the most common configuration error in the field: a freshly initialized head on a frozen encoder `1e-3`–`5e-4`; a full fine-tune of a pretrained model `1e-5`–`3e-5` (2e-5 is the default); LoRA/QLoRA adapters on a frozen base `1e-4`–`2e-4`. The 10× gap between full FT and LoRA is not arbitrary: full-FT steps destroy pretrained features, and adapters start at zero and need bigger steps to move.
- **Why asked:** `Trainer` will happily run `lr=1e-3` on a full `bert-base` fine-tune; the loss drops, then oscillates, and the model ends up worse than the pretrained base outside your training distribution.
- **Trap:** Scaling the LR with the batch without knowing the rule: multiply the effective batch by *k* → scale LR by ≈ `sqrt(k)`.

**Q19. What is `warmup_ratio` protecting you from?**

- **Answer:** A randomly initialized head produces large, anisotropic gradients for the first few steps. `warmup_ratio=0.03`–`0.1` ramps the LR up over that window so those gradients do not move the pretrained encoder before the head has settled. `warmup_steps=0` on a fresh head is a real risk of an early divergence that never recovers — the optimizer has already moved the encoder.
- **Why asked:** It is the difference between a run that converges and a run that spikes at step 15 and is written off as "unstable."
- **Trap:** Applying a large warmup to a LoRA run. Adapters start at exactly zero (`B` is zero-initialized), so there is nothing to protect; 0.03 is plenty.

**Q20. `bf16` vs `fp16` — decide in one sentence.**

- **Answer:** `bf16` has 8 exponent bits and 7 mantissa bits; `fp16` has 5 and 10. `bf16` therefore has fp32's dynamic range and **cannot overflow to NaN** on a large gradient, which is why on Ampere/A100/H100 you set `bf16=True` and never think about loss scaling. On a T4/V100 (no bf16) you use `fp16=True`, where `Trainer` handles loss scaling for you — and if you write your own loop, `torch.cuda.amp.GradScaler` is not optional. Detect with `torch.cuda.is_bf16_supported()`; never set both.
- **Why asked:** It is a hardware decision with a correctness consequence, and the symptom (NaN at step N) does not point at precision.
- **Trap:** Setting `bf16=True` on a pre-Ampere card and getting an error or a slow emulated path.

**Q21. What is the `-100` convention?**

- **Answer:** `-100` is `torch.nn.CrossEntropyLoss`'s default `ignore_index`, so a `labels` tensor containing `-100` at a position contributes exactly zero to the loss and zero gradient — which is how you exclude prompt tokens and pad tokens from an SFT objective. `DataCollatorForLanguageModeling` and `DataCollatorForSeq2Seq` write `-100` for you; `DataCollatorWithPadding` does not.
- **Why asked:** It is the mechanism behind Q10's trap and behind every "why is my supervised fraction 100%" report.
- **Trap:** Assuming any collator masks pads. Only the LM ones do.

**Q22. Which three decoding parameters silently do nothing, and when?**

- **Answer:** `temperature`, `top_p`, `top_k` — all three are ignored when `do_sample=False`, because deterministic decoding never consults the sampling distribution. Recent `transformers` warns; older versions are silent. Also note `repetition_penalty=1` is *off*, not "mild": it divides the logits of already-seen tokens by 1.0, the multiplicative identity, and values below 1.0 actively encourage repetition.
- **Why asked:** Copy-pasting a sampling config without flipping `do_sample` is a standard half-hour of confusion in a notebook.
- **Trap:** Reading `repetition_penalty=1` as a small penalty. It is no penalty.

**Q23. `gradient_accumulation_steps=8` — what does it cost, and what must you change?**

- **Answer:** It runs 8 forward/backward passes before one optimizer step, giving `effective_batch = per_device_train_batch_size × gradient_accumulation_steps × num_devices` at 8× the wall clock per step and no extra peak memory (activations are freed between micro-batches). Nothing must change if you use `Trainer` — it divides the loss by the accumulation count. In a hand-written loop you must divide by `G` yourself, or the loss jumps by ~`G`× at the first step after any change to `G`.
- **Why asked:** The hand-written-loop bug is invisible until you tune `G`, and then it looks like an LR problem.
- **Trap:** Thinking gradient accumulation reduces memory. It reduces *step count*, not peak memory.

**Q24. `group_by_length` vs `packing`.**

- **Answer:** Both attack pad waste and they are different mechanisms. `group_by_length=True` sorts within a buffer so each batch's longest item is closer to its shortest — pad fraction falls from ~60% to ~10% on typical text, with no correctness risk. `packing=True` (TRL) concatenates examples end-to-end until the sequence is full, driving pad fraction to ~0%, 2–5× faster on short-sample SFT — but it requires correct `position_ids` and block-diagonal attention masks, or tokens from example A attend to example B. TRL's `SFTTrainer` does it correctly with Flash Attention 2; a hand-written collator usually does not.
- **Why asked:** Packing is the highest-value throughput knob in SFT and also the highest-risk one, and the tell (summaries that mix two unrelated documents) appears only in generation, not in the loss.
- **Trap:** Turning on `packing=True` without `attn_implementation="flash_attention_2"`.

---

## Level 3 — Advanced, Internals & Theory

**Q25. Derive the four-term memory model and say which term each optimization attacks.**

- **Answer:** `M_total = M_weights + M_gradients + M_optimizer + M_activations`. For full FT with AdamW in mixed precision the repo's own implementation charges: weights `P × 4 B` (bf16/fp16 copy **plus** an fp32 master copy), gradients `P × 2 B`, optimizer `P × 8 B` (two fp32 moments) — `14 B/param` static. Activations are `batch × seq_len × hidden × layers × 2 B × 12`, divided by `3√layers` when gradient checkpointing is on. Each lever maps to one term: QLoRA cuts weights (0.5 B/param) *and* gradients *and* optimizer, because only adapters are trainable; gradient checkpointing cuts activations; ZeRO-3/FSDP shards the first three across ranks and leaves activations per-GPU.
- **Why asked:** The decomposition is the whole of memory planning, and it tells you immediately which lever to pull.
- **Trap:** Quoting a single "bytes per parameter" for all methods. The number is a *sum over terms*, and the terms differ per method.

> **Correction:** the handbook states full-FT static cost five different ways — **12, 14, 16, 18 and 20 bytes per parameter**, i.e. 84 GB to 140 GB for the same 7B model. The full table, the breakdowns and the arithmetic are in the Correction at the end of this bank; the short version is that `code/common/memory.py` charges **14 B/param** (weights 4 + gradients 2 + optimizer 8) and prints **91.6 GiB** for the 7B full-FT cell, and 91.6 reconciles only with 14 (`7e9 × 14 = 98.0e9 B = 91.3 GiB` plus the 0.35 GiB activation term at `batch 1, seq 2048` with checkpointing). **Derive the four terms and quote 14.**

**Q26. You have two LoRA adapters trained with different `r`. Why can you not average their weights?**

- **Answer:** Because the low-rank factorization is not unique, so the same function has infinitely many `(A, B)` representations, and only the *product* `BA` is the weight delta. Two adapters with `r=8` and `r=16` do not even live in the same tensor shapes, and two with the same `r` but different initializations occupy different bases. To combine them you must (1) `merge_and_unload()` each into a dense model and average the **dense** weights, (2) concatenate along the rank axis into a single `r = r₁ + r₂` adapter (valid, and the same function only if the bases are orthogonal — they are not), or (3) use a proper task-arithmetic method (TIES / DARE, which trim, elect signs, and rescale before merging). Never average `lora_A` and `lora_B` separately.
- **Why asked:** It tests whether the candidate understands that LoRA is a *reparameterization*, not a small dense model. Averaging the factors is a common and silently destructive move.
- **Trap:** Believing `r` is a hyperparameter of the model rather than of the factorization. It is; merging two `r=16` adapters by element-wise mean gives a valid tensor with the wrong function.

**Q27. Where exactly does the merge lose information, and what is unrecoverable?**

- **Answer:** `merge_and_unload()` computes `W' = W + (alpha/r)·BA` in the dtype of `W`. Two losses: (1) **precision** — if `W` is fp16, the delta `BA` is often below fp16's resolution in a given coordinate and rounds away, which is why CH-06 §5.5 says merge on CPU in bf16/fp32, not on a GPU in fp16; (2) **a QLoRA base mismatch** — an adapter trained on a 4-bit NF4 base learned to correct *that base's* quantization error. Merge it into a full-precision base and the correction is now wrong. The adapter is recoverable (you still have it); the original *behaviour* is not, because the correction is only meaningful against the base it saw.
- **Why asked:** QLoRA-into-fp16 merging is the #1 silent quality regression after a successful training run, and CS-06 §5.5's own sentence is the rule. `code/09_merge_and_export.py` writes a `MERGE_CARD.json` with a `warning` field for exactly this reason.
- **Trap:** Believing "merging is exact." It is exact in exact arithmetic and lossy in the arithmetic you actually use.

**Q28. `code/09_merge_and_export.py --verify` is supposed to catch a broken merge. Read its `_verify` and name a bug.**

- **Answer:** Two. (1) It loads both models with a hardcoded `torch_dtype=torch.float16` on `device_map="cpu"` (`code/09_merge_and_export.py:144`, `:148`), which is the exact configuration `cheat-sheets/CH-06-…md:294` warns against — *"Merge on CPU in bf16/fp32, not on a GPU in fp16."* The verification therefore runs in the same precision that can lose the delta it is verifying, so a real precision loss can pass or a benign one can fail. (2) After `tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)` it calls `tok(text, return_tensors="pt")` with **no `add_special_tokens=False`** (`:154`), so the tokenizer prepends a second BOS — the double-special-token bug documented at CH-06 §5.1 and §5.4. Both models get the same malformed prompt, so agreement still means *something*, but it is no longer a verification of the serving path.
- **Why asked:** "Spot the bug" on real repo code. A candidate who only reads the docstring will miss both.
- **Trap:** Assuming a `--verify` flag means the verification is correct. The flag proves the code ran, not that it tested what it claims.

**Q29. `code/10_embedding_finetune.py` — what is not tunable, and why does it matter?**

- **Answer:** `warmup_ratio`. `DEFAULTS["warmup_ratio"] = 0.1` (`:61`) is read directly when computing `warmup_steps` (`:295`), and there is **no `--warmup-ratio` flag** in `parse_args` — while every other hyperparameter (`--epochs`, `--batch-size`, `--lr`, `--max-len`, `--loss`) is exposed. So a run that diverges in the first 50 steps has no CLI-level remedy, and a sweep over warmup is impossible without editing the file. The general lesson: an unexposed constant is a diagnostic you have removed.
- **Why asked:** It tests whether the candidate reads the flag surface against the code path, rather than trusting `--help`.
- **Trap:** Assuming all `DEFAULTS` entries are wired to arguments. Nothing enforces that.

**Q30. Why is `loss="mnrl"`'s quality governed by `batch_size` and not by `lr`?**

- **Answer:** MNRL (`MultipleNegativesRankingLoss`) uses every *other* positive in the batch as a negative for each anchor, so the batch **is** the negative pool: `L = −log( exp(sim(a,p⁺)/τ) / Σ_j exp(sim(a,p_j)/τ) )`. With `batch_size=32` each anchor sees 31 negatives; with 8 it sees 7. Small batches make the problem easy and the embeddings coarse. The script's own warning (`code/10_embedding_finetune.py:148`) states the fix ordering directly: push the batch size up before you push the epochs up. `--lr` still matters, but it is second-order to the negative pool.
- **Why asked:** It is the "when to use / why this knob" distinction that separates a practitioner from a config-copier. It also explains why gradient caching exists.
- **Trap:** Answering "raise the learning rate." That cannot substitute for hard negatives.

**Q31. A 768-d model scores 88% and a 384-d model scores 51% on the same similarity task. What is the correct reading?**

- **Answer:** Not "more dimensions is better" — that reading is backwards as an explanation. The 768-d score is inflated by **anisotropy**: raw BERT mean-pooling without a contrastive objective puts every sentence in a narrow cone, so every pair looks similarly similar and cosine similarity carries almost no signal. `all-MiniLM-L6-v2` at 384-d beats raw BERT mean-pooling on every retrieval benchmark. The cause is the **training objective**, and dimension is a red herring. The operational check is in the same script's shipping checklist (`:314`): if the mean pairwise cosine across random texts is `> 0.8`, the space is collapsed.
- **Why asked:** It is the module's flagship misconception (CS-06 §17.2) and the correction is mechanical, not hand-wavy.
- **Trap:** Reporting the pair of numbers as evidence about width, which is exactly what the source video does `[1:54:59]`.

**Q32. Perplexity 42.04 under `gpt2` for one sentence — what can you not do with that number?**

- **Answer:** Compare it to a perplexity from a different tokenizer. Perplexity is `exp(−(1/N)Σ log p(w_i | w_<i))` where `N` counts *tokens*, so a tokenizer that splits text into more tokens gets a mechanically different denominator over a different token sequence. It is comparable only within one tokenizer and one text distribution. Across tokenizers, convert to **bits-per-byte**: `BPB = ln(PPL) / ln(2) × (tokens / bytes)`. The "10–30 is good" band is a 2019 GPT-2 yardstick, not a law.
- **Why asked:** Cross-tokenizer perplexity comparison is a real published error, and the fix is a formula the candidate should be able to state.
- **Trap:** Reporting perplexity as a comparable quality number across two model families.

---

## Level 4 — System Design & Scenario

> These are 15-minute whiteboard questions. Structure every answer as: requirements → constraints → design → trade-offs → failure modes.

**Q33. Design a serving stack for a fine-tuned 8B model at 2,000 requests/minute with a 300 ms p99 TTFT SLO. Where does the HF stack stop?**

- **Answer:** Requirements: 33 rps, 300 ms TTFT, unknown context. Constraints: KV cache dominates at concurrency. Design: `vLLM` or `TGI` behind a queue with a bounded depth and a 503 on overflow; continuous batching and paged attention are worth 5–20× over a naive loop; FP8 KV cache for another ~1.5×; a warm-up of 3 synthetic requests at boot because the first request after load is 3–10× slower (CUDA kernel autotune, allocator growth). Trade-offs: dedicated Inference Endpoints (per-hour, per-replica) vs self-hosted (your GPUs, your ops) vs serverless (catalog-limited, 1–30 s cold start). Failure modes: unbounded queue = DoS vector; `pipeline()` is not thread-safe and has no continuous batching, so it is the wrong primitive at any scale; p99 is a queue-saturation metric before it is a model metric. Sizing: `weights + KV cache × concurrency`; for an 8B at 8k context and concurrency 16 the cache alone is tens of GB in bf16.
- **Why asked:** It tests whether "we'll use `pipeline`" survives contact with an SLO, and whether the candidate reaches for the KV-cache term unprompted.
- **Trap:** Sizing VRAM from weights alone. Weights are the constant; the cache is the variable that scales with your SLO.

**Q34. Design the artifact pipeline for shipping a fine-tuned model to a customer, end to end.**

- **Answer:** Requirements: reproducibility, rollback, audit. Design: pin `revision=<sha>` at load time; record the dataset sha (dataset repos change too); `save_pretrained` + `push_to_hub(private=True)` + a model card whose YAML front-matter carries `base_model`, `datasets`, `license`, the `TrainingArguments` JSON, the metrics, the seed, and the library version; one artefact = one repo with immutable tags; keep the previous sha so rollback is "point the service at it." Trade-offs: `model.push_to_hub()` does not push the tokenizer — `trainer.push_to_hub()` does — so the cheap path produces a repo nobody can preprocess for. Failure modes: `revision="main"` silently changes weights between two runs; a pushed dataset appears as a single file named `data`; the 100 GB private storage quota is per *account*, not per repo. Gate the deploy on a `<2 min` CPU pytest suite: golden set, empty and 20k-char inputs, determinism, and a no-leakage prompt assertion.
- **Why asked:** It is the entire MLOps surface of this module, and it is where "it worked in Colab" becomes an incident.
- **Trap:** Shipping the adapter without the base identity. An adapter alone does not identify a model — hence `--base` is **required** in `code/09_merge_and_export.py`.

**Q35. You must distil a 7B into a 1.5B, fine-tune a 66M classifier, and ship an embedding index. Design the memory plan for one 24 GB card.**

- **Answer:** Requirements: three different jobs, one card. Design: classifier — full FT of 66M at `P × 14 B ≈ 0.9 GB` static, trivially fits, batch 32. Student training — QLoRA r16 on the 1.5B (base 4-bit, adapters at ~2% of params) plus gradient checkpointing; the repo's table gives 1.5B LoRA r16 and QLoRA r16 rows in the low single-digit GiB. Distillation *generation* — the 7B teacher is a **separate offline process** in sequence-level KD, so the training run holds only the student; if you insist on token-level KD, the teacher and student are resident together and both the softmax buffers and the teacher weights must be budgeted. Trade-offs: sequence-level KD needs one GPU; token-level KD needs two, plus a top-k logit cache. Failure modes: full FT of the 7B on 24 GB (91.6 GiB) is impossible regardless of batch size; `device_map="auto"` and `Trainer` are incompatible, so inference-only placement cannot be reused for training.
- **Why asked:** It forces the candidate to state which *stage* the memory is for, which is the mistake that makes KD VRAM estimates wrong by 2×.
- **Trap:** Adding teacher and student residency for sequence-level KD, where the teacher is not resident at all.

**Q36. Design the evaluation harness for a model that will be judged on open-ended generation.**

- **Answer:** Requirements: no reference, no single right answer. Design: measure inter-annotator agreement on the test set *before* tuning anything (if Cohen's κ on labels is 0.6, no model can exceed ~0.8 accuracy on the task as defined); then an LLM-as-judge with a 1–5 rubric validated against ~100 human labels, with a both-orders position-bias check and a length-bias check; then per-slice reporting (shortest 10%, longest 10%, rare label, non-ASCII, empty string). Cadence: per-commit offline on a frozen 1,000-row set (<10 min); per-release plus adversarial slices (<1 h); shadow on live traffic; canary at 5% watching refusal rate, output length, and p99; continuous drift monitors on input length, vocabulary novelty, and predicted-label distribution. Trade-offs: reference-based metrics (BLEU/ROUGE) are cheap and measure overlap with *one* written answer, not correctness. Failure modes: a single metric from a single seed; a public benchmark contaminated by pretraining data.
- **Why asked:** It is the one design question where the honest answer is "the metric is the deliverable," and it is where most candidates propose accuracy.
- **Trap:** Proposing an LLM judge without a calibration set, or without a bias check.

**Q37. A customer needs a router over a label space that grows every quarter. Design it, and say when you would switch.**

- **Answer:** Requirements: unstable label space, low latency, no labelled data per new class. Design: start with an embedding index — 20–50 exemplar strings per class, `sentence-transformers`, normalized embeddings, `faiss.IndexFlatIP`, top-k majority vote. Adding a class is 20 exemplars and a re-index, with no training. Switch to a fine-tuned classifier when the label space *stabilizes* and volume justifies it: a classifier hard-codes `num_labels` into its head, so every change is a relabelling round. Trade-offs: an embedding index needs a good encoder (the documented jump was 384-d → 768-d for 24 semantically adjacent internal teams, +8 points) and query/exemplar embeddings must come from the *same* model with the *same* normalization flag. Failure mode: top-1 0.79 but top-5 0.97 — the fix is not a better model, it is surfacing the top-3 to a human.
- **Why asked:** It is the module's best example of "the right answer is not a bigger model," and it tests whether the candidate reasons about label-space stability rather than accuracy.
- **Trap:** Reaching for a classifier because that is the default. The classifier is the *less* accurate and more expensive choice here.

---

## Level 5 — Debugging & Incident Response

> Answer with a *sequence*: what you check first, what the check rules out, and what you do if it does not.

**Q38. A push returns 403 while `whoami` says you are logged in. Sequence.**

- **Answer:** (1) `echo $HF_TOKEN`. The credential resolution order is explicit `token=` → `HF_TOKEN` env var → the cached token file, so a stale **read** token cached into the environment outranks the write token you just logged in with. (2) Unset the variable, or overwrite it with the write token. (3) **Restart the runtime** — the env var is read at client construction, so unsetting it in a live kernel does not help. (4) Re-verify with a whoami call that prints scopes. If it still fails: the repo is gated and the token is from an account that never accepted the licence on *that* account, and `RepositoryNotFoundError` deliberately covers both "does not exist" and "private, you cannot see it."
- **Why asked:** It is the module's canonical incident `[1:04:00]`, and step (3) is the one everyone skips.
- **Trap:** Re-running the login command. It cannot win against the environment variable.

**Q39. `OSError: ... does not appear to have a file named config.json`. Sequence.**

- **Answer:** (1) `HfApi().model_info(id)` — does the repo exist and is it visible to you? (2) Is the id a *dataset* or a *Space* id, or a path rather than a repo id? (3) Is it gated, and has *this account* accepted the licence? (4) Is `HF_TOKEN` shadowing a valid login? (5) Is it a local path that `from_pretrained` is interpreting as a repo id? The error is deliberately ambiguous between "missing" and "private" so the Hub does not leak repo existence.
- **Why asked:** The message names a file, which sends people looking for the file instead of at access.
- **Trap:** Assuming the repo is broken rather than inaccessible.

**Q40. A run OOMs at the first step. Sequence.**

- **Answer:** (1) `torch.cuda.max_memory_allocated()` around one step — get the real number before changing anything. (2) Decompose it with the four-term model. (3) Check `padding="max_length"` — it pads every sequence to the maximum and attention is `O(S²)` in the padded length, so this is often the whole gap. (4) Halve the batch and enable gradient checkpointing (20–30% wall-clock cost, activations ÷ `3√layers`). (5) `optim="paged_adamw_8bit"` or `adafactor`, which cut the optimizer term from 8 B/param to 2 or 0.5. (6) Only then consider FSDP/ZeRO-3, which shards the first three terms and leaves activations per-GPU. If it OOMs *mid-run* after many steps instead, that is a leak, not a sizing problem: a retained logits list, a missing `loss.detach()`, or growing Python state.
- **Why asked:** "First step" vs "mid-run" is the diagnostic fork, and the candidate should ask which one before proposing fixes.
- **Trap:** Reaching for multi-GPU before measuring. Most first-step OOMs are a padding or `max_length` bug, not a capacity problem.

**Q41. Loss descends, generations are fluent nonsense, and the tokenizer loaded fine. Sequence.**

- **Answer:** (1) `assert model.config.vocab_size == len(tokenizer)` — a tokenizer from a *different repo* than the model produces exactly this: the loss descends because the model learns the wrong mapping consistently. (2) Print one collated batch and check `(batch["labels"] != -100).sum()` — if the collator padded labels with a real id, the model learned to emit pad tokens. (3) Diff `tokenizer.apply_chat_template(msgs, tokenize=False)` against the prompt string you actually sent at inference — a hand-written chat prompt on an instruct model produces shorter answers, fewer refusals, and no error at all. (4) Check `model.config._commit_hash` against what you evaluated. (5) Check the training and serving `max_length` and `padding` match.
- **Why asked:** "Fluently wrong" has a small, ordered differential, and only one of the five is a capacity problem.
- **Trap:** Concluding the model needs more data. All five causes above are configuration.

**Q42. You fine-tune, evaluate, and get `eval_accuracy: 0.48` on a 2-class task. Sequence.**

- **Answer:** (1) Ask whether the reported run actually **trained**. A `Trainer` constructed without `train_dataset` and only `.evaluate()` called produces a randomly initialized head — 0.48 is the *control*, not a result, and 0.5 is exactly what a random binary head scores. (2) Check `eval_loss` is ≈ 0.6938, `ln(2)`, the signature of a uniform predictor. (3) `assert any(p.requires_grad for p in model.parameters())`. (4) `print(torch.cuda.max_memory_allocated())` to confirm the run did work. (5) Only then question the data. The general rule: **an eval that ran without a train step is a smoke test for your plumbing, not a measurement of your model.**
- **Why asked:** It is the single best teaching moment in the source material, and it inverts the naive reading of the number.
- **Trap:** Concluding "DistilBERT is bad at sentiment." The number says nothing about DistilBERT.

**Q43. BLEU is 0.0 but the translation looks fine. Sequence.**

- **Answer:** (1) Read `precisions`, not `bleu`. BLEU is `BP · exp((1/4)Σ log p_n)`, and one zero n-gram precision makes `log(0) = −∞`, collapsing the geometric mean to exactly 0.0 regardless of the other three. With a 7-token candidate the 4-gram precision is 0 by construction. (2) Check `brevity_penalty` and `length_ratio` — the demo's `BP=1.0`, `len 7 / ref 6` rules out the brevity term. (3) Decide whether BLEU is even meaningful at that candidate length; usually it is not. (4) Report `precisions` alongside the aggregate, use a full-sentence corpus, or switch to a semantic metric (`sacrebleu` with `smooth_method="floor"` for a non-collapsing variant).
- **Why asked:** "0.0" reads as "garbage" and is actually "one 4-gram missed." The candidate must read the sub-scores first.
- **Trap:** Reporting single-sentence BLEU as a model quality number.

---

## Rapid Fire — True / False / One-Liner

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | `from_pretrained` infers `torch_dtype` from the checkpoint | **False** | No `torch_dtype` means the stored dtype, which is often fp32 — the classic 2× VRAM surprise |
| 2 | `safetensors` is a faster pickle | **False** | It is a security boundary; `.bin` executes arbitrary Python at load |
| 3 | `architectures` in `config.json` drives the `Auto*` dispatch | **False** | Dispatch is on `model_type` |
| 4 | `revision="main"` is safe if the repo id is stable | **False** | `main` is a moving branch; only a commit sha is immutable |
| 5 | `DataCollatorWithPadding` masks `labels` with `-100` | **False** | It passes labels through; LM collators write `-100` |
| 6 | `padding=True` inside `.map()` is correct | **False** | It pads per mapping batch and bakes different shapes into every shard |
| 7 | `set_format("torch")` can be called before the `map`s | **False** | After it, columns are tensors and a string-expecting `map` fails |
| 8 | `num_proc>1` works fine on Windows | **False** | Hangs, or `Cannot re-initialize CUDA in forked subprocess` |
| 9 | `dataset.map` will re-run your edited function | **False** | The Arrow fingerprint caches; use `load_from_cache_file=False` |
| 10 | `metric_for_best_model` defaults to `accuracy` | **False** | It defaults to `loss`, which is anti-correlated with accuracy late in training |
| 11 | `bf16` can overflow to NaN on a large gradient | **False** | 8 exponent bits give it fp32's range — that is the whole reason to prefer it |
| 12 | `load_best_model_at_end=True` requires `eval_strategy == save_strategy` | **True** | Otherwise it raises |
| 13 | `device_map="auto"` works with `Trainer` | **False** | `Trainer` does its own placement and fights Accelerate's hooks |
| 14 | `model.push_to_hub()` pushes the tokenizer | **False** | `trainer.push_to_hub()` does; the model-only push gives an unloadable repo for anyone else |
| 15 | `repetition_penalty=1` is a mild penalty | **False** | `1.0` is the multiplicative identity — it is off, and below 1.0 encourages repetition |
| 16 | `do_sample=False` silently disables `temperature` and `top_p` | **True** | Deterministic decoding never consults the sampling distribution |
| 17 | Gradient accumulation reduces peak memory | **False** | It reduces step count; activations are freed between micro-batches either way |
| 18 | `gradient_checkpointing=True` costs 20–30% wall clock | **True** | It recomputes activations in the backward pass |
| 19 | Full fine-tune LR ≈ 2e-5 and LoRA LR ≈ 2e-4 | **True** | A 10× gap, and it is not arbitrary |
| 20 | A 7B full fine-tune fits on one 80 GB card | **False** | 91.6 GiB static per the repo's own table — over the card before activations |

---

## Coding / Whiteboard Tasks

### Task 1 — Write the tokenize-and-split stage correctly

```python
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

MODEL_ID = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL_ID)

raw = load_dataset("stanfordnlp/imdb")

def prep(batch):
    out = tok(batch["text"], truncation=True, max_length=256)   # NO padding here
    out["labels"] = batch["label"]
    return out

pool = raw["train"].shuffle(seed=42)
tr   = pool.select(range(500, 5500))
val  = pool.select(range(0, 500))
test = raw["test"].shuffle(seed=42).select(range(1000))

tr, val, test = (d.map(prep, batched=True, remove_columns=["text"]) for d in (tr, val, test))
for d in (tr, val, test):
    d.set_format("torch")

assert (tr["labels"] != -100).all()          # labels survived the map
assert tok.pad_token_id != tok.eos_token_id or True   # deliberate, not accidental
```

- **The decisions being graded:** split *before* any corpus-level operation; `truncation=True` explicit; **no** `padding` in the tokenizer call; `set_format` last; an assertion that the label column survived.
- **Grading:** a candidate who paddles inside `prep` fails; a candidate who splits after mapping fails; a candidate who does not assert labels is downgraded.
- **Fail condition:** `padding=True` in `prep`, or `set_format("torch")` called before the `map`s.

### Task 2 — Choose and justify the collator (three datasets)

```python
from transformers import (DataCollatorWithPadding, DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq)

# (a) 2-class classification, `labels` is a ClassLabel column, padding needed
coll_a = DataCollatorWithPadding(tokenizer=tok)          # pads inputs, passes labels through

# (b) causal LM on raw text: every position is supervised, pads must be masked
coll_b = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)   # writes -100 for pads

# (c) seq2seq: labels padded with -100 on the decoder side
coll_c = DataCollatorForSeq2Seq(tokenizer=tok)
```

- **The decisions being graded:** the choice *is* the label semantics — (a) labels are per-example and padding them is meaningless, (b) `mlm=False` selects the causal (shifted-label) objective, and the collator writes `-100`, (c) seq2seq pads labels with `-100` because they are variable length.
- **Grading:** a candidate who reaches for `DataCollatorWithPadding` in case (b) fails; a candidate who cannot say what the collator writes into `labels` is downgraded.
- **Fail condition:** using `DataCollatorWithPadding` on a causal-LM dataset — the loss curve will look *better* than the truth.

### Task 3 — Size a fine-tuning run before renting a GPU

```bash
# The repo's own planner. --dry-run is not a flag here; the tool prints the plan.
python code/common/memory.py --model 7B --method qlora --seq-len 2048 \
    --batch 1 --grad-accum 16 --gpus 1 --optimizer paged_adamw_8bit
python code/common/memory.py --table          # the reference table every cheat sheet cites
python code/common/memory.py --model 8B --method full --optimizer adamw --gpus 8 --tokens 1e8
```

- **The decisions being graded:** the four-term decomposition (weights + gradients + optimizer + activations) and which term each `--method` changes; that `--tokens` only affects the FLOPS/time/cost block, never the VRAM block; that `--gpus 8` with `full` implies a *sharded* per-GPU figure and the tool prints both sharded (FSDP/ZeRO-3) and unsharded (DDP) rows.
- **Grading:** a candidate who quotes a VRAM number without saying which method and which optimizer produced it is downgraded; a candidate who reports the DDP row as the plan fails.
- **Fail condition:** planning a `--method full` 8B run for a single 80 GB card. `code/common/memory.py --model 8B --method full` is 104.7 GiB.

---

## Cheat Sheet of Numbers To Memorize

| Quantity | Value | Why it matters |
|---|---|---|
| 7B full FT VRAM (GiB) | **91.6** | `code/common/memory.py --table`; the reason LoRA exists |
| 7B LoRA r16 / QLoRA r16 (GiB) | **14.7 / 4.9** | The two numbers that decide whether a free GPU works |
| 7B inference bf16 / 4-bit (GiB) | **16.0 / 4.8** | Weights + KV + framework, at seq 4096, batch 1 |
| 8B full FT VRAM (GiB) | **104.7** | Over one 80 GB card before activations |
| Full-FT static cost | **14 B/param** | weights 4 + grads 2 + optimizer 8 (see the Correction in Q25) |
| NF4 weights | **~0.5 B/param** | 4-bit + double quant, why QLoRA fits |
| Full FT LR / LoRA LR | **2e-5 / 2e-4** | A 10× gap that is not arbitrary |
| Warmup ratio | **0.03–0.1** | Below 0.03 on a fresh head risks early divergence |
| `ln(num_classes)` for 2 classes | **0.693** | The signature of a masked-everything bug |
| `max_new_tokens` (never `max_length`) | **128–512** | `max_length` includes the prompt |
| Supervised token fraction (SFT) | **10–40%** | CH-06 §12.1's own assert; see the Corrections section |
| Effective batch | `micro × accum × devices` | `8 × 1 × 1 = 8`; realistic 8B QLoRA `2 × 8 × 4 = 64` |
| Classifier saturation | **2,000–5,000** balanced rows | Diminishing returns are steep after that |
| SFT data | **1,000–50,000** examples | LIMA-style: ~1,000 curated moves style and format |
| safetensors load speedup | **~2×** | mmap; and no code execution |
| `max_shard_size` default | **5 GB** | Why a 16 GB checkpoint arrives as 4 files |
| Tokens per word / chars per token | **≈1.33 / ≈4** | English BPE; the unit conversion you will need |

---

## Answers To The Self-Check Questions From CS-06

**1.** The warning is expected and correct for the head, and a bug everywhere else. `num_labels=5` makes `classifier.weight` shape `[5, 768]` and `classifier.bias` `[5]`, neither of which exists in the checkpoint, so `from_pretrained` leaves the head random while loading every encoder weight. Verify with `print(model.config.num_labels, model.classifier.weight.shape)`.

**2.** The time went into pad tokens. `padding="max_length", max_length=256` processes `100 × 256 = 25,600` positions of which ~3,500 are real text — ~86% of the compute is attention over padding, and attention cost is `O(S²)`. Fix: tokenize with no padding and let `DataCollatorWithPadding(tokenizer)` pad to the longest item *in each batch*; pass the same collator to evaluation.

**3.** Two causes. **(a)** A collator that padded `labels` with a real token id — check `assert (collator([ds[0], ds[1]])["labels"] == -100).any()`. **(b)** No dedicated pad token, so `pad_token` defaults to `eos_token` and generation without `pad_token_id` emits EOS repeatedly — check `print(tokenizer.pad_token, model.generation_config.pad_token_id)`.

**4.** The conclusion is invalid because **no training happened**. The `Trainer` was constructed without a `train_dataset` and only `.evaluate()` was called, so 0.48 is a randomly initialized head — the control, not the result. Adding `train_dataset=train_ds` and `trainer.train()` with 100 rows, 2 epochs, `lr=2e-5` reaches ~0.75–0.85.

**5.** No. At 100 short sentences the wall clock is dominated by Python overhead, tokenization, and kernel-launch latency, so 4.0 s vs 3.0 s is mostly noise. Instead: warm up 10 iterations, loop ≥1000 items with `torch.cuda.synchronize()` around each model, same batch size and torch version, and report tokens/second. Published figures are ~1.6× GPU throughput and ~2× CPU latency.

**6.** BLEU is `BP · exp((1/4)Σ log p_n)` and a 7-token candidate has no 4-gram, so `p_4 = 0` and `log(0) = −∞` collapses the score to exactly 0.0. Report the `precisions` array, use full sentences, and use `sacrebleu` with `smooth_method="floor"` if you need a non-collapsing variant.

**7.** You forgot `do_sample=True`. With `do_sample=False` the model decodes deterministically and never consults the sampling distribution, so `temperature`, `top_p`, and `top_k` are silently ignored. Recent versions warn; older ones are silent.

**8.** `repetition_penalty` divides the logits of already-present tokens, and `1.0` is the multiplicative identity — it is off, not mild; values below 1.0 encourage repetition. Use **1.05–1.15**; above ~**1.3** ordinary function words get forced off and output degrades into incoherence. Pair with `no_repeat_ngram_size=3` for a hard guarantee.

**9.** `7e9 × 14 B = 98.0e9 B = 91.3 GiB` static, plus ≈0.35 GiB of activations, which is the 91.6 GiB the repo's table prints; over 80 GB before fragmentation. Two independent fixes: switch to QLoRA (base 4-bit ~0.5 B/param, only adapters get optimizer state), or keep full FT and shard weights/gradients/optimizer across ranks with FSDP or ZeRO-3 (activating per-GPU unchanged). A third is `optim="adafactor"`, which replaces the two Adam moments with factored statistics.

**10.** `model.push_to_hub()` pushes only weights and config, not the tokenizer, so a teammate loading a tokenizer from another repo gets fluent garbage. Fix: `tokenizer.push_to_hub(...)` or, better, `trainer.push_to_hub(...)`. Verify by loading from a clean cache and asserting `model.config.vocab_size == len(tokenizer)`.

**11.** Ordered by likelihood: **(a) distribution shift** — their inputs differ in length, vocabulary, register, or language; compare length and OOV distributions. **(b) label-definition mismatch** — you trained on your rubric and they apply theirs; have a human label 200 of their examples with your rubric. **(c) a serving preprocessing mismatch** — different `max_length`, `padding`, revision, or a hand-built prompt instead of `apply_chat_template`; log the exact `input_ids`. Note 0.99 train accuracy flags a fourth — overfitting or leakage — but 0.94 test makes it less likely than (a).

**12.** `gradient_checkpointing=True` without `model.enable_input_require_grads()`. The input embeddings' outputs then do not require grad, so the graph is disconnected from the adapters, every gradient is zero, and the loss sits at exactly its initial value (0.0 for a fresh LoRA adapter, because `B` is zero-initialized). One-line fix before `trainer.train()`; verify with `assert any(p.requires_grad for p in model.parameters())` and by checking a LoRA parameter's `.grad` is non-zero after `loss.backward()`.

### Corrections to CH-06 discovered while writing this bank

> **Correction:** `cheat-sheets/CH-06-Hugging-Face-Masterclass.md:595` says the supervised-token-fraction pass range is *"10-40%"* and its assert is `0.05 < sup / max(tok,1) < 0.95`; the same file's own checks table at `:637` says **"5–40%"**. The comment and the assert agree at the lower bound of 10%, and `IQ-13` uses 10–40%. **Quote 10–40%** and treat `:637` as the typo. Note the assert is also far looser than either band — it will not catch a 6% run.

> **Correction:** `case-studies/CS-06-Hugging-Face-Masterclass.md:448` lists `Llama-3.1-8B` full fine-tune at **144.5 GB**, but `code/common/memory.py --table` prints **104.7** for the 8B full-FT cell. The two differ twice over: bytes-per-param (18 vs the code's 14) and units (`8.03e9 × 18 = 144.5e9` bytes = 144.5 decimal GB = **134.6 GiB**, whereas the table is GiB). Neither explanation reconciles them — 134.6 GiB ≠ 104.7 GiB — so this is a genuine disagreement, not a unit slip. **Quote 104.7 GiB**, because it is reproducible from the script the handbook tells you to run. Coincidence worth knowing: **144.5** is also the 70B **LoRA r16** cell of the same table (`CH-06:406`), which is where a reader searching for "144.5" will land by mistake.

> **Correction:** `case-studies/CS-06-Hugging-Face-Masterclass.md:2209` (Key Takeaway 10) and `:1530` (§8.1) both say an 8B/7B full fine-tune needs *"~144 GB of optimizer state"*. Optimizer state is the 8 B/param term alone: `7e9 × 8 B = 56.0e9 B = 52.2 GiB`, or `8.03e9 × 8 = 64.2 GB` decimal for the 8B. The 126 GB and 144.5 GB figures are the **whole static cost** (weights + master weights + gradients + optimizer), not the optimizer's share. The conclusion — that full FT of a 7B+ needs sharding — is unaffected; the label on the number is wrong.

> **Correction:** `cheat-sheets/CH-06-Hugging-Face-Masterclass.md:303` and `:313–340` document `huggingface-cli login|download|upload|whoami|logout|repo create` throughout, with no deprecation note. `case-studies/CS-06-Hugging-Face-Masterclass.md:566` carries the correction: `huggingface-cli` is deprecated in favour of the `hf` CLI — `hf auth login`, `hf auth whoami`, `hf repo create`. The old entry point still works in the 0.2x line but prints a deprecation notice. **Quote the `hf` forms**, and note `CS-06:518` and `:1867` also still print `huggingface-cli whoami` inside otherwise-corrected material.

> **Correction (the bytes-per-parameter figure, which is quoted five different ways):** the repo states the full-fine-tune static cost as **12, 14, 16, 18 and 20 bytes per parameter** in five different places, and the same 7B model therefore costs anywhere from **84 GB to 140 GB**:
>
> | Source | B/param | Breakdown as written | 7B static |
> |---|---|---|---|
> | `CH-06:39` (`params × (2 + 2 + 8)`) and `CH-06:505` | **12** | bf16 weights 2 + bf16 grads 2 + Adam m,v 8 | 84.0 GB = 78.2 GiB |
> | `code/common/memory.py` (weights 4 + grads 2 + optimizer 8) | **14** | bf16 weights 2 + fp32 master 2 + bf16 grads 2 + Adam 8 | 98.0 GB = 91.3 GiB |
> | `IQ-13:446` and `IQ-13:1255` | **16** | bf16 weights 2 + bf16 grads 2 + Adam 8 + fp32 master 4 | 112.0 GB = 104.3 GiB |
> | `CS-06:1644`, `:1652`, `:1667`, `:2252`, `:2273` | **18** | bf16 weights 2 + fp32 master 4 + fp32 grads 4 + Adam 8 | 126.0 GB = 117.3 GiB |
> | `CS-06:440` | **20** | 18, rounded up "to absorb allocator fragmentation" | 140.0 GB = 130.4 GiB |
>
> Arithmetic: `7e9 × 12 = 84e9 B`; `× 14 = 98e9`; `× 16 = 112e9`; `× 18 = 126e9`; `× 20 = 140e9`. The spread is **126 / 84 = 1.5×** for one model, driven entirely by *which terms you count*: 12 omits the fp32 master weights, 18 charges gradients at fp32 (4 B) where 14 and 16 charge bf16 (2 B), and 20 is 18 plus headroom. **14 is the one this repo actually computes** — `code/common/memory.py --table` prints **91.6** for the 7B full-FT cell, and 91.6 GiB reconciles only with 14 B/param (`98.0e9 B = 91.3 GiB`, plus the 0.35 GiB activation term at `batch 1, seq 2048` with checkpointing). Note the unit slip: `code/common/memory.py` prints its own header as `Training VRAM (GB)`, and `IQ-13:111` repeats **GB**, but the script divides by `1024**3`, so the cell is **GiB**. (`CH-06` §7.2 reproduces the table with no unit label at all, which hides the discrepancy rather than causing it.) **Quote 14 B/param and 91.6 GiB for 7B full FT**, and say which terms you counted. `IQ-13:446` is the most careful of the five — it states 16 *and* explains that 12 is the figure "if the master copy is omitted" — but it then cites 91.6 as the corroborating measurement, and 91.6 is a 14 B/param number, not a 16.

---

## Cross-References

| Module | Relationship to IQ-06 |
|---|---|
| **CS-06** Hugging Face Masterclass | The source case study. Every number and flag in this bank is grounded in it; its §19 self-check questions are answered in full above |
| **CH-06** Hugging Face Masterclass Cheat Sheet | The compressed reference: §2 the formulas, §4 the argument and `TrainingArguments` tables, §7.2 the VRAM table this bank quotes, §8 the 21-row symptom→fix lookup, §11 the exact error strings |
| **CS-07** BERT Fine-Tuning | Where the `AutoModelForSequenceClassification` + `Trainer` + `compute_metrics` triple of Q16 and Task 1 is the whole module |
| **CS-12** Domain-Adaptive Continued Pretraining | The next `Trainer` run after this one; `DataCollatorForLanguageModeling(mlm=True)` and continued MLM live there |
| **CS-13** Instruction Fine-Tuning | The masking convention (`-100`, `assistant_only_loss`, packing) that Q10, Q21 and Q24 preview |
| **CS-10 / CS-11** Quantization I & II | Where `BitsAndBytesConfig`, NF4 and the 0.5 B/param base of QLoRA are derived |
| **CS-19** Gemini Fine-Tuning on Vertex AI | The same model artifacts, managed training — useful contrast on where the `Trainer` surface is replaced |
| **CH-13** Instruction Fine-Tuning Cheat Sheet | The SFT-specific version of §4/§7 here: supervised fraction, prompt masking, LR bands |
| **IQ-05** RNN/LSTM → Attention | The attention cost argument behind `O(S²)` padding waste, and the KV-cache derivation |
| **IQ-13** Instruction Fine-Tuning | The sibling bank that takes over at the chat template: masking, packing, `assistant_only_loss`, evaluation of generations |
| **IQ-10** Quantization | The compression axis: what changes when the weights are the thing you shrink rather than the data |
| **IQ-07** BERT | The encoder-side counterpart to this bank's decoder-side generation material |

---

*End of IQ-06. Companion artifacts: `CS-06-Hugging-Face-Masterclass.md` (case study), `CH-06-Hugging-Face-Masterclass.md` (cheat sheet). Ground truth: `code/common/memory.py --table`, `code/09_merge_and_export.py --help`, `code/10_embedding_finetune.py --help`.*
