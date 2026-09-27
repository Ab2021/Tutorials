# IQ-17 — Interview Questions: Axolotl (YAML-Driven Training at Scale)

| Field | Value |
|---|---|
| **Module** | Frameworks / Tooling — the config-as-artefact branch (the declarative trainer) |
| **Pairs with** | **CS-17** (case study), **CH-17** (cheat sheet) |
| **Total questions** | **47** (12 L1 + 14 L2 + 9 L3 + 4 L4 + 8 L5) + 22 rapid-fire + 4 whiteboard tasks + 10 CS-17 §19 self-check answers |
| **Levels covered** | L1 screening → L5 incident response |
| **Source material** | CS-17 §0–§20 + Appendix A/B, CH-17 §1–§13, the YAML configs quoted inline in CS-17 §5.5 and §6, `code/common/memory.py --table` (flag verified by running `--help`). **Note:** CS-17's colophon names a `code/axolotl/` directory and §5.5/§6 name an `axolotal-config/` directory; **neither exists in this repo** — the configs are quoted inline in the case study, which is where this bank cites them from |
| **Also contains** | Rapid-fire true/false table, 4 coding/whiteboard tasks with grading notes, numbers to memorize, the ten CS-17 §19 self-check answers |
| **Time to work through** | ~5 h at interview pace; ~60 min for a revision pass (L1 + L2 + rapid fire + numbers) |
| **Differentiate from** | **IQ-15** (LLaMA-Factory). CH-17 §9.2 is the key-by-key map between the two; this bank assumes you have read it and does not re-ask IQ-15's questions |

**Ground truth used throughout:** the three identities CS-17 §1.4 reconciles its demo with — `steps_per_epoch = rows_surviving ÷ (micro_batch_size × grad_accum)`, `rows_surviving = steps × grad_accum`, and `tokens_trained = total_steps × micro_batch_size × grad_accum × sequence_len`. VRAM figures are `code/common/memory.py --table` (7B: full FT **91.6 GB**, LoRA r16 **14.7 GB**, QLoRA r16 **4.9 GB**), which prices tensors only; CS-17 §11.4 quotes *observed* peaks and is ~1.7–2× higher on purpose. The packing failure is Axolotl issues **#3453** and **#3608** (CS-17 §4.6.4, Appendix B.2).

---

## How To Use This File

- **L1 — Fundamentals & Vocabulary (12).** Can you say what the tool *is* and name the artefact it produces? A candidate who calls Axolotl "a training library" has not read past the README.
- **L2 — Applied & Implementation (14).** Real keys, real `type:` values, real error strings, and the `preprocess --debug` workflow. This is the level that separates "I ran `axolotl train`" from "I have shipped a config a reviewer signed off on."
- **L3 — Advanced, Internals & Theory (11).** The step-count/LR interaction under packing, the four-part reproducibility checklist, the VRAM derivation, the ZeRO/FSDP topology arithmetic.
- **L4 — System Design & Scenario (4).** Full prompts: requirements → constraints → design → trade-offs → failure modes. 10–15 minutes each, out loud.
- **L5 — Debugging & Incident Response (8).** A symptom and a clock. Answer with a *sequence*, not a list, and say which of the look-alikes you are ruling out at each step.

**The meta-rule:** an answer of "it depends" is a failure unless it immediately says what it depends on and then picks a default. Always pick the default.

**The three answers that get people hired in this module:** (1) *Axolotl is loud on the config and quiet on the data* — the pydantic schema at `load_cfg()` catches key values and illegal combinations before the model loads, and nothing catches a wrong `type:` or a wrong template; (2) *packing correctness is a property of architecture × attention backend × Axolotl version, not of your config* (CS-17 §4.6.4); (3) *a YAML file is necessary but not sufficient for reproducibility* — four more things must be pinned (CS-17 §4.1.4). If you remember three things, remember those.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. In one sentence, what is Axolotl?**

- **Answer:** A **configuration-driven training engine** — a declarative front end over `transformers`, `peft`, `trl`, `datasets`, `accelerate` and `bitsandbytes` in which one YAML file describes data loading, tokenisation, quantisation, training and evaluation, and the CLI (`axolotl train cfg.yml`) turns it into gradients. The instructor's own one-liner is the best one available, and CS-17's Appendix A.1 endorses it as "the best one-line definition in the video": *"Axolotl is a configuration-driven training engine"* [10:09]. The artefact the tool exists to produce is not just a checkpoint — it is **a YAML file a reviewer can read and a machine can re-run** (CS-17 §4.1.2).
- **Why asked:** It is the module's load-bearing fact and it predicts both halves of the tool's behaviour: the superpower (everything is declarative and diffable) and the weakness (the declarative surface moves fast, so tutorial keys go stale).
- **Trap:** Calling it "a new training framework" or "a model." It is a thin layer over the Hugging Face stack, and CS-17 §4.1.3 lists five real downsides to the convenience.

**Q2. What are the two ways to run Axolotl, and which is the production path?**

- **Answer:** **Path A — CLI + YAML** (CS-17 §4.2.1): `axolotl train sft_test.yaml`, with overrides like `--learning-rate 1e-4` accepted after the positional config path. **Path B — the Python API** (CS-17 §4.2.2), five imports and four calls: `DictDefault(...)` → `load_cfg(config)` → `load_datasets(cfg=cfg)` → `train(cfg=cfg, dataset_meta=dataset_meta)`. Path A is production because the config is a committed artefact; Path B exists for notebook spikes and debugger access. The rule: **if you use Path B, still write the YAML out** — a notebook run with no config file is an experiment you cannot re-run.
- **Why asked:** It tests whether the candidate has used the tool or only watched someone use it. The `DictDefault`/`load_cfg` pair is the API's actual entry point and appears in the tool's own error messages.
- **Trap:** Assuming Path B is "the same thing without the YAML." It skips the schema gate you are about to be asked about, so a Path B run can carry keys Path A would have rejected.

**Q3. Name the two keys everyone gets wrong because they use the Hugging Face spelling.**

- **Answer:** `sequence_len` (not `max_seq_length`) and `num_epochs` (not `num_train_epochs`). `sequence_len`'s default is only **512** (CS-17 §4.3.7) and `num_epochs`'s is **1.0** (CS-17 §4.3.6). With `strict: true` the HF spelling fails the config; without it, the key is silently ignored and you get the default — which is how people end up training at 512 tokens while believing they set 2048.
- **Why asked:** It is the cheapest possible test of whether someone reads resolved configs or assumes them. The failure is invisible in every log line.
- **Trap:** "It would have warned me." Only the deprecated-boolean path warns; a never-existed key is not a deprecation, it is a `DictDefault` miss.

**Q4. What happens to a row that is longer than `sequence_len`, by default?**

- **Answer:** It is **dropped**, not truncated — `excess_length_strategy` defaults to `drop` (CS-17 §10, §4.3.7). Nothing prints how many rows were lost. CS-17 §1.4 reconciles the demo's numbers only under this rule: a 10,000-row file became **8,840 rows** (~12% silently deleted). Use `raise` while developing, and always check `steps_per_epoch × gradient_accumulation_steps` against the file's row count.
- **Why asked:** It is the single most useful diagnostic in the tool and almost nobody runs it. It also explains why "the run finished early" is never a mystery if you counted first.
- **Trap:** Reading the loss curve for evidence. CS-17 §15.2's second case is worse than dropping: rows were *truncated mid-JSON* and the loss looked **healthier** with the long rows in, because truncated JSON is easier to predict than complete JSON.

**Q5. What is `val_set_size`'s default, and what does that mean operationally?**

- **Answer:** **0.0** — there is no eval by default, so no eval loss ever appears and there is no overfitting signal at all (CS-17 §9.4 row 13). Set `val_set_size: 0.05` plus `eval_steps: 50`. The two counterpoints worth knowing: it is **mutually exclusive with `test_datasets:`** (CS-17 §10 item 8), and `eval_sample_packing` inherits `sample_packing`, which changes the loss denominator — an unpacked eval is the one that stays comparable across runs.
- **Why asked:** It is the same failure as hosted fine-tuning without a `validation_file`, arriving through a different door, and it is the reason so many "it worked" reports are unfalsifiable.
- **Trap:** Setting `val_set_size` and leaving `eval_sample_packing` inherited. Then your eval loss moves when you change the packing setting, and you attribute it to the model.

**Q6. What does `train_on_inputs: false` do, and what is the alternative expression of the same idea?**

- **Answer:** It excludes the prompt from the loss — the standard prompt-masking policy, and Axolotl's default (CS-17 §4.7.3). The alternative expression is `roles_to_train: ["assistant"]` (default), the dataset-level equivalent. Two companion keys matter: `train_on_eos: turn` (options `turn` / `last` / `all`) decides whether the turn terminator is learnt, and per-message `train:`/`weight:` flags let you override at the message level. The boundary rule catches people out: a token straddling two parts with different flags is **conservatively masked**, so whitespace between the prompt and the answer can eat the first token of the answer.
- **Why asked:** Masking is the module's central implementation decision, and the boundary rule is the part that only shows up if you have read a token dump.
- **Trap:** "Prompt masking is always right." CS-17 §10 item 2 names the legitimate exception — completion-style continuation — and IQ-13's four exceptions apply here unchanged.

**Q7. What is `sample_packing`, in one sentence, and what does it buy?**

- **Answer:** It concatenates several short sequences into one `sequence_len`-sized row so no compute is spent on pad tokens. Measured effect: padding efficiency goes from **~35%** (naive `mean ÷ batch_max` ≈ 300/850 in CS-17 §4.6.1) to **~99%**, and the multiplier is `sequence_len ÷ mean_row_tokens` — which is why Axolotl's own comment says **"2–6× increase in tokens per micro-batch"** rather than a flat 6×.
- **Why asked:** It is the biggest single throughput win in SFT and the biggest single silent-correctness risk. A candidate who cannot state both halves has only read the marketing.
- **Trap:** Treating it as free speed. CS-17 §4.6.4 is an entire subsection on the bug it hides, and §17.7 lists "packing is not free speed" as one of fourteen expensive misconceptions.

**Q8. What does packing require that is not a YAML key?**

- **Answer:** A **varlen attention backend** — `flash_attention_2`/`_3`/`_4`. Axolotl computes `attn_supports_packing` from the chosen `attn_implementation` and gates the multipack patches on it; **you cannot override it from YAML** (CS-17 §4.6.2). With `eager` or `sdpa` there is no varlen kernel, so packing is only possible at "lower packing efficiency" via a 4-D mask — and `sample_packing: true` with `sdpa` is not a slow-but-correct configuration, it is the configuration where the correctness question is live.
- **Why asked:** It is the one place where a *derived* setting silently overrules a written one, which is exactly the class of bug this module is about.
- **Trap:** Setting `sample_packing: true` and `attn_implementation: sdpa` because the T4 has no FA2 path. On a T4, the correct answer is `sample_packing: false` (CS-17 §15.2 does exactly this).

**Q9. `adapter: lora` vs `adapter: qlora` — what changes, and what is the one extra key QLoRA requires?**

- **Answer:** QLoRA quantises the **frozen base** to 4-bit NF4 and requires `load_in_4bit: true`; omitting it is a pydantic validation error at `load_cfg()`, before the model loads (CS-17 §4.3.2 — the tool's loudest and most useful gate). The governing rule is a chain, not a preference: **quantised base ⇒ frozen base ⇒ adapter-only training.** You cannot QLoRA today and full-fine-tune the same run tomorrow. Third option worth naming: `adapter: loftq`. `bnb_4bit_quant_type` defaults to `nf4`; `fp4` is measurably worse for LLM weights.
- **Why asked:** It tests whether the candidate knows *why* the combination is illegal rather than memorising that it is.
- **Trap:** "QLoRA is a quantisation method." It is a training method that uses quantisation as one component — IQ-13 Q14 makes the same correction.

**Q10. What are the two commands you run before every long job?**

- **Answer:** First, `python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" config.yml` — 50 ms, no imports, no GPU, catches a tab character or an unquoted `key:value`. Second, `axolotl preprocess config.yml --debug --debug-num-examples 3` — loads the config, validates it, prints the resolved result **and the actual `input_ids`/`labels`**. `--debug-num-examples N` bounds the output. This pair is CH-17 §4.5's "three-step loop"; the third step is reading the token dump, which no schema can do for you.
- **Why asked:** It is the workflow habit that distinguishes a practitioner from a tutorial-follower, and CH-17 §4.5 frames it as Axolotl's genuine workflow win over LLaMA-Factory.
- **Trap:** Running `preprocess` on a `pretraining_dataset:` or `skip_prepare_dataset: true` config — those are prepared on demand and raise `KeyError: 'input_ids'`.

**Q11. What is the DPO dataset `type:` today, and what are the two dead spellings?**

- **Answer:** `chat_template.default`, with `field_messages`, `field_chosen` and `field_rejected` (CS-17 §4.5.2). `type: dpo` and `type: preference` are gone from the docs — CS-17 §15.4's first attempt failed on exactly this. Related canned types you should be able to name: `chatml.ultra`, `chatml.intel` (which wants `{question, chosen, rejected}`), `chatml.argilla`, `llama3.ultra`, and the escape hatch `user_defined.default` with `field_prompt`/`field_chosen`/`field_rejected` plus the three `*_format` keys.
- **Why asked:** It is the concrete form of the module's "tutorial keys go stale" thesis, and it is the failure that opened a real case study.
- **Trap:** Assuming the old spelling fails loudly. It does not necessarily — you may simply get a type that does not select what you meant.

**Q12. Which key replaced `dpo_beta`, and how do you select IPO?**

- **Answer:** `rl_beta` replaced `dpo_beta` (CS-17 §4.8.2). `rl:` selects the objective and takes `dpo | orpo | kto | simpo | grpo | gdpo | ebft`; **IPO is not one of them** — it is `rl: dpo` plus `dpo_loss_type: [ipo]` (CS-17 §4.8.3). Note also that `sample_packing` should be **false** for preference training: packing across chosen/rejected pairs changes what the implicit reward compares, and CS-17 §15.4 records that as a silent degradation.
- **Why asked:** It is a two-part question where the second part is the one that exposes memorisation. Candidates who know `rl_beta` frequently still answer "`rl: ipo`".
- **Trap:** Carrying the SFT learning rate into DPO. CS-17 §15.4's config uses **5e-6** — 10–40× below the SFT LR, because the two runs are doing different things to the same weights.

---

## Level 2 — Applied & Implementation

**Q13. What does `DictDefault` do, and what is Axolotl's one quiet config failure?**

- **Answer:** `DictDefault` returns `None` for any key it does not find, which is why an unrecognised key is **warned about and dropped rather than rejected**. That is the quiet failure: `training_type: sft` is not an Axolotl key, so it is accepted with no effect; `max_seq_length: 2048` is accepted with no effect and you get 512. The fix is `strict: true` in the YAML — which is `false` by default (CS-17 §4.3, §9.2). CH-17 §11 states the resolution precisely: Axolotl is loud on key **values** and illegal **combinations** (pydantic at `load_cfg`), quiet on unknown **keys**, and completely silent on the **dataset contract**.
- **Why asked:** The widely repeated claim "Axolotl rejects unknown keys loudly" is half true, and the false half is the half that bites. This question tests whether the candidate is repeating the claim or has tested it.
- **Trap:** Answering "it depends on the schema" and stopping. The schema does not check what you typed if the key never existed.

**Q14. Spot the bug: `adapter: qlora`, `load_in_4bit: false`, `micro_batch_size: "2"`, `sequence_len: 512`.**

- **Answer:** Four bugs, three loud and one silent. `adapter: qlora` without `load_in_4bit: true` → **pydantic validation error at `load_cfg()`**, loud. `micro_batch_size: "2"` → a string where an int is required, so also loud at the schema. `sequence_len: 512` is legal but is the **default**, so it is almost certainly a silently-accepted `max_seq_length` in disguise. And the fourth, which nothing checks: whether 512 is long enough for the data — CS-17 §15.2 lost 6% of rows to truncation at 1024.
- **Why asked:** Config review is the actual job. The interviewer is watching whether you distinguish "fails the schema" from "fails the data", because those need different remedies.
- **Trap:** Fixing `load_in_4bit` and declaring the review done. The schema covers the config; **nothing covers the data** (CH-17 §4.5).

**Q15. Translate six Axolotl keys into their LLaMA-Factory names.**

- **Answer:** From CH-17 §9.2: `sequence_len` → `cutoff_len`; `num_epochs` → `num_train_epochs`; `micro_batch_size` → `per_device_train_batch_size`; `lora_r` → `lora_rank`; `sample_packing` → `packing`; `rl_beta` → `pref_beta`. Two more that show the structural difference rather than a rename: Axolotl's `adapter: lora|qlora|omitted` has no single counterpart — LLaMA-Factory composes it from `finetuning_type: lora|full` **plus** `quantization_bit: 4`; and Axolotl's stage selection is the ***absence*** of `rl:` / `reward_model:` / `pretraining_dataset:`, where LLaMA-Factory has an explicit `stage:` key.
- **Why asked:** It is the fastest way to find out whether a candidate's framework knowledge is structural or lexical. Anyone can memorise one file's keys; translating means knowing what the keys *do*.
- **Trap:** Assuming `mask_history` and `train_on_prompt` map cleanly onto `roles_to_train` and `train_on_inputs`. They are the right pairing, but LLaMA-Factory has no equivalent of `train_on_eos` at all — which is why "the model never stops generating" is an Axolotl-shaped bug report.

**Q16. How does dataset wiring differ between the two frameworks, and which one is safer?**

- **Answer:** Axolotl wires datasets **inline** in the training config: `datasets: [{path, ds_type, type, field_messages, message_property_mappings, revision}]`. LLaMA-Factory wires them through a **separate registry** — `dataset: <key>` resolving into `data/dataset_info.json`. Axolotl's inline form means the config is self-contained and diffable, which is the whole thesis of the tool. LLaMA-Factory's registry means the config is not self-contained: you must version two files, and the registry is where its loud "lookup" failures come from. Neither is safer in the abstract; the point is that **which file you must commit is different**.
- **Why asked:** It is the concrete difference behind "config-as-artefact" and it is the one that survives a platform review.
- **Trap:** Answering "Axolotl's is better because it's one file" without noticing that a `path:` pointing at a mutable Hub dataset is still not pinned — you need `revision:` (CS-17 §4.1.4).

**Q17. What is the difference between `type:` and `ds_type:`?**

- **Answer:** `type:` is the **prompt strategy** — which loader class reads your rows (`chat_template`, `alpaca`, `completion`, `chat_template.default`, …). `ds_type:` is the **file format for local data** (`json` / `csv` / `parquet` / `arrow`), and `data_files:` names the file. Setting `ds_type: chat_template` is a config error; setting `type: json` silently selects nothing useful (CH-17 §4.4 trap 2 — "easy to confuse", CS-17 §4.3.8).
- **Why asked:** It is a two-key, one-letter distinction that produces a silent failure in one direction and a loud one in the other, so the answer reveals whether the candidate has debugged a real config.
- **Trap:** Inferring `ds_type` from the file extension. It is not derived; it is declared.

**Q18. Spot the bug: a `{"messages":[{role,content}]}` JSONL trained with `type: completion`.**

- **Answer:** `type: completion` reads a `{"text": "..."}` column and trains on it as **plain text with no role masking and no template** (CS-17 §4.5.2). On a messages file you get either an empty field or a tokenised Python repr of the message list. The signature is unmistakable once you know it: **loss sits around 0.5 and the model answers questions by continuing them.** CS-17 §15.1's first attempt did exactly this — loss stuck at **2.31**, the adapter emitting `[{'role': 'user', 'content':`. The fix took 12 minutes and the restarted loss was **1.42 by step 100**.
- **Why asked:** It is the module's canonical "loss looks great and the model is broken" failure, and §4.5.4 ranks it as the single most expensive wrong choice — **hours to days**, because it is only diagnosable by reading a generation.
- **Trap:** Diagnosing it from the loss curve. The loss *fell*. The only reliable detectors are `preprocess --debug` and a generation.

**Q19. You have run `preprocess --debug`. What are the four questions you ask of the printed token dump?**

- **Answer:** From CS-17 §4.7.4: (1) do I see the model's **own** turn markers — `<|im_start|>` / `<|start_header_id|>` / `[INST]` — and not invented ones? (2) is the **user's text masked** (`-100`) and the **assistant's text labelled**? (3) is the **final EOS/EOT position labelled**, not `-100`? A masked terminator is why a model never learns to stop. (4) does the last labelled token look like the **end** of the answer, or is it truncated mid-sentence by `sequence_len`? Any "no" is a config change, not a hyperparameter search.
- **Why asked:** It is the highest-value 60 seconds in the module, and question (3) is the one candidates forget. CH-17 §12's pre-flight check 5 is the same instruction.
- **Trap:** Checking questions (1) and (2) and stopping. (3) and (4) are the two that produce a model which trains cleanly and behaves badly.

**Q20. `eot_tokens` — what is the constraint, and what breaks if you violate it?**

- **Answer:** Every entry must be a **single tokenizer token** (CS-17 §4.7.2). A multi-token entry shifts the mask offsets, so the terminator is mis-aligned and the model does not learn to stop. The one-liner is a print of "single token" or "*** MULTI-TOKEN: eot_tokens will misalign ***" over the configured list. For Qwen the correct value is `eot_tokens: ["<|im_end|>"]`.
- **Why asked:** It is a one-line check with a multi-hour failure, and it is part of CS-17's "matched triple" — model family (Qwen) → template name (`qwen3`) → terminator token (`<|im_end|>`). Any of the three wrong gives a different symptom, which is the next question.
- **Trap:** Copying `eot_tokens` from another model's config because the templates "look similar." They are not: `<|im_end|>` is ChatML/Qwen, `<|eot_id|>` is Llama-3.

**Q21. `chat_template: tokenizer_default` — when does it fail, and what exactly does the error say?**

- **Answer:** It fails when you ask for the tokenizer's own template and the tokenizer has none: `chat_template is tokenizer_default but tokenizer's chat_template is null` (CH-17 §11). The important part is the **fix**, which is not in the training YAML at all — you supply the template in the **tokenizer config**, because the serving stack needs it too. This is the strongest argument for the key: it is the one setting that cannot drift between train and serve, because both read the same object (CS-17 §4.7).
- **Why asked:** It tests whether the candidate understands that the chat template is a property of the *interface*, not of the training script. The failure is loud here and silent for every other template mistake, which is exactly why it is the safe default.
- **Trap:** Reaching for `chat_template: qwen3`/`chatml`/`llama3`/`jinja` to "fix" the error. Naming a template is a *choice*; `tokenizer_default` is an *inheritance*, and inheritance is the only version that cannot go stale.

**Q22. What does `special_tokens:` do to the embeddings, and what is the asymmetry?**

- **Answer:** It resizes the embedding matrices. The asymmetry: Axolotl **grows** embeddings automatically when the tokenizer has extra tokens, but **shrinks** them only if you set `shrink_embeddings: true` (CS-17 §10 item 16). A resume or merge that grew on one side and not the other produces a **shape mismatch at merge time** — and CS-17 §10 item 17 names vocabulary mismatch as the usual cause of merge failures. Fix it with `axolotl merge-lora`, not with a hand-rolled `PeftModel` merge.
- **Why asked:** It is the mechanism behind an error that reads like a framework bug and is actually a configuration asymmetry. It also tests whether you know the adapter and the base must agree on the vocabulary.
- **Trap:** "The tokenizer handles special tokens." `special_tokens` is the *training-side* declaration; the deployed tokenizer is a separate artefact and must carry the same set.

**Q23. `dataloader_num_workers: 0` with `dataloader_prefetch_factor: 2` — what happens, and what are the two correct fixes?**

- **Answer:** It raises `ValueError: prefetch_factor option could only be specified in multiprocessing. let num_workers > 0 to enable multiprocessing` (CH-17 §11). **`num_workers: 0` is perfectly legal** — it means "load in the main process" and is the only option on Windows and in some sandboxed notebooks. The two fixes: (a) `dataloader_num_workers: 2` and keep the prefetch factor, or (b) keep zero workers and **delete the prefetch key entirely**. Either is correct; what is illegal is the pair. CS-17 §14.2 row 13 shows the working Colab values as `num_workers: 2` with `prefetch_factor: 8`.
- **Why asked:** The source video phrased this as "don't write zero here," which suggests zero is illegal. It is not, and a candidate who repeats the video's phrasing has not run it.
- **Trap:** Quoting a *different* error for the same symptom. CS-17 states this two ways — §4.3.5 quotes the `prefetch_factor` ValueError, §14.2 row 13 quotes `ValueError: num_samples should be a positive integer value, but got num_samples=0`. They are related but distinct: the second also fires when the dataset resolves to **zero rows** after filtering or tokenisation, which is a `type:`/`field_*` bug, not a dataloader bug.

**Q24. What are the four things a YAML file does *not* pin, and what is the fix for each?**

- **Answer:** CS-17 §4.1.4: (1) **library versions** — a lockfile or a Docker digest; `axolotlai/axolotl:main-latest` is *not* a pin; (2) the **dataset revision** — `revision: <sha>` and a `sha256sum` of the file, because a Hub dataset edited upstream silently changes an "identical" run; (3) the **seed and determinism flags** — without a recorded seed the run is a distribution, not a point; (4) the **resolved config as actually run** — Axolotl writes `out/<run>/config.resolved.yml`, and that file, not your hand-edited source, is the record.
- **Why asked:** It is the correction the case study makes to the instructor's claim that re-running the same YAML *is* the reproduction. The four items are the checklist, and interviewers listen for whether you say "resolved config" or just "the YAML."
- **Trap:** Answering with four items that are all *in* the YAML (LR, epochs, batch, model). The four things that matter are precisely the ones the YAML does not contain.

**Q25. `val_set_size` and `test_datasets` — what is the conflict, and why does Axolotl make them exclusive?**

- **Answer:** They are mutually exclusive (CS-17 §10 item 8). Both want to carve a held-out set out of your data, and Axolotl will not let one silently win. The practical decision: `val_set_size: 0.05` for a random in-distribution split you use for early stopping; `test_datasets:` when you have a **frozen, separately-authored** set you want to be able to compare across runs. The second is the one that survives a data refresh; the first is the one you get for free.
- **Why asked:** It is a small mutual-exclusion rule that tells you whether the candidate has thought about what a held-out set is *for*. A split drawn from the same file tells you about overfitting; a frozen set tells you about regression.
- **Trap:** Using the random split as your release gate. It shares provenance with training and will therefore flatter you — the same failure IQ-13 documents as `load_best_model_at_end` selecting the most memorised checkpoint.

**Q26. Which three keys change the number of optimizer steps without changing the effective batch, and which of them changes the LR schedule's meaning?**

- **Answer:** `total_steps` is driven by `rows_surviving ÷ (micro_batch_size × gradient_accumulation_steps) × num_epochs` — or by `max_steps`, which **overrides the epoch-derived schedule entirely** and compresses the LR schedule into it (CS-17 §10 item 10). `num_epochs` changes the step count and therefore the schedule. `gradient_accumulation_steps` changes the step count while leaving activations untouched — and changing it **without recomputing the warmup** shifts the schedule silently (CS-17 §9.4 row 11). The three are not interchangeable: `max_steps` is for a smoke run, `num_epochs` for a real one, `grad_accum` for a VRAM-constrained one.
- **Why asked:** It is the arithmetic that makes "the same config" not the same run, and it feeds directly into the mult-GPU question later.
- **Trap:** Using `max_steps` for a short dev run and then raising it for the real run. Your schedule was calibrated to the wrong horizon.

---

## Level 3 — Advanced, Internals & Theory

**Q27. Why is `sample_packing` the most dangerous silent bug in the tool?**

- **Answer:** Because when the boundary metadata is wrong the model attends across document boundaries, and the two signatures are both consistent with a *healthy* run: **loss → ~0 / perplexity → 1**, or **loss ~1.5–2× the unpacked baseline and flat**. The mechanism: packing relies on `cu_seqlens` — the cumulative offsets array `[0, l₁, l₁+l₂, …]` — to give the kernel a block-diagonal mask and to reset `position_ids` per document. Without it, the kernel simply drops the attention mask and the next document becomes the current document's continuation. Axolotl **#3453** is a real instance: with `sample_packing: true` and `flash_attention_2` on Qwen3.5, `cu_seqlens` never reached the gated-delta-rule kernel and recurrent state leaked across boundaries; switching to `sdpa` "fixed" it only because **SDPA does not pack**. Axolotl **#3608** is the context-parallel variant: default `batch_ring` gave **~1.84× higher loss** against the same model unpacked.
- **Why asked:** It is the module's thesis in one bug — the failure that looks like success. CH-17 §1 item 6 and CS-17 §4.6.4 both lead with it.
- **Trap:** Reading the loss as evidence of efficiency. "Loss went down, so packing works" is exactly backwards: a much lower loss at the same step count is *leakage*, not speed.

**Q28. Walk the A/B regression test that detects it, and state the alarm rule.**

- **Answer:** Identical config except `sample_packing`, `max_steps` fixed at 50–100, run twice (CS-17 §4.6.4). The alarm rule: **if `packing: true` shows a *much* lower loss than `packing: false` at the same step count, you have leakage, not efficiency.** The calibration detail that makes it usable: packing legitimately changes the loss *denominator*, so **some** gap is expected — you calibrate the expected gap once on a known-good model and treat deviations **from that** as the alarm, not deviations from zero. CH-17 §8 puts it in the same terms: it is "the only definitive check, and it belongs in your repo."
- **Why asked:** It is the only question in the bank that tests whether the candidate can design an experiment rather than run one. The "expected gap" nuance is what separates the two.
- **Trap:** Asserting a hard threshold like "packing must be within 5%". There is no such constant — the expected offset is architecture- and backend-specific, which is why CS-17 §4.6.4 says packing correctness is a property of **architecture × backend × Axolotl version**.

**Q29. Packing is correct and fast. Why is your LR schedule now wrong?**

- **Answer:** Packing does not change `gradient_accumulation_steps`, so **the number of optimizer steps per epoch drops by the packing factor**. With 3× packing, a `warmup_steps` value that was 5% of the run becomes 15% of it — a 3× longer warmup *as a fraction*, so the model spends the first sixth of training barely moving (CS-17 §4.6.6). The Colab comment states both remedies in one breath: *"use a slightly higher learning rate to account for fewer steps"* **or** *"reduce the micro_batch_size + gradient_accumulation_steps to achieve closer to the same number of steps/epoch."* The rigorous fix is to **recompute the schedule in token space**: `warmup_steps = 0.05 × steps`, where `steps` is the post-packing count.
- **Why asked:** It is the consequence nobody mentions, and it is the one that makes two packing A/B runs incomparable if you changed the schedule between them.
- **Trap:** Compensating with a raised LR by a guessed factor. CS-17 §15.5 records a team that "raised it 3× and destabilised the run"; the correct move there was a **modest increase, 1.2–1.5×**.

**Q30. When is packing the *wrong* choice?**

- **Answer:** CS-17 §4.6.5 and §4.6.6: rows already near `sequence_len` (efficiency is already ~90%+, so the risk is uncompensated); `eager` or `sdpa` backends; any per-example weighting or custom loss computed over boundaries; interpretable per-example loss; **eval and perplexity reporting**, because packing changes the denominator; sequence-level RL and reward models; and very small debug runs, where the docs recommend `sample_packing: False` **and** `eval_sample_packing: False` "to avoid errors". CS-17 §10 item 1 makes it concrete: 1,800-token rows into a 2,048 window is ~10% gain — not worth any correctness risk.
- **Why asked:** It is the exception side of the module's most-hyped feature, and it is where candidates who have only read about packing run out of material.
- **Trap:** Answering "never — it's strictly better." The tool's own documentation recommends turning it off for the exact runs most people start with.

**Q31. Derive the VRAM for a 7B model: full fine-tune, LoRA, QLoRA.**

- **Answer:** Full FT is the 16-bytes-per-parameter budget — `2 (bf16 weights) + 2 (gradients) + 4 (fp32 master) + 8 (Adam m, v) = 16 B/param`, so 7B ≈ **112 GB**. `code/common/memory.py --table`'s floor is `2 + 2 + 8 = 14 B/param` → **91.6 GiB**, because it folds the master copy away. LoRA r16 prices only the adapter and its optimiser state on top of a frozen bf16 base → **14.7 GB**; QLoRA replaces the base with NF4 at ~**0.5 B/param** → **4.9 GB**. Ratios worth quoting: full FT vs LoRA ≈ **6×**, QLoRA vs full FT ≈ **20×**.
- **Why asked:** It is the arithmetic that decides the topology question, and the interview follow-up is always "why do two credible sources disagree?" — the answer is bytes-per-parameter bookkeeping (14 vs 16), GiB vs GB, and QLoRA's dequantise-on-the-fly transient holding both the NF4 and the bf16 copy.
- **Trap:** Quoting 91.6 GB as the number a 7B full FT *needs*. It is the floor. CS-17 §11.4's observed ~120 GB is what you provision, and CH-17 §7.1 says explicitly: add 10–20% and trust a measurement over any table — including that one.

**Q32. 7B full FT on 8 GPUs — ZeRO-1, ZeRO-2, ZeRO-3 or FSDP2?**

- **Answer:** From CH-17 §7.3: DDP `16P` = **112 GB/GPU** → OOM. ZeRO-1 **38.5 GB**, ZeRO-2 **26.3 GB**, ZeRO-3 **~14 GB**, FSDP2 **~14 GB** (per-layer shard) — plus **4–12 GB** of activations and workspace with checkpointing. The rule is **pick the lowest stage that fits**: ZeRO-3 shards the *parameters*, so every forward pass must re-gather them — more communication for no benefit when the weights already fit. For LoRA on 2–4 GPUs, **ZeRO-2 is usually the right answer**. In current Axolotl the FSDP path is `fsdp_version: 2` + an `fsdp_config:` mapping — the bare `fsdp:` list is rejected and `fsdp_version: 1` is a hard error.
- **Why asked:** It is the one question in this bank with a clean right answer and a clean wrong one — "use ZeRO-3, it shards more" is the wrong one and it is very common.
- **Trap:** Applying LoRA intuition to full FT. The 16 B/param bill is dominated by optimiser state, which is why sharding *states* (ZeRO-1/2) buys most of the win before you ever shard parameters.

**Q33. Why is a quantised base frozen, mechanically?**

- **Answer:** The forward pass dequantises NF4 weights to bf16 for each matmul, so the stored parameter is not the matrix being multiplied — there is no continuous path from a gradient to the 4-bit value. CS-17 §19's answer A6 says it plainly: the quantised weights are **not differentiable**, and a full fine-tune is therefore priced at **roughly 16 bytes per parameter**. The chain to recite is **"quantised base ⇒ frozen base ⇒ adapter-only training"** (CS-17 §4.3.2). CS-17 §10 item 3 adds the consequence people miss: full-FT of an already-Instruct model is *usually* a mistake, because the adapter route costs **~20× less**.
- **Why asked:** It tests the mechanism behind a rule that is otherwise memorised as a config constraint. It also sets up the VRAM derivation above.
- **Trap:** "You can quantise, train, then merge back to bf16 — so it's effectively full FT." Merging recovers a bf16 *artefact*; it does not recover full-FT *optimisation*, because the 4-bit error was present in every forward pass you trained through.

**Q34. Your run reports `steps_per_epoch: 1,105` against a 10,000-row file with `gradient_accumulation_steps: 8`. What happened?**

- **Answer:** `rows_surviving = steps × grad_accum = 1,105 × 8 = 8,840`, so **~12% of the file was deleted**. The cause is `excess_length_strategy`'s default `drop` — rows longer than `sequence_len` are removed, and nothing tells you how many (CS-17 §1.4, §9.4 row 10). This identity is the single most useful diagnostic in the tool: run it *before* the job, compare it against `wc -l`, and you have the count for free.
- **Why asked:** It is a derivation, not a recall question, and the interviewer is checking whether the candidate reasons from the step count or guesses.
- **Trap:** Concluding that the framework lost data and filing a bug. It is documented behaviour on a default you did not change.

**Q35. Loss is flat at ~11.9. Another engineer says it should be flat at ~2.3. Who is right?**

- **Answer:** **11.9.** `ln(V)` for Qwen2.5's 151,936-token vocabulary is ≈ **11.93**; **2.303 is `ln(10)`**, an unrelated constant. The `~2.31` that also appears in CS-17 §15.1 is the *converged* loss of a degenerate run — the `type: completion` first attempt — not an initial one, which is why the two numbers coexist in the file. The diagnostic to teach is "**flat, and nowhere near `ln(V)`**", and the number to compare against is your own tokenizer's `math.log(vocab_size)`: ≈11.9 for Qwen's 152k vocabulary, ≈10.4 for a 32k Llama/GPT-NeoX tokenizer.
- **Why asked:** It is a fact-check disguised as a debugging question, and the *shape* of the right answer — "which tokenizer?" — is what is being graded.
- **Trap:** Picking a side without asking for the vocabulary size. Both numbers are meaningful; they describe different events.
- **See:** the two `> **Correction:**` blocks below, which record the source file's own disagreements on this and on the FSDP key dialect.

> **Correction (CS-17 §9.4 row 1 and §14.2 row 1 vs §15.1).** §14.2 row 1 describes a fully-masked run as *"Loss flat at ~2.3 (ln(V))"*. **`ln(V)` is not 2.3.** For Qwen2.5's 151,936-token vocabulary `ln(V) ≈ 11.93`; for a 32k-vocabulary tokenizer `≈ 10.4`. The `~2.31` figure in §15.1 is a *converged* loss for a degenerate run, not an initial one. The correct diagnostic is **"flat, and nowhere near `ln(V)`"**, compared against your tokenizer's `math.log(vocab_size)`.

> **Correction (CS-17 §15.1 vs §4.3.10 / §4.8.4 / §5.6 — the FSDP key dialect).** §15.1's FSDP block (L2741–2745) uses the **FSDP1-prefixed** names `fsdp_auto_wrap_policy`, `fsdp_transformer_layer_cls_to_wrap`, `fsdp_state_dict_type`, `fsdp_cpu_ram_efficient_loading`, `fsdp_offload_params`. §4.3.10's rename table, §4.8.4 and §5.6 treat the prefixed forms as **renamed or removed** and use the unprefixed `auto_wrap_policy`, `transformer_layer_cls_to_wrap`, `state_dict_type`, `cpu_ram_efficient_loading`, `offload_params`. The current dialect is the **unprefixed one** — §5.6 is titled "The FSDP config, corrected" and is the section that was written against a running install. Copy §5.6's block, not §15.1's.

---

## Level 4 — System Design & Scenario

> These are 10–15-minute whiteboard questions. Answer in the fixed shape: **requirements → constraints → design → trade-offs → failure modes.**

**Q36. 48,000 curated Q&A rows. 2×A100-80GB. Design the run and state what you will measure.**

- **Answer:** *Requirements:* domain Q&A with a refusal boundary and zero fabricated dosages; behaviour change, not knowledge injection. *Constraints:* 2 GPUs, so FSDP2 (`fsdp_version: 2` + `fsdp_config:`) or ZeRO-2; a frozen eval set must exist before the job starts. *Design:* QLoRA, `sequence_len` at p99.5 of rendered lengths (CS-17 §15.1 uses 2048), `micro_batch_size: 2` × `grad_accum: 8` × 2 GPUs = effective 32, `num_epochs: 2`, LR `1e-4` with `warmup_ratio: 0.03`, `sample_packing: true` (with FA2 confirmed and the packing A/B in the repo), `chat_template: qwen3` + `eot_tokens: ["<|im_end|>"]`, `val_set_size: 0.05` with `eval_steps: 100`, `wandb_run_id` set. *Trade-offs:* 2 epochs over a 48,000-row set is **3,000 steps at ~4,100 tok/s ⇒ ≈9.1 GPU-hours ≈ $18** — so the GPU is a rounding error and you should spend it on ablations and evals, not on saving it. *Failure modes:* the eval loss bottoms at **~epoch 1.7** and then rises — early-stop on the composite, not on the last checkpoint; and the `type:` must be `chat_template`, because the first attempt at exactly this dataset used `type: completion` and lost hours producing a model that emitted `[{'role': 'user', 'content':`.
- **Grading:** credit for naming the *data* cost alongside the GPU cost (CS-17 §11.3 prices the 10,000-row analogue at **~$400,000** of expert time against **$4.60** of GPU), for freezing the eval set before the run, and for the packing A/B. Deduct for choosing full FT "because we have A100s" with no ablation.
- **Fail condition:** no held-out set, or an answer that treats `succeeded` as success.

**Q37. Full fine-tune a 70B on one 8×H100 node. Walk the memory arithmetic and the three failures you expect.**

- **Answer:** *Requirements:* full FT because LoRA plateaus **4 points below** on the benchmark (CS-17 §15.3). *Constraints:* `16 B/param × 70 B ≈ 1,120 GB`, ÷8 = **140 GB/GPU** — more than an 80 GB card. *Design:* pure ZeRO-3 is **not enough**; you need ZeRO-3 **plus CPU offload** (`zero3_offload.json`, fetched with `axolotl fetch deepspeed_configs`), or 16 GPUs. `sequence_len: 4096`, `micro_batch_size: 1`, `grad_accum: 16`, 2 epochs, LR `1e-5` (**full FT is ~20× lower than LoRA's**), `optimizer: adamw_torch`, `save_total_limit: 3`. *Trade-offs:* ~6,250 steps, ~118 GPU-hours, **$350–450 at H100 spot** — and the checkpoint is a DeepSpeed ZeRO-3 shard, so serving it requires recombining via `zero_to_fp32` or `axolotl merge-sharded-fsdp-weights`. *Failure modes, all three from the source:* (1) `micro_batch_size: 2` → OOM at step 1, because **on 70B, `micro_batch_size` is always 1**; (2) `paged_adamw_8bit` carried over from LoRA habit → it fights over pinned memory and throughput collapses to **180 tokens/s**; (3) `learning_rate: 1e-4` → grad norm above **400** and loss climbing within 300 steps.
- **Grading:** the memory table must be derived, not recalled; the offload conclusion must follow from 140 > 80; at least two of the three failures should be predicted.
- **Fail condition:** proposing 8 GPUs with plain ZeRO-3 and no offload, or quoting a LoRA learning rate for a full fine-tune.

**Q38. Design a CI gate that would have caught CS-17 §15.1's first attempt before it cost a GPU-hour.**

- **Answer:** *Requirements:* catch a wrong `type:`/template/mask **without** training, in CI, on every config change. *Design, in order:* (1) `yaml.safe_load` for syntax, 50 ms; (2) `axolotl preprocess cfg.yml --debug-num-examples 3` with `strict: true` set in the CI copy of the config, so a typo'd key fails instead of becoming `None`; (3) an assertion over the printed token dump — the model's own turn markers present, the user span `-100`, the final EOS position **labelled**, the last labelled token not mid-sentence; (4) the `steps × grad_accum` identity against the file's row count, so silent `drop` fails the build; (5) a 20-step smoke train with the packing A/B, comparing against a calibrated expected gap. *Trade-offs:* steps 1–4 cost seconds and no GPU; step 5 costs minutes and is the only one that can catch a backend-specific packing bug. *Failure modes:* making step 3 a semantic check instead of a structural one — you must assert on token ids and label positions, not on a rendered string that "looks right."
- **Grading:** credit for `strict: true` (the schema half) **and** for the token-dump assertions (the data half). CH-17 §4.5's table is the map: config is loud, data is silent, so a CI gate needs both halves.
- **Fail condition:** an answer that only runs the validator. That is the same gate the tool already has, and it is the gate that let the bug through.

**Q39. Design the versioning, serving and rollback story for a shipped adapter.**

- **Answer:** *Requirements:* an adapter that is a **delta**, not a model — meaningless without the exact base revision and the exact template (CS-17 §16.1). *Design:* ship `adapter_model.safetensors` + `adapter_config.json` (~160 MB fp16 / ~80 MB bf16) plus `out/<run>/config.resolved.yml` and a `RUN_RECORD.yaml` from `run-record.sh` containing git commit, `axolotl --version`, pip-freeze, the base model revision SHA, the dataset `sha256sum`, the config `sha256sum`, GPU type and count, and the seed. Serve with vLLM `--enable-lora` so rollback is a module swap; name versions `task-stage-model-version-dataset-quarter-recipe-revision` rather than a bare date. *Trade-offs:* three ~160 MB adapters behind one base cost **~0.5 GB** against **~45 GB** for three merged 7B models — a **~29 GB saving** and a seconds-long rollback, at the price of an adapter-aware server. *Failure modes:* an unpinned base revision (the adapter then silently loads onto different weights), template drift at serve time (canary prompt suite), and a rollback that was documented but never tested — CS-17 §16.8's tenth checklist item.
- **Grading:** the base-revision pin and the resolved-config archive are the two non-negotiables; the canary suite is the differentiator.
- **Fail condition:** "we version the model" with no mention of the base, the config, or the data.

---

## Level 5 — Debugging & Incident Response

> Answer with a sequence. Say what you check first, second and third, and name the look-alike you are ruling out at each step.

**Q40. Loss is flat at ~11.9 from step 1 and never moves. You have 20 minutes.**

- **Answer:** (1) This is **every label is `-100`** — zero trainable tokens. (2) Run `axolotl preprocess cfg.yml --debug --debug-num-examples 3` and read the rendered text: are there any non-`-100` label positions? (3) If zero, the cause is one of three: a wrong `type:` (the loader never found an assistant turn), a wrong `chat_template` (the template never marks one), or `train_on_inputs`/`roles_to_train` inverted. (4) Check `eot_tokens` alignment last — a multi-token terminator shifts the offsets. (5) Fix and re-preprocess; the number should now be near `ln(V)` at step 0 and descend.
- **Ruling out:** the look-alike at ~0.1 from step 1, which is *not* this bug — that is a memorisation/val-set problem, and the fix is `val_set_size: 0.05` + more data, not a template change.
- **Grading:** step 2 must come before any hyperparameter change. This is CS-17 §14.4's first of three look-alikes.

**Q41. Loss descends, then dives to ~0 / perplexity ~1. You have a deadline.**

- **Answer:** (1) Do **not** celebrate — near-zero loss on a chat SFT set is a bug, and the first hypothesis is **packing with broken boundary metadata**. (2) Check whether `sample_packing: true` is set and whether `attn_implementation` is a varlen backend (`flash_attention_2`/`_3`/`_4`). (3) Run the A/B: identical config except `sample_packing`, `max_steps: 100` fixed, two output dirs. (4) If packed loss is *much* lower than unpacked at the same step count, you have leakage — disable packing for that architecture and file against your Axolotl version. (5) Second candidate: a three-example dataset where the effective batch exceeds the row count, giving 0 steps or pure memorisation. (6) Third: `save_steps` larger than the whole run, so the "checkpoint" you are reading is not what you think.
- **Ruling out:** the legitimate ~1.5–2× *higher* packed loss, which is the other packing signature and is also a bug — but a different one (attention crossing boundaries rather than recurrent state leaking).
- **Grading:** the candidate must refuse the good news. "Loss → 0 means memorisation or leakage, never excellence" is CS-17 §17.8.

**Q42. Loss looks fine. The model answers fluent, well-shaped text in the wrong language and ignores the instruction. Sequence.**

- **Answer:** (1) This is the **wrong chat template** signature — loss trains normally, output is fluent and wrong-shaped. (2) Render the tokenizer's own template: `tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)` and print `len(ids)` and `tok.convert_ids_to_tokens(ids[:24])`. (3) Compare it against what Axolotl will build from the `chat_template:` **name** you configured — if they differ, you have found the bug. (4) Fix to `chat_template: tokenizer_default`, which cannot drift from serve. (5) Re-check the prompt mask while you are in there, since a template change moves the boundaries. (6) Re-run `preprocess --debug` — and **`rm -rf last_run_prepared/` first**, because the cache is keyed on the data, not on the template.
- **Ruling out:** the third look-alike — loss descends, output is correctly shaped, and it is simply no better than the base. That is **capacity or data quantity**, not a bug: train a 5× larger adapter (`lora_r: 64`) and evaluate; if it is still flat, it is the data.
- **Grading:** CS-17 §14.4's three-way split is the model answer. A candidate who proposes a hyperparameter change here has missed the module.

**Q43. `CUDA out of memory` — at step 1, after N steps, and only at eval. Are these the same incident?**

- **Answer:** No — three different causes and three different first moves. (1) **At step 1:** `micro_batch_size × sequence_len` is too large. Halve the micro-batch, turn on `gradient_checkpointing: true`, and **turn off `sample_packing` if you enabled it blindly** — packing raises peak memory before it lowers it. (2) **After N steps, not step 1:** fragmentation with variable-length batches while packing is off. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (Axolotl exposes a helper for this), enable packing, and sort by length. (3) **Only at eval:** `eval_sample_packing` differs from the training value, or the eval batch is not scaled down. Set `eval_sample_packing: false` and shrink the eval batch.
- **Ruling out:** host-RAM exhaustion, which looks like an OOM but reports as `exitcode: -9` on a multi-GPU job and is caused by offload and `cpu_ram_efficient_loading` pressure — a ZeRO-3 + offload job can want 100 GB+ of system memory.
- **Grading:** the candidate must not answer with a single remedy. Three timings, three causes.

**Q44. A multi-GPU job dies with `exitcode: -9`. `nvidia-smi` shows the GPUs were never full. Sequence.**

- **Answer:** (1) `-9` is **SIGKILL from the host OOM killer**, not a VRAM error — the GPUs being idle is the confirmation, not a contradiction. (2) Check the offload settings: `offload_params` and `cpu_ram_efficient_loading` both raise host memory pressure. (3) Check the machine's system RAM against `16 B/param` sharded plus offload buffers. (4) Reduce offload, or move to a host with more RAM. (5) Only then look at the launcher — if you arrived here from a hang rather than a kill, check that `--nproc_per_node` matches the real GPU count and, in Docker, that `--gpus all --ipc=host` are present.
- **Ruling out:** the startup hang, which is an NCCL/launcher problem (`--nproc_per_node` mismatch, missing `--ipc=host`) and looks adjacent but has no exit code at all.
- **Grading:** "check the GPUs" is the fail answer. This is a **host** memory incident on a GPU job — CS-17 §10 item 20's neighbour.

**Q45. The run restarts from step 0 despite checkpoints on disk. And separately: the adapter loads but the output is unchanged. Two incidents, two sequences.**

- **Answer:** **Restart:** (1) `--resume_from_checkpoint` was not passed — Axolotl does not resume implicitly; (2) or `output_dir` changed, so the trainer cannot see the previous run's state. Confirm the checkpoint contains `optimizer.pt`, `scheduler.pt` and `trainer_state.json` — a checkpoint without optimiser state resumes the weights but not the schedule. Fix: `axolotl train cfg.yml --resume_from_checkpoint ./out/checkpoint-500`. **Unchanged output:** (1) the base was loaded without the adapter — check `adapter_config.json` exists and that `axolotl inference cfg.yml --lora-model-dir ...` was used, not a bare base load; (2) the model is not in `eval()` mode; (3) the adapter was saved before any training happened. (4) If all three pass, compute the "did it change anything" check: run the same prompts against base and adapter at `temperature=0` and count identical outputs — near-total identity means the fine-tune did nothing, and the cause is a schedule or LR problem, not a loading problem.
- **Ruling out:** template mismatch, which produces a *changed* model that behaves wrongly — the opposite symptom.
- **Grading:** credit for separating a state-management incident from a loading incident, and for reaching for an identity check rather than re-running training.

**Q46. Loss differs between a 1-GPU and a 4-GPU run of the same config. What happened, and which fix do you ship?**

- **Answer:** (1) `gradient_accumulation_steps` was kept while the world size changed, so the **effective batch changed by 4×** — the loss is not wrong, the experiment is different. (2) Two defensible fixes: **(a)** divide `grad_accum` by the world size, holding the effective batch and therefore the step count and the LR schedule constant; **(b)** keep `grad_accum` and scale the LR (√ is common, linear aggressive). (3) **Ship (a) for a reproduction** — CS-17 §19's answer A4 is explicit that a reproduction should hold the effective batch fixed, because (b) changes the optimisation dynamics as well as the batch. (4) Record which you chose in the run record, because an unrecorded choice is exactly the confound the next person will be unable to see.
- **Ruling out:** a genuine numerical difference from sharding, which exists but is small and does not produce a systematic gap from step 1.
- **Grading:** the question is graded on picking a default *and* stating what it costs you. "It depends" without a default fails the module's meta-rule.

**Q47. A 220,000-row, 3-epoch run on 4×A10G is estimated at 41 hours. You have a day. Sequence.**

- **Answer:** (1) Measure first — a 20-step smoke run gives `sec_per_step`, and that is the only honest input to the estimate (CS-17 §11.1 step 5; it is the step nobody measures). (2) Turn on packing: **`sample_packing: true` + `pad_to_sequence_len: true`** — ~950 → ~3,400 tok/s, 41 h → 11.5 h. (3) Add `attn_implementation: flash_attention_2` — → ~4,100 tok/s, 9.5 h. (4) Turn **off** `gradient_checkpointing` and drop `micro_batch_size` to 1, since VRAM is now available — → ~5,200 tok/s, 7.5 h. (5) Raise `dataloader_num_workers: 4` with `dataloader_prefetch_factor: 4` — → ~5,900 tok/s, **6.6 h**. Total **6.2×**, at identical effective batch and identical LR, cost **~$98 → ~$16**. (6) **Recompute the LR schedule** after packing changes the step count (Q29), and raise the LR only **1.2–1.5×** if at all.
- **Ruling out:** the tempting move of raising the LR first. CS-17 §15.5 records that the team who did that "raised it 3× and destabilised the run."
- **Grading:** the ordering matters — packing first (biggest, cheapest), then attention backend, then checkpointing, then dataloader. A candidate who starts with the LR has optimised the wrong variable.

---

## Rapid Fire — True / False / One-Liner

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | Axolotl rejects unknown YAML keys. | **False** | `DictDefault` returns `None`; only `strict: true` makes it loud (CH-17 §11) |
| 2 | `sequence_len` defaults to 2048. | **False** | **512** — and an HF `max_seq_length` is silently ignored |
| 3 | Rows longer than `sequence_len` are truncated. | **False** | **Dropped** — `excess_length_strategy: drop` is the default |
| 4 | `val_set_size` defaults to 0.0. | **True** | No eval loss ever appears unless you set it |
| 5 | `adapter: qlora` requires `load_in_4bit: true`. | **True** | A pydantic error at `load_cfg()`, before the model loads |
| 6 | `sample_packing` works on any attention backend. | **False** | Needs a varlen backend; `attn_supports_packing` is derived, not configurable |
| 7 | Packing is always faster *and* always correct. | **False** | 2–6× faster; correctness is architecture × backend × version (issues #3453/#3608) |
| 8 | A much lower packed loss at equal steps means efficiency. | **False** | It means boundary leakage — stop and A/B |
| 9 | Packing leaves your LR schedule unchanged. | **False** | Steps/epoch drop by the packing factor; warmup becomes a larger fraction |
| 10 | `cu_seqlens` is what prevents cross-document attention. | **True** | The cumulative-offsets boundary array |
| 11 | `train_on_inputs` defaults to `false`. | **True** | Prompt masking is the default; `roles_to_train: ["assistant"]` is the other form |
| 12 | `train_on_eos` defaults to `turn`. | **True** | Mask the terminator and the model never learns to stop |
| 13 | A multi-token `eot_tokens` entry is tolerated. | **False** | It shifts the mask offsets; each entry must be one token |
| 14 | `chat_template: tokenizer_default` is the safest setting. | **True** | It cannot drift from what the serving stack uses |
| 15 | The DPO dataset type today is `type: dpo`. | **False** | `chat_template.default`; `dpo`/`preference` are gone |
| 16 | IPO is selected with `rl: ipo`. | **False** | `rl: dpo` + `dpo_loss_type: [ipo]` |
| 17 | `dpo_beta` is the current key. | **False** | `rl_beta` replaced it |
| 18 | `fsdp_version: 1` still works. | **False** | A hard error; the bare `fsdp:` list is rejected too |
| 19 | ZeRO-3 is the right default for LoRA on 4 GPUs. | **False** | ZeRO-2 — pick the lowest stage that fits |
| 20 | A 7B full fine-tune needs ~91.6 GB. | **Half** | That is the `memory.py` floor; provision for CS-17 §11.4's observed ~120 GB |
| 21 | `num_workers: 0` is illegal. | **False** | Zero workers is legal; zero **with** an explicit prefetch factor is not |
| 22 | A YAML file is enough to reproduce a run. | **False** | Four more things: versions, data revision, seed, resolved config |

---

## Coding / Whiteboard Tasks

### Task 1 — Read a token/label dump (10 minutes)

You are given `axolotl preprocess sft.yaml --debug --debug-num-examples 2` output showing `input_ids` and `labels` for one Qwen row. The rendered text is `<|im_start|>system\nYou are a pharmacist.<|im_end|>\n<|im_start|>user\nIs X safe?<|im_end|>\n<|im_start|>assistant\nYes, with caveats.<|im_end|>`.

- **Expected solution sketch:** Decode `input_ids` and confirm the four markers are the model's **own** (`<|im_start|>` for Qwen, not `<|start_header_id|>` for Llama-3). Then walk `labels` and locate the boundaries: every position belonging to the system and user spans should be `-100`; the first non-`-100` should be the first token of `Yes`; the last should be the `<|im_end|>` id, **not** `-100`. Count the non-`-100` positions and divide by the sequence length — for a row like this the supervised fraction lands in IQ-13's 10–40% band.
- **The decisions being graded:** (a) do you check the *terminator* rather than the answer? (b) do you measure the supervised fraction instead of eyeballing it? (c) do you notice that the boundary token straddling prompt and answer is conservatively masked?
- **Grading:** all four CS-17 §4.7.4 questions answered, with the tokenizer's own `log(vocab_size)` computed as the reference for "flat at `ln(V)`".
- **Fail condition:** reading the rendered string and declaring it correct. The string can look perfect while the labels are wrong.

### Task 2 — Write the packing regression test (10 minutes)

Write the repo test that would catch a packing-boundary regression on a new model architecture.

- **Expected solution sketch:** A shell or pytest driver that (1) copies the base config twice with `sample_packing: false` and `true`, `max_steps: 100` fixed and different `output_dir`s; (2) runs `axolotl train` on both; (3) reads the final training loss from each run's log; (4) compares the gap against a **stored expected gap** for that architecture × backend × Axolotl version rather than against zero; (5) fails the build when the packed loss is *lower* than the unpacked loss, or when the gap exceeds the stored tolerance.
- **The decisions being graded:** that the alarm is a **lower** packed loss (leakage) and not merely a different one; that the expected offset is calibrated once and versioned, not hard-coded; that `max_steps` is fixed so the two runs are comparable at equal steps rather than equal epochs.
- **Grading:** a stored per-architecture expectation, and a passing baseline run that establishes it.
- **Fail condition:** asserting `packed_loss ≈ unpacked_loss`. Packing legitimately changes the loss denominator, so that test fails on a healthy run.

### Task 3 — Derive the budget and pick the topology (10 minutes)

10,000 curated rows, mean 512 tokens, p99 1,400, target `sequence_len: 1536`, 3 epochs, one A100-40GB at ~$1.50/hr, `micro_batch_size: 4` × `grad_accum: 4`. Give tokens trained, GPU-hours and dollars, then say what changes if the model were 70B full-FT.

- **Expected solution sketch:** `drop` at 1536 removes ~0.5% → **9,950 rows**; `9,950 ÷ 16 = 622 steps/epoch`; `622 × 3 = 1,866 steps`; `1,866 × 4 × 4 × 1536 = 45.9 M tokens`; at a measured ~4,200 tok/s → `45.9e6 ÷ 4,200 ÷ 3600 ≈ **3.0 GPU-hours**` → **~$4.60**. Then the pivot: the same 10,000 rows at 15 minutes of expert time each is roughly **$400,000** — **four orders of magnitude** apart. For 70B full FT, `16 B/param × 70 B = 1,120 GB`, so 8 GPUs give **140 GB/GPU** and plain ZeRO-3 is not enough; you need ZeRO-3 **plus offload** or 16 GPUs.
- **The decisions being graded:** measuring `sec_per_step` rather than assuming it; applying `drop` before computing the step count; naming the data cost, not just the GPU cost.
- **Grading:** the tokens figure must be derived from the post-drop row count, and the 70B conclusion must follow from the arithmetic.
- **Fail condition:** quoting a GPU cost without the data cost, or proposing 8×H100 with no offload.

### Task 4 — Design the start-up assertion block (10 minutes)

Write the assertions a training script should run before it spends a GPU-hour, given only the config path.

- **Expected solution sketch:** (1) `yaml.safe_load` the file — syntax; (2) assert `strict: true` is present in the CI copy, so a typo'd key cannot survive; (3) assert `sequence_len` and `num_epochs` are the Axolotl spellings, not `max_seq_length`/`num_train_epochs`; (4) load the tokenizer and assert `chat_template` renders and that each `eot_tokens` entry is a **single** token; (5) assert the dataset resolved to a non-zero row count and that `steps × grad_accum` matches the file's line count (or fails with the dropped count printed); (6) if `sample_packing: true`, assert `attn_supports_packing` for the configured `attn_implementation`; (7) assert `val_set_size > 0`; (8) after building the model, assert `model.config._attn_implementation` equals what you asked for — the deprecated-boolean path silently leaves you on SDPA.
- **The decisions being graded:** covering both halves — the config half (`strict`) and the data half (the token dump, the row count, the terminator) — and asserting on the **resolved** implementation rather than the requested one.
- **Grading:** at least six of the eight, with (4), (5) and (8) present.
- **Fail condition:** a block that only validates YAML types. That is a re-implementation of the schema Axolotl already runs.

---

## Cheat Sheet of Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| `sequence_len` default | **512** | Raise it; the HF spelling is silently ignored |
| `num_epochs` default | **1.0** | 1–3 for SFT |
| `val_set_size` default | **0.0** | No eval by default |
| `excess_length_strategy` default | **`drop`** | Silently deletes over-long rows |
| `strict` default | **`false`** | Set `true` in CI |
| `optimizer` default | **`adamw_torch_fused`** | The default changed — record it |
| `lora_r` / `lora_alpha` | **16 / 2 × r = 32** | Only `alpha/r` sets adapter loudness |
| `lora_dropout` default | **0.0** | Illegal with the fused LoRA kernels |
| LR — LoRA / full FT / DPO | **1e-4…3e-4 / 1e-5…5e-5 / 5e-6** | DPO is 10–40× below SFT |
| Warmup | **5–10%** of steps | Use `warmup_ratio`; it and `warmup_steps` are exclusive |
| Packing speedup | **2–6×** | `sequence_len ÷ mean_row_tokens` |
| Padding efficiency | **~35% → ~99%** | Unpacked vs packed |
| Loss flat at `ln(V)` | **≈11.9** (Qwen 152k) / **≈10.4** (32k) | **Not 2.3** — `2.303` is `ln(10)` |
| Healthy train loss (chat SFT) | **0.5 – 2.0** | < 0.4 ⇒ memorisation or empty targets |
| Healthy grad norm | **0.1 – 10** | > 100 ⇒ instability |
| Bytes/param — full FT | **14 (floor) / 16 (budget)** | `memory.py` vs an fp32 master copy |
| Bytes/param — NF4 | **0.5** | 4-bit + double quant |
| VRAM, 7B: full / LoRA r16 / QLoRA r16 | **91.6 / 14.7 / 4.9 GB** | The measured floor; add 10–20% |
| ZeRO per-GPU, 7B on 8 GPUs | **1: 38.5, 2: 26.3, 3: ~14 GB** | Plus 4–12 GB activations |
| 70B full FT | **16 B/param = 1,120 GB → 140 GB/GPU on 8** | Needs offload or 16 GPUs |
| T4 VRAM / video's QLoRA peak | **16 GB / ~5 GB** | **Not 12 GB / not 24 GB** |
| Tokens per word | **≈1.33** | English, ±15% |
| The demo identity | `rows = steps × grad_accum` | 1,105 × 8 = **8,840** from a 10k file |
| Data : GPU cost ratio | **~$400,000 : ~$4.60** | Training is nearly free; data is the cost centre |

---

## Answers To The Self-Check Questions From CS-17

> CS-17 §19 poses ten questions and answers them at §19.11. These are the interview-grade restatements: the number, the mechanism, and the trap an interviewer will follow up with.

**1. A colleague says "we use Axolotl because YAML is easier than Python." Give the stronger argument — the one that survives a compliance review.**
- **The strong argument:** a YAML file + a pinned image + a dataset hash is a **complete, reviewable, diffable description of the training procedure**, and re-running it next quarter reproduces the model. CS-17 §4.1.2's table is the evidence: a 2-line semantic diff instead of a code diff mixed with logic; hashable for provenance (`sha256(config.yml)` *is* the run ID); machine-readable by a scheduler; and — the underrated row — **a config file cannot contain an `if` statement that silently skips a preprocessing step only on Tuesdays.** There is no control flow in YAML, so a whole bug class is structurally absent.
- **The detail that makes it land:** Axolotl writes `out/<run>/config.resolved.yml`, so the reviewer sees the **actual** configuration including every default that fired, not the subset the author remembered to set.
- **Trap:** arguing convenience. "Easier" loses the room; "auditable, hashable, and free of hidden control flow" wins it. And a candidate who claims YAML is *safer* in general has not read CS-17 §4.1.3 — "the cost of the convenience — five real downsides". The sharpest one is the `DictDefault` entry in §3's glossary: it "returns `None` for missing keys instead of raising `KeyError`", which is exactly why **a typo'd key does not error** — it silently becomes `None` and the default is used, with `strict: true` as the guard. That is why the correct answer is "auditable", not "safer".

**2. The model's loss is flat and the output repeats the prompt. What is wrong, and what do you run?**
- **Diagnosis:** the labels are not what you think. Two candidates, and you cannot tell them apart from the loss: `train_on_inputs: true`, or a `chat_template:` that renders the user turn as trainable. Both leave the prompt positions labelled, so the model learns to produce the prompt.
- **Run:** `axolotl preprocess sft_config.yml --debug --debug-num-examples 3`, then print the label tensor and count non-`-100` positions. If the prompt span is labelled, it is a masking bug, not a hyperparameter.
- **Fix:** `train_on_inputs: false` and `roles_to_train: ["assistant"]`; then confirm the rendered token string is the model's own template.
- **Trap:** raising the learning rate so the model "learns faster." It learns the prompt faster.

**3. What exactly does packing do about document boundaries, and is there a loss-curve tell?**
- **Mechanism:** a block-diagonal attention mask plus per-document `position_ids`, both derived from `cu_seqlens` — the cumulative offsets `[0, l₁, l₁+l₂, …]`. The kernel is told where each document begins; it is not asked to infer it.
- **The tell:** **there is none that is unambiguous.** Loss → ~0 / ppl → 1 and loss ~1.5–2× the unpacked baseline are both consistent with a healthy-looking run, and a *lower* packed loss is the more likely leakage signature. That is precisely why the check is an **A/B at fixed `max_steps`** with a pre-calibrated expected gap, not a threshold on the loss.
- **Trap:** asserting the packed and unpacked losses should match. They should not — packing changes the denominator.

**4. Your 4-GPU run diverges from the 1-GPU run at the same seed. Which is right?**
- **Cause:** `gradient_accumulation_steps` was held while the world size changed 4×, so the effective batch — and therefore the step count and the LR schedule — changed. The loss is not wrong; it is a different experiment.
- **Fix (a):** divide `grad_accum` by the world size. Holds the effective batch, the step count and the schedule — the right choice for a reproduction.
- **Fix (b):** keep `grad_accum` and scale the LR — √(world size) is common, linear is aggressive. Defensible when you *want* the larger batch, but it changes the optimisation dynamics as well.
- **Trap:** reporting the difference as "multi-GPU nondeterminism." It is a configuration change you made.

**5. Give the opening bid for a QLoRA SFT config on a 3B model and one 24 GB card.**
- **The bid:** `lora_r: 16`, `lora_alpha: 32`, `lora_target_linear: true`, `sequence_len` at your p99.5 rendered length (filter rows above ~900 tokens first), `micro_batch_size: 1`, `gradient_checkpointing: true`, `paged_adamw_8bit`, `gradient_accumulation_steps: 8`, `sample_packing: true` **with FA2 confirmed**, `num_epochs: 1` then 2, `val_set_size: 0.05`, `strict: true`.
- **Why these:** `r=16`/`alpha=32` fixes `alpha/r = 2`; `lora_target_linear` removes the whole "`q_proj` vs `query`" bug class; the 900-token filter is CS-17 §15.2's lesson, where truncation mid-JSON made the loss look *better*.
- **Trap:** copying the 48,000-row case study's `num_epochs: 2` and `sequence_len: 2048` onto a few hundred rows and a small card.

**6. Why can't you full-fine-tune a model you loaded in 4-bit?**
- **Mechanism:** NF4 stores an approximation; the forward pass dequantises to bf16 per matmul, so there is no differentiable path from a gradient back to the stored parameter. Full FT is priced at **~16 bytes per parameter**, which a 4-bit base has not provisioned.
- **Consequence:** the chain is **quantised base ⇒ frozen base ⇒ adapter-only**. Merging a trained adapter back to bf16 recovers a *bf16 artefact*; it does not recover *full-FT optimisation*, because every forward pass you trained through carried the 4-bit error.
- **Trap:** "QLoRA then merge equals full FT, but cheaper." It is the single most common misreading of the method.

**7. Train loss falls to 0.31 while eval loss rises from 0.9 to 1.4. Give three remedies, and say which you try first.**
- **Read it against CS-17 §12.2's bands first.** Train loss in chat SFT should fall to **0.5–2.0**; **below 0.4 is memorisation** — and 0.31 is below it. The eval-vs-train row says a *steadily widening gap* is the overfitting signal, and its prescribed action is literally the three remedies below. So the numbers name the diagnosis, and you should say so before reciting remedies.
- **Remedy (i) — stop earlier.** Keep intermediate checkpoints and select on the **best eval loss**, which was at an earlier step; CS-17 §12.2's companion gotcha is that "a `save_steps` checkpoint from mid-cosine-decay is often better than the last one," because the final checkpoint has an LR near zero.
- **Remedy (ii) — regularise.** `lora_dropout: 0.05` (or 0.1), or reduce `lora_r`. Cheap, and it directly reduces the capacity that is memorising.
- **Remedy (iii) — data.** Add rows, or rebalance: the eval set may be measuring a slice the model has memorised the train side of, which is a *split* problem wearing an overfitting costume.
- **Try (i) first**, because it is free, needs no retraining, and answers the diagnostic question — *was the model ever good?* If the best checkpoint is still bad, this is a data problem, not an epoch problem.
- **Trap:** lowering the learning rate, or adding epochs. The first does not address capacity; the second makes it strictly worse. The paired trap is operational: **know which of `save_steps` / `saves_per_epoch` is in effect when both are set.** CS-17 §4.3.9 documents both keys *as the same config's run-management block* (and §3's glossary row warns "do not set both with conflicting intent"), while §6.1's production starter sets `save_steps: 200` and `saves_per_epoch: 2` together — so a candidate who cannot say which wins cannot promise the checkpoint they just said they would select on.

**8. A DPO run on top of your SFT adapter made the model worse. What carried over?**
- **First candidate:** `sample_packing: true` left on from the SFT config. Packing across chosen/rejected pairs changes what the implicit reward compares, and it degrades the run silently (CS-17 §15.4).
- **Second candidate:** the learning rate. DPO wants **5e-6**, 10–40× below SFT's 1e-4 — carrying the SFT LR over is the other way to destroy the adapter.
- **Third:** the dataset `type:` — it must be `chat_template.default`, with `field_chosen`/`field_rejected` wired.
- **Trap:** diagnosing it as "DPO is unstable." The run is fine; the config is not.

**9. You must serve three domain variants of a 7B model on one GPU. Compare the options.**
- **Three merged 7B models ≈ 45 GB** of weights — does not fit a 24 GB card, and rollback means reloading a 15 GB artefact.
- **One base + three adapters ≈ 0.5 GB** of adapters, served by vLLM `--enable-lora` with `--max-lora-rank` and `--max-loras` set. Rollback is a module swap measured in seconds.
- **The saving is ~29 GB**, and the cost is an adapter-aware server and the discipline that **an adapter is not a model** — it is meaningless without the exact base revision and the exact chat template.
- **Trap:** comparing only VRAM. The operational difference is the rollback time and the artefact size you version.

**10. What are the five things your run record must contain, plus a bonus?**
- **The five:** (1) the YAML as committed, plus Axolotl's own `out/<run>/config.resolved.yml`; (2) the base model revision SHA and the dataset hash/revision; (3) the Axolotl version and a `pip freeze` (or a Docker digest — `main-latest` is not a pin); (4) the GPU type and count; (5) the eval scores on the frozen set.
- **The bonus:** the **seed** — without it the run is a distribution, not a point.
- **Trap:** recording the model version and calling it provenance. The base revision is the field whose absence makes an adapter unreproducible, because the adapter is a delta against *those* weights.

---

## Cross-References

| Module | Relationship to IQ-17 |
|---|---|
| **CS-17** Axolotl | The source case study. Every number in this bank is grounded in it; §19's ten questions are answered above |
| **CH-17** Axolotl Cheat Sheet | The compressed reference: §4.5 the fail-fast table, §4.4 the eight YAML traps, §7 the VRAM/cost calculator, §8 the symptom→fix lookup, §11 the exact error strings |
| **IQ-15** LLaMA-Factory | The sibling bank. CH-17 §9.2 is the key-by-key map between the two; this bank does not re-ask IQ-15's questions |
| **CS-15 / CH-15** LLaMA-Factory | The other YAML framework: a dataset *registry* instead of inline wiring, a WebUI, and a different failure profile |
| **CS-13 / CH-13** Instruction Fine-Tuning | The theory this config expresses: `-100`, prompt masking, the chat template as a serialisation contract, the epoch table |
| **IQ-13** Instruction Fine-Tuning | The masking and template questions in full; this bank assumes them |
| **CS-16** Unsloth | The single-GPU fast path — the right answer when the question is "one T4, one afternoon, one run" |
| **CS-13 §6.8 / CS-16 §6.4** LoRA configuration | The adapter mechanics behind `lora_r`, `alpha/r`, `lora_target_modules` — the config-level form of the same knobs. The rank/alpha/target-module theory in full is the planned **CS-23** (*LoRA & QLoRA: the PEFT deep dive*), which is mapped in the README but **not yet written** |
| **CS-11 §4.11** QLoRA: NF4, double quantization, paged optimizers | Why NF4 makes the base non-differentiable, and the arithmetic behind the QLoRA freeze: the 4.5 bpw single-quantization figure, `4 + 8/64 + 32/16384 = 4.127` bpw with double quantization, and the 6 GB budget for a 7B. CS-17 §19 A6 gives the contrast — "roughly 16 bytes per parameter" for full FT. For quantisation generally see CS-10 / CS-11 and CH-10 / CH-11 — **CS-06 is the Hugging Face Masterclass, not a quantisation module** |
| **Evaluation — no case study yet** | The eval layers behind CS-17 §12 and the frozen-set discipline have no home module: **CS-22 is *Embedding Models and Embedding FT*** (README Part V, not yet written), not evaluation. The nearest written material is CS-13's held-out-set discipline and `code/common/eval_utils.py` (`exact_match`, `format_compliance`, `refusal_rate`, `win_rate`, `bootstrap_ci`) |
| **Serving — `code/09_merge_and_export.py`, `code/15_serve_vllm.py`** | What happens to the adapter after training: `merge-lora`, vLLM serving, rollback. There is no production/serving case study yet — **CS-24 is *RL Fundamentals & RLHF with PPO*** (README Part VI, not yet written) |
| **CS-04** Fine-Tuning vs RAG vs Agents | Read before any of this; every `type:` question presupposes the answer to "should you be fine-tuning at all" |
| **CS-18 / CS-19** Managed fine-tuning | The opposite trade: no weights, no YAML, and a recurring bill instead of a GPU-hour |
| **`code/common/memory.py --table`** | The measured VRAM floor quoted throughout — run it, do not trust