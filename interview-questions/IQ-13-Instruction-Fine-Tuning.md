# IQ-13 — Interview Questions: Instruction Fine-Tuning (SFT)

| Field | Value |
|---|---|
| **Module** | Instruction Fine-Tuning / Supervised Fine-Tuning (SFT) — stage 2 of the three-stage lifecycle |
| **Pairs with** | CS-13 (case study), CH-13 (cheat sheet) |
| **Total questions** | 113 (28 L1 + 34 L2 + 25 L3 + 11 L4 + 15 L5) + 30 rapid-fire + 6 coding tasks + 10 CS-13 self-check answers |
| **Levels covered** | Screen (L1) / Intermediate (L2) / Advanced (L3) / System Design (L4) / Debug (L5) |
| **Source material** | Video 15 `LLM_Fine-Tuning_15_Instruction_Fine-Tuning_Explained_Domain-Specific_FineTuning.txt`; notebook `Instruction_finetuning_on_domain_specific_dataset.ipynb`; `pharma_instruction_data.jsonl` (5 rows); `code/common/memory.py --table`; `code/common/data_utils.py`; `code/data/sample_sft.jsonl` (60 rows) |

**Ground truth used throughout:** `IGNORE_INDEX = -100`, matching `torch.nn.CrossEntropyLoss(ignore_index=-100)`. VRAM figures are the repo's own measured table (`code/common/memory.py --table`), in **GiB** (1024³): 7B full FT **91.6 GiB**, LoRA r16 **14.7 GiB**, QLoRA r16 **4.9 GiB**. Those GiB figures price full FT at `4 + 2 + 8 = 14 B/param` — the repo's floor; the same run is **16 B/param** with an fp32 master copy (DeepSpeed/FSDP) and **12 B/param** without one, so quote the decomposition, never the bare total. `data_utils.token_stats()` returns `examples, tokens_total, len_p50, len_p95, len_max, with_supervision, without_supervision, supervised_token_frac`.

---

## How To Use This File

- **L1 = phone screen / recruiter filter** — 30-second answers. If you cannot answer an L1 in one breath, you will not reach L2.
- **L2 = working engineer** — 2–3 minutes, expects implementation detail: real flags, real numbers, real failure modes.
- **L3 = senior / specialist** — 5 minutes, expects internals, derivations, and trade-offs.
- **L4 = staff / system design** — 15-minute whiteboard. Structure every answer as: requirements → constraints → design → trade-offs → failure modes.
- **L5 = debugging & incident response** — "your loss does X, what do you check and in what order." Answer with a *sequence*, not a list.

**The meta-rule:** an answer of "it depends" is a failure unless it immediately says what it depends on and then picks a default. Always pick the default.

**The three answers that get people hired in this module:** (1) *mask the prompt and the padding, not just the prompt*; (2) *the chat template at train time must be byte-identical to the one at serve time*; (3) *SFT buys behaviour, not facts — if you need facts it is RAG or continued pretraining.* If you can only remember three things, remember those.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. What is supervised fine-tuning, in one sentence?**

- **Answer:** SFT is next-token prediction with cross-entropy, run on `(prompt, response)` pairs, with the prompt's tokens **excluded from the loss**. The objective is identical to pretraining; only the data distribution and the supervision mask change. Mechanically: `L_SFT = −Σ_{t ∈ R} log p_θ(x_t | x_<t)` where `R` is the set of response token positions. The instructor's framing is exactly right and worth quoting: *"the model will always predict the next token in whatever way we have a data"* [11:44] — SFT is not a new objective, it is a new data distribution for the same one.
- **Why asked:** This is the sentence the entire module hangs on. A candidate who describes SFT as "a new loss" or as "teaching the model facts" has the wrong mental model and will make the wrong architectural decisions downstream.
- **Trap:** Saying SFT is "training on questions and answers." That describes the *data*; the technique is the *mask*. A run that trains on the whole string is still called SFT by the person who wrote it, and it is a different task (prompt-completion).

**Q2. What is the difference between a base model and an instruct model, and why does it change your SFT plan?**

- **Answer:** A **base** model is pretrained only — no chat template, no notion of "assistant"; asked a question it continues the text pattern (a base Llama asked "List three contraindications for metformin" replies "…is a common question asked by patients who…"). An **instruct** model is the base plus SFT (usually plus preference tuning) and ships a `chat_template` and an end-of-turn token. It changes the plan in four ways: (1) an instruct base needs **less data** (the template is already learned) and **lower LR** (it is already near a good optimum — halve it); (2) an instruct base carries an existing refusal prior that over-refusal amplification will sharpen; (3) a base model needs 2–3 epochs where an instruct model needs 1–2; (4) you must not train with a template the base has never seen *unless* you accept teaching the markers from scratch — the notebook does exactly this by training a base TinyLlama on markers it never saw.
- **Why asked:** Fine-tuning the wrong checkpoint is one of the two or three most expensive setup errors in the field, and it is invisible for the first hour.
- **Trap:** "They're the same weights, just different prompts." The weights differ, the tokenizer's special-token set can differ, and the template is part of the model's contract.

**Q3. What does `-100` mean, and why is that number everywhere?**

- **Answer:** `-100` is the default value of `ignore_index` in `torch.nn.CrossEntropyLoss`. Any target position equal to it is dropped from **both the numerator and the denominator** of the mean — it contributes nothing to the loss and nothing to the gradient. In SFT you set `labels[prompt_positions] = -100` so that only the assistant's tokens are graded. The `-100` lives in the **labels** tensor only; the corresponding `input_ids` are untouched, so attention still runs over the full sequence and the model still *reads* the prompt — it is simply not *graded* on producing it. The value is a convention inherited from PyTorch's `NLLLoss`; any sentinel works as long as you pass `ignore_index` consistently.
- **Why asked:** The single highest-leverage line of code in the module. If a candidate cannot say what `-100` does to the denominator, they have never audited a loss.
- **Trap:** "It masks the tokens." It masks the *labels*, not the inputs. Believing otherwise leads people to think the prompt is invisible to the model, and then to "fix" things that are not broken.

**Q4. Why is `-100` = `IGNORE_INDEX` and not `-1` or `0`?**

- **Answer:** Because `0` and `-1` are both plausible token ids (id `0` is often `<unk>` or `<pad>`; negative ids do not exist but `-1` is a common "N/A" sentinel in other libraries), and a sentinel that collides with a real id would silently train on a wrong target. `-100` cannot be a valid token id, and it is the PyTorch default, so every framework that constructs a labels tensor — `trl`, `peft` examples, `stanford_alpaca`'s original script — uses the same constant. This repo pins `IGNORE_INDEX = -100` in `code/common/data_utils.py` and every builder (`build_masked_example`) uses it.
- **Why asked:** Tests whether the candidate knows the constant is a *default* they could change rather than a magic number. It also surfaces people who have written `ignore_index` inconsistently across train and eval.
- **Trap:** Using `-100` in `input_ids` (which raises an embedding index error) or using `-1` in labels with a loss that does not set `ignore_index=-1` (which produces a silently wrong loss, not a crash).

**Q5. What fraction of a typical SFT row's tokens are supervised?**

- **Answer:** **10–40%**, and under 10% is a red flag. Worked example from the repo's own data: a 512-token row of 400 prompt + 112 answer is 22% supervised. The tool that measures it is `data_utils.token_stats(ex)`, which returns `with_supervision`, `without_supervision`, and `supervised_token_frac` over a built dataset. `code/data/sample_sft.jsonl` has 60 rows; run `token_stats` over `build_masked_example(tok, m)` for each of them *before* training, not after.
- **Why asked:** This is the cheapest single number that tells you whether your masking is right. A candidate who has never computed it has been training blind.
- **Trap:** Confusing it with "the fraction of tokens that are padding." They interact — `padding="max_length"` + unmasked labels drives the supervised fraction *toward 100%* while driving the useful fraction toward 6%, which is exactly the notebook's bug. A high supervised fraction is not automatically good.

**Q6. What is a chat template, and why is it the most dangerous object in the pipeline?**

- **Answer:** A chat template is a Jinja2 program stored in `tokenizer_config.json` under `chat_template` that serialises a message list into the exact byte string the model was trained on: `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n` (ChatML), `<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>` (Llama-3), `[INST] … [/INST]` (Mistral/Llama-2), `<start_of_turn>…<end_of_turn>` (Gemma). It is dangerous because a mismatch between the training string and the serving string is **silent**: no exception, no warning, a loss curve that looks fine, and a model that is unusable behind the API. The role markers are *conditioning tokens* — if the model only ever saw `<|im_start|>assistant\n` before an answer, that sequence is the cue to answer, and removing it removes the behaviour.
- **Why asked:** CS-13 and CH-13 both call this the #1 silent quality killer. It is the highest-signal practical question in the bank.
- **Trap:** "The tokenizer handles it." Only if you call `tokenizer.apply_chat_template(...)` — the mere act of loading a tokenizer does nothing — and only with the template that belongs to *your* checkpoint, not the one you copied from a tutorial.

**Q7. What are the learning rates for LoRA, full fine-tuning, and encoder fine-tuning?**

- **Answer:** LoRA/QLoRA: **1e-4 to 2e-4** (2e-4 is the common default; this repo, CH-13, and the notebook all use 2e-4). Full fine-tuning of a decoder LM: **1e-5 to 2e-5** — 10x lower. Encoder fine-tuning (BERT-class classification): **2e-5**, not 2e-4 (CH-07). The 10x gap exists because LoRA's adapters initialise at zero and only move a low-rank delta whose output is scaled by `alpha/r`, so there is no pretrained structure to destroy; a 2e-4 update to fully pretrained weights erases the pretraining distribution inside ~100 steps. Instruct (already-aligned) bases want **half** again: 5e-6–1e-5 full FT, 5e-5–1e-4 LoRA.
- **Why asked:** The most common configuration error in practice, and the one that produces "the fine-tune made it worse" reports.
- **Trap:** Using 2e-4 with full fine-tuning and blaming the data. The loss will spike toward NaN around step 50 and the model will forget everything.

**Q8. How many epochs do you train for, and what is the stop rule?**

- **Answer:** **1–3 epochs**; 2 is the default answer. 1 epoch when you have >20k rows or are fine-tuning an already-instruct model; 2 for 1k–10k rows; 3 is the maximum for domain SFT and only with replay data plus a general-capability eval. Above 3 epochs is a red flag in a code review. The stop rule is **not** train loss — it is whichever checkpoint maximises a composite of format compliance, refusal rate and a general benchmark, because SFT quality decouples from loss after roughly the end of epoch 1. The rule of thumb: stop at ~2–10k example-views per distinct row for small datasets, 1–2 views for large ones.
- **Why asked:** Overfitting on small SFT sets is the most common quality complaint, and "how many epochs" is the knob people copy from tutorials.
- **Trap:** Carrying over the classical-ML habit of 50 epochs, or trusting `load_best_model_at_end` on `eval_loss` alone — in SFT a *flat* eval loss with a falling train loss is as bad as a rising one.

**Q9. What is catastrophic forgetting, and how do you prevent it?**

- **Answer:** Fine-tuning on a narrow distribution degrades capabilities the base had — general reasoning, other languages, instruction following, safety behaviour — because gradient descent moves weights toward the narrow objective with no term preserving prior behaviour. Measured cost is real: a typical 5-epoch narrow run costs **MMLU −6 to −14 points**. Mitigations, in order of effectiveness: (1) **replay** — mix 5–20% general instruction data (or 5–10% raw pretraining text, which must **not** be masked) into the SFT set; (2) **fewer epochs**; (3) **LoRA instead of full FT** (the base is frozen); (4) **lower LR**; (5) early stopping on a composite domain+general metric. The correct order to apply them is replay > epochs > LoRA > LR.
- **Why asked:** It is the failure mode nobody measures and the one that causes incidents; the mitigation ordering is the part most candidates get wrong.
- **Trap:** "Use LoRA and you're safe." LoRA reduces forgetting, it does not eliminate it — multi-epoch LoRA on narrow data still drops general benchmarks by 2–5 points, because the adapter's output is added to every forward pass and shifts the shared representation.

**Q10. LIMA says 1,000 examples; Alpaca used 52,000. Which is right?**

- **Answer:** Both, for different claims. **LIMA** (Meta, 2023) fine-tuned 1,000 *curated* prompt/response pairs onto LLaMA-65B with no RLHF and reached 43% win/tie against GPT-4, 58% against Bard, 65% against Alpaca-65B. **Alpaca** was 52,000 rows generated by `text-davinci-003` from 175 seeds for ~$500 — the canonical cheap recipe, not the quality ceiling. The reconciliation: LIMA does not say "use 1,000", it says *for teaching format and behaviour to a model that already has the capability, curated small data beats uncurated large data*. AlpaGasus filtered Alpaca's 52k to 9k using GPT-4 quality scores and **beat the full 52k on 5 of 6 benchmarks**; Deita's 6k beat 100k+ baselines. Past ~10k rows you are no longer buying behaviour, you are buying coverage and robustness.
- **Why asked:** Candidates quote "1,000 examples" as if it were a hyperparameter, and interviewers use this question to see whether they understand *why*.
- **Trap:** Concluding "data doesn't matter, use 1,000 rows." LIMA's 1,000 were deliberately diverse — 750 community Q&A, 200 authored examples from the model's own failures, 50 from Super-NaturalInstructions — and strictly filtered. 1,000 random rows produce a template-follower.

**Q11. Is SFT the right tool for making the model know your company's facts?**

- **Answer:** No. SFT teaches **behaviour and form** — format, tone, register, schema, refusal policy, stopping. It injects facts only weakly, expensively and unreliably: it takes roughly **10–100 paraphrases per fact** to stick, recall is brittle under rephrasing, and a later preference-tuning run frequently erases it. For facts use **RAG** (CS-04); for domain *terminology* and vocabulary use domain-adaptive continued pretraining (CS-12); for preference over two answers use DPO/ORPO (CS-14). CS-13's decision rule is the answer to recite: *"Do you need the model to behave differently? → SFT. Do you need it to know different things? → continued pretraining for terminology, RAG for retrievable facts."*
- **Why asked:** This is the architecture question that precedes every other decision, and the canonical failed project ("fine-tune on our docs to stop hallucinating") is the one that gets candidates rejected.
- **Trap:** Pointing at a model that memorised a fact. At `r=128` and 20 epochs on 200 examples memorisation is possible — and it is a bug, not a capability. It does not survive a differently-worded question.

**Q12. When should you use structured outputs / constrained decoding instead of SFT?**

- **Answer:** When the requirement is **format-only** — a fixed JSON schema, an enum, a grammar — with no reasoning or content change. Constrained decoding (grammar-based samplers, `outlines`, `guidance`, JSON-schema logit masks, or OpenAI's `response_format`) is deterministic, cannot be forgotten by a later training run, needs no data, and costs nothing at training time. Fine-tune only when the **content** also has to change (the model must know which urgency to assign, not just how to spell the field). The honest hybrid, which is what CS-13's insurance case study actually did: SFT for the content decision + a schema validator and retry on the output. SFT took that project's JSON validity from 88% → 99.6%; constrained decoding would have taken the *format* to 100% but left the *urgency accuracy* at 71%.
- **Why asked:** It is the cheapest decision in the module and the one most often skipped, because fine-tuning is more interesting than `json.loads`.
- **Trap:** "We need 100% valid JSON so we must fine-tune." You need 100% valid JSON so you must constrain the decoder; fine-tuning gets you to 99.5% and a validator gets you the rest.

**Q13. What is NEFTune, in one sentence, and what does it cost at inference?**

- **Answer:** NEFTune adds uniform noise `u ~ U(−α/√(L·d), α/√(L·d))` (`L` = sequence length, `d` = embedding dim, `α` = `neftune_noise_alpha`, default **5**) to the **embedding output on every forward pass during training only**. Inference is completely unchanged — no runtime cost, no architecture change, no extra data, no change to the served model. Reported effect (Jain et al., 2023): **+29.8% on AlpacaEval for LLaMA-2-7B and +8.7% for 13B**, with no measured loss on MMLU/GSM8K. It works by regularising the embedding space so the model cannot memorise exact token sequences — which is precisely the format-rigidity failure SFT produces.
- **Why asked:** It is the highest value-per-line-of-code change in the module, and the "does it cost anything at inference" follow-up separates people who read the flag from people who read the paper.
- **Trap:** Enabling it at inference, or expecting it to help everywhere — Tulu-3-scale ablations found it neutral-to-negative on large clean mixtures, and it perturbs semantic content on long documents. Treat it as a cheap experiment, not a mandatory flag.

**Q14. LoRA vs QLoRA vs full fine-tuning — one line each.**

- **Answer:** **Full FT** updates every weight: highest quality ceiling, ~16 bytes/parameter of optimiser+weight state *with an fp32 master copy* (`2 bf16 weights + 2 bf16 grad + 8 fp32 AdamW m,v + 4 master`; **12** without the master, **14** on this repo's floor), which is why it needs ≥50k examples and multi-GPU to be worth it. The repo's table prices 7B at **91.6 GiB** — that is the 14 B/param floor, not the 16 B/param figure; at 16 it would be 104.3 GiB. **LoRA** freezes the base and trains a low-rank delta `ΔW = BA` on all linear projections: ~0.3–1% of parameters, 7B ≈ **14.7 GiB**, 95–99% of full-FT quality, mergeable losslessly, and it is the default below ~50k rows. **QLoRA** is LoRA on top of a 4-bit NF4 base with double quantization and paged optimisers: 7B ≈ **4.9 GiB**, which is the difference between "needs an A100" and "runs on a 12 GB card"; it costs ~20–40% more wall-clock per step and 1–3 quality points versus LoRA. The decision order is: QLoRA to prove the pipeline and the data → LoRA when the data is proven → full FT only if you have demonstrated LoRA is the bottleneck.
- **Why asked:** The most-asked PEFT question in existence, and the one where candidates reveal whether they have shipped or only prototyped.
- **Trap:** "QLoRA is a quantization method." It is a *training* method that uses quantization as one component, and its 4-bit base cannot represent the fine distinctions that knowledge injection requires.

> **Correction:** this answer used to attach the table's 7B figure (**91.6 GiB**) directly to **16 B/param**. Both numbers are right; they are not the same configuration. `91.6 GiB` is 7B × **14 B/param** — the repo's floor, `4 + 2 + 8` — and `16 B/param` is the same run under DeepSpeed/FSDP with an fp32 master copy, which is **104.3 GiB** for 7B. `python code/common/memory.py --table` states the three conventions explicitly (`12` = `2 + 2 + 8`, no master; `14` = the repo's middle ground; `16` = with master) and is the authority for every VRAM figure in this bank. Quote the decomposition with the number, or a reader comparing this file to the table will see a contradiction that is not one.

**Q15. What is the effective batch size, and what units do you state it in?**

- **Answer:** `effective_batch = per_device_train_batch_size × gradient_accumulation_steps × n_gpus`. For SFT the number that actually matters is **tokens per optimizer step**: `micro_batch × max_seq_length × grad_accum × n_gpus`. The useful band is **32k–128k tokens/step**; the notebook's `1 × 512 × 8 × 1 = 4,096` tokens/step is far below it and its gradient is correspondingly noisy. Examples/step is a useless unit when sequence lengths vary 10x — "batch size 16" means 640 tokens in one dataset and 62,000 in another.
- **Why asked:** Config review. People routinely report a micro-batch as "the batch size", then wonder why two runs with "the same batch size" behave differently on different hardware.
- **Trap:** Forgetting that changing `gradient_accumulation_steps` changes the *number of optimizer steps*, and therefore the warmup length and the cosine schedule — schedules are computed after accumulation.

**Q16. What is gradient accumulation and what does it cost?**

- **Answer:** Run N micro-batches, sum their gradients, take one optimizer step. The GPU only ever holds one micro-batch's activations, so it decouples batch size from VRAM. It costs **no extra FLOPs** and it makes each *step* N times slower in wall clock (total wall clock is roughly unchanged). Two consequences that bite: the effective LR should be tuned for the *global* batch, not the micro-batch; and every scheduler-shaped hyperparameter (`num_training_steps`, `warmup_ratio`, cosine horizon) is computed after accumulation, so raising `grad_accum` from 8 to 32 silently shortens your warmup proportionally.
- **Why asked:** The standard VRAM workaround; tests whether the candidate understands what it buys and what it does not.
- **Trap:** "It's free." It is free in compute, expensive in wall clock, and it changes the optimisation dynamics unless you adjust the LR (roughly `LR ∝ √batch`).

**Q17. What is packing, and what is the trap?**

- **Answer:** Packing concatenates multiple short examples into one `max_seq_length` window so no compute is spent on pad tokens — typically a **2–5x throughput** win on short-response data. The trap is **cross-contamination**: without per-sequence position-id resets, token 3 of example B can attend to example A, because a plain causal mask does not know where one example ends. The fix is per-sample `position_ids` restarting at 0 for each packed sequence; FlashAttention-2's `flash_attn_varlen_func` uses `cu_seqlens` derived from position-id discontinuities to build a block-diagonal mask, and HF exposes the pair through `DataCollatorWithFlattening(return_position_ids=True, return_flash_attn_kwargs=True)`. The failure signature is nasty: loss looks *better* than it should, offline eval looks fine on single-turn prompts, and production is erratic on anything long.
- **Why asked:** CS-13 flags this as a silent-corruption trap; CH-13 lists it as gotcha #4. A candidate who enables packing "for speed" without mentioning position ids has not read the docs.
- **Trap:** Enabling `packing=True` for SFT and shipping a model that answers the previous user's question — or, subtler, one whose loss was artificially low because it was getting free context.

**Q18. What is the supervised token fraction, and what does a very low value mean?**

- **Answer:** `supervised_token_frac = n_supervised / n_total` over the dataset — the share of token positions whose labels are not `-100`. Typical healthy range is **10–40%**; below ~10% is a red flag. A low value means one of four things, in order of likelihood: (1) your prompts are much longer than your answers (long retrieved context + a short answer is >95% prompt by construction, which is expected but means each step teaches you little); (2) `max_seq_length` is large and you are not packing, so most supervised positions are **padding** — the notebook's run supervises 32 of 512 positions, 6.3%; (3) truncation is cutting into the answer so rows have few or zero supervised tokens; (4) some rows are **entirely** masked, which contributes `0/0` to the loss mean and is `NaN` on many `transformers` versions. Measure it with `data_utils.token_stats(ex)` and check `without_supervision` — that counter exists precisely to catch case (4).
- **Why asked:** It is the cheapest health metric in the pipeline and it catches three of the four silent failures at once.
- **Trap:** Reporting a *high* supervised fraction as success. With `padding="max_length"` and `labels = input_ids.copy()`, the fraction goes to ~100% while the useful supervision falls to 6% — the metric improves as the pipeline breaks.

**Q19. Why must you append EOS to every training target?**

- **Answer:** The EOS token (or the template's end-of-turn token — `</s>`, `<|eot_id|>`, `<|im_end|>`, `<end_of_turn>`) is what teaches the model to **stop**. If the response span ends with content and no EOS, the model learns `p(next | answer)` with no terminating mass and at inference it rambles until `max_new_tokens`. It is common and it is silent: the loss falls, the answers look right for the first 50 tokens, and the model never yields the turn. Two ways it happens: rows built by string concatenation without the terminator, and a tokenizer/collator step that strips trailing special tokens "to avoid double tokens".
- **Why asked:** This is the direct cause of the most-reported production symptom ("my fine-tune never stops generating"), and the diagnostic is a one-line check.
- **Trap:** Fixing it with `repetition_penalty` or a stop string in the server. That masks a data bug and degrades legitimate repetition in JSON, lists and code.

**Q20. What is the difference between the Alpaca, ShareGPT and OpenAI `messages` data formats?**

- **Answer:** All three encode the same information; they differ in shape. **Alpaca**: `{"instruction", "input", "output"}` — single-turn only, `input` is context appended to the instruction and is frequently the empty string (not missing). **ShareGPT**: `{"conversations": [{"from": "human"|"gpt", "value": ...}]}` — natively multi-turn, legacy role names. **OpenAI chat**: `{"messages": [{"role": "system"|"user"|"assistant", "content": ...}]}` — what `apply_chat_template`, TRL's `SFTTrainer` and Unsloth consume natively, and the format to build in. The rule from CS-13: **store in `messages`, convert on export.** One canonical internal schema plus one converter per framework, because every dataset bug in production SFT traces to three teams each inventing their own intermediate JSON.
- **Why asked:** Every framework speaks a different dialect and every dataset on the Hub has its own column names (`Context`/`Response` with capitals in the video's demo dataset). The candidate needs to know which is which without looking it up.
- **Trap:** Treating the format and the chat template as the same thing. A format is a container; a template is a serialisation contract with the weights. Converting the data correctly and then rendering it with the wrong template is the classic failure.

**Q21. How do you decide between SFT and DPO?**

- **Answer:** SFT needs `(prompt, response)` demonstrations and teaches *what to do*; DPO needs `(prompt, chosen, rejected)` pairs and teaches *which of two outputs is preferred*. Decision: if you can write the ideal answer, use SFT; if you can only say "this one is better than that one", use DPO; if you need both, do SFT first — DPO's gains *require* a competent SFT checkpoint underneath, because preference tuning on a raw base model consistently underperforms SFT-first. Magnitude matters: SFT is roughly **90% of the achievable value** with format compliance moving 40% → 95%+, while typical DPO gains over SFT are 0.2–0.5 points on MT-Bench out of 10 and 5–15 win-rate points on AlpacaEval. One more reason to capture pairs *while* curating SFT data: if you are already paying an expert to write the good answer, pay 20% more for a plausible-but-wrong answer in the same session — that single decision saves a complete relabelling project later. ORPO folds both objectives into one loss and is attractive when you have pairs but no clean SFT set.
- **Why asked:** The "SFT then what" question decides the shape of a project, and interviewers listen for whether the candidate over-invests in the fashionable stage.
- **Trap:** "DPO is where the quality comes from, so skip SFT." Preference pairs are usually stylistic; without a competent SFT base both completions are bad and one bit per pair teaches almost nothing — and DPO can actively *undo* facts SFT installed.

**Q22. What is `max_seq_length` and how do you choose it?**

- **Answer:** The truncation window for the *rendered* example. Set it from the **p99.5 of your own tokenised lengths**, not from the model's context window and not "8192 because it fits". Truncation is the quietest catastrophic bug in SFT: with prompt masking, a row cut before the answer has few or zero supervised tokens (zero → NaN on some versions, few → the model is trained to stop mid-sentence); without masking, the model is trained on a document that just ends, which is harmless-looking and therefore worse. Measure it: `python -c "..."` over `len(tokenizer(render(r)).input_ids)` at p50/p90/p95/p99/99.5/max, and separately count rows where the *prompt alone* already exceeds the window — that is the count that actually matters. Target less than **0.5%** of rows losing their response.
- **Why asked:** It is invisible in the loss curve, cheap to check, and most people set it by copying a tutorial's 512 or 2048.
- **Trap:** `truncation=True, max_length=512` with `tokenization_side="right"` on Alpaca-formatted data — this deletes the labels and produces a model that emits fragments of answers. Note that the notebook's `padding="max_length", max_length=512` wastes **89%** of every forward pass on a 57-token mean row.

**Q23. What is the loss at the start of training, and why is `ln(vocab)` useful?**

- **Answer:** A model outputting a flat distribution over `V` tokens has loss `ln(V)`: **10.37** for a 32k vocabulary, **11.76** for Llama-3's 128,256. It is useful as a sanity anchor: if your loss sits exactly at `ln(V)` and never moves, the model is learning nothing — almost always because every label is `-100`, the labels tensor is not reaching the loss, or the LR is zero. If your loss starts *well below* `ln(V)` on step 0, that is also a signal: the prompt is probably unmasked (prompt tokens are highly predictable) or your eval set is contaminated. The initial loss for a randomly-initialised *classification head* on `K` classes is `ln(K)` for the same reason.
- **Why asked:** It is the one diagnostic that requires no plotting and no comparison run, and it discriminates two opposite bugs.
- **Trap:** Confusing "loss lower than `ln(V)`" with "the model is good." It means the model finds the data easy, which for a masked-prompt SFT run at step 0 usually means the mask is wrong.

**Q24. Roughly how much VRAM does it take to fine-tune a 7B model, by method?**

- **Answer:** From this repo's own table (`code/common/memory.py --table`, single GPU, gradient checkpointing on, AdamW, bf16 weights, fp32 optimiser states, **GiB**): full FT **91.6 GiB**, LoRA r16 **14.7 GiB**, QLoRA r16 **4.9 GiB**; inference-only bf16 is 16.0 GiB and 4-bit inference is 4.8 GiB. Add 10–20% for allocator and framework overhead and leave headroom. That is the reason LoRA and QLoRA exist: full FT is ~**19x** QLoRA's footprint, and the ratio is driven almost entirely by AdamW's optimiser state (**8 bytes per trainable parameter**, two fp32 moments) plus bf16 gradients (2) on top of the weights — `4 + 2 + 8 = 14 B/param` as the table prices it, `2 + 2 + 8 = 12` in a plain bf16 loop with no fp32 master copy.
- **Why asked:** Every practical decision in fine-tuning is downstream of a VRAM budget, and these three numbers are the ones to have on instant recall.
- **Trap:** Sizing from the parameter count in bf16 alone (7B = 14 GB = **13.0 GiB**) and concluding it fits a 24 GB card for *training*. 14 GB is the inference number; the optimiser state is another 56 GB = **52.2 GiB** (`8 B/param` fp32 `m`,`v`).

**Q25. What is gradient checkpointing, and is it a speed optimisation?**

- **Answer:** Instead of storing every intermediate activation for the backward pass, store each block's *input* and recompute the block's internals during backward. It cuts activation memory by roughly **50–70%** (CS-13's VRAM formula uses a ~5x reduction on the activation term) at a cost of roughly **+25–33% step time**. It is a **memory** optimisation and it is almost always the correct trade for SFT above ~1B parameters — CH-13's hyperparameter table lists it as "usually mandatory".
- **Why asked:** It is misquoted as a speed trick in roughly half of candidate answers, and it is on by default in every good config.
- **Trap:** Enabling it expecting faster training, then misattributing the slowdown to the data loader or the optimizer.

**Q26. What is `warmup_ratio` and what value do you use?**

- **Answer:** The fraction of total steps spent ramping the LR linearly from ~0 to peak, typically **0.03–0.10** (3–10%) for SFT. It exists because AdamW's second-moment estimate `v` is computed from very few samples in the first steps, making the effective step `η/√v̂` large and noisy — without warmup, LLM runs frequently diverge in the first few hundred steps. **The degenerate case matters more than the value:** schedule-shaped hyperparameters are meaningless below ~200 optimizer steps. With 3 total steps (the notebook's run) a cosine schedule is nearly constant and a 5% warmup is 0 steps. For short runs set `warmup_steps` explicitly: `warmup_steps = max(10, int(0.05 * num_steps))`.
- **Why asked:** It shows whether the candidate knows which knobs are *inert* at their scale — a genuinely senior signal.
- **Trap:** Setting `warmup_ratio=0.5` on a 60-step run, which means half the run is at a suboptimal LR; or adding warmup to fix a divergence that is actually caused by an LR 10x too high.

**Q27. What is `train_on_responses_only` / `assistant_only_loss` / `train_on_prompt: false`?**

- **Answer:** They are the same feature under four names: mask everything that is not an assistant span. TRL (≥0.12) exposes `SFTConfig(assistant_only_loss=True)`, which builds the mask from the `{% generation %}` markers in the chat template — the only method that is correct for multi-turn, tool calls, and templates that place a generation header inside a turn. Unsloth's `train_on_responses_only(trainer, instruction_part=..., response_part=...)` string-matches the template instead, so it works with templates that lack generation tags. LLaMA-Factory uses `train_on_prompt: false`; Axolotl uses `train_on_inputs: false` with `roles_to_train: ["assistant"]`.
- **Why asked:** It separates people who write collators from people who write configs — and the interviewer's follow-up is always "what happens when the template lacks the generation tag?"
- **Trap:** Assuming it works. If the template has no `{% generation %}` marker, recent TRL raises `ValueError: The chat template ... does not contain the generation tag`, but older versions **warned and silently did not mask** — a silent failure with a quiet log line. Pin your TRL version and assert that the first non-`-100` label is the first response token.

**Q28. What is the single best piece of advice you would give someone starting their first SFT run?**

- **Answer:** Build the eval set and the well-prompted baseline **before** you train, and print the rendered chat-template string and the masked label span **before** you spend a GPU-hour. Without a baseline you cannot tell whether fine-tuning helped; without a held-out set you cannot tell whether it helped for the right reason; without reading the rendered string you cannot tell whether the training data is even in the format you think it is. The second-best advice: QLoRA, 1 epoch, and only then consider touching a hyperparameter. Together those prevent the four most common failures — no measurement, wrong template, wrong mask, wrong LR.
- **Why asked:** A culture question. It reveals what someone learned the hard way, and the answer is diagnostic of whether they have owned a project or only a notebook.
- **Trap:** "Start with full fine-tuning for the best quality." You cannot afford it, and you do not yet know what quality means for your task.

---

## Level 2 — Applied & Implementation

**Q29. Walk me through your end-to-end SFT pipeline.**

- **Answer:** (1) **Decide whether SFT is even the tool** — if you need facts, stop and build RAG; if you need preference, plan for DPO. (2) **Corpus prep** — dedup, PII scrub, language filter, chunk, with a provenance log. (3) **Continued pretraining** (CS-12) if the domain vocabulary is absent from the base. (4) **Data curation** — author or synthesise 1k–20k rows, filter, score, balance, freeze with a version and a content hash. (5) **Template + masking contract** — pick the model's own template, render it, mask prompt+header+padding, assert `attention_mask == 0 ⇒ labels == -100`, and version the `.jinja` file. (6) **Train** — QLoRA or LoRA, 1–3 epochs, LR 1e-4–2e-4, prompt masked, NEFTune on, wandb on. (7) **Evaluate** — format compliance, refusal rate, length drift, swap-corrected win rate, and a general-capability delta. (8) **Merge, quantise, re-evaluate the quantised artefact.** (9) **Serve with the same template**, with version pinning and a regression gate. (10) **Mine production failures back into the next dataset version.**
- **Why asked:** Tests whether the candidate has owned a project or just a training script. The most common real failure is starting at step 6.
- **Trap:** Presenting the training loop as the pipeline. CS-13's own cost table makes the point numerically: GPU time for an 8k-row run is **~$0.85**, human review is **$300**, and a 2,000-row expert dataset at 20 minutes a row is **~$33,000**. The GPU is not the cost.

**Q30. Write the masking code. Where exactly does the mask boundary go?**

- **Answer:** Three components, and the boundary is at the **assistant header, inclusive**:
  1. **Prompt + assistant header → `-100`.** Everything up to and including `<|im_start|>assistant\n` (or `<|start_header_id|>assistant<|end_header_id|>\n\n`) is masked. The first *trained* label is then the first response token, predicted from the last header token — which is exactly right, because of the causal shift: `logits[t]` predicts `labels[t+1]`.
  2. **Response + EOS → unmasked.** The EOS is part of the target, and it is what teaches stopping.
  3. **Padding → `-100`.** `attention_mask == 0` must imply `labels == -100`. This is the step almost everyone misses.
- **Why asked:** It is the highest-leverage code in the module written on a whiteboard, and the header-inclusion decision is the part that is genuinely subtle.
- **Trap:** Masking *through* the header is a real choice, not a bug — CS-13 §2.3 shows header-masked rows supervising 29 tokens against 32 for prompt-masked. Masking the header means the model is never trained to emit the header it just received, which removes the "double header" failure (`### Response:` emitted twice). Not masking it costs you four tokens of a task you do not want. Mask it.

**Q31. Why can you not compare the loss value of two SFT runs?**

- **Answer:** Because `-100` positions are removed from the **denominator** as well as the numerator. The loss is a mean over non-ignored positions, so changing the mask changes both what is learned and the *scale* of the reported number. Two runs on identical data — one prompt-masked, one unmasked — have losses that differ in kind, not in degree; the unmasked run's number is dragged down by highly predictable prompt tokens and looks better. The same applies between a train split and an eval split with different masking policies, and between a padded and a packed run of the same data.
- **Why asked:** CS-13 lists "loss averaged over a different denominator across runs" as silent failure #10. Interviewers ask it to see whether the candidate treats loss as a metric or as a number.
- **Trap:** "Run B has lower loss so it's better." Compare **task metrics** across runs; use loss only to detect divergence, NaN and within-run checkpoint selection.

**Q32. How do you compute and set the effective batch for a small GPU?**

- **Answer:** Target **32k–128k tokens per optimizer step**. Worked example on a 24 GB card: `per_device_train_batch_size=2`, `max_seq_length=2048`, `gradient_accumulation_steps=8`, 1 GPU → `2 × 2048 × 8 = 32,768` tokens/step, which is the workable minimum. On an A100 you would run `4 × 2048 × 8 = 65,536`. The notebook's `1 × 512 × 8 = 4,096` is below the floor: gradients are very noisy and the loss curve is a sawtooth. If you increase tokens/step by 4x, increase the LR by roughly **2x** (square-root scaling), not 4x — the notebook's `2e-4` at 4,096 tokens/step is a valid small-scale point and would be unstable at 64x the batch.
- **Why asked:** Batch sizing is where small-hardware practitioners and paper-readers diverge.
- **Trap:** Reporting `per_device_train_batch_size` as "the batch size", then being unable to explain a run-to-run difference that is actually a difference in accumulation.

**Q33. Your dataset has 5,000 rows and you have one 24 GB GPU. What is your config?**

- **Answer:** `Llama-3.1-8B-Instruct` (or `Qwen2.5-7B-Instruct`) + QLoRA (LoRA if it fits) `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules="all-linear"` (q,k,v,o,gate,up,down), `bias="none"`; LR **1e-4–2e-4** with cosine and `warmup_ratio=0.03`; **2 epochs**; `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, `max_length=` p99.5 of your tokenised rows (measure first, ~1024–2048); `bf16=True` (not fp16 — no loss scaling, no NaN); `gradient_checkpointing=True`; `optim="paged_adamw_8bit"`; `neftune_noise_alpha=5`; prompt masked (`assistant_only_loss=True`); `logging_steps=5`, `eval_strategy="steps"`, `save_total_limit=2`, `report_to="wandb"`, `seed=42`, `data_seed=42`. Then the five pre-flight checks from CH-13 §12: read the rendered template, count `supervised_token_frac` (want 10–40%), run `memory.py --table`, compute steps and wall clock, and compare p95 length against `max_seq_length`.
- **Why asked:** A config-review question. The interviewer is listening for the *order* — measure, then configure — and for the two flags people forget (`bf16`, prompt masking).
- **Trap:** Copying the defaults from a tutorial's 52k-row recipe onto 5,000 rows at 3 epochs and LR 2e-4, then reporting verbosity and rigid formatting as "the model is just like that."

**Q34. How do you handle multi-turn conversations in SFT?**

- **Answer:** "The response" means **every** assistant turn, not just the last one. Masking only the final `gpt` turn discards most of your supervision; masking nothing teaches the model to write the user's messages. Two correct routes: `assistant_only_loss=True` (TRL), which uses the `{% generation %}` markers per turn, or `train_on_responses_only` (Unsloth), which derives the boundary from the template text. Then verify by printing the labels of a multi-turn row and confirming that each assistant span is unmasked and each user span is `-100`. Two further details: put a **system turn in a meaningful fraction of rows** (otherwise the model never learns to condition on the role and will ignore your system prompt at serving time — CS-13's insurance case study used a deliberate 60/40 split for exactly this reason), and include ≥20% multi-turn rows in any conversational dataset, because a model trained mostly on short conversations loses the thread.
- **Why asked:** Multi-turn masking is the single most common agent-SFT bug (CS-13 §15.4 case 1: masking only user turns trained the model to predict its own tool's JSON output — it started hallucinating fake API responses).
- **Trap:** Using the single-turn manual mask (mask everything before the last `### Response:`) on ShareGPT data. It silently trains on user turns.

**Q35. What are the four overfitting signatures in SFT, and why does validation loss not catch them?**

- **Answer:** (1) **Verbosity** — response length drifts up 30–80% over a long run while task accuracy stays flat; the model is rewarded for high-probability filler. (2) **Format rigidity** — it works on prompts that exactly match the training template and degrades on trivial mutations (extra system message, `Summarize` → `Summarise`, leading whitespace); measure it with a prompt-mutation sweep and treat a >25-point canonical-vs-mutated gap as bad. (3) **Over-refusal** — it declines legitimate in-domain requests; measure with a frozen 200-prompt benign suite, target <2%. (4) **Catastrophic forgetting** — MMLU/GSM8K drop. Validation loss does not catch them because in SFT **val loss is flat or falling while the model gets worse**: later epochs are spent memorising surface forms and response lengths, which *reduces* held-out loss on an in-distribution eval set. CS-13's checkpoint table makes the trend explicit: epoch 1 → 0.91 loss, 92% format compliance, +5% verbosity, −2 pts general; epoch 5 → 0.19 loss, 98% compliance, **+61% verbosity, −14 pts general**.
- **Why asked:** It is the conceptual difference between classical ML and SFT, and it is the reason "our eval loss is fine" is not an answer.
- **Trap:** Watching `eval_loss` with `load_best_model_at_end=True` and declaring victory. It selects the most memorised checkpoint on a small in-distribution eval set. Track format compliance, refusal rate, mean length and an MMLU subset on the same axes.

**Q36. What is catastrophic forgetting's mitigation ordering, and what would you do with a 3-epoch full-FT run that dropped MMLU by 8 points?**

- **Answer:** Order: **replay > fewer epochs > LoRA > lower LR**. Concretely for that run: (1) add **5–20% general instruction data** (or 5–10% raw pretraining text, which must **not** be masked — masking replay text leaves nothing to learn); (2) cut to 1–2 epochs; (3) switch to LoRA so the base is frozen; (4) halve the LR; then (5) early-stop on a composite domain+general metric. If you are already on LoRA at 2 epochs, replay is the only lever left.
- **Why asked:** Candidates know the word "forgetting" and rarely know the ordering, which is the operative part.
- **Trap:** Lowering the LR first. It is the weakest of the four for this failure — a small `η` slows the drift but does not change its direction, so given enough steps the model still converges to the narrow distribution.

**Q37. How do you build an SFT dataset when you have no data?**

- **Answer:** Four sources, in order of value. (1) **Real logged interactions** with expert-written answers — in-distribution by construction, and the source CS-13's video omits; 2,000 well-chosen rows is the enterprise sweet spot. (2) **Document-to-QA synthesis** — chunk the corpus and prompt a strong model for Q&A pairs, with two corrections that everyone misses: generate the question *from* the chunk, then answer with the **chunk hidden**, and discard any pair the generator cannot answer without it; and check instruction-response token-Jaccard, because above ~0.3 you are training a copy task. (3) **Seed + self-instruct** — 175 hand-written, *diverse* seeds, generate, filter by similarity, cap at 3–4 rounds to avoid mode collapse. (4) **Persona/Magpie-style** generation from a pretrained model's template prefix — cheap and diverse, but needs a strong classifier to filter leaked template text. Then run the mandatory three-part filter on all of it: **answerability** (judge with the source withheld), **uniqueness** (embedding cosine ≥0.9 → drop), **non-overlap** (token Jaccard <0.3).
- **Why asked:** The instructor is explicit that this is the hardest part — *"in every company … you will NOT find this kind of data"* [23:20] — and the answerability filter is the detail that separates people who have done it from people who have read about it.
- **Trap:** Generating the question from the chunk *with the answer visible* and shipping pairs whose answers are copies of the prompt. That teaches copying, which is the failure mode CS-13 §1.3 opens with.

**Q38. How do you split data for SFT, and what is the classic leak?**

- **Answer:** Split **by source**, not by row: every row derived from the same document, ticket, user or template goes into the same split. Then deduplicate *across* the splits with MinHash-LSH (Jaccard ≥0.8 on 5-grams of the instruction field). Hold out 10–20% total, separating val (for early stopping) from test (touched once); with a small dataset hold out ≥100 rows so the confidence interval is meaningful. Then **decontaminate**: drop any training row sharing a **13-gram** with any eval prompt or reference answer — the GPT-3 paper's standard, long enough that accidental collision is negligible and short enough to catch a reworded sentence.
- **Why asked:** Leakage detection separates careful engineers from fast ones, and the decontamination step is the one people skip.
- **Trap:** `train_test_split(df, test_size=0.1, random_state=42)` on rows of near-duplicate synthetic data. It leaks, and it inflates every metric — CS-13 §15.2's real case: 14 of 200 eval rows had near-duplicates in train, inflating reported validity from 98.9% to 99.9%. A decontamination result of **0.0%** is itself suspicious; 15%+ means every metric you have reported so far is fiction.

**Q39. How do you evaluate an SFT model?**

- **Answer:** Four layers, all of them. (1) **Mechanical/held-out** — eval loss on a *masked the same way* held-out split, plus contamination count; useful only for divergence detection. (2) **Task metric** — schema-validity rate, exact match, regex/`must_contain` assertions, on a frozen held-out set with confidence intervals (at n=200 and p=0.9 the binomial standard error is ≈2.1 points, so ±3 points is noise). (3) **Comparative** — swap-corrected LLM-judge win rate against the *start* checkpoint, plus three free metrics: format compliance (want ≥95%), refusal rate on a benign 200-prompt suite (want ≤2%), and mean response length (watch for >15% drift). (4) **Regression** — an MMLU-500 or IFEval-200 subset to measure forgetting, run on *every* candidate; it costs ~10 minutes and it is the only way you will notice. And always against two baselines: the **well-prompted base** and the **start checkpoint**. A candidate that beats the start but loses to the currently-served model is not shippable.
- **Why asked:** The most common interview failure in this module is evaluating only the task metric.
- **Trap:** Comparing against a zero-shot base model and declaring victory, or using the last 200 rows of the training file as the eval set. CS-13 §15.5 has both as real post-mortem rows.

**Q40. How do you use LLM-as-judge correctly for SFT comparison?**

- **Answer:** Five non-optional steps: (1) **pairwise**, not pointwise — "which is better, A or B" is far more reliable than "rate 1–5"; (2) **swap positions and average** — run every pair as `(A,B)` and `(B,A)`; if A wins both orders it is a real win, if the orders disagree it is a **tie**; (3) blind and shuffle, stripping model-identifying strings; (4) **calibrate against ≥100 human labels** and report the agreement rate — below ~70% the judge is measuring something else; (5) fix and version the judge prompt. Know the measured biases: position bias accounts for **10–15 points** of win rate on its own, verbosity bias makes longer answers win **60–70%** of "equal quality" pairs, and self-preference is worth ~10 points — so never judge with the same model family you trained.
- **Why asked:** LLM-judge is standard practice and is routinely used wrongly; the swap requirement is the tell.
- **Trap:** Reporting an absolute 1–10 score with a single ordering and no calibration. It measures the judge, not the model.

**Q41. How do you pick LoRA `r` and `alpha`?**

- **Answer:** Start **`r=16, alpha=32`** (`alpha = 2r`, keeping the effective scale `alpha/r = 2`) and treat `r` as capacity. `r=8` for style/tone/format; `r=16–32` for a new behaviour or domain; `r=32–64` for something close to a new capability (CS-13's tool-calling case study used `r=32` because the task was harder than style). Past `r=64` returns flatten on most tasks. `lora_dropout=0.05` (0 for speed, and 0.05–0.1 helps on small data), `bias="none"`. The rule that matters: **`alpha/r` is the number that acts like a learning rate.** Halve `r` and halve `alpha`, or you have silently changed the LR while tuning `r`. CH-13's scaling heuristic is `lr ∝ 1/√r` with `alpha = 2r`.
- **Why asked:** Rank is the one LoRA knob with a real quality effect, and it is most often set by copy-paste from a tutorial that has not been updated since 2023.
- **Trap:** Setting `r=16, alpha=16` and separately sweeping LR — you are tuning the same thing twice. And `r=128` on 500 examples will overfit; at `r=256` on a 7B you have ~1.7B trainable parameters and most of the cost of full FT with none of the simplicity.

**Q42. Which `target_modules` do you use for LoRA, and which choice costs you quality?**

- **Answer:** All linear projections: `q_proj, k_proj, v_proj, o_proj` (attention) **and** `gate_proj, up_proj, down_proj` (MLP) — `target_modules="all-linear"` in `peft ≥ 0.8`. That is ~20M trainable parameters on a 7B (~0.3%). The cost of getting it wrong is measurable: `q_proj, v_proj` only (the notebook's 2021-era setting, ~4M params) is **−2 to −5 points** versus all-linear on domain tasks; attention-only (`q,k,v,o`) is −1 to −2. The MLP blocks are two-thirds of the parameters and hold most of the model's factual storage, so leaving them unadapted is the common mistake. Rarely, adding `lm_head` and `embed_tokens` (+300M) helps — but only when you are injecting new vocabulary, and usually it makes things worse.
- **Why asked:** A one-line answer that separates practitioners from tutorial-followers, and CS-13 lists the upgrade as one of the four "free wins" (2–5 points for one string).
- **Trap:** Forgetting that module names differ by architecture — `c_attn` for GPT-2, `query`/`value` for T5, `Wq`/`Wv` elsewhere. A wrong list either raises immediately or (in some `peft` versions) silently attaches **zero** adapters, leaving you with a loss that never moves and an optimiser with no trainable parameters.

**Q43. Why `tokenizer.pad_token = tokenizer.eos_token`, and what does it hide?**

- **Answer:** Llama/Mistral-class tokenizers ship with **no pad token** because pretraining never needed one; `Trainer` raises `ValueError: Asking to pad but the tokenizer does not have a padding token`, or a naive collator pads with id 0 and corrupts the batch. Setting `pad_token = eos_token` fixes it — and it **hides the worst silent bug in the module**: the pad id *is* the eos id, so "training on padding" looks like "training on EOS" and produces no error and no warning. That is exactly why the notebook's `padding="max_length"` + `labels = input_ids.copy()` bug survives code review. The only defence is an explicit assertion: `for l, a in zip(labels, attention_mask): assert a == 1 or l == -100`.
- **Why asked:** Everyone who has fine-tuned a Llama model has hit the missing pad token; far fewer have realised what setting it to EOS conceals.
- **Trap:** Adding a *new* pad token (`add_special_tokens({"pad_token": "<pad>"})`) without `model.resize_token_embeddings(len(tokenizer))`, which produces an index-out-of-range error or randomly-initialised embeddings.

**Q44. How do you serve a LoRA adapter, and what do you have to keep with it?**

- **Answer:** Two routes. **Merge** — `PeftModel.from_pretrained(base, adapter).merge_and_unload()`, saving the merged model plus the tokenizer **with the training template saved into `tokenizer_config.json`**; simplest, best throughput, one artefact. **Hot-swap** — serve one frozen base with `vllm serve $BASE --enable-lora --lora-modules pharma=out/final --max-lora-rank 16`, which gives dozens of adapters on one set of base weights at ~50–200 MB each instead of 14 GB per model; you **must** pass the same template (`--chat-template templates/acme_pharma_v3.jinja`). Merge in **bf16 or fp32**, never fp16, and never merge a QLoRA adapter into an NF4 base — dequantise to bf16 first, then merge, then re-quantise for serving and **re-run the eval on the quantised artefact**, because NF4/AWQ quantisation of a small SFT model sometimes introduces format drift.
- **Why asked:** Serving decisions made at merge time are hard to reverse, and CS-13's serving table lists the three checks everyone forgets: template byte-equality, special-token handling, and `max_tokens` long enough for the longest expected answer.
- **Trap:** Merging and shipping without re-evaluating, then discovering the quantised artefact emits unclosed tags. Also: loading an adapter with `AutoModelForCausalLM.from_pretrained(adapter_dir)`, which either raises `OSError: no file named pytorch_model.bin` or silently builds a **randomly-initialised** model — this is the notebook's own bug and it is the #1 cause of "the fine-tune did nothing."

**Q45. How would you reduce the inference bill with SFT?**

- **Answer:** By shortening the prompt. If a 4,000-token few-shot or system prompt can be replaced by a 300-token one after SFT, per-request input cost falls ~13x. CS-13's insurance case study did exactly this: 88% JSON validity with a 4,000-token prompt → **99.6% with a 300-token prompt**, plus a latency win. The general calculation: `tokens_in × price_in + tokens_out × price_out` per request, times monthly volume; compare against the training bill, which for that project was **$0.63** of GPU time against an **$18,000** dataset. That asymmetry is the strongest financial case for fine-tuning and the one to lead with — but note where the cost actually landed: the dataset, not the GPU.
- **Why asked:** Senior interviews are mostly about this. It ties the technique to a business outcome rather than a benchmark.
- **Trap:** Justifying the fine-tune by quality alone with no cost, latency or token number attached. Also: forgetting the *output* side — if SFT makes answers 3x longer (the verbosity signature), you can lose on cost even while winning on quality.

**Q46. What does a good SFT data row look like?**

- **Answer:** A `messages` list with the assistant target included: `{"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}`. Four properties: (a) the system turn appears in a meaningful fraction of rows — a deliberate **60/40** split if your serving stack sometimes drops it, so the model degrades gracefully rather than breaking; (b) the assistant content is the **ideal** answer, not a transcript of a real agent's typos; (c) the format and length match what production will send and expect; (d) it is **diverse on three axes** — task type (no single task >25% of the set), instruction phrasing (distinct leading verbs: 43 over 1,000 rows is healthy, 6 is a generator in a rut), and response length (a 10–20x spread between p5 and p95, with a deliberate tail of very short answers like "Yes." — those are the cheapest anti-verbosity intervention there is, target 10–20% of rows under 30 tokens). Also target 3–8% "I don't know" rows and 2–5% refusal-boundary rows.
- **Why asked:** Data is 80% of the project; a candidate who cannot describe their row format has not built one.
- **Trap:** 5,000 paraphrases of one instruction. The conditional distribution `p(output | instruction)` has zero variance in the conditioning variable, so there is nothing to learn about instructions — you have paid full fine-tuning cost for a continued-pretraining run with a decorative prefix.

**Q47. Your task is strict JSON extraction and your prompts are long. Structured outputs or SFT?**

- **Answer:** Start with **constrained decoding** for the format (it is deterministic, needs no data, and cannot be forgotten), and use SFT only for the parts that are genuinely a decision — which category, which urgency, whether a document is missing. If you do SFT, the honest production shape is **SFT + a validator + a retry**, not SFT alone: CS-13's case study reached 99.6% validity with SFT and still shipped a validator for the remaining 0.4%, because a downstream service parses the field and a parse failure is a page. Add three specific disciplines: (1) validate against the schema in the *generator* too, so no training row can contain invalid JSON; (2) re-run the format-compliance metric on the **quantised** served artefact, not the fp16 training artefact; (3) remember that a model which always emits `{}` scores 100% on "valid JSON" — check non-emptiness and required keys, not just parseability.
- **Why asked:** It tests whether the candidate reaches for training before reaching for the cheaper deterministic tool, and whether they understand that "95% valid JSON" is not a shippable answer when a parser is downstream.
- **Trap:** "We need 100% so we must fine-tune." Constrained decoding gives you 100% format; fine-tuning gives you the content. You usually need both, and they are different line items.

**Q48. How do you version and reproduce an SFT run?**

- **Answer:** The adapter is regenerable in 30 minutes; the **dataset is not** — treat it like source code. Directory-per-version with `train.jsonl`, a frozen `eval.jsonl` that is never trained on, a `MANIFEST.json` (dataset id, semver version, `n_train`/`n_eval`, sha256 of each file, source breakdown, filters applied, template sha256, seed, parent version, notes) and a `CHANGELOG.md`. Three rules: the eval set is frozen and versioned with the dataset (changing it invalidates every historical number); the training run records the **dataset hash and the template hash**, not just a version string; never mutate a version in place — a row fix is a patch, a distribution shift is a minor version. For seeding: `set_seed(42)`, `data_seed=42`, `dataloader_num_workers=0`, `dataloader_drop_last=False`. Be honest about the ceiling: bit-exact reproduction is achievable only on the same machine with `full_determinism=True` and no flash-attention; across GPUs you get ±2–5% on downstream eval. Define "reproduced" as "eval metric within ±1 point across three seeds."
- **Why asked:** Reproducibility is an engineering discipline and most candidates have never run the same experiment three times. The dataset-hash-vs-version-string distinction is the tell.
- **Trap:** Promising bitwise-identical reproduction on GPUs. Flash-attention's backward pass is nondeterministic by design.

**Q49. What is the relationship between SFT and the rest of the alignment stack?**

- **Answer:** SFT is stage 2 of 3 and carries roughly **90% of the value**. Pretraining → SFT (behaviour + format, 1k–500k rows, ~90%) → DPO/ORPO (preference over pairs, 1k–50k pairs, ~8%) → RL/GRPO (verifiable reward, 10k+ tasks, ~2%). SFT establishes the *interface*: without it there is nothing for DPO to compute a preference over, because a base model does not produce completions in your format. DPO is a refinement with small effect sizes (~0.2–0.5 MT-Bench points) and it can *undo* SFT-installed facts. RL needs a verifiable reward — a program, not a human — which for open-ended domain text usually means training a reward model, which is another project. And SFT data is the raw material for the later stages: design the SFT export to carry `prompt`/`chosen`/`rejected` so you do not re-do the labelling.
- **Why asked:** Tests whether the candidate over-invests in the fashionable stage and under-invests in the one that carries the value.
- **Trap:** "Do RLHF for quality." For a format and house-style problem, DPO/ORPO on 2k pairs is where style constraints actually live; RL is for verifiable reasoning.

**Q50. What are the three worst-case silent failures in an SFT run, and how do you detect each?**

- **Answer:** (1) **Train/serve template mismatch** — loss falls beautifully, model is unusable behind the API. Detect by rendering the same conversation through the training tokenizer and the serving path and diffing the strings byte for byte; put that assertion in CI. (2) **Padding trained as targets** — loss falls nicely, generations are choppy or terminate oddly; 20–90% of the loss is "emit `<pad>`". Detect with `assert attention_mask == 0 ⇒ labels == -100` over a few hundred rows. (3) **Adapter not actually loaded** — output is byte-identical to the base and you conclude "SFT does nothing." Detect by hashing a weight before and after loading, and by printing `print_trainable_parameters()` / counting `sum(p.requires_grad for p in model.parameters())`. Two honourable mentions: **prompt unmasked** (inspect `labels[:20]`, it should be all `-100`; a loss well below `ln(V)` at step 0 is the tell) and **response truncated away** (count rows where the prompt alone exceeds `max_seq_length`).
- **Why asked:** CS-13 §9.4 lists 14 silent failures; these three are the ones that reach production most often and all three are detectable in under an hour.
- **Trap:** Debugging them by adding data, training longer, or tuning sampling parameters. None of the three is a quality problem, so none of the three responds to any of those.

**Q51. What does `data_utils.token_stats()` return, and what do you do with each key?**

- **Answer:** `examples` (row count — must match your dataset), `tokens_total` (total supervised+unsupervised tokens, i.e. your compute bill), `len_p50`, `len_p95`, `len_max` (the length distribution — compare `p95`/`max` against `max_seq_length`; if `max` exceeds it you are truncating, and if `p95` is far below it you are wasting compute on padding), `with_supervision` (rows with at least one non-`-100` label — must equal `examples`; anything less is a silent `0/0` in the loss mean and a wasted batch slot), `without_supervision` (the complement; the counter exists because this is a real and common bug), and `supervised_token_frac` (want 10–40%). Note the API shape: `load_jsonl` returns a list of **message lists**, so you must run `build_masked_example(tok, m)` first — passing raw conversations to `token_stats` raises `TypeError: list indices must be integers or slices, not str`, and that error means you skipped the build step.
- **Why asked:** It is the repo's own pre-flight tool, and a candidate who has used it can answer this in 30 seconds; one who has not will describe what they wish it did.
- **Trap:** Running it on the *rendered strings* instead of the built examples, or on the training split only. Run it on train *and* eval, with the same masking policy, or the two losses are not comparable.

**Q52. How would you handle a 20:1 length imbalance in your response spans?**

- **Answer:** Recognize that a single long response **dominates the epoch**: one 4,000-token example contributes ~40x the loss of a 100-token one, because SFT loss is a mean over tokens. Three options: (1) **cap response length** at ~3x the median and flag rows above it — the diagnostic is `len(resp) > 3 × median`; (2) **weight rows** so each row contributes equally rather than each token; (3) **deliberately keep a spread** and accept that long examples carry more gradient, because length diversity is a feature (a 10–20x p5-to-p95 spread is the target — the short rows teach stopping, the long rows teach structure). What you must *not* do is pad everything to a fixed length, which converts the imbalance into a padding problem. And note the interaction with packing: with packing, a row whose response is entirely truncated still occupies a slot, so re-check the "no example is 100% masked" assertion *after* enabling it.
- **Why asked:** It tests whether the candidate reasons about the loss as a per-token mean (so long sequences dominate) rather than as a per-example average.
- **Trap:** Oversampling the short rows by duplication. That overfits the specific copies; write new diverse short examples instead.

**Q53. What is `assistant_only_loss` and what happens if your template lacks the marker?**

- **Answer:** `assistant_only_loss=True` (TRL `SFTConfig`, ≥0.12) masks every position that is not inside an assistant span, using the `{% generation %}…{% endgeneration %}` markers in the chat template. It is the only masking method that is correct across multi-turn conversations, tool calls, and templates that put a generation header *inside* a turn. If the template lacks the marker: **recent TRL raises** `ValueError: The chat template ... does not contain the generation tag`; **older versions warned and silently did not mask** — a silent failure with a quiet log line, which is the worst outcome. Check the template first: `print("generation" in tok.chat_template)`. If it is missing, either add the markers to a pinned `.jinja` file (CS-13 §4.3.4 shows the four-line edit) or use Unsloth's `train_on_responses_only`, which derives the boundary from the template text rather than from markers.
- **Why asked:** It is the modern correct answer to masking, and the version-dependence is a genuine production trap. Pin your TRL version.
- **Trap:** Enabling it and assuming it worked. Verify by decoding the first non-`-100` label — it must be the first response token, not the prompt's.

**Q54. What is the difference between `padding="max_length"` and a padding collator, and why does it matter here?**

- **Answer:** `padding="max_length"` pads **every** row to `max_seq_length` at tokenisation time; a padding collator (`DataCollatorForSeq2Seq`, or TRT's default with `padding=False`) pads each *batch* to the longest row in that batch. On the notebook's data (mean ~57 tokens, window 512) `max_length` padding wastes **89% of every forward pass**, and — because the labels copy the input ids — trains the model on 455 pad positions per row. A batch-level collator cuts that waste to the spread *within* a batch, and `group_by_length=True` cuts it further by batching similar lengths together. Packing eliminates it entirely. The rule: set `padding=False` plus a collator (or `packing=True`), and `max_seq_length` from your p99.5 — never use `max_length` padding as a convenience.
- **Why asked:** It is a config line that looks innocuous and costs 8x the compute while corrupting the labels, so it tests whether the candidate reads the flags they copy.
- **Trap:** Believing `padding="max_length"` is needed for "correct" batching. It is the opposite: it is the single largest source of wasted compute and of the pad-as-target bug.

**Q55. How do you decide between adding more data and improving the data you have?**

- **Answer:** Diagnose by *how* it fails. If the model performs the behaviour but inconsistently, or fails on input patterns you have seen, you need **more** data (coverage). If it fails in a way that suggests it never learned the behaviour, or failures cluster on a specific input type, you need **better** data (targeted examples of exactly the failing case). The cheapest experiment: hand-write **50 examples** of the failing pattern, append them, retrain for the same step count. If the failure disappears, it was coverage and 500 targeted rows will fix it. The mechanism behind the "better" branch is that SFT is a **mean-seeking estimator with no negative examples** — every mediocre row pulls the output distribution toward mediocre outputs and there is no loss term that says "this row is wrong." If 20% of your rows are mediocre, ~20% of your gradient points at mediocre behaviour. That is why AlpaGasus removing 43,000 rows *improved* every benchmark: those rows were not neutral, they were harmful.
- **Why asked:** It is the question behind most "should we collect more data" meetings, and the mean-seeking argument is the part that changes people's minds.
- **Trap:** "More data is always better." Duplicated or low-quality data actively hurts. Removing bad rows is a higher expected-value action than doubling your LR sweep.

**Q56. What is decontamination and what does a 0% contamination rate mean?**

- **Answer:** Decontamination is removing training examples that overlap your evaluation set, so the eval measures generalisation rather than memorisation. The standard (from the GPT-3 paper) is **13-gram overlap**: compute all 13-grams of each training row, drop the row if any 13-gram appears in any eval prompt or reference answer. Thirteen tokens is long enough that accidental collisions are negligible and short enough to catch a reworded sentence. Apply it to your *judge* prompts too. Interpretation matters: **0.0% is suspicious** — for a domain dataset built from the same corpus as the eval, some overlap is normal, and a zero usually means the checker is broken or the eval set is from a different distribution; **15%+** means your eval set is inside your training set and every metric you have reported so far is fiction. CS-13's real case: 14 of 200 eval rows were near-duplicates of train, inflating the reported validity figure from 98.9% to 99.9%.
- **Why asked:** It is the step that makes every other number meaningful, and the "0% is suspicious" heuristic is the sign of someone who has actually run it.
- **Trap:** Using sentence-level embedding dedup only. Cosine similarity on sentence embeddings misses verbatim 13-gram leakage, which is exactly the leakage that inflates metrics.

**Q57. What is an "I don't know" row and why is it the most valuable row type?**

- **Answer:** A training row where the correct response is an abstention — "That is not stated in the provided passage", "No, because §4 applies", "This is not supported in the current version." They matter because **the only way to teach the model to abstain is to show it abstaining**: with no abstention examples the model's abstention behaviour is inherited entirely from the base model's RLHF prior, which may be wrong for your domain — it may over-refuse benign requests or, worse, confidently invent. Target **3–8%** of rows. Two supporting row types: **refusal-boundary rows** (2–5% — the model must know what to decline) and **very short answer rows** (10–20% under 30 tokens — these are what teach stopping and are the cheapest anti-verbosity intervention). CS-13's house-voice case study learned this the hard way: its first dataset was all-positive answers, and the tuned model stopped saying "this is not supported"; adding 120 negative rows fixed it.
- **Why asked:** It is the least common row type in real datasets and the most valuable, so asking about it reveals whether the candidate thinks about the negative space of their behaviour.
- **Trap:** Concluding that abstention must come from a safety dataset. Safety data with no task data makes the *dominant* behaviour in the loss be refusal — the "refusal machine" failure mode.

**Q58. How do you handle a system prompt in SFT data?**

- **Answer:** Include it in a **deliberate fraction** of rows and understand the trade. Three facts: (1) if the model never sees a system role, it never learns to condition on one, and your serving harness will send one anyway; (2) if every row has an identical system prompt, the model over-fits that literal string and degrades when the serving stack sends a different one (or none); (3) templates differ — Llama-2 wraps the system prompt inside the first `[INST]` with `<<SYS>>`, Mistral v0.1/v0.2 has no system role at all and some templates silently **drop** it, ChatML has a first-class system turn. The production pattern is CS-13's insurance case study: 40% of rows carry the schema-spec system prompt, 60% do not, deliberately. When the serving stack stripped the system prompt for two weeks, the model degraded gracefully to 94% instead of breaking.
- **Why asked:** It is the quiet half of the train/serve mismatch taxonomy — "system prompt present at train, absent at serve" is a distinct failure from "wrong markers."
- **Trap:** Training with one fixed system prompt and assuming the model "knows" it. You have trained a fragile conditional that breaks the day someone edits the prompt.

**Q59. What is the difference between tokenizer mismatch and template mismatch, and which is worse?**

- **Answer:** A **template mismatch** means the same tokens are arranged in a different string — the model sees the wrong *structure* (no assistant header, wrong role markers, `[INST]` where it expects ChatML). The loss looks fine and the model is wrong in a specific, systematic way. A **tokenizer mismatch** means the ids themselves differ — different vocabulary, different special-token set, different BOS/EOS handling — so the embeddings are garbage relative to the input. Tokenizer mismatch is worse and usually louder, because a vocab-size discrepancy tends to throw an index error, but it can also be silent: a different *revision* of the same tokenizer with a different special-token set changes only a few ids and degrades quality subtly. The check for both is the same two lines: `assert model.config.vocab_size == len(tokenizer)` and `print(repr(tok.apply_chat_template(msgs, tokenize=False)))`, plus `assert tok.encode("<|im_start|>", add_special_tokens=False)` returns **one** id — if a role marker splits into pieces, it carries no signal.
- **Why asked:** It separates two failures that look similar in the symptom column and have completely different fixes.
- **Trap:** Adding a special token with `add_special_tokens` without `resize_token_embeddings`. Now the token exists in the vocabulary and has a *random* embedding, which is worse than it not existing.

**Q60. What is the cost model for an SFT run, in dollars?**

- **Answer:** `6ND / (n_gpus × peak_TFLOPs × MFU)` GPU-seconds, then multiply by **1.3–2x** for real overheads. Worked from CS-13 §11.3: 8,000 rows × 512 tokens × 2 epochs = **8.19M tokens**; QLoRA 8B on one A100-40 with gradient checkpointing runs ~4,000 tokens/s → ~34 minutes → **$0.85** at $1.50/GPU-hr. Add ~$1 of judge evaluation over 200 prompts × 2 models, and **$300** of human review. The comparison that should be in every interview answer: the same project's dataset, if authored by an expert at 20 minutes per row for 2,000 rows, is ~670 person-hours ≈ **$33,000** — 40,000x the GPU bill. For full FT on the same data: 8B + ZeRO-3 on 4 × A100-80 is ~1.2h ≈ $7.20. And the notebook's run — TinyLlama-1.1B, 5 rows — is 3 optimizer steps, ~15 seconds on a T4, **<$0.001**.
- **Why asked:** It distinguishes people who budget from people who hope, and the punchline (the dataset is the cost) is a strong closing signal.
- **Trap:** Quoting peak TFLOPs as achieved throughput without an MFU assumption (realistic MFU is 35–50% good, 20–30% common), or omitting the overhead multiplier and then overrunning 2x.
**Q61. What does NEFTune do, and what is the value you set?**

- **Answer:** NEFTune adds uniform noise to the **embedding layer's output** during training only: `emb += uniform(−α, α) / √(L·d)`, where `L` is sequence length and `d` is hidden size. The `1/√(L·d)` scaling keeps the perturbation a fixed fraction of the activation norm so it does not swamp short sequences. In `SFTTrainer` it is one line: `neftune_noise_alpha=5`. Reported gains on AlpacaEval for LLaMA-2 are **+29.8% for 7B and +8.7% for 13B**; CS-13's hyperparameter table includes it and calls it "free." Two caveats: it applies to **full fine-tuning and LoRA** (the paper's own results are full FT; the QLoRA result is community folklore) and it is train-time only — nothing changes at inference, there is no extra cost, and there is nothing to strip. Values: `5` default, `10–15` for long-sequence tasks. The mechanism is that the noise prevents the model from memorising exact surface strings, which is precisely the overfitting signature the module warns about.
- **Why asked:** It is the highest ratio of quality-gain to implementation-cost in the entire module, and it is a one-line answer that reveals whether someone reads beyond the defaults.
- **Trap:** Using it as a substitute for a held-out eval. A regulariser that improves a judge score can also hide a real regression, and NEFTune's reported wins are all judge-based (AlpacaEval), which the module already flags as a biased metric.

**Q62. Time-boxed: what is your 10-minute diagnosis of a model that "did not learn anything from fine-tuning"?**

- **Answer:** In order, cheapest first. (1) **Was the adapter loaded?** Hash a weight before and after, or print `sum(p.requires_grad for p in model.parameters())`. The notebook's own bug — `AutoModelForCausalLM.from_pretrained(adapter_dir)` — builds a fresh model and produces byte-identical output. (2) **Is any label not `-100`?** `assert any(l != -100 for l in labels)` over 100 rows; `supervised_token_frac` of 0 is the tell. (3) **Is the loss moving at all?** If it sits at `ln(V)` (10.37 for 32k, 11.76 for 128k), nothing is learning: LR 0, wrong parameters in the optimiser, frozen weights. (4) **Is the LR in the right regime?** LoRA at `1e-5` moves nothing; full FT at `2e-4` diverges and then looks flat after the divergence. (5) **Is the eval harness wrong?** Serving with the wrong template, truncating the answer with `max_new_tokens`, or comparing against the *already-served* model rather than the base. Only after those five do you question the data.
- **Why asked:** It is the whole module in one question, and the ordering is the answer — people jump to (5) or to "add more data".
- **Trap:** Re-running training. Four of the five causes are outside the training loop, so re-running reproduces the same failure with a fresh GPU bill.

---

## Level 3 — Advanced, Internals & Theory

**Q63. At the gradient level, what does masking the prompt actually change?**

- **Answer:** Nothing about the forward pass — attention still runs over every prompt token, and every prompt token still contributes to the hidden state that produces the answer's first logits. What changes is the **loss**, in two places: the numerator loses the prompt's per-token losses (`log p(prompt_t | prompt_<t)`, which for natural text is very low — often 0.5–2.0 nats against 3–8 for a structured answer), and the **denominator loses those positions too**, because `CrossEntropyLoss(reduction="mean", ignore_index=-100)` averages only over non-ignored positions. So masking the prompt raises the reported loss while making the gradient point only at the answer. CS-13's arithmetic for a 512-token row with a 400-token prompt and a 112-token answer: unmasked, ~56% of the *loss value* comes from the prompt tokens the model already predicts well; masked, 100% of the gradient is on the answer. That is a 2.3x increase in the effective learning signal for the task you care about — and it is why "our loss went up when we added masking" is expected and not a regression.
- **Why asked:** The single most precise question in the module. An answer that only says "we don't want to train on the prompt" is a vocabulary answer; the denominator point is the mechanism.
- **Trap:** Believing that the prompt's tokens are "free" information. They are not free — they are a large, easily-predicted term that dominates the gradient early in training, so the model spends its first several hundred steps learning to predict your prompt template rather than your answers.

**Q64. Why is `-100` and not `-1` or `0`?**

- **Answer:** Because `-100` is the documented default `ignore_index` of `torch.nn.CrossEntropyLoss`, and it is a sentinel that **cannot be a real token id**. Token ids are non-negative, so any negative value would work arithmetically, but `-1` is the conventional "last element" index and `0` is a valid token id in most vocabularies (`<unk>`, `<pad>` or a real token depending on the tokenizer), so `0` would mean "silently train on token 0." Using `-100` matches what `transformers`' own models and losses expect, matches what `Trainer` and TRL produce, and is what every masking helper in the ecosystem assumes. The other consideration is that `labels` and `input_ids` are the same shape but different semantics, and using a value outside the valid id range makes it impossible to accidentally compute a loss on a masked position.
- **Why asked:** It is a small question with a real answer, and it separates people who have read the `CrossEntropyLoss` signature from people who copy-pasted the constant.
- **Trap:** Assuming `-100` positions are excluded from *attention*. They are excluded from the loss only; you must set the corresponding `attention_mask` entries to **1** (because the prompt *is* attended to) and only set pad positions to `0`. Confusing the two produces a model that cannot see its own instructions.

**Q65. Why does `tokenize(a + b) != tokenize(a) + tokenize(b)`, and what does it cost you?**

- **Answer:** BPE builds tokens by greedily merging the most frequent adjacent byte pairs, and the merges are **position-dependent within a sequence**: the boundary between `a` and `b` is a new adjacency, and the pair straddling it may be a merge learned in pretraining. Concretely, if `a` ends with `"###"` and `b` starts with `" Response"`, tokenizing `a` alone may yield `["###", " Response"]` and tokenizing `a+b` may yield `["### Response"]` as a single token. So the token count is not additive and the *identity* of the last prompt token changes. The cost is a mask that is off by one or more tokens: compute `n_prompt = len(tokenize(prompt))` and slice the full tokenization at `n_prompt`, and the slice boundary lands in the wrong place. That single token is either a masked answer token (you lose supervision) or an unmasked prompt token (you train on the prompt) — and because it is one token out of 512 nobody ever notices without an assertion. The fix is the `build_masked_example` pattern from CH-13 §5.2: render the growing prefix, tokenize it *with `add_special_tokens=False`*, and take only the ids beyond the current length.
- **Why asked:** It is the deepest technical question in the module's required coverage, and it is the reason the manual masking code everyone writes is subtly wrong.
- **Trap:** Fixing it by tokenizing the prompt and the answer separately and concatenating the id lists by hand. That gives a token sequence the model was **never pretrained on** — the boundary tokens are wrong — which is worse than the off-by-one it was meant to fix.

**Q66. Compare `{% generation %}` markers, Unsloth's `train_on_responses_only`, and the manual string-split mask.**

- **Answer:**

| Method | Boundary source | Multi-turn correct | Fails when |
|---|---|---|---|
| `{% generation %}` + `assistant_only_loss=True` | Template semantics, per-turn | Yes | Template lacks the tag — recent TRL raises, older TRL **silently does not mask** |
| Unsloth `train_on_responses_only` | Splits the rendered string on `instruction_part`/`response_part` | Yes | Template has nested/duplicated markers (tool calls, few-shot examples inside one turn) |
| Manual mask (mask everything before the last `### Response:`) | Your own string parsing | **No** — only last turn | Any multi-turn data, any template change, any embedded marker in user text |

- **Answer (continued):** All three share one property: they are **template-dependent**, so a template edit invalidates the mask. The engineering consequence is that the template is a **versioned contract** whose hash must be recorded on the training run, and that the mask must be asserted (first non-`-100` label == first response token) in a test rather than trusted. For QLoRA on Llama-3 with multi-turn data, use `{% generation %}`; for a legacy template you cannot edit, use Unsloth; never use the manual mask on anything but a single-turn demo.
- **Why asked:** This is the "which tool and what is its failure mode" question that distinguishes a senior engineer from a practitioner who knows one library.
- **Trap:** Assuming the string-based method is safer because it does not need template edits. It breaks silently the day someone puts the word `### Response:` inside a user message — and in an agent setting, that is a prompt-injection vector.

**Q67. Derive why full fine-tuning needs about 20x the VRAM of QLoRA.**

- **Answer:** Per-parameter byte cost. **QLoRA**: NF4 base weights **0.5 bytes/param** (frozen) + LoRA adapters at `r=16` all-linear ≈ 0.3% of parameters trained in bf16 with fp32 Adam = `0.003 × (2 + 2 + 8) = 0.036` bytes/param amortised + quantisation constants ~0.13 → ≈ **0.7 bytes/param**. **Full FT**: bf16 weights 2 + bf16 gradients 2 + AdamW's two fp32 moments 8 + fp32 master weights 4 = **16 bytes/param** — the DeepSpeed/FSDP configuration with a master copy; quote it as 12 in a plain bf16 loop with no master (`2 + 2 + 8`) or 14 on this repo's floor (`4 + 2 + 8`). Ratio of the static terms: `16 / 0.7 ≈ 23x`, or `14 / 0.7 = 20x` at the floor; the measured table shows `91.6 / 4.9 = 18.7x`. Two things separate those: the table prices full FT at **14**, not 16, which is ~70% of the 23x→18.7x gap, and QLoRA's activation term (`+0.35 GiB`) is proportionally larger on a 4.9 GiB base, which is the remaining ~30%. The **dominant single term is AdamW's optimiser state at 8 bytes/param** — twice the model weights — and that is the whole reason LoRA and quantised optimisers exist. Note also that this ratio is *per model*, so it applies the same at 70B: 913.8 / 46.8 = 19.5x.
- **Why asked:** The candidate is asked to *derive*, not recall, because the derivation is what lets them size an unfamiliar model instead of looking it up.
- **Trap:** Saying "LoRA is more memory-efficient because it trains fewer parameters." True but not the mechanism — the mechanism is that the *frozen* weights need no gradients and no optimiser state, so the `2 + 8 = 10` bytes/param of grad-plus-Adam (`12` if you also count the frozen weights' own copy, `16` with an fp32 master) that dominate disappear. On Llama-3-70B, LoRA `r=16` trains ~160M parameters (2%) but saves 97% of the optimiser state, because the optimiser state scales with *trainable* parameters, not total.

**Q68. Why does the LR depend on the batch size, and what is the rule?**

- **Answer:** Gradient variance scales as `1/B`, so the standard deviation of the gradient estimate scales as `1/√B`. To keep the *signal-to-noise ratio of the parameter update* constant as you increase `B`, you scale `η ∝ √B` — the square-root rule; the linear rule `η ∝ B` is the "perfectly-scaled" regime that holds for small `B` and then breaks, which is why the sqrt rule is the safer default for LLM fine-tuning. Practically: doubling tokens/step by 2x allows ~1.4x the LR, and the safe band is wide. This is also why the module's LR table has different values for different regimes rather than one number: LoRA's `1e-4–2e-4` is at a small effective batch and trains a tiny fraction of parameters; full FT's `1e-5–2e-5` is at a large batch and moves every weight, so a LoRA-scale LR would destroy the pretrained features. The other asymmetry: LoRA's `B` matrix is initialised at zero, so the adapter's effective step is scaled by 1 as well — the two regimes are not comparable by LR alone, only by the *product* of LR, batch and trainable-parameter count.
- **Why asked:** It is the reason a hyperparameter table has multiple columns, and it is the question that catches people who tune LR by looking for a number instead of for a regime.
- **Trap:** Using the sqrt rule alone to justify a 10x LR increase with a 100x batch. Warmup must also scale, and past a point the linear-scaling and sqrt-scaling regions both fail and you need a fresh LR sweep.

**Q69. What is the superficial-alignment hypothesis and how does it apply to SFT?**

- **Answer:** The hypothesis (from the LIMA and "LIMA is not enough" line of work) is that **instruction following is a surface behaviour that pretraining already installed, and SFT's job is to elicit it, not to teach it.** Evidence: 1,000 curated examples produce a usable assistant; a 1.3B InstructGPT model was preferred over a 175B base GPT-3; and a model fine-tuned on **math-only** data follows instructions in unrelated domains nearly as well as one fine-tuned on general instruction data. The strongest form of the claim is that SFT mostly teaches **format, tone, and turn structure**, and that the knowledge and reasoning come from pretraining. Two consequences for practice: (1) SFT data quality dominates quantity because you are selecting a behaviour, not training a capability; (2) SFT alone will not teach the model facts it does not have — that is the CS-12 continued-pretraining job or a RAG job. The honest counterpoint, which you should give: the hypothesis is partly overstated — behavioural-only SFT does **not** generalise as far on hard reasoning as the original paper claimed, and domain SFT with real task data does measurably improve task accuracy, not just tone.
- **Why asked:** It is the theory that justifies the entire "1,000 rows, 2 epochs, stop" stance of the module, so a candidate who tunes 200k rows has implicitly rejected it.
- **Trap:** Taking it to mean "data does not matter." It means the *distribution and quality* of data matters more than the volume, and that a small set of behaviour-defining examples is worth more than a large set of random ones.

**Q70. Why is held-out loss the weakest evaluation signal in SFT?**

- **Answer:** Four reasons. (1) **The eval set is in-distribution by construction** — you curate it the same way as training, so it measures fitting, not generalisation, and it falls monotonically while the model gets worse at everything else. (2) **The masking policy changes the denominator**, so an eval loss computed with a different mask from training is not comparable to it, and the number can be made to look better by masking less. (3) **It has no notion of correctness** — a model that emits valid JSON with the wrong `urgency` field has an excellent loss on a well-formatted reference. (4) **It cannot see the failure modes that matter**: verbosity, rigid formatting, over-refusal and forgetting are all *invisible* to a held-out loss over in-distribution prompt-answer pairs, and CS-13's checkpoint table shows the divergence explicitly — loss 0.63 → 0.19 while verbosity rises 5% → 61% and general capability falls 2 → 14 points. Keep loss for divergence detection (a spike, a NaN, a plateau at `ln(V)`) and for *nothing else*. Evaluate on task metrics, refusal rate, length drift, format compliance under prompt mutation, and a general-capability delta.
- **Why asked:** "Our eval loss is 0.42, it's working" is the most common wrong answer in this module, and CS-13 lists "training loss < 0.5 means memorisation" in its ten-fact summary.
- **Trap:** Using `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"`. On a small in-distribution eval set that selects the **most memorised** checkpoint, and it will be the worst one you would have picked by hand.

**Q71. How does `CrossEntropyLoss` handle an example where every label is `-100`?**

- **Answer:** The mean over zero valid positions is `0/0`, which is `NaN` in a naive implementation. `torch.nn.functional.cross_entropy(..., ignore_index=-100)` computes the numerator and denominator as separate sums; the denominator is the count of non-ignored positions, so an all-ignored row yields `0/0 = NaN`, and that NaN flows into the batch loss and destroys the entire run from that step onward. Some library versions guard it with a clamp to avoid the division by zero, which is worse in a different way: the row contributes **zero** gradient while still consuming a batch slot and a forward pass, so it is silently wasted capacity. Either way the fix is the same and it belongs in the data pipeline: assert `supervised_token_frac > 0` per row at build time, and use `token_stats`'s `without_supervision` counter to fail the run before it starts. Rows reach the all-masked state two ways: truncation that cuts before the answer (a 4,000-token prompt in a 2,048-token window), and a masking helper that fails to find the response marker.
- **Why asked:** It is a specific, checkable numeric consequence of the masking design, and it is one of the four silent failures CS-13 lists that engineers rarely know exists.
- **Trap:** Filtering the row out *after* the fact instead of fixing truncation, which just moves the problem to the next dataset version.

**Q72. Why is 13-gram overlap the standard for decontamination?**

- **Answer:** It is a length/robustness trade-off with a specific justification. Short n-grams (5–8) collide by chance constantly: a 5-gram of common English has a non-trivial probability of appearing in both a training row and an eval prompt, which produces false positives and a decontamination step that deletes 30% of your data for no reason. Long n-grams (30+) miss paraphrased leakage — the training row that says "the deductible is 500" against an eval row that says "the policy has a $500 deductible" shares few exact long n-grams even though it is the same fact. **13** is the value used in the GPT-3 paper and it has stuck because it is empirically in the sweet spot: long enough that accidental collision is negligible (a 13-token exact match in natural text is almost always copied), short enough to catch a sentence copied with light edits. It is a *lexical* check, so it is complement, not a replacement, for a semantic near-duplicate check (MinHash-LSH with Jaccard ≥0.8 on 5-grams for dedup, or embedding cosine ≥0.9 for paraphrase). Use both: 13-gram against the eval set, MinHash within the training set.
- **Why asked:** It is a specific number with a specific justification, and the interviewer wants the trade-off, not the number.
- **Trap:** Decontaminating with embeddings only. Cosine similarity on a sentence encoder is insensitive to a verbatim 13-gram insertion inside a longer sentence, which is exactly the leakage that inflates a benchmark number.

**Q73. What is wrong with a synthetic dataset generated by "prompt a strong model with your documents"?**

- **Answer:** Four failure modes, all measurable. (1) **Unanswerable pairs** — the generator had the chunk in context, so it produces questions whose answers are only inferable from the chunk, not from the question; the trained model then answers from a prior it does not have, and hallucinates. Detect by re-asking the generator with the chunk **withheld** and dropping pairs it cannot answer. (2) **Copy tasks in disguise** — the question contains most of the answer's content, so the model learns paraphrasing, not reasoning. Detect with token Jaccard between instruction and response; **< 0.3** is the threshold. (3) **Mode collapse** — self-instruct at 3–4 rounds converges to one phrasing family and one length; count distinct leading verbs (43 over 1,000 rows is healthy, 6 is a rut) and measure the p5/p95 response-length ratio (want **10–20x**). (4) **Template leakage** — Persona/Magpie output contains fragments of the generator's own chat template, so the trained model learns to emit `<|start_header_id|>` as content. Filter with a classifier and a simple substring blocklist. On top of that, synthetic data is **stylistically flat**: it has the generator's tone, so if your house voice differs, you need at least a minority of human-written rows or you have fine-tuned to the teacher, not to the house.
- **Why asked:** Synthetic data is the default starting point for most real projects, and CS-13 §4.6 lists all four of these filters. Knowing them is the difference between a dataset and a liability.
- **Trap:** Generating 50,000 rows and calling it a dataset. The pipeline should be able to say exactly how many of the 50,000 survived each filter, and if fewer than ~10% survive, the generator is not aligned with the task.

**Q74. Why does LoRA work? What is the low-intrinsic-dimensionality argument?**

- **Answer:** The claim (Aghajanyan et al., then Hu et al. for LoRA) is that **task adaptation has low intrinsic dimension**: the `ΔW` needed to move a pretrained model from its pretraining objective to a specific downstream task has most of its energy in a small subspace, so `ΔW ≈ BA` with `B ∈ R^{d×r}`, `A ∈ R^{r×k}` and `r ≪ min(d,k)` captures it. For attention projections at `r=8` that is 99%+ of the full fine-tune's quality at 0.01% of the parameters. `A` is Gaussian-initialised, `B` is **zero**, so `BA = 0` at step 0 — the model starts exactly at the pretrained weights and the adapter grows into the task, which is a large part of why LoRA is stable at LRs that would wreck a full fine-tune. Two consequences the theory predicts correctly: **rank saturates** (gains flatten past `r=64` on most tasks, because the intrinsic dimension is task-specific and small), and **the rank needed correlates with task difficulty**, not dataset size — style and format need `r=8`, a new behaviour needs `r=16–32`, something closer to a new capability needs `r=32–64`. The caveat worth stating: the intrinsic-dimension evidence is strongest for *adaptation* tasks and weakest for *knowledge injection*, which is why LoRA is a poor tool for teaching new facts and a good tool for teaching new behaviour.
- **Why asked:** LoRA is used constantly and understood rarely; the interviewer wants the mechanism plus the two predictions it makes (saturation and task-dependence).
- **Trap:** "LoRA works because it has fewer parameters so it regularises." That is a consequence, not the mechanism, and it predicts the wrong thing — it would imply that a *smaller* `r` is always better, which is false for hard tasks.

**Q75. Why did the notebook's adapter report 1,126,400 trainable parameters, and where does that number come from?**

- **Answer:** Work it out per module. TinyLlama-1.1B-Chat has **22 layers**, hidden size **2048**, and **GQA** with 32 query heads and 4 KV heads at head_dim 64, so `q_proj: 2048→2048`, `k_proj: 2048→256`, `v_proj: 2048→256`, `o_proj: 2048→2048`. With `r=8` and `target_modules=["q_proj","v_proj"]`: `q = 8×(2048+2048) = 32,768` and `v = 8×(2048+256) = 18,432` → **51,200 per layer** × 22 layers = **1,126,400**. Exactly the reported number. The general formula is `Δparams_per_module = r × (d_in + d_out)`. Two facts to carry away: GQA makes `k_proj` and `v_proj` **asymmetric** (the output dim is `n_kv_heads × head_dim`, not `hidden_size`), which is why a naive `r × 2d` calculation over-counts by ~40%; and the same formula gives ~20M for `target_modules="all-linear"` at `r=16` on a 7B, which is the number you quote when sizing adapters for storage (20M × 2 bytes ≈ 40 MB per adapter, vs 14 GB for the merged model).
- **Why asked:** It is a closed-form arithmetic question with an exact answer, and it catches the GQA asymmetry that most candidates miss.
- **Trap:** `8 × 2048 × 2 = 32,768` per module times 44 modules = 1,441,792, then hand-waving the 28% discrepancy. The discrepancy is GQA, not "biases and configuration details."

**Q76. What is the "SFT memorizes, RL generalizes" result and what does it mean for a project plan?**

- **Answer:** The observation (Chu et al., 2025, and the reasoning-model literature) is that SFT's cross-entropy objective fits the *token sequence*, so with enough epochs on a small set it memorises exact answers and generalises worse out of distribution — while reinforcement learning with a verifiable reward optimises the *outcome* and generalises better on held-out tasks, at the cost of needing a programmatic checker. The practical reading for a planning answer: (1) SFT is the right tool when the output space is finite and stylistic (JSON shape, house tone, refusal behaviour, tool syntax) and the wrong tool when you need the model to *reason* to an unseen answer; (2) if you are going to do RL later, SFT's job is to install the format and get the pass rate off the floor — a model that never emits a valid tool call cannot receive a reward signal; (3) the epoch budget is the operational expression of the result — 1–3 epochs for SFT, because past that you are buying memorisation. Give the counterpoint too, honestly: the result is most solid for math/code where a verifier exists, and most open-ended enterprise SFT is not in the regime where RL is even available, so "SFT then stop" is the correct plan for the large majority of projects.
- **Why asked:** It is the theoretical result that justifies the module's whole epoch and sizing stance, and it is 2025-era — a candidate who knows it is current.
- **Trap:** Concluding "SFT is bad, do RL." For a JSON-schema or house-voice task there is no verifiable reward and SFT is both the cheaper and the better answer.

**Q77. When *should* the prompt be in the loss? Name the real exceptions.**

- **Answer:** Masking the prompt is the default, not a law. Five exceptions: (1) **Continued pretraining / domain adaptation** on raw text (CS-12) — there is no prompt and answer, so every token is 100% supervised. (2) **Replay data** — if you mix 5–10% raw pretraining text into an SFT run to prevent forgetting, that text **must not be masked**, or you are training on nothing; this is the mistake people make when they add replay and see no effect. (3) **Format-learning from a base model** — a base model that has never seen a turn structure sometimes learns the template faster if the prompt tokens are also trained, because the template itself is the thing being taught; it is a legitimate early-experiment choice, to be dropped once the format holds. (4) **Document/agent traces where the "prompt" is model-generated** — in a tool-calling trace, the model's own tool call is part of the context; whether it is masked depends on whether you want to reinforce emitting it (typically yes, train the tool call, mask the tool *result*). (5) **Very short prompts.** If the prompt is 8 tokens and the answer is 200, masking buys ~4% and costs a code path; but mask anyway, because the code path is the same and the discipline is what prevents the 400/112 case from regressing.
- **Why asked:** Every rule in this module has an exception, and the brief demands that they be named. A candidate who says "never train on the prompt" has memorised a rule without its boundary.
- **Trap:** Turning on replay and leaving `assistant_only_loss=True` globally — the raw-text rows have no assistant turn, so they get masked to all `-100`, contributing a `0/0` and possibly a NaN, while you believe you are doing rehearsal.

**Q78. Explain the packing position-id mechanism precisely, and what breaks without it.**

- **Answer:** In a packed batch, `input_ids` is one long sequence containing several examples concatenated: `[A1..An, B1..Bm, C1..Ck]` padded to `max_seq_length`. With a standard causal mask plus a single position-id range `0..L-1`, every token attends to every earlier token **including tokens from previous packed examples** — example B's first token attends to example A's last 2,000 tokens. Two harms: the loss for B is computed in a context it will never see at inference (so the model learns to condition on irrelevant preceding text, which is a distribution shift at serving), and if you pack *only* the sequence and not the mask, the attention pattern itself is wrong. The fix is a **block-diagonal causal mask**, expressed in practice by resetting `position_ids` to 0 at each example boundary; FlashAttention-2's `flash_attn_varlen_func` reads `cu_seqlens` (cumulative sequence lengths) and applies an implicit block-diagonal mask, so `DataCollatorWithFlattening(return_position_ids=True, return_flash_attn_kwargs=True)` is the HF entry point. The **detection** is the important part: pack a batch containing one example whose answer you know is `"Blue"` and one whose answer you know is `"42"`, and check that the logits for the first token of B's answer do not change when you swap A for a different example. If they change, you are leaking.
- **Why asked:** CS-13 calls packing's cross-contamination one of the hardest-to-detect silent failures, and the "swap an example and see if the logits move" test is the answer that proves the candidate has actually debugged it.
- **Trap:** Assuming `packing=True` handles this. In TRL, `packing=True` with `padding_free=True` and a FlashAttention-2-capable model is the safe configuration; `packing=True` on a model running eager attention with default collation is the unsafe one.

**Q79. Why does epoch 4 usually hurt, mechanically?**

- **Answer:** Because after 2–3 epochs the model has fit the *behavioural* signal in the data and further epochs can only fit the *incidental* signal: exact phrasing, response lengths, the surface form of the template. Those incidents are lower-variance across the dataset than the behaviour is, so they are the last thing gradient descent picks up — and once picked up, they are what the model reproduces at inference, where their probability of being correct is much lower. CS-13's checkpoint table tracks it exactly: epoch 1 → 0.91 loss / 92% format / +5% verbosity / −2 general; epoch 2 → 0.63 / 96% / +8% / −4; epoch 3 → 0.44 / 97% / +18% / −7; epoch 5 → 0.19 / 98% / **+61%** / **−14**; epoch 10 → 0.06 / — / **+90%** / **−26**. Notice that format compliance keeps improving (98% is the best number in the table) while every other metric degrades — the memorisation is *visible as a win* on the metric you are most likely to be watching. The 4-epoch boundary is not a law; it is where the memorisation term starts to dominate on typical 1k–10k-row datasets at typical LRs. With 100k+ well-diversified rows, 3–5 epochs is defensible.
- **Why asked:** It is the numeric heart of the module and it explains why "train longer, it's still improving" is the wrong inference from a loss curve.
- **Trap:** Using `eval_loss` as the stopping criterion at all. Loss is 0.19 at the epoch where the model has lost 14 points of general capability — the curve says "keep going."

**Q80. What are the measured biases in LLM-as-judge, and how do you correct each?**

- **Answer:** Three with published magnitudes. **Position bias**: the first-presented response wins **10–15 points** more often at equal quality; correct by running every pair in both orders and counting an order-disagreement as a tie. **Verbosity bias**: longer answers win **60–70%** of pairs the human raters called equal quality; correct by reporting mean-length drift alongside the win rate and by adding a length-matched human-calibrated subset. **Self-preference**: a judge scores its own family's outputs ~10 points higher; correct by judging with a different model family from the one you trained and by blinding model names — which is harder than it sounds, because fine-tuned models leak their identity through distinctive openings. Plus two operational rules: **calibrate against ≥100 human labels** and report the agreement rate (below ~70% the judge is measuring something else), and **freeze and version the judge prompt** — a changed judge prompt invalidates every historical win rate. The honest framing to give in an interview: an LLM judge is a *regression detector*, not a quality metric, and its value is in relative comparison under a fixed protocol.
- **Why asked:** Everyone uses a judge; almost nobody corrects for it. The magnitudes show the candidate has read the papers.
- **Trap:** Reporting a single-ordering win rate to two significant figures. With 200 pairs the binomial standard error is ±2–3 points before any bias correction, so a "win rate of 64.3%" is noise.

**Q81. Why is a 1,000-row curated dataset competitive with a 52,000-row one?**

- **Answer:** Because SFT is **selection, not distillation**. The objective teaches the model which of its pretrained behaviours to emit, and that selection is determined by the *extremes and the diversity* of the demonstrations, not by their count. Once every behavioural mode you care about is represented — 40–60 distinct task types, a range of phrasings, a range of lengths, refusal boundaries — additional rows of the same mode add almost no information about the mode and a great deal of noise about its surface form. The evidence: LIMA's 1,000 curated examples reached 43% win/tie against GPT-4, 58% against Bard and 65% against Alpaca-65B; **AlpaGasus** removed 43,000 of Alpaca's 52,000 rows and the resulting **9,000-row** set beat the full 52,000 on 5 of 6 benchmarks; **Deita**'s 6,000-row subset beat sets of 100k+. The reverse side is that curated means curated — LIMA's 1,000 were written by experts with a diversity specification, not sampled from a pile. Quoting the sizes without the curation is the mistake; the operational form is "**2,000–10,000 domain rows is the sweet spot, and the curation pipeline is the project.**"
- **Why asked:** It is the module's central empirical claim and the one most likely to be tested against a candidate's instinct that more data is better.
- **Trap:** Using Alpaca as the example of good data. It is 52,000 rows generated by `text-davinci-003` with a licence that restricts commercial use and a quality distribution that AlpaGasus showed to be actively harmful in 43,000 of them.

**Q82. What is the licence trap with Alpaca, and how do you audit a dataset's licence?**

- **Answer:** Alpaca's *data* was generated by `text-davinci-003`, whose terms of use prohibited using outputs to train competing models, and the dataset was released under **CC-BY-NC-4.0** — non-commercial, while the license text itself is a community license for the *code* that the data was folded into, which is not the same thing. So a model trained on Alpaca may be non-commercial and may breach the generator's ToS. The audit: (1) read the dataset card and the **license tag** on the Hub, not the README's licence claim; (2) trace the *generator* — if a dataset was produced by an API model, the API's terms apply on top of the data licence; (3) check for **share-alike** clauses (dolly-15k is CC-BY-SA — the *weights* may be argued to be a derivative work); (4) check whether the model weight licence itself forbids training derivatives. A clean default stack to name: **`dolly-15k`** (human-written, CC-BY-SA), **`ultrachat_200k`** (MIT), **`tulu-3-sft-mixture`** (ODC-BY). The practical point: 90% of tutorial datasets are non-commercial, and the licence is discovered by legal during the launch review.
- **Why asked:** It is the question that separates a hobbyist from someone who has shipped. Nobody asks it in a tutorial and everybody asks it in a legal review.
- **Trap:** "It's on HuggingFace so it's free to use." The license tag is the licence; the README is marketing.

**Q83. What is the actual mechanism by which NEFTune helps?**

- **Answer:** `emb ← emb + U(−α, α)/√(L·d)` — uniform noise added to the *embedding output* before the first transformer block, scaled so the perturbation magnitude is roughly constant in the norm of the activation regardless of sequence length. Three mechanisms by which it helps: (1) it **destroys exact surface memorisation** — the model cannot lock onto the literal token sequence because the same token has a different representation each step, which is exactly the overfitting axis the epoch table exposes; (2) it **smooths the conditional distribution** in the same way dropout smooths a classifier, making the output less peaked and less prone to the verbosity attractor; (3) because it is applied to embeddings and not to hidden layers, the perturbation is *input-space*, which propagates through the whole depth of the network — a cheap and deep regularisation. The scaling detail is what makes it work: without `1/√(L·d)` the noise would be negligible on long sequences and catastrophic on short ones (a 10-token row has a much smaller activation norm), so the paper's normalisation is what makes one `α` work across a mixed-length dataset. `α=5` is the default; `α=10–15` for long-sequence tasks; it is train-time only, so inference is byte-identical.
- **Why asked:** It is a one-flag change with a large reported effect, so the interviewer checks whether the candidate understands *why* it works or just that it is "free accuracy."
- **Trap:** Assuming it works on QLoRA. The published results are full fine-tuning; on QLoRA the effect is smaller and less reliably positive, so do not count it as a guaranteed +2.

**Q84. Why does merging a QLoRA adapter into an NF4 base go wrong?**

- **Answer:** Because the adapter's `ΔW` was learned in the **NF4-dequantised** forward pass and the merge arithmetic assumes both operands are in a common precision. `merge_and_unload()` computes `W_merged = W_base + α/r · BA`, and if `W_base` is an NF4-packed tensor the addition either (a) raises on a dtype/shape mismatch, or (b) dequantises to bf16, adds, and re-quantises — at which point the fine-tuned signal (`ΔW` is small relative to `W`) is partly destroyed by the second round of 4-bit quantisation, and the resulting model's behaviour differs from the adapter's. The correct sequence is: **load the base in bf16, load the adapter, merge in bf16 (or fp32), then quantise the merged model for serving, then re-run the eval on the quantised artefact.** The last step is the one people skip and the one that matters — 4-bit quantisation of a small SFT model can introduce format drift (unclosed JSON tags, dropped stop tokens), and the only way to know is to measure the artefact you will actually serve. There is a second, subtler trap: PEFT's adapter config records the base model *name*, and if that name resolves to a different revision or a different quantisation, the merge is against different weights than the ones the adapter was trained against.
- **Why asked:** Merging is a one-line call with a multi-step correct procedure, and the "re-evaluate the quantised artefact" step is the senior signal.
- **Trap:** `merge_and_unload()` on a model loaded with `load_in_4bit=True`, then saving — which appears to work and silently ships a different model from the one you evaluated.

**Q85. Why do special tokens have to be single tokens, and how do you test it?**

- **Answer:** Because a role marker is a **structural delimiter**, and a delimiter that the tokenizer splits into several ordinary subword pieces carries no reliable signal: the model must learn from scratch that those particular pieces in that particular order mean "the user is speaking", using only the gradient from the few thousand rows you have. A single dedicated token is a distinct embedding the model can learn as a *symbol* in a handful of steps. The test is two lines: `assert len(tok.encode("<|im_start|>", add_special_tokens=False)) == 1`, and the same for `<|im_end|>`, `<|eot_id|>`, `<|start_header_id|>` and the BOS/EOS you rely on. Failures are usually a **checkpoint/tokenizer mismatch** — loading the base tokenizer against an instruct checkpoint, an older `tokenizer.json` revision, or a tokenizer from a different family (the notebook's TinyLlama-1.1B-Chat vs TinyLlama-1.1B case). Note the honest caveat: when the base and instruct checkpoints share a vocabulary, the mismatch is *harmless* — the tokens are the same tokens — so this failure only bites when new special tokens were added without `resize_token_embeddings`, which makes the marker either an out-of-range index or a randomly-initialised embedding.
- **Why asked:** It is a two-line check that invalidates an entire training run, and the follow-up ("is it always fatal?") tests whether the candidate is precise or just cautions.
- **Trap:** Adding a token with `tokenizer.add_special_tokens(...)` and forgetting `model.resize_token_embeddings(len(tokenizer))`. Now the token exists in the vocabulary with a random embedding, which trains — so no error is raised, and quality degrades for a reason that is nearly impossible to see in the loss.

**Q86. Why is the loss on the answer's *first* token the hardest to learn, and what does that imply for evaluation?**

- **Answer:** Because of the causal shift. The first answer token is predicted from the last prompt/header token's hidden state, which has no preceding answer context — the model must map "the conversation has arrived at the assistant turn, and the user asked X" onto the first token of the response. Every subsequent answer token is predicted with the previous answer tokens in context, which is far easier. Two implications. (1) **The first token dominates the learning of the *behaviour*** — it is where "begin with `{`", "begin with `Sure,`" and "begin with a refusal" live; that is why the mask boundary decision (does the last supervised position include the assistant header?) is not cosmetic. (2) **Evaluation must look at the opening**, not just the whole response: if you compute a per-position loss profile, the first 3–5 answer positions typically carry 3–10x the loss of the middle, and a model that nails the opening usually nails the rest. Practically: when comparing two checkpoints, look at the opening 20 tokens of 50 held-out generations by hand — it is the highest-information manual review you can do, and it is where the "double header" and "answers in the user's voice" failures show up.
- **Why asked:** It is a genuine mechanistic insight that follows from the shift-by-one, and it explains several failure modes at once.
- **Trap:** A per-position loss plot that averages over all positions and therefore hides exactly the signal you want. Slice the loss curve at the first N answer positions separately.

**Q87. What is the strongest argument that your SFT project should not happen?**

- **Answer:** If the requirement can be met by **prompt engineering, RAG, or constrained decoding**, all three are cheaper, reversible within a day, and require no data. Specifically: (1) **knowledge** — if the model's failure is "it does not know our policies", that is retrieval, because SFT teaches behaviour not facts and a fine-tune cannot be updated when the policy changes on Tuesday; (2) **format** — if the requirement is "valid JSON matching this schema", constrained decoding guarantees it deterministically and cannot be forgotten; (3) **preference between two acceptable outputs** — that is DPO/ORPO territory, not SFT; (4) **long-context or reasoning** — fine-tuning does not add either. And the two organisational arguments against: the dataset is the cost (~$33,000 for 2,000 expert rows), and **the model needs an owner** — a fine-tuned artefact drifts out of distribution while everyone assumes it is still the model they trained, and it locks your base-model upgrade path, because a new base means retraining and re-evaluating everything. The strongest version of the argument: if you cannot state a **measurable** gap between the prompted model and the required behaviour, on a held-out set, at the right sample size, then you do not yet have a fine-tuning project — you have an evaluation project, and you should do that first.
- **Why asked:** The most senior question in the module. Interviewers are looking for the candidate to argue against their own project without being prompted, because that is the behaviour that saves a quarter.
- **Trap:** Answering with a list of *when SFT is right* instead of a genuine steel-man of not doing it. The answer must contain a number and a stopping condition.
---

## Level 4 — System Design & Scenario

> These are 15-minute whiteboard questions. Answer in the fixed shape: **requirements → constraints → design → trade-offs → failure modes**. The interviewer is grading the shape as much as the content, and grading whether you ask for the numbers you need before designing.

**Q88. Design the fine-tuning pipeline for a 14-person insurance company that needs structured claim triage from free-text adjuster notes. They have 3,000 historical claims with adjuster-written summaries, one A100-40 for two weeks, and no ML team.**

- **Answer — Requirements.** Output is a fixed JSON object (`claim_type`, `severity`, `missing_documents[]`, `urgency`). Must be valid 100% of the time because a downstream service parses it. Latency budget ~2s p95. Must not degrade general capability, because the same model handles internal chat. Must be maintainable by one engineer.
- **Constraints.** 3,000 rows is at the low end of the usable band but it is *real* and in-distribution — the highest-value kind. One A100-40 for 14 days is ~330 GPU-hours, which is ~1000x more than training needs. Small team → one artefact, one deployment path. Regulated domain → every artefact and dataset version auditable.
- **Design.** Base `Llama-3.1-8B-Instruct`. **QLoRA** `r=16, alpha=32, dropout=0.05, target_modules="all-linear"` — QLoRA because it leaves VRAM headroom for a bigger batch on 40 GB and makes five parallel experiments fit in the two weeks (this is a *search* budget, not a training budget). LR **2e-4**, cosine, `warmup_ratio=0.03`, **2 epochs**; `per_device_train_batch_size=4`, `grad_accum=4`, `max_length=1536` (measure p99.5 first), `bf16`, gradient checkpointing, `neftune_noise_alpha=5`, `assistant_only_loss=True`. Data: store in `messages` with the schema in a system turn on **60% of rows** so the model degrades gracefully if the system prompt is ever dropped; add 5% "insufficient information" rows so it can abstain; append EOS. **Evaluate before, during and after**: a frozen 200-row held-out set scored on schema validity, per-field accuracy (`urgency` is the one users complain about), refusal rate and mean length; MMLU-500 for forgetting. Hold back 10% by *claim id*, not by row, and 13-gram-decontaminate.
- **Trade-offs.** Full FT would be marginally better and would cost 91.6 GiB and the ability to run five experiments. LoRA (not Q) would train ~30% faster and use 14.7 GiB. A larger base (70B) is out of the VRAM budget entirely. Skipping the held-out set saves two days and makes the whole project unmeasurable.
- **Failure modes.** (1) Schema drift between the training validator and the serving schema — version both and hash them. (2) The system prompt silently dropped by the serving stack — the 60/40 split is the mitigation, and the eval must run *both* ways. (3) Severity and urgency learn opposite things and drift the model toward all-"high" — monitor the predicted label distribution, not just accuracy. (4) The claim-id split is by row and near-duplicate claims leak — check the 13-gram contamination count is non-zero-but-small. (5) The model overfits verbosity because adjuster summaries are long — cap response length at 3x median.
- **Numbers to close with.** Training: 3,000 rows × 900 tokens × 2 epochs ≈ 5.4M tokens ≈ 25 minutes on one A100-40 ≈ **$0.63**. Dataset authoring was the real cost: 2,000 rows × 20 min ≈ **$33,000** or ~670 person-hours. Expected outcome, grounded in the module's own case study: schema validity 88% → **99.6%**, urgency 71% → **89%**, MMLU **−1.2**.
- **Why asked:** It is the module's flagship applied scenario and it exercises every axis: sizing, method choice, masking, template, evaluation, cost.
- **Trap:** Proposing full fine-tuning because "we have an A100." You have an A100 *for two weeks*, which is a hyperparameter-search budget, and the deliverable is a config you understand, not one big run.

**Q89. Design the train/serve template contract for a platform team that has four teams fine-tuning on one shared inference cluster.**

- **Answer — Requirements.** No team may ship a model whose serving prompt differs from its training prompt. A template change must be a versioned, reviewable artefact. Incidents must be diagnosable from logs without re-running training.
- **Constraints.** Four teams, four models, possibly four template dialects; one vLLM cluster that must serve all of them without per-team custom code; existing models already in production with untracked templates.
- **Design.** A single module `prompt_contract.py` that owns every template, and **one function used by both training and serving**:

```python
# prompt_contract.py — the ONLY place a template lives. Imported by training and by the server.
TEMPLATE_VERSION = "acme-pharma-v3"

def render(messages: list[dict], add_generation_prompt: bool = False) -> str:
    return TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
```
  Then four enforcement mechanisms: (1) the training script writes `template_version` **and the sha256 of the rendered template string** into the adapter's config and into the run manifest; (2) the server loads the template from the same module and exposes `/healthz` returning the template hash; (3) a CI test asserts `render(train_messages) == render(serve_messages)` for a fixture conversation — byte equality, not semantic; (4) every inference log line carries the template version, so a regression can be attributed to a template change.
- **Trade-offs.** One shared module is a bottleneck (a change blocks four teams) but the alternative — four copies — is the failure this design exists to prevent. Serving via `--chat-template` (a file) is more portable than importing the module, at the cost of the file and the module being able to drift; if you do it, hash both and assert equality at boot. Refusing to serve a model whose recorded template hash does not match the cluster's is strict and will block deploys; make it a **warning for 24 hours then an error**, so you can find the existing breakage without a hard stop.
- **Failure modes.** (1) A team edits the template in a notebook and never lands the change — the manifest hash catches it. (2) The base model is upgraded to a new revision whose tokenizer gained a token — hash the *tokenizer* too, not just the string. (3) The serving stack applies a default template when none is passed, which is the silent failure mode the whole design exists for — assert the template is explicitly set at boot and fail loudly if the flag is absent. (4) `add_generation_prompt` on in training by accident, appending a spurious assistant header into the labels (this is the bug in CH-13 §5.2's snippet) — unit-test the mask boundary, not just the string.
- **Why asked:** Template mismatch is the #1 silent killer, and the durable fix is organisational, not technical.
- **Trap:** Solving it with documentation ("we'll remember to keep them in sync"). The mechanism must be a test that fails.

**Q90. You have 60,000 internal documents and the support team wants a model that answers questions about them. Design the project.**

- **Answer — Requirements.** Answer internal policy questions with citations; never invent policy; updateable weekly (policies change); must not regress general chat; 8-person support team, no GPU budget.
- **Constraints.** No GPU budget → no training. 60,000 documents → far too much knowledge to inject by SFT, and it changes weekly. Regulated → citations required, hallucination is a compliance event.
- **Design.** Say the hard thing first: **this is a retrieval project, not a fine-tuning project.** The model's failure is "it does not know our policies," and SFT teaches behaviour, not facts — a LoRA cannot be updated when policy v7 supersedes v6 on Tuesday, and it will confidently quote v6. Design: (1) chunk the documents (400–800 tokens, 10–20% overlap, respecting section boundaries); (2) embed with a domain-appropriate model and build a hybrid BM25 + dense index with a reranker; (3) a prompt that instructs citation-by-chunk-id and requires "not stated in the provided context" when the answer is absent; (4) **then**, if the residual failures are behavioural — wrong output shape, unhelpful refusal, inconsistent tone, failure to say "I don't know" — *that* residual is the SFT project, and it needs 500–2,000 rows, not 60,000 documents.
- **Trade-offs.** RAG adds latency and a retrieval component to operate; SFT adds no latency but freezes the knowledge at training time and cannot cite. A hybrid is correct: RAG for facts, SFT for the citation-citation format and the refusal behaviour. Continued pretraining (CS-12) is the third option and is only justified if the domain vocabulary is *absent* from the base model's world — check first by asking the base model about a domain term.
- **Failure modes.** (1) Fine-tuning anyway, and discovering the model quotes a superseded policy with full confidence — the highest-severity failure in this scenario. (2) Retrieval returns the right chunk and the model ignores it, which *is* an SFT-shaped fix (train on rows where the answer is present in context and the model must use it). (3) The "I don't know" rate is 0%, meaning the model answers everything — measure it, target 3–8% on out-of-corpus prompts. (4) Chunking splits a policy across two chunks and neither is sufficient — test retrieval recall@5 on a labelled set before touching the generator.
- **Why asked:** It tests whether the candidate reaches for training by reflex. The correct answer is "not yet, and here is what would change my mind."
- **Trap:** Agreeing that fine-tuning is the answer because the user asked for fine-tuning. The follow-up question is always "what would you do instead, and how would you prove it is enough?" — and the answer is a well-prompted RAG baseline plus a held-out set.

**Q91. Design the evaluation harness for a fine-tune that ships weekly.**

- **Answer — Requirements.** Every weekly candidate is compared against the currently-served model; a regression blocks the deploy; the whole harness runs in under 20 minutes on one GPU; results are diffable across weeks.
- **Constraints.** Weekly cadence means manual review must be rare and cheap. Small team. Judge calls cost money and time. Model changes (base upgrades, quantisation) as well as data changes.
- **Design.** Four tiers, with a gate on each:

| Tier | What runs | Runtime | Gate |
|---|---|---|---|
| 0 Static | schema validator, `supervised_token_frac`, contamination count, `max_length` truncation count | <1 min, CPU | any failure = do not train |
| 1 Mechanical | format compliance on 200 held-out prompts, refusal rate on a benign 200-prompt suite, mean-length drift | 5 min | ≥95% / ≤2% / <15% drift |
| 2 Task | schema validity, per-field accuracy, swap-corrected LLM-judge win rate vs the *served* model, with CIs | 10 min | win rate ≥65%, no field −2 pts |
| 3 Regression | MMLU-500 and IFEval-200 subset, and a prompt-mutation sweep (canonical vs 10 mutations) | 10 min | MMLU ≥ −1 pt, mutation gap <5 pts |

- **Design (continued).** Everything versioned: harness code, eval set (frozen, dataset-versioned), judge prompt, and the report JSON schema. The report is a single JSON per run, keyed by `{dataset_version, base_model_revision, template_hash, adapter_hash}`, and the diff between two weeks' reports is the release note. Human review is **only** triggered by a red gate — 50 generations read by hand, which is enough to catch the failures the metrics miss (voice, tone, an oddly confident wrong answer).
- **Trade-offs.** A 200-prompt eval set gives ±3 points of noise at p=0.9, so the win-rate gate at 65% has to be interpreted with that band — a 67% result is not a win. Judge calls cost real money at weekly cadence; cache by `(prompt, response)`, which makes re-runs nearly free. A frozen eval set eventually overfits by iteration — the discipline is to add new held-out rows every quarter and *retire* the ones the team has seen too often.
- **Failure modes.** (1) The eval set is regenerated each week, which makes every week incomparable. (2) The judge prompt is edited and historical numbers become meaningless. (3) The gate is measured on the fp16 training artefact and the *served* artefact is 4-bit — evaluate what you ship. (4) The prompt-mutation sweep is skipped because it "always passes", and a template change reintroduces the canonical-only behaviour.
- **Why asked:** Continuous evaluation is the difference between a demo and a product, and the numbers (200 prompts, ±3 points, 20 minutes) show whether the candidate has actually operated one.
- **Trap:** Designing a harness with no noise floor. Any gate stricter than ±3 points at n=200 will block good deploys at random.

**Q92. Design a data-curation pipeline for 200,000 raw internal instructions with no labels.**

- **Answer — Requirements.** Produce a training set of 5,000–20,000 rows with a documented provenance chain; every row justifiable; every filter's effect measured; the pipeline reproducible from the raw dump.
- **Constraints.** No labels, unknown quality distribution, unknown contamination, PII present, no budget for a human review of 200,000 rows (only of the survivors).
- **Design.** Eleven filters in order, each logging its input and output count:

| # | Filter | Mechanism | Expected retention |
|---|---|---|---|
| 1 | Schema | required fields present, roles valid, no empty assistant turn | 90–98% |
| 2 | Length | reject <10 tokens or >`max_seq_length`; flag response >3x median | 90–95% |
| 3 | Exact dedup | sha256 of the normalised instruction | 80–95% |
| 4 | Near-dedup | MinHash-LSH, Jaccard ≥0.8 on 5-grams | 70–90% |
| 5 | Decontamination | drop any row sharing a 13-gram with the frozen eval set | 95–99% |
| 6 | Repetition | n-gram self-similarity within the response | 95–99% |
| 7 | Refusal/meta | drop rows that are apologies, meta-commentary, or template leakage | 90–95% |
| 8 | LLM-judge | score 1–5 on helpfulness+correctness; keep ≥4 | 40–70% |
| 9 | Complexity | drop trivially-short or degenerate pairs (token Jaccard ≥0.3 → copy task) | 70–90% |
| 10 | PII | regex + NER scrub or drop; log the category, never the value | 95–99% |
| 11 | Language ID | keep the target language(s) | 90–99% |

- **Design (continued).** Then **balance** rather than filter: cap any single task type at 25% of the set, cap any instruction prefix at 1%, and verify diversity on three axes — distinct leading verbs (43 over 1,000 rows is healthy, 6 is a rut), response-length p5/p95 spread (10–20x), and ≥20% multi-turn rows if the product is conversational. Finally, **freeze**: `train.jsonl`, `eval.jsonl`, `MANIFEST.json` with sha256s of every file, the template hash, the seed, the filter counts, and the parent version. Every row carries a `provenance` field naming its source document or ticket id.
- **Trade-offs.** Each filter is a recall/precision trade: the judge (8) is the most aggressive and the most expensive — run it on a stratified sample of 2,000 first and measure its agreement against 100 human labels before spending it on 200,000. Near-dedup at 0.8 is aggressive and will merge legitimately similar rows (two claims about the same policy); raise to 0.9 and accept some duplication if the task is templated. PII scrubbing may remove the entity the task depends on — scrub to a placeholder and keep the row rather than dropping it.
- **Failure modes.** (1) Running the judge first, which is 20x the cost for a worse result than cheap filters first. (2) Deduplicating *within* train but not *against* eval — the leak that inflated CS-13 §15.2's reported validity from 98.9% to 99.9%. (3) Not logging per-filter counts, so the pipeline cannot be tuned and nobody can explain why 200,000 became 3,000. (4) A filter that fails open: an exception in the PII scrubber silently passing rows through. Assert `output_count ≤ input_count` for every stage.
- **Why asked:** Data is 80% of the project; the ordering and the logging are the senior signals.
- **Trap:** Designing a pipeline that ends with "and then we pick the best rows." Selection without a stated criterion and a measured retention rate is not a pipeline.

**Q93. Design the rollout plan for replacing a prompted GPT-4 endpoint with a fine-tuned 8B model.**

- **Answer — Requirements.** Match or beat the current quality on the task; cut per-request cost and latency; no user-visible regression; a rollback that works in under 5 minutes.
- **Constraints.** The current system is the baseline and it is *good* — the 8B must beat a frontier model on this narrow task, which is possible only because SFT specialises and the prompt can shrink. Serving cluster exists. Compliance needs an audit trail.
- **Design.** Five phases. (1) **Measure the baseline properly**: the *well-prompted* GPT-4 on the frozen 200-row held-out set, with the same metrics you will use for the candidate, including cost-per-request and p95 latency. Without this you have no target. (2) **Shadow** — run the candidate on live traffic in parallel, log both outputs, compare offline with the judge, change nothing for users. Run this for at least a week to cover traffic variance. (3) **Canary** — 5% of traffic, with automatic rollback on a red metric; compare task validity, refusal rate, length drift and a customer-facing proxy (thumbs, escalation rate). (4) **Ramp** 5 → 25 → 50 → 100% over two weeks, one gate per step. (5) **Retire** the old endpoint but keep it callable for 30 days, because rollback windows are cheap and un-retiring a deleted dependency is not.
- **Trade-offs.** A shorter prompt is the main cost win (a 4,000-token system prompt → 300 tokens is roughly a 13x input-cost reduction), but it removes the few-shot examples that were implicitly specifying the task — the SFT data must contain those specifications explicitly. Shadow traffic doubles serving cost during the phase, which is the price of not having an outage. Keeping the old model warm costs money and is worth it for the first month.
- **Failure modes.** (1) The baseline was the *zero-shot* base model, not the well-prompted GPT-4, which is the single most common way this project is declared a success and then fails in production (CS-13 §15.5 has it as a real post-mortem row). (2) The canary looks clean because live traffic is 90% one task type and the failure is in the other 10% — stratify the canary by task type, not uniformly. (3) Latency regresses because the 8B generates longer answers (the verbosity signature); cap `max_new_tokens` at the baseline's p95 length. (4) Cost is computed at the wrong volume — recompute at peak, not average.
- **Why asked:** It is the business case made concrete, and the phrase "versus the well-prompted baseline" is the tell that a candidate has done this before.
- **Trap:** Framing the win as "we replaced GPT-4 with an open model." The win is "same or better task quality at N% lower cost and M ms lower p95, measured on a frozen set, with a rollback that works." Anything else is a demo.
**Q94. Design masking and packing for a multi-turn agent trace that includes tool calls.**

- **Answer — Requirements.** The model must learn to emit tool-call JSON and to *use* tool results; it must not be trained to predict its own tool's output; interleaved turns of 20–60 messages; long traces (2k–8k tokens); throughput matters because the dataset is 30,000 traces.
- **Constraints.** Multiple role types (`system`, `user`, `assistant`, `tool`), nested structures inside assistant turns (text then a tool call then more text), and a template with `{% generation %}` markers that must be verified to exist.
- **Design.** Store the trace in OpenAI `messages` with `tool_calls`/`tool` roles preserved. Mask `system`, `user` and **`tool`**; unmask `assistant`. That last one is the whole design: a `tool` message is *input* the model reads, not output it produces — training on it teaches the model to hallucinate API responses, which is CS-13 §15.4's first case study defect. Use `assistant_only_loss=True` if the template carries `{% generation %}` around each assistant span including its tool call; if the tool call sits *outside* the generation markers, the template needs the four-line edit from CS-13 §4.3.4 before it is usable. For throughput, enable `packing=True` with `DataCollatorWithFlattening(return_position_ids=True, return_flash_attn_kwargs=True)`, and **re-assert after packing** that no packed sequence contains an all-masked example.
- **Trade-offs.** Packing gives 2–5x throughput on 30,000 traces and is the reason this design is affordable; the cost is the position-id correctness requirement and the difficulty of per-row attribution during debugging (packed indices no longer map to row indices — log the row id in a parallel array). Masking the tool call itself is defensible for *some* agent designs (if you want the model to learn to produce the call it already produced), but the default must be to mask anything the model does not generate at inference.
- **Failure modes.** (1) Training on `tool` messages → the model emits fake API responses; the signature is JSON that looks like a real tool result appearing in the assistant turn *without* a preceding call. (2) A trace truncated mid-tool-call leaves a dangling call — reject traces whose last message is an unmatched tool call, or whose first message is a tool result with no preceding call. (3) Packing without `position_ids` reset → cross-contamination between traces, which in an agent setting means the model conditions on another conversation's tool results. (4) The template's `{% generation %}` markers exist but wrap the *text* of an assistant turn and not its tool call, so the call is masked and the model never learns to emit it — assert that every assistant span has ≥1 unmasked token.
- **Why asked:** Agent SFT is where masking bugs are most damaging and least visible, because a hallucinated tool result looks plausible in a log.
- **Trap:** Treating a tool trace as a normal multi-turn conversation. The `tool` role has no analogue in the Alpaca/ShareGPT mental model and gets masked by neither the naive nor the string-matching approach.

**Q95. Design a fine-tuning project where the requirement is that the model says "I don't know" more often.**

- **Answer — Requirements.** Reduce confident wrong answers on out-of-scope questions without making the model useless; the current model answers everything. Target: abstention rate on a known out-of-scope set ≥90%, on a known in-scope set ≤5%.
- **Constraints.** Abstention is a *behaviour*, so it is SFT-shaped — but the model must also keep answering the in-scope questions it currently handles, so the fix cannot be a blanket refusal prior.
- **Design.** Three parts. (1) **Build the measurement first**, because "says I don't know" is otherwise unfalsifiable: a frozen **out-of-scope** set (200 questions whose answers are genuinely absent from your corpus) and a frozen **in-scope** set (200 questions with known answers). Current model: likely 0% and 95%. (2) **Data**: add **3–8% abstention rows**, written in the house voice, covering the *categories* of absence rather than one phrasing — not-in-corpus, superseded policy, requires-a-human-approval, out-of-scope-entity, and ambiguous-question. Write 5–10 phrasings of each so the model learns the *condition*, not the string. Add 2–5% refusal-boundary rows for things it genuinely must decline. (3) **Train at 1–2 epochs, low LR, LoRA** — this is a small behavioural edit and 3 epochs will overfit the abstention phrasing into a blanket prior.
- **Trade-offs.** Abstention trades coverage for precision, and the exchange rate is the product decision, not the engineer's: moving the threshold changes how many real questions go unanswered. Make it explicit — "at 92% out-of-scope abstention we lose 4% of in-scope answers; is that the trade you want?" The alternative to SFT is a **retrieval-confidence gate at serving time** (abstain when the top retrieval score is below a threshold), which is instantaneous, tunable, and needs no training — always present it, and prefer it when retrieval is in the loop.
- **Failure modes.** (1) **Blanket refusal** — the model learns "when in doubt, refuse" and in-scope accuracy collapses; the two-sided metric is what catches it, so never measure abstention alone. (2) Abstention rows written in one rigid phrasing → the model emits that exact sentence for everything, including cases where a hedged answer was correct. (3) The out-of-scope eval set was written by the same person who wrote the training rows, so it measures phrasing recognition, not behaviour. (4) The "unknown" rows are all in the same domain, so abstention generalises only to that domain.
- **Why asked:** It is a behaviour-only intervention with a two-sided metric, which makes it a clean test of whether the candidate designs for the trade-off or for the metric.
- **Trap:** Measuring abstention rate without the in-scope accuracy on the same chart. Every one-sided safety metric can be maximised by a model that says nothing.

**Q96. Design the VRAM and cost plan for fine-tuning 70B when you only have 4x A100-80.**

- **Answer — Requirements.** Get a usable 70B SFT run on 320 GB of VRAM across 4 GPUs; produce a costed plan before renting anything.
- **Constraints.** From the repo's table: 70B full FT **913.8 GiB**, LoRA r16 **144.5 GiB**, QLoRA r16 **46.8 GiB** (single-GPU figures, gradient checkpointing on). 4 × 80 = 320 GB of *card capacity* — decimal, as the vendors quote it — and you need 10–20% headroom, so the usable budget is ~260 GB.
- **Design.** Do the arithmetic on the table first. **QLoRA r16** is 46.8 GiB for a single GPU — it fits on *one* A100-80, so the other three are for parallel experiments, not for this run, and the answer is "do not use 4 GPUs; use one and run four configs." If you want **LoRA r16** (144.5 GiB, better quality than QLoRA), it fits on 2 GPUs with FSDP or on 1 A100-80 with the adapter's activation memory reduced (`gradient_accumulation_steps` up, micro-batch 1, `max_length` measured rather than guessed). **Full FT** at 913.8 GiB requires 16 GPUs with ZeRO-3 — it does not fit in 320 GB under any sharding configuration you would want, so the honest answer is "not on this hardware." Cost at the module's throughput figures (~380 tokens/s/GPU for 70B QLoRA, vs 4,000 for 8B LoRA): a 10,000-row × 1,024-token × 2-epoch run is 20.5M tokens → at 380 t/s that is ~15 GPU-hours ≈ **$22** on one GPU, or ~3.8 hours wall clock.
- **Trade-offs.** QLoRA's 4-bit base means the adapter learns around quantisation error; the quality gap to LoRA is small (0–2 points typically) and QLoRA's VRAM advantage is 3x, which is what buys the extra experiments. LoRA needs the base in bf16 (140 GB) which forces sharding. Full FT is out of reach and the honest comparison is 16 × A100-80 for ~3 hours ≈ $72 — which is *cheap in dollars* and *expensive in the machine you have to find*.
- **Failure modes.** (1) Sizing from `n_params × 2 bytes` (140 GB = **130.4 GiB**) and concluding 2 GPUs suffice for full FT — the optimiser state is another 560 GB = **521.5 GiB** (`8 B/param`), and gradients another 140 GB; the table prices the whole thing at 913.8 GiB. (2) Forgetting the 10–20% allocator overhead and OOMing at 92% of the calculation. (3) Enabling `packing` without `padding_free`, which raises activation memory and moves you over the line. (4) Choosing 70B when an 8B with a 300-token prompt meets the requirement — size the model from the task, not the reverse.
- **Why asked:** It is the pure arithmetic question that determines whether a plan is fundable, and the "use one GPU, run four experiments" answer is the senior move.
- **Trap:** Answering "rent 16 A100s" without noticing that the requirement never said 70B was necessary. The first trade-off is model size, not sharding strategy.

**Q97. Design the data flywheel: how does this model get better after it ships?**

- **Answer — Requirements.** Every production failure should become a training row; the loop must run without a dedicated labelling team; each cycle must be measurable against the frozen eval set.
- **Constraints.** Production logs are the only scalable source; labelling is the bottleneck (20 min/expert row); you cannot train on user data without consent and PII scrub; the eval set must stay frozen or nothing is comparable.
- **Design.** Four stages. (1) **Capture with a triage signal** — log the prompt, the response, the model version, the template version, and one of: an explicit thumbs-down, a downstream validator failure, an escalation, or a retry. Validator failures are the highest-value and cheapest signal, because they are automatic. (2) **Triage weekly** — cluster failures by embedding similarity, count them, and take the top clusters by volume × severity. This is where "we have 400 failures" becomes "the model is wrong about retroactive dates, 210 times." (3) **Label the top 50–200** with an expert-written ideal answer in the `messages` format, plus — this is the cheap part — a plausible-but-wrong `rejected` answer while you have the expert in the room, which is 20% more time and gives you a DPO set for free. (4) **Retrain monthly** on `previous dataset + the new rows`, at the same epochs and LR, and run the full harness. Add, never replace, unless a row is provably wrong.
- **Trade-offs.** Monthly retraining means the model drifts out of sync with production prompts, so the template contract must be enforced per version. Adding only positive rows from failures over-weights the failure distribution — cap new rows at 10–20% of the dataset per cycle, which also stabilises the eval deltas. The alternative to retraining is a prompt fix or a retrieval fix, and the triage step should route obvious prompt bugs there instead.
- **Failure modes.** (1) Labelling without a triage step, so the flywheel spends expert time on the rarest failure. (2) The eval set silently drifts because the new failures get appended to it, destroying comparability — hold it frozen and add a *separate* rolling set. (3) Training on logged PII, which is a compliance incident and also teaches the model to reproduce that PII. (4) Never retiring a dataset — the set grows monotonically until it is 200,000 rows for the same task, at which point nothing improves and nobody knows why.
- **Why asked:** It is the question that separates "we shipped a model" from "we operate a model," and the rejected-answer-for-free trick is a strong senior signal.
- **Trap:** Designing the flywheel to collect thumbs-up/down only. Explicit feedback is rare (1–3% of traffic) and biased toward extremes; validator failures and retries are dense and unbiased.

**Q98. Design SFT for a model that must never speak in the first person and must always cite a section number.**

- **Answer — Requirements.** Two hard constraints, verifiable mechanically, applied to every output; the model must remain useful; violations must be detectable in production, not just at training time.
- **Constraints.** Both constraints are *format*, which means they are the SFT sweet spot — but they are also exactly the kind of constraint a model can satisfy in training and violate in production under distribution shift (a longer prompt, a new document type).
- **Design.** Treat it as a three-layer problem. (1) **Make it mechanically checkable first**: a validator that flags first-person pronouns and a regex that requires `§\d+(\.\d+)?` — run it over the *training set* and fix any row that violates it, because a single violating row teaches the violation. (2) **Data**: 1,000–3,000 rows, of which 10–20% are adversarial — documents that invite first person ("what do *you* think?"), questions with no citable section, and prompts that attempt to extract a personal opinion. Those rows teach the constraint under pressure, which is the actual requirement. Add 3–8% rows where the correct answer is "no section addresses this." (3) **Train** LoRA 1–2 epochs, LR 1e-4, `assistant_only_loss=True`. (4) **Gate at serving**: the validator runs on the output with one retry at temperature 0, and the violation rate is a monitored metric, not just a training metric.
- **Trade-offs.** A hard serving-side gate means occasional doubled latency (retry) and occasional failures (both attempts violate) — decide up front whether "return a canned refusal" or "return the violation" is the better failure, and the answer is usually the canned refusal. Pure SFT without the gate will be 95–99%, not 100%, and the requirement says "never". **Constrained decoding cannot express these constraints** (a pronoun blocklist and a section-number regex are not a context-free grammar you would want to compile), so SFT plus a validator is the only practical route — which is a useful thing to say out loud, because it shows you considered the cheaper tool and rejected it for a specific reason.
- **Failure modes.** (1) One violating training row, discovered only in production — run the validator over the dataset in CI and fail the build. (2) The constraint holds on canonical prompts and breaks on a system-prompt-less request: include 40% system-prompt-absent rows and test both ways. (3) The model satisfies the regex by emitting a plausible-looking but *wrong* section number, which passes the validator and fails the user — measure whether the cited section actually supports the claim, on a sample, with a human. (4) The retry loop masks a degradation: retry rate is the metric to watch, and it rises before the violation rate does.
- **Why asked:** It is the cleanest example of a constraint that is simultaneously an SFT problem and a serving-gate problem, and the interviewer wants to hear that fine-tuning alone does not achieve "never."
- **Trap:** Promising 100% from SFT. Nothing in SFT is deterministic; the guarantee comes from the validator, and the SFT exists to make the validator's retry rare.

---

## Level 5 — Debugging & Incident Response

> Answer in a fixed shape: **what I look at first, what each observation rules in or out, and the fix**. The interviewer is grading the *order* and the *discriminating power* of each check, not the completeness of the list.

**Q99. Your training loss is completely flat at 10.37 for 2,000 steps with a 32k vocabulary. What do you check, in what order?**

- **Answer.** `10.37 = ln(32,000)`. A flat loss at exactly `ln(V)` means the model is producing a uniform distribution at every position — it is learning nothing at all. Order of checks:
  1. **Are there any non-`-100` labels?** Print `labels[:50]` and the count of `l != -100`. If every label is `-100`, the loss is `0/0` (or a clamped zero) and nothing is optimising. This is the most common cause and the cheapest check. → Fix the masking helper or the truncation, not the LR.
  2. **Is the loss connected to the parameters at all?** Print `sum(p.requires_grad for p in model.parameters())` and `model.print_trainable_parameters()`. If the trainable count is 0, PEFT attached nothing — usually a wrong `target_modules` list for the architecture, or an adapter created against a different module naming scheme. → Fix `target_modules` and verify with a non-zero count before launching.
  3. **Is the LR zero or absurdly small?** With LoRA at `1e-5`, loss barely moves in 2,000 steps; with a cosine schedule and `warmup_ratio=0.03` on a very long run, the first 2,000 steps may be inside warmup. → Print the LR at each step, don't read it from the config.
  4. **Are the labels actually being passed to the loss, or is the model computing loss against `input_ids` shifted from a *different* tensor?** A shape-correct but semantically wrong `labels` argument is silent. → Assert `labels.shape == input_ids.shape` and that `labels[0][:n] == -100`.
  5. **Is the collator dropping the `labels` key?** A custom collator that returns `{input_ids, attention_mask}` with no `labels` makes `Trainer` fall back to `input_ids` — which is a *different bug that produces a moving loss*, so if the loss is flat this is ruled out.
- **Why asked:** It is the single most common "my fine-tune does nothing" report and the discriminator is arithmetic — `ln(V)` is an exact number, not a range.
- **Trap:** Restarting with a higher LR. At `ln(V)` the model is not underfitting, it is disconnected; a higher LR on a disconnected graph produces the same flat line plus a bigger GPU bill.
**Q100. Training loss is 0.05 and eval loss is rising. What happened, and what do you do?**

- **Answer.** Two separate signals pointing the same way: 0.05 train loss on an SFT task is **memorisation** — CH-13's ten-fact summary encodes it as "training loss below ~0.5 means memorisation" — and a rising eval loss confirms it. Order of checks:
  1. **How many epochs?** If >3 on a 1k–10k row set, this is expected and the fix is to stop at 1–2. Check the per-epoch eval curve, not the final number: pick the checkpoint at the *minimum* of a composite metric, not of eval loss (which is the trap in Q70).
  2. **How many examples and how many steps?** 500 rows × 5 epochs at batch 8 is 300 steps, and a 7B LoRA at `2e-4` will memorise that in under 100. Compute steps and compare against `n_rows / effective_batch × epochs`.
  3. **Is the eval set actually held out, and is it decontaminated?** A 0.05 train loss against a *contaminated* eval set produces a *falling* eval loss; against a clean one it rises. Run the 13-gram check — if 15%+ of eval rows share a 13-gram with train, every number here is meaningless and the fix is the split, not the schedule.
  4. **Is the LR in the wrong regime?** `2e-4` full FT is 10x too high and produces a fast descent to a memorised solution; `2e-4` LoRA is correct. Check which method is actually running.
  5. **Are the eval and train losses averaged over the same denominator?** Different masking policies between the two make the comparison meaningless — verify both were built with `build_masked_example` and the same template.
- **Fix:** retrain at 1–2 epochs with the same config, keep the epoch-1 checkpoint, and add the four overfit signatures to the eval harness (verbosity, format rigidity under prompt mutation, refusal rate, MMLU delta). Do not lower the LR as the first move — it slows the memorisation rather than changing its direction, and you will simply arrive at the same place with more steps.
- **Why asked:** It is the canonical SFT-specific failure, and the correct response involves reading the *shape* of the curve rather than reacting to a number.
- **Trap:** Early-stopping on `eval_loss` with `load_best_model_at_end=True`. On an in-distribution eval set the minimum eval loss is often the *most* memorised checkpoint, so the mechanism selects the worst model.

**Q101. Loss is `NaN` at step 0. What is it?**

- **Answer.** In order of likelihood for SFT specifically:
  1. **An all-masked row.** `0/0` in the mean over non-ignored positions. Check `token_stats(...)["without_supervision"]` — if it is non-zero you have found it. Usually caused by truncation cutting before the answer, or a masking helper that failed to locate the response marker on one row out of 5,000. → Drop or fix the row, and add the assertion to the data build, not to the training script.
  2. **fp16 without loss scaling.** `fp16=True` with an unstable LR produces `inf` in the loss and `NaN` after the first backward. → Use `bf16=True`; it has the same exponent range as fp32 and needs no scaler. This is why every modern config in this module says bf16.
  3. **An LR that is 100x too high.** A LoRA-scale LR applied to a full fine-tune diverges within a handful of steps. → Check the actual peak LR after warmup; the first steps at LR≈0 will look fine, so the NaN appears at the end of warmup, not at step 0. If the NaN is truly at step 0, this is ruled out.
  4. **A token id outside the embedding range.** Adding a special token without `resize_token_embeddings` gives an index error at the embedding lookup — usually a loud `IndexError`, but a *negative* or wrapped index can produce a `NaN` embedding. → `assert max(input_ids) < model.config.vocab_size`.
  5. **A `0/0` in the *data* itself** — a `NaN` or `inf` in a float label field, e.g. a JSON `NaN` that `json.loads` accepted. → `assert all(isinstance(l, int) for l in labels)`.
- **Why asked:** It is a short question with a genuinely ordered differential diagnosis, and the all-masked row is the SFT-specific one that generic debugging lists miss.
- **Trap:** Adding `max_grad_norm=1.0` and moving on. Gradient clipping does not fix a `0/0` in the loss; the NaN reappears at the next occurrence of the bad row.

**Q102. Works perfectly in the notebook, produces garbage from the endpoint. Where is the bug?**

- **Answer.** This is the train/serve mismatch, and there are exactly seven places to look, in order of frequency:
  1. **Different template.** The notebook called `apply_chat_template`; the server sends a raw string, or a template from a different model family, or the server's default template because none was passed. → Compare `repr()` of both strings byte for byte; this catches it in one command.
  2. **Missing or extra `add_generation_prompt`.** Training had it off (or on) and serving had it the other way; the model sees a conversation that is one role-marker away from what it learned. → Check the last 20 characters of the served prompt.
  3. **Different system prompt.** Trained with the schema spec, served without (or with a reworded one). If fewer than 100% of training rows carried it, this should *degrade*, not *break* — a hard break means the model never saw a system-less prompt at all.
  4. **Truncation at serving.** `max_model_len` or a server-side context cap silently cuts the input, and if it cuts the *end* it removes the assistant header entirely. → Log the token count of every request.
  5. **Sampling parameters.** The notebook used `temperature=0.0` (greedy) and the server defaults to `temperature=0.7, top_p=0.9`. The model is fine; it is sampling badly. → Test at temperature 0 first; it is free and it is the fastest way to separate a model problem from a decoding problem.
  6. **A quantised artefact was served without evaluation.** The notebook evaluated bf16; the server runs AWQ/NF4 and the format compliance dropped. → Re-run the harness on the served artefact.
  7. **The adapter is not loaded at all.** Output is byte-identical to the base model. → Hash a weight or compare a known prompt's output to the base model's.
- **Why asked:** "Works on my machine" is the most common production incident in this module, and the ordering — template first, sampling fifth — reflects how often each occurs.
- **Trap:** Debugging by regenerating with different prompts. All seven causes are environment differences, and none of them is prompt-sensitive in a way that makes the model behave correctly.

**Q103. The model repeats the user's question back before answering, and sometimes never stops. What are the two bugs?**

- **Answer.** These are two different bugs that co-occur, which is why they are asked together.
  **Repeating the question** = the mask boundary is in the wrong place. If the prompt's final tokens are not masked — specifically, if the assistant header is *inside* the unmasked span — the model is trained to produce "…<|im_start|>assistant\n" as *content*, so it continues by reproducing the header and then the user's turn. Check `labels[:30]`: the first non-`-100` index must be the first answer token, not the header. Two causes: the manual mask sliced at the wrong index (BPE non-concatenativity, Q65), or `add_generation_prompt=True` was left on for every turn while building (the CH-13 §5.2 snippet's actual bug), which appends a spurious assistant header after each assistant message and trains the model to open a new turn.
  **Never stops** = **no EOS in the target**. The response span ends with content and no terminator, so the model never learns the stopping token. Check the last unmasked label: it must be the template's end-of-turn id (`<|im_end|>`, `<|eot_id|>`, `</s>`). Two causes: rows built by string concatenation without the terminator, and a collator or tokenizer step that strips trailing special tokens.
- **Diagnosis order:** decode `labels` back to text with `tokenizer.decode([l for l in labels if l != -100])`. That single line shows both bugs at once — you will see whether the span starts with a header and whether it ends with EOS. It is the highest-information debugging step in the entire module.
- **Why asked:** These are the two most-reported production symptoms and they have completely different fixes; conflating them wastes a week.
- **Trap:** Fixing the non-stopping with a `repetition_penalty` or a stop string. Both mask a data bug and degrade legitimate repetition in JSON, code and lists — and neither fixes the header, so the model still says the header twice.

**Q104. The fine-tune is worse at everything — worse in-domain, worse general, and less obedient. Nothing in the config looks wrong. What is it?**

- **Answer.** The most likely cause by a wide margin is the **eval harness**, not the model. Check in this order:
  1. **Is the "worse" model actually the model?** Hash a weight, or run a prompt through the base model directly and diff the output. If identical, the adapter was never loaded — the notebook's `AutoModelForCausalLM.from_pretrained(adapter_dir)` bug, which builds a *randomly initialised* model in some paths and the base model in others.
  2. **Is the serving template the training template?** "Worse at everything" is exactly the symptom of a systematically wrong prompt format — the model is not broken, it is being asked a question in a language it does not speak.
  3. **Is the comparison against the right baseline?** If the baseline is the *already-served* model at temperature 0 and the candidate is at temperature 0.7, the candidate loses on sampling noise across a diverse eval set. → Re-run both at the same temperature.
  4. **If the harness is clean, then the model is genuinely broken**, and the two config causes are: LR 10x too high (full FT at LoRA-scale LR destroys pretrained features in the first 100 steps — check the loss curve for a spike then a plateau), and **the prompt was not masked while the model was trained as a *completion* model**, which teaches it to complete prompts rather than answer them (CS-13 §15.5's post-mortem row: "we accidentally fine-tuned it to complete the prompt with questions").
  5. **A dataset-wide defect**: every row's answer is a *continuation* of its instruction rather than a response to it, usually because the synthesis pipeline generated instruction and answer as one paragraph and split it at a random point. → Sample 20 rows and read them; this is the check people skip because it is not automatable, and it is the one that finds this class of bug.
- **Why asked:** "Worse at everything" is the strongest possible signal that the bug is upstream of the model, and the answer's first three checks are all harness checks.
- **Trap:** Retraining with a lower LR. If the harness is the bug, the retrain reproduces the failure exactly, at cost, and you now have two models you cannot evaluate.

**Q105. You OOM at a batch size that worked yesterday. Same machine, same config. What changed?**

- **Answer.** OOM is deterministic for a fixed config and a fixed data order, so "same config" is almost never true. Order:
  1. **Sequence length.** If data is sharded or streamed, the *order* changed and a batch of 512-token rows is now 2,048-token rows. `max_length` caps it only if you actually set it. → Log the max length per batch; the memory profile of a batch is set by its longest row.
  2. **Packing turned on.** `packing=True` fills every window to `max_seq_length`, so the average token count per forward pass rises from p50 to max. A config that fit at p50 lengths can OOM at max lengths. → That is the point of packing (you are buying throughput with the VRAM you had spare).
  3. **A config line changed in a *different* file.** `gradient_checkpointing` disabled "temporarily" in a branch, or `gradient_accumulation_steps` reduced so micro-batch went up, or `optim="adamw_torch"` instead of `paged_adamw_8bit`. → Diff the actual resolved config, not the one you remember: `print(trainer.args)`.
  4. **Another process on the GPU.** A leftover eval job, or a `wandb` process holding memory. → `nvidia-smi` and look at per-process memory, not just the total.
  5. **The model or tokenizer changed.** A new base revision with a larger vocab, or `resize_token_embeddings` after adding tokens, adds parameters and optimiser state. → Compare `sum(p.numel() for p in model.parameters())` against the last known value.
  6. **Fragmentation, not capacity.** Repeated allocation of variable-length tensors fragments the allocator; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` often recovers 5–15% on a long-running process. This is the one genuine "same config" cause, and it is recognisable because the failure is at a *random* step rather than at step 1.
- **Why asked:** It tests whether the candidate treats OOM as arithmetic or as weather, and the fragmentation case is a real one with a specific signature.
- **Trap:** Setting `per_device_train_batch_size` down and `gradient_accumulation_steps` up without recomputing the effective batch — you have silently changed the optimisation, and now you are comparing two runs that differ in tokens/step.

**Q106. Two runs with the same seed and the same config give different results. What is non-deterministic?**

- **Answer.** Six sources, and only one of them is fixable at the level people expect:
  1. **FlashAttention's backward pass is nondeterministic by design.** It uses atomics in the gradient accumulation, so floating-point addition order varies between runs. This is the dominant source and it cannot be fixed without disabling flash attention. → `attn_implementation="eager"` (or `sdpa`, which is more deterministic but still not guaranteed) plus `full_determinism=True`.
  2. **`torch.use_deterministic_algorithms(True)` is not set.** Many CUDA kernels (scatter-add, some reductions, `atomicAdd`) are nondeterministic unless the flag is on, and the flag throws on kernels with no deterministic implementation. → Set it, expect a `RuntimeError` telling you which op is the problem, and accept a slower run.
  3. **`dataloader_num_workers > 0`** with a seeded-but-shared RNG in the collator produces a different example order. → `num_workers=0`, `data_seed=42`, `dataloader_drop_last=False`.
  4. **Different hardware.** A100 vs H100, or even two A100s in different nodes, changes kernel selection and reduction order. → Report *which* GPU the number came from.
  5. **Non-deterministic data preparation.** A dataset built by a pipeline with an unseeded shuffle, a `set` iteration order, or a timestamp field differs between builds. → Hash the dataset file, not the dataset version string.
  6. **Sampling at evaluation.** `do_sample=True` in generation, or a judge model with temperature > 0. → Temperature 0 for eval, always.
- **How to set the expectation:** with `full_determinism=True`, `num_workers=0`, eager attention, the same GPU and the same software stack, bit-exact reproduction is achievable. Across GPUs it is not, and the practical definition of "reproduced" is **eval metric within ±1 point across three seeds**. Say that out loud — promising bit-exactness on GPUs is the trap.
- **Why asked:** Reproducibility is a discipline question and the candidate who claims bit-exact GPU reproduction has not tried it.
- **Trap:** Chasing determinism so far that you disable flash attention on a 70B run. The 2–3x speedup is worth far more than the last decimal place of reproducibility; record the GPU and the software stack instead.

**Q107. The model emits the literal string `None`, or an empty answer, for some inputs. What is it?**

- **Answer.** A data-construction artifact that is nearly always in the *input* representation, not the model.
  1. **The `null` → `"None"` conversion.** A template that does `f"### Input:\n{row['input']}"` renders Python's `None` as the literal text `None`, and the notebook's dataset has rows with an empty context stored as `None`. The model then learns that some prompts legitimately contain `None` — and, worse, learns to *emit* it, because rows where the target field was absent had `None` written into the target. → Check the raw JSONL: `grep -c '"input": null'`. Then fix the template (`row.get("input") or ""`) **and** rebuild, because the model has already learned the artifact.
  2. **Empty assistant turn.** A row with `"output": ""` teaches the model that silence is an acceptable response for that input shape. → Filter at data-build time: `assert row["messages"][-1]["content"].strip()`.
  3. **A failed generator that wrote an error string** — `"I cannot help with that"`, `"Error"`, or `"None"` — into the target. → grep the dataset for a blocklist of generator artifacts.
  4. **Truncation to zero supervised tokens on that input shape**, so the model learned from the other rows that this shape gets no supervision and emits the highest-prior token. → This is the same signature as Q101's all-masked rows, seen from the generation side.
- **Diagnostic that settles it in one command:** `tokenizer.decode([l for l in labels if l != -100])` for the offending input's *nearest training row*. If the nearest row's target contains `None` or is empty, it is the data; if its target is clean, the input construction is adding the literal.
- **Why asked:** It is a concrete, memorable bug (CS-13 §4.1 documents the `### Input:\nNone` case) that tests whether the candidate treats generation artifacts as data bugs or as model problems.
- **Trap:** Prompt-engineering around it at serving time (adding "do not output None"). The behaviour is trained in; the fix is to remove the training rows that contain it and retrain, because the model has seen thousands of examples with the artifact and a handful of serving-time instructions is a rounding error against that.
**Q108. Format compliance is 98% and users are complaining. What do you measure next, and what is the likely fix?**

- **Answer.** 98% compliance against a *canonical* prompt is not the production number. Measure, in this order:
  1. **The prompt-mutation gap.** Take 50 held-out prompts and generate 10 mutations of each: a synonym in the instruction (`summarise` → `summarize`), a leading space, a different system prompt (present → absent → reworded), an extra sentence of context, and a reordered field. Compute compliance on canonical vs mutated. A **>25-point gap** means the model learned the *template*, not the *format*, and CS-13 §4.7 documents exactly this signature.
  2. **Compliance on the served artefact.** If training was evaluated on the fp16 adapter and production runs 4-bit, re-measure there; quantisation drift (unclosed tags, dropped stop token) is a real 1–3 point effect.
  3. **Non-emptiness and required keys.** "Valid JSON" is satisfied by `{}`. Check that every required field is present and non-empty, and that the enum fields contain a member of the enum.
  4. **The failure taxonomy, not the rate.** 2% of 100,000 requests is 2,000 failures a day; cluster them. If they are all one input type (a document class, a very long input, a language), the fix is targeted data, not more data.
  5. **The retry rate** at the serving gate, which rises before the violation rate does and is the leading indicator.
- **Likely fix, in priority order:** (a) add 200–500 mutated-prompt rows to the training data — an afternoon's work and usually worth 5–15 points on the mutated set; (b) add the schema validator as a serving gate with one retry at temperature 0, which converts a 2% failure rate into a ~0.04% failure rate; (c) if the failures are concentrated in long inputs, check truncation; (d) only then consider retraining with a larger `r` or more epochs, which is the slowest and least targeted option.
- **Why asked:** It tests the difference between a benchmark number and a production metric, and the mutation sweep is the specific instrument for the gap between them.
- **Trap:** Retraining on more data of the same distribution. The canonical-prompt compliance is already 98%, so more of the same data will push it to 98.5% and leave the mutated prompt at 70%. The distribution is the problem, not the volume.

**Q109. After the fine-tune, the model can no longer follow instructions it used to follow. It is not worse at trivia — MMLU is unchanged. What is it?**

- **Answer.** MMLU unchanged rules out catastrophic forgetting of *knowledge*, so this is a **behavioural** regression rather than a capability regression. Three candidates, in order:
  1. **Over-narrow instruction following — the instruction-tuning distribution collapsed.** Your 5,000 rows all have the same instruction shape (one leading verb, one length, one register), so the model's conditional `p(response | instruction)` now has a narrow support and it fails on instructions that do not match. → Measure the **distinct leading verbs** in the instruction field: 6 distinct verbs over 5,000 rows means the model saw one instruction family. Fix is data diversity, not more epochs.
  2. **Over-refusal or a shifted response prior.** A dataset that is 60% "answer in 3 bullet points with a citation" turns every answer into bullets and citations, including requests where that is wrong. → Run the refusal-rate suite and a format-compliance check on out-of-domain instruction shapes (IFEval-200 is designed for exactly this).
  3. **The instruction data taught a *dominant* behaviour that crowds out others.** If 90% of rows are one task, the model becomes a one-task model and general instruction following degrades to whatever generalisation survives. → Measure the task-type histogram; no single task above 25%.
- **Additional check that is often the actual answer:** the model is not worse, the *comparison* is unfair — the previous behaviour was measured with a different template, a different system prompt, or a different `max_new_tokens`. → Re-run the old model on the identical harness.
- **Fix:** diversify instructions on the three axes (task type ≤25%, distinct leading verbs ≥40 per 1,000 rows, response length p5/p95 spread of 10–20x), add 20% general instruction data as replay, and re-evaluate the *composite* metric rather than the task metric alone. Because MMLU is unchanged, the temptation is to declare it fine; the composite metric is what makes the regression visible.
- **Why asked:** It forces the candidate to separate knowledge forgetting from behaviour narrowing, which are different diagnoses with different fixes and which a single benchmark cannot distinguish.
- **Trap:** Adding replay of pretraining text. Replay of raw text fixes knowledge forgetting, not instruction-following narrowing; for this failure the replay must be *general instruction data*, and it must not be masked.

**Q110. Multi-turn conversations work for the first turn and fall apart after. What do you check?**

- **Answer.** Five checks, in order:
  1. **Is only the last assistant turn unmasked?** If the masking helper masks everything before the final response, the model was never trained on earlier assistant turns, so it has no experience of producing a turn that will be followed by another user turn. The signature is specifically that turn 1 is fine and turn 2 is bad, because the model never saw a turn-2-shaped context during training. → Print the labels of a 3-turn training row; there should be **three** unmasked assistant spans.
  2. **What fraction of the data is multi-turn?** If it is under ~20%, the model's conversational behaviour is inherited from the base and will degrade as the conversation grows. → Count rows with ≥2 user turns.
  3. **Is the context being truncated at serving?** A long conversation exceeds `max_model_len` and the server truncates — often from the *front*, deleting the system prompt, which is the one part the model was trained to condition on. → Log the token count per turn and check which end the server truncates.
  4. **Is the role pattern in training the same as at serving?** Trained with `system, user, assistant, user, assistant`; served with `user, assistant, user, assistant` because the serving stack drops the system turn. → Render a served conversation through the training template and diff.
  5. **Is there a turn-count distribution mismatch** — trained on 2 turns, serving 20 — so the model has no representation for a long history? → Include a long tail: 10% of rows with ≥6 turns.
- **Also check the boring thing:** whether the *user* simulator in your eval is well-behaved. A multi-turn eval where each turn is generated by a model at temperature 0.7 produces an incoherent conversation, and the model looks worse than it is. Use a fixed scripted conversation, or a temperature-0 simulator, and score per turn — the degradation curve (compliance per turn index) is the metric.
- **Why asked:** Multi-turn masking is the module's most common agent-SFT defect and the failure signature (turn 1 OK, turn 2 bad) is specific enough to be diagnostic.
- **Trap:** Adding more multi-turn data without fixing the mask. If only the last turn is unmasked, 100x the data teaches exactly the same thing with 100x the compute.

**Q111. Across five saved checkpoints, response length has grown 61% and the judge scores keep improving. Ship it?**

- **Answer.** No, and the reason is that the judge is measuring the wrong thing. Three checks:
  1. **Is the length growth the *cause* of the judge improvement?** Verbosity bias makes longer answers win **60–70%** of pairs that human raters call equal quality. → Re-run the comparison with length-matched pairs (bucket generations by token count and compare within buckets). If the win rate disappears within buckets, you were measuring length.
  2. **What did the other metrics do?** CS-13's checkpoint table shows the pattern: epoch 1 → 0.91 / 92% format / +5% verbosity / −2 general; epoch 5 → 0.19 / 98% / +61% / **−14 general**. Length growth above ~15% is the first overfitting signature and it is the leading indicator of the other three. → Plot mean length, refusal rate, MMLU delta and format compliance across the five checkpoints on one chart. If length is rising and MMLU is falling, it is memorisation.
  3. **What does a human say?** Fifty generations read by hand from the epoch-2 and the epoch-5 checkpoint, unlabelled. Verbosity is exactly the failure that automated metrics reward and humans notice immediately — the answers are longer, more confident and less useful.
- **The decision:** ship the checkpoint at the *minimum* of a composite metric, not the maximum of a judge score, and define the composite before you look at the numbers. Weight format compliance (≥96% good, 99%+ excellent), refusal rate (≤2%), win rate (≥65%, swap-corrected), MMLU delta (≥ −1), and a length-drift penalty (>15% is a red flag). Note the trap in the framing of the question itself: "the judge scores keep improving" is presented as evidence for shipping, and the correct move is to distrust a metric that improves monotonically across checkpoints of the same run.
- **Why asked:** It is the module's central empirical table dressed as an incident, and it tests whether the candidate treats a judge score as a metric or as a verdict.
- **Trap:** Shipping the last checkpoint because "more training is better" or because the win rate against the *previous* checkpoint improved. Win rate against a degraded model improves monotonically while absolute quality falls.

**Q112. Users report the model answers a *different* question from the one asked. It is fluent, on-topic, and wrong. What is it?**

- **Answer.** This is the most damaging failure in the module because it is not detectable by any format or style metric. Four candidates:
  1. **Prompt truncation.** The prompt is longer than `max_seq_length` and the truncation removed the *middle* or the *beginning* — so the model answers the part it can see. → Compare the input token count against `max_seq_length` per request, and check which end your pipeline truncates. Front truncation is the worst because it removes the system prompt and the framing.
  2. **The answer is a *continuation* of the instruction rather than a response to it.** This is the CS-13 §15.5 post-mortem defect: the synthesis pipeline generated a paragraph and split it, so the "answer" completes the "question's" sentence. The model then learns to continue rather than to answer, and it produces fluent text that addresses the topic and not the question. → Read 20 training rows by hand. This is the one bug in the module that no automated check finds.
  3. **A template bug that puts the answer in the wrong slot** — the response was appended after the *user* marker instead of the assistant marker, so the model learned that the user's turn contains the answer and it produces the next user turn. → Print `repr(render(row["messages"][:2]))` for a training row; the marker before the answer must be the assistant one.
  4. **Distribution shift on the input side.** The training prompts all came from one document class and production sends another; the model pattern-matches to the nearest training shape and answers that question. → Cluster production prompts by embedding and check how many clusters were absent from training. Recall@1 against the training set's nearest neighbour is the diagnostic; a low similarity with a confident answer is the signature.
- **Fix by cause:** reduce `max_seq_length` pressure with packing or a shorter prompt (1); regenerate the dataset (2, and this one means rebuilding, not patching); fix the template and retrain (3); add 200–500 rows of the missing input class (4). In all four cases the serving-side mitigation is the same and it should ship *today*: log the prompt, run a retrieval-style check for "is this input in-distribution", and route the low-similarity tail to the prompted baseline or to a human.
- **Why asked:** It is the failure that survives every automated gate, so it tests whether the candidate has a manual-review habit and whether they can enumerate causes rather than guess.
- **Trap:** Concluding "the model hallucinated." Hallucination is a knowledge problem; this is a *task-framing* problem, and the fixes are completely different (RAG does not help when the model is answering the wrong question).

**Q113. Your last three runs were all "good" and none of them shipped. What is wrong with the process?**

- **Answer.** Three runs that all pass their gates and none of which improves production means the harness is measuring something that does not determine shipping. The causes, in order:
  1. **The baseline is wrong.** If the comparison is against a zero-shot or differently-prompted model, every run looks like a win and none of the wins are real. → The baseline must be the **currently-served model, on the identical harness**. CS-13 §15.5 lists this as a real post-mortem row.
  2. **The eval set is in-distribution and saturated.** If every run scores 96–98% on a curated held-out set, the metric has no headroom, so it cannot discriminate — and it is not measuring the thing users complain about. → Build the eval set from **production failures**, not from the curated data, and add a failure-cluster breakdown so a run that fixes the top cluster is visibly different from one that does not.
  3. **The noise floor is larger than the effect.** At n=200 the binomial standard error is ±3 points, so three "improvements" of 1–2 points are three coin flips. → Increase to n=1000 for the win-rate comparison, or pre-register the decision rule ("we ship if format ≥96% and win rate ≥65% with the CI lower bound above 60%").
  4. **No one owns the decision.** "All good" with no threshold and no owner means the runs are evaluated but not *decided upon*. → Define the composite gate and the rollback before the run, in writing, and name a person.
  5. **The improvement is real but not where the value is.** The task metric improved and the cost, latency or the top user complaint did not move, so no one can justify the migration. → Re-derive the gate from the incident that motivated the project.
- **Process fix to state explicitly:** every run gets a one-page record — dataset hash, template hash, base revision, the config, the four metric families with CIs, and a **decision** with a name and a date. Three runs with no decisions is not a pipeline, it is a hobby with GPUs.
- **Why asked:** It is a process question at the end of a technical interview, and it is the one that most reliably separates someone who has been accountable for a model from someone who has trained models.
- **Trap:** Answering with a tool ("we should use W&B / a proper CI"). Tooling is not the missing piece — the missing piece is a threshold, a baseline and an owner, all three of which can be written on one page.

---

## Rapid Fire — True / False / One-Liner

> Two minutes, no explanation beyond one line. The interviewer reads the statement; the candidate answers immediately.

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | SFT's objective is different from pretraining's. | **False** | Same next-token cross-entropy; only the data distribution and the supervision mask differ. |
| 2 | `IGNORE_INDEX = -100` matches `CrossEntropyLoss(ignore_index=...)`. | **True** | It is PyTorch's documented default and a value no real token id can take. |
| 3 | Ignored positions are excluded from the loss denominator too. | **True** | `reduction="mean"` averages over non-ignored positions only — masking the prompt raises the reported loss. |
| 4 | Masked prompt tokens are excluded from attention. | **False** | Attention runs over the full sequence; only the *labels* are `-100`. Their `attention_mask` stays 1. |
| 5 | `tokenize(a + b)` equals `tokenize(a) + tokenize(b)`. | **False** | BPE merges straddle the join; mask boundaries must be found by incremental tokenisation. |
| 6 | The assistant header should be masked. | **True** | It removes the "double header" failure and costs ~3 supervised tokens. |
| 7 | EOS should be masked out of the target so it is not trained. | **False** | EOS is the *most* important target token — it is what teaches stopping. |
| 8 | Training loss below ~0.5 usually means memorisation. | **True** | CH-13's ten-fact summary: at 0.19 loss the model has lost 14 points of general capability. |
| 9 | Held-out loss is a reliable SFT quality metric. | **False** | It is in-distribution by construction and falls while the model degrades on everything else. |
| 10 | LoRA's learning rate is roughly 1e-4 to 2e-4. | **True** | Full FT is 1e-5 to 2e-5; an encoder (BERT-class) is ~2e-5. |
| 11 | 10 epochs on 1,000 rows is a good way to squeeze out more quality. | **False** | Epoch 10 shows +90% verbosity and −26 points of general capability. |
| 12 | Full fine-tuning needs about 20x QLoRA's VRAM. | **True** | Measured: 7B full FT 91.6 GiB vs QLoRA 4.9 GiB (18.7x; the static ratio at the table's 14 B/param is exactly 20x) — driven by AdamW's 8 bytes/param. |
| 13 | QLoRA stores the base weights at 0.5 bytes per parameter. | **True** | NF4 packs two 4-bit values per byte; the adapters stay in bf16. |
| 14 | Packing is safe with a standard causal mask. | **False** | Without per-sequence `position_ids`/`cu_seqlens` the model attends across examples. |
| 15 | NEFTune adds noise at inference time. | **False** | Train-time only, on the embedding output; serving is byte-identical. |
| 16 | `padding="max_length"` with `labels = input_ids.copy()` trains on pad tokens. | **True** | 455 of 512 positions, i.e. 89% of the loss is "emit `<pad>`". |
| 17 | `tokenizer.pad_token = tokenizer.eos_token` hides a real bug. | **True** | Pad and EOS share an id, so training-on-padding is silent and error-free. |
| 18 | 1,000 curated examples can rival 52,000 uncurated ones. | **True** | LIMA (1,000) and AlpaGasus (9,000 from 52,000, beating the full set on 5/6 benchmarks). |
| 19 | SFT is the right tool for injecting a company's product facts. | **False** | Facts are RAG or continued pretraining; SFT teaches behaviour, not knowledge. |
| 20 | SFT is the right tool for enforcing a JSON output schema. | **Half** | Constrained decoding guarantees the format; SFT handles the *decision* fields. Ship a validator either way. |
| 21 | Preference between two acceptable outputs is an SFT problem. | **False** | That needs pairs — DPO or ORPO. |
| 22 | The chat template must be byte-identical at train and serve. | **True** | It is the #1 silent killer; assert it in CI, not in a wiki page. |
| 23 | An LLM judge's win rate needs swapping to be trustworthy. | **True** | Position bias alone is 10–15 points. |
| 24 | `target_modules=["q_proj","v_proj"]` is the recommended LoRA setting. | **False** | `"all-linear"` — attention-only costs 2–5 points on domain tasks. |
| 25 | Effective batch is measured in tokens per optimizer step, not rows. | **True** | 32k–128k tokens/step is the useful band; "batch size 16" is meaningless across length distributions. |

**Bonus rapid-fire (the interviewer's last three, usually asked only of senior candidates):**

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 26 | Gradient checkpointing makes training faster. | **False** | ~+25–33% step time to cut activation memory 50–70%. |
| 27 | 13-gram decontamination returning 0.0% is a good sign. | **False** | It usually means the checker is broken or the eval set is off-distribution. |
| 28 | Bit-exact reproduction across two different GPUs is achievable. | **False** | Flash-attention's backward is nondeterministic; define reproduction as ±1 point across three seeds. |
| 29 | Replay data must not be masked. | **True** | Masking raw pretraining text leaves nothing to learn — it is the classic "we added replay and saw nothing" bug. |
| 30 | QLoRA adapters can be merged into an NF4 base directly. | **False** | Dequantise to bf16, merge, re-quantise, then re-evaluate the served artefact. |
---

## Coding / Whiteboard Tasks

> Six tasks, 10–20 minutes each. The interviewer is grading the *decisions* rather than the syntax; every task has a specific thing that a weak candidate gets wrong and a strong candidate names out loud.

**Task 1. Write a BPE-safe mask builder that works for multi-turn data, and state the invariant it maintains.**

- **Expected solution sketch:**

```python
IGNORE_INDEX = -100

def build_masked_example(tokenizer, messages, max_len=2048):
    """Return input_ids/attention_mask/labels with assistant spans supervised.

    Invariants maintained:
      I1. labels[i] != -100  =>  messages[j]["role"] == "assistant" for the msg
                                 containing token i.
      I2. attention_mask[i] == 0  =>  labels[i] == -100   (pad is never a target)
      I3. the first non-(-100) label is the first token AFTER the assistant header.
      I4. the last non-(-100) label is the template's end-of-turn / EOS token.
    """
    input_ids, labels = [], []
    for i, msg in enumerate(messages):
        rendered = tokenizer.apply_chat_template(
            messages[: i + 1],
            tokenize=False,
            add_generation_prompt=False,          # never: would append a header
        )
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        new = ids[len(input_ids):] if len(ids) > len(input_ids) else ids
        input_ids += new
        labels += new if msg["role"] == "assistant" else [IGNORE_INDEX] * len(new)

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    if IGNORE_INDEX not in labels:                 # all-masked row -> NaN risk
        raise ValueError("example has no supervised tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
```

- **The four decisions being graded:** (1) the **incremental render-and-slice** — rendering the growing prefix and taking only the new ids, which is the only way to be immune to BPE non-concatenativity; (2) `add_generation_prompt=False` for *every* turn, including the assistant ones, because turning it on inside the loop appends a header into `labels`; (3) the **all-masked guard**, which prevents the `0/0` NaN; (4) padding is handled by the *collator*, so `attention_mask` is all-ones here and pad positions get `-100` from the collator, which must be asserted separately.
- **Grading:** passes for anyone who finds the boundary at all; full marks for naming the invariants and for the incremental slice.
- **Fail condition:** `tokenizer(full_text)` then `n_prompt = len(tokenizer(prompt_only))` then slicing at `n_prompt`. This is the off-by-one that the task exists to detect.

**Task 2. Write an assertion suite that catches the pad-as-target bug and the all-masked-row bug before training starts.**

- **Expected solution sketch:**

```python
def audit_masks(examples, tokenizer, pad_id):
    """Run before every training launch. Cheap; catches three silent failures."""
    problems = {"pad_as_target": 0, "all_masked": 0, "no_eos": 0, "unsupervised_prompt": 0}
    for ex in examples:
        ids, am, lb = ex["input_ids"], ex["attention_mask"], ex["labels"]
        assert len(ids) == len(am) == len(lb)

        # 1. pad must never be a target
        if any(a == 0 and l != -100 for a, l in zip(am, lb)):
            problems["pad_as_target"] += 1

        # 2. every row must supervise something
        sup = [l for l in lb if l != -100]
        if not sup:
            problems["all_masked"] += 1
            continue

        # 3. the last supervised token must be EOS / end-of-turn
        if sup[-1] not in {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}:
            problems["no_eos"] += 1

        # 4. the prompt must be masked (i.e. supervision must not start at index 0)
        if lb[0] != -100:
            problems["unsupervised_prompt"] += 1

    total = len(examples)
    for k, v in problems.items():
        assert v == 0, f"{k}: {v}/{total} rows ({100*v/total:.2f}%)"
    return problems
```

- **The decisions being graded:** (1) the check is that `attention_mask == 0 ⇒ labels == -100`, which is the *invariant* rather than a spot check on a few rows; (2) the all-masked check **continues** rather than inspecting `sup[-1]`, which would otherwise crash on exactly the row you are trying to find; (3) the EOS check is a set of candidate ids, not a single id, because the terminator differs by template; (4) it returns counts *and* asserts, so it can run in CI as a report and as a gate.
- **Grading:** the `attention_mask`-implies-`-100` framing is the full-marks answer; a check on "the loss looks reasonable" is not.
- **Fail condition:** iterating over the *dataset* (rendered strings) instead of the *built examples* — the same `TypeError: list indices must be integers or slices, not str` that CH-13 §12 warns about, which means the candidate skipped `build_masked_example` and is auditing nothing.

**Task 3. Write a VRAM estimator that decides between full FT, LoRA and QLoRA for an arbitrary model size.**

- **Expected solution sketch:**

```python
def vram_gb(n_params_b, method="qlora", r=16, seq_len=1024, micro_bs=1,
            grad_ckpt=True, d_model=None, n_layers=None, overhead=1.15):
    """Static bytes/param + activation term, in the shape of code/common/memory.py.

    Calibrated against the repo's measured table:
      7B  -> full 91.6 | lora 14.7 | qlora 4.9      (GiB, single GPU, grad ckpt on)
      13B -> full 170.0 | lora 27.1 | qlora 9.0
      70B -> full 913.8 | lora 144.5 | qlora 46.8
    """
    N = n_params_b * 1e9
    # static terms, bytes per parameter
    if method == "full":                      # bf16 w + bf16 g + fp32 m,v + fp32 master
        static = 2 + 2 + 8 + 4                # = 16
    elif method == "lora":                    # frozen bf16 base + adapter optimiser state
        frac = (2 * r * (d_model or 4096) * (n_layers or 32)) / N
        static = 2 + frac * (2 + 8 + 4)
    elif method == "qlora":                   # NF4 base + adapter optimiser state + consts
        frac = (2 * r * (d_model or 4096) * (n_layers or 32)) / N
        static = 0.5 + frac * (2 + 8 + 4) + 0.13
    else:
        raise ValueError(method)

    # activation term: grows with layers x seq x batch, reduced ~5x by checkpointing
    act_gb = (micro_bs * seq_len * (d_model or 4096) * (n_layers or 32) * 2) / 1e9
    act_gb *= 0.2 if grad_ckpt else 1.0

    return round(N * static / 1e9 + act_gb + N * 2 / 1e9 * 0.0, 1) * overhead
```

- **The decisions being graded:** (1) the **bytes-per-parameter** decomposition, with `16` for full FT (**with** an fp32 master — `2 + 2 + 8 + 4`; name which configuration you mean, because 12 and 14 are the same run under different trainers) and the **8 bytes of AdamW moments named explicitly** as the dominant term; (2) the `overhead=1.15` multiplier and the statement that real runs add 10–20%; (3) calibration — the interviewer will ask "what does your function say for 7B?" and the answer must land near **91.6 / 14.7 / 4.9**, which is the repo's measured table (GiB, priced at 14 B/param), not `16N = 112 GB` (104.3 GiB); (4) the honest caveat that this is a *decision* tool, not a *guarantee*, and that you verify with `memory.py --table` or a real run before renting.
- **Grading:** anyone who produces a monotone function in `N` passes; the arithmetic that reproduces the measured row is the strong answer.
- **Fail condition:** using `n_params × 2 bytes` for full FT and concluding a 7B trains in 14 GB. The optimiser state is 56 GB = **52.2 GiB** and does not appear in that calculation.
**Task 4. Write a loss-curve diagnostician that takes a training log and names the failure.**

- **Expected solution sketch:**

```python
import math

def diagnose(losses, vocab_size=32000, has_eval=False, eval_losses=None):
    """losses: list of (step, loss). Returns an ordered list of findings."""
    findings = []
    lnV = math.log(vocab_size)                     # 10.37 for 32k, 11.76 for 128,256
    first, last = losses[0][1], losses[-1][1]

    # A. flat at ln(V) -> the graph is disconnected, not underfitting
    if abs(first - lnV) < 0.05 and abs(last - lnV) < 0.05:
        findings.append("FLAT_AT_LNV: every label is -100, or no params require grad, "
                        "or lr == 0. Do not raise the lr.")

    # B. starts well BELOW ln(V) -> the prompt is probably unmasked
    if first < lnV - 1.0:
        findings.append("STARTED_LOW: prompt tokens are in the loss (or the eval set "
                        "is contaminated). Check labels[:20] for non-(-100).")

    # C. NaN / inf anywhere
    if any(not math.isfinite(l) for _, l in losses):
        findings.append("NON_FINITE: all-masked row (0/0), fp16 without scaling, or "
                        "lr too high at the end of warmup.")

    # D. memorisation: very low train loss (+ rising eval)
    if last < 0.5:
        findings.append("MEMORISATION: train loss < 0.5. Check epochs (>3?), n_rows, "
                        "and the four overfit signatures: verbosity, format rigidity, "
                        "refusal, forgetting.")

    # E. sawtooth / high variance -> effective batch too small
    window = [l for _, l in losses[-200:]]
    if window and (max(window) - min(window)) > 0.5 and last > 1.0:
        findings.append("NOISY: tokens/step is below ~32k. Raise grad_accum or "
                        "micro_bs; then raise the lr by ~sqrt(batch ratio).")

    # F. divergence: a spike then a plateau
    peak = max(l for _, l in losses[:max(1, len(losses)//5)])
    if peak > 2 * lnV:
        findings.append("DIVERGED_EARLY: lr too high for this method (full FT at "
                        "LoRA-scale lr?), or fp16 overflow.")

    # G. eval diverging from train
    if has_eval and eval_losses and len(eval_losses) >= 3:
        if eval_losses[-1] > min(eval_losses) * 1.1:
            findings.append("EVAL_RISING: stop earlier. But note eval loss is the "
                            "weakest signal - gate on the composite metric instead.")

    for i, f in enumerate(findings, 1):
        print(f"{i}. {f}")
    return findings
```

- **The decisions being graded:** (1) `ln(V)` as an **exact anchor** (`10.37` / `11.76`), because it converts a vague "the loss is flat" into a diagnosis; (2) the discrimination between *flat at* `ln(V)` and *starting below* it, which are opposite bugs; (3) the ordering — data-level causes (A, C) are checked before schedule-level causes (E, F); (4) the closing caveat that `eval_loss` is the weakest signal, so the memorisation branch points at the four signatures rather than at the loss number.
- **Grading:** a candidate who only implements the "is it going down" check gets nothing above a pass; the `ln(V)` and below-`ln(V)` branches are the whole task.
- **Fail condition:** recommending "lower the LR" or "train longer" as the output for the flat-at-`ln(V)` case. At `ln(V)` the model is disconnected; neither helps.

**Task 5. Write a format-compliance and prompt-mutation harness for a JSON-output model.**

- **Expected solution sketch:**

```python
import json, random

MUTATIONS = [
    lambda s: s.rstrip() + " ",
    lambda s: " " + s,
    lambda s: s.replace("Summarize", "Summarise"),
    lambda s: s + "\nPlease be concise.",
    lambda s: s.replace(".", ". "),
    lambda s: s[0].lower() + s[1:],
]

REQUIRED = ["claim_type", "severity", "urgency", "missing_documents"]
ENUMS = {"severity": {"low", "medium", "high"},
         "urgency":  {"routine", "expedite", "immediate"}}

def score_one(text, schema_required=REQUIRED, enums=ENUMS):
    """Return (valid, non_empty, enums_ok). 'valid JSON' alone is not a metric."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False, False, False
    if not isinstance(obj, dict):
        return False, False, False
    non_empty = all(k in obj and obj[k] not in (None, "", [], {}) for k in schema_required)
    enums_ok = all(obj.get(k) in v for k, v in enums.items() if k in obj)
    return True, non_empty, enums_ok

def harness(model, prompts, n_judge=None, seed=42):
    random.seed(seed)
    report = {"canonical": [], "mutated": [], "per_mutation": {}}
    for p in prompts:
        report["canonical"].append(score_one(model.generate(p)))
        for i, mut in enumerate(MUTATIONS):
            r = score_one(model.generate(mut(p)))
            report["mutated"].append(r)
            report["per_mutation"].setdefault(i, []).append(r)

    def rate(rows, idx):
        return round(100 * sum(r[idx] for r in rows) / max(1, len(rows)), 2)

    can = rate(report["canonical"], 0)
    mut = rate(report["mutated"], 0)
    return {
        "canonical_valid_pct": can,
        "mutated_valid_pct": mut,
        "mutation_gap_pts": round(can - mut, 2),          # want < 5
        "canonical_non_empty_pct": rate(report["canonical"], 1),
        "canonical_enums_ok_pct": rate(report["canonical"], 2),
        "worst_mutations": sorted(
            ((i, rate(rows, 0)) for i, rows in report["per_mutation"].items()),
            key=lambda kv: kv[1])[:2],
    }
```

- **The decisions being graded:** (1) **three** metrics, not one — `valid` alone is satisfied by `{}`, so non-emptiness and enum membership are separate; (2) the **gap** is the headline number (`< 5` points is the target, `> 25` means the model learned the template); (3) the mutations are *semantics-preserving* rewrites of the prompt, not new tasks, so a change in output quality is attributable to the prompt surface; (4) the per-mutation breakdown, so a fixed worst mutation is a trainable signal rather than a mystery; (5) temperature 0 and a frozen seed, or the numbers move between runs.
- **Grading:** the non-emptiness check and the gap metric are the two things that separate strong from adequate.
- **Fail condition:** reporting a single canonical compliance number. It tells you nothing about the failure users will actually see.

**Task 6. Write a cost-and-steps estimator that answers "how long and how much money will this run take?" before it starts.**

- **Expected solution sketch:**

```python
def plan(n_rows, p95_tokens, epochs=2, eff_batch_tokens=32768, micro_bs=2,
         grad_accum=8, n_gpus=1, method="qlora", n_params_b=8,
         gpu_hour_usd=1.50, throughput_tps=None, overhead=1.4):
    """Everything the interviewer wants in one call. Throughput from CS-13 s11."""
    tokens = n_rows * p95_tokens * epochs
    steps = max(1, int(tokens / (eff_batch_tokens * n_gpus)))

    # tokens/s/GPU by method+scale, from the module's measured throughputs
    table = {("lora", 8): 4000, ("qlora", 8): 1800,
             ("qlora", 70): 380, ("full", 8): 1200, ("full", 70): 47}
    tps = throughput_tps or table.get((method, n_params_b), 1000)

    gpu_seconds = tokens / (tps * n_gpus) * overhead
    gpu_hours = gpu_seconds / 3600

    return {
        "tokens_total": f"{tokens/1e6:.2f}M",
        "optimizer_steps": steps,
        "micro_batches_per_step": grad_accum,
        "wall_clock_hours": round(gpu_hours / 1, 2),      # if gpu_hours is wall clock on n_gpus
        "gpu_hours": round(gpu_hours, 2),
        "cost_usd": round(gpu_hours * gpu_hour_usd, 2),
        "warmup_steps_at_0.03": max(10, int(0.03 * steps)),
        "schedule_is_meaningful": steps >= 200,          # else cosine/warmup are inert
    }

# Worked, against the module's own numbers:
print(plan(8000, 512, epochs=2, method="qlora", n_params_b=8))
# -> 8.19M tokens, ~34 min, ~$0.85 on one A100-40  (CS-13 s11.3)
```

- **The decisions being graded:** (1) **tokens per optimizer step** (`micro_bs × seq_len × grad_accum × n_gpus`) is the batch unit, not rows; (2) the **overhead multiplier** (1.3–2x) and an explicit statement that it exists; (3) the **`steps` count feeds the schedule** — `schedule_is_meaningful` encodes the notebook-run trap, where 3 total steps makes cosine and warmup inert and `warmup_steps = max(10, ...)` is the correct guard; (4) the closing comparison: the dataset cost (**~$33,000** for 2,000 expert rows, or **$18,000** in the CS-13 insurance case) against the GPU bill (**$0.63–$0.85**), which is the number that decides whether a project is fundable.
- **Grading:** producing a dollar figure at all passes; the tokens/step framing and the schedule guard are the strong answers; naming the dataset cost as the dominant term is the senior answer.
- **Fail condition:** computing `rows × epochs / batch_size = steps` in **rows**, then reporting a step count that is wrong by the ratio of `seq_len × grad_accum`, and a wall clock off by the same factor.

---

## Cheat Sheet of Numbers To Memorize

> If an interviewer asks for a number and you do not have it, you do not have the answer. These are the module's numbers; every one of them appears somewhere in CS-13 or CH-13.

| Quantity | Value | Why it matters |
|---|---|---|
| `IGNORE_INDEX` | **-100** | `CrossEntropyLoss(ignore_index=...)`'s default; excluded from numerator *and* denominator. |
| Loss for a uniform distribution | **ln(V)** — 10.37 (32k), 11.76 (128,256) | A flat loss at this value means the graph is disconnected, not underfitting. |
| Training loss indicating memorisation | **< 0.5** | CH-13's ten-fact summary; at 0.19 loss the model has lost ~14 points of general capability. |
| Supervised token fraction | **10–40%** (22% for 400 prompt + 112 answer) | Below 10% is a red flag — padding, truncation, or long prompts. |
| LoRA / QLoRA learning rate | **1e-4 – 2e-4** (default 2e-4) | 10–20x full FT's. The #1 config error is using full-FT rates on LoRA. |
| Full fine-tuning learning rate | **1e-5 – 2e-5** (instruct models 5e-6 – 1e-5) | LoRA-scale rates here destroy pretrained features in <100 steps. |
| Encoder (BERT-class) learning rate | **2e-5** | Orders of magnitude smaller than LoRA; a different regime entirely. |
| Epochs | **1–3** (2 default) | Epoch 5 → +61% verbosity, −14 general; epoch 10 → +90%, −26. |
| Warmup | **3–10%** (`warmup_ratio=0.03`); `max(10, 0.03×steps)` | Meaningless below ~200 optimizer steps. |
| Tokens per optimizer step | **32k–128k** | Below ~32k the gradient is prohibitively noisy; scale the LR as `√batch`. |
| LoRA rank / alpha / dropout | **r=16, alpha=2r=32, dropout=0.05**, `target_modules="all-linear"` | `alpha/r` acts like an LR; keep it fixed while sweeping `r`. |
| NEFTune | **alpha=5** (10–15 long sequences), train-time only | +29.8% AlpacaEval (7B), +8.7% (13B) on LLaMA-2. |
| Adapter parameters (7B, all-linear, r=16) | **~20M** (~0.3%), ~40 MB in bf16 | Why 20 adapters fit where one merged model does not. |
| 7B VRAM: full / LoRA r16 / QLoRA r16 | **91.6 / 14.7 / 4.9 GiB** (inference bf16 16.0, 4-bit 4.8) | The repo's measured table; the ~19x ratio is AdamW's 8 bytes/param. |
| 8B VRAM: full / LoRA / QLoRA | **104.7 / 16.7 / 5.6 GiB** | Llama-3.1-8B row of the same table. |
| 13B VRAM: full / LoRA / QLoRA | **170.0 / 27.1 / 9.0 GiB** | |
| 70B VRAM: full / LoRA / QLoRA | **913.8 / 144.5 / 46.8 GiB** | Full FT needs ~16×A100-80 with ZeRO-3; QLoRA fits on one. |
| Bytes per parameter | **2 bf16 / 4 fp32 / 8 AdamW m,v / 0.5 NF4 / 1 int8 optimiser** | Full FT = **16** B/param with an fp32 master copy (`2+2+8+4`), **12** without one (`2+2+8`), **14** on the repo's floor (`4+2+8`); QLoRA ≈ 0.7 B/param amortised. |
| Default overhead multiplier | **×1.15** (measured table), **×1.3–2.0** (real runs) | Never rent exactly your computed requirement. |
| LIMA | **1,000 curated** → 43% win/tie vs GPT-4, 58% vs Bard, 65% vs Alpaca-65B | Quality beats quantity; the module's central empirical claim. |
| Alpaca / AlpaGasus / Deita | 52k raw; **9k beat 52k on 5/6 benchmarks**; 6k beat 100k+ | Curated subsets beat the superset. Alpaca is CC-BY-NC-4.0. |
| Domain sweet spot | **2,000–10,000 rows** (smoke 50–200, product 5k–50k, capability 50k–500k) | Below 200 you cannot measure anything; above ~10k returns flatten fast. |
| Expert data cost | **~$33,000** for 2,000 rows at 20 min/row ($18,000 in the CS-13 case) | 40,000x the GPU bill — the dataset is the project cost. |
| GPU cost, 8k rows × 512 tok × 2 epochs | 8.19M tokens ≈ **34 min ≈ $0.85** on one A100-40 (QLoRA, 8B) | Training is cheap; measurement and data are not. |
| Decontamination | **13-gram** overlap with the eval set | 0.0% is suspicious; 15%+ means every metric is fiction. |
| Near-duplicate threshold | **MinHash Jaccard ≥ 0.8** on 5-grams; embedding cosine ≥ 0.9 | The two dedup thresholds; 0.0% exact-match dups is also suspicious. |
| Copy-task threshold | instruction–response **token Jaccard < 0.3** | Above that, the "answer" is a paraphrase of the question. |
| Length diversity | **10–20x** p5-to-p95 response length; 10–20% of rows < 30 tokens | Short rows are the cheapest anti-verbosity intervention. |
| Abstention / refusal-boundary rows | **3–8% "I don't know"**, 2–5% refusal | Without them the model cannot learn to abstain. |
| Task-diversity caps | no task > **25%**; ≥**40 distinct leading verbs** per 1,000 rows; ≥20% multi-turn | The three diversity axes, with thresholds. |
| Judge biases | position **10–15 pts**, verbosity **60–70%**, self-preference **~10 pts** | Swap orders, match lengths, judge with a different family. |
| Eval-set noise floor | n=200 at p=0.9 → **±3 points** (±2.1 se) | Any gate tighter than this blocks good deploys at random. |
| Scorecard gates | format ≥96% good / 99%+ excellent; refusal ≤2%; win rate ≥65%; MMLU ≥ −1; mutation gap <5 | The composite ship decision, stated before the run. |
| Forgetting at 5 epochs | MMLU **−6 to −14** (typical run −6, GSM8K −12) | Mitigation order: replay > fewer epochs > LoRA > lower LR. |
| Replay proportion | **5–20%** general instruction data, or 5–10% raw text (**unmasked**) | Masking replay text is the classic "we added replay and saw nothing" bug. |
| Truncation tolerance | **<0.5%** of rows losing their response | Measure the prompt token count, not the whole row. |
| Tokens per word / chars per token | **1.33** tokens/word, **~4** chars/token (English) | The two conversions for sizing a dataset from prose. |
| The notebook's run | 5 rows, TinyLlama-1.1B, `1×512×8`, **3 optimizer steps**, ~15 s on a T4, <$0.001 | Why its hyperparameters are meaningless: 5% warmup = 0 steps. |
| CS-13 published outcomes | JSON 88 → **99.6%**, urgency 71 → **89%**, MMLU **−1.2**; house voice 34 → **91%**; tool syntax 71 → **98.2%** | The realistic effect sizes to quote in a projection. |
---

## Answers To The Self-Check Questions From CS-13

> CS-13 §19 poses ten self-check questions. These are the full interview-grade answers: longer than the case study's `<details>` blocks, with the numbers and the traps an interviewer will follow up on. If you can answer all ten out loud in under four minutes each, you are ready for this module.

**1. What exactly does SFT change about a model, and what does it not change?**

- **What it changes:** the **conditional distribution of the response given a prompt** — which format, register, length, structure, and refusal policy the model adopts. Mechanically it moves the weights (or the adapter) so that the probability mass on your target continuations increases, which is a *selection* over behaviours the base model already has.
- **What it does not change:** the tokenizer; the model's knowledge; the ceiling on reasoning or factual recall; and the pretrained representations in any way you would call "new capability." The superficial-alignment evidence is the argument: 1,000 curated examples give a usable assistant, and a 1.3B InstructGPT was preferred over a 175B base GPT-3 — that cannot be new capability, it must be elicitation.
- **The operational test:** if your requirement is a *behaviour* (schema, tone, refusal, stopping, tool syntax), SFT is the right tool and it is cheap. If your requirement is *knowledge* ("it does not know our policies"), SFT is the wrong tool — use RAG (CS-04) for facts that change and continued pretraining (CS-12) for domain vocabulary the base lacks.
- **Why asked:** It is the framing question for the whole module, and the answer determines every downstream decision.
- **Trap:** "It teaches the model about our domain." It teaches the model to *talk about* your domain in the shape you demonstrated. Knowledge injection through SFT is 10⁴–10⁶x less token-efficient than in-context, brittle under paraphrase, and partially eroded by later alignment stages.

**2. Why is prompt masking the default, and when is training on the prompt legitimate?**

- **Why masking is the default:** three distinct mechanisms, and you should give all three. (1) **Gradient dilution** — the prompt is highly predictable (`log p(prompt_t | prompt_<t)` is often 0.5–2.0 nats against 3–8 for a structured answer), so an unmasked loss spends the majority of its budget on a task you already have. CS-13's arithmetic for a 512-token row with a 400-token prompt and a 112-token answer: ~56% of the loss value comes from the prompt. (2) **Loss-scale bias** — because the loss is a mean over tokens, rows with longer prompts contribute proportionally more, so your model's behaviour is determined by whichever rows happen to be longest. (3) **The task you accidentally train** — the model learns to *generate the prompt*, which produces the "completes the question instead of answering it" failure that CS-13 §15.5 records as a real post-mortem.
- **The four legitimate exceptions:** (a) the prompt is *generated* content you want the model to be able to produce (synthesising the user query in a synthetic conversation); (b) you are training a base model that must also continue as a plain LM — this is the continued-pretraining case, where there is no prompt/answer split at all; (c) replay data — if you mix 5–10% raw pretraining text in to prevent forgetting, masking it leaves nothing to learn, which is the classic "we added replay and saw no effect" bug; (d) matching a specific published recipe, where you should measure the effect rather than assume it.
- **Why asked:** It is the module's central implementation decision and the exceptions are what separate a rule from an understanding.
- **Trap:** Extending the exceptions to "my prompt is short so it doesn't matter." A 512-token row with a 400-token prompt and a 112-token answer has a 78% prompt — and the notebook's own 5-row demo masks nothing and reports a loss that looks *better* than the masked version.

**3. What does `-100` mean in a labels tensor, and what happens to those positions in the loss?**

- **What it means:** `-100` is `ignore_index` in `torch.nn.CrossEntropyLoss`, and it is a value no valid token id can take (ids are non-negative). It matches the convention that spread through the ecosystem from `tatsu-lab/stanford_alpaca`, and every masking helper — TRL, Unsloth, LLaMA-Factory, Axolotl — produces it.
- **What happens to those positions:** after the causal shift, a target position equal to `-100` contributes **nothing to the numerator and nothing to the denominator**. `reduction="mean"` divides by the count of non-ignored positions, so masking the prompt both removes the prompt's loss *and rescales the reported number*. That is why two runs with different masking policies have incomparable losses, and why "our loss went up when we added masking" is the expected observation rather than a regression.
- **What does *not* happen:** the forward pass is untouched. `-100` lives in `labels`, never in `input_ids`, and the corresponding `attention_mask` entries stay **1** — the model still *reads* the prompt, it is simply not *graded* on predicting it. Only padding gets `attention_mask = 0`, and the invariant is `attention_mask == 0 ⇒ labels == -100`.
- **Why asked:** It is the one mechanical fact that everything else in the module depends on, and the denominator point is the part that is usually missing.
- **Trap:** Setting `attention_mask = 0` on prompt tokens to "exclude them." That removes the prompt from attention entirely, so the model cannot condition on its own instructions — it becomes a model that answers questions it never saw.

**4. Your loss is dropping nicely but the model answers in a rigid, over-long style. What happened, and what do you change?**

- **Diagnosis:** two of the four overfitting signatures appearing together — **verbosity** and **format rigidity** — which is the normal pattern past ~2–3 epochs. The model has learned your training sample's idiosyncratic length and structure rather than the task. Confirm with the numbers: compare mean response length across checkpoints (a >15% drift is the flag; CS-13's table shows +8% at 2 epochs, +18% at 3, +61% at 5), and run a prompt-mutation sweep to measure the canonical-vs-mutated compliance gap (a >25-point gap is format rigidity).
- **The change, in order of effectiveness:** (1) **add 5–20% replay/general instruction data** — it breaks the mode directly by putting diverse lengths and structures in the gradient; (2) **reduce epochs** — the same dataset at 1–2 epochs usually removes both signatures without any other change; (3) **switch to LoRA** (or lower the rank) if you were full-FT, so the base representation is frozen and only a low-dimensional residual moves; (4) **lower the LR** — weakest of the four, because it slows the drift rather than changing its direction; (5) **add NEFTune** (`neftune_noise_alpha=5`), which attacks surface memorisation directly at the embedding layer for a one-line change; (6) **early-stop on length drift** rather than on loss, because loss will still be falling.
- **Why asked:** It is the most common real quality complaint after a first successful run, and the ordering is the operative part.
- **Trap:** Treating it as a decoding problem and adding `repetition_penalty` or `frequency_penalty` at serving. Those change the sampling distribution without changing the model's learned prior, and they degrade legitimate repetition in JSON, lists and code — which is usually the output format that motivated the fine-tune.

**5. How do you evaluate an SFT model, and which metric is most misleading?**

- **The four layers:** (1) **mechanical** — held-out loss (with the same masking policy as training), format compliance, refusal rate on a benign suite, mean-length drift, and a contamination count; (2) **task** — exact match, regex assertions, JSON-schema validity *and non-emptiness and enum membership*, per-field accuracy, all on a frozen held-out set with confidence intervals; (3) **comparative** — a swap-corrected, blinded LLM-judge win rate against the *currently-served* model and against the start checkpoint, judged by a different model family and calibrated against ≥100 human labels; (4) **regression** — an MMLU-500 or IFEval-200 subset to measure forgetting (expect −1 to −3 points, investigate beyond −3).
- **The most misleading is held-out loss**, and there are four reasons: it is in-distribution by construction; it is *improved* by exactly the memorisation you do not want; it has no notion of correctness, so valid-but-wrong JSON scores well; and it is blind to all four overfit signatures. CS-13's own table is the proof — loss 0.63 → 0.19 while verbosity rises 8% → 61% and general capability falls 4 → 14 points.
- **The second most misleading is a single-ordering judge score**, where position bias alone is worth 10–15 points and verbosity bias makes longer answers win 60–70% of equal-quality pairs.
- **Why asked:** It is the module's most common interview failure — candidates evaluate the task metric and stop.
- **Trap:** Using `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"` on a small in-distribution eval set. That mechanism selects the **most memorised** checkpoint, and it will be the worst model in the run.
**6. Walk through the mask tensor for `[BOS] + prompt(24) + response(32) + PAD`, with `max_length=512`, under `padding="max_length"` + `labels=input_ids.copy()`.**

- **The layout:** indices `0` (BOS) through `24` are the prompt (25 positions including BOS), `25–56` are the response (32 positions), and `57–511` are `<pad>` (455 positions). Total 512.
- **What the notebook's scheme produces:** `labels == input_ids` everywhere, so the loss is computed over **all 512 positions** — 25 prompt positions, 32 response positions, and 455 pad positions. The loss is dominated by an easy-to-predict constant (`<pad>` after `<pad>`), which is why it *looks* good.
- **How much of the loss is real:** 32 of 512 positions, i.e. **6.3%**. Across 5 rows × 3 epochs that is 7,680 supervised label positions of which **6,825 are `<pad>`** (5 × 455 × 3), leaving 480 real answer tokens — about 6%. Note that CS-13 §18 states this as "61,000 label positions on `<pad>`," which is roughly 8–9x too high; the derivable figure is 6,825. The *conclusion* is unaffected and the direction is right, but quote 6,825.
- **The correct scheme:** set indices `0–24` (prompt, **including the assistant header if there is one**) and `57–511` (pad) to `-100`, leaving the 32 response positions supervised. After the causal shift (`shift_logits = logits[..., :-1, :]`, `shift_labels = labels[..., 1:]`) the last response position has no successor inside the sequence, so **31 shifted targets** supervise the answer. That is the number to quote.
- **Why asked:** It is the whole masking design compressed into one arithmetic exercise, and the interviewer is checking whether the candidate counts *shifted targets* rather than positions.
- **Trap:** Forgetting the shift. The number of positions you set to non-`-100` is 32; the number of loss terms that actually exist is 31. Small, but the same off-by-one is what makes the BPE boundary bug invisible.
- **The real fix, beyond masking:** use packing instead of padding and all 455 pad positions disappear, turning 6% of the compute into useful work — a ~2–5x throughput win and a cleaner loss.

**7. When does SFT inject knowledge, and why is that not a reason to use it for facts?**

- **When it works at all:** four conditions must hold together — the fact is repeated across **500+ paraphrases**, the base model has the relevant latent capability (so the fine-tune is eliciting rather than installing), the knowledge is not contradicted by later training, and no preference stage (DPO/RLHF) follows, because those measurably erode what SFT injected. In that narrow configuration SFT does move factual accuracy.
- **Why it is still the wrong tool:** (1) **token efficiency** — in-context injection is 10⁴–10⁶x more token-efficient per fact than weight injection, so the same dollar buys a fraction of the knowledge; (2) **brittleness** — a fact in weights generalises poorly under paraphrase and fails on a question shape it did not see; (3) **staleness** — a policy that changes on Tuesday cannot be updated by a model trained on Monday, which in a regulated domain is a compliance event rather than a quality bug; (4) **no citations** — a RAG system can point at the paragraph, and for policy answers the citation is often the requirement.
- **The correct division of labour:** continued pretraining (CS-12) for *domain vocabulary and language* the base model has never seen; retrieval (CS-04) for *facts that change*; SFT for *behaviour* — output shape, tone, refusal policy, how to use a retrieved chunk, and how to cite it. Note the third one is genuinely an SFT problem even in a RAG system: "always cite the chunk id, and say 'not stated in the provided context' when the answer is absent" is a behaviour, and prompt engineering gets you to ~80% while SFT gets you to ~99%.
- **Why asked:** It is the decision that determines whether the project is three days or three months, and the 10⁴–10⁶x figure is the argument that settles it.
- **Trap:** "We'll fine-tune on the documents." That is continued pretraining mislabelled — and if the rows are `(question, answer)` pairs generated from the documents, it is SFT, which teaches the *shape* of an answer about the documents without reliable retrieval of the content.

**8. What is the difference between a chat template and a data format, and why does conflating them break things?**

- **The distinction:** a **data format** is how your dataset *stores* a conversation — Alpaca's `instruction`/`input`/`output` columns, ShareGPT's `conversations[]` with `from`/`value`, OpenAI's `messages[]` with `role`/`content`, or a DPO triple. A **chat template** is how a conversation is *serialised into the single token string* the model consumes, including the control tokens and role markers (`<|im_start|>`, `<|start_header_id|>`, `[INST]`, `<start_of_turn>`, `</s>`). A format is a container; a template is a **serialisation contract with the weights**.
- **Why conflating them breaks things:** they are independent axes, so four combinations exist and only one is correct. Converting Alpaca to `messages` correctly and then rendering with the wrong template is the classic failure — the pipeline is clean, the loss falls, the samples look plausible in the notebook, and the model is wrong behind the API because it is receiving token sequences it was never trained on. This is CS-13 §4.3.4, and it is the #1 silent killer in the module: no error, no warning, and a loss curve that looks *better* than the correct run because the model is memorising an easier, wrong objective.
- **The engineering response:** store canonically (`messages`), convert on export with one tested script, and keep the template as a **single versioned artefact owned by a shared module** with its sha256 recorded on every training run and asserted at serving boot. The CI test is byte equality of the rendered string between the training path and the serving path for a fixture conversation — not semantic similarity, byte equality.
- **Why asked:** It is the highest-severity silent failure and the fix is organisational, so the answer has to include the enforcement mechanism, not just the distinction.
- **Trap:** "The tokenizer's default template is the right one." It frequently is not — a tokenizer's default may be a generic ChatML that the checkpoint was never trained with. Render it and read it before every run; that single habit prevents the most expensive class of bug in the module.

**9. Why is `padding="max_length"` with `max_length=512` on a 5-row dataset with ~96-token examples a problem, in numbers?**

- **The arithmetic:** every row is padded to 512 tokens. With `labels = input_ids.copy()`, all 512 positions are targets. Over 5 rows × 3 epochs that is **7,680 supervised label positions**, of which **6,825 are `<pad>`** (5 × 455 × 3) and 480 are real answer tokens (5 × 32 × 3) — **6%** of the gradient budget. The prompt's 375 positions (5 × 25 × 3) are also trained, against the intent of the masking design.
- **Why the loss looks deceptively good:** `<pad>` after `<pad>` is the easiest possible prediction, so the reported loss is dominated by a constant that approaches zero. The model appears to converge beautifully while learning, in descending order of gradient share: padding, the prompt template, and the answer.
- **The three fixes, in order of value:** (1) **packing** (`packing=True` with `padding_free` and per-sequence `position_ids`) — eliminates all 6,825 pad positions and converts them into real examples, a 2–5x throughput win; (2) **a padding collator** (`padding=False` plus `DataCollatorForSeq2Seq`) — pads only to the longest row in each batch, which on this data is ~96 tokens rather than 512, an ~80% compute reduction; (3) **mask the padding** — the minimum viable fix, `labels[attention_mask == 0] = -100`, which does not recover the compute but does stop the model learning to emit `<pad>`.
- **The meta-point:** the notebook's config is *mechanically correct and pedagogically hazardous*. It runs, it produces a model, and the number it prints means something different from what it appears to mean. Quote it as a demonstration of an anti-pattern, never as a recipe.
- **Why asked:** It is the most concrete numeric exercise in the module and it tests whether the candidate can derive a figure rather than repeat one.
- **Trap:** Fixing the loss without fixing the compute. Masking the padding leaves you burning 89% of every forward pass on pad tokens; on a 5-row demo that is free, and on 50,000 rows it is the difference between one GPU-day and two GPU-weeks.

**10. You have 3,000 domain rows and a deadline. Which method, which hyperparameters, and which evals?**

- **Method:** **LoRA**, not QLoRA, unless you are memory-bound — 3,000 rows on a 7–8B model fits comfortably in one A100-40 or a 48 GB card at LoRA's 14.7–16.7 GiB, and LoRA avoids the 4-bit quantisation error that costs 0–2 points. `r=16`, `target_modules="all-linear"` (q,k,v,o,gate,up,down), `lora_alpha` = 32 if you keep the `alpha/r = 2` convention CH-13 §4.2 recommends (`alpha = 2r`); CS-13 §19's own answer writes `α=16` with `r=16`, which is `alpha/r = 1` — either works, but pick one and hold it while you sweep `r`, because `alpha/r` behaves like a learning rate. `lora_dropout=0.05`, `bias="none"`. Use an **instruct** base if one exists for your model, since it already has the format prior and 3,000 rows is not enough to install one from scratch.
- **Hyperparameters:** LR **1e-4** (the conservative end of 1e-4–2e-4 for a small dataset — small datasets overfit faster, and 2e-4 on 3,000 rows with 2 epochs is where verbosity starts); **2 epochs**, cosine, `warmup_ratio=0.05`; `tokens_per_step ≈ 64k` — `bs 4 × ga 4 × seq 4096` on a big GPU, or `bs 2 × ga 8 × seq 2048` on a small one (both are in the 32k–128k band); `max_seq_length` at the **p95 of your rendered lengths** with `response_lost < 0.5%`; `bf16=True` (never fp16); `gradient_checkpointing=True`; `optim="paged_adamw_8bit"`; `packing=True` with `padding_free` and per-sequence `position_ids`; `neftune_noise_alpha=5`; **prompt masked** (`assistant_only_loss=True` or `train_on_responses_only`); `seed=42`, `data_seed=42`, `logging_steps=5`.
- **Evals, all four layers:** format compliance **≥95%** on the frozen held-out set (96% good, 99%+ excellent); refusal rate **≤2%** on a 200-prompt benign suite; mean-length drift **<15%**; a swap-corrected, blinded win rate **≥60%** against the base and (if one exists) against the currently-served model; a prompt-mutation gap **<5 points**; and an MMLU-500 subset drop **≤3 points**.
- **Budget:** 3,000 × ~700 tokens × 2 epochs ≈ **4.2M tokens** ≈ **20 minutes on one A100-40** ≈ **$0.50** at $1.50/GPU-hr. Add ~$1 for the judge runs and $300 for human review of the held-out set — against which the real cost of the project is the 3,000 rows themselves, which at 20 minutes of expert time each is roughly **$33,000** if you had to author them from scratch. The GPU is not the cost.
- **The four things to do before launching:** (1) render one row and read the string; (2) run `token_stats` and check `supervised_token_frac` is 10–40% and `without_supervision` is 0; (3) run `memory.py --table` to confirm the VRAM; (4) compute the step count and set `warmup_steps = max(10, 0.05 × steps)` so the schedule is not inert.
- **Why asked:** It is the module's closing question, and the interview answer is graded on whether the candidate states *why* each number is what it is — the conservative LR, the p95 length, the four eval layers — rather than reciting a config.
- **Trap:** Copying a 52k-row tutorial recipe (LR 2e-4, 3 epochs, `padding="max_length"`) onto 3,000 rows. It runs, it finishes, and it produces a verbose, rigidly formatted model with a memorised training set — which is exactly the failure the module's epoch table predicts.

---

## Cross-References

| Module | Relationship to IQ-13 |
|---|---|
| **CS-13** Instruction Fine-Tuning | The source case study. Every number in this bank is grounded in it; its §19 self-check questions are answered in full above |
| **CH-13** Instruction Fine-Tuning Cheat Sheet | The compressed reference: §4.1 master hyperparameter table, §5.2 the mask snippet, §7 the VRAM table, §8 the 17-row symptom→fix lookup |
| **CS-12** Domain-Adaptive Continued Pretraining | The prerequisite in practice: non-instructional fine-tuning before instruction tuning; where knowledge actually goes |
| **CS-14** The Alignment Map (RLHF, PPO, DPO, ORPO) | Where SFT sits as stage 2 of 3, and what the later stages can and cannot fix |
| **CS-15** LLaMA-Factory | No-code SFT; `train_on_prompt: false`, `formatting: alpaca\|sharegpt` — the config-level form of Q27 and Q53 |
| **CS-16** Unsloth | The fast path; `train_on_responses_only` is the masking implementation compared in Q66 |
| **CS-17** Axolotl | YAML SFT; `train_on_inputs: false`, `roles_to_train`, `neat_packing` — the packing discussion made concrete |
| **CS-23** LoRA & QLoRA — the PEFT Deep Dive | The adapter mechanics behind Q41, Q42, Q67 and Q75; rank/alpha/target-module theory in full |
| **CS-25** DPO | The stage after SFT; the `prompt`/`chosen`/`rejected` format referenced in Q21 and Q49 |
| **CS-27** ORPO | SFT and preference in one loss — the alternative to "SFT then DPO" |
| **CS-04** Fine-Tuning vs RAG vs Agents | Read before you start SFT; the decision table behind Q12, Q47 and Q87 |
| **CS-09** Knowledge Distillation | Where synthetic instruction data comes from; the teacher→student framing of Q37 and Q73 |
| **IQ-01** Foundations | The lifecycle framing these questions assume: pretraining, SFT, alignment |
| **IQ-15** | The sibling bank that picks up where SFT ends — preference optimisation and reward-based training |

---

*End of IQ-13. Companion artifacts: `CS-13-Instruction-Fine-Tuning.md` (case study), `CH-13-Instruction-Fine-Tuning.md` (cheat sheet). Ground truth: `code/common/memory.py --table`, `code/common/data_utils.py`, `code/data/sample_sft.jsonl`.*
