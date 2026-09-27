# IQ-13 — Interview Questions: Instruction Fine-Tuning (SFT)

| Field | Value |
|---|---|
| **Module** | Instruction Fine-Tuning / Supervised Fine-Tuning (SFT) — stage 2 of the three-stage lifecycle |
| **Pairs with** | CS-13 (case study), CH-13 (cheat sheet) |
| **Total questions** | 113 (28 L1 + 34 L2 + 25 L3 + 11 L4 + 15 L5) + 25 rapid-fire + 6 coding tasks |
| **Levels covered** | Screen (L1) / Intermediate (L2) / Advanced (L3) / System Design (L4) / Debug (L5) |
| **Source material** | Video 15 `LLM_Fine-Tuning_15_Instruction_Fine-Tuning_Explained_Domain-Specific_FineTuning.txt`; notebook `Instruction_finetuning_on_domain_specific_dataset.ipynb`; `pharma_instruction_data.jsonl` (5 rows); `code/common/memory.py --table`; `code/common/data_utils.py`; `code/data/sample_sft.jsonl` (60 rows) |

**Ground truth used throughout:** `IGNORE_INDEX = -100`, matching `torch.nn.CrossEntropyLoss(ignore_index=-100)`. VRAM figures are the repo's own measured table (`code/common/memory.py --table`): 7B full FT **91.6 GB**, LoRA r16 **14.7 GB**, QLoRA r16 **4.9 GB**. `data_utils.token_stats()` returns `examples, tokens_total, len_p50, len_p95, len_max, with_supervision, without_supervision, supervised_token_frac`.

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

- **Answer:** **Full FT** updates every weight: highest quality ceiling, ~16 bytes/parameter of optimiser+weight state (7B = 91.6 GB measured in this repo), requires ≥50k examples and multi-GPU to be worth it. **LoRA** freezes the base and trains a low-rank delta `ΔW = BA` on all linear projections: ~0.3–1% of parameters, 7B ≈ **14.7 GB**, 95–99% of full-FT quality, mergeable losslessly, and it is the default below ~50k rows. **QLoRA** is LoRA on top of a 4-bit NF4 base with double quantization and paged optimisers: 7B ≈ **4.9 GB**, which is the difference between "needs an A100" and "runs on a 12 GB card"; it costs ~20–40% more wall-clock per step and 1–3 quality points versus LoRA. The decision order is: QLoRA to prove the pipeline and the data → LoRA when the data is proven → full FT only if you have demonstrated LoRA is the bottleneck.
- **Why asked:** The most-asked PEFT question in existence, and the one where candidates reveal whether they have shipped or only prototyped.
- **Trap:** "QLoRA is a quantization method." It is a *training* method that uses quantization as one component, and its 4-bit base cannot represent the fine distinctions that knowledge injection requires.
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

- **Answer:** From this repo's own table (`code/common/memory.py --table`, single GPU, gradient checkpointing on, AdamW, bf16 weights, fp32 optimiser states): full FT **91.6 GB**, LoRA r16 **14.7 GB**, QLoRA r16 **4.9 GB**; inference-only bf16 is 16.0 GB and 4-bit inference is 4.8 GB. Add 10–20% for allocator and framework overhead and leave headroom. That is the reason LoRA and QLoRA exist: full FT is ~**19x** QLoRA's footprint, and the ratio is driven almost entirely by AdamW's optimiser state (**8 bytes per trainable parameter**, two fp32 moments) plus bf16 gradients (2) on top of the 2 bytes of weights.
- **Why asked:** Every practical decision in fine-tuning is downstream of a VRAM budget, and these three numbers are the ones to have on instant recall.
- **Trap:** Sizing from the parameter count in bf16 alone (7B = 14 GB) and concluding it fits a 24 GB card for *training*. 14 GB is the inference number; the optimiser state is another 56 GB.

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
<!-- CONTINUE -->
