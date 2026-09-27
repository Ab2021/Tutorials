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
<!-- CONTINUE -->
