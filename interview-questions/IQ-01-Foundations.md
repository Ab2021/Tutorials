# IQ-01 — Interview Questions: Foundations — Pretraining, Training & the LLM Lifecycle

| Field | Value |
|---|---|
| **Module** | Foundations (CS-01) — the entry module; these questions define the vocabulary every later module assumes |
| **Pairs with** | CS-01 (case study), CH-01 (cheat sheet) |
| **Total questions** | 100 (30 L1 + 30 L2 + 22 L3 + 8 L4 + 10 L5) + 20 rapid-fire + 5 coding tasks |
| **Levels covered** | Screen (L1) / Intermediate (L2) / Advanced (L3) / System Design (L4) / Debug (L5) |
| **Source material** | `LLM_Fine-Tuning_01_*` (syllabus, ~20 min), `LLM_Fine-Tuning_02_*` (pretraining & training, ~64 min) |

---

## How To Use This File

- **L1 = phone screen / recruiter filter** — 30-second answers. If you cannot answer an L1 in one breath, you will not reach L2.
- **L2 = working engineer** — 2–3 minutes, expects implementation detail: real flags, real numbers, real failure modes.
- **L3 = senior / specialist** — 5 minutes, expects internals, derivations, and trade-offs.
- **L4 = staff / system design** — 15-minute whiteboard. Structure every answer as: requirements → constraints → design → trade-offs → failure modes.
- **L5 = debugging & incident response** — "your loss does X, what do you check and in what order." Answer with a *sequence*, not a list.

**The meta-rule:** an answer of "it depends" is a failure unless it immediately says what it depends on and then picks a default. Always pick the default.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. What is the difference between pretraining and fine-tuning?**

- **Answer:** Pretraining is the first, largest stage: self-supervised next-token prediction over 1T–15T tokens of general text, producing a *base/foundation model*. Fine-tuning is a later, much smaller stage that adapts that model to a task or behaviour using 1k–1M labelled examples. Pretraining teaches the model the language; fine-tuning teaches it the job. Compute differs by 3–5 orders of magnitude: pretraining a 7B costs ~13,000 A100-hours (~$23k), fine-tuning one costs 1–10 A100-hours ($2–20).
- **Why asked:** This is the single distinction the whole discipline rests on. If a candidate says "fine-tuning" when they mean "training from scratch," nothing they say afterwards can be trusted.
- **Trap:** Saying "pretraining is unsupervised and fine-tuning is supervised." The right word for pretraining is **self-supervised** — the labels are derived from the data itself (token *t+1* labels tokens *1..t*). Also, continued pretraining (CS-12) is technically still pretraining.

**Q2. What is a foundation model?**

- **Answer:** A large model pretrained on broad data at scale, intended to be adapted to many downstream tasks rather than used for one. In this course, the pretrained model *is* the foundation model — the terms are used interchangeably. Examples: Llama-3.1-8B (base), Mistral-7B-v0.1, Qwen2.5-7B, BERT-base.
- **Why asked:** Vocabulary check. "Foundation model" is used loosely in marketing; the interviewer wants the technical definition.
- **Trap:** Claiming a foundation model must be multimodal or must be an instruct model. Neither is true — `bert-base-uncased` is a foundation model.

**Q3. Base model vs instruct model — what is the difference, mechanically?**

- **Answer:** A **base** model is a transformer + tokenizer + LM head trained only on next-token prediction. It ships no chat template and has no learned notion of "assistant" — given a question it *continues the text pattern* (a base model asked "What is the capital of France?" may emit "What is the capital of Germany?"). An **instruct** model is the base plus SFT on (instruction, response) pairs (usually plus preference alignment), shipped with a `chat_template` and stop tokens.
- **Why asked:** This is the most consequential and most misunderstood distinction in the field. Getting it wrong causes two of the most expensive failure modes in production (fine-tuning the wrong checkpoint; evaluating a base model with a chat benchmark).
- **Trap:** "They're the same model, just with different prompts." No — the weights differ, and the template is part of the model's contract.

**Q4. What is a token?**

- **Answer:** The atomic unit of model input/output — a sub-word piece produced by the tokenizer. Roughly 1 token ≈ 4 characters ≈ 0.75 English words for GPT-4-class tokenizers, and ≈ 3.5 characters for Llama-2's 32k tokenizer. All costs, context limits and VRAM figures are per-token, not per-word.
- **Why asked:** Screening for whether the candidate thinks in the model's units. Every cost and memory estimate depends on it.
- **Trap:** "A token is a word." It is not, and the gap is 30–40% on English and 3–10x on non-Latin scripts.

**Q5. What is a tokenizer, and why does it matter so much?**

- **Answer:** The bidirectional map between text and integer ids (`tokenizer.json`/`vocab.json`), plus the special-token definitions and the chat template. It matters because it must match the model exactly: a mismatched tokenizer, a wrong chat template, or a missing pad token all produce training runs with a beautiful loss curve and a useless model. It is the single largest source of *silent* failures in fine-tuning.
- **Why asked:** Candidates who have actually trained something have been burned by a tokenizer at least once.
- **Trap:** Believing the tokenizer is interchangeable between models. The token id `128009` must mean the same thing to the tokenizer and the embedding matrix.

**Q6. What is cross-entropy loss and why is it the training objective?**

- **Answer:** `L = −(1/T) Σ log p(x_t | x_<t)`, the mean negative log-probability of the correct token. It is used because it is differentiable, it heavily penalises confident wrong answers (`p=0.001` → loss 6.9), and it gives diminishing reward for extra confidence once correct (`0.9` → 0.105, `0.99` → 0.010). Accuracy is piecewise-constant, so its gradient is zero almost everywhere — you cannot backprop through it.
- **Why asked:** Tests whether the candidate understands *why* this specific loss, not just its name.
- **Trap:** "Cross-entropy measures accuracy." It measures calibration-weighted surprise, which is a different thing — low loss with a useless model is common.

**Q7. What is perplexity?**

- **Answer:** `perplexity = exp(cross-entropy loss)` — the effective number of equally likely choices the model is choosing among per token. A uniform model over a 32,000-token vocabulary has loss `ln(32000) = 10.37` and perplexity 32,000. A good general LM on Wikipedia sits around 2.2–2.6 (perplexity 9–13.5).
- **Why asked:** It is the standard pretraining metric and candidates frequently misquote it.
- **Trap:** Comparing perplexity across two models with **different tokenizers** or different test sets. It is not a comparable quantity — a 128k vocabulary has more ways to be wrong.

**Q8. What is an epoch, and what is a step?**

- **Answer:** An **epoch** is one full pass over the training dataset. A **step** is one optimiser update. With gradient accumulation, one step consumes `micro_batch × grad_accum × n_gpus` sequences, so a step is not the same as a forward pass. For SFT, 1–3 epochs is typical; for pretraining, epoch counts are meaningless because you never complete one — the unit is tokens.
- **Why asked:** Basic vocabulary, but the micro/global batch confusion is extremely common in config reviews.
- **Trap:** Reporting "the model trained for 3 epochs" as evidence of quality. For SFT on 500 examples, 3 epochs is a memorisation warning.

**Q9. Hyperparameter vs parameter vs activation?**

- **Answer:** **Parameters** are learned weights (7B = 7e9 of them) — what determines capability, VRAM and compute. **Hyperparameters** are set by you (LR, batch size, LoRA rank, epochs) and control how training proceeds. **Activations** are intermediate tensors produced in the forward pass and retained for the backward pass; they are usually the largest *variable* memory term, scaling with batch and sequence length.
- **Why asked:** Screening for whether the candidate can discuss memory without conflating the three.
- **Trap:** Forgetting activations when sizing VRAM, then hitting an OOM at a batch size that "should have fit."

**Q10. What is LoRA in one sentence?**

- **Answer:** LoRA freezes the base model's weights and learns a low-rank update per target matrix, `ΔW = BA` where `A ∈ R^{r×d}` and `B ∈ R^{k×r}` with `r ≪ min(k,d)` (typically `r` = 8–64), reducing trainable parameters by ~100–1,000x and optimiser memory proportionally.
- **Why asked:** The most-asked PEFT question in existence. Answer it in one breath.
- **Trap:** Saying LoRA makes training *faster*. It reduces memory; compute stays ≈6ND because the forward pass still runs through the frozen base. Reported speedups come from kernel fusion and skipping frozen-weight gradient buffers.

**Q11. What is QLoRA?**

- **Answer:** LoRA on top of a frozen base that has been quantized to **4-bit NF4** with double quantization, plus paged optimisers and (optionally) bf16 compute. It cuts a 7B base from 14 GB (bf16) to ~3.9 GB, which is what makes fine-tuning a 7B possible on a 12–16 GB card. It is ~20–40% slower per step than bf16 LoRA and costs ~1–3 points of quality versus LoRA.
- **Why asked:** The default entry point for anyone fine-tuning on constrained hardware.
- **Trap:** "QLoRA is a quantization method." It is a *training* method that uses quantization as one component.

**Q12. What is quantization, in one sentence?**

- **Answer:** Storing and/or computing with fewer bits per weight — fp32 (4 bytes) → fp16/bf16 (2) → int8 (1) → int4/NF4 (~0.5) — trading a small quality loss for a large memory saving and, for decode, a proportional speedup.
- **Why asked:** Foundational; the subject of two later modules (CS-10, CS-11).
- **Trap:** "Quantization is free." It is not — always re-run the eval after quantizing (int4 typically costs 1–3 points).

**Q13. What is the causal language modelling objective, and which models use it?**

- **Answer:** Predict token *t+1* from tokens *≤t*, left-to-right, with a causal attention mask. It supervises **every** position (100%) and its training objective is identical to its inference procedure, which is why it also enables in-context learning. Used by GPT, Llama, Mistral, Qwen, Gemma, and essentially every modern generative LLM — all decoder-only.
- **Why asked:** The objective is the architecture's defining feature.
- **Trap:** Calling it "regression" or "recursive." The term is **autoregressive**.

**Q14. What is masked language modelling, and which models use it?**

- **Answer:** Mask ~15% of tokens and predict them from **both** directions — bidirectional context. Used by BERT and the encoder-only family (RoBERTa, DistilBERT, ELECTRA, DeBERTa). It supervises only ~15% of positions per sequence and cannot generate, but it produces stronger bidirectional representations for classification, NER, extractive QA and sentence embeddings.
- **Why asked:** Tests whether the candidate dismisses encoders as obsolete.
- **Trap:** "MLM is dead." It is the right tool for discriminative tasks and for embeddings (`code/10_embedding_finetune.py`; CS-22 planned, not yet written) — you cannot train a good sentence encoder with a causal mask and mean pooling alone.

**Q15. There is a third objective — what is it, and who uses it?**

- **Answer:** **Span corruption** (a generalisation of MLM): mask contiguous spans with sentinel tokens and have a decoder reconstruct them. Used by T5, mT5, BART and the UL2 family — encoder-decoder architectures. It is efficient and strong for seq2seq tasks (translation, summarisation, structured extraction).
- **Why asked:** Completes the taxonomy. Most candidates know two of the three.
- **Trap:** Lumping span corruption in with MLM. The distinguishing feature is that the *decoder* reconstructs the removed spans, so there is a real generation target.

**Q16. What is transfer learning, and where does fine-tuning fit?**

- **Answer:** Transfer learning is reusing a representation learned on task A for task B — the *goal*. Fine-tuning is the *mechanism*. The instructor's framing is exact: "Transfer learning is nothing, it is just a way to perform a fine-tuning." Two forms: (1) **feature extraction** — freeze everything, train a new head; (2) **partial/full fine-tuning** — freeze early layers, train later ones (or all of them).
- **Why asked:** Distinguishes people who understand the concept from people who memorised the word.
- **Trap:** Treating them as synonyms, or believing transfer learning is a distinct algorithm. It is not — it is a description of what fine-tuning achieves.

**Q17. What is a chat template, and why does it matter?**

- **Answer:** The exact string format that wraps system/user/assistant turns for a given model family — Llama-3 uses `<|start_header_id|>role<|end_header_id|>`, Mistral uses `[INST]`, Qwen uses ChatML (`<|im_start|>`), Gemma uses `<start_of_turn>`. It matters because applying the wrong one is the #1 *silent* SFT failure: the loss falls smoothly and the model is unusable in the app. Always render it with `tokenizer.apply_chat_template(..., tokenize=False)` and read the string before training.
- **Why asked:** The highest-signal practical question in the entire bank.
- **Trap:** "The tokenizer handles it." Only if you call `apply_chat_template` — and only with the template that belongs to *your* model, loaded from the model's own repo.

**Q18. What is catastrophic forgetting?**

- **Answer:** Fine-tuning on a narrow distribution degrades capabilities the base model had — general reasoning, other languages, instruction following, safety behaviour. It happens because gradient descent moves all weights toward the narrow objective with no constraint preserving prior behaviour. Mitigations: lower LR, fewer epochs, LoRA (which freezes the base), and mixing 5–20% general instruction data into the SFT set.
- **Why asked:** It is the failure mode nobody measures, and it is the one that causes incidents.
- **Trap:** "A low learning rate prevents it." It *mitigates* it. Only freezing the base (LoRA) or mixing in general data substantially avoids it — and even then, safety behaviour can shift.

**Q19. What is RAG, and how is it different from fine-tuning?**

- **Answer:** Retrieval-augmented generation fetches relevant documents at inference time and puts them in the context window. It changes the *input*; fine-tuning changes the *weights*. Fine-tuning teaches form (style, format, schema, tone, refusal behaviour, domain vocabulary); RAG supplies substance (current, verifiable, citable facts). They compose: the instructor's own recommended architecture is a fine-tuned model under a RAG and an agent.
- **Why asked:** The architecture decision that precedes every other decision (CS-04).
- **Trap:** Treating them as alternatives. The correct answer is almost always "both, for different reasons."

**Q20. When would you fine-tune instead of using RAG?**

- **Answer:** When the problem is **behaviour or form**, not knowledge: the model ignores your output schema; it needs a specific tone or persona; it mangles domain jargon; it refuses benign in-domain requests; or — the strongest financial case — your few-shot prompt is 2,000–6,000 tokens and the per-request cost is dominated by prompt length. Rule of thumb: if your training data is *documents*, you want RAG; if it is *demonstrations of behaviour*, you want fine-tuning.
- **Why asked:** Interviews for applied roles use this as the opening architecture question.
- **Trap:** "Fine-tuning is better because it's permanent." Permanence is a liability for facts — it is a snapshot you cannot update or cite.

**Q21. What is an embedding?**

- **Answer:** A dense vector representation. In this course's framing, "an embedding is just a set of numbers — a vector." Two distinct uses: **token embeddings** (the model's input layer, `V × h`) and **sentence/document embeddings** (for retrieval, produced by an encoder like `all-MiniLM-L6-v2` or `bge-large`). Only the second is a retrieval primitive.
- **Why asked:** The word is overloaded and the confusion causes real architectural mistakes.
- **Trap:** Confusing them. "We'll fine-tune the embeddings" means two completely different projects depending on which one is meant.

**Q22. What is VRAM and why does it bind everything?**

- **Answer:** GPU memory. It determines which technique is *possible*, before any question of speed or quality. The sizing rule for full fine-tuning is **~16 bytes per parameter** (bf16 weights 2 + bf16 grads 2 + fp32 Adam m,v 8 + fp32 master 4), so a 7B needs 112 GB before activations ≈ 128 GB total. That single number is why LoRA and QLoRA exist.
- **Why asked:** Every practical decision in fine-tuning is downstream of a VRAM budget.
- **Trap:** Sizing from parameter count in bf16 alone (7B = 14 GB) and concluding it fits on a 24 GB card for training. That is the *inference* number.

**Q23. bf16 vs fp16 — which do you train in and why?**

- **Answer:** **bf16** on Ampere and newer. bf16 has 8 exponent bits, so it has the same dynamic range as fp32 and will not overflow; fp16 has 5 exponent bits and 10 mantissa bits, so it overflows without loss scaling, producing NaN loss. fp16 has more mantissa precision, which matters for *inference* on older hardware (V100, T4) but not for training stability.
- **Why asked:** A NaN loss in an interview story usually traces back to this.
- **Trap:** "fp16 is more precise so it's safer." Backwards — for training, range matters more than precision.

**Q24. What is gradient accumulation?**

- **Answer:** Running N micro-batches, summing their gradients, and taking one optimiser step — so the *global* batch is `micro_batch × accum × n_gpus` while the GPU only ever holds one micro-batch's activations. It costs no extra compute but makes each step N× slower in wall clock, and the effective learning rate should be tuned for the *global* batch, not the micro-batch.
- **Why asked:** The standard workaround for VRAM limits; tests whether the candidate understands what it does and does not buy.
- **Trap:** "Gradient accumulation is free." It is free in FLOPs, expensive in wall-clock, and it changes the optimisation dynamics if you do not adjust the LR.

**Q25. What is gradient checkpointing?**

- **Answer:** Instead of storing every intermediate activation for the backward pass, store only each block's *input* and recompute the rest during backward. It cuts activation memory ~5–10x (for a 7B at seq 2048, from ~9 GB to under 1 GB) at a cost of ~20–33% extra compute. It is a **memory** optimisation, not a speed one.
- **Why asked:** Almost always the correct trade for LLM fine-tuning, and frequently misunderstood as a speed trick.
- **Trap:** Enabling it expecting faster training. If someone reports "gradient checkpointing made training faster," they changed something else too.

**Q26. What is learning-rate warmup and why does it matter?**

- **Answer:** The LR ramps linearly from ~0 to the peak over the first 1–10% of steps. It matters because Adam's second-moment estimate `v` is computed from very few samples early on, making the effective step size `η/√v̂` unstable and large. Without warmup, LLM training frequently diverges in the first few hundred steps. Typical `warmup_ratio` is 0.03–0.05.
- **Why asked:** A cheap way to distinguish people who have run a training job from people who have read about one.
- **Trap:** Setting `warmup_ratio=0` because "the model should learn from step 0." It will, and then it will diverge.

**Q27. What is a LoRA adapter, physically?**

- **Answer:** A small file (a few MB to a few hundred MB) containing the `A` and `B` matrices for each targeted layer, plus the config (`r`, `alpha`, `dropout`, `target_modules`, base model id). It is **not** a standalone model — it must be loaded with its base, or merged into it (`W' = W + (α/r)·BA`).
- **Why asked:** Tests whether the candidate has actually shipped one. Adapter/base mismatch is a common production bug.
- **Trap:** Publishing an adapter without the base model id and revision in its config, which makes it unloadable six months later.

**Q28. What is the 6ND rule?**

- **Answer:** Training compute FLOPs ≈ `6 × N × D`, where N is parameters and D is training tokens. Derivation: the forward pass is 2N FLOPs per token (one multiply and one add per weight), and the backward pass is 4N (two matmuls of the same size — gradient with respect to the input, and gradient with respect to the weights). Inference is `2 × N` FLOPs per token.
- **Why asked:** The most useful formula in the field. It converts any plan into a time and dollar estimate.
- **Trap:** Forgetting the factor of 6 and using 2ND for training, which underestimates cost by 3x. Also: real runs burn 1.5–3x the theoretical FLOPs because of restarts, evaluation and failed runs.

**Q29. What is the Chinchilla rule?**

- **Answer:** For a fixed training-compute budget, model size and training tokens should scale in **equal proportion**: `N_opt ∝ C^0.5` and `D_opt ∝ C^0.5`, giving `D_opt ≈ 20 × N` — about 20 tokens per parameter. A 7B model is compute-optimal at ~140B tokens. Chinchilla (70B / 1.4T tokens) matched Gopher (280B / 300B tokens) at the same compute, showing GPT-3 (175B / 300B) was ~4x oversized for its data.
- **Why asked:** The most-quoted scaling result, and the most commonly misapplied.
- **Trap:** Treating 20 tokens/parameter as a production prescription. Modern models are deliberately *overtrained* 14–94x past Chinchilla (Llama-3-8B: 1,875 tokens/param) because inference cost recurs and training cost does not.

**Q30. What is hallucination, and can fine-tuning fix it?**

- **Answer:** Fluent, confident, wrong output. It occurs because the model is a probability distribution over tokens, not a database — asked about something outside its training distribution it returns the nearest plausible continuation, exactly as the ImageNet ResNet in the video labels a tomato as "strawberry, pitcher, orange" because "tomato" is not in its label set. Fine-tuning **cannot** fix it and usually makes it worse: SFT on documents teaches the model the *style* of confident assertion without adding verifiable facts. Fix it with RAG plus citations, output validation, and calibrated uncertainty prompting.
- **Why asked:** The most common misconception that reaches production.
- **Trap:** "Fine-tune on our docs so it stops hallucinating." This is the canonical failed project (see CS-01 case study 15.4: hallucination rose from 18% to 41%).

---

## Level 2 — Applied & Implementation

**Q31. Walk me through your end-to-end fine-tuning pipeline.**

- **Answer:** (1) Define the task and check whether prompting or RAG solves it — if yes, stop. (2) Collect 1k–10k high-quality pairs, preferably from production logs. (3) Apply the model's chat template and mask prompt tokens to `-100`; inspect the rendered string. (4) Split by source document, dedup across splits, hold out 10–20%. (5) Choose QLoRA by default, LoRA if quality demands, full FT only if demonstrated necessary. (6) Train, monitoring loss and grad-norm, early-stopping on val loss. (7) Evaluate task metric + regression suite + safety suite against a well-prompted base baseline. (8) Merge, quantize, re-evaluate. (9) Serve with version pinning and monitoring. (10) Mine production failures back into the dataset.
- **Why asked:** Tests whether the candidate has owned a project or just a training script.
- **Trap:** Starting at step 6. The most common real-world failure is skipping steps 1 and 7.

**Q32. What learning rate would you use for full fine-tuning vs LoRA, and why the difference?**

- **Answer:** Full FT: `1e-5` to `5e-5`, typically `2e-5`. LoRA: `1e-4` to `3e-4`, typically `2e-4`. QLoRA: `1e-4`, slightly lower than LoRA because NF4 dequantization injects noise into the gradient path. The 10x difference exists because LoRA's adapters initialise at **zero** — there is no pretrained structure to destroy, so large steps are safe — whereas a 2e-4 update to a fully pretrained weight matrix erases the pretraining distribution within ~100 steps.
- **Why asked:** The most common configuration error in practice.
- **Trap:** Using `2e-4` with full fine-tuning and reporting "the model got worse." It did — you overwrote it.

**Q33. How do you choose LoRA rank and alpha?**

- **Answer:** Start `r=16, alpha=32` (or `alpha=r`) and treat `r` as capacity. `r=8` for style/tone/format adaptation; `r=16–32` for a new behaviour or domain; `r=64–128` for something close to a new capability, at the cost of more VRAM and overfitting risk on small data. The effective scaling is `alpha/r`, so raise both together or you silently change the effective learning rate. Empirically, past `r=64` the returns flatten on most tasks.
- **Why asked:** Rank is the one LoRA knob with a real quality effect, and it is frequently set by copy-paste.
- **Trap:** Setting `r=16, alpha=16` and then separately tuning `lr` — you are tuning the same thing twice. And `r=128` on 500 examples will overfit.

**Q34. Which `target_modules` do you use for LoRA, and which do people get wrong?**

- **Answer:** All linear layers: `q_proj, k_proj, v_proj, o_proj` (attention) **and** `gate_proj, up_proj, down_proj` (MLP). The common mistake is applying LoRA only to `q_proj, v_proj` — the original paper's minimal setting — which leaves the MLP blocks (two-thirds of the parameters and most of the model's factual storage) unadapted. Attention-only LoRA is measurably worse on nearly every task.
- **Why asked:** A one-line answer that separates practitioners from tutorial-followers.
- **Trap:** Also forgetting that `target_modules` names differ by architecture (`c_attn` for GPT-2, `query`/`value` for T5, `Wq`/`Wv` for some others). A wrong list either raises an error or silently adapts nothing.

**Q35. How do you compute the global batch size, and what value do you target?**

- **Answer:** `global_batch = per_device_train_batch_size × gradient_accumulation_steps × n_gpus`. In tokens: multiply by `max_seq_length`. For SFT, target roughly 64k–2M tokens per step; 128k is a good default. Example: 4 GPUs × micro-batch 2 × seq 2048 × accum 8 = `4 × 2 × 2048 × 8 = 131,072` tokens per step. If you change the global batch, re-tune the LR — the two are coupled (`LR ∝ sqrt(batch)` as a conservative scaling rule).
- **Why asked:** Config review. People routinely report a "batch size" that is really a micro-batch.
- **Trap:** Confusing `per_device_train_batch_size` with the global batch, then wondering why two runs with "the same batch size" behave differently on different hardware.

**Q36. How do you handle examples longer than `max_seq_length`?**

- **Answer:** First look at the actual distribution — the p99 might be 900 tokens, in which case raising `max_seq_length` from 1024 to 4096 costs 4x the memory for nothing. If truncation is needed, truncate the **input from the left** (or filter), never the right, because right truncation cuts the assistant's answer — which is the only supervised part. Log the truncated fraction; if it is above ~2%, fix the data or the length. Attention memory scales with `s²` without flash attention and `s` with it, so length is expensive in a superlinear way.
- **Why asked:** Truncation bugs are silent and common (CS-01 case study 15.1 lost 6% of examples to it).
- **Trap:** `truncation=True, max_length=512` with the default `truncation_side="right"` — this deletes the labels and produces a model that generates fragments.

**Q37. Why `tokenizer.pad_token = tokenizer.eos_token`?**

- **Answer:** Llama/Mistral-class tokenizers ship with **no pad token**, because pretraining never needed padding. `Trainer` will raise `ValueError: Asking to pad but the tokenizer does not have a padding token`, or a naive collator will pad with id 0 and corrupt the batch. Setting `pad_token = eos_token` and `pad_token_id` fixes it. Two consequences to handle: the loss mask must exclude pad positions so the model does not learn to emit EOS everywhere, and the pad token must not appear in the middle of real sequences.
- **Why asked:** Every person who has fine-tuned a Llama model has hit this in their first ten minutes.
- **Trap:** Adding a *new* pad token instead (`tokenizer.add_special_tokens({"pad_token": "<pad>"})`) without resizing the embedding matrix, which produces an index-out-of-range error or random embeddings.

**Q38. Why mask prompt tokens to `-100`, and how do you verify it?**

- **Answer:** The loss should be computed on the **response** tokens only. If you leave the prompt unmasked, most of the loss comes from learning to generate the *questions* — which is not what you want, and the reported loss looks better than it is because prompt tokens are highly predictable. To verify: reconstruct the supervised span from `label != -100` and decode it. It must equal exactly your assistant response plus the turn's EOS token. If it does not, stop.
- **Why asked:** The single most common silent SFT bug.
- **Trap:** Assuming `DataCollatorForLanguageModeling` masks for you. It does not — for SFT you need `DataCollatorForSeq2Seq`, TRL's SFT trainer with `train_on_inputs=False`, or a hand-written collator.

**Q39. How many epochs do you train for, and how do you decide?**

- **Answer:** 1–3 for SFT, starting at 1 and adding only if the eval says the behaviour has not been adopted. The stopping rule is **eval loss**, not train loss: train at the epoch where eval loss reaches its minimum, which for small datasets is often before epoch 1 completes. With fewer than 2,000 examples, 2–3 epochs is common; above 100k examples, 1–2. Above 3 epochs you are training a memoriser unless you have specific evidence otherwise.
- **Why asked:** Overfitting on small SFT sets is the most common quality complaint.
- **Trap:** Carrying over the classical-ML habit of 50 epochs. At `lr=2e-4` with LoRA on 1,000 examples, epoch 10 produces verbatim memorisation.

**Q40. Packing vs padding — what is the difference and when does packing hurt?**

- **Answer:** Padding fills each sequence to `max_seq_length` with pad tokens so a batch is rectangular; if your examples average 300 tokens and you pad to 2,048, you waste ~85% of the compute. Packing concatenates multiple short examples into one full-length sequence, eliminating that waste. It **hurts** for SFT when it concatenates unrelated examples without masking: the model sees one example's response as the continuation of another's prompt, and learns spurious cross-example dependencies. It is safe when each example is delimited by its own BOS/EOS, or when a masked/packed strategy resets attention at boundaries.
- **Why asked:** Tests whether the candidate understands that a 3x throughput win has a correctness cost.
- **Trap:** Enabling `packing=True` for SFT and seeing loss fall faster, then shipping a model that answers the previous user's question.

**Q41. How do you split data for SFT?**

- **Answer:** Split **by source**, not by row: all rows derived from the same document, conversation, user or template go into the same split. Then deduplicate *across* the splits with MinHash/LSH — row-wise splitting of near-duplicates is the classic cause of "our fine-tune scores 0.97." Hold out 10–20% in total, with separate val (for early stopping) and test (touched once) sets. For small datasets, hold out at least 100 examples so the confidence interval is meaningful.
- **Why asked:** Leakage detection is a skill that separates careful engineers from fast ones.
- **Trap:** `train_test_split(df, test_size=0.1, random_state=42)` on rows. With near-duplicate SFT data this leaks and inflates every metric.

**Q42. What does a good SFT dataset row look like?**

- **Answer:** A list of role-tagged messages with the assistant's target included: `{"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}`. The system turn should be present in a meaningful fraction of rows (otherwise the model stops respecting system prompts, because it never saw the role). The assistant content should be the *ideal* answer, not a transcript of a real support agent's typos. Format and length should match what production will send and expect.
- **Why asked:** Data is 80% of the project; a candidate who cannot describe their row format has not built one.
- **Trap:** All rows sharing one template — 5,000 paraphrases of a single question teaches a template-follower, not a behaviour.

**Q43. How would you estimate VRAM for a LoRA run before renting a GPU?**

- **Answer:** `weights + adapters + adapter_optimiser_state + activations + overhead`. For a 7B LoRA in bf16: base 14 GB (frozen, 2 bytes/param) + ~40M trainable params × 16 bytes ≈ 0.6 GB + activations ~5–10 GB (b=1–2, seq 1024–2048, gradient checkpointing on) + ~2–3 GB fragmentation ≈ **22–28 GB**, so a 24 GB card is tight and a 40 GB card is comfortable. The dominant *variable* term is activations, so sequence length and micro-batch are the levers.
- **Why asked:** Tests whether the candidate reasons from components or guesses.
- **Trap:** Using the full-FT number (16 bytes/param × total params = 112 GB) and concluding a 7B needs 8 GPUs when LoRA needs one.

**Q44. How do you estimate training time before you start?**

- **Answer:** `time = 6ND / (n_gpus × peak_TFLOPs × MFU)`. Example: 7B, 15M tokens, 1 × RTX 4090 at ~50 TFLOP/s effective (QLoRA overhead reduces the 165 TFLOP/s peak): `6 × 7e9 × 1.5e7 = 6.3e17`, `6.3e17 / 5e13 = 12,600 s ≈ 3.5 h`. Then multiply by 1.3–2x for real-world overheads — dataloading, checkpointing, evaluation, and gradient checkpointing's recompute. And sanity-check against a known reference: Llama-2-7B used ~184,000 A100-hours for 2T tokens.
- **Why asked:** Distinguishes people who budget from people who hope.
- **Trap:** Using peak TFLOPs (165 for a 4090) as the throughput. Realistic MFU is 20–50%, and 20–30% is common on a first attempt.

**Q45. Your loss curve falls smoothly but the model is unusable. What do you check first?**

- **Answer:** The **chat template**. A wrong template produces exactly this signature — the model learns a perfectly consistent format that is not the one your application sends. Render the training string and the inference string with `apply_chat_template(..., tokenize=False)` and diff them character by character. Second and third candidates: loss computed on prompt tokens (the model learned to generate questions), and a tokenizer/model mismatch.
- **Why asked:** The most valuable debugging intuition in SFT work.
- **Trap:** Adding more data or training longer. Neither fixes a format bug, and both cost money.

**Q46. More data or better data — how do you decide?**

- **Answer:** Diagnose by *how* it is failing. If the model performs the behaviour but inconsistently, or fails on input patterns you have seen before, you need **more** data (coverage). If it fails in a way that suggests it never learned the behaviour at all, or the failures cluster on a specific input type, you need **better** data (targeted examples of exactly the failing case). The cheapest diagnostic: hand-write 50 examples of the failing pattern and re-run for 100 steps. If the failure disappears, it was a coverage problem and 500 targeted examples will fix it. LIMA shows 1,000 curated examples can produce a well-behaved assistant; 100,000 scraped ones frequently cannot.
- **Why asked:** The question behind most "should we collect more data" meetings.
- **Trap:** "More data is always better." Duplicated or low-quality data actively hurts — it teaches the model to reproduce noise.

**Q47. When would you choose full fine-tuning over LoRA?**

- **Answer:** When you have demonstrated that LoRA is the bottleneck: a measurable quality gap on a held-out task with **≥100k examples** and a diverse distribution; when you are teaching a genuinely new capability or a new language rather than refining a behaviour; when you need to change the tokenizer's embedding space (new vocabulary); or when you need the last 1–3 points and the budget exists. Full FT also avoids the adapter-loading and merging complexity in serving. Otherwise LoRA is the right default: comparable quality below ~10k examples, 3–5x less memory, and a much faster iteration loop.
- **Why asked:** Tests whether the candidate reaches for the expensive tool by default.
- **Trap:** "Full fine-tuning is always better because more parameters change." At small data volumes LoRA often *matches or beats* full FT, because the frozen base acts as a regulariser.

**Q48. How do you merge a LoRA adapter, and what dtype should you use?**

- **Answer:** `model = model.merge_and_unload()` after `PeftModel.from_pretrained(base, adapter)`. Merge in **fp32 or bf16**, not fp16 — merging into a low-precision base loses quality, and merging a QLoRA adapter into an NF4 base is not meaningfully supported (most toolchains refuse, or silently degrade). For QLoRA the correct deployment path is: dequantize the base to bf16, merge the adapter, then re-quantize for serving and **re-run the eval**. Merging is mathematically lossless for plain LoRA at matched precision, and it removes the PEFT dependency and the per-request adapter lookup.
- **Why asked:** Serving decisions made at merge time are hard to reverse.
- **Trap:** `merge_and_unload()` on a QLoRA model and shipping it, then wondering why quality dropped.

**Q49. How do you evaluate a fine-tune?**

- **Answer:** Four layers, and you need all four. (1) **Task metric** — schema-validity rate, F1, exact match — on a held-out, deduplicated, uncontaminated set, reported with bootstrap confidence intervals. (2) **Continuous metric on the same outputs** — token-level F1 or log-probability of the correct answer — because a strict metric can hide smooth improvement (the emergence-as-metric-artefact problem). (3) **Regression suite** — 50–200 items covering reasoning, code, multilingual, long context — to catch catastrophic forgetting. (4) **Safety suite** — refusals and jailbreaks. Plus, always, a **well-prompted base-model baseline** and a **prompt-only baseline**: if `base + 8-shot` matches your fine-tune, you did not need the fine-tune.
- **Why asked:** The most common interview failure is evaluating only the task metric.
- **Trap:** Comparing against a zero-shot base model and declaring victory.

**Q50. How do you build an eval set when you have none?**

- **Answer:** Three sources, in order of value. (1) **Real production inputs with hand-written gold answers** — 100–300 items is enough to start, and they are in-distribution by construction. (2) **Post-cutoff or internal data** the base model provably never saw — the only robust defence against contamination. (3) **Synthetic items generated from a schema or taxonomy**, then human-filtered, to cover the input space systematically. Write the rubric *before* collecting the answers, freeze the set, version it, and never tune on the test split.
- **Why asked:** Teams skip eval-set construction because it is slow and unglamorous; interviewers probe for it deliberately.
- **Trap:** Using the training-set's own distribution to build the eval and then reporting the model is excellent — which it is, on data that resembles its training set.

**Q51. What is a regression suite and why does it matter more than the task metric?**

- **Answer:** A compact fixed set of 50–200 items spanning capabilities unrelated to your task — basic reasoning, arithmetic, code, another language, long-context retrieval, refusal behaviour. It matters because fine-tuning degrades *everything* the base could do, in proportion to how narrow and how long you trained. The task metric cannot see this; the regression suite is the only thing that catches it. It is also cheap: 200 items run in minutes.
- **Why asked:** Catches people who have only ever measured the thing they optimised.
- **Trap:** "We only care about our task, so we don't need it." Your users care about the whole model, and safety behaviour is part of the regression surface.

**Q52. How do you use LLM-as-judge correctly?**

- **Answer:** Use it for **pairwise preference** with a strong judge, a written rubric, and **both orderings evaluated and averaged** (to cancel position bias). Report a win rate with a confidence interval, not an absolute score. Known biases to control: position (prefers first), verbosity (prefers longer), self-preference (prefers its own family's style). Calibrate against a human-labelled subset of 50–100 items and report the agreement rate — if agreement is below ~70%, the judge is not measuring what you think.
- **Why asked:** LLM-judge is now standard, and it is routinely used badly.
- **Trap:** Using a judge from the same family as the model being evaluated, single-ordering, and reporting an absolute 1–10 score with no calibration.

**Q53. How would you serve 50 customer-specific fine-tunes cost-effectively?**

- **Answer:** One shared frozen base plus 50 LoRA adapters, served with a multi-LoRA engine (vLLM's `--enable-lora`, TGI, or S-LoRA). This gives one set of base weights in VRAM plus ~50–200 MB per adapter, versus 50 full models at 14 GB each. Trade-offs: adapter load/unload latency on first request, slightly higher per-token overhead than a merged model, and a hard requirement that all tenants share the same base and tokenizer. Merge per-tenant *only* if a tenant has enough traffic to justify a dedicated deployment.
- **Why asked:** The standard multi-tenant architecture question for a platform role.
- **Trap:** Serving 50 merged models. At 14 GB each that is 700 GB of GPU memory for the same capability.

**Q54. How can fine-tuning reduce your inference bill?**

- **Answer:** By shortening the prompt. If a 6,000-token few-shot system prompt can be replaced by a 180-token one after SFT, per-request input cost falls ~33x. Worked example at mid-tier API pricing: `6,000 in × $3/1M + 400 out × $6/1M = $0.0204/request` versus `180 × $3/1M + 400 × $6/1M = $0.0029/request` — from $18,360/month to $2,610/month at 900k requests, against ~$3 of training compute. This is the strongest financial case for fine-tuning, and it is the one to lead with.
- **Why asked:** Ties the technique to a business outcome. Senior interviews are mostly about this.
- **Trap:** Justifying fine-tuning by quality alone, with no cost or latency number attached.

**Q55. What does `warmup_ratio` do and what value do you use?**

- **Answer:** It sets the fraction of total steps spent ramping the LR from ~0 to its peak, typically `0.03–0.05` (3–5%). It exists to stabilise Adam early, when the second-moment estimate `v` is computed from few samples and the effective step `η/√v̂` is large and noisy. With a very small number of total steps (e.g. 60), a 5% warmup is 3 steps and effectively does nothing useful — set `warmup_steps` explicitly for short runs.
- **Why asked:** Small config detail that reveals whether someone has watched a loss curve from step 0.
- **Trap:** Setting `warmup_ratio=0.5` on a 60-step run, which means half the run is at a suboptimal LR.

**Q56. How do you handle class imbalance in SFT data?**

- **Answer:** Do not duplicate minority examples — that causes overfitting to the specific copies. Instead: (1) **resample with weights** via a `WeightedRandomSampler`; (2) **write new diverse examples** of the minority class (this is the real fix); (3) **reweight the loss** per example if the framework supports it; (4) for classification heads (CS-07), use class weights or focal loss. Note that SFT is not classification — the "class" is a behaviour pattern, and the most common imbalance is that 90% of your examples are the same *format*, which is a diversity problem more than a balance problem.
- **Why asked:** Tests whether the candidate reaches for duplication, which is the standard wrong answer.
- **Trap:** Oversampling by copy-paste. It works in classical ML and reliably overfits in SFT.

**Q57. LoRA vs QLoRA in quality terms — what is the real cost?**

- **Answer:** Roughly 1–3 points on most task metrics, in favour of LoRA, with the gap widening as data volume grows (at 1M+ examples the gap reaches 2–5 points versus full FT). QLoRA's costs: NF4 quantization error in the frozen base, dequantization noise in the gradient path, and 20–40% more wall-clock per step. Its benefit is decisive: a 7B base drops from 14 GB to 3.9 GB, which is the difference between "needs an A100" and "runs on a 12 GB card." Use QLoRA to prove the pipeline and the data, then move to LoRA when the data is proven.
- **Why asked:** Prevents the common mistake of shipping QLoRA quality when a cheap upgrade was available.
- **Trap:** Believing QLoRA matches LoRA exactly. It does not, though the gap is small enough that it rarely matters below ~50k examples.

**Q58. How do you fine-tune BERT for a classification task, concretely?**

- **Answer:** `AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=k)`, `lr=2e-5` (or 3e-5), `batch_size=16–32`, 2–4 epochs, `max_length=256` (look at your p99 first), AdamW with `weight_decay=0.01` and no weight decay on bias/LayerNorm, linear warmup over 10% of steps. Use `[CLS]`-pooled logits. On a 14-class task with ~1,400 training examples, expect F1 ≈ 0.80–0.88 versus ~0.40 trained from scratch. For inference, the encoder is bidirectional, so no causal mask and no chat template; batch with right padding (classification has no generation step, so padding side does not matter).
- **Why asked:** Encoder fine-tuning is still the highest-ROI fine-tuning in industry (CS-07), and it is where most candidates have actually shipped something.
- **Trap:** Using a generative model for classification. You get worse accuracy, higher latency, higher cost, and non-deterministic labels.

**Q59. How would you approach fine-tuning for a non-English language?**

- **Answer:** First check the tokenizer's cost on your language — an English-centric 32k vocabulary consumes 3–10x more tokens per word for Hindi, Thai, or Amharic, which directly multiplies your training and inference cost. Prefer a base with a large multilingual vocabulary (Llama-3's 128k, Gemma's 256k, or a genuinely multilingual model like Aya or Qwen). Then: (1) continue pretraining on in-language text for vocabulary and syntax (CS-12) — SFT alone cannot install a language the base has never seen; (2) SFT on in-language instruction data, ideally translated or authored natively rather than machine-translated from English; (3) evaluate with in-language eval sets, since English benchmarks translate poorly and are contaminated.
- **Why asked:** Tests whether the candidate knows the tokenizer's role in multilingual cost.
- **Trap:** Going straight to SFT on 5,000 translated instruction pairs and expecting fluent output. The base has to have the language before SFT can shape its use.

**Q60. What is the single best piece of advice you would give someone starting their first fine-tune?**

- **Answer:** Build the eval set and the well-prompted baseline *before* you train anything. Without a baseline you cannot tell whether fine-tuning helped; without a held-out set you cannot tell whether it helped for the right reason. The second-best advice: start with QLoRA and 1 epoch, and inspect the rendered chat template string before spending a GPU-hour. Together these prevent the four most common failure modes — no measurement, wrong template, wrong LR, and overfitting.
- **Why asked:** A culture-fit question. It reveals what someone has learned the hard way.
- **Trap:** "Start with full fine-tuning to get the best quality." You cannot afford it, and you do not yet know what quality means for your task.

---

## Level 3 — Advanced / Specialist

**Q61. Derive the 6ND rule from first principles.**

- **Answer:** Consider one token through one weight matrix of `N` parameters. The forward pass computes `y = Wx`, which is `N` multiply-accumulate operations = `2N` FLOPs (a multiply and an add each count as one). The backward pass computes two gradients of the same shape — `∂L/∂x = Wᵀ·∂L/∂y` and `∂L/∂W = ∂L/∂y · xᵀ` — each costing `2N`, so `4N`. Total per token: `2N + 4N = 6N`. Over `D` tokens: **`C_train ≈ 6ND`**. Inference only runs the forward pass, so `C_infer ≈ 2N` per token (plus the attention term, which is `12·L·s²·h` per sequence at sequence length `s`, layers `L`, heads `h` — negligible at short context, dominant at long).
- **Why asked:** It is the single formula that converts a plan into a bill. An interviewer wants to see the derivation, not the memorised result.
- **Trap:** Counting a FLOP as one operation instead of two. `6ND` already assumes "multiply + add = 2 FLOPs." If you derive `3ND` you have made this error.

**Q62. Why do Kaplan et al. (2020) and Chinchilla (2022) disagree?**

- **Answer:** Kaplan concluded that model size should grow much faster than data: `N_opt ∝ C^0.73`, `D_opt ∝ C^0.27` — train big, stop early. Chinchilla showed this was an artefact of two experimental choices: Kaplan held the learning-rate schedule fixed across model sizes (so larger models, which need *more* steps to converge, were penalised) and used a fixed cosine schedule whose decay horizon did not scale with the token budget. Fixing both produced near-equal exponents: `N_opt ∝ C^0.5`, `D_opt ∝ C^0.5`, i.e. `D_opt ≈ 20N`. Chinchilla 70B/1.4T matched Gopher 280B/300B at equal compute, a 4x parameter efficiency gain — which is why GPT-3's 175B/300B is now understood as severely undertrained (~1.7 tokens/param against an optimal 20).
- **Why asked:** Tests whether the candidate understands that a scaling law is only as good as its experimental controls.
- **Trap:** Presenting Kaplan as simply "wrong." Its compute-optimal *frontier* shape was right; the allocation between N and D was off.

**Q63. Explain the "emergence is a mirage" argument.**

- **Answer:** Wei et al. (2022) showed abilities appearing abruptly at scale — near-zero accuracy, then a jump — and argued for qualitative phase changes. Schaeffer et al. (2023) showed that the pattern is largely an artefact of **discontinuous metrics**: exact match and accuracy are step functions, so a smoothly improving model crosses the threshold and looks like a jump. When the same models are scored with a continuous metric (token-level edit distance, log-probability of the correct answer), the curves are smooth and predictable. The practitioner's position: treat "emergence" as a hypothesis, never as a planning assumption. Do not budget for a capability to appear at a parameter count — measure with a continuous metric and accept that some capabilities may never arrive.
- **Why asked:** Distinguishes candidates who read papers from those who read paper *titles*. The follow-up is usually "so how do you evaluate?"
- **Trap:** Refusing to take a side. The correct answer is nuanced but decisive: the discontinuity is mostly metric-induced, while some qualitative transitions (chain-of-thought following) remain genuinely sharp.

**Q64. Masked LM only supervises 15% of tokens yet BERT is compute-efficient. Why?**

- **Answer:** Two reasons. (1) **Bidirectional context**: each prediction uses left and right context, so a single prediction encodes far more constraint than a causal one — the model is solving a harder problem per token, which yields more information per gradient step. (2) **Uniform difficulty**: masking is random, so every position is a target over time, whereas in causal LM early positions have almost no context and contribute little signal. Counterweights: MLM cannot generate, its pretrain/finetune objective mismatch (the `[MASK]` token never appears at fine-tune time) requires the 80/10/10 trick, and its per-token supervision is 15% versus 100%. Modern practice: causal LM for anything generative; MLM (or a contrastive objective) for encoders used in classification, NER or embeddings.
- **Why asked:** Tests whether the candidate treats "causal won" as a universal law.
- **Trap:** "MLM is worse because it uses less data." Per-token supervision rate and per-token information content are different quantities.

**Q65. What exactly does AdamW store, and what does that cost in memory?**

- **Answer:** Per parameter: the first moment `m` (exponential moving average of the gradient, fp32 — 4 bytes) and the second moment `v` (EMA of the squared gradient, fp32 — 4 bytes). Plus, with mixed precision, an fp32 master copy of the weights (4 bytes) because bf16 updates vanish (`1e-3 + 1e-8` rounds to `1e-3`). Plus bf16 weights (2) and bf16 gradients (2). Total ≈ **16 bytes per parameter** for full fine-tuning. Updates: `m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)g²`, bias-correct `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)`, then `θ ← θ − η·m̂/(√v̂ + ε)`. LoRA only stores `m`/`v` for the adapter parameters — for `r=16` on a 7B that is ~40M params × 16 bytes = **0.6 GB**, versus 112 GB. That ratio is the entire reason LoRA exists.
- **Why asked:** The memory arithmetic is the practical core of PEFT.
- **Trap:** Forgetting the fp32 master copy, getting 12 bytes/param instead of 16, and under-provisioning by 25%.

**Q66. LoRA reduces memory but not FLOPs. Explain the asymmetry.**

- **Answer:** LoRA adds a small `BA` path *in parallel* to the frozen `W`, so the forward pass is `y = Wx + (α/r)·B(Ax)`. The `Wx` term still runs at full size — you cannot skip it, because `W` is frozen but not removed. So training FLOPs remain ≈`6ND` (the backward pass no longer computes `∂L/∂W` for the frozen weights, saving roughly one-third of the backward, but that saving is offset by the adapter's own forward/backward and is small in absolute terms). Memory falls because there are no `m`/`v`/gradient/master buffers for `N` parameters. **At inference, however, LoRA is a genuine FLOP reduction**: once merged, `W' = W + (α/r)BA` is a single matrix of the original size, so a merged adapter costs exactly the same as the base. That is why people incorrectly believe LoRA speeds up training. It speeds up *serving a smaller model*, not the training step.
- **Why asked:** Directly tests whether the candidate has read the paper or only the blog posts.
- **Trap:** Claiming "LoRA trains 3x faster." It reduces memory ~3–5x and wall clock maybe 1.1–1.3x from kernel effects.

**Q67. Explain the arithmetic-intensity argument for why decode is bandwidth-bound.**

- **Answer:** Arithmetic intensity is FLOPs per byte of memory traffic. Decode generates one token at a time, so every weight must be read from HBM for a single token's worth of arithmetic: intensity ≈ `2N / 2N bytes = 1 FLOP/byte` at batch size 1 in fp16. A GPU's **ridge point** — peak FLOPs ÷ memory bandwidth — is A100-80GB: `312e12 / 2.0e12 = 156 FLOP/byte`; H100 SXM: `989e12 / 3.35e12 = 295 FLOP/byte`. Since 1 ≪ 156, decode sits far to the left of the ridge point, in the bandwidth-bound region. **Practical consequence:** throughput scales with batch size (which raises intensity) until you hit the ridge point or the KV-cache limit, and time-per-token at batch 1 is approximately `weight_bytes / bandwidth`. For 7B in fp16: `14 GB / 2,000 GB/s ≈ 7 ms/token` ≈ 143 tok/s theoretical ceiling on an A100 — and real engines hit 60–120.
- **Why asked:** It is the reasoning behind nearly every serving optimisation (batching, GQA, quantization, speculative decoding).
- **Trap:** Trying to fix throughput by adding FLOPs (a faster GPU with the same bandwidth). Buy bandwidth instead — or quantize, which reduces weight bytes directly.

**Q68. Why is bf16 the default for training rather than fp16?**

- **Answer:** bf16 has 1 sign + 8 exponent + 7 mantissa bits; fp16 has 1 + 5 + 10. The 8 exponent bits give bf16 the same dynamic range as fp32 (`~1e-38` to `~3e38`), so gradients, loss values and attention logits do not overflow. fp16's 5 exponent bits overflow above 65,504 and underflow below ~6e-5, which requires **loss scaling** (multiplying the loss by e.g. 4096, then unscaling gradients) and still produces NaN in long-context attention or deep models. bf16's 7 mantissa bits are enough because stochastic gradient descent is tolerant of per-element rounding error. The corollary: on pre-Ampere hardware without native bf16 (V100, T4), you must use fp16 with a scaler.
- **Why asked:** Every "my loss went NaN" incident traces here.
- **Trap:** "bf16 is lower precision so it's less accurate." For training, numerical *range* matters far more than mantissa depth.

**Q69. What is the theoretical justification for LoRA?**

- **Answer:** The **low intrinsic dimensionality** hypothesis (Aghajanyan et al., 2020): the update needed to adapt a pretrained model to a downstream task has low intrinsic rank — it lives in a subspace of dimension far below `d`. LoRA operationalises this by parameterising `ΔW = BA` with `r ≪ min(k, d)`. The empirical validation: `r=1` or `2` is sufficient for many style/format adaptations, `r=8–64` for most others, and quality is surprisingly insensitive to `r` in that band. **Critique to know:** the hypothesis is about how *adaptation* updates behave, not a theorem, and recent work (e.g. LoRA vs full-FT studies at 1M+ examples) shows the low-rank constraint does become binding at scale — the gap widens. Higher rank helps more when the data is large and diverse.
- **Why asked:** A theory question that reveals whether the candidate knows the limits of the tool.
- **Trap:** Treating `r` as a free lunch. `r=256` on a 7B is 1.7B trainable parameters — at that point you have most of the cost of full FT with none of the simplicity.

**Q70. How does gradient checkpointing change the memory equation, and what does it cost exactly?**

- **Answer:** Without it, a transformer must retain every intermediate activation for the backward pass. For a Llama-7B-shaped block (`h=4096`, 32 heads, `d_head=128`, SwiGLU `d_ff=11008`) at `b=1, s=2048` in bf16, the retained per-block activations total roughly 822 MB with materialised attention matrices; over 32 layers that is ~26 GB. With FlashAttention (no materialised `s²` matrix) it is ~286 MB/block, ~9 GB for the model. Gradient checkpointing stores only each block's **input** (~16 MB) and recomputes the block's internals during backward, dropping per-block cost to well under 1 GB total at a cost of **one extra forward pass ≈ +33% of forward compute, ≈ +25% of total step time**. Since the backward pass is 2x the forward, total FLOPs go from `6ND` to about `8ND` — a 33% slowdown for a 5–10x memory reduction. It is the correct default for any model above ~1B on a single card.
- **Why asked:** It is the classic trade and the numbers are frequently misquoted as "20%".
- **Trap:** Calling it a speed optimisation. If someone enables checkpointing expecting a speedup, they will be disappointed and may misattribute the slowdown to something else.

**Q71. DPO vs RLHF (PPO) — what is the mechanical difference?**

- **Answer:** PPO is a three-model, online RL loop: a policy generates completions, a separately-trained reward model scores them, a value/critic model estimates the baseline, and the policy is updated with a clipped surrogate objective plus a KL penalty to a reference model. Four models in memory, unstable, sensitive to reward hacking. **DPO** observes that the RLHF objective has a closed-form optimal policy, which lets you substitute it into the preference likelihood and obtain a loss purely over the policy and a frozen reference: `L_DPO = −log σ(β·[log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x))])`. No reward model, no critic, no sampling loop — it is a supervised classification problem over preference pairs. Consequence: DPO is far simpler and cheaper; PPO can exceed it when you have a good reward model and a lot of compute, and online methods (GRPO) are now preferred for reasoning tasks where the reward is verifiable.
- **Why asked:** The alignment section of every senior interview (CS-14 §4.6.1–4.6.10).
- **Trap:** "DPO is just RLHF without the RL." The KL constraint is still there — it is baked into the `π_ref` term, and `β` is its strength. Setting `β` badly is the main DPO failure mode.

**Q72. Why does fine-tuning teach form rather than facts?**

- **Answer:** Because the objective is next-token prediction *over your dataset's distribution*. Gradient descent minimises the average loss across your examples, so it prioritises patterns that appear consistently — formatting, tone, schema, refusal phrasing, vocabulary. A fact that appears once in 5,000 examples contributes ~1/5000 of the gradient signal and is indistinguishable from noise. The base model's facts are stored in a distributed, superpositional way across the MLP weights; SFT does not write new facts into that store, it reshapes the output distribution. Practically: the model becomes *fluent* in your domain's format and register and *confident* in its domain's vocabulary, without becoming more accurate. That is exactly why the canonical failed project is "fine-tune on our docs to fix hallucination."
- **Why asked:** The conceptual heart of CS-04's architecture decision.
- **Trap:** Pointing to a model that memorised a fact from fine-tuning. At `r=128` and 20 epochs on 200 examples, memorisation is possible — and it is a bug, not a capability. It does not survive contact with a differently-worded question.

**Q73. How does tokenizer vocabulary size affect multilingual cost?**

- **Answer:** A tokenizer trained mostly on English allocates few merge operations to other scripts, so non-Latin text is split into many more tokens. Llama-2's 32k SentencePiece tokenizer consumes roughly 4–10x more tokens per word for Hindi, Thai, Burmese or Amharic than for English; Qwen2's 151k and Gemma's 256k vocabularies were sized partly to fix this. The consequences are multiplicative: context window consumption, training tokens (and therefore 6ND cost), inference cost, and effective context available for the actual task. **Practical rule:** before committing to a base model for a non-English task, tokenize 10,000 words of your target language and compute `tokens/word`. If it exceeds ~2.5, budget for a different base or a tokenizer extension — and if you extend the tokenizer, you must resize the embedding matrix and train the new rows (CS-12).
- **Why asked:** A concrete, quantifiable skill that most candidates have never exercised.
- **Trap:** Assuming "it's multilingual" from the model card. Test it with your own data.

**Q74. Explain GQA and why it exists.**

- **Answer:** Grouped-query attention uses fewer key/value heads than query heads (Llama-2-70B: 64 query heads, 8 KV heads; Llama-3-8B: 32 and 8). The KV cache size is `2 × L × n_kv_heads × d_head × bytes_per_element` per sequence. For Llama-2-70B at 4,096 context in fp16 with MHA (64 KV heads): `2 × 80 × 64 × 128 × 2 = 2.62 MB/token`, so 4,096 tokens is 10.7 GB *per sequence* — impossible to batch. With GQA at 8 KV heads it is 1.34 GB, an 8x reduction. Multi-query attention (1 KV head) goes further but degrades quality at scale; GQA is the compromise that nearly every modern model uses. **Consequence for fine-tuning:** if you train with MHA and serve with a GQA conversion (or vice versa) the checkpoints are not interchangeable — the `k_proj`/`v_proj` shapes differ.
- **Why asked:** It is the reason long-context serving is affordable at all (CS-10, CS-16).
- **Trap:** Attaching the KV cache to the model rather than to the sequence batch. It grows with concurrent requests, not just with model size.

**Q75. What is weight decay actually doing in AdamW?**

- **Answer:** It is L2 regularisation decoupled from the adaptive gradient: `θ ← θ − η·(m̂/(√v̂+ε)) − η·λ·θ`. In classic Adam, L2 was added to the gradient *before* the second-moment normalisation, so parameters with large gradients got a smaller effective decay — the regularisation was entangled with the adaptive scaling. `AdamW` applies decay directly to the weights, so every parameter shrinks by the same relative amount per step regardless of its gradient history. Typical `λ = 0.01–0.1`, and the standard practice is to **exclude** bias terms and LayerNorm/RMSNorm weights from decay (they are not scale-free). For LLM fine-tuning, `0.0–0.01` is common; the LoRA literature frequently uses `0.0` or `0.01`.
- **Why asked:** A precise question that separates people who use `AdamW` as a magic string from people who know why it replaced `Adam`.
- **Trap:** "Weight decay prevents overfitting, so more is better." At `λ=0.1` with `lr=2e-4` on a small SFT set you are shrinking the adapter to zero faster than it learns.

**Q76. Why does a low learning rate not fully prevent catastrophic forgetting?**

- **Answer:** Because all parameters move toward the narrow objective with no term that preserves prior behaviour. A small `η` slows the drift but does not change its direction — given enough steps, the model still converges to the narrow distribution. What actually helps, in rough order of effectiveness: (1) **freeze the base** (LoRA/adapters) — the prior capability lives in weights that never move; (2) **mix general instruction data** (5–20%) into the SFT set so the objective itself contains the behaviour you want to preserve; (3) **early stopping on a regression suite**, which makes forgetting observable; (4) **regularisation toward the base** — KL penalties to a reference model, as in RLHF/DPO, or weight-space interpolation/merging. Note that (1) is not a complete guarantee either: LoRA can still shift the model's *style* enough to degrade refusal behaviour, because the adapter's output is added to every forward pass.
- **Why asked:** The mitigation list is a practical checklist, and most candidates only know "lower the LR."
- **Trap:** Believing LoRA makes forgetting impossible. It makes it much less likely; measure it anyway.

**Q77. How do scaling laws interact with fine-tuning?**

- **Answer:** Three distinct interactions. (1) **Base quality flows through**: a better pretrained base raises the ceiling for every downstream fine-tune — fine-tuning cannot exceed the capability of the base on a task the base has never seen. (2) **Fine-tuning data scaling is far weaker**: SFT gains saturate quickly and then reverse (overfitting), which is why LIMA's 1,000 examples is competitive with 50,000 — the curve is steep early and then flat-to-negative, unlike pretraining's clean power law. (3) **Chinchilla does not apply**: there is no `20 × N` rule for SFT; the right units are *diverse behaviours covered*, not tokens processed. And the overtraining correction applies to *continued pretraining*, not to SFT — if you are doing domain-adaptive continued pretraining (CS-12), Chinchilla-style reasoning about `N` vs `D` is relevant again, and the inference-cost argument for overtraining applies with full force.
- **Why asked:** Tests whether the candidate mechanically transfers pretraining intuitions to SFT. They do not transfer.
- **Trap:** "Double the SFT data and get double the improvement." You will more likely get overfitting and a worse model.

**Q78. Quantization-aware training vs post-training quantization — what changes during training?**

- **Answer:** PTQ (GPTQ, AWQ, GGUF k-quants) quantizes an already-trained model and calibrates scales on a small sample; it never updates weights. QAT inserts **fake-quantization** nodes into the forward pass during training — weights are rounded to the target grid and dequantized on the fly — so the model learns to place its weights where the quantization error is smallest, and the straight-through estimator passes gradients through the rounding. Costs: full training compute, access to the training data, and a modified graph. Benefits: 1–3 points over PTQ at the same bit width, and viable 2–3 bit models where PTQ collapses. The relevant middle ground for fine-tuners is **QLoRA**: the base is quantized (PTQ) but the adapters are trained in bf16, which is neither PTQ nor QAT but captures most of QAT's benefit for a fraction of the cost.
- **Why asked:** CS-10/CS-11 territory, but the vocabulary must be set here.
- **Trap:** Confusing QLoRA with QAT. QLoRA does not change how the frozen weights are represented during the update — it never updates them.

**Q79. Explain the KV cache and its scaling.**

- **Answer:** During autoregressive decode, each new token attends to all previous keys and values. Recomputing them every step would make decode `O(s²)` per token in a way that repeats work, so implementations store the K and V tensors: size per sequence = `2 × L × n_kv_heads × d_head × bytes`. It converts decode from quadratic-recompute to linear-memory. Scaling consequences: it grows linearly with context and with concurrent requests, and it is usually the term that limits batch size on a long-context server — not the weights. For Llama-3-8B (GQA, 8 KV heads, 32 layers, `d_head=128`) at 8,192 tokens in fp16: `2 × 32 × 8 × 128 × 2 × 8192 = 1.07 GB` per sequence — so a 4-sequence batch is 4.3 GB before any weights.
- **Why asked:** Every serving capacity plan depends on this, and it is invisible in the model's parameter count.
- **Trap:** Sizing a server from weights alone. At long context, the KV cache dominates.

**Q80. What is the ceiling on any fine-tune?**

- **Answer:** Two ceilings. (1) **The base's capability ceiling** — fine-tuning can elicit and shape behaviour the base is capable of; it cannot install a capability the base lacks. A 1B model will not learn to do multi-step arithmetic through SFT. (2) **The data ceiling** — a model trained on your data can only be as good as the *best* examples in it. If your gold answers are 70% correct, your model converges toward 70%, and it will confidently produce that 70% answer on the 30% where it is wrong. This is why data quality beats data quantity at every scale, and why the LIMA result (1,000 curated examples) is the most important data result in the field. Corollary: before adding data, audit the top of your distribution — the model's behaviour tracks your best examples, not your average.
- **Why asked:** It is the honest answer to "can we make it better?" and it redirects the conversation to data.
- **Trap:** "More compute will fix it." Beyond 2–3 epochs on SFT data, more compute reliably hurts.

**Q81. What does the attention FLOPs term do to the 6ND estimate at long context?**

- **Answer:** Attention costs `12 × L × s² × h` FLOPs per sequence (2 matmuls for `QKᵀ`, 2 for `AV`, each `2·s²·d_head·n_heads`, plus projections) versus `12 × L × s × h²` for the linear parts — so the attention term overtakes the linear term when `s > h`. For a 7B (`h=4096`, `L=32`), the crossover is at ~4,096 tokens. At `s=128k` the attention term dominates completely: `12 × 32 × (1.28e5)² × 4096 ≈ 2.6e15` FLOPs per sequence versus `12 × 32 × 1.28e5 × 4096² ≈ 8.2e12` for the linear terms — 300x. This is why long-context *training* is disproportionately expensive and why FlashAttention (which reduces memory, not FLOPs) plus sequence parallelism exist. It is also why `6ND` is an excellent estimate at 2k context and a poor one at 128k.
- **Why asked:** Tests whether the candidate applies `6ND` outside its validity range.
- **Trap:** Quoting `6ND` for a 128k-context run and being off by two orders of magnitude.

**Q82. A base model has better perplexity than your fine-tune. Is that a problem?**

- **Answer:** Usually not — and this is a case where the number is misleading. Perplexity is measured on a held-out corpus of *general* text; SFT deliberately narrows the model's distribution, so its probability mass is concentrated on your domain's style and it assigns lower probability to out-of-domain text. A fine-tune that improves your task by 30 points while raising general perplexity by 15% is a normal, successful trade. The genuine warning sign is a **large** perplexity increase (say, over 50% on in-domain text) or a rise combined with regression-suite failures — that indicates catastrophic forgetting or a training bug. Always report perplexity **on in-domain held-out text** alongside the task metric and the regression suite; cross-domain perplexity after SFT is not a quality signal.
- **Why asked:** Evaluates whether the candidate can interpret a metric rather than just report it.
- **Trap:** Optimising perplexity after SFT, which pushes toward reproducing the base and undoing the fine-tune.

---

## Level 4 — System Design & Architecture

**Q83. Design a fine-tuning pipeline for a regulated healthcare customer with a strict no-data-egress policy.**

- **Answer:** **Requirements:** PHI must never leave the customer's network; the model must be auditable, versioned and reproducible; the output must be reviewable by a clinician. **Constraints:** no managed APIs (OpenAI, Vertex, Bedrock); the customer has a small on-prem GPU cluster, likely 1–4 × A100-40GB or L40S. **Design:** (1) **Base model**: an open-weights model with a permissive licence — a 7B or 8B class (Llama-3.1-8B or Qwen2.5-7B) or a 3B if it fits better; licence review is a workstream, not a footnote (Llama's 700M-MAU clause, and note that EU AI Act obligations attach to deployers of GPAI-based systems). (2) **Data plane**: all labelling, dedup and training inside the perimeter; de-identify with a PHI scrubber before the data ever reaches the trainer; keep a data-provenance record per example. (3) **Training**: QLoRA on a single GPU first to prove the pipeline, then LoRA if quality demands; 1–3 epochs; a clinician-written eval set of 200–500 real cases held inside the perimeter. (4) **Eval**: task metric + clinician review of a stratified sample + a regression suite + a **refusal/safety suite** (medical advice boundaries are the highest-risk behaviour). (5) **Serving**: vLLM on-prem, adapters versioned by content hash, every response logged with the model+adapter version and the input for audit. (6) **Governance**: signed model cards, a change-control process, and a documented rollback to the previous adapter.
- **Why asked:** Full-stack design under the constraint that eliminates the easiest answers.
- **Trap:** Proposing a managed API or a cloud GPU rental and then discovering the policy forbids it. Also: ignoring the licence and audit requirements, which are usually the real blockers in this sector.

**Q84. Design an eval strategy for a fine-tuned model serving 10M requests/month.**

- **Answer:** Four layers at four cadences. **Pre-release (blocking):** frozen test set of 500–2,000 items with task metric + continuous metric + bootstrap CIs; regression suite of 200 items; safety suite of 150 items; latency and cost profile at p50/p95/p99. A release requires no regression against the incumbent on any layer. **Continuous (per-request, sampled):** log 1–5% of traffic with inputs and outputs; compute online proxies — schema-validity rate, refusal rate, output-length distribution, tool-call success, user-thumbs signals. **Daily:** a fixed "canary" set of 100 items replayed against production to detect silent drift (serving-stack changes, tokenizer updates, quantization re-exports). **Weekly:** LLM-as-judge pairwise against the incumbent on 500 sampled production inputs, with human review of a 50-item calibration subset to keep judge agreement above ~70%. **Quarterly:** refresh the test set from *new* production distributions and re-baseline. Alert on: schema-validity drop >2 points, refusal-rate change >3 points, p95 latency +30%, and any safety-suite failure.
- **Why asked:** Distinguishes "we ran an eval" from "we operate a model."
- **Trap:** Designing only the pre-release layer. At 10M requests/month, drift and distribution shift are the dominant risks, and only continuous measurement catches them.

**Q85. A legal-research assistant: fine-tuning, RAG, or agents?**

- **Answer:** **RAG as the backbone, not fine-tuning** — the requirement is citation-grounded retrieval from a large, changing corpus of statutes and case law. Fine-tuning cannot supply facts, cannot cite, and cannot be updated when the law changes. **Then fine-tune for three specific behaviours** on top: (1) **output format** — always emit claim → citation → quoted span, which is a form problem SFT solves well; (2) **jurisdiction and refusal behaviour** — decline to answer outside the licensed jurisdiction, a behaviour the base model handles erratically; (3) **prompt compression** — replace a 4,000-token few-shot citation-format prompt with a 200-token one, cutting per-request input cost ~20x. **Then an agent layer** for multi-step work: search → read → cross-reference → draft, with tool calls. **Eval:** retrieval recall@k, citation-precision (does the cited span support the claim), and a refusal suite for out-of-scope questions. **Constraint to flag:** legal advice is regulated; the system must present as research assistance with citations, not as advice.
- **Why asked:** The canonical architecture question of the course (CS-04). The interviewer is listening for "all three, for different reasons."
- **Trap:** Picking one. Any single-answer response is wrong, and the interview is over.

**Q86. Design a multi-tenant fine-tuning platform serving 50 customers with per-tenant adapters.**

- **Answer:** **Tiers:** (a) one shared frozen base + 50 LoRA adapters for the long tail; (b) a dedicated base + merged adapter for the 2–3 tenants whose traffic justifies it; (c) QLoRA-trained adapters are merged into a bf16 base and re-quantized for serving as a matter of course. **Serving:** vLLM with `--enable-lora`, `max_loras=8` resident, `max_lora_rank` set to the largest adapter, an LRU cache over adapters, and a warm pool that pre-loads each tenant's adapter on a schedule to kill first-request cold start. **Isolation:** per-tenant data and adapters in separate storage with separate encryption keys; a per-tenant tokenizer check (all tenants must share the base and tokenizer, or they need separate deployments); no cross-tenant data mixing in any training set. **Control plane:** a job queue with per-tenant GPU quotas and priority; training jobs on spot/on-demand workers; adapter registry with content hashes keyed to `(base_model, base_revision, tokenizer_revision, dataset_hash, config_hash)`. **Eval:** per-tenant eval sets, run automatically at job end; a per-tenant quality gate before promotion. **Cost:** the shared base at 14–16 GB plus 8 resident adapters at ~50–200 MB each means one 40–80 GB GPU can serve dozens of tenants; the economics collapse if you merge per tenant.
- **Why asked:** Platform-engineering maturity: caching, quotas, versioning, isolation, cold start.
- **Trap:** Design-by-multiplication (50 GPUs, one per tenant) and missing the adapter-cache/cold-start problem entirely.

**Q87. Fine-tune a 70B for a company on a $10,000 budget. Plan it.**

- **Answer:** **Options at $10k** (A100-80GB at ~$1.79/hr on-demand, ~$1.10/hr reserved): ~5,600 on-demand A100-hours. **Full FT is out** — 70B needs ~1,120 GB of optimiser/weight state, i.e. 16×A100-80GB minimum plus sharding, which puts you at 16 GPUs × 30 h ≈ 480 GPU-hours ≈ $860 *if nothing goes wrong* — and it will, because this is your first run at this scale. **Recommended plan:** (1) **QLoRA on 2–4 × A100-80GB** — a 70B in NF4 is ~35 GB, so 2 cards with FSDP/DeepSpeed ZeRO-3 fit it. At ~1.5 s/step for 5,000 steps that is ~2 h per run. (2) Budget **30 runs** (~$200–800 at 2–4 GPUs) for the iteration loop — data curation, LR, rank, epochs. (3) Reserve **$3,000–5,000 for the final LoRA run** at higher rank and more data. (4) Keep **$1,000–2,000 for evaluation and inference** — you must serve the model to evaluate it properly. **The real risk is not compute, it is data:** with $10k you can afford 30 experiments; you cannot afford to discover at run 25 that your eval set was wrong. Build the eval first, and start with a 7B or 8B to debug the pipeline before touching the 70B — the pipeline bugs are identical and 10x cheaper to find.
- **Why asked:** Budget realism and staging. The right answer names the debug-on-a-small-model step explicitly.
- **Trap:** Spending the whole budget on one full-FT run. Also: omitting the eval/serving budget, which is where the money silently goes.

**Q88. Design a pipeline that turns production failures into training data.**

- **Answer:** **Capture:** log inputs, outputs, model+adapter version, latency, and all downstream signals (user edit distance, thumbs, escalation, regenerations, schema-validation failures, tool-call errors) with a stable request id. **Triage:** a daily job clusters failures (embedding + clustering) and ranks clusters by volume × severity; the top clusters become labelling tasks. **Label:** domain experts write the *ideal* response for a sampled 20–50 items per cluster in a labelling UI that enforces the eval rubric. **Curate:** dedup against the existing training set (MinHash), check against the *frozen test set* to prevent leakage, and hold back 10% as a fresh eval slice. **Train:** append to the SFT set or train a delta-adapter on the new cluster only, then evaluate. **Gate:** run the full layer stack from Q84 — task metric must improve, regression and safety suites must not degrade. **Ship:** version, canary at 5% traffic, watch the online proxies, promote or roll back. **Close the loop:** record which cluster each new example came from and re-measure the failure rate for that cluster after release, so you learn whether the fix worked. **Anti-pattern to name:** training on raw production outputs. If the model generated them, they are its own errors — you are distilling its mistakes.
- **Why asked:** This is the maturity differentiator between a team that fine-tunes once and a team that improves.
- **Trap:** Training on logged outputs without human correction. It is the most common and most damaging shortcut in the loop.

**Q89. A new language the base has never seen: how do you design the training plan?**

- **Answer:** **Stage 1 — tokenizer feasibility.** Tokenize 100k words of the language and measure tokens/word. If it is above ~3, the tokenizer is the bottleneck and you have two paths: choose a base with better coverage (Llama-3's 128k, Gemma's 256k, Aya), or extend the vocabulary with SentencePiece on your corpus, resize the embedding matrix, and train the new rows. **Stage 2 — continued pretraining (CS-12).** SFT alone cannot install a language the base has no representation for. Run domain/linguistic continued pretraining on 1B–50B tokens of in-language text at `lr=1e-5` to `5e-5` — this is where the base's representations are reshaped toward the language, and it is the expensive part. Monitor held-out in-language perplexity. **Stage 3 — SFT.** 5k–50k **natively authored or professionally translated** instruction pairs. Machine-translated English instruction data produces a model with English syntax patterns and poor idiomaticity — this is measurable and it is the most common mistake. **Stage 4 — evaluation.** Build an in-language eval set; translated benchmarks are contaminated and their cultural assumptions fail. Also build a regression suite in the original language to detect forgetting. **Cost shape:** stage 2 dominates (a 7B on 10B tokens ≈ 4.2e19 FLOPs ≈ 1,300 A100-hours ≈ $2,300 at 40% MFU), stages 1/3/4 are rounding errors by comparison.
- **Why asked:** Tests the correct decomposition — most candidates jump to SFT and stop.
- **Trap:** "Translate 5,000 instructions and fine-tune." The base needs the language first, or you get a model that responds in the right script with the wrong grammar.

**Q90. Design the rollback and monitoring plan for a fine-tuned model in production.**

- **Answer:** **Versioning:** every artifact identified by `(base_model, base_revision, tokenizer_revision, adapter_or_merged_hash, dataset_hash, training_config_hash, git_sha, eval_report_id)`. Store the training config and dataset snapshot with the adapter; without them the model is not reproducible and not auditable. **Deployment:** blue/green at the inference service, with the ability to route a percentage of traffic to the candidate (`canary`), and the previous version kept warm so rollback is a routing change, not a cold start. **Rollback triggers, defined numerically in advance:** task metric on the canary set drops >2 points; schema-validity drops >2 points; refusal rate moves >3 points; p95 latency +30%; safety-suite failure; hallucination rate (human-audited sample) rises >5 points. **Monitoring:** per-request structured logs; sampled output storage with PII redaction and a defined retention window; dashboards for the online proxies plus drift on input length, language mix, and topic distribution. **Testing:** the daily canary-set replay (catches serving-stack drift), the weekly judge comparison, and the quarterly re-baseline. **Incident path:** automatic rollback on a hard trigger (safety or schema), human decision on soft triggers, and a post-incident write-up that feeds a new example cluster into the Q88 loop.
- **Why asked:** Production reality is the part of the bar most candidates never reach.
- **Trap:** Monitoring only infrastructure metrics (GPU utilisation, latency, error rate). Those will all be green while the model outputs garbage.

---

## Level 5 — Debugging & Incident Response

Answer these as a **sequence of checks**, cheapest and most likely first, with the discriminating observation at each step.

**Q91. Your training loss is stuck at 10.4 and never moves, on a 32k-vocabulary model.**

- **Answer:** `ln(32000) = 10.37` is the uniform-distribution loss — the model is outputting a flat distribution over the vocabulary. **Sequence:** (1) **Check that the labels are not all `-100`.** If the prompt is fully masked and the mask was applied to the wrong span, there is nothing to learn from, and the model stays at the uniform prior. Decode `labels[labels != -100]` and confirm it is your assistant response. (2) **Check `labels` are actually being passed.** A collator that drops the `labels` key makes the Trainer fall back to causal-LM-on-the-whole-batch or, worse, silently train nothing. (3) **Check the loss is computed over the response, not the prompt**, by computing it by hand for one batch. (4) **Check the LR.** `lr` in the config that never reaches the optimiser (wrong group, `0.0`, or overwritten by a CLI flag) gives exactly a flat line. Print `optimizer.param_groups[0]["lr"]` at step 1. (5) **Check for a frozen model.** `model.requires_grad_(False)` from a feature-extraction setup, or a PEFT config with no `target_modules` matching any layer name — PEFT will train nothing and log a warning most people scroll past. Verify with `sum(p.numel() for p in model.parameters() if p.requires_grad)`; it must be >0. (6) Only then suspect the data.
- **Why asked:** A flat line at `ln(V)` has exactly one meaning, and the checks are cheap.
- **Trap:** Restarting the run or lowering the learning rate. Neither fixes a missing label.

**Q92. Training loss falls to 0.05; eval loss rises to 2.8. What is happening, and what do you do?**

- **Answer:** Textbook overfitting — the model has memorised the training set and generalises worse than before training started. **Sequence:** (1) **Check the epoch count and the effective dataset size.** `0.05` train loss on 800 examples at epoch 3 means each example was seen many times at `lr=2e-4`. (2) **Check for duplicate leakage between train and eval** — run MinHash on the two splits; if the eval set is 40% near-duplicates of train, the "overfitting" is really leakage and both numbers are meaningless. (3) **Check the train/eval loss gap on the *first* epoch.** If they diverge from epoch 1, the dataset is too small or too narrow for the learning rate. **Fixes, in order:** stop at the epoch where eval loss is minimal (use `load_best_model_at_end=True`, `eval_strategy="steps"`, `metric_for_best_model="eval_loss"`); reduce epochs to 1; halve the LR; reduce `r`; add 5–20% general instruction data; and — the real fix — **collect more diverse examples**, because with 800 examples no hyperparameter will save you. If eval loss is flat-to-rising from step 100, `early_stopping_patience=2` would have caught it.
- **Why asked:** The most common real failure in the field, and the interviewer wants the leakage check, which most candidates skip.
- **Trap:** Adding dropout and calling it fixed. It helps marginally; it does not substitute for data.

**Q93. Loss is NaN at step 0.**

- **Answer:** **Sequence:** (1) **dtype** — if `fp16=True` without a loss scaler, or the model was loaded in fp16 and trained in fp16, overflow is the likely cause. Switch to `bf16=True`. (2) **Learning rate too high** — `2e-4` on a full fine-tune (as opposed to LoRA) diverges immediately; check whether the LR is an order of magnitude above the intended one. (3) **Bad data** — a `NaN` or `inf` in the input tensors, or token ids outside `[0, vocab_size)`. Assert `input_ids.max() < model.config.vocab_size` and `input_ids.min() >= 0` on the first batch. (4) **Empty sequences** — an all-masked or zero-length sample produces a `0/0` in the loss mean; filter `len(ids) == 0`. (5) **`attention_mask` all zeros** for a row → softmax over an empty set → NaN. This is the classic one for padded batches built by hand. (6) **Numerically unstable attention** on very long sequences without FlashAttention/SDPA. **Instrumentation:** register a forward hook that `torch.isnan(output).any()`s layer by layer; the first NaN layer localises the bug in one run.
- **Why asked:** A hard, concrete failure with a specific ordered diagnosis; the answer quality is very visible.
- **Trap:** Restarting and hoping. NaN at step 0 is deterministic.

**Q94. The model works perfectly in your notebook and produces garbage through the batch inference endpoint.**

- **Answer:** The classic train/serve skew. **Sequence:** (1) **Chat template.** The notebook applies `apply_chat_template`; the endpoint probably concatenates strings or uses a different template (or the serving engine's default, which may differ from the model's). Diff the exact strings. (2) **System prompt absence** — the most common version: the notebook sends a system turn, the endpoint does not, and the fine-tune learned a behaviour conditioned on a role it never receives. (3) **Padding side.** Decoder-only batched generation requires **left** padding; if the endpoint right-pads, the first generated token for every row except the longest is conditioned on pad tokens, and outputs are truncated or garbled. This one is invisible for batch size 1. (4) **Stop tokens.** If the serving engine's `eos_token_id` does not include the template's end-of-turn token, generation runs on and the response includes hallucinated turns. (5) **Tokenizer revision drift** — the endpoint pulled a different tokenizer revision than the one used in training. (6) **Quantization mismatch** — serving a merged-bf16 model that was trained and evaluated as QLoRA. **Fix:** make the serving path call the same `apply_chat_template` and the same tokenizer artifact, pinned by revision, that the training script used; add a golden-request test that runs in CI against the live endpoint and compares to a recorded notebook output.
- **Why asked:** Every team hits this, and the fix is architectural, not a patch.
- **Trap:** Debugging it by tweaking sampling parameters. Temperature does not fix a template mismatch.

**Q95. Your fine-tuned model repeats the user's question before answering, and sometimes keeps going until `max_new_tokens`.**

- **Answer:** Classic **chat-template mismatch**, specifically: the training data had no assistant-turn prefix (`<|start_header_id|>assistant<|end_header_id|>`) or no EOS after the response. The model learned "after the user's question comes a continuation of the conversation," so at inference it generates the next user turn and continues. **Sequence:** (1) Render the training string for one example with `apply_chat_template(..., tokenize=False)` and inspect it. The assistant's turn must be delimited by the model's own header token and terminated by its EOS (`<|eot_id|>` for Llama-3, `</s>` for Mistral, `<|im_end|>` for Qwen). (2) Confirm the EOS **survives tokenization** — a truncation step or a collator that strips trailing special tokens removes the very supervision that teaches stopping. (3) Check `max_seq_length` — if truncation cut the EOS, only some examples taught stopping. (4) Check the serving stop strings include the template's end-of-turn token. **The fix is data, not decoding:** rebuild with the template applied properly, mask the prompt, and ensure the EOS is present and unmasked. A repetition penalty is a band-aid and degrades quality.
- **Why asked:** The signature failure of the most common silent SFT bug. It tests whether the candidate recognises it from the symptom.
- **Trap:** Adding `repetition_penalty=1.2` and shipping. It masks a training-data bug and costs quality on legitimate repetitions (JSON, lists, code).

**Q96. After fine-tuning, the model is worse at everything — including the task you trained it on.**

- **Answer:** **Sequence:** (1) **Check the LR.** `2e-4` with full fine-tuning destroys the pretrained weights; this is the single most common cause and it produces exactly "worse at everything." (2) **Check the model actually loaded.** A silent failure to load the base adapter (wrong path, wrong class) can leave you evaluating a randomly-initialised head. (3) **Check the eval pipeline did not change with the model** — a new tokenizer, a different template, or a changed prompt in the harness makes the *comparison* invalid rather than the model worse. Re-run the previous model through the new harness first. (4) **Check for catastrophic forgetting** on the regression suite — if only the regression suite degraded and the task metric improved, it is forgetting, not a bug. (5) **Check the data labels.** If an annotation pipeline inverted or shifted labels, the model learned the inverted mapping and is confidently wrong — this looks like "worse at everything" when it is really "consistently wrong in a new way." (6) **Check the base checkpoint.** Fine-tuning a *base* model when you meant to fine-tune an *instruct* model produces a model that is worse at instruction following than the instruct baseline you compared against.
- **Why asked:** A discriminating sequence, and (6) — the base/instruct mix-up — is the one the interviewer is fishing for.
- **Trap:** Immediately blaming the data and rebuilding the dataset. The LR check takes 30 seconds.

**Q97. Benchmark improved 12 points but users say the product got worse.**

- **Answer:** **Sequence:** (1) **Distribution mismatch** — the benchmark was built from the same distribution as the training data (or from public data), so it measures what you trained on, not what users send. Sample 200 real production requests and evaluate on those. (2) **The metric is a proxy that does not track the user's outcome.** A schema-validity improvement can come with a hallucination increase; an exact-match improvement can come with an unhelpful brevity. Build a metric tied to the actual downstream outcome (task completion, escalation rate, edit distance). (3) **Latency or length regression** — a fine-tune that produces 3x longer outputs scores better on a recall-like metric and is worse to use; check output-length distribution before and after. (4) **Regression on an unmeasured capability** — the benchmark did not test reasoning, multilingual, long-context or refusals, so a real degradation there is invisible. Run the regression suite. (5) **Contamination in the benchmark** — if the eval set overlaps the training set, the 12 points are memorisation. Run MinHash across the two. (6) **The change shipped with something else** (prompt, retrieval, temperature) and the model is being blamed by association. **The framing to give the interviewer:** an offline metric is a hypothesis about user value; only the online metric tests it.
- **Why asked:** The most senior-feeling failure mode; it tests judgement about measurement itself.
- **Trap:** Reverting the model without diagnosing. You lose the improvement and learn nothing.

**Q98. OOM at a batch size that fit yesterday.**

- **Answer:** **Sequence:** (1) **Sequence-length distribution changed.** Activation memory scales with `batch × seq` (linear parts) and `batch × seq²` (attention without FlashAttention), so a data reshuffle that introduced a few long examples can raise peak memory far more than the average suggests. Check the p99 length in the current shard versus yesterday's. (2) **You raised `max_seq_length` or disabled packing** — both change the memory profile globally. (3) **Gradient checkpointing got disabled** by a config flag, or the model was switched to `attn_implementation="eager"` (materialised attention matrices) instead of `"flash_attention_2"`/`"sdpa"` — the latter can be a 5–10x activation-memory difference at long context. (4) **Something else is on the GPU** — a stale process, another job, or a leftover evaluation loop holding memory. Check `nvidia-smi` and `torch.cuda.memory_summary()`. (5) **Fragmentation** rather than a genuine leak — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` frequently resolves "it fit yesterday." (6) **A code change that keeps a reference** — an accumulating list of losses or logits on the GPU, or `output_hidden_states=True` left on. **Mitigations:** `gradient_checkpointing_enable()`, `per_device_train_batch_size=1`, raise `gradient_accumulation_steps` to keep the global batch constant, and set `max_seq_length` from your p99 rather than a round number.
- **Why asked:** OOM debugging is daily work, and the sequence matters more than the list.
- **Trap:** Blaming a memory leak in the framework. It is almost always the data or a config change.

**Q99. Your fine-tune will not reproduce — same config, different results.**

- **Answer:** Sources of nondeterminism, in order of impact: (1) **Data ordering.** `dataloader_num_workers > 0` with no fixed seed means the shard order varies; `DataLoader(shuffle=True)` without seeding `torch`, `numpy`, `random` and the sampler. Set `seed=42`, `data_seed=42`, `dataloader_num_workers=0` for reproducibility runs (and accept the throughput hit). (2) **Nondeterministic kernels** — `torch.backends.cuda.matmul.allow_tf32`, flash-attention backward, and atomics in scatter/gather are nondeterministic by design. `torch.use_deterministic_algorithms(True)` helps but costs 10–30% and some ops have no deterministic implementation at all. (3) **Hardware and topology** — different GPU counts change the reduction order and therefore the numerics; results on 1×A100 and 4×A100 will differ. (4) **Different library versions.** `transformers`, `peft`, `trl`, `flash-attn` and CUDA versions change numerics and defaults between releases. Pin *exact* versions and record them. (5) **Data drift** — the dataset was pulled from the Hub at `main` rather than a revision; a dataset update between runs silently changes the training set. Pin `revision=<commit>` for both model and dataset. **Realistic standard:** bit-exact reproduction is not achievable across hardware; aim for *statistically equivalent*, and define that as "eval metric within ±1 point across three seeds." Report the variance — a single run's metric without a seed range is not a result.
- **Why asked:** Reproducibility is an engineering discipline, and most candidates have never run the same experiment three times.
- **Trap:** Promising bit-exact reproduction on GPUs. It is not achievable, and claiming it reveals inexperience.

**Q100. You have a fixed compute budget, mixed-quality data, and one week. What do you do?**

- **Answer:** **Days 1–2 — measurement before modelling.** Build 200 hand-labelled eval items from real inputs (stratified across the input space), plus a 200-item regression suite and a 150-item safety suite. Run the well-prompted base model through all three and record the baseline. This is non-negotiable: without it, the remaining five days produce numbers you cannot interpret. **Day 3 — data audit.** Dedup (MinHash), remove the bottom 10% by a quality heuristic, split by source, and hand-fix 100 examples of the most common failure pattern. Expect this to be the single highest-ROI day. **Day 4 — train.** QLoRA, `r=16, alpha=32`, all linear targets, `lr=1e-4`, 1 epoch, effective batch ~128k tokens, gradient checkpointing on, a 5% warmup, `eval_strategy="steps"`, `load_best_model_at_end`. Run three seeds only if time allows; otherwise one well-instrumented run. **Day 5 — evaluate and decide.** Task metric with CIs, continuous metric, regression suite, safety suite. Compare against the prompt-only baseline. **Days 6–7 — iterate on the *data*, not the hyperparameters.** Take the top 5 failure clusters from the eval, hand-write 20 examples each, retrain. **The discipline:** resist the urge to sweep hyperparameters. On a one-week budget with real data, a 10-point LR sweep is worth less than 200 targeted examples, and the baseline is what tells you whether you succeeded.
- **Why asked:** It compresses the whole module into one decision. The interviewer is checking whether you spend the budget on measurement and data, or on hyperparameters.
- **Trap:** Spending days 1–3 setting up a sweep. Without a baseline and an eval set, the best hyperparameters are unidentifiable.

---

## Rapid Fire — True or False

Answer in under five seconds each. The explanation is one line; if you need more, revisit CS-01.

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | Fine-tuning can teach a model new facts reliably. | **False** | It teaches form; facts need retrieval or pretraining-scale exposure (CS-04). |
| 2 | Causal LM supervises 100% of tokens. | **True** | Every position predicts its successor — vs ~15% for MLM. |
| 3 | Chinchilla says 20 tokens per parameter is optimal for production. | **False** | It is compute-optimal; production models are overtrained 14–94x for inference cost. |
| 4 | `bf16` has a wider dynamic range than `fp16`. | **True** | 8 exponent bits vs 5 — the reason it does not overflow. |
| 5 | LoRA makes the training forward pass cheaper in FLOPs. | **False** | The frozen `Wx` still runs; memory falls, FLOPs do not. |
| 6 | The training loss for a 32k vocabulary model starts at ~10.4. | **True** | `ln(32000) = 10.37` — the uniform prior. |
| 7 | A base model and an instruct model with the same parameter count are interchangeable given the right prompt. | **False** | Different weights, different behaviour, different template. |
| 8 | Gradient checkpointing speeds up training. | **False** | It costs ~+33% FLOPs to save 5–10x activation memory. |
| 9 | Full fine-tuning a 7B needs roughly 112 GB before activations. | **True** | 16 bytes/param × 7e9 = 112 GB. |
| 10 | Perplexity is comparable across models with different tokenizers. | **False** | Different vocabularies mean different normalisation. |
| 11 | MLM is obsolete now that causal models dominate. | **False** | It is still better for classification, NER and embeddings (CS-07; `code/10_embedding_finetune.py`). |
| 12 | Decode is usually bandwidth-bound, not compute-bound. | **True** | Arithmetic intensity ~1 FLOP/byte vs a ridge point of 156–295. |
| 13 | LoRA with `r=8` can match full fine-tuning on a small dataset. | **True** | The frozen base regularises; the gap only opens at large data volumes. |
| 14 | Setting `alpha = r` keeps the LoRA scaling at 1. | **True** | Effective scaling is `alpha/r`. |
| 15 | You should mask prompt tokens to `-100` in SFT. | **True** | Otherwise the model learns to generate questions. |
| 16 | A wrong chat template usually shows up as an obviously broken loss curve. | **False** | The loss falls smoothly — it is a silent failure. |
| 17 | Emergent abilities are entirely a metric artefact. | **False** | Mostly metric-induced, but some transitions are genuinely sharp. |
| 18 | 2N FLOPs per token is the inference cost for a dense transformer. | **True** | Plus an `12·L·s²·h` attention term that dominates at long context. |
| 19 | QLoRA and QAT are the same technique. | **False** | QLoRA trains bf16 adapters on a quantized base; QAT trains through fake-quant nodes. |
| 20 | AdamW keeps the `m` and `v` buffers for frozen LoRA base weights. | **False** | Only for trainable parameters — that is the 3–5x memory saving. |

---

## Coding / Whiteboard Tasks

Five tasks, ordered by difficulty. Each has a time target and a grading rubric.

### Task 1 — VRAM estimator (10 minutes)

Write a function that takes `(params_b, precision, method, seq_len, batch, hidden, layers)` and returns estimated GB, broken out by component. It must handle full FT / LoRA / QLoRA and print a component table.

```python
def vram_gb(params_b, method="qlora", base_dtype_bytes=2, batch=1,
            seq_len=2048, hidden=4096, layers=32, lora_rank=16,
            lora_targets=7, grad_ckpt=True):
    """Estimate training VRAM in GB. method: 'full' | 'lora' | 'qlora'."""
    P = params_b * 1e9
    gb = lambda b: b / 1024**3

    # --- frozen base weights ---
    if method == "qlora":
        base_bytes = P * 0.5          # NF4 + double quant ~= 0.5 B/param
    else:
        base_bytes = P * base_dtype_bytes

    # --- trainable + optimiser state ---
    if method == "full":
        trainable = P
        grad_bytes = trainable * base_dtype_bytes        # 2 B/param
        opt_bytes  = trainable * 12                      # fp32 m + v + master
    else:
        # LoRA on all linear projections. Per target layer: r*(d_in+d_out).
        per_layer = lora_rank * (hidden * 4 + hidden * 3)  # attn 4 + mlp 3
        trainable = layers * per_layer
        grad_bytes = trainable * base_dtype_bytes
        opt_bytes  = trainable * 12

    # --- activations ---
    # FlashAttention, no materialised s^2 matrix. ~34*h*batch*seq bytes per layer,
    # reduced ~5x when checkpointing is on.
    act_per_layer = 34 * hidden * batch * seq_len
    if grad_ckpt:
        act_per_layer /= 5
    act_bytes = act_per_layer * layers

    overhead = 0.12 * (base_bytes + grad_bytes + opt_bytes + act_bytes)  # allocator + CUDA ctx

    parts = {"base weights": base_bytes, "gradients": grad_bytes,
             "optimiser state": opt_bytes, "activations": act_bytes,
             "overhead": overhead}
    total = sum(parts.values())
    for k, v in parts.items():
        print(f"{k:<18} {gb(v):8.2f} GB")
    print(f"{'TOTAL':<18} {gb(total):8.2f} GB")
    return gb(total)

# Expected: full 7B -> ~112 GB optimiser+weights, qlora 7B -> ~12-16 GB
vram_gb(7.0, method="full")
vram_gb(7.0, method="qlora")
```

- **Grading:** full FT returns ~110–130 GB (the `16 bytes/param` rule must be visible); QLoRA returns 10–18 GB; the component breakdown is printed; the activation term responds to `seq_len` and `grad_ckpt`.
- **Fail condition:** using a single `16 bytes/param` for every method, or forgetting optimiser state on the full-FT path.

### Task 2 — Chat-template forensics (10 minutes)

Given a tokenizer, write a function that takes a training example and a production request and proves whether their rendered strings are identical. It must print a character-level diff and flag any of: missing system turn, missing BOS, missing EOS after the assistant turn, wrong role delimiter.

```python
def check_template_parity(tok, example_messages, production_messages, model_name=""):
    train = tok.apply_chat_template(example_messages, tokenize=False,
                                    add_generation_prompt=False)
    prod  = tok.apply_chat_template(production_messages, tokenize=False,
                                     add_generation_prompt=True)  # what serving sends
    print(train)
    print("---")
    print(prod)
    train_ids  = tok(train).input_ids
    prod_ids   = tok(prod).input_ids
    print("\nTrain starts with BOS:",  train_ids[0]  == tok.bos_token_id)
    print("EOS present after assistant turn:",
          tok.eos_token_id in train_ids)
    print("Token counts:", len(train_ids), len(prod_ids))
    # Prompt tokens must be masked; the supervised span should be the answer + EOS.
    answer_ids = tok(example_messages[-1]["content"]).input_ids
    print("Supervised span length:", len(answer_ids) + 1, "(answer + EOS)")
    return train, prod
```

- **Grading:** recognises that the training string must end with the assistant turn *plus* EOS, that the serving string ends with the generation prompt, and that the supervised span is exactly the response tokens plus EOS.
- **Fail condition:** printing the strings without checking BOS/EOS/supervised-span length.

### Task 3 — Training-cost estimator (10 minutes)

Given `(params_b, tokens_b, n_gpus, gpu_peak_tflops, mfu, $/gpu-hour, gpus_available)`, return GPU-hours, dollars and wall-clock, then extend it to a small sweep table.

```python
def train_cost(params_b, tokens_b, n_gpus=1, peak_tflops=312, mfu=0.40,
               price=1.79, overhead=1.6):
    """overhead ~1.6x accounts for restarts, eval, dataloading, checkpointing."""
    flops = 6 * params_b * 1e9 * tokens_b * 1e9
    eff = n_gpus * peak_tflops * 1e12 * mfu
    gpu_hours = (flops / eff) / 3600 * overhead
    return {"gpu_hours": gpu_hours, "cost": gpu_hours * price,
            "wall_clock_h": gpu_hours / n_gpus,
            "flops": flops}

for p in (1, 3, 7, 13, 70):
    for t in (0.1, 1.0, 10.0):
        r = train_cost(p, t)
        print(f"{p:>3}B x {t:>5.1f}B tok -> {r['gpu_hours']:9.0f} A100-h  "
              f"${r['cost']:>10,.0f}  {r['wall_clock_h']:8.1f} h wall")

# Sanity anchor: 7B x 2T tokens ~= 6*7e9*2e12/1e15 = 84e3 theoretical
#   A100-80GB h * 1.6 overhead / 0.40 MFU -> ~336k A100-h.
#   Published Llama-2-7B figure is ~184k A100-h on a different, more efficient stack.
```

- **Grading:** the `6ND` term is correct; a stated MFU assumption; an explicit overhead factor; and a comparison against a published anchor.
- **Fail condition:** quoting peak TFLOPs as achieved throughput, or omitting the MFU assumption.

### Task 4 — Loss-curve diagnostician (15 minutes)

Given a list of per-step `(step, train_loss, eval_loss, grad_norm, lr)` tuples, classify the run into one of: *healthy*, *LR too high*, *overfitting*, *label-masking bug*, *template mismatch*, *diverged*. Print the evidence for the classification.

```python
import math

def diagnose(history, vocab_size=32000):
    """history: list of dicts with step, train_loss, eval_loss, grad_norm, lr."""
    ln_v = math.log(vocab_size)
    first, last = history[0], history[-1]
    ev = [h["eval_loss"] for h in history if h.get("eval_loss") is not None]
    gn = [h["grad_norm"] for h in history]

    if any(not math.isfinite(h["train_loss"]) for h in history):
        return "DIVERGED: non-finite loss — check fp16 overflow or lr"
    if abs(first["train_loss"] - ln_v) < 0.05 and abs(last["train_loss"] - ln_v) < 0.05:
        return f"NOT LEARNING: pinned at ln(V)={ln_v:.2f} — labels all -100 or no grads"
    if max(gn) > 50:
        return "LR TOO HIGH: grad_norm spikes >50 — check warmup and peak lr"
    if ev and last["train_loss"] < 0.3 and ev[-1] > ev[0]:
        return "OVERFITTING: train<0.3, eval rising from the start"
    if 0.4 < first["train_loss"] < ln_v - 1 and last["train_loss"] < 1.0:
        return "SUSPECT MASKING: loss started well below ln(V) — prompt likely unmasked"
    return "HEALTHY: monotone train decrease, eval tracking, grad_norm stable <5"

# Worked fixture: a run pinned at the uniform prior
hist = [{"step": s, "train_loss": 10.37, "eval_loss": 10.37,
         "grad_norm": 0.0, "lr": 2e-4} for s in range(0, 500, 50)]
print(diagnose(hist))
```

- **Grading:** the `ln(V)` rule must appear; the "loss started below `ln(V)`" heuristic (unmasked prompt) is the discriminating insight; each branch names a cause and a check.
- **Fail condition:** a function that only detects divergence, or one that cannot explain *why* each branch was taken.

### Task 5 — Data quality audit (20 minutes)

Write a script that takes a JSONL SFT file and the frozen eval set and reports: duplicate rate (exact + near via MinHash), train/eval leakage, length distribution (p50/p90/p99 versus `max_seq_length`), the fraction of rows missing a system turn or an assistant turn, label-mask sanity (supervised token count per row), and template-render consistency across rows.

```python
import json, hashlib
from collections import Counter
import numpy as np

def audit(train_path, eval_path, max_seq_length=2048, n_perm=128, shingle=5):
    rows = [json.loads(l) for l in open(train_path, encoding="utf-8")]
    ev   = [json.loads(l) for l in open(eval_path, encoding="utf-8")]

    def shingles(text):
        w = text.split()
        return {" ".join(w[i:i+shingle]) for i in range(max(0, len(w)-shingle+1))}

    def minhash(text):
        sig = [10**9] * n_perm
        for sh in shingles(text):
            h = int(hashlib.blake2b(sh.encode(), digest_size=8).hexdigest(), 16)
            for i in range(n_perm):
                sig[i] = min(sig[i], (h * (i*2+1) + i) % (2**32))
        return sig

    def text_of(r):
        return " ".join(m["content"] for m in r["messages"])

    exact = len(rows) - len({hashlib.md5(text_of(r).encode()).hexdigest() for r in rows})
    print(f"exact duplicates: {exact}/{len(rows)} ({exact/len(rows):.1%})")

    sigs = [minhash(text_of(r)) for r in rows]
    ev_sigs = [minhash(text_of(r)) for r in ev]
    # crude Jaccard on signatures; use datasketch.MinHashLSH in production
    leaks = sum(1 for s in sigs for e in ev_sigs
                if sum(a == b for a, b in zip(s, e)) / n_perm > 0.8)
    print(f"train/eval near-duplicate pairs: {leaks}")

    chars = [len(text_of(r)) for r in rows]
    approx_tokens = [c // 4 for c in chars]
    print("token p50/p90/p99:", np.percentile(approx_tokens, [50, 90, 99]).round(0))
    print(f"over max_seq_length: {sum(t > max_seq_length for t in approx_tokens)/len(rows):.1%}")

    missing_sys = sum(1 for r in rows if not any(m["role"] == "system" for m in r["messages"]))
    no_asst     = sum(1 for r in rows if not any(m["role"] == "assistant" for m in r["messages"]))
    print(f"rows without system turn: {missing_sys/len(rows):.1%}")
    print(f"rows without assistant turn: {no_asst}")

    lens = [len(text_of(r).split()) for r in rows]
    print("word-length buckets:", Counter(min(l // 50, 20) * 50 for l in lens).most_common(5))

audit("sft_train.jsonl", "sft_eval.jsonl")
```

- **Grading:** reports all six checks with numbers; uses near-duplicate detection (MinHash/LSH), not just exact hashing; compares the length distribution to `max_seq_length` and states the truncation consequence.
- **Fail condition:** only counting exact duplicates, or measuring lengths without connecting them to truncation and label loss.

---

## Cheat Sheet of Numbers To Memorize

| Quantity | Value | Why it matters |
|---|---|---|
| VRAM for full FT (bf16 + AdamW) | **16 bytes / parameter** | 7B = 112 GB, 13B = 208 GB, 70B = 1,120 GB |
| VRAM for LoRA (frozen bf16 base) | **2 bytes / param + ~0.6 GB** | 7B ≈ 14.6 GB before activations |
| VRAM for QLoRA (NF4 base) | **0.5 bytes / param** | 7B base ≈ 3.9 GB — fits a 12 GB card |
| Training compute | **6 × N × D** FLOPs | 7B × 2T tokens = 8.4e22 FLOPs |
| Inference compute | **2 × N** FLOPs/token | 7B = 14 GFLOP/token |
| Attention compute | **12 × L × s² × h** FLOPs/sequence | Dominates when `s > h` (≈4k for a 7B) |
| Chinchilla optimal ratio | **~20 tokens / parameter** | 7B → 140B tokens; 3B → 60B |
| AdamW state | **2 × 4 bytes (m, v) fp32** | Plus 4 bytes master weights in mixed precision |
| LoRA default | **r=16, alpha=32, lr=2e-4** | All linear targets: q,k,v,o,gate,up,down |
| Full FT learning rate | **1e-5 – 5e-5** (typ. 2e-5) | 10x lower than LoRA because the weights are precious |
| SFT epochs | **1–3** | >3 on small data = memorisation |
| warmup_ratio | **0.03 – 0.05** | Adam's `v` is unstable in the first steps |
| Gradient checkpointing cost | **+25–33% time** | For 5–10x less activation memory |
| A100-80GB peak (bf16) | **312 TFLOP/s**, 2.0 TB/s | Ridge point 156 FLOP/byte |
| H100 SXM peak (bf16) | **989 TFLOP/s**, 3.35 TB/s | Ridge point 295 FLOP/byte |
| Realistic MFU | **35–50% good, 20–30% common** | Peak is never achieved |
| Decode speed ceiling | **weight_bytes / bandwidth** | 7B fp16 on A100 ≈ 7 ms/token ≈ 143 tok/s |
| `ln(32000)` | **10.37** | Uniform-loss baseline for a 32k vocab |
| `ln(128256)` | **11.76** | Uniform-loss baseline for Llama-3's vocab |
| Tokens per English word | **~1.3** (GPT-4 class) | 1 token ≈ 4 chars; 0.75 words |
| Llama-2-7B pretraining cost | **~184,000 A100-hours** | Anchor for the 6ND sanity check |
| A100-80GB rental | **$1.10–2.50 / GPU-hour** | Lambda, RunPod, AWS band |
| 7B QLoRA fine-tune (typical) | **2–6 GPU-hours, $3–15** | The number that makes fine-tuning viable |
| KV cache per token (Llama-2-70B, fp16, MHA) | **2.62 MB** | 4k context = 10.7 GB per sequence |
| KV cache per token (Llama-3-8B, GQA-8, fp16) | **131 KB** | 8k context = 1.07 GB per sequence |
| LIMA result | **1,000 curated examples** | Quality beats quantity for instruction tuning |
| MLM mask rate (BERT) | **15%** | Versus 100% supervision for causal LM |

---

## Answers To The Self-Check Questions From CS-01

These mirror the ten questions at the end of CS-01 §19. Answer them before reading.

1. **"Fine-tune our 7B on product docs to stop hallucination."** The plan confuses knowledge with behaviour. Fine-tuning shapes *form*, not facts: an SFT run over documents teaches the register and vocabulary of confident documentation without adding verifiable information, and the canonical outcome is that hallucination *rises* (the CS-01 case study measures 18% → 41%) because the model becomes more fluent in the domain's assertive style. Add to that: documents are not instruction pairs, so you would have to synthesise questions and answers, and there is no citation path. **Proposal:** RAG over the documents with citations and a retrieval-quality eval, plus a small SFT run of 1,000–3,000 hand-written (question, grounded-answer-with-citation) pairs to fix *format and refusal behaviour*, plus a schema/faithfulness validator on the output. Budget the fine-tune at ~$5–20 of GPU time; budget the retrieval work at weeks.
2. **VRAM for a 13B full FT, bf16, AdamW.** `13e9 × 16 bytes = 208 GB`. Breakdown: bf16 weights `13e9 × 2 = 26 GB`; bf16 gradients `26 GB`; Adam fp32 `m` and `v` `13e9 × 8 = 104 GB`; fp32 master weights `13e9 × 4 = 52 GB`. Subtotal `26 + 26 + 104 + 52 = 208 GB`. Add activations: bf16, batch 1, seq 2048, 40 layers ≈ 8–12 GB with FlashAttention and gradient checkpointing, plus allocator fragmentation ~10%. **Total ≈ 235–245 GB** → four A100-80GB minimum, realistically 8 for headroom, using ZeRO-3 or FSDP with CPU offload. Compare QLoRA on the same model: base ~6.5 GB, adapters ~1 GB, optimiser ~1.5 GB → **~20 GB on one 24 GB card.**
3. **Train loss 2.9 → 0.35, eval loss 2.8.** Causes, in order: (1) **overfitting** — the train/eval divergence is the textbook signature, most likely from too many epochs on too small a dataset (1,000–2,000 examples at 3+ epochs will do this); (2) **train/eval leakage** — near-duplicates across the split inflate the train fit and the eval is measuring memorised items, so check with MinHash before accepting the "overfitting" verdict; (3) **distribution mismatch between the splits** — if the eval set came from a different source, template or time window than the train set, a legitimate fit on train looks like a generalisation failure. Fix order: verify no leakage, then stop at the eval-loss minimum with `load_best_model_at_end=True`, then cut to 1 epoch and halve the LR, then — the actual fix — add diverse training data. A final eval loss of 2.8 is also worth comparing to the *base* model's eval loss on the same set; if the base scored 2.9, the fine-tune has learned nothing at all and the data is the problem.
4. **Causal vs masked supervision.** Causal LM computes a loss at *every* position — `T` predictions for a `T`-token sequence — because each token predicts its successor. MLM replaces ~15% of tokens with `[MASK]` and computes loss only on those positions, so ~`0.15T` predictions per sequence: a ~6.7x difference in supervised targets. It is partly offset because MLM predictions use bidirectional context and are individually more informative, but per-token supervision still favours causal. **Where MLM is still better:** sequence classification and token labelling (NER) — the bidirectional encoder's `[CLS]`/token representations encode both sides of each token, which a causal mask forbids, and a fine-tuned BERT beats a similarly-sized generative model on these tasks at a fraction of the inference cost (CS-07).
5. **7B on 2B tokens: time and cost.** `C = 6ND = 6 × 7e9 × 2e9 = 8.4e19 FLOPs`. A100-80GB bf16 peak `312 TFLOP/s`; assume **MFU = 35%** → `109 TFLOP/s` effective per GPU. `8.4e19 / 1.09e14 = 770,600 s = 214 A100-GPU-hours` theoretical, × 1.6 real-world overhead (restarts, eval, dataloading, checkpointing) = **≈ 342 A100-hours**. At `$1.79/GPU-hour` = **≈ $612**; at reserved `$1.10/hr` = **$376**. Sanity anchor: Llama-2-7B used ~184,000 A100-hours for 2T tokens, i.e. ~184 A100-h per B tokens against a theoretical 107 — so 1.7x overhead, consistent. If you ran this on 4 × A100: ~86 hours wall clock.
6. **Why decode is bandwidth-bound.** Arithmetic intensity is FLOPs per byte moved. At batch 1 in fp16, producing one token requires reading every weight once (`2N` bytes) for `2N` FLOPs → intensity ≈ **1 FLOP/byte**. The A100's ridge point is `312e12 / 2.0e12 = 156 FLOP/byte` and the H100's is 295 — so decode operates two orders of magnitude to the left of the ridge, in the region where memory bandwidth, not compute, sets the ceiling. Practically, `time/token ≈ weight_bytes / bandwidth`: 7B fp16 on an A100 is `14 GB / 2 TB/s ≈ 7 ms` ≈ 143 tok/s. **Levers:** (1) **increase batch size** — this amortises the single weight read across many sequences and raises intensity until you hit the ridge point (the single biggest win; continuous batching does exactly this); (2) **reduce weight bytes** — quantization (int8 halves it, int4 quarters it) buys near-proportional speedup; (3) **speculative decoding** — trades spare compute (which is free, since you are bandwidth-bound) for fewer sequential weight reads; (4) **GQA/smaller KV cache** when the KV cache, not the weights, is what limits batch size.
7. **Base vs instruct, mechanically.** A base model is the raw pretrained transformer: weights trained only on next-token prediction, no chat template, no stop tokens, no notion of roles — asked a question, it continues the text pattern (a base model sees "What is the capital of France?" and may produce "What is the capital of Germany?" as the most likely continuation of a quiz list). An instruct model is that base plus supervised fine-tuning on (instruction, response) pairs, usually plus preference alignment, and it ships a `chat_template` plus an end-of-turn token. **What breaks if you fine-tune the wrong one:** (a) fine-tuning the **base** when you wanted an assistant gives you a model that ignores instructions and has to be prompted in completion style, and every eval against an instruct baseline is meaningless; (b) fine-tuning the **instruct** model when you wanted raw domain completion gives you a model that insists on answering in chat format and wraps your data in assistant framing; (c) using an **instruct** tokenizer/template on a **base** checkpoint silently corrupts training, because the special tokens have no learned embedding. Check `model.config.architectures` and whether `tokenizer.chat_template` is set before you start — this is a 10-second check that prevents a wasted run.
8. **Works in chat UI, garbage on the batch endpoint.** Overwhelmingly a **chat-template / padding mismatch**, not a model problem. Ranked: (a) the endpoint does not apply the model's chat template (string concatenation, or a different template than training used); (b) it omits the system turn the fine-tune was conditioned on; (c) **right padding** — decoder-only batched generation requires left padding, or the shorter rows are conditioned on pad tokens and emit garbage; (d) the serving `eos_token_id` does not include the template's end-of-turn token, so generation runs past the answer; (e) a different tokenizer revision. Confirmation test: send a batch of size 1 with the exact chat string — if that works and batch >1 fails, it is padding; if both fail identically, it is the template. Fix by making serving call the same pinned tokenizer artifact and the same `apply_chat_template` path as training, and add a golden-request CI test.
9. **Chinchilla, and why a team would train 10x past it.** The rule: for a fixed compute budget, `N_opt ∝ C^0.5` and `D_opt ∝ C^0.5`, so `D_opt ≈ 20N` — roughly **20 tokens per parameter**. For a 3B model that is **60B tokens** compute-optimal. **Why overtrain anyway:** training cost is paid once, inference cost is paid per request forever. A smaller model trained well past the compute-optimal point reaches the quality of a larger, compute-optimally-trained model at a fraction of the per-token inference cost. If you expect billions of tokens of lifetime inference, the total cost of ownership is minimised by overtrained small models. The evidence: GPT-3 at 1.7 tokens/param (undertrained), Chinchilla at 20, Llama-3-8B at 1,875 tokens/param (94x past Chinchilla) — because Llama-3-8B must be cheap to serve and it beats models 3x its size. The one-line answer: **Chinchilla minimises training FLOPs; production minimises total cost of ownership, and inference dominates.**
10. **0% exact match, 0.82 token-level F1.** The model has learned the *content* but not the *format* — it produces nearly the right tokens in the wrong structure, so the strict metric scores zero while the continuous metric shows real learning. This is the emergence-as-metric-artefact problem in miniature, and it is the single most common reason a fine-tune looks like a failure when it is nearly working. **Check first:** the evaluation harness's normalisation — whitespace, casing, punctuation, key order in JSON, and any wrapper text like "Answer:" that the model emits. Most 0%-exact-match results are a harness that never normalised. **Fix in order:** (1) fix the harness normalisation and re-score the *existing* model — this alone resolves many cases; (2) if the format is still wrong, the training data's format does not match inference, or the chat template is wrong — render and diff; (3) if the format is right but values are wrong, it is a data-quality problem: the gold answers are inconsistent. Do **not** retrain before doing (1) and (2); both are free, and both are more likely than a genuine training failure.

---

## Cross-References

| Relationship | Module |
|---|---|
| **Source case study** | CS-01 (Foundations: Pretraining, Training & the LLM Lifecycle) |
| **Companion cheat sheet** | CH-01 (formulas, VRAM/cost calculators, symptom→fix card) |
| **Next in sequence** | CS-02 (Transfer Learning), CS-03 (Framework Landscape) |
| **Deepens the architecture decision** | CS-04 (Fine-Tuning vs RAG vs Agents) |
| **Deepens the training math** | CS-05 (RNN/LSTM → Attention), CS-06 (Hugging Face) |
| **Deepens encoder fine-tuning** | CS-07 (BERT: NER, Sentiment, QA) |
| **Deepens memory and cost** | CS-10/CS-11 (Quantization), CS-16 (Unsloth), CS-13 §6.8 / CS-11 §4.11 (LoRA/QLoRA) |
| **Deepens data work** | CS-12 (continued pretraining), CS-13 (SFT) |
| **Deepens alignment questions (Q71)** | CS-14 §4.6.1–4.6.10 (Alignment Map: RLHF/DPO/GRPO/ORPO) |
| **Capstone** | CS-13 + CS-16/CS-17 (CS-28 planned, not yet written) |
