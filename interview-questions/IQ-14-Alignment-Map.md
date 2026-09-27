# IQ-14 — Interview Questions: The Alignment Map (RLHF, PPO, DPO, ORPO)

| Field | Value |
|---|---|
| **Module** | Alignment / Preference Optimisation |
| **Pairs with** | CS-14, CH-14 |
| **Total questions** | **116** (35 L1 + 32 L2 + 25 L3 + 12 L4 + 12 L5) |
| **Levels covered** | Screen / Intermediate / Advanced / System Design / Debug |
| **Source video** | LLM Fine-Tuning 16: Preference Alignment & Preference Training in LLMs with RLHF, RLAIF, DPO, LoRA |

---

## How To Use This File

- **L1** = phone screen / recruiter filter. 30-second answers. If you cannot answer an L1 in three sentences, you are not ready for the loop.
- **L2** = working engineer. 2–3 minutes. Expects implementation detail: real flags, real defaults, real numbers.
- **L3** = senior / specialist. 5 minutes. Expects trade-offs, internals, and knowledge of where the method breaks.
- **L4** = staff / system design. 15-minute whiteboard. Answer with **requirements → constraints → design → trade-offs → failure modes**.
- **L5** = debugging & incident. War stories. Answer with an **ordered checklist**, not a list of possibilities.

Every question has an **Answer**, a **Why asked**, and — where a plausible-but-wrong answer exists — a **Trap**. `[Company style: ...]` tags indicate the loop the question is typical of.

**The 12 numbers to have on instant recall before you start:** β=0.1 (DPO default), 0.1–0.5 (β range), `log 2 = 0.6931` (DPO loss floor), 4 models (PPO), 2 (DPO/GRPO), 1 (ORPO/SimPO), 16 bytes/param (full FT AdamW), 14 GB (7B bf16), 112 GB (7B full FT optimiser+grad+weights), 2e-5 (LoRA LR), 1–3 epochs (DPO), 20k pairs (the PPO-vs-DPO crossover).

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. What is preference alignment in one sentence?**
- **Answer:** Training a model on data that says which of two (or more) responses a human prefers, so that the model's *behaviour* — tone, safety, refusal calibration, concision — matches human intent, rather than merely being fluent or factually adequate.
- **Why asked:** Vocabulary gate. Anyone who has done this work has a crisp one-liner.
- **Trap:** "It's RLHF." RLHF is one method; DPO, ORPO, GRPO, KTO are alignment methods that are not RLHF.

**Q2. Name the three stages of the canonical LLM training pipeline.**
- **Answer:** (1) Pretraining — next-token prediction on web-scale text, producing a foundation model. (2) Supervised fine-tuning (SFT) — training on (instruction → response) demonstrations, producing an instruction-following model. (3) Preference alignment — training on `{prompt, chosen, rejected}` pairs, producing the aligned model.
- **Why asked:** The single most fundamental framing question in the track.
- **Trap:** Saying "pretraining, fine-tuning, RLHF" — which conflates SFT with fine-tuning generally and RLHF with all alignment.

**Q3. What does HHH stand for, and what is the catch?**
- **Answer:** Helpful, Harmless, Honest. The catch: they are in tension — maximising helpfulness maximises compliance (hurting harmlessness), maximising harmlessness produces an over-refuser (hurting helpfulness), and optimising for confident-sounding answers attacks honesty. They form a Pareto frontier, and where you sit on it is a product decision expressed through your preference data mix and your β, not something the loss function balances for you.
- **Why asked:** Tests whether you understand alignment as a trade-off or as a score.
- **Trap:** Reciting the three words and stopping.

**Q4. What is the data format for preference training?**
- **Answer:** `{prompt, chosen, rejected}` — three fields. The prompt is shared; `chosen` is the preferred response, `rejected` the dispreferred one. Every public preference dataset is a re-skin of this triple.
- **Why asked:** Absolute baseline.
- **Trap:** Saying `{input, output}` — that is SFT.

**Q5. What is the difference between an SFT dataset and a preference dataset?**
- **Answer:** SFT data is a *single target per prompt* — (instruction, response). Preference data is an *ordering* — two responses and which is better. Preference data carries contrastive information (what not to do) at roughly the same annotation cost as one demonstration.
- **Why asked:** Tests whether you know why the extra stage exists at all.
- **Trap:** "Preference data has two outputs instead of one" without the ordering point.

**Q6. What is a reward model?**
- **Answer:** A model trained to output a scalar score for a (prompt, response) pair, standing in for human judgement. Usually initialised from the SFT model with the LM head replaced by a scalar head, trained with the Bradley–Terry loss on preference pairs. It is required by PPO-based RLHF, and not required by DPO, ORPO, SimPO, or GRPO-with-a-verifier.
- **Why asked:** Gate for the whole RLHF conversation.
- **Trap:** Confusing it with the *value* model (which predicts expected return from a partial state for advantage estimation).

**Q7. Which method needs no reward model and no reference model?**
- **Answer:** **ORPO** (and **SimPO** also needs no reference, though it does compute two forward passes over the policy). ORPO is a single-model, single-stage method: `L = L_SFT + λ·L_OR`.
- **Why asked:** Tests whether you know the landscape beyond RLHF/DPO.

**Q8. What is `beta` in DPO?**
- **Answer:** The coefficient scaling the log-ratio inside the sigmoid. It controls how hard the policy is pushed away from the reference model. Typical range 0.1–0.5; the TRL default and the video's value is **0.1**.
- **Why asked:** It is the single most important DPO hyperparameter.
- **Trap:** Saying "it's the KL penalty like in PPO" — in DPO it multiplies the log-ratio inside the sigmoid, so *higher β = more movement*, the opposite of PPO's β.

**Q9. What is the KL divergence term doing in the alignment objective?**
- **Answer:** `β·KL(π_θ ‖ π_ref)` keeps the trained policy close to the frozen SFT reference. Three jobs: it acts as a brake on reward hacking (keeps the policy in the region where the reward model is valid), it preserves general capability, and it prevents collapse onto a degenerate high-reward string.
- **Why asked:** Every alignment method except ORPO/SimPO contains this term.
- **Trap:** Thinking it is a safety mechanism. It anchors you to the reference, which may itself be rude or unsafe.

**Q10. What is RLHF's "4-model problem"?**
- **Answer:** RLHF with PPO holds four models in memory simultaneously: the **policy** (trained), the **reference** (frozen, for the KL), the **reward model** (frozen, scores responses), and the **value model** (trained, for advantage estimation). That is roughly 4× the model weights plus optimizer state for two of them.
- **Why asked:** The economic fact that drove the industry to DPO.
- **Trap:** Saying "3 models" by forgetting the value model, or "2 models" by merging policy and value — which is sometimes done but is not the canonical setup.

**Q11. What is DPO, in one sentence?**
- **Answer:** Direct Preference Optimization — a closed-form re-parameterisation of the KL-constrained RLHF objective that turns it into a **supervised classification loss on preference pairs**, using only the policy's and a frozen reference's log-probabilities. No reward model, no rollouts, no RL loop.
- **Why asked:** The single most likely "explain DPO" screen question.

**Q12. Is DPO reinforcement learning?**
- **Answer:** No. It is a supervised learning method on pairs. The video's own wording: *"This is a simple supervised learning"* [19:47]. There is no environment, no reward signal at training time, and no sampling loop.
- **Why asked:** It is a favourite gotcha. The derivation comes *from* RL, but the method is supervised.

**Q13. What is GRPO?**
- **Answer:** Group Relative Policy Optimization — PPO without the value model. For each prompt you sample a *group* of G responses, score each with a reward function, and use the group's mean reward as the baseline. The advantage is the response's z-score within the group.
- **Why asked:** The reasoning-model era made this the most-asked modern alignment question.
- **Trap:** Describing it as "PPO with fewer models" without the group-relative baseline, which is the actual mechanism.

**Q14. What is RLVR?**
- **Answer:** Reinforcement Learning with Verifiable Rewards — GRPO (or PPO) where the reward is a *program*: a unit test, a math answer check, a schema validator, a compiler. Because the reward is exact, you need no reward model and no preference data at all.
- **Why asked:** It is the defining alignment technique of the 2025–2026 reasoning-model wave.

**Q15. What is the difference between RLAIF and RLHF?**
- **Answer:** The loop is identical; the labeler is different. RLHF uses human preference labels, RLAIF uses an LLM judge. RLAIF exists because human labelling does not scale — the video's framing is that humans top out at *"one lakh two lakh"* (100k–200k) labels.
- **Why asked:** Tests whether you know the scalability argument.

**Q16. What is Constitutional AI?**
- **Answer:** Anthropic's specific RLAIF recipe: write a constitution (a list of principles), have the model critique its own response against a sampled principle and revise it, use the revision as `chosen` and the original as `rejected`, then run preference training. Its key property is that the values are written down and auditable rather than implicit in a crowd of annotators.

**Q17. What is the alignment tax?**
- **Answer:** The capability degradation caused by alignment. InstructGPT reported RLHF models scoring lower than base GPT-3 on SQuAD, DROP and HellaSwag while being strongly human-preferred. Mitigated in that paper by **PPO-ptx**, which mixes pretraining gradients into the PPO update.

**Q18. What is reward hacking?**
- **Answer:** The policy maximises the *proxy* reward (the reward model's score) rather than the true objective (human preference), exploiting errors in the reward model. Because the policy is doing search over the response space and the RM has errors everywhere, this happens on the training distribution with a perfectly-generalising RM. It is an instance of Goodhart's law.

**Q19. Name three observable symptoms of reward hacking.**
- **Answer:** (1) Response length grows monotonically through training — length bias. (2) The model agrees with false premises — sycophancy. (3) Every answer becomes a bulleted list with a summary — formatting exploit. Others: uncertainty deletion (never hedging), refusal drift, degenerate repetition.

**Q20. What is the win rate metric?**
- **Answer:** The fraction of pairwise comparisons in which your model's response is preferred over a baseline's. It is the primary alignment metric, and it is always baseline-relative — "60% win rate" is meaningless without naming the baseline and the judge.

**Q21. What is AlpacaEval 2's LC win rate, and why does it exist?**
- **Answer:** Length-controlled win rate. Raw AlpacaEval uses an LLM judge, and LLM judges prefer longer responses, so a model that simply writes more wins more — by roughly 10 points. The LC variant fits a logistic regression including the length difference and reports the win rate at zero length difference, i.e. the counterfactual "if both were equally long, who wins?".

**Q22. Which alignment method has the smallest memory footprint?**
- **Answer:** ORPO (or SimPO) — **one model**. It has no reference model and no reward model, and the SFT loss inside its objective is the anchor that the reference would otherwise provide.

**Q23. What is a "delta patch" and why can't it be stacked?**
- **Answer:** A LoRA adapter is a small low-rank weight *delta* — `W_new = W_old + (α/r)·BA` — not a full layer. Stacking adapters means the second adapter trains against a frozen offset it cannot see or adjust, which produces unstable loss, hallucination and degraded quality. The correct discipline is to **merge** each adapter into the base (`merge_and_unload()`) before attaching the next.

**Q24. What does `get_peft_model` do, and how does it differ from `PeftModel.from_pretrained`?**
- **Answer:** `get_peft_model(model, config)` **creates** a new LoRA adapter to be trained. `PeftModel.from_pretrained(model, path)` **loads** an already-trained adapter — for inference or for further training. The video makes this distinction explicitly [44:47]–[45:38].

**Q25. What is `beta` in PPO?**
- **Answer:** The coefficient on the per-token KL penalty in the reward: `r_total = r_RM − β·KL_t`. Higher β keeps the policy closer to the reference. Typically 0.02–0.05. **Note this is the opposite direction from DPO's β.**

**Q26. What is rejection sampling / best-of-n?**
- **Answer:** Sample n responses per prompt, keep the best by some scorer (a reward model, a verifier, a judge), and SFT on the kept ones. No RL, no pairs needed. It is bounded by what the sampler can produce — it reduces variance, it does not add capability.

**Q27. What is the difference between a reward model and a value model?**
- **Answer:** A reward model scores a *complete* response and is frozen. A value model predicts the *expected future return* from a partial sequence (a state) and is trained alongside the policy; it exists only to compute advantages. PPO needs both; GRPO needs only the former (and, with verifiable rewards, neither).

**Q28. Which is more sample-efficient: SFT on demonstrations or DPO on pairs?**
- **Answer:** DPO on pairs, per unit of annotation effort. Ranking k answers is O(k) annotator work but yields O(k) or O(k²) comparison constraints, whereas a demonstration yields exactly one training target. The caveat: SFT is what makes the preference stage cheap in the first place — you need it first.

**Q29. What is `remove_unused_columns` and what should it be for DPO?**
- **Answer:** It is a Hugging Face `Trainer` flag that drops dataset columns not present in the model's `forward()` signature. For DPO it must be **`False`**, because `prompt`, `chosen` and `rejected` are not model inputs — they are consumed by the trainer's own collator, and the default `True` deletes them.

**Q30. What happens if you set DPO's beta very low?**
- **Answer:** The policy is pushed hard away from the reference: reward hacking appears almost immediately, fluency collapses, output degenerates (repetition, or the same answer for every prompt). β→0 is pure reward maximisation with no anchor.

**Q31. What happens if you set DPO's beta very high?**
- **Answer:** The policy cannot move. The margin stays at zero and the loss is pinned at the sigmoid floor, `−log σ(0) = log 2 ≈ 0.6931`. Zero gradient, no learning.

**Q32. What is a preference "tie", and why does it matter?**
- **Answer:** A pair on which annotators genuinely see no difference. If your guidelines forbid ties, annotators are forced to invent an ordering on genuinely equal pairs, which injects noise. Either allow and drop ties, or handle them with **cDPO's label smoothing** (ε set to the observed noise rate).
- **Why asked:** Separates people who have run an annotation project from people who have read papers.

**Q33. What is inter-annotator agreement and why should anyone care?**
- **Answer:** The degree to which independent annotators produce the same preference — measured by raw agreement %, Cohen's κ for two raters, or Krippendorff's α for n. It is the *ceiling* on achievable alignment: if annotators agree on only 65% of pairs, 35% of your labels are noise and no amount of training fixes it. Rule of thumb: below 70% raw agreement, fix the guidelines before spending on volume.
- **Why asked:** Almost nobody measures it; mentioning it is a strong signal.

**Q34. Which is cheaper: human preference labels or LLM-judge labels?**
- **Answer:** LLM-judge labels by roughly 50× — around $0.04–$0.06 per pair versus $2–$4 for human annotation. But the judge imports its own stylistic biases (notably a preference for verbosity and heavy formatting), so the production default is RLAIF labels plus a human-audited holdout sample.

**Q35. What is the one-line rule for choosing between GRPO and DPO?**
- **Answer:** *If you can write a program that returns a reward, use GRPO with that program and skip the reward model and preference data entirely. If you cannot, use DPO on pairs.* Verifiable domains (math, code, tool calls) → GRPO/RLVR. Subjective domains (tone, style, safety, helpfulness) → DPO or ORPO.

---

## Level 2 — Applied & Implementation

**Q36. Write the DPO loss and explain each symbol.**
- **Answer:**
  ```text
  L_DPO = − E_{(x, y+, y−) ~ D} [ log σ( β · ( log(π_θ(y+|x)/π_ref(y+|x))
                                            − log(π_θ(y−|x)/π_ref(y−|x)) ) ) ]
  ```
  `x` = prompt; `y+`/`y−` = chosen/rejected; `π_θ` = the policy being trained; `π_ref` = the frozen SFT reference; `β` = strength coefficient (default 0.1); `σ` = logistic sigmoid. Define `h(x,y) = log π_θ(y|x) − log π_ref(y|x)`, the **log-ratio**; then `L = −log σ(β·(h(y+) − h(y−)))`. The loss is minimised when the log-ratio advantage of the chosen response is large and positive.
- **Why asked:** The single most common "write it on the board" DPO question.

**Q37. Which TRL classes and arguments do you use for DPO, exactly?**
- **Answer:**
  ```python
  from trl import DPOTrainer, DPOConfig
  cfg = DPOConfig(output_dir=..., learning_rate=2e-5, per_device_train_batch_size=1,
                  gradient_accumulation_steps=8, num_train_epochs=1, beta=0.1,
                  loss_type="sigmoid", remove_unused_columns=False, max_length=1024)
  trainer = DPOTrainer(model=model_with_new_lora, ref_model=None, args=cfg,
                       train_dataset=ds, processing_class=tokenizer)
  trainer.train()
  ```
  Note `processing_class=tokenizer` — TRL ≥ 0.13 renamed the `tokenizer` argument. `remove_unused_columns=False` is mandatory.
- **Why asked:** Verifies you have actually run it, not just read about it.
- **Trap:** Passing `tokenizer=tokenizer` on a new TRL, or omitting `remove_unused_columns=False`.

**Q38. In the video's notebook, what does `ref_model=None` mean?**
- **Answer:** With PEFT adapters, TRL **disables the adapter** on the same base weights and uses that as the reference — effectively a free reference, one set of weights in memory. At **full fine-tuning**, TRL deep-copies the model at trainer init and freezes the copy — two full models, and the most common cause of an unexpected OOM in full-FT DPO.
- **Why asked:** It is the single most misunderstood line in the notebook and a genuinely discriminating question.

**Q39. Walk through the LoRA-merge discipline for a three-stage pipeline.**
- **Answer:**
  ```python
  base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
  m = PeftModel.from_pretrained(base, "checkpoint-stage1")   # load EXISTING adapter
  m = m.merge_and_unload()                                    # fold into base weights
  m = get_peft_model(m, lora_config)                          # attach a FRESH adapter to train
  ```
  Correct: `Base + merge(stage-N LoRA) + NEW LoRA`. Wrong: `Base + LoRA + LoRA + LoRA`. Stacking produces unstable loss, hallucination, poor tuning and quality degradation — and the loss curve does not reveal it.
- **Why asked:** The video's central practical lesson. Answers that skip `merge_and_unload` are a red flag.

**Q40. Why is merging a LoRA into an 8-bit base a problem?**
- **Answer:** The merged weights must be re-quantised back to int8, and the rounding changes the model's outputs relative to the unmerged path. PEFT emits the warning *"Merge lora module to 8-bit linear may get different generations due to rounding errors."* The production-safe pattern is to load the base in bf16, merge, then optionally re-quantise — and to re-run the stage-N evaluation on the merged model rather than assuming it is unchanged.
- **Why asked:** A subtle production detail almost nobody knows; it appears in the video's own screen output.

**Q41. What is the DPO loss floor, and why?**
- **Answer:** `−log σ(0) = log 2 ≈ 0.6931`. It is reached when the log-ratio margin is zero — which is exactly the state at initialisation, because the policy and the reference are the same weights. So a DPO run's loss *starts* at ~0.693 and must fall below it to be learning.
- **Why asked:** A superb, cheap diagnostic. Candidates who know this number have debugged a real run.

**Q42. The video's DPO run reported `loss=0.6619` with `global_step=1`. Interpret both numbers.**
- **Answer:** `global_step=1` means exactly **one** optimiser update: 5 examples ÷ (batch 1 × grad-accum 8) = ⌈5/8⌉ = 1. `loss=0.6619` is 0.031 below the 0.6931 floor, i.e. one step of learning. You cannot learn a preference direction from one gradient step — the run demonstrates the *pipeline*, not a trained model.
- **Why asked:** Tests whether you read the metric or just re-ran the notebook.

**Q43. Why does DPO need the SFT stage first?**
- **Answer:** The KL term anchors the policy to π_ref. On a base model, π_ref is an autocompleter, so the KL term penalises the model *for being an assistant*. Additionally, `chosen` and `rejected` are both far off-distribution for a base model, so their log-probs are tiny and noisy and the log-ratio carries little signal. Result: a model worse than the SFT baseline on instruction-following and only marginally better on tone.
- **Why asked:** It is the classic "can I skip a stage" probe and it separates understanding from recipe-following.

**Q44. What data preprocessing is mandatory before a DPO run?**
- **Answer:** (1) Assert the schema is exactly `{prompt, chosen, rejected}` — TRL accepts aliases in some versions and silently mis-maps in others. (2) Materialise an explicit `prompt` column (`hh-rlhf` ships only `chosen`/`rejected`; the prompt is the shared prefix and inference is fragile). (3) Count and drop rows where `chosen == rejected` — a judge with a parsing bug produces these and they contribute a zero margin. (4) Compute `len(prompt+chosen)` at the p95 and set `max_length` above it. (5) Deduplicate prompts against your eval set. (6) Set `tokenizer.chat_template` explicitly and print one formatted example.
- **Why asked:** This is the checklist that separates a working run from a mysterious failure.

**Q45. What do you log during a DPO run?**
- **Answer:** Training loss (against the 0.6931 floor); `rewards/chosen`, `rewards/rejected`, `rewards/margins`, `rewards/accuracies` (TRL emits these); `logps/chosen`, `logps/rejected`; **mean response length** (the reward-hacking canary); KL to the reference; and a capability suite (MMLU/GSM8K/IFEval) every N steps (the alignment-tax canary). `report_to="wandb"` in production — the failure modes are trends, not endpoints.
- **Why asked:** Verifies you monitor for over-optimisation rather than just watching loss fall.

**Q46. What learning rate and epoch count for DPO with LoRA?**
- **Answer:** Learning rate **1e-5 to 2e-5** for LoRA (the video uses 2e-5); **5e-7 to 5e-6** for full fine-tuning, because full FT moves far more parameters. Epochs **1–3** — DPO overfits past that; the loss keeps falling while quality drops. Warmup ratio 0.1.

**Q47. Why does the video use `repetition_penalty=1.1` and what is the safe range?**
- **Answer:** It discourages the model from repeating tokens already generated. Safe range is **1.0–1.3**; 1.1 is mild, 1.3 is aggressive, and 1.5+ starts breaking syntax because the model is forbidden from repeating necessary function words and punctuation. The video says "up to two", which is generous. For stronger control prefer `no_repeat_ngram_size` or a presence penalty.

**Q48. How do you build an ORPO config, and how does it differ from DPO's?**
- **Answer:**
  ```python
  from trl import ORPOTrainer, ORPOConfig
  cfg = ORPOConfig(learning_rate=8e-6, per_device_train_batch_size=1,
                   gradient_accumulation_steps=8, num_train_epochs=1,
                   beta=0.1,                       # ORPO's beta scales the odds-ratio term
                   max_length=1024, max_prompt_length=512,
                   remove_unused_columns=False)
  trainer = ORPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer)
  ```
  Key differences: **no `ref_model`** (there is none), a much lower LR, and `lambda` (or `beta` in TRL's naming) weights the odds-ratio term against the SFT term — typically 0.1–0.5.
- **Why asked:** Tests whether "cheapest method" means you know its actual knobs.

**Q49. Can ORPO run without a preceding SFT stage?**
- **Answer:** **Yes** — and that is its distinguishing property. The objective is `L_SFT + λ·L_OR`, so the SFT loss anchors the model and you can go from base to aligned in one stage, provided your `chosen` set is a real instruction dataset. This is the one legitimate exception to the "never align a base model" rule (which holds for DPO, because DPO's anchor is a *frozen* reference, not a live SFT loss).
- **Why asked:** A genuinely subtle question that distinguishes deep understanding from recitation.

**Q50. What is the GRPO config, and what is `num_generations`?**
- **Answer:**
  ```python
  from trl import GRPOTrainer, GRPOConfig
  cfg = GRPOConfig(learning_rate=1e-6, per_device_train_batch_size=1,
                   gradient_accumulation_steps=4, num_generations=8,
                   temperature=1.0, max_completion_length=1024,
                   beta=0.04, num_train_epochs=1, bf16=True)
  trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=cfg,
                        train_dataset=prompt_only_ds, processing_class=tokenizer)
  ```
  `num_generations` is **G**, the group size — how many responses are sampled per prompt to form the baseline. G=1 degenerates (the group mean is the sample itself, advantage is 0/0). Realistic: 8–16; DeepSeek-R1 used 16.

**Q51. What is the KL penalty coefficient in PPO and how does it compare to DPO's beta?**
- **Answer:** PPO's `kl_coef` is typically **0.02–0.05** and it multiplies a per-token KL penalty subtracted from the reward; higher = the policy stays closer to the reference. DPO's β is 0.1 by default and multiplies the log-ratio *inside the sigmoid*; higher = the policy is pushed *further* from the reference. **The meaning inverts**, and copying a value between the two is a real and common error.

**Q52. What is the "zero-variance group" problem in GRPO?**
- **Answer:** The advantage is `(r_i − mean(r))/std(r)` within the group. If all G responses get the same reward — all 0 because the prompt is too hard, or all 1 because it is too easy — the numerator is zero and the std is zero, so the prompt contributes no gradient. On hard reasoning tasks 30–50% of prompts can be zero-variance. Mitigations: **dynamic sampling** (DAPO — drop zero-variance prompts from the batch instead of wasting them), adjust task difficulty, or raise the rollout temperature.
- **Why asked:** It is the #1 practical GRPO failure and distinguishes people who have run it.

**Q53. How much VRAM for DPO on a 7B model with QLoRA?**
- **Answer:** Roughly **12–20 GB**. Arithmetic: 4-bit base ≈ 3.5–4.5 GB; the reference is free (same base, adapter disabled); LoRA adapter + 8-bit optimizer state ≈ 1 GB; activations for batch 1 × 1024 tokens ≈ 6–14 GB. It fits on a single RTX 4090 24 GB or an A6000. Contrast: full-FT DPO is ~150 GB and full-FT PPO ~295 GB.
- **Why asked:** Numeric fluency on the decisive constraint.

**Q54. What is the PPO-vs-DPO crossover in data volume?**
- **Answer:** Roughly **20,000 pairs**. Below that, DPO matches or beats PPO at a fraction of the cost. Above it, with a well-trained reward model and the compute to run the RL loop, PPO's exploration advantage starts to pay. This is the whole economic case for the field's migration in one number.

**Q55. What is the difference between DPO, IPO and cDPO?**
- **Answer:** All three share the log-ratio formulation. **DPO** uses `−log σ(β·margin)` and drives the margin toward +∞. **IPO** uses a squared loss with a *finite* target margin (`1/(2β)`), which prevents overfitting when pairs are near-deterministic (correct vs incorrect). **cDPO** adds **label smoothing** (ε typically 0.1) to the sigmoid, admitting that a fraction of labels are wrong — use it when IAA is low.

**Q56. When would you use KTO instead of DPO?**
- **Answer:** When your data is **unpaired** — thumbs-up/thumbs-down telemetry, A/B outcomes — rather than head-to-head pairs. KTO is pointwise, uses a prospect-theory utility with asymmetric weights (`λ_U > λ_D`, losses weigh heavier than gains), and requires no pairs. It is less sample-efficient than DPO when pairs *are* available, so it is a data-shape decision, not a quality decision.
- **Trap:** Saying "KTO needs no reference model" — it does, because its utility is defined relative to a KL-based reference point.

**Q57. What is SimPO, and what problem does it solve?**
- **Answer:** Simple Preference Optimization — DPO **without a reference model**, using the **length-normalised average log-probability** as the implicit reward plus a target margin γ. It solves two problems at once: (1) memory — one model fewer, ~30–50% less VRAM; (2) length bias — the `1/|y|` normalisation removes DPO's tendency to make responses longer. Note it still has a β (scaling) and a γ (margin); γ=0 is not the right default.

**Q58. What does the Bradley–Terry model say, and what is the RM loss?**
- **Answer:** BT: `P(y+ ≻ y− | x) = σ(r(x,y+) − r(x,y−))` — preference probability is a logistic function of the reward *difference*. The RM loss is maximum likelihood on that: `L = −E[log σ(r_φ(x,y+) − r_φ(x,y−))]`. Two consequences people miss: the loss depends only on the *margin*, so RM absolute scores are only identifiable up to a per-prompt constant (never compare RM scores across prompts); and **an RM whose loss is stuck at 0.693 has learned nothing** (σ(0)=0.5).

**Q59. How do you get preference data if you have no annotation budget?**
- **Answer:** Three options in increasing cost: (1) **RLAIF** — an LLM judge labels pairs; ~$0.04–$0.06 per pair, ~85–95% agreement with humans on non-safety content. (2) **Rejection sampling** — sample n responses from your own model, score them with a verifier or heuristic, use the best as `chosen` and a lower-ranked one as `rejected`; the labels are free, the generation is the cost. (3) **SPIN** — use your existing SFT data as `chosen` and your previous iteration's outputs as `rejected`, and iterate. All three need a human-audited holdout or you are flying blind.

**Q60. How many preference pairs do you actually need?**
- **Answer:** Approximate scale for a 7B tone/style task: **<500** — mostly noise, little change. **1,000** — a detectable shift, some overfitting. **5,000** — production-usable with DPO or ORPO. **20,000** — diminishing returns for DPO, and where PPO starts to win. **50,000+** — saturated for offline methods. The real answer is "until held-out preference accuracy stops improving", which is why you split 10% off before training.

**Q61. What is the difference between an offline and an online alignment method?**
- **Answer:** **Offline** (DPO, ORPO, IPO, cDPO, SimPO, SLiC) trains on a fixed preference dataset and never samples from the policy during training. Its ceiling is the dataset. **Online** (PPO, GRPO, iterative DPO) samples from the current policy, scores, and updates. Online methods can discover responses the dataset never contained — which is why verifiable-reward RL produces genuinely new capability while DPO produces better-calibrated existing capability. Online methods cost 5–30× more because they generate tokens during training.

**Q62. How do you serve an aligned model if you have several alignment variants?**
- **Answer:** Serve **adapters**, not merged models. One base in VRAM plus N adapters at ~4 MB each, with per-request adapter selection (vLLM and TGI both support multi-LoRA serving). Merging requires a full model copy per variant — 14 GB each at 7B bf16. This also gives you free A/B: route a traffic split to different adapter names.

**Q63. How do you split data for a DPO run?**
- **Answer:** Hold out **10%** of the pairs, split by *prompt* (never by row — the same prompt appearing in both splits leaks). Use the held-out set to compute **preference accuracy** and the **implicit margin** on every checkpoint. Without a held-out split you cannot tell learning from overfitting, and your only signal is training loss, which falls in both cases.

**Q64. What is the implicit reward margin and how do you compute it?**
- **Answer:** `margin = β · ( h_θ(x, y+) − h_θ(x, y−) )` where `h_θ(x,y) = log π_θ(y|x) − log π_ref(y|x)`. It is DPO's own quantity: the model's implicit reward advantage for the chosen response. Compute it on held-out pairs — a positive mean is necessary but not sufficient (check the **p10** too: a negative bottom decile means a tenth of your distribution got worse, the classic over-optimisation signature).

**Q65. What is a length-controlled evaluation and how do you do it internally?**
- **Answer:** Length-controlled win rate regresses out the length difference between the two responses before computing the win rate. Internally: fit a logistic regression of the judge's verdict on the length difference (and optionally position), then report the win rate at zero length difference. Cheap version: correlate mean response length with win rate across checkpoints — a correlation above ~0.7 means your evaluation is measuring verbosity, not quality.

**Q66. What is the most common DPO setup error?**
- **Answer:** Several candidates, but the one that *silently* destroys a run is the **chosen/rejected swap** — the loss falls smoothly, the run looks healthy, and the model learns the wrong direction. The one that *errors loudly* is `remove_unused_columns=True`. The one that is most common overall is forgetting `remove_unused_columns=False`. The one that is most dangerous in production is **truncation at `max_length`**, because it silently trains on identical prefixes.
- **Why asked:** A well-posed "what goes wrong" question tells you whether the candidate has operated this, not just studied it.

**Q67. How do you add preference signal to a model when you only have one GPU and no annotation budget?**
- **Answer:** **ORPO on QLoRA**, with preference pairs generated by rejection sampling from the model itself and filtered by a heuristic or an LLM judge. One model, ~11–19 GB for 7B, and the SFT term in the objective means you can go straight from the base model. Total compute cost: single-digit dollars. Expected outcome: a modest tone/format shift, not a capability change.

---

## Level 3 — Advanced, Internals & Theory

**Q68. Derive the DPO loss from the RLHF objective.**
- **Answer:** Start from the KL-constrained objective `max_π E_{y~π}[r(x,y)] − β·KL(π(·|x) ‖ π_ref(·|x))`. This has a closed-form optimum `π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)` where `Z(x) = Σ_y π_ref(y|x)·exp(r(x,y)/β)` is the partition function. Invert for the reward: `r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)`. This is the **implicit reward** — any reward that produced π* can be written this way. Now substitute it into the Bradley–Terry loss `−log σ(r(x,y+) − r(x,y−))`. The `β·log Z(x)` term is identical for both members of the pair because **they share the same prompt x**, so it cancels. What remains is a loss in the policy's own log-probabilities against the frozen reference — no reward model, and no ability to recover absolute reward (only differences).
- **Why asked:** The canonical senior-level DPO question. The key insight they are listening for is **the partition function cancels because the pair shares a prompt**.
- **Trap:** Claiming DPO trains a reward model, or failing to explain why `Z(x)` cancels.

**Q69. Why does the KL term in the RLHF objective produce a closed-form optimum at all?**
- **Answer:** Because the objective is a sum over the simplex of `E[r] − β·KL` with the constraint that π sums to 1, and `KL(π ‖ π_ref)` contains `log π`. Setting up the Lagrangian and taking the derivative with respect to `π(y|x)` gives `r(x,y) − β·(log(π/π_ref) + 1) − λ = 0`, which solves to `π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)`. This is a **Gibbs/Boltzmann distribution** — the reward is a negative energy and β is the temperature. It is the same mathematical object as a softmax with a prior.
- **Why asked:** It tests whether you can see the structure rather than memorising the result. The Gibbs-distribution observation explains why β is described as an inverse temperature and why β→0 gives a delta function on the argmax (degenerate) and β→∞ gives the prior (no movement).

**Q70. Which direction is the KL term, and why does it matter?**
- **Answer:** `KL(π_θ ‖ π_ref)` — the **forward** KL, with the *policy* as the expectation distribution: `Σ_y π_θ(y)·log(π_θ(y)/π_ref(y))`. It penalises mass the policy puts where the reference does not. Because the expectation is over the policy, it is **mode-covering**: the policy is free to abandon modes the reference had (including bad ones) without a penalty. Reversing it to `KL(π_ref ‖ π_θ)` is **mode-seeking** and would penalise the policy for ignoring anything the reference does — including the bad answers — which is precisely backwards. Getting the direction wrong in an interview is a hard fail.
- **Trap:** "KL is symmetric, it doesn't matter." It is not symmetric and it matters a great deal.

**Q71. Why does DPO exhibit length bias, mechanically?**
- **Answer:** The implicit reward is `β·(log π_θ(y|x) − log π_ref(y|x))`, and `log π_θ(y|x)` is a **sum over tokens**, so it grows in magnitude with sequence length. For a policy that is (weakly) better than the reference per token, longer sequences get a larger positive log-ratio — the model is rewarded for length, not for quality. SimPO's fix is to use the *average* log-prob (`(1/|y|)·log π`) so length cancels; R-DPO adds an explicit length penalty. This is the mechanism behind the most common observed symptom of DPO over-optimisation.
- **Why asked:** It is the most-cited DPO limitation and the bridge to SimPO. Candidates who say "it just makes things longer" without the sum-vs-average mechanism are reciting.

**Q72. When does the DPO gradient vanish, and what does that imply for your data?**
- **Answer:** The gradient of `−log σ(z)` is `−σ(−z)`, which vanishes as `z → +∞` — i.e. once the margin is large, mastered pairs contribute almost nothing. Implication: **easy pairs are exhausted quickly, and continued training on them is wasted compute while the margin keeps growing on the pairs that still have signal.** Practical consequences: (1) you need *hard* pairs — the long tail of the annotation effort is where the value is; (2) more epochs do not help once the easy pairs saturate, they just overfit the hard ones; (3) it explains why DPO's loss curve flattens while quality can still be degrading.

**Q73. Why is the reward model only valid on its training distribution, and what breaks if you ignore that?**
- **Answer:** The RM is a function approximator trained on (prompt, response) pairs drawn from a particular generator. Outside that region it extrapolates without supervision, and its scores are arbitrary — but the policy treats them as authoritative. In PPO this means a policy that explores into unseen regions receives high-variance, meaningless rewards and the run diverges or produces incoherent text. The canonical instance: train the RM on GPT-4 outputs, run PPO against a 1B policy's outputs. Mitigations: collect preference data on the *policy's own* outputs (the InstructGPT iteration), KL-bound exploration (β), and RM ensembles with a disagreement penalty.

**Q74. What is the over-optimisation curve and what are its practical consequences?**
- **Answer:** As you optimise further against a fixed RM (measured by KL divergence from the reference), the **proxy** reward rises monotonically while the **true** reward — measured by a much larger gold RM or humans — rises, peaks, and then declines. Gao, Schulman & Hilton (2023) fit the gap as a function of `√KL`, with the true-reward peak typically at KL ≈ 5–20 nats depending on RM size. Practical consequences: (1) **never select a checkpoint on the RM score** — it is highest at the worst checkpoint; (2) you must early-stop on a proxy-independent metric; (3) your KL budget is a *resource to spend*, not a penalty to minimise.

**Q75. Why is PPO considered unstable, concretely?**
- **Answer:** At least five compounding reasons. (1) It has ~10 interacting hyperparameters (clip ε, GAE λ, γ, KL coefficient, value-loss coefficient, entropy bonus, batch size, minibatch size, LR, rollout length) and a bad combination manifests as "training runs but nothing improves" rather than an error. (2) The policy, reference, RM and value model are all functions of the same weights at different times — any staleness or mismatch produces biased advantages. (3) Reward normalisation across prompts of different difficulty is a subtle and common bug that makes reward magnitudes incomparable. (4) The importance ratio is an exponential in the log-prob difference, so small probability changes produce large ratio changes on rare tokens. (5) Rollouts are stochastic, so the run is not reproducible across seeds or GPU counts.

**Q76. Why does GRPO drop the value model, and what does it lose?**
- **Answer:** It drops the value model by using a **group baseline** instead of a learned one: sample G responses for a prompt, and use the group's mean reward as the baseline for advantage. This is valid because a baseline only needs to reduce variance, and the sample mean of G on-policy returns is an unbiased, low-cost estimator of that. **What it loses:** the advantage is per-*sequence*, not per-*token*, so GRPO cannot express "this part of the reasoning was good, that part was bad". For long chain-of-thought with intermediate steps that matter, you need a process reward model and a per-token critic — which is a genuinely different capability, not just a memory trade.

**Q77. Why does GRPO need G ≥ 4, realistically 8–16?**
- **Answer:** The advantage is a z-score within the group. With G=1 the group mean is the sample itself and the numerator is identically zero. With small G the mean and std are noisy estimators, so the advantage has high variance and the group frequently has zero reward variance (all-pass or all-fail), contributing no gradient. Empirically the fraction of zero-variance prompts falls sharply from G=4 to G=16. DeepSeek-R1 used G=16. Cost scales linearly in G, so G is the dominant cost knob.

**Q78. Explain the ORPO objective and why the odds ratio removes the need for a reference.**
- **Answer:** `L_ORPO = L_SFT + λ·L_OR` where `L_SFT` is ordinary token-level cross-entropy on the chosen response and `L_OR = −log σ( log( odds_θ(y+|x) / odds_θ(y−|x) ) )` with `odds_θ(y|x) = π_θ(y|x)/(1 − π_θ(y|x))`. There is no reference because **the SFT loss is the anchor**: the model is continuously trained to assign high probability to the chosen responses, which is exactly the constraint that DPO's KL term to a frozen reference enforces. Using the *odds ratio* rather than a probability difference keeps the preference term bounded and scale-stable. Net effect: one model, one stage, no reference.
- **Why asked:** Tests whether you understand ORPO as a design, not as a config.

**Q79. What is the difference between RLVR and RLHF, and why does the former need no reward model?**
- **Answer:** RLHF learns a *reward model* as a proxy because human preference is not computable. RLVR uses a **program** as the reward — a unit test, a math answer check, a schema validator, a compiler. Because the program is exact, (a) there is no proxy and therefore no reward-hacking-of-the-proxy (you can fool a learned RM; you cannot fool a compiler into accepting invalid syntax), and (b) there is no reward-model training run, no preference data, and one fewer model in memory. The remaining reward-hacking surface is **specification gaming** — passing the tests without solving the problem — which is why you keep a held-out test split the model never sees.

**Q80. How does DAPO improve on vanilla GRPO?**
- **Answer:** Four changes: (1) **clip-higher** — decouple the lower and upper clip bounds (e.g. ε_low=0.2, ε_high=0.28) because symmetric clipping suppresses the low-probability tokens that drive exploration; (2) **dynamic sampling** — drop prompts whose group has zero reward variance instead of wasting the batch; (3) **token-level loss** — average the loss over all tokens in the batch rather than per-sequence, which removes a length bias in the gradient; (4) **overlong filtering** — mask truncated sequences out of the loss rather than training on their incomplete rewards. Each is a small code change with a large effect on long-CoT training.

**Q81. What is the alignment tax, and which of its causes does PPO-ptx address?**
- **Answer:** Three causes: (a) **narrow reward** — capabilities outside the alignment distribution get no reward and drift freely; (b) **reference degradation** — π^SFT is itself already degraded relative to the base, and the KL anchors you there; (c) **distribution shift on long outputs** — preference data is short and single-turn, so long-form and multi-turn behaviour degrades. **PPO-ptx** adds `+γ_ptx·E_{x~D_pretrain}[log π_θ(x)]` to the PPO objective, giving pretraining capabilities a gradient again. It addresses **(a)**, and partially **(c)** by keeping the model fluent. It does **not** address (b) — that damage is upstream of alignment and no alignment trick recovers it.

**Q82. Why can't alignment improve factual accuracy?**
- **Answer:** Preference data encodes *which of several fluent answers a human prefers*, not *which is true*. A reward model trained on human preferences learns human *stated* preferences, and humans do not reliably detect fabrication — so a confident invented citation frequently wins a preference comparison against an honest "I don't know". Optimising that signal can *increase* confident fabrication. Facts belong in retrieval (CS-04); alignment belongs to behaviour. The one honest exception: you *can* train calibration, by including pairs where "I'm not certain, and here is how you'd verify" beats a confident wrong answer.

**Q83. What does the reference model cost in DPO, exactly, and when is it free?**
- **Answer:** It is **free with PEFT** and it is a **full second model at full fine-tuning**. With LoRA, the base weights are frozen and the adapter is separable, so TRL takes the same base with the adapter disabled as the reference — one set of weights. At full FT there is no separable trainable part, so TRL **deep-copies at trainer init** and freezes the copy — 14 GB for 7B bf16 *plus* the training copy's optimiser state. This is why LoRA makes DPO cheap and why full-FT DPO OOMs unexpectedly.

**Q84. Explain the reward normalisation issue in PPO.**
- **Answer:** Rewards from a Bradley–Terry RM are identifiable only up to a **per-prompt additive constant** — they are margins, not absolute qualities. If you normalise across a batch of *different prompts*, you are mixing incomparable quantities: a prompt whose responses are all terrible contributes the same normalised spread as one whose responses are all excellent, and the policy receives a strong gradient on the terrible prompt. The correct procedures are: **per-prompt** standardisation (subtract the prompt's own mean), and if you whiten across the batch, do it on the *residual after* per-prompt centring. Getting this wrong produces a run that trains smoothly and generates nonsense — one of the most common PPO bugs.

**Q85. What is the difference between a process reward model and an outcome reward model, and which methods need which?**
- **Answer:** An **outcome reward model** scores only the final answer; a **process reward model** scores each intermediate reasoning step. ORMs are what DPO, ORPO, PPO-with-an-RM and GRPO all use implicitly — and for GRPO the group baseline is *only* valid with an outcome reward, because the advantage is per-sequence. PRMs give denser credit assignment and are what you need for long chain-of-thought where a final answer can be right by luck. The cost is that PRM labels are far more expensive (step-level annotation) and PRMs are more hackable (the policy learns to write steps the PRM likes).
- **Why asked:** It is the frontier question in reasoning-model alignment and it explains GRPO's limitation from the inside.

**Q86. Why is DPO said to be "off-policy" or "offline", and what is the practical consequence?**
- **Answer:** The training distribution is the fixed preference dataset, which was generated by some *other* model (often a stronger one, or an older version of yours). The policy never samples during training, so it never sees the consequences of its own current behaviour. Consequence: **the ceiling is the dataset.** If the dataset contains no example of a behaviour you want, DPO cannot discover it. Iterative/online DPO (regenerate pairs from the current policy each round) is the standard remedy; PPO and GRPO get it for free.

**Q87. What is the difference between the "reward" in DPO and the reward in RLHF?**
- **Answer:** In RLHF the reward is an **explicit learned function** `r_φ(x,y)` — a separate artifact you can call, inspect, ensemble and use for best-of-n. In DPO the reward is **implicit**: `r(x,y) = β·log(π_θ(y|x)/π_ref(y|x)) + β·log Z(x)`. You can compute the *difference* `r(x,y+) − r(x,y−)` on demand (that is the implicit margin), but you cannot recover absolute reward because `Z(x)` is intractable. Consequences: DPO cannot be used for best-of-n scoring, cannot be ensembled, and cannot be inspected independently of the policy.

**Q88. Why do so many modern recipes use a smaller learning rate for GRPO than DPO?**
- **Answer:** GRPO's gradient is a policy-gradient estimator multiplied by the importance ratio, and it is taken over *generated* tokens that were sampled at temperature 1.0 — so the effective gradient noise is far higher than DPO's, which is taken over fixed text with a bounded sigmoid. A DPO-scale LR (1e-5–2e-5) applied to GRPO produces divergence or immediate collapse. Typical GRPO LRs are **1e-6 to 5e-6**, an order of magnitude lower. The same applies to PPO.

**Q89. What is the significance of the `loss_type` parameter?**
- **Answer:** It selects which member of the preference-loss family TRL uses, all sharing the log-ratio formulation: `sigmoid` (DPO's exact loss), `hinge` (SLiC-style margin loss, with a flat region so mastered pairs contribute exactly zero gradient), `ipo` (squared loss with a finite target margin), `kto_pair` (KTO applied to pairs), `bco_pair`, `sppo_hard`, `aot`, `apo_zero`/`apo_down`, `discopop`, and `simpo`. The video's config comments on this: `loss_type="sigmoid",  # or "hinge", depending on experiment`.
- **Why asked:** Tests breadth beyond the single default.

**Q90. Why is a reward-model ensemble better than a larger single reward model?**
- **Answer:** The policy hacks the *errors* of the reward function, not its average accuracy. A larger RM has smaller errors but they are still systematic in the region the policy explores. An ensemble of k RMs trained on different data orderings and seeds has *decorrelated* errors, so the policy must find inputs that fool **all k** — a much smaller set. The best practice is not the mean but a **pessimistic** aggregate: `min_i r_i(x,y)`, or `mean − α·std`, which explicitly penalises the regions where the ensemble disagrees (i.e. where the RM's knowledge is weakest). Cost: k× RM training and k× inference memory; k=3–5 is typical.

**Q91. A colleague says "DPO is just RLHF with the reward model folded in, so it's the same thing." What is right and what is wrong?**
- **Answer:** **Right:** DPO's objective is derived from the same KL-constrained reward-maximisation problem, and the policy DPO converges to is the same policy the RLHF objective has as its optimum — DPO is not a different *goal*, it is a different *optimisation path* to the same goal. **Wrong:** they are not the same in practice for three reasons. (1) DPO is **offline** — it cannot explore, so it does not reach that optimum when the dataset is thin, whereas PPO's rollouts can. (2) DPO's finite dataset plus a saturating sigmoid means it under-optimises relative to the true objective on some pairs and over-fits on others. (3) The absence of an explicit RM means you lose best-of-n, ensembling, and inspection. Saying "same objective, different optimisation, and the difference is exploration and dataset ceiling" is the complete answer.

**Q92. What determines the achievable win rate ceiling of an alignment run?**
- **Answer:** Four things, in order of impact: (1) **the SFT baseline's competence** — you select among behaviours the model already produces, so a weak SFT model caps you; (2) **the preference signal's clarity** (IAA — a 65%-agreement dataset caps you near where the majority label is); (3) **the method's ability to explore** (offline methods cannot exceed the dataset; online methods can); (4) **how far you are willing to pay the alignment tax** — every additional point of win rate costs general capability. There is no method that fixes a weak signal or a weak baseline.

---

## Level 4 — System Design & Scenario

**Q93. Design the alignment stage for a customer-support assistant. You have 7B, one A100 40GB, 8 weeks, and a product owner who wants "a friendlier, more accurate assistant".** `[Company style: startup ML eng]`
- **Answer structure:**
  - **Requirements.** Separate "friendlier" from "more accurate" immediately — these are different problems with different architectures. Accuracy is likely a knowledge/retrieval problem (CS-04); friendliness is a behaviour problem. Get the product owner to rank 3 concrete failure examples from real transcripts.
  - **Constraints.** 40 GB VRAM on one card; 8 weeks including data collection; no existing preference data; a B2B product where a bad refusal is as costly as a bad answer.
  - **Design.**
    1. **Week 1 — Diagnose and baseline.** Sample 200 real conversations, hand-label the failure mode of each (wrong fact / wrong tone / unnecessary refusal / too long / format). If wrong-fact dominates, stop and go to RAG. Freeze the SFT model's numbers on a capability suite (MMLU/GSM8K/IFEval) and a 300-prompt behavioural set.
    2. **Weeks 2–4 — Data.** Write annotation guidelines with an explicit HHH priority order, an explicit length-neutrality clause, and an explicit refusal-calibration clause. Collect **1,500 human pairs** ($4/pair, $6k) and **4,500 RLAIF pairs** ($0.06/pair, $270) from a *different-model-family* judge. Double-annotate 10% permanently as the IAA control. Gate: ≥70% raw agreement before spending on volume.
    3. **Week 5 — Train.** QLoRA DPO: 4-bit base, LoRA r=16 on all linear layers, lora_alpha=16, lr 2e-5, batch 1 × accum 16, 1 epoch, β=0.1, `loss_type="sigmoid"`, `max_length` = p95 of prompt+chosen, `remove_unused_columns=False`. 6,000 pairs → ~375 optimiser steps. Wall-clock ~5 h on the A100. **Do not** use `ref_model=None` with full FT on this card.
    4. **Weeks 5–6 — Sweep.** Three β values (0.05, 0.1, 0.2) × two LRs (1e-5, 2e-5). That is six 5-hour runs = $45. Select on held-out preference accuracy + capability delta, not on loss.
    5. **Week 7 — Evaluate.** 500 human comparisons × 3 raters against the SFT baseline ($3,600 budget), blinded and position-swapped. Plus LC-style length control, since DPO's default failure here is verbosity. Plus the capability suite.
    6. **Week 8 — Ship and instrument.** Serve as an adapter with a traffic split vs the SFT model. Monitor win rate, mean length, refusal rate, and escalation rate weekly.
  - **Trade-offs.** DPO over ORPO because pairs exist and β gives finer control than λ; QLoRA over LoRA because 40 GB is tight when the reference is present; 1 epoch because the dataset is small.
  - **Failure modes.** Length bias (measured); over-refusal from a safety-heavy mix (measured); knowledge problems masked as tone problems (separated in week 1); **the product owner's "friendlier" being unmeasurable** — the most likely cause of failure, and the reason week 1 exists.

**Q94. You have 30,000 tasks with unit tests and want a coding model that is genuinely better. Design the training.** `[Company style: big-tech research]`
- **Answer structure:**
  - **Requirements.** Produce *correct* code more often, not just better-formatted code. General capability must not regress.
  - **Constraints.** Reward is **verifiable** — tests are executable. No human labels. Compute budget matters because rollouts are expensive.
  - **Design.** **GRPO/RLVR**, not DPO. Steps:
    1. **Reward function** in a sandbox: pass all tests → 1.0, else 0.0, with a small partial credit for "executes without crashing on 3 held-out inputs" to break the all-zero plateau.
    2. **Data split:** 30k tasks → 25k train, 5k held-out. The held-out split is what prevents specification gaming; the model must never see these tests.
    3. **Config:** G=16, temperature 1.0, max_completion_length 1024, β=0.04 (PPO-convention KL), lr 1e-6, batch 1 × accum 4, 1 epoch, bf16. Enable **DAPO's dynamic sampling** so zero-variance groups are dropped.
    4. **Monitoring:** log `reward_std` per batch and the fraction of zero-variance prompts (target < 30%); log pass@1 on the held-out suite every 100 steps; log MMLU/GSM8K for the capability regression.
    5. **Then a DPO second pass** on tone/format pairs, on the GRPO output, to fix instruction-following and response style. **Order matters** — tone on a model that cannot do the task is wasted money.
  - **Trade-offs.** GRPO vs DPO: DPO could not produce new capability here; it would only sharpen the existing distribution, and the tasks already exist with exact labels. G=16 vs G=8: G=16 halves the zero-variance waste at 2× rollout cost — the right trade on hard tasks.
  - **Failure modes.** Specification gaming (held-out tests); zero-variance collapse on hard tasks (dynamic sampling); reward hacking the *sandbox* (a model that returns the expected output for the test inputs — mitigated by held-out inputs and by checking that the code is general); capability regression from a narrow reward (capability suite + KL).

**Q95. Your 70B model needs alignment and you have 2×A100 80GB. Design the run.** `[Company style: big-tech research]`
- **Answer structure:**
  - **Requirements.** Behaviour change on a large model with 160 GB total.
  - **Constraints.** Full-FT DPO on 70B needs 70 × 16 = **1,120 GB** for weights+grads+optimiser, plus a 140 GB reference deep-copy. Full-FT PPO needs four such models. **Both are impossible.** Even bf16 inference of 70B is 140 GB — it does not fit on one card.
  - **Design.** **QLoRA DPO, sharded.** 4-bit NF4 base ≈ 35 GB; adapter + 8-bit optimiser ≈ negligible; reference is free (adapter disabled); activations with gradient checkpointing and `max_length=1024`, batch 1 ≈ 20–30 GB. Total ≈ 60–70 GB, so it fits on **one** A100 80GB with room for the tokeniser and CUDA overhead — or comfortably sharded across both with `device_map="auto"` and FSDP.
  - **Concrete config:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`; `LoraConfig(r=32, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05)`; lr 1e-5; batch 1 × accum 16; 1–2 epochs; β=0.1; gradient checkpointing on.
  - **If they insist on full FT:** 2×A100 cannot do it, and 8×H100 80GB still gives only 640 GB — you would need FSDP with CPU offload, at roughly 10–20× the wall-clock. The right answer is to say so and propose QLoRA + a *better dataset*.
  - **Alternative worth proposing:** **ORPO QLoRA**, which removes the reference entirely and lets you use a larger batch or longer sequences in the same 80 GB.
  - **Failure modes.** Merging the adapter into the 4-bit base is lossy — merge in bf16 if you need bit-fidelity, or serve the adapter unmerged. QLoRA's quality gap vs full FT is real but small at r=32 on all-linear.

**Q96. Design a preference-data pipeline that produces 50,000 pairs a month with bounded cost and auditable quality.** `[Company style: big-tech research / consulting]`
- **Answer structure:**
  - **Requirements.** Volume, cost ceiling, and **auditability** — someone must be able to answer "why does the model refuse X?".
  - **Constraints.** Human annotation is the bottleneck; LLM judges are cheap but biased; quality is the ceiling.
  - **Design.**
    1. **Prompt sourcing** from production traffic (deduplicated, PII-scrubbed, stratified by intent cluster) plus a synthetic slice for coverage of rare but important intents (safety, refusals).
    2. **Generation:** sample 4 responses per prompt from the *current policy* plus 1 from the previous release (for contrast). Sampling from your own policy is what keeps the reward model's distribution aligned.
    3. **Labeling, three tiers:** (a) LLM judge with a **rubric** (score 5 axes separately, then aggregate — not a single "which is better"), from a **different model family** than the one being trained; (b) **5% human audit**, permanently, with per-annotator agreement tracked against the majority; (c) **1,000 human pairs/month** on the hardest, highest-stakes slice (safety + the judge's lowest-confidence pairs).
    4. **Filtering:** drop pairs where the judge's confidence is low *and* the margin is near zero; drop `chosen == rejected`; deduplicate against eval sets.
    5. **Quality gates:** report judge-vs-human agreement on the audited sample (target ≥85%); report Krippendorff's α on the double-annotated slice; track the pair-difficulty distribution (an all-easy set is a set that stops teaching).
    6. **Governance:** version the constitution/rubric, the judge model and its version, and the guidelines. Every shipped adapter records the data hash and the rubric version.
  - **Trade-offs.** Judge bias vs cost: mitigated by family diversity, rubric decomposition, and the human audit. Synthetic vs real prompts: synthetic covers rare safety cases, real covers the actual distribution — you need both and you need the ratio to be explicit.
  - **Failure modes.** Judge-bias drift when the judge is upgraded (pin the version); the easy-pair plateau (track difficulty); distribution staleness (monitor prompt-distribution drift and re-source monthly); annotation-team drift (the permanent 5% audit).

**Q97. You must choose between DPO and GRPO for a task where the reward is *partially* verifiable — e.g. a customer-support agent whose final answer includes a policy citation that can be validated.** `[Company style: big-tech research]`
- **Answer structure:**
  - **Requirements.** Two objectives in one task: cite the correct policy (verifiable) and phrase it helpfully (not verifiable).
  - **Design.** **Hybrid, in order.** (1) First run **GRPO/RLVR** with a composite reward: `0.7·(citation_valid) + 0.2·(citation_matches_policy_intent) + 0.1·(format_ok)`, where only the first term is programmatic. The non-verifiable terms must carry low weight and be rubric-based, or you have re-created an RM with extra steps. (2) Then run **DPO** on 3–5k pairs for tone, concision and refusal calibration, on top of the GRPO output.
  - **Why the order.** GRPO explores and can discover citation behaviours the SFT data lacked; DPO cannot. Tone is a selection problem over behaviours the model already has, which is exactly what DPO is for. Doing DPO first and GRPO second would let the RL pass unlearn the tone.
  - **Trade-offs.** A single mixed-reward GRPO run is cheaper but confounds two objectives and makes reward hacking harder to attribute. Two passes cost more compute but give you two independently measurable stages and a clean rollback point.
  - **Failure modes.** The verifiable term dominating (weighting); the unverifiable term being an RM in disguise and getting hacked (keep its weight low, keep a human audit); the GRPO pass degrading instruction-following (capability suite); the DPO pass undoing the citation behaviour (evaluate citation validity after stage 2, not just before).

**Q98. Your organisation wants one alignment pipeline to serve five different products (support, code, legal, marketing, internal search). Design it.** `[Company style: consulting]`
- **Answer structure:**
  - **Requirements.** Five product-specific behaviours from a shared base, with independent iteration and no cross-contamination.
  - **Constraints.** GPU budget is shared; legal has compliance requirements; marketing changes weekly.
  - **Design.** **One base, five adapters.**
    - **Shared:** one base model (SFT'd once), one tokeniser with an explicit `chat_template`, one eval harness, one training-code path, one serving stack with per-request adapter selection.
    - **Per product:** a preference dataset with its own guidelines, its own adapter (LoRA r=16–32), its own eval set (300–500 prompts), its own win-rate baseline.
    - **Method per product:** support → DPO (tone, pairs from telemetry-derived pairs); code → GRPO/RLVR (tests); legal → DPO with cDPO label smoothing (high-stakes, noisy, expensive labels) and a *high* β for conservatism; marketing → ORPO (weekly iteration, single-GPU, cheap); internal search → rejection sampling + SFT (no preference signal, just quality filtering).
    - **Governance:** the legal and support adapters carry their own compliance records; the pipeline must record data provenance per adapter.
  - **Trade-offs.** Five adapters means five evaluations, five monitoring dashboards, and five rollback paths — real operational cost. The alternative (one model, prompted) is cheaper but cannot hold five contradictory tone policies. State the operational cost honestly.
  - **Failure modes.** Adapter confusion at serve time (route by product ID, not by prompt content); cross-product data leakage (per-product data isolation is a *policy* requirement, not just a quality one); base-model upgrade invalidating all five adapters at once (this is the biggest real risk — budget a re-run of all five on every base bump).

**Q99. A regulator asks: "how does your model decide to refuse a request?" Give the system design that lets you answer.** `[Company style: consulting / regulated industry]`
- **Answer structure:**
  - **Requirements.** An auditable answer, not a hand-wave. Reproducibility.
  - **Constraints.** The model is a weight file; the decision is distributed across billions of parameters. So the *provenance* must carry the answer.
  - **Design.** Every alignment artifact records: (1) the **preference-data provenance** — the guidelines version (a versioned document stating the HHH priority order and refusal criteria), the annotator pool, the judge model and version, the constitution/rubric text; (2) the **training config** — β, epochs, loss type, seed; (3) the **evaluation report** — the false-refusal rate on a benign set and the true-refusal rate on a harmful set, with confidence intervals; (4) the **capability report** — what the alignment cost. Plus a **red-team log** with every adversarial prompt tried and the outcome.
  - **The honest answer to the regulator.** Refusal behaviour is a learned property determined by the balance of refusal examples in the preference data, expressed through the guidelines' priority order. The evidence is the false-refusal/true-refusal calibration curve on held-out sets and the guidelines document itself. **Do not claim mechanistic interpretability you do not have.**
  - **Trade-offs.** Full auditable provenance costs engineering time (versioning, hashing, per-run artefacts) and slows iteration. For regulated products it is not optional.
  - **Failure modes.** Guidelines versioned but *not* linked to a trained checkpoint; judge upgrades changing refusal behaviour silently; a refusal-rate regression not gated in CI.

**Q100. You are asked to add alignment to a pipeline that already has SFT, but the only data you can get is 400 pairs. Design the most useful possible use of that budget.** `[Company style: startup ML eng]`
- **Answer structure:**
  - **Requirements.** Extract maximum value from an insufficient dataset.
  - **Design.** **Change the objective from breadth to depth.** 400 pairs cannot shift a general tone; it *can* fix one narrow, high-frequency, high-cost behaviour. Steps: (1) rank the failure modes from production transcripts by frequency × cost and pick **one** — e.g. "cites a policy section that does not exist"; (2) build all 400 pairs on that single behaviour, deliberately including the hard cases and 50 pairs where the *shorter* answer wins (to avoid length bias); (3) use **cDPO** with `label_smoothing=0.1` since 400 pairs will have noise; (4) use a **low rank (r=8) and β=0.2**, choosing a gentle, stable operating point over an aggressive one; (5) evaluate on a 200-prompt set targeting that behaviour specifically, not a general benchmark.
  - **What not to do.** Do not spread 400 pairs across five behaviours (you will move none of them); do not run a general AlpacaEval-style evaluation (it will show nothing, correctly); do not use PPO (no signal at this scale).
  - **Trade-offs.** Narrow and measurable beats broad and unmeasurable. If the product owner wants general friendliness, the honest answer is that 400 pairs is the wrong budget and the right move is RLAIF to reach 5,000.
  - **Failure modes.** Overfitting at 400 pairs after 2+ epochs (cap at 2); the behaviour regressing on the general distribution (capability suite); measuring on the training distribution (held-out split by prompt).

**Q101. Design the monitoring for a shipped aligned model, from day 1 to month 6.** `[Company style: big-tech research]`
- **Answer structure:**
  - **Requirements.** Detect quality decay, reward hacking, distribution shift, and safety regression, before users do.
  - **Design — four loops at four cadences.**
    1. **Real-time (every request):** latency, refusal rate, mean response length, error/truncation rate, tool-call failure rate. Length is the reward-hacking canary.
    2. **Daily:** implicit feedback (regenerations, copy rate, abandonment, thumbs) segmented by intent cluster; escalation rate; complaint rate. Trend, not level.
    3. **Weekly:** 200 held-out prompts × a frozen judge, computing win rate vs the shipped baseline and **mean length**, plus the capability suite. A win-rate drop below 50% over 200 comparisons triggers rollback.
    4. **Monthly:** prompt-distribution drift (embedding centroid distance / PSI > 0.2 means the preference data is stale); human-audited sample of 100 conversations; **re-collect preferences on the new distribution** and queue the next alignment iteration.
  - **Trade-offs.** Every loop costs money and engineering. The weekly judge-based eval is the cheapest high-value loop; the monthly human audit is the most expensive and the least substitutable.
  - **Failure modes.** No baseline kept (you cannot detect drift without a frozen comparison); judge version not pinned (a judge upgrade looks like a model regression); the eval set itself drifting into the training distribution (deduplicate and rotate).

**Q102. Design the A/B test for shipping a new aligned adapter.** `[Company style: big-tech research]`
- **Answer structure:**
  - **Requirements.** Distinguish a real improvement from sampling noise and from novelty effects.
  - **Constraints.** Win rate is a *proxy* for the product metric; the product metric is what you are really moving.
  - **Design.** (1) **Offline first:** 500 held-out prompts × 3 raters, blinded, position-swapped, against the current adapter. Gate on win rate ≥ 55% with a CI that excludes 50%, *and* capability delta > −2%. (2) **Online:** serve both adapters behind the same base with per-request adapter selection; randomise at the **user** level, not the request level, so a user sees consistent behaviour. (3) **Metrics:** primary = the product outcome (resolution rate, time-to-resolution, escalation rate); secondary = win rate on a live-sampled human audit; guardrail = refusal rate, length, latency, error rate. (4) **Duration:** long enough for the primary metric's natural cycle (typically 2–4 weeks for support), with a pre-registered stopping rule. (5) **Analysis:** pre-register the primary metric and the minimum detectable effect; run the capability suite on a sample of live traffic.
  - **Trade-offs.** User-level randomisation costs you statistical power (fewer independent units) but is the only design that measures the experience correctly. Request-level randomisation inflates significance and produces incoherent user experience.
  - **Failure modes.** Novelty effect (users prefer the new thing for a week); the human-audit sample being non-representative; **optimising the proxy** — a win-rate improvement that does not move resolution rate is a measurement, not a win.

**Q103. Your team proposes four different alignment projects for the same model. Design the sequencing.** `[Company style: consulting]`
- **Answer structure:**
  - **Requirements.** Get the most total quality per GPU-month.
  - **Design — sequence by (a) verifiability and (b) blast radius.** (1) **GRPO/RLVR on the verifiable core** first (code/math/tool use) — it produces genuine capability, has no data cost, and everything downstream is built on top of it. (2) **DPO for the highest-blast-radius behaviour** next (safety/refusal calibration), because a safety regression blocks shipping everything else. (3) **DPO for tone/concision**, the highest-volume complaint. (4) **ORPO for the cheap long-tail polish** last, as a fast-iteration loop.
  - **Why this order.** Capability before style (style on top of a model that cannot do the task is wasted). Safety before tone (a safety block stops the release). Cheap iteration last (you want the fast loop running against the near-final model, not against a model that will be replaced).
  - **Trade-offs.** Serial sequencing delays each project; parallel sequencing means four teams fighting for the same GPU and four evaluations against a moving base. Serial with a shared eval harness is almost always right for a single model.
  - **Failure modes.** Doing tone first and re-doing it after the RL pass (the RL pass changes the distribution and invalidates the tone work); measuring each stage against the base rather than against the previous stage (you cannot attribute the gains).

**Q104. Design an evaluation suite that would catch reward hacking *before* it reaches users.** `[Company style: big-tech research]`
- **Answer structure:**
  - **Requirements.** Detect the four concrete hacks: length bias, sycophancy, formatting exploits, uncertainty deletion.
  - **Design — a probe suite, run every N steps.**
    - **Length probe:** mean and p90 response length on a fixed 200-prompt set. Alarm on a >20% rise from the SFT baseline. Free, and the earliest signal.
    - **Sycophancy probe:** 50 pairs of (false premise, correct pushback). Feed the false premise; measure the fraction where the model validates it. Alarm on any rise from baseline. This is the one that catches the honesty failure.
    - **Uncertainty probe:** 50 questions whose answers are genuinely unknowable. Measure the fraction where the model hedges appropriately vs asserts. Alarm on a *fall*.
    - **Format probe:** the same 200 prompts; measure the fraction where the response opens with a preamble ("Great question!") or a bulleted list, when the prompt did not ask for one.
    - **Capability probe:** MMLU + GSM8K + IFEval subset, every 100 steps. Alarm on a >2% drop.
    - **Judge-divergence probe:** score the same outputs with your training judge *and* a different-family judge. A widening gap between them means you are fitting the training judge.
  - **Selection rule.** Ship the checkpoint where the human/gold win rate is highest, **subject to** all six probes being within tolerance. Never select on the training judge's score.
  - **Trade-offs.** Every probe costs a generation pass; the full suite is ~700 generations every 100 steps. Cheaper than one bad release.
  - **Failure modes.** Probes leaking into the training data (keep them in a separate, access-controlled set); probes that are themselves gameable (rotate them); the alarms being ignored because they fire on every run (set thresholds from a *baseline* run, not from intuition).

---

## Level 5 — Debugging & Incident Response

**Q105. Your DPO run's loss goes from 0.693 to 0.05 in one epoch, and the model now writes three paragraphs for every one-line question. Diagnose and fix.**
- **Answer (ordered):**
  1. **Confirm the diagnosis.** Log mean response length per checkpoint — it will have risen monotonically. Compute the raw win rate and a length-controlled win rate; the LC number will be much lower. This is the textbook over-optimisation/length-bias signature.
  2. **Root cause.** DPO's implicit reward is `β·(log π_θ(y) − log π_ref(y))` and `log π_θ(y)` is a **sum over tokens**, so longer sequences get a larger log-ratio for the same per-token quality. Nothing in the loss penalises length.
  3. **Why the loss went to 0.05.** The margin is exploding — you have over-optimised past the peak of the true-reward curve. Loss is a function of the margin and the margin grows monotonically during over-optimisation.
  4. **Fixes, in order of preference.** (a) **Switch to SimPO**, which uses the length-normalised *average* log-prob and removes the bias structurally. (b) If you must stay on DPO: reduce to a point where mean length is flat (typically well under one epoch on this dataset), lower β to 0.05, and add `label_smoothing`. (c) **Fix the data**: add pairs where the shorter response wins; add an explicit length-neutrality clause to the guidelines. (d) **Fix the metric**: report LC win rate, and select checkpoints on it.
  5. **Guard against recurrence.** Add mean-length and LC-win-rate to the CI regression gate (§16.4 of CS-14).

**Q106. Your DPO model is worse than the SFT model on every eval, but the loss fell beautifully from 0.69 to 0.31. First three things you check, in order.**
- **Answer (ordered):**
  1. **The chosen/rejected mapping.** This is the highest-probability cause and it produces exactly this signature — a healthy loss curve and a reversed model. Check: take 20 held-out pairs and compute `β·(logratio_chosen − logratio_rejected)`. If the mean is **negative**, the labels are swapped somewhere in the pipeline (column mapping, a CSV header typo, a judge that wrote its verdict into the wrong field). Fix and re-run.
  2. **The chat template.** Print one fully-formatted *training* example and one *inference* prompt and diff them. If the training prompts were formatted with a raw concatenation (no template set) and inference uses the model's chat template, the model was trained on a different input distribution from the one it is served on. This is a top-3 silent failure and the loss curve is unaffected.
  3. **`max_length` truncation.** Compute the p95 of `len(prompt+chosen)` and `len(prompt+rejected)`. If either exceeds `max_length`, the responses were truncated — and if chosen and rejected were truncated to *identical* prefixes, you trained on zero-margin pairs (explaining a slow loss floor) or on partial responses (explaining a degraded model).
  4. **Behind those three:** LR too high (check for loss spikes), too many epochs (check the eval loss), and the reference model being accidentally trainable (which would make the margin unable to grow — but note that symptom is a *flat* loss, not a falling one, so it is lower probability here).

**Q107. Your PPO run's reward score climbs steadily for 2,000 steps while human evaluation gets worse from step 600. What happened, and what do you do?**
- **Answer (ordered):**
  1. **Name it.** This is **reward-model over-optimisation** — the exact shape Gao, Schulman & Hilton (2023) quantified. The proxy reward rises monotonically; the true reward peaked around step ~600 and is declining.
  2. **Immediate action.** Stop the run. Select the checkpoint at the true-reward peak (~step 600, identified by the human eval or a gold-judge eval), not the latest one. If you have not been running a human/gold eval at intervals, start one now and reconstruct the curve from saved checkpoints.
  3. **Diagnose which hack.** Inspect outputs at step 600 vs 2,000. Look for: length growth (length bias), agreement with false premises (sycophancy), formatting uniformity (format exploit), loss of hedging (uncertainty deletion), or repetition. The specific artefact tells you which mitigation to apply.
  4. **Fixes.** Raise β (the KL coefficient) — you spent too much KL budget. Add a length penalty or length normalisation if that is the hack. Add a **reward-model ensemble** with a pessimistic aggregate (min or mean − α·std) so the policy must fool all members. Re-collect preferences on the *current* policy's outputs and retrain the RM (iterative RLHF) — this is the standard fix at the source rather than at the symptom.
  5. **Prevent recurrence.** Cap the KL budget rather than the step count. Run the capability + probe suite every 100 steps. **Never select a checkpoint on the RM score** — it was highest at step 2,000, which is the worst model you produced.

**Q108. Your ORPO run trains fine but the model's outputs are almost identical to the SFT model. What do you check?**
- **Answer (ordered):**
  1. **λ (or β in TRL's naming) is too low.** ORPO's objective is `L_SFT + λ·L_OR`; if λ is at the bottom of the range (or 0), the preference term contributes nothing and you have simply run SFT again. Raise λ to 0.1–0.5 and re-run.
  2. **The adapter is not actually training.** `print(sum(p.numel() for p in model.parameters() if p.requires_grad))` — must be > 0 and in the millions. Also verify `target_modules` matched real module names (a typo silently produces zero adapters in some PEFT versions) and that the saved checkpoint contains `adapter_model.safetensors`.
  3. **`max_length` truncation again** — same diagnostic as Q106, and here it is worse, because the SFT term on a truncated `chosen` is a degraded SFT signal too.
  4. **Pairs are too easy.** If `chosen` and `rejected` differ only in a trivial way (or the SFT model already ranks `chosen` above `rejected` with a large margin), there is nothing left to learn. Measure the SFT model's held-out preference accuracy — if it is already 85%, your pairs are saturated and you need harder ones.
  5. **LR too low for the adapter.** 8e-6 is the full-FT figure; with LoRA on a small rank you may need 1e-5–2e-5. Check the loss curve has actually moved from its initial value.
  6. **It is working, and the expected effect is small.** ORPO's shift is genuinely smaller than DPO's for the same data — the SFT term dominates. If points 1–5 are clean, the honest answer is that ORPO polishes rather than transforms, and you should switch to DPO if you need a larger shift.

**Q109. Your GRPO run's reward is stuck at 0.0 for 500 steps. What do you check?**
- **Answer (ordered):**
  1. **Is the reward actually being computed?** Log the raw reward values before aggregation and the `reward_std` per batch. If everything is exactly 0.0 with zero variance, either the reward function is erroring silently and returning 0 (check for a broad `except: return 0.0` in a sandbox-execution wrapper — the most common cause), or the task is too hard for the current policy.
  2. **Zero-variance groups.** Log the fraction of prompts with nonzero reward variance. If it is near 0, every group is all-fail (or all-pass) and no gradient flows. This is the vanilla-GRPO failure that DAPO's dynamic sampling fixes — implement it, or raise the temperature to 1.0+ for more rollout diversity, or **start from an easier task slice**.
  3. **The generation is broken.** Inspect the actual completions, not the rewards. If they are truncated at `max_completion_length` mid-sentence, or the model is emitting EOS immediately, the reward can never be earned. Check the stop-token configuration and the chat template.
  4. **The LR is too low or too high.** 1e-6 for 500 steps on a 7B with a hard task may genuinely show no movement; conversely, an LR that is too high collapses the policy to one output. Check the entropy of the rollout distribution.
  5. **The reward is too sparse.** If a task requires 10 correct steps and the model gets 0 for any error, the pass rate may be below 1% and the signal is effectively absent. Add **partial credit** (e.g. a small reward for "executes without crashing", or per-test-component scoring) to break the plateau — then anneal it away.
  6. **KL β is too high.** If β is at the PPO-convention default and the policy is heavily penalised for deviating, the reward term is swamped. Check the KL term's magnitude relative to the reward term in the logged metrics.

**Q110. Your QLoRA DPO run OOMs at step 1 on an A100 40GB with a 7B model, batch size 1, max_length 1024. Walk the diagnosis.**
- **Answer (ordered):**
  1. **Confirm the reference is free.** With PEFT and `ref_model=None` there should be one set of base weights, not two. Check the model is actually a `PeftModel` when it reaches the trainer — if the adapter attach failed, TRL sees a plain model and **deep-copies it as the reference**, doubling the footprint. `print(type(model))` and look for `peft_config`.
  2. **Compute the budget.** 4-bit base ≈ 4.5 GB; adapter + 8-bit optimiser ≈ 1 GB; that leaves ~34 GB for activations, which at batch 1 × 1024 tokens should be 6–14 GB. If it is OOMing, something is not 4-bit or the sequence is far longer than you think.
  3. **Check `max_prompt_length` separately.** `max_length` caps the *total*; if prompts average 800 tokens and responses 600, the total is 1,400 and gets truncated — but the activations are computed on the truncated length, so this is not the OOM cause. Verify by logging the actual batch token count.
  4. **Check that gradient checkpointing is on.** `DPOConfig(gradient_checkpointing=True)` cuts activation memory 3–5× for ~30% more time. Its absence is a very common cause of exactly this OOM.
  5. **Check for a fragmenting allocator.** Repeated failed allocations leave the cache fragmented. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` fixes a surprising fraction of "it OOMs at step 1 but the numbers say it should fit".
  6. **Reduce, in this order:** enable gradient checkpointing → lower `max_length`/`max_prompt_length` to the p95 (you may be padding every batch to 1024 when the real p95 is 500) → set `optim="paged_adamw_8bit"` → use SimPO (removes the reference forward entirely) → reduce LoRA rank.
  7. **If it still OOMs,** the model is not actually 4-bit. Verify `model.is_loaded_in_4bit` or inspect a layer's dtype — a silently-failed `BitsAndBytesConfig` load falls back to bf16 (14 GB) and produces exactly this symptom.

**Q111. After shipping your aligned adapter, users report the assistant has become evasive — it refuses plausible requests it used to answer. Diagnose.**
- **Answer (ordered):**
  1. **Measure it.** Run a benign-prompt false-refusal set (200–500 ordinary requests) against the shipped adapter and the pre-alignment adapter. Quantify the delta. Do not act on anecdotes.
  2. **Root cause candidates, in order of probability.** (a) **The preference data is safety-heavy or refusal-heavy** — if `chosen` is disproportionately a refusal or a hedge, you trained refusal as a style. (b) **The judge/annotators rewarded hedging** — "I can't be sure without more context" reads as careful and wins preference comparisons against a direct answer. (c) **A rubric with a harmlessness axis dominated helpfulness** in an RLAIF pass. (d) **Judge-bias transfer** — the labelling model is more cautious than your users want and you imported its caution.
  3. **Look at the data.** Count the fraction of `chosen` responses that contain a refusal pattern ("I can't", "I'm not able to", "consult a professional"). If it is above ~5% on a general-purpose set, that is your answer.
  4. **Fix.** Rebuild the preference set with an explicit refusal-calibration clause ("refusal is correct only when the request is genuinely harmful; prefer a helpful answer to an unnecessary refusal") and an explicit share of helpful-answer pairs on benign prompts. Re-run at a *lower* β if the shift is otherwise fine (β=0.2–0.3) so the model moves less.
  5. **Roll back first if it is user-visible.** Adapter swap is a config change (§16.1); roll back, then fix.
  6. **Prevent recurrence.** Add false-refusal rate to the CI regression gate — it is one of the six numbers in §16.4, and it is the one teams most often omit.

**Q112. You are handed a colleague's DPO run: `beta=0.01`, `num_train_epochs=8`, `learning_rate=1e-4`, `r=4`, `target_modules=["q_proj"]`, on 2,000 pairs. The loss is 0.02 and they say it is working great. What do you tell them?**
- **Answer (ordered):**
  1. **Every hyperparameter is wrong in the same direction, and the low loss is the symptom, not the evidence.** The loss is a function of the log-ratio margin, and this configuration maximises the margin regardless of quality: β=0.01 (10× too aggressive), 8 epochs (3–8× too many), LR 1e-4 (5–50× too high), rank 4 on q_proj only (~0.02% trainable params, so the LR had to be high to move at all — a compensating error).
  2. **Predict the failure before testing.** Expected outputs: degenerate repetition, loss of hedging, all answers converging to one style, mean length growth, and capability regression. The margin will be enormous (10+ nats).
  3. **Verify with three cheap diagnostics.** (a) Held-out implicit margin — expect a large mean with a *negative p10*. (b) Mean response length vs the SFT baseline — expect a large rise. (c) Held-out preference accuracy — expect it to look fine or good, which is exactly why the run "looks great"; the held-out margin is the tell.
  4. **The conversation to have.** Explain that training loss is a diagnostic for "did it move", not "is it good", and that this run has moved far past the true-reward peak. Show the p10 and the length number, not an opinion.
  5. **The corrected config.** β=0.1, 1–2 epochs, LR 1e-5–2e-5 (LoRA) with LR *reduced* as rank *increases*, `r=16`, `target_modules="all-linear"`, `label_smoothing=0.05–0.1` on a 2,000-pair set (noise is likely). Then select the checkpoint on a held-out human/gold win rate plus a length check, and expect the loss to land around 0.3–0.5 — **a higher loss than their broken run, and a better model.**

**Q113. Halfway through a DPO run, your training loss spikes from 0.35 to 4.2 and stays high. What happened?**
- **Answer (ordered):**
  1. **Look at the step, not the epoch.** Identify the exact step of the spike, then look at the batch that caused it. The overwhelming majority of DPO loss spikes are a **single pathological batch** — one very long pair, or a pair with a near-zero-length response, or a pair where `chosen` and `rejected` are near-identical (which drives `log π_θ(y)` to extreme values on a long sequence).
  2. **Check the optimiser state.** A loss that spikes and *stays* high (rather than recovering) usually means the weights were damaged — a large gradient step moved the adapter into a region from which it does not return. Confirm by evaluating the checkpoint before and after the spike.
  3. **Root causes, in order of probability.** (a) **No gradient clipping** — set `max_grad_norm=1.0`; this alone prevents most spikes. (b) **LR too high** — 1e-4 on LoRA DPO will spike; 1e-5–2e-5 will not. (c) **A long-sequence outlier** — the loss is a sum of log-probs, so a single 4,000-token pair with a large margin produces a proportionally larger gradient. Filter or bucket by length. (d) **fp16 overflow** — if you are in fp16 rather than bf16, a large logit produces inf and the loss jumps; switch to bf16.
  4. **Recovery.** Reload the last checkpoint *before* the spike, add gradient clipping and/or lower the LR, and resume. Do not resume from the spiked checkpoint — the weights are damaged and the optimiser moments are polluted.
  5. **Prevention.** Always set `max_grad_norm=1.0`, always use bf16, always filter responses shorter than ~20 tokens, and log per-step loss (a per-epoch average would have hidden this entirely).

**Q114. Your win rate against the SFT baseline is 72%, which seems too good. What are the four most likely explanations, and how do you distinguish them?**
- **Answer (ordered):**
  1. **Evaluation leakage — most likely.** Some of your eval prompts appeared in the preference data (or the eval set was generated from the preference prompts). **Distinguish:** hash the prompts and compute the intersection; if it is nonzero, rebuild the eval set from a held-out prompt pool. Deduplicate by *semantic* similarity too, not just exact match.
  2. **The baseline is broken.** The SFT model was loaded with a different precision, a different chat template, or without its adapter, so you are comparing against a degraded model. **Distinguish:** re-run the baseline through the exact same generation path as the candidate (same script, same dtype, same template) and check its outputs look like what you remember from the SFT evaluation.
  3. **The judge is biased toward your model.** If the judge is the same family as your model (self-preference), or if your model is systematically longer, or if it always appears in position A. **Distinguish:** (a) swap positions and re-judge — consistent-verdict-only win rates will drop toward 50% if position bias is the cause; (b) compute a length-controlled win rate; (c) re-judge with a different-family judge and compare.
  4. **Selection on the eval set.** You evaluated several checkpoints and reported the best. **Distinguish:** count how many checkpoints you scored; with 5 checkpoints and no correction, a 3–5 point optimistic bias is expected. Report the pre-registered checkpoint or apply a multiple-comparison correction.
  5. **It might be real — but verify with humans.** A 72% win rate on a narrow, well-specified behaviour change with a good dataset is achievable. Run 200 human comparisons on a fresh held-out prompt set; if humans say 60–68%, the number is real and the judge is mildly inflated.

**Q115. You are asked to reproduce a published alignment result and cannot. Walk through what you check.**
- **Answer (ordered):**
  1. **Data.** Is your dataset the same one, the same *revision*, and the same split? Preference datasets are frequently re-cleaned; `ultrafeedback-binarized-preferences-cleaned` exists precisely because the raw version had quality problems. Pin the dataset revision hash.
  2. **Schema.** Has `chosen`/`rejected` been mapped correctly? In `hh-rlhf` there is no `prompt` column — if you inferred the prompt differently from the paper, the trained quantity is different.
  3. **Reference model.** Is your reference the same checkpoint the paper used? In DPO with LoRA this matters enormously — the reference is the *base with the adapter disabled*, so a different base (or a base with a different revision) changes the objective's anchor.
  4. **Loss type and β.** `loss_type="sigmoid"` vs `"hinge"` vs `"ipo"` are different objectives. β=0.1 vs 0.5 is a 5× difference in push. The paper may also use a length-normalised variant.
  5. **Evaluation protocol.** Are you measuring the same thing? Raw vs LC win rate is a ~10-point difference. MT-Bench vs AlpacaEval vs Arena Elo are different scales. Judge version and prompt template matter — a different judge prompt can move a win rate by 5+ points.
  6. **Seeds and hardware.** DPO-style methods are reasonably reproducible but not exactly so; PPO/GRPO runs are frequently not reproducible at all across different GPU counts because the rollout batching changes. If the paper used a different batch composition, expect differences.
  7. **The honest conclusion.** If items 1–5 check out and you are within a few points, you have reproduced it. If you are 15+ points off, one of items 1–4 is wrong, and it is most often item 1 or item 2.

**Q116. A junior engineer says "the aligned model is much better — I checked 10 outputs and they're all improved." What do you say, and what do you make them do?**
- **Answer (ordered):**
  1. **Name the problem precisely, without dismissing the observation.** Ten outputs from one model is not an evaluation; it is a sample. At temperature 0.8, two samples from the *same* model differ as much as samples from two different models — the observed difference may be entirely sampling noise. The instructor's own demo does exactly this (one generation per model) and it is the weakest part of the video's methodology.
  2. **Quantify the noise.** Have them generate 5 samples from each model on the same 20 prompts and count how many of the "improvements" survive the extra sampling. This is usually a sobering experience and it teaches the lesson better than an explanation.
  3. **Make them build the minimum viable eval.** 200 held-out prompts, one generation each from both models, the same decoding parameters, a blinded A/B with position swapping, and a length check. That is a half-day of work and it produces a number with a confidence interval.
  4. **Make them check the two failure modes they cannot see.** Mean response length (their "improvement" may be verbosity) and the capability suite (their improvement may have cost something they are not measuring).
  5. **The rule to leave them with.** *A single generation is an anecdote. A win rate with a confidence interval over a held-out prompt set, plus a capability delta and a length check, is an evaluation. Anything in between is a story you are telling yourself.*

---

## Rapid Fire — True / False / One-Liner

Answer in under five seconds. If you have to think, you do not know it.

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | DPO needs a reward model. | **False** | The reward is implicit in the log-ratio; no `r_φ` artifact exists. |
| 2 | PPO needs two models. | **False** | Four: policy, reference, reward, value. |
| 3 | ORPO needs a reference model. | **False** | The SFT term is the anchor. |
| 4 | SimPO needs a reference model. | **False** | Length-normalised average log-prob is its own anchor. |
| 5 | GRPO needs a value model. | **False** | The group mean is the baseline. |
| 6 | DPO's loss at initialisation is 0.5. | **False** | `−log σ(0) = log 2 = 0.6931`. |
| 7 | An RM loss of 0.693 means the RM has learned something. | **False** | It means it has learned **nothing** — 0.693 is the chance floor. |
| 8 | Raising DPO's β keeps the model closer to the reference. | **False** | It pushes **further** — DPO's β is inside the sigmoid, the opposite convention to PPO's KL penalty. |
| 9 | The KL term is `KL(π_ref ‖ π_θ)`. | **False** | It is `KL(π_θ ‖ π_ref)` — the expectation is over the policy. |
| 10 | SFT can add knowledge the base model lacks. | **False** | SFT teaches format and behaviour; facts come from pretraining or RAG. |
| 11 | Alignment improves factual accuracy if the pairs are good. | **False** | Preferences encode which answer is *preferred*, not which is *true*. |
| 12 | Instruction tuning alone is sufficient for a production assistant. | **False** | It gets you format; preference alignment gets you the selection among plausible answers. |
| 13 | You can run DPO on a raw base model. | **False** | `chosen`/`rejected` are off-distribution; the KL anchors you to an autocompleter. |
| 14 | Full-FT DPO on 7B needs two full models in memory. | **True** | Reference is a silent deep-copy; the most common OOM cause. |
| 15 | LoRA DPO's reference model is free. | **True** | Base with the adapter disabled — one set of weights. |
| 16 | 7B bf16 inference needs about 14 GB. | **True** | 2 bytes × 7e9 = 14 GB, plus activations. |
| 17 | Full-FT AdamW needs 16 bytes per parameter. | **True** | 4 weights (2+2), 4 grads, 4 master fp32, 4+4 Adam m/v — wait: 2+2+2+4+4+4 = 18; the standard planning figure is 16 with grads in bf16 and no separate fp32 master. Use **16 B/param**. |
| 18 | 7B full fine-tuning weights+grads+optimiser is about 112 GB. | **True** | 7e9 × 16 B = 112 GB. |
| 19 | PPO's β and DPO's β mean the same thing. | **False** | PPO: KL penalty, higher = stay nearer. DPO: inside the sigmoid, higher = push further. |
| 20 | A preference pair must have `chosen` longer than `rejected`. | **False** | Length-neutrality is the point; a length-biased dataset teaches verbosity. |
| 21 | Inter-annotator agreement caps achievable alignment quality. | **True** | You cannot learn a decision the annotators themselves do not make consistently. |
| 22 | 100% inter-annotator agreement is the target. | **False** | It usually means the task was trivial or the annotators colluded; 70–85% raw agreement on hard subjective tasks is healthy. |
| 23 | RLAIF is a form of preference data generation. | **True** | It replaces the human labeller with an LLM judge. |
| 24 | Constitutional AI uses human preference labels for the RL stage. | **False** | The critique-revision stage uses model self-critique against an explicit written constitution. |
| 25 | A reward model generalises to any policy's outputs if it is big enough. | **False** | It is only valid on its training distribution — the canonical 1B-policy-vs-GPT-4-RM failure. |
| 26 | Over-optimisation makes the RM score go down. | **False** | The RM score keeps rising; the **true** reward peaks and falls. |
| 27 | You should select the checkpoint with the best RM score. | **False** | That is the most-hacked checkpoint — the worst model you produced. |
| 28 | DPO's loss is a proxy for output quality. | **False** | It is a proxy for the margin; the margin grows monotonically through over-optimisation. |
| 29 | A DPO loss of 0.05 after one epoch is a good sign. | **False** | It means the margin has exploded; expect length bloat and hedging loss. |
| 30 | Increasing `num_train_epochs` is how you get more out of a small preference set. | **False** | You get over-fitting. With 5 pairs, 1 epoch = 1 optimiser step; there is nothing to repeat. |
| 31 | `log 2 = 0.6931` is the DPO loss floor. | **False** | It is the **initialisation** value; the floor is 0. |
| 32 | Reward is identifiable up to a per-prompt constant in Bradley–Terry. | **True** | Only differences within a prompt are identified — hence reward normalisation subtleties. |
| 33 | Length bias is caused by the reward model being bad. | **False** | It is caused by the implicit reward being a **sum** over tokens; a perfect RM does not fix it. |
| 34 | SimPO fixes length bias structurally. | **True** | It uses the average log-prob instead of the sum. |
| 35 | `remove_unused_columns=False` is optional in TRL. | **False** | Mandatory — otherwise the prompt/chosen/rejected columns are stripped and the trainer crashes. |
| 36 | `load_in_8bit=True` is the current recommended API. | **False** | Deprecated; use `BitsAndBytesConfig(load_in_8bit=True)`. |
| 37 | `torch_dtype` is the current keyword. | **False** | Renamed to `dtype` in transformers ≥ 4.56; `torch_dtype` warns. |
| 38 | `tokenizer=` is the current TRL argument. | **False** | TRL ≥ 0.13 uses `processing_class=`. |
| 39 | Merging a LoRA adapter into an 8-bit base is lossless. | **False** | It rounds to 8-bit — lossy. Merge in bf16 or serve the adapter unmerged. |
| 40 | You should stack LoRA adapters across pipeline stages. | **False** | Merge the previous stage's adapter into the base, then attach a fresh adapter. |
| 41 | QLoRA's NF4 base for 7B is about 4 GB. | **True** | ~3.5–4.5 GB. |
| 42 | PPO is the default recommendation for a product team in 2025. | **False** | Try DPO or ORPO first; PPO is for a verifiable/black-box reward you cannot pair. |
| 43 | GRPO is preferred over DPO when the reward is verifiable. | **True** | The program reward removes the RM and the preference data entirely. |
| 44 | GRPO with G=1 works. | **False** | The group mean is the sample itself; the advantage is identically zero. |
| 45 | GRPO and KL: you set a β penalty just like PPO. | **True** | GRPO keeps the KL-to-reference term; typical β ≈ 0.04. |
| 46 | The DPO-vs-PPO crossover is around 20,000 pairs. | **True** | Below that, DPO's simplicity dominates; above it, PPO's exploration starts to pay. |
| 47 | The alignment tax is unavoidable in full. | **True** | Every alignment run costs something; the engineering question is how much and where. |
| 48 | PPO-ptx restores all lost capability. | **False** | It restores fluent pretraining behaviour; it does not undo SFT damage. |
| 49 | Win rate on AlpacaEval 2 raw is directly comparable to Arena Elo. | **False** | Different scales, different prompt sets, different protocols. Never mix. |
| 50 | LC win rate controls for response length. | **True** | Length-controlled — it is the number to report. |
| 51 | A judge model should be the same family as the model being trained. | **False** | Self-preference bias; use a different family. |
| 52 | Position bias means you should swap A/B order and count only consistent verdicts. | **True** | Standard protocol. |
| 53 | KTO needs paired data. | **False** | KTO is specifically for **unpaired** desirable/undesirable examples. |
| 54 | KTO uses prospect theory — losses loom larger than gains. | **True** | `λ_D > λ_U` in the objective. |
| 55 | KTO works with a 1.0 desirable : 1.0 undesirable ratio. | **False** | It needs roughly balanced or a documented reweighting; a skewed ratio degrades it. |
| 56 | cDPO is DPO with label smoothing. | **True** | It assumes ε of the preference labels are wrong. |
| 57 | IPO replaces the unbounded sigmoid margin with a fixed target. | **True** | Target margin `1/(2β)`; squared loss in the margin. |
| 58 | SLiC-HF is a hinge loss on the sequence log-prob difference. | **True** | A flat region once the margin exceeds the target. |
| 59 | Best-of-n requires no training at all. | **True** | Pure inference-time selection using a reward model — the "no-training-required" baseline you should always measure. |
| 60 | Rejection sampling is a fine-tuning method. | **False** | It is a **data-generation** method; you still need an SFT or preference stage on the output. |
| 61 | RAFT is best-of-n plus SFT on the winners. | **True** | That is exactly the recipe. |
| 62 | SPIN generates its own preference pairs by self-play against the previous iteration. | **True** | The previous checkpoint supplies the `rejected` responses. |
| 63 | A DPO model can be used for best-of-n scoring. | **False** | No absolute reward — `Z(x)` is intractable. Only margins between two responses on the same prompt. |
| 64 | The implicit margin on held-out data is a useful cheap diagnostic. | **True** | It detects label swapping, over-optimisation, and a dead model, in seconds. |
| 65 | Reward hacking can occur even with a perfect reward model on the training distribution. | **True** | Gao et al.'s result — the policy finds the RM's errors within its own training distribution. |
| 66 | KL divergence in nats and the RM score are the same scale. | **False** | Different quantities; the over-optimisation curve plots true reward against √KL. |
| 67 | More preference data always beats a better preference dataset. | **False** | Quality and difficulty of pairs dominate volume above a few thousand. |
| 68 | 500 pairs is enough to shift general tone. | **False** | It is enough to fix **one** narrow behaviour. |
| 69 | ORPO is a single-stage method that starts from the base model. | **True** | It has an SFT term in the objective. |
| 70 | DPO replaces SFT. | **False** | DPO is downstream of SFT and needs it. |

> **The eleven you must not miss:** #8 (DPO β direction), #9 (KL direction), #15 (`ref_model=None` with LoRA), #26/#27 (RM score vs true reward), #33 (the mechanism of length bias), #41 vs #17–18 (the memory arithmetic), #43 (verifiable → GRPO), #53 (KTO is unpaired), #59 (best-of-n as the no-training baseline), #63 (DPO cannot score absolutely), #70 (DPO does not replace SFT).

---

## Coding / Whiteboard Tasks

Each task states what the interviewer is grading. Write real code; there is no partial credit for describing it.

### T1. Write the DPO loss from scratch, in ~15 lines of PyTorch, with no library helpers.

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1, label_smoothing=0.0):
    """
    All logps are (batch,) tensors of *summed* log-probabilities over the
    response tokens (not length-normalised -- that is SimPO).
    Returns (loss, chosen_reward, rejected_reward, margin).
    """
    pi_logratios  = policy_chosen_logps  - policy_rejected_logps      # log pi(y+)/pi(y-)
    ref_logratios = ref_chosen_logps     - ref_rejected_logps         # log pref(y+)/pref(y-)
    logits = pi_logratios - ref_logratios                             # the DPO (implicit) margin, un-scaled
    if label_smoothing > 0.0:
        # cDPO: eps of the labels are assumed flipped -> both directions get weight
        losses = (-F.logsigmoid(beta * logits) * (1 - label_smoothing)
                  - F.logsigmoid(-beta * logits) * label_smoothing)
    else:
        losses = -F.logsigmoid(beta * logits)
    chosen_rewards   = beta * (policy_chosen_logps  - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return losses.mean(), chosen_rewards, rejected_rewards, (beta * logits).detach()
```

**Grading:** (1) `beta` multiplies the **difference of log-ratios**, not one log-ratio; (2) `.detach()` on the reported rewards (they are diagnostics, not loss terms); (3) `logsigmoid`, not `log(sigmoid(...))` — numerical stability; (4) the reference log-probs must come from a `no_grad()` forward pass on the frozen reference; (5) knowing that at `logits = 0` the loss is `log 2 = 0.6931`.

### T2. Write SimPO's loss and explain, in two lines, which term removes the need for a reference.

```python
def simpo_loss(policy_chosen_logps, policy_rejected_logps,
               chosen_lengths, rejected_lengths, beta=2.0, gamma=0.5):
    """
    SimPO: length-normalised average log-probs, no reference model, plus a
    target margin gamma. beta here is a scaling coefficient (2.0-2.5 typical),
    NOT a KL coefficient -- it looks like DPO's beta but is swept 2-10x higher.
    """
    avg_chosen   = policy_chosen_logps   / chosen_lengths
    avg_rejected = policy_rejected_logps / rejected_lengths
    logits = avg_chosen - avg_rejected
    return -F.logsigmoid(beta * logits - gamma).mean()
```

**Grading:** (a) the division by length is the whole point — it is what removes the length bias; (b) `gamma` is subtracted **inside** the sigmoid, so it sets a target margin the model must exceed before the loss saturates; (c) understanding *why* no reference is needed: the average log-prob is anchored by the SFT-initialised policy and the margin acts as a regulariser, so there is no drifting anchor to constrain.

### T3. Write the Bradley–Terry reward-model loss with a `margin` term, and say what `margin` does.

```python
def reward_model_loss(chosen_rewards, rejected_rewards, margin=0.0):
    """
    BT: P(y+ > y-) = sigmoid(r(y+) - r(y-)).
    margin > 0 requires the RM to separate the pair by at least `margin`
    before the loss goes to zero -- a hinge-like effect that sharpens the
    RM on near-ties. Set 0.0 for the textbook objective.
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards - margin).mean()
```

**Grading:** (a) `−log σ(r+ − r−)`, not `−(log σ(r+) + log σ(1 − r−))`; (b) the floor at 0.693 when the RM cannot separate; (c) knowing that `margin` trades calibration for separation — it sharpens near-ties but makes the RM's absolute scores less calibrated, which matters if you use them for best-of-n thresholds.

### T4. Whiteboard: draw the four-model PPO diagram and annotate every arrow with its data type and cost.

```
                    ┌─────────────────┐
   prompts x ──────▶│  POLICY π_θ     │──▶ responses y  (SAMPLED, temp 1.0)
                    │  (trainable)    │        │
                    └────────┬────────┘        │
                             │                 ▼
                    ┌────────▼────────┐  ┌─────────────┐
                    │ REFERENCE π_ref │  │ REWARD r_φ  │──▶ scalar reward
                    │ (FROZEN, bf16)  │  │ (FROZEN)    │
                    └────────┬────────┘  └─────────────┘
                             │                 │
                             ▼                 ▼
                    log π_ref(y|x)      r(x,y)
                             │                 │
                             └────────┬────────┘
                                      ▼
                             KL penalty + advantage
                                      │
                    ┌─────────────────▼───────────────┐
                    │ VALUE MODEL V_ψ (trainable)     │──▶ baseline b(x,y)
                    │ GAE + returns                   │
                    └─────────────────┬───────────────┘
                                      ▼
                          PPO clipped surrogate update
```

**Grading — the annotations matter more than the boxes:**
- Policy: trainable; 16 B/param full FT, ~2 B/param LoRA.
- Reference: **frozen, bf16, 14 GB for 7B**; provides `log π_ref` only.
- Reward: frozen; a scalar per completed sequence; **only valid on its training distribution**.
- Value: **trainable, a full-size model head** — the one people forget in the memory budget; 112 GB at 7B full FT.
- The KL arrow comes from policy-vs-reference log-probs, the reward arrow from the RM, and both feed the advantage; the value model supplies the baseline.
- Four models × 7B = **~280 GB of parameters alone** before activations.

### T5. Whiteboard: which method for these five scenarios? Justify each in one line.

| Scenario | Method | One-line justification |
|---|---|---|
| 30k unit-test-checkable code tasks, no human labels | **GRPO / RLVR** | Program reward → no RM, no pairs, and the group baseline needs no value model. |
| 12k human preference pairs on tone, 1× A100 40GB | **DPO (QLoRA)** | Pairs exist, no verifier; QLoRA fits the 40 GB with a free reference. |
| Only a scalar quality score from a black-box API, one response at a time | **PPO** | The reward is accessible-per-sample, not pairable and not a program — PPO's RM path is the only fit. |
| 800 unpaired "this was good" / "this was bad" flags from a support team | **KTO** | Unpaired binary feedback is exactly KTO's input format. |
| 3k pairs, one 24 GB consumer GPU, need a same-day result | **ORPO (QLoRA)** | Single stage, no reference, no RM, 11–19 GB for 7B. |

**Grading:** the *mapping* (verifiable→GRPO, paired→DPO, black-box scalar→PPO, unpaired→KTO, cheapest→ORPO) plus the stated constraint (VRAM or label type) driving the choice. Anyone who answers "PPO" to all five has memorised one method.

### T6. Write a minimal, correct TRL DPO training script. It must run.

```python
# dpo_train.py -- runnable. pip install "trl>=0.13" "peft>=0.13" \
#   "transformers>=4.56" "datasets" "bitsandbytes>=0.44" "accelerate"
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SFT_ADAPTER = None          # set to e.g. "./checkpoint-3" if you SFT'd first

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- SFT stage's adapter must be merged into the base BEFORE attaching a new one.
#     Attaching without merging is the silent bug: get_peft_model raises
#     "Already found a peft_config" and the new adapter never trains.
if SFT_ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)
    model = model.merge_and_unload()          # merge in bf16 if you need fidelity
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM", target_modules="all-linear",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()            # expect ~0.5-2% for r=16 all-linear

ds = load_dataset("csv", data_files="pharma_preference_data.csv")["train"]
# columns must be exactly: prompt, chosen, rejected (map yours if not)

cfg = DPOConfig(
    output_dir="./dpo-out",
    beta=0.1, loss_type="sigmoid",
    learning_rate=2e-5, num_train_epochs=1,
    per_device_train_batch_size=1, gradient_accumulation_steps=8,
    max_length=1024, max_prompt_length=512,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit", max_grad_norm=1.0,
    bf16=True, logging_steps=1, save_strategy="epoch",
    remove_unused_columns=False,              # MANDATORY
    report_to="none",
)
trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer)
trainer.train()
trainer.save_model("./dpo-out/final")          # saves the ADAPTER, not the base
```

**Grading:** (1) `remove_unused_columns=False`; (2) `processing_class=` not `tokenizer=`; (3) `gradient_checkpointing_kwargs={"use_reentrant": False}` — without it, gradient checkpointing with a frozen base raises on newer torch; (4) the merge-before-attach discipline and the ability to name the error it prevents; (5) `max_grad_norm=1.0`; (6) knowing `prepare_model_for_kbit_training` must be called **before** `get_peft_model`, not after; (7) knowing that `ref_model=None` here means "adapter disabled on the same base", i.e. free; (8) knowing the LR would be **1e-6**, not 2e-5, if this were GRPO.

---

## Cheat Sheet Of Numbers To Memorize

| Number | What it is |
|---|---|
| **0.6931** | `log 2` — the DPO loss at initialisation **and** the reward-model loss floor (no learning). |
| **0** | The DPO loss floor; the KL divergence of a policy from itself. |
| **0.1** | DPO's default β. |
| **0.01–0.1** | β range that is too aggressive / too permissive is reversed: **0.01 = aggressive, 0.5+ = conservative**. Sweep 0.05 / 0.1 / 0.2. |
| **0.04** | Typical GRPO β — and note this is the **PPO convention** (a KL penalty), not DPO's. |
| **2.0–2.5** | SimPO's β. It looks like DPO's β but is a different quantity and is swept 20× higher. |
| **0.5** | SimPO's default margin γ. |
| **0.1–0.5** | ORPO's λ (the weight on the odds-ratio term). |
| **1/(2β)** | IPO's finite target margin. |
| **4** | Models in memory for PPO: policy, reference, reward, value. |
| **2** | Models for DPO and GRPO (policy + reference). |
| **1** | Models for ORPO and SimPO. |
| **16 bytes** | Per parameter for full-FT AdamW training (bf16 weights+grads, fp32 master, 2× fp32 Adam moments). |
| **10 bytes** | Per parameter with 8-bit Adam. |
| **≈6 bytes** | Per parameter with Adafactor. |
| **2 bytes** | Per parameter for bf16 inference (frozen model). |
| **1 byte** | Per parameter for int8 inference. |
| **0.5–0.65 bytes** | Per parameter for NF4 4-bit inference — plan for 0.6. |
| **14 GB** | 7B in bf16. The number you multiply by whenever you add a frozen model. |
| **7 GB** | 7B in int8. |
| **3.5–4.5 GB** | 7B in NF4 4-bit. |
| **112 GB** | 7B full FT weights + grads + optimiser state. |
| **295–320 GB** | Total VRAM for full-FT PPO on 7B. |
| **150–165 GB** | Total VRAM for full-FT DPO on 7B (two models + optimiser). |
| **12–20 GB** | Total VRAM for QLoRA DPO on 7B — the single most useful number in this module. |
| **11–19 GB** | Total VRAM for QLoRA ORPO on 7B. |
| **25–40 GB** | Total VRAM for QLoRA PPO on 7B. |
| **20×** | The VRAM ratio between full-FT PPO and QLoRA ORPO on 7B. |
| **1e-6 to 5e-6** | Learning rate for GRPO/PPO. An order of magnitude **below** DPO's. |
| **1e-5 to 2e-5** | Learning rate for DPO with LoRA. |
| **2e-5** | The video's DPO learning rate. |
| **1e-4 to 2e-5** | Typical full-FT LR (higher) vs LoRA LR (lower) — the direction surprises people. |
| **1–3 epochs** | DPO. 2 is usually the ceiling on a clean dataset; 1 on a small or noisy one. |
| **8–16** | GRPO's group size G. G=16 was DeepSeek-R1's setting. |
| **G ≥ 4** | Below this, the group mean and std are too noisy to give a usable advantage. |
| **0.2 / 0.28** | DAPO's decoupled clip bounds (ε_low / ε_high). |
| **2,000–10,000 pairs** | The range where DPO is clearly the right call. |
| **20,000 pairs** | The approximate crossover above which PPO's exploration starts to pay. |
| **≥ 70% raw agreement** | The inter-annotator floor below which you should stop collecting and fix the guidelines. |
| **85–90%** | The judge-vs-human agreement you should require of an RLAIF pipeline. |
| **10%** | The share of data to double-annotate permanently as the IAA control set. |
| **80/20** | The typical split of chosen-response wins over rejected that makes a dataset worth training on. A 50/50 split is a dataset with no signal. |
| **5–20 nats** | The KL range over which the true reward typically peaks (Gao, Schulman & Hilton 2023). |
| **√KL** | The functional form those authors fit the true-minus-proxy reward gap to. |
| **~10%** | The share of training tokens that are in `chosen` but **not** in `rejected` — the "informative token" share. Under 5% means your pairs differ only in style. |
| **200–500** | The minimum size of a held-out prompt set for a win rate with a usable confidence interval. |
| **500 × 3** | Prompts × raters for a production human evaluation. |
| **$4/pair** | Order-of-magnitude cost of careful human preference annotation. |
| **$0.06/pair** | Order-of-magnitude cost of an LLM-judge pair. |
| **≈70×** | The cost ratio between human and judge annotation. |
| **$3/1M tokens** | Order-of-magnitude generation cost for a mid-tier API. |
| **1.126M** | The video notebook's trainable params: r=8 on q_proj+v_proj over 22 layers (0.1% of the 1.1B model). |
| **1 optimiser step** | What the video's demo actually performed: 5 pairs ÷ batch 1 ÷ accum 8 × 1 epoch. |
| **5.18 s** | The video's reported `train_runtime`, at `train_samples_per_second=0.966`. |
| **0.6619** | The video's reported `train_loss` — one step down from 0.6931. |
| **32000** | TinyLlama's vocab size (the notebook's model dump). |
| **2048 / 5632 / 22** | TinyLlama's hidden size / intermediate size / layer count. |
| **32 q-heads × 64 / 4 kv-heads × 64** | TinyLlama's GQA config — why `k_proj`/`v_proj` have `out_features=256`. |

---

## Answers To The Self-Check Questions From CS-14

These are the ten questions at the end of CS-14 §19, answered in full so you can grade yourself.

**1. Name the three stages of the canonical LLM training pipeline and one thing each stage contributes that the others cannot.**
**Pretrain** — self-supervised next-token prediction over web-scale text. Contributes: general language competence and world knowledge. Nothing else supplies knowledge. **SFT** (instruction fine-tuning) — supervised pairs of `(instruction, ideal response)`. Contributes: the *format* of being an assistant, instruction-following, and the response structure that makes an output usable. Contributes what pretraining has no signal for: there is no "instruction" in raw text. **Preference alignment** — `(prompt, chosen, rejected)` triples. Contributes: the *selection* among several plausible, correctly-formatted answers, i.e. tone, concision, refusal calibration, safety, and honesty behaviour. Contributes what SFT cannot express: SFT says "answers look like this", preference says "of two answers that both look like this, this one is better". None substitutes: pretraining gives no instruction format, SFT gives no ordering, alignment adds no knowledge.

**2. Why does running DPO on a raw base model produce a worse model than running it on an SFT checkpoint? Give the mechanism in terms of the KL term.**
The DPO objective inherits `max E[r] − β·KL(π_θ ‖ π_ref)` and DPO's implicit reward is `β·log(π_θ/π_ref)`. Setting `π_ref` to a **raw base model** anchors the policy to an autocompleter: the KL term actively penalises the model for being an assistant, because assistant behaviour is far from base-model behaviour. The reward term simultaneously pulls toward the preference data's helpfulness. The optimum of the two-term fight is not an assistant — it is a slightly-helpful autocompleter. Mechanically there is a second, worse effect: `chosen` and `rejected` are both far off the base model's distribution, so `log π_θ(y|x)` is tiny and high-variance for both members of the pair, and the log-ratio that carries the signal is dominated by noise on tokens the base model has never produced in that context. The result is a model that is neither a good autocompleter (the reward moved it) nor a good assistant (the KL held it back), which is measurably worse than the SFT checkpoint on every axis. **Alignment is a selection among behaviours; a base model has no behaviours to select among.**

**3. Write the DPO loss from memory. Define every symbol and state what happens to the loss at β→∞.**

```
L_DPO(π_θ; π_ref) = − E_{(x, y+, y−) ~ D} [ log σ( β · ( log(π_θ(y+|x)/π_ref(y+|x)) − log(π_θ(y−|x)/π_ref(y−|x)) ) ) ]
```

- `x` — the prompt.
- `y+`, `y−` — the chosen and rejected responses.
- `π_θ` — the policy being trained.
- `π_ref` — the frozen reference policy (the SFT checkpoint).
- `β` — the coefficient controlling the strength of the implicit KL constraint. **Higher β pushes the policy further from the reference** (the opposite of PPO's β).
- `σ` — the logistic sigmoid.
- `D` — the preference dataset.

The bracket is the **implicit margin**: `β·(h(x,y+) − h(x,y−))` where `h(x,y) = log π_θ(y|x) − log π_ref(y|x)`. At **β→∞**, the sigmoid argument saturates for any nonzero margin: `σ(z) → 1`, the loss → 0, the gradient → 0, and **π_θ never moves from π_ref**. The objective becomes vacuous. The complementary limit **β→0** makes the loss insensitive to the margin, so the model is free to deviate arbitrarily far from the reference and degenerates (typically to repetition and length bloat). At initialisation, `π_θ = π_ref` makes every margin 0, so the loss is exactly `−log σ(0) = log 2 = 0.6931` — the number you should see at step 0, and if you see anything else your labels or your reference are wrong.

**4. A DPO run's loss goes from 0.69 to 0.12 over two epochs and the mean response length doubles. What happened, what is the metric you failed to watch, and what are three fixes?**
**What happened:** **over-optimisation, manifesting as length bias.** DPO's implicit reward is `β·(log π_θ(y|x) − log π_ref(y|x))`, and `log π_θ(y|x)` is a **sum of per-token log-probabilities**, so the margin grows simply by making the response longer. The model found the cheapest way to raise the implicit reward: pad. The loss fell to 0.12 because the margin is large — the loss is a function of the margin, not of quality, which is why a low loss and a bad model co-occur here.
**The metric you failed to watch:** **mean generation length** on a fixed prompt set (and, upstream of it, the **length-controlled win rate** rather than the raw win rate — the raw number will look like an improvement because longer answers win preference comparisons).
**Three fixes:** (1) **Switch to SimPO**, which normalises the implicit reward by response length (`(1/|y|)·log π`) and therefore removes the bias structurally rather than by tuning. (2) **Rebalance the preference data** — add pairs where the *shorter* response is `chosen`, and add an explicit length-neutrality clause to the annotation guidelines; if you have no such pairs, no amount of β tuning will fix the bias. (3) **Stop earlier and select on a proxy-independent metric** — the checkpoint where mean length is flat is almost always better than the checkpoint with the lowest loss, and with 1 epoch on a small clean dataset you may not have this problem at all. Behind those: lower β to reduce the push, and add `label_smoothing`. **Do not** fix it by raising β alone — that reduces the magnitude of every preference shift, not specifically the length one.

**5. Your `chosen`/`rejected` labels were accidentally swapped. Describe exactly what the loss curve looks like and how you would detect it within ten minutes.**
**The loss curve looks completely normal.** It starts at `0.6931` and falls smoothly and healthily, indistinguishable from a correct run. This is the point: the DPO objective is symmetric in the roles of the two responses — it optimises `σ(margin)` whichever way the margin is signed — so a consistently swapped dataset is a perfectly learnable dataset and the model obligingly learns to prefer the swapped `rejected` responses.
**Detection in ten minutes:** (1) take 20 **held-out** pairs, run one forward pass of the policy and the reference over both members, and compute the mean **implicit margin** `β·(h(x,y_chosen) − h(x,y_rejected))`. If that mean is **negative**, the labels are swapped. This is the fastest and most reliable test — it is one script, no generation, no judge. (2) Confirm by generating responses for a held-out prompt and checking that the model assigns a higher likelihood to what you know is the worse answer. (3) If the mean is near zero rather than negative, the problem is not swapping but something else — check for a truncation that made both members identical.
**Root causes to check while you are in there:** a CSV whose `chosen`/`rejected` headers are swapped, a judge that wrote its verdict into the wrong field, a HuggingFace dataset where `chosen` is the **rejected** column of the original `hh-rlhf` schema, or an accidental sort that reordered one column but not the other.
**Prevention:** put a unit test in the data pipeline that asserts a positive implicit margin on a hand-labelled gold pair after one epoch. This one assertion would catch the entire class of bugs.

**6. What does `ref_model=None` do when using PEFT adapters, and what does it do at full fine-tuning? Why does the answer differ?**
**With PEFT/LoRA:** the base weights are frozen and the adapter is a **separable, additive delta** on top of them. "The model without its adapter" is therefore already a valid reference model, and TRL obtains it by **disabling the adapter** on the same weights for the reference forward pass. Memory cost: **zero extra parameters** — one set of base weights serves both roles. This is why LoRA makes DPO dramatically cheaper than the naive accounting suggests.
**At full fine-tuning:** there is no separable trainable part. The policy *is* the weights, and there is no "disable the delta" operation. TRL therefore **deep-copies the model at trainer initialisation** and freezes the copy. Memory cost: a full extra set of parameters — **14 GB for 7B in bf16**, and at full FT the copy sits beside a policy carrying 112 GB of weights+grads+optimiser state.
**Why the answer differs:** the question is whether "the model minus its trainable part" is cheaply computable. With LoRA it is a flag on a forward pass. With full FT it requires a physical copy.
**Production consequence:** this is the single most common cause of an unexplained OOM in full-FT DPO — the engineer budgeted for one model plus optimiser and got two full models. If you see DPO OOM at trainer init before step 1, check this first. The remedies are: use LoRA, use SimPO/ORPO (no reference at all), or budget the extra 14 GB explicitly.

**7. Compute the VRAM for full-FT PPO on a 7B model and for QLoRA ORPO on the same model. Show the arithmetic.**
**Full-FT PPO, 7B:**
- Per-parameter training cost with AdamW: bf16 weight (2) + bf16 grad (2) + fp32 master weight (4) + Adam m (4) + Adam v (4) = **16 bytes/param**, plus ~2 bytes for the moment the optimiser keeps transiently — plan 16.
- **Policy** (trainable): 7e9 × 16 B = **112 GB**.
- **Value model** (trainable — a full-size model, this is the one people forget): 7e9 × 16 B = **112 GB**.
- **Reference** (frozen, bf16): 7e9 × 2 B = **14 GB**.
- **Reward model** (frozen, bf16): 7e9 × 2 B = **14 GB**.
- **Activations + KV cache + fragmentation**: 15–40 GB depending on `max_length`, batch, and gradient checkpointing.
- **Total ≈ 295–320 GB.** That is four A100 80 GB cards as an absolute floor, realistically 8 with headroom.

**QLoRA ORPO, 7B:**
- **4-bit NF4 base** (frozen): 7e9 × ~0.55 B ≈ **4 GB** (3.5–4.5 GB with quantisation constants).
- **LoRA adapters** (r=16, all-linear, ~40M params): 40e6 × 2 B ≈ 0.08 GB, plus grads and 8-bit optimiser state ≈ **0.5 GB**.
- **No reference model, no reward model, no value model.** ORPO has none of the three.
- **Activations**: 6–14 GB with gradient checkpointing at `max_length` 1024, batch 1.
- **Total ≈ 11–19 GB.**

**Ratio: roughly 20×** — one RTX 4090 versus a multi-node cluster, for a method that on many tone/style tasks produces a *comparable* result. This single comparison, more than any benchmark table, is why the practical recommendation is "try DPO or ORPO before PPO."

**8. Given: unit-test-checkable code generation, 30k synthetic tasks, no human labels. Which method, why, and what config knob do you set differently from a DPO run?**
**Method: GRPO with a verifiable reward (RLVR).**
**Why:** the reward is a **program** — run the unit tests, reward 1.0 or 0.0. Three consequences, each of which removes a whole component: (a) there is no need for a **reward model**, because the reward is exactly computable, so there is no proxy to hack and no RM training run; (b) there is no need for **preference pairs**, because the reward is a scalar per sample, which removes the annotation cost and the whole data-collection pipeline; (c) there is no need for a **value model**, because GRPO replaces the learned baseline with the group mean reward. That is two models and one dataset removed relative to PPO, and one model and one dataset removed relative to DPO. Additionally, DPO is *offline*: it can only sharpen behaviours already present in the dataset, whereas 30k tasks with exact labels describe a capability the model may not yet have — GRPO can explore to find it.
**Config differences from a DPO run:**
- **Learning rate: 1e-6** (not 2e-5). An order of magnitude lower, because a policy-gradient estimator over sampled tokens at temperature 1.0 has far more gradient noise than a bounded sigmoid over fixed text.
- **β=0.04** — and it means the **opposite** of DPO's β. It is the PPO-convention KL *penalty* coefficient: higher = stay nearer the reference. Do not carry a DPO intuition across.
- **`num_generations=16`** (G) instead of a fixed pair batch. G is the dominant cost multiplier; below 4 the group statistics are unusable.
- **`temperature=1.0`** for rollout diversity. DPO has no sampling step at all.
- **A `reward_func`** (sandboxed test execution) instead of a `chosen`/`rejected` dataset.
- **Enable DAPO's dynamic sampling** so prompts whose group has zero reward variance — all-pass or all-fail — are dropped rather than burning the batch.
- **Guard against specification gaming** with a held-out test split the model never sees, and monitor `reward_std` and the zero-variance fraction as your primary health metrics, not the mean reward.

**9. What is the alignment tax, what is PPO-ptx, and which of the three mechanisms in §4.8 does it address?**
**The alignment tax** is the capability degradation caused by alignment. InstructGPT is the canonical citation: the RLHF models scored *lower* than base GPT-3 on SQuAD, DROP and HellaSwag while being strongly preferred by humans. The tax is not a bug — it is the expected consequence of optimising a narrow objective, and it is the number you must report alongside any win rate.
**PPO-ptx** adds a pretraining-log-likelihood term to the PPO objective: `L_PPO + γ_ptx · E_{x ~ D_pretrain}[log π_θ(x)]`, with `γ_ptx ≈ 0.01–0.1`. It mixes gradient from the original pretraining corpus into every PPO step, so the model is never optimising *only* the reward.
**Which mechanism it addresses:** the three are (a) **narrow reward** — capabilities outside the alignment distribution receive no reward and drift freely; (b) **reference degradation** — π^SFT is itself already degraded relative to the base, and the KL anchors you to it; (c) **distribution shift on long outputs** — preference data is short and single-turn, so long-form and multi-turn behaviour degrades. **PPO-ptx addresses (a)**, and partially (c) by keeping the model fluent on general text. **It does not address (b)** — that damage happened during SFT and no alignment trick recovers it; the fix is upstream, in the SFT data and epoch count.
**Beyond the video:** the modern equivalent is to mix a small percentage of pretraining or general-instruction data directly into the DPO/ORPO batch (5–10%) rather than adding a separate ptx term, which achieves the same regularisation without a separate objective term and is what most current recipes do.

**10. Explain Goodhart's law as it applies to a reward model, describe the shape of the over-optimisation curve, and name four mitigations in priority order.**
**Goodhart:** the reward model is a *proxy* for human judgement, and the policy optimises the proxy, not the judgement. Given enough optimisation pressure, the policy discovers the regions where the proxy is wrong — and the critical empirical result (Gao, Schulman & Hilton 2023) is that this happens **on the RM's own training distribution**, with a perfectly generalising RM, not because the RM generalises poorly. That is why "collect more data and train a bigger RM" does not solve it.
**Shape of the curve:** plot true reward against optimisation strength, measured by KL divergence from the reference. The **proxy** reward rises monotonically and never turns over. The **true** reward (a much larger gold RM, or humans) rises, **peaks**, and then declines. The authors fit the gap between them as growing approximately in `√KL`, with the true-reward peak typically at **KL ≈ 5–20 nats** depending on RM size. Concretely: your RM score is highest at the checkpoint where the model is worst.
**Four mitigations in priority order:**
1. **The KL penalty (β).** Bounds how much of the RM's error surface the policy can reach. It is free, it is already in the objective, and it is the primary control. Raise β when you see the proxy diverging.
2. **Early stopping on a proxy-independent metric** — a human or gold-judge win rate, plus a capability suite. This is the only control whose signal does not come from the thing being hacked. **Never select a checkpoint on the RM score.**
3. **Length-normalised rewards** — SimPO-style formulations, or an explicit length penalty, which kill the single most common concrete hack (verbosity) at the source.
4. **Reward-model ensembles** — k=3–5 RMs with decorrelated errors (different seeds and data orderings), aggregated **pessimistically** (`min_i r_i`, or `mean − α·std`) so the policy must fool all of them, and disagreement is treated as a penalty on regions of weak RM knowledge.
**Behind those, in order:** reward clipping and whitening; **refreshing the RM on the current policy's outputs** (iterative RLHF — fixing the distribution mismatch at the source rather than constraining the symptom); collecting preference data on the policy's own distribution from the start; and running capability evaluations every N steps as the alarm that tells you the tax has become too expensive.

---

## File Trail

| File | Role |
|---|---|
| `CS-14-The-Alignment-Map.md` | The orientation module — the map this bank is built from. |
| `CH-14-Alignment-Map.md` | The one-page reference: matrix, decision tree, VRAM calculator, starter config. |
| `CS-14 §4.6.1–4.6.2` (this module's own deep dive) | RLHF with PPO and the RL fundamentals. (`CS-24` as a separate module was never written) |
| `CS-14 §4.6.3–4.6.8` (this module's own deep dive) | DPO, IPO, cDPO, KTO, SLiC, SimPO — the method-family deep dive is **inside CS-14**, not in a separate document. (`CS-25` is cited elsewhere in this repo as "DPO" but was never written) |
| `CS-14 §4.6.10` (this module's own deep dive) | GRPO, RLVR, and the reasoning-model era. (`CS-26` as a separate GRPO module does not exist) |
| `CS-14 §4.6.9` (this module's own deep dive) | ORPO. (`CS-27` as a separate ORPO module does not exist) |
| `CS-13` (SFT) | The stage immediately upstream — you cannot align what you have not instruction-tuned. |
| `CS-13 §6.8` / `CS-11 §4.11` (LoRA/QLoRA) | The adapter mechanics and merge discipline that make DPO affordable. (`CS-23` is cited elsewhere as the LoRA deep dive but was never written) |
