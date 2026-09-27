# IQ-09 — Interview Questions: Distilling LLMs into Small Language Models

| Field | Value |
|---|---|
| **Module** | LLM → SLM distillation — response distillation, synthetic-data pipelines, on-policy GKD, model collapse, the cost model, self-preference bias |
| **Pairs with** | CS-09 (case study), CH-09 (cheat sheet) |
| **Total questions** | 44 (12 L1 + 12 L2 + 9 L3 + 5 L4 + 6 L5) + 20 rapid-fire + 3 coding tasks + 10 CS-09 self-check answers |
| **Levels covered** | Screen (L1) / Intermediate (L2) / Advanced (L3) / System Design (L4) / Debug (L5) |
| **Source material** | Video 9 `Knowledge Distillation — LLM to SLM`; `code/07_distillation.py`; `code/common/memory.py` |

**Ground truth used throughout:** CS-09 §4.9's sizing table — Qwen2.5-0.5B **47.5 MMLU / 49.6 GSM8K**, Qwen2.5-1.5B **60.9 / 68.5**, Llama-3.1-8B **69.4 / 84.5**, Llama-3.3-70B **86.0 / 95.1**, and R1-Distill-Qwen-1.5B at **83.9 MATH-500** vs GPT-4o's **74.6**. Cost: CS-09 §11.3 — 50k examples, k=3, two-stage judge, GPT-4o ≈ **$1,075** generation + **$9** student training = **$1,084**, of which training is **0.8 %**; the same 50k from expert humans is **$166,500**. Every CLI flag below was verified by running `python code/07_distillation.py --help`.

---

## How To Use This File

- **L1 = phone screen / recruiter filter** — 30-second answers. If you cannot answer an L1 in one breath, you will not reach L2.
- **L2 = working engineer** — 2–3 minutes, expects implementation detail: real flags, real numbers, real failure modes.
- **L3 = senior / specialist** — 5 minutes, expects internals and trade-offs.
- **L4 = staff / system design** — 15-minute whiteboard. Structure every answer as: requirements → constraints → design → trade-offs → failure modes.
- **L5 = debugging & incident response** — "your student does X, what do you check and in what order." Answer with a *sequence*, not a list.

**The meta-rule:** an answer of "it depends" is a failure unless it immediately says what it depends on and then picks a default. Always pick the default.

**The three answers that get people hired in this module:** (1) *supervised fine-tuning on teacher outputs is knowledge distillation, and it is forward KL — which is mode-covering, so the student hallucinates on the teacher's tail*; (2) *self-preference bias means a GPT-4o judge scoring a GPT-4o-teacher student measures stylistic mimicry, not quality*; (3) *the training run is under 1 % of the cost — the generation and filtering are the project.* Everything else here is a consequence.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. What is the structural difference between BERT-era and LLM-era distillation?**

- **Answer:** The output space. A BERT-era teacher emits a small distribution over `num_labels` (3, or 30k for MLM), so you can cache full soft targets cheaply and do token-level logit KD. An LLM teacher emits a distribution over a **150k vocabulary per position**, which is 600 KB/token at fp32 — 25.5 TB for a 42.5M-token corpus. So the LLM era defaults to **response distillation**: sample text from the teacher, filter it, and train the student on it as ordinary SFT. The loss is no longer an explicit KL; it is cross-entropy on sampled trajectories, which is *still* a KL — a forward KL, in the direction that makes the student mode-covering.
- **Why asked:** It is the single framing that explains why every other technique in the module exists.
- **Trap:** Calling response distillation "just fine-tuning on synthetic data." It is a KL with a known direction and a known failure profile (§4.3).

**Q2. Why is SFT on teacher outputs *knowledge distillation*, mathematically?**

- **Answer:** Because a forward KL `KL(p_T‖p_θ)` decomposes into `Σ p_T log p_T − Σ p_T log p_θ`, whose first term is constant in `θ` — so minimising KL is exactly maximising `E_{y~p_T}[log p_θ(y)]`. Training on samples drawn from `p_T` with cross-entropy is a Monte-Carlo estimate of that expectation. **Supervised fine-tuning on teacher generations *is* the gradient of the forward KL, in expectation.** The direction matters: forward KL is mode-covering, so it punishes `p_θ = 0` where `p_T > 0` — the student is driven to spread mass over the teacher's whole support, including low-probability regions it cannot model well, which is where the hallucination on the tail comes from.
- **Why asked:** It converts "we generate data and train" from a recipe into a principled method, and the *direction* of the KL is the part that predicts the failure profile.
- **Trap:** Saying the two are "equivalent by coincidence." The equivalence is exact up to the constant, and it is why the method works at all.

**Q3. What is response distillation, and how does it differ from logit KD?**

- **Answer:** Response distillation uses only the teacher's **sampled text** — no distribution — so it works against a closed API, needs no shared vocabulary, and is the practical default. Logit KD uses the teacher's **full distribution per position**, requires both models resident and a shared vocabulary, and has a storage wall. The trade: logit KD carries strictly more information per token; response distillation carries enough, and is the only one that scales.
- **Why asked:** It is the fork every design starts at, and the candidate should reach for the constraint (API teacher? vocabulary?) rather than a preference.
- **Trap:** Choosing logit KD for quality and then discovering the teacher is an API. Re-scope before writing code — CS-08 §8.2 STOP condition 5.

**Q4. Can you do logit KD from Llama-3.1-8B into Qwen2.5-1.5B?**

- **Answer:** No. `V_Llama = 128,256` and `V_Qwen = 151,936`, and more importantly the token *boundaries* differ, so per-position alignment is meaningless even where ids overlap. The failure presents four ways: a shape mismatch (`RuntimeError: The size of tensor a (128256) must match the size of tensor b (151936)`), the silent variant where `V` happens to match but the id→token mapping differs (finite, falling, meaningless KL), a projection that buys nothing, and `ValueError: Expected input batch_size …` when the two were tokenised separately so `T_t ≠ T_s`.
- **Why asked:** It is the constraint that kills most attempted token-level LLM KD runs, and the *silent* third case is the one that ships.
- **Trap:** Believing vocab *size* equality is the test. The test is `tok_T.get_vocab() == tok_S.get_vocab()` — a full dict comparison, which CH-09 §11 lists as the guard.

**Q5. The notebook's LLM section does token-level KD across two models. Why does it run?**

- **Answer:** Accidentally. `microsoft/phi-2` and `microsoft/phi-1_5` both use the **CodeGen tokenizer, V = 51,200** — so the vocabulary alignment happens to hold. The video never states this, and the demo teaches the wrong lesson: the reason it works is a property of the model *pair*, not of the technique.
- **Why asked:** It is the module's best "spot the hidden precondition" question, and it inverts the demo's apparent message.
- **Trap:** Reproducing the notebook with Llama + Qwen and concluding the code is broken. The code is fine; the pair is wrong.

**Q6. What is model collapse, in one sentence?**

- **Answer:** Shumailov et al. (*Nature*, 2024): if you train a model on its own generated data and **replace** the real data each round, the tails of the distribution lose mass — rare events are never sampled, so never learned, so never sampled again — and variance shrinks to zero across generations.
- **Why asked:** It is the headline risk, and the interviewer wants the *mechanism* (tail mass loss), not the word "collapse."
- **Trap:** Believing distillation itself collapses. A one-way transfer from a fixed teacher has no feedback loop and cannot collapse; distilling a stronger teacher **degrades** (the student inherits the teacher's errors at full weight) but does not collapse.

**Q7. Distillation vs fine-tuning vs RAG vs prompting — when do you *not* need a small model?**

- **Answer:** Distil when you need to remove a per-token API cost or a latency floor on a **narrow, stable** task. Do not distil when the task is broad and open-ended (fine-tune a small open model instead), when the knowledge changes faster than you can re-distil (RAG), or when the traffic is small enough that the API bill never amortises the engineering (the breakeven question, §16.4). The four are not alternatives on one axis — distillation is a *serving-cost* play, RAG is a *knowledge-freshness* play, prompting is a *no-training* play.
- **Why asked:** It is the decision layer above the module, and the honest answer includes "don't."
- **Trap:** Treating distillation as the default because it is the module's subject.

**Q8. Name the three KD modes and when each applies.**

- **Answer:** **Token-level KD** — teacher and student resident, shared vocabulary, soft targets at temperature; highest signal per token, storage wall. **Sequence-level / response KD** — teacher offline, sampled text, ordinary SFT; the practical default. **On-policy GKD** — the *student* generates, the teacher scores every prefix; best when the capacity gap is large, but the teacher must be available during training, which rules out APIs.
- **Why asked:** It is the taxonomy with the *constraint* attached to each, which is what makes it usable.
- **Trap:** Listing them as quality tiers. GKD is not "better sequence-level KD" — it solves a different problem (exposure bias).

**Q9. Give the student sizing law.**

- **Answer:** `Quality ≈ a·log(params) + b·(data quality) + c` in the 1–10B range: each **doubling of parameters ≈ +4–6 MMLU points**, while each *quality tier* of data (raw web → filtered → synthetic textbook) ≈ **+5–15 MMLU points** and is a one-time cost. Practical floors: ~**0.3B** to hold a chat format at all, ~**1.5B** for a usable general assistant and for reliable JSON/function calling (or ~3B with constrained decoding), ~**3–8B** to compete with a 2023-era 70B on a narrow task, and ~**1.5B** for long-chain reasoning *if distilled from a reasoning teacher*.
- **Why asked:** It reframes "how small can I go" as a **data** question, which is the module's thesis.
- **Trap:** Quoting parameter-count floors without the data qualifier. R1-Distill-Qwen-1.5B's 83.9 MATH-500 is the counterexample to every older floor.

**Q10. How small can you go before it is unsafe to ship without a filter?**

- **Answer:** No size is safe. A distilled student inherits the teacher's **responses**, not its safety training — so the student reproduces the teacher's *refusals and non-refusals* only to the extent they appear in the distilled corpus, which is a filter artefact, not a policy. Floors exist for *capability*; there is no floor for safety.
- **Why asked:** It is the question most candidates answer with a number.
- **Trap:** Assuming a 7B student is safer than a 1.5B student. Neither has a safety layer unless you add one.

**Q11. What is the self-preference bias, and why does it invalidate most distillation claims?**

- **Answer:** An LLM judge scores text from its own model family higher, and the bias correlates with the judge's ability to recognise its own generations (Panickssery et al. 2024). So if the teacher is GPT-4o and the judge is GPT-4o, your win rate measures **stylistic mimicry, not quality**. Worse, if your dDPO judge is GPT-4o and the teacher is GPT-4o, you are distilling the judge's bias *as a preference* — a bias expressed as a preference is harder to detect than a bias expressed as a response, because the student internalises it into its reward structure.
- **Why asked:** It is CS-09 §12.4 and it is the reason Orca's AGIEval/BigBench scores were contested.
- **Trap:** Using a different judge from the teacher and calling it solved. Judge family ≠ teacher family is **necessary but not sufficient** — a Claude judge on a Claude-teacher-derived student is still self-preference.

**Q12. The one-line rule for legal exposure.**

- **Answer:** *Distil from open-weights teachers whenever you can; when you must use a closed teacher's outputs, distil a narrow task capability rather than a general assistant, use the vendor's zero-retention endpoint, do not redistribute the teacher's raw outputs, document provenance, and get the terms reviewed.* The crux of every vendor clause is the difference between **competition** and **internal task specialisation**; and the two instruments are separate — the **weights licence binds the artefact you ship**, the **ToS binds the dataset you generated**. Open weights (Llama Community Licence, Gemma Terms, MIT) explicitly permit derivative models, which is precisely why every serious open recipe uses an open teacher or a released dataset.
- **Why asked:** It is the question with a real answer that is not legal advice, and the licence-vs-ToS distinction is the technically correct part.
- **Trap:** Confusing the two instruments — being compliant on one and not the other is the common real-world failure.

---

## Level 2 — Applied & Implementation

**Q13. Walk the end-to-end response-distillation pipeline in order.**

- **Answer:** (1) Build the prompt set — use **real user prompts** if you have them, 500–50,000; they define the distribution that matters. (2) Generate with the teacher, k = 1–3 samples per prompt, greedy or low-temperature, pinning the **dated** model id per example. (3) Filter — refusals, boilerplate, instruction-echoing, truncation, length, n-gram repetition — because every one of those is a training target at full weight. (4) **Decontaminate** against your evaluation set with the 13-gram rule. (5) Train the student as ordinary SFT, mixing in ≥5–10 % human replay. (6) Evaluate with an objective metric *next to* any judge number, plus a prompt-level holdout.
- **Why asked:** It is the pipeline you will actually operate, and step 3 is the one that decides quality.
- **Trap:** Going straight from generation to training. CH-09 §12.3 lists the specific defects that survive unfiltered generation.

**Q14. Why generate k = 3 samples per prompt instead of 1?**

- **Answer:** It buys diversity and a filtering choice at 3× the generation cost and **zero** extra training cost. In CS-09 §11.3's worked example k = 3 costs **$975** against k = 1's **$325** on GPT-4o — 3× generation, and generation is ~90 % of the project. The payoff: you can keep the best candidate per prompt under a judge, or keep all three as three training rows, which is genuinely more data. What it does not buy is quality per row; k = 3 of a bad teacher is three bad rows.
- **Why asked:** It is a cost/benefit decision with a specific number, and the interviewer is testing whether you know generation dominates the budget.
- **Trap:** Raising k to 10 "for diversity." The cost is linear and the marginal diversity is not.

**Q15. What is the filter stack, in order?**

- **Answer:** Apply cheapest-first, because each stage is more expensive than the last: (1) **mechanical** — empty, too short, too long, truncation at `max_new_tokens`, exact duplicates; `code/07_distillation.py` drops anything under 5 words during generation (`if len(resp.split()) < 5: continue`). (2) **format** — instruction echoed back, prompt prefix present, malformed structure, language mismatch. (3) **content** — refusal patterns, hedging boilerplate ("It's important to note…"), degenerate n-gram repetition. (4) **semantic** — dedup by embedding, then a judge or verifier on the survivors. (5) **contamination** — 13-gram overlap against the evaluation set. The ordering matters because stage 4 costs money per row and stages 1–3 are free.
- **Why asked:** It is the quality gate, and the ordering is the engineering judgement.
- **Trap:** Running the judge first. You pay a model to reject rows a regex would have caught.

**Q16. What is the 13-gram decontamination rule and where does it go?**

- **Answer:** Compute the set of 13-gram shingles in every evaluation example, then drop any *generated training row* that shares a 13-gram with any eval example. It goes **after** generation and **before** training, and it belongs in the pipeline as a gate, not a post-hoc audit. The n = 13 choice is the published convention (GPT-3-era) — long enough that a shared 13-gram is not coincidence, short enough to catch paraphrase-adjacent copying.
- **Why asked:** Contamination is the failure that turns a good result into a retraction, and it is invisible in the metrics.
- **Trap:** Decontaminating against the *training* set instead of the *evaluation* set. The direction is the whole point.

**Q17. How do you size the dataset?**

- **Answer:** The rule of thumb band is **20M–200M response tokens** total, i.e. `examples × tokens/example × epochs`. CS-09 §11.3's worked example is 50,000 examples × 850 tokens × 3 epochs = **127.5M tokens** — mid-band. Below ~20M the student underfits the teacher's distribution; above ~200M you are paying for data you cannot absorb, and the correct move is to raise data *quality* instead (the sizing law's +5–15 points per quality tier).
- **Why asked:** It is the only sizing number that governs the project's cost, and it is downstream of the *data* decision, not the model one.
- **Trap:** Scaling examples without checking tokens per example. 50k short rows and 50k long rows are different datasets.

**Q18. A student trained on response distillation is fine on teacher-forced perplexity and poor in free generation. What is it, and what do you do?**

- **Answer:** **Exposure bias.** Training conditions on *teacher* prefixes; inference conditions on the student's *own*. The moment the student makes one token choice the teacher would not have made, it is off the training manifold and unconstrained. Fix in order of cost: (1) more data, so the student's error distribution overlaps the teacher's prefix distribution more; (2) on-policy GKD — the student generates, the teacher scores every prefix of that generation, so training and inference prefixes match; (3) accept the ceiling and constrain decoding.
- **Why asked:** It is the failure that makes "it works on paper" diverge from "it works," and GKD exists to fix it.
- **Trap:** Blaming the data volume first. Measure the teacher-forced-vs-free gap *before* deciding, because that gap is the diagnostic (CS-09 §12.5's third test).

**Q19. What is GKD, and which of its two variants do you pick?**

- **Answer:** On-policy distillation (Agarwal et al. 2024): the student generates `ŷ ~ p_θ`, the teacher scores every prefix of `ŷ`, and you minimise `D(p_T(·|x, ŷ_<t) ‖ p_θ(·|x, ŷ_<t))` at each `t`. Two variants: **teacher-data / forward KL** (mode-covering — the student tries to match the teacher everywhere) and **student-data / reverse KL** (mode-seeking — the student concentrates on what it can actually do well). The paper's finding on T5-XL → T5-small/base is that on-policy beats supervised KD, the advantage **grows with the capacity gap**, and the **student-data / reverse-KL variant wins when you generate the training data from the student**. Practical constraint: the teacher must be resident during student training, so it is a self-hosted-teacher technique.
- **Why asked:** It is the strongest 2025-era signal in the module, and the variant distinction is what separates knowledge from a paper title.
- **Trap:** Reaching for GKD with an API teacher. It cannot work at any volume.

**Q20. What are the four conditions under which recursive synthetic data causes collapse?**

- **Answer:** From CS-09 §4.8: (1) recursive self-training **replacing** real data each round with no verifier → **collapses** (tails never sampled). (2) Recursive self-training **accumulating** data with enough real data → **does not collapse**; Alemohammad et al. 2024 show it is stable or converges to the real distribution, because the real data anchors the tails every round. (3) Distilling from a **stronger** teacher with a verifier → **does not collapse**; it is a one-way transfer, not self-consumption. (4) Distilling from a stronger teacher **without** a verifier → **degrades but does not collapse**; the student inherits the teacher's confident wrong answers and hallucinated citations at full weight. And the special case: distilling **within your own family** (7B → 3B → 1.5B → 0.5B) with no human data and no verifier → **collapses, and fast**, because that is recursive self-training wearing a distillation costume.
- **Why asked:** The headline says "synthetic data collapses models"; the actual claim is much narrower, and the narrow version is the actionable one.
- **Trap:** Answering "replace vs accumulate" as the only distinction. The **verifier** and the **direction of the transfer** are separate conditions.

**Q21. Which single mitigation prevents collapse, and what is the measurable signature?**

- **Answer:** The mitigation: **never let any training mix reach zero human tokens.** Concretely, keep ≥**5–10 % human-authored examples** in every SFT mix, and require a **verifier** (not a judge) for any dataset you will recurse on. Meta's Llama 3 report used synthetic data extensively in post-training with no mode collapse, attributing it to exactly this. The signature: track `distinct-3`, self-BLEU between samples at temperature 1.0, and embedding-space coverage. A collapsing dataset loses `distinct-3` monotonically across generations **while its judge score rises** — the judge likes the confident mode. That divergence is the alarm, and it is why a rising win rate is not evidence of health.
- **Why asked:** It is the operational form of Q20, and the divergence between diversity and judge score is the detail that shows you have run one of these loops.
- **Trap:** Monitoring judge score alone. It moves in the *wrong direction* during collapse.

**Q22. Where does `completion_only_loss` come from and what breaks without it?**

- **Answer:** It masks the prompt tokens out of the training loss so only the response is supervised. Without it the student is trained on the prompt too, and the failure is specific and ugly: loss falls, and generations come out as `!!!!!!` or whitespace — because the model learned that the highest-probability continuation of a prompt is more prompt. CH-09 §11 lists the pair together: `completion_only_loss=True` **and** `ignore_index = pad_token_id`. Set both; they are different masks for different tokens.
- **Why asked:** It is the most common visible defect in a hand-rolled response-distillation training script.
- **Trap:** Setting `completion_only_loss=True` and leaving pads at the pad id, so the model also learns to emit pad tokens at full weight.

**Q23. The student emits the teacher's system prompt. What happened?**

- **Answer:** The system prompt was stored inside the `response` field instead of the prompt side of the schema, so it became a training target. The fix is structural (keep the system prompt separate) and the guard is an assertion — `assert not response.startswith(system_prompt[:40])` — over the whole corpus, not a spot check. This is a filter failure, not a training failure, which is why it survives every loss-curve review.
- **Why asked:** It is a specific, memorable, real defect and it teaches the general rule that dataset schema errors present as model behaviour.
- **Trap:** Diagnosing it as a decoding or repetition-penalty problem.

**Q24. Give the token-level KD loss the way the repo script writes it, and name the two convention traps.**

- **Answer:** On the **top-k truncated** tensors the repo script builds: `s_log = F.log_softmax(s_top / T, dim=-1)`, `t_soft = F.softmax(t_top / T, dim=-1)`, then `kd = F.kl_div(s_log.reshape(-1, k), t_soft.reshape(-1, k), reduction="sum") / n_pos * (T * T)`, `hard = F.cross_entropy(...)`, and `loss = a.alpha * kd + (1 - a.alpha) * hard` — **α weights the SOFT (KD) term**, default 0.7, which is Hinton's convention and CH-08 §4's. The script prints `hard CE`, `KD (x T^2)` and `mixture 0.70 x KD + 0.30 x CE   ← --alpha weights the SOFT term` on every run, and warns when the KD term lands under 10 % of the CE term despite carrying most of the weight. **Trap 1, the α side:** the *same symbol* means the opposite thing in three places in this repo's own material — CS-08 §4.2.4 tabulates Hinton's reading (`α` on hard, 0.1) against the notebook's (`α` on soft, 0.7), a **76 %** spread in the worked loss (0.1576 vs 0.2766), and CH-09 §2 and §4.1 still carry a table asserting this script is *"Inverted vs Hinton"* with default 0.5, which describes an earlier revision of the file. Read the loss line. **Trap 2, the reduction:** `batchmean` on a `(B·L, V)` full-vocabulary tensor divides by the batch dimension and is right; on a `(n_pos, k)` top-k tensor it divides by `k`, not by the position count — so the script uses `"sum"` over `n_pos`. `mean` is wrong for both (it divides by `numel`, a factor of `V` ≈ **150,000** at a 150k vocabulary, and the soft term vanishes).
- **Why asked:** These are the two silent bugs that make a KD run look like plain SFT, and both have bitten this repo's own artifacts.
- **Trap:** Trusting a flag's help string *or a quoted line number*. `--help` and the arithmetic on the loss line are the tiebreakers; a cached citation of "line 74" or "line 261" is not, because the convention has been changed in that file before.

---

## Level 3 — Advanced, Internals & Theory

**Q25. Derive why SFT on teacher outputs is a forward KL, and say what the direction implies.**

- **Answer:** `KL(p_T‖p_θ) = Σ_y p_T(y) log p_T(y) − Σ_y p_T(y) log p_θ(y)`. The first term does not depend on `θ`, so `argmin_θ KL = argmax_θ E_{y~p_T}[log p_θ(y)]` — maximum likelihood on samples from the teacher. Sampling `y ~ p_T` and minimising cross-entropy is a Monte-Carlo estimate of that. **Forward KL is mode-covering:** it assigns infinite cost to `p_θ = 0` wherever `p_T > 0`, so the student is pushed to put mass everywhere the teacher does — including the long low-probability tail it cannot represent. That is the mechanistic origin of "the student is confidently wrong in the teacher's style": it has been *forced* to cover regions where it has no capacity, and it does so by flattening.
- **Why asked:** It is the theoretical spine of the module, and the mode-covering inference is the part that predicts behaviour.
- **Trap:** Saying reverse KL is "the same thing with the arguments swapped." It is mode-*seeking* — it penalizes `p_θ > 0` where `p_T = 0`, which produces a student that is narrow and confident rather than broad and flat. That is exactly the GKD variant choice in Q19.

**Q26. Quantify where the dark knowledge is for a 150k LLM vocabulary, and say why that kills full-vocab caching.**

- **Answer:** The KL mass lives in the top two or three classes, not the tail. CS-08 §4.2.4's extension: a class the teacher gives 0.001 and the student 0.01 contributes `0.001·ln(0.001/0.01) = −0.0023` nats — and the negative sign means a class the student *over*-weights reduces the KL, so it is not a supervision signal at all. Against `KL = 0.02453` at T = 2, a tail class contributes ~10 % of the loss at best and numerical noise at fp16 for everything past the top few hundred. So top-20 captures **>99.9 %** of the KL mass at a **6,500×** storage reduction versus full fp32. For an LLM at V = 150k that is the difference between **25.5 TB** and **6.8 GB** for a 42.5M-token corpus — which is the difference between impossible and routine.
- **Why asked:** It converts "top-k is a trick" into "top-k is a theorem about where the signal is," and it is why the LLM era does not cache full distributions.
- **Trap:** Believing a bigger `T` puts more knowledge in the tail. `T` rescales logits; it does not create signal where the teacher has none.

**Q27. The capacity gap says a 70B teacher can lose to a 7B teacher. R1 distilled 671B into 1.5B and won. Reconcile.**

- **Answer:** The law is specific to **logit KD**, and R1 used **sequence-level KD**. Sampling from `p_T` gives the student *high-probability trajectories only*, so it never has to represent the teacher's full distribution — the representational-mismatch mechanism that drives the capacity gap in logit KD is largely bypassed. Hence the reconciliation: for token-level KD, CH-08 §7.4 governs (sweep 2–10×; >50× and a mid-size teacher often wins); for **response distillation on a verifiable narrow task**, the gap can be >400× and still work, because the trace quality dominates. DeepSeek sampled 800k traces and got MATH-500 **83.9** from a 1.5B student. The practical rule: **sweep the teacher size on a verifiable narrow task rather than assuming either direction** — that is the three-teacher sweep, and it is the diagnostic almost nobody runs.
- **Why asked:** It is the single most interesting tension in the handbook, and it tests whether the candidate knows *which* mechanism the law is about.
- **Trap:** Saying "the capacity gap is wrong." It is correct for logit KD and the wrong lens for response KD. Note that CH-09 §13 already flags this, including that CS-09's own glossary row mis-points at "§12" for the counter-evidence when it is in §15.4 and §4.9.

**Q28. If response distillation is forward KL, why does the R1 result show reasoning being *transferred* rather than just style?**

- **Answer:** Because for a **verifiable** task the sampled trajectories are not arbitrary text — they are *complete reasoning paths to a checkable answer*, and the verifier is what makes them transferable. The distinction the module draws is between distilling answers (Orca-2 distils *reasoning processes*, Phi-1/2 distils *textbook-quality text*) and distilling traces with an answer check. Without a verifier, the student inherits the teacher's confident wrong answers at full weight — degradation, not collapse — and the filter's job is to remove those rows. With a verifier, the tail of plausible-but-wrong traces is pruned before training, so what is left is a distribution the student can actually learn. CS-09 §12.5's second test is the diagnostic: take 100 responses with a reasoning block and check whether the answer *follows from* the trace. Plausible trace, non-following answer → the student learned the *shape* of reasoning, which happens when you filter on the answer only and not on trace validity.
- **Why asked:** It is the difference between R1-Distill and Orca, and the verifier is the mechanism.
- **Trap:** Concluding "CoT distillation works" without the verifier. Filter on trace validity or you teach the format of reasoning.

**Q29. What is the measurable signature of a collapsing dataset, and why is the judge score misleading?**

- **Answer:** Track three quantities on the distillation set across generations: `distinct-3` (unique 3-grams / total), self-BLEU between samples at temperature 1.0, and embedding-space coverage (mean pairwise cosine distance between response embeddings). In a collapsing loop, `distinct-3` **falls monotonically** while the **judge score rises** — because the judge rewards the confident mode the collapse is converging to. That divergence between diversity and judge score is the alarm, and it is why "our win rate keeps improving each generation" is a failure signal, not a success one. The same logic explains why the judge must come from a different family from the teacher (Q11) and why an objective metric must sit next to any win rate.
- **Why asked:** It is the monitoring design for the one failure mode that is invisible in every metric people actually look at.
- **Trap:** Monitoring loss or judge score. Both improve during collapse.

**Q30. Why does a 4-bit teacher change the soft targets, and when does it matter?**

- **Answer:** NF4 quantization perturbs the logits, so the teacher's distribution is the *quantized* teacher's, not the original's — and the KL is computed against whatever distribution you actually got. For **sequence-level KD** it barely matters: the sampled text is usually still correct, which is why a self-hosted 70B at 4-bit is the standard cheap-teacher configuration. For **token-level KD** it matters more, because you are fitting the perturbation: a quantized teacher's soft targets are noisier exactly in the tail, which is where the dark knowledge is. The diagnostic is CH-09 §5.5's entropy test — if the teacher's mean max-probability is near 1 after quantization, the soft targets have collapsed toward one-hot and you are paying for an expensive label smoother.
- **Why asked:** It is the "exceptions and pros/cons" probe: quantization is safe for one KD mode and risky for the other, and candidates usually apply one verdict to both.
- **Trap:** Assuming a quantized teacher is equivalent because the generations look fine. For logit KD, "looks fine" is not the test.

**Q31. What is a prompt-level holdout and why is an example-level split not enough?**

- **Answer:** Split by **prompt**, not by example. If one prompt's k = 3 samples land on both sides of the split, the validation row is a near-duplicate of a training row and your validation number measures memorisation. The failure is silent and it is the default outcome of a naive `train_test_split` on a generated corpus, because the corpus is naturally grouped by prompt. The same logic applies across the whole project: a **prompt-cluster** holdout — an *intent* you generated no data for — is the honest generalisation test, and CS-09 §12.5's fourth diagnostic is that accuracy collapsing on a held-out prompt cluster means you distilled a *prompt distribution*, not a task.
- **Why asked:** It is the evaluation-integrity question, and it is one of CH-09 §12.2's checks.
- **Trap:** Reporting an example-level split as a held-out number. It is a leak, and the tell is a suspiciously high score.

**Q32. Give the honest comparison between a distilled 1.5B and the API it replaces.**

- **Answer:** It is a FLOPs question, not a parameter question, and the answer is often "not yet." Compute the breakeven before starting: `generation + engineering_hours × rate + training` against `monthly_api_saving × months`. CS-09 §16.4's worked case is **$5,075** of setup against a **$2,000/month** API bill → **~2.5 months** to amortise. Against a **$200/month** bill the engineering time **never amortises**, so the correct answer is "do not distil" — and that is a legitimate interview answer. The second-order terms that decide it: the student is a permanent second model in the registry, a monthly refresh costs `+1 generate +1 distil` (≈3× a monthly fine-tune), and if you re-sample the prompt pool each month you pay generation every month — the cache is only reusable if the transfer set did not change.
- **Why asked:** It is the one design question where the expected answer is a number and possibly "no."
- **Trap:** Comparing per-token API price to GPU-hour price without the engineering time or the refresh cadence.

---

## Level 4 — System Design & Scenario

> These are 15-minute whiteboard questions. Structure every answer as: requirements → constraints → design → trade-offs → failure modes.

**Q33. Design a 1.5B student from a 32B open-weights teacher on a $1,000 budget.**

- **Answer:** Requirements: 1.5B student, broad instruction following, $1,000. Constraints: generation dominates. Design: self-host the 32B teacher at int8 (**30.9 GiB**, per CH-09 §7.1 — one 48 GB card or two 24 GB) and generate 50k prompts × k = 3 at 850 tokens each. If you rent instead of using an API, generation is GPU-hours, not per-token — and the API price list (CS-09 §11.2) puts GPT-4o at **$2.50/M in, $10/M out** and a self-hosted 70B at **$0.97/M out**, which is 4.6× cheaper than a hosted 7B at **$0.19** only in the wrong direction: a hosted small model is cheapest per token, and a self-hosted large one is cheapest per *quality* token. Run the two-stage judge: cheap stage over all 150k candidates (**$20.03** on GPT-4o-mini), strong stage over the 45k survivors (**$100.13** on GPT-4o). Filter mechanically first so the judge sees fewer rows. Trade-offs: sequence-level only, because the teacher is a different family from the student; student training is **2.6 A100-hours ≈ $9** for 1.5B at 127.5M tokens — **0.8 %** of the budget. Failure modes: unfiltered generations teach verbosity and hedging; no human replay risks collapse; no 13-gram decontamination invalidates the eval.
- **Why asked:** It forces the candidate to notice that the *training* is free and the *data* is the project.
- **Trap:** Budgeting for GPU training and treating data generation as negligible. CS-09 §11.3 is 1,075 vs 9.

**Q34. Design the teacher-selection experiment.**

- **Answer:** Requirements: pick a teacher and defend it. Design: the **three-teacher sweep** — distil the *same* student from teachers at roughly 3×, 30× and 300× the student's size, identical hyperparameters, and evaluate on a **verifiable narrow task** plus a general set. The expected shape depends on the mode: for token-level KD, non-monotone with a peak in the 2–10× band; for response KD on a verifiable task, monotone up to and past 400× (R1's 671B → 1.5B). Constraints: `n_teachers × (teacher fine-tune + distillation run)` — use a few thousand examples and 2 epochs. Trade-offs: teacher *training length* is a second axis (Cho & Hariharan: student accuracy is non-monotone in teacher training length, peaking early), so the best KD teacher may be one you would not ship. Failure modes: assuming the teacher you already have is right because it is the one you have; skipping the from-scratch and label-smoothing baselines, so you cannot distinguish a KD gain from a capacity gain.
- **Why asked:** CH-09 §5.4 calls this "the one diagnostic almost nobody runs" — and it is the answer to the capacity-gap question in practice.
- **Trap:** Running the sweep on a general benchmark instead of a verifiable narrow task, where the noise swamps the difference between teachers.

**Q35. Design the evaluation for a distilled student, given a teacher whose family matches your judge.**

- **Answer:** Requirements: the judge is the same family as the teacher — so the default win rate is invalid. Design, in order: (1) put an **objective metric** next to every win rate — the objective metric eliminates the entire class of problem where it exists; (2) change the judge to a **different family** from the teacher (for a GPT-4o teacher, judge with Claude or Gemini); (3) **position-swap** every pairwise comparison and convert disagreements to ties; (4) report the judge's identity beside every number plus **inter-judge agreement** across two judges — if they disagree by more than **~15 points** on the same pairs, the win rate is noise and you report both; (5) **length-controlled** win rate to remove the verbosity confound; (6) a **human spot-check on 100 pairs**, the only ground truth available; (7) never let the judge be a model you distilled. Calibration: a distilled 1.5–3B student in the **high-50s** AlpacaEval 2.0 LC is doing well, and anything above that should trigger a **contamination audit** before you celebrate (CS-09 §12.3). Trade-offs: the full stack costs money per evaluation, so run 1–4 per commit and 5–7 per release. Failure modes: reporting a 20 %→65 % win-rate jump after switching judges and calling it an improvement — it is two explanations and the position-swap test distinguishes them.
- **Why asked:** It is the module's evaluation chapter compressed into a design, and the mitigation ordering is the graded part.
- **Trap:** Switching the judge and reporting the new number without reporting the old one.

**Q36. Design the refresh loop for a distilled model whose teacher is a closed API.**

- **Answer:** Requirements: monthly refresh, closed teacher, cost control. Design: (1) **freeze the prompt set** if you can — the generation cache is reusable across months *only* if the transfer set did not change, so a frozen prompt set turns a monthly generation bill into a one-time cost; (2) if you must re-sample the prompt pool, budget generation every month and re-run the cost model; (3) pin the **dated** teacher model id per example, because an alias that moves makes your corpus non-reproducible and mixes distributions; (4) checkpoint every 500 examples and use exponential backoff with a concurrency cap, because `openai.RateLimitError: 429` mid-run is the standard failure; (5) keep the previous student and the previous thresholds — rolling back the model without rolling back the thresholds is the most common post-rollback incident; (6) monitor agreement with a shadow teacher on a 1 % traffic sample. Trade-offs: each refresh is `+1 generate +1 distil` on top of a fine-tune, ≈3× the compute. Failure modes: the alias moves and half your corpus silently comes from a different teacher; the filter's kill-rate drifts and nobody watches it; the prompt pool is re-sampled but the budget is not.
- **Why asked:** It is the operational life of the pipeline, and the frozen-prompt-set observation is the cost lever most candidates miss.
- **Trap:** Assuming a one-time distillation cost. The recurring cost is the generation.

**Q37. A colleague proposes distilling within your own family: 7B → 3B → 1.5B → 0.5B, no human data. Design the response.**

- **Answer:** Requirements: a smaller deployable model, no human data available. Constraints: this is exactly CS-09 §4.8's last row — recursive self-training wearing a distillation costume — and it collapses, and fast. Design the response: (1) refuse the chain as specified; (2) the fix is not "add a verifier" alone and not "add human data" alone — you need **≥5–10 % human-authored examples in every SFT mix** *and* a **verifier** (not a judge) for anything you recurse on; (3) collapse the chain to a **single** step from the strongest member (7B → 1.5B) rather than three hops, because every hop multiplies tail-mass loss and a one-way transfer from a fixed model has no feedback loop; (4) instrument the loop with `distinct-3`, self-BLEU and embedding coverage, and treat a rising judge score with falling `distinct-3` as the alarm; (5) if no human data exists at all, distil *once* from the 7B and stop — do not recurse. Trade-offs: the single-hop student is worse than the chain *would* have been if it worked, which is why the chain is tempting; and the honest alternative is to fine-tune the 1.5B on the same target data instead. Failure modes: the chain looks fine for two generations and degrades at the third; the judge score rises throughout, so every dashboard is green.
- **Why asked:** It is the "when NOT to do it" design, and the answer requires knowing that the *chain structure* is the defect, not just the data mix.
- **Trap:** Proposing to add a judge. A judge is a model from the same distribution and will reward the mode; the rule requires a verifier.

---

## Level 5 — Debugging & Incident Response

> Answer with a *sequence*: what you check first, what the check rules out, and what you do if it does not.

**Q38. Loss falls, generations come out as `!!!!!!` or whitespace. Sequence.**

- **Answer:** (1) You are training on the prompt tokens *and* the pads. Set `completion_only_loss=True` and `ignore_index = pad_token_id` — both, because they mask different tokens. (2) Verify the mask is real: print one batch and check `(labels != -100).float().mean()` is close to the completion fraction, not 1.0. (3) Check the collator: a `DataCollatorWithPadding` passes labels through and pads with the pad id, which is exactly this bug. (4) Check `tok.pad_token` — most Llama/Qwen tokenizers have none, so `pad_token = eos_token` and pad positions carry a *real* target id. (5) Only then look at decoding parameters.
- **Why asked:** It is the most common hand-rolled-SFT defect and every step above is a different cause with the same symptom.
- **Trap:** Adding `repetition_penalty`. The model is not repeating; it is continuing padding.

**Q39. `RuntimeError: The size of tensor a (128256) must match the size of tensor b (32064) at non-singleton dimension 2`. Sequence.**

- **Answer:** (1) Confirm the pair: `V_Llama-3.3-70B = 128,256` and `V_Phi-3-mini = 32,064` — two different families, so this is token-level KD across tokenizers and it cannot work. (2) Do **not** test for the fix by comparing sizes; test `tok_T.get_vocab() == tok_S.get_vocab()`, because the *silent* variant of this bug is equal `V` with a different id→token mapping, where the KL is finite, falling, and meaningless. (3) Switch to sequence-level KD, or pick a same-family student. (4) Note `code/07_distillation.py` already makes this a go/no-go — on `--token-kd` it loads the tokenizers even under `--dry-run` precisely to run the alignment check and exit before spending anything, so a run that reaches this error skipped its own guard.
- **Why asked:** It is the error that defines the LLM-era constraint, and step (2) is the one that generalises.
- **Trap:** Sizing the smaller logit tensor to match. The ids do not mean the same things.

**Q40. `torch.cuda.OutOfMemoryError` on the *second* `from_pretrained`. Sequence.**

- **Answer:** (1) Both models loaded in fp32 — `from_pretrained` does not infer dtype. `torch_dtype=torch.float16` (or bf16). The Phi-2/Phi-1.5 pair is **16.0 GB in fp32 vs 8.0 GB in fp16**, so the second load is the one that tips over. (2) Budget the whole two-model case, not the weights: a 32B int8 teacher is **29.8 GiB** of weights plus a 1.5B QLoRA student (**1.1 GiB**) plus 1.0 GiB overhead plus the **full-vocab softmax buffers** — at `seq 512, V = 151,936` that is **311 MB per tensor × 3 = 0.93 GiB**. Total **~32 GiB** on one device (CH-09 §7.2). (3) The top-k truncation is not only a storage trick — it is what makes the two-model forward pass *fit*, dropping the softmax buffer term to ~0. (4) If it still does not fit, move the teacher to its own process and cache; `device_map="auto"` plus a manual gather is the source of the companion error, `Expected all tensors to be on the same device`.
- **Why asked:** It separates "I know VRAM" from "I have sized a two-model forward pass," and the buffers are the term everyone forgets.
- **Trap:** Reducing batch size first. The buffers scale with `seq × V`, not with batch, so at seq 512 they are small and the weights are the problem.

**Q41. Soft loss is `nan` from step 1. Sequence.**

- **Answer:** (1) fp16 overflow in a 150k-way softmax, or `log(0)` on a zero-probability target. Do the softmax and log-softmax in **fp32**, keep the clamp, and `clip_grad_norm_(1.0)`. (2) Check `T > 0`. (3) Check the teacher's logits are float, not int — `ValueError: Expected input to be a floating point tensor` is the explicit version of this. (4) Check the top-k gather: an `IndexError` or a mis-gathered index puts `-inf`-adjacent values into the log-softmax. (5) If soft loss is constant at exactly `log(V)`, the teacher's distribution has collapsed to uniform — lower `T`, check a quantized teacher, and confirm the teacher generates sane text at all.
- **Why asked:** Three distinct causes with the same symptom, and the last one (`log(V)`) is a *different* bug wearing the same clothes.
- **Trap:** Assuming `nan` means the loss function is wrong. Step (5) is a teacher bug.

**Q42. The distilled student's win rate jumps from 20 % to 65 % when you switch the judge from GPT-4o to Claude. Sequence.**

- **Answer:** (1) Two explanations: **self-preference bias** — the GPT-4o judge was penalising a non-GPT-4o-family student or rewarding its own family elsewhere — or a **genuine** judge-specific artifact such as position or length bias in one of the two judges. (2) The distinguishing test is **position-swapped pairwise judging**: if swapping the order moves the numbers materially, the difference is position bias, not quality. (3) Report both numbers plus the **inter-judge agreement**; if the two judges disagree by more than **~15 points** on the same pairs, the win rate is noise. (4) Add an **objective task metric** on the same prompts — if accuracy is flat while the win rate moved 45 points, you are measuring style. (5) Never resolve it by picking the judge that gives the better number, and never let the judge be a model you distilled.
- **Why asked:** It is CS-09 §19's question 9, and it is deliberately constructed so that both explanations are plausible.
- **Trap:** Concluding the Claude judge is "more accurate." Both are measurements with biases; the position-swap test is what separates them.

**Q43. The student's trace is plausible but the answer does not follow from it. Sequence.**

- **Answer:** (1) This is CS-09 §12.5's second test — the student learned the *shape* of reasoning. (2) Root cause: you filtered on the **answer** only, not on **trace validity**, so traces that reach a correct answer by invalid steps were kept and imitated. (3) Fix: add a trace-validity gate — for a verifiable task, a verifier that checks the intermediate steps, not just the final answer. (4) Check the split between style and substance too: run a judge-based win rate *and* an objective task accuracy on the same prompts; win rate ↑ with accuracy flat is style transfer only, meaning the dataset taught format. (5) Check the prompt-cluster holdout — if accuracy collapses on an intent you generated no data for, you distilled a prompt distribution rather than a task. (6) Capability-ceiling probe: evaluate on a task the teacher is good at that you generated *no* data for; collapse to base-model level is expected, because response distillation transfers no general capability.
- **Why asked:** It is the "learned the style, not the substance" diagnostic, which is the most common disappointing outcome and the one with a specific root cause.
- **Trap:** Collecting more data. More invalid traces teach the shape harder.

**Q44. The filter's kill rate is 40 % on Monday and 8 % on Tuesday, same prompts and teacher. Sequence.**

- **Answer:** (1) The teacher moved. API aliases are not pinned, so `gpt-4o` on Tuesday is not the `gpt-4o` you pinned per example — check the returned model id on both days and the `created` timestamps. (2) The teacher's sampling changed: `temperature > 0`, no seed, so two runs on the same prompt give different output; the filter's kill rate is a distribution statistic and it moves with the sample. (3) A filter rule changed or a dependency shifted — a refusal classifier's threshold, a `tiktoken` version changing the length computation, an embedding model update changing dedup. (4) Check truncation: if `max_new_tokens` is being hit more often on Monday, the kill rate rises and the cause is upstream. (5) The durable fix is to pin the **dated** teacher id per example and store it in the row, so the corpus is reproducible and a mixed-distribution corpus is detectable. Accept that API generation is not bit-reproducible.
- **Why asked:** It is a real operational incident with a reproducibility root cause, and the *dated model id per row* is the design answer.
- **Trap:** Retuning the filter thresholds. That makes the corpus inconsistent across days, which is worse than either kill rate.

---

## Rapid Fire — True / False / One-Liner

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | SFT on teacher outputs is a form of KL minimisation | **True** | Forward KL: `argmin KL = argmax E_{y~p_T}[log p_θ]` |
| 2 | Forward KL is mode-seeking | **False** | It is mode-covering; reverse KL is mode-seeking |
| 3 | Logit KD works from any API teacher | **False** | APIs return text, not distributions |
| 4 | Llama-3.1-8B and Qwen2.5-1.5B can share a token-level KD loss | **False** | 128,256 vs 151,936, and different boundaries |
| 5 | Phi-2 and Phi-1.5 share a tokenizer | **True** | Both CodeGen, V = 51,200 — why the notebook demo runs |
| 6 | Equal vocabulary sizes guarantee safe logit KD | **False** | The id→token mapping must match; the silent variant gives a finite, meaningless KL |
| 7 | Distilling a stronger teacher can cause model collapse | **False** | It degrades — the student inherits errors at full weight; no feedback loop |
| 8 | Distilling within your own family with no human data collapses fast | **True** | Recursive self-training wearing a distillation costume |
| 9 | Accumulating instead of replacing synthetic data prevents collapse | **Only with real data present** | The real data anchors the tails every round |
| 10 | Zero human tokens in the mix is safe if you have a judge | **False** | You need a **verifier**, and ≥5–10 % human replay |
| 11 | A rising judge score across generations is good news | **False** | It rises during collapse; watch `distinct-3` instead |
| 12 | Self-preference bias only matters if judge family == teacher family | **False** | Different family is necessary but not sufficient |
| 13 | A distilled student inherits the teacher's safety training | **False** | It inherits its *responses*; there is no safe size |
| 14 | Generation is the dominant cost | **True** | $1,075 vs $9 for training — 0.8 % |
| 15 | k = 3 samples per prompt costs 3× the training budget | **False** | 3× the generation, zero extra training cost |
| 16 | The 13-gram rule decontaminates against the training set | **False** | Against the **evaluation** set, before training |
| 17 | An example-level train/test split is enough for a generated corpus | **False** | Split by prompt; k samples of one prompt straddle the split |
| 18 | On-policy GKD works with an API teacher | **False** | The teacher must be resident during student training |
| 19 | The reverse-KL GKD variant wins on student-generated data | **True** | Mode-seeking, when the data comes from the student |
| 20 | A per-token API comparison is enough to decide on distillation | **False** | Add engineering time and the monthly refresh cadence |

---

## Coding / Whiteboard Tasks

### Task 1 — Write the response-distillation data filter

```python
import re, hashlib, json

REFUSAL  = re.compile(r"\b(I can'?t|I cannot|I'?m unable|as an AI)\b", re.I)
BOILER   = re.compile(r"(it'?s important to note|it is worth noting)", re.I)
TRUNC_RE = re.compile(r"(\.\.\.|…)\s*$")

def keep(row, prompt, max_words=900, min_words=5, n=3):
    r = row["output"].strip()
    if len(r.split()) < min_words:                 return False, "too_short"
    if len(r.split()) > max_words:                 return False, "too_long"
    if TRUNC_RE.search(r):                         return False, "truncated"
    if REFUSAL.search(r):                          return False, "refusal"
    if BOILER.search(r):                           return False, "boilerplate"
    if prompt[:40].lower() in r[:120].lower():     return False, "instruction_echoed"
    toks = r.split()
    if len({tuple(toks[i:i+n]) for i in range(len(toks)-n+1)}) < 0.6 * (len(toks)-n+1):
        return False, "repetitive"
    return True, "ok"

kept = [r for r in raw if keep(r, r["instruction"])[0]]
print(f"kill rate {1 - len(kept)/len(raw):.1%}")     # watch this across runs (Q44)
```

- **The decisions being graded:** cheapest tests first; each rule maps to a defect CH-09 §12.3 names in the "generate 20 and read them" list; the kill rate is printed because it is the run-over-run reproducibility signal.
- **Grading:** a candidate who runs a judge over all rows before the regexes is downgraded on cost; a candidate who checks refusal patterns but not instruction-echoing misses the defect that produces a student which recites the prompt.
- **Fail condition:** filtering *after* training, or filtering against the training set instead of the evaluation set (the 13-gram gate belongs here too).

### Task 2 — Decontaminate and split by prompt

```python
from collections import defaultdict

def shingles(text, n=13):
    t = text.split()
    return {tuple(t[i:i+n]) for i in range(max(0, len(t) - n + 1))}

eval_sh = set().union(*[shingles(e["output"] or e["instruction"]) for e in eval_rows])
train   = [r for r in raw if not (shingles(r["output"]) & eval_sh)]   # 13-gram gate
print(f"decontaminated: dropped {len(raw) - len(train)} rows")

by_prompt = defaultdict(list)
for r in train:
    by_prompt[r["instruction"]].append(r)

prompts = sorted(by_prompt)                      # split by PROMPT, not by row
val_p  = set(prompts[:len(prompts)//10])
val, tr = [], []
for p, rows in by_prompt.items():
    (val if p in val_p else tr).extend(rows)
assert not ({r["instruction"] for r in val} & {r["instruction"] for r in tr})
```

- **The decisions being graded:** the 13-gram gate drops a whole *row*, not a shingle; the split is on the prompt key so all k samples of a prompt land on the same side; the assertion makes the leak impossible to reintroduce silently.
- **Grading:** a candidate who splits rows fails; a candidate who decontaminates against the training set fails; a candidate who reports a held-out number from a row split is downgraded.
- **Fail condition:** `train_test_split(train)` on a k > 1 generated corpus — the validation rows are near-duplicates of training rows.

### Task 3 — Size the project before spending anything

```bash
# The two modes are a MUTUALLY EXCLUSIVE required group; there is no --mode flag.
python code/07_distillation.py --help

# 1. Size the job. --dry-run on --from-teacher does NOT load the weights.
python code/07_distillation.py --from-teacher --teacher Qwen/Qwen2.5-32B-Instruct \
    --prompts data/prompts.jsonl --n 5000 --out data/seqkd.jsonl --dry-run
# 2. Then generate 20 and READ them (the script's own instruction), then the full run.
python code/07_distillation.py --from-teacher --teacher Qwen/Qwen2.5-32B-Instruct \
    --prompts data/prompts.jsonl --n 5000 --out data/seqkd.jsonl
# 3. Filter, then train the student as ordinary SFT.
python code/01_sft_lora.py --data data/seqkd.jsonl --output out/student

# Teacher VRAM (GiB) for the generation budget, and the student-only training budget:
python code/common/memory.py --model 32B --method qlora --seq-len 512 --batch 1
python code/common/memory.py --table
```

- **The decisions being graded:** the teacher is a *separate offline process* for `--from-teacher`, so the teacher's VRAM is a generation budget and the training run holds only the student; the correct output flag on `01_sft_lora.py` is `--output`, not `--out` — `--out` belongs to `07_distillation.py`'s *generation* step, and the same short name in two scripts is how the confusion starts (`cheat-sheets/CH-08-Knowledge-Distillation.md:339` and `:603` still print `--out` for `01_sft_lora.py`); `--dry-run` on `--from-teacher` skips the weights, and on `--token-kd` it must still load tokenizers because the vocab check is the go/no-go.
- **Grading:** a candidate who sums teacher and student VRAM for sequence-level KD is downgraded — that is the two-model case, which only applies to token-level KD.
- **Fail condition:** writing `--mode from-teacher`. There is no `--mode` flag.

---

## Cheat Sheet of Numbers To Memorize

| Quantity | Value | Why it matters |
|---|---|---|
| Qwen2.5-0.5B / 1.5B MMLU · GSM8K | **47.5 · 49.6 / 60.9 · 68.5** | The 1.5B band is the modern student sweet spot |
| Llama-3.1-8B / Llama-3.3-70B MMLU | **69.4 / 86.0** | The teacher-grade reference points |
| R1-Distill-Qwen-1.5B MATH-500 | **83.9** vs GPT-4o **74.6** | The 671B → 1.5B proof; >400× gap |
| MMLU per doubling of params | **+4–6 points** | The sizing law's first term |
| MMLU per data quality tier | **+5–15 points** | One-time cost — the second term dominates |
| Capability floors | **0.3B chat / 1.5B assistant / 1.5B JSON / 3–8B narrow** | Each set by a named failure |
| Safe size without an output filter | **none** | The student inherits responses, not safety training |
| 50k × k=3 generation, GPT-4o | **$975** | Generation dominates: 30M in / 90M out |
| Judge stage 1 / stage 2 | **$20.03 / $100.13** | Cheap judge over 150k, strong judge over 45k |
| Realistic GPT-4o total | **≈$1,075** | vs $166,500 for expert humans |
| Student training, 1.5B / 8B | **$9 / $48** | 2.6 / 13.6 A100-hours; **0.8 %** of the total |
| Synthetic vs human label cost | **30×–150× cheaper, ~2,000× faster** | The reason the open-model ecosystem exists |
| Self-hosted 70B vs hosted 7B | **$0.97 vs $0.19 per 1M out** | Self-hosting a big teacher is 4.6× the price per token |
| Breakeven | **$5,075 vs $2,000/month → 2.5 months** | Below ~$200/month it never amortises |
| Full logits, V=150k fp32 | **600 KB/token → 25.5 TB** per 42.5M tokens | The storage wall |
| Top-100 / top-20, 8 B/entry | **800 B / 160 B per token → 34 GB / 6.8 GB** | The script's convention |
| Raw text | **~170 MB** per 42.5M tokens | Full logits are ~150,000× the corpus |
| Softmax buffers, B=8 · L=2048 · V=128,256 | **8.41 GB/tensor ≈ 25 GB** for 3 | The term everyone forgets |
| Teacher VRAM, 32B int8 / 70B fp16 | **30.9 / 131.5 GiB** | GiB, weights + 512-token KV + 1 GiB overhead |
| Dataset band | **20M–200M** response tokens | 50k × 850 × 3 = 127.5M |
| Human replay floor | **≥5–10 %** | The single mitigation that prevents collapse |
| Vocabularies | Llama-3 **128,256** · Qwen2.5 **151,936** · Phi-3 **32,064** · CodeGen **51,200** | The four numbers that decide whether logit KD is possible |
| Judge-disagreement alarm | **> ~15 points** | The win rate is noise; report both |
| Good AlpacaEval 2.0 LC for 1.5–3B | **high-50s** | Above that, audit for contamination first |

---

## Answers To The Self-Check Questions From CS-09

**1.** Because `V_Llama-3.1 = 128,256` and `V_Qwen2.5 = 151,936`, and beyond the sizes the token *boundaries* differ, so per-position alignment is meaningless even where ids coincide. Four presentations: **(a)** a shape mismatch — `RuntimeError: The size of tensor a (128256) must match the size of tensor b (151936) at non-singleton dimension 2`; **(b)** the silent variant — equal `V`, different id→token mapping, so the KL is finite, falling, and meaningless, caught only by `assert tok_T.get_vocab() == tok_S.get_vocab()`; **(c)** a projection matrix that appears to fix it and buys nothing, because it maps ids that do not denote the same tokens; **(d)** `ValueError: Expected input batch_size … to match target batch_size …` when the two models were tokenised separately so the sequence lengths differ. Fix: sequence-level KD, or a same-family student.

**2.** **No — the comparison is invalid before it is interesting.** A 3-class base checkpoint with a randomly initialized head scores ~33 %, and a teacher *measured* at 22 % is almost certainly a model that was never trained on the task, which is CS-08 §8.2's STOP condition 2 and the notebook's own failure (CS-09 §14.3). What is actually happening is that the "teacher" is an untrained head, so every improvement over it is vacuous. Eliminate the alternatives in order: **(a)** teacher never trained; **(b)** invalid comparison — different eval sets, preprocessing, or class order; **(c)** leakage — the student trained on the teacher's evaluation rows; **(d)** teacher in train mode, so dropout depressed its measured accuracy. Only after all four is "the student genuinely beats it on a narrow task" the answer.

**3.** `KL(p_T‖p_θ) = Σ p_T log p_T − Σ p_T log p_θ`, and the first term is constant in `θ`, so minimising KL is maximising `E_{y~p_T}[log p_θ(y)]` — maximum likelihood on teacher samples, which is exactly SFT on teacher generations. The **forward** KL. Mode-covering: it assigns infinite cost to `p_θ = 0` where `p_T > 0`, so the student is forced to spread mass over the teacher's entire support, including the low-probability tail it has no capacity to represent. The failure profile that follows: the student is **confidently wrong in the teacher's style**, and it hallucinates on the tail rather than abstaining — because abstention would mean putting near-zero mass where the teacher has some.

**4.** The factor is `T²`, multiplying the soft term only. Omitting it shrinks the soft gradient as `1/T²`, so raising `T` makes the soft term *weaker* — you conclude, wrongly, that soft targets do not help. Applying it to the cross-entropy term as well scales the hard term by the same factor, which is equivalent to multiplying the learning rate by `T²` on that term — the loss curve shifts and nothing improves, because the two terms' *relative* weight (which is what `α` is for) has been corrupted by a third control.

**5.** Three techniques on one 24 GB card: **(1) 4-bit QLoRA for the student** — mandatory, because it is what makes the student's weights, gradients and optimizer state fit at all; **(2) sequence-level KD** rather than token-level, so the teacher is a **separate offline process** and is not resident during training; **(3) an int4 or int8 teacher for the generation stage** (CH-09 §7.1: a 32B teacher is 60.7 GiB fp16, 30.9 int8, **15.9 int4** — the difference between needing two 48 GB cards and one 24 GB). If you insist on token-level KD, the teacher and student are resident together and CH-09 §7.2's two-model budget applies (~32 GiB for a 32B int8 teacher plus a 1.5B QLoRA student), so the top-k truncation becomes **mandatory** rather than an optimisation — it is what removes the 0.93 GiB full-vocab softmax buffer. Sequence-level KD is the one that makes the 24 GB card sufficient.

**6.** Generation: 50,000 × k = 3 = 150,000 calls at 200 input + 600 output tokens = **30M in / 90M out**. On GPT-4o at **$2.50/M in, $10/M out**: `30 × 2.50 = $75` and `90 × 10 = $900`, so **$975**. Two-stage judge: stage 1 over all 150,000 candidates at 850 in / 10 out = 127.5M in / 1.5M out, on GPT-4o-mini at **$0.15/M in, $0.60/M out** = `127.5 × 0.15 = $19.13` + `1.5 × 0.60 = $0.90` = **$20.03**; stage 2 over the 45,000 survivors (30 % survival) = 38.25M in / 0.45M out on GPT-4o = `38.25 × 2.50 = $95.63` + `0.45 × 10 = $4.50` = **$100.13**. Realistic total ≈ **$1,075**. Student training: 50,000 × 850 × 3 epochs = 127.5M tokens, `6 × 1.5e9 × 1.275e8 = 1.148e18` FLOPs; at 312 TFLOPS × 40 % MFU = 1.248e14 FLOP/s that is 9,200 s = **2.6 A100-hours ≈ $9**. **Generation dominates absolutely** — training is **0.8 %** of the $1,084 total. The lever that matters is the filter and the judge, not the optimizer.

**7.** DeepSeek distilled a **671B** MoE into a **1.5B** student by SFT on **800k sampled traces** and got **83.9 on MATH-500**, beating GPT-4o's **74.6**, Claude-3.5-Sonnet's **78.3**, and even a **direct-RL** run on Qwen2.5-32B (**47.0 AIME** where R1-Distill-32B got **72.6**). It is surprising because the capacity gap is **>400×**, far outside logit KD's 2–10× sweet spot. The implication: for **verifiable narrow tasks**, distillation beats RL for the *student*, because RL on a small model lacks the capacity to explore its way to the behaviour, while distillation hands it a distribution it can imitate. The reconciliation with the capacity-gap law is mechanical — sampling from `p_T` gives the student only high-probability trajectories, so it never has to represent the teacher's full distribution, which is where the representational mismatch in logit KD comes from.

**8.** Four conditions (CS-09 §4.8): **(a)** recursive self-training **replacing** real data each round with no verifier — collapses, because tails are never sampled; **(b)** recursive self-training **accumulating** data with enough real data — does not collapse (Alemohammad et al. 2024); **(c)** distilling from a **stronger** teacher with a verifier — does not collapse, it is a one-way transfer with no feedback loop; **(d)** distilling from a stronger teacher **without** a verifier — degrades but does not collapse, because the student inherits confident wrong answers at full weight. Plus the special case that *does* collapse fast: distilling **within your own family** (7B → 3B → 1.5B → 0.5B) with no human data and no verifier. **The only condition that actually matters is (a) — whether the loop is fed by something that is not itself** — which is operationalised as: never let a training mix reach zero human tokens (keep ≥5–10 % human-authored), and require a *verifier* rather than a judge for anything you recurse on.

**9.** Two explanations: **(1) self-preference bias** — an LLM judge scores its own family's text higher (Panickssery et al. 2024), so a Claude judge rates a Claude-family student higher than a GPT-4o judge did, independently of quality; **(2) a genuine judge-specific artifact** — position bias or length/verbosity bias present in one judge and not the other. The test that distinguishes them is **position-swapped pairwise judging**: re-run every comparison with the candidates in the opposite order. If the win rate moves materially on the swap, the effect is position bias in that judge, not quality. Corroborate with an **objective task metric** on the same prompts — flat accuracy alongside a 45-point win-rate swing means you measured style — and report both judges plus their **agreement**, since disagreement above ~15 points makes the number noise.

**10.** When it is a legal problem: when the vendor's terms restrict using outputs to develop models that **compete** with their services, and you are building a general-purpose assistant on their outputs; when you redistribute the teacher's raw outputs; when you use a consumer-tier endpoint for bulk generation. When it is not: **open-weights teachers** — Llama Community Licence, Gemma Terms, MIT releases — explicitly permit distillation and derivative models, which is the sanctioned path and the reason every serious open recipe (Zephyr, Orca, OpenHermes, WizardLM, R1-Distill) uses an open teacher or a released dataset; and a **narrow internal task model** is a different risk profile from a general competitor. **The one-line rule a lawyer can check:** *distil from open-weights teachers whenever you can; if you must use a closed teacher's outputs, distil a narrow task capability rather than a general assistant, use the zero-retention endpoint, do not redistribute the raw outputs, document provenance, and get the terms reviewed* — remembering that **the weights licence binds the artefact you ship and the ToS binds the dataset you generated**, two instruments a company can be compliant on one and not the other.

### Corrections to CS-09 and CH-09 discovered while writing this bank

> **Correction:** `case-studies/CS-09-Knowledge-Distillation-LLM-to-SLM.md:1914` ends its self-check with *"Answers to these ten questions are in `IQ-09-LLM-Distillation.md`"*. That file does not exist and never has — the answers are in **`IQ-09-Distillation-LLM-to-SLM.md`** (this file), matching the CS-09/CH-09 filenames. Separately, `cheat-sheets/CH-09-Distillation-LLM-to-SLM.md:745` says *"the answers file `IQ-09` is not yet written"*, which was true when the card was authored and is now stale. **Cite `IQ-09-Distillation-LLM-to-SLM.md`.**

> **Correction:** `cheat-sheets/CH-09-Distillation-LLM-to-SLM.md:701` titles its pre-flight list *"**The six checks** before you trust a distillation run"*, and the table beneath it is numbered **1 through 8**. The heading was copied from `CH-08 §12.3`, which does have exactly six rows; CH-09 added two LLM-specific checks — *7. an objective metric next to any win rate* and *8. prompt-level holdout* — without renaming the section. Nothing else in either card depends on the count, but a reader who counts rows to confirm they have covered "all six" will stop two checks early, and check 8 in particular (val prompts must be **disjoint**, not just val examples) is the one whose absence produces an inflated validation number. **Quote eight checks.**

> **Correction:** `case-studies/CS-09-Knowledge-Distillation-LLM-to-SLM.md:150` (the "Teacher capacity gap" glossary row) says the LLM-era counter-evidence is in *"§12"*. §12 is **Evaluation — How To Know It Worked**; the counter-evidence (R1-Distill's 671B → 1.5B at MATH-500 83.9) is in **§15.4**, with the sizing law in **§4.9**. `CH-09:798` already catches this and states the correct locations, so it is an unfixed cross-reference in CS-09 rather than an open question. **Read §15.4 and §4.9.**

> **Correction (the α convention in CH-09 is stale, and it was stale in the other direction once already):** three separate places in `cheat-sheets/CH-09-Distillation-LLM-to-SLM.md` assert that `code/07_distillation.py` weights the **HARD** term: §2's formula table (line 56, *"The repo script's loss `L = α·hard_CE + (1−α)·T²·KL` … **Inverted vs Hinton.** `code/07_distillation.py` line 74, 261"*), §4.1's knob table (line 160, *"`--alpha` (**weight on the HARD term**) … **0.5** … Script convention is **inverted** vs Hinton"*), and the Correction block at lines 756–764 (which cites "line 317" and a default of 0.5). All three describe an earlier revision of that file. The current script's `--help` reads *"Weight on the SOFT (KD) term; (1-alpha) goes to the hard-label CE. Hinton's convention, and the one CH-08 §4 uses. 0.7 means 70 % distillation / 30 % ground truth"*, its default is **0.7**, its loss line is `a.alpha * kd + (1 - a.alpha) * hard`, and its own docstring records the fix: *"(The first version of this script had it backwards, which meant `--alpha 0.7` — the exact command CH-08 §6 tells you to run — produced 70 % hard / 30 % KD.)"* So CH-09's §4.2 table (*"one table, three sources, three meanings"*) is the only part of the card that is still complete — the three meanings are real, but the script now sits with Hinton and CH-08, not against them. **Run `--help` and read the loss line; do not trust a quoted flag default or line number in any card, including this one.**
>
> **Correction (same file, the `--out` hint):** `cheat-sheets/CH-09-Distillation-LLM-to-SLM.md:747–754` opens a Correction with *"the script's own closing hint is wrong. `code/07_distillation.py` line 150 prints `python 01_sft_lora.py --data {out_path} --out out/student`"*. The script now prints **`--output`**, so the Correction's premise no longer holds — though its conclusion is still the right command. The two genuinely unfixed instances are in the other card: `cheat-sheets/CH-08-Knowledge-Distillation.md:339` and `:603`. CH-09's own runnable commands (`:220`, `:383`, `:678`) were already correct.

> **Correction:** `cheat-sheets/CH-09-Distillation-LLM-to-SLM.md:709` (check 5) and `:65` (§2's formula table) both give the capacity gap as *"≤ ~10×, or you ran the three-teacher sweep"*, inherited from CH-08 §7.4. That band is correct **for token-level logit KD only**, and the same card says so at §1 row 4 and again at §13 — *"for sequence-level KD on a verifiable narrow task the opposite holds (R1: 671B → 1.5B)"*. It is not a contradiction so much as a scope that is stated twice and applied once; when you quote the 2–10× band, **say which KD mode it governs**, because for the response-distillation default it does not apply at all.

---

## Cross-References

| Module | Relationship to IQ-09 |
|---|---|
| **CS-09** LLM → SLM Distillation | The source case study. Its §4.8 collapse conditions, §11.3 cost model, §12.4 self-preference bias and §15.4 R1 numbers are the ground truth here; its §19 questions are answered in full above |
| **CH-09** LLM → SLM Cheat Sheet | §4.2 the α table — the one part of its α material that is still accurate — plus §7.1 the teacher VRAM table, §7.3 the storage wall, §11 the exact error strings, §12.2 the eight checks. **Its §2 table, §4.1 knob row and `:756–764` Correction all still assert the script weights the hard term; it now weights the soft term. See the Corrections above.** |
| **CS-08** Knowledge Distillation Foundations | **Prerequisite.** Soft labels, `T²`, α, the capacity gap, TAKD, DistilBERT, the Hinton MNIST proof — this bank assumes all of it |
| **CS-14** RLHF, PPO, DPO, ORPO | Where dDPO (Zephyr's second stage) lives — the judge-as-preference stage, and the `β` KL anchor |
| **CS-13** Instruction Fine-Tuning & SFT | The training half of response distillation is exactly SFT: chat templates, `completion_only_loss`, packing, masking. Its §12 *Evaluation — How To Know It Worked* carries the MT-Bench / AlpacaEval / judge-bias discipline behind Q11, Q31 and Q35 |
| **CS-15 / CS-16 / CS-17** LLaMA-Factory, Unsloth, Axolotl | All three consume the `(instruction, response)` JSONL unchanged; use them instead of writing a training loop |
| **CS-10 / CS-11** Quantization I & II | The complementary compression axis, and why int4 is the teacher floor for a self-hosted generator |
| **CS-06** Hugging Face Masterclass | `from_pretrained` dtype behaviour, `device_map`, and the two-model placement problem behind Q40 |
| **CS-04** Fine-Tuning vs RAG vs Agents | The decision layer above this module — whether a small model is needed at all |
| **IQ-08** Knowledge Distillation Foundations | The prerequisite bank: the `T²` derivation, the worked example, the logit-KD loss, the encoder comparison tables |
| **IQ-13** Instruction Fine-Tuning | Where the SFT half of every recipe here is specified in full — masking, packing, chat templates, evaluation of generations |

---

*End of IQ-09. Companion artifacts: `CS-09-Knowledge-Distillation-LLM-to-SLM.md` (case study), `CH-09-Distillation-LLM-to-SLM.md` (cheat sheet). Ground truth: `code/07_distillation.py --help`, `code/01_sft_lora.py --help`, `code/common/memory.py --table`.*
