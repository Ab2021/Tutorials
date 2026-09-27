# IQ-08 — Interview Questions: Knowledge Distillation Foundations

| Field | Value |
|---|---|
| **Module** | Knowledge distillation I — soft targets, temperature, the `T²` factor, `α`, the capacity gap, TAKD, DistilBERT |
| **Pairs with** | CS-08 (case study), CH-08 (cheat sheet) |
| **Total questions** | 44 (12 L1 + 12 L2 + 9 L3 + 5 L4 + 6 L5) + 20 rapid-fire + 3 coding tasks + 10 CS-08 self-check answers |
| **Levels covered** | Screen (L1) / Intermediate (L2) / Advanced (L3) / System Design (L4) / Debug (L5) |
| **Source material** | Video 8 `Knowledge Distillation — Foundations`; `code/07_distillation.py`; `code/common/memory.py` |

**Ground truth used throughout:** the Hinton et al. (2015) MNIST experiment as reported in CS-08 §13.1 — **146** hard-label test errors, **74** soft-target errors, teacher **67**, and ~98.6 % of the 1,010 test 3s correct on a class absent from all training data. The instructor's own worked example (CS-08 §4.2.4): `q = [0.70, 0.20, 0.10]`, `p = [0.50, 0.30, 0.20]`, hard loss **0.6931** nats, `KL@T=1 = 0.08512`, `KL@T=2 = 0.02453` (a **3.5×** drop), `T²·KL = 0.09812`, gradient on the cat logit **−0.21484** with `T²` vs **−0.20000** at T=1 (within **7 %**) and **−0.05371** without (3.7× smaller). Every CLI flag below was verified by running `python code/07_distillation.py --help`.

---

## How To Use This File

- **L1 = phone screen / recruiter filter** — 30-second answers. If you cannot answer an L1 in one breath, you will not reach L2.
- **L2 = working engineer** — 2–3 minutes, expects implementation detail: real flags, real numbers, real failure modes.
- **L3 = senior / specialist** — 5 minutes, expects derivations and trade-offs, not vocabulary.
- **L4 = staff / system design** — 15-minute whiteboard. Structure every answer as: requirements → constraints → design → trade-offs → failure modes.
- **L5 = debugging & incident response** — "your student does X, what do you check and in what order." Answer with a *sequence*, not a list.

**The meta-rule:** an answer of "it depends" is a failure unless it immediately says what it depends on and then picks a default. Always pick the default.

**The three answers that get people hired in this module:** (1) *`T²` exists because the softmax gradient scales as `1/T²`, so without it raising `T` silently shrinks the soft term*; (2) *soft targets are not a regulariser — the Hinton MNIST student classified 98.6 % of a digit class it never saw*; (3) *the capacity gap is non-monotone: a 70B teacher frequently loses to a 7B teacher for a 1B student.* Everything else in this bank is a consequence of those three.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. Say what distillation is in one sentence, with the mechanism.**

- **Answer:** Training a small *student* to reproduce a large *teacher's* output distribution rather than the hard labels — `L = α·T²·KL(q_teacher‖p_student) + (1−α)·CE(y, p_student)` — so the student receives not just "this is a cat" but "this is a cat, a bit dog, hardly rabbit," which is the teacher's learned similarity structure.
- **Why asked:** The one-sentence version distinguishes "compressing a model" from the actual objective. Compression is the *outcome*; distribution matching is the *method*.
- **Trap:** Saying "making a small model from a big one." That describes pruning and quantization too, and neither transfers the teacher's distribution.

**Q2. What is a soft target, and what is a hard target?**

- **Answer:** A hard target is one-hot — `[1, 0, 0]`. A soft target is the teacher's full probability vector, `[0.70, 0.20, 0.10]`. The soft target's *relative magnitudes off the argmax* are the new information, and Hinton called that "dark knowledge."
- **Why asked:** It is the single most important concept in the module, and the interviewer wants to hear "relative magnitudes," not "probabilities."
- **Trap:** Saying the soft target is "a smoother label." That is what label smoothing is, and a smoothing baseline provably cannot learn a class it never saw (CS-08 §13.1).

**Q3. What does temperature do, mechanically?**

- **Answer:** It rescales the logits before the softmax: `p_i(T) = exp(z_i/T) / Σ_j exp(z_j/T)`. Raising `T` flattens the distribution, exposing the relative ordering of the low-probability classes that a confident teacher has compressed toward 0. At `T = 1` you get the teacher's native output; as `T → ∞` the soft target approaches the uniform distribution.
- **Why asked:** It is the knob that makes dark knowledge visible, and the candidate should be able to state the limit behaviour.
- **Trap:** Believing higher `T` is always better. Past ~20 the soft target is nearly uniform and the KL signal is mostly noise the student cannot fit (CH-08 §4). `T = 4` with a range of 2–20 is the default.

**Q4. Why does `T²` appear in the loss?**

- **Answer:** Because `∂/∂z_i [KL(q^T‖p^T)] ∝ 1/T²` — raising `T` flattens the softmax, and the gradient of a flattened softmax shrinks quadratically in `T`. Multiplying the soft term by `T²` restores the gradient magnitude that the temperature scaling removed, so changing `T` changes *what the target looks like* rather than *how much it counts*.
- **Why asked:** It is the most commonly omitted term in the field, and the derivation is one line: for high `T`, `exp(z/T) ≈ 1 + z/T`, so the softmax becomes approximately linear in `z/T`.
- **Trap:** Describing `T²` as a hyperparameter you tune. It is a correction, not a knob — CH-08 §4 says "turn it OFF only if you know why."

**Q5. Where does the `T²` go — the soft term only, or both?**

- **Answer:** The soft term only. The CE term is computed against a one-hot target at `T = 1`; applying `T²` to it would rescale your learning rate for no reason. Concretely, in the corrected loss `alpha * (T*T) * soft + (1.0 - alpha) * hard` the `T*T` multiplies only `soft` — CH-08 §5.1 line 180.
- **Why asked:** Half-remembered implementations apply it to both terms, and the symptom (loss curve shifted, quality unchanged) does not point at the cause.
- **Trap:** Multiplying the whole loss by `T²` "for safety."

**Q6. What are the three families of distillation?**

- **Answer:** **Response-based** (match the output logits — Hinton 2015), **feature-based** (match intermediate hidden states — FitNets), **relation-based** (match relations *between* examples, e.g. `Q·Kᵀ` and `V` relations — RKD, MiniLM). A fourth axis is *when* it is done: offline (a frozen teacher), online (teacher and student trained together, DML), and self-distillation (born-again networks).
- **Why asked:** It is the taxonomy, and the interviewer is probing whether you know it is four axes — family, supervision site, teacher availability, and student size — not three mutually exclusive boxes.
- **Trap:** Listing only "logit and feature." Missing relation-based explains why you cannot say what MiniLM transfers.

**Q7. DistilBERT in one line of numbers.**

- **Answer:** 6 layers (from 12), 66.9 M params (from 109.5 M) — **40 % smaller**, **1.63× faster**, GLUE **77.0** vs **79.5**. Three loss terms: response (soft targets), cosine embedding loss on hidden states, and the MLM loss. It is the reference encoder distillation.
- **Why asked:** It is the canonical artifact, and the *three* loss terms are the detail that separates readers from skimmers.
- **Trap:** Quoting a "3 % quality loss" as if it were the rule. CS-08 §13.3's table shows MobileBERT at 77.7 with only 25.3 M params — *better* than DistilBERT while being 2.6× smaller.

**Q8. Sequence-level KD vs token-level KD — one sentence each.**

- **Answer:** Token-level KD matches per-token distributions at temperature and requires a shared vocabulary; sequence-level KD (Kim & Rush 2016) trains the student on the teacher's *generated sequences* as ordinary hard labels, so it needs no logits and no shared vocabulary. Sequence-level beats training on the original human references because the teacher's output is drawn from a distribution the student can actually reach on the input distribution it will see.
- **Why asked:** It is the practical fork: token-level has a storage wall, sequence-level is what everyone actually ships.
- **Trap:** Calling sequence-level KD "just fine-tuning on synthetic data." The mechanism is a KL argument (CS-09 §4.3), and the difference shows up in the failure profile.

**Q9. What is the capacity gap, and why is it counter-intuitive?**

- **Answer:** The teacher-to-student parameter ratio, and its effect on distillation is **non-monotone**. A gap of **2–10×** is the sweet spot; beyond **50×** a mid-size teacher often *beats* the huge one; beyond **200×** it is usually a waste. The mechanism is that a 70B teacher's distribution lives on a manifold so far from anything a 1B student can represent that the KL signal is mostly noise the student cannot fit.
- **Why asked:** It is the single most counter-intuitive result in the field, and "bigger teacher is better" is the answer that fails the interview.
- **Trap:** Recommending the largest available teacher. CH-08 §7.4 says the fix is *test a mid-size teacher too.*

**Q10. What is TAKD?**

- **Answer:** Teacher-Assistant Knowledge Distillation — insert an intermediate "assistant" model between a huge teacher and a small student to decompose one unbridgeable gap into two bridgeable ones. The assistant is typically sized near the geometric mean, trained on the teacher's soft targets, and then teaches the student.
- **Why asked:** It is the standard fix for a >50× gap, and it costs the extra assistant-training run — which is why CH-08 §8's symptom table offers cheaper fixes first.
- **Trap:** Reaching for TAKD before considering a smaller teacher or a bigger student. Both are cheaper and usually sufficient.

**Q11. Distillation vs fine-tuning vs quantization — one line each.**

- **Answer:** Fine-tuning changes *what a model does*; distillation changes *how much model does it* (fewer FLOPs, same architecture family); quantization changes *how many bytes* it takes (same FLOPs, same architecture). Distillation and quantization **compose** — distil then quantize — and the two are orthogonal axes.
- **Why asked:** It is the decision-layer vocabulary, and confusing distillation with quantization is one of the module's named misconceptions.
- **Trap:** Saying distillation makes inference faster *and* smaller in memory. It reduces FLOPs; the memory win comes from the parameter count and only becomes a bytes win after quantization.

**Q12. What is the single wrong reason people give for distillation working?**

- **Answer:** "It's a regulariser / it's like label smoothing." That hypothesis predicts the student should get *no* information about a class it never saw — and Hinton's MNIST student, trained with all 3s deleted, classified ~98.6 % of the 1,010 test 3s correctly (≈14 errors). A uniform-smoothing baseline can never acquire that class. Soft targets carry the teacher's *structure*, not just noise.
- **Why asked:** It is the module's central experiment, and the interviewer wants the prediction the hypothesis makes, not just the outcome.
- **Trap:** Knowing the 146 → 74 improvement and not knowing the 3s result. The 74 is not the interesting number.

---

## Level 2 — Applied & Implementation

**Q13. Write the KD loss, and say what breaks in each line.**

- **Answer:** `t_soft = F.softmax(t_logits / T, dim=-1)` — the teacher must be *softmaxed*, not log-softmaxed, because `F.kl_div` takes `input` as log-probabilities and `target` as probabilities. `s_log = F.log_softmax(s_logits / T, dim=-1)` — the student must be *log*-softmaxed. Then the reduction, which depends on the shape you built: on a **full-vocab** `(B·L, V)` tensor use `reduction="batchmean"`; on the **top-k truncated** tensor the repo script instead does `F.kl_div(..., reduction="sum") / n_pos`, because `batchmean` there would divide by `top_k` rather than by the position count — the script says so in a comment on the line above. Then `* (T * T)`, outside the call, on the soft term only. And `loss = alpha * kd + (1 - alpha) * hard`, with `alpha` in **Hinton's convention** — weight on the soft term, default **0.7** (`--help`: *"Weight on the SOFT (KD) term… 0.7 means 70% distillation / 30% ground truth"*).
- **Why asked:** All four are silent. `mean` on a full-vocab tensor divides by `L·V` (≈3×10⁸ at L=2048, V=150k); `batchmean` on a top-k tensor divides by `k`. Either way the soft term is annihilated and the run looks like plain SFT.
- **Trap:** Passing probabilities as `input`. `F.kl_div` then returns a mathematically meaningless value that **can be negative**, and nothing raises (CH-08 §8). Second trap: assuming one reduction string is correct everywhere — the right answer depends on whether the tensor is full-vocab or top-k.

**Q14. What is the teacher's entropy telling you before you commit to a run?**

- **Answer:** Whether the soft targets carry any information. If the teacher's mean max-probability is ~0.99 and the entropy of `q` is near zero on your transfer set, the "soft" targets are effectively one-hot and you have built an expensive label smoother. Measure both on a sample of the transfer set first; if they are degenerate, either raise `T`, pick a less overconfident teacher, or accept that you are doing sequence-level KD.
- **Why asked:** It is the single cheapest go/no-go check and it is the one almost nobody runs. CS-08 §13.1 states the transferable rule directly: *"if your soft targets are close to uniform after temperature scaling, you have built an expensive label smoother."*
- **Trap:** Tuning `T` on the student's final accuracy. That costs a full training run per `T` and confounds `T` with the `T²` correction.

**Q15. `code/07_distillation.py` has two modes. How do you invoke each?**

- **Answer:** They are a **mutually exclusive required group** — there is no `--mode` flag, and passing neither is an argparse error. `--from-teacher` takes `--prompts` (a `.jsonl` of prompts); `--token-kd` takes `--text` (a raw `.txt` corpus). Passing the wrong one is the common mistake and argparse catches it. `--dry-run` is available on both, and on `--token-kd` it still loads the tokenizers because the vocabulary-alignment check is the go/no-go — but not the weights.
- **Why asked:** It tests whether the candidate reads a flag surface rather than recalling one.
- **Trap:** Assuming `--dry-run` never loads anything. On `--token-kd` it must load tokenizers, and a mismatch is a hard exit.

**Q16. How would you size a token-level KD job?**

- **Answer:** Two memory terms and one storage term. **Resident:** teacher + student together, so the teacher's weights are charged at inference precision plus its attention/softmax buffers. At `B=8, L=2048, V=128,256` the fp32 softmax buffers alone are `8 × 2048 × 128256 × 4 B = 8.41 GB` per tensor, ≈25 GB for three (CH-09 §7.2). **Storage:** the cached logits — `code/07_distillation.py` charges `k × (4 bytes logit + 4 bytes index)` = **8 B/entry** in its `--dry-run` plan, and the script's `--top-k` default is **100**. **Compute:** teacher forward is `2ND`, student training `6ND`.
- **Why asked:** Token-level KD is the case where candidates underestimate memory by 2×, because they forget the softmax buffers and the cache.
- **Trap:** Budgeting for the student alone. The teacher being resident is the whole reason sequence-level KD exists.

**Q17. Where does `--size-hint` and `--quant-bits` fit?**

- **Answer:** They describe the teacher for the *plan* on the `--token-kd` path so a laptop can size a 32B job: `--size-hint` (default `32B`) picks the preset and `--quant-bits` (default `1.0`) the teacher's precision. They do not change what is loaded. `--teacher-temp` (default `0.8`) is the teacher's *sampling* temperature on the `--from-teacher` path — distinct from `-T/--temperature` (default `2.0`), which is the KD softening temperature on the token path.
- **Why asked:** Two flags both named "temperature" with different jobs is a real source of confusion, and the interview answer is to say which path each belongs to.
- **Trap:** Setting `-T` on the generation path and expecting a softer teacher. `-T` is not read there.

**Q18. What is the supervised fraction in a KD run, and how do you check it?**

- **Answer:** The share of target tokens that carry a real (non-`-100`) label. In token-level KD the hard term must be computed on the *same* positions as the soft term, so the mask that excludes prompt/pad tokens applies to both. Print `(labels != -100).float().mean()` on one batch and compare with the fraction of *completion* tokens — a number near 1.0 means you are training on the prompt, and near 0.0 means the hard term is dead.
- **Why asked:** A dead hard term looks like "KD is not helping" and is actually a masking bug (CH-08 §8, first row).
- **Trap:** Blaming the loss function. Check the mask first.

**Q19. Why is the student's learning rate not special?**

- **Answer:** KD changes the *target*, not the optimization landscape, so the student typically trains at the same LR as ordinary training of that architecture (CH-08 §4). What *is* special is that KD overfits too, so epochs stay at **1–3**, and the teacher must be in `eval()` with `requires_grad_(False)` — which is three separate ways to get a silently wrong run if you skip it.
- **Why asked:** Candidates reach for a KD-specific LR schedule that does not exist, and skip the freezing, which does matter.
- **Trap:** Not calling `teacher.eval()`. Dropout in the teacher at train time makes the soft targets noisy in a way that never appears in the reported loss.

**Q20. How many epochs of KD before you are overfitting?**

- **Answer:** 1–3, and the diagnostic is not the training loss. Watch agreement with the teacher on a *held-out* slice, plus the student's accuracy on the original hard-label validation set — those can diverge in opposite directions, and it is the second one that decides whether you ship. CH-08 §8's row 12 is worth memorizing: *"distillation loss decreases but eval does not improve → you are fitting the teacher's noise."*
- **Why asked:** KD's train loss is bounded below by the teacher's own entropy, so "loss still going down" is a weaker signal than usual.
- **Trap:** Early-stopping on the KD loss.

**Q21. Where does the feature loss go, and how do you weight it?**

- **Answer:** On intermediate hidden states, and the weight is *per-layer* because hidden-state magnitudes grow with depth — a single weight that works at layer 2 is either negligible or dominant at layer 11. DistilBERT used a cosine embedding loss on the last hidden state; FitNets used a regressed projection from the student's hidden size to the teacher's. `hint` layers need a learned linear projection when the widths differ.
- **Why asked:** It tests whether the candidate has seen a feature-KD loss fail, which is almost always a scale problem.
- **Trap:** A single `feature_weight` across all layers, or forgetting the projection when the student is narrower.

**Q22. The teacher and student have different vocabularies. What are your options?**

- **Answer:** Four, in order of preference: (1) sequence-level KD — no alignment needed, always works, and is the default; (2) pick a student in the same family as the teacher, which is the *accidental* reason the notebook's Phi-2 → Phi-1.5 demo runs at all (both use the CodeGen tokenizer, V = 51,200 — CS-09 §4.6); (3) align on the token *set* overlap with a projection matrix and accept the loss of the tail; (4) abandon token-level KD. Option (2) is the one people miss, and it is why "same family" is a design constraint, not a preference.
- **Why asked:** It is the constraint that kills most attempted token-level KD runs, and the failure is a shape mismatch, not a quality problem.
- **Trap:** Trying to force it with a projection. The vocabularies do not merely differ in size — the token *boundaries* differ, so per-position alignment is meaningless.

**Q23. What is the `--top-k` flag trading away?**

- **Answer:** Fidelity against storage, and the trade is extremely favourable. CS-08 §4.2.4 quantifies where the dark knowledge lives: a 4th class the teacher gives 0.001 and the student gives 0.01 contributes `0.001·ln(0.1) = −0.0023` nats. The information is in the top two or three classes. Top-20 captures >99.9 % of the KL mass at a 6,500× storage reduction versus full fp32.
- **Why asked:** It is the number that makes token-level KD feasible at all, and the candidate should be able to say *why* 20 is enough rather than quoting 20.
- **Trap:** Raising `--top-k` to 1000 "for safety." Storage grows linearly, fidelity does not.

**Q24. Give the `α` convention trap in one answer.**

- **Answer:** There are two separate questions here and most candidates conflate them.
  **First, what does the *paper* say?** Nothing about `α` — the symbol does not appear in
  arXiv:1503.02531. Hinton calls it "a weighted average of two different objective functions",
  names the **soft**-target cross-entropy *first* and the **hard**-label cross-entropy
  *second*, and reports "a considerably lower weight on the second objective function" (0.5
  relative on the hard targets in the ASR runs, §4.1). **The soft term carries the larger
  weight.** So the popular interview line "Hinton's `α` weights the hard term" is a
  misattribution, and a candidate who says it has repeated folklore rather than read the paper.
  **Second, where does the confusion come from?** From reimplementers assigning the invented
  symbol `α` to opposite terms. This notebook (cells 20, 39), CH-08 §4 and
  `code/07_distillation.py` all put `α` on the **soft** term — `alpha * kd + (1 - alpha) * hard`,
  default 0.7, matching Hinton's direction. A large family of textbook and blog
  reimplementations puts `α` on the **hard** term; that reading is a faithful transcription of
  Hinton's *sentence* ("lower weight on the second objective") and an unfaithful reading of his
  *physics*. The two give **0.2766 vs 0.1576** on CS-08 §4.2.4's example — 76 % apart for
  identical teacher, student, `T` and data, with nothing changed but the name.
  **The transferable answer:** *the loss is unambiguous, the letter is not; read the mixture
  expression at every call site, and never cite a paper for a symbol it does not contain.*
  (The repo's own script is the cautionary tale — its first version really did weight the hard
  term, so `--alpha 0.7`, the command CH-08 §6 told you to run, once meant 70 % hard / 30 % KD
  and now means the opposite. That run trains either way, its loss falls either way, and in the
  wrong convention you have paid for a teacher and run plain SFT.)
- **Why asked:** It is the highest-value trap in the module, and the *reason* it is a trap is that a copy-pasteable command line plus a plausible flag name is not a specification.
- **Trap:** Reading the flag name instead of the arithmetic on the loss line. `code/07_distillation.py` prints `hard CE`, `KD (x T^2)` and `mixture` on every run precisely so the convention is observable from one forward pass — and it warns when the KD term is under 10 % of the CE term despite carrying most of the weight.

---

## Level 3 — Advanced, Internals & Theory

**Q25. Derive `∂L_soft/∂z_k` for `L_soft = KL(q‖p)`, and explain `T²`.**

- **Answer:** With `p = softmax(z/T)` and `q` the teacher's distribution at the same `T`, the KL gradient with respect to the student's *scaled* logits reduces to `∂L/∂z_k = (1/T)(p_k − q_k)`. Chain through the `1/T` from `z/T` and you get a `1/T²` overall — so the gradient **vanishes quadratically** as you raise `T`. Multiplying the loss by `T²` restores the `(p_k − q_k)` gradient you had at `T = 1`, decoupling "how soft the target is" from "how hard it pushes." Numerically, from CS-08 §4.2.4: at `T = 2` the cat-logit gradient is **−0.21484** with `T²` (within 7 % of the `T = 1` value of −0.20) and **−0.05371** without — 3.7× smaller. As `T → ∞` the soft target tends to uniform and the KL tends to zero, so the objective degenerates into the hard term plus a constant.
- **Why asked:** It is the derivation the module exists to teach, and the interviewer wants the intermediate `1/T` step, not the conclusion.
- **Trap:** Saying "T² compensates for the softer distribution." State the mechanism: the softmax's Jacobian scales as `1/T`, and it appears twice.

**Q26. In the worked example, raw KL fell 0.08512 → 0.02453 when T went 1 → 2. Why is "the loss went down, so T=2 is better" wrong?**

- **Answer:** Because it is a *scaling artefact*, not an improvement. The softmax at `T = 2` is flatter, so the two distributions are closer in KL by construction — a uniform `q` gives exactly 0 at any `T`. The evidence that nothing improved is the `T²`-corrected value: `4 × 0.02453 = 0.09812`, which is back in the same ballpark as `0.08512` at `T = 1`. That is the point — `T²` makes the number *comparable across temperatures* so that changing `T` changes the target's shape, not its weight.
- **Why asked:** It is the quantitative form of the `T²` trap, and a candidate who reports raw KL across `T` will mis-tune the only knob that matters.
- **Trap:** Comparing raw KL values at different temperatures at all.

**Q27. Where is the dark knowledge, quantified?**

- **Answer:** In the top two or three classes. CS-08 §4.2.4 extends the example with a real 4th class the teacher gives 0.001 and the student 0.01: its KL contribution is `0.001·ln(0.001/0.01) = 0.001·(−2.303) = −0.0023` nats — sign included, a class the student over-weights *reduces* the KL. The classes that move the loss are the ones where both models put real mass. This is the arithmetic justification for top-k caching and for `k ≈ 20–100`.
- **Why asked:** It converts "top-k is cheaper" into "top-k is where the signal is," which is the difference between a cost decision and a fidelity decision.
- **Trap:** Believing the long tail of a 150k vocabulary carries the dark knowledge. It carries numerical noise at fp16.

**Q28. State the capacity gap, give the two experimental results behind it, and name two fixes cheaper than TAKD.**

- **Answer:** The gap is `teacher_params / student_params` and its effect is **non-monotone**: 2–10× is the sweet spot, >50× the teacher is often worse than a mid-size one, >200× is usually a waste. Two results: **Cho & Hariharan 2019 (CIFAR-100)** — student accuracy is non-monotone in *teacher training length*, peaking early and declining, so the best teacher for KD may be one you would not ship; and the >50× observation that a mid-size teacher wins. Three mechanisms (CS-08 §4.6.2, and they need different fixes): the teacher's *capacity* to represent a function the student cannot, the teacher's *confidence calibration* being wrong for a small model, and the *distribution mismatch* after the student's own errors compound. Cheaper fixes than TAKD: use a smaller/mid-size teacher (free), or a bigger student (you were going to deploy it anyway), or self-distillation at the student's own capacity.
- **Why asked:** It is the module's headline counter-intuition, and the three mechanisms are separable — which is exactly the structure the interviewer is testing.
- **Trap:** Answering with the observation and not the mechanisms. "70B is too big" is not an explanation.

**Q29. Name the four axes of the taxonomy, and place FitNets, attention transfer and RKD.**

- **Answer:** (1) **What is supervised** — response (output logits), feature (intermediate hidden states), relation (relations between representations or examples). (2) **Where** — output, hidden, attention, embedding. (3) **Teacher availability** — offline, online/DML, self-distillation. (4) **Student size** — compression vs same-size (born-again). **FitNets** supervises an intermediate *hidden state* through a learned regressor from student width to teacher width — alignment problem: widths differ, so you must learn a projection, and *which* layer to hint is a hyperparameter. **Attention transfer** supervises the *spatial* attention map `Σ_c |A_c|²` — alignment problem: it sums over channels, so it crosses *head counts* but requires matching spatial dimensions. **RKD** supervises *pairwise* and *triple-wise* distances between examples in the output/hidden space — alignment problem: none for head count or width, only a distance-metric choice, which is why it composes freely.
- **Why asked:** It is the "when to use" question in taxonomy form: each variant exists to dodge a specific alignment constraint.
- **Trap:** Saying attention transfer matches attention matrices. It matches their *squared sums over channels*, which is exactly how it dodges the head-count constraint.

**Q30. DistilBERT has three loss terms. What happens if you drop the third?**

- **Answer:** The three are the response loss (soft targets on the output), the cosine embedding loss on the last hidden state, and **the MLM loss**. Dropping the MLM term means the student is never trained on the raw masked-language objective, so it loses general-purpose language ability outside the distillation task — the retention gap widens, and the student becomes a narrow task model even though the response loss looks fine. DistilBERT's initialization trick matters here too: the student is initialized from the teacher by **taking every other layer** (CS-08 §4.7.3), which is worth a meaningful accuracy margin over random init for free.
- **Why asked:** It is the "pros/cons and exceptions" probe: a three-term loss where each term exists for a different failure.
- **Trap:** Treating the MLM term as optional regularization. It is what keeps the student a language model.

**Q31. Distillation hurts rare classes most. Why mechanically, and what do you change?**

- **Answer:** Both loss terms under-supervise the tail. A class with 20 examples gets ~20 rows of hard-loss signal, and the teacher's own probability on that class is low and noisy — so the student receives *less signal on the tail from both terms*, while the loss weights them equally. The fixes follow: gate acceptance on **worst-class recall** and macro-F1 (not accuracy); use `nn.CrossEntropyLoss(weight=w)` with `w_c ∝ 1/n_c` — and note the *soft* term has no natural per-class weighting, so the asymmetry is intentional; oversample the tail in the *transfer set*, because the teacher's soft target on a rare class is the dark knowledge you most want and it is the scarcest; and if the tail matters, distil a teacher trained with class weights even if it costs a point of accuracy, because an accuracy-optimised teacher is usually worst exactly where you need it.
- **Why asked:** It is the module's "finding to state in an interview" (CS-08 §12.5) and the diagnostic most often omitted from reports — `recall_score(y, pred, average=None)`, where the `average=None` is the whole point.
- **Trap:** Reporting macro-F1 without per-class recall. Macro-F1 still lets a large tail class mask a small one.

**Q32. Does KD beat training from scratch? Give the honest answer.**

- **Answer:** Distillation reliably beats training from scratch *at the same data*, and it does **not** automatically beat training from scratch with *better data or a longer schedule*. The evidence both ways: Hinton 2015 improves MNIST 146 → 74; Beyer et al. 2022 shows KD beating from-scratch at every student size on ImageNet, but only with strong *consistent* augmentation of the teacher's targets; born-again (Furlanello 2018) improves same-capacity students generation over generation; but **Gunasekar et al. 2023** ("Textbooks Are All You Need") had a 1.3B model trained from scratch on ~7B tokens of curated data beat much larger models — data quality can dominate distillation. And Müller 2019 shows that with a near-uniform teacher, KD ≡ label smoothing, so the "KD gain" can be a smoothing gain. The practical rule: run three arms — (A) hard labels, (B) KD, (E) label smoothing — and choose KD only if it wins by more than seed noise. On encoder classification at 100k examples KD typically wins by **1–4 points** over (A), and can lose when the "teacher" is barely larger than the student.
- **Why asked:** It is the question that separates someone who has run the baseline from someone who has read the abstract. The run-E smoothing baseline is the tell.
- **Trap:** Quoting a paper's gain without asking whether the paper ran the smoothing control.

---

## Level 4 — System Design & Scenario

> These are 15-minute whiteboard questions. Structure every answer as: requirements → constraints → design → trade-offs → failure modes.

**Q33. You must distil `bert-large` into `bert-base` on a single 24 GB card. Design it.**

- **Answer:** Requirements: encoder classification, 100k-ish examples, one 24 GB card. Constraints: teacher + student both resident for token-level KD, plus the cache. Design: teacher forward at `B=32, L=256` costs `2ND ≈ 5.0 GB` with the student; total ≈ **5.0 GB** — comfortable. The three mandatory moves: teacher in `eval()` with `requires_grad_(False)`; cache the top-k logits once (§11.3 — for a 3-class task the full cache is **0.6 MB** for 100k examples, so you can afford full distributions there); and keep the student's MLM term alive. Trade-offs: full-vocab cache at BERT's 30,522-token vocab is **6.1 GB** for 100k examples — still affordable, which is why encoder KD is the one place full-vocab caching is normal and LLM KD never is. Failure modes: the notebook's own BERT result is a *silent failure* (CS-08 §14.3) because the teacher was a base checkpoint with a randomly initialized head — check `STOP condition 2` before writing any code.
- **Why asked:** It forces the candidate to notice that the storage wall is a function of vocabulary size, and that encoder vocabularies are 5× smaller than LLM ones.
- **Trap:** Applying the LLM storage intuition. 6.1 GB vs 25.7 GB is the difference between caching full distributions and needing top-k.

**Q34. Design the "which teacher?" experiment.**

- **Answer:** Requirements: pick a teacher, and defend the pick. Design: a size sweep — train or obtain teachers at roughly 3×, 7×, 30× and 70× the student's size, distil the *same* student from each with identical hyperparameters, and plot student quality against teacher size. The expected shape is **non-monotone with a peak in the 2–10× band**, and the sweep is the only way to find *your* peak because it depends on the student's capacity and the task. Constraints: budget is `n_teachers × (teacher fine-tune + distillation run)`, so use the cheapest fold — few thousand examples, 2 epochs. Trade-offs: the biggest teacher may still be the best *if* the task is knowledge-heavy (reasoning traces, code) rather than representation-heavy; and teacher *training length* is a second axis (Cho & Hariharan) — the best KD teacher may be under-trained relative to your best deployable model. Failure modes: evaluating on the test set you will report on; not running the from-scratch and smoothing baselines, so you cannot tell a KD gain from a capacity gain.
- **Why asked:** It is the module's core design experiment, and CS-09 §5.4 calls it "the one diagnostic almost nobody runs."
- **Trap:** Assuming the teacher you already have is the right one, because it is the one you have.

**Q35. Design a KD pipeline for a 3-class internal taxonomy where the tail class matters.**

- **Answer:** Requirements: 3 classes, imbalanced, tail class is the one that pages someone. Constraints: teacher must be *good on the tail*. Design: (1) train the teacher with class weights even at the cost of ~1 accuracy point — this is the counter-intuitive step and it is the whole answer; (2) build a transfer set that **oversamples the tail**, because the teacher's soft target there is the scarcest and most valuable signal; (3) hard loss with `weight=w, w_c ∝ 1/n_c`; (4) accept on **worst-class recall** and macro-F1, never accuracy; (5) re-fit decision thresholds on the *student's* validation distribution, because a confidence threshold fitted to the teacher is the wrong threshold. Trade-offs: class weighting raises head-class error to buy tail recall, so the acceptance gate must be stated per class before the run. Failure modes: aggregate accuracy hiding a 5-point tail loss; the student matching the teacher on accuracy but *not* on its predictive distribution (Stanton 2021), which breaks any `max_prob > 0.9` routing rule downstream.
- **Why asked:** It is the design question where the correct answer is "change the teacher, not the student" — and it is measured by whether the candidate reaches for worst-class recall unprompted.
- **Trap:** Treating it as a class-imbalance problem solvable entirely on the student side. The soft term has no natural per-class weighting, so the tail has to be fixed upstream.

**Q36. Your team wants to distil a 7B into a 1B for a task where a from-scratch 1B is already 2 points behind the 7B. Design the decision.**

- **Answer:** Requirements: decide whether the run is worth it. Constraint: CS-08 §8.2 STOP condition 1 — *the teacher is less than ~3 points better than a from-scratch small model*. Two points is under the noise floor of most evaluations, so the expected gain does not exceed the measurement error. Design: before spending anything, (a) compute the evaluation's own noise: run the from-scratch 1B with 3 seeds and measure the spread — if it is ±1.5 points, a 2-point gap is not a gap; (b) check STOP conditions 2, 3, 7 and 8 (teacher actually fine-tuned on the task; you can evaluate on held-out data; the eval set is not contaminated with teacher training data; the licence permits derivative models); (c) if the gap survives that, run the three arms (hard, KD, smoothing) at small scale first. Trade-offs: the recurring cost is not the training run — it is a permanent second model in the registry, an extra generate step per refresh (~3× a monthly fine-tune's compute), and two artifacts to monitor. Failure modes: shipping on a 2-point gain that later turns out to be seed noise; discovering the licence problem after the GPU bill.
- **Why asked:** It is the "when NOT to do it" question, and the module has an explicit eight-item list to check against.
- **Trap:** Answering "yes, distil, bigger teacher always helps." Condition 1 exists because the gain must beat the noise floor.

**Q37. Design the monitoring for a distilled model in production.**

- **Answer:** Requirements: two models, one of them now replaced, no labels on live traffic. Design: six monitors — input distribution drift (PSI/KL on token statistics, alert PSI > 0.2); confidence distribution drift (mean max-prob, alert shift > 0.05); **agreement with a shadow teacher on a 1 % traffic sample** (alert on a drop > 3 points) — this is the single best distillation-specific monitor because it needs no labels and gives a leading indicator; per-class recall on a labelled canary slice (alert on any class −5 points); latency p99 at production batch size; and shadow-mode win-rate during rollout (gate at ≥90 % of the teacher's decisions). Trade-offs: the shadow teacher costs serving capacity, so keep it warm for 30 days and then retire the fleet while keeping the weights in the registry. Failure modes: rolling back the model but *not* the student-calibrated thresholds — CS-08 §16.5 names this as the most common post-rollback incident, because the old model's confidence now sits on the wrong side of a cutoff fitted to a different distribution.
- **Why asked:** It is the production half of the module, and the threshold-rollback failure is the one nobody plans for.
- **Trap:** Monitoring only accuracy or only latency. Both are lagging and neither is distillation-specific.

---

## Level 5 — Debugging & Incident Response

> Answer with a *sequence*: what you check first, what the check rules out, and what you do if it does not.

**Q38. Soft loss is ~1e-6 while hard loss is ~2.0. Sequence.**

- **Answer:** (1) Read the reduction. `F.kl_div`'s default is `mean`, which divides by `numel`. On a `(B, L, V)` tensor that divides by `B·L·V` (≈3×10⁸ at L=2048, V=150k); on a flattened `(B·L, V)` tensor it divides by `V` (≈150,000). Either way the soft term is annihilated — by a different number, which is why the bug is slippery. (2) Fix: flatten deliberately and use `reduction="batchmean"`. (3) Then check the `T*T` factor is present, because that is the *other* way the soft term gets small. (4) Re-print both terms with `code/07_distillation.py`'s own diagnostic — it prints `hard CE`, `KD (x T^2)`, `mixture` and `positions` from one forward pass before training, which is exactly the check you want. It also warns explicitly when the KD term is under 10 % of the CE term despite carrying most of the `alpha` weight, which is the signature of this bug.
- **Why asked:** It is CH-08 §8's first row and the most common "KD is not working" report.
- **Trap:** Raising the soft-term weight. That multiplies an already-broken number.

**Q39. The student gets *worse* as you raise T. Sequence.**

- **Answer:** (1) Missing `T*T`. Without it the soft term shrinks as `1/T²`, so "more temperature" appears to hurt and you conclude — wrongly — that soft targets do not help. (2) Verify the `T²` is *outside* the `kl_div` call, i.e. `F.kl_div(...) * (T*T)` and not the temperature argument. (3) Check whether `T` is being applied to both the teacher and the student. It must be the same `T` on both sides, or you are comparing two different distributions and the KL is meaningless. (4) Check for `OverflowError`/NaN in the softmax — `T` applied to already-large logits in fp16 overflows; use bf16 or subtract the max first.
- **Why asked:** It is a three-line differential where the first line is the answer, and the third (asymmetric `T`) is the one that survives even after the fix.
- **Trap:** Concluding "soft targets don't help here" and reverting to hard labels. That conclusion is an artefact of the missing correction.

**Q40. `RuntimeError: The size of tensor a (V1) must match the size of tensor b (V2) at non-singleton dimension 2`. Sequence.**

- **Answer:** (1) Confirm it is a vocabulary mismatch, not a batching bug: print `teacher.config.vocab_size` and `student.config.vocab_size` and `len(teacher_tok)`, `len(student_tok)`. (2) Note the failure is *positional* — the two tokenizers split the same text differently, so even equal vocab sizes do not imply aligned ids; a projection matrix over the id space is meaningless. (3) Choose a route: sequence-level KD (always works), or a same-family student, which is why the notebook's Phi-2 → Phi-1.5 demo runs at all (both use the CodeGen tokenizer, V = 51,200). (4) `code/07_distillation.py` makes this a go/no-go: on `--token-kd` it loads the tokenizers even under `--dry-run` precisely to run the alignment check and exit early, so the run you are debugging should never have started.
- **Why asked:** It is the error that kills most token-level KD attempts, and the interview answer should include "the tool already told you."
- **Trap:** Trying a size fix — padding the smaller logit tensor. The ids do not mean the same things.

**Q41. The student beats the teacher by 38 accuracy points on a 3-class task. Sequence.**

- **Answer:** (1) Ask whether the teacher was trained at all. A base checkpoint with a randomly initialized head scores ~33 % — that is the notebook's own failure (STOP condition 2) and it makes every "improvement" meaningless. (2) Check whether the *comparison* is valid: teacher and student evaluated on the same set, same preprocessing, same class order. (3) Check for leakage — was the teacher's evaluation set inside the student's distillation data? The student can memorise the teacher's answers on the exact rows it trained on. (4) Check the teacher's `eval()`/dropout state during distillation, which makes its soft targets noisy and its *measured* accuracy worse. (5) Only then consider that the student genuinely beats the teacher, which for a *narrow* task with a well-curated transfer set does happen.
- **Why asked:** The 60.8 %-vs-22 % case is CS-08 §19's own question 8, and it is designed so that the naive reading ("the student is better!") is wrong.
- **Trap:** Concluding the student exceeded its teacher. All four explanations above must be eliminated first.

**Q42. Accuracy matches the teacher, but the routing rule `max_prob > 0.9` now fires on 3 % of traffic instead of 11 %. Sequence.**

- **Answer:** (1) This is calibration, not accuracy — Stanton et al. 2021: students match the teacher's *predictions* but not its *predictive distribution*. (2) Measure it: plot the confidence histogram for both models and compute ECE. (3) Re-fit the threshold on the **student's** validation distribution rather than porting the teacher's — this is the fix, and it is a one-line config change. (4) Re-check downstream: any business rule, escalation path, or abstention policy keyed on a confidence number needs the same treatment. (5) The free bonus is that KD often *improves* calibration (it is a form of soft-label training) — so claim it with an ECE number, not with a feeling, and re-derive the threshold on held-out data.
- **Why asked:** It is CS-08 §19's question 9, and it is the failure mode that survives a successful distillation.
- **Trap:** Retraining the student to fix a threshold. The model is fine; the operating point is wrong.

**Q43. Disk fills during logit caching. Sequence.**

- **Answer:** (1) Compute what you were about to write before writing it: full-vocab fp32 for a 150k vocabulary is **600 KB/token**, and for 1M tokens that is **600 GB** — the script prints this arithmetic verbatim in its `--dry-run` plan (`150,000 x 4 bytes (fp32) = 600 KB/token`). (2) Switch to top-k. Top-20 fp16 is **120 B/token** and top-100 is **600 B** at the 6 B/entry convention CH-08 §7.1 uses, or **160 B** and **800 B** at the 8 B/entry convention the script charges — a 33 % difference, so state which you used. (3) Re-check the ratio: full fp32 logits at a 150k vocabulary are ~**150,000×** the size of the corpus they describe, so the text alone is ~4 B/token and a 1M-token corpus is 4 MB. (4) If the run is already partly written, delete and restart — a truncated cache silently mis-aligns indices.
- **Why asked:** It is the "what breaks in production" question, and the arithmetic is the answer: nobody should ever hit this error, because the script computes the number for you first.
- **Trap:** Compressing the cache with `np.savez_compressed` and calling it fixed. CH-08 §11 lists `np.savez_compressed` disk-full as its own row.

**Q44. The distillation loss decreases but eval does not improve. Sequence.**

- **Answer:** (1) You are fitting the teacher's noise — CH-08 §8's last row. Lower `T` to 2–5. (2) Curate the corpus: the teacher's mistakes are transferred *faithfully*, so a teacher wrong on 8 % of the transfer set teaches that 8 % with full confidence. (3) Check the teacher is actually better than the student *on this task* (STOP condition 1: the gain must exceed the evaluation's noise floor). (4) Measure teacher entropy on the transfer set — if the soft targets are near-uniform after temperature scaling, the soft term is a label smoother and the KD loss is measuring nothing (CS-08 §13.1's transferable rule). (5) Check the hard term is not dead: `(labels != -100).float().mean()` on one batch, and whether `kl_div`'s reduction is eating the soft term (Q38).
- **Why asked:** It is the terminal symptom of five different root causes, and the sequence is what is being graded.
- **Trap:** Training longer. The train loss is bounded below by the teacher's entropy, so it will keep descending regardless.

---

## Rapid Fire — True / False / One-Liner

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | Soft targets are a regulariser, like label smoothing | **False** | The MNIST student classified 98.6 % of a class absent from all its training data |
| 2 | `T²` multiplies the whole loss | **False** | Soft term only; the hard term is at `T = 1` against a one-hot |
| 3 | Raising `T` increases the soft-term gradient | **False** | Without `T²` it shrinks as `1/T²` |
| 4 | `α = 0.7` means the same thing in CH-08 and in the script | **False** | CH-08: soft; the script: hard. The script's default is 0.5, CH-08's is 0.7 |
| 5 | `F.kl_div(input, target)` takes probabilities for both | **False** | `input` = log-probs, `target` = probs; wrong inputs give a value that can be negative |
| 6 | `reduction="mean"` on a `(B,L,V)` KL is off by ~`L·V` | **True** | ≈3×10⁸ at L=2048, V=150k |
| 7 | A bigger teacher always makes a better student | **False** | Non-monotone; >50× gap and a mid-size teacher often wins |
| 8 | The capacity-gap sweet spot is 2–10× | **True** | CH-08 §7.4 |
| 9 | Teacher and student must share a vocabulary for sequence-level KD | **False** | Seq-KD needs no logits and no shared vocabulary |
| 10 | Token-level KD requires a shared vocabulary | **True** | The script makes it a go/no-go check even under `--dry-run` |
| 11 | The top-2 or 3 classes carry most of the KL mass | **True** | A 4th class at 0.001 contributes −0.0023 nats |
| 12 | Full-vocab fp32 logits at V=150k are 600 KB/token | **True** | → 600 GB per 1M tokens, ~150,000× the corpus |
| 13 | `T = 0` is a valid way to disable softening | **False** | Soft loss is NaN; the softmax is undefined |
| 14 | KD typically overfits after ~10 epochs | **False** | 1–3 epochs; and the tell is held-out agreement, not train loss |
| 15 | The teacher must be in `eval()` mode during distillation | **True** | Train-mode dropout makes the soft targets noisy |
| 16 | Distillation reduces the student's memory footprint | **Partly** | It reduces FLOPs; the byte win needs quantization too |
| 17 | ALBERT is a distilled model | **False** | Factorized embeddings + parameter sharing, no teacher — CS-08 §13.3 |
| 18 | MobileBERT is deeper than DistilBERT | **True** | 24 layers vs 6, and smaller — depth matters for encoders |
| 19 | KD reliably beats from-scratch training with *better data* | **False** | The honest comparison; run the label-smoothing arm |
| 20 | With a near-uniform teacher, KD ≡ label smoothing | **True** | Müller et al. 2019; measure teacher entropy first |

---

## Coding / Whiteboard Tasks

### Task 1 — Write the token-level KD loss, correctly

```python
import torch
import torch.nn.functional as F

T = 4.0
alpha_soft = 0.7          # NAME the convention in your own code

def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    # 1. student: LOG-softmax at T.  2. teacher: softmax at T (plain probabilities).
    s_log  = F.log_softmax(student_logits / T, dim=-1)
    t_soft = F.softmax(teacher_logits / T, dim=-1)

    # 3. batchmean on a DELIBERATELY flattened (B*L, V) tensor, then the T^2 correction.
    soft = F.kl_div(s_log.reshape(-1, s_log.size(-1)), t_soft.reshape(-1, t_soft.size(-1)),
                    reduction="batchmean") * (T * T)

    # 4. hard term at T=1, on the same masked positions.
    hard = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)),
                           labels.reshape(-1), ignore_index=-100)

    return alpha * hard + (1.0 - alpha) * soft, hard.detach(), soft.detach()

loss, hard, soft = kd_loss(s_logits, t_logits, labels)
print(f"hard {hard:.4f}  soft {soft:.4f}  ratio {soft / hard:.3f}")   # the diagnostic
assert soft > hard * 0.1, "soft term is being annihilated — check reduction and T^2"
```

- **The decisions being graded:** `log_softmax` on the student and plain `softmax` on the teacher (the argument-order trap); an *explicit* flatten plus `"batchmean"` — and the ability to say that on a **top-k** tensor the correct reduction is instead `"sum"` divided by the position count, because `batchmean` there divides by `k`; `T*T` outside the `kl_div` and multiplying only the soft term; a named `alpha` convention; `ignore_index=-100` on the hard term; and the printed ratio as a runtime assertion.
- **Grading:** a candidate who uses the default reduction fails; a candidate who cannot say which convention their `alpha` is in is downgraded; a candidate who applies `T*T` to both terms fails.
- **Fail condition:** `F.kl_div(softmax(s/T), softmax(t/T))` — a mathematically meaningless number that can be negative, and nothing raises.

### Task 2 — Reproduce the worked example, then break it on purpose

```python
import torch, torch.nn.functional as F
q = torch.tensor([0.70, 0.20, 0.10])    # teacher, T=1
p = torch.tensor([0.50, 0.30, 0.20])    # student, T=1
y = torch.tensor(0)

hard = F.cross_entropy(p.log().unsqueeze(0), y.unsqueeze(0))
kl1  = (q * (q / p).log()).sum()
kl2  = (q.log()/2).softmax(0).mul(((q.log()/2).softmax(0) / (p.log()/2).softmax(0)).log()).sum()
print(f"hard {hard:.4f}  KL@T=1 {kl1:.5f}  KL@T=2 {kl2:.5f}  T^2*KL {4*kl2:.5f}")
# expected: hard 0.6931  KL@T=1 0.08512  KL@T=2 0.02453  T^2*KL 0.09812
```

- **The decisions being graded:** that KL is computed manually from the definition; that the T=2 distributions are obtained by softmaxing `log(p)/2`, not by squaring `p`; that raw KL *fell* 3.5× purely from temperature and the `T²` value did not.
- **Grading:** a candidate who reports "T=2 is better because the loss is lower" fails; a candidate who computes `p**(1/T)` instead of `softmax(log(p)/T)` is downgraded (it is the same thing up to normalization — say so).
- **Fail condition:** comparing raw KL across temperatures without applying `T²`.

### Task 3 — Size a distillation run before renting the GPU

```bash
# The two modes are a MUTUALLY EXCLUSIVE required group — there is no --mode flag.
python code/07_distillation.py --help

# Token-level KD: the vocab-alignment check runs even under --dry-run (tokenizers only).
python code/07_distillation.py --token-kd --dry-run --text data/corpus.txt \
    --size-hint 32B -T 4 --alpha 0.5 --top-k 100

# Sequence-level KD: generate, READ 20, then SFT the student on the result.
python code/07_distillation.py --from-teacher --teacher Qwen/Qwen2.5-32B-Instruct \
    --prompts data/prompts.jsonl --n 5000 --out data/seqkd.jsonl --dry-run

# The reference VRAM table this bank quotes.
python code/common/memory.py --table
```

- **The decisions being graded:** that `--from-teacher` and `--token-kd` are exclusive and one is required; that `-T` (KD softening, default 2.0) and `--teacher-temp` (sampling, default 0.8) are different flags on different paths; that `--alpha` weights the **soft (KD)** term in Hinton's convention with default 0.7, so `--alpha 0.7` means 70 % distillation; and the 8 B/entry storage arithmetic the script prints.
- **Grading:** a candidate who passes `--mode` fails; a candidate who cannot state which term `--alpha` weights — from `--help` and the loss line, not from memory — is downgraded, because that convention has been inverted in this repo before.
- **Fail condition:** attempting `--token-kd` with a teacher and student from different families without first confirming the vocabularies match.

---

## Cheat Sheet of Numbers To Memorize

| Quantity | Value | Why it matters |
|---|---|---|
| Hinton MNIST: hard / soft / teacher | **146 / 74 / 67** errors | The proof soft targets carry information |
| Test 3s correct after training with no 3s | **~98.6 %** (≈14 errors) | A smoother cannot do this |
| Worked example hard loss | **0.6931** nats | `−ln(0.50)` |
| `KL@T=1` / `KL@T=2` | **0.08512 / 0.02453** | A 3.5× drop that is a scaling artefact |
| `T²·KL` at T=2 | **0.09812** | Back in the same ballpark — the point of `T²` |
| Gradient with / without `T²` at T=2 | **−0.21484 / −0.05371** | Within 7 % of T=1 vs 3.7× smaller |
| Two α conventions, same example | **0.2766 vs 0.1576** | 76 % apart — read the code |
| `T` default / useful range | **4 / 2–20** | ≥20 is usually noise |
| `α` weight on the **soft** term / default | **0.7 / 0.7** | CH-08 §4, CS-08 §7.1 and `code/07_distillation.py` all agree, and all match Hinton's *direction* — the paper itself defines no `α` |
| Capacity-gap sweet spot / danger | **2–10× / >50×** | Non-monotone; >200× is waste |
| Epochs | **1–3** | KD overfits too |
| Top-k storage, 8 B/entry | **6.8 GB** per 42.5M tokens at k=20 | The script's arithmetic |
| Top-k storage, 6 B/entry | **120 B/token** at k=20 | CH-08 §7.1's convention — 33 % apart |
| Full-vocab fp32, V=150k | **600 KB/token** | 600 GB per 1M tokens; ~150,000× the corpus |
| Raw text | **~4 B/token** | ~170 MB for 42.5M tokens |
| Encoder full-vocab cache, V=30,522 | **6.1 GB** per 100k examples | Why encoder KD can afford full distributions |
| DistilBERT vs bert-base | **66.9 M / 40 % smaller / 1.63× / 77.0 vs 79.5** | The reference point |
| Teacher forward vs student training FLOPs | **2ND vs 6ND** | Teacher is a one-off `bert-large` pass |
| KD overhead vs student-only | **~2.01×** | CS-08 §4.4 |
| Typical KD gain over hard labels | **1–4 points** | Encoder classification, 100k examples |

---

## Answers To The Self-Check Questions From CS-08

**1.** `L = α·T²·KL(q_teacher^T ‖ p_student^T) + (1−α)·CE(y, p_student)` with `q` the teacher's temperature-softened distribution, `p` the student's, `T` the softening temperature, `y` the one-hot label, `α` the mixing weight. Here `α` names the **soft** term, giving 0.2766 on CS-08 §4.2.4's example; naming the hard term with the same letter gives 0.1576 — 76 % apart for identical physics. **Hinton's paper defines no `α` at all**: it calls the loss "a weighted average of two different objective functions", names the soft-target term first, and puts the *lower* weight on the hard targets (0.5 relative in the ASR runs). The notebook's `alpha_soft`, CH-08 §4 and `code/07_distillation.py` (default 0.7, documented as *"Weight on the SOFT (KD) term"*) all follow that direction. Name the term your symbol weights in the code — and never cite a paper for a symbol it does not contain.

**2.** `∂L_soft/∂z_k = (1/T)(p_k − q_k)` for the temperature-scaled logits, hence `∝ 1/T²` once the `z/T` chain rule is included. The `T²` multiplier exists to restore the `(p_k − q_k)` gradient magnitude you had at `T = 1`, so that temperature controls the *shape* of the target rather than its weight. As `T → ∞`, `p^T` and `q^T` both tend to uniform, `KL → 0`, and the objective degenerates to the hard term plus a constant — which is why the useful range stops around 20.

**3.** `L_hard = −ln(0.50) = 0.6931` nats. `KL(q‖p)@T=1 = 0.08512` nats. At `T = 2`, `q^{T=2} = [0.52288, 0.27949, 0.19764]` and `p^{T=2} = [0.41546, 0.32180, 0.26275]`, giving `KL = 0.02453` — a 3.5× reduction that is purely a scaling artefact from flattening the softmax. Applying `T²` gives `0.09812`, comparable to the `T = 1` value. The per-class cat-logit gradient at `T = 2` is `−(p − q) = −0.10742`, so with `T²` it is `2 × 0.10742 = −0.21484` (within 7 % of the `T = 1` value of `−0.20`) and without it `0.10742 / 2 = −0.05371`, 3.7× smaller.

**4.** The four axes: **what** is supervised (response / feature / relation), **where** (output / hidden / attention / embedding), **teacher availability** (offline / online-DML / self), and **student size** (compression vs same-size born-again). **RKD** supervises pairwise and triple-wise *distances between examples* in a representation space — alignment problem: none for width or head count, only the choice of distance metric, which is why it composes freely. **Attention transfer** supervises `Σ_c |A_c|²`, the squared attention map summed over channels — alignment: crosses head counts but requires matching spatial dimensions. **FitNets** supervises an intermediate hidden state through a learned regressor from student width to teacher width — alignment: widths differ so a projection must be learned, and *which* layer to hint is a hyperparameter.

**5.** **Sequence-level KD** trains the student on the teacher's generated sequences as hard labels; **token-level KD** matches per-token distributions at temperature. Only sequence-level is available from a text-only API, because the API returns sampled tokens and discards the distribution. Seq-KD beats training on the original human references because the teacher's output is drawn from a distribution conditioned on a *function class the student can actually represent*, and the teacher's errors are the errors of a model that has already solved the task — the student is imitating a reachable target rather than fitting a human reference that may be off the student's own manifold.

**6.** The gap is `teacher_params / student_params`; the effect is **non-monotone** with a sweet spot at **2–10×** and degradation past **50×**. The two results: **Cho & Hariharan 2019 (CIFAR-100)** — student accuracy is non-monotone in *teacher training length*, peaking early and then declining; and the standard **>50×** observation that a mid-size teacher beats a 70B for a 1B student. The three mechanisms: the teacher's capacity to represent a function the student cannot, the teacher's confidence calibration being wrong for a small model, and the distribution mismatch that compounds once the student's own errors enter. Two cheap fixes: pick a mid-size teacher (free), or grow the student (you were going to deploy it anyway).

**7.** Deltas from `bert-base-uncased`: 12 → **6** layers, 109.5 M → **66.9 M** params (**40 % smaller**), **1.63×** faster, GLUE **79.5 → 77.0**. Three loss terms: the **response loss** on the teacher's soft targets, the **cosine embedding loss** on the last hidden state, and the **MLM loss**. Dropping the MLM term removes the general language-modelling objective, so the student over-specializes on the distillation task and loses general ability — the retention gap widens even though the response loss looks healthy.

**8.** Ordered by likelihood: **(a) the teacher was never trained** — a base checkpoint with a randomly initialized head scores ~33 % on 3 classes, which *is* the 22 %, and it is CS-08 §8.2's STOP condition 2 and the notebook's own failure. **(b) the comparison is invalid** — different evaluation sets, preprocessing, or class order between teacher and student; check the label mapping. **(c) leakage** — the teacher's evaluation rows are inside the student's distillation data, so the student memorised the answers. **(d) teacher in train mode** during distillation, so its measured accuracy is depressed by dropout while its soft targets are noisy. Only after eliminating all four does "the student genuinely beats its teacher on a narrow task" become the answer.

**9.** Calibration changed, not accuracy: the student matches the teacher's *predictions* but not its *predictive distribution* (Stanton et al. 2021). KD is a form of soft-label training, and it typically makes the student *more* confident, so a threshold fitted to the teacher fires on fewer rows. **The fix is to re-fit the threshold on the student's validation distribution**, not to retrain the model — and to re-check every downstream rule keyed on a confidence number, because rolling back the model without rolling back the thresholds is the most common post-rollback incident in this area (CS-08 §16.5).

**10.** Pipeline: (1) build a prompt set covering the target distribution; (2) call the API teacher and store the text at k = 1–3 with a cheap-judge filter first; (3) apply a quality filter — refusals, boilerplate, instruction-echoing, truncation — because these survive generation and are imitated faithfully; (4) **decontaminate** against your evaluation set with the 13-gram rule; (5) train the student on the filtered `(instruction, response)` pairs as ordinary SFT, mixing in ≥5–10 % human replay to slow catastrophic forgetting; (6) evaluate with per-slice and worst-class metrics and a judge whose self-preference bias you have measured. **The one thing you cannot do** is token-level logit KD — the API never returns the distribution, and even if it did, the student's tokenizer is from a different family, so per-position alignment is meaningless (CS-09 §4.4). The notebook's LLM section attempts exactly this and only appears to work because Phi-2 and Phi-1.5 share the CodeGen tokenizer.

### Corrections to CH-08 discovered while writing this bank

> **Settled:** `cheat-sheets/CH-08-Knowledge-Distillation.md` used to contradict itself on top-k
> storage — §2's formula table prescribed `k × (4 + 4) × tokens` (**8 bytes per entry**, worked as
> `20 × 8 × 1M = 160 MB`) while §7.1's table gave top-20 **fp16** = **120 B/token**, top-50 = 300 B,
> top-100 = 600 B, i.e. **6 bytes per entry** (fp16 value + int32 index). A 33 % spread inside one
> card. The row is now precision-explicit — `k × (4 B index + value bytes) × tokens`, with both
> `k=20 fp16 → 120 MB` and `k=20 fp32 → 160 MB` shown — so the two tables agree by construction.
>
> **The reusable point is the one that survives the fix:** the entry size is not a property of
> *top-k storage*, it is a property of *the precision you cached the values in*. `code/07_distillation.py`
> charges 8 B/entry (*"top-k fp32 logit + int32 index"*, arithmetic `a.top_k * 8`); CH-08 §7.1's
> table is fp16 and charges 6. Both are right about their own tensor. **Whenever you quote a
> bytes-per-entry figure, name the value precision in the same breath** — otherwise the number is
> 33 % wrong for half your readers, and silently so.

> **Settled:** `cheat-sheets/CH-08-Knowledge-Distillation.md:328` shows the run-it command
> `python code/07_distillation.py --token-kd --dry-run --text data/corpus.txt --size-hint 32B -T 4
> --alpha 0.7 --top-k 20`, and the same file's §4 table (line 119) defines α as *"Weight on the
> soft (KD) loss vs hard-label CE"* with default **0.7**. These once disagreed — the script used to
> weight the **hard** term, so `--alpha 0.7` produced 70 % hard / 30 % KD — and every card that
> reported the disagreement (`CH-09` §2, §4.1 and its Correction block; the IQ-08 and IQ-09
> corrections that quoted them) has since been reconciled. `code/07_distillation.py --help` now
> reads *"Weight on the SOFT (KD) term… 0.7 means 70 % distillation / 30 % ground truth"*, its loss
> line is `a.alpha * kd + (1 - a.alpha) * hard`, and its docstring records the fix.
>
> **Two lessons outlive the fix.** (1) **Read `--help` and the loss line of the file in front of
> you; never trust a quoted flag default or line number in a card — including this one.** A
> correction block is a snapshot of a defect, and it expires the moment someone fixes the defect;
> several blocks in this bank had to be rewritten for exactly that reason during the same pass that
> fixed the code. (2) **A defect that lives in prose has to be fixed wherever the prose is
> duplicated.** The α convention was wrong in one script and correctly *reported* as wrong in four
> markdown files; fixing the script silently invalidated all four reports.

> **Correction (the reduction, which is shape-dependent):** `cheat-sheets/CH-08-Knowledge-Distillation.md:180` uses `F.kl_div(s_log, t_soft, reduction="batchmean") * (T * T)` on a **full-vocabulary** tensor, where `batchmean` divides by the batch dimension and is right. `code/07_distillation.py` operates on a **top-k truncated** tensor and therefore uses `reduction="sum"` divided by the position count instead — its comment states that `batchmean` there *"would divide by `top_k`, not by the position count."* Both are correct for their own tensor; neither is correct for the other's. When you quote a reduction, **say which tensor it is applied to**.

> **Correction:** `cheat-sheets/CH-08-Knowledge-Distillation.md:338` tells you to filter the generated corpus with `python code/data/make_instruction_data.py --filter --in data/seqkd.jsonl`. Both flags are invented: `make_instruction_data.py`'s argparse has a required mutually exclusive group `(--from-docs | --template)` and otherwise only `--out`, `--n`, `--provider`, `--model`, `--chunk-words`, `--seed`. There is no `--filter` and no `--in`, so the command exits on `unrecognized arguments`. The file does contain a real `quality_filter()`, but it is only called on data the script generated itself; import it, or write the rules yourself from CS-09 §5.3's filter stack.

> **Correction:** `cheat-sheets/CH-08-Knowledge-Distillation.md:339` and `:603` both print `python code/01_sft_lora.py --data … --out out/student`. `code/01_sft_lora.py`'s argparse has **no `--out` flag** — the output directory flag is `--output` (default `./out/sft-lora`). `code/07_distillation.py` used to print the same wrong flag and has since been corrected to `--output`, but the two CH-08 lines have not been. **Use `--output`.**

---

## Cross-References

| Module | Relationship to IQ-08 |
|---|---|
| **CS-08** Knowledge Distillation Foundations | The source case study. Its §4.2.4 worked example, §12.3 calibration, §12.5 worst-class recall, §13.1 MNIST experiment and §13.3 encoder table are the ground truth for this bank; its §19 questions are answered in full above |
| **CH-08** Knowledge Distillation Cheat Sheet | §2 the formulas, §4 the hyperparameter band, §5.1 the loss snippet this bank's Task 1 corrects, §7.1 the storage wall, §8 the 14-row symptom→fix lookup |
| **CS-09** LLM → SLM Distillation | **Direct continuation.** Response distillation, synthetic-data pipelines, cross-tokenizer alignment, model collapse, R1-Distill. Assumes §0–§13 of CS-08 |
| **CS-07** BERT Fine-Tuning & Task Heads | The `bert-large` → `bert-base` setup; the classification head, `num_labels` and `DataCollatorWithPadding` that Q33 depends on |
| **CS-05** RNN/LSTM → Attention | The softmax/logit gradient machinery Q25's derivation assumes |
| **CS-06** Hugging Face Masterclass | `from_pretrained`, `AutoModelForSequenceClassification`, and the randomly-initialized-head warning that Q41 and STOP condition 2 turn on |
| **CS-13** Instruction Fine-Tuning | Where the `(instruction, response)` schema and the masking convention for sequence-level KD live, and its §12 *Evaluation — How To Know It Worked* is the four-layer stack behind the acceptance gates in Q31, Q35 and Q44 |
| **CS-10 / CS-11** Quantization I & II | The orthogonal compression axis; where "distil then quantize" is measured and why int8 is the teacher floor |
| **CS-14** RLHF, PPO, DPO, ORPO | The alignment layer above this: where a distilled student would be preference-tuned, and the `β` KL anchor |
| **CH-08 / CH-09** | The two cards this bank is built from — quote the script's conventions over the tables' where they disagree |
| **IQ-09** LLM → SLM Distillation | The sibling bank: sequence-level KD, on-policy distillation, model collapse, the cost model, self-preference bias |

---

*End of IQ-08. Companion artifacts: `CS-08-Knowledge-Distillation-Foundations.md` (case study), `CH-08-Knowledge-Distillation.md` (cheat sheet). Ground truth: `code/07_distillation.py --help`, `code/01_sft_lora.py --help`, `code/data/make_instruction_data.py --help`, `code/common/memory.py --table`.*
