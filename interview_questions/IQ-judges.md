# Interview Pack · LLM-as-a-Judge & Evaluation Methods

> **Covers:** CS-06 (offline vs online evals), CS-07 (LLM-as-a-judge; reference-based vs
> reference-free), CS-08 (G-Eval — the deterministic LLM-as-a-judge framework) ·
> **Code ground truth:** `CODE-01` (`PATTERNS.md` judge recipe), `CODE-02` (`cot_classify` — the
> verbatim judge spec), `CODE-03` (`validate-evaluator`) · **Role level:** mid → senior · **Format:**
> screen → onsite → system-design

---

## How this domain is assessed

Judging is the highest-leverage and most-abused technique in LLM evaluation. It is the only way to
score subjective quality at scale, and it is the easiest way to produce a number that looks like a
measurement while measuring nothing.

Interviews in this domain are effectively **statistics interviews wearing an ML hat**. The interviewer
wants to know whether you treat a judge as a *classifier with a validation protocol* or as an *oracle*.
Candidates who have shipped a judge usually have a story about discovering that raw agreement was
lying to them; candidates who have not will reach for a Likert scale in the first two minutes.

Two framings from the source lectures anchor the whole domain, and a candidate who has them will sound
different from one who does not:

1. **There are exactly three evaluation methods — a program, a human, or an LLM.** Not four, not a
   spectrum. Every pipeline ever built is executed by one of them, whether it is measuring a component,
   a workflow or a whole application; what varies is which one is *feasible* (CS-07). LLM-as-a-judge
   exists because it sits **between** the other two — it has a program's scalability and a human's
   capacity for judgement.
2. **Metrics split into count-based and judgment-based, and only the second kind needs a judge**
   (CS-08). Count-based metrics (recall, precision, faithfulness) decompose the output into claims,
   mark each, and compute a ratio. Judgment-based metrics (style, correctness, completeness,
   helpfulness, all safety metrics) cannot be decomposed at all — the property lives at the *answer*
   level, not the sentence level. A candidate who cannot say *why* a metric cannot be counted has not
   understood what a judge is for.

| What is being scored | The signal | How it is elicited |
|---|---|---|
| **Taxonomy fluency** | Do you know the three methods, and why some metrics can't be counted? | "Why does this metric need a judge at all?" |
| **Classification thinking** | Do you reach for binary labels + TPR/TNR? | "How would you validate your judge?" |
| **Bias awareness** | Do you name positional bias, self-preference, verbosity? | "What can go wrong with a judge?" |
| **Cost awareness** | Do you know judging is the dominant eval expense? | "How do you judge 50,000 responses a day?" |
| **Method selection** | Do you know when *not* to judge? | "Should we use a judge for this?" |
| **Reference discipline** | Do you distinguish reference-based from reference-free? | "Where does ground truth come from?" |

**The single most predictive question in this pack** is *"what would make you distrust your judge?"*
A strong candidate answers immediately with a concrete failure — imbalanced data hiding a TPR of zero —
and a weak candidate says "I'd spot-check a few outputs."

---

## Tier 1 — Fundamentals (10 Q)

### Q1. What is LLM-as-a-judge?

**What they're really testing:** whether you have a definition beyond "use a model to score things".

**Model answer:**
- LLM-as-a-judge uses a language model to evaluate another model's output against a criterion.
- It is the **third of exactly three methods** — a program, a human, or an LLM. Nothing else executes
  an eval pipeline. Programmatic is free and exact but only possible when a deterministic check exists;
  human is the most reliable and the most expensive, and is unusable at scale; the judge sits between
  them with a program's scalability and a human's capacity for judgement. Most real pipelines use it.
- It exists because the alternative is worse at scale: human review is the highest-quality signal but
  does not scale to per-release gates, and string metrics (BLEU/ROUGE/EM) cannot score open-ended
  quality at all.
- The sharper reason a judge exists is that **some metrics cannot be counted at all.** Count-based
  metrics — recall, precision, faithfulness, answer relevance — work by decomposing the output into
  claims, marking each favourable or unfavourable, and computing a ratio. Judgment-based metrics cannot
  be decomposed:
  - **Style is an answer-level property.** A brand voice like *Why → What → How* "will not exist in
    every single sentence … it will exist at the answer level". Claim-level decomposition destroys the
    very thing being measured.
  - **Correctness breaks on analogies.** Decompose an answer containing an analogy, and the analogy
    becomes an isolated claim that the judge compares against the expected answer, finds unrelated, and
    penalises. Correctness has to be judged over the whole answer.
- So the honest boundary: **if the property can be counted, a judge is waste; if it cannot, a judge is
  the only scalable instrument.**
- The judge takes an input, an output, and usually a criterion; it returns a verdict plus — critically
  — **reasoning**.
- The reasoning is not decoration: it is what you read when a score moves, and it is what makes the
  judge debuggable.
- A judge is a **classifier**, so it has labels, a decision boundary, and two error rates that must be
  reported separately.
- It must be **validated against human labels** before any of its numbers mean anything. That is the
  step the industry skips.
- Note what the **human** method actually covers, because it is not only "someone rates the output".
  Humans do five distinct things in LLM evaluation: **direct grading**, **red teaming** (deliberately
  attacking the system to find where it breaks, before launches), **A/B testing** (two variants live in
  production, users rate, the better one ships), **annotation** (creating golden datasets and rubrics),
  and **human-in-the-loop** (the fallback when programs and judges cannot resolve a case, or when a
  threshold leaves a **grey area**). Every one of those is a live interview topic on its own — and note
  that red teaming and A/B testing are *not* judge substitutes, they are different instruments.

**Red flags:** "we ask GPT-4 to rate it 1–10"; treating the judge's output as ground truth; no
mention of human labels; being unable to say why a metric cannot be computed in code.

### Q2. What is the difference between reference-based and reference-free evaluation?

**What they're really testing:** whether you know where ground truth comes from.

**Model answer:**
- **Reference-based** evaluation compares the output against a known-good reference — a gold answer, a
  labelled class, an ideal response. It is precise and directly interpretable, but it requires you to
  *have* the reference, which is expensive and sometimes impossible.
- **Reference-free** evaluation judges the output on its own terms — is it grounded in the retrieved
  context, is it relevant to the question, is it coherent, is it safe. No gold answer needed.
- The split matters because it determines **how your dataset scales**. Reference-free metrics let you
  evaluate unlimited production traffic; reference-based metrics cap you at your labelled set.
- The two are complementary and usually run together: faithfulness is reference-free (answer vs
  retrieved context), correctness is reference-based (answer vs truth).
- A subtle case: contextual recall *feels* reference-free but is reference-based — it needs an ideal
  answer that the judge decomposes into claims.
- Practical rule: **use reference-free metrics online at scale, reference-based metrics offline on the
  golden set.**

**The one-question diagnostic test** — the source gives a single test that classifies any pipeline you
are shown, and it is worth stating verbatim:

> **"Does your golden dataset hand you a correct answer?"** If yes → reference-based. If no →
  reference-free.

The definitions it is testing:
- **Reference-based:** "you have a reference and/or correct answer and the key things a correct answer
  must contain … for each test case. You grade by comparing the output against the reference." The
  dataset carries **question + correct answer**.
- **Reference-free:** "you have no predefined correct answer. You judge the output's quality directly
  on its own terms against a criteria and rubric." The dataset carries **only the question** — one
  column.

Applying the test to the three worked examples in the source, which is a good self-check:

| Example | Golden dataset contains | Classification |
|---|---|---|
| Retriever / Recall@K | which document IDs hold the answer (1001; 1001 and 1003) | **Reference-based** |
| Human grading of helpfulness | questions only — one column, no answer | **Reference-free** |
| LLM-judged exam marking | the human marker's marks per answer | **Reference-based** — "who is the correct one? The human's evaluation" |

**The trap that follows from this, and it is the single most-missed distinction in the area:** a
**rubric is not a correct answer.** A rubric is *a scale*, not a particular right answer. So the
presence of a detailed rubric does **not** make an evaluation reference-based. The helpfulness example
had a 1–5 rubric and was still reference-free, because no correct answer existed in the dataset. Only a
stated correct answer makes an evaluation reference-based.

**Red flags:** treating them as interchangeable; assuming reference-free means "no ground truth
needed" for the whole program; not knowing that recall-type metrics need a reference; **arguing that a
rubric makes an evaluation reference-based.**

### Q3. What is G-Eval, and how does it differ from a binary judge?

**What they're really testing:** whether you know more than one judge architecture.

**Model answer:**
- First, the honest framing, which the source states flatly: **G-Eval is not a new paradigm — "it is
  also just LLM-as-a-judge."** It is a paper (2023) that fixes the two reasons naive LLM-as-a-judge is
  unreliable, with exactly **two innovations and nothing else**:
  1. **Convert the criteria into evaluation steps via Chain-of-Thought.** A one-line criterion
     ("decide how factually correct this is") is not a tight enough statement — the model perceives it
     differently on each call. So the judge is asked to expand it into **four to five concrete
     evaluation steps**: "a rulebook … a constitution on how we measure correctness". The result is
     that "we haven't left any scope for it to use its own brain."
  2. **Take a probability-weighted score instead of the emitted integer.** The prompt forces a digit
     0–10, so the digit tokens carry the probability mass. Extract the **top-k** (k = 5) token
     log-probabilities, **drop the non-numeric tokens**, **normalise** them to sum to 1, and take the
     **weighted average**.
- The variance mechanism matters, because it explains why the innovation works. The model is often
  nearly tied between adjacent digits — 8 at 51% and 7 at 40% on one run, then 8 at 40% and 7 at 51% on
  the next. The emitted integer is just the argmax, so it flips; taking the weighted average keeps the
  model's own uncertainty instead of discarding it.
- The worked arithmetic, because interviewers ask for it:
  ```
  top-5 tokens:  8 → 0.70,  7 → 0.20,  9 → 0.05,  "the" → 0.01,  ":" → non-numeric
  drop non-numeric          → the three digits now sum to 0.95
  normalise (divide by .95) →  0.7368,  0.2105,  0.0526
  weighted average          →  8(0.7368) + 7(0.2105) + 9(0.0526)  =  7.84
  naive emitted integer     →  8
  normalise to 0–1          →  7.84 / 10 = 0.784   vs threshold 0.7  → pass
  ```
  The paper's own illustration makes the same point: naive **3** versus weighted **2.59**.
- The payoff is determinism, and the source gives the before/after: naive judging moves **60 → 70 →
  75** across runs; G-Eval moves **84 → 83**, with **the same test case failing both times**. That is
  the sentence to quote: *"the whole point of G-Eval is to make the evaluation process more
  deterministic. Less probabilistic."*
- **Requirements and knobs** worth knowing, because they are where implementations silently go wrong:
  - It needs a model that **exposes token log-probabilities** for the top-k. A chat endpoint that does
    not return them cannot run innovation 2 at all. The paper recommends **GPT-4** as the judge.
  - In DeepEval, **`strict_mode=True` silently disables innovation 2** — the weighted calculation does
    not happen and you get the raw integer back, i.e. you have quietly reverted to the naive version.
    Keep it `False`.
  - **Criteria vs evaluation steps** is a real design decision: pass `criteria` and let CoT generate
    the steps *at the start*, while you are still learning the pipeline; once you understand the
    pass/fail picture, **pass your own evaluation steps directly** — because steps regenerated on every
    call vary slightly and reintroduce exactly the variance you removed.
  - Always supply a **scoring rubric** (explicit bands, e.g. 0–4 / 5–8 / 9–10). Without one, "G-Eval
    decides on its own how much to score an answer" — you have fixed the scale's *stability* but not
    its *meaning*.
- It produces a **graded** score, so it captures degrees of quality that a binary judge collapses.
- That expressiveness is also its weakness: a graded score is harder to validate with classification
  metrics, and "the gap between 3 and 4 is noise".
- Use G-Eval for **any judgment-based metric** — correctness, completeness, style, helpfulness, safety
  metrics. Use a **binary judge** when you are detecting a specific failure mode — there, the crisp
  decision and the clean classification metrics are worth more than resolution. You *can* use naive
  LLM-as-a-judge: "that will also work. But what's the problem? The score will jump around a lot."
- The two share the important part: both are only trustworthy after validation against human labels.

**Red flags:** "G-Eval is just a better prompt"; calling GPT-4 the innovation (it is a recommendation);
not knowing about the probability weighting; `strict_mode=True`; leaving the scoring scale to the
judge; regenerating evaluation steps on every call after you know them; using a graded judge to detect
a binary failure.

### Q4. Why binary PASS/FAIL rather than a 1–5 Likert scale?

**What they're really testing:** whether you have thought about measurement noise.

**Model answer:**
- Because **the gap between a 3 and a 4 is noise**, while PASS/FAIL forces a crisp decision the model
  (and the human) can actually make consistently.
- A binary label lets you use **classification metrics** — TPR, TNR, precision, recall — which are
  exactly the tools you need to validate the judge. A Likert score gives you a continuous number with
  no error model.
- Inter-annotator agreement is markedly higher on binary labels; humans disagree about whether
  something is a 3 or a 4 far more than about whether it passes.
- Binary labels compose: several binary judges (grounded, relevant, complete, safe) give a
  diagnostic profile, whereas one 1–5 "quality" score gives an uninterpretable average.
- Exceptions exist — when a genuine ordering is the product requirement (ranking two summaries),
  **pairwise comparison** is more reliable than a Likert scale and still avoids absolute scoring.
- If you need granularity, get it from **more binary judges with narrower criteria**, not from a wider
  scale.

**Red flags:** defending a Likert scale for "nuance"; no mention of classification metrics; never
considering pairwise.

### Q5. How do you validate a judge?

**What they're really testing:** whether you have actually done it.

**Model answer:**
- Collect roughly **100 human-labelled examples**. Fewer and the error bars are too wide to decide
  anything.
- **Split** into examples you may use to iterate on the judge prompt, and a **held-out** set you look
  at only at the end. Tuning against the held-out set destroys it.
- Run the judge, then build the confusion matrix with **"positive" = the judge says FAIL**.
- Report **TPR** (share of real failures caught) and **TNR** (share of real passes cleared)
  **separately**, plus raw FP and FN counts.
- **Never report accuracy or raw agreement.** A judge that rubber-stamps PASS scores ~90% agreement
  when 90% of traces pass, while catching zero failures — accuracy looks excellent and TPR reads ~0.
- Ship only when **both** clear roughly **0.9**, and re-validate whenever the model, prompt or data
  distribution changes.
- Add a second, cheaper validation that catches a different failure: **re-run stability on a frozen
  set.** Run identical settings twice and check that the aggregate score barely moves and that *the
  same test cases* fail. G-Eval's published demonstration is the model for this — 84, then 83, with the
  same single failing case, against a naive judge moving 60 → 70 → 75. TPR/TNR tells you whether the
  judge is *right*; re-run stability tells you whether it is *reproducible*. You need both.

**The exception worth knowing, because it is where the standard protocol does not apply.** The
TPR/TNR protocol assumes a **binary** judge against a **binary** reference. When the reference is an
**ordinal mark** rather than a label — a human marker's score out of 15 on an exam answer — there is no
positive class and no confusion matrix. The right metric becomes **Mean Absolute Error**, and the
source's worked case is a judge marking exam answers against a human expert:

```
MAE = ( Σ | llm_marks − human_marks | ) / n_answers
```

Illustrative result: **2.3** — "on average my LLM, when evaluating answers, deviates from a human by
±2.3 marks." **The ideal is 0**, and the improvement levers are the same three as everywhere else:
a better judge model, a changed system prompt, a changed rubric. Two things are worth saying about it:

- **It is still a reference-based evaluation** by the CS-07 test — the dataset contains the human's
  marks, so a correct answer (in the sense of "the standard of correctness") does exist. What changed
  is the *shape* of the reference, and therefore the metric.
- **MAE is a regression metric, and it has the same blind spot correlation does.** A judge with a low
  MAE can still be systematically permissive, or fail badly on the extreme cases while averaging fine.
  Report the **error distribution**, not just the mean — which answers were missed by a lot is where
  the information is.

So the complete answer to "how do you validate a judge?" is a choice: **binary reference → TPR/TNR on
a held-out split; ordinal reference → MAE plus the error distribution; no reference at all → you cannot
validate against a reference, and you must fall back on human agreement on the criterion itself.**

**Red flags:** reporting a single agreement number; no held-out split; validating with 20 examples;
no re-validation plan; applying a classification protocol to an ordinal reference without noticing.

### Q6. What goes wrong with judges? Name the biases.

**What they're really testing:** whether you know the failure catalogue.

**Model answer:**
- **Positional bias** — in pairwise comparison, the judge prefers whichever option is presented first
  (or last). Mitigate by swapping the order and averaging.
- **Self-preference bias** — a judge rates outputs from its own model family higher. Mitigate by using
  a different family for judging than for generating where feasible.
- **Verbosity bias** — longer answers score higher regardless of quality. Mitigate by controlling for
  length or adding an explicit criterion.
- **Sycophancy / style bias** — confident, well-formatted prose outscores correct but plainly written
  answers.
- **Reference bias** — an unvalidated rubric in the system prompt steers scores more than the actual
  output does.
- **Saturation / central tendency** — the judge clusters around the middle of a scale.
- **The prompt-level control, which is separate from all of the above and is the one people forget.**
  Biases are mitigated *in the judge prompt itself* by naming the anti-patterns explicitly. The source's
  judging prompt carries this instruction, and it is close to a required line in any writing-quality
  judge:
  > *"For each dimension, decide whether the answer genuinely addresses it. **Do not reward verbosity,
  > keyword stuffing, and confident assertions that lack substantiation. Reward structure, relevant
  > examples, and balanced argumentation.**"*
  Without that pair of sentences a judge LLM rewards length and confident phrasing, and the resulting
  metric measures formatting rather than quality. Note the structure: **an explicit prohibition list
  paired with an explicit reward list.** A prompt that only says "score quality 0–10" leaves the judge
  to invent its own notion of quality, which is where every one of these biases comes from.
- The general mitigation is the same for all of them: **binary labels, few-shot critiques from real
  errors, explicit anti-patterns in the prompt, and validation with TPR/TNR on held-out data.**

**Red flags:** "LLMs are biased" with no specifics; no mitigation for positional bias; never having
heard of self-preference.

### Q7. Why does the judge's reasoning matter if you only use the verdict?

**What they're really testing:** whether you have ever debugged a judge.

**Model answer:**
- The aggregate number tells you *that* something changed; it never tells you *what* to fix.
- Requiring chain-of-thought reasoning **before** the verdict both improves the verdict's accuracy and
  gives you a per-case explanation.
- You read the reasoning on every failure — that is how you discover that the judge is failing on a
  particular length, language, or answer shape rather than on the criterion you think you are
  measuring.
- It is also how you build the **few-shot critiques**: the judge's own reasoning on a disputed case,
  corrected by a human, becomes a training example.
- The practical discipline: make reasoning mandatory in the output schema, and make reading it part of
  the workflow rather than optional. `include_reason=True` is not a nicety.
- This is also how you detect reward hacking — the reasoning shows the judge being satisfied by a
  proxy.

**Red flags:** scoring without reasoning; never reading failure cases; treating the judge as a
black box.

### Q8. Pointwise or pairwise judging?

**What they're really testing:** whether you know the reliability difference.

**Model answer:**
- **Pointwise** ("is this output good?") produces an absolute score. It is what you need for
  monitoring a single system over time, and for gating a release on a threshold.
- **Pairwise** ("which of these two is better?") produces a preference. It is substantially **more
  reliable** for subjective criteria — humans and judges both agree more on relative judgements than
  on absolute ones.
- Pairwise suits A/B comparison between two configurations or two models; it is the basis of win-rate
  and Elo-style leaderboards.
- Pairwise has two costs: it needs two candidates, and it is vulnerable to **positional bias**, so you
  swap the order and average.
- Pairwise results are also harder to threshold — "better than the old version" is a legitimate deploy
  criterion but a weaker one than "passes our absolute quality bar".
- The usual architecture: **pairwise for development and comparison, pointwise for monitoring and
  gating.**

**Red flags:** only knowing pointwise; no awareness of positional bias; using pairwise for a release
gate without an absolute bar.

### Q9. What is inter-annotator agreement and why does it matter for judges?

**What they're really testing:** whether you know your ground truth is also noisy.

**Model answer:**
- It is the degree to which two human labellers agree on the same item — commonly measured with
  **Cohen's κ**, which corrects for agreement expected by chance.
- It matters because the judge is validated *against humans*, so if the humans disagree with each
  other, the ceiling on the judge's achievable agreement is low and the label set is not ground truth.
- Low agreement is usually a **criterion problem**, not an annotator problem: the label definition is
  ambiguous. The fix is to rewrite the criterion until two people can apply it independently and
  agree.
- Report it. A judge that matches humans at 0.85 when the humans only agree with each other at 0.70 is
  performing near the ceiling, not badly.
- Practical discipline: write annotation guidelines, run a calibration round on a shared subset, and
  measure κ before trusting the labels.
- This is also the diagnosis path when a judge "disagrees with humans 30% of the time" — check human
  agreement first.

**Red flags:** treating human labels as noise-free; no guidelines; never measuring κ; immediately
retuning the judge without checking the criterion.

### Q10. What is the difference between offline and online evaluation, and why does it matter for judges?

**What they're really testing:** whether you have thought about judge deployment.

**Model answer:**
- **Offline** evaluation runs a frozen dataset before deploy. **Online** evaluation runs on live
  production traffic after deploy, as real users interact with it.
- The framing worth memorising, because it is exact: **offline checks whether the application works
  correctly; online tells you whether it is running normally.** They are complementary, not rivals.
- The reason they are different instruments: **online evaluation works without answers and without a
  golden dataset** — that is its defining feature. Its corollary is the hardest constraint in this
  area: **you cannot measure correctness online**, because no ground truth arrives with live traffic.
  What you can measure is whether today's *distribution* matches the baseline.
- The baseline itself comes from offline evaluation. That is the mechanism by which the two are
  complementary rather than duplicative: offline sets the reference line that online is judged
  against. Baseline 85 with an observed 87 is fine; 75 in the last 24 hours is an alert.
- They must use the **same evaluators**; if online uses a different judge prompt than offline, the
  offline number stops predicting production and the program loses credibility.
- The cost profile differs sharply: offline judging is bounded by your golden set, online judging
  scales with traffic — so online judges are **sampled**, and never randomly. Random sampling spends
  budget on conversations that were fine; **stratified sampling** buckets conversations by category
  and over-weights the problematic ones (thumbs-down, escalations, abrupt endings, repeated rephrased
  questions, money/refund topics). "Not all conversations are the same."
- Reference-based judges fit offline (you have the references); **reference-free** judges —
  faithfulness, answer relevance — are what you can actually run online.
- The loop between them is the point: production failures become golden-set rows. Offline eval →
  deploy → production failures → add those conversations to the dataset → re-run offline → deploy.
  That is the **self-improving loop**.

**Red flags:** "we evaluate before launch"; different judges online and offline; no sampling plan for
online judging cost; random sampling; alerting on a single conversation rather than a windowed
aggregate; treating an offline improvement as an unqualified win.

---

## Tier 2 — Applied / trade-off (10 Q)

### Q11. When should you *not* use a judge?

**What they're really testing:** restraint, which is rarer than enthusiasm.

**Model answer:**
- Whenever a **code check** can answer the question. Schema validity, required-field presence,
  citation present, tool arguments well-formed, database state, latency under budget, exact label
  match — all deterministic, free, and unit-testable.
- When the criterion is a **hard constraint** rather than a judgement: output length, forbidden
  strings, PII patterns.
- When you have **no labels and no plan to get them** — an unvalidated judge produces confident noise,
  which is worse than no metric.
- When the decision does not depend on it. If nothing changes when the score halves, the judge is
  cost without value.
- When the sample is too small to support a threshold above the noise floor.
- The framing to state explicitly: a judge is the **expensive, noisy, last-resort** instrument. Reach
  for code first, human labels second, judge third — and the human labels you do collect are what make
  the judge legitimate.

**Red flags:** "judges can evaluate anything"; no code-check reflex; judging without label plans.

### Q12. How many human labels do you need, and how do you get them cheaply?

**What they're really testing:** whether you understand the dominant cost.

**Model answer:**
- Roughly **100** for a judge to be validatable at all — enough to estimate TPR and TNR with usable
  confidence intervals and still hold some out.
- The binding constraint is the **failure class**, not the total: if the failure you are detecting
  occurs in 5% of traces, 100 random labels give you ~5 positives, which is not enough to estimate
  TPR. **Over-sample the failure class** deliberately for validation.
- Get them cheaply by: labelling only what the judge is uncertain about; using disagreement between
  two cheap judges to route items to humans; and reusing an **annotation queue** over sampled
  production traces rather than building a separate labelling exercise.
- Give annotators a crisp criterion plus a few worked critiques — vague guidelines produce
  low-agreement labels, which caps your judge's achievable performance.
- The output of labelling is not just labels: it is **written critiques**, which become the judge's
  few-shot examples. That is the highest-value form of the effort.
- Never synthetic-label the validation set — you would be validating the judge against a model.

**Red flags:** random sampling for a rare failure; no guidelines; synthetic labels for validation;
using 20 examples.

### Q13. A judge call costs $0.02 and you have 50,000 responses a day. What do you do?

**What they're really testing:** cost engineering under a real constraint.

**Model answer:**
- First do the arithmetic out loud: $0.02 × 50,000 = **$1,000/day** ≈ $30k/month just to observe. That
  is a product decision, not an infrastructure detail.
- Then attack it in order:
  - **Sample** — evaluate a few percent of traffic, chosen deterministically, plus a boosted sample of
    high-risk segments.
  - **Route** — a cheap model or a code check for the easy majority; the expensive judge only for
    ambiguous or high-stakes cases.
  - **Cache** — identical (prompt, judge, input) triples should never be judged twice.
  - **Shrink** — trim the judge prompt and cap the answer length sent to the judge.
  - **Tier** — heavy judging offline on the golden set; light, sampled judging online.
- Instrument the judge spend as its own cost line, otherwise it hides inside "infra".
- Do not reduce coverage of the failure you actually care about — reduce the cost per observation.

**Red flags:** "just judge everything"; no arithmetic; no routing or caching; treating judge cost as
infrastructure.

### Q14. Does judge temperature matter?

**What they're really testing:** whether you care about reproducibility.

**Model answer:**
- Yes — set it to **0**. Reproducibility is a property you need: if the same input yields a different
  verdict on a re-run, your metric moves for reasons unrelated to the system.
- Temperature 0 does not guarantee determinism at the provider level (batching, hardware,
  implementation details can still vary), so do not *rely* on bit-exact reproducibility.
- That is why the more robust design is to **constrain the output format** — a single letter, or a
  strict JSON schema — so small sampling differences do not change the parse.
- And why you should track the **noise floor**: run identical settings twice and measure the spread.
  That spread is what your threshold must exceed.
- For **G-Eval**, the design deliberately uses token probabilities rather than a sampled token, which
  is partly a stability choice.
- Pin and record the judge model version; a silent provider upgrade is a distribution change and
  invalidates re-validation.

**Red flags:** "temperature doesn't matter for judging"; relying on temperature 0 for exact
reproducibility; no noise-floor measurement.

### Q15. How do you make a judge's output parseable?

**What they're really testing:** engineering pragmatism.

**Model answer:**
- Constrain the format explicitly. The canonical three moves, taken verbatim from a production judge
  specification: **reason step by step** → **print only the letter of your answer on the final line** →
  **then repeat that letter**.
- The repetition is the trick: verbose models often append commentary, and having the verdict appear
  as the final token makes extraction robust.
- Modern equivalent: require **strict JSON** with a fixed schema and a constrained enum for the
  verdict, plus a `const true` style field that makes the schema self-describing.
- Always parse defensively — a malformed judge response is an error to be counted and surfaced, not
  silently coerced to PASS or FAIL. Coercion biases your metric in one direction.
- Log the raw judge output alongside the parsed verdict so a parse regression is diagnosable.
- And count parser failures as their own metric: a rising parse-failure rate usually means the judge
  model changed underneath you.

**Red flags:** free-text verdicts parsed with a regex on prose; silently defaulting unparseable output
to PASS; no raw-output logging.

### Q16. How do you judge a 3,000-word answer?

**What they're really testing:** whether you know the practical limits.

**Model answer:**
- Decompose before judging: split the answer into claims or sections and judge **per claim** rather
  than asking one judge for a holistic verdict. This is exactly how faithfulness works — decompose,
  then verify each claim against the context.
- This improves reliability (each judgement is small and local) and gives you partial credit and
  localisation.
- Truncation is a trap: truncating the answer before judging means silent bias toward whatever
  survives, and long answers are exactly where the failures hide.
- Verbosity bias compounds the problem — long answers score higher on holistic scales regardless of
  quality, so length control belongs in the criterion.
- Practically, aggregation matters: per-claim verdicts need a defined roll-up (all-claims-pass?
  fraction-passing?) and the choice should reflect the consequence — for safety, all-claims-pass;
  for quality, the fraction.
- If the criterion genuinely is holistic (coherence of the whole document), judge in stages:
  section-level first, then a document-level pass over the section summaries.

**Red flags:** truncating; one holistic verdict on a very long output; no aggregation rule.

### Q17. How do you compare two judge prompts?

**What they're really testing:** whether you can run a controlled experiment.

**Model answer:**
- Treat it as a **model comparison with the same protocol as judge validation**: the same labelled set,
  a held-out split, and TPR/TNR reported separately for each variant.
- Do **not** pick the winner by eyeballing a few outputs — judge prompts differ in ways that only show
  up on the tails.
- Fix everything else: same judge model, same temperature, same input formatting, same label
  distribution. Vary one thing.
- Report the **per-class breakdown**, not just aggregate TPR/TNR — one prompt may be better at
  catching the severe failure and worse at the mild one.
- Watch for prompt-length confounds: a longer, more detailed prompt often looks better on the
  examples it was written for and worse out of distribution. Test on held-out data.
- Record the prompt version with every run; a judge prompt is a deployable artefact and deserves
  versioning like one.

**Red flags:** "the new prompt looks better"; no held-out set; changing multiple things at once; no
prompt versioning.

### Q18. How do you judge multi-turn conversations?

**What they're really testing:** whether your model of judging scales past one-shot.

**Model answer:**
- The unit of evaluation becomes the **conversation**, not the turn — averaging per-turn scores hides
  exactly the failures that matter (contradicting an earlier turn, failing to carry context forward).
- Add conversational criteria: **context carry-over** (are references resolved correctly),
  **consistency** (does turn 4 contradict turn 2), **correction handling** (does it adapt when the
  user says "no, I meant X"), **information reuse** (does it avoid re-asking known facts).
- Reference-based judging gets harder: the same question has different correct answers at different
  points, so gold answers do not transfer. Prefer **outcome-based** grading where the task has a
  checkable end state.
- Practically, judge the **trajectory** — feed the judge the conversation with roles and turn indices,
  and ask for a verdict on a specific criterion rather than "how good was this".
- Sample whole conversations for human review; turn-level sampling systematically misses
  cross-turn failures.

**Red flags:** "same judge, looped over turns"; turn-level sampling; no consistency criterion.

### Q19. How do you judge in a language the judge is weak in?

**What they're really testing:** whether you treat language as a distribution.

**Model answer:**
- Test it before assuming: judge quality is not uniform across languages, and a criterion that
  validates well in English can have a materially lower TPR in another language. **Validate per
  language** — do not inherit the English TPR/TNR.
- Prefer a judge model with strong performance in the target language; if none is available, consider a
  translation-then-judge pipeline, accepting that translation is itself a lossy step.
- Beware **code-switching**: real user traffic mixes languages mid-utterance, which is where judges
  fail most. Include mixed-language examples in the validation set explicitly.
- Watch for **script and tokenisation effects** — the same content in a different script can tokenise
  several times more expensively and score differently.
- Culturally specific criteria (politeness, formality, humour) do not transfer; define them in terms of
  observable behaviour rather than an abstract quality.
- And budget accordingly: a multilingual judge fleet is more expensive than a single-language one.

**Red flags:** assuming English results transfer; no code-switched examples; no per-language validation.

### Q20. How do you detect that your judge has drifted?

**What they're really testing:** whether you treat your instrument as a system that can break.

**Model answer:**
- Treat the judge as a **production system with its own monitoring**: its outputs can change for
  reasons unrelated to the system under test.
- Track: the **pass rate over time** (a step change is the signal), the **parse-failure rate**, and
  the **score distribution** (a narrowing distribution often means the judge has collapsed to one
  verdict).
- Re-validate on a **frozen, labelled canary set** on a schedule. If TPR/TNR on the canary moves, the
  judge changed even though your prompt did not.
- Pin the judge model version and record it with every run; a provider-side silent upgrade is the most
  common cause.
- Keep a **reference judge output snapshot**: for a fixed set of inputs, store the verdicts. A diff on
  re-run with no code change is definitive evidence of drift.
- The trigger list for re-validation: judge model change, judge prompt change, system prompt change,
  data distribution change, and any significant shift in the pass rate.

**Red flags:** assuming the judge is stable; no canary set; not pinning the model version; no
snapshot diff.

---

## Tier 3 — Senior / staff (8 Q)

### Q21. Design the judge programme for a company shipping five LLM products.

**What they're really testing:** whether you can build shared infrastructure without over-centralising.

**Model answer:**
- **Centralise what is easy to get wrong and expensive to duplicate:**
  - the judge runtime (prompt versioning, model version pinning, caching, parse-failure accounting);
  - the **validation protocol** — a shared harness that takes a judge and a labelled set and returns
    TPR/TNR/FP/FN;
  - the **label store** and annotation tooling, because labelling is the expensive resource;
  - the **score contract**, so a score means the same thing across products.
- **Decentralise what is domain-specific:** failure taxonomies, criteria definitions, and golden
  datasets belong to the teams who know the product.
- Enforce one rule centrally: **no judge ships without a TPR/TNR number on held-out data.** That
  single gate prevents the most common and most damaging failure.
- Operate a **shared judge registry** with owner, version, validation numbers, and last-validated
  date — the same shape as a model registry.
- Budget a **central label pool**; teams will not fund labelling well but will consume it readily.
- Measure the programme on the **offline→online correlation** for shared metrics. That correlation is
  the programme's actual output.

**Red flags:** every team builds its own judge runtime; no validation gate; labels funded per-team so
nobody does it; no versioning of judges.

### Q22. A judge is used as an RL reward. What changes?

**What they're really testing:** whether you understand the training/eval boundary.

**Model answer:**
- Everything about the incentive changes: an eval measures, a reward **optimises**, and anything
  optimised will be **gamed**.
- A judge used as reward will be hacked — the policy finds the wording, length, or format the judge
  systematically over-rewards. This is not a hypothetical; it is the default outcome.
- Mitigations: prefer **verifiable rewards** wherever the task permits (deterministic, ungameable);
  keep the judge reward bounded and combine it with hard constraints; refresh the judge periodically
  so the exploit does not persist; hold out a judge the policy never sees.
- **Reward-model over-optimisation** is a real phenomenon: proxy reward rises while true quality falls
  after some point. Monitor a held-out, human-validated metric and stop when it turns.
- Any eval used as a training signal **stops being a valid held-out measurement** — you need a private
  set for reporting.
- Practically: log the reward distribution and watch for the signature of hacking — reward up, output
  diversity down, length up.

**Red flags:** treating a training reward like an eval metric; no held-out judge; never expecting
hacking; no diversity/length monitoring.

### Q23. How do you judge safety-critical criteria?

**What they're really testing:** whether you choose the right error trade-off.

**Model answer:**
- Choose the error asymmetry explicitly, and choose it toward **TPR**. For safety, a missed violation
  is usually far worse than a false alarm, so you accept a lower TNR.
- Report TPR and TNR separately and state the operating point: "at TPR 0.98 we get TNR 0.75" is a
  decision a reviewer can make; an aggregate F1 is not.
- Where possible, add **deterministic** detectors alongside the judge — pattern matches for PII,
  blocklists, classifier models trained specifically for the category. Judges are the flexible layer,
  not the only layer.
- Do not let a judge be the last line of defence for anything with legal or physical consequences;
  route those to human review and treat the judge as a **triage** system that decides *what* humans
  see.
- Expect adversarial inputs by design: a judge validated on organic traffic has not been validated on
  jailbreak attempts. Red-team the judge itself.
- Keep the safety judge's threshold separate from quality thresholds — they have different error costs
  and should not share a tuning pass.

**Red flags:** optimising for accuracy on safety; no deterministic backstop; no adversarial validation;
one threshold for all criteria.

### Q24. You have two judges that disagree on 20% of cases. Which do you trust?

**What they're really testing:** whether you can adjudicate between instruments.

**Model answer:**
- Neither by assertion — **measure both against human labels** on the same held-out set and compare
  TPR and TNR separately.
- Look at the **direction** of the disagreement. If one judge is systematically more permissive, that
  is a threshold difference; if the disagreement is scattered, it is a criterion-definition problem.
- Check whether the disagreements concentrate on a **subpopulation** — long answers, a language, a
  particular failure subtype. That tells you which judge is failing on what.
- Sample the disagreement set for **human adjudication** — that is where labels are most informative,
  because it is exactly the region where you do not know the answer.
- Consider **ensembling** deliberately: require both judges to agree for high-confidence verdicts, and
  route disagreements to humans. This gives you a calibrated triage rather than an arbitrary winner.
- Never resolve a judge disagreement by averaging scores — averaging two differently-biased measures
  produces a number that is not a measurement.

**Red flags:** picking the one that "looks right"; averaging the two; no human adjudication of the
disagreement set; no subpopulation analysis.

### Q25. How do you keep a judge from being gamed by the system it evaluates?

**What they're really testing:** adversarial thinking.

**Model answer:**
- Assume the incentive exists the moment anyone is measured on the judge's score, and that it exists
  automatically the moment it is a training signal.
- **Isolate**: the system under test must not see the judge's prompt or its few-shot examples, and
  must not be able to write to the judge's inputs.
- **Rotate and hold out**: keep judge prompts and a validation set the system never sees, so
  optimisation cannot target the exact instrument.
- **Prefer verifiable checks** for anything that can be checked — a deterministic assertion cannot be
  talked around.
- **Watch the signature**: rising judge score with falling output diversity, rising length, or rising
  format conformity is the classic hacking fingerprint.
- **Include adversarial examples in validation**: a judge validated only on organic traffic will have
  an optimistic TPR against deliberate manipulation.
- Log and audit the judge's inputs — a sudden shift in input distribution toward judge-pleasing
  patterns is the tell.

**Red flags:** no isolation between system and judge; a single fixed judge prompt used forever; no
diversity monitoring; validation only on organic data.

### Q26. How do you decide between improving the judge and improving the system?

**What they're really testing:** whether you can distinguish measurement error from real change.

**Model answer:**
- First establish **which one actually moved**. Improving the judge changes your *estimate*; improving
  the system changes the *thing being estimated*. Confusing the two wastes quarters.
- Diagnostic: if the human labels disagree with the judge on the failing cases, the judge is wrong. If
  humans agree with the judge, the system is genuinely failing.
- Look at **judge TPR/TNR on a canary set** — if those moved, the instrument drifted.
- Consider **measurement error vs real effect** in the delta: a 2-point move on a 15-row set is inside
  the noise floor and justifies investigation, not action.
- Improve the judge when the failure modes you care about are not represented, when TPR/TNR is below
  bar, or when the reasoning shows the judge is answering a different question.
- Improve the system when the judge is validated and the failures are real — then use error analysis to
  pick the cluster, not the loudest complaint.

**The procedure, in the order the source actually follows it.** This is the part most candidates cannot
describe, and it is a repeating four-step loop:

1. **Run** the metric and read the pass/fail counts.
2. **Read the failure reasons the judge cited**, for the failing test cases specifically — not a sample
   of all outputs, the failures.
3. **Identify the common pattern** across those reasons.
4. **Change exactly one of two knobs** — the **evaluator** (criteria, evaluation steps, scoring rubric)
   or the **generator** (the application's own system prompt) — then re-run.

The two knobs are the whole decision, and the source's three worked cases show that the same loop
points at different knobs each time:

| Metric | First run | Diagnosis from the failure reasons | Knob used | Second run |
|---|---|---|---|---|
| Correctness | 66% (8/15) | the judge penalised the generated answer for *omitting* points the ideal answer had | **Evaluator** — added "do not penalise omissions; only wrong statements count", plus a scoring rubric | **84%** (14/15) |
| Completeness | 68% (5/15) | the generator's prompt told it to answer *concisely from the context* | **Generator** — added "answer thoroughly, cover every distinct part" | **75%** (14/15) |
| Style | 54% | no guidance ever given on the brand voice, *and* the rubric over-demanded analogies | **Both** — generator prompt, plus a counter-line on the rubric | **74%** (9/15) |

- The lesson from that table: **a low score is not evidence that the system is broken.** Completeness
  went from 5/15 to 14/15 with no change to retrieval, to the model, or to the judge's criterion — only
  to the generator's prompt. If you "fix" the judge on that evidence you have fixed nothing.

**Two failure modes that only appear once you are inside this loop, and both are counter-intuitive:**

- **Over-correction — a rubric can be stricter than the requirement.** In the style case, the rubric
  said to reward analogies and examples, so the judge cited *"lacked analogies or examples"* as its
  reason on **every** low score. The requirement was never that every answer contain an analogy. The
  source names this explicitly as an over-correction, and the fix was a counter-line: *"an analogy or
  concrete example is a bonus when the concept is abstract, but a clear, direct, well-explained answer
  is fully acceptable."* The generalisable rule: **when a judge fails a case, read its stated reason as
  evidence about the metric, not only about the application.** A rubric rule that is stricter than the
  product requirement manufactures false failures, and false failures drive wrong fixes.
- **Metrics trade off, so a maximum on one is not the goal.** "If this metric starts to reach too high,
  other metrics will start taking a hit. **Faithfulness** and such will start taking a hit." Pushing
  style to its ceiling trades against groundedness; 74 was accepted as good. A candidate who treats
  every metric as monotone-improving has not shipped a multi-metric suite.

**One distinction that repeatedly causes misdiagnosis:** **correctness and faithfulness are different
metrics**, and conflating them makes the error analysis incoherent. Faithfulness means the answer is
**grounded in the retrieved context**; correctness means the answer is **factually right at the world
level**. There are four combinations, and each has a different fix: correct-and-faithful (ideal);
**correct but not faithful** (the generator ignored the context and answered from its own training
knowledge — a retrieval-adherence bug); **faithful but factually wrong** (the *source material* is
wrong and the model faithfully repeated it — a corpus bug, not a model bug); and both wrong. If your
suite has only one of the two, you cannot tell a corpus problem from a grounding problem.

**Red flags:** tuning the judge whenever scores are low; no canary set; no human adjudication; acting
on a delta inside the noise floor; not reading the judge's stated failure reasons; fixing the generator
when the evaluator was at fault (or vice versa); driving one metric to its maximum and degrading
another; treating correctness and faithfulness as the same thing.

### Q27. What is the biggest unsolved problem in LLM judging?

**What they're really testing:** whether you have a considered view rather than a vendor's.

**Model answer:**
- **Validation at scale.** The protocol is known — held-out human labels, TPR/TNR — but it is
  labour-bound, and the field has settled on a practice where judges ship unvalidated and are trusted
  anyway. That gap between what is known and what is done is the real problem.
- Related: **we have no accepted standard for reporting judge quality**, so two teams' "faithfulness
  0.92" are not comparable quantities.
- **Distribution shift** is structurally unsolved: a judge validated on last quarter's traffic is
  being applied to this quarter's, and re-validation is expensive and rarely scheduled.
- **Correlated failure** is the deepest issue: when the judge and the system share a model family, or
  share training data with the benchmark, agreement can look like correctness while both are wrong in
  the same way.
- The direction that has the most promise is **verifiable rewards** taking over wherever the task
  permits, leaving the judge to the genuinely subjective residue.
- A mature view acknowledges what this means in practice: judge numbers should be reported with their
  validation status attached, as an estimate with error bars, not as a fact.

**Red flags:** claiming judging is solved; naming only a bias; no view on what would fix it; treating
the vendor's metric as the state of the art.

### Q28. How would you evaluate an eval? (Meta-evaluation)

**What they're really testing:** whether you can go one level up.

**Model answer:**
- The eval system is a system, so it has the same properties: it can be wrong, stale, expensive, and
  gameable.
- **Accuracy**: judge TPR/TNR on a held-out set, re-measured on a schedule. This is the primary
  measure and it is the one everyone skips.
- **Stability**: the noise floor — run identical settings twice and measure the spread. If the spread
  approaches your threshold, the eval cannot support the decisions being made on it.
- **Discriminative power**: does the pass rate actually vary across system changes? A suite whose pass
  rate never moves has stopped carrying information.
- **Predictive validity**: the correlation between offline and online for the same metric. This is the
  eval programme's real output and almost nobody measures it.
- **Cost**: per-run spend split into generation and grading, tracked as a first-class number.
- **Coverage**: what fraction of your observed production failure modes the suite can detect at all.
  Coverage gaps are invisible unless you deliberately audit them — take last month's incidents and ask
  which the suite would have caught.

**Red flags:** never having considered evaluating the eval; no offline/online correlation; no coverage
audit; no cost tracking for the harness.

---

## Tier 4 — Debug-this-scenario (5 Q)

### Q29. Your judge agrees with humans 95% of the time but ships regressions constantly. Diagnose.

**What they're really testing:** whether you know why agreement is the wrong statistic.

**Model answer:**
- Almost certainly an **imbalanced label distribution**. If 95% of traces genuinely pass, a judge that
  always says PASS scores 95% agreement while catching **zero** failures.
- Compute the confusion matrix with **positive = FAIL** and report **TPR** and **TNR** separately. The
  TPR will be near zero.
- Corroborating evidence from the field: binary judges reached **>95% precision on *consistent*
  summaries but only ~30–60% recall on *inconsistent* ones** — precision looks excellent while the
  false-negative blind spot is enormous.
- Fix: collect more **positives** deliberately (over-sample the failure class), rewrite the judge with
  few-shot critiques drawn from real failures, and re-validate on a held-out split.
- Also fix the gate: a threshold on raw agreement cannot catch regressions. Gate on TPR-validated
  metrics and on a set that contains a realistic proportion of failures.

**Red flags:** adding more examples of the same kind; tuning the threshold; not computing TPR.

### Q30. Judge scores dropped 8 points overnight with no code change. Diagnose.

**What they're really testing:** whether you separate instrument drift from system change.

**Model answer:**
- First ask: **did the instrument move, or the system?** Check the judge model version — a silent
  provider upgrade is the most common cause of an overnight shift.
- Check the judge prompt version and any shared prompt registry — a prompt change by another team is a
  system change to your eval.
- Check the **input distribution**: a traffic shift toward a query class the system handles badly will
  depress the score with no code change.
- Check the **parse-failure rate**: if the judge model changed its output format, parses fail and
  scores collapse. Unparseable output silently coerced to FAIL is a common bug.
- Run the **canary set** — a frozen set of inputs with stored verdicts. If the verdicts changed, it is
  the judge. If they are stable, it is the inputs.
- Check whether an upstream component changed: corpus refresh, index rebuild, retriever config.

**Red flags:** assuming the system regressed; not checking the model version; no canary set; not
checking parse failures.

### Q31. The judge loves outputs from one model family. Diagnose and fix.

**What they're really testing:** self-preference bias, and whether you have a mitigation.

**Model answer:**
- This is **self-preference bias**: a judge rates outputs from its own family higher, because it
  recognises its own stylistic and formatting conventions as "correct".
- Diagnose by measuring TPR/TNR **split by the generator's family**. A gap in TNR (clearing its own
  family's failures more readily) confirms it.
- Fixes, in order of preference:
  1. **Judge with a different family** than the one generating, where feasible.
  2. **Blind the origin** — strip identifying stylistic markers, normalise formatting, randomise
     presentation order.
  3. **Calibrate per family** — if you must use one judge, validate and report TPR/TNR per source
     model rather than pooled.
- Correlated failure is the deeper risk: if generator and judge share blind spots, their agreement
  looks like correctness. That argues for keeping at least one **verifiable** metric independent of any
  judge.
- Note this also affects **leaderboards**: a judge comparing model outputs can systematically favour
  its own family, which is why harness and judge identity belong in the reporting.

**Red flags:** ignoring the possibility; pooling TPR/TNR across families; no independent verifiable
metric.

### Q32. Your G-Eval scores swing ±0.15 between identical runs. Diagnose.

**What they're really testing:** whether you understand the noise floor and how to reduce it.

**Model answer:**
- First: **is that swing actually large?** Compute the noise floor properly by running the identical
  configuration several times and looking at the spread. If ±0.15 is the floor, your threshold cannot
  be finer than that, and any comparison of two configurations is uninterpretable until you reduce it.
- Causes of high variance in graded judging:
  - **Sampled tokens** rather than probability weighting — G-Eval's design uses token probabilities
    precisely to reduce this, so verify the implementation actually does.
  - **A criterion that is too broad** — "quality" decomposes differently each run. Narrow the criterion
    until the evaluation steps are stable.
  - **Too few examples** — variance scales with 1/√n; if the set is small, the swing is expected.
  - **Non-zero temperature** — set it to 0.
- The structural fix: **use binary labels for anything that will gate a decision**. Graded scores are
  for exploration; gates need crisp decisions and classification metrics.
- Do not tune the threshold to absorb the variance — that just moves the flapping.

**Red flags:** comparing configurations on a noisy metric; widening the threshold; not measuring the
noise floor.

### Q33. The judge disagrees with humans on the longest 10% of answers. Diagnose.

**What they're really testing:** whether you can find a length confound.

**Model answer:**
- Two likely causes, and they need different fixes. **Cause 1: verbosity bias** — the judge
  systematically over-rewards long answers, so on long outputs it diverges from humans who are
  penalising padding.
- **Cause 2: truncation** — the judge is only seeing the first N tokens of the answer, so it is
  judging a different document than the human did. This is common and usually invisible.
- Diagnose by checking: what is the judge's input length limit versus the actual length distribution?
  And is the divergence concentrated at the *top* of the length distribution (truncation) or spread
  across it (verbosity bias)?
- Fixes: raise or remove the judge's input cap and **decompose long answers into claims** judged
  separately; neutralise verbosity by controlling for length explicitly in the criterion, or by
  normalising answer length in the comparison.
- Validate on a **length-stratified** held-out set, not a random one — random sampling under-covers the
  tails where the disagreement lives.
- Longer term, if the product genuinely rewards concision, make that a separate criterion rather than
  hoping a quality judge absorbs it.

**Red flags:** ignoring length; not knowing the judge's input cap; random validation sampling; one
pooled TPR.

---

## Tier 5 — Trap questions & the naive-answer trap (5 Q)

### Q34. *(Trap)* "We'll use the strongest model as the judge and skip validation — it's smarter than our annotators."

**The naive answer:** agree — a frontier model should beat crowd annotators.

**What they're really testing:** whether you understand that capability is not calibration.

**Model answer:**
- Capability and **agreement with your definition** are different things. A smarter judge can be
  confidently wrong about *your* criterion — it will apply its own notion of quality, which may differ
  from your product's.
- Validation is not "checking the model is smart"; it is measuring whether it answers **your** question
  the same way your humans do. There is no substitute, regardless of model strength.
- The specific danger is asymmetric error: a strong judge can post excellent agreement while missing
  most of the failure class you built it to catch. Raw agreement hides this.
- Validation is cheap relative to the decision: ~100 human-labelled examples and one held-out split,
  reported as TPR/TNR. That is hours, not weeks.
- Strong models *do* reduce how much few-shot tuning you need — that is a real benefit. It does not
  remove the need to measure.

**Red flag:** accepting "it's smarter, so it's right".

### Q35. *(Trap)* "We'll use a 1–10 scale so we can track fine-grained improvement."

**The naive answer:** sounds rigorous — more resolution must be better.

**What they're really testing:** whether you know resolution is not accuracy.

**Model answer:**
- A 1–10 scale gives you **resolution without reliability**. The gap between a 6 and a 7 is noise, so
  you have manufactured precision that does not correspond to a difference in the world.
- You also lose your statistical toolkit: classification metrics (TPR/TNR) need a binary decision. With
  a continuous score you have no error model and no validated operating point.
- Inter-annotator agreement collapses on wide scales — humans disagree about 6 vs 7 far more than
  about pass/fail, which caps the judge you can validate against.
- If you need granularity, get it from **more binary judges with narrower criteria** — grounded,
  complete, relevant, on-brand — which gives a diagnostic profile rather than one uninterpretable
  average.
- If you need to detect fine-grained improvement, detect it by **effect size on a metric with a known
  noise floor**, or by pairwise comparison against the previous version, not by widening the scale.

**Red flag:** accepting "more resolution is better" without a noise-floor argument.

### Q36. *(Trap)* "Our judge correlates 0.85 with human scores — we're good."

**The naive answer:** 0.85 correlation sounds strong, so ship.

**What they're really testing:** whether you know what correlation hides.

**Model answer:**
- Correlation measures **whether the ordering agrees on average**; it says nothing about whether the
  judge catches the failures you built it for.
- A judge can correlate 0.85 overall and still have a **TPR of 0.3 on the failure class** — because the
  failure class is rare and contributes little to the correlation.
- Correlation is also sensitive to the sample's composition. A dataset with more variance inflates it.
- Report **TPR and TNR separately with raw FP/FN counts**, which are the numbers a decision can use.
- The related trap is **Spearman vs Pearson**: rank correlation says the ordering agrees, which is
  better than nothing, but a judge could rank perfectly and still place the wrong cases above the
  threshold. Threshold behaviour is what a gate depends on.
- Always report the correlation *and* the confusion matrix.

**Red flag:** treating a correlation coefficient as sufficient validation.

### Q37. *(Trap)* "Let's judge 100% of production traffic so we never miss anything."

**The naive answer:** maximum coverage must be the safest choice.

**What they're really testing:** whether you can reason about eval economics.

**Model answer:**
- Do the arithmetic before agreeing: at $0.02 per judge call and 50,000 responses a day, that is
  **$1,000/day** — roughly $30k/month to *observe*. That is a product decision, not a configuration
  choice.
- Full coverage does not buy full detection anyway: you would still need someone to look at the
  failures, so the bottleneck just moves.
- The correct design is **targeted sampling**: deterministic sampling for a baseline, plus boosted
  sampling of high-risk segments (long conversations, low-confidence, novel query clusters,
  thumbs-down, agent trajectories that errored or retried).
- Route cheaply: code checks and cheap models on everything, the expensive judge on the ambiguous
  minority.
- Spend the saved budget on **human labels**, which are what make every judge number meaningful — that
  is a strictly better use of the same money.
- Track judge spend as its own cost line so this trade-off is visible rather than discovered.

**Red flag:** accepting full coverage without doing the arithmetic.

### Q38. *(Trap)* "Temperature is 0, so our judge is deterministic and we don't need a noise floor."

**The naive answer:** temperature 0 guarantees reproducibility, so the measurement is exact.

**What they're really testing:** whether you know that determinism is not guaranteed end-to-end.

**Model answer:**
- Temperature 0 is not a determinism guarantee. Provider-side batching, hardware non-determinism, and
  implementation details can still produce different outputs for the same input.
- Even with a perfectly deterministic judge, the **system under test** is usually not deterministic —
  and the metric's variance comes from there as much as from the judge.
- The noise floor is what tells you whether a delta is real, and it must be measured, not assumed:
  run identical settings twice and look at the spread of the metric.
- That spread sets your minimum detectable effect and therefore your CI threshold. A threshold below
  it produces flapping builds, which is how eval suites get disabled.
- Determinism is still worth pursuing — set temperature 0, constrain the output format, and pin model
  versions — but pursue it to *reduce* variance, not to assume it away.
- Practical rule: any eval that gates a deploy needs a measured noise floor on the record.

**Red flag:** asserting determinism from temperature alone.

---

## Live-coding / whiteboard prompts (3)

### Prompt 1 — Implement judge validation (45 min)

> Given `labeled = [(input, output, human_verdict), ...]` where `human_verdict` is `"PASS"` or
> `"FAIL"`, write the function that reports the judge's quality. Then explain what you would do if
> TPR came back at 0.4 and TNR at 0.97.

**What a strong answer covers:** the confusion matrix with **positive = FAIL**; TPR and TNR computed
separately; raw FP and FN returned alongside; explicit handling of the degenerate cases (zero
positives or zero negatives) rather than dividing by zero; a held-out split so the numbers are not
fitted; and for the follow-up — recognise 0.4/0.97 as the classic **rubber-stamp** signature, diagnose
it as an imbalanced set with too few positives, and fix it by over-sampling the failure class,
rewriting the judge with few-shot critiques from real failures, and re-validating. The key judgement
call: do **not** tune the threshold to improve the numbers.

**Red flags:** returning a single accuracy value; no degenerate-case handling; proposing to lower the
threshold rather than improve the judge.

### Prompt 2 — Design a judge for a named failure mode (60 min)

> Production traces show the assistant inventing policy details that are not in the retrieved context.
> Write the judge prompt and the validation plan.

**What a strong answer covers:** a **binary PASS/FAIL** definition with the failure stated concretely;
chain-of-thought reasoning emitted **before** the verdict; a constrained final output (single token,
repeated, or strict JSON with an enum); **4–8 few-shot examples given as written critiques**, not a
long rubric; ~100 human labels with deliberate **over-sampling of the failure class**; a fit/held-out
split; TPR and TNR reported separately on the held-out set with a ship bar around 0.9 on both; and a
re-validation trigger list (model change, prompt change, distribution change).

**Red flags:** a Likert scale; a rubric-stuffed system prompt with no critiques; validation on the
few-shot examples themselves; a single accuracy number; no re-validation plan.

### Prompt 3 — Choose the evaluation method (30 min)

> For each, choose reference-based, reference-free, code-based, or human — and say why: (a) is the
> response grounded in the retrieved documents? (b) did the agent call the right tool with valid
> arguments? (c) is this summary better than that one? (d) is the answer factually correct against our
> knowledge base? (e) is the tone appropriate for a bereavement-support context?

**What a strong answer covers:**
- (a) **reference-free judge** — answer vs context; no gold answer needed; runs online at scale.
- (b) **code-based** — tool name and JSON-schema validation are deterministic; a judge here is waste.
- (c) **pairwise human, or a validated pairwise judge** — relative judgement is more reliable than
  absolute, and this is a comparison task by construction.
- (d) **reference-based** — needs a gold answer or a verifiable source; correctness is not
  determinable from the context alone.
- (e) **human labels first, then a validated judge** — the criterion must be made observable
  ("acknowledges the loss before offering information") or it cannot be validated at all.
- The umbrella rule stated explicitly: **code → reference-free judge → reference-based judge → human**,
  ordered by cost, with the caveat that human labels are still required to legitimise the judges.
- And the classification test applied out loud, because it is what the question is really checking:
  **"does the golden dataset hand you a correct answer?"** — (d) yes → reference-based; (a) and (e) no →
  reference-free. A candidate who reaches for "it has a rubric, so it's reference-based" on (e) has
  failed the question: **a rubric is a scale, not a correct answer.**
- A strong candidate also notes the fourth method hiding in the list: **human-in-the-loop** for the
  grey areas, which is not a separate method but the human method invoked as a fallback.

**Red flags:** a judge for (b); an absolute score for (c); reference-free judging for (d); no concrete
criterion for (e); claiming a rubric makes an evaluation reference-based.

---

## Take-home / case-study prompt (1 full brief)

> **Brief.** A team ships an AI writing assistant for internal communications. They currently have one
> LLM judge — a 1–5 "quality" score using a long rubric in the system prompt, unvalidated, running on
> 100% of 30,000 daily generations with a frontier model. The score has been 4.1 for four months. Two
> teams have independently concluded the assistant "is fine", and the eval budget is being questioned.
>
> **Deliverables (≤6 pages):**
> 1. **Diagnosis.** Name the specific defects in the current setup and, for each, the decision it is
>    corrupting. Be precise about why a static 4.1 is a warning rather than a result.
> 2. **Error analysis plan.** How you would get from 30,000 daily traces to a ranked list of named
>    failure modes, including the sampling strategy and why uniform random sampling fails here.
> 3. **Judge redesign.** The label definition, the reasoning requirement, the output format, and the
>    few-shot strategy — plus what you would do with the existing rubric.
> 4. **The validation protocol.** Label volume, how you handle a rare failure class, the split
>    strategy, the two statistics you report and why not accuracy, and the ship criterion.
> 5. **The economics.** The current monthly judge spend, the redesigned spend, and what you do with the
>    difference. Show the arithmetic.
> 6. **Ongoing governance.** Model-version pinning, the canary set, re-validation triggers, and how you
>    would detect judge drift.
> 7. **What you tell the two teams** who concluded the assistant is fine — in plain language, with the
>    decision they own.
>
> **What is being assessed:** whether you treat the judge as a classifier with a validation protocol;
> whether the redesign is driven by observed failure modes rather than metric aesthetics; whether the
> economics are done with arithmetic; and whether the plan includes governance rather than ending at
> "validate once".

---

## Scoring rubric — what separates a hire from a no-hire

| Dimension | No-hire | Mid-level hire | Senior hire |
|---|---|---|---|
| **Taxonomy** | Judges everything | Knows a code check exists for some things | Names the three methods, explains *why* a metric cannot be counted (answer-level properties, the analogy problem), and matches the metric shape to the reference — TPR/TNR for a binary one, MAE for an ordinal one |
| **Label design** | 1–5 Likert scale | Binary PASS/FAIL | Binary, one judge per failure mode, few-shot critiques from real errors |
| **Validation** | "We spot-check outputs" | Knows TPR/TNR exist | Reports both on held-out data with FP/FN, over-samples rare failures, states a ship bar and re-validation triggers |
| **Bias awareness** | "LLMs are biased" | Names one bias | Names positional, self-preference and verbosity, with a specific mitigation for each |
| **Method selection** | Judges everything | Knows code checks exist | Applies code → judge → human by cost, and will argue *against* a judge |
| **Reference discipline** | Conflates the two | Knows the distinction | Places each metric correctly and knows which can run online at scale |
| **Cost awareness** | No cost dimension | Knows judging costs money | Does the arithmetic, routes and samples deliberately, and spends the saving on labels |
| **Reproducibility** | "Temperature 0, so it's deterministic" | Sets temperature 0 | Measures the noise floor, constrains the output format, pins versions |
| **Governance** | Validate once | Mentions re-validation | Canary set, version pinning, drift detection, and a judge registry with owners |
| **Adversarial thinking** | Assumes good faith | Knows reward hacking exists | Designs against gaming, correlated failure, and over-optimisation, unprompted |
| **Meta-awareness** | Never questions the judge | Knows judges can be wrong | Can evaluate the eval: stability, discriminative power, and offline→online correlation |

**The single strongest signal in this domain:** the candidate volunteers that an unvalidated judge
produces **confidence rather than information**, and can say precisely what goes wrong when a team
makes decisions on one — asymmetric error, a green dashboard, and failures passing straight through.
