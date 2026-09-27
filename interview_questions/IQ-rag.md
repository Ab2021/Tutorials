# Interview Pack · RAG Evaluation

> **Covers:** CS-13 (retriever metrics hands-on), CS-14 (three-level eval suite), CS-15 (operational
> evals), CS-16 (RAG safety) · **Role level:** mid → senior (AI/ML engineer, applied scientist,
> LLM platform engineer) · **Format:** phone screen → onsite → system-design → take-home

---

## How this domain is assessed

"How do you evaluate your RAG app?" is asked in roughly **8 of 10 GenAI interviews** `[5:46]`, and
the source's blunt diagnosis is that most candidates cannot answer it: either they never studied
evaluation, or they *have* and the answer comes out as a flat list of metric names with no structure
`[5:49]`–`[6:00]`, `[45:25]`–`[45:35]`.

The interviewer is not testing whether you can name metrics. Metric names are Googleable in thirty
seconds. **They are testing whether you can place a metric in a structure** — because that is what
predicts whether you will be able to localise a regression at 2 a.m. two quarters from now.

What the interviewer is actually scoring, in order of weight:

| What is being scored | The signal | How it is elicited |
|---|---|---|
| **Structural thinking** | Do you name the *level* before the metric? | The opening question, verbatim |
| **Failure localisation** | Can you say whether a bad answer was retrieval or generation? | Follow-up: "correctness dropped 5 points — now what?" |
| **Statistical honesty** | Do you know what your sample size supports? | Trap: "your P99 is 5.3 s from 25 samples" |
| **Economic awareness** | Do you know what a quality gain costs? | "It's better but 2× slower — ship it?" |
| **Production instinct** | Does your answer end at deployment or begin there? | "What happens after you ship?" |
| **Judgement about tooling** | Can you describe the concept without a product name? | "Describe experiment tracking without saying MLflow" |

**The single highest-leverage move in the whole interview** is the opening sentence. Say *"I build an
eval suite and I evaluate at three levels"* — then walk the levels in build order. That one sentence
converts a scattered answer into a framework the interviewer can follow, and the source is explicit
that a candidate who answers this way is judged to know the material "deeply" `[45:39]`–`[45:46]`.

---

## Tier 1 — Fundamentals (10 Q)

### Q1. How do you evaluate your RAG app?

**What they're really testing:** whether you have a *structure* or a *list*.

**Model answer:**
- I build an **eval suite** — multiple evaluation files run together as one testing suite.
- I evaluate at **three levels, in build order**: component, pipeline, application.
- **Component:** retriever alone on **recall + precision**; generator alone on **faithfulness, answer
  relevance, citation accuracy** — with a hand-fed question and hand-fed context, so it is a unit test.
- **Pipeline:** the **RAG Triad** — context relevance, faithfulness, answer relevance.
- **Application:** correctness, completeness, style — then **safety** (toxicity, PII, jailbreak) and
  **ops** (latency, cost per query, token cost).
- Then I run the whole suite as **regression testing**, gate it in **CI** on a threshold against
  baseline, and continue **online evaluation** with drift detection after deploy.

**Red flags:** a flat list of metric names with no levels; starting with "we use RAGAS"; never
mentioning deployment; treating "evaluation" as a one-off pre-launch activity.

### Q2. Why evaluate the retriever and the generator separately?

**What they're really testing:** whether you understand failure localisation.

**Model answer:**
- End-to-end only tells you the answer was bad — not *why*.
- A bad answer has two very different causes: the retriever never surfaced the right context, or the
  generator misused good context.
- Evaluating them in isolation makes each failure unambiguous, because in each case you control the
  other half.
- It is also cheaper: a retriever problem does not require you to re-run generator judges.
- Fix cost differs by an order of magnitude — chunking/embedding changes are cheap; a generation
  problem may mean a model change.
- This follows the software-engineering norm: you test components at the component level, not only
  the assembled application.

**Red flags:** "we just evaluate end-to-end, it's simpler"; not being able to name what a retrieval
failure looks like versus a generation failure.

### Q3. Name the RAG Triad and the pair each metric checks.

**What they're really testing:** precision of recall on the canonical framework.

**Model answer:**
- **Context relevance** — question ↔ retrieved context. Is what we retrieved on-topic?
- **Faithfulness** — answer ↔ context. Did the answer come from the context, or is it hallucinated?
- **Answer relevance** — answer ↔ question. Does the answer actually address what was asked?
- They exist *as a triad* because a RAG pipeline contains exactly three entities — question, context,
  answer — and therefore exactly three pairwise relationships to check.
- All three are LLM-judged, so the triad costs three judge calls per query.

**Red flags:** naming only two; describing faithfulness as "answer ↔ question" (that is answer
relevance); not knowing why there are three.

### Q4. What is faithfulness?

**What they're really testing:** whether you conflate grounding with correctness.

**Model answer:**
- Faithfulness asks whether every claim in the answer is **supported by the retrieved context**.
- It is *not* the same as correctness. An answer can be perfectly faithful and still wrong, if the
  retrieved context was wrong or the corpus is outdated.
- It is the metric that catches hallucination — the model inventing appointment times, figures or
  policy details that were never in context.
- Computed by an LLM judge that decomposes the answer into claims and checks each against the context.
- It is the **most commonly monitored online metric**, because it degrades when the world changes
  under a static corpus.

**Red flags:** "faithfulness means the answer is correct"; not distinguishing it from correctness.

### Q5. How do you build a golden dataset for RAG evaluation?

**What they're really testing:** whether you know that labels are the expensive part, and where the
traps are.

**Model answer:**
- Four routes, roughly in order of quality-per-effort:
  1. **Hand-author** — best quality, not scalable.
  2. **LLM-draft + mandatory human review** — the practical balance point, and what the source used.
  3. **A library synthesizer** (e.g. DeepEval's `Synthesizer`) — fully automated, but it produced
     academic, over-optimised questions no real user would ask about that corpus.
  4. **Harvest production logs with positive feedback** — realistic, but cannot bootstrap a project.
- **Start with (2), graduate to (4) once you have traffic.**
- The row should be `question | ideal_answer`, **not** `question | chunk_id`.
- Write the ideal answer as a **composition of atomic claims, deliberately** — the judge must
  decompose it, and a dense paragraph decomposes unstably run to run.
- Size: 15 rows detects large moves only; treat 1–2 point deltas as noise until repeat runs of
  identical settings agree.

**Red flags:** "we use the chunk IDs"; proposing a synthesizer and stopping there; no mention of
human review; claiming a 15-row set supports fine-grained tuning.

### Q6. Why should the ideal answer be written atomically?

**What they're really testing:** whether you understand that judge stability depends on input shape.

**Model answer:**
- Contextual recall works by having a judge **decompose the ideal answer into atomic claims**, then
  asking which claims appear in each retrieved chunk.
- If the ideal answer is one dense paragraph, the decomposition varies between runs, so scores are not
  comparable across runs and you cannot tell a real improvement from judge noise.
- Writing it as discrete claims is a **deliberate design choice**, not stylistic preference — you are
  designing the input for the judge's parser.
- The same discipline applies to judge prompts generally: constrain the output format so the result is
  machine-comparable.

**Red flags:** treating the ideal answer as prose; not knowing that recall is computed by claim
decomposition at all.

### Q7. What is contextual precision and how does it differ from precision?

**What they're really testing:** rank awareness.

**Model answer:**
- **Plain precision** is `correct_retrieved / all_retrieved` — a set statistic that ignores order.
- **Contextual precision** is **rank-aware**: it averages the prefix precision at each rank, so a
  useful chunk at position 1 scores very differently from the same chunk at position 5.
- Concretely, both `✓✓✗✗✗` and `✗✗✗✓✓` have plain precision 2/5 = 0.4, but prefix precision gives
  **0.713** versus **0.130**.
- Rank matters because the generator weights early context more, and because attention degrades over
  long contexts.

**Red flags:** reciting one precision formula and stopping; assuming set-based precision is sufficient.

### Q8. What operational metrics matter for a RAG app?

**What they're really testing:** whether you know evaluation does not end at quality.

**Model answer:**
- Three dimensions: **latency, cost, reliability**.
- Latency: mean, median, P50/P95/P99, min, max — never mean alone — plus **TTFT**, which is what the
  user actually experiences.
- Cost: **cost per query** (not per month), split into input and output tokens, measured as a
  distribution so you can see the tail.
- Reliability: **success rate, error rate, timeout rate, retry rate** — four separate numbers, broken
  down by cause (LLM API, retriever, reranker, timeout, rate limit, parser, internal exception).
- Crucially, **nothing here needs an LLM judge or a golden dataset** — a stopwatch, a token counter
  and a success/failure flag. These scripts are **free to run**, so they can gate every commit.
- **Throughput is out of scope** for offline evals — it needs load and stress testing.

**Red flags:** "we monitor latency"; quoting a single average; no mention of cost per query; thinking
ops evals need a judge.

### Q9. What is a trace, and why does it matter?

**What they're really testing:** whether you understand the bridge from offline to online evaluation.

**Model answer:**
- A trace is an instrumented record of one interaction — the question, retrieved documents, the
  answer, and the operational facts: latency, cost, token count, and user feedback (thumbs up/down).
- Tracing is what makes **online evaluation** possible at all: without it you have no data on the
  deployed system.
- It also feeds the offline loop — a production failure becomes a golden-dataset row.
- Platforms named in the source: LangSmith, LangFuse, Confident AI.
- The design principle: **a score is only actionable if it sits next to the latency and cost of the
  same interaction.** Otherwise you cannot tell whether a quality gain was worth its price.

**Red flags:** describing traces as "logging"; not connecting traces to offline dataset growth.

### Q10. What is drift in a RAG app?

**What they're really testing:** whether you think about the system after launch.

**Model answer:**
- Drift is a deployed app's metrics degrading relative to the world it operates in.
- The source's concrete mechanism: keep a **24-hour graph** of one online metric — say faithfulness —
  and if it drops over the **last 8 hours**, treat it as drift.
- Respond with an alert plus a remediation path, not just a dashboard.
- Common causes: corpus staleness, a provider model change, shifted user query distribution, an
  upstream data-pipeline break.
- Drift is undetectable without tracing, which is why online evaluation is not optional.

**Red flags:** confusing drift with variance; no alerting mechanism; "we'd notice from user
complaints."

---

## Tier 2 — Applied / trade-off (10 Q)

### Q11. How do you pick the CI threshold for a deploy gate?

**What they're really testing:** whether you know a threshold is a statistical decision.

**Model answer:**
- The source gives **3 units** as its example: if a metric is more than 3 units below baseline, block
  the deploy.
- But the *number* is a judgement, not a constant. It must be set per metric, against that metric's
  run-to-run variance.
- If a golden set is small, run the **same configuration twice** and see how much the score moves;
  that spread is your noise floor, and a threshold below it produces flapping builds.
- Scale matters: 3 units on a 0–100 scale is a very different rule than 3 units on a 0–1 scale.
- Prefer a **relative** threshold (e.g. no metric may drop more than 2% relative) when metrics have
  different scales.
- Always gate on the **delta vs baseline**, never the absolute value — absolute thresholds reward
  gaming and punish legitimate model migrations.

**Red flags:** quoting 3 without qualification; no notion of noise floor; gating on absolute scores.

### Q12. A change improves correctness and faithfulness but doubles latency and cost. Do you ship?

**What they're really testing:** whether quality is your only axis.

**Model answer:**
- Read both ledgers, then ask which constraint is **hard**.
- In the source's example the change took correctness 91 → 95 and faithfulness 94 → 96, but mean
  latency 2.3 s → 4.1 s and cost 72 paise → 1.08.
- If there is a per-query cost budget or a P95 latency ceiling, and the change breaches it, the
  quality gain does **not** earn a deploy — you go back to the reduction menu for whichever dimension
  broke.
- The correct answer is never "quality wins" unilaterally. It depends on whether the constraint is a
  business constraint or a preference.
- If neither dimension is hard-constrained, shipping a quality gain is usually right — but ship it
  behind a flag and keep the old config as a rollback.
- Segment the cost: if the doubling is concentrated in the long-context query class, you can route
  only that class to the expensive path.

**Red flags:** "ship it, quality is what matters"; "don't ship, latency is what matters"; not asking
whether there is a budget.

### Q13. Which retrieval lever should you pull first?

**What they're really testing:** whether you have empirical intuition about RAG knobs.

**Model answer:**
- From the source's measurements, ranked by observed effect:

| Lever | Effect | Requires re-embed? |
|---|---|---|
| **Chunk geometry** (750/100 → 1000/150) | recall 80 → **97** — the largest single gain | Yes |
| **Embedding model** (small → large) | recall 92 → **99**, precision flat | Yes |
| **Reranker** | precision 83 → 85, but **cost ~5 points of recall** | No |
| **k** (5 → 3) | lost ~1 point (noise at n = 15) | No |

- So: **chunking first, then embedding model, then a reranker** — and **k last**, because recall and
  precision trade off directly through k and it is the one lever with no free lunch.
- Note the reranker result was disappointing with a stock sentence-transformer; a better-quality
  reranker is the source's explicit recommendation.
- Always force a **re-embed** when chunk size, overlap or the embedding model changes — the loader
  reuses an existing store and silently runs against the stale index. This bit the source twice.

**Red flags:** "tune k"; recommending a reranker first; not knowing that chunking changes require
re-embedding.

### Q14. How large does the golden set need to be?

**What they're really testing:** statistical humility.

**Model answer:**
- There is no universal N; it is set by the **size of the effect you need to detect**.
- The source ran 15 rows, which detected large moves (recall 80 → 97) but not small ones — the
  speaker attributed run-to-run fluctuation to row count and treated a 1-point change as noise.
- Practical rule: run identical settings twice and measure the spread. If your threshold is below
  that spread, you need more rows.
- Cover failure modes, not just volume — a stratified set of 100 rows spanning your known failure
  clusters beats 1,000 random rows.
- For reliability metrics the answer is different: 25–50 samples is far too few, and you push toward
  **1,000** to observe rare errors.

**Red flags:** "more is better"; quoting a number without connecting it to detectable effect size.

### Q15. What do you do with a production failure?

**What they're really testing:** whether you close the loop.

**Model answer:**
- Fold it back into the **offline golden dataset**.
- When a real conversation makes the app misbehave, take those specific examples and add them to the
  offline eval and golden datasets, so the next version is evaluated against them.
- The source calls this explicitly **a loop** — it is what makes the suite get better over time
  instead of going stale.
- This requires tracing plus a way to flag bad interactions (thumbs-down, or a human triage queue).
- Prioritise by frequency × severity, not by whoever complained loudest.
- This is also the mechanism by which online failures become regression tests — the same principle as
  promoting production incidents into a software test suite.

**Red flags:** "we'd fix the prompt"; no mechanism for getting failures into the dataset; treating
online and offline evaluation as separate worlds.

### Q16. DeepEval or RAGAS?

**What they're really testing:** whether you can articulate a tooling decision without tribal loyalty.

**Model answer:**
- Both cover the RAG Triad metrics, so for pure RAG neither is a capability gap.
- The source chose **DeepEval** for two stated reasons: RAGAS's metrics were already covered by the
  course material, and DeepEval is **substantially broader** — agents, multi-turn conversations,
  images, multi-modal — whereas RAGAS cannot do all of that.
- DeepEval is built on **pytest**, so it fits an existing Python testing workflow and the mental model
  transfers immediately.
- The source's bet is that DeepEval becomes a de-facto standard within a year or two.
- The framework-generic point: the framework choice is a bet on **ecosystem breadth**, not on metric
  quality — and learn the concept, because a tool is a week's work.

**Red flags:** "RAGAS is the standard"; being unable to name a difference; tool-first reasoning.

### Q17. What does experiment tracking buy you that a spreadsheet does not?

**What they're really testing:** whether you have run more than three experiments.

**Model answer:**
- A spreadsheet stores a score. Experiment tracking stores a score **with its configuration** —
  model, chunk size, overlap, k, judge model, golden-set version.
- Without the configuration attached, a number is not reproducible and you cannot attribute a change.
- It makes comparisons systematic: run 12 → run 13 → diff, automatically.
- The three maturity levels are: (1) plain baseline capture with manual comparison; (2) experiment
  tracking with a tool; (3) dashboard plus CI/CD.
- Tools named: **MLflow** (the canonical example), **Confident AI** (DeepEval's platform),
  **Weights & Biases**.
- Note there is no settled standard in the LLM world the way MLflow became one for ML — so learn the
  concept, not the product.

**Red flags:** "we keep a Google Sheet"; logging scores without configs; naming only one tool.

### Q18. Your eval suite takes 40 minutes and costs $30 per run. How do you make it fit a PR check?

**What they're really testing:** whether you understand the cost structure of evaluation.

**Model answer:**
- Split the suite by cost. **Operational evals are free** — no judge, no golden data — so they run on
  every commit.
- Code-based evaluators (schema valid, citation present, latency under budget) are near-free and run
  on every commit.
- Judge-based evals are the expensive tier: run a **stratified subset** on PRs and the full set
  nightly or pre-release.
- Cache aggressively: identical (prompt, judge, input) triples should never be judged twice — the
  source's own cost run showed heavy provider-side prefix caching from repeated questions, which is
  the same idea.
- Deterministic sampling (so the same trace is always scored the same way) matters when you sample.
- Cheap models for cheap criteria: not every judge needs a frontier model.

**Red flags:** running everything on every commit and calling it "CI"; no notion of a judge budget;
sampling non-deterministically.

### Q19. How would you evaluate a RAG app whose corpus changes daily?

**What they're really testing:** whether you can design for a moving target.

**Model answer:**
- The golden set must be **scoped to queries whose answers are stable**, plus a separate set for
  time-sensitive queries evaluated against a pinned corpus snapshot.
- Version the corpus: every eval run records the corpus version and the index build id.
- Expect **faithfulness to be stable while correctness drifts** — grounding does not depend on the
  world being right.
- Add **freshness checks** as their own band: does the retriever surface documents newer than N days
  for queries that need them?
- Run the suite on corpus updates as well as code updates — a corpus refresh is a deploy.
- Watch for silent index staleness: a re-embed that did not happen is indistinguishable from a
  retrieval regression until you check the store's build timestamp.

**Red flags:** assuming a static corpus; not versioning the index; treating corpus updates as
non-events.

### Q20. Should the judge be the same model as the generator?

**What they're really testing:** awareness of self-preference bias.

**Model answer:**
- Prefer not to, but the reason is subtler than "self-preference".
- **Self-preference bias** is real: models rate their own outputs higher, inflating scores when the
  judge shares a family with the generator.
- More important in practice: the judge must be **validated against human labels** (TPR/TNR) before
  any of its numbers mean anything. Family overlap is one of the things that moves TPR/TNR.
- A stronger judge than the generator is the common choice, but strength is not the same as
  agreement — validate.
- Cost matters: judging on every query with a frontier model can dominate your bill. Route it.
- Pin the judge model version and record it with every run — a silent judge upgrade invalidates
  baseline comparisons.

**Red flags:** "we use the same model, it's convenient"; never validating the judge; not pinning the
judge version.

---

## Tier 3 — Senior / staff (8 Q)

### Q21. Design the evaluation system for a RAG product from scratch. What do you build first?

**What they're really testing:** sequencing and pragmatism at scale.

**Model answer:**
- **Phase 0 — error analysis before any evaluator exists.** Sample 30–50 real traces, open-code
  free-text notes, axial-code into named failure modes, rank by frequency × severity. Writing evals
  before this measures imagined failures.
- **Phase 1 — the cheap tiers.** Tracing plus operational evals (free, always-on) and code evaluators
  for the deterministic failure modes. Many "quality" problems are schema or citation problems.
- **Phase 2 — one judge, validated.** Pick the top failure cluster; write a single binary judge for
  *that cluster only*; label 100+ examples; validate TPR/TNR on a held-out split; ship only when both
  clear ~0.9.
- **Phase 3 — the RAG Triad** once retrieval and generation are individually trustworthy.
- **Phase 4 — CI gate** with a per-metric threshold set above the noise floor; baseline captured on
  the first full run.
- **Phase 5 — online.** Same judges on sampled traffic, drift graph, failure harvesting into the
  golden set.
- Sequence matters more than completeness: a validated judge for one real failure mode beats twelve
  unvalidated metrics.

**Red flags:** proposing twelve metrics in week one; no error-analysis phase; a judge with no
validation plan; no answer to "what do you build *first*".

### Q22. How do you decide what NOT to evaluate?

**What they're really testing:** prioritisation maturity.

**Model answer:**
- Evaluation budget — engineer time and judge spend — is finite. Everything you add has a maintenance
  cost, and a stale metric is worse than no metric because it creates false confidence.
- Rank candidate failure modes by **frequency × severity × fixability**. Evaluate what you can act on.
- Prefer **code checks over judges** wherever the property is checkable: schema, citation presence,
  tool arguments, database state. A judge spent on a regex is waste — slower, costlier, noisier, and
  not unit-testable.
- Drop metrics that never move. If a metric has been green for six months across every change, it is
  not providing decision value; demote it to a periodic audit.
- Avoid vanity metrics that are not connected to a decision. Ask of each: *what would I do
  differently if this number halved?*
- Don't measure what you cannot act on within a sprint.

**Red flags:** "measure everything and see"; refusing to deprioritise; adding metric #14 to a broken
pipeline.

### Q23. Your judge scores 92% agreement with humans. Ship it?

**What they're really testing:** whether you know why raw agreement is a trap.

**Model answer:**
- No — agreement is the wrong statistic, especially on imbalanced data.
- A judge that rubber-stamps PASS scores ~90% agreement when 90% of traces genuinely pass, while
  catching **zero** failures. Its TPR is ~0 and its accuracy looks excellent.
- Report **TPR and TNR separately** on a held-out split, plus raw FP/FN counts.
- Corroborating finding: binary judges reached >95% precision on *consistent* summaries but only
  ~30–60% **recall** on *inconsistent* ones — the false-negative blind spot accuracy hides.
- Ship only when **both** TPR and TNR clear ~0.9; for safety-critical criteria, weight TPR higher,
  since a missed failure is usually worse than a false alarm.
- Re-validate whenever the agent, prompt, or data distribution changes — a validated judge is not
  validated forever.

**Red flags:** "92% is great, ship it"; reporting a single accuracy number; no held-out split; no
re-validation plan.

### Q24. How do you stop reward hacking in an agentic/RAG evaluation?

**What they're really testing:** adversarial thinking about your own harness.

**Model answer:**
- The general principle: **grade the world state, not the transcript.** Where the outcome is
  checkable (a database row, a file, a passing test suite), assert on that instead of on prose.
- Where output grading is unavoidable, **isolate the grader**: the grading process must not share
  state, filesystem or network with the system under test, or the agent can influence its own score.
- Prefer **verifiable rewards over judgeable ones** — deterministic, cheap, ungameable.
- Watch for **eval-specific overfitting**: if the same small golden set gates every deploy, teams
  optimise for it. Hold out a private set and refresh it.
- **Contamination** is the same problem at the model level: public benchmarks leak into training.
  Canary strings and freshness checks help.
- Log the harness alongside the score. The same model with a different harness can rank differently —
  a real observed effect, not a theoretical one.

**Red flags:** transcript-only grading; graders running in the same container; a frozen public golden
set as the only gate; never having considered that anyone would game it.

### Q25. How do you make evaluation cheap enough to run on every commit?

**What they're really testing:** cost engineering.

**Model answer:**
- Tier by cost. **Free tier** — ops evals and code evaluators, every commit.
- **Cheap tier** — small stratified judge subset, every commit.
- **Expensive tier** — full golden set plus the RAG Triad, nightly and pre-release.
- The dominant cost is judge tokens, so reduce calls rather than reduce coverage: cache identical
  judge inputs, use cheaper models for cheaper criteria, and batch.
- Prefer code evaluators: a schema check costs zero and never drifts.
- **Deterministic sampling** so a retried run doesn't re-score different traces, which otherwise
  makes runs incomparable and doubles spend.
- Keep a fixed golden set so the expensive tier's cost is predictable and can be budgeted.
- Instrument the eval suite's own cost — grading spend is part of the run's economics, not overhead.

**Red flags:** "we just run it all"; no caching; no cost attribution for the eval harness itself.

### Q26. How do you evaluate multi-turn RAG conversations differently from single-turn?

**What they're really testing:** whether your mental model scales past one-shot RAG.

**Model answer:**
- Single-turn metrics (faithfulness, context relevance) are **per-turn** and still apply, but they
  miss the failure modes unique to dialogue.
- Add: **context carry-over** — did the system resolve the pronoun/reference correctly?;
  **consistency** — does turn 4 contradict turn 2?; **information reuse** — did it avoid re-asking
  what it already knows?; **correction handling** — when the user says "no, I meant X", does it adapt?
- The unit of evaluation becomes the **conversation**, not the turn, so scoring needs to aggregate
  over a trajectory rather than average independent turns.
- Reference-based metrics are harder: the same question has different correct answers at different
  points in a conversation. Use outcome-based grading where possible.
- Sample whole conversations for human review rather than isolated turns — turn-level sampling hides
  the failures you care about.

**Red flags:** "same metrics, just looped"; turn-level sampling for conversational failures.

### Q27. What is the difference between model evaluation and application evaluation, and why does the company need both?

**What they're really testing:** whether you can connect evals to org decisions.

**Model answer:**
- **Model evals** answer "which model should we build on" — standardized evals (benchmarks) for a
  broad capability read, plus **custom model evals** on your own data when selecting for your
  application.
- **Application evals** answer "is our product good enough to ship" — the RAG Triad, correctness,
  completeness, safety, ops.
- They serve different decisions and different owners: model evals are episodic and inform a build
  decision; application evals are continuous and gate every deploy.
- Using a leaderboard score to make a product decision is the classic category error — benchmarks
  measure capability under their own harness, not your users' satisfaction.
- Conversely, application evals cannot tell you whether a *different* base model would be better;
  only a head-to-head custom model eval can.
- Both belong in the same suite so no one has to choose between them at release time.

**Red flags:** conflating the two; using MMLU to justify a product decision; no custom model eval
before committing to a provider.

### Q28. How would you present RAG evaluation status to a non-technical executive?

**What they're really testing:** translation without dilution.

**Model answer:**
- Lead with the **decision**, not the metrics: "we can ship this week; here is the one thing that
  got worse."
- Use three numbers with plain names: **"did it find the right information"** (retrieval),
  **"did it stick to what it found"** (faithfulness), **"what does it cost per question"** (cost).
- Express thresholds as commitments, not statistics: "we block a release if any of these drops more
  than 3 points" is understandable and defensible; "TPR 0.91" is not.
- Always show the **cost/quality trade** explicitly, because that is the decision they own.
- Never present a single aggregate "quality score" — it hides exactly what they need to decide.
- Say what is *not* measured: throughput, rare errors, adversarial robustness. Unmeasured is not the
  same as safe.

**Red flags:** dumping metric tables; one aggregate score; hiding the cost dimension; claiming more
certainty than the sample size supports.

---

## Tier 4 — Debug-this-scenario (5 Q)

### Q29. Faithfulness dropped from 0.94 to 0.81 overnight. No code changed. Diagnose.

**What they're really testing:** systematic triage under ambiguity.

**Model answer:**
- First, rule out **measurement artefacts**: did the judge model version change? Did the golden set
  change? Did the judge's prompt or sampling temperature change? A metric can move without the system
  moving.
- Then check **upstream inputs**: did the corpus/index update? Was the index rebuilt with a different
  embedding model? Is the vector store's build timestamp current?
- Then check **the world**: a provider model deprecation or silent upgrade is a common cause of
  overnight shifts — pin versions.
- Compare the **retrieval ledger** on the same window: if recall also fell, the problem is upstream of
  generation. If recall is flat and only faithfulness fell, the generator or its prompt changed
  behaviour.
- Then read the actual failing traces with the judge's reasoning attached — the aggregate number
  cannot tell you what broke.
- Finally, check query distribution: a traffic shift toward a query class the corpus does not cover
  will depress faithfulness with no code change at all.

**Red flags:** immediately tuning the prompt; blaming the model; not checking whether the *metric*
moved versus the *system*.

### Q30. Retrieval recall is 0.97 but users say the answers are wrong. What's going on?

**What they're really testing:** whether you know recall is not the goal.

**Model answer:**
- High recall with bad answers points at the **generation or the context assembly**, not retrieval.
- Check **context relevance**: you may be retrieving the right documents *plus* a large amount of
  noise that the generator attends to.
- Check **precision and rank**: recall 0.97 with the useful chunk at position 9 means the generator
  may effectively ignore it.
- Check **k and context length**: stuffing 15 chunks may exceed the useful context window and dilute
  attention.
- Check **faithfulness**: if faithfulness is also low, the generator is not grounded in what was
  retrieved.
- Check **correctness against the golden set**: "users say it's wrong" may be a subset of queries
  where the corpus is genuinely silent — retrieval succeeded, the answer does not exist.
- Check **completeness**: correct but partial answers read as wrong to users.

**Red flags:** pushing recall higher; assuming users are wrong; not separating retrieval failure from
generation failure.

### Q31. Your P95 latency improved from 3.0 s to 2.0 s. Are you done?

**What they're really testing:** whether you know percentiles can lie.

**Model answer:**
- Not necessarily — **failed and timed-out requests are excluded from the latency distribution**.
- If the timeout rate rose from 2% to 8% while the P95 fell, the system did not get faster; it started
  dropping more requests, and the survivors are the fast ones.
- **Always report the timeout rate on the same line as the percentile.**
- Check the sample size: if the P99 equals the P95, the sample is too small for a tail statistic.
- Check whether the measured path changed — a cache hit rate increase will move latency without any
  code improvement, and will not generalise to production traffic.
- Confirm identical setup and identical questions between the two runs; the absolute number is
  machine-dependent, so only the **differential** is portable.

**Red flags:** celebrating the percentile; no timeout metric; comparing runs with different question
sets.

### Q32. Cost per query came out at 2 paise, but finance says it's 4× that in production. Who's wrong?

**What they're really testing:** whether you trust your own harness.

**Model answer:**
- Probably the eval is wrong, and the cause is usually **prompt caching**.
- The eval harness sends the same question repeatedly, so the provider's automatic prefix cache fires
  aggressively — in the source's run, 1,109 of 1,753 input tokens were cached — and the measured cost
  understates reality.
- Production questions differ from each other, so the real cache hit rate is lower.
- Fix: recompute without the cache credit, or use **distinct questions per sample**.
- Other divergence sources: production answers are longer (output tokens usually cost ~4× input), the
  production query mix is different, and retries add hidden calls.
- The general lesson: **verify that the eval's traffic resembles production traffic** before quoting
  its unit economics to anyone.

**Red flags:** defending the number; not knowing about prefix caching; quoting a total rather than a
distribution.

### Q33. You change the embedding model and nothing moves. Debug.

**What they're really testing:** whether you have hit this specific, common bug.

**Model answer:**
- First suspicion: **the vector store was not rebuilt.** The loader checks for an existing store and
  reuses it, so a chunking or embedding change silently runs against the old index. This bit the
  source twice.
- Fix: delete the store directory and re-run. Then confirm the index build timestamp moved.
- Second: confirm the change actually took effect — log the resolved model name, dimensions and
  chunk parameters with the run. Config that is not logged is config that may not have applied.
- Third: check whether the metric can move at all — if recall was already 0.99, there is no headroom,
  and you should be looking at precision or at a harder question subset.
- Fourth: check whether the golden set is large enough to detect the effect (15 rows, ~1 point is
  noise).
- Fifth: a cached judge could be returning cached verdicts for unchanged inputs — clear the judge
  cache when inputs are unchanged but the system changed.

**Red flags:** concluding "embedding models don't matter"; re-running without checking the index;
not logging config with the score.

---

## Tier 5 — Trap questions & the naive-answer trap (5 Q)

### Q34. *(Trap)* "Just tell me the one metric I should track for RAG."

**The naive answer:** faithfulness, or "the RAG Triad score".

**What they're really testing:** whether you will collapse a structure under pressure.

**Model answer:**
- There is no single metric, and offering one signals you have a list rather than a framework.
- If forced to pick one for a **dashboard headline**, faithfulness is the most defensible: it is the
  metric that degrades when the world moves under a static corpus, and it is the one users
  experience as "it made something up".
- But a headline metric is a *monitoring* choice, not an *evaluation* choice. The deploy gate needs
  retrieval and generation metrics separately, or you cannot localise a regression.
- The honest answer: "one *headline*, several *gates*" — and say which is which.
- Add a paired operational number (cost per query) to the headline, because quality without economics
  is not a decision.

**Red flag:** naming one metric without qualification; treating the headline as the whole system.

### Q35. *(Trap)* "Correctness is 1.0 — can you ship?"

**The naive answer:** yes.

**What they're really testing:** whether you know completeness is a separate axis.

**Model answer:**
- Not necessarily. **Completeness is a different metric from correctness.**
- If the user's question has two parts and the answer only covers one, the answer is *correct* — the
  part it answered is right — and *incomplete*. The source's example scores completeness **0** even
  though correctness is perfect.
- A correct-but-partial answer is a real failure mode that a single "correctness" metric hides
  entirely.
- Also check what correctness was measured against — the golden set's distribution, not "all
  questions".
- And check the other bands: safety and ops can still fail a release on their own.

**Red flag:** shipping on one green number; never having heard of completeness.

### Q36. *(Trap)* "Recall is 0.95, so shouldn't we raise k to get the last 5%?"

**The naive answer:** yes, raise k.

**What they're really testing:** whether you understand the recall/precision trade.

**Model answer:**
- No — recall and precision trade off directly through k.
- Raising k from 5 to 10 will pull in the missing gold chunks, but it also pulls in noise, and **the
  noise is what you hand to the generator**. You may fix a retrieval metric while degrading the
  actual answer.
- The correct move is to fix the other levers first — chunk geometry, a better embedding model, a
  reranker — and only then tune k, because k is the one lever with no free lunch.
- Measure the *downstream* effect, not the retrieval metric alone: if faithfulness or correctness
  falls as recall rises, k was the wrong lever.
- Also check headroom: 0.95 may be the corpus's ceiling if some golden answers genuinely do not exist
  in the corpus.

**Red flag:** optimising a single retrieval metric in isolation.

### Q37. *(Trap)* "Our eval suite passes 100% of the time. Isn't that great?"

**The naive answer:** yes, the system is excellent.

**What they're really testing:** whether you recognise an eval that is measuring nothing.

**Model answer:**
- No — a suite that never fails is not passing, it is **saturated**, and it has stopped carrying
  information.
- An eval that cannot fail cannot gate anything. Its value is exactly its ability to distinguish a
  good change from a bad one.
- Diagnose: is the golden set too easy? Too small? Are the thresholds set below the noise floor, so
  everything clears? Are the failure modes you actually care about simply not represented?
- Fix: harvest real production failures into the set, add adversarial and edge cases, and re-measure
  the noise floor.
- The same phenomenon appears at benchmark scale as **contamination and saturation** — a score near
  100% on a public benchmark usually means the benchmark has been outgrown or leaked, not that the
  model is solved.
- Track the **pass rate over time**; a suite whose pass rate never varies is a maintenance cost with
  no decision value.

**Red flag:** treating 100% as a success criterion rather than a warning sign.

### Q38. *(Trap)* "We'll validate the judge after we ship — we don't have time now."

**The naive answer:** agree, ship, validate later.

**What they're really testing:** whether you understand that an unvalidated judge produces
confidence, not information.

**Model answer:**
- An unvalidated judge gives you a number that *looks* like a measurement but has unknown error bars
  — and teams make decisions on it, which is worse than having no number.
- The specific danger is **asymmetric error**: a judge can post high agreement while missing most real
  failures, so the metric reads green while the failures you built it to catch go straight through.
- Validation is not expensive: ~100 human-labelled examples and one held-out split, reported as TPR
  and TNR. That is hours, not weeks.
- The honest intermediate: ship behind a flag and treat the judge's output as **directional only**
  until validated — but do not put it in a CI gate.
- The step is skipped because it is unglamorous, which is exactly why the source calls it the single
  most-skipped step in the industry.

**Red flag:** treating validation as optional polish rather than a precondition for trust.

---

## Live-coding / whiteboard prompts (3)

### Prompt 1 — Compute rank-aware contextual precision (30 min)

> Given a list of retrieved chunks and a set of indices known to be relevant, implement
> `contextual_precision(retrieved, relevant_indices)` that averages prefix precision at each rank.
> Then demonstrate it on `✓✓✗✗✗` versus `✗✗✗✓✓` and explain why plain precision is insufficient.

**What a strong answer covers:** the prefix-precision definition; the 0.713 vs 0.130 result; an
explicit statement that rank matters because the generator weights early context; handling of the
zero-relevant-chunks edge case; and a note that at k = 1 the two definitions coincide.

**Red flags:** computing `len(intersection)/len(retrieved)` and calling it done; no edge-case handling.

### Prompt 2 — Design the eval suite directory layout (45 min)

> Sketch the repository for a RAG app plus its eval suite: where product code lives, where each
> evaluation band lives, and how a single entry point runs them all and produces a baseline
> comparison. Then write the CI step that gates a deploy on a 3-unit regression.

**What a strong answer covers:** `src/` with `retriever.py`, `generator.py`, `rag_pipeline.py`,
`main.py`; `evals/` with one file per band (retriever, generator, pipeline, application, safety,
ops); a root `run_evals.py`; a compare-to-baseline step with a per-metric threshold; and a note that
the threshold must exceed the noise floor, which is measured by repeat runs. Bonus: distinguishing
the free tier (ops, code checks) from the expensive tier (judges) and running them on different
cadences.

**Red flags:** one giant eval file; no baseline concept; a hardcoded absolute threshold with no
noise-floor justification.

### Prompt 3 — Write a binary judge for a named failure mode (60 min)

> Here are 40 production traces in which the assistant invented a policy detail not present in the
> retrieved context. Write the judge prompt, define the label, and describe exactly how you would
> validate it before putting it in CI.

**What a strong answer covers:** a **binary PASS/FAIL** label, never a 1–5 Likert; chain-of-thought
reasoning emitted **before** the verdict; few-shot examples given as **written critiques**, not a
long rubric; a constrained output format so the verdict parses deterministically; validation by
hand-labelling ~100 examples, splitting into fit/held-out, and reporting **TPR and TNR separately**
plus FP/FN counts; a ship criterion of roughly ≥0.9 on both; and a re-validation trigger when the
agent, prompt or data distribution changes.

**Red flags:** a Likert scale; rubric-stuffed system prompt with no critiques; reporting a single
accuracy number; no held-out split.

---

## Take-home / case-study prompt (1 full brief)

> **Brief.** You are the first evaluation engineer at a company shipping a customer-support RAG
> assistant. The corpus is ~40,000 internal help-centre articles, refreshed weekly. Traffic is
> ~50,000 questions/day. There is an existing "quality score" — a single number from an LLM judge
> that nobody has validated — currently reading 0.91, and the team is nervous because support
> escalations are rising.
>
> **Deliverables (≤6 pages):**
> 1. **Diagnosis of the current setup.** Name the three specific things wrong with a single
>    unvalidated judge score, and say what decision each failure is corrupting.
> 2. **Error analysis plan.** How you would get from 50,000 daily traces to a ranked list of named
>    failure modes in two weeks, including the sampling strategy and why pure random sampling fails
>    here.
> 3. **The eval suite.** The three levels, the metrics in each, which need a golden dataset and which
>    do not, and which are code evaluators rather than judges. Justify every metric by naming the
>    decision it informs.
> 4. **The judge validation plan.** Label definition, label volume, split strategy, the two
>    statistics you will report and why not accuracy, and your ship criterion.
> 5. **The CI gate.** The threshold, how you derived it, and what happens on a breach. State the
>    per-metric noise floor explicitly.
> 6. **The economics.** Cost per query broken into input and output tokens, the per-query budget, and
>    what you would cut first if the budget halved. Note any measurement artefacts that would make
>    your number optimistic.
> 7. **Online plan.** What you trace, which metrics you run online, the drift window and alert rule,
>    and the mechanism that puts production failures back into the offline golden set.
>
> **What is being assessed:** structural thinking, statistical honesty about sample sizes, the
> code-evaluator-versus-judge split, cost awareness, and whether the plan ends at deployment or
> begins there.

---

## Scoring rubric — what separates a hire from a no-hire

| Dimension | No-hire | Mid-level hire | Senior hire |
|---|---|---|---|
| **Structure** | Flat metric list; no levels | Names the three levels, in order | Names the levels *and* explains why build order is evaluation order |
| **Localisation** | "Evaluate end-to-end" | Separates component and pipeline evaluation | Adds the failure-attribution argument and what each class of failure costs to fix |
| **Statistical honesty** | Quotes accuracy; ignores sample size | Knows TPR/TNR matter | Reports both on held-out data, states a noise floor, refuses to quote tail percentiles from n = 25 |
| **Economics** | No cost dimension | Names cost per query | Reads the quality ledger against the cost ledger and says which constraint is hard |
| **Judge design** | Reaches for a Likert judge | Binary, with reasoning | Binary, few-shot critiques, validated, version-pinned, re-validated on distribution change |
| **Tooling** | Tool-first ("we use RAGAS") | Names the right tool and a reason | Describes the concept tool-independently and treats the framework as a bet on ecosystem breadth |
| **Production instinct** | Stops at deploy | Mentions monitoring | Tracing, drift window with explicit numbers, and a failure-harvesting loop into the golden set |
| **Prioritisation** | Measures everything | Measures what users complain about | Ranks by frequency × severity × fixability and will argue for measuring *less* |
| **Adversarial thinking** | Assumes good faith | Knows contamination exists | Designs against reward hacking, eval overfitting, and self-preference bias unprompted |
| **Communication** | Metric dump to executives | Explains metrics in plain words | Leads with the decision, expresses thresholds as commitments, and states what is *not* measured |

**The single strongest signal in a senior interview:** the candidate mentions that the eval harness
itself has a cost and a failure mode — that the judge may be wrong, the index may be stale, the
sample may be too small, and the cache may be flattering the number. That is the difference between
someone who runs evals and someone who trusts them correctly.
