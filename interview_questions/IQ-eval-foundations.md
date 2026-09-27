# Interview Pack · LLM Evaluation Foundations

> **Covers:** CS-01 (model evals vs application evals), CS-02 (the eval curriculum), CS-03 (why
> multiple eval pipelines), CS-04 (the complete workflow), CS-05 (LLM model eval capabilities) ·
> **Role level:** junior → mid (AI engineer, applied scientist, data scientist moving into LLM work) ·
> **Format:** screen → onsite → system-design → take-home

---

## How this domain is assessed

This pack covers the vocabulary layer. Interviewers use it to answer one question fast: **does this
person know the difference between evaluating a model and evaluating a product?** Candidates who
conflate the two will confidently cite MMLU when asked whether a feature should ship, and that single
slip ends the loop.

The domain assesses three things:

| What is being scored | The signal | How it is elicited |
|---|---|---|
| **Vocabulary precision** | Do you separate model evals from application evals? | "How would you choose which LLM to use?" |
| **Procedural fluency** | Can you recite a complete workflow, in order? | "Walk me through building an eval for a new feature" |
| **Cost realism** | Do you know what each eval method costs? | "Why not just use human review for everything?" |

**The trap that catches most candidates** is treating "evaluation" as a single activity. It is at
least four: choosing a base model, testing a component, testing a pipeline, and monitoring a product
in production. Each has a different dataset, a different cadence, and a different owner.

---

## Tier 1 — Fundamentals (10 Q)

### Q1. What is an LLM eval?

**What they're really testing:** whether you have a definition or a vibe.

**Model answer:**
- An eval is a **systematic, repeatable test** used to judge an LLM application's output.
- Three properties do the work: it is *systematic* (a defined procedure, not a spot check), it is
  *repeatable* (the same input configuration gives a comparable number), and it produces a *judgement*
  you can act on.
- An eval has four parts: a **task**, a **success criterion**, a **dataset**, and an **evaluation
  method**.
- The success criterion is the part teams skip, and skipping it produces an eval that cannot fail —
  which means it cannot decide anything.
- "Repeatable" is what distinguishes an eval from a demo: if you cannot re-run it and compare, you
  have an anecdote.

**Red flags:** "we look at the outputs"; no mention of a dataset or a criterion; treating a
leaderboard score as an eval of your application.

### Q2. What is the difference between a model eval and an application eval?

**What they're really testing:** the single most important distinction in the domain.

**Model answer:**
- **Model evals** answer *which model should we build on.* They split into **standardized evals**
  (published benchmarks like MMLU and GSM8K) and **custom model evals** — your own dataset, built from
  your own task, used to pick between candidate models.
- **Application evals** answer *is our product good enough to ship.* They test the assembled system:
  retriever, generator, prompts, guardrails, tools.
- They serve different **decisions**, have different **cadences** (model evals are episodic, at
  selection time; application evals are continuous, on every change), and different **owners**.
- A benchmark score tells you a model's capability under *that benchmark's* harness; it does not tell
  you whether your feature will satisfy your users.
- The classic category error is using a leaderboard to justify a product decision. The reverse error
  — using only application evals — leaves you unable to tell whether a different base model would be
  better.

**Red flags:** using the terms interchangeably; "a benchmark is an eval"; never having run a custom
model eval before committing to a provider.

### Q3. What counts as an LLM application? Do they all evaluate the same way?

**What they're really testing:** whether your eval strategy is a template or an understanding.

**Model answer:**
- Five shapes: a **plain chatbot**, a **RAG-based chatbot**, an **agent**, a **multi-modal** app (e.g.
  one that also generates images), and a **fixed-schema output** app (e.g. an email classifier that
  must emit *support / refund / technical* and nothing else).
- No, the strategy differs. A **fixed-schema** app is evaluated like a classifier: exact match on the
  label, accuracy, confusion matrix. Cheap, deterministic, no judge needed.
- **RAG** needs retrieval metrics plus the RAG Triad. **Agents** need trajectory and world-state
  grading. **Multi-modal** needs modality-specific judges.
- Chatbot-only is the easy case and is largely subsumed by RAG and agents.
- The practical consequence: before choosing metrics, **classify your application shape**, because the
  shape determines whether your metrics are string comparisons or validated judges.

**Red flags:** one eval plan for everything; not knowing that fixed-schema apps are classifier-shaped;
treating "chatbot" as the only shape.

### Q4. What is a golden dataset?

**What they're really testing:** whether you know where the cost actually lives.

**Model answer:**
- A **golden dataset** is your curation of inputs paired with the correct/expected outputs — the
  ground truth you evaluate against.
- For a classification-shaped task it is two columns: the input, and the human-assigned label.
- Sizing: the source's illustration uses 3 rows, but states you would normally build **50 to 500
  rows**.
- The best source of data is **your own historical traffic** — real inputs your system has already
  seen — labelled by a human you sit down with. Synthetic data has its place but does not replace
  real distribution.
- Cost is dominated by labelling, not by tooling. That is why the eval-method choice matters so much:
  a cheaper method lets you afford a bigger set.
- The golden set is a **living artefact**: production failures get added to it, so it grows toward the
  distribution that actually breaks you.

**Red flags:** "we'll generate it with an LLM and ship it"; no human review; no plan for growing it;
confusing the golden set with the prompt examples.

### Q5. What are the three evaluation methods, and when do you pick each?

**What they're really testing:** cost/quality triage.

**Model answer:**
- **Automated** — write code that computes the metric. Correct when the output is discrete and
  checkable (a class label, a schema, a citation, a number). Cheapest, fastest, deterministic, and it
  can gate every commit.
- **Human** — correct when outputs are long free text and the judgement is genuinely subtle. Highest
  quality, but the source's phrasing is blunt: in production you have to *pay a salary* — it does not
  scale to every release.
- **LLM-as-judge** — the middle option: you test through an LLM. Scales like automation, handles
  subjective criteria, but produces a number you must **validate against human labels** before you
  trust it.
- The decision rule in order: **can code check it? → code. No? Is a judge validated for it? → judge.
  No? → human, on a sample.**
- Most teams over-invest in judges and under-invest in code checks and human labels — but the human
  labels are what make the judge trustworthy in the first place.

**Red flags:** jumping to a judge for something a regex would catch; "we use humans for everything";
using a judge without ever collecting human labels.

### Q6. Walk me through building an eval for a new feature.

**What they're really testing:** procedural fluency — can you recite a workflow unprompted?

**Model answer:**
- Define the **task** and the **target** explicitly.
- Define the **success criterion** — and make it measurable. For a routing system: "if out of 100
  emails it routes 90 correctly, the system is 90% accurate."
- Build the **dataset** (50–500 rows, ideally from real traffic, human-labelled).
- Choose the **evaluation method** (automated / human / judge) based on output shape.
- **Run** the system and **evaluate** the results.
- **Analyse** the failures — this is error analysis, and it is what tells you what to fix.
- **Improve** and **iterate**. Repeat until the numbers stop moving in a way you care about.
- **Deploy**, **monitor**, and feed production failures back into the dataset. The loop never closes.

**Red flags:** starting with metrics; no success criterion; no analysis step between "run" and
"improve"; treating deployment as the end.

### Q7. Why does one LLM application need several eval pipelines?

**What they're really testing:** whether you understand failure localisation.

**Model answer:**
- Because **an LLM application can break in many places.** "One LLM-based application may have several
  LLM evals" is the organising claim.
- For a RAG chatbot there are at least two obvious failure points: the **retriever** and the
  **generator** — and they fail in two independent ways. (a) The retriever fetches wrong documents, so
  the generator answers wrongly *on* the wrong documents. (b) The retriever is correct but the
  generator **ignores it, hallucinates, and answers wrongly**.
- Those two causes need **different fixes** — a reranker versus a prompt/model change — so a single
  end-to-end score cannot tell you what to do.
- There is a third, **workflow-level** eval: the failure may only be visible in the *interaction*
  between retriever and generator.
- A real RAG app typically runs a retriever eval, an embedding-model eval, a whole-workflow eval, and
  a latency eval **simultaneously**.

**Red flags:** "we have one quality score"; no ability to name the two failure modes; not knowing
that the fix differs by cause.

### Q8. What makes a good success criterion?

**What they're really testing:** whether you can turn a fuzzy goal into a measurable one.

**Model answer:**
- It must be **measurable**, **decision-relevant**, and **agreed before you look at the results**.
- Measurable: it maps to a number and a threshold. "90% routing accuracy" is measurable; "good UX" is
  not.
- Decision-relevant: you must be able to state what you would *do differently* if it halved. If
  nothing changes, it is a vanity metric.
- Agreed up front: picking the threshold after seeing the score is how teams ship regressions.
- It should be **plural but small** — a headline criterion plus the constraint criteria (latency,
  cost, safety) that can veto a release on their own.
- For subjective outputs, the criterion must still specify the *failure mode*, not just "quality" —
  "does not invent policy details" is checkable; "is helpful" is not.

**Red flags:** criteria chosen post hoc; a single aggregate score; "we know it when we see it".

### Q9. What is the relationship between offline and online evaluation?

**What they're really testing:** whether your model of evaluation includes production.

**Model answer:**
- **Offline** evaluation runs a frozen dataset pre-deploy: fast, repeatable, and the basis of the
  deploy decision.
- **Online** evaluation runs on sampled production traffic post-deploy: realistic, and the only way to
  see distribution shift, drift, and long-tail failures.
- They must use the **same evaluators**, or the offline number does not predict online behaviour.
- Online needs **tracing** — latency, cost, tokens, and user feedback captured per interaction.
- The loop between them is the point: production failures become golden-set rows, so the offline
  suite tracks reality instead of diverging from it.
- Offline alone means quality degrades silently; online alone means you cannot gate a release.

**Red flags:** "we evaluate before launch"; no tracing; treating the two as separate projects with
separate metrics.

### Q10. Why not just deploy and see what happens?

**What they're really testing:** whether you understand what evaluation buys.

**Model answer:**
- Because deploying without evaluation makes **the user the test set**, and the failure mode is a
  production incident rather than a number.
- Without a baseline you cannot tell an improvement from a regression — you have no comparison point,
  so every change is a guess.
- You cannot localise: a bad answer gives you no information about whether retrieval or generation
  broke.
- You cannot iterate fast: the feedback loop becomes user complaints, which are slow, sparse, and
  biased toward the loudest users.
- You cannot roll back confidently, because you do not know what "worse" means.
- The source's framing: build the system, then the real question is *"can we deploy this directly?
  No."* — evaluation comes before deployment, not after.

**Red flags:** "we ship and watch the metrics"; no baseline; treating user complaints as the eval
signal.

---

## Tier 2 — Applied / trade-off (10 Q)

### Q11. How big should the golden dataset be?

**What they're really testing:** whether you connect set size to decision quality.

**Model answer:**
- It is set by **the size of the effect you need to detect**, not by a rule of thumb.
- The source's guidance for a classification-shaped task: normally **50 to 500 rows**.
- Coverage beats volume: a stratified set spanning your known failure clusters is worth more than a
  larger random sample, because pure random sampling under-covers rare failures.
- There is a hard floor for judges: you need **~100 human-labelled examples** to validate a judge with
  TPR/TNR at all.
- Compute the noise floor — run identical settings twice and measure the spread. A threshold below it
  produces flapping builds.
- Standard error of a proportion is the intuition: at p = 0.9, n = 15 gives SE ≈ 0.077, so a 1–2
  point move is noise; n = 100 gives SE ≈ 0.030. A 15-row set detects large moves only.

**Red flags:** "a few hundred is standard" with no reasoning; no noise-floor concept; judging with 20
labels.

### Q12. Iterating: prompt fix or model switch?

**What they're really testing:** whether you know the cost-ordered ladder.

**Model answer:**
- Order by cost: **prompt fix → retrieval/context fix → model switch**. The source's example climbed
  exactly this ladder.
- Prompt first: cheapest, fastest, reverts in seconds. In the example, a prompt fix took routing
  accuracy 80% → **90%**.
- Then the data/context layer: better retrieval, better chunking, better few-shot examples — still
  cheap and reversible.
- Model switch last: the heaviest LLM took it **90% → 95%**, but it costs more per call, adds latency,
  and is harder to roll back. Justify it with the cost ledger, not just the quality ledger.
- Stop when the marginal gain no longer clears the noise floor, or when the remaining failures are not
  worth their cost.
- Record which lever produced which delta — that knowledge is what makes the *next* decision cheap.

**Red flags:** reaching for a bigger model first; not knowing what a prompt fix is worth; no stopping
rule.

### Q13. Why not use human evaluation for everything?

**What they're really testing:** cost realism.

**Model answer:**
- Humans are the highest-quality signal and the only source of the labels that validate your judges —
  so they are indispensable, but they are *not* a scalable per-release gate.
- The source's phrasing is direct: in production, human evaluation means you have to **pay a salary**.
- Throughput is the binding constraint: a release cycle is not gated by how much a human can read in a
  week.
- Inter-annotator disagreement is a real cost too — you need guidelines, calibration rounds, and
  agreement measurement before human labels are trustworthy enough to validate against.
- The right shape: humans on a **stratified sample** and on the **judge-validation set**, automation
  on everything that repeats.
- Human effort should be minimised and focused: free-text observations from humans, machine
  structuring of those observations.

**Red flags:** "humans are the ground truth so we use them for everything"; no annotation guidelines;
ignoring inter-annotator agreement.

### Q14. How do you handle a fixed-schema output application?

**What they're really testing:** whether you can recognise an easy eval and not over-engineer it.

**Model answer:**
- It is a **classification problem**, so evaluate it like one: exact match against the gold label,
  accuracy, and a **confusion matrix** to see *which* classes are being confused.
- No LLM judge is needed, and using one would be a mistake — slower, costlier, noisier, and not
  unit-testable.
- Report **per-class** precision and recall, not just overall accuracy, because class imbalance will
  hide a systematically broken class.
- The threshold is a business decision: which error is worse? Routing a billing email to technical
  support may cost far less than routing it to refunds.
- Add schema-validation as a code check on every call: *did the model emit one of the allowed labels
  at all?* Invalid output is a different failure from wrong output and needs its own metric.
- This shape is where you should demand **the most automation**, freeing human labelling budget for
  the harder systems.

**Red flags:** building a judge for a 3-class classifier; reporting only overall accuracy; no
schema-validity check.

### Q15. A model has the best benchmark scores. Is it the right choice for our app?

**What they're really testing:** whether benchmarks are your only evidence.

**Model answer:**
- Not necessarily — benchmark scores measure capability under *that benchmark's* harness, on *that
  benchmark's* data, which is not your task's distribution.
- Run a **custom model eval**: your own dataset, your own criterion, the candidate models head to
  head. That is the only direct evidence for *your* decision.
- Weigh the operational dimensions too: latency, cost per query, context-window needs, license and
  hosting constraints, and tool-calling reliability if you are building an agent.
- Watch for **contamination and saturation**: a near-perfect public score may mean the benchmark was
  outgrown or leaked into training, not that the model is solved.
- The same model under a different harness can rank differently, so compare models *inside your own
  application's scaffolding*.
- Treat benchmark scores as a shortlist filter, never as the decision.

**Red flags:** picking the leaderboard leader; no custom eval; ignoring latency and cost; not knowing
that harness can flip the ranking.

### Q16. What do you monitor after deployment?

**What they're really testing:** whether "monitor" means something specific to you.

**Model answer:**
- **Operational**, continuously: latency percentiles plus timeout rate, cost per query, token usage,
  success/error/retry rates split by cause.
- **Quality**, on a sample: the same metrics you evaluated offline — faithfulness, relevance,
  correctness — run online on sampled traffic.
- **Drift**: a rolling window per metric, with an alert rule. The concrete shape used in the source's
  RAG work: a 24-hour graph, alert if a metric degrades over the last 8 hours.
- **User feedback**: thumbs up/down, escalations, regeneration rate, abandonment.
- **The failure-harvesting loop**: production failures get added to the golden dataset, so the offline
  suite improves from what production taught you.
- Everything hangs off **tracing** — without per-interaction records, none of the above exists.

**Red flags:** "we watch latency"; no quality metric online; no drift alert; no feedback loop into the
dataset.

### Q17. How do you choose between building and buying eval tooling?

**What they're really testing:** judgement about where your scarce engineering goes.

**Model answer:**
- Name the layer first: **process, runner, judge, benchmark adapters, platform.** Different layers are
  buy/build decisions for different reasons.
- **Buy** the commodity layers: benchmark adapters, trace storage, dashboards, standard metric
  implementations. Nobody wins by writing their own histogram.
- **Build** what encodes your product's judgement: your golden dataset, your failure-mode taxonomy,
  your custom judges. That is the part competitors cannot copy.
- The runner is the genuine judgement call: a registry-shaped framework usually wins unless your
  evaluation has a shape nothing supports (long-horizon execution grading is the usual example).
- Beware the platform trap: a five-component self-hosted observability stack is a real operational
  cost. Adopt it when a second person needs to look at traces, not before.
- The decision rule: buy the plumbing, build the judgement, and keep the golden dataset portable so
  you can change tools without losing your ground truth.

**Red flags:** building a runner from scratch reflexively; buying a platform on day one; letting the
golden dataset be locked inside a vendor's format.

### Q18. What is the cost of an eval program?

**What they're really testing:** whether you have thought about evals as an expense.

**Model answer:**
- Four cost lines: **human labelling** (usually dominant at the start), **judge tokens** (dominant at
  scale), **infrastructure** (containers, GPUs for execution grading, trace storage), and
  **engineering time** (the recurring one that teams forget).
- The engineering time is what makes maintaining twelve metrics expensive — a stale metric is not free,
  it is a liability.
- Cost-reduction levers: code checks instead of judges; caching identical judge inputs; cheaper models
  for cheaper criteria; deterministic sampling so retries do not re-score; tiering the suite by cadence.
- **Instrument the eval suite's own spend.** Grading is part of the run's economics, not overhead —
  execution-grading harnesses track judge token usage for exactly this reason.
- The framing that keeps it honest: the eval suite is a gating mechanism that lets you change, push
  and deploy without fear. That is what the spend buys, and the spend should be sized to it.

**Red flags:** no idea what the suite costs to run; treating human labelling as free; no cost
attribution for the harness itself.

### Q19. How do you evaluate a multi-modal application?

**What they're really testing:** whether you know where your experience ends.

**Model answer:**
- Decompose by modality: the text path and the image path have different failure modes and need
  different metrics.
- For generation: judge-based criteria on fidelity to the prompt, plus deterministic checks (does the
  image render, does it meet the required dimensions, does it contain the requested object).
- For understanding: classification-style metrics on labelled examples (is the description correct,
  is the OCR right).
- Cross-modal consistency is its own metric: does the caption match the image, does the answer cite
  the visual evidence it claims?
- Tooling matters here — most mature runners and judge libraries cover multi-modal; narrower
  RAG-focused libraries often do not, which is a real reason to prefer the broader framework.
- Note that multi-modal is a minority of production systems; if your interview is for a text-first
  role, say so honestly rather than over-claiming.

**Red flags:** "same as text"; no cross-modal consistency check; no deterministic render check.

### Q20. What is the difference between an eval and a test?

**What they're really testing:** whether you can be precise about your own discipline.

**Model answer:**
- A **test** asserts a specific expected behaviour on a specific input — pass/fail, deterministic,
  binary. It belongs in CI and it either passes or the build is broken.
- An **eval** measures quality on a distribution — it produces a *number* on a dataset, and the
  question is whether that number moved meaningfully.
- Both belong in the suite: code-based evaluators are essentially tests (schema valid, citation
  present, latency under budget), and judge-based evaluators are measurements.
- The practical consequence is the **threshold**: a test needs no threshold, an eval does, and the
  threshold must exceed the noise floor.
- Confusing them is why teams write "evals" that fail loudly on noise and get disabled.
- Software-testing concepts transfer usefully — pytest as the runner paradigm, fixtures, CI gating —
  which is a real argument for eval libraries that build on pytest.

**Red flags:** using the words interchangeably; no threshold concept; no distinction between
deterministic and measured checks.

---

## Tier 3 — Senior / staff (8 Q)

### Q21. You are joining as the first eval engineer. What do you do in the first 30 days?

**What they're really testing:** prioritisation, and whether you start with data or with tooling.

**Model answer:**
- **Days 1–5: look at real data.** Pull 30–50 real traces and read them. Open-code free-text notes.
  Do not write a single evaluator.
- **Days 5–10: axial-code into named failure modes**, count frequencies, and rank by frequency ×
  severity × fixability. This ranking *is* your roadmap.
- **Days 10–20: the cheap tier.** Tracing plus operational evals (free, no judge, no golden data) and
  code evaluators for the deterministic failure modes. Many "quality" problems are schema, citation or
  format problems.
- **Days 20–30: one judge, validated.** Pick the top failure cluster, write one binary judge for it,
  get ~100 human labels, report TPR/TNR on a held-out split.
- Throughout: **capture a baseline** on the first full run, and resist tool procurement until you know
  what you are measuring.
- The deliverable at day 30 is not a framework — it is a ranked failure taxonomy, a baseline, and one
  validated judge.

**Red flags:** starting with framework selection; building twelve metrics; no error-analysis phase; no
baseline.

### Q22. How do you decide what not to evaluate?

**What they're really testing:** whether you will argue for measuring less.

**Model answer:**
- Rank candidate failure modes by **frequency × severity × fixability** and evaluate what you can act
  on. A metric for a failure you cannot fix is a permanent red dashboard.
- Prefer **code checks over judges** wherever the property is checkable — a judge spent on a schema
  check is waste.
- Drop metrics that never move. If a metric has been green across every change for six months, it is
  not providing decision value; demote it to a periodic audit.
- Apply the test: *what would I do differently if this number halved?* If the answer is "nothing", the
  metric is decoration.
- Watch the maintenance cost: each metric needs ownership, a dataset, thresholds, and re-validation
  when the system changes. A metric is a standing liability with a one-time benefit.
- It is a legitimate senior answer to say the suite should have **three metrics, not twelve** — if
  those three are the ones that have ever changed a decision.

**Red flags:** "measure everything"; refusing to deprioritise; adding metric #14 to a broken pipeline.

### Q23. How do you make evaluation part of the engineering culture?

**What they're really testing:** organisational awareness.

**Model answer:**
- Make the eval command the **fastest path to feedback** on a change — if running the suite is slower
  than deploying and eyeballing, nobody will run it.
- Put the gate in **CI**, automatically, so the decision does not require anyone to remember.
- Make failures **self-explaining**: every judge result carries its reasoning, so the person who broke
  it can see why without asking the eval owner.
- Give the suite a **visible number** on a dashboard that the whole team looks at anyway — the same
  place latency and cost appear.
- Budget the maintenance explicitly: name an owner, and treat re-validation after a model change as
  required work rather than an interruption.
- Lead with the *decision* the suite enables — "we can ship on Fridays now" — rather than the metric
  count.

**Red flags:** relying on conviction and docs; a suite that takes 40 minutes; no named owner; metrics
without reasoning attached.

### Q24. How do you choose the base model for a product?

**What they're really testing:** whether you can run a selection process, not just read a leaderboard.

**Model answer:**
- Step 1: use **standardized evals** as a shortlist filter — cheap, broad, and good for eliminating
  obviously-wrong candidates.
- Step 2: build a **custom model eval** on your own task: a dataset from real traffic, your own
  success criterion, the shortlisted models run head to head *inside your application's harness*.
- Step 3: weigh the operational dimensions — latency, cost per query, context window, tool-calling
  reliability, hosting/licensing constraints.
- Step 4: check contamination and saturation before trusting a suspiciously high public score.
- Step 5: decide, then **keep the custom eval** as the regression check when the provider ships a new
  version — a silent model update is a system change.
- Record the harness with every number; the same model under different scaffolding can rank
  differently.

**Red flags:** choosing the leaderboard leader; no custom eval; ignoring cost and latency; no
regression check on silent model updates.

### Q25. How do you evaluate something where you cannot define the correct answer?

**What they're really testing:** whether you can design an eval under genuine uncertainty.

**Model answer:**
- Accept that you cannot measure correctness, and measure something adjacent that you *can* define:
  **groundedness** (is it supported by the provided context?), **consistency** (does it contradict
  itself or the corpus?), **coverage** (does it address all parts of the input?), **constraint
  adherence** (schema, tone, length, refused-when-appropriate).
- Prefer **relative** judgements over absolute ones: pairwise comparison ("which of these two is
  better, and why?") is far more reliable for judges than an absolute quality score.
- Use **preference data** where you have it — user thumbs, regeneration rate, dwell time.
- Where the task is executable, **grade the world state** instead of the prose: check the database,
  the file, the passing test suite. That is the escape hatch from unjudgeable outputs.
- Be explicit in reporting that the metric is a proxy. A proxy presented as ground truth is how eval
  programs lose credibility.

**Red flags:** forcing a 1–5 "quality" score; no pairwise option; presenting a proxy as truth.

### Q26. How do evaluation and training interact?

**What they're really testing:** whether you see the RL connection.

**Model answer:**
- A benchmark is, structurally, a **frozen RL environment**: a task, a reward signal, and an episode.
  The reward signal is the eval.
- Therefore **verifiable rewards beat judgeable ones** wherever the task allows — deterministic,
  cheap, ungameable, and they can be computed at training scale.
- The failure mode is **reward hacking**: if the reward is a proxy the model can game, training will
  find the exploit. Container isolation and world-state grading exist specifically to close this.
- Evaluation drives training data too: error analysis on a deployed system identifies the failure
  modes worth collecting demonstrations for.
- The loop runs both ways: evals select the model, then evals gate what training produced.
- The caution: any eval used as a training signal stops being a valid held-out measurement. Hold out a
  private set.

**Red flags:** treating training and evaluation as unrelated; not knowing what reward hacking is;
training on the eval set.

### Q27. How do you report evaluation results to leadership?

**What they're really testing:** translation without dilution.

**Model answer:**
- Lead with the **decision**: "we can ship"; "we should not ship"; "here is the trade-off you own."
- Use three plain-language numbers: did it find the right information, did it stick to what it found,
  what does a question cost.
- Express thresholds as **commitments**: "any release that drops more than 3 points on these does not
  ship" is defensible; "TPR 0.91" is not.
- Always surface the cost/quality trade explicitly — that is the decision leadership actually owns.
- Never present one aggregate "quality score" — it hides exactly the thing they must decide.
- State what is **not** measured: throughput, rare errors, adversarial robustness. Unmeasured is not
  the same as safe.
- Where the sample is small, say so. Over-claiming certainty that later collapses costs more trust
  than the original caveat would have.

**Red flags:** metric tables; one aggregate score; hidden cost dimension; certainty beyond the sample
size.

### Q28. What does a mature eval program look like at 200 engineers?

**What they're really testing:** whether you can think beyond one team.

**Model answer:**
- **Shared substrate**: one trace store where quality, latency, cost and feedback sit on the same
  object for the same interaction. Without this, every team rebuilds it badly.
- **Shared contracts**: a common score schema, so a score means the same thing across teams and can be
  aggregated.
- **Centralised judgement, decentralised ownership**: judges and the validation discipline are
  centralised (they are easy to get wrong and expensive to duplicate); datasets and failure taxonomies
  are owned by the teams who know the domain.
- **A golden-dataset practice**, not a golden-dataset file: versioned, reviewed, grown from production
  failures, with a rule that a fixed bug gets a row.
- **Tiered cadence**: free checks on every commit, sampled judges on PRs, full suite nightly and
  pre-release.
- **An eval on the evals**: track the suite's own cost, its pass rate over time, and its judge
  TPR/TNR. A suite whose pass rate never varies has stopped working.

**Red flags:** assuming each team should build its own everything; no shared score schema; no notion
of the eval suite's own health.

---

## Tier 4 — Debug-this-scenario (5 Q)

### Q29. Your team has an eval suite. Every run passes. What do you check?

**What they're really testing:** whether you can recognise a saturated eval.

**Model answer:**
- A suite that never fails is not passing — it is **saturated**, and it has stopped carrying
  information.
- Check the **difficulty of the dataset**: is it made of easy cases? Was it generated by the same LLM
  that is being evaluated?
- Check the **thresholds**: set below the noise floor, everything clears.
- Check the **coverage**: are the failure modes the team actually worries about represented at all?
- Check whether it is **stale**: does the dataset contain last quarter's production failures?
- Fix by harvesting real failures, adding adversarial and edge cases, and re-measuring the noise
  floor. Track the pass rate over time as its own health metric.

**Red flags:** "100% is great"; no dataset review; no threshold review.

### Q30. Accuracy is 95% but users are unhappy. Diagnose.

**What they're really testing:** whether you can find the gap between your metric and reality.

**Model answer:**
- First: **what is the dataset's distribution versus production's?** A 95% on a balanced set can be
  terrible on a production mix dominated by two classes.
- Check **per-class metrics and the confusion matrix**. 95% overall can hide one class at 40% — and
  that class may be the one users complain about.
- Check **what the metric does not measure**: style, completeness, latency, tone. Users experience all
  of it.
- Check whether the unhappy users are a **cohort** — a language, a region, a product line — that the
  dataset under-represents.
- Check the **feedback channel**: are complaints sampled from the loudest users, or do they reflect a
  real distribution shift?
- Fix: re-stratify the dataset toward production's actual mix, add the underrepresented cohorts, and
  add the missing dimensions as separate metrics.

**Red flags:** trusting the number over the users; no per-class breakdown; no cohort analysis.

### Q31. Your golden dataset was built six months ago and scores keep improving, but production quality is flat.

**What they're really testing:** whether you understand dataset drift and overfitting.

**Model answer:**
- This is the classic signature of **eval overfitting**: the team has optimised against a frozen set
  while the production distribution moved away from it.
- Check whether the dataset contains any recent production failures — if not, it is measuring last
  quarter's system.
- Check whether the same set is used for iteration *and* for the gate. If so, it is a training set
  wearing a test set's clothes.
- Fix: harvest recent production failures into the set, hold out a **private** set that is never used
  during development, and refresh the public set on a cadence.
- Also verify the production measurement itself — are the online metrics using the same evaluators as
  offline? If not, the two numbers are not comparable and "flat" may be a measurement artefact.
- Long term: track the correlation between offline and online for the same metric. That correlation
  *is* the eval program's credibility.

**Red flags:** "we need a harder dataset" with no reference to production; no private hold-out; no
offline/online correlation check.

### Q32. Your judge and your human reviewers disagree 30% of the time. What do you do?

**What they're really testing:** systematic debugging of an eval instrument.

**Model answer:**
- First, quantify the disagreement correctly: report **TPR and TNR separately** with raw FP/FN counts.
  "30% disagreement" alone is uninterpretable — it could be concentrated entirely in one direction.
- Check the **label definition**. Most disagreements are definitional: the humans and the judge are
  answering slightly different questions. Rewrite the criterion until a human can apply it
  unambiguously.
- Check the **human side** too — inter-annotator agreement. If humans disagree with each other 25% of
  the time, the judge is not the problem; the criterion is.
- Look at the **few-shot examples**: are they drawn from the same distribution as the disagreement
  cases? Add critiques from the disputed cases.
- Check for **systematic bias**: does the judge fail more on longer answers, a particular language, or
  outputs from one model family (self-preference)?
- Do not simply tune until agreement rises — you can always overfit a judge to a label set. Re-validate
  on a held-out split and re-check the noise floor.

**Red flags:** "we'll just retune the prompt"; reporting a single agreement number; never checking
human inter-annotator agreement.

### Q33. Leadership wants one number to track. What do you give them?

**What they're really testing:** whether you can resist a reasonable-sounding request without being
obstructive.

**Model answer:**
- Push back once, with the reason: a single aggregate hides the thing they need to decide — a
  high-quality-but-expensive system and a cheap-inaccurate one can produce the same number.
- Offer the alternative that satisfies the underlying need: **one headline metric plus the constraint
  metrics that can veto a release.** That is one number to watch, with the vetoes visible.
- The headline should be the metric that users experience as failure — for RAG, faithfulness
  ("did it make something up?"); for a classifier, per-class recall on the class that matters most.
- Always pair the headline with **cost per query**, because that is the trade leadership owns.
- Express it as a **commitment**: "we block a release if this drops more than 3 points."
- Document the composite if they insist on a single score — and label it explicitly as a composite so
  nobody mistakes it for a measurement.

**Red flags:** inventing an unvalidated composite and presenting it as quality; refusing outright
without an alternative; hiding the cost dimension.

---

## Tier 5 — Trap questions & the naive-answer trap (5 Q)

### Q34. *(Trap)* "Isn't a benchmark score just an eval?"

**The naive answer:** yes, equivalently.

**What they're really testing:** vocabulary precision under a leading question.

**Model answer:**
- No. A benchmark is **one kind** of eval — a *standardized* eval — and it measures a model's
  capability on someone else's task under someone else's harness.
- An application eval measures *your* system on *your* task with *your* success criterion.
- They answer different questions: "which model is broadly capable" versus "should we ship this
  feature."
- Benchmarks are legitimately useful as a **shortlist filter** for model selection; they are not
  evidence about your product.
- The tell that someone has conflated them is a product decision justified by a leaderboard rank.

**Red flag:** agreeing; no notion of a custom model eval.

### Q35. *(Trap)* "We'll write the evals after we ship, based on real failures."

**The naive answer:** that sounds pragmatic and data-driven, so it seems wise.

**What they're really testing:** whether you can identify a plan that makes users the test set.

**Model answer:**
- The *instinct* is right — production failures are the best source of eval data — but the *sequence*
  is wrong.
- Shipping first means you have **no baseline**, so you cannot tell whether the next change helps or
  hurts. Every subsequent decision is a guess.
- You also cannot localise: a bad answer with no component evals gives you no information about
  whether retrieval or generation broke.
- "We'll learn from failures" also assumes the failures are visible, frequent, and attributable.
  Real failures are sparse, and the ones that matter most are the rare severe ones.
- The correct version of this instinct: ship **behind a flag** to a small cohort, with tracing and
  operational evals already in place, and harvest the failures into the golden set. Production becomes
  a data *source*, not the first evaluation.
- Nothing about this requires a large upfront investment — error analysis on 30–50 traces plus tracing
  is days, not quarters.

**Red flag:** agreeing that shipping first is fine because "real data is better".

### Q36. *(Trap)* "Our app is a classifier, so we don't need evaluation infrastructure."

**The naive answer:** agree — classification is a solved problem with known metrics.

**What they're really testing:** whether you notice the hidden pipeline.

**Model answer:**
- It is true that a fixed-schema classifier needs no **judge** — and reaching for one would be a
  mistake.
- But "no judge" is not "no evaluation infrastructure." You still need a **golden dataset** (50–500
  rows), a **baseline**, a **CI gate** with a threshold above the noise floor, **per-class** metrics
  rather than overall accuracy, and a **schema-validity** check on every call.
- You also need the operational band: latency, cost per query, error rate split by cause.
- And the harvesting loop: production misroutes become dataset rows.
- The general principle: the eval *method* gets simpler for a classifier; the eval *discipline* does
  not. Teams that conflate the two end up with no gate at all.

**Red flag:** concluding that simplicity of method means no infrastructure needed.

### Q37. *(Trap)* "We'll use the same LLM to generate the dataset and to judge the outputs."

**The naive answer:** efficient — one model, one integration, no human cost.

**What they're really testing:** whether you understand correlated failure.

**Model answer:**
- Two distinct problems. First, a **synthetic dataset generated by the model under test** inherits
  that model's blind spots and biases — the failures it cannot imagine are exactly the failures it
  will not produce.
- Second, a **judge from the same family as the generator** exhibits self-preference bias: it rates
  its own outputs higher, inflating scores.
- The deeper issue is that **correlated errors are invisible**: if the generator and the judge are
  wrong in the same way, agreement looks like correctness.
- The fix is not "never use synthetic data" — dimension-based synthetic generation is a legitimate way
  to manufacture diversity. It is that synthetic data must be **reviewed by a human** before it becomes
  ground truth, and the judge must be **validated against human labels** (TPR/TNR) on held-out data.
- Minimum viable discipline: ~100 human-labelled examples, a held-out split, and a different model
  family for judging than for generating where feasible.

**Red flag:** accepting the efficiency argument; never mentioning human review or validation.

### Q38. *(Trap)* "Our eval suite runs in 40 minutes and costs $30. Is that fine?"

**The naive answer:** "sure, that's cheap."

**What they're really testing:** whether you connect eval cost to iteration speed.

**Model answer:**
- It depends entirely on **how often you want to run it**, and the answer is usually "not as often as
  you should."
- $30 and 40 minutes is fine **nightly** and pre-release; it is fatal **per commit**, because
  developers will route around it.
- The fix is not to shrink coverage but to **tier by cost** — this is the design decision, not a
  budget problem:
  - free tier: operational evals and code evaluators, every commit;
  - cheap tier: a stratified judge subset, every commit;
  - expensive tier: the full golden set and the pipeline metrics, nightly and pre-release.
- Then attack the expensive tier's unit cost: cache identical judge inputs, use cheaper models for
  cheaper criteria, and sample **deterministically** so retries do not re-score different items.
- Also worth asking: is $30 of that spend buying decisions? A judge that has never changed a decision
  is pure cost.

**Red flag:** accepting the number without asking the cadence question.

---

## Live-coding / whiteboard prompts (3)

### Prompt 1 — Design a golden dataset for a routing classifier (40 min)

> A support system reads an incoming email and routes it to *billing*, *technical*, or *customer
> support*. Design the golden dataset you would build to evaluate it, and state the metrics you would
> compute.

**What a strong answer covers:** a two-column artefact (email text, human-assigned label); a source
strategy led by **historical traffic**, human-labelled; a size grounded in the 50–500 row range and
justified by the effect size you need to detect; a stratification plan so rare classes and hard cases
are represented; the metrics: **overall accuracy, per-class precision/recall, and a confusion matrix**
rather than accuracy alone; the business-weighting question (*which misroute costs most?*); and a
**schema-validity** check for invalid outputs as a separate failure class. Bonus: the plan for growing
it — production misroutes become rows.

**Red flags:** accuracy alone; a randomly sampled set; no human labelling; no notion of class
imbalance.

### Prompt 2 — Draw the complete evaluation workflow (30 min)

> Whiteboard the end-to-end process from "we have an idea for an LLM feature" to "it is running in
> production and improving."

**What a strong answer covers:** define task/target → define success criterion → build dataset →
choose eval method → run → evaluate → analyse (error analysis) → improve → iterate → deploy → monitor
→ harvest failures into the dataset → repeat. The two things graders look for: that **the criterion
comes before the dataset**, and that the loop **closes** — deployment feeds the dataset. Bonus:
annotating where the eval method branches (code / human / judge) and where the offline/online boundary
sits.

**Red flags:** starting with metrics; a linear diagram with no feedback loop; deployment as the
terminal node.

### Prompt 3 — Choose an evaluation method (30 min)

> For each of the following, say whether you would use automated code checks, human review, or an LLM
> judge, and why: (a) an email router's label; (b) whether a RAG answer is grounded in its retrieved
> context; (c) whether a summary is "well-written"; (d) whether a JSON response matches the required
> schema; (e) whether a support reply is appropriately empathetic.

**What a strong answer covers:**
- (a) **automated** — discrete gold label, accuracy + confusion matrix.
- (b) **validated judge** — subjective and semantic; must be validated with TPR/TNR; measure
  faithfulness.
- (c) **judge**, or **pairwise human** for calibration — genuinely graded quality; pairwise comparison
  is more reliable than an absolute score.
- (d) **automated** — this is a schema check; a judge here is waste.
- (e) **judge, validated against human labels** — tone is subjective, but the label must be made
  concrete (e.g. "acknowledges the user's problem before offering a fix") or the judge is
  unvalidatable.
- The overarching rule stated explicitly: **can code check it? → code. Then judge. Then human on a
  sample.**

**Red flags:** judges for (a) and (d); no validation step for (b) or (e); no concrete label definition
for tone.

---

## Take-home / case-study prompt (1 full brief)

> **Brief.** A 12-person team has shipped an LLM-powered internal knowledge assistant (RAG over ~5,000
> policy documents, ~800 queries/day). They have no evaluation beyond "it seems okay" and a thumbs-up
> count in the UI. Two weeks ago a policy document was updated and nobody noticed whether answers
> changed. Engineering wants to refactor the retrieval layer and is blocked because they cannot tell
> whether the refactor helps.
>
> **Deliverables (≤6 pages):**
> 1. **Application shape classification.** Which of the five application shapes is this, and what does
>    that imply for the metric families you will use?
> 2. **The first 30 days.** A concrete week-by-week plan. Justify the ordering, and say explicitly what
>    you will *not* build in month one and why.
> 3. **The evaluation method choices.** For each metric you propose, say whether it is an automated
>    code check, a human review, or an LLM judge — and justify that choice by the output's shape.
> 4. **The golden dataset plan.** Source, size, labelling process, how you handle the stale-policy
>    problem, and how the set grows.
> 5. **The gate.** What CI blocks a deploy, what the threshold is, and how you derived it.
> 6. **The model-versus-application split.** Where does a custom model eval fit, if at all? What
>    decision would it inform that application evals cannot?
> 7. **The cost.** Four cost lines with an estimate, and the first thing you would cut if the budget
>    halved.
> 8. **What you will tell leadership in week four** — in plain language, with the decision they own.
>
> **What is being assessed:** whether the plan starts with data rather than tooling; whether the
> eval-method choice is derived from output shape rather than fashion; whether the gate is
> statistically grounded; and whether the plan closes the loop from production back into the dataset.

---

## Scoring rubric — what separates a hire from a no-hire

| Dimension | No-hire | Mid-level hire | Senior hire |
|---|---|---|---|
| **Model vs application** | Uses the terms interchangeably | States the distinction | Explains that they serve different decisions, cadences and owners, and that a leaderboard cannot justify a product decision |
| **Workflow fluency** | Names metrics | Recites the workflow in order | Recites it *and* says where it breaks — criterion before dataset, loop must close |
| **Eval-method choice** | Reaches for a judge | Picks a method per output shape | Applies "code → judge → human" and explains why a judge on a schema check is waste |
| **Dataset thinking** | "We'll generate one" | Knows 50–500 rows and human labelling | Adds noise floor, stratification, a private hold-out, and the harvesting loop |
| **Cost awareness** | No cost dimension | Knows humans are expensive | Names four cost lines and tiers the suite by cadence so it survives contact with CI |
| **Production instinct** | Stops at deploy | Mentions monitoring | Tracing, drift windows with numbers, and offline/online using the *same* evaluators |
| **Prioritisation** | Measures everything | Measures what users complain about | Ranks by frequency × severity × fixability and will argue for measuring less |
| **Statistical honesty** | Quotes accuracy | Knows sample size matters | Reports TPR/TNR, states the noise floor, and refuses to over-claim from a small sample |
| **Communication** | Metric dump | Plain-language explanation | Leads with the decision, expresses thresholds as commitments, and names what is not measured |

**The single strongest signal at this level:** the candidate says that evaluation is **not one
activity** — that selecting a model, testing a component, testing a pipeline, and monitoring a product
are four different jobs with four different datasets — and then places whatever metric is under
discussion into the correct one.
