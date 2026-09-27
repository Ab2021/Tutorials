# Interview Pack · Production, Online & Operational Evaluation

> **Covers:** the `06-production` domain — CS-20 (evals that thrive in production — the harness as the
> evaluation target, the HHH+R frame, task completion / adherence / navigation, the tool metrics, and
> the ROI-first cost target), CS-21
> (observability, traces, alerts, red teaming), CS-22 (scaling evals), CS-23 (pricing AI agents — the
> value-first pricing order, attribution × autonomy, ceiling cost, credits, the 80% rule and the eval
> bill as a COGS line), CS-24
> (tooling as an eval accelerator) — together with the production half of CS-06 (logging → signals →
> dashboard → alert) and CS-15 (operational evals: faster, cheaper) · **Code ground truth:** `CODE-06`
> (Langfuse — traces, scores, experiments, prompts, four storage planes), `CODE-07` (search_evals —
> decimal cost accounting, agent/grader cost split) · **Role level:** senior → staff · **Format:**
> screen → onsite → system-design

---

## How this domain is assessed

This is the domain that separates people who have *run* an LLM product from people who have *built
demos*. Everything upstream — datasets, judges, benchmarks — is preparation. Production evaluation is
where the preparation meets real traffic, real money and real users, and where the questions are
suddenly about sampling, alerting, cost, PII and incident response rather than about prompt design.

The organising thesis, and the thing to say early: **a score is only actionable if it sits next to the
latency and cost of the same interaction.** Quality, price and speed are one measurement, not three.
A candidate who separates them has not operated a system.

The second framing, from CS-20, is **what the evaluation is pointed at**. You are not evaluating a
model and you are not evaluating a tool — you are evaluating a **harness**: the LLM plus its context
(knowledge and memory), its tools, its skills and its guardrails. That is the object that goes to
production, so that is the object the eval must describe. It also settles a question candidates
fumble: when someone says "we want to evaluate our link-validator tool," the answer is to wrap it in
an agent and evaluate **the harness**, then infer the tool's quality from it.

The third framing is **ordering by cost**. Among the dozens of evaluators a platform will offer you,
run the cheap objective gates before the expensive subjective ones — task completion, then adherence,
then navigation, then quality, then context, then guardrails and cost. Each layer is cheaper than the
next, and a failure at an early layer makes the later ones uninterpretable.

The fourth thing this domain tests is **restraint**. The strongest candidates argue against
evaluating everything, against alerting on single traces, and against the dashboard that nobody reads.

| What is being scored | The signal | How it is elicited |
|---|---|---|
| **Trace-first thinking** | Do you build the record before the metric? | "Where does the number come from?" |
| **Harness identity** | Do you evaluate the harness, not the model or the tool? | "What exactly are you scoring?" |
| **Metric ordering** | Cheap objective gates before expensive subjective ones | "Which evaluator do you run first?" |
| **Sampling discipline** | Deterministic *and* stratified, not random | "How do you choose what to evaluate?" |
| **Alerting judgement** | Window, threshold, owner, remediation | "How do you set up drift detection?" |
| **Economics** | Cost per query as a first-class metric | "What does this cost to run?" |
| **Operational honesty** | Percentiles with timeout rate; samples with n | "How fast is it?" |
| **Loop closing** | Production failure → dataset → fix → release | "What happens after an incident?" |

**The single most predictive question in this pack** is *"your dashboard is green and users are
complaining — what do you do?"* A strong candidate immediately asks what the dashboard is *not*
measuring: coverage, segmentation, timeout rate, cost, and the failure classes nobody wrote an
evaluator for.

---

## Tier 1 — Fundamentals (10 Q)

### Q1. What is production evaluation, and how does it differ from offline evaluation?

**What they're really testing:** whether you have the split precise, not approximate.

**Model answer:**
- **Offline** evaluation runs an eval pipeline over the application *before* deploy, against a frozen
  golden dataset with known answers. It gates releases, compares variants one variable at a time, and
  catches regressions.
- **Online** (production) evaluation runs on **live production traffic after deployment, as real users
  interact with it.** Its defining feature: **it works without answers and without a golden
  dataset.**
- The precise framing: **offline checks whether the application works *correctly*; online tells you
  whether it is running *normally*.** They are complementary, not rivals.
- The constraint that follows: **online evaluation cannot measure correctness**, because no ground
  truth arrives with live traffic. What you compare is today's *distribution* against a **baseline
  that offline evaluation produced**. Baseline 85 with 87 observed is fine; 75 in the last 24 hours is
  an alert.
- Production evaluation exists because three things only appear live: **unanticipated inputs** (your
  golden set held 500 anticipated questions; production brings code-switching, ambiguity, angry rants
  that hide the question, and prompt injection), **emergent failures that only appear at scale** (a
  launch brings concurrent thousands and a latency spike that cannot be reproduced on a laptop), and
  **drift** (prices, curricula, policies and documents change; the golden set silently goes stale
  while offline scores still look healthy).
- They must share the **same evaluators**, or the offline number stops predicting production.

**Red flags:** "we evaluate before launch"; equating online with "monitoring uptime"; believing an
online judge can measure correctness; no baseline concept.

### Q2. What is a trace, and why is it the unit of production evaluation?

**What they're really testing:** whether you build the record before the metric.

**Model answer:**
- A **trace** is one interaction recorded end to end: the input, the output, the **retrieved
  documents**, the model and prompt versions, latency, token counts, cost, and errors — plus, for an
  agent, every step inside it.
- A **span / observation** is a step *within* a trace — an LLM call, a retrieval, a tool call. Spans
  are what let you evaluate a sub-component (retriever quality) rather than only the end-to-end
  result, and they are where you localise a failure.
- The trace is the unit because **everything else attaches to it**: scores, human feedback, dataset
  membership, experiment membership, prompt versions. That attachment is what turns a pile of metrics
  into a system where **a bad trace leads directly to the prompt change that fixes it**.
- It is also the **join key for late signals**. A signal that arrives after the conversation — an
  escalation email a day later, a thumbs-down submitted on the next screen — must be attached back by
  a stable identifier. Without conversation and session IDs preserved end to end, the signal is lost.
- And it is what makes the **self-improving loop** possible: a production failure is added to the
  golden dataset from the trace, the offline suite re-runs, and the fix is gated.

**Red flags:** logging only the input and output; no per-step spans; no stable IDs for join keys; a
"quality dashboard" not built from real interactions.

### Q3. Captured signals versus computed signals.

**What they're really testing:** whether you know which half of your metrics are free.

**Model answer:**
- **Captured signals** already exist in the interaction — you only have to store them. Thumbs up/down,
  latency in milliseconds, prompt and completion tokens, cost per conversation, error flag and status
  code. No model, no judge, no cost.
- **Computed signals** must be calculated by building an **evaluator**: faithfulness, answer
  relevance, correctness, hallucination, toxicity, bias and fairness.
- The distinction is an engineering priority order. **Captured signals are free, deterministic and
  always available; computed signals cost money, vary run to run, and need validation.** Build the
  captured half first — a large share of production incidents are visible in latency, error rate and
  cost alone, with no judge involved.
- The trap: teams reach for the judge first because quality feels like the interesting problem, and
  end up with an expensive, noisy metric while the free deterministic ones sit uncollected.
- Note also that most computed signals are not LLM-specific — the source's caution is that LLM-based
  software "is not special" here; error rate, latency and cost are ordinary software concerns that the
  LLM world sometimes forgets it still has.

**Red flags:** no captured-signal layer; judging before logging; treating cost and latency as
infrastructure rather than product metrics.

### Q4. What is drift, and how do you detect it?

**What they're really testing:** whether you have a detector or just an intention.

**Model answer:**
- **Drift** is a metric degrading relative to the world — not because your code changed, but because
  the inputs, documents, prices or user behaviour changed underneath you. Your golden dataset and your
  whole eval pipeline silently become obsolete, while the offline score still looks healthy and users
  start complaining.
- It is detectable **only** by a windowed online metric. Offline cannot see it by construction, because
  offline replays the frozen dataset.
- A working detector has three parts, and a dashboard alone is not one:
  - **A rolling window on one metric.** A 24-hour rolling graph on a single metric (for example
    faithfulness), with the alert firing on degradation over the **last 8 hours**. The window matters:
    too short and variance triggers it, too long and you learn about the regression a day late. Pick it
    against the metric's own noise floor.
  - **A threshold**, expressed relative to the offline-derived baseline.
  - **A remediation path** — an owner and a defined response. An alert without one is noise, which is
    how alert channels get muted.
- Distinguish drift from **instrument drift**: the judge model being silently upgraded, or a judge
  prompt changing. Keep a **canary set** of frozen inputs with stored verdicts; if the verdicts move
  with no code change, the instrument moved, not the system.
- Drift is also the reason the **golden dataset must keep receiving production failures** — a static
  dataset drifts away from reality while continuing to look green.

**Red flags:** "we'd notice from user complaints"; no window; no threshold; no owner; not
distinguishing system drift from judge drift.

### Q5. Why must production sampling be deterministic?

**What they're really testing:** whether you have debugged a flaky metric.

**Model answer:**
- **Deterministic sampling** means membership in the evaluated set is a function of a stable trace
  identifier — a hash of the trace ID against the sampling config — not a random draw at request time.
- The reason: with a `Math.random()` sampler, a **retried or replayed run scores a different set of
  items**, so two runs of the same configuration are not comparable. You lose the ability to attribute
  a movement to anything.
- It also **doubles your spend on retries**, since the same logical interaction can be judged twice
  while other traces are never judged at all.
- Determinism makes the metric reproducible: given the config and the trace store, anyone can
  reconstruct exactly which traces were evaluated.
- Practical discipline: **record the sampling configuration with the run**, alongside the judge
  version and prompt version, so a historical number can be reproduced and explained.
- Determinism is a separate property from **stratification** — see the next question. You want both:
  membership stable across replays, *and* the set chosen to include the cases worth looking at.

**Red flags:** random sampling per request; no sampling config recorded; not realising retries
re-score different items.

### Q6. Random sampling versus stratified sampling.

**What they're really testing:** whether you know that fairness is not the same as usefulness.

**Model answer:**
- Random sampling is the wrong default. The principle, stated flatly in the source: **"not all
  conversations are the same."** Random sampling spends most of your budget on conversations that were
  fine.
- **Stratified sampling** divides conversations into categories first, then draws more samples from
  the problematic categories. Which categories count as problematic is a product judgement, but the
  list is fairly universal: **thumbs-down**, conversations that **ended abruptly**, **escalations**,
  **repeated rephrased questions** (the user asked again because the first answer failed), and
  anything touching **money — refunds, fees, pricing, admission**.
- Implementing it usually needs a small **classifier** to bucket conversations, which is effectively
  clustering. That classifier is cheap and deterministic, and it pays for itself immediately.
- The two properties compose: **stratify to choose the set, then sample deterministically within each
  stratum** so replays are comparable.
- One caveat worth stating: pure stratification means you lose the unbiased estimate of the *overall*
  rate. Keep a small uniform random sample alongside the boosted strata so you can still report an
  unbiased headline number — evaluate the boosted sample for *diagnosis*, the uniform one for
  *reporting*.

**Red flags:** "we sample 1% at random"; no strata; no awareness that boosting a stratum biases the
headline rate.

### Q7. What is an annotation queue, and why is it load-bearing?

**What they're really testing:** whether you know where human labels come from in a live system.

**Model answer:**
- An **annotation queue** routes a sample of traces to humans for review, with a place to record what
  was right and wrong. It is where human labels are produced continuously, rather than in a one-off
  labelling project.
- It is load-bearing for three reasons:
  1. **It is the source of judge validation.** Every judge needs TPR/TNR against human labels on
     held-out data. Without a queue, that validation is a project you keep not starting.
  2. **It is the source of the golden dataset.** Labelled traces with a written critique become
     golden-set rows and few-shot examples directly.
  3. **It is the only instrument that can adjudicate the cases the machines disagree on.** When the
     judge and a code check disagree, the queue is where you find out who is right.
- Routing should be **targeted, not uniform**: send the ambiguous cases (judge low confidence, two
  judges disagreeing), the high-stakes ones, a uniform sample for calibration, and anything flagged by
  a user.
- Design details that matter in practice: **crisp guidelines plus worked critiques** (vague guidelines
  produce low-agreement labels, which caps the judge you can validate), and **measuring
  inter-annotator agreement** so you know your ground truth is actually ground truth.
- **The hard part is getting an answerable question out of an expert, and CS-22 has the method.** You
  do not ask a lawyer "is this clause good?" — you cannot label that, and neither can a judge. You ask
  **why** they think it is good, and convert the reason into a **yes/no question about a fact**. Their
  illustration: a lawyer says "the indemnity should have insurance in it"; the checkable question
  becomes "**does the indemnification clause have insurance or not?**" That sentence is the whole
  technique — **subjectivity is not eliminated, it is *decomposed* until what remains is checkable.**
  Two consequences: the labelled set becomes reproducible, and the resulting criteria are cheap enough
  to automate later.
- **And when the task is find-the-items, the right metrics are precision and recall, not accuracy.**
  CS-22's worked example on contract key-term extraction, and the numbers are worth being able to
  reproduce:
  - The lawyer names **10** key terms. The agent names **12** — all 10 correct ones plus **2** extras.
    **Precision = 10 / 12 ≈ 83%** (of what it flagged, how much was right → penalises **false
    positives**).
  - In the other direction, the agent finds only **8** of the lawyer's 10.
    **Recall = 8 / 10 = 80%** (of what should have been found, how much was found → penalises **false
    negatives**).
  - **Accuracy alone would have hidden both errors**, and the two failures have opposite product
    consequences: a false positive sends the user a clause that was never there; a false negative
    silently drops a risk. **Which one you can tolerate decides which metric you optimise**, and that
    is a domain judgement, not a statistical one.

**Red flags:** no annotation path; labelling everything; no guidelines; no agreement measurement;
treating human labels as noise-free; asking experts for quality judgements instead of decomposing them
into yes/no facts; optimising accuracy on a find-the-items task.

### Q8. Your P95 latency improved. Should you celebrate?

**What they're really testing:** whether you publish a percentile without its timeout rate.

**Model answer:**
- No — not until you look at the **timeout rate**. Timed-out requests are excluded from the latency
  distribution, so a "faster" P95 can simply mean you are now **dropping more requests**. The
  distribution improved because the slow tail got cut off, not because the system got faster.
- The rule: **never publish a percentile without the timeout rate and the sample size.** All three, or
  the number is not interpretable.
- Also publish the shape, not one number: mean, median, P50, P95, P99, min, max. A mean alone hides a
  bimodal distribution where most requests are fast and a meaningful minority are unusable.
- Separate **TTFT (time to first token)** from total latency. With streaming, **the user experiences
  TTFT** — 1.6 s to first token feels different from 3.6 s total, even though the totals may match. A
  product can have an excellent total latency and a terrible perceived latency, or the reverse.
- And put the percentile next to **quality and cost** for the same interaction. A latency win that
  halves quality is not a win, and you only see that if the numbers share a screen.
- For agents, latency is also a *behavioural* metric: step count and retry count drive it, so a
  latency regression is often an agent-efficiency regression.

**Red flags:** a P95 with no timeout rate; mean-only reporting; TTFT conflated with total latency;
latency on a different dashboard from quality and cost.

### Q9. How do you compute and report cost per query?

**What they're testing:** whether you treat cost as a measured quantity.

**Model answer:**
- The basic arithmetic, per call:
  `cost = (input_tokens / 1e6 × input_rate) + (output_tokens / 1e6 × output_rate)`
- **Split input from output** — output tokens typically cost around **4×** input, so a change that
  shifts the balance between them moves the bill in a way a single total hides.
- Track it as a **distribution, not a total**. A typical cheap query with a small number of expensive
  outliers is a different business from a uniformly expensive one, and the mean conceals which you
  have.
- Express the budget as a **per-query ceiling** — for example, a stated ceiling on a complex query —
  because that is the number a product decision can act on. "We spent $40k last month" is not
  actionable; "the expensive path costs 6× the cheap path and runs on 12% of traffic" is.
- **Use exact decimal arithmetic** for aggregation, not floats. Floating-point drift compounds across
  a long multi-step agent run and makes per-component attribution wrong.
- **Separate the lines**: the system under test, the judge/grader, and any retrieval or reranking
  infrastructure. You cannot optimise what you have not separated, and grading cost is easy to forget
  because it never appears in the product.
- **Instrument the eval suite's own spend** as a first-class number. An eval harness whose cost is
  invisible is an eval harness that grows without limit.
- Two traps: identical repeated inputs inflate provider **prefix caching**, so measured cost
  **understates** real cost; and cost measured on a handful of samples from a developer machine will
  not match production throughput.

**Red flags:** a single monthly total; float arithmetic; agent and judge cost pooled; no per-query
ceiling; no cost dimension on the quality dashboard.

### Q10. You have judge scores, code checks and human labels. Should they all be the same thing?

**What they're testing:** whether you have one score contract or three incompatible ones.

**Model answer:**
- No — but they must all attach to the **same object** through **one contract**, so a score is a
  first-class event with a name, a value, a source, a timestamp and an optional comment. The value is
  the *only* thing that varies by source.
- The three sources have genuinely different properties, and treating them as interchangeable is the
  error:
  - **Code checks** are deterministic, free, and always right about what they measure — but they
    measure a narrow slice (schema validity, required fields, forbidden strings, tool arguments).
  - **Judges** are flexible and cover subjective quality — but they are noisy, cost money, and are
    only meaningful after validation against human labels.
  - **Human labels** are the highest-quality signal and the ground truth your judges are measured
    against — but they are slow, expensive, and themselves noisy (measure inter-annotator agreement).
- The practical consequence of one contract: you can put a code check, a judge score and a thumb on the
  same trace and compare them, which is how you discover that your judge disagrees with your code
  check on a particular failure class.
- And it gives you the ordering rule for the whole system: **code → judge → human**, in increasing
  cost, stopping at the cheapest level that can answer the question. A human reviewing something a
  boolean could have settled is the most expensive mistake in the stack.
- **CS-22 gives the measured version of why judges must be validated first, and it is the strongest
  evidence in the pack.** When the team computed the **precision and recall of their LLM judge**
  against the lawyers' labels, both came in **below 40%** — which the speaker calls "a **flip of a
  coin**." A judge that is wrong more often than it is right is not a cheap substitute for a human
  label; it is a random number generator wearing a rubric. Note what this costs a team that skips
  validation: they would have shipped a gate that blocked good releases and passed bad ones, and
  nothing in the dashboard would have said so.
- **So state the deployment bar as a number, and hold it.** CS-22's rule: an LLM judge may **replace**
  the human expert — "**you can replace this because this can run at scale**" — only once it clears
  roughly **80% precision** (restated as **70%** when the speaker is challenged in Q&A). Below that,
  the instruction is explicit and worth quoting: "**don't do it. Still do whatever the lawyer is
  saying and hire the lawyers. Don't compromise your product because you don't have money.**" That is
  the sentence to bring to the interview: **the alternative to a validated judge is not a cheap judge,
  it is paying the expert.**
- The **escalation ladder** is the framework to draw, and it is the answer to "how do I get from
  humans to scale":
  | Rung | Method | Cost | When |
  |---|---|---|---|
  | 1 | **Human expert** (yes/no against ground truth) | Highest | Always works; start here |
  | 2 | **Precision / recall** against that ground truth | Medium | Once the expert has labelled a set |
  | 3 | **Similarity / coherence** (BLEU, ROUGE) | Low | When a reference answer exists |
  | 4 | **LLM as a judge** | Low | **Only after it passes its own precision/recall test** |
  | 5 | **Scale** | Lowest | Only above the judge bar |
  The rule that makes the ladder safe: **you move up a rung only when the rung you are leaving has
  measured the rung you are entering.** Skipping from 1 to 4 is how teams end up with a coin-flip
  judge and a green dashboard.

**Red flags:** separate dashboards per source; a judge score treated as ground truth; no code-check
layer; human review used where an assertion would do; deploying a judge without measuring its
precision and recall; assuming a cheap judge is better than no judge.

---

## Tier 2 — Applied / trade-off (10 Q)

### Q11. How do you decide what to evaluate in production?

**What they're testing:** whether you understand that targeting is the whole game.

**Model answer:**
- You never evaluate everything, and you should not want to. The framework is **targeting**: a
  **decision model** that maps a trace to the set of evaluators that apply to it, based on properties
  of the trace.
- The properties worth targeting on:
  - **Product surface** — a retrieval path needs faithfulness and context relevance; a tool-calling
    path needs argument validity and tool-selection accuracy; a classifier needs label accuracy.
  - **Risk class** — anything touching money, safety, legal or PII gets the expensive evaluators and a
    human in the loop; a low-stakes FAQ answer does not.
  - **Signal class** — thumbs-down, escalation, abrupt ending, repeated question. These get judged
    heavily because they are where the failures are.
  - **Novelty** — a query cluster the system has not seen before gets judged, because that is where
    emergent failures appear.
  - **Cost tier** — cheap deterministic checks on 100% of traffic; the expensive judge on a sampled
    slice.
- The payoff is that targeted evaluation gives **more decision value per dollar** than blanket
  evaluation, and it produces interpretable numbers ("faithfulness on refund conversations") rather
  than one undifferentiated average.
- Say the arithmetic out loud to justify it: a judge at a couple of cents per call on 50,000 daily
  interactions is around **$1,000 a day** to observe. That is a product decision, and targeting is how
  you make it affordable without giving up coverage of what matters.
- **Within the targeted set, order the evaluators by cost.** CS-20's rule, and it is the cleanest
  answer to "there are 30 or 50 evaluators, where do I start": run the cheap objective gate first, and
  only read the expensive subjective ones on what passes. The order is **task completion**
  (binary — did the job get done at all) → **task adherence** (did it follow its constraints without
  waste) → **task navigation** (was the plan right, and was it followed) → **quality** (the
  domain rubric, authored with an expert) → **context** (retrieval, groundedness, completeness,
  relevance) → **guardrails and cost**. The reason is not just economy: a failure at an earlier layer
  makes the later layers uninterpretable. There is no point scoring helpfulness on a task that never
  completed.
- Note the honest gap in the field, and report it as one. When the audience asked CS-20's speaker for a
  prioritisation rubric across the evaluator catalogue, he did not have one and said so — the ordering
  rule above is the partial answer he did give, and the full minimum/medium/large tiering was promised
  and never delivered. A candidate who claims a crisp universal priority order is overclaiming; the
  defensible claim is that the order is **by cost and objectivity**, and that your specific ordering
  depends on where your failures are.

**Red flags:** "we judge everything"; one evaluator for all traffic; no risk classing; no cost tiering;
scoring quality before checking whether the task completed.

### Q12. How do you set an alert that people will actually respond to?

**What they're testing:** whether you have operated an on-call rotation.

**Model answer:**
- Four required parts, and an alert missing any of them is noise:
  1. **One metric.** Not a composite score. A single named quantity, so the page is unambiguous.
  2. **A window.** Aggregate over a window — a rolling 24-hour graph with the alert firing on
     degradation over the last 8 hours is a reasonable shape. The window must be set against the
     metric's noise floor: too short and variance pages you at 3am, too long and you learn about a
     regression a day late.
  3. **A threshold** relative to the offline-derived baseline. Not an absolute number invented at the
     keyboard.
  4. **An owner and a remediation path.** What the responder is supposed to do. An alert with no
     defined response is a notification, and notifications get muted.
- Alert on **windowed aggregates, never on a single conversation.** One bad trace is noise; the
  aggregate over the last hour is the signal.
- **Set the threshold above the noise floor.** If the metric swings 3 points between two runs of
  identical settings, a 2-point threshold produces flapping, and flapping produces muted channels.
  Measure the noise floor first, then set the threshold.
- Tier the severity: a paging alert for a hard quality or availability break, a ticket for a slow
  drift, a dashboard-only line for something you are watching.
- And review the alerts periodically. **Alert fatigue** is a failure mode of the alerting system, not
  of the responders, and the fix is deleting alerts nobody acts on.
- **Concrete thresholds make the answer credible**, and CS-21 gives a usable agent set. Alert when:
  - **ground-truth accuracy drops below 30%** — a floor, not a target; below it the agent is not
    reliably doing the job at all;
  - **tool-calling failure rises above 40%** — the agent's actions are broken, which is a different
    fault from a bad answer;
  - **guardrail triggers exceed 5%** — the agent is refusing or blocking too often.
  Say what those numbers are *relative to*: they are floors and ceilings on the failure side, and the
  healthy side needs its own targets — CS-21's stated launch gate is **60–75% helpfulness** before
  releasing to **1% of customers**. A floor and a gate are different controls, and a mature answer has
  both.
- **The over-triggering alert is the one teams forget**, and it has a famous counter-example. Google's
  first support agent was so policy-bound that **six out of ten customers were simply denied access** —
  not by errors or timeouts, but by the agent saying "I can't answer this question, it's against my
  policy." Every health metric was green, because the failure produced a valid response. This is why a
  guardrail needs an alert on its **trigger rate**, not only on its breaches: **both under- and
  over-triggering are failures of a guardrail**, and only one of them looks like a failure in the logs.

**Red flags:** thresholds with no baseline; alerting on single traces; a composite quality score as
the page condition; no owner; no noise-floor measurement; never pruning; no alert on guardrail
**over**-triggering.

### Q13. How do you handle PII in production traces?

**What they're testing:** whether you have shipped something with real user data.

**Model answer:**
- Trace stores are a **new and often unowned copy of user data**, and they accumulate: every
  conversation, every retrieved document, every tool argument. Treat the trace store as a
  data-protection surface with an owner, not as a debugging convenience.
- The core control: **mask or blur PII before storage**, at write time — phone numbers, card numbers,
  dates of birth, government identifiers. Masking on read is not equivalent: the sensitive value is
  still at rest and still exfiltratable.
- Practical layering:
  - **Detect** with a combination of pattern matching for structured identifiers and a classifier for
    the unstructured cases.
  - **Mask at the logging boundary** — the logging path is the single choke point where every trace
    passes, so it is the right place for the control. Non-blocking logging must still be *ordered*
    after masking.
  - **Restrict access by role**, so that the full unmasked record is available to the few who need it
    under audit, and the masked record is the default.
  - **Retention limits** — set a retention period and enforce it, because the risk grows with the
    store, and long-horizon debugging argues for keeping more than you should.
- There is a real tension to name: masking degrades the trace's usefulness for debugging, and retrieval
  content that gets masked may make a faithfulness score meaningless. The resolution is usually
  **tiered storage** — masked by default, with a separately controlled, access-audited path for the
  rare case that needs the original.
- And the uncomfortable one: **the judge is a third party**. Any computed signal that sends trace
  content to an external model API is a data-egress path and belongs in the same review as the rest.

**Red flags:** no masking; masking only on read; unlimited retention; unmasked content sent to a judge
API without review; the trace store owned by nobody.

### Q14. How do you close the loop from a production failure to a fix?

**What they're testing:** whether your eval system has an output other than a number.

**Model answer:**
- The loop is the point of the whole system, and it has a specific shape:
  **offline eval → deploy → production failures → add those conversations to the offline dataset →
  re-run offline eval → redeploy → new failures → repeat.** That is the **self-improving loop**.
- The mechanics that make it real rather than aspirational:
  - **Add to dataset** from the trace view, so a bad conversation becomes a golden-set row without a
    separate data-engineering task.
  - **Annotation queue** for recording what was right and wrong, which produces both the label and the
    written critique.
  - **Prompt versioning**, so the fix is a versioned object rather than an edit — a bad trace leads to
    the playground, to a version, to an experiment, to a release.
  - **An experiment** comparing the candidate config against the current one on the frozen set, with
    the CI gate deciding whether it ships.
- The release discipline matters: **treat a prompt change as a release.** A prompt edit that is not
  evaluated and not versioned is an unmonitored deploy.
- What the loop must not become: a golden set that only ever grows from incidents and never gets
  rebalanced, until it is a museum of past bugs with no coverage of the current traffic.

**Red flags:** a fix that ships without a dataset row; no prompt versioning; an annotation queue that
feeds nothing; a golden set that never gets pruned or rebalanced.

### Q15. How does production evaluation differ for an agent versus a chatbot?

**What they're testing:** whether your production model scales to multi-step systems.

**Model answer:**
- The unit changes. A chatbot's trace is roughly one request-response; an agent's trace is a
  **trajectory** — a sequence of decisions, tool calls and observations, sometimes spanning minutes to
  hours. Per-turn averaging hides the failures that matter.
- New metrics appear, and they are operational as much as qualitative: **step count**, **tool-selection
  accuracy**, **argument validity**, **retry rate**, **recovery rate**, **loop detection**. A correct
  answer in 40 steps where 5 suffice is a cost and latency problem even though the quality metric is
  green.
- CS-20 draws the sharper version of this, and it is worth using because it separates three things
  candidates blur into "did it work":
  - **Task completion** — the objective *did the job get done*. Binary, cheap, automatable, and the
    numerator of any outcome-based pricing.
  - **Task adherence** — did the agent follow the instructions and constraints while doing it. This is
    the **cost and time control**, and it is the metric that catches the failure completion cannot see:
    a run that finishes the task while ignoring its constraints "is taking 60 minutes because I can't
    follow instructions while it's a six minutes task, and my cost is **10x** than what it should be."
    Bad adherence causes retry loops, and retry loops are where the margin goes.
  - **Task navigation** — was the *plan* correct, and was it followed correctly. Adherence asks "did I
    obey the rules"; navigation asks "was the plan right in the first place." Separating them is what
    lets you tell a planning failure from an execution failure.
- The tool surface becomes a metric family of its own, and every member has a cost consequence:
  **tool calling accuracy** (right tool, right arguments), **tool dispatch** (which tool was chosen for
  a step — "every tool call is a cost, double cost to you"), **tool parallelism** (were independent
  calls issued concurrently — this is latency and cost, directly), and **tool utilization** (did the
  agent actually *use* what the tool returned). The last is the one teams miss: a tool can be called
  successfully and its output silently ignored, which no call-level success metric detects.
- For retrieval-backed paths, the context metrics CS-20 names are **retrieval**, **groundedness**,
  **completeness** and **relevance** — with the explicit warning that relevance is checked while
  retrieval *volume* is not, and "am I retrieving too much... and that is costing me too much money
  downstream." Over-retrieval is quality-neutral and cost-negative, so it only shows up if you
  instrument tokens retrieved per query.
- Latency stops being a single number — an agent's P95 is a function of step count, and the tail is
  where the product breaks. **Timeout rate** becomes a first-class metric because agent runs hit
  timeouts rather than erroring.
- Cost becomes the dominant operational concern: **agent cost plus grader cost**, tracked per
  component, in decimal arithmetic, because a multi-step run's cost compounds.
- Evaluation needs **partial credit**. Binary pass/fail on a long run throws away all the information
  about where it went wrong, so world-state assertions per requirement plus trajectory metrics are
  what let you localise a regression.
- And the safety surface is larger: an agent takes **actions**, so a failure can be destructive rather
  than merely wrong. That argues for a deterministic guard layer in the production path (spend limits,
  allowlists, confirmation gates), separate from and in addition to the evaluation.
- Finally, name the object: for an agent you are evaluating a **harness** — LLM plus context (knowledge
  and memory), tools, skills and guardrails. That is why the metric list above looks the way it does,
  and why the same agent scored against a different harness is a different measurement.
- **Know where the harness came from, because it changes what you own.** CS-24 documents the 2025 shift
  from "build your own agent" to **adopt a harness**: context assembly, memory lifecycle, MCP-server
  authoring, guardrails and multi-agent routing stopped being the builder's job. The pieces supplied
  out of the box are **file-system access with plain search rather than embeddings**, the ability to
  **add files and folders**, and **memory + a browser plugin + connectors**. Two consequences for an
  evals engineer: (a) the retrieval you now depend on is **grep, not semantic search** — it trades
  recall for determinism, with no index to keep in sync and no ANN failure mode, but also no semantic
  generalisation, so measure its hit rate on a known-answer set exactly as you would a real retriever;
  and (b) **you cannot evaluate what you did not instrument**, so adopting a harness means establishing
  what telemetry the harness exposes before you assume you can measure the trajectory at all.
- **Do not mistake the harness's in-loop "evaluate" step for an eval suite.** CS-24's loop is
  `context → inspect sources → decide & act → evaluate → plan → "is the job done?"`, and the audience
  question about success criteria is answered with **the prompt, the skills file and the `CLAUDE.md`
  files** — a textual, human-authored **termination condition**, with no dataset, judge, threshold or
  golden answers. In-loop self-checking gives you **termination**; only an offline suite gives you
  **evidence**. The disciplined version of that in-loop step is trajectory evaluation (CS-17), and the
  measurement method is CS-20 to CS-22.

**Red flags:** judging an agent with chatbot metrics; no step or retry metrics; no timeout rate; cost
pooled; no production-side guardrails; treating task completion as sufficient without adherence or
navigation; no tool-utilization metric.

### Q16. How do you scale an eval suite so it stays useful?

**What they're testing:** whether you have been through the "the suite got too slow and got disabled"
cycle.

**Model answer:**
- Scale by **tiering the cadence to the cost**, so that fast feedback never blocks on the expensive
  layer:
  - **Per commit** — free deterministic checks only: schema validity, required fields, code-based
    assertions, unit tests. Seconds.
  - **Per pull request** — a sampled subset of the judge suite on a small representative set, plus the
    regression gate. Minutes.
  - **Nightly** — the full offline suite across all evaluators and the whole golden set.
  - **Pre-release** — the full suite plus execution-heavy evaluation (container rollouts, long-horizon
    agent runs) and red-team probes, which are hours and do not belong in CI.
- The rule that keeps it alive: **the gate must be above the noise floor.** A CI gate that flakes gets
  disabled within a month, and a disabled gate is worse than no gate because it produces false
  confidence. Measure the metric's spread across two runs of identical settings, then set the
  threshold above it — a gate that blocks on a drop larger than a defined number of units is workable;
  a strict equality gate is not.
- **Keep the suite's own cost and runtime visible.** An eval harness that takes an hour and costs
  real money will be run less and less, so instrument both and treat a regression in either as a bug.
- **Prune deliberately.** Evaluators that never move, and golden-set rows that every system passes,
  carry no information. Coverage of *observed* failure modes is the metric — audit it by taking last
  month's incidents and asking which the suite would have caught.

**Red flags:** one monolithic suite; no tiers; a strict-equality gate; no noise floor; no suite-level
cost tracking; never pruning.

### Q17. How do you price an agent product?

**What they're testing:** whether cost is a design constraint in your mental model.

**Model answer:**
- Do the costing in the right direction. The common mistake is to build the agent, measure what it
  costs, and then try to price it. CS-20's **ROI-first** rule inverts that: derive the target cost from
  the **displaced human's willingness to pay**, set the bar there, and then work backwards to whether
  the agent can be built inside it. His stated target is "**$1 or less than $1, ideally 50 cents** to
  run the agent" — and the worked example is a daily-briefs agent that costs **$5** to run when the
  beneficiary will pay **$1** ("it's a 2 minutes job for me... I want to give a dollar for this job and
  it costs you $5"). The verdict is: do not build it. Only after the ROI exists do "multi-agent, 20 tool
  calls, all that" become design questions.
- Note the direction of that constraint, because it is the reusable part: it is a
  **willingness-to-pay ceiling derived from the human alternative**, not a compute budget. Price from
  the cost of the work being displaced, not from your inference bill.
- Then build the cost model bottom-up, per unit of work:
  `cost_per_task = Σ (agent step costs) + grader cost + retrieval/reranking cost`
  where each LLM step is `(input_tokens / 1e6 × input_rate) + (output_tokens / 1e6 × output_rate)`.
- Then attach the **distribution** across tasks, not the mean, because agents are heavy-tailed: a
  small fraction of tasks consume most of the budget. The outliers are where the design work is. CS-23
  gives this rule a name and a threshold, and it is the most transferable idea in that session: price
  on **ceiling cost**, defined as the **average of the P80-and-above band**, never on the mean. The
  reason is cohort composition, not arithmetic — "the sales will sell to everybody who's willing to
  buy, and your first 100 customers will be hungry customers... and they will buy you and make you
  bankrupt." Early customers are the ones nobody else could serve, so they sit in the tail by
  construction. Mean cost is the margin story; ceiling cost is the pricing story. The worked ceiling:
  **$100 per run**, three hours maximum, five downstream agents, six services/tools/checks/guardrails/
  knowledge.
- **Set the price in this order, and do not reorder it.** CS-23's spine is: (1) **value created** =
  hours saved × an hourly rate the customer already pays — the worked figure is **6 hours saved at
  $200/hour = $1,200 of work**; (2) **attribution and autonomy**, decided honestly; (3) the **charge
  matrix** — a countable unit of the job, not a token count (one page = 500 words, a playbook of
  **50 checks**, the bundle is **one contract**); (4) a target **margin** (the worked example is
  **40%**); (5) **ceiling cost** across the cost buckets plus the hidden ones; (6) work **backwards**
  from the sellable price into eval depth, model size, agent complexity, tool restrictions and the
  accuracy you can afford. Step 6 is what makes this an eval question rather than a finance question.
- **Attribution × autonomy is the pricing map**, and it decides whether outcome pricing is available
  to you at all. **High attribution + high autonomy** — the agent finishes the job and you can stand
  behind the result — is "the holy grail" and permits outcome-based pricing. Almost everything shipped
  today is **high attribution, low autonomy**: the vendor finds the risk, the customer verifies it and
  owns the outcome ("you cannot trust us, please verify, please validate, all the responsibility is
  yours"). That is the Claude Code analogy exactly — the model writes most of the code; the code that
  reaches production is the customer's problem. Low attribution forbids outcome pricing outright.
- **Know why outcome pricing is blocked, because the blockers are eval problems.** Three reasons
  given: agents hallucinate, so you cannot stand behind an outcome; they do **not learn from their
  mistakes** the way a human hire does; and there is no **10x-on-the-bet curve** — with a human you
  accept variance because one great hire returns a multiple, "that's the bet we make on humans. We
  can't make the same bets here." The second and third are exactly the trajectory-evaluation and
  learning-loop problems of CS-17/CS-18. The sentence to carry out: **pricing model power is gated on
  eval power.**
- **Count the cost buckets in the right proportion.** Three buckets: **model cost** (token spend),
  **agent/harness cost** (retrievals, tool calls, guardrails, knowledge base), **infra cost** (app
  services, containers, observability). The ordering observed in CS-23's own estimate is worth
  memorising because it inverts the intuition: **infra is the smaller line, model cost is the
  smallest, and harness cost is the largest.** Then add the lines nobody counts — **training,
  evaluation, monitoring, human-in-the-loop, customisation and forward-deployed engineering** —
  *before* the margin, not after. HITL here includes the vendor's own CEO, sales and CTO, which makes
  it a per-customer cost rather than a per-business one.
- **The eval bill is a COGS line, not an invoice line.** CS-23's Harvey example: roughly **$100 to
  run** and **more than $1,000 to evaluate**, against a **$1,000 charge** — a **$100 loss per
  customer**, with the gap covered by funding. This is the sharpest available statement of the point
  in the bullet above it: evaluation is not overhead you pay instead of shipping, it is the largest
  under-counted item in cost of goods, and it is "the biggest ticket item here" once you count it.
  Teams that cannot name their eval cost per customer do not know their unit economics.
- **Credits are the current best answer to cost volatility, and they have exactly one failure mode.**
  A credit is a non-token billing unit the vendor internally maps to tokens, tool calls and retrievals
  ("this query took five credits"). They do three jobs: the customer sees a round number instead of a
  token count; the vendor **absorbs model-price shocks** by shrinking what a credit buys while holding
  the headline price and margin; and they **re-align sales**, because a flat-subscription vendor has
  no reason to track usage and a credit vendor does. The failure mode is a credit anchored to nothing
  the buyer recognises — "let's say I bought $25 of credit, I don't know how much coding will I get
  out of that $25." The fix is to anchor the credit to the charge matrix: one contract credit = a
  200-page contract with 10 questions.
- **The 80% rule.** Size the retainer so **80% of users never hit its limit**, and check that small,
  medium and large workloads all fit inside it. If ordinary users hit the ceiling, every interaction
  arrives with an upsell and "your customers will just throw the product." It is a churn rule
  expressed as a pricing parameter.
- **Instrument per request, or none of this is computable.** CS-23's schema: **user_id, customer_id,
  feature, prompt, agent_run_id, deployment**. User ID and customer ID are separate because one is an
  individual and one is an enterprise; the pair is what makes cost-to-serve attributable. Dashboard it
  as total runs, user growth, committed revenue and **margin trend**, with drill-down to customer and
  then to user.
- **Know the competitive anchors**, because interviewers ask for numbers and the numbers are the
  argument. Same buyer, same job, three different axes of metering — and a human baseline underneath
  all of them:

| Player | Model | Number |
|---|---|---|
| **Sierra** | outcome-based | ~**$200–350k** |
| **Decagon** | per-conversation / per-resolution | ~**$400k** |
| **Finn / Intercom** | platform fee + per-resolution | ~**$40k** + **$1/resolution** · **65%** resolution guarantee or **$65k back** |
| **Human baseline** | hourly | **$35/hour** · ~**5 tickets** → **$7 per resolution** |

  The human line is the load-bearing one: at $35/hour and five tickets, a human resolution costs $7,
  so Finn's $1 is roughly a **7× undercut** — which is the value argument, and also the reason the
  vendor must be confident about resolution rate. A cheaper per-resolution price paired with a weaker
  guarantee can be more expensive in expectation, and a per-resolution price says nothing about the
  fixed platform fee above it.
- **Where the growth comes from financially, so you are not naive about it.** The Rule of 40:
  growth % + profit % must reach **40**. With growth at 100%, 60% losses are fundable — "and that's
  why every company is raising every 3 months, 6 months." The SaaS warning underneath: subscription
  pricing worked because software margins ran **70–80%** and marginal cost was near zero; agents break
  it because token cost is non-deterministic, scales with success, and is invisible at the point of
  sale. The spread is roughly **100×** — a median customer at **$20** and a P90 customer at **$2,000**
  ("your 10% customers are charging you 100x"), with the concrete instance being a user who runs it
  three hours a day, costs about **$2,000**, and pays **$20**.
- The levers, in order of leverage:
  - **Step count.** Cost is roughly linear in steps, so an efficiency improvement is a cost
    improvement. This is why step efficiency is a metric and not a nicety.
  - **Model routing.** Cheap model for easy steps, expensive model for hard ones. Route on measured
    confidence, not on vibes.
  - **Context discipline.** Trim retrieved context and conversation history; input tokens are the
    silent majority of most bills.
  - **Caching.** Prefix caching for stable system prompts and retrieved context; and never re-judge
    an identical (input, judge, prompt) triple.
  - **Timeouts and retry policy.** Do not retry permanent failures, and cap retries — a runaway retry
    loop is both a cost incident and a latency incident.
- **Express the price as a per-query ceiling tied to the product's value**, because that is the only
  form in which the trade-off is decidable. And put it on the same dashboard as quality and latency, so
  a quality gain that triples cost is visible at the moment it is proposed.
- Finally, be honest about the **grader's share**. Evaluation cost is part of the cost of running the
  product, and a team that has not measured it does not know its own unit economics. CS-20's anchor
  figure: an evaluation run over a test set took **16 minutes** and consumed roughly **10x the tokens
  of serving the application** — a ratio to plan against, not a number to quote, since the source never
  gives the test-set size, token count or model. The mechanism is what matters and it is worth stating:
  an eval re-runs the whole harness over a frozen corpus, so its cost scales with the **corpus**, not
  with traffic — and it is billed even when production traffic is zero.
- Close the loop financially, which is where all of this is going. Track **per customer**: revenue,
  attributed spend, margin — and **flag negative margins**. If you cannot attribute spend per customer,
  you cannot tell which customers are loss-making, and the eval stack is precisely the instrumentation
  that makes per-customer pricing decidable. That is CS-20's real thesis: **evals are the cost
  accounting system, not the QA system.**

**Red flags:** mean-only cost; no step-count lever; no routing; no caching; grader cost omitted; cost
on a separate dashboard; pricing from your inference bill rather than from the displaced human's
willingness to pay; no per-customer margin attribution; **pricing on the mean instead of the P80+
ceiling**; **metering in tokens rather than in a charge matrix the buyer recognises**; claiming
outcome-based pricing while the customer still verifies the output; omitting training, evaluation,
monitoring, human-in-the-loop or forward-deployed engineering from cost of goods; a retainer that
ordinary users hit.

### Q18. What is red teaming, and how does it differ from your regression suite?

**What they're testing:** whether you understand why coverage audits exist.

**Model answer:**
- **Red teaming** is adversarial probing of the live system by people actively trying to make it fail
  — prompt injection, jailbreaks, data exfiltration attempts, tool misuse, edge-case inputs designed
  to break assumptions.
- CS-21 names three attack classes worth using as a checklist, because they cover different surfaces:
  **injection** (hostile instructions arriving through user input, retrieved documents or tool output),
  **corruption** (poisoning the context or a data source the agent trusts), and **sandwiching**
  (wrapping a hostile instruction inside content the agent has been told to treat as benign). Each
  targets a different trust boundary, which is why one passing does not imply the others do.
- It is **not** the regression suite, and the difference is structural: your regression suite is built
  from failures you have already seen, so **by construction it cannot catch a failure class you have
  not imagined.** Red teaming is the only instrument that looks for those.
- Therefore it needs a **separate cadence** — a regular exercise, not a per-commit job — and its
  output is not a score but **new failure classes**, which get converted into golden-set rows and
  evaluators. That conversion is what makes red teaming pay off; a red-team report that produces no
  new evaluator has produced nothing durable.
- Two further distinctions:
  - A **judge validated on organic traffic has not been validated on adversarial inputs.** Expect
    materially lower TPR on deliberate manipulation, and validate the judge against adversarial
    examples separately.
  - **Production guardrails are a different control from evaluation.** A spend limit, an allowlist or a
    confirmation gate stops harm *at the moment of action*; an evaluator tells you afterwards. You need
    both, and neither substitutes for the other.
- Where an agent takes irreversible actions, red teaming should specifically target **the guardrails
  themselves**, not only the model's willingness to attempt the action.
- CS-20 places red teaming correctly relative to evaluation: it is "**beyond evaluations**." It is not
  a scored run of your existing suite — it is a campaign of launched attacks whose output is a
  qualitative finding, and whose value is entirely in what you convert it into. Treat the platform's
  red-team mode the same way you treat a pen test: scheduled, scoped, and followed by remediation work
  items.
- And pair it with **alerts on the live system**, which is the other half of the operational control:
  a live threshold on the metric you care about, firing to a named owner. CS-20's honest note on this
  is worth repeating in an interview — getting alerts working is "a little involved but possible," and
  that is the normal experience rather than a sign you have done something wrong.

**Red flags:** red teaming treated as a subset of the regression suite; a one-off exercise; no
conversion of findings into evaluators; no production-side guardrails; judge validated only on organic
data; no live alerting owner.

### Q19. How do you manage prompt versions?

**What they're testing:** whether you treat prompts as deployable artefacts.

**Model answer:**
- **A prompt is a deployable artefact and deserves the same treatment as code**: versioned, attributed,
  reviewable, rollback-able, and evaluated before it ships.
- Concretely, a prompt becomes a **versioned object** referenced by the application, not a string
  literal in the source. That gives you:
  - **Attribution** — every trace records the prompt version that produced it, so a metric movement
    can be traced to a change.
  - **Comparison** — two versions can be run side by side on the frozen dataset as an experiment, with
    the CI gate deciding.
  - **Rollback** — reverting is a pointer change, not a code revert.
  - **Caching** — a stable prompt object is what makes prefix caching effective.
- The failure this prevents is the most common one in production LLM systems: **a prompt edit made
  directly in production, by someone, at some point, with no evaluation and no version bump.** The
  metric moves, nobody can explain it, and there is no diff to look at.
- Practical gate: **a prompt change is a release.** It goes on the versioned path, gets evaluated
  against the frozen set, and clears the same threshold as a code change.
- And remember that prompt versions are part of the **eval's own provenance**: a judge prompt change
  alters the metric, so judge prompts need versioning too, for the same reason.

**Red flags:** prompts as string literals; no version recorded in the trace; prompt edits shipped
without evaluation; judge prompts unversioned.

### Q20. How do you know your online metrics are trustworthy?

**What they're testing:** whether you can meta-evaluate your own measurement.

**Model answer:**
- Apply the same suspicion to the metric that you apply to the system. Five checks:
  1. **Is it reproducible?** Same config, same traces, same verdicts. If a re-run disagrees, the
     metric's noise floor is too high for the decisions made on it.
  2. **Is the instrument stable?** Keep a **canary set** — frozen inputs with stored verdicts. If the
     verdicts move with no code change, the judge or the model version changed underneath you. Pin
     the judge model version explicitly.
  3. **Is it predictive?** Measure the **correlation between the online metric and the offline metric**
     for the same quantity. This is the eval programme's real output and almost nobody measures it.
     If offline faithfulness and online faithfulness do not track, one of them is measuring the wrong
     thing, and the offline number is not a gate you can trust.
  4. **Does it discriminate?** Does the metric actually move when the system changes? A number that
     never moves carries no information, however plausible its value.
  5. **What does it not cover?** Coverage is invisible unless audited. Take last quarter's incidents
     and ask which ones the metric would have caught. The ones it would not have are your blind spots.
- Plus the honesty check: **sampling must be recorded and representative.** A metric computed over a
  stratified sample that over-weights failures is a *diagnostic* number, not an unbiased headline;
  reporting it as the rate is a common and quiet error.

**Red flags:** never having questioned the metric; no canary set; no offline/online correlation; no
coverage audit; a stratified rate reported as an unbiased one.

---

## Tier 3 — Senior / staff (8 Q)

### Q21. Design the observability stack for an LLM product.

**What they're testing:** whether you can build the substrate, in the right order.

**Model answer:**
First, name the four things the stack has to carry, because "observability" is a vendor word that
hides four different capabilities. CS-21's decomposition, and it is the cleanest available:
**monitoring** (uptime, failed requests, response time — the pre-agent discipline, "always there"),
**evaluation** (is it *right* — the newer layer), **guardrails** (did it stay inside its policies and
privileges), and **red teaming** (voluntary attack). Only the first is old; the other three are the
new work, and a vendor selling "observability" may mean one, some, or all four.

Then the transport decision, which has a defensible default: **emit OpenTelemetry**. It is a five-year-old
open standard that "every cloud provider, as well as most of the new people" support, and its value is
portability — traces, logs and metrics in a format any tool can process, so "if I go to AWS tomorrow I
should be able to set it up and run it." The practical cost is small: wiring OTel into a cloud monitor
is on the order of **one day of developer effort**. Choosing a proprietary trace format instead is a
lock-in decision made by accident.

Build order — each layer is a prerequisite for the next, and skipping one produces a dashboard that
cannot be interpreted:
1. **Logging / tracing layer.** A structured per-turn record, written **non-blocking** (logging must not
   add latency to the path it measures), with **conversation, turn, session and user IDs** as stable
   join keys, **PII masked at write time**, and a durable queryable store.
2. **Captured signals.** Thumbs up/down, latency in ms, token counts, cost, error flag and status
   code. Free, deterministic, no model.
3. **A baseline**, produced by the offline evaluation. Without it, no number on the dashboard can be
   judged good or bad — this is the layer teams skip, and it is why their dashboards are decorative.
4. **Time-windowed aggregation.** Last 1h / 24h / week / 6 months, so both spikes and slow drift are
   visible.
5. **Alerting** — metric + comparison + window + **owner + remediation path**, routed where someone
   will see it.
6. **Evaluators** — the computed signals: reference-free judges where no ground truth exists
   (faithfulness, relevance), code checks where determinism is available.
7. **Sampling and targeting** — deterministic, stratified, with a decision model mapping trace
   classes to evaluator sets.
8. **The feedback loop** — Add-to-dataset from the trace view, plus an annotation queue, feeding the
   offline golden set.

The architecture principle worth stating: **this is not a batch job, it is a property of the trace
store.** Everything — scores, datasets, experiments, prompt versions, human feedback — attaches to one
object, and that attachment is what makes a bad trace lead directly to the fix.

**Red flags:** a metric store with no trace store; blocking logging; no baseline layer; no sampling
layer; no path from the dashboard back to a dataset.

### Q22. Build or buy an evaluation platform?

**What they're testing:** whether you can make an infrastructure decision with reasons.

**Model answer:**
- Decompose the platform before deciding, because the answer differs by component. A mature platform
  has **four planes**:
  | Plane | Responsibility |
  |---|---|
  | **Ingest/query** | UI, API, auth, entitlements |
  | **Async work** | evaluation runs, experiments, exports, retention, telemetry ingestion |
  | **Storage** | traces/observations at volume vs metadata vs blobs |
  | **Contracts** | the shared table definitions everything writes through |
- **Buy the ingest, storage and UI planes.** They are undifferentiated heavy lifting — a trace store at
  volume is a genuinely hard distributed-systems problem (columnar storage for spans, a relational
  store for metadata, object storage for blobs, and a retention job that actually runs).
- **Own the contracts and the evaluators.** The score contract, the failure taxonomy, the criteria
  definitions and the golden dataset are your product knowledge and cannot be bought.
- Build in-house only what is genuinely specific: a decision model encoding your risk classes, a
  domain-specific deterministic evaluator, and the harness that runs your particular system under test.
- The decision rule for anything borderline: **own it if it encodes a judgement only your team can
  make; buy it if it is a solved engineering problem.**
- And name the exit cost before you buy. Traces are the system of record for evaluation; migrating them
  is the expensive part, so an open schema and an export path are requirements, not nice-to-haves.
- **The capability checklist to test any platform against**, drawn from CS-20's Foundry walkthrough.
  Whatever you buy or build, the pipeline must support all of these, and the absence of any one is a
  reason to look elsewhere:
  | Capability | Why it is load-bearing |
  |---|---|
  | **Trace capture** | The raw material — without it there is no dataset |
  | **Dataset creation from traces** | Turns live traffic into a frozen eval corpus |
  | **Turn-scope selection** (single vs multi-turn) | Multi-turn and single-turn failures differ |
  | **Recurring evaluation** | "Every day you can just evaluate 100 interactions recurringly" |
  | **Synthetic data generation** | The bootstrap option before you have traffic |
  | **Field mapping** | Query, response, context, ground truth |
  | **Auto-detected tool definitions** | Removes manual wiring; the schema is usually inferable |
  | **Out-of-the-box evaluators** | Task, quality and safety categories — the starting set, not the answer |
  | **Alerts on drops** | Threshold, window, owner — an alert with no owner is decoration |
  | **Red teaming** | Attack launching, "beyond evaluations" |
- **And be honest about how the platform market is shaped.** Evaluator catalogues grow because
  platforms are frequently **paid per evaluation**, which is an incentive to offer more evaluators
  rather than the right ones — "you feel like a kid in a candy land." Treat the catalogue size as a
  vendor incentive, not as a feature count, and bring your own prioritisation.
- **A worked example of the vendor-opinion trap, because it is the right answer in an interview.** On
  platform choice, CS-20's speaker is emphatic — cloud platforms (Azure > AWS > Vertex, in his
  ranking) over open-source tooling, and specifically that Arize and BrainTrust "do less but say more
  and price more." Record it as **one practitioner's opinion**: the transcript contains no benchmark,
  no feature matrix and no pricing data behind it, and the speaker sells a course and discloses that
  the platform does not pay him. The transferable discipline is not the ranking — it is that you should
  be able to say **what evidence would settle your own platform choice**, and in this case none was
  offered.

**Red flags:** building a trace store from scratch; buying the evaluators wholesale; no score contract;
no export path; no exit-cost consideration; choosing a platform on catalogue size; repeating a
vendor-adjacent opinion as if it were a finding.

### Q23. How do you govern evaluation spend?

**What they're testing:** whether you can run evaluation as a budgeted function.

**Model answer:**
- Put evaluation spend on its own line, split into **generation cost** (running the system under test)
  and **grading cost** (judges, human annotation, execution containers). Pooled, it is invisible and
  grows without a decision.
- Establish the **cost per evaluation run** for each tier — per-commit, per-PR, nightly, pre-release —
  and report it. A nightly suite that costs a meaningful amount per night is an annual budget item and
  should be a deliberate one.
- The levers, in order of leverage:
  - **Target** — cheaper evaluators on the broad traffic, expensive ones only on the risk classes.
  - **Sample** — stratified, but with a defined budget per stratum rather than an open-ended one.
  - **Route** — code checks and small models for the easy majority.
  - **Cache** — never judge an identical triple twice.
  - **Tier** — heavy evaluation offline on a bounded golden set; light, sampled evaluation online.
- The trade to name explicitly: **budget spent on judging is budget not spent on human labels**, and
  human labels are what make every judge number meaningful. When the two compete, labels usually win
  — a smaller number of validated metrics beats a large number of unvalidated ones.
- Finally, report the eval spend **per model release**, so the cost of a decision is visible alongside
  the decision. "We ship weekly and each release costs this much to evaluate" is a sentence a budget
  owner can act on.
- **The human-label arithmetic every candidate should be able to do**, because it is usually the
  largest single line and it is always underestimated. CS-21's worked contract-redlining example:
  one human-answered question costs **25 cents**; a contract contains roughly **20 risks**; each risk
  needs **20–30 questions** to cover → about **1,000 questions** per contract → **$250 to evaluate one
  contract.** Two consequences follow immediately. First, **human evaluation does not scale to
  production volume** — 1,000 contracts a month is $250,000, so human labels belong on the calibration
  slice and nowhere else. Second, this is precisely the arithmetic that justifies automating: an
  evaluator that costs a fraction of a cent per run and agrees with the human labels is worth building,
  and you can compute the payback period rather than assert it.
- **And put the run-away costs in the same conversation**, because they are what makes the budget
  argument land. CS-21's two war stories: a single agent session fixing Outlook ran **three hours** and
  billed **$786** for work a human does in five minutes; and a company exhausted its **entire IT budget
  in three months**, with expenses growing **9× month on month.** Neither is a quality failure — the
  agents were doing the job — and neither is visible in a quality dashboard. They are cost failures,
  and they are why cost per session, per user and per task belongs on the same screen as quality.

**Red flags:** no separate line; no per-tier cost; capping human labels to fund more judging; no cost
per release; treating eval cost as infrastructure overhead; no per-session or per-user cost ceiling;
assuming an agent's spend is naturally bounded, so no runaway-cost control is put in place.

### Q24. How do you run eval gates in CI without the team disabling them?

**What they're testing:** whether you have seen a good gate die.

**Model answer:**
- The failure mode is predictable: the gate flakes, an urgent fix ships with the gate bypassed, the
  precedent is set, and within a month nobody looks. So the design goal is **a gate that is trusted**.
- The requirements for that:
  - **Above the noise floor.** Measure the metric's spread across two runs of identical settings.
    Set the blocking threshold above that spread — a drop larger than a defined number of units blocks;
    anything smaller reports. A gate finer than the noise is a coin flip, and a coin flip is what gets
    disabled.
  - **Deterministic.** Fixed seeds and a frozen dataset, so the only thing that varies is the code.
  - **Fast enough to survive.** The blocking layer must be the cheap layer: code checks and a small
    sampled judge run. Everything expensive moves to nightly.
  - **Attributable.** When it blocks, the output must say **which metric moved, by how much, on which
    examples, and which change caused it** — not just exit code 1. A gate that makes the developer do
    the forensics is a gate they will bypass.
  - **Bypassable with a record.** A documented override path with a reason and a follow-up, rather
    than an unrecorded `--skip`. If the escape hatch is hidden, people find an uglier one.
  - **Owned.** A named person responsible for its health, who prunes stale evaluators and recalibrates
    thresholds as the system drifts.
- One structural point: **do not put execution-heavy evaluation in CI.** Container rollouts and
  long-horizon agent runs belong on nightly or pre-release cadences; putting hours of wall-clock in the
  merge path guarantees the gate gets bypassed.

**Red flags:** a strict-equality threshold; no noise floor; expensive checks in the PR path; a gate
message that does not name the failing examples; no override record; no owner.

### Q25. What does eval infrastructure look like at three maturity levels?

**What they're testing:** whether you can plan a programme rather than a project.

**Model answer:**
- **Level 1 — the script.** A golden dataset, one eval script, run manually before a release. Value:
  you know whether a change helped. Limitation: results are not comparable over time, because nothing
  records the config.
- **Level 2 — experiment tracking.** Runs are recorded with their configuration (dataset version,
  model version, prompt version, judge version), results are comparable across runs, and a baseline is
  captured so a delta is meaningful. Value: you can answer "did this change help, relative to what?"
  This is the level at which a CI gate becomes legitimate, because there is a baseline to gate against.
- **Level 3 — platform plus CI plus production.** Tracing in production, scores attached to traces,
  sampled online evaluation, drift alerting, annotation queues feeding the dataset, prompt versions
  wired to experiments, and a CI gate on the offline suite. Value: the loop is closed and the system
  improves without a human remembering to run something.
- The progression is not optional and cannot be skipped: **a CI gate without recorded baselines is
  meaningless**, and **online evaluation without a trace store is impossible.** Each level is the
  prerequisite for the next.
- What to say about where to start: the highest-return first move is usually **tracing** — it is a
  prerequisite for everything downstream and it immediately improves debugging even before any
  evaluation exists. The lowest-return first move is buying a dashboard.
- **Add the business-level framing and the two numbers that justify the whole programme.** The reason
  this matters commercially is stark: something like **80–90% of companies have agents but only ~15%
  reach production**, and the stated reason is that nobody is doing evaluations and evaluations are
  costly. Evaluation is the **gate on production**, not a quality nicety — so the funding argument is
  the production rate, not the elegance of the metrics.
- **The launch gate, stated as a ramp rather than a threshold**, which is CS-20's sharpest operational
  contribution. You do not go from "manual evals" to "launched"; you go:
  | Stage | Gate |
  |---|---|
  | **Internal / manual** | Observe the real number — in CS-20's case, a scorecard of **22%** task completion, **52%** helpfulness, **78%** retrieval relevance, guardrails passing |
  | **Set the target** | Name the bar: "I want us to be 60% helpful, 70% honest" |
  | **Canary** | "I can launch to **one or two percent** of my customers" |
  | **Ramp** | Widen as the numbers hold |
  | **Steady state** | Continuous evaluation plus per-customer margin tracking |
  The useful part for an interview is the posture: **you ship at 22% deliberately, to 1–2% of users, with the targets written down.** A candidate who insists on a quality bar before any launch has not run a product; a candidate who launches to everyone without a number has not run an eval programme.
- **And budget the effort explicitly.** CS-20's allocation: "at least **30–40% of your time should go
  into evals**," with the work framed as "**breaking your agents** and making them eval ready." Note
  the verb — the posture is adversarial testing, not score-tracking, and it matches CS-16 and CS-21.
  If you state an effort allocation, state that one; it is the only such number in the source.
- **CS-22 states the same launch gate with all three numbers attached, and that is the version to
  quote**: "we can only launch this product to **1% user** when we reach helpfulness of **60%**,
  honesty of **70%** and less than **5% harmlessness**." Three things are worth noticing in that
  sentence. It contains **two floors and a ceiling** — helpfulness and honesty must rise *above* a
  threshold, harmlessness must stay *below* one. **Harmlessness is stated as a rate, not a pass**,
  because a safety property is a frequency rather than a boolean. And the gate is attached to a
  **user percentage** rather than to a date, which is what makes it a rollout decision instead of a
  deadline. That is what a launch criterion looks like when it is written down rather than argued
  about.
- **Then connect the eval stack to the product targets**, because that is what turns it from
  measurement into a programme. CS-22's three team-level targets, stated as rates of change:
  **+20% completed tasks**, **+40% helpfulness**, **−30% cost**. The metric *structure* underneath
  them is the part to copy: a **northstar** (total tasks completed and *accepted* by users — the
  adoption link), an **L1 operating metric** (task completion rate plus helpfulness, honesty and
  harmlessness) reviewed week on week, and a separate **technical performance** layer (P95/P99,
  recovery, first meaningful response time). And the management posture that goes with it — asked
  *how* engineering should hit the numbers, the PM's answer is that the *how* is "none of your
  business": **the PM owns the target, engineering owns the method.**

**Red flags:** jumping to level 3 tooling without a dataset; a CI gate with no baseline; no config
recorded with results; "we'll add observability later"; no canary stage; a launch bar with no number
attached; treating evals as a phase rather than a standing 30–40% of the work; a northstar metric with
no adoption link; a PM specifying the implementation.

### Q26. How do you handle multi-tenant evaluation?

**What they're testing:** whether your quality model survives a second customer.

**Model answer:**
- The core problem: **an aggregate metric across tenants hides the tenant that is failing.** A
  comfortable 92% overall can contain one customer at 60%, and per-tenant volume weighting means the
  unhappy customer is invisible in the average.
- So evaluate **per tenant where the failure cost is per tenant**, and report the **distribution across
  tenants**, not the mean. Watch the worst decile, not the average.
- Segment the metrics by anything that changes the input distribution: tenant, language, plan tier,
  region, document corpus. A retrieval system whose corpus is per-tenant will have per-tenant recall,
  and a single recall number is meaningless.
- The practical constraints:
  - **Sample size per tenant.** A small tenant has too little traffic for a percentage to be
    meaningful. Report per-tenant numbers with sample sizes and confidence intervals, and do not fire
    alerts on a tenant with 20 monthly interactions.
  - **Data isolation.** Trace stores now contain multiple customers' data, which raises the PII and
    contractual stakes considerably. Access control and retention are per-tenant concerns.
  - **Cost attribution.** Per-tenant cost is a commercial question, not just an engineering one — some
    tenants will be running at a loss and you want to know.
- And the release-gate consequence: if a change improves the aggregate while regressing one large
  tenant, that is a regression. **Gate on the worst segment, not the mean.**

**Red flags:** one global number; volume-weighted averages; no per-tenant cost; alerting on
statistically meaningless tenant samples; no data isolation.

### Q27. Walk me through responding to a production quality regression.

**What they're testing:** whether you have run an incident.

**Model answer:**
1. **Confirm and scope.** Is it a real regression or instrument drift? Check the canary set; check the
   judge model version and the judge prompt version; check the parse-failure rate. If the frozen
   verdicts moved with no code change, you are debugging the measurement, not the system.
2. **Bound it.** When did it start (correlate with deploy and prompt-version timestamps), what
   fraction of traffic, which segments — tenant, language, query class, product surface. A regression
   that hits 2% of traffic in one segment is a different incident from one that hits everything.
3. **Check the cheap explanations first.** Deploy correlation, prompt version change, upstream corpus
   or index rebuild, provider model upgrade, traffic-mix shift (a launch bringing a new query class
   the system handles badly — which is not a regression in the code at all).
4. **Read the traces.** Automated failure analysis on the newly-failing class converts "quality
   dropped 6 points" into "it fails when the user references a prior order". This is the step that
   turns a metric into a fix.
5. **Mitigate before diagnosing.** Roll back the prompt version or the model, or route the affected
   segment to a fallback. The trace store makes the rollback cheap because the version is a pointer.
6. **Fix and gate.** The fix goes through the normal path: a golden-set row for the failure, an
   evaluator if the failure class was not covered, an experiment, and the CI gate.
7. **Post-incident, fix the detector.** If the regression was found by users rather than by the
   system, that is the primary finding. Add the coverage; recalibrate the threshold; if the alert
   fired and nobody acted, fix the routing.

The recurring theme: **most "quality regressions" are one of four things** — a real code/prompt change,
an upstream data change, a traffic-mix change, or the instrument moving. Separating them is the first
move, and the canary set plus version pinning is what makes it a five-minute question instead of a
day.

**Red flags:** jumping straight to prompt tuning; no version pinning; no canary set; no segmentation;
not asking whether the detector should have caught it.

### Q28. What is the relationship between evaluation and product decisions?

**What they're testing:** whether you can connect the machinery to the business.

**Model answer:**
- Evaluation exists to make decisions, and a metric that does not change a decision is cost without
  value. So work backwards from the decisions:
  - **Should we ship this change?** → offline suite + CI gate on recorded baselines.
  - **Which of two designs is better?** → offline experiment on a frozen set, one variable at a time.
  - **Is the live system still healthy?** → online metrics against offline-derived baselines, with
    drift alerting.
  - **Where should we invest next?** → the ranked failure taxonomy from error analysis, weighted by
    volume and cost.
  - **What does this cost?** → cost per query as a distribution, on the same dashboard as quality.
- The four things that make evaluation decision-grade, and the absence of any one makes it theatre:
  a **baseline** to compare against, a **measured noise floor** so a delta is interpretable, **cost and
  latency alongside quality** for the same interaction, and a **named owner** for each number.
- The uncomfortable implication to state: this means **most metrics should be deleted**. If nothing
  changes when a number halves, it is decoration. A small set of validated, owned, decision-linked
  metrics is worth more than a dashboard of forty.
- And the programme-level output: the **offline→online correlation** for your shared metrics. That
  single number tells you whether your offline gates predict anything, which is the difference between
  an evaluation programme and an evaluation ritual.
- **Then draw the ownership line, because a candidate who cannot is a candidate who will build
  dashboards nobody acts on.** CS-21 states the split as its central deliverable, and it is worth
  reproducing because it is counter-intuitive in one direction:

  | The **product manager** owns | Engineering owns |
  |---|---|
  | The **success criteria** — what good means | **Traces** — emitting and storing them |
  | **Cost targets** — per session, per user, per task | **Dashboards** — building and maintaining them |
  | **Helpfulness targets** — the launch gate | **Alerts** — wiring and routing them |
  | **Failure definitions** — what counts as broken | **Guardrail enforcement** — the actual controls |
  | **Thresholds and alert placement** — where the line sits | |

  The counter-intuitive half is the right-hand column: engineering builds the alerting *mechanism*, but
  the PM decides *what the threshold is and where the line goes*. And the genuinely new claim is
  **cost**: CS-21's argued shift is that cost per session, per user and per task used to be an
  engineering concern and is now a product one, "because it started impacting customer behavior and
  customer cost." His framing of why is worth quoting: starting cost is trivial but "people are paying
  as they are using the product, and if you're not able to manage that cost for them, they are going to
  leave your product."

**Red flags:** metrics with no decision attached; no baseline; no noise floor; quality separated from
cost; never having deleted a metric; engineering setting the quality bar; no named owner for cost per
user.

---

## Tier 4 — Debug-this-scenario (5 Q)

### Q29. Latency P95 improved 20% but users complain it's slower. Diagnose.

**What they're testing:** the timeout-rate reflex.

**Model answer:**
- First hypothesis, and the most likely one: **you are dropping more requests.** Timed-out requests are
  excluded from the latency distribution, so a "faster" P95 can mean the slow tail is now being killed
  rather than served. **Check the timeout rate first** — if it rose, the improvement is an artefact.
- Second: **the mean versus the tail.** If you optimised the median and the P99 got worse, the average
  user experience improved while your heaviest users got worse. Report the full shape — P50, P95, P99,
  min, max — with the sample size.
- Third: **TTFT versus total latency.** With streaming, users perceive **time to first token**. If the
  change moved work from before the first token to after it, the total improved while the *perceived*
  latency got worse. This is a very common and very invisible regression.
- Fourth: **whose P95?** Segment by tenant, query class and product surface. An aggregate improvement
  can conceal a regression on a segment that a specific set of users experiences every day.
- Fifth: **is it actually latency?** For an agent, "slower" often means more steps or more retries —
  the user sees more intermediate output and more waiting, which is a trajectory problem showing up as
  a latency problem.
- The fix for the reporting failure: **never publish a percentile without its timeout rate and sample
  size**, and always publish TTFT separately.

**Red flags:** accepting the P95 at face value; not checking timeouts; mean-only analysis; TTFT merged
into total; no segmentation.

### Q30. Cost per query doubled overnight with no code change. Diagnose.

**What they're testing:** whether you can find a cost regression.

**Model answer:**
- Work through the layers that sit between your code and the bill:
  1. **Provider pricing or model routing changed.** A silent model upgrade, a deprecated model falling
     back to a pricier one, or a rate-card change. Check the model version recorded on the traces.
  2. **Traffic mix shifted.** A new query class that hits a longer prompt path, a batch job that started
     running, a tenant onboarded. Cost is a distribution — check whether the *shape* changed or the
     whole curve moved.
  3. **Prompt or context growth.** Retrieved context got longer (k increased, chunks got bigger, a
     document refresh added length), or conversation history is accumulating because a trim step
     broke. **Input tokens are usually the silent majority**, so this is a frequent culprit.
  4. **Step count increased.** For an agent, a doubled cost is usually a doubled number of steps —
     look for retry loops, a tool that started failing and being retried, or a stopping condition that
     stopped triggering.
  5. **Caching stopped working.** Prompt prefix caching or a response cache being invalidated — for
     example because the prompt version or a timestamp in the prompt changed on every request.
     Ironically, this often *decreases* measured cost when it is working, so its loss shows up as a
     sudden increase.
  6. **Retry policy.** A non-retryable status being retried, or an unbounded retry loop.
- Where to look: traces give you tokens per step and cost per trace, so the diagnosis is usually a
  group-by away — cost by model version, by step count, by prompt version, by tenant. **If you cannot
  group by those, that is the actual finding.**
- And note the measurement trap: if cost is computed with float arithmetic and aggregated over long
  runs, the drift itself can look like a cost increase. Use decimal.

**Red flags:** looking only at request volume; no per-model or per-step cost breakdown; not checking
cache hit rate; float cost arithmetic.

### Q31. Your dashboard is green and users are complaining. What now?

**What they're testing:** whether you can think about what your measurement does not cover.

**Model answer:**
- Assume the dashboard is measuring a **subset** of what matters, and the complaints are pointing at
  the complement. The productive question is: **what is green because nothing is measuring it?**
- Run through the standard coverage gaps:
  - **Segmentation.** The aggregate is green; one tenant, language or query class is not. Group by every
    dimension you have and look for the worst cell.
  - **Failure severity.** Your metric may average a formatting nit and a wrong refund into one number.
    Check whether the *severity* distribution moved, not just the rate.
  - **Timeout rate.** Green latency with a rising timeout rate means requests are being dropped rather
    than served — those users never appear in your quality metric at all.
  - **Silent users.** Thumbs-down is a voluntary signal and heavily biased; users who simply leave
    produce no signal. Compare engagement and repeat-question rates against baseline.
  - **The unmeasured surface.** Every evaluator covers a criterion someone wrote. Ask what the system
    does that no evaluator looks at — tone, over-refusal, unhelpful-but-correct answers, latency on the
    expensive path.
- Then go to the qualitative evidence: **read the actual complaints and the traces behind them.** Ten
  real traces beat a week of speculation, and they will usually reveal a failure class you never
  enumerated. That class becomes an evaluator — which is the fix to the *detector*, not just to the
  incident.
- Finally, check the obvious: **is the complaint about quality at all?** Users often report cost,
  slowness or a billing surprise as "quality". Latency and cost belong on the same dashboard as
  quality precisely so this is visible.

**Red flags:** assuming users are wrong; adding a new metric without reading the traces; not segmenting;
not checking timeout rate or engagement.

### Q32. The CI eval gate is being bypassed every week. Diagnose.

**What they're testing:** whether you have seen a gate die and know why.

**Model answer:**
- Bypass is a **design outcome**, not a discipline problem. Work through the causes in likelihood
  order:
  1. **The gate is below the noise floor.** If the metric swings more than the threshold between
     identical runs, the gate is a coin flip. Measure the spread across two identical runs; if the
     threshold is inside it, no amount of exhortation will help. **This is the most common cause.**
  2. **It is too slow.** If a PR waits tens of minutes, urgency wins. Move everything expensive to
     nightly and keep the blocking layer to deterministic checks and a small sampled run.
  3. **It is not actionable.** "Metric dropped, exit 1" without naming which metric, by how much, and
     on which examples forces the developer to do forensics, which they will skip.
  4. **It is too strict in the wrong dimension.** Blocking on any movement at all, including
     improvements in one metric with a small expected dip in another, makes the gate an obstacle to
     good changes.
  5. **The baseline is stale.** Gating against a baseline captured months ago produces false failures
     that everyone learns to ignore.
- The fixes map one-to-one: recalibrate the threshold above the measured noise floor; tier the suite so
  only cheap checks block; make the failure output name the metric, the delta, the threshold and the
  failing examples; allow expected trade-offs to be declared; and refresh baselines on a schedule with
  an owner.
- One more: make the **escape hatch recorded** — an override that requires a reason and files a
  follow-up. An unrecorded `--skip` is what turns a gate into a formality; a recorded override is a
  decision, and the record is what lets you see whether the gate is wrong or the changes are.

**Red flags:** blaming the team; not measuring the noise floor; a slow blocking layer; an
uninformative failure message; no baseline refresh; hidden skip flags.

### Q33. An alert fired and nobody responded for six hours. Diagnose.

**What it's testing:** whether you treat alerting as a system with its own failure modes.

**Model answer:**
- Treat this as an **alerting incident**, separate from whatever the alert was about. Diagnose in three
  layers:
  1. **Did it reach a human?** Routing — was the channel one people read, was it the right rotation,
     was it outside working hours, did a rate limit or a webhook failure swallow it. A Slack channel
     nobody owns is not an alert.
  2. **Was it actionable?** If the alert says a metric dropped with no owner, no context and no
     remediation path, the responder had nothing to do. **An alert without a defined response is a
     notification.** Check whether the last ten firings of this alert were acted on — if most were
     dismissed, the alert has trained its audience to dismiss it.
  3. **Was it believable?** If the alert has a history of firing spuriously — because its threshold is
     inside the noise floor — the six-hour delay is rational behaviour. **Alert fatigue is a property
     of the alerting system, not of the responders.**
- The systemic fixes: one metric per alert; a threshold above the measured noise floor; a window (for
  example a rolling 24-hour graph firing on degradation over the last 8 hours); a named owner; an
  explicit remediation path; severity tiers so only genuine breaks page; and a periodic review that
  **deletes** alerts nobody acts on.
- And the meta-question worth asking: was the alert even the right detector? If the failure was
  eventually found by users, the finding is a coverage gap, and the response is a new evaluator rather
  than a louder alarm.

**Red flags:** punishing the responders; not checking whether the alert was actionable; keeping noisy
alerts "just in case"; no owner; no deletion policy.

---

## Tier 5 — Trap questions (6 Q)

### Q34. *(Trap)* "Let's run the quality judge on 100% of production traffic so we never miss anything."

**The naive answer:** agree — maximum coverage is the safest choice.

**What they're testing:** whether you reason about eval economics.

**Model answer:**
- Do the arithmetic before agreeing: at a couple of cents per judge call and 50,000 daily
  interactions, that is roughly **$1,000 a day**, around **$30k a month**, to *observe*. That is a
  product decision, not a configuration toggle.
- Full coverage does not buy full detection anyway. The bottleneck just moves: someone still has to
  look at the failures, and a 100%-coverage dashboard nobody reads detects nothing.
- The correct design is **targeted and sampled**:
  - Cheap deterministic checks and captured signals on 100% of traffic — schema validity, latency,
    cost, error rate.
  - The expensive judge on a **deterministic, stratified** sample: a small uniform slice for an
    unbiased headline, plus boosted strata for the cases where failures live (thumbs-down, escalations,
    abrupt endings, repeated questions, money topics).
  - Risk-class targeting: anything touching money, safety or PII gets the expensive evaluators and
    human review.
- Spend the difference on **human labels** — the annotation queue is what makes every judge number
  meaningful, and it is chronically underfunded.
- And instrument the judge's own spend as a first-class line, so this trade-off is a decision rather
  than a discovery.

**Red flag:** accepting full coverage without doing the arithmetic.

### Q35. *(Trap)* "We'll add evaluation after we launch — we need to move fast."

**The naive answer:** agree — evals can come later; shipping is the priority.

**What they're testing:** whether you know what is irrecoverable.

**Model answer:**
- Some of this is genuinely deferrable, and some is not, and the difference is **whether the data is
  captured at the moment it exists**.
- **Irrecoverable if deferred — tracing.** You cannot reconstruct the conversations that happened
  before you started logging. Every week without a trace store is a week of production behaviour that
  is permanently unobservable, and it is the prerequisite for every downstream capability: online
  evaluation, drift detection, dataset harvesting, incident response. Deferring it does not defer the
  work, it discards the evidence.
- **Irrecoverable if deferred — PII masking.** Data you stored unmasked cannot be unm-stored. The
  control has to be at the logging boundary from the first request, which means it has to exist at
  launch.
- **Recoverable later** — the judge suite, the CI gate, benchmark comparisons, dashboards. These can
  be built on the trace store afterwards, because the traces are the substrate.
- So the honest answer is not "you must do everything first" but "**you must capture from day one**".
  A tracer plus captured signals plus PII masking is a small amount of work at launch and an enormous
  amount of work to retrofit.
- The framing that usually lands: **shipping without tracing is shipping without a flight recorder.**
  You will have the incident. You will not have the data.

**Red flag:** accepting "later" for tracing and PII masking.

### Q36. *(Trap)* "Our average latency is 800 ms, which is fine."

**The naive answer:** agree — 800 ms is a reasonable number.

**What they're testing:** whether you know an average hides the distribution that users experience.

**Model answer:**
- An average is the least useful latency statistic. A mean of 800 ms is consistent with a system where
  everyone waits 800 ms, and with one where 90% of requests take 200 ms and 10% take 6 seconds. **Those
  are different products**, and the mean cannot tell them apart.
- Publish the **shape**: P50, P95, P99, min, max, with the sample size. The tail is what users
  remember and what they churn over.
- **Never publish a percentile without the timeout rate.** Timed-out requests are excluded from the
  distribution, so a good-looking P95 can coexist with a growing pile of dropped requests. The
  timeout rate is the number that reveals it.
- Report **TTFT separately from total latency**. With streaming, the user perceives time to first
  token; a system with an excellent total and a slow first token feels broken.
- Segment the distribution. A global mean hides the tenant, language or query class where the system is
  unusable.
- For agents, latency is a **behavioural** metric: step count and retry count drive it, so a latency
  investigation is often an agent-efficiency investigation.
- And the last piece: put latency on the **same dashboard as quality and cost**. A latency improvement
  that halves quality is not an improvement, and the mean will never tell you.

**Red flag:** defending an average as sufficient.

### Q37. *(Trap)* "We sample 1% of traffic uniformly at random — that's the fair and unbiased way."

**The naive answer:** agree — random sampling is the statistically correct choice.

**What they're testing:** whether you confuse unbiasedness with usefulness.

**Model answer:**
- Random sampling is *unbiased* and *wasteful*. The principle: **not all conversations are the same.**
  A uniform 1% spends almost all of your budget on conversations that went fine, because most
  conversations go fine.
- The failures concentrate, and they concentrate in detectable places: **thumbs-down**, **escalations**,
  **conversations that ended abruptly**, **repeated rephrased questions** (the user asked again because
  the first answer failed), and anything touching **money**. A sample that ignores those places is
  measuring the easy majority.
- Use **stratified sampling**: bucket conversations, then over-weight the problematic strata. The
  bucketing usually needs a small classifier or clustering step, which is cheap and deterministic.
- But keep the statistical point intact: **boosting a stratum biases the headline rate.** A
  failure-enriched sample gives you a *diagnostic* number, not the population rate. So keep a **small
  uniform random sample alongside the boosted strata** — the uniform slice for reporting an unbiased
  headline, the boosted strata for finding out what is wrong.
- And make the selection **deterministic** as well as stratified, so a replayed run evaluates the same
  items and two runs are comparable.
- So the complete answer has three parts: **stratify for diagnosis, keep a uniform slice for
  reporting, and make both deterministic for reproducibility.**

**Red flag:** defending uniform random sampling as sufficient, or reporting a boosted rate as the
population rate.

### Q38. *(Trap)* "The judge score is our quality metric — we track it on the dashboard and gate on it."

**The naive answer:** agree — a validated judge is exactly what you want as the headline.

**What they're testing:** whether you know what a single number cannot carry.

**Model answer:**
- A judge score can legitimately be *a* metric. Treating it as *the* metric has four specific
  problems:
  1. **It has no cost or latency attached.** Quality in isolation is not actionable — a quality gain
     that triples cost is not a gain, and you only see that if both numbers share a screen.
  2. **It is not ground truth online.** Online, there is no human perspective in the traffic, so the
     judge is measuring against its own criterion, not against correctness. It can drift, and it can
     be wrong in a correlated way with the system it judges.
  3. **It is a single number compressing many criteria.** One "quality" score cannot tell you that
     groundedness improved while completeness fell. Use several narrow binary judges so the profile is
     diagnostic, and keep **failure severity** separate — a severity scale averaged into a quality
     score loses its meaning.
  4. **It hides coverage.** What the score does *not* measure is invisible: the timeout rate, the
     unmeasured surface, the tenant it is failing. Coverage has to be audited separately.
- So the corrected design: **one contract, many signals.** Attach judge scores, code checks, human
  labels, latency, cost and error rate to the same trace so they can be read together; keep the judge
  validated (TPR/TNR on held-out labels) and version-pinned so you know when the instrument moves; add
  a canary set to separate instrument drift from system drift; and gate on several named metrics, not on
  one composite.
- The framing to state: **a single quality number is a press release; a trace with a score, a latency
  and a cost attached is a decision.**

**Red flag:** accepting a single judge score as the quality metric and the gate condition.

### Q39. *(Trap)* "Our agent evaluates itself inside its loop, so we don't need an offline eval suite."

**The naive answer:** agree — self-evaluation is efficient, and it runs on every request rather than on
a sample.

**What they're testing:** whether you can tell a **control signal** from a **measurement**.

**Model answer:**
- Distinguish the two jobs. An **in-loop check is a termination condition** — it answers "should I keep
  going, or am I done?" An **eval suite is evidence** — it answers "is the system good, and is it better
  or worse than last week?" Those are different questions, and the first cannot answer the second.
  CS-24's loop is `context → inspect sources → decide & act → evaluate → plan → "is the job done?"`, and
  when an audience member asks what the loop compares against, the answer is the **prompt, the skills
  file and the `CLAUDE.md` files** — a textual, human-authored termination condition. No dataset, no
  judge, no threshold, no golden answers.
- Four specific failures of self-evaluation as the *only* evaluation, and they are independent:
  1. **It is self-referential.** The same model that produced the output is judging it. The errors it
     cannot detect are correlated with the errors it makes, so the check is blind in exactly the place
     you need sight.
  2. **It leaves no artifact.** No frozen dataset, no versioned criterion, no threshold, no trend. You
     cannot tell whether this week is worse than last, and you cannot reproduce a failure.
  3. **It cannot compute the metrics you will be asked for.** Without ground truth there is no accuracy,
     no precision, no recall — and per CS-22 the precision/recall measurement is precisely what
     license the whole judge-based program.
  4. **It cannot be validated or gated.** There is nothing to regress against in CI and nothing to
     calibrate against human labels, so the loop can degrade silently while reporting health.
- Say what the in-loop check is **good** for, so the answer is not purely negative: an agent that can
  detect it is stuck and stop is strictly better than one that loops forever, and a completion check is
  how you bound cost. Keep it. Just do not promote it.
- The corrected architecture has both, doing different jobs:
  - **In the loop:** step budget, completion check, fallback and escalation — cheap, deterministic
    where possible, and owned by engineering.
  - **Offline:** a golden set harvested from production failures, versioned, run in CI as a release
    gate.
  - **Around the loop:** **trajectory evaluation** (CS-17) — score the path, not just the final answer,
    because an agent that reaches the right answer by an invalid path passes a completion check and
    fails the business.
- State the principle you are applying: **anything the system judges about itself is a control signal;
  anything an independent instrument judges is a measurement.** A production system needs both, and
  confusing them is how teams end up with a green dashboard and no evidence.

**Red flag:** equating the agent's own self-check with an evaluation suite, or arguing that in-loop
evaluation removes the need for offline datasets, judges and gates.

---

## Live-coding / whiteboard prompts (3)

### Prompt 1 — Instrument a system for online evaluation (45 min)

> Here is a RAG endpoint. Make it evaluable in production. List what you change and in what order.

**What a strong answer covers:** a **trace** per interaction recording input, output, **retrieved
document IDs**, model and prompt versions, latency, token counts, cost and error status, with
conversation/session/user IDs as stable join keys; logging written **non-blocking** so it does not add
to the latency it measures; **PII masked at write time**; **captured signals** stored first (thumbs,
latency, cost, error) before any judge is introduced; a **baseline** produced by running the offline
suite; a **deterministic, stratified** sampling layer; **targeting** so the expensive judge applies to
a subset; **one score contract** so a code check, a judge verdict and a thumb are the same kind of
event; an **annotation queue** for human labels; and **Add-to-dataset** closing the loop back to the
offline golden set. The ordering argument is the assessed part — tracing and masking first because
they are irrecoverable, judges last because they depend on everything above.

**Red flags:** starting with the judge; blocking logging; no masked PII; no join keys for late signals;
no baseline layer.

### Prompt 2 — Design the alerting policy (45 min)

> Design drift detection and alerting for a customer-facing RAG assistant.

**What a strong answer covers:** **one metric per alert**, not a composite; a **rolling 24-hour graph
with the alert firing on degradation over the last 8 hours**; a **threshold set above the measured
noise floor** (run identical settings twice; measure the spread) and stated relative to the
**offline-derived baseline**; **windowed aggregates only — never a single conversation**; a **named
owner and a remediation path** for each alert; **severity tiers** so only genuine breaks page and drift
files a ticket; a **canary set** with frozen verdicts to separate **judge/model drift from system
drift**; segmentation so a per-tenant regression is visible against a green aggregate; latency alerts
that carry the **timeout rate** so a drop does not look like a win; and a periodic review that
**deletes** alerts nobody acts on. A strong candidate also notes that the first alert worth building is
usually a cheap captured-signal one — error rate or latency — not a judge.

**Red flags:** a composite score as the page condition; alerting on single traces; a threshold invented
without measuring the noise floor; no owner; no canary set.

### Prompt 3 — Cost and latency incident (30 min)

> Cost per query doubled overnight and P95 latency is up 30%, with no deploy. Where do you look?

**What a strong answer covers:** treat them as likely one incident, because for agents **step count
drives both** — check traces for a rise in steps, retries or loops (a tool that started failing and is
being retried, or a stopping condition that stopped firing). Then the other layers: **prompt and
context growth** (retrieval `k` increased, chunks got larger, a document refresh added length,
conversation history accumulating because a trim step broke) — **input tokens are usually the silent
majority of a bill**; **caching loss** (prefix caching invalidated by a prompt version or timestamp
changing per request); a **provider model upgrade or rate-card change**, checked against the model
version recorded on traces; a **traffic-mix shift** (a new query class on the expensive path, a batch
job, a new tenant); and **retry policy** retrying non-retryable failures. The diagnostic answer is the
strong part: everything above should be a **group-by on the trace store** — cost and latency by model
version, step count, prompt version and tenant — so if you cannot group by those, **that is the actual
finding**. Also note the measurement trap: float cost arithmetic over long runs can itself look like a
cost increase; use decimal.

**Red flags:** looking only at request volume; no step-count hypothesis; no per-model/per-prompt
breakdown; not checking cache hit rate; no timeout rate on the latency side.

---

## Take-home / case-study prompt (1 full brief)

> **Brief.** A B2B company runs a customer-support RAG assistant for 40 enterprise tenants, handling
> roughly 60,000 conversations a month. They have: a golden dataset of 300 rows built at launch eight
> months ago, a judge suite run manually before each monthly release, and no tracing. Two weeks ago a
> large tenant escalated, saying answer quality had "fallen off a cliff". The team's dashboard shows
> the offline judge score at 0.91, unchanged since launch. The CTO wants an evaluation plan and a
> budget.
>
> **Deliverables (≤6 pages):**
> 1. **Diagnose the escalation.** Explain why the offline score can be 0.91 and flat while a tenant is
>    failing, naming each structural reason. Say what you would look at first, given there are no
>    traces.
> 2. **The capture layer.** What you instrument, the fields, the join keys, the non-blocking
>    requirement and the PII position — and why these are the items that cannot be deferred.
> 3. **The evaluation layer.** Captured signals before computed ones: which of each, and why in that
>    order. Where the baseline comes from and what breaks without it.
> 4. **Sampling and targeting.** A concrete sampling design for 60,000 monthly conversations across 40
>    tenants, covering stratification, determinism, the uniform slice for unbiased reporting, and
>    per-tenant sample-size limits.
> 5. **Alerting.** One worked alert — metric, window, threshold, owner, remediation path — plus how you
>    set the threshold and how you would distinguish judge drift from system drift.
> 6. **Economics.** Estimate the monthly evaluation cost for your design, split into generation,
>    grading and human labelling, with the arithmetic shown. State the per-query cost ceiling
>    implication.
> 7. **Governance.** Prompt and judge versioning, the annotation queue, the canary set, retention and
>    access control across tenants.
> 8. **The answer to the CTO.** In plain language: what happened, what you cannot know without traces,
>    what it costs, and what you would have known and when.
>
> **What is being assessed:** whether you put capture before measurement; whether the diagnosis is
> structural rather than a list of guesses; whether sampling is both stratified and deterministic and
> whether you understand the reporting-bias consequence; whether the alert has an owner and a
> remediation path; whether the economics are done with arithmetic; and whether you can say plainly
> which of the eight months of history is permanently unrecoverable.

---

## Scoring rubric — what separates a hire from a no-hire

| Dimension | No-hire | Senior hire | Staff hire |
|---|---|---|---|
| **Trace-first thinking** | Talks about dashboards and metrics | Builds a trace store | Insists capture and PII masking are non-deferrable, and explains what is irrecoverable |
| **Harness identity** | "We're evaluating the model" | Knows the app matters | Says plainly that the object is the **harness** — LLM + context + tools + skills + guardrails — and that swapping the harness makes it a different measurement |
| **Metric ordering** | Runs the judge first | Knows some evals are cheaper | Orders by cost and objectivity — completion → adherence → navigation → quality → context → guardrails — and explains that an early-layer failure makes later layers uninterpretable |
| **ROI-first design** | Prices after building | Knows cost matters | Derives the target from the **displaced human's willingness to pay** and works backwards to whether the agent can be built inside it |
| **Offline vs online** | Conflates them | Knows both exist | States correctness vs normality, and that online structurally cannot measure correctness against a baseline offline produced |
| **Sampling** | Random or 100% | Samples at a fixed rate | Stratifies for diagnosis, keeps a uniform slice for unbiased reporting, makes it deterministic, and knows boosting biases the headline |
| **Captured vs computed** | Reaches for a judge | Knows both categories | Builds the free deterministic layer first and treats judging as the last resort |
| **Alerting** | A threshold on a dashboard | Adds a window | Metric + window + noise-floor-calibrated threshold + owner + remediation path, with severity tiers and a deletion policy |
| **Threshold literacy** | Waits for something to break | Has a threshold | Names floors **and** gates — accuracy floor **30%**, tool-failure ceiling **40%**, guardrail-trigger ceiling **5%**, helpfulness gate **60–75%** — and alerts on guardrail **over**-triggering, not just breaches |
| **Responsibility split** | One team owns "quality" | Knows PMs and engineers both care | Draws the line: PM owns criteria, cost/helpfulness targets, failure definitions, thresholds and alert **placement**; engineering owns traces, dashboards, alerts and guardrail enforcement |
| **Human-label economics** | "We'll have humans check" | Knows labels cost money | Does the arithmetic — **25¢/question × ~20 risks × 20–30 questions ≈ 1,000 questions ≈ $250 per contract** — and concludes human labels belong on the calibration slice |
| **Latency reporting** | Mean latency | Reports P95 | Never a percentile without the timeout rate and n; TTFT separate; segmented; latency on the same dashboard as quality and cost |
| **Cost** | No cost dimension | Knows it costs money | Decimal arithmetic, input/output split, per-query ceiling, step-count lever, agent and grader lines separated, cache traps named |
| **Pricing design** | Prices from the inference bill | Knows cost must be a design constraint | Prices **value-first**: hours saved × the customer's own hourly rate → a **charge matrix** the buyer recognises → target margin → **ceiling cost (P80+ band, averaged)** → works **backwards** into model size, eval depth and tolerable accuracy; checks **attribution × autonomy** before considering outcome pricing; applies the **80% rule** to the retainer |
| **COGS completeness** | Counts tokens | Counts tokens and infra | Counts **harness cost as the largest line**, then adds training, evaluation, monitoring, human-in-the-loop and forward-deployed engineering **before** the margin — and can quote the Harvey shape: **$100 to run, $1,000+ to evaluate, $1,000 charged** |
| **Loop closing** | Fixes the bug | Adds it to the dataset | Versioned prompt, experiment, CI gate, and a coverage audit against last quarter's incidents |
| **Meta-measurement** | Trusts the metric | Knows judges can drift | Canary set, offline→online correlation, discriminative-power check, and an honest statement of coverage gaps |
| **Decision link** | Reports numbers | Reports numbers someone reads | Deletes metrics that change no decision, and can state what each remaining metric decides |

**The single strongest signal in this domain:** the candidate says that **a score is only actionable
next to the latency and cost of the same interaction**, and can then explain why the trace — not the
dashboard — is the system of record.

**The second strongest**, and the one that separates a senior hire from a staff hire: the candidate
can say **what the evaluation is *for*.** Not "quality" — the answer CS-20 argues for and CS-21
operationalises is that the eval stack is the **cost accounting system that makes pricing decidable**.
Task completion is the gate, adherence is the cost control, per-customer margin is the business
decision, and you cannot price an outcome you cannot measure. A candidate who frames evaluation as a
QA function will build dashboards; a candidate who frames it as the instrumentation behind per-customer
unit economics will build a product.
