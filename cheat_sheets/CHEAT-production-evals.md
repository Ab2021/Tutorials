# Cheat Sheet · Production & Online Evaluation

> **Covers:** CS-20 (evals that thrive in prod — the harness as the evaluation target, HHH+R, task
> completion / adherence / navigation, tool metrics, ROI-first), CS-21 (observability, traces, alerts,
> red teaming), CS-22 (scaling evals), CS-23 (pricing agents — value-first order, ceiling cost, credits,
> the 80% rule, the eval bill as COGS), CS-24 (the harness shift, and the in-loop "evaluate" that is not
> an eval suite) · **Code ground truth:** `CODE-06` (Langfuse — the platform all of this lives in),
> `CODE-07` (search_evals — cost accounting) · **Use when:** your system is live and the question is no
> longer "does it work" but "is it still working, for whom, and at what price".

---

## The 60-second version

Offline evaluation answers *should we ship*. Online evaluation answers *is it still working, for the
users we actually have, at the price we actually pay*. They must use the **same evaluators**, or the
offline number stops predicting the online one.

The precise statement of the split: **offline checks whether the application works *correctly*; online
tells you whether it is running *normally*.** Online evaluation's defining feature is that it works
**without answers and without a golden dataset** — which means the one thing it structurally cannot do
is measure correctness, because no ground truth arrives with live traffic. What you compare is today's
distribution against a **baseline that your offline evaluation produced**.

The organising idea is that **a score is only actionable if it sits next to the latency and cost of
the same interaction**. That is why evaluation in production is a property of the *trace store*, not a
batch job: traces, scores, datasets, experiments, prompt versions and human feedback all attach to one
object, and a bad trace leads directly to the prompt change that fixes it.

Four things make online evaluation work: **tracing** (the substrate), **deterministic sampling** (so
retries score the same items), **targeting** (you never evaluate everything), and **drift alerting**
(a window, a threshold, and a remediation path).

**The object you are evaluating is a harness, not a model and not a tool.** LLM + **context**
(knowledge = how *your* world works, memory = current state) + **tools** + **skills** + **guardrails**.
That is what ships, so that is what the evaluators must describe. Corollary: to evaluate a *tool*,
wrap it in an agent and evaluate the harness around it.

**And order the evaluators by cost** — cheap objective gates before expensive subjective ones.
A failure at an earlier layer makes every later layer uninterpretable:

```
task completion  ->  task adherence  ->  task navigation  ->  quality  ->  context  ->  guardrails/cost
(binary, free)      (cost control)      (was the plan right) (expert rubric)  (4 metrics)   (checks)
```

---

## Core concepts

| Concept | Meaning | Why it matters |
|---|---|---|
| **Trace** | One interaction: input, output, retrieved docs, latency, cost, tokens | The unit everything attaches to |
| **Observation / span** | A step *inside* a trace (an LLM call, a retrieval, a tool) | Lets you evaluate a sub-step, not just the whole |
| **Score** | Any evaluative signal — judge, code check, human label, end-user thumb | One contract for four very different sources |
| **Harness** | LLM + context + tools + skills + guardrails | **This** is the evaluation target — not the model, not the tool |
| **Knowledge** | Static context: how *your* world works | The half of context that RAG supplies |
| **Memory** | Dynamic context: current state, current document, conversation history | The half that makes multi-turn work |
| **Task completion** | Did the job get done — binary | The objective gate; the numerator of outcome pricing |
| **Task adherence** | Did it follow its instructions and constraints while doing so | **The cost control** — bad adherence means retry loops, the 60-min-vs-6-min failure |
| **Task navigation** | Was the *plan* right, and was it followed | Separates planning failure from execution failure |
| **Tool dispatch** | Which tool was chosen for a step | "Every tool call is a cost, double cost to you" |
| **Tool parallelism** | Were independent calls issued concurrently | Direct latency and cost lever |
| **Tool utilization** | Did the agent actually *use* the tool's output | A tool can be called successfully and its output ignored |
| **Groundedness** | Is the assertion backed by retrieved content | The anti-hallucination context metric |
| **Completeness** | How much of the needed information retrieval actually supplied | Recall side — know how much is missing |
| **ROI-first agent** | Design to a cost target derived from the displaced human's willingness to pay | Decides *which* agents are worth building at all |
| **Online eval** | Evaluators run on sampled production traffic | Real distribution, real failures |
| **Offline eval** | Evaluators run on a frozen dataset | Repeatable; the deploy gate |
| **Deterministic sampling** | The same trace is always in or out of scope for a given config | A `Math.random()` sampler silently makes retries incomparable |
| **Stratified sampling** | Bucket conversations by category, then over-weight the problematic ones | "Not all conversations are the same" — random sampling spends budget on the ones that were fine |
| **Targeting** | Which evaluator applies to which subset | You never evaluate everything — and shouldn't |
| **Annotation queue** | Human review routed to sampled traces | Produces the labels that validate your judges |
| **Drift** | A metric degrading relative to the world | Detected only by a windowed online metric |
| **Baseline** | The reference value a live metric is judged against | It comes from your **offline** evaluation — without one, no dashboard number is interpretable |
| **Red teaming** | Adversarial probing of the live system — CS-20's "**beyond evaluations**" | The failures an eval suite was never designed to catch |
| **Prompt version** | A versioned, cacheable prompt object | Closes the loop: bad trace → playground → version → experiment |
| **Dataset / experiment** | Frozen set + candidate config + side-by-side diff | Offline evaluation with a UI, and the CI-gating substrate |
| **Negative-margin flag** | Alerting when a customer costs more than they pay | The eval stack is a **cost accounting system**, not a QA system |

---

## Formulas & metrics

**Sampling**

```
sampled_volume = traffic_rate * sampling_pct
judge_spend    = sampled_volume * judge_cost_per_call
```
Sampling must be **deterministic** — a function of a stable trace identifier, not a random draw — so a
retried or replayed run scores the *same* items. A non-deterministic sampler doubles spend and makes
runs incomparable.

**Drift detection**

```
window      = 24 h rolling graph on ONE metric (e.g. faithfulness)
alert rule  = metric degrades over the LAST 8 h
response    = alert + a defined remediation path (not just a dashboard)
```
The window matters: too short and variance triggers the alert; too long and you learn about the
regression a day late. Pick it against the metric's own noise floor.

**Cost in production**

```
cost_per_session = Σ cost_per_trace
cost_per_query   = (in_tok/1e6 * in_rate) + (out_tok/1e6 * out_rate)
```
Track it as a **distribution**, not a total — a typical 2-paise query with 1.5-rupee outliers is a
different business from a flat 2-paise one. Split input from output: output tokens typically cost
**~4×** input.

**Latency in production**

```
report = mean, median, P50, P95, P99, min, max   +  TIMEOUT RATE  +  sample size
TTFT   = time to first token (needs streaming to observe)
```
**Never publish a percentile without the timeout rate.** Timed-out requests are excluded from the
distribution, so a "faster" P95 can mean you are dropping more requests.

**Reliability**

```
success rate | error rate (1 - success) | timeout rate | retry rate
```
Four numbers, **categorised by cause**: LLM API | retriever | reranker | timeout | rate limit |
parser/format | internal exception. One generic error rate tells you nothing actionable.

---

## Decision rules

1. **Use the same evaluators offline and online.** Otherwise the offline number is not predictive and
   the whole program loses credibility.
2. **Trace first.** Without per-interaction records, no online metric, drift alert or feedback loop
   exists.
3. **Sample deterministically**, record the sampling config with the run, and **stratify** — over-weight
   thumbs-down, escalations, abrupt endings, repeated rephrased questions and money/refund topics.
   Random sampling is the wrong default.
4. **Target, don't blanket.** Define which evaluator applies to which subset — full-traffic judge
   evaluation is a budget hole with no decision value.
5. **Route humans to a stratified sample**, not to everything; their output is the judge-validation
   set.
6. **One metric per alert, with a window and a remediation path.** An alert without an owner and a
   response is noise.
7. **Put quality, cost and latency on the same dashboard.** A quality gain that triples cost is only
   visible if the two numbers share a screen.
8. **Version prompts and evaluate prompt changes as releases.** A prompt edit is a deploy.
9. **Red-team the live system on a cadence**, separately from the regression suite — adversarial
   failures are not in your golden set by construction.
10. **Harvest production failures into the golden dataset** — this is the loop that keeps offline
    evaluation honest.
11. **Instrument the eval suite's own spend.** Grading is part of the run's cost.
12. **Tier the cadence**: free checks per commit, sampled judges on PRs, full suite nightly and
    pre-release.
13. **Treat throughput separately.** It cannot be measured by replaying a handful of requests from a
    laptop; it needs load and stress testing with concurrent traffic.
14. **Order the evaluators by cost** — task completion → adherence → navigation → quality → context →
    guardrails. An early-layer failure makes every later layer uninterpretable.
15. **Never let completion stand alone.** A run that finishes the task while ignoring its constraints
    is the 60-minutes-for-a-6-minute-job failure: completion green, margin destroyed. Measure
    **adherence** or you cannot see it.
16. **Count every tool call.** Dispatch, parallelism and **utilization** — a tool that is called and
    whose output is ignored is invisible to any call-success metric.
17. **Instrument retrieval *volume*, not just relevance** — over-retrieval is quality-neutral and
    cost-negative, and only shows up as tokens-retrieved-per-query.
18. **Price from the displaced human's willingness to pay, not from your inference bill.** Set the
    agent cost target (≤ **$1**, ideally **50¢**), then work backwards to whether it can be built.
19. **Never deploy a judge before measuring its precision and recall against expert labels.** A judge
    below the bar is not a cheap expert — it is a coin flip. Below the bar, **pay the expert.**
20. **Turn subjective criteria into yes/no facts by asking *why*.** "Should have insurance in it" →
    "does the clause have insurance or not?"
21. **On find-the-items tasks, report precision and recall, not accuracy.** Accuracy hides both the
    false positive and the false negative, and they have opposite product consequences.
22. **Price from value, before you price from cost.** Hours saved × the customer's own hourly rate →
    charge matrix → margin → **ceiling cost** → backwards into model size, eval depth and tolerable
    accuracy. A price set from your inference bill is a price set from the wrong end.
23. **Price on the P80+ ceiling, never on the mean.** Your first customers are the hungriest ones; the
    mean is the margin story, the ceiling is the pricing story.
24. **Count the eval bill inside cost of goods**, before the margin — not as overhead you pay instead of
    shipping. The Harvey shape ($100 run / $1,000+ eval / $1,000 charged) is the cautionary case.
25. **Anchor any credit to a unit of the customer's work**, never to tokens. An unanchored credit is a
    black box and the CFO escalates.
26. **Size the retainer so 80% of users never hit it** — a violated 80% rule shows up as churn, not as
    a pricing error.
27. **Do not read the agent's in-loop "evaluate" step as an eval suite.** In-loop self-checking gives
    termination; only an offline suite with golden answers gives evidence.

---

## The escalation ladder (CS-22) — never skip a rung

```
1 HUMAN EXPERT      yes/no against ground truth        always works, costs experts
2 PRECISION/RECALL  accuracy vs that ground truth      once the expert has labelled a set
3 SIMILARITY        BLEU / ROUGE vs a reference        when a reference answer exists
4 LLM AS A JUDGE    must pass ITS OWN precision/recall test
5 SCALE             only above the judge bar
```

**You move up a rung only when the rung you are leaving has measured the rung you are entering.**

**The measured finding that justifies the rule:** when a team computed their judge's precision and
recall against expert labels, **both came in under 40%** — "a **flip of a coin**." A judge that is
wrong more often than right is a random number generator wearing a rubric.

**The deployment bar:** replace the human expert only above ~**80% precision** (restated as **70%**).
Below it: "**don't do it. Still do whatever the lawyer is saying and hire the lawyers. Don't
compromise your product because you don't have money.**" The alternative to a validated judge is not a
cheap judge — **it is paying the expert.**

**Turn a subjective criterion into a checkable one by asking *why*.** "The indemnity should have
insurance in it" (unlabelable) → "**does the indemnification clause have insurance or not?**"
(checkable). Subjectivity is not eliminated, it is **decomposed** until what remains is yes/no.

**On find-the-items tasks use precision and recall, not accuracy.** Expert names **10** key terms; agent
names **12** (all 10 + 2 extras) → **precision = 10/12**; agent finds **8** of 10 → **recall = 8/10**.
Accuracy hides both. A false positive sends a clause that was never there; a false negative silently
drops a risk — **which one you tolerate is a domain judgement, not a statistical one.**

---

## The four pillars, and who owns what

"Observability" is a vendor word hiding four capabilities. Only the first is old.

| Pillar | Question | Owner |
|---|---|---|
| **Monitoring** | Is it up? uptime, failed requests, response time | Engineering |
| **Evaluation** | Is it **right**? vs ground truth and criteria | PM sets criteria; engineering builds the runners |
| **Guardrails** | Did it stay inside policy and privilege? | PM defines; engineering enforces |
| **Red teaming** | Can we break it on purpose? injection / corruption / sandwiching | Scheduled exercise, "beyond evaluations" |

**Transport: emit OpenTelemetry.** A 5-year-old open standard every cloud supports; portability is the
point ("if I go to AWS tomorrow"). Wiring it into a cloud monitor ≈ **1 day** of dev effort. A
proprietary trace format is a lock-in decision made by accident.

**The responsibility line (CS-21's deliverable):** the **PM owns** success criteria, cost targets
(per session / user / task), helpfulness targets, failure definitions, thresholds and alert
*placement*. **Engineering owns** traces, dashboards, alerts and guardrail enforcement. Engineering
builds the *mechanism*; the PM decides *where the line goes*.

**Alerting checklist — the three attack classes to red-team:** **injection** (hostile instructions via
user input, retrieved docs, or tool output), **corruption** (poisoning a trusted context or source),
**sandwiching** (a hostile instruction wrapped inside content marked benign). Different trust
boundaries — one passing does not imply the others do.

---

## The launch ramp (CS-20)

Do not go from "manual evals" to "launched". Go:

| Stage | Gate |
|---|---|
| **Internal / manual** | Observe the real number — e.g. **22%** task completion, **52%** helpfulness, **78%** retrieval relevance, guardrails passing |
| **Set the target** | Name the bar — "60% helpful, 70% honest" |
| **Canary** | Launch to **1–2%** of customers |
| **Ramp** | Widen as the numbers hold |
| **Steady state** | Continuous evaluation + per-customer margin tracking, **flagging negative margins** |

**Author the criteria with a domain expert:** write the questions a helpful human would answer; ask for
**pass/fail first, then why** — the "why" is what teaches you their world and tells you which prompt,
instruction or context to change. Convert every failure into a backlog item.

---

## Pricing an agent (CS-23) — value first, ceiling cost, credits

**The order is the framework. Never reorder it.**

```
1. VALUE          hours saved x the customer's own hourly rate
                  worked: 6 hrs saved @ $200/hr  =  $1,200 of work
2. ATTRIBUTION    high/high -> outcome pricing allowed ("the holy grail")
   & AUTONOMY     high attribution + LOW autonomy = almost everything shipped
                  -> price the UNIT OF WORK, not the outcome
3. CHARGE MATRIX  a countable unit of the job, not a token count
                  worked: 1 page = 500 words; playbook of 50 checks; bundle = 1 contract
4. MARGIN         worked example: 40%
5. CEILING COST   average of the P80-and-above band  (NEVER the mean)
6. WORK BACKWARDS -> eval depth, model size, agent complexity, tolerable accuracy
```

**Ceiling cost, not average cost.** Price on `avg(cost | P80 and above)`. The first 100 customers are
the hungriest ones — the ones nobody else could serve — so they sit in the tail by construction:
*"they will buy you and make you bankrupt."* Mean cost is the **margin** story; ceiling cost is the
**pricing** story. Worked ceiling: **$100/run**, max **3 hours**, **5** downstream agents, **6**
services/tools/checks/guardrails/knowledge.

**The three cost buckets, in the observed proportion** — this inverts the intuition:

| Bucket | Contents | Observed size |
|---|---|---|
| **Agent / harness** | retrievals, tool calls, guardrails, knowledge base | **largest** |
| **Infra** | app services, containers, observability | smaller |
| **Model** | token spend | **smallest** |

Then add the lines nobody counts, **before** the margin: **training, evaluation, monitoring,
human-in-the-loop, customisation, forward-deployed engineering (FDE)**. HITL includes the vendor's own
CEO, sales and CTO — so it scales **per customer**, not per business.

**The eval bill is a COGS line, not an invoice line.** The Harvey shape: **$100 to run**, **$1,000+ to
evaluate**, **$1,000 charged** → **$100 loss per customer**, covered by funding. *"Nobody's accounting
for that. If you count for that then the largest biggest ticket item is here."*

**The competitor map** (same buyer, same job, three metering axes — and the human baseline under all
of them):

| Player | Model | Number |
|---|---|---|
| **Sierra** | outcome-based | ~**$200–350k** |
| **Decagon** | per-conversation / per-resolution | ~**$400k** |
| **Finn / Intercom** | platform fee + per-resolution | ~**$40k** + **$1/resolution** · **65%** guarantee or **$65k back** |
| **Human** | hourly | **$35/hr** · ~**5 tickets** → **$7/resolution** |

The human line carries the argument: $7 vs $1 is a **~7× undercut** — and a cheaper per-resolution
price with a weaker guarantee can be **more expensive in expectation**, with a fixed platform fee
sitting above it that the per-resolution number says nothing about.

**Credits** — the metering unit to reach for, with exactly one failure mode.

- They give the customer a **round number** instead of a token count.
- They let the vendor **absorb model-price shocks**: when models get dearer, a credit buys less and
  margin holds.
- They **re-align sales** — a flat-subscription vendor has no reason to track usage; a credit vendor
  does.
- ⚠ **The failure mode is an unanchored credit.** *"I bought $25 of credit, I don't know how much
  coding will I get out of that $25."* Fix: anchor the credit to the charge matrix — **1 contract
  credit = a 200-page contract with 10 questions**.

**The 80% rule.** Size the retainer so **80% of users never hit its limit**, with small/medium/large
workloads all fitting inside it. Otherwise every interaction carries an upsell and *"your customers
will just throw the product."*

**Why the SaaS template breaks, and why outcome pricing is blocked.**

| Claim | The reason |
|---|---|
| SaaS margins were **70–80%**, so a flat seat price worked | near-zero marginal cost + system of record |
| Agents break it | token cost is non-deterministic, scales with success, invisible at sale |
| The spread is **100×** | median customer **$20** vs P90 customer **$2,000** |
| Outcome pricing is blocked (1) | agents hallucinate — you cannot stand behind the outcome |
| Outcome pricing is blocked (2) | they **do not learn from mistakes** like a human hire |
| Outcome pricing is blocked (3) | no **10x-on-the-bet curve** — *"that's the bet we make on humans. We can't make the same bets here"* |
| Why loss-making agent companies survive | the **Rule of 40**: growth % + profit % ≥ **40** — 100% growth permits 60% losses |
| Why pure usage fails | low attribution + AWS pay-as-you-go inheritance; **AgentForce on its 3rd pricing iteration in year one** |

**Instrument per request, or none of it is computable:** `user_id`, `customer_id`, `feature`, `prompt`,
`agent_run_id`, `deployment`. User ≠ customer (individual vs enterprise) — the pair is what makes
cost-to-serve attributable. Dashboard: total runs, user growth, committed revenue, **margin trend**,
drilled to customer, then to user.

> ⚠ **CS-23's caveats.** Same host as CS-20/21/22 — mutually corroborating but **not independent**.
> Its hours-saved arithmetic is garbled in the transcript (only `$200 × 6 hrs = $1,200` reconciles).
> The Azure estimator demo shows an implausible identical **$600** across two very different models,
> which the host calls a tool bug. A cost-dashboard session and a pricing calculator are announced and
> **never delivered** — like CS-20's "Mahesh minimum list".

---

## The harness shift, and the "evaluate" that is not an eval (CS-24)

The 2025 division of labour: you no longer build memory loading, MCP servers, guardrail plumbing and
context assembly. You **adopt a harness** — and the harness is what is bought.

```
before:  build agent + memory + MCP servers + guardrails
         + "get the right context all the time"
after:   adopt a harness  ->  plumbing is not your problem
```

**What the harness supplies, out of the box:**

| Mechanism | What it replaces |
|---|---|
| **File-system access with plain search** (not embeddings) | building and syncing an embedding index |
| **Add files / folders** | deciding what the model may see |
| **Memory + browser plugin + connectors** | a search API; writing MCP servers |

**The eval trap in this session, and the reason to read it.** The agentic loop is
`context → inspect sources → decide & act → evaluate → plan → "is the job done?"`. That **`evaluate`
step is a runtime self-check inside the trajectory — not an eval suite.** It has no dataset, no judge,
no threshold and no golden answers; done-ness lives in the **prompt + skills file + `CLAUDE.md`**, which
is a textual termination condition written by a human.

> ⚠ When someone says "our agent evaluates itself in the loop, so we don't need an offline eval," this
> is the confusion. **In-loop self-checking gives you termination. It does not give you evidence.**
> Discipline the in-loop step with trajectory evaluation (CS-17) and measure quality offline
> (CS-20–CS-22).

**Two caveats worth stating.** The session credits **two** unblocks, not one — the loop *and* the jump
in model capability ("Opus 4 onwards the models became really smart at this also"); a harness around a
weaker model does not reproduce the result. And the filename promises an "n8n limitation" the
transcript never states — describe the three capabilities, do not invent the defect.

---

## Thresholds & defaults worth memorising

| Item | Value |
|---|---|
| Drift window | **24 h** graph, alert on degradation over the **last 8 h** |
| CI regression gate | block if any metric drops **> 3 units** vs baseline |
| Latency budget | P95 ≤ **3,000 ms**; TTFT ≤ **1,200 ms** |
| Cost budget | express as a **per-query ceiling** (e.g. 50 paise on a complex query) |
| Reliability samples | 25–50 far too few → toward **1,000** for rare errors |
| Output vs input token rate | ≈ **4 ×** |
| Judge ship bar | TPR ≥ 0.9 **and** TNR ≥ 0.9 — plus, per CS-22, **≥ 80%** precision (restated **70%**) before a judge replaces the expert; one measured **< 40%** is "a flip of a coin" |
| Storage split | ClickHouse for traces/observations at volume; Postgres for metadata; S3 for blobs |
| Perceived latency | the user experiences **TTFT**, not total — 1.6 s to first token feels different from 3.6 s total |
| Cache caution | repeated identical questions inflate provider prefix caching → cost is understated |
| **Eval run cost** | ~**10×** the serving token spend · **16 min** for one run · scales with **corpus**, not traffic |
| **Launch gate** | helpfulness **60%**, honesty **70%**, harmlessness **< 5%** — **then** release to **1%** of users |
| **Team targets (CS-22)** | **+20%** completed tasks · **+40%** helpfulness · **−30%** cost |
| **Human-label cost** | **25¢**/question × ~**20** risks × **20–30** questions ≈ **1,000** Q ≈ **$250** per contract |
| **Alert floors** | ground-truth accuracy **< 30%** · tool-calling failure **> 40%** · guardrail triggers **> 5%** |
| **Runaway-cost war stories** | 3-hour agent session billed **$786** for 5 min of human work · a company's IT budget gone in **3 months**, **9×** MoM growth |
| **Over-guarding** | Google's first support agent denied **6 of 10** customers — every health metric green |
| **Eval time budget** | **30–40%** of your time on evals ("breaking your agents") |
| **Agents reaching production** | ~**80–90%** of companies have agents, only ~**15%** reach production — evals are the gate |
| **ROI-first cost target** | ≤ **$1**, ideally **50¢** per agent run |
| **Recurring eval cadence** | e.g. **100 interactions/day** evaluated on a schedule |
| **Pricing order (CS-23)** | value → attribution/autonomy → charge matrix → margin → **ceiling cost** → backwards into engineering |
| **Ceiling cost band** | average of the **P80 and above** band — **never** the mean |
| **Worked ceiling** | **$100**/run · 3 h max · **5** downstream agents · **6** services/tools/guardrails/KB |
| **Target margin (worked)** | **40%** |
| **Value anchor (worked)** | **6 h** saved × **$200/h** = **$1,200** of work |
| **Cost bucket ordering** | harness **largest** > infra > model **smallest** |
| **The 100× spread** | median customer **$20** vs P90 customer **$2,000** |
| **The 80% rule** | **80%** of users must never hit the retainer limit |
| **Rule of 40** | growth % + profit % ≥ **40** (100% growth funds 60% losses) |
| **Human resolution cost** | **$35/h** ÷ ~**5 tickets** = **$7**/resolution → **$1**/resolution is a ~**7×** undercut |
| **Competitor anchors (CS-23)** | Sierra outcome ~**$200–350k** · Decagon ~**$400k** · Finn ~**$40k** + **$1**/resolution, **65%** guarantee or **$65k** back |
| **Harvey unit economics** | **$100** to run · **$1,000+** to evaluate · **$1,000** charged = **$100 loss** per customer |
| **Cost instrumentation** | `user_id`, `customer_id`, `feature`, `prompt`, `agent_run_id`, `deployment` |
| **Harness connectors (CS-24)** | **2,137** out-of-the-box (Gmail, Slack, M365, SharePoint, OneDrive, Salesforce, Snowflake) |
| **Done-ness (CS-24)** | prompt + **skills file** + **`CLAUDE.md`** — a textual termination condition, **not** a threshold |

---

## Tool commands

```bash
python3 -m evals.eval_latency       # P50/P95/P99, TTFT — free, no judge, no golden data
python3 -m evals.eval_cost          # per-query cost, input/output split
python3 -m evals.eval_reliability   # success / error / timeout / retry, by cause
python run_evals.py                 # the full offline suite
python compare_to_baseline.py       # CI gate; exit != 0 blocks the deploy
```

```python
# Online path: trace -> targeting -> deterministic sampling -> evaluator -> score
with langfuse.start_as_current_span(name="rag-answer") as span:
    docs = retriever.search(question)
    answer = llm(question, docs)
    span.update(input=question, output=answer,
                metadata={"k": 5, "retrieved": [d.id for d in docs]})

langfuse.score_current_trace(name="groundedness", value=1.0,
                             comment="Every claim traceable to a retrieved passage.")
langfuse.flush()
```

**The four planes to understand before self-hosting a platform** (from `CODE-06`):

| Plane | Responsibility |
|---|---|
| Ingest/query | UI, API, auth, entitlements |
| Async work | evaluation runs, experiments, exports, retention, OTel ingestion |
| Storage | traces/observations at volume vs metadata vs blobs |
| Contracts | the shared table definitions everything writes through |

**The evaluation-path files worth reading in any platform:** the eval orchestrator, the judge executor,
the **code-based** evaluators directory, the **deterministic sampling** module, the score-event writer,
the observation-level evaluator, the **decision model** (which evaluator applies to which target), and
the trace filter utilities.

---

## Top 10 mistakes

1. **Online and offline using different evaluators**, so the offline number predicts nothing.
2. **No tracing**, so a "quality dashboard" is built from something other than real interactions.
3. **Non-deterministic sampling** — retries score different items and runs become incomparable.
4. **Evaluating 100% of traffic** with a judge, then discovering the bill.
5. **A drift alert with no window, no threshold and no remediation path.**
6. **Publishing a latency percentile with no timeout rate and no sample size.**
7. **Quality on one dashboard, cost on another** — so the trade-off is invisible.
8. **Treating a prompt change as a non-release** with no evaluation and no version bump.
9. **No red teaming**, so adversarial failures are discovered by users.
10. **A golden set that never receives production failures**, so offline evaluation drifts away from
    reality while looking green.

---

## If you only remember three things

1. **A score is only actionable if it sits next to the latency and cost of the same interaction** —
   evaluation in production is a property of the trace store, not a batch job.
2. **Same evaluators offline and online; deterministic sampling; target, don't blanket.** Without
   those three, online numbers are not comparable to anything.
3. **Window + threshold + remediation path** for drift, and always publish a percentile together with
   its timeout rate and sample size.

**And two from the agent tier:** you are evaluating a **harness** (LLM + context + tools + skills +
guardrails), not a model — and **order the evaluators by cost**, because a failure at an early layer
makes every later layer uninterpretable. Completion is green; **adherence is the margin.**

**And two from the money tier:** **the eval bill is a cost-of-goods line, not an invoice line** — the
Harvey shape is `$100 to run / $1,000+ to evaluate / $1,000 charged`, and a team that cannot name its
eval cost per customer does not know its unit economics. Price from **value** and on the **P80+
ceiling**, never from your inference bill and never on the mean. And when someone says the agent
already evaluates itself in its loop — that is a **termination condition**, not evidence.
