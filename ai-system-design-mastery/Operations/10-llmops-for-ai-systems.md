# LLMOps for AI Systems

**TL;DR** — LLMOps is MLOps with three inversions. First, **quality is a first-class production metric**: a fast, available system producing bad outputs is failing, and no traditional monitoring stack will tell you, because the failure has a 200 status code. Second, **the thing you ship is not an artifact but a versioned release manifest** — code, prompt, model snapshot, index, guardrails, and config each version independently, and `release_id` is what makes an answer reproducible. Third, **evaluation and observability are the same system viewed from two angles**: offline evals are what you run before shipping, online sampling is what you run after, and the same judge calibration pipeline serves both. The capstone insight for 2026 is that "LLM-as-judge" is now the *worst-case fallback, not the default*: a serious stack runs **distilled judges inline on every trace, a frontier judge on a 1–5% sample to detect judge drift, and humans on disagreements to set ground truth** — because serving a frontier judge on every production trace stopped being affordable above roughly 100,000 requests a day.

> **60-second interview answer:** "I'd define LLMOps as the operational discipline around three facts. The system is composed of six independently-versioned things — code, prompt, model snapshot, index, guardrails, config — so I don't deploy a model, I deploy a pinned release manifest with a `release_id` that appears in every trace. Quality is a production metric that infrastructure monitoring cannot see, so I run a layered judge architecture: distilled judges inline on every trace for the failure taxonomy they cover, a frontier judge on a sampled one to five percent to catch drift between the two, and human review on disagreements to keep the gold set honest. And evaluation is continuous, not a phase — the same harness runs in CI before the release and on production traffic after it. The number I'd defend hardest is the sampled quality score, because it's the only signal that catches the failure that matters: everything green and the answers worse."

---

## 1. Purpose

Define how an AI system is versioned, released, observed, evaluated, costed, governed, and repaired over its whole life.

**LLMOps vs MLOps — the four differences that matter:**

| | Classical MLOps | LLMOps |
|---|---|---|
| **The artifact** | One trained model file | **Six independently-versioned assets** in a manifest |
| **Correctness** | Objective metric (F1, AUC) | **Subjective, multi-dimensional; no single ground truth** |
| **Determinism** | Same input → same output | **Same input may produce different output** |
| **The dependency** | Your own serving stack | **A third-party API that changes under you** |
| **Failure visibility** | Metrics move | **Metrics stay green while quality drops** |
| **Cost shape** | Fixed compute | **Per-token, variable, easily 10x on a bad day** |
| **Evaluation** | Before deploy | **Continuously, before and after** |

**The framing to lead with:** *"Classical MLOps asks 'did the model regress?'. LLMOps has to ask 'did the model regress, the prompt regress, the index go stale, the provider change the snapshot, or the retrieval recall drop?' — and the answer to all five looks identical from the outside: the answers got worse and every dashboard is green. That's why the defining LLMOps capability isn't deployment, it's measurement."*

**In scope:** the release manifest, the operational lifecycle, tracing and metrics, the layered eval stack, quality-drift detection, cost operations, governance evidence, incident response.

**Out of scope:** the mechanics of each subsystem — those live in [deployment](04-deployment-release-and-cicd.md), [capacity](05-capacity-planning-and-autoscaling.md), [cost](06-cost-and-finops.md), [reliability](07-reliability-patterns.md), [security and compliance](08-security-guardrails-and-compliance.md), and [DR](09-dr-failover-and-backup.md). This file is the view that connects them.

---

## 2. Scope — the six independently-versioned assets

```
   A RELEASE IS NOT A MODEL. IT IS A MANIFEST.

   ┌──────────────────────────────────────────────────────┐
   │  release_id: rel_2026_09_27_a3f1                     │
   ├──────────────────────────────────────────────────────┤
   │  code:        git sha 8c3f1d2                        │
   │  prompt:      template@v47 (sha 2ab9...)             │
   │  model:       claude-sonnet-4-6-20260115  (PINNED)   │
   │  embeddings:  text-embedding-3-large@2024-02 (PINNED)│
   │  index:       knowledge_v7 @ snapshot 2026-09-27T02  │
   │  guardrails:  guardrails@v12 (sha 7f1c...)           │
   │  config:      runtime.yaml sha 9d4e...               │
   │  chunker:     chunking.yaml sha 1a2b...              │
   ├──────────────────────────────────────────────────────┤
   │  RULES                                               │
   │  ① every field immutable once released               │
   │  ② the pointer is the ONLY deploy change             │
   │  ③ release_id appears in EVERY trace and log         │
   │  ④ rollback = repoint, not rebuild                   │
   └──────────────────────────────────────────────────────┘
```

**Why this matters more than any other LLMOps idea:** *"If `release_id` isn't in the trace, then when quality drops I cannot answer the only question that matters — what changed? Six assets move independently, and if I can't attribute an answer to a specific combination, I'm debugging by anecdote. So the manifest and the `release_id` in every trace is the foundation everything else in LLMOps stands on."*

**The six assets and how each fails:**

| Asset | Versions | Changes how often | Its failure mode |
|---|---|---|---|
| Prompt | Git, template ID | Weekly | Silent quality shift; injection surface change |
| Model | Pinned snapshot | Quarterly | Provider mutation; capability regression |
| Index | Snapshot + alias | Daily | Staleness; embed-drift; ACL gap |
| Guardrails | Git | Monthly | Over/under-blocking; latency |
| Config | Git + IaC | Continuous | Param drift; cache keys; timeouts |
| Code | Git sha | Daily | Everything ordinary |

---

## 3. SLIs and SLOs — the unified operational contract

### 3.1 The metric taxonomy

```
   ┌─────────────────────────────────────────────────────────┐
   │  OPERATIONAL — table stakes, classical                  │
   │   request rate · error rate · p50/p95/p99 latency       │
   │   TTFT · availability · saturation                      │
   ├─────────────────────────────────────────────────────────┤
   │  QUALITY — the LLMOps differentiator                    │
   │   sampled judge score · faithfulness · relevance        │
   │   hallucination rate · refusal rate · schema-valid rate │
   │   thumbs up/down · regeneration rate · task completion  │
   ├─────────────────────────────────────────────────────────┤
   │  COST — variable, per-token, attributable               │
   │   cost/request · cost/conversation · cost/task ·        │
   │   cost/outcome · tokens in/out · cache hit rate         │
   ├─────────────────────────────────────────────────────────┤
   │  AGENT — for trajectories, not answers                  │
   │   steps/task · tool-error rate · loop detection ·       │
   │   partial-completion rate · trajectory score · Pass^k   │
   └─────────────────────────────────────────────────────────┘
```

### 3.2 The unified SLI table

| Class | SLI | Target | Alert | Owner |
|---|---|---|---|---|
| Ops | Availability | 99.9% | < 99.9% / 5 m | Platform |
| Ops | Error rate | ≤ 0.5% | > 5% / 5 m | Platform |
| Ops | Latency p95 | per route | > SLA / 5 m | Platform |
| Ops | TTFT p95 | ≤ 800 ms | > 1.5 s / 5 m | Platform |
| **Quality** | **Sampled judge score** | **≥ 4.2 / 5.0** | **< 3.8 / 30 m** | **Eval owner** |
| Quality | Faithfulness | ≥ 0.85 | < 0.80 / 1 h | Eval owner |
| Quality | Hallucination rate | ≤ 0.03 | > 0.06 / 1 h | Eval owner |
| Quality | Schema-valid rate | ≥ 0.995 | < 0.98 / 15 m | Platform |
| Quality | Refusal rate | 1–5% | outside band / 1 h | Eval owner |
| Quality | Regeneration rate | ≤ 8% | > 15% / 1 h | Product |
| Quality | Thumbs-up rate | ≥ 80% | < 70% / 24 h | Product |
| Cost | Cost per request | target | > 1.5× baseline / 1 h | FinOps |
| Cost | Cache hit rate | ≥ 30% semantic | < 20% / 2 h | FinOps |
| **Agent** | **Trajectory success** | **≥ 0.85** | **< 0.75 / 1 h** | Agent owner |
| Agent | **Pass^4** | **≥ 0.60** | < 0.45 | Agent owner |
| Agent | Partial completion | ≤ 0.02 | > 0.05 / 1 h | Agent owner |
| Agent | Mean steps/task | ≤ 8 | p99 > 20 | Agent owner |
| Judge | **Judge-human agreement** | **κ ≥ 0.6** | κ < 0.4 | Eval owner |
| Judge | Judge drift | ≤ 2 pts vs frontier | > 3 pts | Eval owner |

**The two rows people don't build, and the reason they matter:**

*"Judge-human agreement is where I'd spend a review cycle. A quality score can fall because the system changed or because the judge changed, and those need opposite responses. So I sample human labels monthly and track Cohen's kappa against the judge — below 0.4 is 'fair' or 'poor' agreement and means the judge is no longer measuring what I think it's measuring. Without that row, a judge regression looks exactly like a product regression, and I'd spend a week debugging the model when I should have been fixing the grader."*

*"And judge-drift against a frontier judge is the cheap early version of the same check: run a frontier judge on 1–5% of traces and compare it to the distilled judge's verdict. If they diverge by more than about three points, the distilled judge needs recalibration — it's trained on a fixed failure taxonomy and anything outside that taxonomy regresses to a default score."*

---

## 4. Architecture

```
══════════════ THE LLMOPS PLATFORM ══════════════

   ┌────────────────── CONTROL PLANE ──────────────────┐
   │  · Prompt registry (git, versioned, review)        │
   │  · Eval registry (datasets, judges, thresholds)    │
   │  · Release manifest store                          │
   │  · Gate orchestrator (CI quality/safety/cost)      │
   │  · Model & index catalogs (pins, versions)         │
   └───────────────────────┬────────────────────────────┘
                           │  deploys
                           ▼
   ┌────────────────── DATA PLANE ─────────────────────┐
   │  request → gateway → orchestrator → generate       │
   │  every hop emits a span                            │
   └───────────────────────┬────────────────────────────┘
                           │  traces + metrics + logs
                           ▼
   ┌────────────────── OBSERVABILITY ──────────────────┐
   │  trace store (spans: retrieval, rerank, gen)       │
   │  metrics (Prometheus) · logs (hashed + secured)    │
   │  cost attribution · guardrail trip log             │
   └───────────────────────┬────────────────────────────┘
                           │  sampled traces
                           ▼
   ┌────────────────── EVAL LAYER ─────────────────────┐
   │  ① DISTILLED JUDGES — inline, every trace          │
   │     taxonomy: groundedness, instruction-following, │
   │     toxicity, PII, off-topic, refusal              │
   │     emits a LABEL + a CONFIDENCE                   │
   │            │                                       │
   │            ├─ low confidence / high stakes ──┐     │
   │            ▼                                 ▼     │
   │  ② FRONTIER JUDGE — 1–5% sample      ③ PRM step   │
   │     calibration + drift detection      scoring     │
   │            │                                       │
   │            ▼   disagrees?                          │
   │  ④ HUMAN REVIEW → GOLD SET                         │
   │            │                                       │
   │            ▼                                       │
   │  ⑤ QUARTERLY DISTILLED JUDGE RETRAIN               │
   └───────────────────────┬────────────────────────────┘
                           │  drift · alerts · findings
                           ▼
   ┌────────────────── FEEDBACK LOOP ──────────────────┐
   │  quality drop → freeze pipeline → investigate →    │
   │  fix → candidate eval set → gate → release         │
   └───────────────────────────────────────────────────┘
```

### 4.1 Why the eval layer is layered

```
   THE COST CONSTRAINT THAT FORCES THE SHAPE

   Frontier judge on every trace:
     1M requests/day × $0.02 per judgement = $20,000/day
                                            = $600,000/month
     ⇒ unaffordable above roughly 100K req/day

   Distilled judges inline on every trace:
     1M × $0.0006 = $600/day = $18,000/month
     ⇒ ~97% cheaper, ~10x lower latency (sub-100ms short responses)
     ⇒ 88–92% agreement with the frontier judge

   Frontier judge on 2% sample:
     20,000 × $0.02 = $400/day = $12,000/month
     ⇒ affordable, and it's what tells you the distilled judge drifted

   Human review on disagreements:
     ~200/month × $15 = $3,000/month
     ⇒ this is where ground truth comes from, and nothing replaces it

   ────────────────────────────────────────────────────────────
   TOTAL ≈ $33,000/month vs $600,000 for frontier-on-everything
```

**The shape of the disagreement, which is the catch:** *"A distilled judge is trained on a fixed failure taxonomy — groundedness, instruction-following, toxicity, PII, off-topic, refusal. Inside that taxonomy it agrees with a frontier judge 88 to 92% of the time and lands within two or three points of human gold labels. Outside it, it regresses to a default score. So it's excellent for the failures you've already seen and blind to a freshly-released attack vector, a new category of user intent, or a domain-specific factuality check. That's exactly why the frontier sample exists, and exactly why I'd never ship a distilled judge as the only judge."*

### 4.2 Tracing — the spans that must exist

```python
async def rag_query(query: str) -> str:
    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("release_id", RELEASE_ID)      # ① mandatory
        span.set_attribute("tenant_id", tenant)           # ② for slicing
        span.set_attribute("session_id", session_id)

        with tracer.start_as_current_span("guardrail_input") as s:
            s.set_attribute("verdict", verdict)           # ③ safety
            s.set_attribute("score", score)

        with tracer.start_as_current_span("embed_query") as s:
            s.set_attribute("model", EMBED_MODEL_PIN)     # ④ which embedder
            s.set_attribute("embedding_dim", dim)

        with tracer.start_as_current_span("vector_search") as s:
            s.set_attribute("collection", "knowledge_v7") # ⑤ which index
            s.set_attribute("results_count", len(results))
            s.set_attribute("top_score", results[0].score)
            s.set_attribute("acl_filtered", n_filtered)   # ⑥ isolation proof

        with tracer.start_as_current_span("rerank") as s:
            s.set_attribute("in_count", len(results))
            s.set_attribute("out_count", len(reranked))

        with tracer.start_as_current_span("generate") as s:
            s.set_attribute("model", MODEL_PIN)           # ⑦ pinned snapshot
            s.set_attribute("input_tokens", n_in)
            s.set_attribute("output_tokens", n_out)
            s.set_attribute("cache_read_tokens", n_cached) # ⑧ cost truth
            s.set_attribute("ttft_ms", ttft)
            s.set_attribute("finish_reason", reason)

        with tracer.start_as_current_span("guardrail_output") as s:
            s.set_attribute("verdict", verdict)

        with tracer.start_as_current_span("cost") as s:
            s.set_attribute("usd", cost)                  # ⑨ attributed
```

**The three attributes people omit and then need:** `cache_read_tokens` (without it cost attribution is wrong by 30–70% on a cached system), `acl_filtered` (the only per-request proof that tenant isolation ran), and `release_id` (without it no incident is attributable).

### 4.3 Agent trajectories — what to trace beyond the answer

A final-answer score hides the three failure shapes that matter most:

| Failure | The answer looks | The trajectory shows |
|---|---|---|
| **Right answer, wrong reasoning** | Fine | Guessed after a botched calculation |
| **Right answer, dangerous path** | Fine | Four destructive tool calls before a safe one succeeded |
| **Right answer, runaway cost** | Fine | 47 retrieval calls where 2 sufficed |

**Trajectory failure modes to score for:**

| Mode | Signature |
|---|---|
| Reasoning-action mismatch | Chain-of-thought says one thing; tool call does another |
| Over-retrieval | More retrieval calls than the task needs |
| Tool flailing | Same tool with slight variations until something works |
| Premature commitment | Answer written before all evidence is in |
| Self-jailbreaking | The agent's own intermediate reasoning bypasses its own policy |

**The two methods:** process reward models score each step independently, and an auxiliary **auditor agent** — usually a different model from the one being graded — replays the trajectory asking "was this step justified?" at each node. *"The trajectory-judging layer exists because an agent that gets the right answer after four destructive tool calls is not a success, and no final-answer grader will ever tell you that."*

### 4.4 Pass^k — the agent reliability metric

```
   Pass^1 = success rate on a single trial          (traditional)
   Pass^k = probability of succeeding on ALL k trials

   EXAMPLE
     Pass^1 = 70%
     Pass^4 = 12%

   ⇒ "The agent works on the easy path but cannot recover
      from any small perturbation."

   That gap is the signal you need BEFORE rolling an agent
   out at scale, and it is invisible in a single-run success
   rate. A 70% Pass^1 sounds shippable; a 12% Pass^4 is a
   system that fails a user four times in a row on a bad day.

   PAIR IT WITH VERIFIER-BASED GRADING
     Determine success from deterministic post-conditions —
     the order is cancelled, the refund exists, the seat is
     changed — not from an LLM-graded transcript. Post-conditions
     are cheap, unambiguous, and impossible to talk your way past.
```

### 4.5 Memory-equipped systems — per-operation evals

For any system with memory, aggregate accuracy hides the stage where the error originates.

| Stage | What is measured | Typical failure |
|---|---|---|
| **Extraction** | The fact written matches the source | Stored "user is allergic to peanuts" when the source said "dislikes peanuts" |
| **Update** | The update is correct relative to prior state | New memory contradicts an older one with no resolution |
| **QA** | The answer is grounded in stored memories | Answered from parametric knowledge while appearing to cite memory |

*"A system can score very high on standard hallucination benchmarks while making catastrophic extraction errors, because the aggregate metric averages away the stage that's broken — and the stage is the only part you can actually fix. So every write, update, and read gets its own eval with its own error rate, and I'd watch the extraction rate hardest: a 5% extraction error compounds across thousands of operations into a wholly unreliable agent."*

---

## 5. Standard operating procedures

### SOP-1 — The operational lifecycle

```
   ┌────────────────────────────────────────────────────────────┐
   │  ① VERSION   everything in a manifest, nothing unversioned  │
   │  ② GATE      CI: static → unit → golden → quality → safety  │
   │              → cost, then shadow → canary → full            │
   │  ③ OBSERVE   traces, metrics, logs, cost, guardrail trips   │
   │  ④ EVALUATE  distilled inline + frontier sample + humans    │
   │  ⑤ COST      attribution per request, alert on deviation    │
   │  ⑥ GOVERN    audit trail, cards, disclosure, retention      │
   │  ⑦ REPAIR   incident → rollback → root cause → new test     │
   └────────────────────────────────────────────────────────────┘
   Every stage produces an artifact. The artifacts ARE the ops
   system: manifests, gate reports, traces, eval scores, cost
   records, audit logs, incident reports.
```

### SOP-2 — Standing up LLMOps from zero

**The 90-day sequence, in dependency order:**

```
DAYS 1–14 — VISIBILITY
  □ 1. release_id in every trace and log                    ← do this first
  □ 2. Token counts + cost per request, no exceptions
       (including notebooks and internal tools)
  □ 3. Latency split: TTFT vs total, per route
  □ 4. Error rate by class (retryable vs not)
  □ 5. Guardrail trip counters
  ⇒ You cannot operate what you cannot see. Nothing else starts
    until these five exist.

DAYS 15–30 — THE MANIFEST
  □ 6. Prompts out of code into a registry, versioned
  □ 7. Model snapshots PINNED (never a floating alias)
  □ 8. Index versioned with an alias pointer
  □ 9. release_id assembled from all six assets, immutable
  □ 10. Rollback = repoint the alias. Test it. Time it.
  ⇒ Target: a 90-second rollback, exercised.

DAYS 31–60 — EVALUATION
  □ 11. Build a golden set: 200–500 examples, hand-labelled
  □ 12. Define 5–10 quality criteria with a rubric
  □ 13. Wire the eval harness into CI as a BLOCKING gate
  □ 14. Add guardrail + injection + PII suites
  □ 15. Publish a quality score with a threshold, not a dashboard
  ⇒ Target: no release ships without a gate report.

DAYS 61–90 — CONTINUOUS OPERATION
  □ 16. Distilled judges inline on 100% of traces
  □ 17. Frontier judge on a 1–5% sample
  □ 18. Human review on disagreements → gold set updates
  □ 19. Quality SLO with its own alert and pager
  □ 20. Cost attribution per team/feature/tenant
  □ 21. Incident runbook + first drill
  ⇒ Target: quality regressions page someone.
```

**The sequencing rationale:** *"The temptation is to start with evals because they're the intellectually interesting part. But evals without `release_id` and cost attribution produce scores you can't act on, because you can't tell what changed or what it cost. So visibility comes first, then the manifest — which is what makes change attributable — and only then the eval layer, which is what makes the change detectable."*

### SOP-3 — The eval harness

```python
class EvaluationPipeline:
    def __init__(self, sut, evaluators, dataset):
        self.sut = sut                    # system under test
        self.evaluators = evaluators      # one per metric
        self.dataset = dataset            # list[EvalSample]

    def run(self) -> EvalReport:
        results = []
        for sample in self.dataset:
            prediction = self.sut.generate(sample.input)
            scores = {
                ev.name: ev.evaluate(
                    input=sample.input,
                    prediction=prediction,
                    reference=sample.expected_output,
                    context=sample.context,
                )
                for ev in self.evaluators
            }
            results.append({"id": sample.id, "prediction": prediction,
                            "scores": scores, "metadata": sample.metadata})

        # Aggregate — and ALWAYS break down by slice, never only overall
        report = EvalReport()
        for ev in self.evaluators:
            vals = [r["scores"][ev.name] for r in results]
            report.add_metric(ev.name, {
                "mean": statistics.mean(vals),
                "std":  statistics.stdev(vals),
                "min":  min(vals), "max": max(vals),
                "p10":  percentile(vals, 10),   # the tail matters
            })
        for category in {r["metadata"]["category"] for r in results}:
            subset = [r for r in results if r["metadata"]["category"] == category]
            report.add_breakdown(category, self.aggregate(subset))
        return report
```

**The slice rule:** *"An aggregate score hides a broken slice. A 4.3 overall can be 4.6 on short questions and 2.8 on multi-hop ones, and only one of those numbers tells me what to fix. So every report breaks down by category, difficulty, and tenant, and the gate blocks on a slice regression even if the aggregate improved — that's the rule that catches the most real regressions."*

**Evaluator selection by task:**

| Task | Automated evaluator | Why |
|---|---|---|
| Classification / extraction | Exact match, schema validity | Objective; no judge needed |
| Summarisation | ROUGE + factual-consistency judge | Overlap is not quality — pair them |
| Code generation | **Execution against tests** | Execution is ground truth |
| RAG answer | Faithfulness + relevance + context precision/recall | Four separate failure sites |
| Agent task | Trajectory score + **verifier post-conditions** + Pass^k | Final answer hides the path |
| Open-ended chat | Judge + human sample | No ground truth exists |

### SOP-4 — Judge design and calibration

```python
JUDGE_PROMPT = """
Evaluate the response to the question.

Question: {question}
Response: {response}
Reference (if available): {reference}

Rate 1–5 on each criterion with a brief justification:
  correctness · relevance · completeness · clarity

Ignore response length when scoring.
Return JSON:
{"correctness": {"score": N, "reason": "..."}, ... , "overall": N}
"""
```

**The four judge biases and their mitigations — memorise this table:**

| Bias | Signature | Mitigation |
|---|---|---|
| **Position** | Prefers the first (or last) option | **Swap positions and re-run; keep only consistent verdicts** |
| **Length** | Prefers longer responses | Instruct explicitly to ignore length; normalise |
| **Self-preference** | Prefers its own model family's style | **Use a different model as judge than the one under test** |
| **Format** | Prefers a particular markup style | Diverse calibration examples |

```python
def calibrated_pairwise_judge(q, a, b):
    r1 = pairwise_judge(q, a, b)
    r2 = pairwise_judge(q, b, a)                    # positions swapped
    r2_adjusted = "A" if r2["winner"] == "B" else "B"
    if r1["winner"] == r2_adjusted:
        return {"winner": r1["winner"], "confidence": "high"}
    return {"winner": "tie", "confidence": "low"}   # inconsistent → tie
```

**The consistency check is the whole point:** *"Running the judge twice with the positions swapped costs two calls and converts an unreliable signal into a reliable one. A verdict that survives the swap is high confidence; one that flips is a tie, and a tie tells me the question was genuinely close rather than that one answer was better. Without the swap, position bias alone can manufacture a two-to-five point quality difference that doesn't exist."*

**Calibrating the judge against humans:**

```python
kappa = cohen_kappa_score(human_labels, judge_labels)
# < 0.20 poor · < 0.40 fair · < 0.60 moderate · < 0.80 substantial · ≥ 0.80 almost perfect
# TARGET: κ ≥ 0.60 (moderate or better)
# ALERT:  κ < 0.40 — the judge is not measuring what you think
```

**Human annotation discipline:**

```
   RULES THAT MAKE A GOLD SET WORTH HAVING
   ① A written rubric with 3–5 examples at EACH score level
   ② Two annotators on a 10–20% overlap set
   ③ Report Cohen's kappa; below 0.6 the rubric is ambiguous,
      not the annotators
   ④ Adjudication records kept — the disagreements and their
      resolutions ARE the ground truth
   ⑤ The set is versioned and NEVER tuned to make a failing
      release pass
   ⇒ A gold set edited to fix a red build is no longer a gold set.
     It's a rubber stamp with extra steps.
```

### SOP-5 — The online evaluation loop

```python
class OnlineEvaluator:
    sample_rate = 0.10          # 10% to frontier; 100% to distilled

    def maybe_evaluate(self, request, response, trace):
        # 100% — distilled, cheap, inline
        d = distilled_judge(request, response)
        record("judge_distilled", d.label, d.confidence)

        # Low confidence OR high-stakes → frontier, immediately
        if d.confidence < 0.7 or is_high_stakes(request):
            f = frontier_judge(request, response)
            record("judge_frontier", f.score)
            if abs(f.score - d.score) > 3:
                record("judge_drift", {d: d.score, f: f.score})
                enqueue_human_review(request, response, d, f)
            return

        # Otherwise a sample
        if random.random() < self.sample_rate:
            asyncio.create_task(self.evaluate_async(request, response))
```

**The drift detector:**

```python
def detect_quality_drift(current, baseline, threshold=0.10):
    cur_m, base_m = mean(current), mean(baseline)
    drift = abs(cur_m - base_m) / base_m
    stat, p = stats.ttest_ind(current, baseline)
    return {"current_mean": cur_m, "baseline_mean": base_m,
            "drift_pct": drift * 100,
            "is_significant": drift > threshold and p < 0.05,
            "p_value": p}
    # Require BOTH: a 10% move AND statistical significance.
    # A 10% move on 30 samples is noise. Effect size + significance.
```

**What to do when degradation is detected — in this order:**

```
   ① FREEZE the release pipeline (prevent a second variable)
   ② Check the release timeline — did anything ship in the last 60 min?
   ③ Slice the drop: which category, which tenant, which route?
   ④ Separate the four causes, which look identical upstream:
        · the model changed      → check served vs manifest snapshot
        · the prompt changed     → check template version in the trace
        · the index changed      → check freshness + retrieval recall
        · the judge changed      → check judge-human κ and frontier drift
   ⑤ Sample the actual responses — read 20 of them
   ⑥ Roll back if a release correlates; investigate after
   ⑦ Post-incident: add the signal that would have caught it earlier
```

**Step ④ is the whole LLMOps debugging skill:** *"Four different causes present identically — the answers got worse. The trace is what separates them, because the trace recorded which model snapshot served, which prompt template ran, which index answered, and which judge scored. Without those four attributes in the trace, all I can do is guess, and I'll usually guess wrong and spend the day on the model when the actual change was a chunker config."*

### SOP-6 — Cost operations

The discipline, in the order that produces value:

```
   ① ATTRIBUTION FIRST — tag every request with team, feature,
     tenant, model+route, release_id. Untagged spend is unmanageable.
   ② SHOWBACK → SOFT CHARGEBACK → HARD CHARGEBACK, in that order.
     Never skip straight to hard: teams that see a bill before they
     own it fix their own biggest waste.
   ③ TRACK UNIT ECONOMICS — cost per request / conversation / task /
     OUTCOME. Only the last one is a business metric.
   ④ CACHE AGGRESSIVELY — static-first prompt ordering, purge
     volatiles, cache minimums of 1,024–2,048 tokens.
   ⑤ ROUTE BY DIFFICULTY — cascade cheap → expensive, escalate on
     low confidence. 45–85% savings at ~95% quality.
   ⑥ BATCH what isn't interactive — roughly 50% off.
   ⑦ RESERVE / SPOT for predictable self-hosted load.
   ⑧ DEGRADE UNDER PRESSURE — but NEVER disable eval sampling.
     Eval sampling is 1–5% of spend and it is the control that
     tells you the degradation is working.
```

**The "bill doubled, traffic flat" ordered cause list** — because this page fires often and the answer is empirical:

| Rank | Cause | Share of cases |
|---|---|---|
| 1 | **Cache-hit collapse** | ~40% |
| 2 | Reasoning-token inflation | ~15% |
| 3 | Retry multiplier | ~12% |
| 4 | Output-length growth | ~10% |
| 5 | Context growth | ~8% |
| 6 | Agent step creep | ~7% |
| 7 | Routing regression (frontier where small would do) | ~5% |
| 8 | Mix shift (a new heavy feature) | ~3% |

*"Check them in that order, not in the order they occur to you. Cache collapse alone is about forty percent of these pages, and it's the cheapest to check — one dashboard panel — so it goes first."*

### SOP-7 — Governance as a byproduct

```
   THE PRINCIPLE: compliance is a VIEW over telemetry and artifacts
   you already produce. Do not build a parallel process.

   What you already have        → What it satisfies
   ─────────────────────────────────────────────────────────────
   Traces with release_id       → audit trails, decision logging
   Eval harness in CI           → pre-deployment eval evidence
   Red-team suites              → adversarial testing evidence
   Judge + human labels         → measurement evidence
   Cost attribution             → post-market monitoring inputs
   Release gate reports         → sign-off records
   Incident runbooks            → incident response process
   Cards stack in git           → model/system/data documentation
```

**The artifacts to produce, by audience:**

| Artifact | Contents | Lives in |
|---|---|---|
| Model card | Intended use, data summary, eval results, limitations | Repo, versioned |
| System card | End-to-end risks, mitigations, threat model, SLOs | Repo, versioned |
| Data card | Provenance, licensing, known biases, retention | Repo, versioned |
| Audit record | Who/what/when, retrieval, tool calls, proposed/taken/not-taken, `release_id` | Immutable store |
| Gate report | Per-release eval, safety, and cost results | Release record |

**Retention floors:** high-risk system logs ≥ 6 months (EU convention); financial/health decisions 5–7 years; general interaction logs 90 days–1 year; red-team and eval artifacts 3 years.

### SOP-8 — The LLMOps incident runbook

```
T+0   CLASSIFY
      □ Infrastructure (error rate up) or quality (error rate flat)?
      □ Did a release ship in the last 60 min? → ROLL BACK FIRST.
        90 seconds, and it eliminates the largest single hypothesis.
      □ Is it one tenant, one route, or global?

T+3m  ATTRIBUTE — read the trace attributes
      □ served model snapshot vs manifest      → model regression
      □ prompt template version                → prompt change
      □ index snapshot + retrieval recall      → staleness / drift
      □ judge κ and frontier drift             → the GRADER changed
      □ cache_read_tokens collapse             → cost event
      □ acl_filtered == 0 where expected       → SECURITY EVENT

T+10m MITIGATE BY CLASS
      infra     → breakers, failover, degradation ladder
      model     → repin to the last known-good snapshot
      prompt    → revert the template version
      index     → flip the alias to the previous snapshot
      judge     → recalibrate; do NOT roll back the product for it
      security  → disable the surface, preserve logs, page security

T+30m COMMUNICATE
      □ Status page if user-visible
      □ Escalate if unresolved and semantic

T+POST LEARN — the four questions
      □ Which signal caught it? If a user complaint, the SLO has a gap.
      □ Which control failed?
      □ Which control would have caught it? Add it AND its test.
      □ Was it model, prompt, index, judge, code, or config?
        "The LLM" is not a root cause.
```

---

## 6. Failure modes and mitigations

| # | Failure | Symptom | Root cause | Mitigation |
|---|---|---|---|---|
| 1 | **Silent quality degradation** | Dashboards green, users unhappy | No quality SLI | Sampled judge as a paged SLI |
| 2 | **Judge drift** | Quality score moves, system unchanged | Judge model changed under you | Frontier sample + human κ |
| 3 | **Aggregate hides a slice** | Overall fine, one cohort broken | No per-category breakdown | Gate on slice regressions |
| 4 | **Unattributable regression** | "Answers got worse, no idea why" | `release_id` not in the trace | `release_id` in every span |
| 5 | **Eval set tuned to pass** | Gate always green | Set edited to fix a red build | Own PR, second reviewer, never ships with the change |
| 6 | **Gate flake** | Retries become routine | Non-deterministic evals | Flake rate < 3% SLO; fix or quarantine the case |
| 7 | **Prompt drift in production** | Behaviour changes between deploys | Prompts edited outside the registry | Registry only; template version in the trace |
| 8 | **Floating model alias** | Quality changes with no release | Alias not pinned | Pin snapshots; alert on served≠manifest |
| 9 | **Index staleness** | Answers miss recent facts | Ingestion lag | Freshness SLI; lag alert |
| 10 | **Embedding drift** | Recall quietly falls | Model updated without a re-index | Pin embedder; re-index on any change |
| 11 | **Cost runaway** | Bill 3x, traffic flat | Cache collapse / step creep | Attribution + the ordered cause list |
| 12 | **Agent step creep** | p99 steps/task climbing | A tool failing slowly | Step budget + no-progress detector |
| 13 | **Pass^1 optimism** | Ships, then fails repeatedly | No Pass^k measurement | Measure Pass^k; require ≥ 0.60 |
| 14 | **Trajectory blind spot** | Right answer, dangerous path | Only the final answer graded | Trajectory scoring + PRM |
| 15 | **Memory error compounding** | Agent increasingly wrong over time | Extraction errors accumulate | Per-operation memory evals |
| 16 | **Distilled judge blind spot** | Novel failure unscored | Outside the training taxonomy | Frontier sample + low-confidence escalation |
| 17 | **Missing cost truth** | Cost attribution off by 50% | `cache_read_tokens` not recorded | Record cache read/write tokens |
| 18 | **Eval sampling disabled under pressure** | Cannot tell if degradation worked | Cost-cutting removed the control | Never disable eval sampling |
| 19 | **Orphaned shadow environment** | Shadow results differ subtly | Config drift between shadow and prod | Deploy from the same manifest |
| 20 | **Unversioned prompt hotfix** | No record of what changed | Edited live to fix an incident | Emergency path still writes a manifest |

---

## 7. Capacity and cost of LLMOps

### 7.1 The operating budget

| Function | Cost | Notes |
|---|---|---|
| Tracing (100% of requests) | ~$0.000002/req | Span storage dominates |
| Distilled judges (100%) | **~$0.0006/req** | Self-hosted, sub-100ms |
| Frontier judge (2% sample) | ~$0.0004/req amortised | $0.02 per judgement |
| Human review (~200/mo) | ~$3,000/month | Where ground truth comes from |
| Eval suite in CI | ~$59/release | Dominated by shadow cost |
| Guardrail stack | ~$0.0000027/req | ~490 ms, < 1% of inference |
| Observability platform | $500–5,000/month | Scale-dependent |
| On-call + operational labour | 0.5–1.5 FTE | The largest line, usually unbooked |
| **Total at 1M req/day** | **~$33,000/month + labour** | ~2–4% of inference spend |

**The comparison to hold in an interview:** *"The full layered eval stack is around $33,000 a month at a million requests a day, against $600,000 if I ran a frontier judge on every trace. And it's two to four percent of the inference spend it's protecting. If someone proposes cutting it, the number to put next to it is the cost of a week of shipping silently degraded answers, which is a number nobody ever calculates until it happens."*

### 7.2 Sampling-rate selection

| Traffic | Distilled inline | Frontier sample | Human review |
|---|---|---|---|
| < 10K req/day | 100% | 10% | 50/week |
| 10K–100K | 100% | 5% | 50/week |
| 100K–1M | 100% | 2% | 25/week |
| 1M–10M | 100% | 1% | 10/week |
| > 10M | 100% | 0.5% + all low-confidence | 10/week |

**The rule that overrides the table:** *"Every trace the distilled judge scores with low confidence goes to the frontier judge regardless of the sample rate, and every high-stakes trace does too. The sample rate governs the *baseline* calibration rate; it never governs escalation. If I had to cut the frontier budget, I'd cut the random sample to zero and keep only the escalations, because a low-confidence escalation is worth more than a hundred random samples."*

---

## 8. Metrics and alerts

```yaml
llmops_alerts:
  # ── Quality (the ones that matter) ────────────────────────
  - name: quality_score_drop
    condition: sampled_judge_score < 3.8
    window: 30m
    severity: critical
    action: page_eval_owner, freeze_pipeline

  - name: quality_slice_regression
    condition: any_slice_score_delta < -0.4 vs 7d
    window: 1h
    severity: critical
    note: aggregates hide slices — this is the one that catches it

  - name: judge_human_agreement_low
    condition: cohens_kappa < 0.40
    window: monthly
    severity: warning
    note: the GRADER drifted, not the system

  - name: judge_drift_vs_frontier
    condition: abs(distilled_score - frontier_score) > 3.0
    window: 1h
    severity: warning

  - name: faithfulness_drop
    condition: faithfulness < 0.80
    window: 1h
    severity: critical

  - name: hallucination_rate
    condition: unsupported_claim_rate > 0.06
    window: 1h
    severity: critical

  - name: refusal_rate_out_of_band
    condition: refusal_rate < 0.01 OR > 0.08
    window: 1h
    severity: warning
    note: BOTH directions are failures

  - name: regeneration_rate
    condition: regeneration_rate > 0.15
    window: 1h
    severity: warning
    note: users re-asking is the cheapest quality signal there is

  # ── Attributability ───────────────────────────────────────
  - name: release_id_missing
    condition: traces_without_release_id > 0
    severity: critical
    note: without it, no incident is attributable

  - name: model_snapshot_mismatch
    condition: served_snapshot != manifest_snapshot
    severity: critical

  - name: index_freshness_lag
    condition: index_lag > sla
    window: 1h
    severity: warning

  # ── Agents ────────────────────────────────────────────────
  - name: trajectory_success_drop
    condition: trajectory_success < 0.75
    window: 1h
    severity: critical

  - name: pass_k_regression
    condition: pass_4 < 0.45
    severity: warning

  - name: agent_step_p99
    condition: p99_steps_per_task > 20
    severity: warning

  - name: partial_completion_spike
    condition: tasks_ending_partial > 0.05
    window: 1h
    severity: warning

  - name: tool_loop_detected
    condition: same_tool_same_args > 2 in one task
    severity: warning

  - name: memory_extraction_error
    condition: memory_extraction_error_rate > 0.05
    window: 24h
    severity: warning
    note: compounds across operations

  # ── Cost ──────────────────────────────────────────────────
  - name: cost_per_request_deviation
    condition: cost_per_request > 1.5 × 7d_baseline
    window: 1h
    severity: warning
    runbook: caches → reasoning tokens → retries → output length

  - name: cache_hit_collapse
    condition: semantic_cache_hit_rate < 0.20 OR prompt_cache_hit_rate < 0.50
    window: 2h
    severity: warning
    note: cause #1 for "bill doubled, traffic flat"

  - name: eval_sampling_disabled
    condition: eval_sample_rate == 0
    severity: critical
    note: the control you must never remove
```

**Three dashboards, and who looks at them:**

| Dashboard | Audience | Panels |
|---|---|---|
| **Golden signals** | On-call | Availability · latency · errors · saturation · breakers |
| **Quality** | Eval owner, product | Judge score + slices · faithfulness · hallucination · refusal · regeneration · judge κ · drift |
| **Unit economics** | FinOps, eng lead | Cost/request · cost/outcome · cache hit rates · cost by team/feature/tenant · budget burn |

**Plus one live panel that isn't a dashboard:** the *release manifest currently serving*, with its six asset versions and the previous release one click away. *"During an incident, the first thing I want on screen is what's running and what ran before it, with a rollback button. That's not a dashboard, it's a control, and it's the single highest-value UI in the whole LLMOps stack."*

---

## 9. Change management

| Change | Owner | Reviewer | Notice | Gate |
|---|---|---|---|---|
| Prompt edit | Prompt owner | Eval owner | 1 day | Full eval + injection suite |
| Model pin change | Platform | Eval owner | 2 weeks | 72 h shadow, 2 h bakes |
| Embedding model change | ML owner | Eval + platform | 1 quarter | **Full re-index + dual-index eval** |
| Chunker config | ML owner | Eval owner | 1 month | Full rebuild + quality comparison |
| Guardrail threshold (tighten) | Security | — | None | FP rate check |
| Guardrail threshold (loosen) | Security | Security lead | 1 week | Injection suite must pass |
| Judge prompt or model | Eval owner | Eval owner + 1 | 1 week | **Re-verify κ against human labels** |
| Eval-set change | Eval owner | **Second reviewer, own PR** | 1 week | **Never ships with the change it measures** |
| Quality SLO threshold | Eval owner | Product | 2 weeks | Cost/benefit written down |
| Cost budget per route | FinOps | Eng lead | 1 week | Quality impact assessment |
| Sampling rate reduction | Eval owner | Eval owner + FinOps | 1 week | **Never to zero** |

**Three four-eyes rules worth stating:** *"Eval-set changes get their own pull request and a second reviewer, and they never ship in the same release as the thing they're measuring — otherwise a release can pass its own exam by moving the pass mark. Judge changes require re-verifying kappa against human labels, because a judge change silently redefines what every historical score meant. And loosening a guardrail requires a full injection-suite pass, because tightening only over-blocks and I'll see that in the false-positive metric, while loosening fails silently."*

---

## 10. Audit and compliance

| Record | Contents | Retention |
|---|---|---|
| Release manifests | All six asset versions, `release_id`, approver | Life of the system + 3 y |
| Gate reports | Eval, safety, cost results per release | 3 years |
| Audit records | Retrieval, prompts, outputs, tool calls, proposed/taken/not-taken | High-risk ≥ 6 mo; financial 5–7 y |
| Judge calibration records | Human labels, κ, drift measurements | 2 years |
| Gold-set versions | Every version + who changed it and why | Forever |
| Cost attribution | Per request, tagged | 1–2 years |
| Incident reports | Classification, timeline, root cause, corrective action | 3–5 years |
| Red-team artifacts | Suite version, results, findings | 3 years |
| Cards stack | Model, system, data cards per version | Life + 3 y |

**The line that ties governance to engineering:** *"Security gives you the OWASP controls and data governance, evals and CI give you the red-team and measurement evidence, observability gives you the audit logs and post-market monitoring, and release gates give you the sign-off record. Compliance is a view over telemetry and artifacts you already produce — if I'm building something twice, I've architected it wrong."*

**The GDPR point that catches people:** *"A deletion request cannot reach into model weights or into an immutable audit log. So the honest position is that you minimise what enters training or memory in the first place, you keep a mapping you can delete, and you're explicit with legal about what a deletion can and cannot reach. Discovering that limit during a request is the wrong time."*

---

## 11. Interview Q&A

**Q1. What is LLMOps and how is it different from MLOps?**
Four differences I'd lead with. First, the artifact: in classical MLOps you ship a model file; in LLMOps you ship a manifest of six independently-versioned things — code, prompt, model snapshot, index, guardrails, config — and the `release_id` that pins the combination is what makes an answer reproducible. Second, correctness: F1 and AUC are objective; LLM quality is subjective and multi-dimensional, so I need several independent measures and a judge I've calibrated against humans. Third, determinism: the same input can produce different output, so "did it regress?" is a statistical question, not a diff. Fourth, and most importantly, failure visibility: a classical system that degrades moves a metric, whereas an LLM system can degrade while everything on the dashboard stays green — the failure has a 200 status code. So the defining LLMOps capability isn't deployment, it's measurement.

**Q2. What metrics would you track in production?**
Four groups. Operational, which is table stakes: request rate, error rate, p50/p95/p99, TTFT for streaming, availability, saturation. Quality, which is what makes this different: a sampled judge score, faithfulness and relevance for RAG, hallucination rate, refusal rate, schema-valid rate, plus user signals — thumbs up/down, regeneration rate, task completion. Cost: cost per request, per conversation, per task, and per *outcome*, with token counts split by direction and, critically, cache-read versus cache-write tokens. And agent metrics for anything multi-step: trajectory success, Pass^k, steps per task, partial-completion rate, tool-error rate. The framing I'd give is that a fast, available LLM system producing bad outputs is still failing, so quality has to be a paged SLI and not a dashboard someone checks on Fridays.

**Q3. How do you detect quality degradation in production?**
Continuously, with a layered eval architecture rather than a single judge. Distilled judges run inline on 100% of traces for the failure taxonomy they cover, and they emit a confidence score, not just a label. A frontier judge runs on a one-to-five-percent sample to detect drift between the distilled judge and the larger model, and it also runs on every low-confidence or high-stakes trace regardless of the sample rate. Humans review disagreements and feed the gold set. Alongside that I watch drift statistically — a ten-percent degradation that's also significant at p under 0.05, because a ten-percent move on thirty samples is noise — plus the cheap behavioural signals that move first: regeneration rate, refusal rate in both directions, and the output-token distribution. The mistake is relying on a single judge, because a quality score can fall because the system changed or because the judge changed, and those need opposite responses.

**Q4. Why not just use an LLM judge on everything?**
Cost, mostly. A frontier judge on every trace is roughly $0.02 per judgement, which at a million requests a day is $20,000 a day — about $600,000 a month. It stopped being affordable above roughly a hundred thousand requests a day. A distilled judge is about 97% cheaper and roughly ten times lower latency, at sub-100ms for short responses, and it agrees with a frontier judge 88 to 92% of the time and lands within two or three points of human gold labels. So the shape is distilled inline, frontier for calibration on a sample, and humans for ground truth. The catch is that a distilled judge is trained on a fixed failure taxonomy — groundedness, instruction-following, toxicity, PII, off-topic, refusal — and outside that taxonomy it regresses to a default score, so it's excellent for failures you've already seen and blind to a new attack vector or a novel intent. That's exactly why the frontier sample exists.

**Q5. How do you evaluate an agent differently from a chatbot?**
You grade the trajectory, not just the answer, and you measure reliability with Pass^k rather than a single-run success rate. Final-answer grading misses three important shapes: the right answer reached by wrong reasoning, the right answer reached after four destructive tool calls, and the right answer reached with forty-seven retrieval calls instead of two. So I score each step — process reward models score steps independently, and an auditor agent, usually a different model from the one being graded, replays the trajectory asking whether each step was justified. The failure modes I watch for are reasoning-action mismatch, over-retrieval, tool flailing, premature commitment, and self-jailbreaking. And for the reliability question, Pass^1 of 70% with Pass^4 of 12% tells me the agent works on the easy path but can't recover from any perturbation — which is invisible in a single-run number and is exactly what I need to know before scaling out.

**Q6. What's the most common LLMOps mistake you see?**
Shipping a prompt change outside the versioning system. Prompts get edited to fix an incident or tune a response, they live in code or a config blob, and nobody records which version produced which answer. Then quality drops three weeks later and there's no way to attribute it, because the trace doesn't say which template ran. The fix is unglamorous — prompts in a registry with template IDs, and the template version recorded in every trace — but without it every other LLMOps capability degrades, because you can't debug what you can't attribute. The related mistake is using a floating model alias instead of a pinned snapshot, which means the provider can change your system's behaviour with no release on your side.

**Q7. How do you handle a quality regression?**
First, freeze the pipeline so I don't add a second variable. Then check whether anything shipped in the last sixty minutes and roll back if so — it's a ninety-second fix and it eliminates the largest single hypothesis. Then read the trace attributes to separate four causes that look identical from the outside: did the served model snapshot differ from the manifest, did the prompt template version change, did the index change or go stale, or did the judge itself drift. Each of those has a different fix — repin the snapshot, revert the template, flip the index alias, or recalibrate the judge — and if I can't separate them in the trace, I'll guess wrong. Then sample twenty actual responses and read them, because reading is still the highest-signal diagnostic. And post-incident, I add the signal that would have caught it earlier, because if a user complaint was the detection mechanism, my SLO has a gap.

**Q8. How would you stand up LLMOps from nothing?**
Ninety days in dependency order. First two weeks, visibility: `release_id` in every trace and log, token and cost accounting for every request with no exceptions including notebooks, latency split into TTFT and total, error rate by class, and guardrail trip counters. You cannot operate what you cannot see. Weeks three and four, the manifest: prompts out of code into a registry, model snapshots pinned, the index versioned with an alias pointer, and a tested ninety-second rollback. Weeks five through eight, evaluation: a hand-labelled golden set of a few hundred examples, five to ten quality criteria with a rubric, the eval harness wired into CI as a blocking gate, plus guardrail and injection suites — the target being that no release ships without a gate report. Days sixty to ninety, continuous operation: distilled judges inline, a frontier sample, human review on disagreements, a paged quality SLO, cost attribution per team and feature, and an incident runbook with its first drill. The sequencing matters because evals without attribution produce scores you can't act on.

**Q9. What's the cost of the eval stack?**
At a million requests a day, roughly $33,000 a month. About $18,000 for distilled judges inline at six hundredths of a cent per request, $12,000 for a frontier judge on a two-percent sample at two cents per judgement, and around $3,000 for human review of a couple of hundred disagreements. Against that, running the frontier judge on every trace would be about $600,000 a month. It's two to four percent of the inference spend it's protecting. And if the budget gets cut, the sampling rate comes down — never to zero, and never the escalations, because a low-confidence escalation is worth more than a hundred random samples.

**Q10. How do you know your judge is still valid?**
Two checks. Cohen's kappa against human labels on a sample, monthly, with a target of at least 0.6 — moderate agreement or better — and an alert below 0.4. And frontier-judge drift on the sampled traces, alerting when the distilled and frontier verdicts differ by more than about three points. The distinction that matters is what each tells me: if kappa falls, the judge no longer measures what I think it measures and I fix the judge; if kappa holds and the score fell, the system changed and I fix the system. Without both checks, a judge regression is indistinguishable from a product regression, and I'll spend the week debugging the model instead of the grader.

**Q11. How do you evaluate a system with memory?**
Per operation, not in aggregate. Memory has three stages that fail differently: extraction, where the fact written to memory must match the source; update, where a new memory must be correct relative to prior state and must not silently contradict an older one; and QA, where the answer must be grounded in stored memories rather than in parametric knowledge dressed up as recall. The insight is that a system can score very high on standard hallucination benchmarks while making catastrophic extraction errors, because the aggregate averages away the stage that's broken — and the stage is the only part I can actually fix. So every write, update, and read gets its own eval, and I'd watch the extraction error rate hardest, because a five-percent error compounds over thousands of operations into a wholly unreliable agent.

**Q12. What do you do about cost?**
Attribution first, because untagged spend is unmanageable: every request tagged with team, feature, tenant, model and route, and `release_id`, including cache-read versus cache-write tokens, because without those the cost model is wrong by thirty to seventy percent on a cached system. Then the maturity ladder — showback, then soft chargeback, then hard chargeback, never skipping a stage. Then unit economics, and only the last one is a business metric: cost per request, per conversation, per task, and per outcome. Then the levers in value order: prompt caching, which alone was about a third of the reduction in a cost model I worked through and is roughly a day of work; routing by difficulty with a cheap-first cascade at forty-five to eighty-five percent savings and about ninety-five percent of quality; batching what isn't interactive for about fifty percent off; and reserved or spot capacity for predictable self-hosted load. And when the bill doubles with traffic flat, the ordered cause list starts with cache-hit collapse, which is about forty percent of those cases.

**Q13 (pushback). Isn't this a lot of infrastructure for a product that's mostly a prompt over an API?**
The infrastructure should match the blast radius, and I'd scale it explicitly rather than applying everything everywhere. If the system is read-only, over public data, with no tools and no memory, then the operational core is: a versioned prompt, a pinned model, cost and latency tracking, a small golden set in CI, and an audit log. That's a few days of work, not a platform. What changes the calculus is any of four things — private data, side effects, a long-lived agent, or a regulated domain. Side effects mean the action gate and trajectory evaluation. Long-lived agents mean durable execution and Pass^k. Regulated domains mean the cards stack and retention. And the one thing I'd never skip regardless is the sampled quality score with an alert, because that's the only thing standing between me and shipping silently worse answers for a month.

**Q14. How do evals and observability relate?**
They're the same system viewed from two angles, and treating them as separate is the mistake. Offline evals are what I run before shipping, on a golden set, as a blocking CI gate. Online sampling is what I run after shipping, on production traffic, as an SLI with a pager. Both use the same judge, the same rubric, the same calibration against human labels, and — critically — the same trace schema, so a production failure can be turned into a new eval case and a case that fails in CI can be traced back to the production traffic that motivated it. The loop closes in both directions: production findings become eval cases, and eval failures block the release that would have caused the production problem.

**Q15. What's the single most important LLMOps capability?**
Attribution. Every other capability depends on being able to answer "what produced this output, and with which version of which asset?" — because that's the only way to act on a quality drop, a cost spike, or a security incident. Put `release_id` in every trace, pin every asset, and record the served model snapshot, the prompt template version, the index snapshot, and cache-read tokens, and suddenly everything else becomes tractable. Without it, all the dashboards in the world just tell you that something is wrong and leave you guessing which of six moving parts did it.

---

## 12. Cheat-sheet summary

| Dimension | Value |
|---|---|
| **The release unit** | **A manifest, not a model** |
| **Six versioned assets** | code · prompt · model · index · guardrails · config |
| **The four manifest rules** | Immutable on release · pointer is the deploy · `release_id` in every trace · rollback = repoint |
| **The four LLMOps inversions** | Quality is a metric · manifest not artifact · eval is continuous · the dependency is third-party |
| **Metric groups** | Operational · Quality · Cost · Agent |
| **Golden signals** | Availability · latency · errors · saturation |
| **Quality SLI** | **Sampled judge ≥ 4.2/5 · alert < 3.8** |
| **Faithfulness** | ≥ 0.85 · alert < 0.80 |
| **Hallucination** | ≤ 0.03 · alert > 0.06 |
| **Schema-valid** | ≥ 0.995 |
| **Refusal band** | 1–5% · **both directions alert** |
| **Regeneration rate** | ≤ 8% — the cheapest quality signal |
| **Judge kappa target** | **κ ≥ 0.6** · alert < 0.4 |
| **Judge drift alert** | Δ > 3 points vs frontier |
| **Layered judge stack** | **Distilled inline 100% → frontier 1–5% → humans on disagreement** |
| **Distilled judge economics** | **~97% cheaper · ~10x lower latency · 88–92% frontier agreement** |
| **Frontier-on-everything** | **~$600K/month at 1M req/day — unaffordable above ~100K req/day** |
| **The stack cost** | **~$33K/month at 1M req/day · 2–4% of inference** |
| **Distilled judge limit** | Fixed taxonomy — **regresses to a default outside it** |
| **Judge biases** | Position · length · self-preference · format |
| **Position-bias fix** | **Swap and re-run; inconsistent → tie** |
| **Self-preference fix** | Judge with a **different model family** |
| **Agent grading** | **Trajectory, not final answer** |
| **Pass^k** | **Pass^1 70% + Pass^4 12% = works on the easy path only** |
| **Pass^4 target** | ≥ 0.60 |
| **Agent grading method** | PRM per step + auditor agent on a sample |
| **Trajectory failure modes** | Reasoning-action mismatch · over-retrieval · tool flailing · premature commitment · self-jailbreaking |
| **Agent grading truth** | **Verifier post-conditions**, not LLM transcript scoring |
| **Memory evals** | **Per operation: extraction · update · QA** |
| **Extraction error compounds** | 5% per op → unreliable over thousands of ops |
| **Drift detection** | **Effect size AND significance** — 10% and p < 0.05 |
| **Four causes of a quality drop** | Model · prompt · index · **judge** |
| **Trace must carry** | `release_id` · tenant · template version · model pin · index snapshot · `cache_read_tokens` · `acl_filtered` |
| **Debugging order** | Freeze → roll back → read trace attributes → read 20 responses |
| **Cost levers, value order** | Cache → cascade routing → batch → reserved/spot |
| **"Bill doubled, traffic flat"** | **#1 = cache-hit collapse (~40%)** |
| **Cascade savings** | 45–85% at ~95% quality |
| **Batch discount** | ~50% |
| **Cost attribution tags** | team · feature · tenant · model+route · `release_id` · `step_index` · `cache_state` |
| **Chargeback stages** | showback → soft → **hard** — never skip |
| **Never disable** | **Eval sampling** |
| **Governance** | A **view** over traces, evals, gates — not a parallel process |
| **High-risk log retention** | ≥ 6 months |
| **Four-eyes rules** | Eval-set change · judge change · guardrail loosening |
| **90-day plan** | Days 1–14 visibility · 15–30 manifest · 31–60 evals · 61–90 continuous |
| **The one capability** | **Attribution** — everything else depends on it |
| **The rule that never breaks** | **If a user complaint is how you found out, your SLO has a gap.** |

---

## 13. Cross-links

- [Deployment, Release and CI/CD](04-deployment-release-and-cicd.md) — the manifest, the gates, and the 90-second rollback
- [Capacity Planning and Autoscaling](05-capacity-planning-and-autoscaling.md) — the capacity half of operations
- [Cost and FinOps](06-cost-and-finops.md) — attribution, unit economics, and the cost ladder
- [Reliability Patterns](07-reliability-patterns.md) — semantic validation and the quality SLO
- [Security, Guardrails and Compliance](08-security-guardrails-and-compliance.md) — governance as a byproduct
- [DR, Failover and Backup](09-dr-failover-and-backup.md) — protecting the eval sets this file depends on
- [RAG Evaluation and Observability](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) — the RAG metric deep dive
- [Agentic System Design](../HLD/05-ai-systems/06-agentic-system-design.md) — trajectory design and step budgets
- [Memory and Context Engineering](../HLD/05-ai-systems/08-memory-and-context-engineering.md) — memory operations and their evals
- [Production RAG System](../HLD/05-ai-systems/01-production-rag-system.md) — the system this discipline operates
- [Operationalisation Cheatsheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
