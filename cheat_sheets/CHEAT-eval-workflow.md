# Cheat Sheet · The Evaluation Workflow, End to End

> **Covers:** CS-03 (why multiple eval pipelines), CS-04 (the complete workflow), CS-06 (offline vs
> online) · **Code ground truth:** `CODE-03` (evals-skills — the process as agent skills), `CODE-01`
> (`PATTERNS.md` — the patterns), `CODE-06` (Langfuse — where it all lives) · **Use when:** you are
> starting eval work from zero, or your existing eval suite is not changing any decisions.

---

## The 60-second version

The order of operations is the whole content of this cheat sheet, and it is the thing teams get
wrong. **You do not start by writing evaluators.** You start by looking at real traces.

```
1. ERROR ANALYSIS      sample 30-50 real traces -> open-code -> axial-code -> rank by freq x severity
2. DATASET             build the golden set from what you found (LLM-draft + human review)
3. EVALUATOR           code check if checkable in code; else ONE binary judge for ONE cluster
4. VALIDATE            ~100 human labels, held-out split, TPR + TNR separately, both >= ~0.9
5. BASELINE            first full run becomes the baseline; every run after is a diff
6. CI GATE             re-run on every push; block if a metric drops past the per-metric threshold
7. ONLINE              same evaluators on sampled production traces + tracing + drift alerts
8. HARVEST             production failures -> back into the golden set -> go to step 1  <- THE LOOP
```

Everything else — which framework, which dashboard, which judge model — is a detail that can change
without breaking the method. The order cannot.

---

## Core concepts

| Stage | What it produces | The failure if you skip it |
|---|---|---|
| **Error analysis** | A ranked list of *named, observed* failure modes | You build evaluators for failures you imagined |
| **Dataset** | A frozen golden set with reference answers | You cannot measure anything twice |
| **Evaluator** | A code check or a validated judge | You have an opinion, not a measurement |
| **Validation** | TPR/TNR on held-out labels | You have a number with unknown error bars |
| **Baseline** | The comparison point | "Better" and "worse" are undefined |
| **CI gate** | An automatic deploy decision | Regressions ship because nobody compared |
| **Online** | Production quality, cost, latency, drift | Quality degrades silently for weeks |
| **Harvest** | A golden set that grows | The same failures recur forever |

**The two axes that organise everything.** *Offline vs online* — offline is a frozen dataset, fast
and repeatable and pre-deploy; online is sampled production traffic, realistic and post-deploy. They
must use **the same evaluators**, or the offline number does not predict the online one. *Model vs
application* — model evals choose the base model (benchmarks + custom evals on your data); application
evals decide whether to ship. Different decisions, different owners, different cadences.

**Why multiple pipelines rather than one.** Different applications fail differently: a fixed-schema
classifier is evaluated by exact label match; a RAG chatbot needs retrieval metrics plus the triad; an
agent needs trajectory and world-state grading. One "quality score" spanning all of them cannot
localise a failure, so it cannot tell you what to fix.

---

## Formulas & metrics

**Sampling for error analysis**

```
stratified sample = cluster representatives  +  random picks
```
Pure random under-covers rare failure modes. Pure cluster sampling over-covers prototypes. You need
both, because the rare-but-severe failure is exactly the one worth a judge.

**Sample size intuition**

```
noise floor  = spread of the metric across two runs of IDENTICAL settings
usable gate  = threshold > noise floor
```
If you cannot measure the noise floor, you cannot set a threshold. This is the single most useful
piece of arithmetic in eval work.

**Judge validation** — see `CHEAT-judge-design.md` for the full treatment:
```
TPR = TP/(TP+FN)   TNR = TN/(TN+FP)   ship when BOTH >= ~0.9 on held-out data
```

**Pass rate**
```
pass@k = 1 - C(n-c, k)/C(n, k)     [capability: one success is enough]
pass^k = (per-trial success)^k     [reliability: every trial must succeed]
p = 0.75  ->  pass@10 ~ 1.0   vs   pass^10 ~ 0.056
```

**Cost**
```
run_cost = rollout_cost + judge_cost          # both belong in the number
cost_per_query = (in_tok/1e6 * in_rate) + (out_tok/1e6 * out_rate)
```
Output tokens typically cost ~4× input. Instrument the **eval suite's own** spend — grading is part
of the run's economics, not overhead.

---

## Decision rules

1. **Look at your data before you write a single evaluator.** Non-negotiable.
2. **Build the review interface.** You must *see* variance to cluster it — colour for categories,
   spacing for hierarchy, opacity for importance. A JSON dump does not produce insight.
3. **Humans produce observations, machines produce structure.** The annotator leaves free-text notes;
   the tooling sorts notes into failure modes, tracks coverage, and proposes new samples.
4. **Code check before judge.** If the property is a schema, a regex, a tool argument or a database
   state, a judge is strictly worse: slower, costlier, noisier, and not unit-testable.
5. **One judge, one failure mode, binary labels.** Compose judges; do not multiplex one.
6. **Validate before you trust.** Held-out split, TPR and TNR separately.
7. **Capture the baseline on the first full run**, store the configuration with it.
8. **Gate on the delta, never the absolute.** Absolute thresholds reward gaming and punish
   legitimate migrations.
9. **Tier the suite by cost.** Ops evals and code checks are free → every commit. Judge subsets →
   every commit. Full golden set → nightly and pre-release.
10. **Use deterministic sampling online.** A retried run must score the *same* traces, or runs are
    incomparable and you pay twice.
11. **Re-validate the judge on any distribution change** — model, prompt, traffic mix, corpus.
12. **Close the loop.** Production failures become golden-set rows, or the suite goes stale.

---

## Thresholds & defaults worth memorising

| Item | Value |
|---|---|
| Traces to sample for error analysis | **30–50**, stratified (cluster reps + random) |
| Human labels for judge validation | **≥ 100** |
| Judge ship bar | **TPR ≥ 0.9 and TNR ≥ 0.9** |
| CI regression gate | **> 3 units** below baseline blocks the deploy (set per metric, above the noise floor) |
| Few-shot critiques | **4–8** |
| Golden set size | 15 rows detects large moves only; **1–2 point deltas are noise** |
| Reliability samples | 25–50 too few → push toward **1,000** |
| Drift window | **24 h** graph, alert on degradation over the **last 8 h** |
| Baseline storage | Every run must record: model, chunk size, overlap, k, judge model, dataset version |

---

## Tool commands

```bash
# --- the full loop, minimal form ---
# 1. error analysis (agent-skill form)
npx skills add <repo> --skill error-discovery
#   -> "Can you help me do error analysis on traces.jsonl?"

# 2. validate the judge you wrote
npx skills add <repo> --skill validate-evaluator
#   -> "Validate this judge against labels.csv using TPR/TNR."

# --- the suite, from repo root ---
python3 -m evals.eval_retriever      # component level
python3 -m evals.eval_latency        # ops — free, no judge, no golden data
python run_evals.py                  # everything
python compare_to_baseline.py        # exit non-zero if any metric dropped past threshold

# --- force a clean re-embed ---
rm -rf chroma_store/                 # else the loader silently reuses the stale index
```

```python
# tracing — everything hangs off the trace
with langfuse.start_as_current_span(name="rag-answer") as span:
    docs = retriever.search(question)
    answer = llm(question, docs)
    span.update(input=question, output=answer,
                metadata={"k": 5, "retrieved": [d.id for d in docs]})
langfuse.score_current_trace(name="groundedness", value=1.0, comment="...")
```

**Tool roles, one line each:**

| Layer | Job | Examples |
|---|---|---|
| Process | the order of operations | `CODE-03` skills, `CODE-01` PATTERNS |
| Runner | execute evals, resolve datasets | OpenAI evals, Evalscope, DeepEval, promptfoo |
| Judge | score subjective criteria | validated binary judges, G-Eval |
| Benchmarks | published capability numbers | MMLU, GSM8K, SWE-bench, 205 adapters in Evalscope |
| Execution grading | run the artefact, then grade | `frontier-evals` (fresh container per stage) |
| Platform | traces + scores + datasets + experiments | Langfuse |

---

## Top 10 mistakes

1. **Writing evaluators before doing error analysis.** The single most expensive shortcut.
2. **One aggregate "quality score"** that cannot localise a failure.
3. **Spending a judge on something a code check would catch** — schema, citation presence, regex.
4. **Skipping validation**, or reporting accuracy instead of TPR/TNR.
5. **No baseline and no stored configuration**, so no run is comparable to any other.
6. **Threshold below the noise floor**, so the gate flaps and everyone learns to ignore it.
7. **Evaluating everything on every commit**, then abandoning CI because it is too slow and too
   expensive.
8. **Non-deterministic online sampling**, so retries score different traces and runs cannot be
   compared.
9. **Stopping at deployment** — no tracing, no drift alert, no online evaluators.
10. **A golden set that never grows**, so last month's production failures are still invisible
    offline.

---

## If you only remember three things

1. **Error analysis → dataset → evaluator → validate → baseline → CI gate → online → harvest.** The
   order is the method.
2. **Code check before judge; binary before graded; validated before trusted; TPR and TNR, never
   accuracy.**
3. **Close the loop** — production failures must flow back into the golden set, or your eval suite
   measures last quarter's system.
