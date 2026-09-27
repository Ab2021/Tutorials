# Cheat Sheet · Benchmarks & Leaderboards

> **Covers:** CS-09 (how to read a leaderboard), CS-10 (the evolution of knowledge benchmarks),
> CS-11 (saturation vs contamination), CS-12 (selecting the right LLM — custom model evals) ·
> **Code ground truth:** `CODE-02` (openai/evals — the registry-driven runner: an eval is YAML plus
> optional code), `CODE-04` (EvalScope — 205 benchmark adapters behind one CLI, plus arena and
> stress-testing modes) · **Use when:** someone shows you a score and expects you to act on it.

---

## The 60-second version

A **benchmark** is "a standardised test used to measure a particular model capability." A
**leaderboard** is where those results get published so models can be compared "on a common set of
evaluations." The benchmark is the exam; the leaderboard is the notice board.

**A benchmark has exactly four components**, and a number is meaningless unless all four are known:

| # | Component | What it covers |
|---|---|---|
| 1 | **Dataset + task** | questions **and** answers, plus what the model must do — "like a golden dataset" |
| 2 | **Run configuration** | prompt construction (zero-shot vs few-shot, CoT on/off), decoding (temperature ≈ 0, max tokens), scoring strategy (pass@1 / pass@k / majority@k), tool access |
| 3 | **Scoring method** | **extraction** of the answer from free-form output, then comparison to ground truth |
| 4 | **Aggregation method** | how per-item scores become one headline number — a mean, or a **weighted** mean |

All four are documented in the benchmark's **research paper**. Everything a benchmark score can do to
mislead you is one of those four being different, hidden, or degrading.

**Four separate reasons a number is untrustworthy**, and they are independent:
**contamination** (the dataset entered pretraining, so you cannot tell thinking from memorising),
**saturation** (everyone clusters at 92–94% and the benchmark stops discriminating),
**configuration gaming** (the lab's own model gets the favourable settings, the rival gets the defaults
— a **5–10%** swing), and **aggregation hiding** (a 57-subject average concealing a weak subject).

**Seven knowledge benchmarks, one lifecycle.** Each was built to fix the previous one's failure, and
every one of them saturates: **MMLU** (breadth) → **TruthfulQA** (reliability) → **AGIEval** (borrow
real human exams) → **GPQA** (depth) → **MMLU-Pro** (repair MMLU) → **SimpleQA** (calibration) →
**HLE** (breadth × depth, private test set).

**And the rule that governs all of it:** leaderboards are a **filtering tool, not a decision tool.**
Shortlist 3–5 models, then run **your own** evaluation. Never accept the number thrown in your face.

---

## Custom model evals — the three stages

| Stage | What you do | Output |
|---|---|---|
| **1. Requirements** | Write your own constraints: capability, volume, latency ceiling, **budget** | A filter |
| **2. Leaderboard shortlist** | Apply the requirements to the boards | **5–10** candidates |
| **3. Custom model eval** | Run those on **your own data**, graded **your own way** | The decision |

**The cost filter, worked (CS-12, Text-to-SQL — note: *not* RAG, the schema is in the prompt):**

```
rate        = $10 / 1M input , $50 / 1M output
per query   = (400/1e6 × $10) + (100/1e6 × $50) = $0.004 + $0.005 = $0.009
per month   = $0.009 × 50,000/day × 30            = $13,500  ≈ ₹12.82 lakh
budget      = ₹3 lakh/month  ->  ~4x over  ->  need ~1/4 the price
```

**146 models** → compute monthly cost for each → drop everything over budget (**50–60 removed**)
→ min–max normalise rating and speed → `score = 9 × rating' + 1 × latency'` (**90:10**)
→ top 10 → **5** candidates to the custom eval.

**Prompt caching is the first lever:** cache hit on the static schema prefix costs **$1 instead of
$10**; TTL **5 minutes**. Order of operations: **cost model → shortlist → caching → custom eval** —
caching changes which models clear the filter.

**Grade by executing, not by comparing strings.** For any executable output (SQL, code, JSON, a tool
call):

1. **Run** the model's output and the golden output.
2. **Compare result tables**, never the text.
3. Normalise: row counts, int-vs-decimal, decimal precision.
4. **Sort rows** — unless ordering is semantically meaningful.
5. `accuracy = correct / total`.

**What the custom eval found:** ~**90% / 85% / 80% / 65% / 55%**, with the **week's headline model
last**. The top two were separated on **API reliability**, not accuracy, and the **PM made the call**.
Public rank did not predict task accuracy.

> ⚠ **CS-12's figures are illustrative, not a study.** Its own numbers disagree — volume quoted as
> both **50,000** and **5,000**/day, cache-write as both **$50** and **$12.50**, three unreconciled
> budgets (₹30 L / ₹3 L / ₹1 L). Use the **method**; re-derive the numbers.

---

## Core concepts

| Term | Meaning |
|---|---|
| **Benchmark** | a standardised test measuring one model capability |
| **Leaderboard** | a public ranking table over a common set of evaluations |
| **Saturation** | all frontier models cluster at one high score; the benchmark no longer discriminates |
| **Contamination** | the dataset is public **and** the model's pretraining scraped it |
| **Canary string** | a marker injected into a dataset so its appearance in an answer reveals it was trained on |
| **Static vs dynamic benchmark** | dataset frozen since the paper vs refreshed against a recent window |
| **Goodhart's Law** | when a measure becomes a target, it becomes less useful as a measure |
| **Rank bias** | over-weighting rank position when the underlying scores are 0.2 apart |
| **Configuration gaming** | favourable run settings for your own model, defaults for the rival |
| **Aggregation hiding** | the headline average that conceals a weak subject |
| **Knowledge capability** | the model's **parametric** world knowledge — what it retained from training |
| **Breadth vs depth** | many subjects at a basic level (MMLU) vs few subjects at research level (GPQA) |
| **Google-proof** | solvable only by a domain specialist, not by a non-specialist with search and 30 minutes |
| **Calibration** | whether the model knows that it does not know |
| **Eval harness** | "a piece of code that you write in order to execute model evaluation" |
| **Composite score** | one number joining several benchmarks — useful, but its weights are usually hidden |

**The four leaderboard types, ranked by usefulness:** benchmark-specific (least useful — one exam),
**multi-benchmark composite** (most useful; also carries cost, latency, speed, context), human-preference
(LMArena — biased toward long, confident, well-formatted answers), and application-specific
(**BFCL** for tool calling, **MTEB** for RAG embeddings).

**Why the leaderboard's information does not transfer** — the Kaggle analogy. Kaggle data is clean and
the problem statement is clear, which is why Kaggle success does not predict job success. Benchmarks
are clean data with clear problem statements too. Production has ambiguous requests, missing
information, company-specific data, tool failures and edge cases — and whether the model handles *that*
is exactly what the leaderboard did not measure.

---

## Formulas & metrics

```
accuracy                  = correct / total
micro accuracy            = the same ratio computed PER SUBJECT
MC1                       = argmax over the log-probabilities assigned to each option
MC2                       = normalized probability mass summed over all true answers
correct-given-attempted   = correct / attempted          # attempted excludes abstentions
F-score                   = 2*(correct * cga) / (correct + cga)   # harmonic mean
calibration (HLE)         = RMS(self-reported confidence, actual correctness)
```

**Aggregation, exactly**

```
simple    score = correct_items / total_items            # 920/1000 = 92%
weighted  score = SUM(n_s * score_s) / SUM(n_s)          # over subjects s
          NEVER SUM(score_s) / 57                        # MMLU's 57 subjects are unequal
```

Scoring strategies are **not** interchangeable and must be disclosed: **pass@1** (one showing),
**pass@k** (correct if *any* of k attempts is right — "a more lenient strategy"), **majority@k**
(the mode of k answers). Comparing a pass@k number against a pass@1 number is not a comparison.

---

## Decision rules

1. **Never compare two models evaluated under different settings.** The run configuration must be
   identical across everything being compared.
2. **Always ask which scoring strategy was used** — pass@1, pass@k or majority@k.
3. **Always ask who ran the evaluation.** Discount a self-reported lab number; it behaves like a car
   brochure's mileage claim.
4. **If the benchmark is public and old, assume contamination risk.**
5. **If everyone scores 92–94%, the benchmark is saturated** — stop quoting it.
6. **For multi-subject benchmarks, demand per-subject scores**, never only the average.
7. **Before treating a plateau as a capability ceiling, check the invalid-question rate** —
   ~6.5–8% of MMLU's questions were never answerable, capping every model near 92.
8. **When a benchmark is graded by an LLM judge, do not compare results across years.**
9. **When a claim says a model beat humans on exam X, ask what else exam X tests — and does not.**
10. **On a calibration benchmark, do not score abstention as failure** — knowing you do not know is
    the thing being measured.
11. **Smoke-test with `--limit` before paying for a full run.** 20 items cost ₹3–4; 8,000 cost ₹2,300.
12. **Use an eval harness, not hand-rolled plumbing** (`lm-evaluation-harness`, Inspect, HELM), so
    results are standardised and comparable.
13. **Write down your own constraints first** — latency need, cost ceiling, context need, public vs
    on-premise — *before* opening a leaderboard.
14. **Read the definitions before the number**: what is scored, how, by whom, inference budget,
    reasoning on/off, dataset age, update cadence, private test set, confidence intervals, and
    **composite weights**.
15. **Scroll past the top three.** Ranks 10, 12, 15 and 20 are generally cheaper and often sufficient.
16. **Apply the budget filter before the eval, not after.** Convert every candidate's per-token rate
    to a **monthly cost on your own volume** and cut everything over budget first — 146 models can
    become 5 before a single evaluation runs.
17. **Check caching before you drop a model.** A cache hit at **$1 instead of $10** on a static prefix
    can move a candidate from 4× over budget to inside it.
18. **Grade by executing, not by comparing strings.** Run the output; compare results, not text;
    normalise types and precision; sort unless order is semantic.
19. **Write the weights down.** `9 × rating' + 1 × latency'` is a **decision**; an unstated composite
    weighting is an opinion you have not admitted to holding.

**The five-step reading procedure (copy-pasteable)**

1. Write down application type, latency need, cost ceiling, context need, deployment constraint.
2. Choose the board that matches your work — agent board / LMArena-type / **MTEB** for RAG /
   Artificial Analysis for budget.
3. Read the definitions: scoring, scorer, budget, reasoning mode, dataset age, cadence, private test
   set, saturation, confidence intervals, composite weights.
4. Shortlist the **top 3–5** models against your criteria.
5. Run **your own** evaluation on all of them, and pick from that.

---

## Thresholds & defaults worth memorising

| Item | Value |
|---|---|
| Decoding for comparability | **temperature ≈ 0** |
| Saturation cluster (illustrative) | **95 / 94 / 92** — or "top 10 between 92 and 94" |
| Saturation progression | 25% → 36% → 50% → 70% → **90–95%** |
| Meaningless rank gap | **0.2** points (84.3 at rank 3 vs 84.1 at rank 5) |
| Shortlist before your own eval | **3–5** models |
| Configuration-gaming swing | **5–10%** |
| MMLU | **14,000** questions · **57** subjects · **4** options · **6.5–8%** invalid |
| MMLU score path | **84%** by generation vs **87%** by log-probability — same model |
| TruthfulQA | **817** questions · **38** categories · GPT-3 **58%** vs human **94%** |
| AGIEval | **20** exams · **8,000+** questions · human baseline **67%** avg / **91%** top |
| GPQA | **546 / 443 / 198** (extended / main / diamond) · GPT-4 **39%** → o1 **78%** |
| MMLU-Pro | **10** options · **14** categories · **12,000** questions |
| SimpleQA | **4,326** questions · same model **88%** on MMLU vs **40%** here |
| HLE | **2,500** questions · **100+** subjects · **1,000** experts · private test set |
| CS-12 cost anchor | **$0.009**/query @ 400-in/100-out, $10/$50 per M → **₹12.82 L/month** @ 50k/day |
| Prompt caching | cache hit **$1** vs **$10** input rate · **5-minute** TTL |
| CS-12 shortlist funnel | **146** models → **~50–60** cut by budget → top **10** → **5** evaluated |
| GSM8K | ~**8,500** questions · full run ₹**2,300** · 20-item smoke ₹**3–4** |

---

## Top 10 mistakes

1. **Using a leaderboard as a decision tool** instead of a filter.
2. **Quoting a number without its configuration** — zero-shot or few-shot, CoT or not, which
   pass@k, temperature, max tokens, tools.
3. **Comparing models run under different settings**, or comparing judge-graded results across years.
4. **Treating an invalid-question ceiling as a model capability ceiling.**
5. **Reading "beat humans on exam X" as "surpassed human intelligence."**
6. **Averaging subject percentages without weighting by question count.**
7. **Accepting a frontier lab's self-reported number**, or believing a rank difference with no
   confidence interval.
8. **Quoting a saturated benchmark** because it is familiar.
9. **Assuming a knowledge benchmark predicts agentic or open-ended performance** — and assuming
   pretraining filters protect against alignment-stage contamination.
10. **Treating a composite score as transparent when its weights are hidden.**

---

## If you only remember three things

1. **A benchmark has four components** — dataset + task, run configuration, scoring, aggregation —
   and a number means nothing unless all four are known and identical across the models compared.
   The same model scores **84% or 87% on MMLU** depending on the scoring path alone.
2. **Leaderboards are a filtering tool, not a decision tool.** Shortlist 3–5, then run your own
   evaluation — because benchmark performance does not transfer, and contamination plus Goodhart's
   Law can inflate a score without improving the model.
3. **Two forces kill a benchmark** (contamination, saturation) and **two forces distort a reported
   number** (configuration gaming, aggregation hiding) — so read the **stars, not the number**:
   what is scored, how, by whom, on what data, at what age, with what confidence interval, and with
   what weights.
