# Cheat Sheet · Metric Formulas & Statistics

> **Covers:** CS-07 (judges), CS-08 (G-Eval), CS-10 (classical metrics), CS-13 (retriever metrics),
> CS-15 (operational metrics), CS-17 (trajectory/agentic), CS-18 (pass@k / pass^k) ·
> **Use when:** you need the exact definition, the failure mode, and the "never use it for" of any
> eval metric.

---

## The 60-second version

Every metric here answers a specific question and has a specific way of lying to you. The skill is
not knowing the formula — it is knowing the **shape of the answer**: is it rank-aware, is it
reference-based, does it need a judge, and what does it hide?

Three rules cover most of the ground:
1. **Rank-aware beats set-based** whenever a consumer reads the list in order (retrieval; context
   stuffing).
2. **Recall-type metrics hide false alarms; precision-type metrics hide misses.** Report both, or
   report TPR/TNR where the "positive" class is the failure.
3. **Any metric on an imbalanced set lies when reported alone.** Prevalence changes what a number
   means.

---

## Core concepts

| Family | Needs a reference? | Needs a judge? | Answers |
|---|---|---|---|
| **Retrieval ranking** | Yes (relevance labels) | No | Did we find the right documents, and early? |
| **Retrieval semantic** | Yes (ideal answer) | **Yes** | Did we find the right *content*, regardless of chunking? |
| **Classification** | Yes (gold labels) | No | Did we call the label right? |
| **Text overlap** | Yes (reference text) | No | How similar is the surface form? |
| **Semantic similarity** | Yes (reference text) | No (embeddings) | How similar is the meaning? |
| **LLM-judged** | Sometimes | **Yes** | Is this good, grounded, relevant, safe? |
| **Trajectory** | Sometimes | Sometimes | Was the *path* correct, not just the outcome? |
| **Pass-rate** | Yes (verifier) | No | Can it do it at all / every time? |
| **Operational** | **No** | **No** | Is it fast, cheap, reliable? |

The last row matters: **operational metrics need neither a golden dataset nor a judge**, which is why
they are free to run on every commit.

---

## Formulas

### Retrieval — ranking

```
Precision@k  = |relevant ∩ top_k| / k
Recall@k     = |relevant ∩ top_k| / |all relevant|          ("what fraction of gold did we surface")
Hit Rate@k   = 1 if any relevant doc appears in top_k, else 0;  averaged over queries
MRR          = mean over queries of  1 / rank_of_first_relevant
MAP          = mean over queries of  AP,  where
               AP = (1/R) * Σ_{k=1..n} Precision@k * rel(k),  R = total relevant for that query
```

**DCG / NDCG** — the only one that handles *graded* relevance (a document can be partly relevant):

```
DCG@k  = Σ_{i=1..k}  rel_i / log2(i + 1)
IDCG@k = DCG@k of the IDEAL ordering (relevant docs sorted best-first)
NDCG@k = DCG@k / IDCG@k          -> in [0, 1], comparable across queries
```

### Retrieval — semantic (re-chunking-proof)

```
Contextual Recall    = claims_found_in_retrieved / total_claims_in_ideal_answer
Contextual Precision = mean over ranks of prefix precision
                       A: 1/1, 2/2, 2/3, 2/4, 2/5  -> 0.713   (✓✓✗✗✗)
                       B: 0/1, 0/2, 0/3, 1/4, 2/5  -> 0.130   (✗✗✗✓✓)
                       both have plain precision 2/5 = 0.4
```

**Why the semantic pair wins:** a gold **chunk ID** label dies the moment you change chunk size — the
same passage gets a new ID and every row is invalid. An **ideal answer** survives re-chunking. The
cost is that you now need a judge, and the ideal answer must be written as **atomic claims by design**
so the judge's decomposition is stable.

### Classification

```
Accuracy  = (TP + TN) / N                       <- lies on imbalanced data
Precision = TP / (TP + FP)     "of my positives, how many were right"
Recall    = TP / (TP + FN)     "of the real positives, how many did I find"  = TPR
F1        = 2 * (P * R) / (P + R)
TNR       = TN / (TN + FP)     "of the real negatives, how many did I clear"
FPR       = 1 - TNR
Cohen's κ = (p_o - p_e) / (1 - p_e)             <- agreement corrected for chance
```

**For judges, set "positive" = the judge says FAIL.** Then `Recall = TPR` is *the share of real
failures the judge catches* — the number that matters. `Accuracy` will look excellent while TPR is 0.

### Correlation (judge ↔ human)

```
Pearson r    linear agreement        sensitive to scale/offset, and to outliers
Spearman ρ   rank agreement          use when only the ORDER matters
Kendall τ    rank agreement, small-n use
```
Report **both** a correlation and TPR/TNR. A judge can correlate well on average and still miss the
rare failure class entirely.

### Text overlap (classical)

```
Exact Match (EM) = 1 if normalize(pred) == normalize(gold), else 0
Token F1         = harmonic mean of token-level precision & recall   (SQuAD style)
BLEU             = clipped n-gram PRECISION × brevity penalty        (translation)
ROUGE-N          = n-gram RECALL vs reference                        (summarisation)
ROUGE-L          = longest-common-subsequence based
METEOR           = unigram alignment + stemming/synonyms, F-mean
```
**Never use these as a quality metric for open-ended generation.** They measure surface form. A
correct answer phrased differently scores zero on EM and poorly on BLEU.

### Semantic similarity

```
BERTScore = greedy token matching on contextual embeddings, then F1
```
Better than BLEU/ROUGE for paraphrase, still reference-bound, and it will happily rate a fluent
hallucination as similar if it shares vocabulary.

### LLM-judged

```
G-Eval      : generate evaluation steps from a criterion -> score with token probabilities,
              weighted by each step's importance          [graded; harder to validate]
Binary judge: PASS/FAIL + CoT reasoning first              [validatable with TPR/TNR]
Pairwise    : A vs B, pick the winner -> Win Rate = wins / comparisons
Elo         : iterative rating from pairwise outcomes       [needs many comparisons]
```

### Trajectory / agentic

```
Tool-selection accuracy  = correct tool chosen / tool calls made
Argument accuracy        = tool calls with valid+correct args / tool calls
Step efficiency          = optimal_steps / actual_steps        (or excess steps)
Recovery rate            = failures followed by a successful retry / total failures
Outcome / world-state    = assertion on the DB/filesystem/env AFTER the run  <- the strongest signal
```

### Pass-rate

```
pass@k = 1 - C(n-c, k) / C(n, k)      n samples drawn, c correct; compute as a running product
pass^k = estimated (per-trial success)^k
p = 0.75  ->  pass@10 ≈ 1.0   vs   pass^10 ≈ 0.056
```
Naive `1 - (1-p)^k` is **biased high for small n** — never report it.

### Operational

```
cost_per_query = (in_tok/1e6 * in_rate) + (out_tok/1e6 * out_rate)     out_rate ≈ 4 × in_rate
end_to_end     = retrieval + generation                                (~0.7 s + ~2.9 s observed)
report latency = mean, median, P50, P95, P99, min, max — never mean alone
TTFT           = time to FIRST token (requires streaming to observe)
TPOT           = time per output token
reliability    = success rate | error rate (1 - success) | timeout rate | retry rate
```

**The latency trap:** timed-out and failed requests are **excluded** from the percentile. A P95 that
improves from 3.0 s → 2.0 s while the timeout rate rises 2% → 8% means you are *dropping* more
requests. **Always publish the timeout rate on the same line as the percentile.**

**The small-sample trap:** with n = 25, the P99 *is* essentially the maximum. If P99 == P95, the
sample is too small for a tail statistic.

### Statistics you will actually need

```
Standard error of a proportion = sqrt(p(1-p)/n)
   p = 0.9, n = 15  ->  SE ≈ 0.077   (a 1-2 point move is noise)
   p = 0.9, n = 100 ->  SE ≈ 0.030
Bootstrap CI : resample the per-item scores with replacement ~1000×, take the 2.5/97.5 percentiles
Noise floor  : run IDENTICAL settings twice; the spread IS your floor. A threshold below it flaps.
Prevalence correction: TPR/TNR are prevalence-independent; PPV/NPV are NOT.
```

---

## Decision rules

1. **Rank-aware if anything reads the list in order.** Use contextual precision / NDCG, never plain
   precision over a set.
2. **Recall and precision trade off through k.** Fix chunking, embedding and reranking first; tune k
   last.
3. **For a judge, make FAIL the positive class** and report TPR/TNR. Never accuracy.
4. **Prefer the reference that survives parameter changes** — ideal answer over chunk ID.
5. **Never use BLEU/ROUGE/EM for open-ended generation**; use a validated judge or a semantic metric.
6. **Measure cost as a distribution per query**, not a monthly total.
7. **Report percentile + timeout rate + sample size together**, or the latency number is
   uninterpretable.
8. **Grade the world state where the task is executable** — an assertion on the database after the
   run beats any transcript metric.
9. **Use pass@k for capability, pass^k for reliability.** They tell opposite stories about the same
   model.
10. **Compute the noise floor before setting any threshold.**
11. **Exclude cold start** — discard the first runs *in total*, not per question.
12. **Recompute any cost measured with repeated identical questions**: provider prefix caching
    inflates the hit rate and understates production cost.

---

## Thresholds & defaults worth memorising

| Item | Value |
|---|---|
| Judge ship bar | TPR ≥ 0.9 **and** TNR ≥ 0.9 |
| Metric pass threshold | 0.7 (DeepEval default in the source's suite) |
| CI regression gate | block on a drop > 3 units vs baseline, set above the noise floor |
| Reliability samples | 25–50 too few → toward **1,000** |
| Latency repeats | 5× per question minimum; discard first 2 runs as cold start |
| Golden set to detect small moves | 15 rows detects only large moves; **1–2 points = noise** |
| Latency budget | P95 ≤ 3,000 ms; TTFT ≤ 1,200 ms |
| p = 0.75 | pass@10 ≈ 1.0 vs pass^10 ≈ 0.056 |
| Output token rate | ≈ 4 × input rate |

---

## Tool commands

```bash
python3 -m evals.eval_retriever     # recall / precision / contextual pair
python3 -m evals.eval_latency       # P50/P95/P99, TTFT — no judge, no golden data
python3 -m evals.eval_cost          # per-query cost, input vs output split
python3 -m evals.eval_reliability   # success / error / timeout / retry by cause
```

```python
# DeepEval — metric with a threshold and forced reasoning
metric(threshold=0.7, model=judge, include_reason=True)
#   score < threshold -> FAIL ; score >= threshold -> PASS
#   ALWAYS read the reason on failures; the aggregate cannot tell you what to fix
```

---

## Top 10 mistakes

1. **Reporting accuracy for a judge** — hides a TPR of 0 behind a 90% number.
2. **Plain precision instead of rank-aware contextual precision.**
3. **BLEU/ROUGE/EM for open-ended answers** — surface metrics on a semantic task.
4. **Quoting a mean latency** with no percentiles, no timeout rate, and no sample size.
5. **Quoting a tail percentile from n = 25.**
6. **Labeling the golden set with chunk IDs**, then changing chunk size.
7. **Tuning k to fix recall** while precision (and the actual answer quality) degrades.
8. **Naive `1-(1-p)^k`** instead of the unbiased combinatorial estimator.
9. **Cost measured with repeated identical questions**, inflated by prefix caching.
10. **Setting a threshold with no noise floor**, so the gate flaps and gets ignored.

---

## If you only remember three things

1. **Rank-aware for retrieval; TPR/TNR for judges; world-state for executables.** Match the metric
   family to the question.
2. **Recall hides false alarms, precision hides misses** — and prevalence changes what any single
   number means.
3. **Percentiles without timeout rates and sample sizes are not measurements**, and BLEU is not a
   quality metric for open-ended generation.
