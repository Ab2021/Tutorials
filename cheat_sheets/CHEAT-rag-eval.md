# Cheat Sheet · RAG Evaluation

> **Covers:** CS-13 (retriever metrics hands-on), CS-14 (three-level eval suite + interview framing),
> CS-15 (operational evals), CS-16 (RAG safety) · **Use when:** you have a retriever + generator
> pipeline and must decide whether it is shippable, or you are answering "how do you evaluate your
> RAG app?" in an interview.

---

## The 60-second version

Evaluate in the order you build. **Component level** first — retriever alone, then generator alone —
so a bad answer is never ambiguous. **Then pipeline**: the RAG Triad. **Then application**:
correctness, completeness, style. **Then safety**, then **ops**. Each band is one file under
`evals/`; `run_evals.py` runs them all and diffs against a stored baseline. A metric more than
**3 units** below baseline blocks the deploy. After deploy, keep the same metrics running online and
watch a **24-hour drift graph**; harvest production failures back into the golden set.

The single most common interview failure is reciting `recall, precision, faithfulness, answer
relevance` as a flat list. The metrics are not the answer — **the level each metric belongs to is the
answer.**

---

## Core concepts

| Term | Meaning | Where it lives |
|---|---|---|
| **Component level** | Retriever and generator evaluated separately, as each is built | `evals/eval_retriever.py`, `eval_generator.py` |
| **Pipeline level** | The wired retriever→generator path | RAG Triad file |
| **Application level** | Whole product, user-facing quality | correctness / completeness / style file |
| **RAG Triad** | Context relevance, faithfulness, answer relevance — one per pipeline entity | pipeline file |
| **Golden dataset** | Curated question (+ ideal answer) set used as ground truth | `evals/` fixture |
| **Baseline** | Metric values from the first full run — everything is judged relative to it | experiment 1 |
| **Regression testing** | Running the whole suite on the whole app after a change | `run_evals.py` |
| **Drift** | A deployed app's metrics degrading relative to the world | online dashboard |
| **Citation accuracy** | Whether the cited document/line is the one actually used | generator band |
| **Completeness** | Whether the answer covers *every* part of a multi-part question | application band |
| **TTFT** | Time to first token — what the user actually experiences | ops band |

**The three entities → three metrics.** A RAG pipeline contains exactly three things: the **user
question**, the **retrieved context**, the **generated answer**. The triad exists because there are
exactly three pairs between them.

---

## Formulas & metrics

**Retrieval**

| Metric | Formula | Answers |
|---|---|---|
| **Recall** | `correct_retrieved / all_correct_that_exist` | Of the documents I should have found, how many did I bring? |
| **Precision** | `correct_retrieved / all_retrieved` | Of the documents I brought, how many were useful? |
| **Contextual Recall** (RAGAS) | `claims_found_in_retrieved / total_claims_in_ideal_answer` | Reference-based, survives re-chunking |
| **Contextual Precision** (DeepEval) | mean of prefix precision at each rank | Rank-aware precision |

**The precision trap.** Two retrievers returning 5 chunks of which 2 are correct have *identical*
plain precision (2/5). They are not equivalent:

```
A: ✓ ✓ ✗ ✗ ✗   prefix precision  1/1, 2/2, 2/3, 2/4, 2/5  -> 0.713
B: ✗ ✗ ✗ ✓ ✓   prefix precision  0/1, 0/2, 0/3, 1/4, 2/5  -> 0.130
```

Rank matters because the generator weights early context more. **Always use rank-aware contextual
precision, never plain precision over a set.**

**Why not gold chunk IDs?** The label is tied to the chunking, not the content. Change chunk size
from 750 → 1000 and the same passage gets a new ID — every gold row dies and you relabel everything.
Defensible only if the corpus is cleanly separated documents **and** chunking is frozen forever.

**Generation / quality**

| Metric | Pair checked | Judges |
|---|---|---|
| **Context relevance** | question ↔ retrieved context | Is the retrieved context on-topic? |
| **Faithfulness** | answer ↔ context | Did the answer come from context, or is it hallucinated? |
| **Answer relevance** | answer ↔ question | Does the answer address the question? |
| **Citation accuracy** | answer ↔ source | Is the cited document/line correct? |
| **Correctness** | answer ↔ truth | Is the answer right? |
| **Completeness** | answer ↔ question parts | Did it answer *all* of it? |

> A two-part question answered partially is **correctness 1.0, completeness 0**. A single
> "correctness" metric hides this entirely.

**Operational**

```
cost_per_query = (input_tokens/1e6 * in_rate) + (output_tokens/1e6 * out_rate)
```
Output rate is typically **~4× the input rate**. Report cost **per query** (not per month) and as a
*distribution* — a typical 2-paise query with 1.5-rupee outliers is a different business than a flat
2-paise one.

```
end_to_end_latency = retrieval + generation        (~0.7 s + ~2.9 s in the source's demo)
```

Reliability = four separate numbers, never one: **success rate**, **error rate** (its complement),
**timeout rate** (a hung request is not an error), **retry rate**.

---

## Decision rules

1. **Build → evaluate → advance.** Never build the whole app then test it. Evaluate the retriever
   before the generator exists.
2. **Test the generator with hand-fed question + context.** You control the input, so a failure is
   unambiguously the generator's. This is a unit test, not a pipeline test.
3. **Fix k last.** Recall and precision trade off directly through k. Exhaust chunk geometry,
   reranker and embedding model first — k is the one lever with no free lunch.
4. **Force a re-embed** whenever chunk size, overlap or embedding model changes — delete the store
   directory first, or the loader silently reuses the stale index.
5. **Read the judge's reasoning on every failure.** The aggregate number does not tell you what to
   fix; `include_reason=True` does.
6. **Gate on the delta, not the absolute.** Deploy only if the change is incrementally better than
   baseline; otherwise block.
7. **Read both ledgers.** If quality rose but latency/cost doubled, check which constraint is hard
   before celebrating.
8. **Never quote a tail percentile from a small sample.** With n = 25 the P99 *is* the maximum.
9. **Publish the timeout rate on the same line as any latency percentile** — timeouts are excluded
   from the distribution, so a "faster" P95 can mean you are dropping more requests.
10. **Recompute cost without the prompt-cache credit.** Sending the same question repeatedly inflates
    provider-side prefix caching; production questions differ, so the real rate is lower.

---

## Thresholds & defaults worth memorising

| Item | Value | Source |
|---|---|---|
| CI deploy gate | block if any metric drops **> 3 units** vs baseline | CS-14 |
| Metric pass threshold | **0.7** (DeepEval `threshold=0.7`) | CS-13 |
| k (top-k retrieval) | **5** baseline; k=3 rejected (lost 1 point at n=15) | CS-13 |
| Chunk geometry | **750/100 → 1000/150** was the largest single gain (recall 80 → 97) | CS-13 |
| Drift window | **24 h graph**, alert if a metric degrades over the **last 8 h** | CS-14 |
| Latency budget | system P95 ≤ **3,000 ms**; TTFT ≤ **1,200 ms** | CS-15 |
| Cost budget | express as **per-query ceiling** (e.g. 50 paise/complex query) | CS-15 |
| Reliability samples | 25–50 is too few; push toward **1,000** to see rare errors | CS-15 |
| Latency repeats | **5× per question** minimum; discard first 2 runs as cold start | CS-15 |
| Golden set size | 15 rows detects large moves only — **1–2 point deltas are noise** | CS-13 |

---

## Tool commands

```bash
# Retriever component eval (from repo root; needs __init__.py in BOTH src/ and evals/)
python3 -m evals.eval_retriever

# Ops evals — no judge, no golden data, free to run on every commit
python3 -m evals.eval_latency
python3 -m evals.eval_cost
python3 -m evals.eval_reliability

# Full suite + baseline diff
python run_evals.py
python compare_to_baseline.py     # exit non-zero if any metric dropped > 3

# Force a clean re-embed
rm -rf chroma_store/
```

```python
# DeepEval shape
LLMTestCase(input=..., expected_output=ideal, retrieval_context=chunks, actual_output=placeholder)
metric(threshold=0.7, model=judge, include_reason=True)
evaluate(test_cases=[...], metrics=[...])   # score < threshold = FAIL
```

**Library choice:** DeepEval over RAGAS here — broader (agents, multi-turn, multi-modal) and built
on **pytest**. Learn the *concept*, not the tool; a tool is a week's work.

---

## Top 10 mistakes

1. Answering the interview question with a flat metric list and no levels.
2. Building the whole RAG app and *then* evaluating it — no failure localisation.
3. One single "score" for everything, so a change cannot be attributed.
4. Using plain precision instead of rank-aware contextual precision.
5. Labelling the golden set with chunk IDs, then changing chunk size.
6. Changing the embedding model without deleting the vector store.
7. Setting k high "because recall went up" — and handing the generator more noise.
8. Deploying on vibes: no baseline, no threshold, no gate.
9. Optimising the retriever when the generator is 4× the latency.
10. Stopping at deploy — no online eval, no drift alert, and a golden set that never grows.

---

## If you only remember three things

1. **Three levels, in build order:** component → pipeline (RAG Triad) → application. Name the level
   *before* the metric.
2. **Recall and precision trade off through k; rank matters; re-chunking invalidates ID-based labels.**
3. **A quality gain that doubles cost and latency is not automatically a win** — read both ledgers,
   then gate the deploy on the delta against baseline.
