# Cheat Sheet · LLM-as-a-Judge: Design & Validation

> **Covers:** CS-07 (LLM-as-a-judge), CS-08 (G-Eval), CS-04 (the complete eval workflow) ·
> **Code ground truth:** `CODE-01` (awesome-evals `PATTERNS.md`), `CODE-02` (OpenAI evals
> `cot_classify`), `CODE-03` (evals-skills `validate-evaluator`) · **Use when:** you are about to
> write a judge, or you are about to trust one someone else wrote.

---

## The 60-second version

A judge is a **classifier**, and it must be treated like one. That single reframing is the whole
discipline: classifiers have labels, thresholds, held-out data, and — critically — **two** error
rates that must be reported separately.

The recipe: run **error analysis first** (you cannot write an evaluator for a failure mode you have
not observed); write **one binary judge** for **one** failure cluster; put the signal in few-shot
**critiques** rather than a long rubric; require reasoning **before** the verdict; then validate
against ~100 human labels on a held-out split and report **TPR and TNR separately**. Ship only when
both clear roughly **0.9**. Re-validate whenever the agent, prompt or data distribution changes.

The step teams skip is validation. It is also the only step that tells you whether the number means
anything.

---

## Core concepts

| Concept | The rule | Why |
|---|---|---|
| **Error analysis first** | Sample 30–50 real traces → open-code notes → axial-code into named failure modes → rank by frequency × severity | Evals written from imagination measure imagined failures |
| **One judge, one failure mode** | Write a judge for a single named cluster | A judge that scores "quality" measures nothing in particular |
| **Binary PASS/FAIL** | Never a 1–5 Likert | "The gap between a 3 and a 4 is noise, but pass/fail forces a crisp decision and lets you use classification metrics" |
| **Reasoning before verdict** | Chain-of-thought, then the label | The reasoning is what you read when it fails; `include_reason=True` is not optional |
| **Few-shot *critiques*, not rubrics** | Pair PASS/FAIL examples with a domain expert's one-line written *why* | The signal lives in the critiques; a long rubric dilutes it |
| **Held-out split** | Fit examples and validation examples must be disjoint | Otherwise you measure memorisation |
| **TPR and TNR, never accuracy** | Report both plus raw FP/FN | Accuracy lies on imbalanced data |
| **Re-validation triggers** | Agent change, prompt change, data-distribution change | A validated judge is not validated forever |
| **Version pinning** | Record the judge model and prompt version with every run | A silent judge upgrade invalidates every baseline |

**The critique-shadowing move.** Have a domain expert label examples *and write one sentence on each
explaining the call*. Promote 4–8 of those critiques into the judge prompt as few-shot examples. This
is the load-bearing part — not the rubric. It transfers the expert's *judgement*, not just their
labels.

---

## Formulas & metrics

```
TPR (recall / sensitivity)  = TP / (TP + FN)     share of REAL failures the judge catches
TNR (specificity)           = TN / (TN + FP)     share of REAL passes the judge clears
PPV (precision)             = TP / (TP + FP)     of the judge's FAIL calls, how many were right
NPV                         = TN / (TN + FN)
FP rate                     = 1 - TNR
```

Where **"positive" = the judge says FAIL** — i.e. the event you are trying to detect is a *failure*.

```python
def validate(judge, labeled):
    tp=fp=tn=fn=0
    for inp, out, human in labeled:
        verdict = judge(inp, out)          # "PASS" or "FAIL"
        fail = (verdict == "FAIL")         # positive = catches a real failure
        human_fail = (human == "FAIL")
        if   fail and human_fail: tp += 1
        elif fail and not human_fail: fp += 1
        elif not fail and not human_fail: tn += 1
        else: fn += 1
    tpr = tp/(tp+fn) if tp+fn else 0.0
    tnr = tn/(tn+fp) if tn+fp else 0.0
    return {"TPR": tpr, "TNR": tnr, "FP": fp, "FN": fn}
# Ship only when BOTH TPR and TNR clear ~0.9 on held-out data.
```

**The trap this exposes.** A judge that rubber-stamps PASS scores **~90% agreement when 90% of traces
genuinely pass** — while catching **zero** failures. Its TPR reads ~0 and its accuracy looks
excellent. Raw agreement on imbalanced data is not a validation.

**The empirical corroboration.** Binary judges reached **>95% precision on *consistent* summaries but
only ~30–60% recall on *inconsistent* ones**. Precision looks great; the false-negative blind spot is
enormous, and only TPR surfaces it.

**Prevalence correction.** A judge validated on a 50/50 split will have a *different* PPV in
production if the real failure rate is 2%. Report TPR/TNR (which are prevalence-independent) and
recompute PPV/NPV for the deployment prevalence before quoting an expected false-alarm rate.

**Pass-rate estimators (the other half of judging).**

```
pass@k  (unbiased, OpenAI human-eval):
    pass@k = 1 - C(n-c, k) / C(n, k)      computed as a running product for stability
    n = samples drawn, c = of those, how many were correct
    NEVER use 1 - (1-p)^k naively: biased high for small n

pass^k  (all k trials must succeed — reliability):
    (per-trial success)^k ,  estimated from the same n, c
```

With **p = 0.75**: `pass@10 ≈ 1.0` but `pass^10 ≈ 0.056`. They tell opposite stories about the same
model.

| Reach for | When |
|---|---|
| **pass@k** | One success out of k is enough — capability, best-of-k, anything with a verifier |
| **pass^k** | Every one of k independent trials must succeed — reliability, customer-facing policy adherence |

**G-Eval, for contrast.** G-Eval is the *rubric-scoring* judge: it generates evaluation steps from a
criterion, then scores with token probabilities weighted by step importance. It is more expressive
than a binary classifier and correspondingly harder to validate. Use it when the criterion is
genuinely graded (e.g. summarisation coherence), not when you are detecting a specific failure.

---

## Decision rules

1. **Do error analysis before writing any evaluator.** No exceptions. If you cannot name the failure
   mode in a sentence, you cannot write a judge for it.
2. **One judge per failure mode.** Compose multiple judges rather than asking one judge to score five
   things.
3. **If it is checkable in code, do not spend a judge on it.** Schema validity, regex, tool-call
   arguments, citation presence, database state — all cheaper, faster, and unit-testable.
4. **Binary before graded.** Only escalate to a graded scale when a genuine ordering is required.
5. **Reasoning first, verdict last.** Constrain the output format so the verdict parses
   deterministically.
6. **Fit few-shot examples from real errors.** Synthetic examples teach the judge your imagination.
7. **Hold out the validation set before you iterate.** Looking at the held-out set while tuning turns
   it into a training set.
8. **Report TPR, TNR, and raw FP/FN counts.** Never a single accuracy number.
9. **Set the ship bar per criterion.** Safety-critical criteria weight TPR higher (a missed failure
   is worse than a false alarm); cost-sensitive criteria weight TNR higher.
10. **Re-validate on any distribution change** — new model, new prompt, new traffic mix, new corpus.
11. **Pin the judge model version.** Record it with every run.
12. **Budget the judge.** Cache identical judge inputs; use cheap models for cheap criteria; route
    expensive judges to a sampled subset.

---

## Thresholds & defaults worth memorising

| Item | Value |
|---|---|
| Judge ship criterion | **TPR ≥ 0.9 and TNR ≥ 0.9** on held-out data (both, not either) |
| Human labels needed | **≥ 100** expert-labelled examples |
| Few-shot critiques in prompt | **4–8** |
| Trace sample for error analysis | **30–50** real traces |
| Error-analysis clustering | **Open coding** (free-text notes) → **axial coding** (named clusters) → **prioritise** (frequency × severity) |
| Binary vs Likert | **Binary**, unless a genuine ordering is required |
| Judge temperature | 0 for reproducibility |
| `include_reason` | Always on — the aggregate number does not tell you what to fix |
| Re-validation trigger | Any change to agent, prompt, or data distribution |
| pass@k vs pass^k | p = 0.75 → **pass@10 ≈ 1.0**, **pass^10 ≈ 0.056** |

---

## Tool commands & code shapes

**A judge is a prompt with a constrained output — that is the whole specification.** OpenAI evals
expresses it as YAML; the load-bearing part is the prompt's three moves:

```yaml
# registry/evals/modelgraded/closedqa.yaml  (shape, not the literal file)
args:
  model: gpt-4o
  choice_scores: { "Y": 1.0, "N": 0.0 }
instructions: |
  - Read the question, the criteria, and the submission.
  - Reason step by step about whether the submission satisfies the criteria.
  - Print only the letter of your answer on the final line: Y or N.
  - Then repeat that letter one final time.
```

Three moves to copy verbatim: **reason step by step** → **print only the letter** → **repeat the
letter**. The repetition is what makes the verdict robustly parseable from a verbose completion.

**Validation harness** — use the `validate()` function above; point it at your own `labeled`
list of `(input, output, "PASS"|"FAIL")` triples.

**Minimal CLI shape:**

```bash
# error analysis on real traces (agent-skill form)
npx skills add <repo> --skill error-discovery      # -> "do error analysis on traces.jsonl"
npx skills add <repo> --skill validate-evaluator   # -> splits, TPR/TNR, bias correction
```

---

## Top 10 mistakes

1. **Skipping error analysis** and writing judges from imagination.
2. **Reporting accuracy or raw agreement** instead of TPR/TNR — hides the false-negative blind spot.
3. **Validating on the same examples used as few-shot examples** — measures memorisation.
4. **A 1–5 Likert scale** where a binary decision was needed — noise dressed as resolution.
5. **A long rubric in the system prompt** instead of few-shot critiques — the signal is in the
   critiques.
6. **One giant judge** scoring relevance, tone, format and safety at once — uninterpretable when it
   moves.
7. **Spending a judge on a regex** — schema checks, citation presence and tool-argument assertions
   belong in code.
8. **Not re-validating** after a model or prompt change — the judge's TPR silently drifts.
9. **Unpinned judge model version** — a provider upgrade invalidates every historical baseline.
10. **Judging everything** — judge cost should be routed by criterion and by sampling, or the bill
    eats the team's iteration budget.

---

## If you only remember three things

1. **Error analysis → one binary judge for one observed failure mode → reasoning before verdict.**
2. **Validate with TPR and TNR separately on a held-out split; ~90% agreement can mean zero
   coverage.** Ship only when both clear ~0.9, and re-validate on every distribution change.
3. **If it is checkable in code, never spend a judge on it.**
