# Codebase · `awesome-evals` (BenchFlow)

> **Repo path:** `zip_resources/… → extracted/awesome-evals-main/awesome-evals-main`
> **Language:** Markdown (plus a GitHub Action) · **License:** see `LICENSE`
> **Scale:** 149 `.md` files; `README.md` with 13 thematic sections + talks + mentions + landscape
> **Role in the stack:** Not a runner — a **curated, annotated canon** plus a **runnable pattern
> playbook**. It is the bibliography and the recipe book that the other six repos implement.

---

## 1. What it is / what it is not

**It is:** an opinionated, *verified* reading list for agent evaluation, and — more usefully —
`PATTERNS.md`, a playbook of ten evaluation patterns given as real code with worked examples,
action items and pitfalls. The README claims a specific provenance method worth noting: a
**depth-4 recursive citation crawl** over **11.6k papers** ranked by in-degree, targeted
practitioner-web discovery for sources citation graphs miss, **47 talks/podcasts transcribed and
deep-noted** with verbatim quotes and timestamps, and per-section gap audits with adversarial
verification. It advertises **443+ curated links · 143 deep reading notes**.

**It is not:** executable. There is no test suite, no CLI, no dataset. Its value is *judgement* —
which sources are load-bearing, and which patterns actually hold up.

**Why it belongs in a codebase dossier:** `PATTERNS.md` is the single best short source of
*correct* judge code and the calibration discipline around it. It is the thing you copy from.

---

## 2. Repository anatomy

```
awesome-evals-main/
├── README.md          # the canon: 13 numbered sections + talk index + companies + gaps
├── PATTERNS.md        # 10 runnable patterns with code, worked examples, pitfalls
├── SCAN.md            # the self-updating audit workflow (runs in CI, monthly/seasonally)
├── CONTRIBUTING.md    # the "non-BS" bar: show your work, one-line why, verify URL, prune dead
├── MENTIONS.md        # eval mentions mined out of general agent talks/posts
├── notes/             # 143 deep reading notes, split into articles/ papers/ talks/
└── docs/index.html    # rendered index
```

The `notes/` split — `articles/`, `papers/`, `talks/` — maps to the three source genres the field
actually publishes in. `talks/` is the interesting one, because practitioner talks carry the
operational detail that papers omit.

---

## 3. The 10 files that matter

| # | Path | Why |
|---|---|---|
| 1 | `PATTERNS.md` § LLM-as-judge aligned to humans | The judge recipe: binary, few-shot critiques, TPR/TNR validation |
| 2 | `PATTERNS.md` § pass@k / pass^k | The unbiased estimator + why naive `1-(1-p)^k` is wrong |
| 3 | `PATTERNS.md` § Error analysis: open → axial coding | The highest-ROI activity in the field; the procedure |
| 4 | `PATTERNS.md` § Trajectory & tool-use evaluation | How to grade *how*, not just *what* |
| 5 | `PATTERNS.md` § Outcome / environment-state grading | World-state assertions as the ground-truth escape hatch |
| 6 | `PATTERNS.md` § CI gating & regression datasets | Turning evals into a merge gate |
| 7 | `PATTERNS.md` § Verifiable reward / RL-environment rubric | Where eval and training meet |
| 8 | `PATTERNS.md` § Synthetic test-data generation | Escaping the "we have no data" deadlock |
| 9 | `PATTERNS.md` § Contamination-resistant eval design | Designing so your number *stays* meaningful |
| 10 | `README.md` § 6 (benchmark vs eval) and § 8 (judge biases) | The two sections that reframe what you thought you were measuring |

---

## 4. Core content: the patterns, in detail

### 4.1 LLM-as-judge aligned to humans — the load-bearing pattern

**When to use (stated in the file):** "you need to scale a quality check across thousands of agent
traces that can't be verified with code/string match (tone, faithfulness, task-completion, 'did it
follow the instruction'), **and you have at least ~100 human-labeled examples to validate against**."

The four rules it insists on:

1. **Build the judge from error analysis, not imagination.** Write evaluators for failure modes you
   *observed*. (Hamel's rule: write evals only after error analysis.)
2. **Binary pass/fail, never 1–5 Likert.** "The gap between a 3 and a 4 is noise, but pass/fail forces
   a crisp decision and lets you use classification metrics."
3. **Put the real signal in few-shot *critiques*, not in the system prompt.** Pass/fail labels paired
   with a domain expert's written reasoning ("critique shadowing") — **not** a long rubric.
4. **Validate with TPR and TNR separately, never raw agreement.** "A judge can post 80% agreement
   while missing most real failures."

The code is reconstructed faithfully from OpenAI evals' `closedqa.yaml` `cot_classify` spec and
Braintrust's `LLMClassifier`. The validation function is the part to steal:

```python
def validate(judge, labeled):
    tp=fp=tn=fn=0
    for inp, out, human in labeled:
        verdict = judge(inp, out)          # "PASS" or "FAIL"
        fail = (verdict == "FAIL")         # "positive" = catches a real failure
        human_fail = (human == "FAIL")
        if   fail and human_fail: tp += 1
        elif fail and not human_fail: fp += 1
        elif not fail and not human_fail: tn += 1
        else: fn += 1
    tpr = tp / (tp + fn) if tp + fn else 0.0   # share of real failures the judge catches
    tnr = tn / (tn + fp) if tn + fp else 0.0   # share of real passes the judge clears
    return {"TPR": tpr, "TNR": tnr, "FP": fp, "FN": fn}
# Ship only when BOTH TPR and TNR are high (e.g. >= 0.9) on held-out data.
```

**Worked example (the strongest thing in the file):** Hamel's **NurtureBoss** real-estate assistant.
Error analysis on real traces clustered failures into named buckets; the biggest was the bot
**hallucinating appointment availability/times that were never in context**. The team wrote a binary
judge for exactly that cluster and validated TPR/TNR on held-out labels **rather than accuracy** —
because the failure class was rare, so a judge could score high agreement while catching almost none
of the actual hallucinations. Corroborating data point from Eugene Yan: binary judges hit
**>95% precision on *consistent* summaries but only ~30–60% recall on *inconsistent* ones** — the
false-negative blind spot that raw accuracy hides and TPR exposes.

**Pitfall stated verbatim:** "Raw agreement lies on imbalanced data: a judge that rubber-stamps PASS
scores ~90% agreement when 90% of traces pass while catching zero failures — TPR would read ~0."

### 4.2 pass@k vs pass^k — the two questions people conflate

- **pass@k** — reach for it when *one* success out of k attempts is enough (capability, best-of-k,
  anything with a verifier).
- **pass^k** — reach for it when *every* one of k independent trials must succeed (reliability;
  customer-facing agents that must follow policy the same way every time).

The file's headline number is the whole point:

> with **p = 0.75, pass@10 ≈ 1.0 but pass^10 ≈ 0.056** — and they tell opposite stories about the
> same model.

Never report the naive `1 - (1-p)^k`: when you drew `n` samples and `c` were correct, that plug-in
estimate is **biased high for small n**. Use the unbiased combinatorial estimator
`pass@k = 1 - C(n-c, k) / C(n, k)`, computed as a running product for numerical stability (never form
the giant binomials). The code is taken verbatim from OpenAI `human-eval`
(`human_eval/evaluation.py`); the pass^k helper is labelled reconstructed because pass^k is not in
human-eval.

### 4.3 Error analysis: open → axial coding → prioritize

The three-stage procedure the file encodes: **open coding** (free-text notes on a sample of real
traces) → **axial coding** (cluster the notes into named failure modes) → **prioritize** (count
frequency × severity to pick what to fix and what to build a judge for). This is the same workflow
packaged as an agent skill in `CODE-03` (`error-discovery`).

### 4.4 The remaining patterns

| Pattern | Core idea |
|---|---|
| Code-based assertions / unit tests | If a property is checkable in code (schema, regex, tool-call arguments, DB state), **never** spend a judge on it |
| Trajectory & tool-use evaluation | Grade the path: tool selection, argument correctness, step efficiency, recovery |
| Outcome / environment-state grading | Assert on the **world state after** the agent runs (τ-bench style: check the database, not the transcript) |
| CI gating & regression datasets | Promote production failures into a frozen regression set that gates merges |
| Verifiable reward / RL-environment rubric | A benchmark is a frozen RL environment; "verifiable beats judgeable" |
| Synthetic test-data generation | Dimension-based tuple generation to manufacture diversity instead of sampling the same distribution |
| Contamination-resistant eval design | Hold-out, freshness, private sets, and canary strings |

---

## 5. Worked example — the full judge-alignment loop

The pattern file's own end-to-end story, generalised:

1. **Sample** ~30–50 real traces (not synthetic).
2. **Open-code**: free-text notes per trace.
3. **Axial-code**: cluster into named failure modes; count frequencies.
4. **Pick the top cluster** (frequency × severity). For NurtureBoss: *invented appointment times*.
5. **Write one binary judge** for that cluster only — PASS/FAIL with CoT reasoning *before* the verdict.
6. **Expert-label 100+ examples** and write a one-line *critique* per example (the "why").
7. **Promote 4–8 critiques into few-shot examples** in the judge prompt. (This is the load-bearing
   part — not the rubric.)
8. **Validate on a held-out split** reporting TPR + TNR + FP/FN counts. Ship only when both clear
   ~0.9.
9. **Re-validate** whenever the agent, prompt, or data distribution changes.

---

## 6. Strengths, weaknesses, when to use it

| | Assessment |
|---|---|
| **Strengths** | Every entry says *what it is and why it belongs* — no link-dump. The judgement is practitioner-accurate: binary judges, TPR/TNR over accuracy, few-shot critiques over rubrics, pass@k vs pass^k disambiguation. The provenance method (citation crawl + adversarial gap audit + dead-link pruning) is itself a template for building a canon. `SCAN.md` makes curation reproducible and machine-assisted. |
| **Weaknesses** | It is a **moving target by design** — links rot, and the 2025–2026 material dominates, so it is weak on foundations. It is opinionated and vendor/community-adjacent (BenchFlow-maintained), so the "companies & landscape" section is not neutral. No runnable test suite: code in `PATTERNS.md` is illustrative and several snippets are explicitly *reconstructed*, not verbatim. |
| **Use it when** | You need to build or audit a judge; you need to cite a canonical source; you want the shortest path from "we have traces" to "we have a validated evaluator". |
| **Don't use it as** | Your only source. Pair it with primary specs (`CODE-02` YAML, `CODE-03` skills) before shipping a judge. |

---

## 7. Minimal usable example

You do not "run" this repo. You use it in three moves:

```bash
# 1. Find the canonical source for a claim
grep -rn "pass@k" PATTERNS.md README.md

# 2. Copy the judge-validation harness, then point it at your own labels
#    (PATTERNS.md -> "LLM-as-judge aligned to humans" -> validate())
#    labeled = [(input, output, "PASS"|"FAIL"), ...]  # >=100 expert-labelled

# 3. Check the provenance claim before you cite it
grep -rn "nurtureboss\|NurtureBoss" -i PATTERNS.md
```

To run the curation workflow yourself, follow `SCAN.md`: generate a subscription token with
`claude setup-token`, store it as a repo secret, and let `.github/workflows/eval-scan.yml`
(daily 08:23 UTC + manual dispatch) open a `scan/<date>` PR. The workflow never pushes to `main`.

---

## 8. Reading order for a newcomer

| Step | Path | Time |
|---|---|---|
| 1 | `README.md` § Must-read starter set | 15 min — 12 sources, read these first |
| 2 | `PATTERNS.md` § LLM-as-judge aligned to humans | 20 min — the single most valuable page |
| 3 | `PATTERNS.md` § pass@k / pass^k | 10 min |
| 4 | `PATTERNS.md` § Error analysis | 15 min |
| 5 | `PATTERNS.md` § Trajectory, § Outcome/world-state | 25 min |
| 6 | `README.md` § 6 (benchmark vs eval), § 9 (agent-specific), § 10 (safety) | 30 min |
| 7 | `PATTERNS.md` § CI gating, § Contamination-resistant design | 20 min |
| 8 | `notes/talks/` | as needed |
| 9 | `SCAN.md` + `CONTRIBUTING.md` | 15 min — if you want to contribute or self-curate |

---

## Cross-references

- **Concepts:** CS-07 (LLM-as-a-judge), CS-08 (G-Eval), CS-11 (contamination/saturation),
  CS-17 (agentic/trajectory evals), CS-20/CS-22 (production evals at scale).
- **Sibling codebases:** `CODE-02` (OpenAI evals — the `cot_classify` spec this file's judge code is
  reconstructed from), `CODE-03` (evals-skills — the same discipline as agent-executable skills),
  `CODE-04` (Evalscope — where the benchmark canon becomes runnable).
