# Codebase · `evals-skills` (AI Evals Course / Parlance Labs)

> **Repo path:** `zip_resources/… → extracted/evals-skills-main/evals-skills-main`
> **Language:** Markdown + agent plugin manifests (`SKILL.md` files) · **License:** MIT
> **Scale:** 11 files, 9 skills, 1 workflow diagram
> **Role in the stack:** The **process** layer, packaged for agents. It encodes the *order of
> operations* of good eval work (error analysis **before** evaluator writing, validation **before**
> trust) so a coding agent cannot skip the steps that practitioners skip.

---

## 1. What it is / what it is not

**It is:** a set of nine installable **agent skills** — Markdown instruction packs that a coding
agent (Claude Code, Codex, or any `SKILL.md`-aware harness) loads on demand — that make an agent do
product-specific eval work correctly. The README draws the boundary explicitly: these are for
**product-specific** evals, "not foundation model benchmarks".

**It is not:** a library, a runner, or a metric collection. There is no executable eval code here;
the deliverable of most skills is a *review interface*, a *judge prompt*, or *validation numbers*.

**Why it belongs in a codebase dossier:** because the discipline it enforces is the difference
between an eval that predicts production behaviour and an eval that flatters it. The README states
the motivation plainly: these skills "guard against common mistakes we've seen helping **50+
companies** and teaching thousands of students".

---

## 2. Repository anatomy

```
evals-skills-main/
├── README.md                       # the routing map + rationale
├── workflow.png                    # the error-discovery loop, visualised
├── .claude-plugin/marketplace.json # Claude plugin manifest
├── .codex-plugin/plugin.json       # Codex plugin manifest
├── .agents/plugins/marketplace.json
└── skills/
    ├── evals-start/SKILL.md            # entry point / router
    ├── eval-audit/SKILL.md             # audit an existing pipeline
    ├── error-discovery/SKILL.md        # + review-loop.md   <- the important one
    ├── generate-synthetic-data/SKILL.md
    ├── write-code-eval/SKILL.md
    ├── write-judge-prompt/SKILL.md
    ├── validate-evaluator/SKILL.md
    ├── evaluate-rag/SKILL.md
    └── build-review-interface/SKILL.md
```

Each skill also ships an `agents/openai.yaml` adapter, so the same skill content targets both
Claude and Codex harnesses — a portability detail worth copying if you write skills for a team.

---

## 3. The 9 skills

| Skill | What it does | When it fires |
|---|---|---|
| `evals-start` | **Router.** Looks at your situation and sends you to the right skill. | Always the entry point |
| `eval-audit` | Inspects an existing eval setup, surfaces problems **with prioritized severity** | "I already have an eval pipeline" |
| `error-discovery` | Builds a review app, selects diverse samples, organises your notes into failure modes | "I have traces but haven't analysed them" |
| `generate-synthetic-data` | Diverse synthetic inputs via **dimension-based tuple generation** | You have too few / too homogeneous cases |
| `write-code-eval` | Code checks for **objective** failure modes | The failure is checkable in code |
| `write-judge-prompt` | LLM-as-judge for **subjective** quality criteria | The failure needs judgement |
| `validate-evaluator` | Calibrate judges against human labels via **data splits, TPR/TNR, bias correction** | Before you trust any judge number |
| `evaluate-rag` | Retrieval **and** generation quality in RAG pipelines | RAG systems (see CS-13/CS-15) |
| `build-review-interface` | Custom annotation UIs for human trace review | You need humans in the loop |

**The routing logic is the design.** The README's claim: most situations route to `eval-audit` or
`error-discovery`, and `error-discovery` "is the most important. Error discovery involves
qualitative and quantitative analysis of your traces to find failure modes. **You should only write
evals after doing this step.**"

That ordering constraint is the whole thesis — and it is exactly what practitioners get wrong.

---

## 4. Core content: the error-discovery loop, step by step

The README specifies the skill's behaviour precisely, which makes it a good spec to copy even if you
never install it:

1. **Read the dataset** and infer the **content type** (articles, agent traces, code, structured
   output, …).
2. **Design visual encoding based on what varies in the data**, using Gestalt principles —
   *colour for categories, spacing for hierarchy, opacity for importance*. (This is why the skill
   builds a UI rather than dumping JSON: the annotator must *see* variance.)
3. **Build a single-file HTML review app** served by a **Python stdlib server, no dependencies.**
4. **Cluster the data and pick a diverse initial sample** — cluster representatives **plus** random
   picks. (Pure random sampling under-covers rare failure modes; pure cluster sampling over-covers
   prototypes.)
5. **Run an interactive loop**: monitor annotations → categorise failure modes → **propose new
   samples to increase coverage**.

The human's role is explicitly narrow: **you read and leave free-text notes.** The agent sorts the
notes into failure modes, tracks coverage, and picks new samples to fill gaps. That division —
humans produce *observations*, machines produce *structure* — is the operational core.

Usage is a single sentence: `Can you help me do error analysis on traces.jsonl?`

---

## 5. Why this repo matters more than its size suggests

| Principle enforced | Where | Why it matters |
|---|---|---|
| Error analysis precedes evaluator writing | `error-discovery` | Evals written from imagination measure imagined failures |
| Objective checks are code, not judges | `write-code-eval` vs `write-judge-prompt` split | Judges are expensive, slow and biased; never spend one on a regex |
| Judges must be **validated**, not assumed | `validate-evaluator` (splits, TPR/TNR, bias correction) | The single most-skipped step in the industry |
| Audit before extend | `eval-audit` with **prioritized severity** | Prevents adding metric #14 to a broken pipeline |
| Sampling must be stratified, not random | cluster reps + random picks | Rare-but-severe failures are invisible to uniform sampling |
| Human effort should be minimised and focused | free-text notes only, machine does the sorting | Annotation budget is the binding constraint |

**Analyst note:** the `write-code-eval` / `write-judge-prompt` dichotomy is the highest-value idea
here and it maps directly onto the "verifiable beats judgeable" thesis (CS-17). If a failure mode can
be reduced to a schema check, a tool-argument assertion, or a database-state assertion, a judge is
strictly worse: slower, costlier, noisier, and it cannot be unit-tested.

---

## 6. Installation & usage

```bash
# Install all skills
npx skills add https://github.com/ai-evals-course/evals-skills

# Install one skill only
npx skills add https://github.com/ai-evals-course/evals-skills --skill error-discovery

# Keep them current
npx skills check
npx skills update
```

Then, from your agent:

```
Can you help me do error analysis on traces.jsonl?
```

The plugin manifests (`.claude-plugin/marketplace.json`, `.codex-plugin/plugin.json`) mean the same
skills install into Claude Code and Codex without editing content.

---

## 7. Strengths, weaknesses, when to use it

| | Assessment |
|---|---|
| **Strengths** | Encodes the correct *order of operations*, which is rarer and more valuable than any metric library. Dependency-free review app (stdlib only) means it works in any environment, including air-gapped ones. Explicit validation step with TPR/TNR. Cross-harness portability. Explicitly scoped to product evals, avoiding the benchmark/product confusion that ruins most eval projects. |
| **Weaknesses** | Markdown-only: no runnable eval library, so you still assemble the harness yourself (RAGAS/DeepEval/Langfuse/`CODE-02`). The `error-discovery` UI is deliberately throwaway — it is an annotation tool, not a product. Coverage is limited to what the course teaches: the README concedes "production monitoring, regression suites, and cost" live only in the paid course. As agent skills they depend on a capable agent harness and can be skipped or ignored by the model. |
| **Choose it over…** | …reading a blog post and hoping your team follows it. If your failure mode is *process drift* rather than *missing tooling*, this is the right artifact. |
| **Don't choose it for** | Benchmarking models (use `CODE-04`), or as your only eval infrastructure (use it as the method, plus a runner and a platform). |

---

## 8. Minimal usable example

```bash
npx skills add https://github.com/ai-evals-course/evals-skills --skill error-discovery
npx skills add https://github.com/ai-evals-course/evals-skills --skill validate-evaluator

# In your agent session:
#   "Can you help me do error analysis on traces.jsonl?"
#  -> agent profiles the data, builds a single-file HTML review app on a stdlib server,
#     picks a diverse sample (cluster reps + random), and starts the annotation loop.
#
#   "Validate this judge against labels.csv using TPR/TNR."
#  -> data splits, TPR/TNR, bias correction.
```

If you do not install skills, the transferable artefact is the loop itself: **profile data → build
review UI → stratified sample → free-text notes → machine-clustered failure modes → new samples →
only then write evaluators → validate with TPR/TNR.**

---

## 9. Reading order for a newcomer

| Step | Path | Time |
|---|---|---|
| 1 | `README.md` | 10 min — the routing map |
| 2 | `skills/evals-start/SKILL.md` | 5 min — see how routing is written |
| 3 | `skills/error-discovery/SKILL.md` + `review-loop.md` | 25 min — the core method |
| 4 | `skills/validate-evaluator/SKILL.md` | 15 min — the discipline everyone skips |
| 5 | `skills/write-code-eval/SKILL.md` then `write-judge-prompt/SKILL.md` | 20 min — in that order |
| 6 | `skills/evaluate-rag/SKILL.md` | 15 min — bridges to CS-13/CS-14/CS-15 |
| 7 | `skills/eval-audit/SKILL.md`, `generate-synthetic-data/SKILL.md`, `build-review-interface/SKILL.md` | 25 min |

---

## Cross-references

- **Concepts:** CS-04 (the complete eval workflow — this repo is its operational form), CS-07/CS-08
  (judges and their validation), CS-13/CS-14/CS-15 (RAG evaluation), CS-20 (production evals), CS-22
  (scaling evals).
- **Sibling codebases:** `CODE-01` (`PATTERNS.md` is the prose+code twin of these skills, including
  the same TPR/TNR validation function), `CODE-02` (where the binary judge prompt format comes from),
  `CODE-06` (Langfuse — where annotations and human labels actually get stored at scale).
