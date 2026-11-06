# Case Study Collection INDEX — Template (v2.0)

> Purpose: maintain a searchable, reviewable catalog of case studies by **decision**, **reusable pattern**, **domain**, **evidence maturity** and **theory links**. Create/update this INDEX *before* drafting a new case. This file is a blank collection template; placeholder rows are **not** actual cases.
>
> Companion writing guide: [Integrated Case Study Outline Template](Case_Study_Outline_Template_v2_Integrated.md).

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | `[Team / role]` |
| Last reviewed | `YYYY-MM-DD` |
| Review cadence | `[Monthly / quarterly / on trigger]` |
| Index version | `1.0` |
| Canonical case-ID format | `CASE-YYYY-NNN` |
| Canonical tags | Defined below; do not introduce synonyms without review |
| Source of truth | `[Repo or directory]` |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| `CASE-YYYY-NNN` | `[Title](cases/CASE-YYYY-NNN.md)` | `[Domain]` | `[analytical / predictive-ml / llm-agentic / streaming-systems / incident / hybrid]` | `[Question]` | `[Canonical tag]` | `[0–2 tags]` | `[observed / published / estimated / simulated / illustrative / not-measured]` | `[proposed / pilot / production / scaled / paused / retired]` | `YYYY-MM-DD` | `[One sentence]` |

**Catalog policy:** Add a case *planned* if not yet written, but explicitly label it `not-measured / proposed` and leave the result empty rather than filling in a hypothetical outcome. Verify that relative Markdown links resolve at release time.

## 3. Canonical reusable pattern vocabulary

Keep **roughly 15–20 active canonical patterns**. The 18 sample definitions below are editorial starting points and may be refined for your collection. Prefer mechanism names over generic industry/topic tags.

| Tag | Pattern / diagnostic signature | When applicable | Boundary / when not to use |
|---|---|---|---|
| `saturation-curve` | Marginal gain flattens after increased exposure | Channel response or system scaling plateaus | No dose/scale response has been measured |
| `diminishing-returns` | Each additional unit produces smaller incremental benefit | Resource allocation at the margin | Marginal effect is unknown or highly unstable |
| `budget-reallocation` | Shift finite budget under explicit constraints | Optimization of competing interventions | No reliable comparator or constrained objective |
| `causal-confounding` | Observed association may be driven by selection or common causes | Before/after, observational optimization or uplift | Randomization already addresses the suspected confound |
| `data-leakage` | Model uses information unavailable at decision time | Predictive model unexpectedly overperforms | Feature is legitimately available before the prediction cut-off |
| `label-delay` | Ground truth arrives late or inconsistently | Fraud, insurance, clinical or delayed outcome labels | Labels are immediate and stable |
| `distribution-shift` | Operating data changes relative to training/evaluation | Performance decay or new domain deployment | Difference is only random sampling noise |
| `evaluation-mismatch` | Offline metric fails to capture task/business success | ML/LLM/system evaluation gap | Metric already measures the end decision and guardrails |
| `rag-vs-long-context` | Retrieval/context strategy changes with cost, quality or workload | Document/question-answer architecture selection | Choice is untested for the actual context/price regime |
| `agentic-blast-radius` | Tool-authorized actions magnify risk of model error | Agents with side-effectful tools | Workflow is read-only with no consequential external action |
| `human-in-the-loop` | Human confirmation limits risk or improves quality | High-impact/uncertain decisions | Review latency invalidates the workflow or no qualified reviewer exists |
| `stream-backpressure` | Producer/consumer rate mismatch accumulates backlog | Streaming pipelines and queue-based systems | Bottleneck lies elsewhere, not in flow-rate imbalance |
| `partition-skew` | Uneven partition work drives tail latency or lag | Kafka/Flink/partitioned data processing | Load is uniform and imbalance is not observed |
| `checkpoint-recovery` | Recovery correctness and duration depend on saved state | Stateful streams or long-running training | Workflow is stateless or restart semantics differ |
| `cost-latency-tradeoff` | Performance improvement increases spend or resource load | Serving/architecture optimization | Cost and latency are not binding constraints |
| `rollout-regression` | Deployment changes cause regressions despite passing pre-release tests | A/B, canary, feature flag or release incidents | No rollout event or observed post-change divergence |
| `model-monitoring` | Post-deployment observations detect drift, failure or degradation | Live predictive/LLM systems | Not deployed, or no meaningful feedback signal exists |
| `incident-postmortem` | Structured incident analysis yields preventive changes | Failure with customer/operational impact | There was no actual incident; use an analysis-limit tag instead |

### Pattern definition checklist

For each new or renamed pattern, record **definition**, **signature**, **necessary preconditions**, **counterexample**, **primary/secondary case links**, **theory links**, and **deprecated synonyms**. Require the pattern to appear in at least one real/planned case, otherwise avoid adding it.

## 4. Cross-case similarity and contrast map

| Pattern | Supporting case(s) | Contrasting case(s) | Theory/methodology link | Boundary learned |
|---|---|---|---|---|
| `[tag]` | `[Case link(s)]` | `[Case link(s)]` | `[Theory module](theory/module.md)` | `[When not to generalize]` |

## 5. Questions and future-case backlog

| Open question | Originating case | Needed evidence | Proposed next case or experiment | Priority | Owner role | Trigger / review date |
|---|---|---|---|---|---|---|
| `[Unanswered question]` | `[Case link]` | `[Holdout / data / benchmark / incident replay]` | `[Follow-up case]` | `[High/Med/Low]` | `[Role]` | `YYYY-MM-DD` |

## 6. Review and retirement log

| Case ID | Date | What changed | Evidence or market trigger | Action | Owner role |
|---|---|---|---|---|---|
| `CASE-YYYY-NNN` | `YYYY-MM-DD` | `[Pricing, regulation, tool versions, results, deployment status]` | `[Verifiable source]` | `reviewed / updated / superseded / retired` | `[Role]` |

## 7. Pre-publish index checks

- [ ] All case IDs are unique; planned vs published states are explicit.
- [ ] Relative case links and theory links resolve.
- [ ] Each case uses one canonical primary tag and no more than two secondary tags.
- [ ] Evidence maturity and implementation stage are **distinct** fields.
- [ ] Last-reviewed date and re-review trigger are present on the case and the INDEX.
- [ ] Every claim of measured impact has a source or evidence artifact and scope.
- [ ] Cross-case comparisons state *what changed* (population, method, scale, time, constraints).
- [ ] Outdated cases are marked superseded/retired, not silently deleted.

---

*INDEX template v2.0 · November 2026 · Designed for the companion 15-section integrated case-study template.*
