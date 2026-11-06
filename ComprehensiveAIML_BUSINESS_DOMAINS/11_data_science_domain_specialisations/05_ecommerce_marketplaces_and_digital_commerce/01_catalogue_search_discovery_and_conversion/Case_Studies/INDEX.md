# Catalogue, Search, Discovery and Conversion - Case Study Collection

> **Mandatory authoring contract:** Every authored case must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 |
| Review cadence | Quarterly and on material market/regulatory change |
| Index version | 2.0-planned |
| Canonical case-ID format | CASE-05-01-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-05-01-01 | Search relevance recovery | Catalogue, Search, Discovery and Conversion | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-05-01-02 | Catalogue enrichment | Catalogue, Search, Discovery and Conversion | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-05-01-03 | Recommendation cold start | Catalogue, Search, Discovery and Conversion | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-05-01-04 | Checkout funnel diagnosis | Catalogue, Search, Discovery and Conversion | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-05-01-05 | Review abuse detection | Catalogue, Search, Discovery and Conversion | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |

## 3. Canonical reusable patterns

| Tag | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| data-leakage | Information unavailable at decision time inflates performance | Unexpectedly strong predictive result | Feature is truly available before the cutoff |
| label-delay | Outcome truth arrives late or changes | Fraud, claims, retention, downstream operational labels | Immediate, stable labels |
| distribution-shift | Operating population differs from training/evaluation | New market, season, channel, or rollout | Change is sampling noise only |
| evaluation-mismatch | Offline metric does not represent the business decision | Ranking, forecast, LLM, agent, or multi-objective products | Metric captures decision and guardrails |
| cost-latency-tradeoff | Quality, cost, and response time conflict | Online, real-time, or edge operation | Cost/latency are non-binding |
| human-in-the-loop | Qualified review changes risk/quality trade-off | High-impact, uncertain, or regulated work | Review is infeasible or ineffective |
| rollout-regression | Release degrades a live guardrail despite tests | Canary, A/B, feature-flag, or migration work | No rollout comparison exists |
| model-monitoring | Production signals reveal drift or degradation | Deployed model/system with feedback | No meaningful feedback signal |
| causal-confounding | Observational relationship is not intervention effect | Attribution, uplift, promotion, policy decisions | Valid randomization addresses concern |
| incident-postmortem | Operational impact requires causal reconstruction | Actual incident or replay exercise | No failure event/replay exists |

## 4. Planned cross-case map

| Pattern | Supporting planned cases | Theory/method links | Boundary to test |
|---|---|---|---|
| evaluation-mismatch | All planned cases | Relevant method and evaluation modules | Whether proxy metric changes decision quality |
| distribution-shift | Market, season, policy, or user changes | Forecasting, MLOps, and domain indexes | Population and operational conditions |
| human-in-the-loop | High-impact operational cases | Governance, UX, and domain workflows | Review capacity and override quality |

## 5. Backlog and review

Before drafting any listed case, replace the PLAN ID with a CASE ID, add a relative file link, choose one primary pattern and no more than two secondary patterns, state evidence/stage independently, and set a concrete review trigger. The full 15-section integrated case-study outline is mandatory for each drafted case.

## 6. Pre-publish checks

- [ ] Case ID is unique and linked.
- [ ] Evidence maturity is distinct from implementation stage.
- [ ] Decision, metric, action, and outcome refer to the same population.
- [ ] Assumptions, constraints, failure modes, and a quick-reference card are present.
- [ ] Company-specific claims have primary/public evidence or are labelled illustrative.
- [ ] Local links resolve and the review trigger is actionable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR021 | Evolve an authorized near-real-time search index safely | Supporting curriculum; shared brief | hybrid | How should search remain correct across indexing, deletion, ranking and shard failures? | `materialized-view` | `authorization-scope`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR024 | Bound phrase-search memory while preserving exact matches | Supporting curriculum; shared brief | hybrid | How can common-term phrase queries be evaluated without materializing huge posting lists? | `bounded-work` | `correctness-contract`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR029 | Return useful graph candidates under explicit traversal budgets | Supporting curriculum; shared brief | incident | How should a recommendation API respond when graph expansion reaches a node, edge, memory or time limit? | `bounded-work` | `correctness-contract`, `incident-postmortem` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR032 | Balance autocomplete freshness, memory and abuse controls | Supporting curriculum; shared brief | hybrid | Which terms enter a bounded suggestion index, and how can safe new trends appear promptly? | `materialized-view` | `bounded-work`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR021 | [REF-IR021](../../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md) | Exact global top-k needs a valid distributed protocol; figures in the source are scenario claims, not search-cluster benchmarks |
| PLAN-2026-IR024 | [REF-IR024](../../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Search/phrase_search_postings.md) | Rarity affects work, not semantic order; deleted documents and analyzer versions must share a consistent snapshot |
| PLAN-2026-IR029 | [REF-IR029](../../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md) | Depth and output-size limits alone do not bound expansion; source product degrees and outage figures are illustrative claims |
| PLAN-2026-IR032 | [REF-IR032](../../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Autocomplete/frequency_admission_and_freshness.md) | Sketches cannot enumerate terms; approximate frequencies and novelty thresholds must be evaluated rather than copied |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
