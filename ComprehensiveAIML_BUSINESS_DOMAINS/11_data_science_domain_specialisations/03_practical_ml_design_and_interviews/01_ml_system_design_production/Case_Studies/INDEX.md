# ML System Design Production - Case Study Collection

> **Mandatory authoring contract:** Every authored case follows the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and linked case master. Authored drafts are unreviewed, and their evidence, implementation stage and editorial status are distinct; other catalog rows remain plans.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 |
| Review cadence | Quarterly and on material market/regulatory change |
| Index version | 2.0-planned |
| Canonical case-ID format | CASE-03-01-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-03-01-01 | Fraud model platform | ML System Design Production | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-03-01-02 | Personalisation service | ML System Design Production | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-03-01-03 | Forecasting platform | ML System Design Production | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-03-01-04 | Content moderation system | ML System Design Production | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-03-01-05 | ML incident review | ML System Design Production | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| CASE-03-01-005 | [Delivery-risk model monitoring and drift](01_model_monitoring_drift_incident.md) | ML System Design Production | incident | Continue, contain, or use a verified fallback after a sustained distance shift | model-monitoring | distribution-shift, label-delay | illustrative | proposed | not reviewed | Draft; constructed PSI and hypothetical unit failure do not establish degradation or intervention benefit |

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
| model-monitoring | CASE-03-01-005 | [Production ML design plan](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-ml-design) | Feature shift is not quality degradation; contrast with planned incident review |

## 5. Questions and future-case backlog

The delivery-risk case asks how to select a safe fallback before labels mature; its causal intervention effect and thresholds remain unmeasured.
The fraud, personalisation, forecasting and moderation cases remain planned, with distinct decision contracts to be specified.
The incident-review plan is retained separately: it may cover an actual incident only if evidence is available and appropriately sourced.
Before drafting a planned case, assign a canonical ID, bounded decision, evidence level, and review trigger.

## 6. Review and retirement log

| Case or scope | Editorial state | Last substantive review | Trigger |
|---|---|---|---|
| CASE-03-01-005 | Draft; unreviewed | None | Change to outcome maturity, feature contract, or intervention policy |
| Planned cases | Planning only | None for case content | Evidence and case decision available |

## 7. Pre-publish checks

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
| PLAN-2026-IR026 | Select data structures from operation mix and failure semantics | Supporting curriculum; shared brief | hybrid | Which data structure fits the update/query workload while retaining the required correctness and resource bounds? | `correctness-contract` | `bounded-work`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR026 | [REF-IR026](../../../../../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md) | Eight teaching units plus rapid prompts; approximate structures, local proofs and service guarantees must be distinguished |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
