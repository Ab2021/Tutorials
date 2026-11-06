# Fraud Modelling - Case Study Collection

> **Mandatory authoring contract:** Every case follows the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md). Drafts are not reviewed or published; evidence, implementation stage and editorial status are separate.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 |
| Review cadence | Quarterly and on material market/regulatory change |
| Index version | 2.0-planned |
| Canonical case-ID format | CASE-01-01-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-01-01-001 | [Real-time payment-card fraud detection](01_real_time_payment_card_fraud_detection.md) (draft; crosswalk PLAN-01-01-01) | Fraud Modelling | predictive-ml | Select approve, challenge or decline under friction and latency constraints | evaluation-mismatch | label-delay, cost-latency-tradeoff | illustrative; not-measured | proposed | not reviewed | Numerical example only; production impact unknown |
| PLAN-01-01-02 | Insurance claims fraud detection | Fraud Modelling | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-01-03 | Account takeover and application fraud | Fraud Modelling | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-01-04 | AML transaction monitoring | Fraud Modelling | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-01-05 | Marketplace fraud-ring graph detection | Fraud Modelling | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |

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

## 4. Similarity and contrast map

| Pattern | Supporting case | Contrasting case | Theory/method link | Boundary learned |
|---|---|---|---|---|
| evaluation-mismatch | CASE-01-01-001 | Planned insurance claims case | [Fraud evaluation](../../../02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md) | Synchronous card decisions differ from asynchronous claims investigation. |
| label-delay | CASE-01-01-001 | Planned account takeover case | Point-in-time labels and selective observation | Authentication outcomes need not reveal subsequent fraud. |
| cost-latency-tradeoff | CASE-01-01-001 | Planned AML monitoring case | Decision deadlines and fallback | Synchronous authorization cannot wait for manual investigation. |

## 5. Questions and future-case backlog

| Open question | Originating case | Needed evidence | Proposed next case or experiment | Priority | Owner role | Trigger |
|---|---|---|---|---|---|---|
| Does the challenge policy reduce total fraud and friction? | CASE-01-01-001 | Mature randomized outcomes and appeals | Controlled issuer trial | High | Fraud product owner | Before rollout |
| Are features timely and replayable under outage? | CASE-01-01-001 | Timestamped production traces and recovery exercises | Load and failure experiment | High | Platform owner | Before launch |
| Are selected labels biasing calibration? | CASE-01-01-001 | Independent audits and maturity curves | Label-audit study | High | Model risk owner | Before threshold approval |
| Is fraud displaced to subsequent attempts? | CASE-01-01-001 | Linked account-level follow-up | Longer-window randomized analysis | Medium | Risk analytics owner | Before scaling |

## 6. Review and retirement log

| Case ID | Date | What changed | Evidence or market trigger | Action | Owner role |
|---|---|---|---|---|---|
| CASE-01-01-001 | 2026-11-07 | Draft registered; no substantive review | Illustrative source brief | Review before publication or implementation | Curriculum maintainer |

## 7. Pre-publish checks

- [ ] The draft receives substantive technical and numerical review.
- [ ] Open questions, source limits, and proposed gates remain explicit.
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
| PLAN-2026-IR036 | Evaluate incremental investigator value from graph fraud signals | Supporting curriculum; shared brief | predictive-ml | Which graph-enriched alerts add confirmed value at a fixed investigation capacity beyond claim-level scoring? | `delayed-selective-labels` | `human-in-the-loop`, `data-leakage` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR036 | [REF-IR036](../../../../../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md) | Connectivity and shared providers are not proof of fraud; source reference notes are unavailable and labels are delayed/selected |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
