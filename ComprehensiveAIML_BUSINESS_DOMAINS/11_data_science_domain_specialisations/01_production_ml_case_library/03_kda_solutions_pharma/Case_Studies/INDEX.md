# KDA Solutions Pharma - Case Study Collection

> **Mandatory authoring contract:** Every authored case follows the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master. Evidence, implementation stage and editorial status remain separate; source directory names are topic leads, not evidence of source contents.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 |
| Review cadence | Quarterly and on material market/regulatory change |
| Index version | 2.0-planned |
| Canonical case-ID format | CASE-01-03-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-01-03-001 | [HCP promotional-response drivers](01_HCP_Promotional_Response_Driver_Analysis_Launch_Brand.md) | KDA Solutions Pharma | analytical | Test an incremental approved HCP offer against usual scheduling | causal-confounding | evaluation-mismatch, label-delay | not-measured | proposed | not reviewed | Fictional illustration; source contents unavailable, outcome not measured; draft unreviewed |
| PLAN-01-03-02 | Patient adherence and persistency | KDA Solutions Pharma | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-03-03 | Brand-performance decomposition | KDA Solutions Pharma | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-03-04 | Payer formulary access drivers | KDA Solutions Pharma | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-03-05 | Omnichannel next-best action attribution | KDA Solutions Pharma | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |

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

## 4. Cross-case similarity and contrast map

| Pattern | Supporting case(s) | Contrasting case(s) | Theory/methodology link | Boundary learned |
|---|---|---|---|---|
| causal-confounding | CASE-01-03-001, illustrative design | Planned omnichannel attribution case, not yet authored | Randomized policy contrast and positivity, in case Section 5 | Association is not intervention effect; interference may require cluster assignment |
| evaluation-mismatch | CASE-01-03-001 | Planned adherence case, not yet authored | Outcome definition and guardrails, in case Section 9 | Prescriber response is not a patient-health outcome |
| label-delay | CASE-01-03-001 | Planned brand-performance case, not yet authored | Outcome maturity and missingness, in case Section 9 | Late events and absent coverage cannot be interpreted as zero response |

## 5. Questions and future-case backlog

| Open question | Originating case | Needed evidence | Proposed next case or experiment | Priority | Owner role | Trigger / review date |
|---|---|---|---|---|---|---|
| Can the proposed randomized offer be reviewed and measured without patient identifiers? | CASE-01-03-001 | Authorized source, eligibility, consent and protocol review | Proposed HCP policy experiment | High | Commercial decision owner | Before any test or launch |
| What prescriber-response evidence did the original source contain? | CASE-01-03-001 | Original source contents | Revisit source crosswalk | High | Curriculum maintainer | Source becomes accessible |
| Which decisions and data support the remaining planned titles? | PLAN-01-03-02 through PLAN-01-03-05 | Original source contents or separately sourced evidence | Draft each planned case | Medium | Curriculum maintainer | Before drafting each case |

## 6. Review and retirement log

| Case ID | Date | What changed | Evidence or market trigger | Action | Owner role |
|---|---|---|---|---|---|
| CASE-01-03-001 | 2026-11-07 | Illustrative draft added; not reviewed | Topic lead only; original contents unavailable | Review before publication and before any actual use | Curriculum maintainer |

## 7. Pre-publish index checks

- [ ] Case ID is unique and linked; planned titles remain distinct from authored cases.
- [ ] Evidence maturity, implementation stage and document state are independently accurate.
- [ ] Decision, metric, action, and outcome refer to the same population.
- [ ] Assumptions, constraints, failure modes, and a quick-reference card are present.
- [ ] Source contents, access limits, and unmeasured outcomes are disclosed.
- [ ] Links resolve and review triggers are actionable.
- [ ] Review dates are actual; no review is implied by the draft creation date.