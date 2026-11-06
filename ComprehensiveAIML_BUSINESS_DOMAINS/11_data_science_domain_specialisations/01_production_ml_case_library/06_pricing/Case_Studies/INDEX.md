# Pricing and Revenue Management - Case Study Collection

> **Mandatory authoring contract:** Every authored case must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. Evidence, implementation stage and editorial status remain separate. Drafts are not reviewed or published.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 |
| Review cadence | Quarterly and on material market/regulatory change |
| Index version | 2.0-planned |
| Canonical case-ID format | CASE-01-06-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-01-06-001 | [Dynamic retail pricing](01_Dynamic_Pricing_Ecommerce_Retail.md) | Pricing and Revenue Management | hybrid | Retain a fixed rule or test a bounded inventory-aware public price policy? | causal-confounding | evaluation-mismatch; model-monitoring | not-measured; illustrative | proposed | Not reviewed | Constructed example only; causal response and real outcomes unknown |
| PLAN-01-06-02 | Airline and hotel revenue management | Pricing and Revenue Management | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-06-03 | Insurance rating engine | Pricing and Revenue Management | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-06-04 | Rideshare surge pricing | Pricing and Revenue Management | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-06-05 | SaaS usage-based pricing | Pricing and Revenue Management | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |

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

| Pattern | Supporting planned cases | Theory/method links | Boundary to test |
|---|---|---|---|
| evaluation-mismatch | All planned cases | Relevant method and evaluation modules | Whether proxy metric changes decision quality |
| distribution-shift | Market, season, policy, or user changes | Forecasting, MLOps, and domain indexes | Population and operational conditions |
| human-in-the-loop | High-impact operational cases | Governance, UX, and domain workflows | Review capacity and override quality |
| causal-confounding | CASE-01-06-001; other cases planned | [Leakage and preprocessing](https://scikit-learn.org/stable/common_pitfalls.html) | Randomization, interference, and inventory differ by market |
| inventory horizon | CASE-01-06-001 versus planned airline and hotel case | Price and capacity decisions | Replenishable stock differs from perishable room or seat capacity |

## 5. Questions and future-case backlog

| Question / planned case | Evidence needed | Priority / owner |
|---|---|---|
| Does a permitted price change cause contribution uplift? | Approved cluster experiment and mature finance data | Before policy launch / pricing science |
| How much substitution crosses experimental clusters? | Shared-inventory mapping and exposure study | Before randomization / experimentation |
| What inventory horizon is appropriate? | Replenishment records and validated multi-day simulation | Before scarce-stock rollout / inventory |
| Are jurisdiction and customer safeguards adequate? | Legal determination and customer-journey review | Before exposure / legal and product |
| Airline and hotel revenue management; insurance rating; rideshare surge; SaaS usage pricing | Case-specific bounded decision, evidence and template-complete drafts | Planned / curriculum maintainer |

Before drafting another listed case, replace its PLAN ID with a CASE ID, add a relative file link, choose one primary pattern and no more than two secondary patterns, state evidence/stage independently, and set a concrete review trigger.

## 6. Review and retirement log

| Item | Editorial status | Technical / numerical / interview review | Reviewer / date | Trigger |
|---|---|---|---|---|
| CASE-01-06-001 (formerly PLAN-01-06-01) | Draft, unreviewed | Pending; no business experiment or implementation | None / not reviewed | Eligibility, price rules, experiment or finance definition changes |
| Remaining PLAN entries | Planned, not authored | Not applicable until drafted | None | Draft assignment or source change |

## 7. Pre-publish checks

- [ ] Case ID is unique and linked.
- [ ] Evidence maturity is distinct from implementation stage.
- [ ] Decision, metric, action, and outcome refer to the same population.
- [ ] Assumptions, constraints, failure modes, and a quick-reference card are present.
- [ ] Company-specific claims have primary/public evidence or are labelled illustrative.
- [ ] Local links resolve and the review trigger is actionable.