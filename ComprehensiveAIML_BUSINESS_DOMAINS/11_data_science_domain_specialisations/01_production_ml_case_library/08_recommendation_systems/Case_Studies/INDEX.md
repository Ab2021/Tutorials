# Recommendation Systems - Case Study Collection

> **Mandatory authoring contract:** Every authored case follows the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master. This collection contains draft content and planned cases; evidence, implementation stage and editorial status remain separate. Publication checks remain open.

The Netflix and Spotify drafts have been expanded and remain unreviewed; published claims and hypothetical analyses are distinguished in each case.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 |
| Review cadence | Quarterly and on material market/regulatory change |
| Index version | 2.1-draft |
| Canonical case-ID format | CASE-01-08-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-01-08-001 | [Netflix GenRec versus a mature production ranker](01_Netflix_GenRec_Ranking.md) | Media recommendation ranking | predictive-ml | Assess a catalog-aware LLM ranker against a mature baseline | evaluation-mismatch | cost-latency-tradeoff; distribution-shift | published | pilot | not reviewed | Netflix reports significant online gains in an approximately four-week A/B evaluation on batch surfaces; magnitude, metric definitions and general rollout are not established; editorial status: draft |
| CASE-01-08-002 | [Spotify personalization versus experimentation](02_Spotify_Personalization_Experimentation.md) | Music personalization | predictive-ml | Separate recommendation serving from causal system evaluation | evaluation-mismatch | label-delay; cost-latency-tradeoff | published | production | not reviewed | Spotify describes separate stacks and experiment scale, not a measured lift; editorial status: draft |
| PLAN-01-08-01 | Video candidate-generation and ranking | Recommendation Systems | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-08-02 | E-commerce product recommendations | Recommendation Systems | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-08-03 | Music session recommendation | Recommendation Systems | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-08-04 | News-feed ranking | Recommendation Systems | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-08-05 | Ads recommendation ranking | Recommendation Systems | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |

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
| evaluation-mismatch | CASE-01-08-001; planned cases | Ranking and online experimentation | Offline MRR versus member-level outcomes |
| distribution-shift | CASE-01-08-001; market, season, policy, or user changes | Forecasting, MLOps, and domain indexes | Freshness, population and operational conditions |
| cost-latency-tradeoff | CASE-01-08-001 | Model serving and context engineering | Compaction savings depend on the whole workload |
| human-in-the-loop | High-impact operational cases | Governance, UX, and domain workflows | Review capacity and override quality |

## 5. Questions and future-case backlog

The authored GenRec case is a separately sourced case in the video recommendation family; PLAN-01-08-01 remains open because its broader candidate-generation versus popularity-baseline decision has not been authored.
Remaining planned cases need distinct decisions, evidence and baselines.
CASE-01-08-002 addresses stack separation, not the session-model decision in PLAN-01-08-03, which remains open. Its unresolved questions include assignment mechanics, effect sizes, model specifications, and safeguards.
Open GenRec questions include online metric definitions, randomization, precise effect sizes and any subsequent rollout.

## 6. Review and retirement log

| Date | Entry | Editorial consequence |
|---|---|---|
| 2026-11-07 | Added CASE-01-08-001 from the 30 July 2026 Netflix primary source. | Draft; independent review and publication checks remain open. |
| 2026-11-07 | Added CASE-01-08-002 from Spotify Engineering's January 2026 article. | Unreviewed draft; independent review and publication checks remain open. |

## 7. Pre-publish checks

- [ ] Case ID is unique and linked.
- [ ] Evidence maturity is distinct from implementation stage.
- [ ] Decision, metric, action, and outcome refer to the same population.
- [ ] Assumptions, constraints, failure modes, and a quick-reference card are present.
- [ ] Company-specific claims have primary/public evidence or are labelled illustrative.
- [ ] Local links resolve and the review trigger is actionable.