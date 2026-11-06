# NLP, LLM and Document Intelligence - Case Study Collection

> **Authoring contract:** Follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md). Case evidence, implementation stage and editorial status are separate. The linked RAG case is an unreviewed draft; remaining entries are plans, not completed cases.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 (planning catalog only; the new draft is not substantively reviewed) |
| Review cadence | Quarterly and on material market, security, source or policy change |
| Index version | 2.1-draft |
| Canonical case-ID format | CASE-01-05-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-01-05-001 | [Authorized policy answers with an enterprise knowledge assistant](01_Enterprise_RAG_Knowledge_Assistant.md) | Enterprise policy knowledge | llm-agentic | Whether cited, authorized assistance merits a controlled pilot against manual portal search | authorization-aware-retrieval | evaluation-mismatch; human-in-the-loop | illustrative | proposed | not reviewed | No implementation or outcome measured; editorial status: draft. Replaces PLAN-01-05-01. |
| PLAN-01-05-02 | LLM content moderation at scale | NLP, LLM and Document Intelligence | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-05-03 | E-commerce semantic search | NLP, LLM and Document Intelligence | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-05-04 | Document entity-extraction pipeline | NLP, LLM and Document Intelligence | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-01-05-05 | Support-ticket summarisation | NLP, LLM and Document Intelligence | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |

## 3. Canonical reusable patterns

| Tag | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| authorization-aware-retrieval | Relevant-looking passages may be inaccessible, expired, or insufficient | User-specific knowledge retrieval with authoritative permissions and versioned sources | Not a substitute for request-time access checks, source authority, or evidence entailment; see CASE-01-05-001 |
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
| authorization-aware-retrieval | CASE-01-05-001 (illustrative draft) | PLAN-01-05-03 semantic search (planned) | [NLP/LLM topic home](../INDEX.md) | Employee-specific access checks are not necessarily the same as public catalog search. |
| evaluation-mismatch | CASE-01-05-001 (illustrative draft) | PLAN-01-05-02 moderation (planned) | [Source coverage brief](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-cases-05) | Retrieval coverage, moderation accuracy and employee task outcomes answer different questions. |
| human-in-the-loop | CASE-01-05-001 (proposed escalation) | PLAN-01-05-04 extraction (planned) | [Course standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) | Policy-owner interpretation differs from review of extracted fields. |

## 5. Questions and future-case backlog

| Open question | Originating case | Needed evidence | Proposed next case or experiment | Priority | Owner role | Trigger / review date |
|---|---|---|---|---|---|---|
| Does hybrid retrieval improve supported answers over manual search? | CASE-01-05-001 | Frozen role-aware labels and paired comparison | Offline evaluation and approved task study | High | Evaluation lead | Before pilot approval |
| Can current authorization be enforced during revocation? | CASE-01-05-001 | Entitlement semantics and adversarial revocation fixtures | Security assessment | High | Security owner | Before production data access |
| Which patterns distinguish moderation from policy answering? | PLAN-01-05-02 | Moderation scope, data and error costs | Bounded moderation case | Medium | Curriculum maintainer | When authored |

## 6. Review and retirement log

| Case ID | Date | What changed | Evidence or market trigger | Action | Owner role |
|---|---|---|---|---|---|
| CASE-01-05-001 | 2026-11-07 | Draft authored from the PLAN-01-05-01 topic; no substantive review claimed | Source crosswalk and authoring contract | Await independent editorial and technical review | Curriculum maintainer |

## 7. Pre-publish checks

Publication remains pending; the following checks have not been independently completed.

- [ ] Case ID is unique and linked.
- [ ] Evidence maturity is distinct from implementation stage.
- [ ] Decision, metric, action, and outcome refer to the same population.
- [ ] Assumptions, constraints, failure modes, and a quick-reference card are present.
- [ ] Company-specific claims have primary/public evidence or are labelled illustrative.
- [ ] Local links resolve and the review trigger is actionable.