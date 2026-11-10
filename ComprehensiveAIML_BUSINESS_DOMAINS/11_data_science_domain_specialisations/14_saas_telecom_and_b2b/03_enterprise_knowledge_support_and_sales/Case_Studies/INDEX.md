# Enterprise Knowledge, Support and Sales - Case Study Collection

> **Mandatory authoring contract:** Every authored case must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 |
| Review cadence | Quarterly and on material market/regulatory change |
| Index version | 2.0-planned |
| Canonical case-ID format | CASE-14-03-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-14-03-01 | Permission-aware support agent | Enterprise Knowledge, Support and Sales | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-14-03-02 | Sales call intelligence | Enterprise Knowledge, Support and Sales | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-14-03-03 | Contract review | Enterprise Knowledge, Support and Sales | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-14-03-04 | Ticket routing | Enterprise Knowledge, Support and Sales | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| PLAN-14-03-05 | Knowledge freshness incident | Enterprise Knowledge, Support and Sales | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |

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
| PLAN-2026-IR003 | Answer support questions from the applicable policy version | Supporting curriculum; shared brief | llm-agentic | When should a support assistant answer, abstain or escalate a policy question? | `evaluation-mismatch` | `authorization-scope`, `materialized-view` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR004 | Customize tenant assistants without crossing isolation boundaries | Supporting curriculum; shared brief | llm-agentic | How should tenant adapters be trained, selected, versioned and rolled back? | `authorization-scope` | `atomic-release-bundle`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR012 | Preserve answer evidence through ingestion and filtered retrieval | Supporting curriculum; shared brief | llm-agentic | Which chunking and index design retrieves every required authorized span within the service budget? | `materialized-view` | `authorization-scope`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR015 | Publish reviewed coaching scripts from current product evidence | Supporting curriculum; shared brief | llm-agentic | Which generated scripts are fit for an approved, versioned coaching registry? | `human-in-the-loop` | `eligibility-before-ranking`, `atomic-release-bundle` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR003 | [REF-IR003](../../../../../INTERVIEW_CASES/Agentic_AI_System/Applied_AI_engineering/grounded_policy_rag.md) | Citations and entailment checks require validation; never transfer the example refund window into a new business |
| PLAN-2026-IR004 | [REF-IR004](../../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements |
| PLAN-2026-IR012 | [REF-IR012](../../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md) | Reranking cannot recover a missing candidate; the source embedding limit and tenant count are not platform requirements |
| PLAN-2026-IR015 | [REF-IR015](../../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md) | Human approval and source maintenance remain necessary; the example word target is not a compliance standard |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->

## 7. Review and retirement log

| Date | Scope | Evidence observed | Editorial decision | Trigger for recheck |
|---|---|---|---|---|
| 2026-11-09 | Existing catalog and reference-plan rows | Rows are planning briefs; source examples do not establish local deployments, measurements, or authored cases. | Retain all rows as `PLAN`; do not assign `CASE` IDs or claim results. | A template-complete local case, primary evidence, or a changed source-reference boundary. |
