# Module 25 Case Studies â€” Model Serving, Inference Optimisation and Cost

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Multi-model inference gateway | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | reduce p99 LLM latency | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR002 | Select model routes within hard customer constraints | Proposed home | llm-agentic | Which eligible provider and fallback path satisfy residency, capability, deadline and spend requirements? | `eligibility-before-ranking` | `cost-latency-tradeoff` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR004 | Customize tenant assistants without crossing isolation boundaries | Supporting curriculum; shared brief | llm-agentic | How should tenant adapters be trained, selected, versioned and rolled back? | `authorization-scope` | `atomic-release-bundle`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR007 | Balance interactive inference latency and admitted throughput | Proposed home | hybrid | Which scheduling and KV-memory policy meets latency and isolation requirements at the admitted load? | `resource-admission` | `cost-latency-tradeoff`, `authorization-scope` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR014 | Separate extraction correctness from routing economics | Supporting curriculum; shared brief | llm-agentic | How should extraction contracts and model routes be accepted without confusing valid syntax with business success? | `evaluation-mismatch` | `idempotent-side-effects`, `cost-latency-tradeoff` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR002 | [REF-IR002](../../../../INTERVIEW_CASES/Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md) | Revalidate provider capabilities and economics when authoring; source tables are not current service guarantees |
| PLAN-2026-IR004 | [REF-IR004](../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements |
| PLAN-2026-IR007 | [REF-IR007](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_Inference/continuous_batching_serving.md) | GPU geometry, concurrency limits and speedups require workload measurements; speculative decoding can be slower |
| PLAN-2026-IR014 | [REF-IR014](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md) | Contains two reference problems: author separate narratives; valid JSON and cheap token prices do not establish successful outcomes |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
