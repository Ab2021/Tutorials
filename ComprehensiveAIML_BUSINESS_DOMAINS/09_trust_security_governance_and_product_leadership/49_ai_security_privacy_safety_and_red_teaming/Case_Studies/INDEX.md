# Module 49 Case Studies â€” AI Security, Privacy, Safety and Red Teaming

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Agent data-exfiltration simulation | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | secure RAG design review | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR004 | Customize tenant assistants without crossing isolation boundaries | Supporting curriculum; shared brief | llm-agentic | How should tenant adapters be trained, selected, versioned and rolled back? | `authorization-scope` | `atomic-release-bundle`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR008 | Block assistant releases on auditable quality and safety gates | Supporting curriculum; shared brief | llm-agentic | Does a candidate pass the required suites with enough evidence to justify promotion? | `evaluation-mismatch` | `data-leakage`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR004 | [REF-IR004](../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements |
| PLAN-2026-IR008 | [REF-IR008](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md) | A judge score is not a deployment guarantee; workload volume and evaluation deadlines remain unspecified |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
