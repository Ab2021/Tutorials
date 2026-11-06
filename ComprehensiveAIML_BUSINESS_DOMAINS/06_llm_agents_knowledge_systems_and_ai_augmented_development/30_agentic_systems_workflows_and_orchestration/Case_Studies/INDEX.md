# Module 30 Case Studies â€” Agentic Systems, Workflows and Orchestration

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Claims operations agent | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | research workflow with approval gates | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR005 | Release and recover an agent as a compatible bundle | Supporting curriculum; shared brief | hybrid | Which model, prompt, tools, index and policy versions can be promoted and recovered together? | `atomic-release-bundle` | `rollout-regression`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR006 | Recover procurement workflows without replaying approvals | Proposed home | llm-agentic | How can a purchase proceed only after approval of its exact version despite retries and restarts? | `idempotent-side-effects` | `human-in-the-loop`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR005 | [REF-IR005](../../../../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md) | A prompt-only rollback cannot restore all dependencies; example dwell times and canary percentages are not defaults |
| PLAN-2026-IR006 | [REF-IR006](../../../../INTERVIEW_CASES/Agentic_AI_System/Langraph/durable_human_approval_graph.md) | External APIs may not support exactly-once effects; approval waiting times and run counts are exercise assumptions |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
