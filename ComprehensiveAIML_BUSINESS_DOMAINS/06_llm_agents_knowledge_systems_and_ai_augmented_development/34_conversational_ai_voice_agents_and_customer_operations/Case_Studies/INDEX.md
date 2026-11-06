# Module 34 Case Studies â€” Conversational AI, Voice Agents and Customer Operations

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Voice scheduling assistant | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | contact-centre QA automation | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR015 | Publish reviewed coaching scripts from current product evidence | Proposed home | llm-agentic | Which generated scripts are fit for an approved, versioned coaching registry? | `human-in-the-loop` | `eligibility-before-ranking`, `atomic-release-bundle` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR016 | Distinguish speech, speaker and extraction errors in call analytics | Supporting curriculum; shared brief | hybrid | Which customer objections can be reported with reliable speaker attribution and evidence? | `evaluation-mismatch` | `authorization-scope`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR017 | Adapt domain ASR while preserving supported-domain quality | Supporting curriculum; shared brief | hybrid | Should a speech adapter be promoted, and to which evaluated product and language domains? | `rollout-regression` | `distribution-shift`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR015 | [REF-IR015](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md) | Human approval and source maintenance remain necessary; the example word target is not a compliance standard |
| PLAN-2026-IR016 | [REF-IR016](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md) | Transcript matching does not prove audio fidelity; aggregate coaching themes must not become unvalidated individual scores |
| PLAN-2026-IR017 | [REF-IR017](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md) | Model-generated transcripts are not independent gold; audio resampling does not restore missing frequencies |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
