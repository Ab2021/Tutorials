# Module 18 Case Studies â€” NLP, Speech and Document Intelligence

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Claims document extraction | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | call-centre transcription quality | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR010 | Compare specialized models with authorized shared-model baselines | Supporting curriculum; shared brief | llm-agentic | When do repository-aware completion or language adapters justify their added complexity? | `data-leakage` | `authorization-scope`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR011 | Compare document VLMs with OCR for adjuster assistance | Supporting curriculum; shared brief | hybrid | Which claim fields can be extracted reliably, and which bundles require adjuster review? | `human-in-the-loop` | `evaluation-mismatch`, `delayed-selective-labels` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR016 | Distinguish speech, speaker and extraction errors in call analytics | Proposed home | hybrid | Which customer objections can be reported with reliable speaker attribution and evidence? | `evaluation-mismatch` | `authorization-scope`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR017 | Adapt domain ASR while preserving supported-domain quality | Supporting curriculum; shared brief | hybrid | Should a speech adapter be promoted, and to which evaluated product and language domains? | `rollout-regression` | `distribution-shift`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR010 | [REF-IR010](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md) | Split this reference into two independent future narratives; code-test passage does not prove security or language accuracy |
| PLAN-2026-IR011 | [REF-IR011](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Insurance_Document_VLM.md) | Neither a similarity match nor a VLM fraud suspicion justifies denial; source scale and patch arithmetic need fresh sizing |
| PLAN-2026-IR016 | [REF-IR016](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md) | Transcript matching does not prove audio fidelity; aggregate coaching themes must not become unvalidated individual scores |
| PLAN-2026-IR017 | [REF-IR017](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md) | Model-generated transcripts are not independent gold; audio resampling does not restore missing frequencies |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
