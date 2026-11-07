# Module 21 Case Studies â€” Fine-Tuning, Alignment and Model Customisation

## Authoring standard

Every authored case follows the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md). The authored case below is draft and unreviewed; its proposed intervention has no measured outcome or deployment. Other entries remain planning briefs.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | [Specialise a support model](01_support_domain_adaptation.md) | Compare adaptation with frozen and manual baselines using independent safety and quality gates. | draft; unreviewed |
| 2 | compact a domain model for edge | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR004 | Customize tenant assistants without crossing isolation boundaries | Proposed home | llm-agentic | How should tenant adapters be trained, selected, versioned and rolled back? | `authorization-scope` | `atomic-release-bundle`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR009 | Allocate a domain-training budget with restart headroom | Supporting curriculum; shared brief | hybrid | Should a domain model be trained from scratch or continued from an existing checkpoint within an all-in budget? | `cost-latency-tradeoff` | `data-leakage`, `atomic-release-bundle` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR010 | Compare specialized models with authorized shared-model baselines | Supporting curriculum; shared brief | llm-agentic | When do repository-aware completion or language adapters justify their added complexity? | `data-leakage` | `authorization-scope`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR017 | Adapt domain ASR while preserving supported-domain quality | Proposed home | hybrid | Should a speech adapter be promoted, and to which evaluated product and language domains? | `rollout-regression` | `distribution-shift`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR004 | [REF-IR004](../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements |
| PLAN-2026-IR009 | [REF-IR009](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md) | The reference compute and GPU figures are assumptions; an internal legal checkpoint is not validated legal advice |
| PLAN-2026-IR010 | [REF-IR010](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md) | Split this reference into two independent future narratives; code-test passage does not prove security or language accuracy |
| PLAN-2026-IR017 | [REF-IR017](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md) | Model-generated transcripts are not independent gold; audio resampling does not restore missing frequencies |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->

## 1. Collection metadata

Module 21 cases cover bounded model-customisation decisions and their operational consequences.
The collection owner is the curriculum maintainer; evidence, implementation stage and editorial status are independent.
The case below is illustrative and does not claim any real customer, training run or deployment.

## 2. Case catalog

| Case ID | Title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-2026-FT01 | [Should support draft generation use a domain adapter?](01_support_domain_adaptation.md) | Support model adaptation | llm-agentic | Whether to pilot an agent-only adapter | `evaluation-mismatch` | `data-leakage`, `atomic-release-bundle` | not-measured; illustrative sizing | proposed | never | No run, measured impact or deployment; draft, unreviewed |

## 3. Canonical reusable patterns

| Pattern | Meaning in this collection |
|---|---|
| `evaluation-mismatch` | A proxy or offline metric does not by itself establish the decision outcome. |
| `data-leakage` | Information unavailable at decision time or shared across partitions biases evaluation. |
| `atomic-release-bundle` | Model, tokenizer, configuration and evidence must be versioned and promoted together. |

## 4. Similarity and contrast map

| Case or brief | Relationship | Important difference |
|---|---|---|
| CASE-2026-FT01 | Adapted support drafts with independent evaluation | Single fictional workflow; human review and current policy are mandatory. |
| PLAN-2026-IR004 | Related adapter lifecycle | Tenant isolation is the central question there, not established by this case. |
| PLAN-2026-IR010 | Related comparison with a shared baseline | Its proposed repository and language settings differ from support drafting. |

## 5. Questions and future-case backlog

- What sample size would support an agreed severe-risk upper bound in an actual deployment?
- How should an actual business define the minimum economically useful improvement?
- The edge compact-model case remains planned; no edge deployment is implied.
- Reference-informed briefs above remain planned and are not claimed as covered cases.

## 6. Review and retirement log

| Item | State | Trigger or next review |
|---|---|---|
| CASE-2026-FT01 | Draft; no substantive review recorded | Review when sources, model, policy, rights, rubric or serving assumptions change. |
| Collection | Not published | Independently check numerical work, source access, links and all case gates. |

## 7. Pre-publish checks

- Verify the case's equations, arithmetic and applicable sources independently.
- Check that all 15 sections, interview answers and local links are complete.
- Confirm no illustrative quantity is represented as a measured or deployed result.
- Record reviewer, date and any unresolved limitations before changing editorial status.
