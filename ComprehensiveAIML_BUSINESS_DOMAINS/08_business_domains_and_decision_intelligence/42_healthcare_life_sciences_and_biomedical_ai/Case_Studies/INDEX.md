# Module 42 Case Studies — Healthcare, Life Sciences and Biomedical AI

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Module 42 curriculum maintainer. |
| Audience | Learners designing bounded healthcare, life-sciences, and biomedical-AI decisions without converting planned work into deployment claims. |
| Domain boundary | Clinical documentation, validation, workflow, evidence, safety, privacy, and operational decision design; not medical advice, clinical protocol, legal determination, or implementation certification. |
| Index version | 1.0. |
| Last reviewed | 2026-11-10 for the authored CASE-M42-001 draft and retained planning row. |
| Review cadence | Recheck on source-body access, local workflow/policy evidence, material template revision, or new case publication. |
| Canonical case-ID format | `CASE-M42-NNN` for authored cases; `PLAN-M42-NNN` for planned briefs. |
| Source of truth | This `INDEX.md` is the canonical collection index. |
| Publication-state convention | `content status` is editorial; `evidence` and `stage` describe the case intervention separately. |
| Canonical active patterns | `draft-only-human-approval`, `time-bounded-provenance`, `abstention-with-owned-fallback`, `validation-plan-with-adjudication`, `manual-baseline-comparator`, `permission-bounded-workflow`. |
| Review triggers | Change to intended use, source authorization, patient/session boundary, data provenance, model/prompt, write permission, reviewer workflow, validation protocol, or source availability. |

The visible source inventory and curriculum briefs establish topics to teach.

They do not establish a deployment, outcome, clinical validity, legal compliance, or local policy.

Evidence, implementation stage, and editorial status are deliberately separate in the catalog.

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-M42-001 | [Clinical Documentation Assistant: Draft Evidence, Review Capacity, and the Write Boundary](01_clinical_documentation_assistant.md) | Healthcare clinical documentation | llm-agentic | Whether to authorize only a draft-only evaluation with clinician approval and no autonomous record/care/order action, versus retain manual documentation. | draft-only-human-approval | time-bounded-provenance; abstention-with-owned-fallback | illustrative | proposed | not reviewed | Draft manuscript; constructed workload arithmetic and state traces only. Local workflow, source body, policy, implementation, and outcome evidence remain OPEN. |
| PLAN-M42-002 | ECG model validation plan (planned; no file yet) | Biomedical signal validation | predictive-ml | Design the decision, architecture, metrics, risks, and validation plan for an ECG model without claiming clinical deployment or benefit. | validation-plan-with-adjudication | manual-baseline-comparator; permission-bounded-workflow | not-measured | proposed | not reviewed | Original planned row retained. Source inputs, reference standard, implementation, and results remain unresolved. |

| Case ID | Editorial status | Source/reference status | Next editorial gate |
|---|---|---|---|
| CASE-M42-001 | draft | Source inventory and curriculum brief consulted; original source body and local evidence OPEN. | Independent editorial and source review; reconcile any supplied local policy before any implementation wording. |
| PLAN-M42-002 | planned | No authored case or linked source plan is available in this collection. | Preserve plan until a bounded authored case has its own evidence and validation record. |

## 3. Canonical reusable patterns

| Pattern | Definition and mechanism | Diagnostic signature | Preconditions | Counterexample / avoid condition | Related cases | Theory or method link | Deprecated synonyms |
|---|---|---|---|---|---|---|---|
| draft-only-human-approval | Generate a non-binding draft outside the system of record; an accountable human reviews, edits, rejects, or discards before using an independent write path. | A generated artifact exists but assistant-originated record writes remain zero. | Named reviewer, manual fallback, separate write permission, auditable draft version. | Avoid when no reviewer capacity or the assistant can mutate consequential records. | CASE-M42-001. | [Clinical Decision Support](../Cheat_Sheets/01_Clinical_Decision_Support_Evidence_and_Handoff.md). | “human in the loop” when it lacks explicit authority or action boundary. |
| time-bounded-provenance | Freeze authorized inputs at a declared decision/request cutoff and retain source identity/version/access basis for each output. | Every reviewed claim can be traced to an allowed source manifest; later data does not appear in earlier draft. | Stable identity, source availability timestamps, immutable manifest. | Avoid claims of time-valid evaluation when source availability is unknown. | CASE-M42-001. | [Clinical Decision Support](../Cheat_Sheets/01_Clinical_Decision_Support_Evidence_and_Handoff.md). | “latest chart context.” |
| abstention-with-owned-fallback | Withhold an output when required source, scope, identity, or confidence condition fails, then route work to a named baseline owner. | Abstentions have reason, owner, terminal disposition, and remain in workload denominator. | Manual path and visible reason exist. | Avoid silent blank/negative output with no ownership. | CASE-M42-001. | [Human Factors](../Cheat_Sheets/03_Human_Factors_Alert_Design_and_Recovery.md). | “no result” without routing. |
| validation-plan-with-adjudication | Define reference criteria, sampling, reviewer rubric, disagreement handling, and decision gates before interpreting performance. | Quality rate has explicit numerator, denominator, rubric, and adjudicated status. | Reference source, independent review capacity, versioned protocol. | Avoid when acceptance clicks are treated as ground truth. | PLAN-M42-002; CASE-M42-001 uses the pattern for draft quality. | Clinical decision-support evidence sheet. | “validated” based on one aggregate score. |
| manual-baseline-comparator | Compare a proposed workflow with the actual current manual/rule process under matched scope and denominators. | Proposed time or quality includes abstentions, corrections, rejections, and pending work. | Measurable baseline and comparable task boundary. | Avoid comparison with imagined manual work or denominator changes. | CASE-M42-001; PLAN-M42-002. | Case master evaluation guidance. | “before/after improvement” without comparator design. |
| permission-bounded-workflow | Make read scope, write scope, actors, and forbidden actions explicit and testable. | Negative permission test verifies a prohibited action cannot occur. | Identity, authorization, audit events, disable control. | Avoid broad delegated tokens or implied approval. | CASE-M42-001; PLAN-M42-002 planned. | Human-factors recovery sheet. | “read-only” without endpoint and credential review. |

A pattern is reusable only when its preconditions and counterexample remain visible.

Pattern tags are mechanisms, not vendor names, hospital names, or claims of completed deployments.

## 4. Cross-case similarity and contrast map

| Pattern | Supporting case(s) | Contrasting case(s) | Theory/methodology link | Boundary learned |
|---|---|---|---|---|
| draft-only-human-approval | CASE-M42-001. | Planned ECG validation plan differs because its core output is a validation decision rather than drafted prose. | Clinical decision-support handoff concepts. | Human approval must be an explicit action and separate from system write authority. |
| time-bounded-provenance | CASE-M42-001. | Planned ECG validation will need signal-acquisition and annotation time boundaries rather than note-source spans. | Clinical decision-support time boundary; biomedical signal pipeline. | Event time alone does not establish permitted availability. |
| abstention-with-owned-fallback | CASE-M42-001. | ECG plan may use indeterminate/quality-reject routing rather than note-draft abstention. | Human-factors recovery lifecycle. | An unserved item must have an owner and a terminal route. |
| validation-plan-with-adjudication | CASE-M42-001 design and PLAN-M42-002 planned. | Documentation adjudicates factual/text categories; ECG validation will require a separately defined reference standard. | Clinical decision-support evidence and handoff. | A rubric/label is not interchangeable across domains. |
| manual-baseline-comparator | CASE-M42-001 and PLAN-M42-002 planned. | Documentation compares authoring burden; ECG plan will compare validation/interpretation workflow, not note time. | Case master evaluation design. | Same comparator pattern, different unit and guardrails. |
| permission-bounded-workflow | CASE-M42-001. | ECG plan should not be assumed to share record-write or order interfaces. | Human-factors ownership and recovery. | Do not import an interface or authority model from another case. |

The ECG comparison remains a planned learning contrast, not empirical support.

No cross-case performance conclusion is available from the catalog.

## 5. Questions and future-case backlog

| Open question | Originating case | Needed evidence | Proposed next case or experiment | Priority | Owner role | Trigger / review date |
|---|---|---|---|---|---|---|
| Which documentation note types have source-complete, encounter-scoped inputs and a meaningful manual fallback? | CASE-M42-001. | Local workflow map, source inventory, access and template configuration. | Offline snapshot-contract review before any draft evaluation. | High. | Clinical documentation governance lead and data owner. | Before any implementation request. |
| What is actual manual documentation time, correction burden, and reviewer capacity by note type? | CASE-M42-001. | Comparable timing protocol and observed task data. | Paired/manual workflow measurement proposal. | High. | Quality lead. | Before a time-saving claim. |
| Can factual omission, attribution, negation, medication, and unit errors be adjudicated reliably? | CASE-M42-001. | Rubric, double review, disagreement record, adjudication protocol. | Annotation calibration exercise. | High. | Quality lead. | Before reporting quality rate. |
| How should a proposed system prove no assistant-originated record writes or cross-session leaks? | CASE-M42-001. | Permission inventory, negative tests, request/session audit trace. | Design-review security test plan. | High. | Application and security owners. | Before any connected prototype. |
| What ECG population, reference standard, acquisition constraints, and failure categories should govern validation? | PLAN-M42-002. | Bounded planning brief and primary/local source evidence. | Author the retained ECG validation case without copying documentation assumptions. | High. | Biomedical validation lead. | When source material is available. |
| What local governing artifacts define retention, reviewer authorization, and source access? | CASE-M42-001. | Authoritative local policy/artifact, version, scope, and review date. | Evidence reconciliation, not a generic compliance claim. | High. | Privacy/security and governance owners. | When artifacts are supplied or change. |

Backlog entries are questions that can change a decision.

They are not claims that an experiment, policy review, or system already exists.

## 6. Review and retirement log

| Case ID | Date | What changed | Evidence or market trigger | Action | Owner role |
|---|---|---|---|---|---|
| CASE-M42-001 | 2026-11-10 | Authored initial draft-only documentation-assistant case and registered stable case ID. | Curriculum authoring requirement; source body unavailable. | Marked editorial status draft, intervention proposed, evidence illustrative; retained OPEN evidence limits. | Module 42 curriculum maintainer. |
| PLAN-M42-002 | 2026-11-10 | Expanded catalog around original planned ECG validation row without authoring an unrequested case. | Collection-index structure requirement. | Retained as planned with no completion claim. | Module 42 curriculum maintainer. |

No case is retired or superseded.

A future retirement entry must retain the stable ID, reason, date, owner, and successor link where one exists.

## 7. Pre-publish index checks

- [x] Unique IDs distinguish the authored `CASE-M42-001` draft from retained `PLAN-M42-002`.
- [x] The authored case link resolves to the canonical draft manuscript.
- [x] The original ECG validation plan remains visible and is not claimed complete.
- [x] Case type and pattern tags describe the actual draft-only mechanism rather than a blanket default.
- [x] Evidence (`illustrative`/`not-measured`), stage (`proposed`), and editorial status (`draft`/`planned`) remain separate.
- [x] The authored decision specifies owner, baseline, source/time boundary, write boundary, and excluded actions.
- [x] Catalog limitation text does not claim medical efficacy, legal compliance, deployed outcome, or local policy approval.
- [x] Existing links target authored local manuscript, local sheets, or the course template; planned ECG case has no fabricated link.
- [x] Review log dates describe index/editorial action, not an independent substantive case review.
- [ ] Independent source review is OPEN because the original scenario body and local governing artifacts are unavailable.
- [ ] Any implementation claim is blocked pending source/access contract, negative permission tests, reviewer rubric/adjudication, and actual evaluation evidence.
