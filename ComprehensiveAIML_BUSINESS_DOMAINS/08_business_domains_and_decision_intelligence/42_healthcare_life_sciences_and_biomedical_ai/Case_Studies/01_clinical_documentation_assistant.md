---
case_id: "CASE-M42-001"
case_title: "Draft-Only Clinical Documentation Assistant with Clinician Approval"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Healthcare — clinical documentation workflow"
primary_pattern: "draft-only-human-approval"
secondary_patterns: ["time-bounded-provenance", "abstention-with-owned-fallback"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Clinical documentation governance lead"
source_index: "INDEX.md"
review_trigger: "Any change to note scope, source access, model/prompt version, write permission, reviewer workflow, or local governance artifact"
---

# Clinical Documentation Assistant: Draft Evidence, Review Capacity, and the Write Boundary

## 1. The One-Line Framing

> Given one completed encounter and its authorized, time-bounded source artifacts, produce a draft note with claim-level provenance for a clinician to review, edit, approve, or discard, subject to no autonomous care, order, problem-list, medication, or record-write action, compared with manual documentation from the same encounter.

The unit of work is one encounter-note draft, not a patient lifetime, a diagnosis, or an order.

The output is a non-binding draft in a review workspace.

The beneficiary is the clinician who remains accountable for the signed note.

The owner of the decision is the clinical documentation governance lead.

The operational question is whether a proposed draft-only workflow merits a tightly bounded evaluation.

The primary success condition is a lower estimated documentation time per eligible encounter without an unacceptable review burden or unresolved unsafe draft.

This is an illustrative design exercise.

All workload values are constructed teaching inputs.

No named organization, clinic, clinician, model vendor, or deployment is described.

No medical efficacy, legal compliance, billing validity, safety outcome, or realized time saving is claimed.

The baseline is manual documentation in the existing authoring workflow.

Manual documentation remains the fallback for every excluded, abstained, unavailable, or rejected draft.

The proposed system may read only explicitly authorized encounter-scoped source snapshots.

The proposed system may create a draft artifact outside the clinical record.

The proposed system may not submit a signed note.

The proposed system may not create, modify, discontinue, or route an order.

The proposed system may not alter medications, diagnoses, allergies, problem lists, or care plans.

The proposed system may not send patient-facing communication.

The proposed system may not infer a fact when the evidence is absent.

The clinician may accept, edit, reject, or discard a draft.

Only the clinician uses the existing record authoring workflow to write an approved note.

The primary reusable pattern is `draft-only-human-approval`.

The secondary patterns are `time-bounded-provenance` and `abstention-with-owned-fallback`.

The simpler framing, “summarize the encounter,” is insufficient.

A summary can silently omit a material fact, add an unsupported fact, flip a negation, misattribute a statement, or alter a medication unit.

Those failures change the review task even when prose reads fluently.

The actual design problem is therefore evidence-bounded drafting plus accountable review.

**Evidence declaration:** illustrative workflow and illustrative arithmetic; source-body and local workflow evidence are OPEN.

**Claim CLM-01:** under stated illustrative inputs, net time is estimated rather than observed.

**Claim CLM-02:** any proposed time benefit is conditional on clinician review, correction, abstention, and unsafe-completion accounting.

**Claim CLM-03:** the approval boundary is a system invariant, not a model-quality metric.

## 2. Context & Background

Clinical documentation often requires a clinician to assemble a coherent note from an encounter and its authorized artifacts.

The note may contain history, observations, assessment language, medication discussion, plan language, and attribution to speakers or sources.

This case does not define what a clinically appropriate note contains.

That content belongs to local professional practice and approved workflow.

The case instead defines the boundaries of a proposed drafting aid.

Before the proposed aid, the clinician reviews source materials and authors the note manually.

The clinician may use templates, dictation, copy-forward, or existing editor aids in the baseline.

Those baseline variants must be measured rather than assumed uniform.

The proposed workflow begins only after an encounter is identified and eligible under a local configuration.

An encounter coordinator or workflow integration provides the encounter identifier and requested note type.

A policy gate resolves whether the requesting clinician is authorized for that encounter and note type.

A snapshot builder selects allowed source artifacts with an explicit cutoff time.

A draft generator receives only that snapshot and a versioned drafting instruction.

A claim extractor associates generated clauses with source spans or labels them unsupported.

A review workspace displays the draft, source evidence, omissions checklist, uncertainty, and action buttons.

The clinician edits or discards the draft.

The existing record system receives content only through the clinician-controlled authoring path.

The case treats that existing path as outside the assistant's write permission.

The post-review audit records draft version, source snapshot, reviewer disposition, and permitted quality labels.

It does not treat a draft as an order or care recommendation.

Stakeholders include the clinician author, clinical documentation governance lead, privacy/security owner, application owner, reviewer-quality lead, and data engineering owner.

The clinician author owns the content decision for a particular note.

The governance lead owns intended-use boundaries and stop decisions for the proposal.

The application owner owns availability of the draft workspace and the disable switch.

The data owner owns source authorization, lineage, and snapshot contracts.

The quality lead owns annotation rubrics and adjudication procedures.

The security owner owns access-control review and incident handling.

The baseline persists because it has known ownership and does not depend on a generative component.

The baseline can still contain errors or take time.

That does not make an unvalidated draft acceptable.

A proposed aid must be compared with the actual manual process, not an imagined error-free or impossibly slow manual process.

The relevant timing boundary is note drafting after an encounter, not real-time decision support during care.

A source fact is eligible only if it is in the approved snapshot before draft generation.

A later correction, later lab result, later signature, or later note is not silently imported into an earlier draft.

If the note workflow requires a later source, the request must be re-created with a new cutoff.

The geographical, language, specialty, and patient-population scopes are unknown for this illustrative exercise.

No claim generalizes across them.

No source establishes a local retention rule, role policy, evaluation threshold, staffing norm, or authorization practice.

The visible source inventory lists a clinical decision-support topic.

Its underlying source body was unavailable at authoring time.

The source inventory establishes topic provenance only.

It does not establish that this workflow existed or worked in practice.

The local clinical decision-support cheat sheet supplies reusable time-boundary and handoff concepts.

It does not supply evidence of performance for this proposed assistant.

The human-factors sheet supplies an alert/recovery vocabulary.

It does not supply a validated reviewer workload threshold.

A documentation draft is not automatically low risk because it is not an order.

Text may be copied into a record and may affect later interpretation.

Therefore provenance and review must make uncertainty and changes visible.

The case avoids calling a draft “safe.”

Instead it names testable failure modes and a containment route.

**Workflow invariant W-01:** no generated content becomes a record write through an assistant credential.

**Workflow invariant W-02:** every displayed claim is either linked to an allowed source span or visibly marked unsupported.

**Workflow invariant W-03:** an abstained request has a named manual fallback, not an empty draft that appears complete.

**Workflow invariant W-04:** a clinician's approval is distinct from viewing, downloading, or copying a draft.

## 3. The Business Question

| Decision contract field | Bounded definition |
|---|---|
| Question | Should the governance lead authorize an illustrative, draft-only evaluation for eligible encounters, or retain manual documentation until provenance, review capacity, and recovery gates are satisfied? |
| Decision owner | Clinical documentation governance lead, with clinician and application-owner input. |
| Trigger and deadline | A proposed drafting capability is available for design review; no operational deadline is asserted. |
| Available actions | Retain manual baseline; run offline replay only; run a clinician-reviewed simulation; authorize a bounded proposed pilot after gates; or stop the proposal. |
| Value | Potential reduction in clinician authoring time and more structured source review, both unmeasured. |
| Downside | Unsupported, omitted, negated, misattributed, wrong-unit, stale, or leaked content may consume review time or be copied into a note. |
| Objective | Minimize estimated clinician minutes per eligible encounter while keeping unsafe completion, leakage, and write-boundary violations at zero in executed checks. |
| Hard constraints | No autonomous clinical action; no assistant record write; clinician approval before any record content; encounter-scoped access; auditable provenance; owned abstention. |
| Scope | Draft support for one eligible encounter note at a time under a defined source cutoff and reviewer session. |
| Out of scope | Diagnosis, triage, treatment selection, prescribing, orders, billing, coding, patient messaging, autonomous chart updates, and legal/compliance determination. |

The stakeholder question is paraphrased because no stakeholder interview artifact is available.

“Can we reduce time” is not sufficient without a denominator.

The denominator is eligible encounter-note requests in a stated window.

The numerator for manual time is clinician minutes spent authoring and reviewing under the baseline measurement protocol.

The numerator for proposed net time includes source preparation attributable to the workflow, draft review, correction, rejection handling, and manual fallback for abstentions.

The objective is not maximize words generated.

The objective is not maximize draft acceptance.

The objective is not minimize model latency alone.

The objective is not maximize note length.

A high acceptance rate could conceal rushed review.

A low correction count could reflect missing annotation rather than factual correctness.

A short draft could save time by omitting relevant content.

A long draft could shift work from authoring to deletion.

The decision gate therefore requires both workload and content-quality measures.

The decision is reversible only before the assistant is granted any expanded integration permission.

The proposed stage has no expanded permission.

The governance lead may stop evaluation if a forbidden-write attempt, patient/session leakage, or unresolved severe factual category occurs.

The application owner must be able to disable draft generation without disabling the baseline record workflow.

The clinician retains a manual path even during any proposed evaluation.

The manual path is the no-change comparator.

The decision should remain “do not proceed” if the evaluation cannot establish source lineage and review ownership.

The decision should remain “do not proceed” if reviewer capacity is modeled only by average time without correction and fallback time.

The decision should remain “do not proceed” if abstentions are excluded from the reported time denominator.

The decision should remain “do not proceed” if unsafe completed drafts have no independent adjudication.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Every source span has encounter identity, availability time, source system, and version. | Data | Snapshot replay and schema inspection. | Unverified | Provenance cannot be shown. | Do not generate; manual baseline. | Source-schema change / data owner. |
| A-02 | The requester has a current relationship and permitted role for the encounter. | Access | Authorization test matrix. | Unverified | Inappropriate exposure. | Deny request and log reason. | Role-policy change / security owner. |
| A-03 | Clinicians can review draft claims with source context in the proposed workspace. | Human workflow | Usability evaluation with a defined rubric. | Unverified | Apparent savings hide review failure. | Manual authoring. | UI change / quality lead. |
| A-04 | A note type can be bounded by an approved template and source set. | Workflow | Local configuration review. | Unverified | Scope creep and uncontrolled omissions. | Mark note type ineligible. | Template revision / governance lead. |
| A-05 | Review labels can be sampled and adjudicated independently. | Evaluation | Annotation protocol and calibration exercise. | Unverified | Unsafe completion cannot be estimated. | Do not report quality rate. | Rubric revision / quality lead. |
| A-06 | A session identifier can be bound to one authenticated user and request context. | Security | Session isolation tests. | Unverified | Patient or session leakage. | Disable requests for affected path. | Auth/session change / application owner. |
| C-01 | Assistant has no credential or API route that writes clinical-record content. | Authority | Permission inventory and negative integration test. | Required | Autonomous write boundary is violated. | Revoke route; retain manual workflow. | Integration change / application owner. |
| C-02 | Assistant never creates or changes an order, medication, diagnosis, allergy, problem list, or care plan. | Authority | Endpoint deny-list and audit query. | Required | Out-of-scope clinical action. | Disable assistant; incident route. | New tool/action / governance lead. |
| C-03 | All draft claims need evidence or an unsupported marker. | Quality | Claim-to-source audit sample. | Required | Fluent fabrication is hidden. | Abstain or force review warning. | Prompt/model/parser change / quality lead. |
| C-04 | Negation, attribution, and medication-unit content receive separate error labels. | Quality | Annotator guide and adjudication. | Required | Aggregate factual metric masks high-severity errors. | Block use for affected note type. | Rubric/model change / quality lead. |
| C-05 | Requests without eligible sources or with stale/ambiguous identity must abstain. | Safety | Constructed request traces. | Required | System invents completeness. | Manual workflow with visible reason. | Source outage / data owner. |
| C-06 | Manual baseline remains continuously available. | Resilience | Failure-path walkthrough. | Required | Draft outage blocks documentation. | Use existing manual editor. | Baseline dependency change / application owner. |
| C-07 | Evaluation labels do not flow into generation inputs at draft time. | Leakage | Feature-time audit and held-out replay. | Required | Retrospective quality is inflated. | Exclude field and rerun. | New data field / data owner. |
| C-08 | No local policy determination is implied by this manuscript. | Editorial | Source ledger review. | Confirmed | Readers may mistake a design exercise for compliance advice. | Mark OPEN and obtain local review. | Policy source arrives / governance lead. |

The most design-reversing assumptions are A-01, A-03, and A-05.

Without A-01, source grounding cannot be inspected.

Without A-03, review time cannot be interpreted as a usable workflow measure.

Without A-05, a low unsafe-completion estimate has no defensible reference process.

An LLM may produce grammatically plausible statements that have no evidentiary support.

A retrieval layer does not by itself prove that a generated claim is supported.

An evidence span can be stale, misjoined, or attributed to the wrong speaker.

A correct source span can still be transformed into an incorrect claim.

A correct claim can still appear in the wrong encounter if isolation fails.

A correct medication name can still have an incorrect dose or unit.

A correct assertion can become false when a negation is dropped.

A correct observation can become false when attributed to the wrong person.

The system must treat these as distinct failure categories.

A generic “hallucination rate” is not enough for the decision.

No local retention period is stated because it is not in evidence.

No local response-time objective is stated because it is not in evidence.

No local note-signature authority is asserted because it is not in evidence.

The proposed design should be reviewed against actual governing artifacts before any implementation authorization.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Proposed specification |
|---|---|
| Unit of analysis | One `(encounter_id, note_request_id, draft_version)` request; one encounter may have multiple explicit requests, never an implicit reuse. |
| Decision time | `t_request`, when an authenticated clinician requests a draft; all source availability must be at or before the snapshot cutoff `t_cutoff`. |
| Source cutoff | `t_cutoff` is captured with the request and cannot move during generation; a new request creates a new snapshot. |
| Inputs | Authorized encounter-scoped source artifacts, source identifiers, availability timestamps, source versions, and note-template configuration. |
| Output | A draft text plus claim list, evidence references, uncertainty markers, abstention reason when applicable, and immutable draft metadata. |
| Outcome labels | Reviewer correction labels and independently adjudicated claim labels; neither is a clinical outcome. |
| Population | Only encounter-note types explicitly enabled by local configuration; all other requests are excluded and routed manually. |
| Exclusions | Ambiguous encounter identity, missing requester authorization, unavailable source lineage, unsupported language/template, stale required source, or unresolved prior session. |
| Join semantics | `encounter_id` plus source-specific immutable artifact ID; one-to-many source artifacts remain separately represented; no text-only join. |
| Split concept | If offline evaluation occurs, split by encounter and time so no source artifact or later correction from a held-out encounter enters development. |
| Label maturity | A draft is reviewed only after its request; adjudication occurs after a defined delay; incomplete labels remain pending, not negative. |
| Time zone | Every timestamp uses an explicit timezone and clock source; comparisons require normalized instants. |

`encounter_id` identifies the bounded workflow context.

`note_request_id` identifies a single user-requested draft attempt.

`draft_version` identifies a regenerated or edited assistant artifact.

`review_session_id` identifies the authenticated reviewer session.

`source_artifact_id` identifies a source object without relying on mutable display text.

`source_version` identifies the version observed at snapshot assembly.

`source_available_at` is the earliest time the assistant could access that version.

`source_authz_basis` identifies the policy/configuration decision used for access review.

`claim_id` identifies a generated atomic assertion or explicitly marked uncertainty.

`claim_type` is one of asserted, negated, attributed, medication, unit, temporal, uncertain, or administrative.

`evidence_status` is supported, ambiguous, unsupported, excluded, or not-applicable.

`review_disposition` is approved-with-edits, approved-without-edits, rejected, discarded, abstained-manual, or pending.

`unsafe_completion` is an adjudication label, not a user-interface event.

`unsafe_completion` requires a predeclared rubric and a complete reviewed denominator.

The proposed input contract excludes future notes, later addenda, post-encounter outcomes, and cross-encounter free-text retrieval.

The proposed input contract excludes other patients' text.

The proposed input contract excludes unscoped prior-session conversation.

The proposed input contract excludes model memory from an earlier request.

The proposed input contract excludes external web search.

The proposed input contract excludes tool calls that can mutate the clinical record.

The proposed input contract excludes any data field whose authorization or time availability is unknown.

A source may be allowed for a clinician yet not allowed for the proposed assistant.

The access decision is therefore not inferred from clinician visibility alone.

The draft generator receives a compact structured source packet rather than unrestricted record navigation.

Each packet retains source boundaries so the reviewer can inspect origin.

Each packet records omitted source categories and the reason they were unavailable or excluded.

A missing source is not silently represented as a negative fact.

A stale source is not silently represented as current.

A source from another encounter is a hard rejection, not a low-confidence input.

### 5b. Tools & Stack

| Component | Proposed role | Permission boundary | Versioned artifact |
|---|---|---|---|
| Request gateway | Authenticate user, validate request schema, rate-limit, create request ID. | Reads identity/session claims; cannot write record content. | Gateway configuration. |
| Authorization policy service | Resolve encounter-scoped eligibility and source scopes. | Returns allow/deny and reason; does not return record content itself. | Policy version and decision log. |
| Snapshot builder | Materialize authorized source packet at `t_cutoff`. | Read-only source adapters; no cross-encounter retrieval. | Snapshot manifest and hashes. |
| Draft generator | Produce structured draft candidates from packet and template. | No tools, no external retrieval, no record-write credential. | Model ID, prompt ID, decoding config. |
| Claim linker | Segment output and attach evidence references or unsupported markers. | Operates only on snapshot and draft. | Parser/linker version. |
| Review workspace | Display draft, sources, warnings, and actions. | Cannot call clinical write endpoint as assistant. | UI build and schema version. |
| Audit store | Append request, source, generation, review, and failure events. | Append-only access by defined roles. | Event schema and retention configuration. |
| Quality workspace | Sample completed reviews for blinded annotation/adjudication. | De-identified/minimum-necessary process to be defined locally. | Rubric and sampling version. |
| Kill switch | Disable new generation and show manual fallback. | Does not disable baseline editor. | Change record and control state. |

The stack is proposed, not implemented.

Specific products, model vendors, frameworks, infrastructure, and costs are intentionally not asserted.

The design does not require an agent that invokes tools.

The design should prefer a constrained generation call over tool-using autonomy.

The generator should receive explicit task instructions to quote uncertainty rather than complete missing facts.

The generator should receive a fixed output schema.

The output schema should reject unrecognized claim fields.

The gateway should reject malformed request identifiers before source access.

The policy service should decide deny before the snapshot builder fetches content.

The snapshot builder should record every adapter queried and every adapter skipped.

The review workspace should make unsupported claims impossible to confuse with supported claims.

The audit store should not become a shadow clinical record.

Audit design must minimize sensitive text retention and use locally approved controls.

### 5c. Method Summary

The proposed method is `authorized request -> fixed snapshot -> constrained draft -> claim evidence linking -> clinician review -> clinician-controlled record authoring or discard`.

The method does not include autonomous finalization.

The method does not include autonomous action selection.

The method does not include an order recommendation endpoint.

The simplest credible baseline is manual note authoring from the existing workflow.

A second considered alternative is template-only assistance with no generated prose.

A third considered alternative is extractive evidence display without a drafted narrative.

A fourth considered alternative is unrestricted conversational generation.

The unrestricted conversational alternative is not selected because it weakens encounter scope, reproducibility, and source-boundary control.

The template-only alternative may win if provenance linking or reviewer burden is not acceptable.

The extractive alternative may win if factual completeness is more important than prose assembly.

No candidate has been tested in this case.

| Candidate | Evidence considered | Strength | Limitation or avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Manual baseline | Existing workflow is the comparator by definition. | Known author ownership and no assistant write path. | May require more authoring time; actual time remains unmeasured. | Retained baseline. | Baseline workflow materially changes. |
| Template-only fields | Reasoned alternative. | Constrains structure and reduces generation risk. | Still needs factual entry and may not reduce assembly time. | Not tested. | Provenance model fails. |
| Extractive evidence board | Reasoned alternative. | Makes sources visible without asserting prose. | May shift synthesis effort entirely to clinician. | Not tested. | Draft corrections dominate. |
| Constrained draft-only generator | Reasoned proposed method. | Can present a draft with source-linked claims and abstention. | Needs rigorous review and cannot act autonomously. | Proposed, not tested. | Gates in Section 9 pass. |
| Unrestricted chat assistant | Reasoned alternative. | Flexible interaction. | Weak reproducibility, context isolation, and source control. | Rejected for this scope. | Only under a separately defined safety architecture. |

The generator begins with a template schema whose fields identify permitted section labels.

The generator emits each atomic claim with a source reference list.

The claim linker validates that cited source references exist in the snapshot manifest.

A cited source reference does not alone validate semantic support.

The reviewer may mark a claim as supported, unsupported, omitted, incorrect, ambiguous, misattributed, negation-error, medication-error, unit-error, or not-reviewed.

The reviewer may mark a draft as unusable without classifying every claim.

The quality workflow samples both accepted and rejected drafts.

Sampling only accepted drafts would underestimate residual problems.

Sampling only edited drafts would misrepresent unreviewed acceptance behavior.

The method treats a reviewer edit as a signal of burden, not a ground-truth diagnosis of error severity.

The method treats reviewer approval as a workflow event, not proof of factual perfection.

### 5d. Data Provenance, Quality, Exclusions, and Leakage Controls

| Control ID | Control | Mechanism | Failure signal | Containment |
|---|---|---|---|---|
| P-01 | Encounter scope | Require exact `encounter_id` on request, artifacts, and response. | Any artifact has different encounter identity. | Reject packet; do not generate. |
| P-02 | Time cutoff | Freeze manifest at `t_cutoff`; compare `source_available_at <= t_cutoff`. | Artifact arrives after cutoff. | Exclude artifact and record reason. |
| P-03 | Source provenance | Keep adapter, artifact ID, version, hash, extraction time, and access basis. | Missing lineage field. | Abstain or route manual. |
| P-04 | Session isolation | Bind request to authenticated `review_session_id` and server-side authorization check. | Request/session mismatch or stale token. | Deny and invalidate session. |
| P-05 | Output isolation | Keep drafts in request-scoped storage with no cross-request retrieval. | Draft appears under another request. | Disable path; incident investigation. |
| P-06 | Future-information exclusion | Deny artifacts available after `t_cutoff`, including later corrections. | Time audit violation. | Rebuild snapshot and rerun evaluation. |
| P-07 | Label isolation | Keep review/adjudication labels out of generation packet. | Label field reaches prompt packet. | Stop evaluation; remove path. |
| P-08 | Duplicate handling | Preserve source identity and deduplicate only identical immutable artifact/version pairs. | Text-similar but distinct artifacts merged. | Reconstruct packet and audit. |
| P-09 | Missingness visibility | Emit explicit missing/stale/unavailable markers. | Draft implies absence from missing source. | Abstain or force warning. |
| P-10 | Prompt injection boundary | Treat source text as data, not instructions; use structured delimiters and output schema. | Source text changes system task or requests tools. | Reject/escape source and log. |

Provenance begins before generation.

The source adapter must identify what it returned.

The snapshot manifest must identify what the assistant was permitted to see.

The manifest must identify what was not available.

The manifest must identify when each artifact became available.

The manifest must identify which policy version authorized access.

The manifest must be immutable for a particular draft version.

A regenerated draft creates a new output version and references the original or a newly built manifest explicitly.

The reviewer must be able to distinguish regenerated content from clinician edits.

Clinician edits must not be attributed to the model in quality metrics.

The proposed system should never train on drafts, clinician edits, or record content without a separate approved data-use decision.

This case makes no claim that such use is permitted.

A review event is not permission to repurpose the content.

A copied draft is not proof that every included source was appropriate.

A complete-looking packet is not proof that it contains all relevant encounter information.

The system must disclose configured source scope to the reviewer.

The system must disclose missing required fields before presenting a draft as usable.

A time-safe snapshot can still be clinically incomplete.

A provenance-complete claim can still be wrong because the source itself is wrong or ambiguous.

A high evidence-linking rate can still hide omissions.

Quality review must therefore sample both additions and omissions.

### 5e. Reproducibility and Method-Selection Ledger

| Artifact or source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | `../../../../Templates/directory.txt` | consulted 2026-11-10 | Visible clinical decision-support topic provenance. | Local. | Does not provide source-body workflow or result evidence. |
| SRC-02 | `../../../00_SOURCE_COVERAGE_AGENTS.md#scenario-07` | consulted 2026-11-10 | Curriculum brief for bounded decision support, time boundary, authorization, review, and sensitivity. | Local. | Not empirical documentation-assistant evidence. |
| SRC-03 | `../Cheat_Sheets/01_Clinical_Decision_Support_Evidence_and_Handoff.md` | draft, consulted 2026-11-10 | Time-boundary, abstention, and denominator method reference. | Local. | Does not validate this proposed system. |
| SRC-04 | `../Cheat_Sheets/03_Human_Factors_Alert_Design_and_Recovery.md` | draft, consulted 2026-11-10 | Ownership, acknowledgment, and recovery concepts. | Local. | Does not establish local workflow behavior. |
| RUN-01 | Section 6 hand arithmetic | constructed 2026-11-10 | Illustrative workload and approval-boundary smoke inputs. | Local. | Not a clinical dataset or software run. |
| CFG-01 | Proposed request/snapshot/output contracts in Section 8 | version 1.0 | Design-review artifact. | Local manuscript. | No implemented interface exists. |

A reproducible proposed calculation requires the same input counts, units, and formulas.

A reproducible system evaluation would additionally require a versioned source snapshot, request schema, policy decision, model identifier, prompt identifier, decoding configuration, claim-linker version, reviewer rubric, and sampling rule.

A random seed would not reconstruct source availability, session state, or reviewer behavior.

A model version alone would not reconstruct a prompt or source packet.

A prompt alone would not reconstruct a model service behavior.

A snapshot hash alone would not establish that access was authorized.

A reviewer label alone would not establish that the reviewer saw the same draft version.

The ledger separates what is available from what is proposed.

The ledger separates illustrative arithmetic from evidence of real-world performance.

The method-selection reason is containment of scope and authority.

The selection does not assert that constrained generation is clinically effective.

## 6. Analysis Walkthrough

The decisive current result is not a performance improvement.

The decisive current result is a checked illustrative workload model showing which quantities must be counted before a proposed evaluation can claim net documentation-time savings.

The same walkthrough checks that an approval boundary does not become a hidden write path.

**Evidence:** all numerical inputs in this section are ILLUSTRATIVE teaching values.

**Status:** arithmetic is checked; no clinical workflow, model, or outcome evaluation has been executed.

| Step | Hypothesis or decision | Input snapshot | Method and parameters | Intermediate output | Interpretation | Evidence/status |
|---|---|---|---|---|---|---|
| 1 | Define eligible workload. | 120 scheduled requests in one illustrative shift. | Partition by eligibility and abstention. | 90 displayed drafts, 20 abstentions, 10 ineligible requests. | Coverage must retain abstentions in the eligible denominator. | Derived/illustrative. |
| 2 | Compare net time. | Manual mean 12.0 min; proposed components stated below. | Sum draft review, correction, fallback, rejection restart, and unsafe-completion follow-up. | 891 proposed clinician-minutes versus 1,320 manual clinician-minutes. | Estimated difference is 429 clinician-minutes. | Derived/illustrative. |
| 3 | Count clinician capacity. | 90 displayed drafts and 20 abstentions. | Retain every eligible request and every proposed-path workload component. | 891 clinician-minutes across 110 eligible requests, or 8.10 clinician-minutes per eligible request. | Generation does not eliminate clinician work; 429 minutes is not a staffing result. | Derived/illustrative. |
| 4 | Assess unsafe completion denominator. | 3 unsafe completed drafts among 72 adjudicated approved drafts. | Divide by adjudicated completed drafts; report pending and rejected drafts separately. | 4.17% illustrative rate. | It is not a rate over all 120 requests, all eligible requests, or all displayed drafts. | Derived/illustrative. |
| 5 | Check approval boundary. | 90 draft actions, 72 clinician approvals. | Require zero assistant-originated record writes and zero nonclinician approvals. | 0 forbidden writes; 72 clinician-controlled authoring actions. | Approval is necessary but not quality proof. | Constructed state trace. |

### 6.1 Worked workload and net-time calculation

Define `N_total` as all draft requests in the illustrative shift.

Define `N_ineligible` as requests rejected before source access.

Define `N_eligible` as requests that pass request and source eligibility.

Define `N_abstain` as eligible requests that deliberately produce no draft and route to manual documentation.

Define `N_draft` as eligible requests that receive a displayed draft.

Define `N_complete` as displayed drafts with a completed clinician review disposition.

Define `N_pending` as displayed drafts without a completed disposition by the measurement cutoff.

Define `t_manual` as baseline clinician minutes per manual note.

Define `t_open` as clinician minutes to open and orient to a proposed draft.

Define `t_review` as clinician minutes to inspect draft content and evidence.

Define `t_correct` as added clinician minutes for drafts requiring correction.

Define `t_finalize` as clinician minutes in the existing authoring workflow after approval.

Define `t_fallback` as clinician minutes to complete an abstained request manually.

Define `t_reject` as clinician minutes to reject/discard a draft and restart manually.

Define `t_unsafe_followup` as added review or remediation minutes for an unsafe completed draft discovered by quality sampling.

The following equations are workload identities under the stated fixed-average assumptions.

They are not staffing rules.

They are not clinical recommendations.

They are not claims about a measured organization.

```text
FORMULA [DOC-01]: N_eligible = N_total - N_ineligible
WHERE: N_total [requests/shift]; N_ineligible [requests/shift]; N_eligible [requests/shift].
CONSTRAINT: each request has exactly one eligibility disposition; counts use one fixed shift.
NUMERICAL EXAMPLE: N_total=120; N_ineligible=10; N_eligible=120-10=110 requests.
INTERPRETATION: the relevant service denominator is 110 eligible requests, not 120 total requests.
```

```text
FORMULA [DOC-02]: N_draft = N_eligible - N_abstain
WHERE: N_draft [drafts/shift]; N_abstain [eligible requests routed without a draft/shift].
CONSTRAINT: abstention is logged before review outcome and has a manual fallback.
NUMERICAL EXAMPLE: N_eligible=110; N_abstain=20; N_draft=110-20=90 drafts.
INTERPRETATION: 20 abstentions are service work, not missing rows to remove from a time analysis.
```

```text
FORMULA [DOC-03]: coverage = N_draft / N_eligible
WHERE: coverage [proportion]; N_draft [drafts]; N_eligible [eligible requests].
CONSTRAINT: numerator excludes abstentions; denominator includes all eligible requests in the same window.
NUMERICAL EXAMPLE: coverage=90/110=0.818181...=81.82%.
INTERPRETATION: the draft path covers 81.82% of eligible requests under this illustrative configuration.
```

```text
FORMULA [DOC-04]: T_manual = N_eligible * t_manual
WHERE: T_manual [clinician-minutes/shift]; t_manual [clinician-minutes/eligible request].
CONSTRAINT: a fixed illustrative average is used; variation by note type and clinician is excluded.
NUMERICAL EXAMPLE: N_eligible=110; t_manual=12.0 min; T_manual=110*12.0=1320.0 min.
INTERPRETATION: manual comparison must use eligible requests, including requests that would abstain in the proposed path.
```

The baseline and proposed totals deliberately cover the same 110 eligible requests.

The 20 abstentions remain manual fallback workload in both the baseline comparison and proposed total.

The 10 ineligible requests are outside this service comparison because neither path serves them under the stated scope.

```text
FORMULA [DOC-05]: T_draft = N_draft*(t_open+t_review+t_finalize) + N_correct*t_correct
WHERE: T_draft [clinician-minutes]; N_correct [drafts needing correction]; all t values [minutes/draft].
CONSTRAINT: correction time is incremental beyond opening, review, and finalization; correction count refers to reviewed drafts.
NUMERICAL EXAMPLE: N_draft=90; t_open=0.5; t_review=3.0; t_finalize=1.0;
                   N_correct=36; t_correct=4.0;
                   T_draft=90*(0.5+3.0+1.0)+36*4.0=405+144=549 min.
INTERPRETATION: even accepted drafts require 405 minutes of clinician opening, review, and finalization before corrections.
```

```text
FORMULA [DOC-06]: T_abstain = N_abstain*t_fallback
WHERE: T_abstain [clinician-minutes]; t_fallback [manual completion minutes per abstention].
CONSTRAINT: abstention is explicit; fallback time includes manual authoring only, not any unrelated care time.
NUMERICAL EXAMPLE: N_abstain=20; t_fallback=12.0 min; T_abstain=20*12.0=240 min.
INTERPRETATION: abstentions preserve the manual baseline workload for those requests.
```

```text
FORMULA [DOC-07]: T_reject = N_reject*t_reject
WHERE: T_reject [clinician-minutes]; N_reject [rejected displayed drafts]; t_reject [minutes/rejection and restart].
CONSTRAINT: rejected drafts are not counted as finalized draft savings.
NUMERICAL EXAMPLE: N_reject=9; t_reject=8.0 min; T_reject=9*8.0=72 min.
INTERPRETATION: rejection is a workload component even if no generated text is retained.
```

```text
FORMULA [DOC-08]: T_unsafe = N_unsafe_completed*t_unsafe_followup
WHERE: T_unsafe [quality/remediation minutes]; N_unsafe_completed [adjudicated completed drafts];
       t_unsafe_followup [minutes per discovered unsafe completion].
CONSTRAINT: count only independently adjudicated cases; discovery delay and downstream effect are not modeled.
NUMERICAL EXAMPLE: N_unsafe_completed=3; t_unsafe_followup=10.0 min; T_unsafe=3*10.0=30 min.
INTERPRETATION: this adds an illustrative review burden, not a measure of patient harm.
```

```text
FORMULA [DOC-09]: T_proposed = T_draft + T_abstain + T_reject + T_unsafe
WHERE: T_proposed [clinician-minutes/shift]; terms defined in DOC-05 through DOC-08.
CONSTRAINT: components are mutually exclusive where counted; no double-count of correction within rejection.
NUMERICAL EXAMPLE: T_proposed=549+240+72+30=891 min.
INTERPRETATION: proposed workflow time includes visible review, fallback, rejection, and discovered unsafe-completion burden.
```

```text
FORMULA [DOC-10]: Delta_T = T_manual - T_proposed
WHERE: Delta_T [clinician-minutes/shift]; positive means estimated time lower than manual baseline.
CONSTRAINT: both totals cover the same 110 eligible requests and comparable task scope.
NUMERICAL EXAMPLE: Delta_T=1320-891=429 min=7.15 clinician-hours.
INTERPRETATION: under constructed inputs, estimated net time is lower by 429 minutes; this is not an observed saving.
```

The figures in DOC-01 through DOC-10 correct the simplified first table.

The first table separated the 90 draft-path requests to expose review burden.

The full comparator restores the 20 abstentions to the manual denominator.

The 10 ineligible requests are excluded from both totals because the proposed workflow never serves them.

An evaluation must document how the manual baseline handles those ineligible requests if it later reports all-request workflow time.

The arithmetic is dimensionally consistent because every time total is a count times minutes per count.

`T_manual`, `T_draft`, `T_abstain`, `T_reject`, and `T_unsafe` are all clinician-minutes per shift.

`Delta_T` has the same unit.

The draft-path correction count is 36 of 90 displayed drafts in this illustration.

That correction proportion is 40%.

It is not a factual-error rate because corrections may be stylistic or structural.

The rejection count is 9 of 90 displayed drafts in this illustration.

Those 9 are not also included as approved completions.

For a complete disposition accounting example, let 72 be approved drafts, 9 be rejected drafts, and 9 be pending drafts.

Then `72 + 9 + 9 = 90` displayed drafts.

A pending draft cannot be counted as a completed success or failure until disposition is known.

### 6.2 Unsafe completion and abstention denominators

```text
FORMULA [DOC-11]: unsafe_completion_rate = N_unsafe_completed / N_adjudicated_completed
WHERE: N_unsafe_completed [completed drafts with an adjudicated unsafe category];
       N_adjudicated_completed [completed drafts sampled and adjudicated under the same rubric].
CONSTRAINT: do not place rejected, abstained, pending, or unadjudicated drafts in this denominator.
NUMERICAL EXAMPLE: N_unsafe_completed=3; N_adjudicated_completed=72;
                   unsafe_completion_rate=3/72=0.041666...=4.17%.
INTERPRETATION: 4.17% describes the constructed adjudicated completed-draft subset only.
```

```text
FORMULA [DOC-12]: abstention_rate = N_abstain / N_eligible
WHERE: N_abstain [eligible requests without a draft]; N_eligible [eligible requests].
CONSTRAINT: abstention reason is logged at request time; ineligible requests are excluded.
NUMERICAL EXAMPLE: abstention_rate=20/110=0.181818...=18.18%.
INTERPRETATION: 18.18% of eligible requests use manual fallback in this illustration.
```

```text
FORMULA [DOC-13]: completed_draft_rate = N_complete / N_draft
WHERE: N_complete [reviewed drafts with terminal clinician disposition]; N_draft [displayed drafts].
CONSTRAINT: pending drafts stay in the denominator and are not silently dropped.
NUMERICAL EXAMPLE: N_complete=81; N_draft=90; completed_draft_rate=81/90=90.00%.
INTERPRETATION: terminal review completion is a workflow measure, not approval quality.
```

The 72 approvals in the approval-boundary trace are a subset of the 81 completed reviews.

The other 9 completed reviews are rejections.

The 9 pending reviews remain open at the measurement cutoff.

The unsafe-completion denominator is 72 only if all 72 approved drafts are adjudicated.

If only 30 approved drafts are adjudicated, the rate denominator is 30, not 72.

If adjudication samples without a valid sampling design, the result cannot be generalized.

If an unsafe category is discovered after note signing, it must still retain its discovery and classification timestamp.

The case does not quantify patient or clinical harm from an unsafe completion.

The residual-harm term in the workload model is a review/remediation-time proxy only.

It must never be presented as patient benefit or harm measurement.

### 6.3 Approval-boundary state trace

The following constructed trace is a mechanism check, not a deployed audit log.

Precondition: the assistant service account lacks record-write permissions.

Precondition: the clinician uses a separate authenticated record-authoring session.

Precondition: the review workspace uses an explicit `approve_for_copy` action that is not a record write.

Input: request `R-17` for encounter `E-44` by clinician session `S-9`.

State 1: gateway authenticates `S-9` and assigns `R-17`.

State 2: policy service permits defined source scope for `E-44`.

State 3: snapshot builder creates manifest `M-17` at `t_cutoff`.

State 4: generator creates draft `D-17-v1` outside the record.

State 5: claim linker attaches evidence references and one visible uncertainty marker.

State 6: review workspace shows `D-17-v1` to `S-9` only.

State 7: clinician edits content and selects `approve_for_copy`.

State 8: workspace records approval event with clinician identity, not assistant identity.

State 9: clinician independently uses existing record editor to author the final note.

State 10: audit receives a record-authoring reference if locally approved, without assistant content write.

Invariant check A: assistant-originated clinical record writes equals 0.

Invariant check B: order, medication, diagnosis, allergy, problem-list, and care-plan mutations by assistant equal 0.

Invariant check C: approval event actor equals authenticated clinician, not service account.

Invariant check D: record-write actor, if recorded, equals clinician session in existing workflow.

Invariant check E: failed authorization at State 2 leads to no snapshot and no draft.

Invariant check F: missing source lineage at State 3 leads to abstention and manual fallback.

Invariant check G: session mismatch at State 6 leads to no draft display.

A UI button label cannot establish these invariants.

Only permission inventory, negative tests, and audit inspection could establish them in an implementation.

### 6.4 Sensitivity and uncertainty

The time estimate is sensitive to correction burden, abstention rate, rejection restart time, and the manual baseline.

The following table changes one input at a time while holding other illustrative inputs fixed.

| Stress or test | Original condition | Alternative | Recomputed proposed time | Change in conclusion | Uncertainty and limitation | Decision consequence |
|---|---|---|---:|---|---|---|
| Higher correction burden | `N_correct=36`, `t_correct=4` min. | `N_correct=54`, `t_correct=5` min. | `405+270+240+72+30=1017` min. | Estimated saving falls to `1320-1017=303` min. | Correction categories and time are unmeasured. | Measure correction taxonomy before proceeding. |
| More abstention | `N_abstain=20`, `N_draft=90`. | `N_abstain=40`, `N_draft=70`; hold 40% correction proportion. | `70*4.5 + 28*4 + 40*12 + 7*8 + 30 = 993` min. | Estimated saving is 327 min. | Rejection count is rounded assumption. | Report coverage with time, not alone. |
| Slower manual baseline | `t_manual=12` min. | `t_manual=9` min. | `T_manual=990` min. | Estimated saving is only 99 min. | Baseline variation may dominate model conclusion. | Collect paired/manual time data. |
| More unsafe follow-up | `3*10=30` min. | `6*20=120` min. | `549+240+72+120=981` min. | Estimated saving is 339 min. | This is not a patient-harm measure. | Do not reduce quality sampling. |
| No net benefit boundary | Original proposed components. | `t_manual=8.1` min. | `T_manual=891` min. | `Delta_T=0`; no estimated time saving. | Equality ignores variability and implementation overhead. | Do not justify work on time alone. |

For the more-abstention stress, the totals are checked as follows.

Draft open/review/finalize time is `70 * 4.5 = 315` minutes.

Correction time is `28 * 4 = 112` minutes.

Abstention fallback time is `40 * 12 = 480` minutes.

Rejection restart time is `7 * 8 = 56` minutes.

Unsafe follow-up time remains `30` minutes.

The total is `315 + 112 + 480 + 56 + 30 = 993` minutes.

The manual total remains `110 * 12 = 1320` minutes.

The difference is `1320 - 993 = 327` minutes.

The no-net-benefit boundary solves `110 * t_manual = 891`.

Therefore `t_manual = 891 / 110 = 8.1` minutes per eligible request.

Below 8.1 minutes, this constructed proposed workflow consumes more time than the constructed manual baseline.

At 8.1 minutes, it has no estimated time advantage before model operations or implementation overhead.

Above 8.1 minutes, it has a modeled time difference but still no demonstrated quality or clinical benefit.

A confidence interval is not calculated because there is no sample of observed times.

A future measurement should define whether estimates use encounter-level paired differences, clinician-level clustering, or another justified method.

A future measurement should not treat repeated notes from one clinician as independent without checking clustering.

### 6.5 What was tried first and why it did not work

No approach was actually tested in this case.

Therefore no rejected implementation result is claimed.

The unrestricted conversational alternative was rejected at design review, not after an observed failure.

Its reason is that unrestricted context and tools would make encounter scoping, prompt reproducibility, and write-boundary enforcement harder to demonstrate.

The template-only alternative remains credible.

It should be tested as a comparator if a proposed evaluation is authorized.

The extractive evidence-board alternative also remains credible.

It could be preferable if generated narrative correction time is high.

## 7. The Finding

The supported recommendation is to retain manual documentation as the operational baseline and authorize no autonomous or assistant-originated record action.

A draft-only evaluation is only a proposed next step, contingent on explicit source provenance, encounter/session isolation, clinician review, independent adjudication, and an enforceable zero-write boundary.

Under constructed inputs covering 110 eligible requests, the proposed workflow estimates `429` fewer clinician-minutes than a `12.0` minute manual baseline, but this is derived arithmetic rather than a measured saving.

The estimate reverses at a manual baseline of `8.1` minutes per eligible request and excludes unmeasured variability, implementation overhead, clinical impact, and downstream consequences.

The main boundary is that fluent draft text, reviewer approval, and a source link do not establish factual completeness, medical efficacy, legal compliance, or safe clinical use.

## 8. System/Solution Design

### 8a. Offline / Evaluation Path

The proposed offline path begins with a locally approved retrospective source snapshot definition.

It must never use later note edits or labels as draft-time inputs.

A cohort builder selects requests using predeclared eligibility rules.

A snapshot validator checks identity, authorization, time cutoff, source version, and schema completeness.

A packet builder produces a minimum-necessary structured representation.

A versioned generator produces a candidate draft.

A versioned claim linker attaches source evidence references.

A quality sampler selects completed, rejected, abstained, and pending-path records according to a declared plan.

Reviewers receive a rubric that distinguishes omission, addition, negation, attribution, medication, unit, provenance, and leakage categories.

An adjudicator resolves predefined disagreements without seeing the generating configuration where blinding is feasible.

An analysis job computes denominators by request state.

A governance report records result, uncertainty, deviations, and decision.

```text
[Eligible request cohort + cutoff]
              |
              v
[Authorization and time validator] --reject--> [Manual-baseline-only ledger]
              |
              v
[Versioned source snapshot manifest]
              |
              v
[Constrained draft generator] -> [Claim/evidence linker]
              |                          |
              v                          v
      [Draft version store]       [Unsupported/ambiguous markers]
              |                          |
              +-------------> [Blinded quality sample]
                                           |
                                           v
                           [Rubric + adjudication + denominator report]
                                           |
                                           v
                           [Governance decision; no record-write action]
```

The offline path has no training requirement in this case.

If a future model is adapted, training data rights, versioning, and evaluation separation would require a new decision record.

The current design does not assume fine-tuning.

The current design does not assume retained prompt histories.

The current design does not assume external retrieval.

The snapshot validator must fail closed for unknown encounter identity.

The snapshot validator must fail closed for source availability after cutoff.

The snapshot validator must fail closed for missing authorization basis.

The cohort builder must retain excluded records with exclusion reasons for denominator auditing.

The quality sampler must retain abstentions to evaluate whether abstention reasons are concentrated in a subgroup or note type.

### 8b. Online / Serving / Review Path

A proposed request path is shown below.

```text
[Clinician authenticated session + encounter + note type]
                         |
                         v
              [Request schema validation]
                         |
             deny ------+------ malformed or unauthenticated
             |                   |
             v                   v
      [Manual fallback]       [Audit deny reason]
                         |
                         v
      [Encounter-scoped authorization policy]
                         |
             deny ------+------ unauthorized or unsupported scope
             |                   |
             v                   v
      [Manual fallback]       [Audit deny reason]
                         |
                         v
   [Fixed source snapshot at t_cutoff + provenance manifest]
                         |
          abstain ------+------ missing, stale, ambiguous, or unlined source
             |                   |
             v                   v
      [Manual fallback]   [Visible abstention reason + audit]
                         |
                         v
 [Constrained generator with no tools and no write credential]
                         |
                         v
       [Claim linker + uncertainty and source markers]
                         |
                         v
 [Clinician review: edit / approve-for-copy / reject / discard]
       |                  |                  |             |
       v                  v                  v             v
 [audit]        [clinician-controlled]     [audit]       [audit]
                [existing record editor]
                         |
                         v
          [Clinician writes final note; assistant never writes]
```

The request schema includes `request_id`, `encounter_id`, `note_type`, `requester_id`, `session_id`, `requested_at`, and optional template configuration ID.

The gateway validates field types and rejects missing identifiers.

The gateway attaches server-observed request time separately from client-supplied time.

The authorization policy returns an allow or deny decision with policy version and reason code.

The policy should not return a broad unrestricted chart token to the generator.

The snapshot builder fetches only configured source classes.

Each fetched artifact carries identity, version, availability time, and source-system metadata.

The snapshot manifest records all included artifact IDs.

The snapshot manifest records all excluded artifact IDs when known.

The snapshot manifest records why each exclusion occurred.

The generator receives the manifest-derived payload and cannot call source adapters.

The generator cannot invoke orders, medication systems, messaging systems, or record APIs.

The generator cannot access another request's draft cache.

The claim linker operates after generation and before display.

If the linker fails, the workspace must show an abstention or clearly label the draft unavailable.

The workspace must not silently show unlinked prose as a completed draft.

The workspace shows source identifiers and relevant excerpts according to permitted access.

The workspace differentiates assistant text, clinician edits, and source text.

The workspace displays a configured warning for unsupported or ambiguous claims.

The workspace offers reject and discard as first-class actions.

The workspace does not make approval the default action.

The workspace must preserve a manual fallback link.

The workspace records approval as an intent to use content, not as proof of truth.

The existing record editor remains the sole path for note persistence.

The record editor should perform its own authentication and authorization checks.

The assistant does not proxy the clinician's record-write request.

A copy operation, if provided, should be explicit and should not auto-submit or auto-sign.

A system integration must not convert clipboard or copy state into approval state.

A network timeout after generation must not retry a record write because none exists.

A timeout before display must return an owned manual fallback.

A timeout during review preserves the draft only under defined session and retention rules; otherwise it requires a new request.

A model-service timeout creates an abstention rather than a fabricated partial note.

An authorization-service timeout denies generation and routes manually.

A source-adapter timeout yields an explicit incomplete-source abstention unless a locally approved partial-source rule exists.

A claim-linker timeout yields no draft display.

A review-workspace outage leaves the manual baseline available.

### 8c. Key Design Decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| DEC-01 | Draft-only output outside record. | Direct record draft, autonomous write, manual only. | Proposed containment reasoning. | Extra copy/review step versus authority control. | Any assistant-originated record write. | New integration request. |
| DEC-02 | Fixed encounter/time snapshot. | Live chart context, broad retrieval. | Time-boundary method reference. | May abstain more often. | Source after cutoff or different encounter. | Source contract change. |
| DEC-03 | Claim-level evidence and unsupported marker. | Whole-note citation, no provenance. | Proposed auditability requirement. | More interface complexity. | Claim appears supported without valid source ID. | Linker/rubric revision. |
| DEC-04 | Explicit abstention with manual fallback. | Silent blank output, low-confidence prose. | Workflow reasoning. | Lower coverage versus hidden incompleteness. | Abstention has no owner or reason. | Capacity/workflow change. |
| DEC-05 | Clinician-controlled existing record write. | Assistant API write, delegated signer. | Hard authority constraint. | Less automation. | Service account can mutate record. | Permission model change. |
| DEC-06 | Sample independent adjudication. | Reviewer acceptance only. | Evaluation design requirement. | Added quality workload. | Completed drafts lack auditable quality denominator. | Rubric or sampling change. |
| DEC-07 | Disable-first rollback. | Continue degraded generation. | Proposed recovery control. | Lost convenience during investigation. | Leakage or forbidden-write signal. | Incident or source mismatch. |

### 8d. Architecture Artifacts and Operational Readiness

| Area | Proposed contract |
|---|---|
| Interfaces | Versioned request, policy response, snapshot manifest, draft, claim, review-event, and abstention schemas; unknown fields are rejected or explicitly versioned. |
| Identity | Authenticated clinician identity, server-side session binding, encounter relationship check, least-privilege service identities. |
| State | Request-scoped immutable manifest and draft versions; append-only audit events; no shared conversational memory across requests. |
| Retention | Not specified by this case; local policy must define minimum necessary retention, deletion, and access review before implementation. |
| Reliability | User-facing measure would be percentage of eligible requests receiving either a usable review workspace or a visible manual fallback; target remains unmeasured and must be locally set. |
| Resilience | Bounded retries for idempotent request status only; no retry path may write a record; dependency failure produces denial or abstention. |
| Recovery | Disable new generation, preserve manual baseline, isolate affected request/session scope, investigate manifest and audit lineage, re-enable only after defined evidence. |
| Observability | Request counts, eligibility reasons, abstention reasons, source-lineage failures, session mismatches, draft display failures, review dispositions, forbidden-write attempts, and quality-label denominators. |
| Security | Threat model covers unauthorized access, cross-encounter leak, cross-session leak, prompt injection, overbroad tokens, audit exposure, and assistant write escalation. |
| Economics | Only clinician-time equations are illustrative; model, infrastructure, storage, privacy-review, and implementation costs are unpriced and OPEN. |
| Human workflow | Clinician approves/rejects; governance lead stops; application owner disables; data owner repairs source lineage; quality lead adjudicates sampled content. |

The proposed request-state machine has explicit terminal states.

`denied` means authorization or schema failed before source packet assembly.

`abstained` means the request was eligible enough to enter processing but lacked a required condition for a trustworthy draft.

`generated` means a draft exists outside the record.

`under_review` means a reviewer can inspect a draft in an authorized session.

`approved_for_copy` means a clinician selected the explicit review action.

`rejected` means the clinician declined the draft.

`discarded` means the draft is intentionally not used.

`expired` means session or retention rules ended availability without approval.

`error` means a system failure prevented normal completion.

Every terminal state must retain a reason code.

No terminal state is equivalent to a signed note.

No state transition invokes a care, order, or record mutation API.

The one exception is a clinician independently using the existing record application, which lies outside assistant authority.

The system must reject duplicate request IDs or make duplicate handling idempotent.

A retry must return the existing status or create a clearly versioned new request.

A retry must not combine source packets from two request times.

A cache key must include encounter identity, request ID, session/authorization scope, and source manifest version.

A cache key based only on a patient identifier is insufficient.

A cache key based only on note type is unsafe.

A shared prompt-history buffer is prohibited for this scope.

A developer or support console must not expose drafts without role-appropriate authorization.

Logs should use identifiers and structured categories where possible rather than unnecessary full note text.

If text is retained for quality review, the local governance process must authorize the access and retention design.

This manuscript does not establish that authorization.

### 8e. Successful and failed request reasoning

A successful request is not “the model returned text.”

A successful request path requires request validation, permitted scope, complete required provenance, a draft or explicit abstention, a reviewable display, and auditability.

A successful clinician workflow may still reject the draft.

Rejection can be the correct outcome when the draft is unhelpful or unsupported.

A failed request can fail safely when it produces a visible manual fallback.

A failed request fails unsafely when it produces content for the wrong encounter, exposes another session, silently omits required source, or mutates a record.

Recovery begins by stopping new generation for the affected scope.

Recovery does not begin by retrying a draft into the record.

Recovery requires preserving enough audit lineage to determine affected requests.

Recovery requires notifying the named operational and governance owners through a local process not specified here.

Recovery requires deciding whether already displayed drafts need a review action.

Recovery requires confirming that the manual baseline remains available.

Recovery verification requires replaying the fault path in a nonclinical test environment where feasible.

## 9. Evaluation & Validation

The evaluation question is not whether generated prose sounds plausible.

The evaluation question is whether the proposed draft-only workflow can be shown, under a predeclared protocol, to reduce clinician documentation burden relative to manual documentation without violating hard boundaries or exceeding defined content-quality guardrails.

No evaluation has been executed for this case.

Every threshold below is a proposed governance parameter, not an approved clinical criterion.

| Evaluation field | Proposed definition |
|---|---|
| Primary metric | Net clinician minutes per eligible encounter: manual baseline minutes minus proposed-path clinician minutes, including review, correction, rejection, abstention fallback, and discovered unsafe-completion follow-up. |
| Numerator | Sum of measured clinician task minutes under a defined observation protocol. |
| Denominator | Eligible encounter-note requests in the same note-type, clinician, and window definition; abstentions remain included. |
| Unit | Clinician-minutes per eligible encounter, plus total clinician-minutes for capacity planning. |
| Baseline | Manual documentation in the existing workflow for matched eligible encounter-note requests. |
| Proposed pass rule | A governance-defined lower confidence bound or predeclared practical margin for net time must favor the proposed workflow, while all hard guardrails pass; no threshold is approved in this case. |
| Hard guardrail | Zero assistant-originated record writes, zero autonomous care/order changes, zero confirmed cross-encounter/session leaks in executed tests, and zero unresolved critical access-control failures. |
| Content guardrail | Report separate rates and counts for factual addition, factual omission, negation, attribution, medication, unit, provenance, and leakage errors; do not aggregate away severe classes. |
| Review guardrail | Report median and distribution of review/correction time; reviewer completion and pending denominators; no claim from approved drafts alone. |
| Test design | Offline replay and rubric-based human review first; any later live study requires separate local approval and protocol. |
| Sample | To be determined from observed variability, desired precision, reviewer clustering, note types, and error-category rarity. |
| Uncertainty | Use a prespecified approach appropriate to paired or clustered encounter-level time measurements and sampled adjudication; no interval is available now. |
| Validity threat | Selection into eligibility, reviewer behavior change, leakage from later notes, incomplete labels, and nonrepresentative source availability. |
| Artifact | Versioned protocol, manifest audit, model/prompt configuration, reviewer rubric, adjudication guide, sampling ledger, and analysis query. |
| Decision | Not executed; no pass/fail result exists. |

### 9.1 Content-quality taxonomy

| Error category | Definition | Numerator | Denominator | Why separate | Required review action |
|---|---|---|---|---|---|
| Factual addition | A draft asserts a material fact not supported by authorized source evidence. | Adjudicated drafts/claims with unsupported addition. | Adjudicated completed drafts or claims, stated explicitly. | Fluent additions can appear credible. | Correct/reject; investigate source/linker/generator. |
| Factual omission | A required source-supported fact is absent under the predeclared note rubric. | Adjudicated drafts with qualifying omission. | Adjudicated drafts with applicable rubric item. | Omission is invisible to citation-only metrics. | Correct or reject; assess source scope/template. |
| Negation error | A claim reverses presence/absence, denial/affirmation, or uncertainty polarity. | Adjudicated negation errors. | Applicable negated/affirmed claim opportunities. | Low frequency can conceal high consequence. | Stop affected template if threshold breached. |
| Attribution error | A statement is assigned to the wrong speaker, source, time, or subject. | Adjudicated attribution errors. | Applicable attributed claims. | Correct words can be false under wrong attribution. | Correct/reject; inspect entity binding. |
| Medication error | Medication name, route, dose, frequency, timing, or status is wrong or unsupported. | Adjudicated medication errors. | Drafts/claims containing medication content, stated explicitly. | Medication information needs category-level visibility. | Block medication rendering if unresolved. |
| Unit error | A numeric unit, magnitude, frequency, or conversion is wrong or unsupported. | Adjudicated unit errors. | Numeric/medication claims with units. | Correct number with wrong unit remains incorrect. | Block affected rendering; add unit validation. |
| Provenance failure | Claim lacks valid source reference or reference is not in snapshot. | Claims with invalid/absent lineage. | All displayed claims. | Source display is a core boundary. | Abstain/display warning; repair linker. |
| Temporal leakage | Draft uses source unavailable by `t_cutoff`. | Requests with violating source. | All generated requests. | Invalidates decision-time evaluation. | Stop path; rebuild snapshot. |
| Patient/session leakage | Draft/source data crosses encounter or authenticated session boundary. | Confirmed leakage events. | Requests/sessions tested, stated explicitly. | Access boundary is categorical, not average quality. | Disable affected path and investigate. |

A claim-level denominator can differ from a draft-level denominator.

The report must say which denominator is used.

A draft with two unit errors contributes two claim errors but one affected-draft count.

Both may be useful if reported distinctly.

A draft with no medication content is not automatically a true negative for medication error.

The medication denominator requires applicability.

An omission denominator requires a predeclared reference rubric.

Without a reference rubric, absence cannot be consistently classified.

A reviewer correction is not automatically a factual addition error.

A reviewer may edit style, chronology, or local template formatting.

Quality labels must distinguish content issue from stylistic preference.

The adjudicator must have access to authorized source evidence and the rubric, not merely the draft.

### 9.2 Human review and adjudication protocol

A proposed review sample begins with a frozen request and draft version.

The sampler records whether the draft was approved, rejected, discarded, abstained, pending, or errored.

The sampler includes approved drafts because approval is not perfect validation.

The sampler includes rejected drafts because rejection reasons reveal workload and quality failure.

The sampler includes abstentions because abstention can conceal systematic source availability problems.

The sampler includes pending drafts because pending work affects capacity and censoring.

A reviewer receives the draft, source evidence, source-scope declaration, and annotation rubric.

Where feasible, the reviewer should not be told the generator configuration when judging content.

The reviewer labels each applicable category independently.

A second reviewer labels a defined overlap sample.

Disagreement is recorded rather than overwritten.

An adjudicator resolves disagreements using the rubric and source snapshot.

If source evidence is ambiguous, the label should permit `indeterminate` rather than force a false binary.

Indeterminate labels require a reported denominator treatment.

If indeterminate labels are excluded, the report must give their count and reason.

If a reviewer cannot assess a category because a source is unavailable, that is a source-coverage finding, not a negative error label.

The rubric must define materiality separately from grammatical quality.

The rubric must define whether an incorrect capitalization or abbreviation qualifies as a factual error.

The rubric must define medication-name and unit normalization rules.

The rubric must define how copied clinician edits are separated from draft content.

The rubric must define whether template placeholders are errors or abstentions.

The protocol must lock the rubric version before tallying a primary result.

Rubric changes create a new analysis version.

A calibration exercise can measure annotation agreement.

Agreement alone does not establish that the rubric reflects clinical truth.

A low agreement result should halt interpretation of a fine-grained quality rate until categories are clarified.

### 9.3 Manual comparison design

The strongest practical comparison should measure comparable manual and proposed workflow tasks.

A before/after comparison can be confounded by note type, staffing, template change, and learning.

A concurrent randomized workflow study may reduce some confounding but requires separate local design and approval.

This case does not prescribe a clinical-study design.

A paired simulation can compare the same de-identified source packet under manual and proposed review tasks.

A paired simulation may not reproduce interruption, real workload, or final documentation behavior.

An offline replay can establish provenance and error taxonomy feasibility.

An offline replay cannot establish time savings in live work.

A live shadow mode can measure technical outputs without exposing a draft for use.

A shadow mode cannot establish clinician review burden if no clinician reviews it.

A proposed evaluation should predeclare note-type strata.

A proposed evaluation should predeclare how repeated encounters from one clinician are handled.

A proposed evaluation should predeclare how interventions are stopped.

A proposed evaluation should predeclare how missing time observations are handled.

A proposed evaluation should predeclare whether correction time is timed by task observation, self-report, or instrumentation.

Each timing method has bias and intrusiveness tradeoffs.

A timer that starts at draft display may miss source orientation before display.

A timer that ends at approval may miss record-editor finalization.

A manual baseline timer must use a comparable start and stop definition.

The protocol must define interruptions and multitasking treatment.

The protocol must define whether break time is excluded and how.

The protocol must not treat model latency as clinician time saved unless it changes the workflow interval.

### 9.4 Proposed evaluation gates

| Gate | Required evidence | Decision if not met | Owner |
|---|---|---|---|
| G-01 Scope | Explicit note type, encounter, source, and write-permission contract. | Do not generate. | Governance lead. |
| G-02 Access | Negative tests for unauthorized encounter/session and overbroad scope. | Do not expose draft. | Security owner. |
| G-03 Provenance | Snapshot manifest and claim-link validity on all audited requests. | Abstain or repair pipeline. | Data owner. |
| G-04 Temporal validity | No source with `source_available_at > t_cutoff` in evaluation packet. | Invalidate affected analysis. | Data owner. |
| G-05 Write boundary | Permission inventory plus negative test show zero assistant record mutations. | Stop proposal. | Application owner. |
| G-06 Review workflow | Reviewer can find evidence, edit, reject, and access manual fallback in scenario test. | Redesign interface. | Quality lead. |
| G-07 Quality taxonomy | Rubric, calibration, and adjudication protocol operational. | Do not report safety/content rate. | Quality lead. |
| G-08 Capacity | All eligible requests, abstentions, corrections, rejections, and pending work included. | Do not claim net savings. | Governance lead. |
| G-09 Recovery | Timeout, denial, source-mismatch, session-mismatch, and disable traces end in manual fallback. | Do not authorize wider study. | Application owner. |
| G-10 Decision record | Evidence labels, deviations, uncertainty, and stop authority documented. | Remain proposed. | Governance lead. |

## 10. Action Taken

No action has been taken as of 2026-11-10.

No system has been deployed.

No clinical note has been generated by a real system described here.

No clinician workflow change has been implemented.

No measured time saving exists.

No model quality result exists.

No local policy or compliance conclusion exists.

The action record is therefore a proposed recommendation rather than an implementation claim.

| Action field | Record |
|---|---|
| Recommendation | Retain manual baseline; consider only an offline, draft-only design evaluation after G-01 through G-10 have evidence. |
| Approval/status | Not approved; proposed design exercise. |
| Implementer and handoff | No implementer assigned; if authorized later, application owner, data owner, quality lead, security owner, and clinician governance lead require named artifacts. |
| Effective scope | None. |
| Release strategy | No deployment; offline replay and constructed failure traces first. |
| Stop/rollback | Do not enable; if implementation later exists, disable generation on access, provenance, leakage, or write-boundary failure while preserving manual baseline. |
| Excluded actions | Autonomous note finalization, record writes, orders, medication changes, diagnosis changes, patient communication, and broad conversational chart access are excluded. |

The recommended first artifact is a source and access contract.

The recommended second artifact is a reviewer rubric.

The recommended third artifact is a negative permission test demonstrating no assistant write capability.

The recommended fourth artifact is an offline replay with frozen source manifests.

The recommended fifth artifact is a capacity study that retains abstentions and corrections.

The order matters because a time estimate is not useful if the input boundary is unknown.

The order matters because a fluent draft is not useful if its sources cannot be inspected.

The order matters because a reviewer cannot correct what the interface hides.

The order matters because a proposed release must be stoppable without blocking manual documentation.

## 11. Outcome & Results

**Outcome not yet measured.**

No draft-only system described by this case has been implemented or evaluated.

No baseline timing study has been observed.

No treatment timing study has been observed.

No content-quality rate has been observed.

No patient, care, safety, financial, or legal outcome has been measured.

The following table distinguishes constructed arithmetic from measured results.

| Metric | Baseline value/window/N | Treatment or after value/window/N | Absolute change | Relative change | Uncertainty | Evidence/source | Limitation |
|---|---|---|---|---|---|---|---|
| Net clinician time | Not measured. | Illustrative `T_manual=1320` versus `T_proposed=891` clinician-minutes across 110 eligible requests. | Illustrative `429` minutes. | Not reported as realized change. | No sample interval. | Derived Section 6 inputs. | Constructed averages; not observed workflow. |
| Draft coverage | Not measured. | Illustrative `90/110=81.82%`. | Not applicable. | Not applicable. | No sample interval. | Derived Section 6 inputs. | Configuration assumption, not service observation. |
| Abstention rate | Not measured. | Illustrative `20/110=18.18%`. | Not applicable. | Not applicable. | No sample interval. | Derived Section 6 inputs. | No actual source-availability distribution. |
| Unsafe completion | Not measured. | Illustrative `3/72=4.17%` among adjudicated completed drafts. | Not applicable. | Not applicable. | No sampling design. | Derived Section 6 inputs. | Not a patient-harm metric or real quality rate. |
| Assistant record writes | Not measured implementation. | Constructed trace expects `0`. | Not applicable. | Not applicable. | Requires permission and audit tests. | Section 6.3 mechanism trace. | A design trace is not a system test. |

The next measurement plan belongs to a separately approved protocol.

The protocol owner should be the governance lead with named quality, data, application, and security collaborators.

The proposed window should be stated before measurement.

The proposed population should be stated before measurement.

The proposed note-type mix should be stated before measurement.

The proposed stop rule should be stated before measurement.

The decision trigger should state whether the result authorizes a narrower evaluation, retains manual baseline, or stops the work.

A completed evaluation still would not establish medical efficacy unless it measured an appropriate clinical outcome under a suitable design.

A completed evaluation still would not establish legal compliance without applicable authoritative review.

A completed evaluation still would not authorize autonomous care or record changes under this case scope.

## 12. Failure Modes & What Went Wrong

No incident is claimed.

The following are hypothetical or constructed failure mechanisms for design and interview use.

| Failure ID | Trigger and symptom | Detection or discriminating test | Confirmed cause or hypothesis | Impact/blast radius | Containment | Prevention/owner | Verification/limitation |
|---|---|---|---|---|---|---|---|
| F-01 Future source leakage | Later addendum appears in a pre-cutoff draft. | Compare each artifact `source_available_at` with `t_cutoff`. | Hypothesis: warehouse snapshot used latest state. | Affected request cohort and invalid retrospective evaluation. | Stop affected analysis; rebuild packets. | Immutable manifest / data owner. | Replay shows zero post-cutoff artifacts; does not prove content quality. |
| F-02 Factual addition | Draft states an unsupported observation. | Claim reviewer finds no supporting source span under rubric. | Hypothesis: generative completion or weak linker. | Affected draft and any copied content. | Reject/correct; sample similar template. | Claim-level evidence / quality lead. | Adjudicated sample; rare errors may remain. |
| F-03 Factual omission | Required rubric item absent despite source evidence. | Compare draft with reference checklist and source. | Hypothesis: template gap or compression. | Affected note completeness. | Manual correction; assess note type. | Omission rubric / quality lead. | Requires a valid reference standard. |
| F-04 Negation reversal | “No” becomes affirmative or vice versa. | Targeted negation test set and claim labels. | Hypothesis: sentence compression or tokenization. | Potentially material factual distortion. | Stop affected rendering/template. | Negation-aware test corpus / quality lead. | Constructed tests do not estimate field rate. |
| F-05 Attribution swap | Patient statement appears as clinician observation. | Compare actor/source field in evidence and draft. | Hypothesis: flattening multi-speaker transcript. | Misleading record statement. | Reject/correct and inspect packet. | Preserve speaker/source metadata / data owner. | Needs diverse source forms. |
| F-06 Medication detail error | Name, dose, route, frequency, status, or unit is wrong. | Medication-specific adjudication and unit parser checks. | Hypothesis: numeric copying or unsupported completion. | High-priority content category. | Suppress medication generation for scope. | Separate medication schema / governance lead. | No claim of clinical harm quantification. |
| F-07 Unit conversion error | Correct value rendered with wrong unit. | Unit consistency check against source representation. | Hypothesis: normalization defect. | Numeric claims in affected notes. | Reject/correct; disable conversion. | Preserve source unit and explicit transform / application owner. | Test set cannot cover every notation. |
| F-08 Wrong encounter leak | Source from encounter A displays in encounter B. | Synthetic cross-encounter request test; manifest identity audit. | Hypothesis: cache key missing encounter scope. | Confidentiality and factual integrity boundary. | Disable path; incident route. | Encounter-bound cache key / security owner. | Requires real implementation test. |
| F-09 Cross-session leak | Reviewer sees prior user's draft. | Concurrent synthetic sessions with distinct identities. | Hypothesis: shared browser/server state. | Confidentiality boundary. | Invalidate sessions; disable workspace. | Session-bound access checks / security owner. | UI tests do not prove all infrastructure paths. |
| F-10 Silent abstention | Missing source yields a short confident draft. | Inject missing required source and inspect output. | Hypothesis: default completion behavior. | Hidden incompleteness. | Replace with visible manual fallback. | Required-source gate / application owner. | Must test each note template. |
| F-11 Reviewer overload | Draft queue grows or review is rushed. | Measure pending, correction, rejection, and time distribution. | Hypothesis: capacity assumed from generation volume. | Delayed documentation and degraded review. | Limit generation; retain manual path. | Capacity gate / governance lead. | Counts do not measure attention quality alone. |
| F-12 Approval laundering | Approval click is interpreted as factual validation. | Compare approval logs with adjudication labels. | Hypothesis: metric misuse. | Misleading quality claim. | Relabel dashboards and require sampling. | Separate workflow and quality metrics / quality lead. | Adjudication still has uncertainty. |
| F-13 Forbidden write path | Service account mutates a record object. | Permission inventory and negative endpoint tests. | Hypothesis: integration scope creep. | Hard authority violation. | Revoke credentials; disable system. | Deny-list and least privilege / application owner. | Requires live configuration inspection. |
| F-14 Prompt injection | Source text attempts to alter task or request access. | Adversarial structured-source test. | Hypothesis: untrusted text treated as instruction. | Draft integrity or boundary escalation. | Escape/reject source; no tool execution. | Structured delimiters/no tools / security owner. | Test corpus is incomplete. |
| F-15 Stale source | Source was authorized but no longer current for requested workflow. | Freshness field and template requirement check. | Hypothesis: snapshot scope lacks freshness rule. | Outdated draft context. | Abstain; request new snapshot. | Freshness contract / data owner. | Local freshness requirement remains OPEN. |
| F-16 Duplicate request mix-up | Retry attaches old draft to new request. | Idempotency and request-version trace. | Hypothesis: nonunique request IDs. | Wrong draft/source pairing. | Reject duplicate; create versioned request. | Idempotency key / application owner. | Requires concurrency tests. |
| F-17 Audit gap | Draft displayed without sufficient lineage event. | Reconcile display log with manifest and review event. | Hypothesis: asynchronous logging loss. | Unreviewable affected scope. | Treat as failed request; investigate. | Append-only event contract / application owner. | Logging does not validate content. |
| F-18 Source mutation | Artifact changes after packet assembly. | Compare manifest version/hash at review. | Hypothesis: mutable source pointer. | Reproducibility failure. | Preserve immutable copy/reference. | Versioned source contract / data owner. | Retention feasibility is OPEN. |

### 12.1 Failure diagnostics and recovery traces

**Ambiguous symptom:** reviewers report that a draft “looks wrong.”

Competing cause one is factual addition by the generator.

Competing cause two is a correct statement linked to the wrong encounter.

Competing cause three is an omitted negation in source extraction.

Competing cause four is a stale source packet.

Competing cause five is a reviewer interface rendering source evidence from another draft.

The first diagnostic is to inspect request ID, encounter ID, session ID, manifest version, and source artifact identity.

The second diagnostic is to compare source availability times to cutoff.

The third diagnostic is to inspect claim text and source span for negation and attribution.

The fourth diagnostic is to compare rendered evidence IDs with the manifest.

The fifth diagnostic is to inspect whether the draft had an unsupported marker ignored by the reviewer.

The immediate containment is to prevent further generation for the affected scope if identity or session mismatch is plausible.

The immediate containment is not to silently edit the audit history.

The immediate containment is not to re-run a draft against live sources and overwrite the original evidence.

The recovery record should retain original request state, reason, affected scope, containment time, and owner.

The recovery record should distinguish confirmed cause from hypothesis.

The recovery record should define whether already displayed drafts require notification or review under a local process.

The recovery record should verify that manual documentation remains available.

**Failed dependency trace:** authorization service times out.

Precondition: no cached broad authorization token is allowed for fallback generation.

Input: valid request reaches policy service, but response times out.

Expected transition: request becomes `denied` or `abstained` with `authorization_unavailable` reason.

Expected user experience: visible message directs clinician to manual workflow.

Forbidden transition: snapshot builder uses stale allow decision without configured validity.

Forbidden transition: generator receives sources before authorization decision.

Recovery: application owner investigates dependency; new requests remain manual until the policy route is trustworthy.

Verification: constructed trace shows no manifest, no draft, and no record mutation.

**Failed dependency trace:** source adapter returns an artifact with a different encounter ID.

Expected transition: snapshot validation rejects packet.

Expected user experience: visible `source_identity_mismatch` abstention and manual fallback.

Forbidden transition: silently drop identity field and continue.

Forbidden transition: use text similarity to decide that the artifact is probably relevant.

Recovery: isolate adapter/cache path and inspect all manifests using that adapter version.

Verification: replay rejects mismatched artifact before generator invocation.

**Failed dependency trace:** review workspace loses session binding.

Expected transition: workspace denies draft display and requires reauthentication.

Expected user experience: no content shown to unbound session.

Forbidden transition: display cached draft based only on browser tab or request ID.

Recovery: invalidate affected sessions and disable review workspace if cross-session exposure is possible.

Verification: concurrent synthetic sessions cannot retrieve each other's draft identifiers.

**Failed dependency trace:** model returns malformed claim schema.

Expected transition: claim validator rejects result and emits `generation_schema_invalid` abstention.

Expected user experience: manual fallback, not a partial unmarked note.

Forbidden transition: render unvalidated prose without evidence markers.

Recovery: pin previous configuration or disable generation; inspect model/prompt/version change.

Verification: malformed-output test never reaches review approval action.

### 12.2 Why existing alarms might miss a failure

A generation-success counter can be green while source lineage is wrong.

A low latency graph can be green while review time is high.

A high approval rate can be green while omissions are unmeasured.

A zero model-error log can be green while the model is not asked to label its own errors.

A record-write log can be zero while a copy integration bypasses expected audit fields.

A session-authentication success log can be green while server-side authorization is not rechecked at display.

A cache hit metric can be green while cache keys omit encounter scope.

An abstention count can look low because missing sources are converted into confident prose.

A quality sample can look good because it excludes rejected or pending drafts.

A reviewer survey can look positive while rare medication-unit errors are absent from the sample.

Therefore monitoring must connect technical events to the bounded decision and the relevant denominator.

## 13. Transferable Lesson

| Pattern field | Content |
|---|---|
| Canonical pattern | `draft-only-human-approval`. |
| General principle | For consequential text assistance, constrain the system to evidence-bounded drafts and make the accountable human’s review, rejection, and existing write path explicit. |
| Preconditions | A defined unit of work, authorized source scope, stable time cutoff, visible provenance, manual fallback, reviewer capacity, and a nonassistant write boundary. |
| Mechanism | Fixed source manifests constrain what is presented; claim-level evidence exposes additions; abstention avoids fabricated completeness; human review and separate write permissions preserve authority. |
| Supporting evidence | This case offers derived illustrative arithmetic and constructed traces only; no outcome or deployment measurement exists. |
| Avoid when | The workflow cannot define source scope, reviewer ownership, record-write separation, or a meaningful manual fallback. |
| Portability test | Can each output claim be tied to an authorized, time-valid input, and can the user reject it without losing the baseline workflow? |
| Adaptation | In another domain, replace encounter and clinician with the domain unit and accountable reviewer, but retain provenance, abstention, and authority boundaries. |

The lesson is a design hypothesis, not an empirical generalization.

The mechanism is strongest when the output is advisory text and the human can inspect sources.

The mechanism weakens when the output must be acted on immediately without time for review.

The mechanism weakens when source artifacts are incomplete or inaccessible.

The mechanism weakens when a reviewer is nominally present but lacks time or authority.

The mechanism weakens when the assistant can mutate the system of record.

The mechanism can transfer to legal, financial, or operational drafting only after replacing the domain-specific error taxonomy.

A legal drafting system may need clause and authority errors instead of medication-unit errors.

A financial drafting system may need currency, account, and period errors instead of encounter-source errors.

A support-ticket drafting system may need customer tenancy and action-permission checks.

The common structural constraint is that a fluent draft is not a verified decision.

The common structural counterexample is a workflow where users rubber-stamp drafts because review capacity is not measured.

The first portability test is to enumerate what the assistant can read, what it can write, and who owns final action.

The second portability test is to enumerate factual categories whose errors cannot be safely averaged into one score.

The third portability test is to include abstentions and corrections in the workload denominator.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

| Relationship | Actual target | Mechanism shared or changed | Boundary learned |
|---|---|---|---|
| uses-method | [Clinical Decision Support: Evidence, Decision-Time Data, and Clinician Handoff](../Cheat_Sheets/01_Clinical_Decision_Support_Evidence_and_Handoff.md) | Uses decision-time availability, abstention, denominator, and clinician-handoff concepts. | That sheet does not validate documentation drafting. |
| uses-method | [Human Factors: Alert Design, Review, and Recovery](../Cheat_Sheets/03_Human_Factors_Alert_Design_and_Recovery.md) | Uses ownership, terminal disposition, recovery, and audit distinctions. | It is a draft sheet and has no local workload evidence. |
| contrasts-with | Planned ECG model validation plan (Case 02; no file yet) | Documentation outputs draft text; ECG case will evaluate a model-validation decision. | Do not equate source-grounded drafting with diagnostic validation. |
| uses-method | [Case Study Content Master Template](../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md) | Supplies decision, evidence, evaluation, failure, and interview structure. | Template is an authoring standard, not clinical evidence. |

### 14b. What this case leaves open

| Open question | Why current evidence is insufficient | Data or test that would resolve it | Priority/trigger | Owner role |
|---|---|---|---|---|
| Which note types have an adequately bounded source contract? | No local templates or source maps were provided. | Configuration review and source-manifest prototype. | Before any evaluation. | Governance lead and data owner. |
| What is actual manual time and its variance? | All times are constructed inputs. | Comparable observed or instrumented workflow study with declared timing rules. | Before time-saving claim. | Quality lead. |
| Can reviewers reliably label omissions and attribution errors? | No local rubric calibration exists. | Double-review calibration and adjudication exercise. | Before quality-rate claim. | Quality lead. |
| What source retention and access scheme is permissible? | Local policy and source body are OPEN. | Authoritative local governance review. | Before implementation. | Security/privacy owner. |

### Interview pack

#### Q-01 / framing / decision boundary

**Candidate prompt and explicitly given facts:** A team says, “The assistant only drafts notes, so it is harmless.” What is the bounded decision and what is wrong with that statement?

**What the candidate should clarify or assume:** Clarify note type, encounter scope, source cutoff, reviewer role, existing write path, and what “draft” can influence.

**Strong answer:** The decision is whether to authorize a draft-only evaluation versus retain manual documentation, not whether to automate care. A draft can still add, omit, negate, misattribute, or misstate units, and it may be copied into a record. The system must therefore preserve clinician approval, source provenance, abstention, and no assistant record-write capability.

**Check and conclusion:** Identify the input, output, owner, baseline, and hard boundary from Section 1; reject “harmless” as an evidence claim.

**Acceptable alternatives and conditions:** A candidate may propose template-only assistance if claim-level provenance cannot be delivered.

**Follow-up:** If the team asks for automatic note signing, the scope changes and this case no longer authorizes it.

**Red flags/common wrong answer:** “A clinician is in the loop, so all risk is transferred.” Viewing or clicking is not a quality guarantee.

**Scoring:** 0 ignores scope; 1 says human review; 2 names baseline and write boundary; 3 explains specific error categories and action consequence.

**Answer support:** Sections 1, 3, and 8.

#### Q-02 / data / time boundary

**Candidate prompt and explicitly given facts:** A source artifact describes an earlier event but was entered after `t_cutoff`. May the draft use it?

**What the candidate should clarify or assume:** Distinguish clinical event time from source availability time and verify authorization.

**Strong answer:** Not for the frozen request. The relevant condition is whether the assistant could access the authorized artifact by `t_cutoff`, not whether the event described happened earlier. Exclude it, record the reason, and create a new request/snapshot if the workflow needs the later artifact.

**Check and conclusion:** Apply `source_available_at <= t_cutoff`; do not reconstruct earlier availability from text.

**Acceptable alternatives and conditions:** A locally approved later-addendum workflow may use it only with a new explicitly versioned request.

**Follow-up:** What if the source is available but from another encounter? Reject it as an identity failure, not merely stale data.

**Red flags/common wrong answer:** “Use the most recent chart because it is more accurate.” That changes the decision-time problem.

**Scoring:** 0 uses event time only; 1 mentions freshness; 2 names cutoff and new snapshot; 3 also names authorization and audit consequence.

**Answer support:** Sections 5a, 5d, 8b, and F-01.

#### Q-03 / calculation / complete workload denominator

**Candidate prompt and explicitly given facts:** There are 120 requests, 10 ineligible, 20 abstentions, and 90 displayed drafts. What are eligible count, coverage, and abstention rate?

**What the candidate should clarify or assume:** Counts use one shift; abstentions are eligible requests routed to manual fallback.

**Strong answer:** `N_eligible=120-10=110`. Coverage is `90/110=81.82%`. Abstention rate is `20/110=18.18%`. The 10 ineligible requests are outside the service denominator, while abstentions remain inside it.

**Check and conclusion:** `90+20=110`; do not divide coverage by 120 or exclude abstentions.

**Acceptable alternatives and conditions:** Report fractions before rounding.

**Follow-up:** If 10 pending drafts exist, do they change coverage? No; coverage counts displayed drafts, but they affect review-completion reporting.

**Red flags/common wrong answer:** `90/120=75%` without explaining the changed denominator.

**Scoring:** 0 incorrect arithmetic; 1 gets one rate; 2 gets all counts; 3 explains why abstention remains in denominator.

**Answer support:** DOC-01 through DOC-03 and DOC-12.

#### Q-04 / calculation / net time

**Candidate prompt and explicitly given facts:** Use `N_draft=90`, `t_open=0.5`, `t_review=3`, `t_finalize=1`, `N_correct=36`, `t_correct=4`, `N_abstain=20`, `t_fallback=12`, `N_reject=9`, `t_reject=8`, `N_unsafe_completed=3`, and `t_unsafe_followup=10`. Manual time is 12 minutes for 110 eligible requests. Compute net time.

**What the candidate should clarify or assume:** Correction time is incremental; rejection restart is not double-counted in draft finalization; all values are illustrative.

**Strong answer:** `T_draft=90*4.5+36*4=549`. `T_abstain=20*12=240`. `T_reject=9*8=72`. `T_unsafe=3*10=30`. `T_proposed=891` minutes. `T_manual=110*12=1320` minutes. `Delta_T=429` minutes, or 7.15 hours, estimated only.

**Check and conclusion:** Verify `549+240+72+30=891`; do not call 429 a realized saving.

**Acceptable alternatives and conditions:** State 7 hours 9 minutes if converting to hours/minutes.

**Follow-up:** At what manual time does modeled benefit disappear? `891/110=8.1` minutes.

**Red flags/common wrong answer:** Compare 90 drafts against 110 manual notes while omitting abstention fallback.

**Scoring:** 0 omits components; 1 arithmetic partially correct; 2 correct total; 3 identifies denominator and evidence limitation.

**Answer support:** DOC-05 through DOC-10 and Section 6.4.

#### Q-05 / quality / plausible wrong metric

**Candidate prompt and explicitly given facts:** The dashboard reports 95% of claims have a source link. Can the team claim factual accuracy?

**What the candidate should clarify or assume:** A link can be invalid, semantically weak, stale, or attached to an incomplete claim.

**Strong answer:** No. Source-link rate measures a provenance property with a defined denominator, not factual correctness or completeness. Evaluate additions, omissions, negation, attribution, medication, and unit errors separately with source-aware adjudication.

**Check and conclusion:** Inspect whether cited IDs exist in the manifest and whether claim semantics are supported; report unsupported and ambiguous categories.

**Acceptable alternatives and conditions:** A candidate may use source-link validity as a release gate, but not as a quality conclusion.

**Follow-up:** What if all links are valid but the draft omits a required source-supported fact? The omission metric can still fail.

**Red flags/common wrong answer:** “Citations eliminate hallucination.” Citations can be irrelevant or incomplete.

**Scoring:** 0 equates citations with truth; 1 notes links may be wrong; 2 names separate taxonomy; 3 explains denominator and adjudication.

**Answer support:** Sections 5c, 5d, and 9.1.

#### Q-06 / debugging / ambiguous wrong draft

**Candidate prompt and explicitly given facts:** A clinician says a draft has the right words but the wrong person appears to have said them. How do you diagnose it?

**What the candidate should clarify or assume:** Preserve source-speaker metadata and do not assume the error is generative.

**Strong answer:** Treat this as possible attribution error. Inspect claim ID, source artifact ID, actor/source field, encounter ID, manifest version, and rendered evidence. Competing causes include flattened multi-speaker input, wrong encounter cache, source rendering mismatch, or generation rewrite.

**Check and conclusion:** Compare source speaker and claim attribution; if identity mismatch is plausible, contain the path rather than only editing text.

**Acceptable alternatives and conditions:** A candidate may first reproduce with a frozen manifest if no active leakage signal exists.

**Follow-up:** If the source belongs to another encounter, what changes? It becomes a hard access/identity incident path.

**Red flags/common wrong answer:** “Ask the model to be more careful.” That does not discriminate cause.

**Scoring:** 0 treats as grammar; 1 says inspect prompt; 2 lists source/manifest checks; 3 names containment and competing hypotheses.

**Answer support:** Sections 5d, 9.1, F-05, and Section 12.1.

#### Q-07 / security / session leakage

**Candidate prompt and explicitly given facts:** Two concurrent reviewers report seeing each other’s drafts. What is the immediate action?

**What the candidate should clarify or assume:** Determine whether this is confirmed cross-session exposure or a mistaken identifier before broad conclusions.

**Strong answer:** Stop or disable the affected review path, invalidate affected sessions, preserve audit evidence, and keep manual documentation available. Inspect server-side session binding, authorization checks at display, cache keys, and request-to-encounter mapping.

**Check and conclusion:** A cache key must include encounter, request, session/authorization scope, and manifest version; browser-tab identity alone is insufficient.

**Acceptable alternatives and conditions:** Scope containment may be narrower if evidence shows one isolated test environment, but it must be justified.

**Follow-up:** Can the team keep generating drafts while the workspace is disabled? Not if it creates inaccessible sensitive artifacts or obscures scope; retain manual path.

**Red flags/common wrong answer:** “Refresh the browser.” That may destroy diagnostic evidence and does not contain exposure.

**Scoring:** 0 only fixes UI; 1 mentions logout; 2 names disable/audit/manual fallback; 3 adds cache and authorization diagnosis.

**Answer support:** Sections 8d, F-09, and Section 12.1.

#### Q-08 / design / approval boundary

**Candidate prompt and explicitly given facts:** A product manager proposes that clicking “approve” should automatically save the note to the record. Is that within scope?

**What the candidate should clarify or assume:** Distinguish user intent, assistant action, record write, and local authorization.

**Strong answer:** No. This case requires the assistant to have no clinical record-write capability. Approval is a review event; the clinician separately authors or writes through the existing record workflow. Automatic save changes the authority and integration contract and requires a new decision.

**Check and conclusion:** Verify service permissions, endpoint deny-list, actor identity, and audit trace show zero assistant-originated writes.

**Acceptable alternatives and conditions:** A separate system might support clinician-initiated record integration only after new governance and technical controls, but it is outside this case.

**Follow-up:** Does copy-to-clipboard satisfy the boundary? Only if it does not auto-submit, auto-sign, or turn copy state into approval.

**Red flags/common wrong answer:** “The clinician clicked approve, so the assistant can save.” That conflates human intent with agent authority.

**Scoring:** 0 permits auto-save; 1 says clinician must approve; 2 separates review from write; 3 names negative permission tests and scope change.

**Answer support:** Sections 1, 6.3, 8b, and DEC-05.

#### Q-09 / evaluation / reviewer burden

**Candidate prompt and explicitly given facts:** Draft acceptance rises from 70% to 90% after a UI change, while average review time rises. Is this an improvement?

**What the candidate should clarify or assume:** Need eligible denominator, correction/rejection/pending burden, content quality, and comparable workflow windows.

**Strong answer:** It is inconclusive. Acceptance is a workflow disposition, not a utility metric. Calculate net clinician minutes including review, correction, rejections, abstentions, pending work, and any quality follow-up; inspect whether the UI made approval easier rather than drafts better.

**Check and conclusion:** Compare predeclared primary metric and content guardrails under comparable note mix.

**Acceptable alternatives and conditions:** Acceptance can be a descriptive secondary metric with explicit denominator.

**Follow-up:** What if review time rose because reviewers now inspect evidence more carefully? Then lower acceptance may not mean worse quality; adjudication is needed.

**Red flags/common wrong answer:** “90% acceptance proves the new model is better.”

**Scoring:** 0 accepts metric at face value; 1 mentions time; 2 includes all workload components; 3 explains approval laundering and comparator.

**Answer support:** Sections 3, 6, 9, and F-12.

#### Q-10 / changed constraint / abstention

**Candidate prompt and explicitly given facts:** A new source adapter becomes unreliable, doubling abstentions from 20 to 40 of 110 eligible requests. What should change?

**What the candidate should clarify or assume:** Manual fallback remains available; correction proportion and other illustrative inputs are held fixed only for the arithmetic example.

**Strong answer:** Report coverage falling to `70/110=63.64%` and abstention rising to `40/110=36.36%`. Recompute workload rather than claiming the original net time: the Section 6 stress gives 993 proposed minutes and 327 estimated saved minutes under stated assumptions. Investigate source reliability and do not hide abstentions.

**Check and conclusion:** `70+40=110`; coverage and time change together.

**Acceptable alternatives and conditions:** Pause the note type if the source is required for factual review.

**Follow-up:** If abstentions have no manual owner, can the system continue? No; the failure is workflow incompleteness.

**Red flags/common wrong answer:** “Coverage is still fine because generated drafts look good.”

**Scoring:** 0 ignores denominator; 1 calculates one rate; 2 includes time recomputation; 3 identifies fallback/eligibility decision.

**Answer support:** Section 6.4 and C-05.

#### Q-11 / medication-unit failure / changed risk category

**Candidate prompt and explicitly given facts:** A model draft correctly copies a medication number but changes the unit. Which aggregate metric could hide this, and what is the bounded response?

**What the candidate should clarify or assume:** Do not quantify medical severity; focus on documentation quality and authority boundary.

**Strong answer:** A generic factual-accuracy or acceptance rate can hide category rarity. Report medication and unit errors with applicable denominators, correct or reject the affected draft, and consider suppressing medication generation for the note scope until source representation and rendering controls are validated.

**Check and conclusion:** Compare source number and unit as a single semantic quantity; do not accept numeric string match alone.

**Acceptable alternatives and conditions:** Render medication data as an evidence-only extract rather than generated prose if local design permits.

**Follow-up:** Does clinician approval prove the unit is correct? No; use independent adjudication for estimation.

**Red flags/common wrong answer:** “The model was 99% accurate overall.” Overall rate lacks category and denominator.

**Scoring:** 0 relies on aggregate metric; 1 names unit error; 2 gives separate denominator and response; 3 distinguishes review from adjudication.

**Answer support:** Section 9.1 and F-06/F-07.

#### Q-12 / uncertainty / unsupported conclusion

**Candidate prompt and explicitly given facts:** The constructed model shows 429 fewer clinician-minutes per shift. What can and cannot be concluded?

**What the candidate should clarify or assume:** Inputs are illustrative, averages are fixed, and no actual sample exists.

**Strong answer:** It shows arithmetic consequences of stated inputs and identifies sensitivity to baseline time, correction, abstention, rejection, and unsafe-follow-up assumptions. It cannot show observed time savings, statistical uncertainty, clinical benefit, patient outcome, legal compliance, or deployment readiness.

**Check and conclusion:** Cite the 8.1-minute no-benefit boundary and absence of a confidence interval.

**Acceptable alternatives and conditions:** State that the calculation is useful for planning what to measure.

**Follow-up:** What would make an interval appropriate? A defined observed sample and analysis design that handles pairing/clustering.

**Red flags/common wrong answer:** “The system saves seven hours.” That removes evidence label and uncertainty.

**Scoring:** 0 treats model as measurement; 1 calls it illustrative; 2 names limits; 3 names sensitivity and next evidence.

**Answer support:** Sections 6, 7, 9, and 11.

#### Q-13 / recovery / failed dependency

**Candidate prompt and explicitly given facts:** The authorization service times out after a clinician submits a request. What should the system do?

**What the candidate should clarify or assume:** No broad cached token is authorized for fallback; manual workflow exists.

**Strong answer:** Deny or abstain with visible `authorization_unavailable` reason, show manual fallback, create no source manifest, generate no draft, and perform no record action. Investigate before re-enabling automated requests.

**Check and conclusion:** The audit should show request, timeout, reason, no snapshot, no draft, and no write.

**Acceptable alternatives and conditions:** A narrowly scoped, locally approved cached policy could exist only if its validity and use are defined; this case does not assume one.

**Follow-up:** Why not retry indefinitely? It delays workflow and may act on stale authorization; use bounded operational retry outside user action.

**Red flags/common wrong answer:** “Use the last successful authorization.”

**Scoring:** 0 generates anyway; 1 says show error; 2 names fallback/no snapshot; 3 explains stale authorization boundary.

**Answer support:** Section 8e and Section 12.1.

#### Q-14 / method selection / alternative

**Candidate prompt and explicitly given facts:** Why might an extractive evidence board be preferable to generated narrative?

**What the candidate should clarify or assume:** The goal is a bounded documentation task, not maximum automation.

**Strong answer:** If source provenance, corrections, or factual-category errors make generated prose costly to review, an extractive board can preserve source visibility while leaving synthesis to the clinician. It may save less authoring time, but it reduces unsupported narrative completion risk. Compare both against manual workflow with the same review-time and content rubric.

**Check and conclusion:** Do not call either option superior without a comparator and measured workflow burden.

**Acceptable alternatives and conditions:** Template-only fields may also be preferable when standardized structure dominates.

**Follow-up:** When might the generated draft still be worth evaluating? When source scope, claim linking, review capacity, and quality gates are demonstrably controlled.

**Red flags/common wrong answer:** “Use the most advanced model because it writes better.”

**Scoring:** 0 technology-first; 1 names transparency; 2 compares tradeoffs; 3 ties choice to metrics and gates.

**Answer support:** Section 5c and Section 9.

#### Q-15 / source evidence / honest status

**Candidate prompt and explicitly given facts:** What evidence remains OPEN, and why does that matter to the recommendation?

**What the candidate should clarify or assume:** Distinguish local curriculum sources from local clinical workflow and empirical evidence.

**Strong answer:** The original source body, local source/access policy, workflow map, manual timing distribution, reviewer rubric calibration, implementation permissions, and outcome measurements are OPEN. Consequently the recommendation is only to retain manual baseline and consider gated design evaluation, not to claim deployment, efficacy, compliance, or saving.

**Check and conclusion:** Review the source ledger and outcome section; no observed or published performance claim exists.

**Acceptable alternatives and conditions:** Name a first required artifact, such as a source/access contract.

**Follow-up:** If a vendor case study appears, is it enough? It may inform an alternative but does not establish this local workflow or permission model.

**Red flags/common wrong answer:** “The topic inventory proves hospitals use this design.”

**Scoring:** 0 invents evidence; 1 says more research; 2 lists material gaps; 3 ties gaps to bounded decision.

**Answer support:** Sections 2, 5e, 10, and 11.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Healthcare clinical documentation workflow; one encounter-note request at a time. |
| Problem type | Draft-only text assistance with clinician review and no autonomous clinical-record action. |
| Method | Authorized time-bounded source snapshot, constrained draft, claim evidence markers, abstention/manual fallback, clinician-controlled existing write path; manual documentation baseline. |
| Primary result | Not measured. Illustrative arithmetic estimates `429` fewer clinician-minutes across 110 eligible requests under stated inputs only. |
| Key lesson | A fluent draft is not a verified note: retain provenance, factual-category review, abstention, capacity accounting, and a zero assistant-write boundary. |
| Tags | `draft-only-human-approval`; `time-bounded-provenance`; `abstention-with-owned-fallback`. |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-M42-001 / 1.0 / 2026-11-10. |
| Evidence level | Illustrative arithmetic and constructed state traces; source-body, local workflow, policy, and outcome evidence OPEN. |
| Implementation stage | Proposed; no deployment or evaluated system claimed. |
| Baseline / primary metric | Manual documentation; net clinician-minutes per eligible encounter including review, correction, rejection, abstention fallback, and discovered unsafe-completion follow-up. |
| Key limitation | No observed workflow, quality, clinical, legal, or deployment result; approval and citation do not prove factual completeness. |
| Review trigger / owner | Any change to scope, source access, model/prompt, write permission, reviewer workflow, or governing artifact / clinical documentation governance lead. |
