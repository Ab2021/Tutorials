---
case_id: "CASE-18-01"
case_title: "Claims document extraction with evidence-linked abstention"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Document intelligence / claims intake"
primary_pattern: "human-in-the-loop"
secondary_patterns: ["evaluation-mismatch", "authorization-scope"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Claims-operations product owner"
source_index: "INDEX.md"
review_trigger: "A change to document schema, access policy, reviewer capacity, or independent gold labels"
---

# Claims Document Extraction: Evidence-Linked Fields or Reviewer Routing

## 1. The One-Line Framing

**Illustrative design exercise; not a company measurement, deployment, or claims determination.**
Given an intake bundle of rendered claim documents, produce an evidence-linked candidate for a bounded field or route it to an authorised reviewer so the claims-operations owner can reduce transcription work without releasing an unsupported value, compared with the existing human extraction workflow.
The output is one of `ACCEPT_WITH_EVIDENCE`, `ROUTE_TO_REVIEW`, `REQUEST_SOURCE`, or `REJECT_ACCESS`; it is never a claim disposition.
The binding constraints are source-page provenance, bundle/claim binding, access scope, schema version, reviewer authority, and queue capacity.
The primary success condition is that every released candidate has validated required evidence; quality and time values below are illustrative and require an independent pilot.

## 2. Context & Background

A document-intake workflow receives scans, PDFs, attachments, and corrected pages before an authorised reviewer enters structured fields into a claims system.
The decision owner is the claims-operations product owner; reviewers own final field confirmation and any consequential downstream action.
The baseline is manual review of rendered pages under existing access controls. It may be slow, but it is the comparator for accuracy, evidence, cost, and authority.
A naive OCR-only solution misses that text can be correct while its page, claim, form version, region, or permission is wrong.
This case deliberately avoids insurer-specific policies, coverage terms, adjudication, payment, fraud, or legal conclusions.

## 3. The Business Question

**Illustrative stakeholder question:** “Can intake show reviewers a small set of evidence-linked candidate fields while routing uncertain or ineligible bundles, without exporting unsupported values?”
Decision owner: claims-operations product owner, with security and schema owners controlling their respective gates.
Actions: retain manual baseline; pilot evidence-linked assistance on approved templates; request better source; route to reviewer; block access/export.
Objective: reduce reviewer transcription minutes per eligible field while maintaining required-evidence coverage and bounded accepted-field error in an independently reviewed population.
Hard constraints: no automatic claim decision; no cross-claim evidence; no output without source/version/region lineage; no bypass of access policy.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Check | Impact if wrong | Fallback | Owner |
|---|---|---|---|---|---|---|
| A-01 | Bundle-to-claim binding is available before extraction. | data | Compare intake record to page hashes. | Evidence may cross claims. | Route bundle. | Intake owner |
| A-02 | Approved templates have controlled schema versions. | data | Registry lookup. | Field semantics may differ. | Unknown-template route. | Schema owner |
| A-03 | Reviewer capacity can absorb routed volume. | operations | Queue age and staffed minutes. | Safety routing creates delay. | Throttle or keep manual. | Operations lead |
| C-01 | Access decision precedes page display and retrieval. | security | Negative authorization tests. | Disclosure risk. | Reject access. | Security owner |
| C-02 | Human approval remains required for consequential use. | authority | Workflow audit. | Model output is over-relied on. | Block export. | Operations owner |
The most design-reversing assumptions are bundle binding, template/version control, and reviewer capacity.

## 5. Data, Tools & Method


### 5a. Data Specification

- Unit of analysis: one candidate field in one rendered page region, bound to a source document version and intake bundle.
- Input keys: bundle ID, claim binding token, document version, page hash, page ordinal, region coordinates, field schema ID, access decision, and extraction run ID.
- Inputs: rendered pixels, embedded text only when it agrees with rendering, OCR tokens with coordinates, layout regions, approved template/schema registry, and immutable intake timestamps.
- Outcome: independent reviewer disposition of the candidate field, evidence adequacy, correction, or route reason; outcome is delayed and unavailable at extraction time.
- Population: authorised eligible bundles for explicitly approved templates; excluded are inaccessible, unknown-template, corrupted, and unbound pages.
- Split: claim/bundle-disjoint temporal train, validation, and test partitions; template/issuer holdout is an additional onboarding test.
- Time contract: feature availability is the intake timestamp; later correction notes, reviewer edits, and payment outcomes are prohibited from input.

### 5b. Tools & Stack

- Proposed intake service renders a canonical page image and records page hashes; it does not alter the original source.
- Proposed OCR service emits tokens, coordinates, confidence, language, and engine version.
- Proposed layout and extraction service produces field candidates with raw spans, normalized values, and schema/template version.
- Proposed policy service returns an access decision before retrieval or display.
- Proposed reviewer console shows only authorised source regions, candidate evidence, uncertainty reason, and correction controls.
- Proposed audit store retains source/version hash, run/configuration, candidate, route, reviewer disposition, and export status.

### 5c. Method Summary

1. Validate source accessibility, page integrity, bundle/claim binding, and template/schema eligibility.
2. Render the page and compare text layer with visual/OCR evidence; preserve conflicts.
3. Generate field candidates only from permitted regions under the approved schema.
4. Attach raw span, coordinates, document/page/version identity, and transformation lineage to every candidate.
5. Apply hard gates for authority, required evidence, conflicts, version, and schema validity before any score policy.
6. Apply a validated routing policy with capacity monitoring; output accepted evidence-linked candidates or explicit route reasons.
7. Let the authorised reviewer confirm, correct, reject, request source, or escalate; never convert a score into a claim decision.

### 5d. Data provenance, quality, exclusions and leakage controls

- Every source is identified by intake record, content hash, version/effective timestamp, access decision, and page hash.
- The canonical rendered image is the audit reference; embedded text is a secondary representation that may conflict.
- Deduplicate by content/page hash before splitting; retain lineage for corrected or superseded pages.
- Do not fit template selection, normalization dictionaries, thresholds, or calibration on the final test population.
- Record missingness and exclusions: inaccessible, encrypted, unbound, unknown-template, redacted, unreadable, or no-required-region pages.
- Audit earliest availability of all inputs and remove reviewer corrections, downstream status, and future replacement pages from extraction-time features.

### 5e. Reproducibility and method-selection ledger

| Artifact ID | Version / date | Purpose | Access | Limitation |
|---|---|---|---|---|
| SRC-01 | intake snapshot; illustrative | source/page identity | restricted in a real deployment | no source corpus supplied |
| CFG-01 | schema + route policy; illustrative | candidate gating | proposed | requires approval |
| RUN-01 | hand calculation below | capacity/routing example | reproducible arithmetic | not a system run |
| GOLD-01 | future independent review set | evaluation oracle | not available | outcome not measured |

## 6. Analysis Walkthrough

The decisive design choice is to gate evidence and authority before using a confidence/routing score.
**ILLUSTRATIVE workload:** 1,200 eligible candidate fields/day arrive during 8 staffed hours. A proposed policy accepts 55% with evidence and routes 45%; no acceptance is released without hard-gate success.
Accepted candidates: `1,200 × 0.55 = 660 fields/day`; routed candidates: `1,200 × 0.45 = 540 fields/day`.
If a reviewer spends an illustrative 1.8 staffed minutes per routed field, routed work is `540 × 1.8 = 972 staffed minutes/day = 16.2 staffed hours/day`.
Two reviewers provide `2 × 8 = 16 staffed hours/day`; under this simplified mean-rate model, capacity is short by `0.2 hours = 12 minutes/day`.
Therefore the proposed policy cannot be represented as sustainably staffed by two reviewers at these values; it must reduce routed volume, reduce validated review time, add capacity, or remain manual.
This is a fluid capacity calculation, not a queueing, tail-latency, quality, or deployment guarantee.

```text
FORMULA [CDX-01]: routed_work_minutes = N * r * m
WHERE: N candidate fields/day; r routed fraction [0,1]; m staffed reviewer minutes/routed field.
CONSTRAINT: constant average workload; excludes breaks, rework, arrivals variation, and escalation.
NUMERICAL EXAMPLE: 1200 * 0.45 * 1.8 = 972 staffed minutes/day.
INTERPRETATION: 972/60 = 16.2 staffed hours/day, exceeding two 8-hour reviewers by 0.2 hour.
BOUNDARY: if r or m rises, shortfall grows; capacity must not be inferred from accepted-field accuracy.
```

Sensitivity: at routed fraction 0.40, work is `1200 × 0.40 × 1.8 = 864 minutes = 14.4 hours`; two reviewers have 1.6 hours nominal headroom. At 0.50, work is 1,080 minutes = 18 hours; the shortfall is 2 hours. These differences change staffing feasibility, not correctness evidence.

## 7. The Finding

For this illustrative design, evidence-linked extraction is suitable only as a proposed reviewer-assistance workflow, not an automatic claims process.
The calculated routing workload slightly exceeds two reviewers under the stated assumptions, so the proposed 45% route policy requires adjustment or additional authorised capacity before any pilot.
No quality, cost, latency, or business result has been measured; the only verified result in this document is the displayed arithmetic.
A hard evidence/authority gate remains mandatory even if a future model has high calibrated confidence.

## 8. System/Solution Design

- `Intake source -> access/bundle gate -> canonical render + hash -> OCR/layout -> schema candidate -> evidence/conflict gate -> routing policy -> reviewer console -> approved export/audit`.
- State owned by intake: original source, version, page hashes, access decision, bundle/claim binding.
- State owned by extraction: run/configuration, OCR/layout tokens, candidates, evidence spans, gate results; it cannot mutate the original source.
- State owned by review: reviewer disposition, correction, reason, escalation; it is the only path to approve a routed field.
- Failed request path: inaccessible/unknown/unbound/conflicting inputs produce explicit route or reject states with no export.
- Recovery: a corrected source invalidates affected candidates by source version and reruns only under the updated lineage.
- Idempotency key: source version + page hash + schema version + extraction configuration; retries must not create duplicate review tasks.
- Operational readiness remains proposed: measure queue age, routing rate, evidence-gate failures, source-version invalidations, reviewer overrides, and access-denial events.

## 9. Evaluation & Validation

- Primary decision metric: accepted-field error rate among independently reviewed accepted fields, with numerator = accepted fields whose approved value/evidence is wrong and denominator = independently reviewed accepted fields.
- Coverage metric: accepted fields divided by eligible field opportunities; report separately for all received, eligible, accepted, routed, and independently sampled accepted populations.
- Evidence metric: fraction of displayed candidates with required source/version/page/region/raw-span lineage; missing lineage is a hard failure, not an accuracy miss.
- Reviewer metric: staffed minutes per routed field and correction rate, stratified by template, scan quality, language, and bundle length.
- Safety/authority metric: count of export attempts lacking access, bundle binding, required evidence, or approved schema; target is policy-defined and requires owner approval.
- Validation split: claim/bundle-disjoint, time-separated independent gold labels; test unknown templates separately rather than folding them into a friendly average.
- Comparator: authorized manual baseline and a no-assistance route; do not compare only against OCR text.
- Uncertainty: future sampling plan must report intervals and reviewed counts; no confidence interval is claimed here because no gold dataset was observed.

## 10. Action Taken

No production action was taken.
The proposed action is a gated design review: approve field/schema scope, define independent gold collection, establish reviewer-capacity policy, and run access/bundle/version negative tests before any pilot.
The owner must explicitly choose the maximum acceptable accepted-field error, required evidence fields, reviewer escalation SLA, and stop conditions.
If capacity is below the routed workload, retain manual workflow or reduce scope; do not relax evidence gates to make the chart look feasible.

## 11. Outcome & Results

**Evidence status: ILLUSTRATIVE / VERIFIED arithmetic only / NOT MEASURED operationally.**
Derived result: 540 routed fields/day at 1.8 minutes each require 16.2 staffed reviewer hours/day.
Derived sensitivity: two 8-hour reviewers are short by 0.2 hour at the stated 45% route fraction and have 1.6 nominal hours at 40%.
Not established: extraction accuracy, calibration, reviewer-time distribution, queue tail, cost, adoption, policy compliance, or any organisational outcome.
A pilot outcome may only be reported with a locked protocol, independent gold, source/version trace, specified population, time window, and approved metric definitions.

## 12. Failure Modes & What Went Wrong

### FM-01 — low-resolution fax
- Symptom: an extraction or route decision is unreliable under low-resolution fax.
- Mechanism: character strokes merge after rasterization.
- Discriminating diagnostic: OCR confidence by token and effective DPI.
- Immediate recovery: route pages below validated scan-quality threshold.
- Verification: compare reviewer field agreement after re-scan.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-02 — handwritten adjustment
- Symptom: an extraction or route decision is unreliable under handwritten adjustment.
- Mechanism: cursive differs from the printed template.
- Discriminating diagnostic: handwriting flag plus field-level confidence.
- Immediate recovery: require reviewer transcription; do not infer a value.
- Verification: audit accepted transcription against double entry.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-03 — rotated page
- Symptom: an extraction or route decision is unreliable under rotated page.
- Mechanism: orientation transform is wrong.
- Discriminating diagnostic: orientation score and text-line angle.
- Immediate recovery: rotate then rerun layout detection.
- Verification: verify anchors align to expected page coordinates.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-04 — mixed document bundle
- Symptom: an extraction or route decision is unreliable under mixed document bundle.
- Mechanism: pages belong to multiple claim packets.
- Discriminating diagnostic: claim identifier conflict and page-sequence discontinuity.
- Immediate recovery: split bundle for reviewer reconciliation.
- Verification: verify every page has exactly one approved bundle owner.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-05 — stale form version
- Symptom: an extraction or route decision is unreliable under stale form version.
- Mechanism: field label is reused with changed semantics.
- Discriminating diagnostic: template/version classifier and effective-date lookup.
- Immediate recovery: abstain unless the schema version is approved.
- Verification: compare schema hash to the source-record version.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-06 — missing policy number
- Symptom: an extraction or route decision is unreliable under missing policy number.
- Mechanism: identifier is absent or obscured.
- Discriminating diagnostic: required-field completeness check.
- Immediate recovery: route without entity linkage.
- Verification: verify no downstream join was created.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-07 — duplicate upload
- Symptom: an extraction or route decision is unreliable under duplicate upload.
- Mechanism: same pages arrive with different filenames.
- Discriminating diagnostic: perceptual hash and canonical page hash.
- Immediate recovery: deduplicate before extraction.
- Verification: verify one retained source lineage.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-08 — cropped signature region
- Symptom: an extraction or route decision is unreliable under cropped signature region.
- Mechanism: capture omits a required region.
- Discriminating diagnostic: page-boundary and required-region coverage check.
- Immediate recovery: request complete capture.
- Verification: verify region coverage against page dimensions.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-09 — table row drift
- Symptom: an extraction or route decision is unreliable under table row drift.
- Mechanism: cell boundaries shift across rows.
- Discriminating diagnostic: row/column consistency and amount-format check.
- Immediate recovery: send table to structured review.
- Verification: verify reconciled row totals.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-10 — currency ambiguity
- Symptom: an extraction or route decision is unreliable under currency ambiguity.
- Mechanism: symbol or locale formatting is ambiguous.
- Discriminating diagnostic: currency source and amount-pattern conflict.
- Immediate recovery: hold monetary field for reviewer.
- Verification: verify currency against authoritative document context.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-11 — date ambiguity
- Symptom: an extraction or route decision is unreliable under date ambiguity.
- Mechanism: numeric date order is locale-dependent.
- Discriminating diagnostic: locale evidence and impossible-date test.
- Immediate recovery: retain raw string and route.
- Verification: verify normalized date with source image.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-12 — negated clause
- Symptom: an extraction or route decision is unreliable under negated clause.
- Mechanism: a nearby negation changes the extracted status.
- Discriminating diagnostic: evidence span includes sentence and qualifier.
- Immediate recovery: require reviewer confirmation.
- Verification: verify span entails displayed status.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-13 — cross-page continuation
- Symptom: an extraction or route decision is unreliable under cross-page continuation.
- Mechanism: field value begins on one page and ends on another.
- Discriminating diagnostic: continuation markers and page adjacency.
- Immediate recovery: assemble only with bundle-level evidence.
- Verification: verify both page references remain attached.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-14 — stamp obscures text
- Symptom: an extraction or route decision is unreliable under stamp obscures text.
- Mechanism: overlay hides printed characters.
- Discriminating diagnostic: occlusion detector and image-layer check.
- Immediate recovery: abstain rather than hallucinate characters.
- Verification: verify reviewer can see an unoccluded source.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-15 — non-English rider
- Symptom: an extraction or route decision is unreliable under non-English rider.
- Mechanism: language model chooses the wrong language.
- Discriminating diagnostic: language identification confidence and script coverage.
- Immediate recovery: route to qualified-language reviewer.
- Verification: verify reviewed translation preserves field meaning.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-16 — barcode mismatch
- Symptom: an extraction or route decision is unreliable under barcode mismatch.
- Mechanism: barcode payload disagrees with visible identifier.
- Discriminating diagnostic: decode result versus OCR comparison.
- Immediate recovery: treat as conflict, not automatic correction.
- Verification: verify source-system identifier before release.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-17 — template lookalike
- Symptom: an extraction or route decision is unreliable under template lookalike.
- Mechanism: similar layout belongs to a different issuer.
- Discriminating diagnostic: issuer/logo/template evidence disagreement.
- Immediate recovery: use unknown-template route.
- Verification: verify template approval against controlled registry.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-18 — late corrected page
- Symptom: an extraction or route decision is unreliable under late corrected page.
- Mechanism: replacement page arrives after extraction.
- Discriminating diagnostic: version timestamp and page fingerprint change.
- Immediate recovery: invalidate affected fields and re-review.
- Verification: verify superseded values are not exported.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-19 — encrypted attachment
- Symptom: an extraction or route decision is unreliable under encrypted attachment.
- Mechanism: content is unavailable to extractor.
- Discriminating diagnostic: decryption/access state.
- Immediate recovery: route to authorized intake process.
- Verification: verify no placeholder extraction entered the record.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-20 — redaction overlap
- Symptom: an extraction or route decision is unreliable under redaction overlap.
- Mechanism: redaction covers a required field.
- Discriminating diagnostic: redaction geometry intersects field region.
- Immediate recovery: record unavailable rather than reconstruct.
- Verification: verify redaction policy and reviewer disposition.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-21 — long narrative note
- Symptom: an extraction or route decision is unreliable under long narrative note.
- Mechanism: field is implied rather than explicitly stated.
- Discriminating diagnostic: evidence-span absence and low entailment score.
- Immediate recovery: do not populate structured field.
- Verification: verify reviewer identifies explicit source span.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-22 — multi-claim packet
- Symptom: an extraction or route decision is unreliable under multi-claim packet.
- Mechanism: one bundle contains several valid claim identifiers.
- Discriminating diagnostic: entity clustering conflict.
- Immediate recovery: split into separate work items.
- Verification: verify each output has isolated provenance.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-23 — scan shadow
- Symptom: an extraction or route decision is unreliable under scan shadow.
- Mechanism: edge shadow resembles a digit or punctuation.
- Discriminating diagnostic: illumination/contrast diagnostic.
- Immediate recovery: reprocess image then compare candidates.
- Verification: verify stable result across approved preprocessing.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-24 — checkbox ambiguity
- Symptom: an extraction or route decision is unreliable under checkbox ambiguity.
- Mechanism: mark is faint or partially outside box.
- Discriminating diagnostic: mark coverage and neighboring-box check.
- Immediate recovery: route binary decision to reviewer.
- Verification: verify image crop and reviewer rationale.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-25 — footer leakage
- Symptom: an extraction or route decision is unreliable under footer leakage.
- Mechanism: page footer is mistaken for a field value.
- Discriminating diagnostic: region role classifier and coordinate policy.
- Immediate recovery: exclude footer region from field candidate set.
- Verification: verify no forbidden region is cited.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-26 — header carryover
- Symptom: an extraction or route decision is unreliable under header carryover.
- Mechanism: repeated header duplicates an entity name.
- Discriminating diagnostic: repetition across pages and coordinate role.
- Immediate recovery: prefer designated entity field only.
- Verification: verify field provenance is not a header.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-27 — PDF text-layer conflict
- Symptom: an extraction or route decision is unreliable under PDF text-layer conflict.
- Mechanism: embedded text differs from rendered pixels.
- Discriminating diagnostic: text-layer/rendered-token comparison.
- Immediate recovery: preserve conflict and route.
- Verification: verify reviewer selects authoritative rendition.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-28 — blank page insertion
- Symptom: an extraction or route decision is unreliable under blank page insertion.
- Mechanism: separator page shifts page numbering.
- Discriminating diagnostic: blank-page detector and source index reconciliation.
- Immediate recovery: retain page identity independent of ordinal.
- Verification: verify page-to-source mapping.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-29 — quality threshold gaming
- Symptom: an extraction or route decision is unreliable under quality threshold gaming.
- Mechanism: high confidence is produced on an easy subset.
- Discriminating diagnostic: coverage by template and scan-quality slice.
- Immediate recovery: monitor selective coverage; retain review sample.
- Verification: verify calibration on routed and accepted populations.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-30 — review queue overload
- Symptom: an extraction or route decision is unreliable under review queue overload.
- Mechanism: valid routed items wait beyond service target.
- Discriminating diagnostic: arrival/service rates and queue age.
- Immediate recovery: throttle intake or add authorized capacity.
- Verification: verify queue age and error rate recover.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-31 — reviewer override loop
- Symptom: an extraction or route decision is unreliable under reviewer override loop.
- Mechanism: same disputed field cycles between model and reviewer.
- Discriminating diagnostic: override reason codes and repeat count.
- Immediate recovery: finalize reviewer authority and create issue record.
- Verification: verify loop stops with one final disposition.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-32 — schema deployment mismatch
- Symptom: an extraction or route decision is unreliable under schema deployment mismatch.
- Mechanism: extractor emits a field absent from target schema.
- Discriminating diagnostic: contract-version validation.
- Immediate recovery: block export and roll back mapping.
- Verification: verify producer/consumer versions match.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-33 — downstream coercion
- Symptom: an extraction or route decision is unreliable under downstream coercion.
- Mechanism: text amount is silently rounded or truncated.
- Discriminating diagnostic: typed parse audit and precision comparison.
- Immediate recovery: store raw value and reject lossy coercion.
- Verification: verify exported amount matches approved value.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-34 — access-scope leak
- Symptom: an extraction or route decision is unreliable under access-scope leak.
- Mechanism: review screen shows another claim page.
- Discriminating diagnostic: authorization decision and page-to-claim binding.
- Immediate recovery: revoke session and investigate exposure.
- Verification: verify negative authorization tests.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-35 — audit-log gap
- Symptom: an extraction or route decision is unreliable under audit-log gap.
- Mechanism: extraction lacks model or source version.
- Discriminating diagnostic: required lineage fields absent.
- Immediate recovery: block release until lineage is complete.
- Verification: verify immutable audit record.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-36 — reviewer fatigue
- Symptom: an extraction or route decision is unreliable under reviewer fatigue.
- Mechanism: long bundles reduce correction quality.
- Discriminating diagnostic: error rate by bundle length and reviewer workload.
- Immediate recovery: split work and sample second review.
- Verification: verify agreement returns to control range.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-37 — rare template drift
- Symptom: an extraction or route decision is unreliable under rare template drift.
- Mechanism: new template has no labelled validation examples.
- Discriminating diagnostic: unknown-template rate and novelty embedding distance.
- Immediate recovery: abstain and create controlled onboarding set.
- Verification: verify approval before automatic route.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-38 — label leakage
- Symptom: an extraction or route decision is unreliable under label leakage.
- Mechanism: post-decision adjudication note enters input.
- Discriminating diagnostic: availability timestamp exceeds extraction cut-off.
- Immediate recovery: remove field and rerun split.
- Verification: verify point-in-time feature manifest.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### FM-39 — selective-label bias
- Symptom: an extraction or route decision is unreliable under selective-label bias.
- Mechanism: only routed documents receive gold labels.
- Discriminating diagnostic: route propensity and sampled accepted audits.
- Immediate recovery: sample accepted outputs for independent review.
- Verification: verify metrics report reviewed and all-released populations separately.
- Boundary: retain the source/version and route reason; do not silently substitute a plausible value.

### Field-decision analysis 01: low-resolution fax
**Decision grain:** `low-resolution fax` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** character strokes merge after rasterization can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use OCR confidence by token and effective DPI; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** route pages below validated scan-quality threshold; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** compare reviewer field agreement after re-scan; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `low-resolution fax` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 02: handwritten adjustment
**Decision grain:** `handwritten adjustment` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** cursive differs from the printed template can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use handwriting flag plus field-level confidence; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** require reviewer transcription; do not infer a value; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** audit accepted transcription against double entry; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `handwritten adjustment` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 03: rotated page
**Decision grain:** `rotated page` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** orientation transform is wrong can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use orientation score and text-line angle; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** rotate then rerun layout detection; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify anchors align to expected page coordinates; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `rotated page` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 04: mixed document bundle
**Decision grain:** `mixed document bundle` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** pages belong to multiple claim packets can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use claim identifier conflict and page-sequence discontinuity; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** split bundle for reviewer reconciliation; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify every page has exactly one approved bundle owner; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `mixed document bundle` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 05: stale form version
**Decision grain:** `stale form version` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** field label is reused with changed semantics can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use template/version classifier and effective-date lookup; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** abstain unless the schema version is approved; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** compare schema hash to the source-record version; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `stale form version` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 06: missing policy number
**Decision grain:** `missing policy number` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** identifier is absent or obscured can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use required-field completeness check; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** route without entity linkage; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify no downstream join was created; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `missing policy number` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 07: duplicate upload
**Decision grain:** `duplicate upload` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** same pages arrive with different filenames can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use perceptual hash and canonical page hash; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** deduplicate before extraction; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify one retained source lineage; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `duplicate upload` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 08: cropped signature region
**Decision grain:** `cropped signature region` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** capture omits a required region can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use page-boundary and required-region coverage check; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** request complete capture; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify region coverage against page dimensions; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `cropped signature region` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 09: table row drift
**Decision grain:** `table row drift` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** cell boundaries shift across rows can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use row/column consistency and amount-format check; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** send table to structured review; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify reconciled row totals; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `table row drift` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 10: currency ambiguity
**Decision grain:** `currency ambiguity` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** symbol or locale formatting is ambiguous can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use currency source and amount-pattern conflict; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** hold monetary field for reviewer; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify currency against authoritative document context; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `currency ambiguity` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 11: date ambiguity
**Decision grain:** `date ambiguity` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** numeric date order is locale-dependent can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use locale evidence and impossible-date test; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** retain raw string and route; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify normalized date with source image; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `date ambiguity` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 12: negated clause
**Decision grain:** `negated clause` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** a nearby negation changes the extracted status can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use evidence span includes sentence and qualifier; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** require reviewer confirmation; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify span entails displayed status; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `negated clause` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 13: cross-page continuation
**Decision grain:** `cross-page continuation` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** field value begins on one page and ends on another can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use continuation markers and page adjacency; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** assemble only with bundle-level evidence; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify both page references remain attached; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `cross-page continuation` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 14: stamp obscures text
**Decision grain:** `stamp obscures text` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** overlay hides printed characters can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use occlusion detector and image-layer check; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** abstain rather than hallucinate characters; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify reviewer can see an unoccluded source; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `stamp obscures text` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 15: non-English rider
**Decision grain:** `non-English rider` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** language model chooses the wrong language can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use language identification confidence and script coverage; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** route to qualified-language reviewer; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify reviewed translation preserves field meaning; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `non-English rider` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 16: barcode mismatch
**Decision grain:** `barcode mismatch` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** barcode payload disagrees with visible identifier can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use decode result versus OCR comparison; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** treat as conflict, not automatic correction; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify source-system identifier before release; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `barcode mismatch` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 17: template lookalike
**Decision grain:** `template lookalike` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** similar layout belongs to a different issuer can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use issuer/logo/template evidence disagreement; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** use unknown-template route; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify template approval against controlled registry; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `template lookalike` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 18: late corrected page
**Decision grain:** `late corrected page` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** replacement page arrives after extraction can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use version timestamp and page fingerprint change; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** invalidate affected fields and re-review; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify superseded values are not exported; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `late corrected page` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 19: encrypted attachment
**Decision grain:** `encrypted attachment` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** content is unavailable to extractor can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use decryption/access state; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** route to authorized intake process; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify no placeholder extraction entered the record; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `encrypted attachment` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 20: redaction overlap
**Decision grain:** `redaction overlap` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** redaction covers a required field can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use redaction geometry intersects field region; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** record unavailable rather than reconstruct; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify redaction policy and reviewer disposition; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `redaction overlap` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 21: long narrative note
**Decision grain:** `long narrative note` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** field is implied rather than explicitly stated can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use evidence-span absence and low entailment score; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** do not populate structured field; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify reviewer identifies explicit source span; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `long narrative note` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 22: multi-claim packet
**Decision grain:** `multi-claim packet` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** one bundle contains several valid claim identifiers can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use entity clustering conflict; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** split into separate work items; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify each output has isolated provenance; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `multi-claim packet` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 23: scan shadow
**Decision grain:** `scan shadow` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** edge shadow resembles a digit or punctuation can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use illumination/contrast diagnostic; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** reprocess image then compare candidates; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify stable result across approved preprocessing; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `scan shadow` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 24: checkbox ambiguity
**Decision grain:** `checkbox ambiguity` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** mark is faint or partially outside box can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use mark coverage and neighboring-box check; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** route binary decision to reviewer; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify image crop and reviewer rationale; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `checkbox ambiguity` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 25: footer leakage
**Decision grain:** `footer leakage` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** page footer is mistaken for a field value can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use region role classifier and coordinate policy; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** exclude footer region from field candidate set; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify no forbidden region is cited; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `footer leakage` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 26: header carryover
**Decision grain:** `header carryover` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** repeated header duplicates an entity name can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use repetition across pages and coordinate role; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** prefer designated entity field only; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify field provenance is not a header; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `header carryover` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 27: PDF text-layer conflict
**Decision grain:** `PDF text-layer conflict` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** embedded text differs from rendered pixels can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use text-layer/rendered-token comparison; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** preserve conflict and route; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify reviewer selects authoritative rendition; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `PDF text-layer conflict` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 28: blank page insertion
**Decision grain:** `blank page insertion` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** separator page shifts page numbering can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use blank-page detector and source index reconciliation; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** retain page identity independent of ordinal; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify page-to-source mapping; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `blank page insertion` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 29: quality threshold gaming
**Decision grain:** `quality threshold gaming` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** high confidence is produced on an easy subset can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use coverage by template and scan-quality slice; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** monitor selective coverage; retain review sample; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify calibration on routed and accepted populations; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `quality threshold gaming` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 30: review queue overload
**Decision grain:** `review queue overload` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** valid routed items wait beyond service target can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use arrival/service rates and queue age; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** throttle intake or add authorized capacity; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify queue age and error rate recover; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `review queue overload` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 31: reviewer override loop
**Decision grain:** `reviewer override loop` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** same disputed field cycles between model and reviewer can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use override reason codes and repeat count; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** finalize reviewer authority and create issue record; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify loop stops with one final disposition; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `reviewer override loop` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 32: schema deployment mismatch
**Decision grain:** `schema deployment mismatch` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** extractor emits a field absent from target schema can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use contract-version validation; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** block export and roll back mapping; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify producer/consumer versions match; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `schema deployment mismatch` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 33: downstream coercion
**Decision grain:** `downstream coercion` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** text amount is silently rounded or truncated can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use typed parse audit and precision comparison; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** store raw value and reject lossy coercion; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify exported amount matches approved value; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `downstream coercion` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 34: access-scope leak
**Decision grain:** `access-scope leak` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** review screen shows another claim page can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use authorization decision and page-to-claim binding; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** revoke session and investigate exposure; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify negative authorization tests; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `access-scope leak` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 35: audit-log gap
**Decision grain:** `audit-log gap` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** extraction lacks model or source version can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use required lineage fields absent; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** block release until lineage is complete; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify immutable audit record; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `audit-log gap` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 36: reviewer fatigue
**Decision grain:** `reviewer fatigue` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** long bundles reduce correction quality can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use error rate by bundle length and reviewer workload; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** split work and sample second review; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify agreement returns to control range; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `reviewer fatigue` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 37: rare template drift
**Decision grain:** `rare template drift` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** new template has no labelled validation examples can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use unknown-template rate and novelty embedding distance; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** abstain and create controlled onboarding set; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify approval before automatic route; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `rare template drift` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 38: label leakage
**Decision grain:** `label leakage` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** post-decision adjudication note enters input can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use availability timestamp exceeds extraction cut-off; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** remove field and rerun split; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify point-in-time feature manifest; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `label leakage` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

### Field-decision analysis 39: selective-label bias
**Decision grain:** `selective-label bias` is evaluated for one schema field, one page hash, and one document version; a bundle-level score cannot authorize a field-level export.
**Competing explanation:** only routed documents receive gold labels can resemble a valid extraction, while a wrong claim binding can produce the same displayed text on the wrong record.
**Evidence test:** use route propensity and sampled accepted audits; this test changes the decision only when its source and configuration version are recorded with the candidate.
**Routing consequence:** sample accepted outputs for independent review; routing is a work item with a reason code, not a missing value that downstream code may coerce.
**Acceptance boundary:** verify metrics report reviewed and all-released populations separately; otherwise retain the raw candidate and mark the field unavailable rather than normalizing uncertainty away.
**Economic consequence:** this failure consumes reviewer minutes and may delay a bundle, so measure its rate by template and scan channel before changing a global threshold.
**Version consequence:** if the page or schema changes after a candidate is produced, invalidate its export eligibility and compare the new page hash rather than copying the prior disposition.
**Counterexample:** a high OCR score does not resolve `selective-label bias` when required page-region, claim-binding, or access evidence is absent.
**Recovery evidence:** the final record must distinguish source request, reviewer correction, reviewer rejection, and accepted-with-evidence; those states have different denominators in quality reporting.

## 13. Transferable Lesson

A document system should make the unit of trust smaller than the document: one field plus its authorized source/version/page/region evidence.
Scores can prioritize work; they cannot repair missing authority, missing provenance, unknown schema semantics, or missing human decision rights.
Evaluation must preserve the operational denominator: all received, eligible, accepted, routed, reviewed, and released populations answer different questions.
Capacity is a correctness constraint when the safety design routes exceptions to people; a route policy that cannot be staffed is not a safe operational design.
The same pattern transfers to invoice, clinical-document, onboarding, and contract extraction, but each domain needs its own authorised owner, schema, evidence policy, and validation plan.

## 14. Connections & Cross-References

Related reusable material: Module 18 NLP task contracts and document intelligence sheets; Module 20 permission-aware retrieval; Module 22 document-VLM planning material. These are conceptual links, not evidence for this proposed workflow.

### Interview pack
#### Q1. Why is field accuracy not sufficient?
**Answer:** A correct string can still be from the wrong page, wrong claim, wrong form version, or unauthorized context. The release unit is an evidence-linked field decision, so evaluation must include provenance, schema validity, routing, and reviewer correction burden.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** Ask for the denominator: all documents, only reviewed documents, or only fields emitted?
**Acceptable alternatives:** Use a field metric only as one component of a declared decision metric.
**Red flags:** Claims that a high F1 alone permits automation.
**Scoring:** 3 for provenance and denominator; 2 for metric limitation; 1 for naming F1.
**Answer support:** Sections 5a, 9.

#### Q2. What is the safe baseline?
**Answer:** The baseline is authorized human extraction from the rendered source with existing intake controls. It is slower but defines the correctness, authority, and audit comparator; OCR-only is not an operational baseline.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** What baseline cost and queue data must be measured before a pilot?
**Acceptable alternatives:** A rules-assisted human workflow is acceptable if it is the actual current process.
**Red flags:** Calling manual review a failure without measuring its service level.
**Scoring:** 3 for bounded baseline and comparator; 2 for partial answer; 1 for tool name.
**Answer support:** Sections 2, 3.

#### Q3. How do you avoid leakage?
**Answer:** Freeze the document and metadata at intake time, split by claim/bundle and time, and exclude later adjudication notes, payment outcomes, and reviewer corrections from prediction-time inputs. Fit template or normalization choices within the training partition.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** How would you detect a duplicate crossing the split?
**Acceptable alternatives:** Use a held-out issuer or time period when that matches the release question.
**Red flags:** Randomly splitting pages from the same bundle.
**Scoring:** 3 for time and entity boundaries; 2 for one boundary; 1 for generic split.
**Answer support:** Sections 4, 5d, 9.

#### Q4. When should the system abstain?
**Answer:** It abstains when required evidence is missing, authority is absent, schema/template version is unknown, candidates conflict, or a validated risk/coverage policy routes the item. Abstention is a controlled output, not an empty field.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** What disposition and evidence must an abstention retain?
**Acceptable alternatives:** Escalate to an authorized reviewer or request a better source image.
**Red flags:** Treating low confidence as permission to guess.
**Scoring:** 3 for concrete trigger and disposition; 2 for threshold only; 1 for vague review.
**Answer support:** Sections 5c, 8.

#### Q5. How is calibration used?
**Answer:** Calibration estimates the observed error rate among outputs with similar scores on a declared independent population. It can set a routing policy with capacity constraints, but it does not establish validity on a new template or grant authority.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** What happens if scan-quality mix shifts?
**Acceptable alternatives:** Use conformal/risk controls only after checking their assumptions and coverage.
**Red flags:** Applying one global threshold across all issuers without slice evidence.
**Scoring:** 3 for population and boundary; 2 for threshold; 1 for definition only.
**Answer support:** Sections 6, 9.

#### Q6. Why keep the raw value?
**Answer:** Raw evidence preserves what the document actually expressed. Normalization is a separately versioned transformation, allowing a reviewer to see whether a locale, currency, or date rule changed the downstream value.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** Which value should be displayed in an audit?
**Acceptable alternatives:** Store raw image coordinates, raw string, normalized value, rule version, and reviewer disposition.
**Red flags:** Overwriting raw text with normalized data.
**Scoring:** 3 for traceability; 2 for raw retention; 1 for generic audit.
**Answer support:** Sections 5a, 5e.

#### Q7. How do you evaluate a routing threshold?
**Answer:** For each threshold, calculate accepted coverage, accepted error against independent gold, routed volume, reviewer capacity, and the error cost of accepted mistakes. Choose only among policies that satisfy hard safety/authority constraints.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** What if the reviewer queue cannot absorb the routed volume?
**Acceptable alternatives:** Throttle, defer automation, or add capacity only with an authorized operational plan.
**Red flags:** Optimizing accepted accuracy after dropping hard examples.
**Scoring:** 3 for all five quantities; 2 for metrics and capacity; 1 for accuracy alone.
**Answer support:** Sections 6, 9.

#### Q8. What is the unit of split?
**Answer:** The bundle/claim entity, not a page or field, because pages share template, claimant, identifiers, and narrative context. Time cut-offs prevent future correction patterns from leaking backwards.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** When could issuer-level holdout be needed?
**Acceptable alternatives:** Use issuer/template holdout to test onboarding robustness.
**Red flags:** Splitting one bundle across train and test.
**Scoring:** 3 for entity and time; 2 for entity; 1 for random split.
**Answer support:** Sections 5a, 9.

#### Q9. How do OCR and extraction errors differ?
**Answer:** OCR is a transcription-stage error; extraction is selection and normalization from text/layout. A correct OCR token can be assigned to the wrong schema field, and a poor OCR token can sometimes be corrected by visible evidence only through reviewer action.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** Which diagnostic isolates the stage?
**Acceptable alternatives:** Compare rendered crop, OCR tokens, candidate span, schema rule, and final output.
**Red flags:** Blaming the language model for a rasterization failure.
**Scoring:** 3 for stage separation and diagnostic; 2 for difference; 1 for tool names.
**Answer support:** Sections 5c, 12.

#### Q10. What makes a citation useful?
**Answer:** It identifies source document version, page, bounding region, raw text, extraction run, and any human correction. A page number alone is insufficient when pages are duplicated, replaced, or bundled.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** How do citations behave after a correction page arrives?
**Acceptable alternatives:** Invalidate or version all affected citations and preserve the prior audit record.
**Red flags:** Citing a retrieved summary rather than the rendered source.
**Scoring:** 3 for all provenance elements; 2 for page plus version; 1 for citation mention.
**Answer support:** Sections 5d, 8.

#### Q11. How do you estimate review cost?
**Answer:** Use released volume, accepted/routed fractions, average staffed reviewer minutes, loaded cost, rework, and exception handling. Keep proposed values illustrative until observed time-study evidence exists.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** What cost is often omitted?
**Acceptable alternatives:** Queue delay, quality sampling, onboarding, rework, and failed exports.
**Red flags:** Pricing only model calls.
**Scoring:** 3 for units and limitations; 2 for formula; 1 for a dollar total.
**Answer support:** Sections 6, 11.

#### Q12. What would change if false acceptance is much costlier?
**Answer:** Tighten hard evidence gates, raise the evidence/risk requirement, expand reviewer routing, and re-evaluate whether capacity can sustain the policy. Do not simply report a higher accuracy target.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** What if routing capacity is fixed?
**Acceptable alternatives:** Reduce scope to fields whose evidence can be validated or defer automation.
**Red flags:** Lowering the threshold to protect throughput.
**Scoring:** 3 for policy and capacity consequence; 2 for threshold direction; 1 for vague conservatism.
**Answer support:** Sections 3, 4, 6.

#### Q13. How should a new template be introduced?
**Answer:** Classify it as unknown, collect authorized examples, define the schema and evidence rules, validate by independent bundle/time splits, then explicitly approve any automatic route. Until then, route to review.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** Which success claim is forbidden during onboarding?
**Acceptable alternatives:** Do not claim aggregate performance transfers from a similar template.
**Red flags:** Auto-enrolling a visually similar form.
**Scoring:** 3 for controlled lifecycle; 2 for review first; 1 for retraining.
**Answer support:** Sections 5d, 9.

#### Q14. Why is a model confidence score not an authority signal?
**Answer:** Confidence is an internal or calibrated statistical quantity under a defined population; authority comes from business policy and an authorized human/process. A highly confident extraction can still be forbidden to release.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** Give a concrete hard gate.
**Acceptable alternatives:** Require claim binding and permission checks before display even at high confidence.
**Red flags:** Equating probability with permission.
**Scoring:** 3 for separation and example; 2 for separation; 1 for definition.
**Answer support:** Sections 3, 4.

#### Q15. How do you validate a migration?
**Answer:** Replay a frozen source set through old and new pipelines, reconcile page counts, hashes, fields, citations, permissions, deletions, and downstream schema behaviour; investigate differences rather than averaging them away.
**Reasoning:** The decision must preserve source lineage, authority, and an independently evaluable denominator before it optimizes speed.
**Follow-up:** What is the rollback unit?
**Acceptable alternatives:** Version source snapshot, parser, OCR, extractor, schema mapping, and index/output contract together.
**Red flags:** Comparing only aggregate extraction accuracy.
**Scoring:** 3 for reconciliation dimensions; 2 for replay; 1 for A/B label.
**Answer support:** Sections 5e, 9.

## 15. Quick Reference Card

| Field | Decision rule | Critical condition |
|---|---|---|
| Unit | One field + source/version/page/region evidence | Never release a bare string. |
| Baseline | Authorized manual review | Compare workflow, not OCR alone. |
| Accept | All hard evidence, authority, schema, and policy gates pass | A score is secondary to hard gates. |
| Route | Evidence conflict, unknown, low validated risk, or required review | Record route reason and source lineage. |
| Capacity | `N × r × m` staffed minutes/day | Use staffed minutes and model only declared assumptions. |
| Evaluate | Report accepted error, evidence coverage, route rate, workload, and slices | Separate reviewed, accepted, and all-received populations. |
| Evidence | Illustrative design; arithmetic verified | No deployment or quality outcome is claimed. |
| Stop | Access, bundle, source-version, schema, or authority gate fails | Block export and escalate. |
