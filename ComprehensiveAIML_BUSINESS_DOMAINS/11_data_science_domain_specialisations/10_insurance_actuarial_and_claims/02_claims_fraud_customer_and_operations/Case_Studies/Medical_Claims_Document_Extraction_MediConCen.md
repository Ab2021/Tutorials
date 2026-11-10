---
case_id: "CASE-10-02-006"
case_title: "Medical claims document extraction: separating field accuracy from safe processing"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: "production"
content_status: "draft"
case_type: "llm-agentic"
domain: "Insurance / medical claims document processing"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "cost-latency-tradeoff"]
evidence_level: "published"
source_basis: "published"
as_of_date: "2026-11-07"
owner_role: "Claims operations and data governance leads"
source_index: "INDEX.md"
review_trigger: "Material source correction, new independent evaluation, or change in intended deployment scope"
---
# Medical claims document extraction: separating field accuracy from safe processing
## 1. The One-Line Framing
Given documents belonging to one medical insurance claim, extract supported fields for a claims processor while preserving evidence and routing uncertainty to qualified review.
The decision is whether to automate a document-processing step rather than rely on a manual-first workflow.
The primary proposed success condition is accurate, timely, appropriately reviewed processing at the claim level, not merely valid output or individual-field accuracy.
The relevant baseline is manual document reading and processing; a quantitative baseline is unavailable.
The company-facing evidence is a published vendor customer story, not an independent evaluation.
The source reports an April 2025 platform launch and results but does not provide the data needed to reproduce those results.
All designs, examples, controls, thresholds and arithmetic below are explicitly hypothetical unless attributed to the source.
The scenario concerns administrative document extraction, not clinical diagnosis or advice and not an instruction to approve or deny care.
The primary reusable pattern is `evaluation-mismatch`; secondary patterns are `human-in-the-loop` and `cost-latency-tradeoff`.
The simpler question, “How accurate are extracted fields?”, omits whether all required fields for an individual claim are correct and whether processing actually benefits people.
## 2. Context & Background
### 2.1 Published setting
[Google Cloud's MediConCen customer story](https://cloud.google.com/customers/mediconcen) describes a Hong Kong-based company serving insurers and medical institutions.
It describes a blockchain-based platform for exchanging information between organizations with different internal systems.
It also describes optical character recognition (OCR), which converts image content into machine-readable text.
The story says OCR alone could not understand or summarize extracted text and insurers previously had to review every claim manually.
It says the new Google Cloud-enabled platform launched in April 2025.
The story describes an algorithm orchestrating nearly 100 Gemini instances across hundreds of smaller tasks.
It says Vertex AI is used for model fine-tuning and mentions Cloud Run and Cloud Key Management.
The source does not describe the exact orchestration graph, prompts, thresholds, population, or claim-level validation protocol.
Do not infer that the proposed architecture later in this case reproduces the company's system.
### 2.2 Workflow and stakeholders
A claim may arrive with a form, invoices, receipts, and other documents relevant to the insurer's administrative process.
The person submitting a claim expects its information to be processed correctly and without unnecessary delay.
A claims operator needs a legible, supported representation of submitted information.
An insurer retains responsibility for decisions, customer communication and applicable policies.
A document processor needs an authorized source and an unambiguous claim association.
A reviewer needs access to the relevant document region and an ability to correct or flag a field.
A platform operator needs to observe queues, failures and recovery without exposing raw sensitive content in general logs.
A privacy and security owner needs to review the purpose, access, storage and deletion of claim data.
A service owner needs a defensible accounting of turnaround, review workload and operating costs.
### 2.3 Boundaries
Actual claim volume, channels, jurisdictions, language mix, product mix, partners and peak load are not disclosed in the source.
The source mentions expansion contexts in Vietnam, Indonesia and Malaysia, but does not quantify deployment in those markets.
Neither the case nor source supplies a regulatory applicability determination.
Clinical prior-authorization outcomes are not supplied and are outside this administrative extraction case.
No inference is made about individual medical care, benefit entitlement or fraud.
## 3. The Business Question
Illustrative stakeholder question: should an insurer introduce machine-assisted extraction for claims documents, and under what evidence and controls could some cases safely bypass routine document review?
Decision owner: the insurer's accountable claims operations lead, with privacy, security, legal and technology approval as applicable.
Trigger: a proposal to reduce manual transcription and administrative waiting while retaining defensible controls.
Deadline: not provided; any deployment schedule is a scenario choice.
Available actions: retain manual processing, use OCR with manual correction, introduce assisted extraction with full review, or consider narrowly scoped automation after evidence gates.
No option is presumed implemented in the illustrative insurer workflow.
The operational objective is to improve correctly completed claims per unit time without unacceptable errors, delays, privacy exposure or inequitable performance.
The primary proxy is the proportion of eligible claims with every required extracted field correct against an adjudicated reference.
The business outcome must also measure the complete end-to-end process, not extraction in isolation.
A rise in straight-through processing can hide an increase in wrongly processed claims if quality is not measured in both reviewed and unreviewed paths.
The no-change option remains credible if the data, audit coverage or human recovery capacity is insufficient.
Clinical judgment, treatment recommendations and prior-authorization decisions are excluded.
## 4. Assumptions & Constraints
### 4.1 Scenario assumptions, not company facts
A-01: The intake system can assign each document to one claim and tenant.
Check: reconcile source identifiers and attachments against a protected intake manifest.
If false, stop automated processing and route to an authorized intake specialist.
Owner: intake operations; trigger: mismatched or missing manifest identifiers.
A-02: Authorized reference labels can be obtained without exposing data beyond the evaluation team.
Check: confirm permissions, rubric, available reviewer time and access boundaries.
If false, restrict the pilot to manual work until a defensible evaluation set exists.
Owner: privacy and evaluation leads; trigger: new data use or retention purpose.
A-03: Human reviewers can understand the documents and resolve ambiguities.
Check: sample document languages and quality; assess reviewer agreement and coverage.
If false, provide appropriate language expertise or defer automation for the affected segment.
Owner: claims operations; trigger: new language or document type.
A-04: Users can recover a misrouted claim without losing source evidence or corrections.
Check: exercise reconciliation and replay using permitted test data.
If false, do not automate downstream handoff.
Owner: platform operations; trigger: changes to storage or claims-system interfaces.
A-05: Source pages contain sufficient evidence for the required fields.
Check: measure absent and unreadable evidence separately from wrong extractions.
If false, request missing material through the authorized workflow rather than inventing a value.
Owner: claims operations; trigger: source template or upload changes.
### 4.2 Constraints to establish before release
C-01: Access must be restricted to permitted claim data and approved purposes; exact obligations require local legal review.
C-02: An extraction system must not autonomously make clinical decisions.
C-03: A claimed field must point to source evidence or be explicitly marked missing, ambiguous or unsupported.
C-04: Review queues must have capacity and escalation routes before any automated routing is enabled.
C-05: External service terms, residency, retention, encryption and subprocessors require customer-specific verification.
C-06: A model confidence score is not a calibrated probability unless independently validated for the field and population.
C-07: Release owners must decide quantitative error, latency, cost and slice limits before examining confirmatory results.
The three highest-leverage unknowns are label quality, claim/document linkage, and how errors differ by population slice.
If any fails, the proposed automation recommendation can reverse even if an aggregate field score looks favorable.
## 5. Data, Tools & Method
### 5a. Data Specification
The claim is the unit for workflow success; a field instance within a claim is the unit for field-level extraction.
A claim may include multiple pages, multiple documents and repeated field types.
A field instance is identified by tenant, claim, document, page or region, semantic field name and occurrence number.
The ground truth is the value adjudicated from authorized source documents, not automatically a later model output.
A reference record should distinguish absent, unreadable, conflicting, not applicable and intentionally redacted information.
Dates should include an explicit interpretation and ambiguity flag; amounts need both value and currency.
A claim received timestamp and a document availability timestamp are distinct from a later correction timestamp.
Do not use later documents or downstream adjudication to represent information available at initial extraction.
The source does not disclose data volume, sample window, time zone, entity counts, inclusion criteria or gold-label instructions.
Consequently, no company extraction score can be independently reconstructed from the public story.
For a proposed evaluation, freeze an eligible incoming cohort, document the time window, and keep all pages and repeats from one claim in the same split.
Where feasible, separate providers or templates across a robustness holdout to expose template memorization.
Keep temporally later data untouched for a final evaluation and record label maturation cutoffs.
Reconcile document counts, exclusions, joins and duplicates before scoring; report the excluded population.
### 5b. Tools & Stack
Published: OCR, Gemini, Vertex AI fine-tuning, Cloud Run and Cloud Key Management are named in the [customer story](https://cloud.google.com/customers/mediconcen).
Published: the story describes task orchestration and a blockchain-based information-exchange architecture.
Published: it mentions BigQuery as a planned fraud-detection use, not evidence of an already operating fraud solution.
Unknown: exact model and API versions, prompt content, topology, storage schema, hardware, costs and failure handling.
Proposed below: an abstract authenticated intake, document store, extractor, evidence validator, reviewer queue and claims-system adapter.
The proposal deliberately does not assume that a particular cloud product or proprietary feature implements any control.
### 5c. Method Summary
Manual-first baseline: trained personnel read source pages and enter fields; it is interpretable but consumes reviewer capacity.
OCR plus human correction: a text layer assists entry but may lose layout or misread characters; performance is not tested here.
Assisted extraction: an extraction component proposes structured fields and evidence, with routine human review; this is a proposed pilot.
Conditional automation: an insurer may consider a restricted no-routine-review path only after independent quality and operational gates pass.
No alternatives have been compared experimentally in this case.
Extraction input: authorized document bundle and a versioned requested-field schema.
Transformation: normalize pages and preserve the relationship between text, coordinates and original source.
Output: structured candidate values, field status, source references and version identifiers.
Post-processing: validate formats, reconcile evidence and flag contradictions; never turn a syntactically valid value into an assumed true value.
Decision: send the package to human review or a separately authorized downstream processing path.
### 5d. Provenance, quality and leakage
Original submitted documents remain the authoritative evidence; transformed text is a derived artifact.
Each proposed transformation retains its input document version and page mapping.
A protected manifest associates each document version with a claim and authorized tenant.
Missing attachments, duplicate uploads and incompatible versions need explicit states, not silent dropping.
The evaluator must separate source absence from extraction error and report both.
A corrected claim may contain information unavailable when the first processing decision occurred.
Freeze decision-time inputs for evaluation and do not leak later corrections into the extractor's input.
A model improvement dataset must not accidentally contain the final holdout claims or near-duplicate copies.
Use authorized de-identification and retention controls appropriate to the actual context; de-identification is not assumed perfect.
The source crosswalk refers to source directories that are unavailable; their contents were not inspected and support no claim here.
### 5e. Reproducibility and method-selection ledger
SRC-01: [MediConCen customer story, Google Cloud](https://cloud.google.com/customers/mediconcen), public web page, accessed 2026-11-07; publication date is not stated in the page text.
SRC-01 supports company/vendor-reported launch, architecture descriptions and headline results, subject to the source's missing methodology.
SRC-02: [Google SRE workbook, implementing SLOs](https://sre.google/workbook/implementing-slos/), method reference, accessed 2026-11-07; it is not evidence of MediConCen's service levels.
No source dataset, claim-level labels, execution logs, configuration or independent evaluation artifact is available here.
RUN-01: arithmetic in Section 6 is hand-constructed for teaching; it is not a system run or company measurement.
A real reproduction would require an approved data snapshot, reviewer rubric, model and prompt versions, code revision, runtime configuration and evaluation outputs.
Record a release manifest for schema, prompts, model version, preprocessing, policy checks and routing thresholds together.
A seed alone would not guarantee exact reproducibility of a hosted or nondeterministic model.
### 5f. Proposed field-contract register
The following field contracts are hypothetical examples, not a statement of MediConCen's schema.
Every example requires an authorized source and a documented reason for collecting the field.
A missing or contradictory source is an explicit state, never permission to guess.
#### Field 01: Claim identifier
Meaning: the intake claim key.
Proposed representation: exact identifier.
Required provenance: linkage to the authorized intake manifest.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a digit substitution joins another claim.
Reviewer response: verified intake key.
Proposed quality check: exact match after allowed normalization.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 02: Document identifier
Meaning: the submission document key.
Proposed representation: exact identifier.
Required provenance: manifest entry and document version.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: duplicate identifiers collapse distinct attachments.
Reviewer response: verified manifest association.
Proposed quality check: one-to-one or explicitly versioned match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 03: Page number
Meaning: the source page index.
Proposed representation: positive integer.
Required provenance: original page ordering.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a detached page points reviewers to unrelated evidence.
Reviewer response: page manifest.
Proposed quality check: index and document-version reconciliation.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 04: Member identifier
Meaning: the member key printed on an authorized document.
Proposed representation: exact identifier.
Required provenance: visible identifier and intake context.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: an adjacent member number is selected.
Reviewer response: manual identity reconciliation.
Proposed quality check: exact-match scoring on eligible values.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 05: Patient name
Meaning: the submitted name.
Proposed representation: text with language metadata.
Required provenance: source region and accepted spelling rules.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: transliteration or similar names cause a false merge.
Reviewer response: authorized identity workflow.
Proposed quality check: exact or preapproved equivalence rubric.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 06: Insurer name
Meaning: the named insurance organization.
Proposed representation: text.
Required provenance: document heading or explicit designation.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: provider name is mistaken for insurer.
Reviewer response: manual role review.
Proposed quality check: semantic role and value accuracy.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 07: Provider name
Meaning: the named submitting provider.
Proposed representation: text.
Required provenance: source region with provider role.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a billing agent is mistaken for the provider.
Reviewer response: manual role review.
Proposed quality check: semantic role and value accuracy.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 08: Provider identifier
Meaning: the provider identifier on the source.
Proposed representation: exact identifier.
Required provenance: source region and identifier scheme.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: an unrelated tax or invoice number is selected.
Reviewer response: manual identity reconciliation.
Proposed quality check: scheme and exact-value match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 09: Invoice number
Meaning: the invoice identifier.
Proposed representation: exact identifier.
Required provenance: invoice heading and document association.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a receipt number is mistaken for invoice number.
Reviewer response: manual reconciliation.
Proposed quality check: exact-value and document-link match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 10: Receipt number
Meaning: the receipt identifier.
Proposed representation: exact identifier.
Required provenance: receipt heading and source region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: an invoice number is copied into the receipt field.
Reviewer response: manual reconciliation.
Proposed quality check: exact-value and document-link match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 11: Service date
Meaning: the date explicitly describing a service.
Proposed representation: date with interpretation.
Required provenance: source date label and supporting region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a billing date is assigned as service date.
Reviewer response: review date role and ambiguity.
Proposed quality check: role-aware date comparison.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 12: Invoice date
Meaning: the invoice issue date.
Proposed representation: date with interpretation.
Required provenance: invoice date label.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a service date is assigned as invoice date.
Reviewer response: review date role and ambiguity.
Proposed quality check: role-aware date comparison.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 13: Submission date
Meaning: the recorded submission date.
Proposed representation: timestamp with zone when available.
Required provenance: intake event rather than later processing time.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a model supplies a document date as intake time.
Reviewer response: use authoritative intake event.
Proposed quality check: timestamp and provenance match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 14: Service description
Meaning: the description as written.
Proposed representation: text.
Required provenance: bounded source region without clinical inference.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a summary adds unsupported detail.
Reviewer response: retain original wording for review.
Proposed quality check: source-supported semantic comparison.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 15: Service code
Meaning: the code explicitly shown.
Proposed representation: code and coding-system name.
Required provenance: source label and code context.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a number is interpreted under the wrong code system.
Reviewer response: review code-system and source.
Proposed quality check: code plus system exact match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 16: Quantity
Meaning: the stated number of units.
Proposed representation: number and unit.
Required provenance: line-item source region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a quantity is read from an adjacent row.
Reviewer response: review line-item alignment.
Proposed quality check: numeric and line association match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 17: Unit price
Meaning: the stated price per unit.
Proposed representation: decimal and currency.
Required provenance: line-item source region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: decimal or currency is silently changed.
Reviewer response: review amount and currency.
Proposed quality check: exact decimal and currency match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 18: Line amount
Meaning: the amount for one invoice line.
Proposed representation: decimal and currency.
Required provenance: line-item source region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: an adjacent line amount is attached to the service.
Reviewer response: review row association.
Proposed quality check: amount, currency and row match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 19: Subtotal
Meaning: the stated subtotal.
Proposed representation: decimal and currency.
Required provenance: labeled subtotal region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a grand total is misread as a subtotal.
Reviewer response: review subtotal label.
Proposed quality check: value and semantic role match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 20: Tax amount
Meaning: the stated tax amount.
Proposed representation: decimal and currency.
Required provenance: labeled tax region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: an unlabeled fee is assumed to be tax.
Reviewer response: review original label.
Proposed quality check: value and explicit-presence match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 21: Discount amount
Meaning: the stated discount.
Proposed representation: decimal and currency.
Required provenance: labeled discount region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a negative line item is treated as a discount without support.
Reviewer response: review original label.
Proposed quality check: value and explicit-presence match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 22: Total billed
Meaning: the stated total billed.
Proposed representation: decimal and currency.
Required provenance: invoice total region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a prior balance or paid amount is substituted.
Reviewer response: review total and line reconciliation.
Proposed quality check: value, currency and role match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 23: Amount paid
Meaning: the stated paid amount.
Proposed representation: decimal and currency.
Required provenance: payment-specific source region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: the amount billed is treated as paid.
Reviewer response: review payment evidence.
Proposed quality check: value and role match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 24: Currency
Meaning: the explicitly stated currency.
Proposed representation: currency code or unresolved marker.
Required provenance: currency symbol and context.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a shared currency symbol is mapped to the wrong currency.
Reviewer response: retain ambiguity for review.
Proposed quality check: document-supported code match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 25: Document language
Meaning: the language of a relevant region.
Proposed representation: language tag or mixed marker.
Required provenance: page text and reviewer assessment.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a mixed-language page is assigned a single unsupported language.
Reviewer response: language-capable review.
Proposed quality check: region-level language agreement.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 26: Document type
Meaning: the document function.
Proposed representation: controlled type or unknown.
Required provenance: document structure and source context.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a receipt is classified as an invoice.
Reviewer response: review document association.
Proposed quality check: type and unknown-state accuracy.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 27: Submission channel
Meaning: the authenticated intake channel.
Proposed representation: controlled channel identifier.
Required provenance: intake metadata.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a page header is mistaken for an upload channel.
Reviewer response: use authoritative intake metadata.
Proposed quality check: event provenance match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 28: Attachment count
Meaning: the number of received source attachments.
Proposed representation: nonnegative integer.
Required provenance: intake manifest.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: missing pages are hidden by an incorrect count.
Reviewer response: reconcile manifest and stored objects.
Proposed quality check: count and manifest match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 29: Line-item count
Meaning: the number of visible invoice lines.
Proposed representation: nonnegative integer.
Required provenance: source invoice region.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: continuation pages are omitted.
Reviewer response: review page completeness.
Proposed quality check: line-count and page-coverage match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
#### Field 30: Correction status
Meaning: whether an authorized correction exists.
Proposed representation: versioned status.
Required provenance: correction event history.
Eligibility: score only where the reference rubric defines an applicable value.
Missing-source handling: record absent or unreadable rather than manufacturing a value.
Ambiguous-source handling: preserve the competing evidence and route for adjudication.
Principal failure: a stale extracted value overwrites an approved correction.
Reviewer response: reconcile event sequence.
Proposed quality check: state-transition and version match.
Privacy boundary: show only the minimum authorized evidence needed for this field.
Version boundary: retain the source and schema versions used to produce the candidate.
Operational boundary: a correct value alone does not authorize a claim decision.
## 6. Analysis Walkthrough
### 6.1 Source claim ledger
CLM-01: The source says the Google Cloud-enabled platform launched in April 2025; this is a published deployment claim, not proof of the proposed design.
CLM-02: The source reports 98% field-level extraction accuracy; the field definition, denominator, sample and uncertainty are not given.
CLM-03: The source reports a 100% higher straight-through processing rate; the baseline and absolute rate are not given.
CLM-04: The source reports claim processing time falling from days to hours; distributions and measurement windows are not given.
CLM-05: The source describes nearly 100 Gemini instances orchestrated across hundreds of smaller tasks and use of Vertex AI fine-tuning.
All five claims are attributed to [Google Cloud's MediConCen customer story](https://cloud.google.com/customers/mediconcen), accessed 2026-11-07.
The narrative attributes improvements to its technology; because no evaluation design or control is supplied, this case does not establish causality.
### 6.2 Worked calculation: relative and absolute rates
Illustrative only: suppose the initial straight-through processing rate is 20% of eligible claims and the final rate is 40%.
Absolute change = 40% − 20% = 20 percentage points.
Relative change = (40% − 20%) / 20% = 1.00 = 100% higher.
A 100% relative increase is doubling, not a 100-percentage-point increase.
An initial rate of 20% and final rate of 40% are constructed solely to demonstrate arithmetic; neither is a company measurement.
If a baseline rate is unknown, no absolute point change or final rate can be computed from the relative change alone.
The source does not supply the baseline or final rates and does not define the population used for the stated change.
### 6.3 Worked calculation: field versus claim
Illustrative only: consider 100 claims with exactly five required field instances each, or 500 required field instances.
Suppose 490 field instances are correct and ten are wrong: field accuracy is 490 / 500 = 98%.
If the ten errors occur on ten different claims, 90 / 100 = 90% of claims have all five required fields correct.
If all ten errors are concentrated into two claims, 98 / 100 = 98% of claims have all five required fields correct.
Both arrangements have identical 98% field accuracy and very different all-required-fields-correct rates.
The examples assume every field is scored, every claim has the same number of required fields and no other errors occur.
Neither arrangement describes the company, and neither determines financial, clinical or customer impact.
### 6.4 Optional independence illustration
Under the additional illustrative assumption that each of five fields is independently correct with probability 0.98, the probability all five are correct is 0.98^5 ≈ 0.9039.
That is approximately 90.39%, not 98%.
The independence assumption is often unreasonable because page quality, language and template errors can affect several fields together.
The previous explicit allocation example does not require independence and directly shows why per-claim scoring is needed.
### 6.5 Sensitivity and unresolved comparison
If required fields increase to ten under the same illustrative independence assumption, 0.98^10 ≈ 0.8171, or 81.71%.
The change from five to ten fields changes the hypothetical all-correct probability by approximately −8.68 percentage points.
This sensitivity is mathematical, not an estimate of a real workload.
The highest-leverage evaluation step is to measure complete claims with adjudicated source evidence and representative field counts.
No initial approach was actually tested and rejected in this case; the alternatives above are only considered.
A different claim mix, intervention, labeling process or review policy could change the observed result.
## 7. The Finding
The defensible decision is to gather claim-level, slice-level and workflow evidence before authorizing unreviewed processing in a new setting.
The [customer story](https://cloud.google.com/customers/mediconcen) reports 98% field-level extraction accuracy, a 100% higher straight-through rate and processing time reduced from days to hours, but supplies neither the evaluation design nor enough baseline detail to reproduce or attribute those results.
The illustrative calculations show that a field-level score does not establish an all-required-fields-correct rate, and doubling a rate is not a 100-percentage-point increase.
A controlled, privacy-reviewed evaluation with representative adjudicated claims and a human fallback is the next justified action, not proof that any particular automation threshold has been met.
## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
The following diagram and all associated interfaces are a proposed design, not an assertion about MediConCen.
```text
Authorized source + intake manifest -> protected snapshot -> eligibility and deduplication
    -> reviewer labeling + independent adjudication -> time/group split
    -> candidate configuration -> frozen evaluation -> approval decision
    -> versioned model/prompt/schema/routing bundle -> controlled release
                                      \-> failure: reject bundle and retain prior workflow
```
Training is optional; a system may use a fixed service without fine-tuning.
If fine-tuning is proposed, document the authorization basis, training snapshot, exclusion rules, model version and deletion implications.
Keep the final test cohort separate from threshold tuning and from model-improvement feedback.
Reviewer corrections are selected operational labels and cannot automatically be treated as a representative evaluation set.
An evaluation steward should maintain independent random audits of automatically processed claims, subject to authorization.
### 8b. Online / Serving / Real-Time Path
```text
Authenticated sender -> tenant and claim authorization -> manifest + protected document store
    -> schema/page checks -> extraction candidate + source references -> deterministic validation
    -> routing decision -> human review and correction -> authorized claims adapter
                        \-> eligible automation gate -> authorized claims adapter
                        \-> invalid/ambiguous/dependency failure -> quarantine + review
    -> protected audit events + reconciliation <- adapter acknowledgment and reviewer feedback
```
The authenticated intake identifies the tenant, source system, claim key, document version and permitted purpose.
The document store owns source versions; a manifest records attachment identities and integrity checks.
The extractor returns candidates, evidence references and statuses, never an unqualified instruction to approve treatment.
The validator rejects schema failures and identifies contradictions, missing evidence and unresolved values.
The router sends uncertain, unsupported, excluded or failed work to an authorized queue.
The claims adapter sends a versioned candidate package and waits for a durable acknowledgment.
A downstream business decision remains under the insurer's separately governed process.
A protected audit event records transitions and versions without routinely copying document text into general logs.
### 8c. Key Design Decisions
DEC-01: Keep source evidence alongside candidates; alternative is value-only extraction.
Tradeoff: evidence retrieval adds storage and access complexity but supports meaningful correction.
Failure boundary: a source reference that points to the wrong page is worse than an explicit missing reference.
DEC-02: Default to review for missing evidence or unsupported types; alternative is automatic completion.
Tradeoff: review costs more time but avoids converting ambiguity into an invented value.
DEC-03: Version model, prompt, schema and policy together; alternative is independent release.
Tradeoff: coordinated releases slow iteration but simplify attribution and rollback.
DEC-04: Audit a probability sample from automation as well as review; alternative is using corrections alone.
Tradeoff: audit effort consumes capacity but reduces selection bias in measured quality.
DEC-05: Use a protected intake manifest and idempotent adapter; alternative is implicit object names and blind retries.
Tradeoff: state management adds implementation work but makes duplicates and loss detectable.
All decisions are proposed and require validation, approval and a bounded rollout before use.
### 8d. Interfaces and operational readiness
Intake request: tenant_id, claim_id, submission_id, document_id, version, content reference and event timestamp.
Extractor request: authorized content reference, requested schema version, document version and processing purpose.
Extractor response: field name, occurrence, value or explicit null state, evidence location, candidate version and error status.
Evidence location: document version, page identifier and region reference when available.
Adapter request: claim ID, package version, idempotency key, field payload and provenance references.
Adapter response: accepted, rejected with reason, or unknown outcome requiring reconciliation.
Human review event: reviewer role, field decision, correction reference, reason code and timestamp.
A model confidence value, if included, is descriptive until field-specific calibration supports its intended use.
Authorization is checked at both request and evidence retrieval, not just at the initial upload.
A tenant identifier is not itself proof of authorization.
Duplicate delivery must not create duplicate downstream side effects.
The idempotency key should identify a logical claim package version, not merely an attempt.
Retries use bounded attempts and backoff; ambiguous downstream outcomes require lookup before reissue.
Malformed or unauthorized requests fail closed; errors return non-sensitive codes and an operational correlation ID.
A stale document or schema version must not silently replace a newer reviewer correction.
The state owner records received, extracting, review-pending, ready, submitted, acknowledged and quarantined states.
State transitions should use compare-and-set or an equivalent explicit concurrency rule.
A source correction creates a new immutable document version and invalidates dependent candidates as needed.
Backup and restore must include manifests, versions, state and audit references, not only the document objects.
Reconciliation compares intake manifests, stored objects, processing states and downstream acknowledgments.
Proposed service indicator: fraction of eligible submissions receiving a completed supported extraction package within an agreed time window.
Its numerator, denominator, window and target require owner approval; no numerical service objective is established here.
Measure end-to-end latency from authenticated receipt to acknowledged package; separately record queue, extraction and review times.
Do not add individual component 99th percentiles and call the sum the end-to-end 99th percentile.
At cold start or overload, apply bounded admission and queue work rather than bypassing checks.
If extraction dependencies fail, preserve intake and route to manual processing within agreed capacity.
If review capacity is exhausted, hold work safely and escalate; do not silently relax review criteria.
If the adapter times out, inspect downstream state before retrying an externally visible operation.
Rollback restores a compatible model, schema, routing and adapter bundle; incompatible in-flight work needs explicit migration or review.
Privacy owners define retention, deletion, access review and incident response for source and derived data.
Security controls should cover identities, least privilege, encryption, keys, secrets, tenant isolation and audit access.
A vendor's certification or key-management feature is not proof that a particular deployment complies with applicable requirements.
### 8e. Illustrative capacity and cost model
The following quantities are constructed assumptions, not observed company data or quoted service prices.
Suppose 1,000 claims arrive in a day and each requires an average of 4 pages: 1,000 × 4 = 4,000 pages per day.
Suppose 25% of the day's claims arrive in one hour: 250 claims per peak hour, or about 4.17 claims per minute.
At four pages each, the peak-hour workload is 1,000 pages, or about 16.67 pages per minute.
If one worker can sustainably process 100 pages per hour under a benchmark, the arithmetic lower bound at the peak is 1,000 / 100 = 10 workers.
The assumed throughput is not a measurement; the lower bound omits variance, failures, headroom and review.
If review requires 3 minutes for each of 250 peak-hour claims, workload is 750 reviewer-minutes, or 12.5 reviewer-hours.
At 60 productive minutes per reviewer-hour, that is an arithmetic lower bound of 12.5 concurrent reviewers for that peak hour.
Real staffing needs account for breaks, uneven arrivals, skill mix, rework and service targets.
If only 20% of the 250 claims require review, the same illustrative assumptions give 50 × 3 = 150 minutes, or 2.5 reviewer-hours.
The lower review scenario is not a recommended routing rate and must not bypass quality gates.
Cost per correctly processed claim = total fully allocated processing cost / number of claims verified correct under the agreed rubric.
Total cost includes ingestion, pages or requests, model tokens, compute, storage, networking, retries, review labor and failed work.
No current prices are asserted; obtain dated price quotes and include contract discounts and minimum charges before budget approval.
The denominator must include errors found by independent audit; do not count an unreviewed claim as correct merely because it was processed.
Track capacity by document size, language, concurrency, queue age and workload spikes rather than daily average alone.
## 9. Evaluation & Validation
### 9.1 Proposed primary metric
Primary metric: number of eligible claims with every required field correct, source-supported and associated with the right claim divided by number of eligible claims scored.
The unit is a claim; the analysis window, required field set and eligibility rules must be fixed before the confirmatory audit.
A claim with a missing required extraction is not silently excluded from the denominator.
Where the source itself lacks required evidence, report a distinct source-incomplete disposition and its prevalence.
Predefine whether source-incomplete claims fail the end-to-end metric or form a separate explicitly reported stratum.
Field-level accuracy is the number of correctly extracted eligible field instances divided by the number of eligible reference field instances.
Measure field-specific accuracy and missing, wrong, unsupported and ambiguous outcomes separately.
Straight-through processing rate is the count of eligible claims completed without routine human intervention divided by all eligible claims in the same cohort.
Specify what counts as routine intervention and how post-processing audit or later correction is handled.
Report correctly processed straight-through claims separately from gross straight-through rate.
Turnaround is measured from a defined receipt timestamp to a defined completed-processing timestamp, including waiting and rework.
Report the distribution, including median and tail percentiles, along with incomplete or censored claims.
### 9.2 Gold labeling and cohort
Construct a representative eligible cohort from a documented time window and include all channels and document types in scope.
Sample before seeing model outcomes and retain sampling probabilities for stratified estimates.
Use language-capable, qualified reviewers and a written field-by-field evidence rubric.
Blind initial reviewers to the model's proposed value where practicable to reduce anchoring.
Use independent duplicate labeling on a sample and adjudicate disagreements using original source evidence.
Measure reviewer disagreement, missing evidence and adjudication rates by field and slice.
Include unreviewed automated claims in an independent audit; human-corrected records alone are selected labels.
Cluster uncertainty estimates at least at the claim level because fields within a claim are dependent.
Consider provider, template or time clustering when those sources of dependence are material.
Report confidence intervals and the sampling approach, including how small slices are handled.
A small slice with too few labels is an uncertainty finding, not evidence that the slice passes.
A predeclared randomization design could compare assisted and baseline workflows if ethical and operational approval permits.
Randomize at a unit that limits contamination, document eligibility and exposure, and monitor sample-ratio imbalance.
Prespecify the analysis population, observation window, stopping rules, intervention and outcome definitions.
Without a valid comparator, changes over time may reflect workload, policy, population or staffing differences.
### 9.3 Proposed deployment gates, not passed tests
Before data access, privacy and security owners approve the specific data purpose, access and retention plan.
Before shadow use, document owners approve schema, source references, error handling and a representative evaluation rubric.
Before assisted use, reviewers demonstrate that they can see evidence, correct fields and escalate ambiguity.
Before any no-routine-review path, owners set and test field, complete-claim, slice and adverse-impact thresholds in advance.
Before release, operations verifies adequate review and escalation capacity under expected and peak load.
Before promotion, the team checks independent audit coverage, uncertainty bounds, cost and end-to-end latency.
A slice with insufficient evidence is excluded from automation or routed to review, not silently treated as passing.
At any stage, missing data authorization, broken claim linkage or uncontrolled side effects blocks promotion.
Thresholds, statistical power, evaluation dates and actual pass/fail results are not available in the public story.
This case reports no executed experiment or passed gate.
### 9.4 Minority slices and fairness
Define slices with a defensible purpose, sufficient privacy protection and lawful access to relevant attributes.
Examples include document language, handwriting, scan quality, document type, channel and provider template.
Protected or sensitive attributes must not be inferred from clinical records merely to create a slice.
For every slice, report sample size, source-incomplete rate, field accuracy, claim completeness, review rate and uncertainty.
Investigate a gap before increasing automation; verify whether it arises in capture, extraction, review or downstream handoff.
Where sample sizes are small, gather more appropriate evidence or retain manual review instead of declaring equality.
Avoid publishing small-cell counts or identifiable examples in unrestricted reports.
Monitor whether reviewer language coverage and queue delays vary by slice as well as model extraction errors.
## 10. Action Taken
Published action: the [customer story](https://cloud.google.com/customers/mediconcen) says MediConCen launched its platform in April 2025 and describes the technology choices summarized in Section 2.
The source does not disclose the approval record, customer rollout scope, release thresholds or rollback plan.
Proposed action for a new insurer: obtain authorization and run a bounded, manual-first evaluation before considering conditional automation.
The proposed design has no recorded approval, deployment, handoff or measured outcome in this case.
A claims operations lead would own the workflow; privacy, security and platform owners would sign off on their respective boundaries.
The proposed release sequence is offline assessment, side-effect-free shadow processing, assisted review, then narrowly scoped automation only if gates pass.
Any material regression, authorization failure, tenant mix-up or unresolvable adapter state would stop promotion and trigger containment.
Returning to manual processing must include reconciliation of in-flight work and preservation of corrections.
The source does not establish a clinical prior-authorization outcome or authorize automation of clinical decisions.
## 11. Outcome & Results
Source declaration: the following are vendor/company-reported outcomes, not independently verified measurements; illustrative calculations elsewhere are not company outcomes.
CLM-02: 98% field-level extraction accuracy is reported by [Google Cloud's MediConCen story](https://cloud.google.com/customers/mediconcen).
Its baseline, scoring rubric, population, sample, denominator, time window and uncertainty are not disclosed.
CLM-03: 100% higher straight-through processing rate is reported by the same source.
The baseline and final rates, eligible population, dates and absolute percentage-point change are not disclosed.
CLM-04: processing time reduced from days to hours is reported by the same source.
The source does not provide a median, tail, sample, distribution, measurement window or unchanged-workload comparator.
The source narrative credits the technology; the case cannot establish causal attribution without a defensible comparison.
Review effort, sustained duration, cost, adverse outcomes, slice quality and customer-specific adoption are not measured in the accessible evidence.
No quantitative clinical prior-authorization outcome is reported.
No outcome is established for the proposed design; a future evaluation owner must define cohort, window, controls and decision thresholds before measurement.
A 98% field-level result is not zero error and is not an all-claims-correct result.
A security feature or vendor statement is not proof of compliance in the actual deployment.
## 12. Failure Modes & What Went Wrong
No actual incident is reconstructed below; each entry is a hypothetical diagnostic scenario.
For every scenario, containment must preserve the source, avoid unauthorized decisions and notify the accountable owner as appropriate.
### F-01: Wrong claim association
Status: hypothetical; no company incident is alleged.
Symptom: a page appears under the wrong claim.
Discriminating check: compare intake manifest, tenant and document identifiers.
Possible cause: intake misassociation or a broken join; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: quarantine both packages and stop downstream writes.
Accountable role: intake owner.
Prevention and verification: cross-tenant and duplicate-key association tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-02: Wrong page citation
Status: hypothetical; no company incident is alleged.
Symptom: a plausible value cites an unrelated page.
Discriminating check: inspect page-version and region mapping against original.
Possible cause: coordinate transform or page ordering error; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: send affected bundle to review.
Accountable role: document pipeline owner.
Prevention and verification: page rotation and reordered-page reference tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-03: Unsupported value
Status: hypothetical; no company incident is alleged.
Symptom: a candidate contains a value absent from the source.
Discriminating check: blindly inspect the cited region and adjacent pages.
Possible cause: generation inferred a value or cited wrong evidence; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: mark unsupported and require review.
Accountable role: extraction owner.
Prevention and verification: gold examples with intentionally absent values.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-04: Wrong field role
Status: hypothetical; no company incident is alleged.
Symptom: invoice date appears in a service-date field.
Discriminating check: compare nearby labels and date roles.
Possible cause: semantic role confusion rather than unreadable text; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: hold the claim for role-aware review.
Accountable role: claims quality lead.
Prevention and verification: date-role contrast set.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-05: Currency ambiguity
Status: hypothetical; no company incident is alleged.
Symptom: a currency symbol maps to the wrong code.
Discriminating check: inspect source context and currency rubric.
Possible cause: ambiguous symbol or unjustified default; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: retain unresolved currency for review.
Accountable role: claims quality lead.
Prevention and verification: mixed-currency and shared-symbol tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-06: Decimal error
Status: hypothetical; no company incident is alleged.
Symptom: the extracted total differs by a factor of ten.
Discriminating check: compare original amount, separators and locale.
Possible cause: OCR error or locale normalization; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: hold the amount and reconcile invoice lines.
Accountable role: extraction owner.
Prevention and verification: decimal and locale regression suite.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-07: Missing attachment
Status: hypothetical; no company incident is alleged.
Symptom: a claim lacks an invoice referenced by the manifest.
Discriminating check: compare received objects and manifest entries.
Possible cause: partial upload or ingestion loss; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: request retransmission through authorized intake.
Accountable role: intake owner.
Prevention and verification: partial-upload and checksum tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-08: Duplicate submission
Status: hypothetical; no company incident is alleged.
Symptom: the same package is processed more than once.
Discriminating check: compare stable submission keys and downstream acknowledgment.
Possible cause: retry without idempotency or repeated upload; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: prevent additional side effects and reconcile state.
Accountable role: adapter owner.
Prevention and verification: duplicate-delivery and retry tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-09: Stale correction
Status: hypothetical; no company incident is alleged.
Symptom: an old extraction replaces a reviewer correction.
Discriminating check: compare event versions and write order.
Possible cause: concurrent update without version protection; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: restore the authorized correction and freeze retries.
Accountable role: state owner.
Prevention and verification: out-of-order event and compare-and-set tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-10: Reviewer overload
Status: hypothetical; no company incident is alleged.
Symptom: review queue age rises despite steady arrivals.
Discriminating check: compare arrival, completion and staffing by skill.
Possible cause: capacity shortage or model error surge; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: hold safely and escalate staffing.
Accountable role: operations lead.
Prevention and verification: peak-load and queue-drain exercise.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-11: Language gap
Status: hypothetical; no company incident is alleged.
Symptom: one language has worse unsupported-field rates.
Discriminating check: compare blinded labels by language with uncertainty.
Possible cause: capture, model or reviewer language mismatch; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: route affected language to qualified review.
Accountable role: evaluation lead.
Prevention and verification: language-stratified adjudication audit.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-12: Handwriting gap
Status: hypothetical; no company incident is alleged.
Symptom: handwritten fields have lower correctness.
Discriminating check: compare handwritten and typed evidence at the same field type.
Possible cause: OCR limitation or label disagreement; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: exclude affected items from automation.
Accountable role: evaluation lead.
Prevention and verification: handwriting-stratified evaluation.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-13: Template shift
Status: hypothetical; no company incident is alleged.
Symptom: a new provider layout causes row errors.
Discriminating check: compare errors and page layouts by template version.
Possible cause: new layout changes line association; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: route new template to review.
Accountable role: document pipeline owner.
Prevention and verification: unseen-template holdout.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-14: Low-quality scan
Status: hypothetical; no company incident is alleged.
Symptom: multiple fields become unreadable.
Discriminating check: inspect source resolution and blur before extraction.
Possible cause: capture degradation rather than model drift; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: request a clearer document or authorized manual review.
Accountable role: intake owner.
Prevention and verification: blur and compression stress tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-15: Selected labels
Status: hypothetical; no company incident is alleged.
Symptom: reviewed claims look worse than automation claims.
Discriminating check: compare audit sampling and routing probabilities.
Possible cause: review selection biases observed corrections; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: audit an independent sample of both routes.
Accountable role: evaluation lead.
Prevention and verification: sampling-frame and weighting checks.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-16: Model drift
Status: hypothetical; no company incident is alleged.
Symptom: a previously stable field becomes unreliable.
Discriminating check: compare source mix, versions and blinded audits.
Possible cause: model, source mix or upstream preprocessing change; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: pause affected automation.
Accountable role: model owner.
Prevention and verification: versioned temporal holdout and release comparison.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-17: Schema change
Status: hypothetical; no company incident is alleged.
Symptom: the adapter rejects otherwise valid packages.
Discriminating check: compare sender and receiver schema versions.
Possible cause: incompatible deployment or undocumented required field; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: hold in-flight work and restore compatible bundle.
Accountable role: integration owner.
Prevention and verification: forward/backward compatibility tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-18: Unknown adapter outcome
Status: hypothetical; no company incident is alleged.
Symptom: a timeout occurs after a downstream write.
Discriminating check: query downstream idempotency and acknowledgment state.
Possible cause: ambiguous commit rather than definite failure; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: reconcile before retrying.
Accountable role: adapter owner.
Prevention and verification: timeout-after-commit fault injection.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-19: Dependency outage
Status: hypothetical; no company incident is alleged.
Symptom: extraction requests time out.
Discriminating check: compare dependency health and queue saturation.
Possible cause: upstream service failure or local resource exhaustion; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: preserve input and use manual fallback.
Accountable role: platform owner.
Prevention and verification: bounded-timeout and recovery exercise.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-20: Tenant access error
Status: hypothetical; no company incident is alleged.
Symptom: a reviewer can access unrelated evidence.
Discriminating check: test authorization with tenant-scoped identities.
Possible cause: missing access check or overbroad role; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: revoke access and follow incident process.
Accountable role: security owner.
Prevention and verification: negative cross-tenant authorization tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-21: Sensitive logging
Status: hypothetical; no company incident is alleged.
Symptom: document content appears in ordinary logs.
Discriminating check: inspect restricted sample of logs under approved access.
Possible cause: unredacted exception or debugging output; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: restrict access and apply incident response.
Accountable role: security owner.
Prevention and verification: log redaction and secret-scanning tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-22: Insufficient audit sample
Status: hypothetical; no company incident is alleged.
Symptom: a minority slice has a wide interval.
Discriminating check: calculate slice sample size and uncertainty.
Possible cause: low prevalence or incomplete audit coverage; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: retain review and collect approved additional labels.
Accountable role: evaluation lead.
Prevention and verification: stratified sample and precision planning.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-23: False automation success
Status: hypothetical; no company incident is alleged.
Symptom: straight-through rate rises while corrections rise later.
Discriminating check: link completion events to later authorized corrections.
Possible cause: premature closure or downstream error detection; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: pause affected routing and investigate.
Accountable role: operations lead.
Prevention and verification: matured-cohort correction analysis.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-24: Reference disagreement
Status: hypothetical; no company incident is alleged.
Symptom: gold reviewers disagree on a source value.
Discriminating check: blindly compare rubric, original region and reviewer rationales.
Possible cause: source ambiguity or unclear labeling rule; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: adjudicate and record unresolved cases.
Accountable role: evaluation lead.
Prevention and verification: double-label and rubric-calibration exercise.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-25: Wrong document version
Status: hypothetical; no company incident is alleged.
Symptom: evidence references an older attachment.
Discriminating check: compare document hashes and version timestamps.
Possible cause: cache staleness or incorrect manifest lookup; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: hold package and regenerate from current authorized source.
Accountable role: document pipeline owner.
Prevention and verification: version replacement and cache invalidation tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-26: Excessive cost
Status: hypothetical; no company incident is alleged.
Symptom: cost per correct claim rises above budget.
Discriminating check: partition costs by calls, pages, retries and review.
Possible cause: retry storm, inefficient routing or difficult inputs; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: bound retries and route safely to manual work.
Accountable role: service owner.
Prevention and verification: cost-per-correct-claim sensitivity review.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-27: Latency tail
Status: hypothetical; no company incident is alleged.
Symptom: median improves but old claims accumulate.
Discriminating check: inspect percentiles, censored work and queue age.
Possible cause: slow review path or unbounded retries; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: escalate aged work and stop harmful routing.
Accountable role: operations lead.
Prevention and verification: end-to-end tail and backlog audit.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-28: Misinterpreted rate
Status: hypothetical; no company incident is alleged.
Symptom: stakeholders equate doubling with 100 points.
Discriminating check: recompute relative and absolute changes from same cohort.
Possible cause: denominator misunderstanding; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: correct the report and avoid unsupported outcome claims.
Accountable role: analytics owner.
Prevention and verification: independent calculation review.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-29: Invalid evidence control
Status: hypothetical; no company incident is alleged.
Symptom: valid JSON is mistaken for true data.
Discriminating check: inspect values against original source and rubric.
Possible cause: schema validation confused with semantic correctness; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: require source-backed adjudication.
Accountable role: evaluation lead.
Prevention and verification: syntactically valid but false-value tests.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
### F-30: Unsafe rollback
Status: hypothetical; no company incident is alleged.
Symptom: a prior model cannot read new in-flight schema.
Discriminating check: compare bundle compatibility and state migration rules.
Possible cause: independent version rollback; this remains a hypothesis until tested.
Potential impact: an incorrect, delayed or improperly exposed claim package.
Immediate containment: stop affected processing and reconcile manually.
Accountable role: release owner.
Prevention and verification: restore-and-replay compatibility drill.
Recovery requires an auditable reconciliation of affected in-flight claims.
Do not infer that this scenario occurred from its inclusion in the register.
## 13. Transferable Lesson
Pattern: `evaluation-mismatch` means an easily measured proxy may not represent the actual business decision.
A field-level score is useful for diagnosis but cannot determine how many complete claims are correct.
Review and evidence provenance may reduce operational risk, but their effect requires evaluation in the target workflow.
Prerequisites include authorized source data, claim-level labels, reliable document linkage and qualified reviewers.
The supporting evidence is the public headline and the worked arithmetic, not an independently reproduced company experiment.
Avoid transferring the design when ground truth cannot be audited, the review queue cannot absorb exceptions, or source access is unauthorized.
A portability test asks whether all required fields, evidence and downstream actions can be independently reconstructed for a sample of target claims.
A transfer to another document domain needs new field definitions, harm analysis, reviewers and operating gates.
The difference between field and claim grain is structural; the particular cloud architecture and numerical thresholds are local choices.
## 14. Connections & Cross-References
### 14a. Related cases, theory, and cheat sheets
Method reference: [Google SRE workbook: implementing SLOs](https://sre.google/workbook/implementing-slos/) informs the proposed service-indicator design, not company outcomes.
Collection navigation: [Case Studies index](INDEX.md) lists related planned operational cases; no local comparator is represented as completed here.
The local planning brief for document vision and OCR is a reference-informed proposal, not evidence that its source directories were inspected.
The customer source is [Google Cloud: MediConCen](https://cloud.google.com/customers/mediconcen), accessed 2026-11-07; no publication date is visible in the accessible page text.
### 14b. Open-question register
Q-01: What were the eligible population, scoring rubric, sample and field denominator behind 98%?
Resolution: obtain an authorized evaluation protocol and independently adjudicated field and claim samples.
Owner: evaluation lead; priority: before using the headline for rollout decisions.
Q-02: What were the baseline and final straight-through rates and their windows?
Resolution: obtain cohort-level counts and compare matched definitions across periods.
Owner: analytics lead; priority: before estimating absolute process impact.
Q-03: How do errors, review delay and missing evidence differ across language, template and scan-quality slices?
Resolution: conduct a privacy-reviewed, stratified independent audit with uncertainty intervals.
Owner: claims quality lead; priority: before expanding automation to a new slice.
Q-04: Does the proposed workflow improve complete-claim outcomes, latency and cost relative to a suitable baseline?
Resolution: run an approved controlled comparison with prespecified outcomes and mature follow-up.
Owner: claims operations lead; priority: before deployment authorization.
### 14c. Interview pack
All questions and answers concern the proposed scenario unless explicitly attributed to the published source.
#### Interview 01
Candidate question: What is the bounded decision?
Answer: Whether and how to introduce extraction assistance or gated automation for claims documents.
Reasoning: The output is a source-supported administrative package, not a clinical decision.
Follow-up: What would force manual processing?
Follow-up answer: Missing evidence, unavailable audit coverage, unauthorized data use or inadequate review capacity.
Interviewer check: the candidate must name the output package and the review gate, and must not describe extraction as an approval decision.
#### Interview 02
Candidate question: Who owns the outcome?
Answer: The insurer claims operations lead owns the workflow, supported by evaluation, platform, privacy and security roles.
Reasoning: A service provider cannot replace the insurer’s accountability for its own process.
Follow-up: Who may change access rules?
Follow-up answer: Only the accountable access owner through an approved policy and audit process.
Interviewer check: the candidate must separate insurer accountability from vendor service delivery and must not assign ownership to the model or the vendor.
#### Interview 03
Candidate question: What is the evaluation grain?
Answer: Use field instances for extraction diagnostics and claims for complete-package success.
Reasoning: Multiple field observations within a claim are correlated and claim field counts can vary.
Follow-up: How do you estimate uncertainty?
Follow-up answer: Use a sampling design and interval method that respects claim and relevant provider or time clusters.
Interviewer check: the candidate must state that field-instance and claim-level grains answer different questions, and must not average field errors into a claim verdict.
#### Interview 04
Candidate question: How do absent fields enter the metric?
Answer: Distinguish source-incomplete, unreadable and extraction-missing cases before scoring.
Reasoning: Silently excluding difficult documents biases quality upward.
Follow-up: What if the field is required but absent?
Follow-up answer: Follow the prespecified end-to-end disposition and report its frequency.
Interviewer check: the candidate must separate source absence from extraction failure and must not silently drop unreadable documents from the denominator.
#### Interview 05
Candidate question: Why is 98% field accuracy not 98% claim accuracy?
Answer: A claim fails the all-correct criterion when any required field is wrong.
Reasoning: Errors may be dispersed or concentrated, so the joint distribution matters.
Follow-up: What does independence imply for five fields?
Follow-up answer: Under the illustrative assumption, 0.98^5 is approximately 90.39%.
Interviewer check: the candidate must show the joint-probability reasoning explicitly and must reject treating a field average as a claim guarantee.
#### Interview 06
Candidate question: Interpret 100% higher straight-through processing.
Answer: It means twice the baseline rate if the source uses relative change convention.
Reasoning: For example, 20% to 40% is a 20-point increase and a 100% relative increase.
Follow-up: Can the company baseline be inferred?
Follow-up answer: No; the source does not disclose the baseline or final rate.
Interviewer check: the candidate must distinguish relative change from percentage-point change and must not invent the undisclosed baseline.
#### Interview 07
Candidate question: How should reviewers use evidence?
Answer: Show the original authorized region and allow correction, ambiguity and escalation.
Reasoning: A fluent value or model confidence is not proof of source support.
Follow-up: What if the citation is wrong?
Follow-up answer: Treat the candidate as unsupported and route it to review.
Interviewer check: the candidate must require a verifiable source region for each value and must not accept model confidence as evidence.
#### Interview 08
Candidate question: What is the best baseline?
Answer: Compare with the actual manual workflow and, if useful, OCR with manual correction.
Reasoning: A baseline must use the same eligible cohort and definitions as the candidate.
Follow-up: What if workload changes?
Follow-up answer: Use a controlled or carefully adjusted comparison and report remaining confounding.
Interviewer check: the candidate must define the comparison cohort before choosing a baseline and must not compare a tuned system to an untuned manual straw man.
#### Interview 09
Candidate question: How do you prevent duplicate actions?
Answer: Use stable logical package idempotency keys and reconcile uncertain outcomes before retrying.
Reasoning: A network timeout does not prove the downstream write failed.
Follow-up: What if acknowledgment is lost?
Follow-up answer: Query the downstream system for the key before a bounded reissue.
Interviewer check: the candidate must identify a stable logical key and a reconciliation step, not merely a retry counter.
#### Interview 10
Candidate question: What should happen during a service outage?
Answer: Preserve authorized intake, bound retries, queue or route to manual handling and escalate.
Reasoning: Relaxing evidence checks to maintain throughput trades availability for unmeasured risk.
Follow-up: What constrains the fallback?
Follow-up answer: Manual capacity, skills, privacy controls and backlog limits.
Interviewer check: the candidate must preserve evidence controls during degradation and must not relax verification to restore throughput.
#### Interview 11
Candidate question: How is a minority slice evaluated?
Answer: Audit appropriate document slices separately with counts, uncertainty, review rates and missingness.
Reasoning: A high aggregate can conceal poor outcomes in a small subgroup.
Follow-up: What if the sample is too small?
Follow-up answer: Do not infer a pass; retain review or gather authorized additional data.
Interviewer check: the candidate must state the slice's denominator and uncertainty and must not generalise a small-slice pass.
#### Interview 12
Candidate question: What does a controlled comparison add?
Answer: It can separate the workflow intervention from changes in population, staffing and time when properly designed.
Reasoning: Randomization, stable eligibility and contamination controls support causal interpretation.
Follow-up: What if randomization is unavailable?
Follow-up answer: Use a justified observational design and explicitly state residual confounding.
Interviewer check: the candidate must name the identification strategy and its residual confounding rather than implying a measured causal effect.
#### Interview 13
Candidate question: How should capacity be estimated?
Answer: Multiply peak arrivals by pages and benchmark per-worker throughput, then add headroom and review capacity.
Reasoning: A daily average hides burst and queue effects.
Follow-up: What is the illustrative peak?
Follow-up answer: A quarter of 1,000 claims in an hour is 250 claims or 1,000 pages at four pages each.
Interviewer check: the candidate must carry peak arrivals through pages, workers and review capacity with consistent units.
#### Interview 14
Candidate question: What is the right cost denominator?
Answer: Use correctly processed claims under a defensible audited rubric.
Reasoning: Raw calls or gross straight-through claims can reward cheap but erroneous work.
Follow-up: Which costs matter?
Follow-up answer: Include model requests, compute, storage, network, retries, review labor and failed work.
Interviewer check: the candidate must charge failed and reworked claims to the denominator, not only successful ones.
#### Interview 15
Candidate question: What should be released together?
Answer: Version the model, prompt, schema, preprocessing, policy checks and routing rules as a compatible bundle.
Reasoning: Rolling back only one part can leave in-flight work incompatible.
Follow-up: How do you validate rollback?
Follow-up answer: Exercise restoration and reconciliation against representative versioned state.
Interviewer check: the candidate must enumerate every coupled artifact and describe an actual restoration exercise.
#### Interview 16
Candidate question: What does the public story prove about security or clinical outcomes?
Answer: It reports product features and vendor/company claims, not deployment-specific compliance or clinical outcomes.
Reasoning: An advertised security feature is not a legal applicability analysis or a control audit.
Follow-up: What must happen before data use?
Follow-up answer: Confirm purpose, access, contractual terms and applicable obligations with authorized owners.
Interviewer check: the candidate must not convert a marketing or feature description into a compliance or clinical conclusion.
## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Medical insurance claims document processing |
| Problem type | Decide when extraction assistance or gated automation is justified |
| Method | Compare source-backed extraction and qualified review against the manual-first baseline |
| Primary result | Published: the [vendor customer story](https://cloud.google.com/customers/mediconcen) reports 98% field accuracy, a 100% higher straight-through rate and processing from days to hours; methods and denominators are undisclosed |
| Key lesson | Field-level accuracy does not establish complete-claim correctness or causal benefit |
| Tags | `evaluation-mismatch`; `human-in-the-loop`; `cost-latency-tradeoff` |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-10-02-006 / 1.0 / 2026-11-07 |
| Evidence level | Published vendor/company claims; proposed controls and calculations are illustrative |
| Implementation stage | Reported platform production; illustrative evaluation and controls proposed |
| Baseline / primary metric | Manual-first processing; claim-level accuracy and relevant operational outcomes, not measured here |
| Key limitation | Baseline rates, sample, denominator, observation windows and causal evaluation are undisclosed |
| Review trigger / owner | Source correction, independent evaluation or intended-scope change / claims operations and data governance leads |

Illustrative arithmetic only: 20% to 40% is a 20-percentage-point and 100% relative increase.
Proposed controls require source-backed fields, uncertainty handling, authorized review and no autonomous clinical decisions; they have not been executed or approved in this case.
Editorial status: draft, unreviewed.
