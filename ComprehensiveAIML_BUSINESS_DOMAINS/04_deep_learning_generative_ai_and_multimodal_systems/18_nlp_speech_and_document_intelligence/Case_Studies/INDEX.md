# Module 18 Case Studies — NLP, Speech and Document Intelligence

## 1. Collection metadata

This draft collection retains original cases for claims extraction and transcription quality, plus
reference-informed future decisions. No case manuscript, measured model result, deployment, or
organisation-specific workflow is claimed here. The source filenames and interview references are
planning inputs, not evidence that their assumed workloads or outcomes apply to a new case.

| Field | Value |
|---|---|
| Editorial state | Draft; no completed local canonical case |
| Evidence / implementation state | not-measured / proposed |
| Scope | Text, document, and speech decisions with human verification |
| Review trigger | New primary evidence, labelled evaluation data, or completed canonical manuscript |

## 2. Case catalog

| Case ID | Linked title / canonical home | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-18-01 | Claims document extraction — `01_claims_document_extraction.md` (planned) | Document intelligence | hybrid | Which extracted fields can be shown and which bundles require reviewer routing | `human-in-the-loop` | `evaluation-mismatch`, `authorization-scope` | not-measured | proposed | never | No manuscript, evaluation, or operational outcome |
| CASE-18-02 | Call-centre transcription quality — `02_call_centre_transcription_quality.md` (planned) | Speech analytics | hybrid | Whether a transcript has adequate evidence and speaker attribution for aggregate reporting | `evaluation-mismatch` | `human-in-the-loop`, `authorization-scope` | not-measured | proposed | never | No manuscript, evaluation, or operational outcome |
| PLAN-2026-IR010 | Repository-aware completion / language adapter (planned shared brief) | Code and multilingual assistance | llm-agentic | When added adaptation complexity is justified | `data-leakage` | `authorization-scope`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 | Planning brief only |
| PLAN-2026-IR011 | VLM/OCR adjuster assistance (planned shared brief) | Document intelligence | hybrid | Which claim fields require review | `human-in-the-loop` | `evaluation-mismatch`, `delayed-selective-labels` | not-measured | proposed | 2026-11-06 | Planning brief only |
| PLAN-2026-IR016 | Customer-objection aggregation (planned) | Speech analytics | hybrid | Whether objections have reliable speaker and evidence attribution | `evaluation-mismatch` | `authorization-scope`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 | Planning brief only |
| PLAN-2026-IR017 | Domain ASR adaptation (planned shared brief) | Speech recognition | hybrid | Which product/language routes an adapter may be promoted to | `rollout-regression` | `distribution-shift`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 | Planning brief only |

## 3. Canonical reusable patterns

| Pattern | Meaning | Boundary |
|---|---|---|
| `human-in-the-loop` | The system prepares fields, transcript, spans, or recommendations for an authorised reviewer. | Confidence cannot replace review authority. |
| `evaluation-mismatch` | WER, field F1, and offline accuracy need not predict a valid workflow decision. | Include evidence coverage, calibration, correction effort, and abstention. |
| `authorization-scope` | Audio, document regions, and derived text are governed by purpose and access scope. | A usable extraction may still be unauthorised to disclose. |
| `data-leakage` | A split can expose future, same-repository, same-speaker, or duplicate-template information. | Random row splits do not establish generalisation. |
| `rollout-regression` | An adaptation can improve a target slice while degrading a supported route. | Promotion requires independent target and regression evaluation. |

## 4. Similarity and contrast map

| Item | Shared reasoning | Difference that changes validation |
|---|---|---|
| CASE-18-01 | Both document cases require provenance and review routing. | Claims cases need field/schema, page-region, and temporal contract checks. |
| CASE-18-02 | Both speech cases need a human-gold comparison. | Reporting requires speaker-role and evidence-span validity, not only transcription. |
| PLAN-2026-IR017 | ASR quality can vary by language and route. | Adapter promotion adds replay and cross-route regression requirements. |
| PLAN-2026-IR010 | Language-aware systems need independent splits. | Code correctness and authorisation differ from ASR or document evidence. |

## 5. Questions and future-case backlog

1. Define whether the unit is a field, page region, utterance, speaker turn, call, or report before selecting a metric.
2. Keep human gold independent of system-generated text and establish speaker, language, channel, and scan-quality slices.
3. Treat OCR, ASR, extraction, attribution, and aggregation as separable error stages with their own diagnostics.
4. Route uncertain cases to authorised reviewers; do not use an unsupported model score to determine a claim or person outcome.
5. Test time, speaker, template, repository, and language isolation in every evaluation split.

## 6. Review and retirement log

| Item | State | Required action |
|---|---|---|
| CASE-18-01 | Planned | Author canonical case with field-level evidence and reviewer-workload validation. |
| CASE-18-02 | Planned | Author canonical case with a checked WER/role/evidence diagnostic trace. |
| IR010/011/016/017 | Planning only | Preserve reference limits; do not transfer reference figures as measurements. |
| Collection | Draft | Reconcile case links, evidence, stage, and editorial status after each manuscript. |

## 7. Pre-publish checks

- [ ] Each real case has the 15 ordered sections and a final quick-reference card.
- [ ] Each data contract names source, unit, cut-off, labels, permissions, exclusions, and split boundaries.
- [ ] Quantitative examples have units, a checked calculation, sensitivity, and a failure boundary.
- [ ] Each case has 12–18 answered interview questions and a concrete recovery path.
- [ ] Catalog rows link only to real canonical manuscripts and retain planned/shared labels honestly.
