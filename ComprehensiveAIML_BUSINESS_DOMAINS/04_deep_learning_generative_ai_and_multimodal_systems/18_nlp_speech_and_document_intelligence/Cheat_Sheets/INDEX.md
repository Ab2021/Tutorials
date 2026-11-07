# Module 18 Cheat Sheets — NLP, Speech and Document Intelligence

## 1. Overview
For NLP practitioners who can work with rates and labeled data, this collection supports task selection, document extraction and speech evaluation. Only ASR is authored, as an unreviewed draft; readiness requires independent technical and source review, not merely a card.

## 2. Topic ladder
| Level | Topic and outcome | Prerequisite / skill | Status |
|---|---|---|---|
| Foundation | NLP task matrix: select a task for an input and output; compare classification, extraction and generation using a counterexample. | Basic data labels and evaluation. | Planned |
| Core | Document extraction checklist: trace fields to source pages and validate a schema; diagnose an unsupported field. | Task selection; provenance and schema basics. | Planned |
| Core/production | [03 ASR transcription evaluation](03_asr-transcription-evaluation.md): compute edit rates and diagnose slice regressions. | Rates, sequence alignment, consent and reference transcription. | Draft; unreviewed; source files unavailable |

## 3. Quick topic lookup
| Question | Exact section | Use and warning |
|---|---|---|
| How are word errors counted? | [Formulas](03_asr-transcription-evaluation.md#4-core-formulas-with-full-derivation) | Use a consistent tokenization; empty references need separate handling. |
| Why did a score improve while users complain? | [Failure modes](03_asr-transcription-evaluation.md#8-failure-modes--edge-cases) | Compare frozen normalization and critical slices; aggregate WER hides harm. |
| Which NLP task or extraction checklist? | Topic ladder above | Both are planned, not links to completed sheets. |

## 4. Reading paths
- **Beginner:** Learn rates and reference transcription, then read [definitions](03_asr-transcription-evaluation.md#3-foundational-definitions) and [worked examples](03_asr-transcription-evaluation.md#5-worked-numerical-example); task matrix is a planned extension.
- **Practitioner:** Read the ASR draft from definitions through [selection](03_asr-transcription-evaluation.md#7-comparison--selection-table) and failures; compare normalization and language slices.
- **Advanced:** Read the entire draft, including streaming and diagnostics; document extraction and multilingual adaptations remain planned, not prerequisites available here.

## 5. Prerequisite skills
Compute ratios and edit operations, distinguish words from characters, and interpret paired comparisons; the draft's [definitions](03_asr-transcription-evaluation.md#3-foundational-definitions) provide local reference. Understand consent, transcript labeling and timestamp boundaries before evaluating real audio.

## 6. What this collection does NOT cover
The NLP task matrix and document extraction checklist are planned. ASR evaluation does not establish downstream clinical safety, speaker attribution or audio privacy compliance. Proposed interview-derived briefs below are planning references, not authored sheets.

## 7. Master symbol glossary
| Symbol | Meaning; unit / type / domain | First use |
|---|---|---|
| WER | Word edit rate; dimensionless scalar, nonnegative, potentially above 1; undefined for zero reference words. | [Formulas](03_asr-transcription-evaluation.md#4-core-formulas-with-full-derivation) |
| CER | Character edit rate; dimensionless scalar under the declared character and normalization policy. | [Worked example](03_asr-transcription-evaluation.md#5-worked-numerical-example) |

## 8. Quick Reference Card directory
| Card | Rule / use | Critical condition and avoid |
|---|---|---|
| [ASR card](03_asr-transcription-evaluation.md#12-quick-reference-card) | Compare normalized edits per reference word. | Nonempty reference and shared normalization; do not infer safety from aggregate WER. |

## 9. Common Issues locator
The final Common Issues artifact is not authored; its issue count is unknown, not zero. Until then use the draft's [hypothetical ASR failures](03_asr-transcription-evaluation.md#8-failure-modes--edge-cases): for improved WER with complaints, compare normalization and critical slices; for silence insertions, inspect voice boundaries and consented audio. These are not observed incident counts.

## 10. Empirical sources, versions and collection maintenance
The original source directories are unavailable and were not inspected. ASR claims and illustrative arithmetic remain subject to source and independent review; production performance is not measured. Consult the [authoring standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [master template](../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md); recheck normalization, evaluation policy and sources when they change. No review date is asserted. All sheets require the template's 12 sections, checked calculations, diagnostic failures, answered interview questions and final card.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

### Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR010 | Fill-in-the-middle; repository/time splits; authorized code retrieval; multilingual sampling; language adapters | Sandboxed functional correctness and latency for code; native-reviewed grounding and worst-language regressions for support. Boundary: Split this reference into two independent future narratives; code-test passage does not prove security or language accuracy. | [REF-IR010](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md) | planned; not written |
| PLAN-2026-IR011 | Page and region provenance; thumbnail/tiling tradeoffs; schema validation; claim-level temporal splits | Field accuracy and calibration by template, handwriting and scan quality; reviewer workload, evidence coverage and latency. Boundary: Neither a similarity match nor a VLM fraud suspicion justifies denial; source scale and patch arithmetic need fresh sizing. | [REF-IR011](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Insurance_Document_VLM.md) | planned; not written |
| PLAN-2026-IR016 | Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation | Human-gold WER, domain-term WER, role error, objection precision/recall and end-to-end report coverage. Boundary: Transcript matching does not prove audio fidelity; aggregate coaching themes must not become unvalidated individual scores. | [REF-IR016](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md) | planned; not written |
| PLAN-2026-IR017 | Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment | Paired WER with uncertainty by domain and code-switch slice, real-time factor and supported-route regression gates. Boundary: Model-generated transcripts are not independent gold; audio resampling does not restore missing frequencies. | [REF-IR017](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->

## 11. Collection acceptance checklist
- [ ] Author the task matrix, extraction checklist and final Common Issues artifact with at least 25 substantive issues.
- [ ] Independently review the ASR draft, formulas, examples, sources and all 12 sections.
- [ ] Validate real links, anchors, reading paths and glossary as the collection changes.
- [ ] Record reviewed versions and evidence; do not treat proposed briefs or illustrative figures as production results.
