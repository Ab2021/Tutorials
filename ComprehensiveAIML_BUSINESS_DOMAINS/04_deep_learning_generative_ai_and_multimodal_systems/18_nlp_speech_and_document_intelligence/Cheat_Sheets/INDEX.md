# Module 18 Cheat Sheets â€” NLP, Speech and Document Intelligence

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. The catalog includes an unreviewed draft; expand this index to all 11 collection-index sections and create a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | NLP task matrix | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | document extraction checklist | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | [ASR transcription evaluation](03_asr-transcription-evaluation.md) | WER/CER alignment, streaming latency, language and privacy slices, diagnostics and interview questions. | draft; unreviewed; source files unavailable |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR010 | Fill-in-the-middle; repository/time splits; authorized code retrieval; multilingual sampling; language adapters | Sandboxed functional correctness and latency for code; native-reviewed grounding and worst-language regressions for support. Boundary: Split this reference into two independent future narratives; code-test passage does not prove security or language accuracy. | [REF-IR010](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md) | planned; not written |
| PLAN-2026-IR011 | Page and region provenance; thumbnail/tiling tradeoffs; schema validation; claim-level temporal splits | Field accuracy and calibration by template, handwriting and scan quality; reviewer workload, evidence coverage and latency. Boundary: Neither a similarity match nor a VLM fraud suspicion justifies denial; source scale and patch arithmetic need fresh sizing. | [REF-IR011](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Insurance_Document_VLM.md) | planned; not written |
| PLAN-2026-IR016 | Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation | Human-gold WER, domain-term WER, role error, objection precision/recall and end-to-end report coverage. Boundary: Transcript matching does not prove audio fidelity; aggregate coaching themes must not become unvalidated individual scores. | [REF-IR016](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md) | planned; not written |
| PLAN-2026-IR017 | Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment | Paired WER with uncertainty by domain and code-switch slice, real-time factor and supported-route regression gates. Boundary: Model-generated transcripts are not independent gold; audio resampling does not restore missing frequencies. | [REF-IR017](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
