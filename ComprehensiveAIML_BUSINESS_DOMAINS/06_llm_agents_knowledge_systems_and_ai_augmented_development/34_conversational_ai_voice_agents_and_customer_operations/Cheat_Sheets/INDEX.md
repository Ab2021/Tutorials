# Module 34 Cheat Sheets â€” Conversational AI, Voice Agents and Customer Operations

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | voice-agent architecture | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | conversation QA rubric | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | handoff policy | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR015 | Product/date eligibility; grounded generation; native-language review; source dependencies; supersession triggers | Unsupported product claims, expired evidence, read-aloud time, reviewer agreement and agent usability. Boundary: Human approval and source maintenance remain necessary; the example word target is not a compliance standard. | [REF-IR015](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md) | planned; not written |
| PLAN-2026-IR016 | Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation | Human-gold WER, domain-term WER, role error, objection precision/recall and end-to-end report coverage. Boundary: Transcript matching does not prove audio fidelity; aggregate coaching themes must not become unvalidated individual scores. | [REF-IR016](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md) | planned; not written |
| PLAN-2026-IR017 | Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment | Paired WER with uncertainty by domain and code-switch slice, real-time factor and supported-route regression gates. Boundary: Model-generated transcripts are not independent gold; audio resampling does not restore missing frequencies. | [REF-IR017](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
