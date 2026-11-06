# Claims, Fraud, Customer and Operations - Cheat Sheets

> **Mandatory authoring contract:** Every authored sheet must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Purpose

These planned reference cards support repeatable decisions in Claims, Fraud, Customer and Operations. They complement case studies; they do not replace the assumptions, evidence, and evaluation required in a full case.

## Planned references

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | Claims automation controls | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 2 | Fraud investigation guide | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 3 | Customer fairness checklist | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |

## Required card fields

Every card must state: intended decision and reader; preconditions and inputs; default recommendation; metrics/formulas or architecture pattern; trade-offs; failure signals; security/privacy/governance constraints; and links to deeper material. Review tool-, market-, and regulation-sensitive cards quarterly.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR005 | Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment | Bundle-compatibility tests, rollback drills, guardrail breaches and unresolved side effects after recovery. Boundary: A prompt-only rollback cannot restore all dependencies; example dwell times and canary percentages are not defaults. | [REF-IR005](../../../../../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md) | planned; not written |
| PLAN-2026-IR011 | Page and region provenance; thumbnail/tiling tradeoffs; schema validation; claim-level temporal splits | Field accuracy and calibration by template, handwriting and scan quality; reviewer workload, evidence coverage and latency. Boundary: Neither a similarity match nor a VLM fraud suspicion justifies denial; source scale and patch arithmetic need fresh sizing. | [REF-IR011](../../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Insurance_Document_VLM.md) | planned; not written |
| PLAN-2026-IR014 | Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing | Gold field accuracy and reconciliation for extraction; full-path cost and grounded quality by tier for routing. Boundary: Contains two reference problems: author separate narratives; valid JSON and cheap token prices do not establish successful outcomes. | [REF-IR014](../../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md) | planned; not written |
| PLAN-2026-IR015 | Product/date eligibility; grounded generation; native-language review; source dependencies; supersession triggers | Unsupported product claims, expired evidence, read-aloud time, reviewer agreement and agent usability. Boundary: Human approval and source maintenance remain necessary; the example word target is not a compliance standard. | [REF-IR015](../../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md) | planned; not written |
| PLAN-2026-IR016 | Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation | Human-gold WER, domain-term WER, role error, objection precision/recall and end-to-end report coverage. Boundary: Transcript matching does not prove audio fidelity; aggregate coaching themes must not become unvalidated individual scores. | [REF-IR016](../../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md) | planned; not written |
| PLAN-2026-IR031 | N+1 detection; adjacency-list indexing; per-step authorization; cycle checks; stable assembly; snapshot reads | Reference traversal comparisons, round trips, rows scanned, unauthorized descendants, deep/wide trees and truncation behavior. Boundary: One SQL statement can still do excessive work; an outer LIMIT is not a universal recursion work bound. | [REF-IR031](../../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md) | planned; not written |
| PLAN-2026-IR036 | Conservative entity resolution; typed claim-actor graphs; motifs and supernodes; point-in-time features; reversible merges | Temporal baseline comparison, precision at capacity, mature outcomes, sampled low-risk audits and appeal/false-positive slices. Boundary: Connectivity and shared providers are not proof of fraud; source reference notes are unavailable and labels are delayed/selected. | [REF-IR036](../../../../../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
