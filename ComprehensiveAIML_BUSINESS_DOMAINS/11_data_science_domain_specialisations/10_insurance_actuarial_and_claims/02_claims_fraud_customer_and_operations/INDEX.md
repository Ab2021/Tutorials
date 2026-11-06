# Claims, Fraud, Customer and Operations

> **Family:** [Insurance, Actuarial and Claims](../INDEX.md)  
> **Purpose:** Claims intake, triage, damage assessment, fraud, recovery, customer service, and operational capacity.  
> **Company / context examples:** P&C and health insurers

## Detailed topic map

| # | Topic | Status |
|---:|---|---|
| 1 | claims journey | planned |
| 2 | document extraction | planned |
| 3 | damage vision | planned |
| 4 | fraud | planned |
| 5 | triage | planned |
| 6 | severity | planned |
| 7 | subrogation | planned |
| 8 | customer communication | planned |
| 9 | call centre | planned |
| 10 | workforce | planned |
| 11 | monitoring | planned |
| 12 | human review | planned |

## Applied collections

- [Case Studies](./Case_Studies/INDEX.md): decision-centred cases using the local v2 catalog and integrated outline standard.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): concise references for recurring design, analysis, and operating decisions.

## Scope boundary

Use public company material only where cited. Treat company-specific systems as unknown unless a primary or clearly labelled published source supports the claim. Learning cases may be simulated or illustrative, but cannot claim measured production impact.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR005 | Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment | Which model, prompt, tools, index and policy versions can be promoted and recovered together? | Bundle-compatibility tests, rollback drills, guardrail breaches and unresolved side effects after recovery |
| PLAN-2026-IR011 | Page and region provenance; thumbnail/tiling tradeoffs; schema validation; claim-level temporal splits | Which claim fields can be extracted reliably, and which bundles require adjuster review? | Field accuracy and calibration by template, handwriting and scan quality; reviewer workload, evidence coverage and latency |
| PLAN-2026-IR014 | Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing | How should extraction contracts and model routes be accepted without confusing valid syntax with business success? | Gold field accuracy and reconciliation for extraction; full-path cost and grounded quality by tier for routing |
| PLAN-2026-IR015 | Product/date eligibility; grounded generation; native-language review; source dependencies; supersession triggers | Which generated scripts are fit for an approved, versioned coaching registry? | Unsupported product claims, expired evidence, read-aloud time, reviewer agreement and agent usability |
| PLAN-2026-IR016 | Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation | Which customer objections can be reported with reliable speaker attribution and evidence? | Human-gold WER, domain-term WER, role error, objection precision/recall and end-to-end report coverage |
| PLAN-2026-IR031 | N+1 detection; adjacency-list indexing; per-step authorization; cycle checks; stable assembly; snapshot reads | Should a subtree API use a recursive CTE or bounded level-order batching for its workload? | Reference traversal comparisons, round trips, rows scanned, unauthorized descendants, deep/wide trees and truncation behavior |
| PLAN-2026-IR036 | Conservative entity resolution; typed claim-actor graphs; motifs and supernodes; point-in-time features; reversible merges | Which graph-enriched alerts add confirmed value at a fixed investigation capacity beyond claim-level scoring? | Temporal baseline comparison, precision at capacity, mature outcomes, sampled low-risk audits and appeal/false-positive slices |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
