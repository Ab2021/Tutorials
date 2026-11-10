# KDA Solutions Pharma — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-01-03-001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-01-03-001 | [HCP promotional-response drivers for a fictional launch](./01_HCP_Promotional_Response_Driver_Analysis_Launch_Brand.md) | KDA Solutions Pharma | analytical | decision record in the linked case | causal-confounding | evaluation-mismatch, label-delay | not-measured | proposed | 2026-11-10 | authored |
| CASE-01-03-002 | [Patient adherence and persistency in a fictional support programme](./02_Patient_Adherence_And_Persistency.md) | KDA Solutions Pharma | analytical | decision record in the linked case | metric-definition-drift | immortal-time-bias, informative-censoring | not-measured | proposed | null | authored |
| PLAN-01-03-02 | Patient adherence and persistency | KDA Solutions Pharma | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-01-03-03 | Brand-performance decomposition | KDA Solutions Pharma | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-01-03-04 | Payer formulary access drivers | KDA Solutions Pharma | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-01-03-05 | Omnichannel next-best action attribution | KDA Solutions Pharma | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 2 authored case(s) and 4 planned commitment(s). Authored cases share point-in-time decision logging and an explicit baseline; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-01-03-02` — Patient adherence and persistency. Not authored; no file exists, so no link is offered.
- `PLAN-01-03-03` — Brand-performance decomposition. Not authored; no file exists, so no link is offered.
- `PLAN-01-03-04` — Payer formulary access drivers. Not authored; no file exists, so no link is offered.
- `PLAN-01-03-05` — Omnichannel next-best action attribution. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [HCP promotional-response drivers for a fictional launch](./01_HCP_Promotional_Response_Driver_Analysis_Launch_Brand.md) | index regenerated from disk | not-measured / proposed unchanged |
| 2026-11-10 | [Patient adherence and persistency in a fictional support programme](./02_Patient_Adherence_And_Persistency.md) | index regenerated from disk | not-measured / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
