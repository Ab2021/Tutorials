# Catalogue, Search, Discovery and Conversion — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-05-01-001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-05-01-001 | [Search relevance recovery after an offline-NDCG rollout](./01_Search_Relevance_Recovery.md) | Catalogue, Search, Discovery and Conversion | hybrid | decision record in the linked case | evaluation-mismatch | rollout-regression, human-in-the-loop | illustrative | proposed | null | authored |
| CASE-05-01-002 | [Catalogue enrichment](./02_Catalogue_Enrichment.md) | Catalogue, Search, Discovery and Conversion | data-quality | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-01-003 | [Recommendation cold start](./03_Recommendation_Cold_Start.md) | Catalogue, Search, Discovery and Conversion | recommendation | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-01-004 | [Checkout funnel diagnosis](./04_Checkout_Funnel_Diagnosis.md) | Catalogue, Search, Discovery and Conversion | analytical | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-01-005 | [Review abuse detection](./05_Review_Abuse_Detection.md) | Catalogue, Search, Discovery and Conversion | predictive-ml | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-01-006 | [Authorized near-real-time index evolution](./06_Authorized_Near_Real_Time_Index.md) | Catalogue, Search, Discovery and Conversion | streaming-systems | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | authored |
| CASE-05-01-007 | [Bounded phrase-search memory](./07_Bounded_Phrase_Search.md) | Catalogue, Search, Discovery and Conversion | incident | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-01-008 | [Bounded graph recommendation traversal](./08_Bounded_Graph_Recommendations.md) | Catalogue, Search, Discovery and Conversion | incident | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-01-009 | [Autocomplete freshness and abuse controls](./09_Autocomplete_Freshness_Controls.md) | Catalogue, Search, Discovery and Conversion | streaming-systems | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| PLAN-05-01-01 | Search relevance recovery | Catalogue, Search, Discovery and Conversion | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-01-02 | Catalogue enrichment | Catalogue, Search, Discovery and Conversion | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-01-03 | Recommendation cold start | Catalogue, Search, Discovery and Conversion | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-01-04 | Checkout funnel diagnosis | Catalogue, Search, Discovery and Conversion | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-01-05 | Review abuse detection | Catalogue, Search, Discovery and Conversion | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR021 | Evolve an authorized near-real-time search index safely | Catalogue, Search, Discovery and Conversion | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR024 | Bound phrase-search memory while preserving exact matches | Catalogue, Search, Discovery and Conversion | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR029 | Return useful graph candidates under explicit traversal budgets | Catalogue, Search, Discovery and Conversion | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR032 | Balance autocomplete freshness, memory and abuse controls | Catalogue, Search, Discovery and Conversion | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 9 case file(s), 2 passing structural lint, and 9 planned commitment(s). Cases require independent review of point-in-time decision logging and explicit baselines; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-05-01-01` — Search relevance recovery. Not authored; no file exists, so no link is offered.
- `PLAN-05-01-02` — Catalogue enrichment. Not authored; no file exists, so no link is offered.
- `PLAN-05-01-03` — Recommendation cold start. Not authored; no file exists, so no link is offered.
- `PLAN-05-01-04` — Checkout funnel diagnosis. Not authored; no file exists, so no link is offered.
- `PLAN-05-01-05` — Review abuse detection. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR021` — Evolve an authorized near-real-time search index safely. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR024` — Bound phrase-search memory while preserving exact matches. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR029` — Return useful graph candidates under explicit traversal budgets. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR032` — Balance autocomplete freshness, memory and abuse controls. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Search relevance recovery after an offline-NDCG rollout](./01_Search_Relevance_Recovery.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Catalogue enrichment](./02_Catalogue_Enrichment.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Recommendation cold start](./03_Recommendation_Cold_Start.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Checkout funnel diagnosis](./04_Checkout_Funnel_Diagnosis.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Review abuse detection](./05_Review_Abuse_Detection.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Authorized near-real-time index evolution](./06_Authorized_Near_Real_Time_Index.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Bounded phrase-search memory](./07_Bounded_Phrase_Search.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Bounded graph recommendation traversal](./08_Bounded_Graph_Recommendations.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Autocomplete freshness and abuse controls](./09_Autocomplete_Freshness_Controls.md) | index regenerated from disk | illustrative / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
