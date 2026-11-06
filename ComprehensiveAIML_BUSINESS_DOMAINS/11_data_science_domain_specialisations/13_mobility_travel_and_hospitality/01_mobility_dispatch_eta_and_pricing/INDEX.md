# Mobility Dispatch, ETA and Pricing

> **Family:** [Mobility, Travel and Hospitality](../INDEX.md)  
> **Purpose:** On-demand transport, dispatch, routing, ETA, driver supply, dynamic pricing, safety, and service recovery.  
> **Company / context examples:** Ride-hailing, delivery, fleet services

## Detailed topic map

| # | Topic | Status |
|---:|---|---|
| 1 | marketplace dynamics | planned |
| 2 | demand | planned |
| 3 | driver supply | planned |
| 4 | dispatch | planned |
| 5 | routing | planned |
| 6 | ETA | planned |
| 7 | pricing | planned |
| 8 | safety | planned |
| 9 | fraud | planned |
| 10 | weather | planned |
| 11 | experiments | planned |
| 12 | operations | planned |

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
| PLAN-2026-IR018 | CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics | Which guarantees are needed for road routing, distributed admission and phrase retrieval? | Reference Dijkstra comparisons, admissible-heuristic checks, concurrent window simulations and brute-force phrase/ranking oracles |
| PLAN-2026-IR019 | Coherent traffic snapshots; CH compatibility; fallback Dijkstra; map matching; route versus ETA objectives | How should a navigation service publish legal optimal routes while separately evaluating predicted arrival time? | Base-graph shortest-path oracle, closure and turn-restriction replays, ETA error and routing tail latency |
| PLAN-2026-IR023 | Shortcut expansion; closure snapshots; CH compatibility; directed nonnegative Dijkstra fallback | When must a route query fall back to the live base graph after a closure? | Nested-shortcut and adversarial-closure queries compared with base Dijkstra, including unreachable pairs |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
