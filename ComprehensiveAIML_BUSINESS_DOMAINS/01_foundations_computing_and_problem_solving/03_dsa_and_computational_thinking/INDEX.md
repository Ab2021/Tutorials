# Module 03 â€” Data Structures, Algorithms and Computational Thinking

> **Part:** [Part 01 â€” Foundations, Computing and Problem Solving](../INDEX.md)  
> **Prerequisites:** 02  
> **Purpose:** Reason about correctness, complexity, memory, and the algorithmic choices behind scalable systems.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | arrays trees graphs and heaps | [DSA patterns and invariants draft](Cheat_Sheets/01_DSA_Patterns_and_Invariants.md); remaining structure scope planned |
| 2 | hashing and sorting | planned |
| 3 | dynamic programming | planned |
| 4 | greedy and graph algorithms | planned |
| 5 | complexity and trade-offs | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored course-topic drafts

- [DSA Patterns and Invariants](Cheat_Sheets/01_DSA_Patterns_and_Invariants.md) — pattern selection, complexity assumptions, invariants and oracle validation; source inventory only.

## Canonical case studies

- [Design a Rate Limiter](Case_Studies/01_Design_Rate_Limiter.md) — illustrative bounded-admission decision; draft.
- [Implement Top-K Stream Processing](Case_Studies/02_Implement_Top_K_Stream_Processing.md) — illustrative deterministic snapshot decision; draft.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR018 | CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics | Which guarantees are needed for road routing, distributed admission and phrase retrieval? | Reference Dijkstra comparisons, admissible-heuristic checks, concurrent window simulations and brute-force phrase/ranking oracles |
| PLAN-2026-IR019 | Coherent traffic snapshots; CH compatibility; fallback Dijkstra; map matching; route versus ETA objectives | How should a navigation service publish legal optimal routes while separately evaluating predicted arrival time? | Base-graph shortest-path oracle, closure and turn-restriction replays, ETA error and routing tail latency |
| PLAN-2026-IR022 | Token-bucket invariants; atomic refill and debit; leased budgets; expiry; failure policy | Which admission policy remains within its global budget during contention or coordinator failure? | Aggregate-admission histories, refill boundaries, coordinator outages, lease races and rejected-request latency |
| PLAN-2026-IR023 | Shortcut expansion; closure snapshots; CH compatibility; directed nonnegative Dijkstra fallback | When must a route query fall back to the live base graph after a closure? | Nested-shortcut and adversarial-closure queries compared with base Dijkstra, including unreachable pairs |
| PLAN-2026-IR024 | Rarest-term anchor; lazy postings; skips; relative positions; repeated terms; matching analyzers | How can common-term phrase queries be evaluated without materializing huge posting lists? | Brute-force token-position oracle, peak memory and tail latency over frequent and repeated-term phrases |
| PLAN-2026-IR025 | Half-open intervals; deterministic ties; exchange proof; weighted interval DP boundary | When is earliest-finish greedy optimal, and when must scheduling become a weighted optimization problem? | Exhaustive small-instance optimum comparisons and counterexamples for weighted or buffered variants |
| PLAN-2026-IR026 | Cache policies; ranks/range queries; sketches; prefix/nearest-neighbor search; graphs; scheduling; online windows | Which data structure fits the update/query workload while retaining the required correctness and resource bounds? | Operation traces, exact small-data oracles, measured memory/latency and stated approximation or distributed guarantees |
| PLAN-2026-IR027 | Fenwick tree lowbit traversal; inclusive ranges; point deltas; wide arithmetic; atomic visibility | Which update and snapshot contract makes range totals correct and efficient? | Randomized update/query comparisons with a simple array, boundary/overflow checks and update-visibility tests |
| PLAN-2026-IR028 | Hash collision amplification; catastrophic regex backtracking; pre-parse limits; bounded queues and downstream budgets | Which per-request limits and algorithm changes stop CPU exhaustion without rejecting legitimate traffic unnecessarily? | Adversarial near-match inputs, collision workload, CPU per request, timeout cascades and legitimate-input acceptance |
| PLAN-2026-IR030 | Hash flooding; BFS OOM; regex backtracking; recursive DFS depth; ORM N+1; overflow; unstable ties; floating search termination | Which invariant failed, how can it be contained, and which replay would prevent recurrence? | Reproduce each mechanism separately with bounded adversarial traces; track latency, memory, database calls and deterministic outputs |
| PLAN-2026-IR031 | N+1 detection; adjacency-list indexing; per-step authorization; cycle checks; stable assembly; snapshot reads | Should a subtree API use a recursive CTE or bounded level-order batching for its workload? | Reference traversal comparisons, round trips, rows scanned, unauthorized descendants, deep/wide trees and truncation behavior |
| PLAN-2026-IR035 | Score buckets; Fenwick count summaries; deterministic ties; deduplicated events; cross-shard moves; closing watermarks | How should score events support fast display ranks and a reproducible final prize result? | Reference sorted snapshots, duplicate/late events, negative scores, bucket crossings and exact final-rank replay |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
