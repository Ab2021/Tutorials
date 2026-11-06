# Evaluation, Reliability and Incidents

> **Family:** [Practical ML Design and Interviews](../INDEX.md)  
> **Purpose:** Evaluation across predictive ML, GenAI, agents, streaming, edge, and domain systems; incident analysis and prevention.  
> **Company / context examples:** All production systems

## Detailed topic map

| # | Topic | Status |
|---:|---|---|
| 1 | evaluation design | planned |
| 2 | offline/online gaps | planned |
| 3 | human evaluation | planned |
| 4 | drift | planned |
| 5 | SLOs | planned |
| 6 | tracing | planned |
| 7 | red teaming | planned |
| 8 | canaries | planned |
| 9 | incident command | planned |
| 10 | postmortems | planned |
| 11 | continuous improvement | planned |

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
| PLAN-2026-IR008 | Versioned evaluation manifests; contaminated benchmarks; calibrated judges; over-refusal; sandboxed code scoring | Does a candidate pass the required suites with enough evidence to justify promotion? | Paired quality comparisons, critical-failure review, uncertainty, judge drift and missing-suite failure tests |
| PLAN-2026-IR030 | Hash flooding; BFS OOM; regex backtracking; recursive DFS depth; ORM N+1; overflow; unstable ties; floating search termination | Which invariant failed, how can it be contained, and which replay would prevent recurrence? | Reproduce each mechanism separately with bounded adversarial traces; track latency, memory, database calls and deterministic outputs |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
