# Module 47 â€” Cybersecurity, Security Operations and AI Defense

> **Part:** [Part 08 â€” Business Domains and Decision Intelligence](../INDEX.md)  
> **Prerequisites:** 09,12,15,30,31,49  
> **Purpose:** Use data, ML, agents, and automation for detection and response while securing the AI attack surface.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | threat modelling | planned |
| 2 | SIEM and detection engineering | planned |
| 3 | anomaly and graph analytics | planned |
| 4 | SOC agents | planned |
| 5 | AI attack and defence | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR028 | Hash collision amplification; catastrophic regex backtracking; pre-parse limits; bounded queues and downstream budgets | Which per-request limits and algorithm changes stop CPU exhaustion without rejecting legitimate traffic unnecessarily? | Adversarial near-match inputs, collision workload, CPU per request, timeout cascades and legitimate-input acceptance |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [SOC automation](../../00_SOURCE_COVERAGE_AGENTS.md#scenario-11) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
