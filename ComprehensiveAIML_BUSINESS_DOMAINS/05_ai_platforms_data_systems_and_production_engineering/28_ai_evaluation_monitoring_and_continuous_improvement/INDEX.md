# Module 28 â€” AI Evaluation, Monitoring and Continuous Improvement

> **Part:** [Part 05 â€” AI Platforms, Data Systems and Production Engineering](../INDEX.md)  
> **Prerequisites:** 08,11,20,26  
> **Purpose:** Create valid offline and online evaluation systems for predictive models, LLMs, agents, and multimodal products.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | test-set design | planned |
| 2 | LLM and agent evaluations | planned |
| 3 | human evaluation | planned |
| 4 | production monitoring | planned |
| 5 | feedback and regression gates | planned |

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
| PLAN-2026-IR001 | Dataset lineage; synthetic-data feedback; deduplication limits; language and domain coverage | Which documents should be admitted or quarantined before the next training run? | Held-out long-tail factual accuracy, diversity and per-slice retention; compare filtering with provenance controls |
| PLAN-2026-IR008 | Versioned evaluation manifests; contaminated benchmarks; calibrated judges; over-refusal; sandboxed code scoring | Does a candidate pass the required suites with enough evidence to justify promotion? | Paired quality comparisons, critical-failure review, uncertainty, judge drift and missing-suite failure tests |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Observability versus evaluation, run tracing, evaluation frameworks, feedback loops and monitoring](../../00_SOURCE_COVERAGE_AGENTS.md#agent-observability) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Fine-tuning Module 09 / cheat sheet 11 — evaluation and debugging](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-evaluation) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [Six inference clusters, tool contracts, mock responses, manifest, archive and explainer](../../00_SOURCE_COVERAGE_AGENTS.md#inference-clusters) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |
| [LangSmith observability, production deployment and dated emerging patterns](../../00_SOURCE_COVERAGE_AGENTS.md#lang-ops) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
