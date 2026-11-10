# Module 28 â€” AI Evaluation, Monitoring and Continuous Improvement

> **Part:** [Part 05 â€” AI Platforms, Data Systems and Production Engineering](../INDEX.md)  
> **Prerequisites:** 08,11,20,26  
> **Purpose:** Create valid offline and online evaluation systems for predictive models, LLMs, agents, and multimodal products.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | test-set design | planned |
| 2 | LLM and agent evaluations | planned |
| 3 | human evaluation | [authored](Cheat_Sheets/04_Model_Evaluation_and_Production_Monitoring.md) |
| 4 | production monitoring | [authored](Cheat_Sheets/04_Model_Evaluation_and_Production_Monitoring.md) |
| 5 | feedback and regression gates | planned |
| 6 | training, language-model, benchmark, judge, failure and profiling diagnostics | planned |
## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored source submodules

Visible Module 09 source titles 9.1–9.6 are mapped from [`Train/Templates/directory.txt`](../../../Templates/directory.txt). These are new drafts because the original source lesson bodies were unavailable.

- [9.1 — Training Metrics and Loss Analysis](9.1_Training_Metrics_and_Loss_Analysis.md) — draft; masked-token objective/reductions, optimization and data instrumentation, curve triage, checkpoint gates and diagnostics.
- [9.2 — Perplexity and Language Model Metrics](9.2_Perplexity_and_Language_Model_Metrics.md) — draft; held-out likelihood contract, corpus aggregation, paired comparison, uncertainty assumptions and release gates.
- [9.3 — Evaluation Benchmarks: Comprehensive Design](9.3_Evaluation_Benchmarks_Comprehensive.md) — draft; manifest/provenance, slice aggregation, contamination controls, scorer validity and release gates.
- [9.4 — LLM-as-Judge and Human Evaluation](9.4_LLM_as_Judge_and_Human_Evaluation.md) — draft; rubric/protocol design, judge-human calibration, bias tests, adjudication and promotion limits.
- [9.5 — Debugging Common Training Failures](9.5_Debugging_Common_Training_Failures.md) — draft; evidence-preserving triage, data/objective/precision/distributed diagnostics and recovery controls.
- [9.6 — Infrastructure Benchmarking and Profiling](9.6_Infrastructure_Benchmarking_and_Profiling.md) — draft; workload/metric contracts, tail latency, profiling methodology and capacity claims.

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
