# Module 27 — Datasets, Tools, and Benchmarks

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/27_datasets_tools_benchmarks.md`](../../../course/27_datasets_tools_benchmarks.md)
> **Folder:** `modules/04_value_governance/27_datasets_tools_benchmarks/`
> **Version:** 2026.5

---

## 1. Module purpose

This module is the evidence shelf: which public datasets fit which decision, how benchmarks mislead, and which tools earn a place in a supply chain stack.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 02, 07, 18 |
| Unlocks | 28 |
| Assessment | Dataset and benchmark audit (Week 33) |
| Survey file | `27_datasets_tools_benchmarks.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_dataset-fit-tests.md`](01_dataset-fit-tests.md) | The Seven Fit Tests | Grain, horizon, coverage, censoring, licence, provenance, and decision relevance. | 1,200–1,800 | planned |
| 2 | [`02_provenance-classes.md`](02_provenance-classes.md) | Data Provenance Classes | Competition, regulatory, company-published, synthetic, and scraped data. | 1,200–1,800 | planned |
| 3 | [`03_retail-demand-datasets.md`](03_retail-demand-datasets.md) | Retail and Demand Datasets | M5, M4, Favorita, Rossmann, Online Retail II, Dunnhumby, Instacart. | 1,200–1,800 | planned |
| 4 | [`04_manufacturing-datasets.md`](04_manufacturing-datasets.md) | Manufacturing, Quality, and Maintenance Datasets | C-MAPSS, Bosch, SECOM, Scania, MetroPT, SKAB. | 1,200–1,800 | planned |
| 5 | [`05_logistics-trade-external.md`](05_logistics-trade-external.md) | Logistics, Trade, and External Signals | Comtrade, LPI, GDELT, ACLED, weather, and port data. | 1,200–1,800 | planned |
| 6 | [`06_benchmark-suites.md`](06_benchmark-suites.md) | Benchmark Suites | What each competition or suite measures and what it omits. | 1,200–1,800 | planned |
| 7 | [`07_benchmark-failure-modes.md`](07_benchmark-failure-modes.md) | The Six Ways a Benchmark Misleads | Contamination, leakage, metric mismatch, regime change, selection, and scale. | 1,200–1,800 | planned |
| 8 | [`08_m5-foundation-caution.md`](08_m5-foundation-caution.md) | The M5 Lesson and Foundation-Model Caution | What a winning solution proves and what it does not. | 1,200–1,800 | planned |
| 9 | [`09_tool-selection.md`](09_tool-selection.md) | Tool Selection | Forecasting, optimization, simulation, data, evaluation, and orchestration. | 1,200–1,800 | planned |
| 10 | [`10_licensing-reproducibility.md`](10_licensing-reproducibility.md) | Licensing and Reproducibility | Terms of use, citation, versioning, and a reproducible data card. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Seven Fit Tests ->
02 Data Provenance Classes ->
03 Retail and Demand Datasets ->
04 Manufacturing, Quality, and Maintenance Datasets ->
05 Logistics, Trade, and External Signals ->
06 Benchmark Suites ->
07 The Six Ways a Benchmark Misleads ->
08 The M5 Lesson and Foundation-Model Caution ->
09 Tool Selection ->
10 Licensing and Reproducibility
```

## 5. What this module does not cover

- It does not redistribute proprietary data or guarantee that a public dataset will remain available.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 10 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`27_datasets_tools_benchmarks.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
