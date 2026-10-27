# Module 03 — Data Engineering and Interoperability

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
> **Folder:** `modules/01_foundations/03_data_engineering_interoperability/`
> **Version:** 2026.5

---

## 1. Module purpose

This module turns the data landscape into reproducible, tested, documented pipelines that preserve supply chain semantics from ingestion to a demand-and-inventory mart.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 02 |
| Unlocks | 09, 19, 21 |
| Assessment | Reproducible planning-data pipeline (Week 5) |
| Survey file | `03_data_engineering_interoperability.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_layered-architecture.md`](01_layered-architecture.md) | Layered Architecture and the Medallion Pattern | Raw, staging, conformed, and mart layers with clear contracts between them. | 1,200–1,800 | planned |
| 2 | [`02_batch-microbatch-streaming.md`](02_batch-microbatch-streaming.md) | Batch, Micro-Batch, and Streaming | Latency choices and their consequences for planning and execution. | 1,200–1,800 | planned |
| 3 | [`03_ingestion-patterns.md`](03_ingestion-patterns.md) | Ingestion Patterns | Full, incremental, change-data-capture, and file-drop ingestion with idempotency. | 1,200–1,800 | planned |
| 4 | [`04_sql-supply-chain-patterns.md`](04_sql-supply-chain-patterns.md) | SQL Patterns That Carry Supply Chain Semantics | As-of joins, snapshot logic, allocation, and slowly changing dimensions. | 1,200–1,800 | planned |
| 5 | [`05_python-structure-tests.md`](05_python-structure-tests.md) | Python Project Structure, Types, and Tests | A layout and test strategy that survives handover. | 1,200–1,800 | planned |
| 6 | [`06_dbt-models-tests.md`](06_dbt-models-tests.md) | dbt Models, Tests, and Documentation | Staging, intermediate, and mart models with freshness and uniqueness tests. | 1,200–1,800 | planned |
| 7 | [`07_orchestration.md`](07_orchestration.md) | Orchestration and Scheduling | Dependencies, backfills, retries, and the planning cutoff. | 1,200–1,800 | planned |
| 8 | [`08_edi-x12-edifact.md`](08_edi-x12-edifact.md) | EDI: X12 and EDIFACT | 850, 855, 856, 810, 846, 852, 940, 945, and their EDIFACT analogues. | 1,200–1,800 | planned |
| 9 | [`09_gs1-epcis.md`](09_gs1-epcis.md) | GS1 Identifiers and EPCIS 2.0 | GTIN, GLN, SSCC, and event data that supports traceability. | 1,200–1,800 | planned |
| 10 | [`10_apis-contracts-mdm.md`](10_apis-contracts-mdm.md) | APIs, Data Contracts, and MDM | Schema ownership, entity resolution, and breaking-change control. | 1,200–1,800 | planned |
| 11 | [`11_quality-quarantine-lineage.md`](11_quality-quarantine-lineage.md) | Quality Tests, Quarantine, Versioning, and Lineage | Failing loudly, isolating bad rows, and reproducing any past plan. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Layered Architecture and the Medallion Pattern ->
02 Batch, Micro-Batch, and Streaming ->
03 Ingestion Patterns ->
04 SQL Patterns That Carry Supply Chain Semantics ->
05 Python Project Structure, Types, and Tests ->
06 dbt Models, Tests, and Documentation ->
07 Orchestration and Scheduling ->
08 EDI: X12 and EDIFACT ->
09 GS1 Identifiers and EPCIS 2.0 ->
10 APIs, Data Contracts, and MDM ->
11 Quality Tests, Quarantine, Versioning, and Lineage
```

## 5. What this module does not cover

- It does not teach causal inference or forecasting; it makes their inputs trustworthy and reproducible.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 11 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`03_data_engineering_interoperability.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
