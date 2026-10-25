# Module 03 — Data Engineering and Interoperability

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/03_data_engineering_interoperability.md`](../../course/03_data_engineering_interoperability.md)
> **Folder:** `modules/03_data_engineering/`
> **Version:** 2026.3

---

## 1. Module purpose

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 02 |
| Unlocks | 06, 15, 26 |
| Assessment | Reproducible cohort pipeline (Week 7) |
| Survey file | `03_data_engineering_interoperability.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_architecture-patterns.md`](01_architecture-patterns.md) | Architecture Patterns | Lakehouse, medallion layers, event-driven ingestion, and clinical constraints on each. | 1,200-1,800 | complete |
| 2 | [`02_sql-for-clinical-data.md`](02_sql-for-clinical-data.md) | SQL for Clinical Data | Window functions, temporal joins, deduplication, and the queries that define cohorts. | 1,200-1,800 | complete |
| 3 | [`03_python-engineering.md`](03_python-engineering.md) | Python Engineering | Packaging, typing, testing, configuration, and reproducible environments. | 1,200-1,800 | complete |
| 4 | [`04_orchestration.md`](04_orchestration.md) | Orchestration | DAGs, idempotency, backfills, retries, and scheduling clinical pipelines. | 1,200-1,800 | complete |
| 5 | [`05_fhir-ingestion.md`](05_fhir-ingestion.md) | FHIR Ingestion | Resources, references, bundles, search parameters, and flattening to analytics tables. | 1,200-1,800 | complete |
| 6 | [`06_dicom-pipelines.md`](06_dicom-pipelines.md) | DICOM Pipelines | Series assembly, de-identification, pixel handling, and imaging cohort construction. | 1,200-1,800 | complete |
| 7 | [`07_data-versioning-and-lineage.md`](07_data-versioning-and-lineage.md) | Data Versioning and Lineage | Snapshots, hashes, provenance, and answering 'where did this number come from'. | 1,200-1,800 | complete |
| 8 | [`08_deidentification-engineering.md`](08_deidentification-engineering.md) | De-identification Engineering | Safe Harbor, expert determination, dates, free text, and re-identification risk. | 1,200-1,800 | complete |
| 9 | [`09_security-engineering.md`](09_security-engineering.md) | Security Engineering | Least privilege, encryption, audit logs, secrets, and pipeline threat models. | 1,200-1,800 | complete |
| 10 | [`10_testing-and-observability.md`](10_testing-and-observability.md) | Testing and Observability | Data tests, freshness checks, schema contracts, and alerting on pipeline failure. | 1,200-1,800 | complete |
| 11 | [`11_fhir-bulk-export-and-analytics.md`](11_fhir-bulk-export-and-analytics.md) | FHIR Bulk Export and Analytics | Asynchronous Bulk FHIR $export, SMART backend services, NDJSON, and Parquet data lakes. | 1,200-1,800 | complete |
| 12 | [`12_da-vinci-and-us-core-interoperability.md`](12_da-vinci-and-us-core-interoperability.md) | Da Vinci and US Core Interoperability | US Core IG, Da Vinci CRD/DTR/PAS prior auth triad, and CMS-0057-F mandate. | 1,200-1,800 | complete |
| 13 | [`13_cds-hooks-and-smart-on-fhir.md`](13_cds-hooks-and-smart-on-fhir.md) | CDS Hooks and SMART on FHIR | Event-driven CDS Hooks, Cards, prefetch optimization, and OAuth2 app launch. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Architecture Patterns ->
02 SQL for Clinical Data ->
03 Python Engineering ->
04 Orchestration ->
05 FHIR Ingestion ->
06 DICOM Pipelines ->
07 Data Versioning and Lineage ->
08 De-identification Engineering ->
09 Security Engineering ->
10 Testing and Observability
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 0 |
| draft | 0 |
| complete | 10 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

