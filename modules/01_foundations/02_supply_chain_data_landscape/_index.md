# Module 02 — The Supply Chain Data Landscape

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
> **Folder:** `modules/01_foundations/02_supply_chain_data_landscape/`
> **Version:** 2026.5

---

## 1. Module purpose

This module maps where supply chain data actually lives, how trustworthy each source is, and what access, licensing, and quality constraints shape a planning use case.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 00, 01 |
| Unlocks | 03 |
| Assessment | Data-source map (Week 4) |
| Survey file | `02_supply_chain_data_landscape.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_systems-of-record.md`](01_systems-of-record.md) | Systems of Record | ERP, WMS, TMS, OMS, POS, PLM, CRM, and what each system can and cannot answer. | 1,200–1,800 | complete |
| 2 | [`02_master-data.md`](02_master-data.md) | Master Data | Item, location, supplier, customer, BOM, and the identifiers that join them. | 1,200–1,800 | complete |
| 3 | [`03_transaction-data.md`](03_transaction-data.md) | Transaction Data | Orders, shipments, receipts, invoices, inventory snapshots, and their grain. | 1,200–1,800 | complete |
| 4 | [`04_reference-and-external.md`](04_reference-and-external.md) | Reference Data and External Signals | Calendar, weather, macro, trade, and geospatial sources. | 1,200–1,800 | complete |
| 5 | [`05_telemetry-iot-vision.md`](05_telemetry-iot-vision.md) | Telemetry, IoT, and Computer Vision | Sensor, telematics, RFID, and image data as planning evidence. | 1,200–1,800 | complete |
| 6 | [`06_data-movement.md`](06_data-movement.md) | Data Movement: EDI, API, Flat File, Event | How each transport constrains latency, completeness, and schema stability. | 1,200–1,800 | complete |
| 7 | [`07_access-and-licensing.md`](07_access-and-licensing.md) | Access Pathways, Licensing, and Personal Data | Commercial data, partner data, privacy, and contractual limits on use. | 1,200–1,800 | complete |
| 8 | [`08_data-quality-dimensions.md`](08_data-quality-dimensions.md) | Data-Quality Dimensions | Completeness, validity, consistency, timeliness, uniqueness, and accuracy. | 1,200–1,800 | complete |
| 9 | [`09_master-data-drift.md`](09_master-data-drift.md) | Master-Data Drift, Worked | Item changes, pack-size changes, and the forecast history that silently breaks. | 1,200–1,800 | complete |
| 10 | [`10_source-selection-trust.md`](10_source-selection-trust.md) | Source Selection and the Hierarchy of Trust | Choosing a source for a decision and documenting its known defects. | 1,200–1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Systems of Record ->
02 Master Data ->
03 Transaction Data ->
04 Reference Data and External Signals ->
05 Telemetry, IoT, and Computer Vision ->
06 Data Movement: EDI, API, Flat File, Event ->
07 Access Pathways, Licensing, and Personal Data ->
08 Data-Quality Dimensions ->
09 Master-Data Drift, Worked ->
10 Source Selection and the Hierarchy of Trust
```

## 5. What this module does not cover

- It does not build pipelines or define schemas; Module 03 turns the landscape into engineered data.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 0 |
| draft | 0 |
| complete | 10 |

All ten topic files are written and meet the quality bar in `MODULE_EXPANSION_SPEC.md` §5: at
least 1,000 lines, all 17 sections present, arithmetic checked, sources real and graded. The
module total is 12,900 lines. Status moves from `complete` to `reviewed` when a second reader
signs off, and to `revised` after corrections.

## 7. Sources for this module

See section 14 of the survey file (`02_supply_chain_data_landscape.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
