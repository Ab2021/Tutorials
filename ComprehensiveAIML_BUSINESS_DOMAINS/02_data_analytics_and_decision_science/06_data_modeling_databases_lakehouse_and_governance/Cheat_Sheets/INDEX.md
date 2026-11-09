# Module 06 Cheat Sheets — Data Modelling, Databases, Lakehouse and Governance

## 1. Overview

This collection teaches the reusable controls behind durable analytical data products: declared grain, model choice, query access patterns, lineage, cataloguing, and governed change. It is for analysts, analytics engineers, and reviewers deciding whether a dataset can support a metric or operational decision. Readiness means the reader can state an input/output contract, calculate a quality signal with its denominator, select a storage/model pattern under stated constraints, and diagnose a mismatch without inventing production evidence.

## 2. Topic ladder

| Level | File | Outcome | Prerequisites | Key skill | Status |
|---|---|---|---|---|---|
| Foundation | [01 Data contracts](01_Data_Contract_Checklist.md) | Specify grain, quality, delivery, and change controls; compute coverage | SQL keys and aggregation | `V/E`, freshness watermark | Draft; expanded 2026-11-09; not independently reviewed |
| Core | [02 data model grain and join safety](02_Data_Model_Grain_and_Join_Safety.md) | Select model and query grain; detect fan-out | 01 | functional dependencies, join cardinality | Draft; not independently reviewed |
| Core | 03 lakehouse tables, snapshots, and governance | Select table/version/lineage controls | 01–02 | snapshot and mutation invariants | Planned |
| Advanced | 04 metadata, lineage, and catalogue review | Trace a metric to source and owner | 01 | lineage/reconciliation | Planned |
| Advanced | 05 authorized retrieval, search indexing, and bounded hierarchies | Defend deletion, ACL, and traversal controls | 01–04 | bounded work and snapshot reads | Planned; reference-informed |
| Final | 06 Common Issues | Diagnose at least 25 distinct controls | 01–05 | discriminating checks | Planned; no artifact exists |

## 3. Quick topic lookup

| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| “What does one row mean?” | [01 §3](01_Data_Contract_Checklist.md#3-foundational-definitions) | before metrics or joins | a primary key alone does not state grain |
| “Are we complete?” | [01 §4–5](01_Data_Contract_Checklist.md#4-core-formulas-with-full-derivation) | independent expected count exists | `R/R` is not completeness |
| “Can this field change safely?” | [01 §7–8](01_Data_Contract_Checklist.md#7-comparison--selection-table) | producer release | parser compatibility can hide semantic drift |
| “Why did a metric jump?” | [01 §8](01_Data_Contract_Checklist.md#8-failure-modes--edge-cases) | after release | do not attribute change before checking grain/join/unit |

## 4. Reading paths — three options

- **Beginner:** 01, then the planned 02; practice writing a key, grain, expected-count, and freshness statement.
- **Practitioner:** 01 → 02 → 03 → 04; add reconciliation, lineage, and release authority to one live dataset design.
- **Advanced:** all files in ladder order, then Common Issues. Optional forward extension: Module 09 for CDC/event-time controls.

## 5. Prerequisite skills

Readers need SQL `GROUP BY`, joins and null handling; basic ratios and UTC timestamps; and the ability to distinguish source event time from processing time. The contract sheet explicitly supplies the data-product vocabulary. Vendor-specific table formats and catalog APIs are deliberately not assumed.

## 6. What this collection does NOT cover

It does not certify legal compliance, prescribe a warehouse vendor, prove proprietary source-system behavior, or replace security threat modelling. Streaming delivery semantics hand off to Module 09; causal validity handoffs go to Module 08; forecasting feature readiness handoffs go to Module 10.

## 7. Master symbol glossary

| Symbol / term | Meaning | Units / type | Domain | First appearance |
|---|---|---|---|---|
| `E` | independently expected records | records/window | nonnegative integer | [01 §3](01_Data_Contract_Checklist.md#3-foundational-definitions) |
| `R` | unique received records | records/window | `0..E` unless replay/excess is diagnosed | [01 §3](01_Data_Contract_Checklist.md#3-foundational-definitions) |
| `V` | valid received records | records/window | `0..R` | [01 §3](01_Data_Contract_Checklist.md#3-foundational-definitions) |
| `c` | completeness `R/E` | dimensionless | known `E>0` | [01 §4](01_Data_Contract_Checklist.md#4-core-formulas-with-full-derivation) |
| `v` | validity `V/R` | dimensionless | `R>0` | [01 §4](01_Data_Contract_Checklist.md#4-core-formulas-with-full-derivation) |
| watermark | last asserted-complete cutoff | UTC instant | declared population | [01 §4](01_Data_Contract_Checklist.md#4-core-formulas-with-full-derivation) |

## 8. Quick Reference Card directory

| File / card | Core rule | Use condition | Critical assumption | Avoid condition |
|---|---|---|---|---|
| [01 §12](01_Data_Contract_Checklist.md#12-quick-reference-card) | identity + `V/E` + complete watermark | release a declared decision dataset | scope and denominator match | source population is unknown |

## 9. Common Issues locator

`06_25_Common_Issues.md` is planned and does not yet exist. No issue count is claimed. Until authored, use the five diagnostic entries in [01 §8](01_Data_Contract_Checklist.md#8-failure-modes--edge-cases) as topic-local examples, not a collection troubleshooting artifact.

## 10. Empirical sources, versions and collection maintenance

| File | Evidence / sources | Status | Review trigger | Open limitation |
|---|---|---|---|---|
| 01 | DERIVED formulas; VERIFIED illustrative arithmetic; Apache Avro, OpenLineage, and scikit-learn documentation linked in §10 | Draft | change in definitions, source/version, or contract mechanism | no source body, real run, or measured dataset is claimed |
| 02–06 | no artifact | Planned | authoring | no claims or outcomes available |

`directory.txt` was consulted as an inventory. It contains visible vector-database entries but no accessible, authoritative data-contract source body; filenames are not treated as factual evidence.

## 11. Collection acceptance checklist

- [ ] Every ladder artifact exists and retains the required 12 sections.
- [ ] Every formula has validity, derivation, units, checked arithmetic, and a boundary.
- [ ] Every sheet has 10–15 answered interview questions and four or more diagnostic failures.
- [ ] The final Common Issues artifact contains at least 25 complete distinct issues and is last.
- [x] Existing 01 links, glossary, reading paths, evidence boundary, and final card are navigable.
- [ ] Primary sources and any empirical claims have been independently reviewed.
