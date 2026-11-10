# Module 06 â€” Data Modelling, Databases, Lakehouse and Governance

> **Part:** [Part 02 â€” Data, Analytics and Decision Science](../INDEX.md)  
> **Prerequisites:** 02,05  
> **Purpose:** Model entities, build reliable data products, and choose warehouse, lakehouse, or operational patterns deliberately.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | dimensional and normalized models | planned |
| 2 | transactional and analytical databases | planned |
| 3 | lakehouse patterns | [authored](Case_Studies/01_model_an_ecommerce_lakehouse.md) |
| 4 | metadata lineage and catalogues | planned |
| 5 | data contracts | [authored](Cheat_Sheets/01_Data_Contract_Checklist.md) |
## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored course-topic drafts

- [Data Contract Checklist](Cheat_Sheets/01_Data_Contract_Checklist.md) — ownership, grain, quality and change controls; visible source inventory only.
- [Data Model Notation](Cheat_Sheets/03_Data_Model_Notation.md) — entity identity, cardinality, optionality and temporal intervals; illustrative derivation and trace only.
- [Data Model Grain and Join Safety](Cheat_Sheets/02_Data_Model_Grain_and_Join_Safety.md) — distinct coverage/multiplicity diagnostics and constructed executable SQLite oracles; no production run is claimed.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR012 | Structure-aware chunking; actual tokenizer limits; HNSW and exact filtered baselines; CDC; deletion propagation | Which chunking and index design retrieves every required authorized span within the service budget? | All-required-span success, filtered recall, duplicate context, ACL violations, freshness lag and per-tenant latency |
| PLAN-2026-IR021 | WAL and immutable segments; positional phrase queries; field-weighted BM25; exact top-k; ACLs and tombstones | How should search remain correct across indexing, deletion, ranking and shard failures? | Brute-force Boolean/phrase and global top-k oracles, delete visibility, retrieval quality, index lag and merge capacity |
| PLAN-2026-IR031 | N+1 detection; adjacency-list indexing; per-step authorization; cycle checks; stable assembly; snapshot reads | Should a subtree API use a recursive CTE or bounded level-order batching for its workload? | Reference traversal comparisons, round trips, rows scanned, unauthorized descendants, deep/wide trees and truncation behavior |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
