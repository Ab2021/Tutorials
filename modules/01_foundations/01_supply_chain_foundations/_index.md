# Module 01 — Supply Chain Foundations for Data Scientists

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
> **Folder:** `modules/01_foundations/01_supply_chain_foundations/`
> **Version:** 2026.5

---

## 1. Module purpose

This module gives the physical and commercial vocabulary an analyst needs before touching a model: flows, entities, planning processes, inventory, service, cost, and working capital.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 00 |
| Unlocks | 07, 10, 11, 12, 13, 14, 15, 16, 17 |
| Assessment | Supply-chain decision map (Week 2) |
| Survey file | `01_supply_chain_foundations.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_three-flows-one-network.md`](01_three-flows-one-network.md) | Three Flows, One Network | Physical, information, and financial flows and how a decision propagates across them. | 1,200–1,800 | complete |
| 2 | [`02_decision-stack.md`](02_decision-stack.md) | The Decision Stack | Strategic, tactical, operational, and execution decisions with cadence and horizon. | 1,200–1,800 | complete |
| 3 | [`03_network-entities.md`](03_network-entities.md) | Network Entities and Product Flow | Nodes, lanes, DCs, plants, suppliers, customers, and the bill of materials. | 1,200–1,800 | complete |
| 4 | [`04_planning-processes.md`](04_planning-processes.md) | Planning Processes: S&OP, IBP, and Execution | Who meets, when, with what number, and what they can actually change. | 1,200–1,800 | complete |
| 5 | [`05_inventory-and-service.md`](05_inventory-and-service.md) | Inventory Types and Service Metrics | Cycle, safety, in-transit, and obsolete stock; CSL, fill rate, OTIF. | 1,200–1,800 | complete |
| 6 | [`06_cost-and-working-capital.md`](06_cost-and-working-capital.md) | Cost Taxonomy and Working Capital | Ordering, holding, shortage, transport, DIO, DSO, DPO, and cash conversion. | 1,200–1,800 | complete |
| 7 | [`07_manufacturing-basics.md`](07_manufacturing-basics.md) | Manufacturing: BOM, Routing, MRP, Capacity | How a product is built and how the plan explodes into components and load. | 1,200–1,800 | complete |
| 8 | [`08_transport-and-incoterms.md`](08_transport-and-incoterms.md) | Transportation Modes and Incoterms 2020 | Mode economics, responsibility transfer, and landed-cost composition. | 1,200–1,800 | planned |
| 9 | [`09_warehousing-and-sourcing.md`](09_warehousing-and-sourcing.md) | Warehousing and Sourcing | Receiving to shipping, and the Kraljic portfolio as a sourcing lens. | 1,200–1,800 | planned |
| 10 | [`10_archetypes-and-metric-conflicts.md`](10_archetypes-and-metric-conflicts.md) | Supply Chain Archetypes and Metric Conflicts | Efficient versus responsive chains and the metrics that trade against each other. | 1,200–1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Three Flows, One Network ->
02 The Decision Stack ->
03 Network Entities and Product Flow ->
04 Planning Processes: S&OP, IBP, and Execution ->
05 Inventory Types and Service Metrics ->
06 Cost Taxonomy and Working Capital ->
07 Manufacturing: BOM, Routing, MRP, Capacity ->
08 Transportation Modes and Incoterms 2020 ->
09 Warehousing and Sourcing ->
10 Supply Chain Archetypes and Metric Conflicts
```

## 5. What this module does not cover

- It does not cover forecasting, optimization algorithms, or data engineering; those are Modules 07, 11, and 03.
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

See section 14 of the survey file (`01_supply_chain_foundations.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
