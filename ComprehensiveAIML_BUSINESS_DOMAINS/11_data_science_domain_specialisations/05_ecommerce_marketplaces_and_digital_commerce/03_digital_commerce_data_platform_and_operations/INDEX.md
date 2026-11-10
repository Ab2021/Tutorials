# Digital Commerce Data Platform and Operations

> **Family:** [E-commerce, Marketplaces and Digital Commerce](../INDEX.md)  
> **Purpose:** The shared data, experimentation, ML, and operations platform supporting commerce at scale.  
> **Company / context examples:** Retail and e-commerce platforms

## Detailed topic map

| # | Topic | Status |
|---:|---|---|
| 1 | event tracking | [authored](Cheat_Sheets/01_Commerce_Event_Schema_and_Identity.md) |
| 2 | customer and product 360 | planned |
| 3 | feature platform | [authored](Cheat_Sheets/04_25_Common_Issues.md) |
| 4 | experimentation | planned |
| 5 | real-time inventory | planned |
| 6 | order orchestration | [authored](Case_Studies/04_Order_Status_Agent.md) |
| 7 | analytics reliability | [authored](Cheat_Sheets/02_Inventory_Order_and_Analytics_Reliability.md) |
| 8 | privacy | planned |
| 9 | FinOps | [authored](Cheat_Sheets/03_FinOps_Observability_Incidents.md) |
| 10 | observability | [authored](Cheat_Sheets/03_FinOps_Observability_Incidents.md) |
| 11 | incident response | [authored](Case_Studies/01_Inventory_Freshness_Incident.md) |
## Applied collections

- [Case Studies](./Case_Studies/INDEX.md): decision-centred cases using the local v2 catalog and integrated outline standard.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): concise references for recurring design, analysis, and operating decisions.

## Scope boundary

Use public company material only where cited. Treat company-specific systems as unknown unless a primary or clearly labelled published source supports the claim. Learning cases may be simulated or illustrative, but cannot claim measured production impact.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR033 | Single-flight scope; jitter; fill admission; consistent hashing; old/new ring epochs; version fences and tombstones | How should cache fills and ring migration preserve freshness while limiting database amplification? | Cold-fleet and skewed-key load replays, database concurrency, invalidation lag and stale-fill/delete races |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
