# Enterprise Knowledge, Support and Sales

> **Family:** [SaaS, Telecom and B2B](../INDEX.md)  
> **Purpose:** RAG, agents, documents, CRM, support operations, and sales intelligence under access controls.  
> **Company / context examples:** Enterprise software and B2B functions

## Detailed topic map

| # | Topic | Status |
|---:|---|---|
| 1 | knowledge ingestion | planned |
| 2 | permissions | planned |
| 3 | RAG | planned |
| 4 | agents | planned |
| 5 | CRM | planned |
| 6 | ticketing | planned |
| 7 | sales intelligence | planned |
| 8 | documents | planned |
| 9 | evaluation | planned |
| 10 | security | planned |
| 11 | human review | planned |
| 12 | change management | planned |

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
| PLAN-2026-IR003 | Authorized retrieval; effective policy versions; claim-level citations; embedding migrations | When should a support assistant answer, abstain or escalate a policy question? | Unsupported-claim rate, eligible-span recall, abstention quality and policy-version errors against expert labels |
| PLAN-2026-IR004 | LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints | How should tenant adapters be trained, selected, versioned and rolled back? | Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix |
| PLAN-2026-IR012 | Structure-aware chunking; actual tokenizer limits; HNSW and exact filtered baselines; CDC; deletion propagation | Which chunking and index design retrieves every required authorized span within the service budget? | All-required-span success, filtered recall, duplicate context, ACL violations, freshness lag and per-tenant latency |
| PLAN-2026-IR015 | Product/date eligibility; grounded generation; native-language review; source dependencies; supersession triggers | Which generated scripts are fit for an approved, versioned coaching registry? | Unsupported product claims, expired evidence, read-aloud time, reviewer agreement and agent usability |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
