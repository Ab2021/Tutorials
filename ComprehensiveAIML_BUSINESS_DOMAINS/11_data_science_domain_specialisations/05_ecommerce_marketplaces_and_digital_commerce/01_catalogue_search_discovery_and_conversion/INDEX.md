# Catalogue, Search, Discovery and Conversion

> **Family:** [E-commerce, Marketplaces and Digital Commerce](../INDEX.md)  
> **Purpose:** E-commerce discovery from product data and query understanding through ranking, content, conversion, and trust.  
> **Company / context examples:** Flipkart, marketplace and direct-commerce contexts

## Detailed topic map

| # | Topic | Status |
|---:|---|---|
| 1 | catalogue quality | planned |
| 2 | query understanding | planned |
| 3 | retrieval | planned |
| 4 | ranking | planned |
| 5 | recommendations | planned |
| 6 | content | planned |
| 7 | reviews | planned |
| 8 | conversion | planned |
| 9 | checkout | planned |
| 10 | fraud | planned |
| 11 | experimentation | planned |
| 12 | SEO | planned |

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
| PLAN-2026-IR021 | WAL and immutable segments; positional phrase queries; field-weighted BM25; exact top-k; ACLs and tombstones | How should search remain correct across indexing, deletion, ranking and shard failures? | Brute-force Boolean/phrase and global top-k oracles, delete visibility, retrieval quality, index lag and merge capacity |
| PLAN-2026-IR024 | Rarest-term anchor; lazy postings; skips; relative positions; repeated terms; matching analyzers | How can common-term phrase queries be evaluated without materializing huge posting lists? | Brute-force token-position oracle, peak memory and tail latency over frequent and repeated-term phrases |
| PLAN-2026-IR029 | Enqueue-time deduplication; paged adjacency; independent work budgets; deterministic traversal; explicit partial results | How should a recommendation API respond when graph expansion reaches a node, edge, memory or time limit? | High-degree and cyclic graph replays, peak memory, examined edges, output completeness and candidate utility |
| PLAN-2026-IR032 | Trie admission; time-window sketches; bounded candidate stores; immutable snapshots; real-time overlays; CDN versioning | Which terms enter a bounded suggestion index, and how can safe new trends appear promptly? | Memory including two generations, suggestion relevance, freshness lag, abusive-query tests and per-locale tail latency |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
