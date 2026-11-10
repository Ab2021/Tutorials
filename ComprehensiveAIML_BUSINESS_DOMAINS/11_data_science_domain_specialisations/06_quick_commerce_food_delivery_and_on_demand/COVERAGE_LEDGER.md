# Quick-commerce family coverage ledger

**Scope:** this ledger is confined to the Quick Commerce, Food Delivery and On-Demand family. It records authoring evidence, not an assertion that a planned catalog is complete. **Checked:** 2026-11-09. **Source-body boundary:** the three linked interview references were readable; the source inventory names and focused-plan rows are planning inputs, not evidence of proprietary implementations or outcomes.

## Status vocabulary

- **planned:** a catalog commitment with no teaching artifact.
- **draft:** an authored artifact whose structure, numerical work, links, and independent review are still open.
- **complete authorship:** all required sections and local checks exist; this state is not claimed below until an artifact is actually present.
- **source uncertainty:** the listed source establishes only the stated visible mechanism; missing or clipped source text remains open.

## Teaching-home reconciliation

| Teaching home | Original commitments | Reference-informed commitments | Authored artifact evidence | Current truthful state | Next acceptance evidence |
|---|---|---|---|---|---|
| Inventory and assortment | dark-store operations; local assortment; inventory accuracy; perishables; substitutions; replenishment; forecasting; waste; vendor fill; availability promises; experiments | IR027: Fenwick lowbit traversal, inclusive range totals, point deltas, wide arithmetic, atomic visibility | No individual sheet or case existed at check time | planned | Per-artifact 12/15-section content, checked traces, cases, navigation, and issue artifact |
| Dispatch, ETA and operations | order state; demand forecast; dispatch; batching; routing; ETA; rider supply; surge; service recovery; weather; fraud and safety; real-time monitoring | IR019: coherent traffic snapshot, CH compatibility, Dijkstra fallback, map matching, route/ETA distinction; IR023: shortcut expansion, closure snapshot, directed nonnegative fallback, unreachable pairs | No individual sheet or case existed at check time | planned | Per-artifact 12/15-section content, oracle replays, cases, navigation, and issue artifact |
| Growth, pricing and unit economics | cohort economics; acquisition; promotion; delivery fees; membership; retention; CLV; price sensitivity; market-level supply; fraud; causal measurement; governance | No destination-specific reference row found in the local plan | No individual sheet or case existed at check time | planned | Per-artifact 12/15-section content, checked economics examples, cases, navigation, and issue artifact |
| Family cross-domain collections | coupled inventory–dispatch–pricing decision support | no additional named source row | No individual cross-domain artifact existed at check time | planned | One bounded service-admission case and reusable integrated decision sheet if retained in catalog |

## Readable source-reference register

| ID | Visible source commitment | Teaching boundary | Evidence status | Required verification / limitation |
|---|---|---|---|---|
| REF-IR027 | Fenwick tree range totals: one-based lowbit traversal; inclusive `prefix(r)-prefix(l-1)`; point deltas; `O(log n)` operations; snapshot hazard | Inventory totals only; it is not distributed inventory correctness | SOURCE READ | Reference uses illustrative quantities and assumes fixed ordering and one logical writer; randomized array oracle, overflow, replay, and visibility checks remain to be executed per authored artifact |
| REF-IR019 | Navigation teaching brief: coherent snapshots, CH compatibility, base-graph fallback, map matching, and distinct routing/ETA objectives | Delivery routing and ETA evaluation; not a company route-service claim | PLAN ONLY | The named broader source text was not used as a benchmark or implementation record; public/source body evidence remains open |
| REF-IR023 | Closure-aware routing: directed nonnegative Dijkstra baseline; recursively validate shortcut paths; filtering invalid shortcuts alone does not prove exactness; fall back to base graph for exact contract | Delivery routing only; time-dependent and turn-state routing need a separate contract | SOURCE READ | Reference labels scenario requirements and example values illustrative; no production latency or incident result is claimed |

## Collection gates still open

| Collection | Required authored sheets | Required authored cases | Common Issues requirement | Navigation state |
|---|---:|---:|---:|---|
| Inventory sheets | 3 listed references + IR027 teaching outcome | n/a | >=25 complete issues in final artifact | catalog exists; not yet 11-section compliant |
| Inventory cases | n/a | 5 listed cases + IR027 decision outcome | n/a | catalog has six sections plus reference appendix; not yet seven-section compliant |
| Dispatch sheets | 3 listed references + IR019 + IR023 teaching outcomes | n/a | >=25 complete issues in final artifact | catalog exists; not yet 11-section compliant |
| Dispatch cases | n/a | 5 listed cases + IR019 + IR023 decision outcomes | n/a | catalog has six sections plus reference appendix; not yet seven-section compliant |
| Economics sheets | 3 listed references | n/a | >=25 complete issues in final artifact | catalog exists; not yet 11-section compliant |
| Economics cases | n/a | 5 listed cases | n/a | catalog has six sections; not yet seven-section compliant |
| Family sheets/cases | integrated decision-support artifact if retained | bounded cross-domain case if retained | >=25 complete cross-domain issues if collection is authored | catalogs are brief-only |

## Local verification record

No numerical or navigation smoke test is recorded as passed in this ledger because no individual teaching artifacts were present to test. The reference examples themselves contain checkable teaching arithmetic, but their source text is not a verification of an authored artifact. Re-run local link, heading-order, final-card, numerical-oracle, and issue-count checks after artifacts exist.
