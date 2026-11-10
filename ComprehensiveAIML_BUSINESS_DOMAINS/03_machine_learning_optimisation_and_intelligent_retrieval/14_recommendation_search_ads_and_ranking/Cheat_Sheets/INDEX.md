# Module 14 â€” Recommendation, Search, Ads and Ranking — Cheat Sheets

## 1. Overview

This collection carries 6 reference card(s) and 1 troubleshooting artifact(s) for `Module 14 â€” Recommendation, Search, Ads and Ranking`. Cards teach a reusable method, calculation, or selection rule; the Common Issues file turns those rules into diagnostics. Status is measured, not asserted: a card is `authored` only when `Tools/manuscript_lint.py` exits 0 for it.

## 2. Topic ladder

| Level | Cards | Purpose |
|---|---|---|
| core | Ranking metrics and candidate generation, Recommendation architectures, Online Experimentation and Bandits, Retrieval index correctness and BM25, Phrase and positional retrieval, Autocomplete admission and freshness | Build from prerequisites to production judgement |
| advanced | Common issues: recommendation and retrieval | Build from prerequisites to production judgement |

## 3. Quick lookup

| If you need to… | Read |
|---|---|
| work on ranking metrics and candidate generation | [Ranking metrics and candidate generation](./01_ranking_metrics.md) |
| work on recommendation architectures | [Recommendation architectures](./02_recsys_architectures.md) |
| work on online experimentation and bandits | [Online Experimentation and Bandits](./03_online_experimentation_checklist.md) |
| work on retrieval index correctness and bm25 | [Retrieval index correctness and BM25](./04_retrieval_index_correctness.md) |
| work on phrase and positional retrieval | [Phrase and positional retrieval](./05_phrase_and_positional_retrieval.md) |
| work on autocomplete admission and freshness | [Autocomplete admission and freshness](./06_autocomplete_admission_and_freshness.md) |
| diagnose a failure that keeps recurring | [Common issues: recommendation and retrieval](./07_25_Common_Issues.md) |

## 4. Reading paths

| Path | Sequence |
|---|---|
| Beginner | foundation cards first, then the Quick Reference Card (§12) of each core card |
| Practitioner | core cards in ladder order, then one case in `../Case_Studies/` that cites them |
| Advanced | advanced cards, then the Common Issues artifact and its discriminating checks |

## 5. Prerequisite skills

| Skill | Why it is needed | Where it is available |
|---|---|---|
| Unit topic map | Defines what this collection is responsible for and what it excludes | `../INDEX.md` |
| Decision framing | Cards assume a stated decision, population, and cut-off | sibling case collection |
| Arithmetic discipline | Every card carries checked numbers with units | the cards themselves |

## 6. Exclusions and handoffs

| Excluded here | Handed to |
|---|---|
| Decision-context narrative, baselines, and outcomes | `../Case_Studies/` |
| Organisational policy and legal determination | governance modules, not this collection |
| Vendor-specific or unverified current API behaviour | treat as OPEN until sourced at authoring time |

## 7. Master symbol glossary

| Symbol / term | Meaning | Units | Shape / type | Domain | First use |
|---|---|---|---|---|---|
| — | no card defines a symbol yet | — | — | — | — |

## 8. Card directory

| # | Card | Level | Evidence | Status |
|---|---|---|---|---|
| 1 | [Ranking metrics and candidate generation](./01_ranking_metrics.md) | core | DERIVED | authored |
| 2 | [Recommendation architectures](./02_recsys_architectures.md) | core | DERIVED | authored |
| 3 | [Online Experimentation and Bandits](./03_online_experimentation_checklist.md) | core | DERIVED | draft (lint open) |
| 4 | [Retrieval index correctness and BM25](./04_retrieval_index_correctness.md) | core | DERIVED | authored |
| 5 | [Phrase and positional retrieval](./05_phrase_and_positional_retrieval.md) | core | DERIVED | authored |
| 6 | [Autocomplete admission and freshness](./06_autocomplete_admission_and_freshness.md) | core | DERIVED | authored |
| 7 | [Common issues: recommendation and retrieval](./07_25_Common_Issues.md) | advanced | DERIVED | draft (lint open) |

## 9. Common Issues locator

- [Common issues: recommendation and retrieval](./07_25_Common_Issues.md) — the unit's final troubleshooting artifact; entries use stable `CI-` ids.

## 10. Sources, versions, maintenance

| Field | Value |
|---|---|
| Evidence vocabulary | DERIVED / VERIFIED / SOURCED / ILLUSTRATIVE / HEURISTIC / OPEN |
| Version policy | Content is dated by its own frontmatter; unstable API, price, and regulatory facts are rechecked at edit time |
| Maintenance trigger | A card's validity condition, a source body, or a tool's semantics changes |
| Review cadence | Quarterly, plus on any material change to a cited source |

## 11. Acceptance checklist

- [x] Every linked card exists on disk.
- [x] Status column is measured by the lint, not asserted.
- [x] Planned commitments remain listed without links.
- [x] A final Common Issues artifact exists with at least 25 distinct issues.
- [ ] Independent technical review of each card.
