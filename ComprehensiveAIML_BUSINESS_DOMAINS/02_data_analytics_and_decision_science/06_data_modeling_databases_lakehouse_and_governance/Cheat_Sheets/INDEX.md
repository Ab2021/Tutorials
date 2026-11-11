# Module 06 â€” Data Modelling, Databases, Lakehouse and Governance — Cheat Sheets

## 1. Overview

This collection carries 3 reference card(s) and 0 troubleshooting artifact(s) for `Module 06 â€” Data Modelling, Databases, Lakehouse and Governance`. Cards teach a reusable method, calculation, or selection rule; the Common Issues file turns those rules into diagnostics. Status is measured, not asserted: a card is `authored` only when `Tools/manuscript_lint.py` exits 0 for it.

## 2. Topic ladder

| Level | Cards | Purpose |
|---|---|---|

## 3. Quick lookup

| If you need to… | Read |
|---|---|
| work on data contracts: grain, quality, delivery, and change control | [Data Contracts: Grain, Quality, Delivery, and Change Control](./01_Data_Contract_Checklist.md) |
| work on data model grain and join safety | [Data Model Grain and Join Safety](./02_Data_Model_Grain_and_Join_Safety.md) |
| work on data model notation | [Data Model Notation](./03_Data_Model_Notation.md) |

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
| 1 | [Data Contracts: Grain, Quality, Delivery, and Change Control](./01_Data_Contract_Checklist.md) | 1 | DERIVED | authored |
| 2 | [Data Model Grain and Join Safety](./02_Data_Model_Grain_and_Join_Safety.md) | 2 | DERIVED | authored |
| 3 | [Data Model Notation](./03_Data_Model_Notation.md) | 1 | DERIVED | authored |

## 9. Common Issues locator

- Not yet authored. The collection is incomplete until this artifact exists with at least 25 distinct issues.

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
- [ ] Exactly one Common Issues artifact passes lint, including the distinct issue-ID check.
- [ ] Independent technical review of each card.
