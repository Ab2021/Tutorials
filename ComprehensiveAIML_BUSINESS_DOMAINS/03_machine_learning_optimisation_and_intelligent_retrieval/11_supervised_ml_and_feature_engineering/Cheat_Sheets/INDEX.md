# Module 11 â€” Supervised Machine Learning and Feature Engineering — Cheat Sheets

## 1. Overview

This collection carries 5 reference card(s) and 1 troubleshooting artifact(s) for `Module 11 â€” Supervised Machine Learning and Feature Engineering`. Cards teach a reusable method, calculation, or selection rule; the Common Issues file turns those rules into diagnostics. Status is measured, not asserted: a card is `authored` only when `Tools/manuscript_lint.py` exits 0 for it.

## 2. Topic ladder

| Level | Cards | Purpose |
|---|---|---|
| core | Linear trees and boosting, Imbalance and Calibration | Build from prerequisites to production judgement |
| advanced | Common issues: supervised ML | Build from prerequisites to production judgement |

## 3. Quick lookup

| If you need to… | Read |
|---|---|
| work on model validation and error analysis | [Model Validation and Error Analysis](./01_Model_Validation_and_Error_Analysis.md) |
| work on problem formulation and losses | [Problem Formulation and Losses](./02_Problem_Formulation_and_Losses.md) |
| work on linear trees and boosting | [Linear trees and boosting](./03_Linear_Trees_and_Boosting.md) |
| work on feature engineering patterns | [Feature engineering patterns](./04_Feature_Engineering_Patterns.md) |
| work on imbalance and calibration | [Imbalance and Calibration](./05_Imbalance_and_Calibration.md) |
| diagnose a failure that keeps recurring | [Common issues: supervised ML](./06_25_Common_Issues.md) |

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
| 1 | [Model Validation and Error Analysis](./01_Model_Validation_and_Error_Analysis.md) | 2 | DERIVED | authored |
| 2 | [Problem Formulation and Losses](./02_Problem_Formulation_and_Losses.md) | 1 | DERIVED | draft (lint open) |
| 3 | [Linear trees and boosting](./03_Linear_Trees_and_Boosting.md) | core | DERIVED | draft (lint open) |
| 4 | [Feature engineering patterns](./04_Feature_Engineering_Patterns.md) | 2 | DERIVED | draft (lint open) |
| 5 | [Imbalance and Calibration](./05_Imbalance_and_Calibration.md) | core | DERIVED | draft (lint open) |
| 6 | [Common issues: supervised ML](./06_25_Common_Issues.md) | advanced | DERIVED | draft (lint open) |

## 9. Common Issues locator

- [Common issues: supervised ML](./06_25_Common_Issues.md) — the unit's final troubleshooting artifact; entries use stable `CI-` ids.

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
