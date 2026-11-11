# Credit Risk, Markets and Customer Finance — Cheat Sheets

## 1. Overview

This collection carries 12 reference card(s) and 1 troubleshooting artifact(s) for `Credit Risk, Markets and Customer Finance`. Cards teach a reusable method, calculation, or selection rule; the Common Issues file turns those rules into diagnostics. Status is measured, not asserted: a card is `authored` only when `Tools/manuscript_lint.py` exits 0 for it.

## 2. Topic ladder

| Level | Cards | Purpose |
|---|---|---|
| troubleshooting | 25 Common Issues | Diagnose and prevent recurring failures |

## 3. Quick lookup

| If you need to… | Read |
|---|---|
| work on credit data and point-in-time construction | [Credit data and point-in-time construction](./01_Credit_Data_and_Point_in_Time.md) |
| work on underwriting policy under loss constraints | [Underwriting policy under loss constraints](./02_Underwriting_Policy.md) |
| work on scorecards and calibration | [Scorecards and calibration](./03_Scorecards_and_Calibration.md) |
| work on portfolio risk aggregation | [Portfolio risk aggregation](./04_Portfolio_Risk.md) |
| work on stress testing and scenario translation | [Stress testing and scenario translation](./05_Stress_Testing.md) |
| work on market signal time-series discipline | [Market signal time-series discipline](./06_Market_Time_Series.md) |
| work on collections prioritization | [Collections prioritization](./07_Collections_Prioritization.md) |
| work on customer offers and value measurement | [Customer offers and value measurement](./08_Customer_Offers.md) |
| work on fair-lending assessment as a measurement exercise | [Fair-lending assessment as a measurement exercise](./09_Fair_Lending_Assessment.md) |
| work on explainability and adverse-action reasoning | [Explainability and adverse-action reasoning](./10_Explainability_and_Adverse_Actions.md) |
| work on credit model governance | [Credit model governance](./11_Model_Governance.md) |
| work on portfolio monitoring and intervention | [Portfolio monitoring and intervention](./12_Portfolio_Monitoring.md) |
| diagnose a failure that keeps recurring | [25 Common Issues](./13_25_Common_Issues.md) |

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
| 1 | [Credit data and point-in-time construction](./01_Credit_Data_and_Point_in_Time.md) | 2 | DERIVED | draft (lint open) |
| 2 | [Underwriting policy under loss constraints](./02_Underwriting_Policy.md) | 2 | DERIVED | draft (lint open) |
| 3 | [Scorecards and calibration](./03_Scorecards_and_Calibration.md) | 2 | DERIVED | draft (lint open) |
| 4 | [Portfolio risk aggregation](./04_Portfolio_Risk.md) | 2 | DERIVED | draft (lint open) |
| 5 | [Stress testing and scenario translation](./05_Stress_Testing.md) | 2 | DERIVED | draft (lint open) |
| 6 | [Market signal time-series discipline](./06_Market_Time_Series.md) | 2 | DERIVED | draft (lint open) |
| 7 | [Collections prioritization](./07_Collections_Prioritization.md) | 2 | DERIVED | draft (lint open) |
| 8 | [Customer offers and value measurement](./08_Customer_Offers.md) | 2 | DERIVED | draft (lint open) |
| 9 | [Fair-lending assessment as a measurement exercise](./09_Fair_Lending_Assessment.md) | 2 | DERIVED | draft (lint open) |
| 10 | [Explainability and adverse-action reasoning](./10_Explainability_and_Adverse_Actions.md) | 2 | DERIVED | draft (lint open) |
| 11 | [Credit model governance](./11_Model_Governance.md) | 2 | DERIVED | draft (lint open) |
| 12 | [Portfolio monitoring and intervention](./12_Portfolio_Monitoring.md) | 2 | DERIVED | draft (lint open) |
| 13 | [25 Common Issues](./13_25_Common_Issues.md) | troubleshooting | DERIVED | draft (lint open) |

## 9. Common Issues locator

- [25 Common Issues](./13_25_Common_Issues.md) — the unit's final troubleshooting artifact; entries use stable `CI-` ids.

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
