# Time-Series Playbook - Cheat Sheets

> **Mandatory authoring contract:** Follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md). This collection remains in draft: its authored sheet has not received independent review, the other references remain planned, and the final Common Issues artifact does not yet exist.

## 1. Overview

This collection supports forecasting decisions for analysts, engineers, and planners familiar with basic averages, timestamps, and supervised learning. Readiness requires explicit decision timing, honest evidence, actionable diagnostics, and completed review; a draft is not publication.

## 2. Topic ladder

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | Forecast method chooser | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 2 | Hierarchy reconciliation | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 3 | Forecast monitoring checklist | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 4 | [Rolling-origin validation and demand accuracy](04_Rolling_Origin_Validation_And_Demand_Accuracy.md) | As-of replay, forecast accuracy, leakage, replenishment implications, and answered interviews. | draft; not reviewed |

The authored sheet is usable with the external foundational skills listed above; the other ladder entries are planned rather than mandatory prerequisites.

## 3. Quick topic lookup

| Question | Location | Boundary |
|---|---|---|
| How should forecasts be replayed? | [Concept map](04_Rolling_Origin_Validation_And_Demand_Accuracy.md#2-concept-map--mental-model) | Match operational cadence and availability. |
| Which metric handles zero demand? | [Formulas](04_Rolling_Origin_Validation_And_Demand_Accuracy.md#4-core-formulas-with-full-derivation) | WAPE requires positive aggregate actuals. |
| Why is accuracy suspicious? | [Failure modes](04_Rolling_Origin_Validation_And_Demand_Accuracy.md#8-failure-modes--edge-cases) | Check feature vintages and population before attribution. |

## 4. Reading paths

- **Beginner:** Learn averages, absolute values, and timestamp ordering; then read sections 1–6 of the draft.
- **Practitioner:** Read sections 1–9 in order and apply the replay to an actual decision contract.
- **Advanced:** Read all sections, including results, interview reasoning, and operational limits; other topics in the ladder remain planned.

## 5. Prerequisite skills

Readers need arithmetic with units and percentages, basic supervised learning, and a practical understanding of event versus availability timestamps. The sheet explains its forecasting-specific terms and equations.

## 6. Exclusions and handoffs

Full model fitting, causal promotion estimation, inventory optimization, and comprehensive hierarchy reconciliation are outside this sheet. The [parent playbook](../INDEX.md) lists related topics as planned; do not treat them as published handoffs.

## 7. Master symbol glossary

| Symbol | Meaning and unit | Type and domain | First use |
|---|---|---|---|
| `i` | Series identifier | Index | [Definitions and formulas](04_Rolling_Origin_Validation_And_Demand_Accuracy.md#4-core-formulas-with-full-derivation) |
| `o` | Forecast origin | Time index | Same |
| `h` | Forecast horizon, periods | Positive integer | Same |
| `y`, `f` | Actual and forecast, target units | Numeric cells | Same |
| `e` | Forecast minus actual, target units | Numeric cell | Same |
| `n` | Number of scored pairs | Positive integer | Same |
| `w` | Scoring weight, dimensionless unless specified | Nonnegative numeric cell | Same |
| `m`, `T` | Seasonal period and training length, periods | Integers with `T > m > 0` | Same |
| `q` | Quantile level | Scalar strictly between zero and one | Same |
| `N`, `K`, `H`, `P` | Series, origins, horizons, feature counts | Nonnegative integers | [Cost analysis](04_Rolling_Origin_Validation_And_Demand_Accuracy.md#9-complexity--cost-analysis) |

## 8. Quick Reference Card directory

| File | Rule | Critical condition |
|---|---|---|
| [Rolling-origin card](04_Rolling_Origin_Validation_And_Demand_Accuracy.md#12-quick-reference-card) | Replay decision-time information and score matched cells. | WAPE requires positive total actual demand. |

## 9. Common Issues locator

The final Common Issues artifact is **planned**, with zero authored issues in that artifact. The draft sheet's separate [failure-mode examples](04_Rolling_Origin_Validation_And_Demand_Accuracy.md#8-failure-modes--edge-cases) do not satisfy the collection artifact requirement.

## 10. Sources, versions, and maintenance

The authored sheet is version 1.0, draft, and not independently reviewed. It distinguishes derived and illustrative examples from open evidence. Its cited preprocessing source is an unpinned stable documentation page; recheck it before publication. Revisit the sheet when decision cadence, data availability, labels, or source definitions change.

## 11. Collection acceptance checklist

- [x] One actual draft is indexed with its status and a card link.
- [ ] Independently review the draft's arithmetic, sources, links, and template compliance.
- [ ] Author planned references and the final Common Issues artifact.
- [ ] Complete collection-wide prerequisite, evidence, navigation, and publication checks.
