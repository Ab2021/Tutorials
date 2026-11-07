# Marketing Analytics Playbook — Cheat Sheets

## 1. Overview
For marketing analysts and experiment owners comfortable with rates and business metrics, this collection supports measurement choices and incremental-effect decisions. The causal inference sheet is an unreviewed draft; readiness requires independent review and evidence specific to the campaign and population.

## 2. Topic ladder
| Level | Topic and outcome | Prerequisite / skill | Status |
|---|---|---|---|
| Foundation | Channel measurement map: connect a channel decision to an outcome and distinguish attribution from incrementality. | Campaign and conversion concepts. | Planned |
| Core | MMM assumptions: identify a defensible marketing mix model design and test a confounding counterexample. | Regression and time-series basics. | Planned |
| Core | Lifecycle metric definitions: define cohort, denominator and horizon; diagnose inconsistent retention. | Customer and event grain. | Planned |
| Core/production | [21 Causal inference and A/B experimentation](21_Causal_Inference_Marketing.md): define an estimand, compute lift and diagnose assignment failures. | Randomization, rates and uncertainty. | Draft; not independently reviewed |

## 3. Quick topic lookup
| Question | Exact section | Use and warning |
|---|---|---|
| How is incremental lift calculated? | [Formulas](21_Causal_Inference_Marketing.md#section-4--core-formulas-with-full-derivation) | Fix population, assignment and horizon; relative lift is undefined at zero control rate. |
| Why are assignment counts unexpected? | [Failure modes](21_Causal_Inference_Marketing.md#section-8--failure-modes--edge-cases) | Compare immutable assignment counts to the configured allocation before interpreting outcomes. |
| Which channel or lifecycle definition? | Topic ladder above | The map and lifecycle guide are planned, not authored. |

## 4. Reading paths
- **Beginner:** Learn rates, control groups and assignment; read [definitions](21_Causal_Inference_Marketing.md#section-3--foundational-definitions) and [worked example](21_Causal_Inference_Marketing.md#section-5--worked-numerical-example). The channel map remains planned.
- **Practitioner:** Read the causal draft through [selection](21_Causal_Inference_Marketing.md#section-7--comparison--selection-table) and failures; define the unit, estimand, comparator and horizon before interpreting lift.
- **Advanced:** Read the complete draft, including observational limits and economics; MMM and lifecycle metrics are planned extensions, not available prerequisites.

## 5. Prerequisite skills
Calculate proportions and differences, interpret uncertainty and random assignment, and recognize confounding, interference and cohort grain. The draft's [definitions](21_Causal_Inference_Marketing.md#section-3--foundational-definitions) introduce estimands and counterfactuals; consult its source references before applying an estimator.

## 6. What this collection does NOT cover
Channel measurement, MMM assumptions and lifecycle metric guides remain planned. Attribution, predictive accuracy and observational association alone do not establish incremental causal impact. This draft does not replace legal, privacy or business approval for a campaign.

## 7. Master symbol glossary
| Symbol | Meaning; unit / type / domain | First use |
|---|---|---|
| Y_i(1), Y_i(0) | Unit i's potential outcomes under treatment and control; outcome units, scalar for the specified horizon. | [Definitions](21_Causal_Inference_Marketing.md#section-3--foundational-definitions) |
| ATE | Average treatment effect in the declared population; outcome units per unit; a causal estimand, not automatically an observed difference. | [Definitions](21_Causal_Inference_Marketing.md#section-3--foundational-definitions) |
| Absolute lift | Treatment minus control conversion rate; dimensionless difference, often reported in percentage points. | [Definitions](21_Causal_Inference_Marketing.md#section-3--foundational-definitions) |

## 8. Quick Reference Card directory
| Card | Rule / use | Condition and avoid |
|---|---|---|
| [Causal inference card](21_Causal_Inference_Marketing.md#section-12--quick-reference-card) | Compare assigned groups for a defined estimand and outcome horizon. | Assignment and measurement assumptions matter; do not treat attributed sales as causal lift. |

## 9. Common Issues locator
The final Common Issues artifact is not authored; its count is unknown. Use [causal inference failures](21_Causal_Inference_Marketing.md#section-8--failure-modes--edge-cases): for allocation mismatch compare immutable assignments with the planned ratio; for duplicate assignment trace identity and first eligibility. These are hypothetical diagnostics, not observed incident counts.

## 10. Empirical sources, versions and collection maintenance
The original source directories were unavailable and not inspected. The causal draft's scenarios and arithmetic are illustrative, not measured campaign performance; source and independent review remain open. Consult [authoring standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the [master template](../../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). Recheck randomization, measurement, privacy rules and sources when they change; no review date is asserted. Each sheet requires 12 sections, checked examples, failure diagnostics, answered interview questions and a final card. Cards must state the decision, reader, inputs, assumptions, tradeoffs, failure signals and governance; review tool-, market- and regulation-sensitive cards quarterly.

## 11. Collection acceptance checklist
- [ ] Author the three planned guides and a final Common Issues artifact with at least 25 substantive issues.
- [ ] Independently review the causal draft, its 12 sections, arithmetic, assumptions and sources.
- [ ] Verify links, anchors, glossary and reading paths as files are added.
- [ ] Record actual review dates and distinguish illustrative campaign inputs from measured results.
