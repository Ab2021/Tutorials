# Module 11 Cheat Sheets — Supervised ML and Feature Engineering

## 1. Overview

This collection teaches the decision contracts behind tabular supervised learning: define the prediction before fitting, construct available features, select a loss and threshold under operational constraints, and validate errors under the dependencies deployment must survive. It assumes probability, basic linear algebra, and a binary confusion matrix. A learner is ready to use a method only when they can state its information boundary, calculate its operating metric, name a disqualifying assumption, and propose a bounded diagnostic. The [authoring standard](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [cheat-sheet master](../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md) govern all individual artifacts.

## 2. Topic Ladder

| Level | File | Outcome | Prerequisites | Status |
|---|---|---|---|---|
| 1 | [02 Problem Formulation and Losses](02_Problem_Formulation_and_Losses.md) | turn an action, horizon and error cost into a target/loss contract | probability | draft; unreviewed |
| 1 | planned `03_Linear_Trees_and_Boosting.md` | compare linear, tree and boosted predictors under feature and deployment constraints | formulation | planned |
| 1 | planned `04_Feature_Engineering_Patterns.md` | design time-safe transformations and encodings | formulation | planned |
| 1 | Classification metrics — original planned scope | derive confusion-matrix, rank and probability metrics with prevalence, capacity and maturity boundaries | probability; formulation | planned; validation and calibration drafts do not replace focused metric coverage |
| 2 | planned `05_Imbalance_and_Calibration.md` | select prevalence-aware metrics and capacity threshold | probability | planned |
| 2 | Model selection — original planned scope | compare credible baselines and candidates using leakage-safe nested selection, uncertainty and deployment constraints | features; classification metrics; formulation | planned; the linear/tree/boosting comparison remains a separate commitment |
| 2 | [01 Model Validation and Error Analysis](01_Model_Validation_and_Error_Analysis.md) | reject leaky evaluation and calculate policy metrics | formulation, confusion matrix | draft; unreviewed |
| 3 | planned `06_25_Common_Issues.md` | diagnose collection-wide failures | all above | planned |

## 3. Quick Topic Lookup

| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| “Can this feature be used?” | [01 §8](01_Model_Validation_and_Error_Analysis.md#8-failure-modes--edge-cases) | prediction time is known | source date is not availability time |
| “Why did a score collapse?” | [01 §7](01_Model_Validation_and_Error_Analysis.md#7-comparison--selection-table) | split dependency identified | do not switch to an easier split |
| “Which threshold fits 48 reviews/day?” | [01 §5](01_Model_Validation_and_Error_Analysis.md#5-worked-numerical-example) | labels are mature | precision alone ignores missed positives |
| “Which loss matches the action?” | [02 §4](02_Problem_Formulation_and_Losses.md#4-core-formulas-with-full-derivation) | action costs are specified | calibrated probabilities and capacity are separate checks |

## 4. Reading Paths — Three Options

- **Beginner:** [formulation](02_Problem_Formulation_and_Losses.md) → planned features → [validation](01_Model_Validation_and_Error_Analysis.md). Objective: describe a time-safe prediction contract.
- **Practitioner:** formulation → classification metrics → linear/trees/boosting → features → imbalance/calibration → model selection → validation → Common Issues. Objective: choose and challenge a feasible model policy.
- **Advanced:** all files in ladder order, then case studies. Objective: defend split, preprocessing, threshold and error analysis under a changed constraint.

## 5. Prerequisite Skills

| Skill | Needed for | Available handoff |
|---|---|---|
| conditional probability and rates | calibration and precision/recall | module 05 foundations |
| matrix/vector notation | linear predictors | [Module 05 mathematical foundations](../../../01_foundations_computing_and_problem_solving/05_math_probability_statistics_and_linear_algebra/INDEX.md) |
| timestamps and joins | point-in-time features | [Module 06 data modelling and governance](../../../02_data_analytics_and_decision_science/06_data_modeling_databases_lakehouse_and_governance/INDEX.md) |
| business error costs and capacity | threshold policy | module 01 decision framing |

## 6. What This Collection Does NOT Cover

It does not certify fairness, causal effects, privacy compliance, online experimentation, or production deployment. It hands causal questions to experimentation material, privacy questions to Module 16, and graph-specific fraud signals to Module 15. A file marked planned is not a working lesson.

## 7. Master Symbol Glossary

| Symbol/term | Meaning | Units/type | Domain | First appearance |
|---|---|---|---|---|
| `t0` | committed decision time | timestamp | per decision | [02 §3](02_Problem_Formulation_and_Losses.md#decision-time-t0) |
| `Y` | mature outcome | binary | `{0,1}` | [02 §3](02_Problem_Formulation_and_Losses.md#target-y) |
| `s(x)` | model score | dimensionless | model-specific | [02 §3](02_Problem_Formulation_and_Losses.md#score-sx) |
| `p(x)` | calibrated probability estimate | probability | `[0,1]` | [02 §3](02_Problem_Formulation_and_Losses.md#probability-px) |
| `tau` | action threshold | score/probability | policy-specific | [02 §4](02_Problem_Formulation_and_Losses.md#pf-02--cost-derived-probability-threshold) |
| `TP/FP/FN/TN` | post-policy counts | records | non-negative integers | [01 §3](01_Model_Validation_and_Error_Analysis.md#3-foundational-definitions) |

## 8. Quick Reference Card Directory

| File | Card | Core rule | Critical assumption | Avoid when |
|---|---|---|---|---|
| [02](02_Problem_Formulation_and_Losses.md#12-quick-reference-card) | formulation card | specify action, `t0`, label horizon, loss, capacity and baseline before fitting | probability is calibrated only when evidenced | no authorized action or mature outcome exists |

## 9. Common Issues Locator

`06_25_Common_Issues.md` is planned. No issue count is claimed. Until it exists, use the six bounded diagnostics in [01 §8](01_Model_Validation_and_Error_Analysis.md#8-failure-modes--edge-cases), which are not a substitute for the required final collection artifact.

## 10. Sources, Versions and Maintenance

| File | Source/evidence | Status | Review trigger |
|---|---|---|---|
| [01](01_Model_Validation_and_Error_Analysis.md) | scikit-learn pitfalls guide; Gneiting & Raftery; Davis & Goadrich; illustrative arithmetic | draft; unreviewed | prediction contract, labels, or population changes |
| [02](02_Problem_Formulation_and_Losses.md) | MacKay; Gneiting & Raftery; scikit-learn pitfalls; illustrative arithmetic | draft; unreviewed | action, horizon, loss, capacity, label, or population changes |
| planned files | visible `directory.txt` inventory only | planned | author obtains primary method references |

## 11. Collection Acceptance Checklist

- [ ] [Problem formulation and losses](02_Problem_Formulation_and_Losses.md) now has a restored File Header and final card, but extra `## 11.1`/`## 11.2` headings and repeated generic prompts remain; template and substantive acceptance are incomplete.
- [x] The authored sheet links its source inventory and method references.
- [ ] Every original family and topic outcome has an individual substantive artifact.
- [ ] All individual artifacts meet the 1,000-substantive-line course gate.
- [ ] Final Common Issues has 25 complete issues and all 12 headings.
- [ ] Independent numerical, navigation and editorial review is recorded.
