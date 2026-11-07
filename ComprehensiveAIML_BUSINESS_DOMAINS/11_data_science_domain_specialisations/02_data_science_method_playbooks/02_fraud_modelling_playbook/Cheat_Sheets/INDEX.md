# Fraud Modelling Playbook — Cheat Sheets

> **Status:** Three sheets are authored as drafts, not reviewed or published. The collection is incomplete; the planned final Common Issues artifact does not exist. The [course standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [master template](../../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md) govern publication.

## 1. Overview

This collection serves analysts and fraud decision owners who need to evaluate scores and turn evidence into operational decisions. Its first draft addresses mature outcomes, calibration, action value and constrained review capacity. It assumes basic binary classification, probability and expected value. Readiness means being able to calculate a decision and explain its evidence and limitations, not merely naming a metric. The broader fraud-modelling syllabus is mapped in the [source coverage plan](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-methods-02).

## 2. Topic ladder

| Level | File / topic | Outcome | Prerequisites | Key skill | Status |
|---|---|---|---|---|---|
| 3 — production | [01 Fraud evaluation, calibration and capacity](01_Fraud_Evaluation_Calibration_and_Capacity.md) | Compute and defend a capacity-constrained action policy from specified inputs. | Binary classification, conditional probability and expected value. | Confusion counts, calibration, expected value, top-K allocation. | draft; not reviewed |
| 3 — production | [02 Calibration and probability scoring](02_Calibration_and_Probability_Scoring.md) | Derive, calculate and diagnose probability quality separately from ranking and action value. | Binary outcomes, conditional probability, logarithms and temporal validation. | Proper scores, reliability, prior shift, selective labels. | draft; not reviewed |
| 3 — production | [03 Imbalanced fraud learning and thresholds](03_Imbalanced_Fraud_Learning_and_Thresholds.md) | Calculate rare-event metrics and choose a feasible policy while identifying sampling and label limitations. | Sheets 01–02; conditional probability and temporal validation. | Base rates, cost thresholds, review capacity and selective labels. | draft; not reviewed |
| Planned | Fraud feature library | Specify point-in-time features and leakage controls. | Data and model fundamentals. | Feature availability and entity grain. | planned; not written |
| Planned | Alert-queue capacity guide | Design an operational queue and staffing model. | Evaluation and capacity basics. | Queue and workload constraints. | planned; not written |
| Planned | Investigation feedback schema | Record actions, outcomes and observation mechanisms. | Fraud workflow basics. | Label lineage and audit sampling. | planned; not written |

The authored sheets address the inventory's `04_Imbalanced_Learning.md` and `06_Evaluation_Metrics_Fraud.md` topic names; the original listed contents were not available. They do not complete the whole source collection. Planned titles are not links to completed artifacts.

## 3. Quick topic lookup

| Question | Location | Use condition / warning |
|---|---|---|
| Why did precision change? | [Sections 3–5](01_Fraud_Evaluation_Calibration_and_Capacity.md#3-foundational-definitions) | Fix eligibility, maturity and denominators first. |
| What threshold should we use? | [Section 4](01_Fraud_Evaluation_Calibration_and_Capacity.md#4-core-formulas-with-full-derivation) | Expected benefits need identified action effects and costs. |
| How should constrained slots be assigned? | [Section 5](01_Fraud_Evaluation_Calibration_and_Capacity.md#5-worked-numerical-example) | Top-K benefit requires additive values and identical slot use. |
| What caused an apparent performance shift? | [Section 8](01_Fraud_Evaluation_Calibration_and_Capacity.md#8-failure-modes--edge-cases) | Diagnostic cases are hypothetical, not observed incidents. |
| How reliable are predicted probabilities under distribution shift? | [Calibration and probability scoring, Sections 4–5](02_Calibration_and_Probability_Scoring.md#4-core-formulas-with-full-derivation) | Prior correction requires stable class-conditional score distributions; mature labels alone do not remove selection bias. |
| How do imbalance, sampling and delayed labels affect a threshold? | [Imbalanced learning, Sections 4–5](03_Imbalanced_Fraud_Learning_and_Thresholds.md#4-core-formulas-with-full-derivation) | Prior correction and capacity optimization require their stated assumptions. |
| What should an interview answer establish? | [Section 11](01_Fraud_Evaluation_Calibration_and_Capacity.md#11-interview--exam-questions) | Include assumptions, boundary and uncertainty. |

## 4. Reading paths

- **Beginner:** Review [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), then read sections 1–3 and 5 of the draft; return to derivations once conditional probability is familiar.
- **Practitioner:** Read sections 1–8 in order, then the card; work through both capacity and calibration examples before using a real policy.
- **Advanced:** Read the entire draft, including cost, formal conditions and interview answers; assess selective labels and intervention identification before applying utility estimates.

These paths begin with the first existing sheet; continue to [calibration](02_Calibration_and_Probability_Scoring.md) and [imbalanced learning and thresholds](03_Imbalanced_Fraud_Learning_and_Thresholds.md) for probability and rare-event decision details. They do not imply planned prerequisites or additional sheets already exist.

## 5. Prerequisite skills

- Math: binary confusion matrices, conditional probability, expected value, units and basic optimization; [NIST probability reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35.htm).
- Domain: authorization, fraud outcomes, chargebacks, investigation and customer friction as workflow concepts.
- Systems: timestamped event cohorts, joining versioned labels and computing grouped counts; SQL fundamentals.
- Evaluation: independent holdouts and training-only preprocessing; [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

## 6. Exclusions and handoffs

Model architecture, feature-library design, detailed graph-fraud methods, regulatory advice and AML reporting are outside the authored sheet. Feature, queue and feedback-schema topics remain planned above. The parent [fraud modelling playbook](../INDEX.md) is the actual teaching home; its broader material is not implied to be complete.

## 7. Master symbol glossary

All symbols first appear in the [draft definitions](01_Fraud_Evaluation_Calibration_and_Capacity.md#3-foundational-definitions) or [derivations](01_Fraud_Evaluation_Calibration_and_Capacity.md#4-core-formulas-with-full-derivation); indices identify events.

| Symbol | Meaning | Units / type | Domain / caution |
|---|---|---|---|
| Y, L, R | Target outcome, observed label, audit inclusion indicator | binary event variables | Y and L need not coincide. |
| S, p | Score, calibrated counterfactual fraud probability | scalar; probability dimensionless | Score need not be a probability; p in [0,1]. |
| TP, FP, FN, TN | Confusion-matrix counts | events | Same defined population and observation horizon. |
| pi | Fraud prevalence | dimensionless | [0,1]; target and training priors may differ. |
| TPR, FPR, PPV | Sensitivity, false-positive rate, precision | dimensionless | Respective denominators must be positive. |
| K, N | Review capacity, eligible event count | slots or events per stated period | Nonnegative integers; slots need not be interchangeable. |
| Delta_i | Incremental review benefit | currency/event | Depends on an identified action effect. |
| L_i, r_i | Conditional loss and fraction prevented | currency/event, dimensionless | In E3 L is loss, distinct from observed label L in Section 3; r in [0,1]. |
| c_i, h_i | Review and other incremental harm costs | currency/event | Cost scope must be stated. |
| q_i | Audit inclusion probability | dimensionless | Strictly positive for design-based estimation of each target event. |
| x_i | Review decision | binary | 1 means select; 0 means do not select. |
| B, n_B, bar_p_B, bar_y_B | Reliability bin, bin count, mean probability and outcome rate | bin / events / dimensionless scalars | Bin is nonempty; outcomes and forecasts use the same target. |
| a, b, s | Logistic calibration intercept, slope and scalar score | scalar; dimensionless | Map and score version must match; positive slope preserves ordering. |
| BS, LL, ECE | Brier loss, log loss and expected calibration error | dimensionless, nats/event, dimensionless | ECE depends on chosen bins; proper scores mix probability qualities. |
| C_FP, C_FN, t | False-positive and false-negative costs and decision threshold | currency/event, currency/event, dimensionless | The threshold assumes the two-action cost model and no capacity constraint. |
| q_i, R_i | Audit inclusion probability and realized inclusion indicator | dimensionless probability and binary indicator | Positive known q_i is required for design-based estimation of each target event. |

## 8. Quick Reference Card directory

| Card | Rule | Use condition | Avoid when |
|---|---|---|---|
| [Fraud evaluation and capacity](01_Fraud_Evaluation_Calibration_and_Capacity.md#12-quick-reference-card) | Select up to K positive incremental benefits. | Equal slots, additive benefits and credible action effects. | Review duration differs or labels/effects are unsupported. |
| [Calibration and probability scoring](02_Calibration_and_Probability_Scoring.md#12-quick-reference-card) | Compare mature outcome frequencies with predicted probabilities; score with Brier and log loss. | Same outcome, eligible population and representative labels. | Selective observation or intervention changes the estimand. |
| [Imbalanced fraud learning and thresholds](03_Imbalanced_Fraud_Learning_and_Thresholds.md#12-quick-reference-card) | Combine base-rate-aware metrics with action value and capacity. | Representative mature outcomes and identified incremental benefits. | Selective labels, unsupported action effects or incompatible review slots. |

## 9. Common Issues locator

The collection-wide final Common Issues artifact is **planned, not written**; it has zero completed entries and no issue anchors. The draft's [40 hypothetical diagnostic cases](01_Fraud_Evaluation_Calibration_and_Capacity.md#8-failure-modes--edge-cases) are local teaching cases, not a completed collection-wide Common Issues file.

## 10. Sources, versions and maintenance

Each draft's source ledger records references and limitations. Numerical scenarios are illustrative and production effects are not measured. Files 01–03 are version 1.0, draft, and not reviewed; source/link review remains open. Revisit on changes to label maturity, audit sampling, action policy, reviewer capacity, calibration method or evaluation population. No review date or production result is asserted.

## 11. Collection acceptance checklist

- [x] Three actual draft sheets are linked; planned topics remain clearly marked.
- [x] The drafts contain the twelve prescribed sections, mathematical conditions, worked calculations, diagnostics and interview answers.
- [ ] Independent review of arithmetic, source links, content and navigation.
- [ ] Author remaining planned sheets and complete the prerequisite ladder.
- [ ] Author the final Common Issues artifact with at least 25 distinct complete issues.
- [ ] Complete publication review, evidence registry and all applicable collection gates.
