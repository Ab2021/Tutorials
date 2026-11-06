# Fraud Modelling Playbook — Cheat Sheets

> **Status:** One sheet is authored as a draft, not reviewed or published. The collection is incomplete; the planned final Common Issues artifact does not exist. The [course standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [master template](../../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md) govern publication.

## 1. Overview

This collection serves analysts and fraud decision owners who need to evaluate scores and turn evidence into operational decisions. Its first draft addresses mature outcomes, calibration, action value and constrained review capacity. It assumes basic binary classification, probability and expected value. Readiness means being able to calculate a decision and explain its evidence and limitations, not merely naming a metric. The broader fraud-modelling syllabus is mapped in the [source coverage plan](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-methods-02).

## 2. Topic ladder

| Level | File / topic | Outcome | Prerequisites | Key skill | Status |
|---|---|---|---|---|---|
| 3 — production | [01 Fraud evaluation, calibration and capacity](01_Fraud_Evaluation_Calibration_and_Capacity.md) | Compute and defend a capacity-constrained action policy from specified inputs. | Binary classification, conditional probability and expected value. | Confusion counts, calibration, expected value, top-K allocation. | draft; not reviewed |
| Planned | Fraud feature library | Specify point-in-time features and leakage controls. | Data and model fundamentals. | Feature availability and entity grain. | planned; not written |
| Planned | Alert-queue capacity guide | Design an operational queue and staffing model. | Evaluation and capacity basics. | Queue and workload constraints. | planned; not written |
| Planned | Investigation feedback schema | Record actions, outcomes and observation mechanisms. | Fraud workflow basics. | Label lineage and audit sampling. | planned; not written |

The authored sheet covers the `06_Evaluation_Metrics_Fraud.md` source topic and portions of imbalance and capacity decision-making; it does not complete the whole source collection. Planned titles are not links to completed artifacts.

## 3. Quick topic lookup

| Question | Location | Use condition / warning |
|---|---|---|
| Why did precision change? | [Sections 3–5](01_Fraud_Evaluation_Calibration_and_Capacity.md#3-foundational-definitions) | Fix eligibility, maturity and denominators first. |
| What threshold should we use? | [Section 4](01_Fraud_Evaluation_Calibration_and_Capacity.md#4-core-formulas-with-full-derivation) | Expected benefits need identified action effects and costs. |
| How should constrained slots be assigned? | [Section 5](01_Fraud_Evaluation_Calibration_and_Capacity.md#5-worked-numerical-example) | Top-K benefit requires additive values and identical slot use. |
| What caused an apparent performance shift? | [Section 8](01_Fraud_Evaluation_Calibration_and_Capacity.md#8-failure-modes--edge-cases) | Diagnostic cases are hypothetical, not observed incidents. |
| What should an interview answer establish? | [Section 11](01_Fraud_Evaluation_Calibration_and_Capacity.md#11-interview--exam-questions) | Include assumptions, boundary and uncertainty. |

## 4. Reading paths

- **Beginner:** Review [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), then read sections 1–3 and 5 of the draft; return to derivations once conditional probability is familiar.
- **Practitioner:** Read sections 1–8 in order, then the card; work through both capacity and calibration examples before using a real policy.
- **Advanced:** Read the entire draft, including cost, formal conditions and interview answers; assess selective labels and intervention identification before applying utility estimates.

These paths use one existing sheet rather than implying that planned prerequisites or additional sheets already exist.

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

## 8. Quick Reference Card directory

| Card | Rule | Use condition | Avoid when |
|---|---|---|---|
| [Fraud evaluation and capacity](01_Fraud_Evaluation_Calibration_and_Capacity.md#12-quick-reference-card) | Select up to K positive incremental benefits. | Equal slots, additive benefits and credible action effects. | Review duration differs or labels/effects are unsupported. |

## 9. Common Issues locator

The collection-wide final Common Issues artifact is **planned, not written**; it has zero completed entries and no issue anchors. The draft's [40 hypothetical diagnostic cases](01_Fraud_Evaluation_Calibration_and_Capacity.md#8-failure-modes--edge-cases) are local teaching cases, not a completed collection-wide Common Issues file.

## 10. Sources, versions and maintenance

The draft's [source ledger](01_Fraud_Evaluation_Calibration_and_Capacity.md#10-key-theorems--results-reference) records references and limitations. Its numerical scenarios are illustrative and its production effects are not measured. File 01 is version 1.0, draft, and not reviewed; source/link review remains open. Revisit on changes to label maturity, audit sampling, action policy, reviewer capacity or evaluation population. No review date or production result is asserted.

## 11. Collection acceptance checklist

- [x] One actual draft sheet is linked; planned topics remain clearly marked.
- [x] The draft contains the twelve prescribed sections, mathematical conditions, worked calculations, diagnostics and interview answers.
- [ ] Independent review of arithmetic, source links, content and navigation.
- [ ] Author remaining planned sheets and complete the prerequisite ladder.
- [ ] Author the final Common Issues artifact with at least 25 distinct complete issues.
- [ ] Complete publication review, evidence registry and all applicable collection gates.
