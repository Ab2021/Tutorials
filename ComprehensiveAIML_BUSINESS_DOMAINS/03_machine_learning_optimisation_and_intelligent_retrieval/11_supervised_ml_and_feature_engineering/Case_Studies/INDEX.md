# Module 11 Case Studies — Supervised Machine Learning and Feature Engineering

## 1. Collection metadata

Decision cases connecting supervised estimation, feature availability, calibration, business cost and operational constraints. Follow the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md). Evidence, implementation stage and editorial status are separate. Original source bodies and product-specific evidence remain unavailable; authored examples are explicitly illustrative.

## 2. Case catalog

| Case ID | Canonical title | Domain/type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation | Editorial status |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-M11-01 | [Credit-risk model design](01_credit_risk_model_design.md) | lending / predictive-ml | approve, review or decline under cost and capacity | point-in-time risk scoring | calibration; bounded review | illustrative inputs; derived arithmetic | proposed | null | 1,419 lines; synthetic costs 10,650 versus 13,750; real declined-applicant outcomes and policy value not identified | authored draft; publication/source review open |
| PLAN-M11-02 | Churn model with leakage remediation | retention / predictive-ml | remove unavailable signals before choosing an intervention | temporal validation | feature availability; treatment distinction | not measured | proposed | null | not authored | planned |

## 3. Canonical reusable patterns

| Pattern | Mechanism | Use condition | Failure warning |
|---|---|---|---|
| Point-in-time feature contract | Require evidence availability no later than the decision snapshot. | Historical replay and serving inputs. | A timestamped table does not establish what was known. |
| Probability-to-action policy | Compare calibrated risk with explicit cost, hard rules and approved actions. | Probabilities are valid for the decision population. | Ranking performance alone does not identify an economic threshold. |
| Capacity-bounded review | Admit work atomically against declared slots; use an owned overflow route. | Review can add useful information. | A wide uncertainty band can exceed actual staffing. |
| Selected-label boundary | Separate observable originated-loan outcomes from declined-applicant counterfactuals. | Evaluating lending policies. | A mature holdout does not resolve selection by itself. |

## 4. Similarity / contrast map

| Dimension | Credit risk | Churn leakage remediation |
|---|---|---|
| Decision unit | application at underwriting | customer at a defined intervention opportunity |
| Outcome clock | repayment horizon from defined origination | retention horizon from a frozen observation cutoff |
| Core leakage | later verification or repayment evidence | post-contact or post-churn fields |
| Additional identification limit | declined-loan outcomes are counterfactual | churn prediction does not identify treatment benefit |
| Shared requirement | frozen feature/label availability, baseline and independent temporal evaluation | same, with domain-specific actions and costs |

## 5. Questions / future-case backlog

Retain and author the original churn-with-leakage-remediation case as a complete bounded decision, not another credit-risk copy. Specify customer/action timing, intervention eligibility, unavailable-field counterexamples, an untouched temporal holdout, treatment-effect limits and a checked business-cost example. Product economics, jurisdiction, lawful data use and applicable fairness requirements need authorized evidence; the credit case does not establish them.

## 6. Review / retirement log

| Pass | Item | Event | Evidence/stage effect |
|---|---|---|---|
| current authoring pass | CASE-M11-01 | Canonical module manuscript authored; corrected the walkthrough's inconsistent cost summary and made synthetic declined-applicant labels explicit. | Remains illustrative/proposed. |
| current integration pass | CASE-M11-01 | Executed the 100-score fixture: 45 approvals, 40 reviews, 15 declines; band counts 40/55/24; verified delayed-feature, equality and threshold boundaries plus 3,100 synthetic cost difference. | Arithmetic/state verification only; no trained-model, live-policy or compliance claim. |

## 7. Pre-publish checks

- [x] Both original case commitments retained; one canonical credit manuscript, not a duplicated part-level case.
- [x] Credit case has 15 ordered sections, 15 answered interview questions and the final Section 15 card.
- [x] Numerical/state smoke executed against the written fixture; no repeated long prose detected.
- [ ] Author the original churn case and its complete interview pack.
- [ ] Obtain appropriate source, product, data, policy and applicability evidence.
- [ ] Complete independent substantive publication review; authored length and smoke checks do not establish operational suitability.