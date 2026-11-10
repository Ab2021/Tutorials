# Part 03 Case Studies — Machine Learning, Optimisation and Intelligent Retrieval

## 1. Collection Metadata

**Scope:** cross-module decision cases for predictive, prescriptive, retrieval, graph, and privacy methods. **Editorial status:** planned catalog; no part-level individual case exists. **Evidence rule:** source names establish scope, not an implementation or measured result. **Contract:** [course standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md). A module case remains owned by its module collection.

## 2. Case Catalog

| Case ID | Linked title if authored | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-M11-01 (original PLAN-M11-01) | [Credit-risk model design](../11_supervised_ml_and_feature_engineering/Case_Studies/01_credit_risk_model_design.md) | lending | predictive-ml | action threshold under cost/capacity | calibrated classification | validation | illustrative; derived arithmetic | proposed | null | module 11 canonical authored draft; publication/source review open; no observed policy benefit |
| PLAN-M11-02 | Churn leakage remediation | retention | predictive-ml | remove unavailable signals | temporal validation | features | not-measured | proposed | null | module 11 planned |
| PLAN-M12-01 | Payment anomaly detection | payments | predictive-ml | review queue | anomaly triage | calibration | not-measured | proposed | null | module 12 planned |
| PLAN-M12-02 | Campaign segmentation | marketing | analytical | segment/action choice | cluster validation | human review | not-measured | proposed | null | module 12 planned |
| PLAN-M13-01 | Last-mile routing | logistics | analytical | feasible route selection | constrained optimization | simulation | not-measured | proposed | null | module 13 planned |
| PLAN-M13-02 | Workforce scheduling | operations | analytical | staffing policy | robust scheduling | simulation | not-measured | proposed | null | module 13 planned |
| PLAN-M14-01 | E-commerce recommender | commerce | predictive-ml | eligible ranked candidates | retrieval-ranking | experimentation | not-measured | proposed | null | module 14 planned |
| PLAN-M14-02 | Search ranking regression | search | incident | diagnose quality change | retrieval-ranking | observability | not-measured | proposed | null | module 14 planned |
| PLAN-M15-01 | Fraud-ring investigation | fraud | predictive-ml | alert capacity allocation | graph evaluation | entity resolution | not-measured | proposed | null | module 15 planned |
| PLAN-M15-02 | Enterprise entity-resolution graph | enterprise data | analytical | merge/reject identities | graph construction | reversible decisions | not-measured | proposed | null | module 15 planned |
| PLAN-M16-01 | Cross-hospital risk model | healthcare | hybrid | privacy-safe collaboration | federated learning | validation | not-measured | proposed | null | module 16 planned |
| PLAN-M16-02 | Private analytics release review | analytics | analytical | release/no-release | privacy threat model | differential privacy | not-measured | proposed | null | module 16 planned |

## 3. Canonical Reusable Patterns

- Prediction cases need a decision-time feature contract, mature labels, baseline, and threshold/capacity consequence.
- Optimization cases need variables, objective, feasibility, solver/heuristic boundary, and an infeasibility or sensitivity path.
- Retrieval/ranking cases need eligibility, candidate/rank separation, exposure boundary, and online-validation plan.
- Graph cases need snapshot time, entity-resolution policy, traversal/feature budget, and non-graph comparator.
- Privacy cases need threat model, protected asset, mechanism, residual risk, and release decision.

## 4. Similarity / Contrast Map

| Contrast | Shared concern | Critical difference |
|---|---|---|
| M11 credit risk vs M12 anomaly | review capacity | labels versus weak/absent labels |
| M13 routing vs M14 recommender | constrained selection | route feasibility versus user exposure/utility |
| M14 retrieval vs M15 graph candidates | candidate generation | index contract versus graph snapshot/budget |
| M15 entity resolution vs M16 federation | sensitive linkage | identity merge policy versus distributed training privacy |

## 5. Questions / Future-Case Backlog

No part-level case is authored. A future case must combine at least two module methods without duplicating a module decision; declare source basis, owner, baseline, numerical oracle, outcome evidence, and 12–18 answered interview questions. Candidate themes: privacy-aware graph fraud triage; capacity-constrained recommender exploration; routing under forecast uncertainty.

## 6. Review / Retirement Log

| Date | Artifact | Action | Reason |
|---|---|---|---|
| 2026-11-09 | catalog | navigation expanded | preserves twelve original candidate threads without claiming cases |

## 7. Pre-Publish Checks

- [x] Original twelve candidate threads preserved and linked to numeric module homes.
- [x] Evidence, stage, and editorial state are distinct.
- [ ] Every `PLAN` has a 15-section `CASE` artifact with a final card.
- [ ] Every authored case has primary sources or a plainly illustrative basis.
- [ ] Each numerical walkthrough, link, and interview pack has independent review.
