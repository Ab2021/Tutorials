# Part 03 Cheat Sheets — Machine Learning, Optimisation and Intelligent Retrieval

## 1. Overview

This is the cross-module navigation layer for reusable prediction, discovery, optimisation, retrieval, graph, and privacy decision methods. Module collections contain the teaching artifacts; this index does not turn a planned title into a completed sheet. Readiness means a learner can locate the appropriate module method, state its input/output contract, calculate or trace its key mechanism, and identify the failure that requires a different method. [Course standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the [master sheet template](../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md) apply.

## 2. Topic Ladder

| Level | Module collection | Focus | Status |
|---|---|---|---|
| Foundation | [11 Supervised ML](../11_supervised_ml_and_feature_engineering/Cheat_Sheets/INDEX.md) | contracts, features, loss, calibration, validation | one draft; remainder planned |
| Foundation | [12 Unsupervised ML](../12_unsupervised_ml_anomaly_detection_and_segmentation/Cheat_Sheets/INDEX.md) | clustering, anomaly, change, review | one draft; remainder planned |
| Core | [13 Optimisation](../13_optimization_operations_research_and_simulation/Cheat_Sheets/INDEX.md) | formulation, solvers, scheduling, simulation | planned |
| Core | [14 Ranking/Search](../14_recommendation_search_ads_and_ranking/Cheat_Sheets/INDEX.md) | candidates, ranking, retrieval, experiments | planned |
| Advanced | [15 Graph ML](../15_graph_ml_knowledge_graphs_and_network_analytics/Cheat_Sheets/INDEX.md) | graph modelling, GNNs, knowledge graphs | one short draft; remainder planned |
| Advanced | [16 Privacy ML](../16_privacy_preserving_ml_and_federated_learning/Cheat_Sheets/INDEX.md) | threat models, DP, federation, secure aggregation | planned |

## 3. Quick Topic Lookup

| Question | Home | Exact use condition | Failure warning |
|---|---|---|---|
| “Is a feature available at decision time?” | [M11 validation](../11_supervised_ml_and_feature_engineering/Cheat_Sheets/01_Model_Validation_and_Error_Analysis.md) | prediction time and lineage known | source date is not availability |
| “Which anomaly method fits review capacity?” | [M12](../12_unsupervised_ml_anomaly_detection_and_segmentation/Cheat_Sheets/INDEX.md) | labels may be incomplete | anomaly score is not proof of fraud |
| “Can this plan meet constraints?” | [M13](../13_optimization_operations_research_and_simulation/Cheat_Sheets/INDEX.md) | variables/objective/constraints explicit | infeasibility needs a certificate/explanation |
| “Did ranking change user value?” | [M14](../14_recommendation_search_ads_and_ranking/Cheat_Sheets/INDEX.md) | exposure and randomization defined | offline relevance is not online causality |
| “Can graph evidence transfer?” | [M15](../15_graph_ml_knowledge_graphs_and_network_analytics/Cheat_Sheets/INDEX.md) | graph snapshot/time boundary explicit | future edges leak |
| “What privacy guarantee applies?” | [M16](../16_privacy_preserving_ml_and_federated_learning/Cheat_Sheets/INDEX.md) | threat and release mechanism specified | encryption alone is not DP |

## 4. Reading Paths — Three Options

- **Beginner:** M11 validation → M12 anomaly triage → M13 formulation. Goal: define a decision and evaluate evidence.
- **Practitioner:** Beginner path → M14 ranking → M15 graph evaluation → M16 threat modelling. Goal: select and challenge an operational approach.
- **Advanced:** all six module collections in numerical order, then their case collections. Goal: compose methods without confusing evidence, optimization, and privacy boundaries.

## 5. Prerequisite Skills

Probability and rates; linear algebra and basic algorithms; SQL-like point-in-time joins; event-time semantics; and decision costs/capacity. Module prerequisites are linked in each module INDEX. These are handoffs, not evidence that every prior lesson is complete.

## 6. What This Collection Does NOT Cover

This index does not provide legal advice, certify fairness or security, replace causal evaluation, or claim a system has been deployed. Hardware-specific training, agents, and domain-specialisation cases have separate course homes.

## 7. Master Symbol Glossary

| Symbol | Meaning | Units/type | Domain | Home |
|---|---|---|---|---|
| `Y` | outcome/label | binary or numeric | stated horizon | M11 |
| `tau` | action threshold | score units | policy-specific | M11 |
| `x` | decision features | vector | point-in-time available | M11 |
| `C` | capacity | actions per interval | non-negative | M11/M12/M15 |
| `G=(V,E)` | graph snapshot | nodes/edges | time-bounded | M15 |
| `epsilon` | DP privacy parameter | dimensionless | mechanism-specific | M16 planned |

## 8. Quick Reference Card Directory

| Card | File | Rule | Critical assumption | Avoid when |
|---|---|---|---|---|
| Validation card | [M11 §12](../11_supervised_ml_and_feature_engineering/Cheat_Sheets/01_Model_Validation_and_Error_Analysis.md#12-quick-reference-card) | evaluate the future decision procedure | timestamps and labels are known | final holdout influenced selection |
| Other cards | module indexes above | planned | no card exists yet | treating catalog titles as cards |

## 9. Common Issues Locator

No part-level Common Issues file exists. Each module's final troubleshooting artifact is planned and must contain at least 25 complete issues under the master template. Existing draft failure tables are not substitutes.

## 10. Empirical Sources, Versions and Collection Maintenance

The visible [directory inventory](../../../Templates/directory.txt) and [source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) establish scope only. The graph-source brief assigns four direct technical titles to M15; IR025 belongs M13; IR018/020/021/024/029/032 extend M14; IR029/036 extend M15. New factual claims require primary references recorded in the individual artifact. Recheck navigation when module status, source assignment, or a prerequisite changes.

## 11. Collection Acceptance Checklist

- [x] Six module collection homes and original 18 reference families are navigable.
- [x] Direct source/addition ownership is stated without inventing source access.
- [ ] Every referenced family has a substantive individual sheet.
- [ ] Part-level final troubleshooting file exists with 25 complete issues.
- [ ] Every individual sheet meets the full 12-section, 1,000-substantive-line gate.
- [ ] Independent navigation, numerical and editorial review is recorded.
