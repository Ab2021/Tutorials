# Practical ML Design and Interviews — Cross-Domain Cheat Sheets

## 1. Overview
This reserved collection is for methods whose decision contract genuinely spans ML system design, product sense, and evaluation/reliability. It contains no individual sheet. Domain-specific material remains in child collections; no empty cross-domain title is treated as coverage.

## 2. Topic ladder
| Level | Topic | Outcome | Prerequisites | Status |
|---|---|---|---|---|
| 1 | Cross-domain decision contract | align unit, owner, input, action, baseline, and timing | child-domain foundations | Planned |
| 2 | Metric-to-operation trace | map an offline metric to a user/operator action and guardrail | product metrics; evaluation design | Planned |
| 3 | Release evidence and recovery contract | select gates and reversible failure handling across domains | system and reliability basics | Planned |
| Final | Common Issues | diagnose boundary failures crossing child domains | all rows | Not authored |

## 3. Quick topic lookup
| Question | Destination | Condition | Warning |
|---|---|---|---|
| How do I reconstruct a feature at decision time? | [System-design feature store](../01_ml_system_design_production/Cheat_Sheets/03_Feature_Store_Design.md) | temporal history exists | corrected snapshot is not serving history |
| How do I decide if ML is justified? | [Product-sense collection](../02_ml_product_sense/Cheat_Sheets/INDEX.md) | decision and baseline can be specified | no authored sheet yet |
| How do I gate a release or replay an incident? | [Evaluation collection](../03_evaluation_reliability_and_incidents/Cheat_Sheets/INDEX.md) | manifest/oracle can be specified | no authored sheet yet |

## 4. Reading paths
- **Beginner:** child system-design feature-store sheet; then product and evaluation catalogs as planning maps.
- **Practitioner:** feature-store sheet → drift case → child product/evaluation topic ladders.
- **Advanced:** all child-domain artifacts plus future cross-domain decision/release contracts.

## 5. Prerequisite skills
Decision contracts, temporal data semantics, metric definitions, experiment vocabulary, failure isolation, and explicit evidence labels.

## 6. What this collection does NOT cover
It does not duplicate child-domain lessons, claim a complete cross-domain framework, or certify a production workflow.

## 7. Master symbol glossary
| Term | Meaning | Units/type | Domain | First source |
|---|---|---|---|---|
| decision contract | owner/input/action/timing/constraint agreement | structured record | bounded workflow | child collections |
| evaluand | system-version-population-window under assessment | manifest | comparison scope | evaluation collection |
| fallback | independently verified alternate action path | policy/state | failure handling | drift case |

## 8. Quick Reference Card directory
No cross-domain card is authored. See the [feature-store card](../01_ml_system_design_production/Cheat_Sheets/03_Feature_Store_Design.md#12-quick-reference-card).

## 9. Common Issues locator
No cross-domain Common Issues artifact exists; required count before completion is 25 distinct complete issues.

## 10. Empirical sources, versions and maintenance
This directory has no empirical source claims. Child indexes retain their source-plan provenance and uncertainty. Revisit when a cross-domain artifact is authored or any linked decision contract changes.

## 11. Collection acceptance checklist
- [x] Eleven-section navigation preserves child-domain handoffs.
- [ ] Cross-domain individual sheets exist.
- [ ] Final 25-issue artifact exists.
- [ ] Evidence and independent review are complete.
