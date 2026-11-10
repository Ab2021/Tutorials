# ML System Design Production — Cheat Sheets

## 1. Overview
This collection teaches decision-time-correct production ML design. The authored feature-store sheet covers temporal feature eligibility, revision ordering, parity, and recovery. Remaining source-plan and original-map commitments are retained as unauthored commitments rather than represented as completed material. Readiness is the ability to specify an input/output contract, choose the smallest viable architecture, calculate a bounded capacity estimate, and diagnose a violated invariant.

## 2. Topic ladder
| Level | File / commitment | Testable outcome | Prerequisites | Status |
|---|---|---|---|---|
| 1 | System framing, data contracts, feature platform | State prediction-time inputs, outputs, ownership, and invalid states | SQL joins; supervised-learning leakage | Planned |
| 2 | [03 Feature Store Design](03_Feature_Store_Design.md) | Reconstruct a point-in-time feature value and detect parity failure | timestamps; keyed data; leakage | Draft, unreviewed |
| 2 | Training, serving, experiments, evaluation | Compare candidate model paths with a frozen cohort and metric contract | framing; data contract | Planned |
| 3 | Monitoring, security, rollout, cost, incidents | Select gates, recovery, and bounded resource policy | prior rows | Planned |
| Final | Common Issues | Diagnose cross-cutting production failures | all rows | Not authored |

## 3. Quick topic lookup
| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| What could the model know at cutoff? | [Feature Store §4](03_Feature_Store_Design.md#4-core-formulas-with-full-derivation) | event and availability times are recorded | latest warehouse snapshot leaks corrections |
| Why do offline and online vectors differ? | [Feature Store §8](03_Feature_Store_Design.md#8-failure-modes--edge-cases) | matched request context exists | aggregate similarity is not parity |
| How should a shifted model be contained? | [Drift case](../Case_Studies/01_model_monitoring_drift_incident.md) | labels are delayed; contract may fail | drift alone is not outcome harm |
| How do cache, range-query, sketch, prefix, nearest-neighbor, graph, scheduling and online-window choices differ? | Source commitment `PLAN-2026-IR026` | operation mix and correctness contract are explicit | no authored local sheet yet |

## 4. Reading paths
- **Beginner:** data-contract commitment → Feature Store §§1–5 → Feature Store §12. Assumes basic joins and UTC timestamps.
- **Practitioner:** Feature Store → drift case → planned evaluation/monitoring commitments. Objective: separate rapid contract containment from mature-outcome evaluation.
- **Advanced:** all authored files, then planned serving, rollout, cost, incident and IR026 commitments. Optional extension: adversarial small-data operation traces.

## 5. Prerequisite skills
Typed entity keys; relational joins; UTC/event/availability times; train/validation/test isolation; deterministic transformations; basic queue and storage arithmetic; least-privilege access. The linked feature-store sheet defines the collection’s temporal notation.

## 6. What this collection does NOT cover
It does not certify a vendor feature store, a company architecture, model quality, regulatory compliance, or a universal launch threshold. Model-family selection and domain-specific business decisions are handed to their respective collections. Source bodies behind inventory names were not accessible and remain uncertain.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units / type | Valid range | First appearance |
|---|---|---|---|---|
| `e` | typed entity key | identifier | authorized entity namespace | [Feature Store §3](03_Feature_Store_Design.md#3-foundational-definitions) |
| `v_r` | feature logical/event time | UTC instant | no later than cutoff for eligibility | [Feature Store §4](03_Feature_Store_Design.md#4-core-formulas-with-full-derivation) |
| `a_r` | availability time | UTC instant | no later than cutoff for eligibility | [Feature Store §4](03_Feature_Store_Design.md#4-core-formulas-with-full-derivation) |
| `H` | maximum feature age | seconds | nonnegative policy bound | [Feature Store §4](03_Feature_Store_Design.md#4-core-formulas-with-full-derivation) |
| PSI | binned population-shift statistic | dimensionless | descriptive only | [Drift case §6](../Case_Studies/01_model_monitoring_drift_incident.md#6-analysis-walkthrough) |

## 8. Quick Reference Card directory
| File/card | Core rule | Use condition | Avoid condition |
|---|---|---|---|
| [Feature Store card](03_Feature_Store_Design.md#12-quick-reference-card) | rank only values available by cutoff | publication history and definition version exist | exact replay is claimed from corrected snapshots only |
| [Drift case card](../Case_Studies/01_model_monitoring_drift_incident.md#15-quick-reference-card) | contain a broken contract before attributing outcome harm | compatible fallback is verified | fallback shares the fault |

## 9. Common Issues locator
No collection-wide Common Issues artifact exists. The feature-store sheet includes local hypothetical diagnostic entries; they are not a completed 25-issue collection artifact. Required issue count: 25 distinct complete entries before this locator can claim completion.

## 10. Empirical sources, versions and maintenance
`SRC-MLD-01`: source-plan brief [ds-ml-design](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-ml-design), reviewed for planning on 2026-11-06; supports topic scope, not source-body claims. `SRC-IR026`: [reference-informed plan](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md), supports a proposed operation-mix brief only. Illustrative arithmetic in authored files is internally checked there; no production benchmark is claimed. Revisit when input timing, transformation versions, access policy, fallback dependencies, or source access changes.

## 11. Collection acceptance checklist
- [x] Canonical index has 11 sections and links only authored artifacts.
- [x] Authored feature-store sheet has 12 sections and final card.
- [x] Authored drift case is explicitly illustrative and unreviewed.
- [ ] Original topic-map commitments have individual authored sheets.
- [ ] Source-plan commitments, including IR026, have focused authored coverage.
- [ ] Final 25-issue Common Issues artifact exists.
- [ ] Independent technical and source review is complete.
