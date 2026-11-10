# Store Operations and Merchandising — Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-04-01-NN |
| Evidence policy | Illustrative teaching inputs are not company results; sources unavailable in this subtree remain OPEN. |
| Editorial state | Draft; independently reviewed status has not been claimed. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog
| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-04-01-01 | [Shelf Availability Prediction](./01_Shelf_Availability_Prediction.md) | Store Operations and Merchandising | predictive-ml | stockout-risk score per SKU-store-day; owner: store operations manager | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-01-02 | [Labour Scheduling](./02_Labour_Scheduling.md) | Store Operations and Merchandising | analytical | shift allocation by role and 30-minute interval; owner: store operations manager | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-01-03 | [Shrink Detection](./03_Shrink_Detection.md) | Store Operations and Merchandising | predictive-ml | review priority for transaction-event clusters; owner: loss-prevention lead | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-01-04 | [Planogram Compliance](./04_Planogram_Compliance.md) | Store Operations and Merchandising | analytical | exception queue for aisle-SKU layouts; owner: merchandising lead | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-01-05 | [Store Experiment Governance](./05_Store_Experiment_Governance.md) | Store Operations and Merchandising | analytical | holdout and rollout decision; owner: retail analytics lead | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |

## 3. Canonical reusable patterns
| Pattern | Use | Boundary |
|---|---|---|
| evaluation-mismatch | A score must support a constrained action. | Proxy quality does not establish decision value. |
| human-in-the-loop | Review capacity or override is material. | Review quality and capacity must be measured. |
| distribution-shift | Seasons, channels, or policy versions change. | A shift alert does not prove harm. |
| data-leakage | Outcome-adjacent state may arrive late. | Availability time, not event label, determines eligibility. |

## 4. Similarity / contrast map
All cases share point-in-time decision logging and a baseline. Store cases emphasize physical execution; omnichannel cases emphasize identity, promise, and routing; pricing cases emphasize incrementality, margin, and customer trust. The policy-retrieval case uses abstention rather than an automatic commercial action.

## 5. Questions / future-case backlog
Open questions: obtain original source bodies; test denominator reconciliation; define operator override quality; validate causal designs before claiming uplift. No additional case is silently substituted for an existing indexed decision.

## 6. Review / retirement log
| Date | Artifact | Status | Evidence |
|---|---|---|---|
| 2026-11-09 | Collection reconciliation | Draft authored | Local navigation and illustrative arithmetic pending independent review. |

## 7. Pre-publish checks
- [x] Every catalog row has a local case link and bounded decision.
- [x] Evidence, implementation stage, and editorial status are separate.
- [x] Cases retain 15 ordered sections and a final card.
- [ ] Independent source/evidence review remains open; no company result is claimed.
- [ ] Independent editorial review remains open.
