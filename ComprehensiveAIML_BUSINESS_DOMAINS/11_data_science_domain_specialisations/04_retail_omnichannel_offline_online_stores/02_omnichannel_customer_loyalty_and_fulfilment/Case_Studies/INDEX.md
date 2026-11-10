# Omnichannel Customer, Loyalty and Fulfilment — Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-04-02-NN |
| Evidence policy | Illustrative teaching inputs are not company results; sources unavailable in this subtree remain OPEN. |
| Editorial state | Draft; independently reviewed status has not been claimed. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog
| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-04-02-01 | [BOPIS Promise Accuracy](./01_BOPIS_Promise_Accuracy.md) | Omnichannel Customer, Loyalty and Fulfilment | predictive-ml | pickup promise window; owner: fulfilment manager | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-02-02 | [Loyalty Next Best Action](./02_Loyalty_Next_Best_Action.md) | Omnichannel Customer, Loyalty and Fulfilment | predictive-ml | ranked contact or no-contact action; owner: loyalty manager | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-02-03 | [Returns Risk Intervention](./03_Returns_Risk_Intervention.md) | Omnichannel Customer, Loyalty and Fulfilment | predictive-ml | intervention eligibility at order time; owner: returns operations lead | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-02-04 | [Cross Channel Attribution](./04_Cross_Channel_Attribution.md) | Omnichannel Customer, Loyalty and Fulfilment | analytical | budget-learning estimate; owner: commercial measurement lead | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-02-05 | [Omnichannel Fulfilment Allocation](./05_Omnichannel_Fulfilment_Allocation.md) | Omnichannel Customer, Loyalty and Fulfilment | analytical | fulfilment node allocation; owner: order-routing manager | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-02-06 | [Authorized Policy Retrieval](./06_Authorized_Policy_Retrieval.md) | Omnichannel Customer, Loyalty and Fulfilment | llm-agentic | answer, abstain, or escalate decision; owner: customer-care policy owner | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |

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
