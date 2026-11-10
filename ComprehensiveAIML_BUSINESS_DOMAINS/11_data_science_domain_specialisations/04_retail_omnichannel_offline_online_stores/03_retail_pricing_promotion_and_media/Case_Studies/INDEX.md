# Retail Pricing, Promotion and Retail Media — Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-04-03-NN |
| Evidence policy | Illustrative teaching inputs are not company results; sources unavailable in this subtree remain OPEN. |
| Editorial state | Draft; independently reviewed status has not been claimed. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog
| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-04-03-01 | [Markdown Optimisation](./01_Markdown_Optimisation.md) | Retail Pricing, Promotion and Retail Media | analytical | markdown schedule by SKU-location-week; owner: category manager | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-03-02 | [Promotion Uplift Measurement](./02_Promotion_Uplift_Measurement.md) | Retail Pricing, Promotion and Retail Media | analytical | continue, stop, or test promotion decision; owner: promotion manager | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-03-03 | [Retail Media Incrementality](./03_Retail_Media_Incrementality.md) | Retail Pricing, Promotion and Retail Media | analytical | campaign reporting decision; owner: retail-media measurement lead | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-03-04 | [Personalised Offer Controls](./04_Personalised_Offer_Controls.md) | Retail Pricing, Promotion and Retail Media | predictive-ml | eligible offer or no offer; owner: CRM manager | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |
| CASE-04-03-05 | [Price Change Incident](./05_Price_Change_Incident.md) | Retail Pricing, Promotion and Retail Media | incident | rollback or correct price publication; owner: pricing operations lead | evaluation-mismatch | human-in-the-loop; distribution-shift | illustrative | proposed | not reviewed | Not measured; illustrative decision design only. |

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
