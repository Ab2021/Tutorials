# Insurance, Actuarial and Claims — Cross-Domain Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reconciled | 2026-11-09 |
| Review cadence | Quarterly and whenever source access, model-risk policy, or domain interfaces change |
| Index version | 2.1-planned |
| Canonical case ID | `CASE-10-00-NNN` |
| Scope | Cases that require a pricing/reserving, claims-operations, and model-risk interface |
| Evidence policy | Illustrative, published, and not-measured claims remain distinct from document state and implementation stage. |

## 2. Case catalog
| Case ID | Case title / relative link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-10-00-001 | Release a delayed-outcome claim model | Cross-domain insurance operations | hybrid | Whether a versioned claim model can be released with compatible policy, schema, and rollback evidence | rollout-regression | model-monitoring, human-in-the-loop | not-measured | proposed | null | Planned; no case file or measured result. |
| PLAN-10-00-002 | Reconcile pricing indication with claims emergence | Cross-domain actuarial control | analytical | Whether emerging claims are comparable enough to challenge a pricing assumption | evaluation-mismatch | label-delay, distribution-shift | not-measured | proposed | null | Planned; does not infer insurer experience. |

## 3. Canonical reusable patterns
| Pattern | Signature | Use when | Boundary |
|---|---|---|---|
| `evaluation-mismatch` | An offline score or rate is not the operational decision outcome. | A metric is offered as evidence of value. | The outcome, population, time window, and guardrails are already aligned. |
| `label-delay` | Mature truth arrives after the action. | Claims, fraud, reserve, and customer outcomes evolve. | A stable immediate adjudicated outcome is available. |
| `rollout-regression` | A compatible-looking release breaks a live guardrail. | Model, schema, threshold, policy, or dependency changes. | No live or replayable control path exists. |
| `model-monitoring` | Production signals can reveal degradation or control failure. | Feedback, data contracts, or review queues exist. | Signals lack a meaningful decision connection. |
| `human-in-the-loop` | Qualified review changes the error/capacity trade-off. | Actions are high-impact or evidence is ambiguous. | Review authority or capacity is absent. |

## 4. Similarity and contrast map
| Planned case | Supporting sheets | Contrast / decision boundary |
|---|---|---|
| PLAN-10-00-001 | [Model risk and change control](../Cheat_Sheets/01_Insurance_Model_Risk_and_Change_Control.md), planned MLOps sheets | A rollback record differs from proof that delayed model quality is acceptable. |
| PLAN-10-00-002 | [Non-life pricing](../01_pricing_underwriting_and_reserving/Cheat_Sheets/03_NonLife_Pricing_Ratemaking.md), [reserving](../01_pricing_underwriting_and_reserving/Cheat_Sheets/05_Claims_Reserving_Methods.md) | Emerging paid loss is not automatically an ultimate-loss or pricing-validity measure. |

## 5. Questions and future-case backlog
Before authoring, assign a CASE ID, choose one primary and at most two secondary patterns, identify owner/action/unit/time/baseline, and write a 15-section case with 5a–5e, a checked quantitative walkthrough, recovery reasoning, and 12–18 answered interview prompts. Open questions: what outcome maturity permits promotion; which policy/schema/model versions must recover atomically; and which pricing-to-claims reconciliation is decision-time valid.

## 6. Review and retirement log
| Date | Scope | State | Next trigger |
|---|---|---|---|
| 2026-11-09 | Reconciled empty cross-domain collection with child-domain authored and planned material. | Planned; no cross-domain case authored. | A bounded source-backed or clearly illustrative decision is drafted. |

## 7. Pre-publish checks
- [x] Catalog preserves explicit planned state and evidence limits.
- [x] Child-domain theory links are actual paths.
- [ ] Each catalog row has a complete 15-section case and Section 15 card.
- [ ] Each case has primary evidence or explicit illustrative basis, quantitative check, and independent review.
- [ ] Cross-domain links and case IDs are verified after case authoring.
