# Dispatch, ETA and Delivery Operations — Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Owner | Curriculum maintainer |
| Index version | 3.0-planned |
| Case ID format | CASE-06-02-NNN |
| Evidence boundary | illustrative or not-measured unless primary evidence is cited |
| Review trigger | routing/ETA contract, source access, or template changes |

## 2. Case catalog
| Case ID | Case title / link | Domain | Type | Decision | Primary pattern | Secondary | Evidence | Stage | Last reviewed | Result / limitation | Editorial status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-06-02-01 | ETA calibration | delivery | predictive-ml | choose calibrated ETA policy | evaluation-mismatch | model-monitoring | not-measured | proposed | null | no result | planned |
| PLAN-06-02-02 | Rider supply forecast | delivery | predictive-ml | plan supply | distribution-shift | label-delay | not-measured | proposed | null | no result | planned |
| PLAN-06-02-03 | Batching policy experiment | delivery | analytical | select batching rule | causal-confounding | cost-latency-tradeoff | not-measured | proposed | null | no result | planned |
| PLAN-06-02-04 | Dispatch outage incident | delivery | incident | contain dispatch failure | incident-postmortem | rollout-regression | not-measured | proposed | null | no incident evidence | planned |
| PLAN-06-02-05 | Late-order recovery | delivery | analytical | select recovery action | human-in-the-loop | evaluation-mismatch | not-measured | proposed | null | no result | planned |
| PLAN-2026-IR019 | Routing correctness and ETA quality | delivery | hybrid | publish legal route and separately score ETA | snapshot-consistency | correctness-contract | not-measured | proposed | null | planning brief only | planned |
| PLAN-2026-IR023 | Closure-aware exact route fallback | delivery | analytical | fall back after closure | correctness-contract | snapshot-consistency | not-measured | proposed | null | source method, no benchmark | planned |

## 3. Canonical reusable patterns
| Pattern | Use when | Boundary |
|---|---|---|
| snapshot-consistency | closure feed and query must agree | a newer change applies to a later query unless a stronger contract exists |
| correctness-contract | route must be legal and exact | negative weights/time-dependent turn state require another contract |
| evaluation-mismatch | route quality and ETA quality differ | an accurate route does not establish ETA calibration |
| incident-postmortem | diagnosing an outage | replay does not establish production incidence |

## 4. Similarity / contrast map
ETA, supply, and batching are decision-quality cases; outage and recovery are operational cases. IR019 distinguishes routing and ETA objectives. IR023 establishes that rejecting stale shortcuts is not an exact-search proof; base-graph Dijkstra is the safe exact fallback under the stated directed nonnegative contract.

## 5. Questions and future-case backlog
Promotion requires a CASE ID/link, one owner/action/unit/time/baseline, Sections 1–15 with 5a–5e, successful and failed recovery paths, source ledger, checked calculation/sensitivity, and 12–18 fully answered interview prompts. Open evidence: no local source establishes a named delivery company's routing architecture or result.

## 6. Review / retirement log
| Date | Event | Decision |
|---|---|---|
| 2026-11-09 | Read closure-routing source and reconciled IR019/IR023 | retained planned; no source timings or company claims imported |

## 7. Pre-publish checks
- [ ] Every linked case has all 15 ordered sections and final card last.
- [ ] Route oracle covers nested shortcuts, closures, directed edges, and unreachable destinations.
- [ ] ETA score and route correctness use separate definitions and decision-time snapshots.
- [ ] Evidence, implementation stage, editorial status, and source limitations are distinct.
- [ ] Links and numerical/state-oracle checks are verified.
