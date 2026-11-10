# Dispatch, ETA and Delivery Operations — Cheat Sheets

**Collection state:** planned; no individual sheet is authored. Contract: [course standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [sheet master](../../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). Source boundaries are in the family [coverage ledger](../../COVERAGE_LEDGER.md).

## 1. Overview
This collection will teach real-time delivery decisions: define order state, select dispatch/batching objectives, evaluate ETA, protect riders/customers, and preserve routing correctness under closure changes. It teaches methods through illustrative data, not company deployments. Readiness requires a decision-time state contract, a checked route/ETA calculation, and a safe fallback.

## 2. Topic ladder
| Level | File / topic | Outcome | Prerequisites | Status |
|---|---|---|---|---|
| Foundation | 01 ETA evaluation | Define calibration, error, and decision guardrails | timestamps, quantiles | planned |
| Core | 02 Dispatch objective guide | Compare allocation/batching objectives and constraints | 01 | planned |
| Core | 03 Rider and customer guardrails | Select guardrails and escalation actions | 01–02 | planned |
| Advanced | 04 Snapshot routing and ETA separation | Keep legal shortest path and ETA evaluation separate | directed graphs | planned; IR019 |
| Advanced | 05 Closure-aware fallback | Expand shortcut validity and use base Dijkstra when exactness is required | 04 | planned; IR023 |
| Final | 06 Common Issues | Triage >=25 delivery failures | 01–05 | planned |

## 3. Quick topic lookup
| Question | File | Section | Use condition | Failure warning |
|---|---|---|---|---|
| Is ETA calibrated enough to promise? | 01 | 4–9 | prediction and realized arrival share a cutoff | mean error can hide late-tail harm |
| Which order/rider pair should dispatch? | 02 | 4–8 | capacity and objective are explicit | local travel minimization can worsen fairness |
| Can a route use a stale shortcut? | 05 | 4–8 | closure version is pinned | filtering a shortcut alone does not prove optimality |
| Is route quality the same as ETA quality? | 04 | 3–7 | route and forecast are separately scored | a legal route may have a poor ETA |

## 4. Reading paths — three options
- **Beginner:** 01 → 02 → 03; assumes basic arithmetic and time semantics.
- **Practitioner:** 01 → 02 → 03 → 04 → 05 → 06; assumes graph basics and operations metrics.
- **Advanced:** 05 → 04 → 01 → 02 → 03 → 06; assumes nonnegative directed shortest paths; forward references are not prerequisites.

## 5. Prerequisite skills
Event timestamps, time-zone handling, percentiles, constrained optimization vocabulary, directed weighted graphs, immutable version snapshots, Dijkstra's settled-distance invariant, and data-leakage controls.

## 6. Exclusions and handoffs
This collection does not establish road-law compliance, rider employment policy, company route-stack behavior, or time-dependent/turn-state routing guarantees. Inventory availability handoff goes to Inventory; incentive and price decisions go to Growth/Pricing.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units / type | Domain | First planned use |
|---|---|---|---|---|
| `y` | realized arrival time | minutes | completed eligible order | 01 |
| `yhat` | ETA prediction | minutes | decision-time prediction | 01 |
| `G=(V,E)` | directed road graph | graph | nonnegative weights for Dijkstra | 04 |
| `c(e)` | edge travel cost | time units | open edge | 04 |
| `S_v` | immutable closure snapshot | versioned set | one request | 05 |

## 8. Quick Reference Card directory
No authored cards or anchors exist. Planned cards cover ETA evaluation, dispatch objective, guardrail response, snapshot route contract, closure fallback, and issue triage.

## 9. Common Issues locator
Planned `06_25_Common_Issues.md`: at least 25 complete issues on state transitions, late events, stale supply, batching, map matching, closures, ETA calibration, weather shifts, safety/fraud signals, observability, and recovery. No issues are currently authored.

## 10. Sources, versions and maintenance
| Source ID | Source / reviewed | Supports | Limitation / trigger |
|---|---|---|---|
| REF-IR019 | planning brief, reviewed 2026-11-09 | required routing/ETA distinction | source body unavailable; not a result claim |
| REF-IR023 | [closure-aware shortest paths](../../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Routing/closure_aware_shortest_paths.md), reviewed 2026-11-09 | stale-shortcut boundary, Dijkstra fallback | illustrative requirements; recheck graph contract changes |

## 11. Collection acceptance checklist
- [ ] Every file has 12 ordered sections, metadata, checked examples, >=4 diagnostics, 10–15 answered questions, and card last.
- [ ] IR019 routes are checked against a base-graph oracle and ETA separately.
- [ ] IR023 covers nested shortcuts, closures, directed edges, and unreachable pairs.
- [ ] `06_25_Common_Issues.md` has >=25 complete non-duplicate issues.
- [ ] Links, prerequisite order, glossary and evidence labels are verified.
