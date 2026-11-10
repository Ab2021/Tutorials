# Quick Commerce Inventory and Assortment — Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Owner | Curriculum maintainer |
| Index version | 3.0-planned |
| Case ID format | CASE-06-01-NNN |
| Evidence boundary | illustrative or not-measured unless a primary source is cited |
| Review trigger | inventory decision contract, source access, or course template changes |

## 2. Case catalog
| Case ID | Case title / link | Domain | Type | Decision | Primary pattern | Secondary | Evidence | Stage | Last reviewed | Result / limitation | Editorial status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-06-01-01 | Substitution recommendation | inventory | predictive-ml | choose eligible substitute | evaluation-mismatch | human-in-the-loop | not-measured | proposed | null | no result | planned |
| PLAN-06-01-02 | Local assortment optimisation | inventory | analytical | choose local SKU set | cost-latency-tradeoff | evaluation-mismatch | not-measured | proposed | null | no result | planned |
| PLAN-06-01-03 | Perishable waste forecast | inventory | predictive-ml | order with expiry risk | distribution-shift | label-delay | not-measured | proposed | null | no result | planned |
| PLAN-06-01-04 | Availability-promise accuracy | inventory | analytical | publish availability promise | evaluation-mismatch | model-monitoring | not-measured | proposed | null | no result | planned |
| PLAN-06-01-05 | Stockout root-cause analysis | inventory | incident | diagnose stockout | incident-postmortem | data-leakage | not-measured | proposed | null | no result | planned |
| PLAN-2026-IR027 | Exact inventory range totals | inventory | analytical | select update/snapshot contract | snapshot-consistency | correctness-contract | not-measured | proposed | null | source-derived method only | planned |

## 3. Canonical reusable patterns
| Pattern | Use when | Boundary |
|---|---|---|
| evaluation-mismatch | proxy differs from promise/customer outcome | metric and action share population/window |
| snapshot-consistency | a query reads a multi-cell update | atomic cells alone do not isolate a snapshot |
| correctness-contract | exact totals are required | fixed order and one logical writer are assumptions |
| distribution-shift | season/store/assortment changes | distinguish from sampling noise |
| incident-postmortem | reconstructing a failure | a hypothetical replay is not an observed incident |

## 4. Similarity / contrast map
Substitution and promise cases optimize customer-facing decisions; assortment and waste cases optimize stock decisions; stockout analysis diagnoses a failure; IR027 supplies an exact total mechanism rather than distributed inventory truth. Every authored case must link to the relevant sheet and explicitly state whether values are illustrative.

## 5. Questions and future-case backlog
Before promotion, each plan needs a unique CASE ID, bounded owner/action/unit/time/baseline, 15 ordered sections including 5a–5e, input-output architecture where applicable, checked numerical walkthrough and sensitivity, honest evidence/stage/results, 12–18 answered Section-14 prompts, and final Section 15 card. Open question: which public primary source, if any, supports a real implementation claim?

## 6. Review / retirement log
| Date | Event | Decision |
|---|---|---|
| 2026-11-09 | Reconciled local catalog with REF-IR027 | retained as planned; no source result inferred |

## 7. Pre-publish checks
- [ ] Every CASE link resolves and contains the complete 15-section template.
- [ ] Owner, action, unit, baseline, deadline, evidence, stage, and editorial state are distinct.
- [ ] Illustrative calculations are labeled and independently checked.
- [ ] Case outcomes are not implied where no measurement exists.
- [ ] Local links, source boundaries, and final-card placement are verified.
