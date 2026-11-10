# Quick Commerce Inventory and Assortment — Cheat Sheets

**Collection state:** planned; no linked individual sheets are authored. Contract: [course standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [sheet master](../../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). See the family [coverage ledger](../../COVERAGE_LEDGER.md) for source uncertainty.

## 1. Overview
This collection will teach an operator or analyst to convert store-level stock, demand, expiry, and substitution signals into bounded availability decisions. It covers reusable methods, not proprietary systems or measured company performance. Readiness means the learner can state the decision-time data contract, calculate the stated illustrative quantity, reject invalid assumptions, and select a fallback.

## 2. Topic ladder
| Level | File / topic | Testable outcome | Prerequisites | Status |
|---|---|---|---|---|
| Foundation | 01 Availability KPI tree | Decompose promise failure into inventory, pick, substitute, and delivery components | ratios and event timestamps | planned |
| Core | 02 Substitution policy | Select a policy with customer consent, margin, and availability constraints | 01 | planned |
| Core | 03 Perishables forecast guide | Calculate lead-time demand and expiry-risk decision inputs | 01 | planned |
| Advanced | 04 Exact inventory range totals | Trace Fenwick point update/range query and select snapshot contract | integer arithmetic; 01 | planned; IR027 |
| Final | 05 Common Issues | Diagnose at least 25 distinct inventory/assortment failures | 01–04 | planned |

## 3. Quick topic lookup
| Question | File | Section | Use condition | Failure warning |
|---|---|---|---|---|
| Why did a promise fail? | 01 | 4–8 | completed order history exists | do not equate availability with delivery completion |
| Can this item be substituted? | 02 | 7 | consent and candidate data are known | ranking cannot override allergy or exclusion rules |
| How much perishable stock is defensible? | 03 | 4–7 | lead time and remaining shelf life are known | censored stockouts bias demand |
| How do I total bins after a correction? | 04 | 4–9 | fixed bin order, point delta | cell atomicity is not snapshot isolation |

## 4. Reading paths — three options
- **Beginner:** 01 → 02 → 03; learns KPI definitions, consented substitution, and expiry-aware forecasting.
- **Practitioner:** 01 → 02 → 03 → 04 → 05; assumes ratio arithmetic and versioned-event basics.
- **Advanced:** 04 → 01 → 03 → 02 → 05; assumes data-structure and concurrent-read foundations; use 04 only as an inventory-total component, not a distributed inventory system.

## 5. Prerequisite skills
Ratios, weighted averages, dimensional checks, inclusive intervals, integer overflow, event time versus processing time, and point-in-time data joins. The Fenwick item requires one-based indexing and bitwise `i & -i`; it will define both locally.

## 6. Exclusions and handoffs
Routing and ETA go to Dispatch; offer economics go to Growth/Pricing; warehouse layout and actual safety policy require operational owners. This collection will not claim Zepto, Swiggy Instamart, or Blinkit implementation details without primary evidence.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units / type | Valid range | First planned use |
|---|---|---|---|---|
| `A` | availability rate | dimensionless ratio | [0,1] | 01 |
| `D_L` | demand over lead time | units | nonnegative count | 03 |
| `i` | one-based bin index | integer | 1..n | 04 |
| `delta` | point inventory adjustment | signed integer units | policy bounded | 04 |
| `V` | snapshot version | monotonic integer | versioned state | 04 |

## 8. Quick Reference Card directory
No card anchors exist until files are authored. Planned cards: 01 KPI decomposition; 02 substitution eligibility; 03 expiry-aware order decision; 04 Fenwick update/query and snapshot precondition; 05 issue triage.

## 9. Common Issues locator
Planned final artifact: `05_25_Common_Issues.md`, with at least 25 complete issues covering stock accuracy, censored demand, expiry, substitutions, vendor fill, promise logic, overflow, replay, and snapshot visibility. No issue IDs or anchors exist yet.

## 10. Sources, versions and maintenance
| Source ID | Source / date | Supported scope | Limitation / trigger |
|---|---|---|---|
| REF-IR027 | [Fenwick inventory totals reference](../../../../../INTERVIEW_CASES/DSA/index_basics/Range_Queries/fenwick_inventory_totals.md), reviewed 2026-11-09 | lowbit, inclusive range, point delta, snapshot caveat | illustrative workload; recheck if data contract changes |
| OPEN-INV-01 | public company material not obtained | company workflow/outcome | do not make company-specific claims |

## 11. Collection acceptance checklist
- [ ] Every planned file exists with 12 ordered sections, metadata, checked examples, four distinct failures, 10–15 answered questions, and final card.
- [ ] IR027 has an array-oracle, boundary, overflow, replay, and visibility check.
- [ ] Final issue artifact has >=25 complete distinct issues.
- [ ] Links, prerequisites, glossary, evidence labels, and card anchors are verified.
