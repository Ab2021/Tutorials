# Module 07 Cheat Sheets — Analytics, Metrics, BI and Product Sense

## 1. Overview
This collection teaches analysts to turn an operational question into an auditable measure and a reversible decision. Readiness means defining grain, numerator, denominator, cutoff, owner, guardrail, and verification—not merely producing a chart. All worked values are illustrative unless explicitly marked otherwise.

## 2. Topic ladder
| Level | File | Testable outcome | Prerequisites | Status |
|---|---|---|---|---|
| Foundation | [01 Metric Trees](01_Metric_Trees_and_North_Stars.md) | Define an outcome, drivers, and guardrail | SQL aggregation | Draft |
| Foundation | [02 Funnel, Cohort, Retention](02_Funnel_Cohort_and_Retention_Analysis.md) | Reconcile ordered conversion and retention denominators | 01 | Draft |
| Core | [03 Dashboard Design](03_Dashboard_Design_and_Operational_Alerts.md) | Select action-ready views and alerts | 01–02 | Draft |
| Core | [04 Segmentation](04_Segmentation_and_Decision_Use.md) | Select stable, decision-relevant segments | 01–03 | Draft |
| Advanced | [05 Decision Memos](05_Stakeholder_Narratives_and_Decision_Memos.md) | Defend an action with uncertainty and alternatives | 01–04 | Draft |
| Final | [06 25 Common Issues](06_25_Common_Issues.md) | Diagnose metric-contract failures | 01–05 | Draft |

## 3. Quick topic lookup
| Question | File / section | Use condition | Warning |
|---|---|---|---|
| What should the north-star exclude? | [01 §3](01_Metric_Trees_and_North_Stars.md#3-foundational-definitions) | Before setting an objective | A proxy is not the outcome. |
| Why did retention move? | [02 §8](02_Funnel_Cohort_and_Retention_Analysis.md#8-failure-modes--edge-cases) | Cohort result changes | Check cohort entry and cutoff. |
| Is an alert actionable? | [03 §7](03_Dashboard_Design_and_Operational_Alerts.md#7-comparison--selection-table) | Designing monitoring | Threshold is not diagnosis. |
| Can we act on this segment? | [04 §8](04_Segmentation_and_Decision_Use.md#8-failure-modes--edge-cases) | Slice differs | Small slices are unstable. |
| How should a decision be narrated? | [05 §11](05_Stakeholder_Narratives_and_Decision_Memos.md#11-interview--exam-questions) | Owner needs a memo | Preserve uncertainty. |

## 4. Reading paths — three options
- **Beginner:** 01 → 02 → 03; define one metric and reconcile its components.
- **Practitioner:** 01 → 02 → 03 → 04 → 05; attach a decision owner and guardrail.
- **Advanced:** all files through 06; run the issue diagnostics on a frozen metric snapshot.

## 5. Prerequisite skills
SQL grouping and joins, ratio arithmetic, UTC cutoffs, and the distinction between an entity and an event. Experiment attribution is handed off to Module 08.

## 6. What this collection does NOT cover
It does not prove causal effects, prescribe a BI vendor, establish company performance, or replace privacy/security review. Causal estimands belong in Module 08; source contracts belong in Module 06.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units/type | Domain | First use |
|---|---|---|---|---|
| `N` | qualifying count | entities/events | nonnegative integer | 01 §4 |
| `D` | eligible denominator | entities/events | positive integer for rates | 01 §4 |
| `M` | declared metric | dimensionless or stated unit | fixed population/window | 01 §4 |
| `r_h` | retention at horizon `h` | proportion | fixed cohort | 02 §4 |
| `EV` | expected value | value-units/decision | stated probability/cost model | 05 §4 |

## 8. Quick Reference Card directory
| Card | Core rule | Use condition | Avoid condition |
|---|---|---|---|
| [01 §12](01_Metric_Trees_and_North_Stars.md#12-quick-reference-card) | Declare denominator and guardrail | metric design | unknown eligibility |
| [02 §12](02_Funnel_Cohort_and_Retention_Analysis.md#12-quick-reference-card) | Fix cohort entry/cutoff | retention analysis | mixed cohorts |
| [03 §12](03_Dashboard_Design_and_Operational_Alerts.md#12-quick-reference-card) | Alert requires owner/action | operations view | no response policy |
| [06 §12](06_25_Common_Issues.md#12-quick-reference-card) | Reconcile components | anomaly response | snapshot is mutable |

## 9. Common Issues locator
[06_25_Common_Issues.md](06_25_Common_Issues.md) contains 25 complete HYPOTHETICAL issues, `ISSUE-001`–`ISSUE-025`, across all four required categories. It is the final collection artifact.

## 10. Empirical sources, versions and collection maintenance
All examples are DERIVED/ILLUSTRATIVE and arithmetic is marked VERIFIED within the files; no company benchmark or operational outcome is claimed. Recheck on metric-definition, source, owner, cutoff, or decision-policy change. Editorial status remains draft and independent review is open.

## 11. Collection acceptance checklist
- [x] Six indexed artifacts exist in prerequisite order and retain 12 sections.
- [x] Final troubleshooting artifact has 25 complete issue entries.
- [x] Cards, glossary, lookup, reading paths, sources, and exclusions are linked.
- [ ] Independent technical/source review completed; no reviewed/published status is claimed.
