# Quick Commerce Growth, Pricing and Unit Economics — Cheat Sheets

**Collection state:** planned; no individual sheet is authored. Contract: [course standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [sheet master](../../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). See the family [coverage ledger](../../COVERAGE_LEDGER.md).

## 1. Overview
This collection will teach bounded decisions about contribution, fees, offers, membership, retention, price response, market supply, abuse, causal measurement, and governance. It does not claim a company price policy or impact. Readiness is the ability to state the unit of analysis, derive a contribution or estimand under assumptions, select a test, and reject a misleading proxy.

## 2. Topic ladder
| Level | File / topic | Outcome | Prerequisites | Status |
|---|---|---|---|---|
| Foundation | 01 Unit-economics model | Compute contribution with explicit revenue/cost inclusions | arithmetic, units | planned |
| Core | 02 Offer experiment checklist | Specify randomization, estimand, guardrails, and stopping rule | 01, experiment basics | planned |
| Core | 03 Market cohort scorecard | Compare cohorts without confusing mix shift for retention | 01–02 | planned |
| Final | 04 Common Issues | Diagnose >=25 pricing/growth failures | 01–03 | planned |

## 3. Quick topic lookup
| Question | File | Section | Use condition | Failure warning |
|---|---|---|---|---|
| Is an order contribution-positive? | 01 | 4–7 | revenue/cost time basis is explicit | gross merchandise value is not contribution |
| Did an offer change behavior? | 02 | 4–9 | eligibility and randomization are recorded | post-treatment conditioning biases effect |
| Is retention improving? | 03 | 3–8 | cohort entry and maturity are fixed | changing acquisition mix can mimic retention |

## 4. Reading paths — three options
- **Beginner:** 01 → 02 → 03.
- **Practitioner:** 01 → 02 → 03 → 04, assuming basic probability and accounting vocabulary.
- **Advanced:** 02 → 03 → 01 → 04, assuming causal-inference vocabulary; the path still uses the 01 contribution definition before an action recommendation.

## 5. Prerequisite skills
Percentages versus currency, contribution accounting, cohort construction, confidence intervals, randomization units, interference, selection bias, and decision-time feature availability.

## 6. Exclusions and handoffs
This collection does not determine legal price compliance, tax/accounting treatment, actual market power, or company results. Dispatch capacity effects hand off to Delivery; availability effects hand off to Inventory. Any policy or legal claim needs a dated authoritative source.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units / type | Domain | First planned use |
|---|---|---|---|---|
| `CM` | contribution margin per order | currency/order | comparable completed-order window | 01 |
| `R` | recognized order revenue | currency/order | defined accounting basis | 01 |
| `C` | incremental order cost | currency/order | same time/currency basis as `R` | 01 |
| `tau` | causal effect / estimand | outcome units | stated eligible population | 02 |
| `t0` | cohort entry time | timestamp | fixed cohort rule | 03 |

## 8. Quick Reference Card directory
No authored card anchors exist. Planned cards: 01 contribution inclusion rule; 02 experiment preflight; 03 cohort comparability; 04 issue triage.

## 9. Common Issues locator
Planned `04_25_Common_Issues.md` will have >=25 complete issues covering denominator drift, cost omission, interference, coupon leakage, abuse false positives, delayed retention, fee fairness, supply feedback, metric gaming, and governance escalation. No issue is authored yet.

## 10. Sources, versions and maintenance
No primary empirical pricing or company-performance source has been reviewed for this planned collection. Illustrative calculations must be labelled; recheck cost definitions, offer terms, and applicable policy before any decision-specific claim.

## 11. Collection acceptance checklist
- [ ] Each planned file exists with 12 ordered sections, metadata, calculations, sensitivity, >=4 diagnostics, 10–15 answered questions, and final card.
- [ ] Contribution inputs share currency, population, and timing basis.
- [ ] Experiment and cohort examples include competing explanations and guardrails.
- [ ] Final issue file contains >=25 complete distinct issues.
- [ ] Links, glossary, source labels, and prerequisites are locally verified.
