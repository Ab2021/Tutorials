# Module 44 Cheat Sheets — Retail, E-Commerce, Media and Personalisation

## 1. Overview

This collection separates traffic, order-state, merchandise and contribution denominators before choosing a commerce intervention. The current manuscript is an illustrative analytical reference, not merchant financial reporting or evidence of a production deployment.

Follow the [course authoring standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [cheat-sheet master](../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). Editorial status, source availability and measured outcomes are independent.

## 2. Topic ladder

| Level | Reference | Testable outcome | Required predecessor | Editorial state |
|---|---|---|---|---|
| Foundation | [Commerce metric tree and order reconciliation](01_Commerce_Metric_Tree_and_Order_Reconciliation.md) | Reconcile eligible visits, paid/completed order cohorts, merchandise adjustments and contribution without changing grain. | Ratios, keyed joins and event timestamps. | Draft; 1,540 physical lines observed; independent substantive review open. |
| Core | Recommendation-system launch checklist | Specify candidate eligibility, exposure, ranking, experiment unit and rollback before release. | Reconciled traffic/order definitions; recommendation and experiment basics. | Original catalog commitment; not authored here. |
| Core | Pricing experiment guide | Separate incremental contribution from conversion changes under assignment, inventory and interference constraints. | Reconciled contribution bridge; causal experiment basics. | Original catalog commitment; not authored here. |
| Final | 25 Common Commerce Analytics Issues | Diagnose at least 25 distinct assumption, numerical, scaling and conceptual failures with discriminating checks. | All applicable completed references. | Required final collection artifact; not authored. |

## 3. Quick topic lookup

| Question | Location | Validity condition | Failure warning |
|---|---|---|---|
| Which order count belongs in a conversion rate? | [Definitions](01_Commerce_Metric_Tree_and_Order_Reconciliation.md#3-foundational-definitions) | Visit eligibility and paid-state time are fixed. | Paid orders, completed orders and customers are different units. |
| Why does a revenue-like number not match cash? | [Worked example](01_Commerce_Metric_Tree_and_Order_Reconciliation.md#5-worked-numerical-example) | Cohort, currency, cancellation/refund policy and snapshot agree. | Analytical merchandise amount is not recognized revenue or cash. |
| How can replay change a business metric? | [Failure diagnostics](01_Commerce_Metric_Tree_and_Order_Reconciliation.md#8-failure-modes--edge-cases) | Stable event identity and order-state transitions are retained. | Duplicate or out-of-order records can change counts and money. |
| What is the operational cost of reconciliation? | [Complexity and sizing](01_Commerce_Metric_Tree_and_Order_Reconciliation.md#9-complexity--cost-analysis) | Payload and processing boundaries are stated. | A derived sizing example is not a measured engine benchmark. |

## 4. Reading paths

- **Beginner:** definitions → worked example → final card. Reproduce the paid/completed distinction before using a rate.
- **Practitioner:** formulas → worked example → failure diagnostics → sizing. Reconstruct the order and money bridges using one fixed snapshot.
- **Advanced:** complete the practitioner path, then defend the stated invariants against duplicate events, delayed refunds and eligibility changes. Recommendation launch and pricing experiments remain planned extensions, not available prerequisites.

## 5. Prerequisite skills

| Skill | Required use |
|---|---|
| Ratios and units | Distinguish orders/visit, currency/order and total currency. |
| Keys and join multiplicity | Count distinct orders once rather than inflating measures through line-item joins. |
| Event and observation time | Define cohort entry separately from the completion/refund snapshot. |
| State transitions | Distinguish payment, cancellation, completion and refund without treating them as interchangeable flags. |
| Cost boundaries | State which variable costs are included before interpreting contribution. |

## 6. Exclusions and handoffs

The sheet does not establish accounting, tax, legal, consent or jurisdiction-specific rules. Finance and policy owners must approve operational definitions. Ranking architecture, causal price-response estimation and full media-rights methods are distinct teaching outcomes; their names in this module do not make them authored.

Use the [module index](../INDEX.md) for the wider original syllabus. Unavailable source bodies remain unavailable; local mathematical examples are not reconstructions of those bodies.

## 7. Master symbol glossary

These definitions follow [the manuscript's notation table](01_Commerce_Metric_Tree_and_Order_Reconciliation.md#31-core-notation). All counts, snapshots and currency components refer to the declared population.

| Symbol | Meaning | Units | Shape/type | Domain and boundary | First use |
|---|---|---|---|---|---|
| `V` | Eligible visits in window `W` | visits | scalar integer | Nonnegative; one count per eligible visit identifier. | Sheet 01 §3.1 |
| `N_paid` | Distinct orders reaching paid state in `W` | orders | scalar integer | Nonnegative; paid-at time determines cohort entry. | Sheet 01 §3.1 |
| `N_done` | Paid-cohort orders completed by the snapshot | orders | scalar integer | `0 <= N_done <= N_paid`; not all calendar-window shipments. | Sheet 01 §3.1 |
| `CVR_paid` | Paid-order conversion at visit grain | dimensionless | scalar ratio | `N_paid/V` only when `V>0`; not customer conversion. | Sheet 01 §3.1 |
| `G_item` | Included item amounts before cancellations and refunds | currency | scalar amount | One declared currency and tax treatment; not cash. | Sheet 01 §3.1 |
| `R_cancel` | Item amount removed through cancellation | currency | scalar amount | Nonnegative; same included orders as `G_item`. | Sheet 01 §3.1 |
| `R_refund` | Item refunds against included orders | currency | scalar amount | Nonnegative; refund observation window must be stated. | Sheet 01 §3.1 |
| `M_net` | Analytical net merchandise amount | currency | scalar amount | `G_item-R_cancel-R_refund`; not recognized revenue. | Sheet 01 §3.1 |
| `AOV_net` | Net merchandise per completed included order | currency/order | scalar ratio | Defined only when `N_done>0`; not payment-time basket size. | Sheet 01 §3.1 |
| `CM` | Contribution after the listed variable components | currency | scalar amount | Local cost bridge; not operating profit. | Sheet 01 §3.1 |
| `K_raw` | Raw recommended candidate count | SKUs | scalar integer | Nonnegative; before eligibility filtering, not impressions. | Sheet 01 §3.1 |
| `K_unavail` | Candidates unavailable at the decision snapshot | SKUs | scalar integer | `0 <= K_unavail <= K_raw`; not lost demand. | Sheet 01 §3.1 |
| `K_eligible` | Candidates remaining after eligibility filtering | SKUs | scalar integer | `K_raw-K_unavail` only if no other filter applies. | Sheet 01 §3.1 |
| `tau` | Decision or observation snapshot | timestamp | instant | UTC or fully qualified timezone; not a date alone. | Sheet 01 §3.1 |
| `epsilon` | Permitted reconciliation residual | currency | scalar tolerance | Nonnegative; rounding policy must justify it. | Sheet 01 §3.1 |

## 8. Quick Reference Card directory

| Card | Core rule | Use condition | Critical assumption | Avoid |
|---|---|---|---|---|
| [Commerce reconciliation](01_Commerce_Metric_Tree_and_Order_Reconciliation.md#12-quick-reference-card) | Keep opportunities, order states and money bridges distinct until keyed reconciliation. | Population, cohort and snapshot are explicit. | Events have stable identity and consistent units. | Calling analytical merchandise or contribution accounting revenue or profit. |

Recommendation and pricing cards remain unwritten; there are no fabricated card links.

## 9. Common Issues locator

The final 25-issue collection artifact is not authored. The current sheet's [Section 8 diagnostics](01_Commerce_Metric_Tree_and_Order_Reconciliation.md#8-failure-modes--edge-cases) are local teaching scenarios, not a claim of 25 completed collection issues or observed merchant incidents.

## 10. Sources, versions and maintenance

| Artifact or authority | Supported use | Evidence limit | Recheck trigger |
|---|---|---|---|
| [Course authoring standard](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) | Required structure, numerical reasoning, evidence and review gates. | Not external certification. | Standard changes. |
| [Cheat-sheet master](../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md) | Individual and collection format. | Does not validate manuscript content. | Template revision. |
| Sheet 01 | Local analytical definitions, illustrative arithmetic and diagnostic mechanisms. | Independent review and merchant-specific evidence remain open. | Cohort, event schema, currency, snapshot or policy changes. |
| Original collection catalog | Commerce metric tree, recommendation launch and pricing experiment commitments. | A planned title is not an authored method or accessible source body. | Scope or canonical teaching-home changes. |

## 11. Collection acceptance checklist

- [ ] Independently review all substantive derivations, boundaries and examples in Sheet 01.
- [x] Navigation smoke after this revision: 783 course Markdown files, 3,564 local links, zero missing paths or anchors; all 15 manuscript glossary links resolve. This is not substantive manuscript acceptance.
- [ ] Author the retained recommendation launch and pricing experiment references.
- [ ] Author the final artifact with at least 25 complete, distinct troubleshooting issues.
- [ ] Verify sources and applicable workflow/accounting definitions before operational claims.
- [ ] Complete independent editorial review; draft presence and physical length do not establish publication readiness.
