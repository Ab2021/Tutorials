# Module 07 Case Studies — Analytics, Metrics, BI and Product Sense

## 1. Collection metadata
Illustrative decision cases for metric governance and product analysis. Evidence, implementation stage, and editorial status are independent; neither case claims a company deployment or measured outcome.

## 2. Case catalog
| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-2026-ANA001 | [Marketplace Conversion Diagnosis](01_Marketplace_Conversion_Diagnosis.md) | marketplace analytics | analytical | change conversion intervention | decision-ready-metric-contract | component-reconciliation, guardrail-governance | illustrative | proposed | null | checked arithmetic only; no company outcome |
| CASE-2026-ANA002 | [Executive Metric Tree Decision](02_Executive_Metric_Tree.md) | product analytics | analytical | adopt versioned metric tree | decision-ready-metric-contract | component-reconciliation, guardrail-governance | illustrative | proposed | null | checked arithmetic only; no deployment |

## 3. Canonical reusable patterns
- **decision-ready-metric-contract:** bind owner, action, entity grain, eligibility, cutoff, numerator, denominator, and guardrail.
- **component-reconciliation:** preserve identity-level components and independently recompute a decision-facing aggregate.
- **guardrail-governance:** block or defer action when the declared adverse condition fails.

## 4. Similarity and contrast map
Both cases use a frozen snapshot and reversible action. The conversion case emphasizes an intervention decision; the metric-tree case emphasizes executive operating review. Neither provides causal attribution; Module 08 is the handoff for that question.

## 5. Questions and future-case backlog
Open questions: how an experiment changes the recommendation, how metric review scales under many segments, and how source-contract changes are approved. These are evidence/validation gaps, not claims of completed work.

## 6. Review and retirement log
Created 2026-11-09 as draft illustrative teaching cases. Recheck if metric definitions, population, decision policy, or external evidence changes. Independent technical review remains open.

## 7. Pre-publish checks
- [x] Catalog preserves all required fields and links.
- [x] Cases use 15 ordered sections and final cards.
- [ ] Independent source/evidence review; no reviewed or published status is claimed.
