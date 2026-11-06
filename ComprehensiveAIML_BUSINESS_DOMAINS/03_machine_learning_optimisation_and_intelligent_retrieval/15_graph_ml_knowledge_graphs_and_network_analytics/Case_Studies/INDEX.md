# Module 15 Case Studies â€” Graph ML, Knowledge Graphs and Network Analytics

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Fraud-ring investigation | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | enterprise entity resolution graph | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR029 | Return useful graph candidates under explicit traversal budgets | Proposed home | incident | How should a recommendation API respond when graph expansion reaches a node, edge, memory or time limit? | `bounded-work` | `correctness-contract`, `incident-postmortem` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR036 | Evaluate incremental investigator value from graph fraud signals | Proposed home | predictive-ml | Which graph-enriched alerts add confirmed value at a fixed investigation capacity beyond claim-level scoring? | `delayed-selective-labels` | `human-in-the-loop`, `data-leakage` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR029 | [REF-IR029](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md) | Depth and output-size limits alone do not bound expansion; source product degrees and outage figures are illustrative claims |
| PLAN-2026-IR036 | [REF-IR036](../../../../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md) | Connectivity and shared providers are not proof of fraud; source reference notes are unavailable and labels are delayed/selected |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
