# Payments, AML and Financial Crime — Cheat Sheets

## 1. Overview

This collection carries 12 reference card(s) and 1 troubleshooting artifact(s) for `Payments, AML and Financial Crime`. Cards teach a reusable method, calculation, or selection rule; the Common Issues file turns those rules into diagnostics. Status is measured, not asserted: a card is `authored` only when `Tools/manuscript_lint.py` exits 0 for it.

## 2. Topic ladder

| Level | Cards | Purpose |
|---|---|---|
| troubleshooting | 25 Common Issues | Diagnose and prevent recurring failures |

## 3. Quick lookup

| If you need to… | Read |
|---|---|
| work on payment flow decisioning | [Payment flow decisioning](./01_Payment_Flow_Decisioning.md) |
| work on fraud detection and capacity-aware thresholds | [Fraud detection and capacity-aware thresholds](./02_Fraud_Detection_and_Thresholds.md) |
| work on aml transaction monitoring | [AML transaction monitoring](./03_AML_Transaction_Monitoring.md) |
| work on sanctions screening with name-match uncertainty | [Sanctions screening with name-match uncertainty](./04_Sanctions_Screening.md) |
| work on conservative entity resolution | [Conservative entity resolution](./05_Entity_Resolution.md) |
| work on typed fraud graphs and motif signals | [Typed fraud graphs and motif signals](./06_Typed_Fraud_Graphs.md) |
| work on streaming fraud controls and recovery | [Streaming fraud controls and recovery](./07_Streaming_Fraud_Controls.md) |
| work on case-management prioritization | [Case-management prioritization](./08_Case_Management.md) |
| work on investigator evidence ux | [Investigator evidence UX](./09_Investigator_UX.md) |
| work on financial-crime model-risk controls | [Financial-crime model-risk controls](./10_Model_Risk_Controls.md) |
| work on privacy-preserving investigation access | [Privacy-preserving investigation access](./11_Privacy_and_Access.md) |
| work on auditability and evidence reconstruction | [Auditability and evidence reconstruction](./12_Auditability_and_Evidence.md) |
| diagnose a failure that keeps recurring | [25 Common Issues](./13_25_Common_Issues.md) |

## 4. Reading paths

| Path | Sequence |
|---|---|
| Beginner | foundation cards first, then the Quick Reference Card (§12) of each core card |
| Practitioner | core cards in ladder order, then one case in `../Case_Studies/` that cites them |
| Advanced | advanced cards, then the Common Issues artifact and its discriminating checks |

## 5. Prerequisite skills

| Skill | Why it is needed | Where it is available |
|---|---|---|
| Unit topic map | Defines what this collection is responsible for and what it excludes | `../INDEX.md` |
| Decision framing | Cards assume a stated decision, population, and cut-off | sibling case collection |
| Arithmetic discipline | Every card carries checked numbers with units | the cards themselves |

## 6. Exclusions and handoffs

| Excluded here | Handed to |
|---|---|
| Decision-context narrative, baselines, and outcomes | `../Case_Studies/` |
| Organisational policy and legal determination | governance modules, not this collection |
| Vendor-specific or unverified current API behaviour | treat as OPEN until sourced at authoring time |

## 7. Master symbol glossary

| Symbol / term | Meaning | Units | Shape / type | Domain | First use |
|---|---|---|---|---|---|
| — | no card defines a symbol yet | — | — | — | — |

## 8. Card directory

| # | Card | Level | Evidence | Status |
|---|---|---|---|---|
| 1 | [Payment flow decisioning](./01_Payment_Flow_Decisioning.md) | 2 | DERIVED | draft (lint open) |
| 2 | [Fraud detection and capacity-aware thresholds](./02_Fraud_Detection_and_Thresholds.md) | 2 | DERIVED | draft (lint open) |
| 3 | [AML transaction monitoring](./03_AML_Transaction_Monitoring.md) | 2 | DERIVED | draft (lint open) |
| 4 | [Sanctions screening with name-match uncertainty](./04_Sanctions_Screening.md) | 2 | DERIVED | draft (lint open) |
| 5 | [Conservative entity resolution](./05_Entity_Resolution.md) | 2 | DERIVED | draft (lint open) |
| 6 | [Typed fraud graphs and motif signals](./06_Typed_Fraud_Graphs.md) | 2 | DERIVED | draft (lint open) |
| 7 | [Streaming fraud controls and recovery](./07_Streaming_Fraud_Controls.md) | 2 | DERIVED | draft (lint open) |
| 8 | [Case-management prioritization](./08_Case_Management.md) | 2 | DERIVED | draft (lint open) |
| 9 | [Investigator evidence UX](./09_Investigator_UX.md) | 2 | DERIVED | draft (lint open) |
| 10 | [Financial-crime model-risk controls](./10_Model_Risk_Controls.md) | 2 | DERIVED | draft (lint open) |
| 11 | [Privacy-preserving investigation access](./11_Privacy_and_Access.md) | 2 | DERIVED | draft (lint open) |
| 12 | [Auditability and evidence reconstruction](./12_Auditability_and_Evidence.md) | 2 | DERIVED | draft (lint open) |
| 13 | [25 Common Issues](./13_25_Common_Issues.md) | troubleshooting | DERIVED | draft (lint open) |

## 9. Common Issues locator

- [25 Common Issues](./13_25_Common_Issues.md) — the unit's final troubleshooting artifact; entries use stable `CI-` ids.

## 10. Sources, versions, maintenance

| Field | Value |
|---|---|
| Evidence vocabulary | DERIVED / VERIFIED / SOURCED / ILLUSTRATIVE / HEURISTIC / OPEN |
| Version policy | Content is dated by its own frontmatter; unstable API, price, and regulatory facts are rechecked at edit time |
| Maintenance trigger | A card's validity condition, a source body, or a tool's semantics changes |
| Review cadence | Quarterly, plus on any material change to a cited source |

## 11. Acceptance checklist

- [x] Every linked card exists on disk.
- [x] Status column is measured by the lint, not asserted.
- [x] Planned commitments remain listed without links.
- [x] A final Common Issues artifact exists with at least 25 distinct issues.
- [ ] Independent technical review of each card.
