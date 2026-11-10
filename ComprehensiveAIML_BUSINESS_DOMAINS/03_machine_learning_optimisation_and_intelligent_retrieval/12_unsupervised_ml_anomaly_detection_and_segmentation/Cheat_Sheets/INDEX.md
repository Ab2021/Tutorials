# Module 12 â€” Unsupervised ML, Anomaly Detection and Segmentation — Cheat Sheets

## 1. Overview

This collection carries 5 reference card(s) and 1 troubleshooting artifact(s) for `Module 12 â€” Unsupervised ML, Anomaly Detection and Segmentation`. Cards teach a reusable method, calculation, or selection rule; the Common Issues file turns those rules into diagnostics. Status is measured, not asserted: a card is `authored` only when `Tools/manuscript_lint.py` exits 0 for it.

## 2. Topic ladder

| Level | Cards | Purpose |
|---|---|---|
| core | Clustering and Embeddings, Segmentation Evaluation | Build from prerequisites to production judgement |

## 3. Quick lookup

| If you need to… | Read |
|---|---|
| work on anomaly detection triage | [Anomaly Detection Triage](./01_Anomaly_Detection_Triage.md) |
| work on clustering and embeddings | [Clustering and Embeddings](./02_Clustering_and_Embeddings.md) |
| work on change-point detection | [Change-point detection](./03_Change_Point_Detection.md) |
| work on segmentation evaluation | [Segmentation Evaluation](./04_Segmentation_Evaluation.md) |
| work on human review loops | [Human review loops](./05_Human_Review_Loops.md) |
| diagnose a failure that keeps recurring | [Common issues: anomaly detection and segmentation](./06_25_Common_Issues.md) |

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
| 1 | [Anomaly Detection Triage](./01_Anomaly_Detection_Triage.md) | 2 | DERIVED | authored |
| 2 | [Clustering and Embeddings](./02_Clustering_and_Embeddings.md) | core | ILLUSTRATIVE | draft (lint open) |
| 3 | [Change-point detection](./03_Change_Point_Detection.md) | 2 | DERIVED | draft (lint open) |
| 4 | [Segmentation Evaluation](./04_Segmentation_Evaluation.md) | core | DERIVED | draft (lint open) |
| 5 | [Human review loops](./05_Human_Review_Loops.md) | 2 | DERIVED | draft (lint open) |
| 6 | [Common issues: anomaly detection and segmentation](./06_25_Common_Issues.md) | 2 | DERIVED | draft (lint open) |

## 9. Common Issues locator

- [Common issues: anomaly detection and segmentation](./06_25_Common_Issues.md) — the unit's final troubleshooting artifact; entries use stable `CI-` ids.

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
