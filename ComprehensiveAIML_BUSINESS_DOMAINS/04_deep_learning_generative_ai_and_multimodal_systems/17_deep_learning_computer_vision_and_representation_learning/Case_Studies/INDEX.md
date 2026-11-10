# Module 17 Case Studies — Deep Learning, Computer Vision and Representation Learning

> **Collection scope:** bounded applied decisions for computer vision and representation learning. The collection is a navigation and review record, not evidence that any proposed intervention was deployed or measured.

## 1. Collection Metadata

| Field | Value |
|---|---|
| Collection owner | Module 17 curriculum maintainer |
| Canonical index | `Case_Studies/INDEX.md` |
| Case ID namespace | `CASE-04-17-NN` |
| Evidence policy | Evidence, implementation stage, and editorial status are separate fields. Illustrative values are not observed results. |
| Source basis | The original local catalog supplied two planned topics; inaccessible source bodies remain OPEN source verification. |
| Last collection review | 2026-11-10 — index structure and row 1 reconciled; independent publication review remains OPEN. |
| Entry rule | A linked case must exist, preserve the 15-section master structure, and state its evidence boundary. |

This collection follows the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master. An authored manuscript can still describe a proposed intervention and can still require technical or editorial review.

## 2. Case Catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary pattern 1 | Secondary pattern 2 | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CASE-04-17-01 | [Visual quality inspection: pass, review, or reject routing](01_visual_quality_inspection.md) | Manufacturing vision | predictive-ml | Whether to validate a human-gated visual triage policy that fits line and reviewer capacity | cost-sensitive triage with human review | distribution-shift monitoring | capacity-constrained routing | illustrative | proposed | not reviewed | Constructed workload/cost ledger and validation plan only; no production accuracy, vendor benchmark, deployment, or observed outcome. Editorial status: draft. |
| PLAN-04-17-02 | Medical-image triage evaluation | Medical imaging | predictive-ml | Design the decision, architecture, metrics, risks, and validation plan | evaluation-mismatch | model-monitoring | — | not-measured | proposed | not reviewed | Original planned row retained; no canonical manuscript is linked yet. Editorial status: planned. |

**Catalog interpretation:** `PLAN` denotes a retained planning commitment, not a completed case. `CASE-04-17-01` is the stable authored identifier. A case result/limitation field does not substitute for its evidence declaration or its editorial status.

## 3. Canonical Reusable Patterns

| Pattern | Use when | Required decision checks | Anti-pattern |
|---|---|---|---|
| Cost-sensitive triage with human review | A score routes finite human attention rather than issuing final authority | Define action costs, raw confusion counts, review capacity, human authority, and fallback | Calling every positive score a final reject |
| Distribution-shift monitoring | Sensing conditions, part mix, revision, or process can change after validation | Track capture health, score/calibration slices, labels, and recovery ownership | Treating a histogram shift as automatic retraining evidence |
| Capacity-constrained routing | A model creates a queue for reviewers or downstream correction | Use staffed-time units, arrival bursts, service rate, queue-age guardrails, and overflow route | Equating shift-average capacity with no waiting |
| Evaluation-mismatch | A proxy metric might not answer the operational question | Bind unit, label, split, metric denominator, action, and harm model | Reporting accuracy without prevalence or decision consequence |
| Model-monitoring | A delayed label and changing population govern safety of a prediction policy | Version models/policies, retain traceability, sample accepted items, and define stop conditions | Monitoring only server uptime |

A reusable pattern is a decision mechanism, not a vendor architecture. The manufacturing case uses the first three patterns only under its explicit illustrative assumptions; the medical-image row must develop its own authority, evidence, and evaluation boundary when authored.

## 4. Similarity / Contrast Map

| Dimension | Visual quality inspection | Medical-image triage evaluation (planned) | Design consequence |
|---|---|---|---|
| Unit of decision | One physical part inspection event | One imaging-study triage event, subject to future case definition | Do not transfer denominators or labels unexamined. |
| Human role | Reviewer retains final quality disposition in the proposed design | Clinical authority and workflow are not yet specified | No manufacturing autonomy rule implies a clinical rule. |
| Primary bottleneck | Conveyor timing, image acquisition, reviewer and rework queues | Planned evaluation topic; actual bottleneck remains OPEN | Capacity model must match the real service system. |
| Sensing shift | Light, camera, pose, part revision, lot/process change | Future imaging protocol/site/population issues require separate evidence | Shift controls are domain-specific. |
| Error consequence | Illustrative correction and escape-cost ledger | Not measured and not transferable | Never reuse the $ values or threshold from row 1. |
| Label maturity | Delayed quality adjudication and rework outcomes | Future case must define reference standard and timing | A routing code is not automatically a ground-truth label. |
| Status | Draft manuscript; proposed intervention; illustrative evidence | Planned catalog commitment | An existing manuscript does not complete the second row. |

## 5. Questions / Future-Case Backlog

1. What independently adjudicated defect taxonomy, severity policy, and visibility rules would support a real manufacturing validation?
2. What representative burst trace and reviewer handling-time study are needed before turning the row-1 mean capacity calculation into an operational queue policy?
3. How should a future case evaluate defect localization when some accepted quality outcomes have no bounded visible region?
4. Which image capture controls can demonstrate that a proposed change in illumination or camera hardware has not invalidated a calibrated operating point?
5. What is the bounded decision owner, input grain, baseline, evidence basis, and final authority for the retained medical-image triage evaluation case?
6. How will the future medical-image case distinguish prioritization, diagnosis, and final clinical disposition rather than merging them into one “accuracy” task?
7. What source bodies or primary evidence, if any, become accessible for either original planned topic, and what precise claims would they support?
8. Which course case, if any, should separately cover physical measurement or non-visual inspection rather than overextending the visual-quality case?

Backlog entries are not links and do not indicate that an artifact exists.

## 6. Review / Retirement Log

| Date | Item | Event | Evidence / status change | Decision | Open limit |
|---|---|---|---|---|---|
| 2026-11-10 | Collection index | Expanded original short planning catalog to the seven required collection sections | Editorial navigation update; no source result added | Retain both original topic commitments | Independent publication review OPEN |
| 2026-11-10 | CASE-04-17-01 | Created illustrative visual-quality-inspection manuscript | Evidence: illustrative; stage: proposed; editorial: draft | Link canonical manuscript | No model run, plant data, deployment, or measured outcome |
| 2026-11-10 | PLAN-04-17-02 | Reconciled original medical-image triage evaluation row | Evidence: not-measured; stage: proposed; editorial: planned | Preserve unlinked planned row | Full manuscript and source verification remain OPEN |

Retire a case link only when a replacement canonical teaching home is linked and the decision is recorded here. Do not retire a planned row merely because a neighboring case is complete.

## 7. Pre-Publish Checks

- [x] Seven ordered collection sections are present.
- [x] Catalog preserves the original visual-quality-inspection and medical-image-triage commitments.
- [x] Row 1 links only to an existing canonical manuscript.
- [x] Row 2 remains visible as an unlinked planned commitment.
- [x] Catalog uses 11 required case fields plus a separate result/limitation field retained from the local catalog intent.
- [x] Evidence, implementation stage, and editorial status are stated separately.
- [x] Row 1 limitation says illustrative calculations are not deployment, production accuracy, vendor benchmark, or observed result.
- [ ] Independent technical review of row 1 is OPEN.
- [ ] Publication review of row 1 is OPEN.
- [ ] Source-body verification for original planning material is OPEN.
- [ ] Row 2 manuscript, evaluation record, and publication checks remain incomplete.
