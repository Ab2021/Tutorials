# Module 22 Case Studies — Multimodal Models, Video, Voice and World Models

## 1. Collection metadata

This collection holds bounded, decision-centred teaching cases for multimodal systems.
Each case must preserve the 15-section case structure, separate evidence from implementation
stage and editorial status, and label illustrative quantities as illustrative. The collection
contains no completed case manuscript yet; the two original case commitments below remain
planned. Source names are inventory evidence, not inspected source bodies.

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Source index | [module index](../INDEX.md) |
| Editorial state | Draft collection; not published |
| Evidence boundary | No organisation, deployment, device benchmark, or measured outcome is asserted |
| Review trigger | A new case manuscript, primary evidence, or changed modality contract |

## 2. Case catalog

| Case ID | Linked title / canonical home | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-22-01 | Multimodal maintenance copilot — `01_multimodal_maintenance_copilot.md` (planned) | Industrial maintenance assistance | hybrid | Whether an evidence-linked multimodal draft may be shown to a technician | `human-in-the-loop` | `evaluation-mismatch`, `authorization-scope` | not-measured | proposed | never | No manuscript, evaluation, deployment, or outcome exists yet |
| CASE-22-02 | Video safety-event review — `02_video_safety_event_review.md` (planned) | Video review workflow | hybrid | Whether an event candidate should be routed to a human reviewer | `human-in-the-loop` | `distribution-shift`, `evaluation-mismatch` | not-measured | proposed | never | No manuscript, evaluation, deployment, or outcome exists yet |
| PLAN-2026-IR011 | Document VLM versus OCR for adjuster assistance (planned shared brief) | Document intelligence | hybrid | Which fields can be extracted and which bundles require review? | `human-in-the-loop` | `evaluation-mismatch`, `delayed-selective-labels` | not-measured | proposed | 2026-11-06 | A planning brief, not a completed case |

## 3. Canonical reusable patterns

| Pattern | Meaning | Boundary |
|---|---|---|
| `human-in-the-loop` | The model prepares evidence and a recommendation; an authorised person makes the consequential decision. | Confidence does not grant decision authority. |
| `evaluation-mismatch` | Offline proxy quality can differ from the complete reviewer workflow outcome. | Measure evidence coverage, correction burden, and abstention as well as model scores. |
| `authorization-scope` | A modality input or retrieved document may only be exposed to an authorised workflow. | A valid prediction does not make an unauthorised input usable. |
| `distribution-shift` | Lighting, camera, language, equipment, template, or task changes can invalidate a learned association. | Aggregate performance does not establish every slice is safe to route. |
| `delayed-selective-labels` | Human-confirmed outcomes arrive late and only for routed items. | Do not treat routed-label accuracy as population accuracy. |

## 4. Similarity and contrast map

| Item | Similarity | Difference that changes validation |
|---|---|---|
| CASE-22-01 maintenance copilot | Multimodal evidence is assembled before a human action. | Maintenance requires asset, manual-version, and work-order provenance. |
| CASE-22-02 video safety-event review | A model surfaces candidates for human review. | Video requires time-window, camera, frame-rate, and event-boundary contracts. |
| PLAN-2026-IR011 document extraction | It combines visual regions and text with review routing. | A claim-field extraction is not a video event nor a maintenance instruction. |
| Module 18 claims document extraction | Both need page/region provenance. | The canonical home for the claims case is Module 18; this collection must not duplicate it. |

## 5. Questions and future-case backlog

1. For CASE-22-01, define the exact evidence a technician needs to reject an incorrect visual or textual suggestion.
2. For CASE-22-01, test whether stale manuals, missing asset identity, or image quality independently change the routing decision.
3. For CASE-22-02, specify event-time boundaries, retained frames, review queue capacity, and the cost of missed versus over-routed events.
4. For CASE-22-02, evaluate camera, lighting, activity, and rare-event slices before proposing a threshold.
5. For PLAN-2026-IR011, retain its planned shared relationship and do not duplicate it as a Module 22 case without a distinct decision contract.

## 6. Review and retirement log

| Item | State | Trigger / required action |
|---|---|---|
| CASE-22-01 | Planned | Author the canonical manuscript, independently check its worked example, then add a real link and review record. |
| CASE-22-02 | Planned | Author the canonical manuscript, independently check its queue/time example, then add a real link and review record. |
| PLAN-2026-IR011 | Planning only | Keep source and primary-evidence access limits explicit if it becomes a manuscript. |
| Collection | Draft | Reconcile headings, links, evidence labels, and catalog rows after every manuscript. |

## 7. Pre-publish checks

- [ ] Each authored file has all 15 ordered sections; Section 15 is its final content.
- [ ] Each case names a bounded owner, decision, baseline, data grain, time contract, and evidence basis.
- [ ] Each quantitative example and material boundary has been independently recomputed.
- [ ] Each interview pack contains 12–18 answered questions with reasoning, follow-up, alternatives, red flags, scoring, and answer support.
- [ ] No illustrative workload, model behaviour, or proposed architecture is presented as measured deployment evidence.
- [ ] Every catalog link targets a real canonical manuscript; shared cases are linked rather than duplicated.
