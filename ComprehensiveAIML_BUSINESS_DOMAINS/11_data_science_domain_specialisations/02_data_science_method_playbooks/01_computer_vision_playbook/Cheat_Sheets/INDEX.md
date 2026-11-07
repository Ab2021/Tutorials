# Computer Vision Playbook — Cheat Sheets

## 1. Overview
For vision practitioners comfortable with coordinates and basic rates, this collection supports task selection, metric interpretation and data-quality diagnosis. Object detection is the only authored sheet and is unreviewed; readiness requires independent review and task-specific evaluation, not just a card.

## 2. Topic ladder
| Level | Topic and outcome | Prerequisite / skill | Status |
|---|---|---|---|
| Foundation | CV task-to-model selector: choose classification, detection or segmentation for a stated output; explain a counterexample. | Images, labels and model outputs. | Planned |
| Core | Detection and segmentation metrics: select and calculate a task-appropriate metric; distinguish box and mask overlap. | Set overlap and rates. | Planned |
| Core | Vision data quality guide: identify label, acquisition and split problems in a sample pipeline. | Dataset and annotation basics. | Planned |
| Core/production | [04 Object detection](04_Object_Detection.md): compute box overlap and matching, interpret AP and diagnose deployment failures. | Coordinates, ratios and ranking. | Draft; not reviewed |

## 3. Quick topic lookup
| Question | Exact section | Use and warning |
|---|---|---|
| How are boxes compared? | [Formulas](04_Object_Detection.md#section-4--core-formulas-with-full-derivation) | Use the same coordinate frame and positive union; IoU alone does not determine matching. |
| Why are boxes shifted? | [Failure modes](04_Object_Detection.md#section-8--failure-modes--edge-cases) | Check resize and padding transforms before changing the model. |
| Which segmentation metric? | Topic ladder above | The separate metric guide is planned; object detection is not mask evaluation. |

## 4. Reading paths
- **Beginner:** Review image coordinates and ratios; read [definitions](04_Object_Detection.md#section-3--foundational-definitions) and [worked example](04_Object_Detection.md#section-5--worked-numerical-example). Task selection remains planned.
- **Practitioner:** Read the detection draft from definitions through [selection](04_Object_Detection.md#section-7--comparison--selection-table) and failures, checking matching and annotation policy.
- **Advanced:** Read the full detection draft, including operations and the card; use the data-quality and segmentation guides when authored.

## 5. Prerequisite skills
Interpret image coordinate frames, rectangle area, set intersection, precision/recall and ranked predictions; the draft's [definitions](04_Object_Detection.md#section-3--foundational-definitions) provide local reference. Understand annotation policy and data splits before comparing models.

## 6. What this collection does NOT cover
The task selector, detection/segmentation metric guide and vision data-quality guide are planned. The detection draft does not establish segmentation performance, clinical safety or real-world detector reliability; use a task-specific evaluation and governance process.

## 7. Master symbol glossary
| Symbol | Meaning; unit / type / domain | First use |
|---|---|---|
| W, H | Image width and height; positive scalar pixel counts. | [Definitions](04_Object_Detection.md#section-3--foundational-definitions) |
| IoU | Intersection over union; dimensionless scalar in [0,1] for positive union. | [Definitions](04_Object_Detection.md#section-3--foundational-definitions) |
| TP, FP, FN | Matched, unmatched predicted and unmatched reference counts under the declared matching policy; nonnegative integers. | [Definitions](04_Object_Detection.md#section-3--foundational-definitions) |

## 8. Quick Reference Card directory
| Card | Rule / use | Condition and avoid |
|---|---|---|
| [Detection card](04_Object_Detection.md#section-12--quick-reference-card) | Compute IoU and apply the stated matching and ranking policy. | Shared coordinate frame and annotation rules are essential; AP is not a deployment guarantee. |

## 9. Common Issues locator
The final Common Issues artifact is not authored; its count is unknown. Use [detection failures](04_Object_Detection.md#section-8--failure-modes--edge-cases): for shifted boxes inspect inverse scaling and padding; for axis swaps inspect coordinates on nonsquare images. These are hypothetical diagnostics, not incident counts.

## 10. Empirical sources, versions and collection maintenance
The original source directories were unavailable and not inspected. Detection examples are illustrative, not measured production performance; the draft and its source claims require independent review. Consult [authoring standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the [master template](../../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). Recheck metric implementation, annotation policy and dataset versions when changed; no review date is asserted. Each sheet requires 12 sections, checked examples, failure diagnostics, answered interview questions and a final card. Cards must specify decision, reader, inputs, assumptions, tradeoffs, failure signals and governance; review tool-, market- and regulation-sensitive cards quarterly.

## 11. Collection acceptance checklist
- [ ] Author the three planned guides and a final Common Issues artifact with at least 25 substantive issues.
- [ ] Independently review the detection draft, its 12 sections, calculations, sources and diagnostic claims.
- [ ] Verify links, anchors, glossary and reading paths as files are added.
- [ ] Record versions and evidence without presenting illustrative results as observed performance.
