---
title: "Object Detection: Geometry, Matching, Evaluation and Operations"
topic: "Object detection evaluation policy, box geometry, matching, AP arithmetic, and production operations"
collection: "computer-vision-playbook"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Object Detection: Geometry, Matching, Evaluation and Operations

## 1. File Header

> **Purpose.** Replace "mAP went down, retrain it" with a defensible chain: define the instance, fix the coordinate frame, fix the matching policy, compute the metric by hand, then choose an operating point by cost.
> **Level.** Core. Assumes comfort with fractions, ratios, and basic probability; does not assume any particular detector architecture.
> **Outcome.** Given a table of reference boxes, predictions with scores, a label set, and an evaluation policy, you can produce per-class average precision, explain every threshold change in it, and defend a production confidence threshold with an explicit cost model.
> **Prerequisites.** Precision and recall definitions; the difference between a probability and a score; enough linear algebra to know that an image resize is an affine map.
> **Exclusions.** Segmentation metrics computed from masks (IoU per mask, panoptic quality), tracking metrics that require identity through time (MOTA, IDF1, HOTA), and training recipes (loss weighting, augmentation schedules, anchor design search). Those are separate sheets in this collection.
> **Version / review.** Draft 1.0, not independently reviewed. Every number is either derived from the formulas in Section 4 or labelled `ILLUSTRATIVE`.
> **Evidence.** DERIVED mathematics with hand-checked arithmetic. The four-image toy benchmark in Sections 5 and 9 is constructed, not measured; it exists so every intermediate value in an AP computation is visible.
> **Retrieval cue.** If mAP moved and nobody changed the model, go to Section 8.3 first (matching policy), then Section 9.4 (score-threshold and suppression interaction). If mAP is stable but the product is failing, go to Section 7 (threshold selection by cost).
> **Notation.** One convention throughout: display math appears in fenced `text` blocks, symbols are defined at first use, and all box coordinates are continuous `xyxy`.

### 1.1 The two questions this sheet separates

A detector answers one question: for this image, which objects are where?
Evaluation answers a different question: under a fixed matching policy, how good is the ranking of predictions?
Deployment answers a third: which single threshold converts that ranking into actions I can afford?

Confusing the three produces the classic false alarm. A team sees mAP fall from 0.84 to 0.76 after a data refresh and concludes the model regressed. In the worked example below, exactly that drop appears when the evaluation IoU threshold moves from 0.50 to 0.75 with the model held byte-identical. Nothing was retrained; the *question* changed.

A second false alarm goes the other way. A team keeps mAP constant by loosening the IoU threshold from 0.75 to 0.50 and reports the improvement. The detector did not improve; the acceptance criterion for "correct location" got cheaper. Section 8.3 gives the discriminating check: recompute both numbers from the same prediction dump.

### 1.2 What "the same instance" means

Everything downstream depends on the definition of one physical instance.
A crowd of overlapping people is not three instances if your annotation protocol says "count only fully visible torsos."
A rack of identical parts is not one instance if the downstream robot needs each part's pose.
Before comparing two detectors, write down: the instance definition, the minimum box side in pixels, the treatment of occlusion and truncation, and whether the model may emit multiple boxes for the same object.

The rest of the sheet assumes the annotation policy is fixed and known. Where the policy is unknown, metrics are not comparable, and no amount of arithmetic repairs that.

## 2. Concept Map / Mental Model

```text
  raw image (W x H, acquisition metadata)
        |
        |  resize / letterbox  (affine, invertible if recorded)
        v
  model input tensor (S x S), recorded scale s and pads (px, py)
        |
        |  forward pass
        v
  dense candidate boxes + class scores + objectness
        |
        |  decode (anchor/grid -> absolute coordinates)
        v
  boxes in MODEL coordinates
        |
        |  inverse affine: undo pad, undo scale
        v
  boxes in ORIGINAL image coordinates
        |
        |  score threshold -> class-wise NMS -> top-k
        v
  final detections  --------------------------->  consumer (alert, count, crop, track)
        |                                                  |
        |                                                  | feedback: cost of FP vs FN
        v                                                  v
  evaluation: reference boxes + policy              operating point selection
        |  (IoU threshold, maxDets, area bins, ignore rules)
        v
  TP / FP / FN per class  ->  PR curve  ->  AP  ->  mAP
```

Read the diagram as four separate contracts that must be written down independently.

1. **Geometry contract.** Which coordinate frame is a box in when it crosses a module boundary, and which transform inverts the resize. A box that silently stays in model coordinates is wrong by the pad and scale, and the error is invisible in aggregate metrics computed with the same bug on both sides.
2. **Selection contract.** Score threshold, per-class or class-agnostic suppression, IoU threshold for suppression, top-k cap, and tie-breaking on equal scores. These are inference-time decisions and they change the *set* of boxes the evaluator sees.
3. **Matching contract.** One-to-one assignment, class-aware by default, IoU threshold, order of assignment (by score, not by best overlap), handling of ignored regions, and whether unmatched predictions beyond `maxDets` count as false positives.
4. **Reporting contract.** Which classes are averaged, macro or micro, whether "background" is a class, which IoU thresholds are in the average, and whether the number is a single-threshold AP or the COCO-style sweep.

Three mental-model statements that prevent most mistakes:

- **The PR curve is a ranking artefact.** It is built by sorting predictions by score and walking down. Nothing in it depends on an absolute score value except the order. Therefore any monotone rescaling of scores leaves AP unchanged, and any non-monotone post-processing (clipping, score blending) can change it.
- **IoU threshold and score threshold are orthogonal.** The first lives in matching; the second lives in selection. Raising the score threshold removes the low-score tail of the curve; raising the IoU threshold deletes true positives from the middle of the curve.
- **NMS is not evaluation matching.** NMS decides which boxes survive to be scored. Matching decides which surviving boxes count as correct. Running NMS at IoU 0.7 and evaluating at IoU 0.5 is legal; assuming they are the same parameter is not.

## 3. Foundational Definitions

### 3.1 Image frame and box representation

An image has width `W` and height `H` in pixels.
Coordinates are continuous with `x` increasing rightward and `y` increasing downward, origin at the top-left corner.
A box is the 4-tuple `(x1, y1, x2, y2)` with `x1 < x2` and `y1 < y2`, interpreted as a half-open region.
Half-open means the box covers `x` from `x1` up to but not including `x2`, so its width is exactly `x2 - x1` in pixels and its area is `(x2-x1)(y2-y1)` in square pixels.
This convention is the one used by the COCO annotation format and by most modern training code.

Two other conventions exist and each needs different arithmetic.

| Convention | Width formula | Area formula | Where it bites |
|---|---|---|---|
| Continuous half-open `xyxy` | `x2 - x1` | `(x2-x1)(y2-y1)` | Default here; matches float coordinates |
| Inclusive integer pixels | `x2 - x1 + 1` | `(x2-x1+1)(y2-y1+1)` | Legacy annotation tools that name pixel indices |
| `xywh` (centre or corner) | `w` given directly | `w*h` | Model outputs and some JSON APIs |

Mixing the first two inflates every small box's area and systematically raises IoU for tiny objects.
A 4x4 inclusive box and a 4x4 half-open box differ by 125% in area (25 versus 16 square pixels).
The fix is mechanical: convert once at ingest, store a provenance flag, and never convert back.

### 3.2 Reference instances and predictions

A **reference instance** (ground truth) is a tuple of image identifier, category, and box, produced by an annotation protocol.
The protocol decides occlusion handling, truncation, minimum size, and which pixels belong to which category in ambiguous overlaps.
Ground truth is a measurement, not physical truth: two annotators following the same protocol disagree on a measurable fraction of boxes, and that disagreement bounds the meaningful resolution of any metric.

A **prediction** is a tuple of image identifier, category, box, and a scalar score.
The score is a ranking key. Whether it is calibrated (Section 6.4) is an additional, separate property that matters for thresholding but not for AP.

### 3.3 Intersection over Union

For boxes `A` and `B`, define the intersection rectangle from the coordinate-wise inner bounds:

```text
IW = max(0, min(A.x2, B.x2) - max(A.x1, B.x1))          [pixels, >= 0]
IH = max(0, min(A.y2, B.y2) - max(A.y1, B.y1))          [pixels, >= 0]
I  = IW * IH                                            [square pixels]
U  = area(A) + area(B) - I                              [square pixels]
IoU(A,B) = I / U                                        [dimensionless, in [0,1]]
```

`IoU` is undefined when `U = 0`, which happens only if both boxes are degenerate. Guard by rejecting zero-area boxes at ingest; do not rely on a division-by-zero policy deep in a metric library.

Four properties of IoU drive everything else in this sheet, and Sections 4.4 and 8.2 derive their consequences.

1. **Bounded.** `0 <= IoU <= 1`, with equality to 1 only for identical boxes.
2. **Scale invariant.** Multiplying all four coordinates of both boxes by a positive constant leaves IoU unchanged, because numerator and denominator both scale by the square of that constant.
3. **Translation sensitive in an area-dependent way.** The same absolute pixel shift costs far more IoU on a small box than on a large one. This is why a single IoU threshold is not a uniform accuracy requirement across object sizes.
4. **Not a metric in the mathematical sense.** `1 - IoU` violates the triangle inequality on rectangles, so nearest-neighbour reasoning, clustering, and any algorithm that assumes a metric space can produce counter-intuitive results.

### 3.4 Matching, true positives, and the three error types

**Matching** assigns each prediction to at most one reference instance (or to none), and each reference instance to at most one prediction (or to none), subject to a class-compatibility rule and an IoU rule.

Given a matching:

- **True positive (TP):** a prediction matched to a reference instance at or above the IoU threshold.
- **False positive (FP):** a prediction matched to nothing, or matched below threshold, or matched to an instance already claimed by a higher-scoring prediction.
- **False negative (FN):** a reference instance left unmatched.

Two consequences follow immediately and are the source of most reported confusion.

- **FP + FN need not equal anything.** They are separate counters over different populations; only TP + FN equals the reference count, and TP + FP equals the number of scored predictions.
- **A duplicate detection is a false positive, not a second true positive.** If two boxes overlap one person at IoU 0.88 and 0.92, the higher-scoring one is the TP and the other is an FP. This "duplicate penalty" is deliberate: it makes precision sensitive to over-firing.

### 3.5 Scores, objectness, and calibrated probability

Some detectors emit a single class score per box.
Others emit an objectness score multiplied by a conditional class score.
The product is the ranking key either way, but the two factors are not interchangeable for diagnostics: low objectness means "no object here," low conditional class score means "object, wrong class." Section 8.6 separates these.

A **calibrated** score satisfies `P(matched and correct | score = s) = s` over the evaluation population.
Detector scores are routinely not calibrated, and are typically overconfident at the high end after NMS because suppression removes near-duplicates that would otherwise contribute ranking diversity.
Calibration affects threshold choice, not AP: AP depends only on the ordering of scores within a class.

### 3.6 The evaluation policy object

Everything in Section 4 takes one explicit policy. Write it as data, not as defaults buried in a library call.

| Field | Example value | What it changes |
|---|---|---|
| `iou_threshold` | 0.50 or a sweep 0.50:0.05:0.95 | Which predictions count as correctly localized |
| `class_aware` | true | Whether a person box on a bicycle counts as an error |
| `max_dets` | 100 per image per class | Recall ceiling; see Section 8.4 |
| `area_bins` | all, small (<32^2), medium, large (>96^2) | Which instances are scored at all |
| `ignore_regions` | crowd boxes with `iscrowd=1` | Whether predictions inside them are FP or ignored |
| `score_floor` | 0.0 or 0.001 | The tail of the PR curve that contributes to AP |
| `sort_tiebreak` | stable by (score desc, box id asc) | Reproducibility when scores collide |

The `score_floor` field is under-appreciated. AP integrates over the whole ranking; truncating the dump at score 0.05 silently deletes false positives from the tail and inflates AP. Two teams comparing 0.84 and 0.87 are often comparing two floors, not two models.

### 3.7 Operating point

An **operating point** is the pair (score threshold, post-processing configuration) that converts a ranking into a decision stream.
It is chosen by cost, not by F1, unless the cost of an FP and an FN happen to be equal.
Section 5.2 and Section 7 make that choice explicit, and Section 5.3 shows a case where the F1-optimal threshold and the cost-optimal threshold differ by 0.2 in score, which is a factor of several in alert volume.
