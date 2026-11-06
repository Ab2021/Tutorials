# Object Detection: Geometry, Matching, Evaluation and Operations

## Section 1 — File Header

> **ID:** CV-CS-04. **Level:** Core methods and production.
> **Purpose:** Compute detection metrics correctly and diagnose a deployed detector.
> **Outcome:** Given boxes, scores, labels, and an evaluation policy, determine matches, AP, and a defensible operating point.
> **Prerequisites:** Basic arithmetic, probability, image arrays, and precision/recall; [course standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md).
> **Exclusions:** Pixel segmentation metrics, multi-object tracking identity metrics, and model-training recipes are separate topics.
> **Version/review:** Draft 0.1; not independently reviewed or published.
> **Evidence:** Mathematical derivations and illustrative, constructed examples; no deployment or performance benchmark is claimed.
> **Retrieval cue:** Unexpected mAP changes: start with matching in Section 4 and diagnostics in Section 8.
> **Source inventory:** This sheet covers the visible directory entry `DS_Domain/Computer_Vision_Cheat_Sheets/04_Object_Detection.md`; the inventory is not evidence of accessible original lesson content.

Object detection predicts both an object category and its location.
The central decision is not merely whether an image contains a class.
It is whether each physical instance has a usable, correctly classified, sufficiently localized prediction.
Evaluation policy and operating policy answer related but different questions.
A ranking metric evaluates a range of confidence thresholds.
An operating point selects one threshold for a specific downstream action.

**Takeaway:** Define the instance, coordinate convention, and evaluation policy before comparing detectors.

## Section 2 — Concept Map / Mental Model

```text
Image + acquisition metadata
   | dataflow
   v
Resize / letterbox with recorded transform
   | dataflow
   v
Detector -> class scores + model-coordinate boxes
   | dataflow
   v
Inverse transform -> original-image coordinates -> suppression / selection
   | dataflow                                      | operational feedback
   v                                               v
Per-image outputs                        latency / input / output monitoring
   |                                                |
   v                                                v
Reference labels + evaluation policy -> matching -> TP / FP / FN -> PR / AP
   |                                                   |
   +------------------- review & drift investigation <-+
```

The arrows represent dataflow or an explicit feedback process, not proof of causation.
The output contract is a set of scored boxes per image and category.
The evaluation contract additionally specifies reference instances and matching behavior.
One-to-one matching prevents multiple predictions from receiving credit for one normal reference object.
The IoU threshold defines how much spatial agreement is required for a match.
Changing the IoU threshold changes the matching problem rather than merely relabeling a chart.
NMS is an inference-time selection procedure, not evaluation matching.
Tracking adds identity through time and cannot be evaluated solely with frame-level AP.
Segmentation predicts spatial masks, for which box overlap is only a proxy.
Unlabeled operational traffic can expose proxy drift but cannot directly establish recall.

**Takeaway:** Keep model output, post-processing, matching, and measurement as separate stages.

## Section 3 — Foundational Definitions

### 3.1 Image and coordinate frame
An image has width `W` and height `H` in pixels.
Use `(x,y)` with x increasing rightward and y increasing downward.
This sheet uses continuous, half-open `xyxy` boxes.
A box is `(x_min,y_min,x_max,y_max)` with nonnegative width and height.
Coordinates may lie on the image boundary, such as `x_max=W`.
A box's geometric area has units of square pixels.
For normalized coordinates, divide x values by W and y values by H.
Normalized values are dimensionless and must retain the original image dimensions for conversion back.
An inclusive integer-pixel convention instead counts endpoints and needs different arithmetic.
Never add one to continuous box widths merely because a different dataset uses inclusive pixel indexes.
A zero-area box is invalid for ordinary instance matching.
Clipping boxes to image bounds is a policy choice and must be applied consistently.
An image crop has its own origin and must be mapped into the reference frame before comparison.

### 3.2 Reference, prediction, and class
A reference or ground-truth instance is a labeled box and category for an image.
Ground truth is a measurement protocol, not infallible physical truth.
Annotation policy specifies occlusion, truncation, ambiguous classes, and unlabeled regions.
A prediction consists of image ID, class, box, and a scalar confidence score.
A confidence score need not be a calibrated probability.
Some detectors output objectness and conditional class scores separately.
Their combination must follow the actual model's documented inference contract.
A class-aware evaluator compares predictions only to eligible references of the same class.
A wrong-class prediction can therefore be an FP for its predicted class and leave an FN for the true class.
A class-agnostic localization diagnostic asks a different question and must not replace class-aware AP.

### 3.3 Overlap and identity
Intersection over union, IoU, is intersection area divided by union area for two valid boxes.
It ranges from zero to one when the union is positive.
IoU of one requires identical regions up to zero-area boundaries.
IoU of zero includes separated boxes and boxes touching only at an edge.
IoU measures relative spatial overlap, not class correctness or probability.
A match associates a prediction with an eligible reference according to a specified algorithm.
An IoU threshold is a matching rule, not the score threshold used to select predictions.
A duplicate detection is an additional prediction for an already matched object.
Depending on the evaluator, ignore or crowd references can have special matching and denominator rules.
Do not apply the normal one-to-one rule to those special cases without checking the implementation.

### 3.4 Counts and rankings
A true positive, TP, is a prediction accepted as a correct match.
A false positive, FP, is a nonignored prediction that does not receive an eligible match.
A false negative, FN, is an eligible reference left unmatched at the selected threshold.
Precision is the fraction of selected predictions that are true positives.
Recall is the fraction of eligible reference objects matched by selected predictions.
Precision and recall are undefined for zero predicted and zero reference denominators respectively unless an explicit reporting convention is imposed.
An empty class should not automatically receive an AP of zero or one without stating the evaluator policy.
Average precision, AP, summarizes a class's precision-recall ranking under a defined interpolation rule.
Mean AP, mAP, averages AP values across a declared set of classes and possibly IoU thresholds.
The aggregation weights and inclusion of classes without references matter.

### 3.5 Operational quantities
The score threshold is the minimum score for a selected deployment output.
A low threshold typically increases output volume; it does not guarantee improved downstream utility.
Non-maximum suppression, NMS, removes overlapping candidate boxes under a specified ranking and overlap rule.
NMS may run per class or class-agnostically.
An NMS overlap threshold is distinct from an evaluator's IoU match threshold.
A slice is a named subpopulation such as camera, lighting condition, or object size.
A latency percentile describes a distribution of individual end-to-end request durations.
A monitored proxy is a measurable operational signal that does not require verified labels.
A labeled audit estimates quality on its sampled population and label policy.

**Takeaway:** The score, IoU, class, reference eligibility, and coordinate system have separate semantics.

## Section 4 — Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

#### Rectangle intersection and union
Validity: both axis-aligned rectangles use the same continuous coordinate frame and have positive area.
The intersection extends from the greater lower bound to the lesser upper bound on each axis.
A negative overlap length means there is no overlap, so clamp each length to zero.
Union area follows inclusion-exclusion: add both areas and subtract the intersection once.
```text
FORMULA [D1]: wI=max(0,min(x2A,x2B)-max(x1A,x1B))
              hI=max(0,min(y2A,y2B)-max(y1A,y1B))
              I=wI*hI; U=area(A)+area(B)-I; IoU=I/U
WHERE: coordinates and lengths are pixels; I and U are square pixels; IoU is dimensionless.
CONSTRAINT: U>0, identical coordinate frame, continuous xyxy convention.
NUMERICAL EXAMPLE: A=(0,0,10,10); B=(5,0,15,10).
                   I=5*10=50; U=100+100-50=150; IoU=1/3.
INTERPRETATION: This pair fails an IoU>=0.5 match despite overlapping.
```
For identical positive-area rectangles, intersection and union coincide, giving IoU one.
For disjoint rectangles, intersection is zero and IoU is zero.
Symmetry follows because the min and max expressions are symmetric in A and B.
The denominator is not the area of the ground-truth box alone.
Intersection over reference area asks a different question.

#### Precision and recall
Validity: a fixed dataset, class, threshold, reference eligibility, and one-to-one matching policy.
Every selected nonignored prediction is either TP or FP.
Every eligible reference is either matched or unmatched.
Therefore selected predictions total `TP+FP`, and eligible references total `TP+FN`.
```text
FORMULA [D2]: precision=TP/(TP+FP); recall=TP/(TP+FN)
WHERE: TP, FP, FN are nonnegative integer instance counts.
CONSTRAINT: denominators positive; ignored predictions/references follow the named evaluator.
NUMERICAL EXAMPLE: TP=2, FP=1, FN=1 gives precision=2/3 and recall=2/3.
INTERPRETATION: Two thirds of selected outputs are correct and two thirds of references are found.
```
Precision measures output purity, not the fraction of all physical objects found.
Recall measures coverage of eligible labels, not the accuracy of every reported box.
No true-negative count is needed for ordinary detection precision or recall.
The number of possible background boxes is not a natural finite TN denominator.
An image with no eligible objects can still contribute false-positive predictions.

### 4.2 General Case

#### From per-image matches to a dataset ranking
Fix a class and an IoU threshold.
For each image, sort predictions by decreasing confidence with a deterministic tie rule.
For each prediction in that order, compare against eligible references in that same image.
For the simple evaluator used in Section 5, choose the highest-IoU currently unmatched reference meeting the threshold.
If none qualifies, mark FP; otherwise mark TP and reserve that reference.
Never match across image IDs or categories.
Then merge the per-image results and sort them globally by score using the same tie policy.
The reference count is the number of eligible objects over all evaluated images.
At each ranked prefix, cumulative TP and FP determine the precision and recall using D2.
The number of predictions can exceed the number of reference objects.
After each prediction, cumulative TP increases by at most one.
Each reference is reserved at most once in this simple policy.
The process terminates after the finite list of predictions is processed.
The trace in Section 5 demonstrates both a TP and a duplicate FP.
The exact reference-selection behavior can differ by benchmark implementation.
In particular, some implementations first find the best-overlap reference even if already matched and then label the detection a duplicate rather than falling back to a second unmatched reference.
Do not assume two libraries agree on ambiguous multi-object matches without a fixture.

#### AP with a right-to-left precision envelope
Validity: a chosen class, one IoU threshold, a finite ranked list, and at least one eligible reference.
Define `r_k` and `p_k` as recall and precision after rank k.
Define the precision envelope at recall r as the maximum precision among ranked points at recall at least r, or zero if no such point exists.
The envelope removes local decreases when measuring maximum achieved precision at or above each recall level.
For the continuous all-points convention, integrate this stepwise envelope over recall from zero to one.
Equivalently sum recall increments times the corresponding envelope precision at each distinct achieved recall.
The uncovered recall interval after the maximum achieved recall contributes zero.
```text
FORMULA [D3]: AP_all = sum_j (r_j-r_(j-1)) * p_interp(r_j)
WHERE: j indexes distinct positive recall levels; r_0=0;
       p_interp(r)=max({p_k : r_k>=r} union {0}).
CONSTRAINT: ranked nonignored predictions, fixed matching policy, positive reference count.
NUMERICAL EXAMPLE: recall levels 1/2 and 1; envelope precisions 1 and 2/3.
                   AP_all=(1/2)*1+(1/2)*(2/3)=5/6.
INTERPRETATION: Ranking the duplicate before the second TP reduces AP from one.
```
AP is dimensionless; multiplying by 100 reports percentage points.
AP is not generally equal to precision at the final operating threshold.
It is not necessarily the arithmetic average of precision over all predictions.
It also is not the area under an arbitrary straight-line plot through raw PR points.

### 4.3 Key Special Cases / Variants

| Variant | Definition and assumption | Use when | Avoid when |
|---|---|---|---|
| All-points interpolated AP | D3 over distinct recall jumps | This explicit teaching convention or matching evaluator | Reporting a benchmark whose sampling convention differs |
| VOC 2007 11-point AP | Mean interpolated precision at recall 0, 0.1, ..., 1 | Reproducing that protocol | Assuming it equals all-points or COCO AP |
| COCO-style AP | Mean sampled interpolated precision over 101 recall thresholds, then configured IoU thresholds and valid categories | Comparing outputs using the named COCO evaluator settings | Claiming generic `mAP` uniquely implies these settings |
| Fixed-threshold precision/recall | D2 after score threshold and matching | Capacity and safety decisions at deployment | Replacing ranking evaluation with a single favorable cutoff |
| Micro counts | Sum TP, FP, FN across declared classes then apply D2 | Overall count-weighted operational summary | Hiding poor rare-class performance |
| Macro AP | Arithmetic mean of valid per-class APs under stated policy | Equal-class reporting | Treating it as volume-weighted business impact |
| Class-agnostic localization | Ignore category for diagnostic matching | Separating localization from classification errors | Claiming it is standard class-aware detection AP |

VOC 2007's 11-point definition samples the precision envelope at eleven recall levels.
Some other VOC-era implementations use a continuous all-points interpolation instead.
The COCO evaluator normally uses ten IoU thresholds from 0.50 to 0.95 in increments of 0.05.
It samples 101 recall levels and has configurable categories, area ranges, and maximum detections.
Its summary AP conventionally uses the `all` area range and a maximum of 100 detections per image.
COCO also handles crowd and ignore annotations through implementation-specific rules.
Its tensors use sentinel values for settings without eligible references; aggregation excludes invalid entries.
Thresholds, area slices, category sets, and caps must accompany a published number.
The official evaluator is a source of implementation details, not evidence of this sheet having executed it.
See [COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) and [VOC challenge evaluation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/index.html).
The linked source can change; pin a commit and package version for a real experiment.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

At score threshold above all scores, no predictions are selected.
If eligible references exist, recall is zero.
Precision has zero denominator and must be reported according to an explicit convention.
If no eligible references exist for a class, recall and ordinary AP for that class are undefined.
An evaluator may exclude such a class rather than giving it zero.
For two identical valid boxes, IoU is one.
For boxes sharing only a boundary, continuous intersection area is zero.
A reference with zero area violates the positive-union premise if both boxes are degenerate.
At a stricter IoU threshold, a fixed pair may cease to qualify.
A full matching outcome can change in more complex ways when competing references are involved.
Do not assume a particular TP is retained under every alternate matching implementation.
For a fixed ranked list and a fixed matching policy, appending lower-scored FPs after the last TP does not improve the all-points AP in D3.
It can still reduce precision at the deployment threshold.
A perfect class ranking at IoU 0.50 need not give perfect AP at IoU 0.75.
A detector can produce high AP and still be unusable if it violates a hard latency deadline.

**Takeaway:** AP is an explicitly defined ranking functional layered on a matching policy, not a universal primitive.

## Section 5 — Worked Numerical Example

**Evidence:** ILLUSTRATIVE, hand-constructed data; arithmetic derived below, not an observed experiment.
**Decision:** Compare score ranking and choose whether the selected predictions meet a hypothetical review-capacity constraint.
**Population:** Two images, one class, two reference objects total.
**Coordinates:** Continuous original-image pixels, `xyxy`, positive boxes, no ignore regions.
**Policy:** IoU >= 0.5, best currently unmatched reference, descending scores, no score ties, no NMS.
Each image contains exactly one eligible reference and all predictions have the same class.

| Image | Reference | Box in pixels | Area |
|---|---|---|---:|
| A | G1 | (0,0,10,10) | 100 px² |
| B | G2 | (20,0,30,10) | 100 px² |

| Global rank | Prediction | Image | Score | Box in pixels | Area |
|---:|---|---|---:|---|---:|
| 1 | P1 | A | 0.90 | (0,0,10,10) | 100 px² |
| 2 | P2 | A | 0.80 | (0,0,10,10) | 100 px² |
| 3 | P3 | B | 0.70 | (20,0,28,10) | 80 px² |
| 4 | P4 | B | 0.60 | (40,0,50,10) | 100 px² |

P1 and G1 have intersection 100 px² and union 100 px².
Their IoU is 1.00, and P1 reserves G1.
P2 has IoU 1.00 with G1 but G1 is already reserved.
P2 is therefore an FP duplicate in the stated policy.
P3 and G2 have intersection 80 px² and union 100 px².
Their IoU is 0.80, so P3 reserves G2.
P4 has no overlap with G2 and is an FP.
The complete ranked trace follows.

| Rank | Score | Result | Cumulative TP | Cumulative FP | Precision | Recall |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 0.90 | TP | 1 | 0 | 1 | 1/2 |
| 2 | 0.80 | FP duplicate | 1 | 1 | 1/2 | 1/2 |
| 3 | 0.70 | TP | 2 | 1 | 2/3 | 1 |
| 4 | 0.60 | FP background | 2 | 2 | 1/2 | 1 |

At recall 1/2, the maximum precision at that recall or any later recall is 1.
At recall 1, the maximum precision at that recall is 2/3.
D3 gives `AP_all=(1/2)*1+(1/2)*(2/3)=5/6`, approximately 0.8333.
Independent area check: the first half of recall contributes 0.5.
The second half contributes 1/3, giving 5/6 in total.
The result lies between zero and one and is lower than the perfect-ranking value one.
The duplicate's precision drop is not itself an additional recall interval.
For VOC 2007 11-point sampling, six levels from 0 through 0.5 have precision one.
Five levels from 0.6 through 1.0 have precision 2/3.
Thus the 11-point result is `(6+5*(2/3))/11=28/33`, approximately 0.8485.
For 101 uniformly spaced recall levels, 51 levels from 0 through 0.50 have precision one.
The other 50 levels have precision 2/3.
The sampled value is `(51+50*(2/3))/101=253/303`, approximately 0.8350.
This single-threshold sampled value alone is not the full COCO summary AP.
COCO summary also averages the applicable IoU thresholds, classes, and configured evaluation settings.
At a score cutoff of 0.70 inclusive, selected outputs are P1, P2, and P3.
The counts are TP=2, FP=1, FN=0, giving precision 2/3 and recall 1.
At a cutoff of 0.90 inclusive, only P1 is selected.
The counts become TP=1, FP=0, FN=1, giving precision 1 and recall 1/2.
A reviewer who can process only one result in this tiny example would select the higher cutoff.
That capacity assumption is illustrative, not a recommendation for a real queue.
A wrong path is to count P2 as a second TP because its IoU is one.
That would double-credit G1 and falsely overstate coverage.
Another wrong path is to use only the last precision value, 1/2, as AP.
The last precision is a valid fixed-cutoff statistic, but it is not the ranking integral.
If P2 is moved after P3, the first two predictions are both TP and D3 becomes one.
That counterfactual changes the ranking, not the underlying references.
For a large dataset, store the same image, class, score, box, and ignore metadata.
Recompute matches separately for every IoU threshold and evaluator configuration.
Data labeling uncertainty, annotation gaps, and cluster dependence require additional analysis.
The example does not estimate generalization, reliability, or deployed business impact.

**Takeaway:** A one-to-one trace exposes duplicate errors that a box-overlap-only calculation misses.

## Section 6 — Visual / Geometric Intuition

```text
Continuous x coordinate, pixels, increases rightward ->
y=0     A: [0----------------10]
        B:          [5----------------15]
y=10    both boxes have vertical span [0,10]
        overlap x=[5,10], width=5 px; overlap height=10 px
```

Caption: The two boxes from D1 share a 5-by-10 rectangle; drawing is schematic.
The union is not the bounding rectangle around both boxes in general.
Here it happens to be a 15-by-10 rectangle because the boxes align vertically and overlap.
For diagonally offset boxes the enclosing rectangle can contain pixels outside the union.
IoU divides overlap by the actual union, not the enclosing rectangle.

```text
Precision (dimensionless)
1.00 |========               envelope through recall 0.5
0.67 |        ============== envelope through recall 1.0
0.00 +-----------------------> Recall (dimensionless)
       0       0.5           1.0
```

Caption: Section 5's interpolated precision envelope; the two widths are each 0.5.
Its step geometry is a conceptual area depiction, not a linearly interpolated raw PR curve.
The FP at rank two lowers raw precision without increasing recall.
Interpolation uses the best later precision for each recall requirement.
A better-ranked later true positive can raise the envelope above a preceding raw point.

**Takeaway:** IoU is a geometric union ratio; AP is area under a specified recall-indexed envelope.

## Section 7 — Comparison / Selection Table

| Approach | Preconditions | Quality and cost boundary | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|
| Human or simple rule baseline | Clear annotation and manageable workload | Requires measured agreement and labor cost | Establishing useful minimum | Rules cannot capture variable scenes | Baseline proposal; not measured |
| One-stage detector | Labeled instances and compatible hardware | Dense candidates can make post-processing costly | Latency-sensitive candidate evaluation | No target-device benchmark or small-object quality evidence | Architectural description; not benchmark |
| Two-stage detector | Proposal and region processing supported | Extra stages add implementation and compute tradeoffs | Evaluating proposal-based alternatives | Deadline or memory budget is violated | Architectural description; not benchmark |
| Set-prediction detector | Suitable trained model and decoding contract | May reduce reliance on hand-designed suppression, but performance is model-specific | Evaluating end-to-end set prediction | Assuming no duplicate or latency issues without tests | Architectural description; not benchmark |
| Tiled inference | Coordinate-preserving crop and merge policy | More passes and boundary duplicates | Small objects where full-image resolution loses detail | Compute budget or tile merge quality fails | Conditional design; not measured |
| Fixed camera region rule | Stable environment and validated geometry | Low overhead but brittle to viewpoint or layout changes | Controlled baseline | Uncontrolled scenes or missed out-of-region objects | Conditional design; not measured |

Selection order is constraint-first.
First reject a candidate that cannot satisfy privacy, safety, or data access requirements.
Next enforce deadline and device-memory limits under the expected workload.
Then evaluate labeled per-class and per-slice quality with the intended cost of mistakes.
Compare the simplest viable baseline with more complex detectors using identical splits and policies.
If small-object recall fails, test higher input resolution or tiles against latency and merge errors.
If crowded objects are suppressed, inspect NMS and evaluate alternatives on a crowd-labeled slice.
If labels are sparse or unreliable, improve the evaluation protocol before declaring a winner.
If constraints conflict, escalate the tradeoff and specify the missing measurement.
No architecture in this table is claimed to be fastest or most accurate in general.
A fair benchmark needs model weights, image size, preprocessing, runtime, device, batch, warmup, and concurrency.
It also needs class definitions, score thresholds, evaluation version, confidence intervals, and sample composition.
Record whether preprocessing and post-processing are included in timing.
Use the same test set only after model and threshold choices have been frozen.
Report measured results separately from illustrative sizing assumptions.

**Takeaway:** Select architectures against hard constraints and comparable measured workloads, not family names.

## Section 8 — Failure Modes & Edge Cases

All scenarios in this section are HYPOTHETICAL teaching diagnostics, not observed incidents.
Each diagnostic distinguishes at least two plausible mechanisms before prescribing a fix.
A negative discriminating test weakens that hypothesis; it does not prove the alternative.

### F-01 — Box offset after resize
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Predictions shifted by a constant margin after letterboxing.
- **Hypothesis:** Padding was not removed before inverse scaling.
- **Competing explanation:** An incorrect crop origin produces a similar offset.
- **Discriminating check:** Overlay corner landmarks before and after inversion.
- **Expected signal:** A shift matching padding supports the first hypothesis.
- **Bounded action:** Apply the exact recorded inverse transform.
- **Prevention and verification:** Preserve transform metadata per image and test asymmetric padding.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-02 — Axis swap
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** Boxes look transposed on nonsquare images.
- **Hypothesis:** An xy versus yx convention was mixed.
- **Competing explanation:** Width and height may instead have been swapped during resize.
- **Discriminating check:** Use a wide image with an off-center synthetic box.
- **Expected signal:** Transposition swaps corner coordinates rather than merely stretching them.
- **Bounded action:** Convert explicitly at the interface.
- **Prevention and verification:** Document order and test nonsquare frames.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-03 — Inclusive endpoints
- **Category:** NUMERICAL_EDGE; **provenance:** HYPOTHETICAL.
- **Symptom:** Small-box IoU differs between two evaluators.
- **Hypothesis:** One computes inclusive integer area and the other continuous area.
- **Competing explanation:** Rounding or clipping can also change overlap.
- **Discriminating check:** Compare a one-pixel endpoint fixture with arithmetic under both conventions.
- **Expected signal:** An exact one-pixel discrepancy identifies endpoint semantics.
- **Bounded action:** Convert annotation coordinates into the evaluator convention.
- **Prevention and verification:** Keep one explicit convention throughout the pipeline.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-04 — Degenerate boxes
- **Category:** NUMERICAL_EDGE; **provenance:** HYPOTHETICAL.
- **Symptom:** NaN overlap or unstable loss appears.
- **Hypothesis:** A box has zero or negative extent.
- **Competing explanation:** Nonfinite model output may also create NaN.
- **Discriminating check:** Log finite coordinates and pre-clamp extents separately.
- **Expected signal:** Zero width or negative height identifies geometry failure.
- **Bounded action:** Reject or repair invalid records according to contract.
- **Prevention and verification:** Count invalid boxes before and after each transformation.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-05 — Duplicate detections
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** Many overlapping outputs inflate FP counts.
- **Hypothesis:** Suppression is disabled or too permissive.
- **Competing explanation:** Two adjacent real objects may legitimately overlap.
- **Discriminating check:** Inspect duplicate predictions against independently annotated crowded examples.
- **Expected signal:** Several predictions on one object support the duplicate hypothesis.
- **Bounded action:** Tune suppression on a held-out set, preserving crowded recall.
- **Prevention and verification:** Monitor duplicate rate alongside labeled crowd recall.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-06 — Crowded-scene misses
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Recall falls when objects are close together.
- **Hypothesis:** Aggressive NMS suppresses valid nearby instances.
- **Competing explanation:** Annotation may mark only visible portions inconsistently.
- **Discriminating check:** Compare pre-NMS and post-NMS matched recall on crowd-labeled frames.
- **Expected signal:** A post-NMS-only loss supports suppression as a cause.
- **Bounded action:** Change suppression policy or model only after measuring duplicates.
- **Prevention and verification:** Include crowd-slice recall in release gates.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-07 — Wrong labels
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** Boxes overlap targets but class AP is low.
- **Hypothesis:** A class ID mapping or taxonomy changed.
- **Competing explanation:** The model may truly confuse visually similar classes.
- **Discriminating check:** Compare class-agnostic matches with a class confusion matrix.
- **Expected signal:** A systematic permutation indicates an ID mapping problem.
- **Bounded action:** Restore the versioned label map; retrain only if needed.
- **Prevention and verification:** Validate class IDs against known fixtures at deployment.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-08 — Score calibration drift
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** The production score histogram shifts.
- **Hypothesis:** Image conditions or model version changed score distribution.
- **Competing explanation:** A score transform or threshold change can cause the same symptom.
- **Discriminating check:** Stratify distributions by model version, camera, and preprocessing version.
- **Expected signal:** A change isolated to a deployment version supports pipeline change.
- **Bounded action:** Repair the changed interface or recalibrate with representative labeled data.
- **Prevention and verification:** Track thresholded quality and calibration on audited samples.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-09 — Recall without labels
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** A dashboard claims recall improved using only output counts.
- **Hypothesis:** Output count is being mistaken for matched reference coverage.
- **Competing explanation:** The underlying number of objects might have increased.
- **Discriminating check:** Obtain a probability sample with complete independent annotations.
- **Expected signal:** A count change without reference counts cannot identify recall.
- **Bounded action:** Compute recall against audited eligible objects.
- **Prevention and verification:** Label the unlabeled count as a proxy, not a quality metric.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-10 — Small-object failure
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Small objects have low AP while large objects do not.
- **Hypothesis:** Downsampling destroys discriminative pixels.
- **Competing explanation:** Small-object annotations can also be less consistent.
- **Discriminating check:** Compare native-resolution crops and independently reviewed labels.
- **Expected signal:** Lost detail after preprocessing supports the resolution hypothesis.
- **Bounded action:** Evaluate higher resolution or tiling under latency limits.
- **Prevention and verification:** Track size-bucket recall and inference cost together.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-11 — Tile seam errors
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Objects near crop boundaries generate misses or duplicates.
- **Hypothesis:** Tiles lack overlap or merged boxes retain tile-local coordinates.
- **Competing explanation:** The base detector may also struggle with truncated objects.
- **Discriminating check:** Compare seam versus tile-interior metrics and map boxes into global space.
- **Expected signal:** A seam-specific concentration indicates crop or merge effects.
- **Bounded action:** Add justified overlap and global-coordinate merging.
- **Prevention and verification:** Keep a seam-specific held-out regression set.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-12 — NMS class policy
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** A valid class disappears near a different class.
- **Hypothesis:** Class-agnostic NMS suppressed it.
- **Competing explanation:** Incorrect class scoring could produce the same final symptom.
- **Discriminating check:** Compare candidates before suppression with class-aware output.
- **Expected signal:** A high-score candidate removed only by cross-class suppression supports NMS.
- **Bounded action:** Select class-aware suppression where duplicate semantics permit.
- **Prevention and verification:** Test overlapping different-class objects explicitly.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-13 — Threshold mismatch
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** Online alert counts disagree with offline counts.
- **Hypothesis:** Different score thresholds or inclusive comparisons are used.
- **Competing explanation:** Preprocessing or model version can also differ.
- **Discriminating check:** Replay the same outputs through both threshold functions.
- **Expected signal:** A mismatch only at cutoff values indicates comparison semantics.
- **Bounded action:** Version and share the threshold policy.
- **Prevention and verification:** Add exact-boundary test cases.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-14 — IoU policy mismatch
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** Reported AP differs across teams.
- **Hypothesis:** The teams used different IoU thresholds or AP interpolation.
- **Competing explanation:** Different image subsets may be responsible.
- **Discriminating check:** Compare image IDs, evaluator config, and matched records on one fixture.
- **Expected signal:** Matching identical but AP different points to interpolation or aggregation.
- **Bounded action:** Recompute with a pinned evaluator and frozen dataset.
- **Prevention and verification:** Store evaluator configuration beside every metric.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-15 — Cross-image matching
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** A prediction is credited to an object in another frame.
- **Hypothesis:** The matching key omitted image identity.
- **Competing explanation:** Image ID reuse after ingestion can produce the same effect.
- **Discriminating check:** Inspect image identifiers on every matched pair.
- **Expected signal:** Any differing ID violates the matching contract.
- **Bounded action:** Partition matching by canonical image ID.
- **Prevention and verification:** Reject ID collisions and retain dataset namespace.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-16 — Reference duplication
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Recall falls despite visually complete detections.
- **Hypothesis:** A single object was annotated twice.
- **Competing explanation:** The detector may instead have missed a second adjacent object.
- **Discriminating check:** Review images with unusually overlapping same-class references.
- **Expected signal:** One physical instance with duplicate labels supports annotation error.
- **Bounded action:** Deduplicate with human adjudication.
- **Prevention and verification:** Add label consistency checks and audit histories.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-17 — Missing annotations
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Many seemingly correct predictions count as FP.
- **Hypothesis:** The labeling policy missed visible objects.
- **Competing explanation:** The detector may be hallucinating plausible objects.
- **Discriminating check:** Blindly review a random sample of apparent FPs.
- **Expected signal:** Confirmed unlabeled objects demonstrate incomplete reference coverage.
- **Bounded action:** Correct labels or mark known unlabeled regions using supported rules.
- **Prevention and verification:** Measure annotation completeness, not just annotator speed.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-18 — Ignored-region policy
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** Crowded-scene AP changes after evaluator migration.
- **Hypothesis:** The new evaluator treats crowd or ignore regions differently.
- **Competing explanation:** A different detection cap can also change crowded-scene results.
- **Discriminating check:** Inspect per-detection ignore flags and max-detection settings.
- **Expected signal:** Differences confined to ignored regions identify policy mismatch.
- **Bounded action:** Use the benchmark-compliant evaluator or explicitly rename the metric.
- **Prevention and verification:** Pin implementation and include crowd fixtures.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-19 — Empty class
- **Category:** NUMERICAL_EDGE; **provenance:** HYPOTHETICAL.
- **Symptom:** Macro AP changes when a class has no references.
- **Hypothesis:** One implementation includes a zero while another excludes invalid AP.
- **Competing explanation:** The class list itself may have changed.
- **Discriminating check:** Count eligible references by class and inspect aggregation masks.
- **Expected signal:** A class with zero references has undefined recall under D2.
- **Bounded action:** Declare and implement an explicit exclusion policy.
- **Prevention and verification:** Publish class support with macro metrics.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-20 — Score ties
- **Category:** NUMERICAL_EDGE; **provenance:** HYPOTHETICAL.
- **Symptom:** AP varies between otherwise identical runs.
- **Hypothesis:** Equal scores are sorted inconsistently.
- **Competing explanation:** Nondeterministic model outputs can also reorder predictions.
- **Discriminating check:** Freeze predictions and compare deterministic tie-breaking runs.
- **Expected signal:** A difference with identical scores and boxes isolates ordering.
- **Bounded action:** Use a stable documented secondary key.
- **Prevention and verification:** Add tied TP/FP ordering fixtures.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-21 — Dataset leakage
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Offline AP is high but new-camera quality is poor.
- **Hypothesis:** Near-duplicate frames crossed train and test boundaries.
- **Competing explanation:** New camera lighting or scene changes may also explain loss.
- **Discriminating check:** Group frames by camera, sequence, and collection time before comparing.
- **Expected signal:** Near-identical cross-split images support leakage.
- **Bounded action:** Resplit by independent deployment unit and retrain.
- **Prevention and verification:** Enforce group-aware data lineage in dataset creation.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-22 — Selection-biased audit
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Audited precision is high while complaints rise.
- **Hypothesis:** Only high-score detections were sampled for labeling.
- **Competing explanation:** The incoming population may have shifted.
- **Discriminating check:** Compare audit inclusion probabilities across scores and cameras.
- **Expected signal:** Zero inclusion probability for low-score events prevents population inference.
- **Bounded action:** Sample both detections and images with known inclusion probabilities.
- **Prevention and verification:** Track sample coverage and report uncertainty.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-23 — Delayed ground truth
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Recent recall appears artificially low.
- **Hypothesis:** Labels have not matured for recent frames.
- **Competing explanation:** An actual detector regression remains possible.
- **Discriminating check:** Plot label completeness against frame age and compare mature cohorts.
- **Expected signal:** Improving completeness with age indicates delayed references.
- **Bounded action:** Delay finalized metrics or explicitly mark immature windows.
- **Prevention and verification:** Monitor labeling lag and cohort maturity.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-24 — Detection cap
- **Category:** SCALE_FAILURE; **provenance:** HYPOTHETICAL.
- **Symptom:** Recall drops only in very crowded images.
- **Hypothesis:** A top-K or max-detections cap truncates valid predictions.
- **Competing explanation:** Model or suppression quality might also worsen in crowd scenes.
- **Discriminating check:** Compare pre-cap versus post-cap matchable detections by object count.
- **Expected signal:** A sharp cutoff at K supports cap truncation.
- **Bounded action:** Raise or redesign the cap after measuring cost and false positives.
- **Prevention and verification:** Track cap-hit rate and crowded-slice recall.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-25 — Latency regression
- **Category:** SCALE_FAILURE; **provenance:** HYPOTHETICAL.
- **Symptom:** End-to-end p99 latency increases after deployment.
- **Hypothesis:** Candidate volume made NMS expensive.
- **Competing explanation:** Queueing, decoding, or device contention can also dominate.
- **Discriminating check:** Trace stage durations and candidate counts at matched load.
- **Expected signal:** A correlated NMS duration increase supports post-processing as cause.
- **Bounded action:** Bound candidates or optimize suppression while measuring recall.
- **Prevention and verification:** Alert on stage and end-to-end latency under load.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-26 — Memory pressure
- **Category:** SCALE_FAILURE; **provenance:** HYPOTHETICAL.
- **Symptom:** Out-of-memory failures occur on large images.
- **Hypothesis:** Input resolution or batch size exceeds available device memory.
- **Competing explanation:** A tensor retention leak may also increase memory over time.
- **Discriminating check:** Compare peak allocation across size and request count.
- **Expected signal:** Size-dependent immediate peaks favor capacity; monotonic growth favors retention.
- **Bounded action:** Reduce batch/resolution or fix retained references as indicated.
- **Prevention and verification:** Load-test target resolutions and alert on memory headroom.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-27 — Color-order mismatch
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Accuracy drops on one serving path.
- **Hypothesis:** RGB and BGR channels were exchanged.
- **Competing explanation:** Different normalization constants can produce similar symptoms.
- **Discriminating check:** Feed a known color patch through both preprocessing paths.
- **Expected signal:** Reversed channel values isolate channel order.
- **Bounded action:** Align preprocessing with training contract.
- **Prevention and verification:** Keep golden tensor fixtures per runtime.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-28 — Quantization regression
- **Category:** NUMERICAL_EDGE; **provenance:** HYPOTHETICAL.
- **Symptom:** A quantized model misses low-contrast objects.
- **Hypothesis:** Activation quantization loses relevant resolution.
- **Competing explanation:** Preprocessing mismatch can produce similar differences.
- **Discriminating check:** Compare float and quantized outputs on identical preprocessed tensors.
- **Expected signal:** Differences present only after quantization support precision loss.
- **Bounded action:** Recalibrate or retain precision for sensitive layers.
- **Prevention and verification:** Gate quantization with slice metrics and hardware measurements.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-29 — Model rollback mismatch
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Rollback restores weights but output schema remains broken.
- **Hypothesis:** Post-processing and label-map versions were not rolled back.
- **Competing explanation:** The input service may also have changed.
- **Discriminating check:** Compare complete model bundle and request contract versions.
- **Expected signal:** A mixed-version bundle supports the compatibility hypothesis.
- **Bounded action:** Roll back the atomic bundle and verify known examples.
- **Prevention and verification:** Version weights, preprocessing, taxonomy, and thresholds together.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-30 — Distribution shift
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Performance drops after a new camera is installed.
- **Hypothesis:** Viewpoint, lighting, or optics changed the input distribution.
- **Competing explanation:** A configuration error can also explain the drop.
- **Discriminating check:** Compare labeled camera slices and inspect preprocessing metadata.
- **Expected signal:** A camera-specific degradation with valid config suggests shift.
- **Bounded action:** Collect representative data and evaluate adaptation safely.
- **Prevention and verification:** Monitor camera-level inputs and audited quality.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-31 — Occlusion label ambiguity
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** Annotators disagree on the extent of boxes.
- **Hypothesis:** One labels visible bounds and another labels full inferred extent.
- **Competing explanation:** A coordinate transform error can look similar.
- **Discriminating check:** Review annotation instructions and compare raw image overlays.
- **Expected signal:** Systematic full-versus-visible differences indicate policy ambiguity.
- **Bounded action:** Adjudicate and version a consistent box policy.
- **Prevention and verification:** Measure inter-annotator agreement by occlusion slice.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-32 — False alarm burst
- **Category:** SCALE_FAILURE; **provenance:** HYPOTHETICAL.
- **Symptom:** Review queue grows although request rate is stable.
- **Hypothesis:** Per-image prediction count or score distribution shifted.
- **Competing explanation:** A downstream worker outage can also grow the queue.
- **Discriminating check:** Measure ingress alert rate and reviewer completion rate separately.
- **Expected signal:** Ingress growth with stable service supports model output change.
- **Bounded action:** Adjust routing or threshold only after labeled impact analysis.
- **Prevention and verification:** Set queue capacity alerts and preserve an escalation path.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-33 — Camera outage
- **Category:** SCALE_FAILURE; **provenance:** HYPOTHETICAL.
- **Symptom:** Detection count collapses to zero for one site.
- **Hypothesis:** Frames stopped arriving or were rejected upstream.
- **Competing explanation:** The scene could genuinely be empty.
- **Discriminating check:** Compare frame heartbeat and decode success with image content.
- **Expected signal:** Missing frames distinguish outage from a valid empty scene.
- **Bounded action:** Restore ingestion and surface unknown rather than no-object status.
- **Prevention and verification:** Alert on missing heartbeat independently of detection counts.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-34 — Privacy exposure
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Diagnostic logs contain identifiable image crops.
- **Hypothesis:** Unredacted debug payloads bypass retention policy.
- **Competing explanation:** A downstream export may also have copied them.
- **Discriminating check:** Trace storage access and retention for sampled log records.
- **Expected signal:** Raw identifiable content in an unauthorized sink confirms exposure.
- **Bounded action:** Restrict access, stop capture, and follow incident policy.
- **Prevention and verification:** Minimize logged imagery and test retention controls.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-35 — High AP low utility
- **Category:** CONCEPT_CONFUSION; **provenance:** HYPOTHETICAL.
- **Symptom:** A detector meets aggregate AP but fails an operational target.
- **Hypothesis:** The operating threshold or rare-class coverage is unsuitable.
- **Competing explanation:** Latency and downstream review capacity may be the bottleneck.
- **Discriminating check:** Measure per-class errors and end-to-end outcomes at the deployed cutoff.
- **Expected signal:** Good offline ranking with poor cutoff recall isolates operating policy.
- **Bounded action:** Choose a validated threshold or redesign the workflow.
- **Prevention and verification:** Gate launch on business guardrails as well as AP.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-36 — Clipping mismatch
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Boxes at image borders have inconsistent IoU.
- **Hypothesis:** One system clips coordinates before evaluation and another does not.
- **Competing explanation:** Different truncated-object annotation policies may also matter.
- **Discriminating check:** Replay boundary boxes with and without clipping.
- **Expected signal:** Changes limited to border fixtures support clipping mismatch.
- **Bounded action:** Adopt the dataset and evaluator clipping contract.
- **Prevention and verification:** Version and test coordinate boundary behavior.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-37 — Excessive compression
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Night-scene performance worsens after bandwidth reduction.
- **Hypothesis:** Compression removes small or low-contrast features.
- **Competing explanation:** Exposure or sensor gain could have changed simultaneously.
- **Discriminating check:** Compare paired original and compressed frames with fixed preprocessing.
- **Expected signal:** A paired degradation after compression supports the hypothesis.
- **Bounded action:** Adjust encoding or route affected scenes to review.
- **Prevention and verification:** Monitor codec settings and evaluate representative night slices.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-38 — Timestamp misalignment
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Overlays appear correct on static scenes but drift on motion.
- **Hypothesis:** Predictions and displayed frames have mismatched timestamps.
- **Competing explanation:** Rolling shutter or slow inference may contribute.
- **Discriminating check:** Compare frame IDs and capture timestamps across the path.
- **Expected signal:** A consistent ID offset indicates alignment failure.
- **Bounded action:** Associate results with immutable frame identifiers.
- **Prevention and verification:** Monitor age of result and dropped-frame counts.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-39 — Class imbalance
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Overall metrics look stable while rare-class recall collapses.
- **Hypothesis:** Frequent classes dominate count-weighted reporting.
- **Competing explanation:** Rare labels may also have high sampling uncertainty.
- **Discriminating check:** Inspect per-class support, recall, and interval estimates.
- **Expected signal:** A rare-class decline hidden in micro totals supports aggregation masking.
- **Bounded action:** Use class-level guardrails and expand the audit sample.
- **Prevention and verification:** Publish support counts and uncertainty with every slice.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

### F-40 — Threshold overfitting
- **Category:** VIOLATED_ASSUMPTION; **provenance:** HYPOTHETICAL.
- **Symptom:** Validation operating point performs poorly on test.
- **Hypothesis:** Threshold was repeatedly tuned on the test set.
- **Competing explanation:** A genuine distribution shift is another plausible explanation.
- **Discriminating check:** Review experiment history and compare untouched time or site holdout.
- **Expected signal:** Repeated test-driven choices indicate selection leakage.
- **Bounded action:** Choose thresholds on validation and evaluate once on held-out data.
- **Prevention and verification:** Keep a sealed final evaluation and record selection decisions.
- **Negative result:** Retain the alternative hypothesis and inspect the next pipeline boundary.

**Takeaway:** Diagnose the affected boundary with a discriminating fixture before changing the model or threshold.

## Section 9 — Complexity & Cost Analysis

### 9a. Algorithmic complexity

Define `N` as scored candidates for one image and class after candidate generation.
Define `G` as eligible references for that image and class.
Define `M` as the number of retained predictions after post-processing.
Define `T` as the number of evaluator IoU thresholds.
Define `P` as the total nonignored predictions for one class across images.
These are input sizes, not model parameter counts.
The bound depends on implementation, spatial indexing, and the number of classes.
A comparison sort of N candidates takes `O(N log N)` time and `O(N)` auxiliary storage in a straightforward implementation.
A naive greedy NMS may compare a retained candidate with each remaining candidate.
Its worst-case pairwise IoU count is `N(N-1)/2`, which is `O(N²)`.
It can use `O(N)` extra working state without materializing every pairwise overlap.
A vectorized implementation that materializes an overlap matrix can instead need `O(N²)` memory.
A simple per-image matching loop computes up to `N*G` IoUs per threshold.
Caching pairwise overlaps can reduce repeated geometric computation across T thresholds.
However, independent matching state is still necessary for each threshold.
A dense cached matrix takes `O(NG)` space and can be wasteful for crowded frames.
Global ranking takes `O(P log P)` comparison time per class under a straightforward sort.
Storing raw predictions and reference annotations requires space proportional to their recorded fields.
These counts exclude network inference, decoding, input I/O, and framework overhead.
Training cost depends on architecture, image size, optimization steps, augmentation, batch, and hardware.
It is not derivable from the NMS bound.
Updating a detector requires retraining or replacing weights, validation, and artifact distribution.
Recovery cost depends on checkpoint size, deployment topology, and traffic routing.
Neither training nor recovery has a universal complexity determined only by N and G.

| Stage | Work model | Working space | Operational caveat |
|---|---|---|---|
| Decode and resize | Implementation and image pixels | Image tensors | Codec, copies, and memory layout matter |
| Model inference | Architecture-dependent | Weights and activations | Batch and precision affect device use |
| Candidate sort | O(N log N) | O(N) | Per-class or global selection changes N |
| Naive NMS | O(N²) worst case | O(N) streaming | Dense pair matrix may require O(N²) |
| Naive matching | O(TNG) per image/class | O(NG) if cached | Ignore, crowd, and cap policies add work |
| Global ranking | O(P log P) | O(P) typical | Disk and distributed shuffle may dominate |
| Rebuild/recovery | Artifact and deployment dependent | Versioned bundle | Partial rollout can mix incompatible versions |

### 9b. Worked sizing and economics

**Evidence:** ILLUSTRATIVE arithmetic only; no runtime, device, price, or deployment measurement is implied.
Suppose there are 100 images per second.
Suppose preprocessing produces 1,000 candidate boxes per image for one suppression group.
Suppose the naive implementation reaches the full pairwise worst case.
For this count, every unordered pair is examined once.
```text
FORMULA [D4]: comparisons/image = N(N-1)/2
WHERE: N is the number of candidates in one suppression group.
CONSTRAINT: naive worst-case pairwise comparison; early suppression can reduce work.
NUMERICAL EXAMPLE: N=1000 gives 1000*999/2=499500 comparisons/image.
INTERPRETATION: At 100 images/s this is 49,950,000 pair comparisons/s.
```
The operation is an IoU comparison, not a floating-point operation or a duration.
If N doubles to 2,000, the worst-case count becomes 1,999,000 per image.
That is approximately four times the original count, not twice.
Actual NMS work depends on grouping, candidate pruning, overlap patterns, and implementation.
The assumed steady arrival rate does not describe burstiness or peak traffic.
For memory, suppose a temporary dense `1000 x 1000` float32 overlap matrix is materialized.
```text
FORMULA [D5]: bytes = rows * columns * bytes_per_element
WHERE: dimensions are element counts and element size is bytes/element.
CONSTRAINT: one dense float32 matrix only; excludes inputs, outputs, allocator overhead, and copies.
NUMERICAL EXAMPLE: 1000*1000*4=4,000,000 bytes, approximately 3.81 MiB.
INTERPRETATION: Concurrent matrices and additional tensors require more memory than this lower bound.
```
A MiB is 1,048,576 bytes; 4,000,000/1,048,576 is approximately 3.8147 MiB.
At four concurrent requests the matrix payload alone would be 16,000,000 bytes, approximately 15.26 MiB.
This does not imply that an actual implementation materializes such a matrix.
No GPU type, driver, runtime version, or device memory is specified for this hypothetical model.
Those must be supplied and measured in an actual capacity plan.
Measure end-to-end latency under representative concurrency and input distributions.
Report p50, p95, and p99 from individual request durations, together with errors and timeouts.
Do not add independently measured stage p99s and call the sum end-to-end p99.
Count queue time, preprocessing, inference, suppression, serialization, and network where relevant.
For economics, divide the observed fully allocated cost for a declared period by useful successfully processed images in that period.
Include compute, storage, data transfer, labeling, human review, and incident labor when those costs are in scope.
State currency, price date, utilization, inclusion rules, and failure treatment.
No currency value is asserted here because neither cost nor volume is measured.

### 9c. Production monitoring contract

The following is a proposed operational plan, not a claim that monitoring exists.
Assign an owner and a response playbook for each metric before deployment.
Log model bundle ID, preprocessing version, camera ID, timestamp, and schema version.
Use a stable opaque image ID rather than embedding sensitive image content in identifiers.
Aggregate dashboards over a declared window and expose the window's image count.
Check ingestion heartbeat independently from zero-detection frequency.
Track frame decode errors separately from empty-scene predictions.
Track input image dimensions, channel order checks, and invalid image rate.
Track acquisition metadata availability without retaining unnecessary personal information.
Track prediction count per image and per class as unlabeled proxies.
Track score distributions before and after deployed thresholds.
Track box size, aspect ratio, and image position distributions.
Track candidate counts before and after NMS and the rate of cap hits.
Track invalid coordinates and out-of-frame boxes before clipping.
Track the proportion of results produced by each model bundle and hardware/runtime path.
Track image age at inference completion when stale results matter.
Track successful inference throughput alongside dropped, timed-out, and retried requests.
Track queue length and queue age where humans or downstream services consume detections.
Track end-to-end request latency distributions at representative offered load.
Break latency down by decode, preprocessing, inference, post-processing, and transport.
Track memory utilization, device saturation, and worker restarts as resource signals.
Use proxy changes to trigger investigation, not to claim a measured recall decline.

### 9d. Labeled quality, uncertainty, and response

Sample complete images independently of whether the detector returned a prediction.
A prediction-only audit cannot find objects in images with zero predictions.
Use a known sampling design that covers sites, times, object sizes, and rare conditions.
Retain sampling probabilities if sampling is intentionally stratified or unequal.
Label all eligible objects under a written annotation policy.
Use blind review or adjudication where detection outputs might bias annotators.
Report sample size, number of eligible objects, and number of independent acquisition groups.
Report per-class precision and recall at the deployed score and IoU thresholds.
Report AP using the pinned evaluator and its full configuration.
Report crowd, size, lighting, camera, and site slices when support is sufficient.
Quantify uncertainty using an appropriate resampling unit such as camera or acquisition sequence.
Frames from one video are correlated; treating every frame as independent can overstate precision.
A confidence interval does not fix biased selection or incorrect labels.
Report label maturity and the time from image capture to completed review.
For a rare slice, an absence of errors in a tiny audit is not proof of safety.
When a guardrail fails, compare the same slice against the previous approved bundle.
Check data contract and annotation changes before attributing causation to model weights.
Route uncertain or unsafe cases to a designated human process where applicable.
Rollback the whole compatible model bundle rather than weights alone.
Verify restoration with known fixtures and a new prospective labeled sample.
Record incident decisions and residual uncertainty without exposing unnecessary images.
Choose alert thresholds using baseline variation, response capacity, and impact.
Do not import illustrative thresholds as universal service-level objectives.

### 9e. Security and governance

Restrict image and annotation access to roles that need it.
Encrypt data in transit and at rest according to the deployment's actual controls.
Specify retention, deletion, consent, and lawful-use requirements for the applicable jurisdiction.
Check whether image-derived crops, embeddings, and debug traces carry personal information.
Avoid publishing unredacted examples merely to explain a metric.
Track artifact provenance, model signatures, and dependency versions.
Test malformed images and resource exhaustion at the input boundary.
Use bounded retries and backpressure so an overloaded inference service cannot amplify traffic indefinitely.
Define what a downstream consumer should do on timeout or unknown status.
A timeout is not evidence that the image contains no object.
A safety-critical decision needs a separately validated system safety case.

### 9f. Evaluation and deployment audit fixtures

These fixtures are proposed checks, not executed tests or evidence of a deployed system.
Each fixture should use the production evaluator and bundle being considered for release.
Record the exact expected result before inspecting the implementation output.
Retain input images, reference metadata, predictions, and policy settings in a versioned test artifact.

**Fixture A — coordinate round trip.**
Start with an off-center box in a nonsquare image.
Apply the actual forward resize and asymmetric letterbox transform.
Apply the recorded inverse transform to all four corners.
Compare the restored box against the original before rounding.
Allow only the documented numeric tolerance.
A shifted corner indicates a different transform, padding convention, or rounding policy.
Check the transform separately for every original image size.

**Fixture B — duplicate reservation.**
Create one normal reference and two identical same-class predictions with different scores.
At a qualifying IoU, expect one TP and one FP under the simple one-to-one policy.
Swap the scores and verify the higher-scored prediction is selected first.
Do not extrapolate this expectation to crowd annotations.
This fixture isolates matching from any NMS applied upstream.
Run the matching test on raw predictions or explicitly disable NMS.

**Fixture C — separate image identity.**
Create identical boxes in two different images, but label an object in only one.
Place a prediction in the other image.
Expect no match across image IDs even though coordinates coincide.
Then create a correct prediction on the labeled image.
Expect that prediction to match without affecting the other image.
Namespace IDs when combining datasets to avoid accidental collisions.

**Fixture D — missing and ignored references.**
Construct one image with no eligible reference and one background prediction.
Under the simple policy, the prediction is an FP.
Then add an ignore or crowd region supported by the intended benchmark.
Determine the expected outcome from the pinned evaluator rather than assuming ordinary matching.
Record whether ignored predictions enter the precision denominator.
Record whether ignored references enter the recall denominator.
This isolates protocol semantics from the detector's visual quality.

**Fixture E — boundary precision.**
Construct a prediction with IoU exactly equal to the evaluation threshold.
Document whether the implementation uses inclusive comparison.
Construct another pair just below the threshold with enough numeric margin to avoid accidental rounding ties.
Compare both outputs against the intended policy.
Perform the same exercise for a score equal to the deployment cutoff.
Threshold comparison and floating-point precision are separate configuration concerns.

**Fixture F — support and aggregation.**
Create one class with labeled instances and one class with none.
Compute the supported class's AP with the pinned evaluator.
Inspect how the unsupported class is represented before averaging.
Confirm the displayed macro denominator matches the evaluator's declared valid-class policy.
Adding an unsupported category should not silently be interpreted as measured failure.
Separately report the category support counts.

**Fixture G — ranking and post-processing.**
Use the four predictions and two references from Section 5.
Evaluate the raw ranked predictions with NMS disabled.
Check the all-points result against 5/6 under the stated simple policy.
Enable the intended NMS and recompute from its actual output.
Any changed AP now reflects both selection and ranking, not matching alone.
Use this fixture to distinguish model-output changes from evaluation-code changes.

**Fixture H — full request behavior.**
Send a valid image through the complete serving path with a unique frame ID.
Verify the response contains the same ID, bundle version, and declared coordinate frame.
Send a malformed image and verify it is reported as an error rather than an empty valid scene.
Simulate a timeout and verify downstream systems receive an unknown or failure state.
For recovery, restore the approved bundle and replay the valid fixture.
Check operational telemetry and access controls without retaining unneeded raw image data.

Document which checks ran, their environment, and the result; a proposed fixture is not a passed test.
If a release uses a different evaluator or post-processing policy, recompute its expected outputs.

**Takeaway:** Complexity estimates locate possible bottlenecks; labeled audits and end-to-end measurements establish actual behavior.

## Section 10 — Key Theorems / Results Reference

| Result | Statement | Essential assumptions | Practical consequence | Does not imply |
|---|---|---|---|---|
| Inclusion-exclusion | Union area equals sum of areas minus intersection | Measurable regions and common frame | D1 calculates overlap without double-counting | Bounding-box enclosure equals union |
| IoU range | IoU is in [0,1] | Nonnegative intersection and positive union | Out-of-range values indicate invalid arithmetic | Identical class labels or correct detections |
| Matching uniqueness | At most one simple-policy TP per eligible reference | One-to-one reservation and no special ignore rule | Duplicate boxes cannot increase TP for one object | Every benchmark uses identical reference selection |
| Recall accounting | TP + FN equals eligible reference count | Fixed reference set and completed matching | Validate the count ledger at each cutoff | Unlabeled physical objects are counted |
| Precision accounting | TP + FP equals selected nonignored predictions | Fixed policy and no unreported ignore | Validate the prediction ledger | A meaningful true-negative count exists |
| Envelope monotonicity | Interpolated precision cannot increase as required recall increases | D3 envelope on a fixed ranked list | Enables stepwise AP integration | Raw precision is monotonic |
| AP range | All-points AP lies in [0,1] | Valid recall increments and precision values | Sanity-check arithmetic | All implementations give the same AP |
| NMS bound | Naive greedy NMS makes at most N(N-1)/2 pair comparisons | One suppression group, each pair compared at most once | Plan worst-case candidate scaling | A latency guarantee or bound for every implementation |

The matching and accounting results are invariants of the specified simple evaluator.
They are not claims about benchmark-specific crowd handling.
The AP envelope is nonincreasing because increasing required recall shrinks the set over which the maximum is taken.
All-points AP is a sum of nonnegative recall increments multiplied by values between zero and one.
The recall increments sum to at most one, establishing the bound.
The official [COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) defines its actual matching, ignore and accumulation behavior.
The [PASCAL VOC 2007 documentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/index.html) documents its competition protocol.
Neither source should be treated as proof of an experiment conducted for this sheet.

**Takeaway:** These identities and invariants have explicit preconditions; implementation behavior must be separately pinned.

## Section 11 — Interview / Exam Questions

Each prompt is illustrative interview practice, not an attributed employer question.
Scores use 0 for wrong, 1 for partial, 2 for correct with assumptions, and 3 for a defended tradeoff.

### Q01 — Level 1 — Definition
**Prompt:** What distinguishes a detection from an image classification?
**Clarification:** Assume one image may contain multiple objects.
**Answer:** A detection identifies an instance location and class, typically with a score; image classification assigns image-level labels.
**Reasoning / check:** Two dogs need two instance boxes even if the image-level label is simply dog.
**Alternative or boundary:** Segmentation is appropriate if pixel-level shape is required.
**Follow-up:** Why not infer object counts directly from the classification label?
**Follow-up answer / red flag:** A class label does not encode instance multiplicity.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Section 3.2.

### Q02 — Level 1 — Definition
**Prompt:** What is the difference between score threshold and IoU threshold?
**Clarification:** Consider a fixed evaluator.
**Answer:** A score threshold selects predictions; the IoU threshold defines sufficient box overlap for a match.
**Reasoning / check:** Their numeric values are independent even when both lie between zero and one.
**Alternative or boundary:** An NMS IoU threshold is a third distinct setting.
**Follow-up:** What happens if you increase the score cutoff?
**Follow-up answer / red flag:** It selects no more outputs, but precision need not monotonically improve.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Sections 3 and 4.

### Q03 — Level 1 — Definition
**Prompt:** Is an objectness score automatically a calibrated probability?
**Clarification:** Assume a model emits a number between zero and one.
**Answer:** No; a bounded score may rank candidates without matching observed event frequencies.
**Reasoning / check:** Calibration must be assessed against a defined event, matching policy, and labeled population.
**Alternative or boundary:** A calibrated score can still drift after a population shift.
**Follow-up:** Can AP establish calibration?
**Follow-up answer / red flag:** No; many monotone score transformations preserve ranking and AP while changing calibration.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Section 3.2.

### Q04 — Level 2 — Geometry
**Prompt:** Compute IoU of boxes (0,0,10,10) and (5,0,15,10).
**Clarification:** Use continuous xyxy pixels and positive areas.
**Answer:** Intersection is 5*10=50 px²; union is 100+100-50=150 px²; IoU is 1/3.
**Reasoning / check:** The dimensionless result is below 0.5 and agrees with D1.
**Alternative or boundary:** An inclusive pixel convention yields different arithmetic and must be stated.
**Follow-up:** What if the boxes only touch at x=10?
**Follow-up answer / red flag:** Intersection area and IoU are zero under this continuous convention.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Section 4.1.

### Q05 — Level 2 — Matching
**Prompt:** One ground truth has two same-class predictions at IoU 1 with scores 0.9 and 0.8; what are the results?
**Clarification:** Use one-to-one matching with no ignore regions.
**Answer:** The first is TP; the second is a duplicate FP because the object is already reserved.
**Reasoning / check:** The ledger has one TP, one FP, and zero FN.
**Alternative or boundary:** An ignored or crowd reference may use special evaluator-specific rules.
**Follow-up:** Would an IoU of one turn the duplicate into a second TP?
**Follow-up answer / red flag:** No; geometry is necessary but reference eligibility and one-to-one matching also matter.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Sections 4.2 and 5.

### Q06 — Level 2 — AP trace
**Prompt:** Derive AP for TP, FP, TP with two references under the all-points envelope.
**Clarification:** Scores are strictly decreasing and all other policy choices match Section 5.
**Answer:** Precision at recalls 1/2 and 1 is 1 and 2/3; AP=(1/2)*1+(1/2)*(2/3)=5/6.
**Reasoning / check:** The duplicate FP decreases precision without creating a recall increment.
**Alternative or boundary:** Eleven-point and 101-point sampling produce different results.
**Follow-up:** What if both TPs precede the FP?
**Follow-up answer / red flag:** The AP becomes one although the final precision remains 2/3.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Section 5.

### Q07 — Level 2 — Invariant
**Prompt:** Why can the interpolated precision envelope not increase with required recall?
**Clarification:** Use the definition in D3.
**Answer:** The candidate set of points at or above recall r shrinks as r increases.
**Reasoning / check:** The maximum over a subset cannot exceed the maximum over the original set.
**Alternative or boundary:** Raw precision can increase after a true positive even while the envelope remains nonincreasing.
**Follow-up:** Does this establish calibration?
**Follow-up answer / red flag:** No; it is an ordering property of the evaluation functional.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Sections 4.2 and 10.

### Q08 — Level 3 — Diagnosis
**Prompt:** Offline AP fell after a serving change; where do you start?
**Clarification:** Assume the same evaluation set is available and do not assume the weights changed.
**Answer:** Freeze paired outputs and compare bundle, preprocessing, coordinates, class map, thresholds, and evaluator settings.
**Reasoning / check:** Replay known fixtures and inspect the first stage whose outputs diverge.
**Alternative or boundary:** If inputs and transforms match, investigate score, model, and label changes.
**Follow-up:** How would a consistent coordinate offset change your hypothesis?
**Follow-up answer / red flag:** It favors transform or padding mismatch over a general model-quality decline.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Sections 8 and 9.

### Q09 — Level 3 — Ambiguous symptom
**Prompt:** The review queue is growing while request volume is stable; diagnose it.
**Clarification:** You have alert arrival and reviewer-completion timestamps.
**Answer:** Compare alert ingress, service completions, queue age, and per-image detection count.
**Reasoning / check:** Higher alert ingress suggests output or threshold drift; lower completions suggests review capacity failure.
**Alternative or boundary:** Both may change at once, and labels are needed to judge quality.
**Follow-up:** Why not immediately raise the score cutoff?
**Follow-up answer / red flag:** It could hide real objects and reduce recall without addressing a reviewer outage.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Sections 8 and 9.4.

### Q10 — Level 3 — Constraint change
**Prompt:** A detector has good AP but cannot meet the target-device deadline. What do you choose?
**Clarification:** Hard latency is a binding deployment requirement.
**Answer:** Reject the infeasible configuration; benchmark a simpler model, lower resolution, or candidate pruning.
**Reasoning / check:** Measure per-slice quality and end-to-end latency for each feasible option.
**Alternative or boundary:** A more capable device is defensible if budget and deployment constraints permit it.
**Follow-up:** What if the deadline becomes softer but rare small-object recall is mandatory?
**Follow-up answer / red flag:** Reconsider higher resolution or tiles, while measuring memory and latency.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Sections 7 and 9.

### Q11 — Level 3 — Sampling
**Prompt:** Can you measure recall by labeling only detections sent to human review?
**Clarification:** Assume images with no detection are never sampled.
**Answer:** No; missed objects in unreviewed images are outside the sample and the denominator is not observable.
**Reasoning / check:** Sample complete images independently of detections and label all eligible instances.
**Alternative or boundary:** Known probability sampling and suitable weights can support population estimates.
**Follow-up:** Does a narrow confidence interval remove this bias?
**Follow-up answer / red flag:** No; precision of a biased estimate does not establish validity.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Section 9.4.

### Q12 — Level 3 — Evaluation definition
**Prompt:** Two teams report different mAP on identical predictions; what settings can explain it?
**Clarification:** Neither report initially specifies its full policy.
**Answer:** Compare IoU grid, interpolation, reference eligibility, category aggregation, detection caps, and score tie ordering.
**Reasoning / check:** Reproduce a single small fixture before scaling to the full dataset.
**Alternative or boundary:** Different annotation files or image subsets are also possible.
**Follow-up:** Why is the term mAP alone insufficient?
**Follow-up answer / red flag:** It does not uniquely specify the averaging axes or evaluation protocol.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Sections 4.3 and 5.

### Q13 — Level 3 — Verification and release
**Prompt:** How do you establish that a rollback actually restored behavior?
**Clarification:** A bundle includes model, preprocessing, taxonomy, thresholds, and runtime.
**Answer:** Restore the compatible bundle atomically; check known fixtures, error rates, latency, and labeled quality on a prospective sample.
**Reasoning / check:** Record which traffic received each bundle and examine affected slices.
**Alternative or boundary:** Shadow evaluation can reduce risk when immediate reversal is not required.
**Follow-up:** Why is restoring only the weight file inadequate?
**Follow-up answer / red flag:** Other versioned components can still shift coordinates, scores, or class labels.
**Scoring focus:** Apply the 0–3 rubric above to the answer and its assumptions.
**Answer support:** Sections 8 and 9.

**Takeaway:** Strong answers name the evaluator, show the arithmetic or trace, and defend the operational tradeoff.

## Section 12 — Quick Reference Card

| Item | Value |
|---|---|
| Topic | Object detection geometry, ranking, matching, and deployment decisions. |
| Core formula | IoU = intersection / union for valid same-frame boxes; precision = TP/(TP+FP); recall = TP/(TP+FN) when denominators are positive. |
| Key assumption | State coordinate convention, eligible references, one-to-one or benchmark-specific matching, IoU threshold, and AP interpolation. |
| Complexity | Naive NMS: O(N²) comparisons worst case; simple evaluation: O(TNG) matching per image/class, plus ranking. |
| Use when | Localize and classify multiple instances, and evaluate under a documented cost and measurement policy. |
| Avoid when | Labels, coordinate transforms, or safety and latency requirements are unavailable or incompatible. |
| Go deeper | [Course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md); segmentation and tracking are separate planned handoffs. |
| Diagnostic | Compare paired predictions and matching records before attributing metric shifts to weights. |
| Evidence boundary | Draft, unreviewed; examples and capacity inputs are illustrative, not measured production results. |
