---
collection: "module-17-vision"
file_id: "04"
topic: "Object detection and segmentation"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Label ontology, IoU threshold, prevalence, camera resolution, or review-capacity change."
notation_style: "plain-text"
prerequisites: ["01_Vision_Evaluation_and_Error_Taxonomy.md", "02_CNNs_Attention_and_Transfer_Learning.md"]
pattern_tags: ["detection", "segmentation", "evaluation"]
source_status: "OPEN — visible source names only"
owner_role: "Vision ML reviewer"
source_index: "INDEX.md"
---

# Detection and Segmentation

> **Purpose:** Define the output grain and matching protocol before selecting a box or mask model.
> **Prerequisites:** [Vision evaluation](01_Vision_Evaluation_and_Error_Taxonomy.md) and grouped split design.
> **Key outcome:** Given boxes or masks and threshold rules, compute IoU, classify match outcomes, and identify when image classification is insufficient.
> **This file does NOT cover:** clinical diagnosis, tracking, or measured detector comparisons.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED overlap arithmetic; VERIFIED illustrative trace; SOURCED method definitions; not measured.
> **Retrieval cue:** If AP improves but the review queue fails, start at Sections 5, 7 and 8.

## 1. File Header

Detection predicts object instances, usually boxes plus classes and scores. Segmentation predicts pixels or regions. The inventory names `04_Object_Detection.md`, `05_Image_Segmentation.md`, and `14_CV_Evaluation_Metrics.md` at [`directory.txt`](../../../../Templates/directory.txt#L552-L563). Their original bodies are unavailable; this is a new, bounded teaching sheet.

**Takeaway:** Choose the least detailed output that can support the real downstream action.

## 2. Concept Map / Mental Model

```text
image -> encoder -> candidate boxes/masks + scores -> class/overlap filtering
labels -> matching protocol (class, IoU threshold, one-to-one rule) -> TP/FP/FN
TP/FP/FN + score threshold + review capacity -> operational decision
error slices: object size | occlusion | class | site | annotator disagreement
```

**Takeaway:** A detector score has no meaning without an explicit matching and threshold protocol.

## 3. Foundational Definitions

| Term | Meaning | Units / shape | Boundary |
|---|---|---|---|
| box | [x_min,y_min,x_max,y_max] spatial extent | pixels in declared frame | coordinate convention explicit |
| mask | pixel membership prediction | H-by-W binary/probability grid | resolution and class policy explicit |
| IoU | overlap divided by union | [0,1] | same coordinate/mask frame |
| TP/FP/FN | matched prediction / unmatched prediction / missed label | counts | one-to-one matching rule |
| NMS | suppress redundant candidates | procedure | can remove nearby true instances |
| AP | area under a precision-recall construction | dimensionless | interpolation and IoU threshold stated |

**Takeaway:** Counts are outcomes of a protocol, not inherent properties of model outputs.

## 4. Core Formulas with Full Derivation

Validity before use: prediction and label refer to the same class and coordinate system. Union contains all predicted-only, label-only, and overlap area.

```text
FORMULA [DET-01]: IoU = I/(P + G - I)
WHERE: I intersection area [pixels^2]; P predicted area [pixels^2]; G ground-truth area [pixels^2].
CONSTRAINT: P+G-I>0; class and coordinate rules are separate from geometry.
NUMERICAL EXAMPLE: I=60, P=80, G=80 gives IoU=60/(80+80-60)=0.60.
INTERPRETATION: at a 0.50 threshold this pair can match; at 0.75 it cannot.
```

Validity before use: a frozen score threshold and one-to-one matching have produced count totals.

```text
FORMULA [DET-02]: precision=TP/(TP+FP); recall=TP/(TP+FN)
WHERE: TP, FP, FN are nonnegative matched-event counts; both results dimensionless.
CONSTRAINT: denominators nonzero; report support and score/IoU rules.
NUMERICAL EXAMPLE: TP=18, FP=6, FN=2 gives precision=18/24=0.75 and recall=18/20=0.90.
INTERPRETATION: 25% of alerted candidates are false under this illustrative protocol.
```

For a binary mask, Dice=2I/(P+G); it weights overlap differently from IoU and is not interchangeable in a stated acceptance criterion.

Counterexample: a large background can make pixel accuracy high when a tiny lesion is entirely missed. Use foreground-sensitive metrics and an action-relevant review protocol.

**Takeaway:** IoU threshold selection changes the definition of success.

## 5. Worked Numerical Example

**ILLUSTRATIVE, VERIFIED trace.** A frame contains two labelled bolts. At score threshold 0.60, three same-class predicted boxes appear. Greedy matching by descending score yields one pair IoU=0.80 (TP), one pair IoU=0.60 (TP at IoU>=0.50), and one duplicate overlapping the first label (FP because that label is already matched). Thus TP=2, FP=1, FN=0, precision=2/3, recall=1.0.

Raising the IoU requirement to 0.75 makes the 0.60 pair unmatched: TP=1, FP=2, FN=1, precision=1/3, recall=1/2. This does not prove a detector changed; only the evaluation contract changed. If reviewers can inspect at most 30 alerts/hour, calculate alert volume across a held-out workload before lowering the score threshold to chase recall.

Sensitivity: if the second label has ambiguous boundaries, an adjudication study may move IoU from 0.60 to 0.72 without model change. Do not call that a model regression without label evidence.

**Takeaway:** Show the matching trace so readers can reproduce TP, FP, and FN.

## 6. Visual / Geometric Intuition

```text
label:      +---------+
prediction:     +---------+
intersection:   +-----+
union:      +-------------+

one label can match one prediction only; a second duplicate is an FP.
mask output preserves boundary shape; a box trades geometry for simpler annotation/action.
```

**Takeaway:** One-to-one matching prevents duplicate predictions from inflating apparent detection success.

## 7. Comparison / Selection Table

| Output | Use when | Strength | Avoid when | Gate / fallback |
|---|---|---|---|---|
| image classification | one image-level disposition suffices | cheapest labels/serving | location changes action | add box or review |
| boxes | location and count matter | practical annotation | irregular boundary determines quantity | mask/contour task |
| semantic mask | every pixel class matters | region coverage | instances must be separated | instance segmentation |
| instance mask | separate objects/boundaries matter | precise action area | labels/cost cannot support it | boxes + human review |

**Takeaway:** Annotation precision must buy a decision benefit, not an aesthetic metric.

## 8. Failure Modes & Edge Cases

| Failure | Competing causes | Diagnostic | Bounded correction | Verification |
|---|---|---|---|---|
| duplicate alerts | weak NMS or nearby true objects | inspect candidate geometry and label count | tune suppression only after edge-case policy | duplicate FP rate |
| small-object recall collapse | resize loss or annotation omissions | size-slice labels and raw-resolution audit | change resolution/tiling or label policy | held-out size recall |
| AP/queue mismatch | low threshold or prevalence shift | alert volume by score and hour | threshold with capacity constraint | shadow queue |
| mask boundary disagreement | label ambiguity or model boundary error | independent annotation overlap | adjudication protocol; uncertainty route | inter-annotator agreement |
| coordinate error | resize/letterbox transform bug or labels wrong | round-trip a known box | centralise transform metadata | pixel-overlay test |

**Takeaway:** A correct metric cannot repair an incorrect coordinate or label contract.

## 9. Complexity & Cost Analysis

Dense proposal or mask heads scale with feature-map positions and classes; postprocessing often scales with candidate count. Naive NMS pair checks are O(M^2) for M candidates, although implementations may prune/sort. A 1,000-candidate image has up to 499,500 unordered candidate pairs. This is a worst-case comparison count, not a latency claim. Mask annotation/review cost can dominate inference cost; measure annotation time and inter-annotator disagreement for the target workflow.

**Takeaway:** Candidate count and annotation protocol are first-class capacity variables.

## 10. Key Theorems / Results Reference

- IoU is a set-overlap ratio with range [0,1] for nonempty union.
- [Lin et al., *Microsoft COCO*](https://arxiv.org/abs/1405.0312), 2014, documents a detection evaluation context; it does not define every local business threshold.
- [Ronneberger et al., *U-Net*](https://arxiv.org/abs/1505.04597), 2015, is a primary segmentation architecture reference; reviewed 2026-11-09.

**Takeaway:** Benchmark protocols must be declared, not presumed from an architecture name.

## 11. Interview / Exam Questions

1. **Compute IoU for intersection 25, predicted 40, label 50.** 25/(40+50-25)=25/65≈0.3846.
2. **Why is a duplicate box an FP?** One-to-one matching already assigns the label to the highest eligible prediction.
3. **When does a mask beat a box?** When boundary/area affects the decision and labels can be reliably acquired.
4. **Why can pixel accuracy mislead?** Background dominance permits high accuracy while missing small foreground objects.
5. **A new camera loses small-object recall. First check?** Coordinate/preprocessing and resolution before retraining.
6. **Why report AP threshold details?** IoU, class, score and interpolation choices define the quantity.
7. **Changed constraint: reviewers halve.** Recompute threshold/alert-volume tradeoff and consider abstention.
8. **How detect label ambiguity?** Independent annotations and disagreement analysis, not a model score alone.
9. **What does NMS risk?** Removing a genuine adjacent instance.
10. **What is a plausible wrong fix for queue overload?** Claiming higher AP solves operational volume without threshold/prevalence analysis.

**Takeaway:** Interview answers should connect geometry to an operational decision.

## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Geometry | DET-01; declare coordinates and class rule |
| Decision counts | DET-02 after frozen threshold and one-to-one matching |
| Output grain | Classification, box, or mask must match action |
| Evaluation | Report IoU thresholds, support, slices, and alert volume |
| Diagnostics | Audit labels, transforms, size slices, duplicates, capacity |
| Avoid | Treating AP, Dice, and pixel accuracy as interchangeable |
| Evidence | Illustrative trace only; source bodies unavailable |
