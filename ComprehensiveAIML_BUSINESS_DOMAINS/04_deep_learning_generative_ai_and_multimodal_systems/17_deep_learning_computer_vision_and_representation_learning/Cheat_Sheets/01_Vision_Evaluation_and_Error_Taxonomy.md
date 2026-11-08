---
collection: "deep-learning-computer-vision-and-representation-learning"
file_id: "01"
topic: "Vision evaluation and error taxonomy"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Vision Evaluation and Error Taxonomy

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) lists visible source titles only. This is a new course-topic draft, not a reconstruction of unavailable source bodies.

## 1. Start with the task contract

Declare whether output is class, box, mask, keypoint, track or embedding; the input population, label protocol, decision threshold, latency/hardware limit and consequences of false outcomes. A benchmark score does not establish safety, fairness or deployment fitness.

## 2. Metric selection

| Task | Core measure | Critical qualifier |
|---|---|---|
| Classification | precision, recall, calibrated risk | class prevalence and threshold |
| Detection | AP at declared IoU thresholds | matching, size and confidence rules |
| Segmentation | IoU/Dice | background policy and object size |
| Retrieval | recall@k / mAP | gallery, duplicate and identity protocol |

For masks, `IoU=intersection/union`. A prediction with 60 overlapping pixels, 20 predicted-only and 20 label-only pixels has IoU `60/(60+20+20)=0.60`.

## 3. Dataset partitions

Split by the dependency that exists at deployment: patient/device/site/time/video/scene, not just images. Keep near duplicates and sequential frames in one partition. Freeze test labels and selection protocol; repeated tuning on test data converts it into validation data.

## 4. Error taxonomy

Classify errors by label validity, visibility/occlusion, scale, illumination, motion/blur, viewpoint, background confounder, domain/site/device shift, class ambiguity, threshold and annotation boundary. Each bucket needs support count, rate, examples and an intervention hypothesis.

## 5. Calibration and thresholding

A model confidence is actionable only after calibration against representative data. Choose threshold from false-positive/false-negative cost and review capacity. Evaluate threshold curves overall and in supported slices; an aggregate threshold may fail a high-cost subgroup.

## 6. Evaluation procedure

1. Freeze task, label and split contracts.
2. Establish a simple baseline and data-quality audit.
3. Report primary metric with support and uncertainty.
4. Slice by meaningful conditions and inspect errors.
5. Stress-test plausible shifts and input corruption.
6. Run deployment shadow/review path with feedback and rollback rule.

## 7. Failure modes

| Failure | Signal | Mitigation |
|---|---|---|
| Frame leakage | validation score falls on video-level split | group by recording/scene |
| Label noise masked as model error | reviewer disagreement cluster | adjudicate/probabilistic label policy |
| Small-object miss | size-slice recall collapse | resolution/data/model review |
| Shortcut/background learning | domain shift failure | diversified data and controlled tests |
| Metric–workflow mismatch | AP improves, reviewer queue overloads | threshold/capacity evaluation |

## 8. Interview checks

1. **Why image-random split fails for video?** Adjacent frames leak scene identity.
2. **What does IoU measure?** Overlap divided by union under declared masks.
3. **Why inspect slices?** Aggregate scores can hide failure mechanisms.
4. **What is calibration for?** Turning scores into decision-relevant probability estimates.
5. **What makes a taxonomy useful?** It links observable errors to testable interventions.

## 9. Quick Reference Card

Record task/output, labels, split group, metric protocol, threshold/cost, slices, shift tests, human review, latency/hardware and rollback. Re-evaluate after data, device, policy or label changes.

**Limitations:** The numerical IoU is illustrative. No device, clinical, image or benchmark performance is claimed.
