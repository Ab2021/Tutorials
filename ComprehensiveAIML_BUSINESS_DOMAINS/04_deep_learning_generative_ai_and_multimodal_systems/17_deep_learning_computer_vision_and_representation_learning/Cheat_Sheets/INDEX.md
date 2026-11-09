# Module 17 Cheat Sheets — Deep Learning, Computer Vision and Representation Learning

## 1. Overview

This collection teaches visual representation selection, ViT token budgeting, detection/mask evaluation, self-supervised pair contracts, and error-taxonomy review. It is for readers who must convert image evidence into a bounded decision. All four newly linked sheets are drafts; original source bodies were unavailable. Every visible source title in [`directory.txt`](../../../../Templates/directory.txt#L547-L566) is inventory provenance, not a claim that its content was inspected.

## 2. Topic ladder

| Level | File | Outcome | Prerequisites | Key skill | Status |
|---|---|---|---|---|---|
| Foundation | [02 CNNs, attention and transfer](02_CNNs_Attention_and_Transfer_Learning.md) | compute geometry/weights and choose adaptation baseline | convolution, grouped splits | CNN-01–03 | draft |
| Core | [03 Vision transformers](03_Vision_Transformers.md) | budget patch tokens and select attention structure | 02 | VIT-01–02 | draft |
| Core | [04 Detection and segmentation](04_Detection_and_Segmentation.md) | trace matching and choose output grain | evaluation protocol | DET-01–02 | draft |
| Core | [05 Self-supervision](05_Self_Supervision_and_Contrastive_Learning.md) | audit pair semantics and contrastive loss | 02 | SSL-01 | draft |
| Evaluation | [01 Vision evaluation](01_Vision_Evaluation_and_Error_Taxonomy.md) | specify task, split, slices and error taxonomy | all above | metric contract | incomplete draft |
| Production | Common Issues | diagnose cross-topic failures | all sheets | issue protocol | not authored |

## 3. Quick topic lookup

| Question | File / section | Use condition | Warning |
|---|---|---|---|
| What does stride do to a feature map? | 02 §4 | dense convolution | padding/dilation must be stated |
| Will a resolution change fit attention? | 03 §4–5 | patch ViT | score count is not measured latency |
| Is a box a true positive? | 04 §4–5 | declared matching protocol | IoU threshold changes result |
| Why did contrastive loss rise? | 05 §5, §8 | pair data available | it may be a false negative |
| Which visual errors require review? | 01 §4–7 | task evaluation | existing file needs full-template expansion |

## 4. Reading paths

- **Beginner:** 02 → 01; learn representation geometry then task/split contract.
- **Practitioner:** 02 → 03 → 04 → 01; choose encoder and output under a frozen evaluation protocol.
- **Advanced:** 02 → 03 → 05 → 04 → 01; add contrastive-data and attention-cost reasoning.
- **Optional:** Read 05 before 03 when unlabelled visual data is the binding resource.

## 5. Prerequisite skills

Readers need matrix/vector multiplication, cross-entropy, train/validation/test separation, probability calibration basics, and pixel coordinate conventions. The module [INDEX](../INDEX.md) states programme-level prerequisites. No external source is treated as a completed prerequisite.

## 6. Exclusions and handoffs

This collection does not provide clinical validation, safety certification, video tracking, 3D reconstruction, generative image models, or measured hardware benchmarks. Hand off document/VLM tasks to Module 22; domain policy and human-review design remain workflow-specific.

## 7. Master symbol glossary

| Symbol/term | Meaning | Units | Shape/type | Valid range | First use |
|---|---|---|---|---|---|
| H,W | image height/width | pixels | scalar | positive | 02, 03 |
| P | patch side/padding by context | pixels | scalar | define locally | 02, 03 |
| N | token count or parameter count by local formula | count | scalar | nonnegative | 02, 03 |
| D | embedding width | features | scalar | positive | 03, 05 |
| IoU | intersection over union | dimensionless | scalar | [0,1] | 04 |
| TP/FP/FN | matched decision counts | count | scalar | nonnegative | 04 |
| tau | contrastive temperature | dimensionless | scalar | >0 | 05 |

## 8. Quick Reference Card directory

| Card | Rule | Critical assumption | Avoid |
|---|---|---|---|
| [02 §12](02_CNNs_Attention_and_Transfer_Learning.md#12-quick-reference-card) | geometry and transfer gate | grouped target evidence | random correlated splits |
| [03 §12](03_Vision_Transformers.md#12-quick-reference-card) | N=(H/P)(W/P) | tokenizer declared | treating count as latency |
| [04 §12](04_Detection_and_Segmentation.md#12-quick-reference-card) | one-to-one IoU matching | frozen protocol | metric substitution |
| [05 §12](05_Self_Supervision_and_Contrastive_Learning.md#12-quick-reference-card) | audit positives/negatives | invariance is valid | pretext loss as outcome |

## 9. Common Issues locator

No final Common Issues artifact exists. Required target: at least 25 distinct complete issues using the four course categories. Existing failure tables are local diagnostics, not a collection-level issue artifact.

## 10. Sources, versions and maintenance

| Source ID | Primary source / scope | Date | Reviewed | Limitation / trigger |
|---|---|---:|---:|---|
| INV-17 | [`directory.txt` lines 547–566](../../../../Templates/directory.txt#L547-L566) | inventory | 2026-11-09 | visible names only; source bodies unavailable |
| SRC-17-01 | [ViT](https://arxiv.org/abs/2010.11929) | 2020 | 2026-11-09 | architecture paper, not local benchmark |
| SRC-17-02 | [SimCLR](https://arxiv.org/abs/2002.05709) | 2020 | 2026-11-09 | objective context, not target evidence |
| SRC-17-03 | [COCO](https://arxiv.org/abs/1405.0312) | 2014 | 2026-11-09 | protocol context; local thresholds open |

Recheck inputs, label policy, device, target population and any model implementation before publication.

## 11. Acceptance checklist

- [x] Four focused sheets have real paths and 12 ordered headings.
- [ ] Every authored individual file has the required 1,000 substantive lines; these drafts do not yet meet that course gate.
- [ ] The existing evaluation draft requires expansion to all 12 headings and answered interview pack.
- [ ] Augmentation, source-specific biosignal variants, 25-issue artifact, two bounded cases and full case catalog remain unauthored.
- [x] Formula examples are explicitly illustrative and hand-checked, not executed benchmarks.
- [x] Source-body uncertainty is visible.
