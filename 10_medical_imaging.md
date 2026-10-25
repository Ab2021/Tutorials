# Module 10 — Medical Imaging

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 32 hours (12 lecture, 4 case clinic, 16 lab)
> **Track fit:** MedTech & Imaging core; elective for others
> **Prerequisites:** Modules 01, 02, 06, 07
> **Tools:** PyTorch, MONAI, pydicom, nibabel, ITK-SNAP or 3D Slicer, Weights & Biases
> **Industry question:** "Does this imaging model improve the radiology pathway without creating new harm, and can we prove it on the scanners and patients we will actually see?"

---

## 1. Why industry cares

Medical imaging is one of the most mature areas for clinical AI, and one of the most prone to inflated claims.

- Imaging is high-volume, high-cost, and backlogged.
- Radiologists face fatigue and workforce shortages.
- Many tasks are repetitive and measurable.
- Regulatory pathways for imaging AI are relatively established.
- Reimbursement for some imaging AI exists.

But imaging AI fails when:

- Training and deployment scanners differ.
- Labels come from reports with errors.
- The model learns shortcuts (laterality markers, chest tubes, scanner artifacts).
- Validation is on a curated academic dataset.
- The model does not fit the reading workflow.
- The population has different disease prevalence.
- The model is not calibrated or monitored.

This module teaches the full imaging pipeline from DICOM to regulated product.

## 2. Learning outcomes

By the end of this module, you can:

- Explain the physics and workflow of major imaging modalities.
- Load, de-identify, and preprocess DICOM data correctly.
- Design annotation and labeling workflows for imaging.
- Train classification, detection, and segmentation models.
- Prevent patient, site, and scanner leakage.
- Evaluate imaging models with task-appropriate metrics and subgroup analysis.
- Interpret saliency and error patterns critically.
- Map an imaging product to FDA/CE pathways and workflow roles.
- Design a monitoring plan for scanner and protocol drift.
- Communicate imaging AI performance to radiologists and executives.

## 3. Mental model: the imaging pathway

```
Order → Protocol → Acquisition → Reconstruction → PACS → Read → Report → Action
   │         │            │              │           │       │
   │         │            │              │           │       └─ Communication
   │         │            │              │           └─ Triage / concurrent / second read
   │         │            │              └─ Metadata, series, windowing
   │         │            └─ Scanner, dose, contrast, technique
   │         └─ Body part, indication, contrast
   └─ Clinical question
```

An imaging model is a component in this pathway, not a standalone oracle.

## 4. Modalities and data characteristics

| Modality | Physics | Typical tasks | Data issues |
|---|---|---|---|
| X-ray | X-ray attenuation | Fracture, pneumonia, tuberculosis | Portable technique, positioning, low contrast |
| CT | Attenuation from multiple angles | Stroke, pulmonary embolism, cancer, trauma | Slice thickness, contrast phase, kernel |
| MRI | Magnetic resonance | Brain, spine, prostate, cardiac | Sequence variability, intensity non-standardization |
| Ultrasound | Sound waves | Cardiac, obstetric, abdominal | Operator dependence, speckle, angle |
| PET | Radiotracer uptake | Oncology, neuro | Tracer, uptake time, resolution |
| Mammography | Low-dose X-ray | Breast cancer screening | Breast density, positioning, 2D/3D |
| Pathology | Microscopy | Cancer diagnosis, grading | Gigapixel images, stain variation |
| OCT | Light interferometry | Retina | Acquisition artifacts, segmentation |

## 5. DICOM

### 5.1 Key identifiers

- `PatientID` (replace).
- `StudyInstanceUID`.
- `SeriesInstanceUID`.
- `SOPInstanceUID`.
- `AccessionNumber` (replace).
- `StudyDate` (shift).
- `Modality`.
- `BodyPartExamined`.
- `Manufacturer`, `ManufacturerModelName`, `SoftwareVersions`.
- `SliceThickness`, `PixelSpacing`, `Rows`, `Columns`.
- `RescaleSlope`, `RescaleIntercept`.

### 5.2 Pixel data

- Apply rescale slope/intercept for CT Hounsfield units.
- Handle MONOCHROME vs RGB, photometric interpretation.
- Respect window center/width for display, but model input may use different normalization.
- Preserve spatial orientation and spacing.
- Handle 2D, 3D, and multi-frame data.

### 5.3 De-identification

- Remove direct identifiers.
- Remove private tags.
- Burned-in annotations: inspect pixels with OCR.
- Date shifting consistent per patient.
- Remove scanner serial numbers if they create site leakage.
- Validate with a tool and manual audit.

## 6. Preprocessing

| Step | Purpose | Pitfall |
|---|---|---|
| Resampling | Consistent voxel spacing | Over-smoothing |
| Orientation | Anatomical consistency | Left/right flips |
| Cropping | Remove irrelevant regions | Cutting off pathology |
| Normalization | Intensity consistency | Site-specific windowing |
| Augmentation | Robustness | Unrealistic transformations |
| Patch extraction | Memory efficiency | Losing context |
| Registration | Align images | Distortion |

**Rule:** preprocessing must be identical at training and inference. Version it.

## 7. Labels and annotation

### 7.1 Label sources

- Radiology report (NLP-extracted).
- Structured report.
- Expert annotation.
- Pathology confirmation.
- Clinical outcome.
- Registry.

### 7.2 Annotation quality

- Define the finding precisely.
- Use multiple readers for a subset.
- Measure inter-reader agreement.
- Adjudicate disagreements.
- Include negative and uncertain cases.
- Record annotation provenance.

### 7.3 Weak labels

Report-derived labels are noisy. A report may mention a finding that is not present, or omit a finding. Use uncertainty labels and validation.

## 8. Tasks and architectures

### 8.1 Classification

- CNN or ViT.
- Multi-label for multiple findings.
- Ordinal for severity.
- Attention maps for localization.

### 8.2 Detection

- Bounding boxes.
- Lesion detection.
- 2D vs 3D.
- Metrics: sensitivity at fixed false positives per image.

### 8.3 Segmentation

- U-Net and variants.
- Dice, IoU, Hausdorff distance.
- 2D vs 3D.
- Annotation cost.

### 8.4 Foundation models

- MedSAM for promptable segmentation.
- BiomedCLIP for vision-language.
- Self-supervised pretraining on unlabeled images.
- Fine-tuning for label-efficient tasks.

## 9. Leakage and shortcuts

| Shortcut | Why it happens | Detection |
|---|---|---|
| Laterality marker | Model learns "L" vs "R" | Crop/erase markers, test |
| Chest tube | Associated with severity | Ablation, saliency review |
| Scanner artifact | Site-specific | Site-held-out validation |
| Report text | Label leakage | Use image-only at inference |
| Patient overlap | Multiple images per patient | Patient-level split |
| Series overlap | Adjacent slices | Study-level split |
| Prevalence | Case-control sampling | Report real-world prevalence |

**Rule:** if a radiologist would not use the feature, test whether the model relies on it.

## 10. Evaluation

### 10.1 Metrics

| Task | Metrics |
|---|---|
| Classification | AUROC, AUPRC, sensitivity/specificity, calibration |
| Detection | Sensitivity at fixed FP/image, mAP, localization accuracy |
| Segmentation | Dice, IoU, Hausdorff, surface distance |
| Triage | Time to review, time to treatment, missed critical findings |
| Workflow | Reading time, alert burden, user trust |

### 10.2 Validation

- Patient-level split.
- Site-held-out.
- Scanner-held-out.
- Temporal validation.
- Prospective silent mode.
- Randomized or stepped-wedge impact study.

### 10.3 Subgroup analysis

- Age, sex, race/ethnicity.
- Body habitus.
- Comorbidities.
- Scanner and protocol.
- Care setting.

## 11. Workflow roles

| Role | Description | Regulatory implication |
|---|---|---|
| Triage | Prioritize worklist | Often lower risk if clinician reviews all |
| Concurrent | Alert during read | May be CDS or device |
| Second read | Independent check | Workflow and liability |
| Autonomous | No clinician review | Highest evidence burden |
| Quantitative | Measure lesion, flow, etc. | Device if clinical claim |

**Industry lesson:** the same model can have different regulatory risk depending on workflow role.

## 12. Industry case studies

### 12.1 IDx-DR

IDx-DR was authorized for autonomous detection of diabetic retinopathy in primary care, with a clear intended use and prospective evidence.

**Lesson:** narrow scope and workflow integration can support autonomy.

Source: FDA De Novo DEN180001. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

### 12.2 CheXpert

CheXpert demonstrated large-scale chest X-ray classification with uncertain labels, but performance varied by finding and population.

**Lesson:** multi-label performance is not one number; report per-finding results.

Source: Irvin, J. et al., *AAAI*, 2019. https://doi.org/10.1609/aaai.v33i01.3301590

### 12.3 MedSAM

MedSAM adapted promptable segmentation to medical images across modalities.

**Lesson:** foundation models reduce annotation burden but need modality-specific validation.

Source: Ma, J. et al., *Nature Communications*, 2024. https://doi.org/10.1038/s41467-024-44824-z

### 12.4 Stroke large-vessel occlusion triage

Several FDA-cleared systems triage CT angiography for large-vessel occlusion, aiming to reduce time to thrombectomy.

**Lesson:** the outcome is time to treatment, not just detection accuracy. Workflow and communication are part of the product.

Source: FDA 510(k) database; peer-reviewed workflow studies.

### 12.5 Mammography AI

Mammography AI has been evaluated in screening programs, with variable results across populations and workflows. Some studies show improved cancer detection; others show increased recalls.

**Lesson:** screening AI must be evaluated on recall rate, interval cancers, and overdiagnosis, not only AUC.

Source: McKinney, S. M. et al., "International evaluation of an AI system for breast cancer screening," *Nature*, 2020. https://doi.org/10.1038/s41586-019-1799-6

## 13. Labs

### Lab 10.1 — DICOM to volume

**Objective:** Load a CT series, apply rescale, and build a 3D volume.

**Steps:**

1. Read series metadata.
2. Sort slices by position.
3. Apply slope/intercept.
4. Resample to isotropic spacing.
5. Visualize in three planes.
6. Document orientation and spacing.

**Deliverable:** volume and preprocessing report.

**Acceptance criteria:** correct orientation; units; no identifier leakage.

### Lab 10.2 — Chest X-ray classifier

**Objective:** Train a multi-label classifier with patient-level splits.

**Steps:**

1. Parse labels from reports or provided annotations.
2. Split by patient.
3. Train CNN and pretrained backbone.
4. Evaluate per-finding AUROC and calibration.
5. Inspect false positives and saliency.

**Deliverable:** model and evaluation.

**Acceptance criteria:** patient-level split; per-finding metrics; error analysis.

### Lab 10.3 — Segmentation

**Objective:** Train a U-Net for a public segmentation task.

**Steps:**

1. Preprocess images and masks.
2. Train with Dice loss.
3. Evaluate Dice and Hausdorff.
4. Inspect failure cases.
5. Test on a held-out site if available.

**Deliverable:** segmentation model.

**Acceptance criteria:** correct mask alignment; boundary errors analyzed.

### Lab 10.4 — Shortcut detection

**Objective:** Detect a planted shortcut (marker, scanner, or metadata).

**Steps:**

1. Train a baseline.
2. Inspect saliency and top features.
3. Erase suspected shortcut.
4. Retrain and compare.
5. Report the impact.

**Deliverable:** shortcut audit.

**Acceptance criteria:** shortcut identified; performance drop quantified.

### Lab 10.5 — Site-held-out validation

**Objective:** Evaluate cross-site performance.

**Steps:**

1. Split by site.
2. Train on sites A and B, test on C.
3. Compare calibration and subgroup performance.
4. Analyze scanner and protocol differences.
5. Propose adaptation or monitoring.

**Deliverable:** external validation report.

**Acceptance criteria:** site differences documented; no pooled-only claim.

### Lab 10.6 — Workflow simulation

**Objective:** Estimate triage impact on reading time.

**Steps:**

1. Model worklist arrival and reading times.
2. Simulate triage at different thresholds.
3. Estimate time to critical finding.
4. Estimate alert burden.
5. Recommend a workflow role.

**Deliverable:** simulation and memo.

**Acceptance criteria:** assumptions explicit; sensitivity analysis.

## 14. Assignment: imaging evidence package

Write a 2,000-word package:

1. Intended use and workflow role.
2. Modality and acquisition.
3. Label definition and annotation.
4. Dataset and splits.
5. Preprocessing.
6. Model and baseline.
7. Performance and calibration.
8. Site and subgroup validation.
9. Shortcut analysis.
10. Regulatory pathway and monitoring.

## 15. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Clinical framing | 20% | Generic model | Clear task and workflow | Regulatory role and outcomes |
| Data quality | 20% | Uses all images | Patient-level split | Site/scanner validation |
| Modeling | 20% | Single architecture | Baselines and tuning | Robustness and efficiency |
| Evaluation | 25% | Dice/AUC only | Task metrics and calibration | Subgroup and workflow impact |
| Governance | 15% | Ignores regulation | Identifies pathway | Evidence package and monitoring |

## 16. Regulatory, ethical, and commercial lens

- **FDA:** most imaging AI is a device. Pathway depends on risk and predicate.
- **CE/MDR:** clinical evaluation and post-market surveillance.
- **Reimbursement:** some imaging AI has CPT codes; many do not. Check payer coverage.
- **Liability:** workflow role affects responsibility. Autonomous use has the highest bar.
- **Equity:** skin tone, body habitus, scanner access, and referral patterns can bias performance.
- **Commercial:** integration with PACS and radiology workflow is a major adoption barrier.

## 17. Failure modes and red flags

- Random split by image.
- Labels extracted from reports without validation.
- No site-held-out test.
- Ignoring scanner metadata.
- Unrealistic augmentation.
- Saliency maps treated as explanations.
- No calibration.
- No workflow evaluation.
- Claiming generalizability from a single academic center.
- No plan for protocol drift.

## 18. Interview and discussion questions

1. **How do you split imaging data?**
   Hint: patient-level, site-held-out, temporal.
2. **What is a shortcut in imaging AI?**
   Hint: marker, tube, scanner artifact, report leakage.
3. **Why is Dice not enough?**
   Hint: boundary errors, clinical relevance, small lesions.
4. **How do you evaluate a triage model?**
   Hint: time to review, time to treatment, missed findings, alert burden.
5. **What metadata matters for imaging?**
   Hint: scanner, protocol, slice thickness, contrast, kernel.
6. **How do you handle uncertain labels?**
   Hint: uncertainty categories, multiple readers, adjudication.
7. **What is the difference between detection and segmentation?**
   Hint: localization vs pixel-level boundary.
8. **How would you validate a mammography model?**
   Hint: screening population, recall rate, interval cancers, subgroup.
9. **What is a foundation model in imaging?**
   Hint: pretrained representation, promptable segmentation, few-shot.
10. **How do you monitor scanner drift?**
    Hint: metadata, image statistics, calibration, subgroup performance.
11. **What workflow role has the highest regulatory bar?**
    Hint: autonomous diagnosis without clinician review.
12. **How do you explain a false positive to a radiologist?**
    Hint: show image region, model uncertainty, similar cases, and known limitations.

## 19. Key terms

| Term | Definition |
|---|---|
| DICOM | Imaging data and metadata standard |
| PACS | Picture archiving and communication system |
| HU | Hounsfield units for CT |
| Window/level | Display mapping of intensity |
| Dice | Segmentation overlap metric |
| Hausdorff distance | Boundary distance metric |
| Triage | Prioritize worklist |
| Concurrent read | AI alert during interpretation |
| Second read | Independent AI check |
| Autonomous | AI without clinician review |
| Shortcut | Spurious feature correlated with label |
| Site-held-out | Validation on unseen institution |

## 20. Sources and further reading

### Datasets and tools
- MONAI: https://monai.io/
- pydicom: https://pydicom.github.io/
- 3D Slicer: https://www.slicer.org/
- MedSAM: https://github.com/bowang-lab/MedSAM
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC and https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
- Medical Segmentation Decathlon: http://medicaldecathlon.com/

### Methods and evidence
- Irvin, J. et al., "CheXpert," *AAAI*, 2019. https://doi.org/10.1609/aaai.v33i01.3301590
- Ma, J. et al., "Segment anything in medical images," *Nature Communications*, 2024. https://doi.org/10.1038/s41467-024-44824-z
- McKinney, S. M. et al., "International evaluation of an AI system for breast cancer screening," *Nature*, 2020. https://doi.org/10.1038/s41586-019-1799-6
- FDA De Novo DEN180001 (IDx-DR): https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
- FDA 510(k) database: https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm

### Guidance
- FDA, *AI/ML SaMD Action Plan*. https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
- DICOM Standard: https://www.dicomstandard.org/
- Royal College of Radiologists, AI guidance. https://www.rcr.ac.uk/

> **Next:** [Module 11](11_physiological_signals_wearables.md) extends the signal-processing mindset to waveforms and wearables.
