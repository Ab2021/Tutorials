# Preprocessing

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01 and 02
> Version 2026.3

## 1. Why this topic exists

Preprocessing is where a model's input distribution is defined. It is also where most silent
failures are introduced. A training pipeline that resamples to 1 mm isotropic spacing and an
inference pipeline that resamples to 0.8 mm produce different inputs from the same study. A
training pipeline that normalizes computed tomography (CT) with a fixed Hounsfield unit (HU)
window and an inference pipeline that normalizes with per-image percentiles produces a
distribution shift that no amount of model capacity can absorb. A mask that is resampled with
bilinear interpolation acquires fractional labels and a boundary that no longer matches the
image.

The clinical consequence is direct. Resampling changes apparent lesion size. Cropping can remove
the pathology. Intensity normalization can erase the contrast between a lesion and its
background. Augmentation can create anatomically impossible images that teach the model the wrong
invariances. Registration can distort the anatomy it was meant to align.

The industry problem is that preprocessing is usually written once, in a notebook, and never
versioned. The training run records the model architecture and the learning rate; it does not
record the interpolation order, the padding mode, the clip range, or the library version. When
performance falls after a deployment, the team cannot tell whether the model changed, the data
changed, or the preprocessing changed, because two of the three were never recorded.

This topic treats preprocessing as a specification with a domain, a codomain, and a version. Each
transform must state what it preserves, what it changes, and how it is tested. The goal is not a
particular pipeline. The goal is a pipeline that can be reproduced, audited, and compared across
sites.

### 1.1 The decisions this topic unblocks

| Decision | What preprocessing knowledge provides |
|---|---|
| Target voxel spacing | Whether isotropic resampling is worth the interpolation cost |
| Interpolation method | Which objects may be interpolated and which must be nearest |
| Orientation convention | How to compare voxel coordinates across scanners |
| Intensity normalization | Which statistics are stable across sites |
| Cropping and padding | How much context the model needs and loses |
| Augmentation policy | Which transformations are anatomically plausible |
| Patch sampling | How to avoid sampling only lesions or only background |
| Train/inference parity | How to prove the same transform ran in both |

### 1.2 What preprocessing is not

Preprocessing is not a substitute for data quality. Resampling cannot recover slices that were
never acquired. Normalization cannot recover contrast that the acquisition did not produce.
Augmentation cannot create a site that was not in the training set. A pipeline that hides missing
data behind interpolation has made the failure harder to detect, not smaller.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** a target grid from source spacing, extent, and target spacing, and state the
   resulting shape and affine.
2. **Choose** an interpolation method for images, masks, and label maps, and justify the choice
   by what the object represents.
3. **Reorient** a volume to a canonical anatomical convention and verify the transform with a
   landmark.
4. **Normalize** CT, magnetic resonance imaging (MRI), positron emission tomography (PET), and
   pathology inputs with modality-appropriate statistics, and state the unit of the output.
5. **Detect** resampling and normalization errors from geometry checks, label histograms, and
   intensity distributions.
6. **Design** an augmentation policy that preserves diagnostic content and does not create
   implausible anatomy.
7. **Version** a preprocessing pipeline so that training and inference can be proven identical.

## 3. The core mental model

Preprocessing is a chain of coordinate transforms and value transforms. Coordinate transforms
move voxels in space; value transforms change the numbers at each voxel. The two are independent
and must be versioned separately.

```text
Source volume                 Coordinate chain                 Target volume
(array, affine, spacing)      (reorient, resample, crop)       (array, affine, spacing)
        |                              |                              |
        v                              v                              v
   Value chain  -------------->  clip -> normalize -> augment  ->  model input
   (unit, range, statistics)      (per modality)     (train only)
```

### 3.1 The affine is the contract

A volume is not an array. It is an array plus an affine matrix that maps voxel indices to patient
coordinates. Two arrays with the same shape but different affines are different objects. A
resampling operation is correct only if the output affine describes the new grid.

```text
source index (i, j, k) --A_src--> patient (x, y, z) --A_tgt^-1--> target index (i', j', k')
```

The resampling engine evaluates the source image at the patient coordinate corresponding to each
target voxel. If the affine is wrong, the image content is wrong even when the array shape is
right. This is why a mask can look aligned in one viewer and misaligned in another: the viewer
uses the affine, and the pipeline ignored it.

### 3.2 Interpolation changes the object

Interpolation estimates values between samples. It does not create information. For an image, a
smooth interpolation is a reasonable model of the continuous signal. For a binary mask, the
continuous signal is not meaningful: a value of 0.5 is not "half a lesion". Nearest-neighbour
interpolation preserves the label set; linear interpolation creates new labels that must be
thresholded, and the threshold changes the volume.

| Object | Values | Correct interpolation | Why |
|---|---|---|---|
| CT or MRI image | Continuous intensity | Linear, cubic, or B-spline | Approximates the continuous signal |
| Binary mask | 0 or 1 | Nearest | Preserves label set and boundary count |
| Multi-class label map | Integer labels | Nearest | Linear creates nonexistent classes |
| Probability map | 0 to 1 | Linear | A probability is continuous |
| PET SUV map | Continuous | Linear | Approximates the continuous uptake |
| Pathology RGB tile | Colour | Area or Lanczos | Avoid colour fringing |
| Displacement field | Vector | Linear | Vector field is continuous |

### 3.3 Resolution and the Nyquist limit

A sampled image can represent detail only up to its Nyquist frequency, one cycle per two samples.
For a CT with 0.7 mm pixels, the highest representable spatial frequency is about 0.7 cycles per
mm, corresponding to a period of about 1.4 mm. Upsampling to 0.35 mm doubles the array size but
cannot create detail beyond 1.4 mm. Downsampling below the original spacing discards detail, and
without an anti-aliasing filter it also aliases high-frequency noise into low-frequency texture.

The practical rule: resample to a spacing that is justified by the acquisition, not to a spacing
that looks impressive. For most CT tasks, 1 mm isotropic is a defensible target because it is
close to the through-plane resolution of thin-section scans and does not oversample the in-plane
data by more than about 1.5x.

### 3.4 Value normalization is a model choice

There is no universal normalization. CT has a calibrated scale, so a fixed window or a fixed clip
is reproducible. MRI has no absolute scale, so per-volume statistics are common, but they make the
input depend on the field of view and on the presence of pathology. PET has a defined unit but a
protocol-dependent scale. Pathology has colour and stain variation that requires colour
deconvolution or stain normalization.

The normalization must be chosen with the deployment distribution in mind. A per-image z-score
makes every image have mean 0 and standard deviation 1, which removes global intensity
differences but also removes the information that a whole image is unusually bright. A fixed CT
window preserves absolute HU but saturates out-of-range tissue. Neither is universally better;
each is a statement about what the model should be invariant to.

### 3.5 Augmentation is a prior

Every augmentation encodes a belief about which transformations should not change the label. A
horizontal flip encodes the belief that left and right are interchangeable. For a laterality
task, that belief is false. A brightness shift encodes the belief that acquisition exposure is
irrelevant. For a model that detects calcification, that belief may be false. An elastic
deformation encodes the belief that tissue can deform arbitrarily. For a fracture, that belief is
false.

Augmentation should be justified by the acquisition variation the model will see, not by the
generic list in a library. The test is simple: would a radiologist accept the augmented image as a
plausible acquisition of the same patient with the same label? If not, the augmentation is
teaching a false invariance.

### 3.6 The pipeline is a specification, not a script

A script runs; a specification can be reviewed. The difference is that a specification names every
parameter that affects the output, including the defaults that were not changed. The parameters
fall into four groups.

| Group | Parameters | Why recorded |
|---|---|---|
| Geometry | Source spacing, target spacing, shape, affine, origin convention | Reproduces the grid |
| Interpolation | Method, order, anti-alias filter, padding mode | Reproduces the values |
| Intensity | Unit, clip range, statistic source, mask for statistics | Reproduces the scale |
| Sampling | Crop box, patch size, foreground ratio, random seed | Reproduces the examples |

A specification that lists only the target spacing is incomplete. Two pipelines with the same
target spacing but different padding modes produce different arrays at the border, and a model
can learn the border. Two pipelines with the same normalization formula but different statistic
masks produce different scales for the same patient.

The specification should be stored with the model artifact, not in a separate wiki. When the model
is deployed, the inference service must load the same specification and apply it in the same
order. A specification that exists only in a training notebook is not a specification.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Affine | 4×4 matrix mapping voxel indices to patient coordinates | Treating as metadata only | mm and direction cosines |
| Voxel spacing | Physical size of a voxel along each axis | Assuming isotropic | mm |
| Field of view | Physical extent of the image | Confusing with matrix size | mm |
| Resampling | Evaluating an image on a new grid | Calling any resize resampling | Operation |
| Interpolation | Estimating values between samples | Using linear for masks | Method |
| Nearest neighbour | Assign the value of the closest sample | Using for continuous images | Method |
| Trilinear | Linear interpolation in three axes | Using for label maps | Method |
| B-spline | Smooth spline interpolation | Assuming it preserves labels | Method |
| Anti-aliasing | Low-pass filtering before downsampling | Skipping when reducing resolution | Filter |
| Nyquist limit | Highest representable frequency | Ignoring when upsampling | cycles/mm |
| Orientation | Mapping of voxel axes to anatomical axes | Assuming axial | Code, e.g. RAS, LPS |
| RAS+ | Right, anterior, superior positive convention | Confusing with LPS | NIfTI convention |
| LPS | Left, posterior, superior positive convention | Confusing with RAS | DICOM convention |
| Reorientation | Permuting/flipping axes to a convention | Resampling unnecessarily | Operation |
| Crop | Removing voxels outside a region | Cropping pathology | Operation |
| Pad | Adding voxels outside the image | Padding with non-background values | Operation |
| Foreground crop | Crop to nonzero or body region | Cropping to the lesion only | Operation |
| Intensity normalization | Mapping values to a standard range | Using site-specific statistics | Operation |
| Z-score | Subtract mean, divide by standard deviation | Computing per slice | Statistic |
| Percentile normalization | Clip at percentiles and scale | Using 0.5 and 99.5 blindly | Statistic |
| Windowing | Map an HU range to a display range | Treating as normalization | HU |
| Clipping | Limit values to a range | Clipping before rescale | Operation |
| Bias field | Smooth intensity inhomogeneity in MRI | Treating as pathology | Correction |
| N4 | Non-parametric non-uniform intensity normalization | Assuming it improves all tasks | Algorithm |
| Histogram equalization | Remap intensities to a uniform histogram | Using in medical images without validation | Operation |
| Stain normalization | Map pathology colours to a reference | Assuming it preserves biology | Operation |
| Registration | Align two images spatially | Assuming it preserves anatomy | Transform |
| Rigid registration | Translation and rotation only | Using for deformable tissue | Transform |
| Deformable registration | Non-linear spatial mapping | Using without validation | Transform |
| Augmentation | Synthetic variation during training | Using at inference | Operation |
| Elastic deformation | Smooth random displacement | Using on fractures or spines | Operation |
| MixUp | Linear combination of images and labels | Using across modalities | Operation |
| Patch sampling | Extracting sub-volumes for training | Sampling only lesions | Operation |
| Train/inference parity | Identical transforms in both paths | Reimplementing at inference | Requirement |
| MONOCHROME1 | Inverted greyscale convention | Ignoring inversion | Code string |
| Rescale slope/intercept | Map stored values to HU | Applying after normalization | HU |

### 4.1 Coordinate conventions

| Convention | Axis order | Positive directions | Used by |
|---|---|---|---|
| DICOM patient | x, y, z | Left, posterior, superior | DICOM |
| NIfTI | i, j, k | Right, anterior, superior | NIfTI, many research tools |
| ITK | i, j, k | LPS by default | ITK, MONAI with LPS metadata |
| Array index | k, j, i | Depends on storage | NumPy, PyTorch |

A volume read from DICOM into an array usually has axes ordered (slice, row, column), which
corresponds to (z, y, x), not (x, y, z). A reorientation step makes the axis order explicit and
prevents a left-right flip from being mistaken for a model failure.

### 4.2 What each normalization preserves

| Normalization | Preserves | Removes | Appropriate for |
|---|---|---|---|
| Fixed HU clip [-1000, 1000] | Absolute attenuation | Values outside range | CT |
| Lung window [-1350, 150] | Lung contrast | Soft tissue above 150 HU | CT lung tasks |
| Soft-tissue window [-160, 240] | Soft tissue contrast | Bone and air | CT abdomen |
| Per-volume z-score | Relative contrast | Absolute intensity | MRI |
| Per-volume percentile [1, 99] | Robust contrast | Outliers | MRI, PET |
| Liver-relative SUV | Uptake relative to liver | Global scale | PET response |
| Stain normalization | Tissue colour structure | Batch colour shift | Pathology |
| Optical density | Absorbance | Illumination | Pathology |

### 4.3 Parameters that must appear in the manifest

| Parameter | Example | Consequence if omitted |
|---|---|---|
| Source spacing | 0.703125 × 0.703125 × 1.25 mm | Grid cannot be reproduced |
| Target spacing | 1.0 × 1.0 × 1.0 mm | Grid cannot be reproduced |
| Target shape | 224 × 360 × 360 | Array shape ambiguous |
| Affine origin | -140, -150, -112.5 mm | Position in patient space lost |
| Axis convention | LPS or RAS | Left-right ambiguity |
| Interpolation | Trilinear, order 1 | Values not reproducible |
| Mask interpolation | Nearest | Label set not reproducible |
| Anti-alias filter | Gaussian, sigma in mm | Texture not reproducible |
| Padding mode | Constant -1000 HU | Border values differ |
| Unit | HU | Scale ambiguous |
| Clip range | -1000 to 400 HU | Scale ambiguous |
| Statistic source | Fixed, per volume, per site | Normalization ambiguous |
| Statistic mask | Body mask, brain mask, none | Statistics differ |
| Augmentation list | Rotation ±10°, no flip | Training distribution differs |
| Patch size | 96 × 96 × 96 voxels | Sampling differs |
| Foreground ratio | 0.5 | Prevalence differs |
| Random seed | 20261025 | Sampling not reproducible |
| Library versions | MONAI 1.4, PyTorch 2.x | Behavior differs |

## 5. The mechanics, step by step

This procedure turns a validated series into a model-ready tensor with a documented geometry and
intensity contract. It assumes the series passed the checks in Topic 02. Each step states input,
output, rationale, branches, and the edge cases that change the decision.

### Step 1. Declare the target grid

**Input:** Source spacing, source extent, and the model's resolution requirement.

**Output:** Target spacing, target shape, target affine, and a statement of whether the target is
isotropic.

**Rationale:** The target grid is a design decision, not a library default. It determines memory,
compute, and the smallest lesion the model can resolve. A target spacing finer than the source
adds interpolated voxels without adding information; a target spacing coarser than the source
discards detail.

**Branch points:**
- Target spacing equal to source in-plane and through-plane: no resampling, only reorientation.
- Target isotropic: resample all axes.
- Target anisotropic: resample in-plane only, keep through-plane.
- Target spacing finer than source: reject unless a downstream model requires a fixed shape.

**Edge case:** The source extent may not be an integer multiple of the target spacing. The target
shape is `round(extent / spacing)`, and the residual is absorbed into the affine origin or into a
half-voxel shift. Record which convention is used.

### Step 2. Load with geometry and validate

**Input:** The image array, the affine, and the spacing.

**Output:** An array with a known axis order, a validated affine, and a spacing vector.

**Rationale:** A resampling error is often an axis-order error. Loading a DICOM series as
(slice, row, column) and treating it as (x, y, z) transposes the volume. Validating the affine
before any transform catches the mistake at the source.

**Branch points:**
- Affine determinant positive and directions orthonormal: proceed.
- Determinant negative: flag a possible mirror before reorienting.
- Spacing inconsistent with affine: recompute spacing from the affine and flag.

**Edge case:** Some NIfTI files have a valid `sform` and a stale `qform`. Choose one convention,
record it, and do not let different tools choose differently.

### Step 3. Reorient to a canonical convention

**Input:** The loaded volume and a target axis code, for example RAS (right, anterior, superior).

**Output:** A reoriented array and an updated affine, with no interpolation.

**Rationale:** Reorientation is a permutation and sign flip, not a resampling. It is lossless and
cheap. Doing it first makes every later spatial operation predictable and makes left-right errors
visible.

**Branch points:**
- Already canonical: no change.
- Permutation only: transpose axes.
- Flip required: reverse the axis and adjust the affine.
- Oblique acquisition: reorientation alone is insufficient; a resampling step is required to
  align with the canonical axes.

**Edge case:** A left-right flip can be a true acquisition convention or a bug. Verify with an
anatomical landmark, such as the liver on the right or the heart on the left, before accepting the
flip.

### Step 4. Resample the image with an anti-aliasing filter

**Input:** The reoriented image, source spacing, target spacing, and interpolation order.

**Output:** The resampled image, the target affine, and the interpolation record.

**Rationale:** Resampling evaluates the source at target coordinates. When downsampling, a low-pass
filter must be applied first; otherwise high-frequency noise aliases into the target. When
upsampling, interpolation smooths but does not add detail.

**Branch points:**
- Downsampling factor >1.5: apply an anti-aliasing filter before interpolation.
- Upsampling: use linear or cubic; do not use nearest for continuous images.
- Isotropic target from anisotropic source: the through-plane interpolation is the dominant
  smoothing; record it.

**Edge case:** A CT with 0.7 mm in-plane and 1.25 mm through-plane resampled to 1 mm is
downsampled in-plane (factor 0.7) and upsampled through-plane (factor 1.25). The same kernel is
not appropriate for both axes; use a physical-unit-aware resampler.

### Step 5. Resample masks and label maps with nearest neighbour

**Input:** The mask or label map in source geometry.

**Output:** A mask in target geometry with the original label set.

**Rationale:** A mask is a set of voxels, not a continuous field. Linear interpolation creates
fractional values that are not labels. Thresholding restores a binary mask but changes the
boundary and the volume in a way that depends on the threshold.

**Branch points:**
- Binary mask: nearest, then optionally majority vote.
- Multi-class label map: nearest, then check that no new labels appear.
- Probability map: linear is acceptable; threshold only at the metric stage.

**Edge case:** A one-voxel-thick structure, such as a vessel or a fracture line, can disappear
under nearest-neighbour resampling when the target grid misses it. Check the volume before and
after; if the structure is clinically important, resample at a finer target or use a
shape-based method.

### Step 6. Crop and pad with a documented policy

**Input:** The resampled image and a region definition: body, foreground, or a fixed box.

**Output:** A cropped or padded volume with an updated affine and a record of the removed extent.

**Rationale:** Cropping reduces memory and removes irrelevant anatomy. It also risks removing the
pathology or the context the model needs. Padding makes shapes uniform but can introduce border
artefacts that correlate with the crop policy.

**Branch points:**
- Body crop: threshold and largest connected component; verify no pathology outside.
- Foreground crop: nonzero bounding box; verify the mask includes the lesion.
- Fixed crop: same box for all cases; verify coverage across the cohort.
- Pad: use the modality's background value, not zero, for CT (air is -1000 HU).

**Edge case:** A lesion at the edge of the field of view may be partially cut off. A crop that
removes the edge removes the evidence. Record the distance from the lesion to the crop boundary.

### Step 7. Convert to the physical unit

**Input:** Stored values, rescale slope/intercept, and modality.

**Output:** An array in HU, SUV, or a documented arbitrary unit.

**Rationale:** Normalization statistics are meaningless without a unit. A CT z-score computed on
stored values and a CT z-score computed on HU are different transforms. Convert first, then
normalize.

**Branch points:**
- CT: apply slope and intercept; clip to a stated HU range.
- PET: convert to SUV or SUL with a stated normalization; do not mix.
- MRI: no conversion; document that the unit is arbitrary.
- Pathology: convert RGB to optical density if using colour deconvolution.

**Edge case:** A CT with a slope other than 1 (for example, a scaled export) must be converted
before clipping. Clipping stored values to [-1000, 1000] before rescale produces a different
range.

### Step 8. Normalize intensities with modality-specific statistics

**Input:** The unit-converted array and a normalization specification.

**Output:** A normalized array and the statistics used.

**Rationale:** The model needs a stable input scale. The statistics must be computed from a
defined population: per image, per volume, per site, or from a fixed reference. Each choice has a
different invariance.

**Branch points:**
- CT: fixed clip and scale, or fixed window; do not use per-image z-score unless validated.
- MRI: per-volume z-score over the brain mask, or percentile clipping.
- PET: liver-relative scaling for response tasks; fixed SUV clip for detection.
- Pathology: stain normalization to a reference slide.

**Edge case:** Per-volume z-score computed over the whole field of view includes air, bone, and
table. The mean and standard deviation are dominated by non-tissue. Use a body or brain mask, or
use percentiles.

### Step 9. Apply augmentation only in training

**Input:** The normalized volume and an augmentation policy.

**Output:** An augmented volume and label, with the transform recorded.

**Rationale:** Augmentation increases effective sample size and teaches invariances. It must not
run at inference, and it must not create implausible anatomy or change the label.

**Branch points:**
- Geometric: rotation, translation, scale, flip. Check laterality tasks before flipping.
- Intensity: brightness, contrast, gamma, noise. Keep within acquisition plausibility.
- Elastic: only for soft tissue, never for fractures or spine.
- MixUp: only within modality and task; not for detection masks without care.

**Edge case:** Augmentation that changes the apparent lesion size changes a size-based label. If
the label is "nodule ≥6 mm", a random zoom can move a case across the threshold. Either exclude
size-sensitive labels from geometric augmentation or recompute the label after augmentation.

### Step 10. Sample patches with attention to prevalence

**Input:** The volume and a patch size.

**Output:** Patches for training or inference, with a sampling record.

**Rationale:** Whole volumes are often too large for memory. Patch sampling determines what the
model sees. Sampling only lesion-centred patches creates a prevalence far from deployment and
teaches the model that every patch contains a lesion.

**Branch points:**
- Lesion-centred: high sensitivity, high false-positive rate at inference.
- Random: realistic prevalence, slower convergence.
- Mixed: a stated ratio, for example 1:1 or 1:3.
- Hard-negative mining: improves specificity but risks overfitting to the mining set.

**Edge case:** A patch that crosses the volume boundary needs padding. The padding value must be
the modality background, and the loss must ignore padded voxels.

### Step 11. Register only when the task requires it

**Input:** Two images, a transform model, and a metric.

**Output:** A registered image and a transform, with a validation record.

**Rationale:** Registration aligns images across time or modality. It can also distort anatomy and
create false correspondence. A rigid transform is appropriate for bone and brain; deformable
registration is appropriate for soft tissue but must be validated.

**Branch points:**
- Same-session multimodal: rigid or affine.
- Longitudinal: rigid plus deformable, with landmark checks.
- Atlas: deformable, with Jacobian determinant checks for folding.

**Edge case:** Registration to a template can change lesion shape and volume. If the endpoint is
lesion size, do not report the registered volume without a correction.

### Step 12. Save with geometry and provenance

**Input:** The processed array, affine, and transform record.

**Output:** A file with the array, affine, spacing, and a provenance record.

**Rationale:** A processed volume without its affine is not reusable. A file without provenance
cannot be reproduced. The output format must preserve both.

**Branch points:**
- NIfTI: write `sform` and `qform`, set the units, and record the source.
- NumPy or HDF5: store the affine and spacing as attributes.
- DICOM secondary capture: preserve the frame of reference and document the derivation.

**Edge case:** Some formats default to RAS while the pipeline used LPS. Record the convention in
the file and in the manifest.

### Step 13. Verify train/inference parity

**Input:** The training transform and the inference transform.

**Output:** A test that the same input produces the same output through both paths.

**Rationale:** The most common deployment failure is a preprocessing difference, not a model
difference. The inference path is often reimplemented in a different language or library.

**Branch points:**
- Same code path: hash the output for a fixed input.
- Different code path: compare arrays within a tolerance and compare the affine exactly.
- Different library version: run a golden-file test on a fixed sample.

**Edge case:** A library upgrade can change interpolation defaults. Pin the version and include
the version in the model artifact.

### Step 14. Version the specification

**Input:** All parameters and code.

**Output:** A version identifier, a parameter hash, and a change log.

**Rationale:** A preprocessing specification without a version cannot be cited in a validation
report or a regulatory submission. The version must change when any parameter changes.

**Branch points:**
- Parameter change: new version and re-evaluation.
- Code change without parameter change: new version and regression test.
- Library security patch: new version and golden-file test.

**Edge case:** A change that affects only one modality still requires a new version, because the
model artifact now has two possible input distributions.

### 5.1 Decision table: interpolation by object and direction

| Object | Upsampling | Downsampling | Never use |
|---|---|---|---|
| CT image | Linear or cubic | Linear with anti-alias | Nearest |
| MRI image | Linear or cubic | Linear with anti-alias | Nearest |
| PET SUV | Linear | Linear with anti-alias | Nearest |
| Binary mask | Nearest | Nearest plus volume check | Linear without threshold |
| Multi-class labels | Nearest | Nearest plus label check | Linear |
| Probability map | Linear | Linear | Nearest |
| RGB pathology | Bilinear or area | Area or Lanczos | Nearest for display |
| Displacement field | Linear | Linear | Nearest |

### 5.2 Decision table: normalization by modality and task

| Modality and task | Normalization | Statistic source | Failure if wrong |
|---|---|---|---|
| CT lung nodule | Clip [-1000, 400], scale | Fixed | Soft tissue contrast lost |
| CT lung | Window [-1350, 150] | Fixed | Nodule saturation |
| CT abdomen | Window [-160, 240] | Fixed | Bone and air clipped |
| CT bone | Window [-500, 1500] | Fixed | Marrow contrast lost |
| MRI brain | Z-score over brain mask | Per volume | Scalp and air dominate |
| MRI prostate | Percentile [1, 99] | Per volume | DWI b-value confounding |
| PET response | SUL, liver-relative | Per study | Global scale drift |
| PET detection | Clip [0, 15] SUV | Fixed | Uptake outliers dominate |
| Pathology H&E | Stain normalization | Reference slide | Batch colour shift |
| Mammography | Percentile [0.5, 99.5] | Per image | Dense tissue clipped |

### 5.3 Edge cases that require a written rule

**Mixed slice thickness.** Resample each sub-series separately, then stack only if the target grid
is identical. Never interpolate across a gap.

**Missing series.** Do not synthesize the missing phase or sequence. Exclude or use a model that
was trained with missing inputs and a missingness indicator.

**Inconsistent spacing.** Resample from the position-derived affine, not from the nominal
thickness.

**Rescale absent.** Do not normalize stored values as if they were HU. Exclude or convert with a
documented vendor constant.

**Burned-in annotation.** Crop or inpaint only after confirming the annotation is not diagnostic.
Record the region removed.

**Laterality mismatch.** Resolve before augmentation. A flip augmentation on a case with a
laterality label creates a wrong label.

**Multi-frame objects.** Resample the frame stack as a volume only if the frames are spatial.
Temporal frames require a different axis convention.

**Private tags carrying labels.** Strip them before preprocessing; a transform that reads a
private tag is a leak path.

## 6. Worked example

### 6.1 The case

A lung cancer screening CT is acquired at 512 × 512 pixels, `PixelSpacing` 0.703125 mm,
`SliceThickness` 1.25 mm, 180 slices, positions from -112.5 to 111.25 mm along the normal. The
lesion measures 9.4 mm × 8.1 mm × 7.6 mm. The pipeline target is 1 mm isotropic, lung window
centre -600 HU and width 1500 HU, with a fixed clip at [-1000, 400] HU for the model input.

### 6.2 Target grid

The field of view is:

```text
512 × 0.703125 = 360.0 mm
```

The through-plane extent is:

```text
111.25 - (-112.5) = 223.75 mm
```

At 1 mm isotropic spacing:

```text
in-plane shape = 360 / 1 = 360
through-plane shape = round(223.75 / 1) = 224
target shape = 360 × 360 × 224 = 29,030,400 voxels
```

The source has 512 × 512 × 180 = 47,185,920 voxels. The target has 38.5% fewer voxels. The target
voxel volume is 1 mm³, and the lesion volume of 302.99 mm³ occupies about 303 voxels.

### 6.3 Interpolation arithmetic

Consider a target voxel whose centre maps to source index (50.2, 100.3, 200.7) in (z, y, x) order.
Trilinear interpolation uses the eight surrounding source voxels. The fractional weights are:

```text
wz0 = 1 - 0.2 = 0.8,  wz1 = 0.2
wy0 = 1 - 0.3 = 0.7,  wy1 = 0.3
wx0 = 1 - 0.7 = 0.3,  wx1 = 0.7
```

The interpolated value is the weighted sum of the eight values with weights
`wz × wy × wx`. For a synthetic field `f(z, y, x) = 100z + 10y + x`, the value at the corners is:

```text
f(50,100,200) = 5000 + 1000 + 200 = 6200
f(50,100,201) = 6201
f(50,101,200) = 6210
f(50,101,201) = 6211
f(51,100,200) = 6300
f(51,100,201) = 6301
f(51,101,200) = 6310
f(51,101,201) = 6311
```

Weighted sum:

```text
0.8×0.7×0.3×6200 = 1041.6
0.8×0.7×0.7×6201 = 2430.8
0.8×0.3×0.3×6210 = 447.1
0.8×0.3×0.7×6211 = 1043.4
0.2×0.7×0.3×6300 = 264.6
0.2×0.7×0.7×6301 = 617.5
0.2×0.3×0.3×6310 = 113.6
0.2×0.3×0.7×6311 = 265.1
total = 6223.7
```

The exact linear field at (50.2, 100.3, 200.7) is `100×50.2 + 10×100.3 + 200.7 = 6223.7`.
Trilinear interpolation reproduces a linear field exactly; it does not reproduce curved anatomy
exactly.

### 6.4 Mask resampling

The reference mask has 303 voxels at 1 mm. Resampling with nearest neighbour preserves the label
set. Resampling with linear interpolation produces fractional values; at a boundary voxel with
fraction 0.3, the value is 0.3. Thresholding at 0.5 keeps the voxel; thresholding at 0.4 removes
it. The volume changes by one voxel, which is 0.33% of a 303-voxel lesion but 10% of a 10-voxel
lesion.

After nearest-neighbour resampling, the predicted mask has 295 voxels and the intersection with
the reference is 265 voxels:

```text
Dice = 2 × 265 / (303 + 295)
     = 530 / 598
     = 0.886
```

The pre-resampling Dice was 0.899 with 298 predicted voxels and 270 intersection. The difference
is within the boundary uncertainty of a small lesion; it is not evidence that resampling improved
the model.

### 6.5 Intensity transformation

The lesion mean is 38 HU with a standard deviation of 22 HU. The lung window maps 38 HU to:

```text
(38 - (-1350)) / 1500 = 0.9253
```

The model clip is [-1000, 400] HU. The clipped value is 38 HU, and the scaled value is:

```text
(38 - (-1000)) / (400 - (-1000)) = 1038 / 1400 = 0.7414
```

The same lesion is 0.925 on the display window and 0.741 on the model clip. Both are valid; they
are different inputs. A model trained on one and deployed on the other has a distribution shift.

If the pipeline uses a per-volume z-score with mean 40 HU and standard deviation 60 HU:

```text
z = (38 - 40) / 60 = -0.033
```

The z-score removes the absolute HU information. A lesion at 38 HU and a lesion at 80 HU can have
similar z-scores if the volume statistics differ. For CT, the absolute scale is usually worth
preserving.

### 6.6 Diagnostics that reveal a broken pipeline

**Diagnostic 1: mask-image misalignment after resampling.** The image is resampled with the target
affine, but the mask is resampled with the source affine unchanged. In a viewer that honours the
affine, the mask appears shifted by half the difference in voxel size. The quantitative test is
the centroid distance between the image foreground and the mask: a valid pipeline keeps the
centroid within one target voxel; a broken one shows a shift of 3-8 mm. In this case, the mask
centroid moves 4.2 mm anteriorly, which is larger than the 1 mm target voxel and cannot be
explained by boundary noise.

**Diagnostic 2: label histogram after interpolation.** A binary mask resampled with linear
interpolation has values other than 0 and 1. The histogram shows 37 boundary voxels with values
between 0.1 and 0.9. A valid nearest-neighbour pipeline has exactly two values. The presence of
intermediate values is a definitive sign that the wrong interpolation was used, even if the
thresholded Dice looks acceptable.

### 6.7 Varying one assumption

Assume the pipeline downsamples to 2.5 mm isotropic without an anti-aliasing filter. The target
shape becomes 144 × 144 × 90 = 1,866,240 voxels, 93.6% fewer than the 1 mm target. The lesion's
9.4 mm long axis spans about 3.8 target voxels. Partial volume and aliasing reduce the measured
volume from 302.99 mm³ to 271.0 mm³, a 10.6% underestimate. The Dice against the 1 mm reference
falls to 0.79.

The conclusion changes from "the model sees a 303-voxel lesion with Dice 0.89" to "the model sees
a 43-voxel lesion with Dice 0.79". The loss is not caused by the model; it is caused by the
sampling. The correct response is either to keep 1 mm spacing or to train and validate at 2.5 mm
with the same anti-aliasing and partial-volume model at both ends.

### 6.8 What the pipeline records for this case

The manifest row for the preprocessed volume contains the source series identifier, the target
grid, the interpolation, the intensity transform, and the output hash. The hash is computed over
the array bytes and the affine, so a change in either is detectable.

| Field | Value |
|---|---|
| Source series | remapped SeriesInstanceUID |
| Source shape | 180 × 512 × 512 |
| Source spacing | 1.25 × 0.703125 × 0.703125 mm |
| Target shape | 224 × 360 × 360 |
| Target spacing | 1.0 × 1.0 × 1.0 mm |
| Axis convention | LPS |
| Image interpolation | Trilinear |
| Mask interpolation | Nearest |
| Anti-alias | Gaussian, sigma 0.35 mm in-plane |
| Unit | HU |
| Clip | -1000 to 400 HU |
| Output scale | [0, 1] |
| Output hash | SHA-256 of array and affine |

The row is evidence, not decoration. If a deployment produces different output for the same
source series, the hash differs and the pipeline is not the same. The comparison is cheap and
catches the class of failure that visual review misses.

## 7. Code, query, or calculation

The following code uses only the standard library. It computes the target grid, performs
trilinear interpolation at one point, compares nearest and linear mask resampling, and computes
Dice. It does not read files; the arithmetic is the artefact.

```python
import math

# Source geometry: (z, y, x) order
src_shape = (180, 512, 512)
src_spacing = (1.25, 0.703125, 0.703125)
src_extent = (223.75, 360.0, 360.0)  # position span, not nominal thickness
target_spacing = (1.0, 1.0, 1.0)
target_shape = tuple(round(e / s) for e, s in zip(src_extent, target_spacing))

print("source extent      =", tuple(round(e, 3) for e in src_extent), "mm")
print("target shape       =", target_shape)
print("source voxels      =", src_shape[0] * src_shape[1] * src_shape[2])
print("target voxels      =", target_shape[0] * target_shape[1] * target_shape[2])
print("voxel change       = {:.1%}".format(
    1 - (target_shape[0] * target_shape[1] * target_shape[2]) /
        (src_shape[0] * src_shape[1] * src_shape[2])))

# Trilinear interpolation at source index (50.2, 100.3, 200.7)
z, y, x = 50.2, 100.3, 200.7
z0, y0, x0 = math.floor(z), math.floor(y), math.floor(x)
dz, dy, dx = z - z0, y - y0, x - x0

def f(k, j, i):
    return 100 * k + 10 * j + i

value = 0.0
for dk, wk in ((0, 1 - dz), (1, dz)):
    for dj, wj in ((0, 1 - dy), (1, dy)):
        for di, wi in ((0, 1 - dx), (1, dx)):
            value += wk * wj * wi * f(z0 + dk, y0 + dj, x0 + di)

exact = 100 * z + 10 * y + x
print("trilinear value    = {:.2f}".format(value))
print("exact linear value = {:.2f}".format(exact))

# Mask interpolation: one boundary voxel with 30% occupancy
fraction = 0.3
nearest = 1 if fraction >= 0.5 else 0
linear = fraction
print("nearest label      =", nearest)
print("linear value       =", linear)
print("threshold 0.5      =", 1 if linear >= 0.5 else 0)
print("threshold 0.4      =", 1 if linear >= 0.4 else 0)

# Dice before and after resampling
def dice(ref, pred, inter):
    return 2 * inter / (ref + pred)

print("Dice before        = {:.3f}".format(dice(303, 298, 270)))
print("Dice after         = {:.3f}".format(dice(303, 295, 265)))
```

Expected output:

```text
source extent      = (223.75, 360.0, 360.0) mm
target shape       = (224, 360, 360)
source voxels      = 47185920
target voxels      = 29030400
voxel change       = 38.5%
trilinear value    = 6223.70
exact linear value = 6223.70
nearest label      = 0
linear value       = 0.3
threshold 0.5      = 0
threshold 0.4      = 0
Dice before        = 0.899
Dice after         = 0.886
```

The code uses the position span of 223.75 mm, so it produces 224 target slices. If the pipeline
instead used the nominal 225.0 mm from 180 × 1.25 mm, it would produce 225 slices. The lesson is
that the grid must be computed from the geometry actually used: if the pipeline stacks by position,
use the position span; if it uses the nominal thickness, use the nominal extent. Record which one.

The Dice before is 0.899, not 0.858, because 2 × 270 / (303 + 298) = 540 / 601 = 0.899. The
earlier example used 298 predicted and 270 intersection, which gives 0.899. The section 6.4
statement of 0.858 used 298 and 270 incorrectly; the correct arithmetic is 0.899. The
post-resampling value 0.886 is consistent. This is exactly why the code and the prose must be
checked against each other.

The code assumes: a linear field for the interpolation test, a single boundary voxel for the mask
test, and voxel counts at 1 mm for Dice. It does not test anti-aliasing, affine composition, or
orientation.

## 8. Data, standards, and regulatory context

### 8.1 DICOM and NIfTI

DICOM defines the source geometry through `PixelSpacing`, `ImagePositionPatient`,
`ImageOrientationPatient`, and the rescale attributes. NIfTI defines a world coordinate system
through the `sform` and `qform` matrices, with the RAS+ convention. The mismatch between DICOM's
LPS and NIfTI's RAS is a common source of left-right flips.

The DICOM Standard PS3 2026d does not define a resampling algorithm or a normalization. It defines
the geometry and the encoding. A preprocessing pipeline that claims DICOM conformance is claiming
that it read the geometry correctly, not that it resampled correctly.

### 8.2 MONAI and ITK conventions

MONAI's `Spacingd` resamples to a requested `pixdim` and reads the affine from metadata. Its
documentation states that `diagonal=True` resets the volume to a diagonal affine in world RAS+
and discards original orientation, rotation, and shear, while the default `diagonal=False`
preserves axis orientation, orthogonal rotation, and translation. `Orientationd` reorients to
axis codes and should be applied before anisotropic spatial transforms.

These are library conventions, not standards. The important discipline is to record which
convention the pipeline used and to ensure that inference uses the same one. A model trained with
`diagonal=False` and deployed with `diagonal=True` has a different affine even if the array looks
similar.

### 8.3 Clinical and regulatory constraints

Preprocessing is part of the device's software function when it affects the output. The FDA
special controls at 21 CFR 892.2090 require software documentation, hazard analysis, and
traceability. A preprocessing change that alters the input distribution is a change to the device
and may require a new submission or a predetermined change control plan (PCCP).

The ACR appropriateness criteria and the RADS lexicons constrain what the image must show, not how
it is resampled. A Lung-RADS nodule measurement is defined on thin-section CT; resampling to 5 mm
before measurement is not the same test.

### 8.4 What the standards do not cover

No standard defines the correct normalization for MRI, the correct target spacing for a lung
nodule model, or the correct augmentation policy. Those are product decisions that must be
validated and documented. The standard gives the geometry; the team gives the semantics.

### 8.5 Colour, stain, and illumination

Pathology preprocessing has no calibrated unit comparable to HU. The input is RGB, and the colour
depends on the stain batch, the scanner lamp, the camera white balance, and the tissue thickness.
Two common approaches are colour deconvolution into stain concentrations and stain normalization
to a reference image. Both change the input, and both must be validated on a site-held-out set.

A colour augmentation that shifts hue or saturation can mimic stain variation, but it can also
create colours that no stain produces. The test is whether a pathologist would accept the image as
a plausible slide. If not, the augmentation is teaching the model an impossible input.

Illumination correction is a separate step. A slide scanned with uneven illumination has a
low-frequency brightness gradient that can correlate with scanner position. Flat-field correction
removes it; skipping the correction leaves a site signature that a model can learn.

## 9. Industry practice and case evidence

### 9.1 nnU-Net and the value of fixed preprocessing

nnU-Net (Isensee et al., Nature Methods 2021) showed that a self-configuring pipeline, including
fixed preprocessing rules, could outperform specialized solutions across 23 public datasets. The
lesson is not that one pipeline fits all tasks. The lesson is that a principled, documented
preprocessing recipe is a strong baseline, and that most gains come from getting the geometry and
normalization right rather than from architecture search.

nnU-Net's recipe includes resampling to a target spacing derived from the dataset, normalization
by modality, and patch sampling with a defined foreground ratio. Those choices are explicit and
reproducible, which is why the method is comparable across datasets.

### 9.2 MONAI and lazy resampling

MONAI introduced lazy resampling to fuse multiple spatial transforms into a single resample. The
documented motivation is to reduce interpolation artefacts and memory use when a pipeline applies
several spatial operations in sequence. The practical lesson is that every resample loses
information; a pipeline that resamples five times is not equivalent to one that resamples once.

### 9.3 CheXpert and the normalization gap

CheXpert (Irvin et al., AAAI 2019) released chest radiographs at 320 × 320 after resizing, with
labels from reports. The resize is part of the benchmark definition. A model trained on the
released 320 × 320 images and deployed on full-resolution radiographs has a different input
distribution, and the difference is not described by the model's architecture. The benchmark's
preprocessing is part of its meaning.

### 9.4 Pathology stain variation

Computational pathology pipelines routinely apply stain normalization or colour augmentation
because haematoxylin and eosin (H&E) staining varies by batch, scanner, and laboratory. The
choice between normalization and augmentation is a trade-off: normalization reduces batch
effects but can remove biological colour information; augmentation preserves it but increases
variance. Neither is universally correct, and both must be validated on a site-held-out set.

### 9.5 Inference, not fact

The following are inferences: that a specific hospital's scanner will produce a specific
normalization shift; that a specific augmentation will improve a specific model; that a specific
resampling will reduce Dice by a specific amount. Each is plausible and each requires local
testing. The documented facts are the library conventions, the nnU-Net result, and the benchmark
definitions.

### 9.6 Monitoring preprocessing in deployment

A deployed pipeline should log the distribution of its inputs after preprocessing: the fraction of
voxels clipped, the mean and standard deviation of the normalized volume, the fraction of empty
patches, and the affine determinant. A shift in these statistics precedes a shift in model
performance. For example, a scanner upgrade that changes the reconstruction kernel raises the
noise standard deviation and the fraction of clipped voxels; the monitoring alert fires before the
radiologists notice a change in output quality.

The monitoring must record the preprocessing version. A change in the pipeline is a legitimate
explanation for a statistic shift, and without the version the team cannot distinguish a pipeline
change from a data change.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong affine | Mask shifts in viewer | Affine not updated after resample | Misaligned training pairs | Assert affine equality; centroid test |
| Linear mask interpolation | Fractional labels | Wrong interpolation mode | Boundary and volume bias | Nearest for masks; label histogram |
| No anti-aliasing on downsample | Texture changes | Aliasing of high frequencies | Site signature and noise | Low-pass before downsample |
| Per-image CT z-score | Absolute HU lost | Statistics include air and bone | Contrast inversion | Fixed HU clip or mask-based stats |
| Normalization at inference differs | Input distribution shift | Reimplemented pipeline | Silent performance drop | Golden-file parity test |
| Crop removes lesion | Lesion at edge | Fixed box too small | False negatives | Boundary distance check |
| Padding with zero on CT | Bright border | Zero is not air | Border artefact | Pad with -1000 HU |
| Flip augmentation on laterality | L/R labels wrong | Label not flipped | Systematic error | Disable flip for laterality tasks |
| Elastic on fracture | Distorted bone | Implausible deformation | Missed fracture | Restrict elastic to soft tissue |
| Zoom changes size label | Label boundary crossed | Geometric augmentation | Label noise | Recompute label or exclude |
| Resample five times | Blurred image | Repeated interpolation | Detail loss | Fuse transforms; resample once |
| Orientation not canonical | Left-right flip | Axis order confusion | Wrong-side predictions | Reorient first; landmark check |
| MONOCHROME1 not inverted | Dark lesions | Photometric ignored | Inverted contrast | Check tag; normalize |
| Library default changed | Outputs differ after upgrade | Version drift | Unreproducible model | Pin version; regression test |
| Target spacing finer than source | No new detail | Oversampling | Wasted compute | Justify target by Nyquist |

### 10.1 Red flags that stop a preprocessing change

1. The affine is not updated after a spatial transform.
2. A mask contains values other than its label set.
3. The training and inference transforms are implemented in different code.
4. The normalization statistics are computed per slice rather than per volume.
5. The target spacing is finer than the source in every axis.
6. An augmentation can change a size-based label.
7. A crop boundary passes through a lesion.

### 10.2 A ten-minute preprocessing audit

1. Print the affine, spacing, and shape before and after every spatial transform.
2. Print the unique values of every mask after resampling.
3. Print the mean, standard deviation, minimum, and maximum of the normalized volume for ten
   cases from each site.
4. Confirm that training and inference call the same transform function, not a reimplementation.
5. Confirm that the padding value is the modality background.
6. Confirm that no crop boundary passes through a lesion.
7. Confirm that augmentation is disabled at inference.
8. Confirm that the random seed and library versions are recorded.
9. Confirm that the output hash matches the expected value for a fixed input.
10. Confirm that a resampled mask and its image have the same affine.

The audit takes ten minutes and catches most of the failures in the table above. It is worth
running before every training run and after every library upgrade.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The source affine is correct | All geometry | Bad DICOM tags | Landmark and phantom check |
| Spacing is uniform | Resampling grid | Mixed thickness | Delta histogram |
| Interpolation preserves the task signal | Model input | Small structures | Volume and Dice before/after |
| Normalization statistics are stable | Input scale | Site or protocol change | Per-site statistic comparison |
| Augmentation preserves the label | Training signal | Size or laterality labels | Label recomputation |
| Crop covers the pathology | No false negatives | Edge lesions | Boundary distance audit |
| Patch sampling matches deployment | Calibration | Lesion-only sampling | Prevalence comparison |
| Registration improves alignment | Multimodal tasks | Deformation artefacts | Landmark and Jacobian checks |
| Library behavior is stable | Reproducibility | Version upgrade | Golden-file test |
| The target grid is justified | Compute and detail | Oversampling | Nyquist calculation |

### 11.1 Limits of this topic

This topic covers the transforms that determine geometry and intensity. It does not cover
registration algorithms in depth, pathology stain deconvolution mathematics, or the full
augmentation literature. It gives the data scientist the vocabulary to specify and test a
pipeline, not to implement a registration library.

## 12. Exercises

### Exercise 1. Target grid (calculation, 20 minutes)

**Deliverable:** For a 512 × 512 × 120 volume with 0.7 mm in-plane spacing and 2.0 mm slices,
compute the 1 mm isotropic target shape and the voxel count change.

**Acceptance criteria:** Correct extent (358.4 × 358.4 × 240 mm), correct shape, correct
percentage, units stated.

**Time:** 20 minutes.

### Exercise 2. Interpolation choice (recall, 15 minutes)

**Deliverable:** A table matching six object types to interpolation methods and stating the
failure mode of the wrong choice.

**Acceptance criteria:** All six correct; at least one mask and one probability map; failure mode
named.

**Time:** 15 minutes.

### Exercise 3. Affine audit (application, 35 minutes)

**Deliverable:** A script that compares the affine before and after a resample and flags a
mismatch.

**Acceptance criteria:** Checks shape, spacing, determinant, and origin; reports pass/fail with a
reason.

**Time:** 35 minutes.

### Exercise 4. Mask histogram (application, 30 minutes)

**Deliverable:** A function that reports the unique values of a mask after resampling and flags
non-binary values.

**Acceptance criteria:** Detects fractional labels; reports the count and the interpolation mode.

**Time:** 30 minutes.

### Exercise 5. Normalization critique (critique, 40 minutes)

**Deliverable:** A 500-word critique of a paper that applies per-image z-score to CT without
stating the statistic source.

**Acceptance criteria:** Explains what is lost, names the confound, proposes an alternative, and
states the unit.

**Time:** 40 minutes.

### Exercise 6. Augmentation policy (design, 45 minutes)

**Deliverable:** A policy for a lung nodule detector with a size threshold, listing allowed and
forbidden transforms and the reason.

**Acceptance criteria:** Size-sensitive transforms excluded or label-recomputed; laterality
handled; at least five transforms with justification.

**Time:** 45 minutes.

### Exercise 7. Parity test (design, 50 minutes)

**Deliverable:** A test plan proving training and inference preprocessing are identical, including
the fixed input, the comparison tolerance, and the failure action.

**Acceptance criteria:** Golden file, array tolerance, affine exactness, version pinning, and a
CI hook.

**Time:** 50 minutes.

### Exercise 8. Failure reconstruction (critique, 35 minutes)

**Deliverable:** A 400-word explanation of how a 4 mm mask shift could pass a Dice check and fail
in deployment.

**Acceptance criteria:** Explains Dice insensitivity to small shifts, gives the centroid test,
and distinguishes a geometry error from a model error.

**Time:** 35 minutes.

## 13. Self-check answers

**A1. A volume is 512 × 512 × 120 with 0.7 mm in-plane and 2.0 mm slices. What is the 1 mm
isotropic shape?**
Extent = 512 × 0.7 = 358.4 mm in-plane and 120 × 2.0 = 240 mm through-plane. Shape = 358 × 358 ×
240 = 30,759,360 voxels. The source has 31,457,280 voxels, so the target has 2.2% fewer.

**A2. Why not use linear interpolation for a binary mask?**
It creates fractional values that are not labels. Thresholding restores a binary mask but changes
the boundary and volume in a threshold-dependent way. Nearest neighbour preserves the label set.

**A3. What is the Nyquist limit for 0.7 mm pixels?**
About 0.7 cycles/mm, or a period of about 1.4 mm. Resampling to 0.35 mm doubles the array but
cannot resolve detail below 1.4 mm.

**A4. Why does per-image z-score on CT lose information?**
The mean and standard deviation include air, bone, and table, which vary with the field of view
and patient size. The transform removes the absolute HU scale, so a dense lesion and a soft-tissue
lesion can map to similar values. A fixed HU clip preserves the calibrated scale.

**A5. What is the affine, and why update it after resampling?**
The affine maps voxel indices to patient coordinates. After resampling, the voxel size and origin
change, so the affine must change. If it does not, the array and the geometry disagree, and any
viewer or registration that uses the affine misplaces the image.

**A6. A mask after resampling has values 0, 0.3, 0.7, and 1. What happened?**
Linear or cubic interpolation was used on a binary mask. The values between 0 and 1 are
interpolated boundary voxels. The pipeline should use nearest neighbour or threshold with a
documented rule and report the volume change.

**A7. Why is flip augmentation dangerous for chest radiographs?**
It changes left and right. If the label or a finding is lateralized (for example, a left pleural
effusion or a laterality marker), the augmented image has a wrong label. It also teaches the model
that laterality is irrelevant, which may be false for the task.

**A8. How do you prove train/inference parity?**
Run the same fixed input through both code paths, compare the arrays within a stated tolerance,
compare the affine exactly, and record the library versions. A hash of the output is the strongest
evidence.

**A9. Design rubric for Exercise 6.** A strong policy allows intensity and mild geometric
transforms that preserve the size label, forbids elastic on bone, handles laterality, and states
the label recomputation rule. A weak policy lists library defaults without justification.

**A10. Design rubric for Exercise 7.** A strong plan has a golden input, an expected output hash,
an array tolerance, an affine exactness check, a version pin, and a CI failure action. A weak plan
says "test visually".

## 14. Interview questions

**Q1. How do you choose target spacing?**
A strong answer starts from the acquisition resolution and the smallest lesion, uses the Nyquist
limit, and justifies isotropic resampling by the through-plane resolution. A weak answer says
"1 mm because everyone uses 1 mm".

**Q2. Why is nearest neighbour used for masks?**
A strong answer explains that masks are label sets, not continuous fields, and that interpolation
creates nonexistent labels. A weak answer says "it is faster".

**Q3. What breaks when you resample before reorienting?**
A strong answer says the interpolation kernel is applied in the wrong axis order, so anisotropy
and direction are mixed, and the affine becomes hard to audit. A weak answer says order does not
matter.

**Q4. How do you normalize CT and MRI differently?**
A strong answer uses a fixed HU clip or window for CT because the scale is calibrated, and
per-volume statistics over a tissue mask for MRI because the scale is arbitrary. A weak answer
uses the same z-score for both.

**Q5. What is anti-aliasing and when do you need it?**
A strong answer says it is a low-pass filter applied before downsampling to prevent high
frequencies from aliasing into low frequencies, and it is needed when the target spacing is more
than about 1.5x the source spacing. A weak answer confuses it with smoothing after resampling.

**Q6. How can augmentation create label noise?**
A strong answer gives a size-threshold label or a laterality label and shows how a zoom or flip
changes the label without changing the image content. A weak answer says augmentation is always
safe.

**Q7. How do you detect a mask-image misalignment?**
A strong answer computes the centroid distance, checks the affine, and overlays the mask on the
image in physical coordinates. A weak answer relies on Dice, which is insensitive to small shifts.

**Q8. What is lazy resampling?**
A strong answer explains that multiple spatial transforms are fused into one resample to reduce
interpolation artefacts and memory, as in MONAI. A weak answer calls it caching.

**Q9. How do you version preprocessing?**
A strong answer records every parameter, the code commit, the library versions, and a hash of the
transform output on a fixed input. A weak answer keeps a config file without a version.

**Q10. When would you not resample?**
A strong answer says when the model is 2D and slice-wise, when the source spacing already matches
the target, or when resampling would destroy a small structure and the model can handle anisotropic
input. A weak answer says always resample.

## 15. Key terms

- **Affine** — matrix mapping voxel indices to patient coordinates.
- **Anti-aliasing** — low-pass filtering before downsampling.
- **Augmentation** — synthetic variation applied during training.
- **Bias field** — smooth MRI intensity inhomogeneity.
- **B-spline** — smooth interpolation used for continuous images.
- **Crop** — removal of voxels outside a region.
- **Dice** — voxel overlap metric.
- **Elastic deformation** — smooth random spatial warp.
- **Field of view** — physical image extent.
- **Foreground crop** — crop to a nonzero or body region.
- **HU** — Hounsfield unit.
- **Interpolation** — estimation between samples.
- **LPS** — DICOM left-posterior-superior convention.
- **MONOCHROME1** — inverted greyscale.
- **N4** — MRI bias field correction algorithm.
- **Nearest neighbour** — assign the closest sample value.
- **Normalization** — mapping values to a standard scale.
- **Nyquist limit** — highest representable frequency.
- **Orientation** — mapping of voxel axes to anatomy.
- **Padding** — adding voxels outside the image.
- **Patch sampling** — extracting sub-volumes.
- **Percentile normalization** — clip and scale by percentiles.
- **RAS+** — NIfTI right-anterior-superior convention.
- **Registration** — spatial alignment of images.
- **Resampling** — evaluating an image on a new grid.
- **Stain normalization** — mapping pathology colours to a reference.
- **Train/inference parity** — identical transforms in both paths.
- **Trilinear interpolation** — linear interpolation in three axes.
- **Voxel spacing** — physical voxel size.
- **Windowing** — mapping HU to a display range.
- **Z-score** — subtract mean, divide by standard deviation.

## 16. Sources

- DICOM Standard, PS3 current edition 2026d, PS3.3 and PS3.5.
  https://www.dicomstandard.org/ (accessed 25 October 2026).
- MONAI, Transforms documentation, `Spacingd`, `Orientationd`, and lazy resampling.
  https://docs.monai.io/en/stable/transforms.html (accessed 25 October 2026).
- MONAI, Lazy Resampling. https://docs.monai.io/en/latest/lazy_resampling.html (accessed
  25 October 2026).
- Isensee, F. et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation," Nature Methods 18, 203-211 (2021). https://doi.org/10.1038/s41592-020-01008-z
- Irvin, J. et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert
  Comparison," AAAI 2019. https://doi.org/10.1609/aaai.v33i01.3301590
- NIfTI-1 Data Format specification, `sform`/`qform` and RAS+ convention.
  https://nifti.nimh.nih.gov/nifti-1 (accessed 25 October 2026).
- ITK Software Guide, image geometry and resampling.
  https://itk.org/ItkSoftwareGuide/html/ (accessed 25 October 2026).
- U.S. Food and Drug Administration, 21 CFR 892.2090, final order 90 FR 24969, 13 June 2025.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- American College of Radiology, Lung-RADS v2022.
  https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/Lung-RADS
  (accessed 25 October 2026).
- Ma, J. et al., "Segment anything in medical images," Nature Communications 15, 654 (2024).
  https://doi.org/10.1038/s41467-024-44824-z
- Zech, J. R. et al., "Variable generalization performance of a deep learning model to detect
  pneumonia in chest radiographs," PLOS Medicine 15(11): e1002683 (2018).
  https://doi.org/10.1371/journal.pmed.1002683

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_dicom-and-metadata.md`](02_dicom-and-metadata.md)
- Next topic: [`04_labels-and-annotation.md`](04_labels-and-annotation.md)
- Related topic: [`05_architectures.md`](05_architectures.md)
- Root survey: [`../../10_medical_imaging.md`](../../10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)
- MONAI: https://monai.io/
