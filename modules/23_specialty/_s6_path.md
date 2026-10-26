## 6. Worked example

A laboratory has built a prostate biopsy grading model and reports a quadratic weighted kappa
of 0.96 against the reference uropathologist. The buyer's question is whether the model is
ready for a workflow in which it pre-orders cases for review.

The evaluation set is 200 biopsies from 200 patients, graded on the five-category ISUP grade
group scale: group 1 (Gleason 3+3), group 2 (3+4), group 3 (4+3), group 4 (8), and group 5
(9 or 10). The reference distribution is 70, 60, 30, 20, and 20 cases.

### 6.1 The patient-level confusion matrix

Rows are the reference read. Columns are the model output for the same case.

| Reference \ model | GG1 | GG2 | GG3 | GG4 | GG5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| GG1 | 53 | 14 | 2 | 1 | 0 | 70 |
| GG2 | 10 | 37 | 11 | 2 | 0 | 60 |
| GG3 | 1 | 9 | 13 | 6 | 1 | 30 |
| GG4 | 0 | 2 | 6 | 8 | 4 | 20 |
| GG5 | 0 | 0 | 2 | 5 | 13 | 20 |
| Total | 64 | 62 | 34 | 22 | 18 | 200 |

The diagonal is exact agreement: 53 + 37 + 13 + 8 + 13 = 124 of 200, or 62 percent. Exact
agreement alone is a poor summary for an ordinal scale, because a one-grade miss and a
four-grade miss are not the same error. Quadratic weighted kappa fixes that by penalising the
squared distance between categories.

### 6.2 The kappa arithmetic

With five categories the normalising denominator is (5 - 1) squared, which is 16. The weighted
observed disagreement is the sum of squared distances times the cell counts, divided by 16:

```text
Distance 1 cells: 14 + 10 + 11 + 9 + 6 + 6 + 4 + 5 = 65
Distance 2 cells:  2 +  2 +  1 + 1 + 2 + 2 = 10
Distance 3 cells:  1
Distance 4 cells:  0

sum of squared-distance counts = 1(65) + 4(10) + 9(1) + 16(0) = 114
weighted observed disagreement   = 114 / 16 = 7.125
```

The expected disagreement is computed from the margins, with expected count
row total times column total divided by 200, and the same squared-distance weighting:

```text
sum over cells of (i-j)^2 x row_i x col_j = 133,440
expected count-weighted total            = 133,440 / 200 = 667.2
weighted expected disagreement           = 667.2 / 16 = 41.7

QWK = 1 - 7.125 / 41.7 = 1 - 0.17086 = 0.8291
```

The reported 0.96 is not this number. The next two sections show where it came from.

### 6.3 Diagnostic one: the split

The reported evaluation split at the tile level, drawing tiles from all 200 slides into both
training and test sets. The same model, evaluated that way, produces the following matrix.

| Reference \ model | GG1 | GG2 | GG3 | GG4 | GG5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| GG1 | 66 | 4 | 0 | 0 | 0 | 70 |
| GG2 | 4 | 52 | 4 | 0 | 0 | 60 |
| GG3 | 1 | 3 | 24 | 2 | 0 | 30 |
| GG4 | 0 | 0 | 3 | 16 | 1 | 20 |
| GG5 | 0 | 0 | 0 | 2 | 18 | 20 |
| Total | 71 | 59 | 31 | 20 | 19 | 200 |

```text
sum of squared-distance counts = 1(23) + 4(1) + 9(0) + 16(0) = 27
weighted observed   = 27 / 16 = 1.6875
weighted expected   = 678.8 / 16 = 42.425
QWK = 1 - 1.6875 / 42.425 = 1 - 0.03978 = 0.9602
```

The gap is 0.9602 minus 0.8291, which is 0.1311. The model did not change. The split changed.
Tiles from the same slide share stain, scanner, focus, and tissue architecture, so a tile-level
split places near-duplicates on both sides and the test set stops being independent.

The diagnostic is to compute both numbers and report the gap. A gap above roughly 0.05 on an
ordinal scale means the split, not the model, is doing the work. Recomputing costs one training
run and no new data.

### 6.4 Diagnostic two: dichotomisation

The second diagnostic asks what happens when the ordinal scale is collapsed for clinical use.
Most prostate decisions turn on "clinically significant cancer", conventionally grade group 2
or higher. Collapse the model output at that boundary and compute the operating point.

```text
True positives  (reference GG2-5, model GG2-5):
  row GG2: 37 + 11 + 2 + 0 = 50
  row GG3:  9 + 13 + 6 + 1 = 29
  row GG4:  2 +  6 + 8 + 4 = 20
  row GG5:  0 +  2 + 5 + 13 = 20
  total = 119
False negatives (reference GG2-5, model GG1) = 10 + 1 + 0 + 0 = 11
False positives (reference GG1, model GG2-5) = 14 + 2 + 1 + 0 = 17
True negatives  (reference GG1, model GG1)   = 53

Sensitivity = 119 / 130 = 0.9154
Specificity =  53 /  70 = 0.7571
```

A team reporting "91.5 percent sensitivity for clinically significant cancer" has told the
truth and has hidden the consequence. Convert the operating point to the deployment prevalence,
which for a screening biopsy population is about 4 percent:

```text
PPV = (0.04 x 0.9154) / (0.04 x 0.9154 + 0.96 x 0.2429)
    = 0.036616 / (0.036616 + 0.233184)
    = 0.036616 / 0.269800
    = 0.1357 -> 13.6 percent
```

A clinician who reads "91.5 percent sensitivity" and expects nine of ten flagged cases to be
cancer will find one in seven. The sensitivity is real and the expectation is wrong, because
specificity of 0.7571 multiplied by a large disease-free group produces seven false positives
for every true positive at this prevalence.

The diagnostic is to compute PPV at the prevalence of the population the model will serve,
before anyone quotes a sensitivity. It takes the counts already in the matrix.

### 6.5 Vary one assumption: the spectrum

The evaluation set had 35 percent grade group 1, which is a referral population. A screening
biopsy population has more benign-adjacent disease. Reweight the rows to 55, 25, 10, 6, and 4
percent, which gives 110, 50, 20, 12, and 8 cases.

| Reference \ model | GG1 | GG2 | GG3 | GG4 | GG5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| GG1 | 83 | 22 | 3 | 2 | 0 | 110 |
| GG2 | 8 | 31 | 9 | 2 | 0 | 50 |
| GG3 | 1 | 6 | 9 | 4 | 0 | 20 |
| GG4 | 0 | 1 | 4 | 5 | 2 | 12 |
| GG5 | 0 | 0 | 1 | 2 | 5 | 8 |

Recomputing on this matrix gives a QWK of 0.7792, a change of minus 0.0499 from 0.8291. The
model is unchanged and the population is more typical of screening, so the honest expectation
for deployment is 0.78, not 0.83.

This is the varied assumption that changes the conclusion. At 0.83 the model is within 0.03 of
the published inter-pathologist ceiling for general pathologists, which supports a
second-read claim. At 0.78 it is at the ceiling rather than near it, which supports a triage
claim and not an autonomous one. The regulatory and commercial difference between those two
claims is larger than the metric difference.

### 6.6 The category-count effect

There is a third number a vendor can produce without changing the model at all: collapse the
five grade groups into three tiers (GG1, GG2-3, GG4-5) and recompute. The QWK falls to 0.7582,
a change of minus 0.0709.

That direction is counterintuitive, because collapsing categories is often assumed to increase
agreement. It falls here because the quadratic weights are normalised by (k - 1) squared, so
the penalty for a two-category miss grows from 4/16 to 1 when the scale shortens. A five-tier
kappa and a three-tier kappa are different quantities, and neither is wrong. The defect is
reporting one and comparing it to the other.

| Scale | Categories | QWK | Comparable to |
|---|---|---:|---|
| ISUP grade group | 5 | 0.8291 | Five-tier literature only |
| Collapsed tiers | 3 | 0.7582 | Nothing published at five tiers |
| Dichotomised at GG2 | 2 | 0.7571 specificity, 0.9154 sensitivity | Screening studies |

### 6.7 What the case does not establish

The case is a measurement example, not a validation of any product. It does not show that the
model improves a pathologist's accuracy, because the reference is a single read and the
comparison is offline. It does not establish transportability to another scanner, another
laboratory's stain protocol, or a biopsy set with a different grade distribution. It does not
establish that a 0.78 QWK is sufficient for a triage claim, because that depends on the review
budget and the harm of a missed high-grade case.

| Choice | Value used here | Alternative that changes the answer | How to disclose it |
|---|---|---|---|
| Split level | Patient | Tile | Report the leakage gap |
| Category count | Five | Three, or two | Report the scale with the kappa |
| Spectrum | 35 percent GG1 | 55 percent GG1 | Report the grade distribution |
| Reference standard | Single uropathologist | Consensus panel | Report inter-reader kappa |
| Aggregation | Attention MIL | Mean pooling | Report the rule and the tile counts |
| Scanner | One model | Two generations | Report per-scanner performance |
| Equivocal handling | Not modelled | Explicit equivocal class | Report the work-up path |

An analysis that discloses these seven choices can be reproduced and criticised. One that
discloses none produces a number that cannot be checked.

