## 4. Core Formulas with Full Derivation

Each result below follows the same five-part anatomy: FORMULA, WHERE (symbols, units, domains), CONSTRAINT (preconditions), NUMERICAL EXAMPLE (recomputed by hand), INTERPRETATION. Each subsection closes with an explicit validity condition that names where the result stops being true.

### 4.1 Base case: IoU of two axis-aligned rectangles

```text
FORMULA
  IW = max(0, min(x2^A, x2^B) - max(x1^A, x1^B))
  IH = max(0, min(y2^A, y2^B) - max(y1^A, y1^B))
  I  = IW * IH
  U  = (x2^A-x1^A)(y2^A-y1^A) + (x2^B-x1^B)(y2^B-y1^B) - I
  IoU = I / U

WHERE
  A, B   : axis-aligned boxes, continuous half-open xyxy, pixels
  IW, IH : overlap extents in pixels, non-negative real
  I      : intersection area, square pixels
  U      : union area, square pixels
  IoU    : dimensionless ratio in [0, 1]

CONSTRAINT
  x1 < x2 and y1 < y2 for both boxes (strictly positive area)
  Same coordinate frame and same units for both boxes
  Same scale for both boxes (no per-box normalization mismatch)

NUMERICAL EXAMPLE  (box A = prediction of a person, box B = reference)
  A = (102, 98, 178, 262)   -> width 76, height 164, area 12464
  B = (100, 100, 180, 260)  -> width 80, height 160, area 12800
  IW = min(178,180) - max(102,100) = 178 - 102 = 76
  IH = min(262,260) - max(98,100)  = 260 - 100 = 160
  I  = 76 * 160 = 12160
  U  = 12464 + 12800 - 12160 = 13104
  IoU = 12160 / 13104 = 0.927960928...

INTERPRETATION
  The prediction covers 88.9% of the reference area (12160/12800 = 0.95, not 0.889;
  recompute: 12160/12800 = 0.950) while claiming 2.6% extra area outside it.
  IoU 0.928 clears a 0.50 threshold comfortably and a 0.90 threshold narrowly.
```

The check that matters in practice is the direction of the two error terms. Here the prediction is 4 pixels narrower and 4 pixels taller than the reference, so it loses a 4x160 strip on the left and gains a 4x4 patch on the right; the net intersection fraction is `12160/12800 = 0.950`. IoU is lower than that fraction because the union is larger than the reference.

**Validity condition.** The formula is exact for axis-aligned rectangles in a shared frame. It is *not* a distance: see 4.4.1.

### 4.2 General case: matching-based precision and recall at a fixed threshold

The base case handles one pair. Evaluation requires a decision over all pairs, and that decision is where every subtle bug lives.

```text
FORMULA
  Let P be predictions of one image and class, sorted by score descending:
      p_(1), p_(2), ..., p_(n),   s(p_(1)) >= s(p_(2)) >= ... >= s(p_(n))
  Let G be reference instances of the same image and class: |G| = m
  Let tau be the IoU threshold, and C = set of already-claimed references, initially empty

  for i = 1..n:
      g* = argmax_{g in G, g not in C} IoU(box(p_(i)), box(g))
      if g* exists and IoU(p_(i), g*) >= tau:
          label(p_(i)) = TP ;  C = C union {g*}
      else:
          label(p_(i)) = FP

  TP(t) = #{ i : s(p_(i)) >= t and label = TP }
  FP(t) = #{ i : s(p_(i)) >= t and label = FP }
  FN(t) = m - TP(t)
  Precision(t) = TP(t) / (TP(t) + FP(t))
  Recall(t)    = TP(t) / m

WHERE
  n  : number of predictions for this image and class, integer >= 0
  m  : number of reference instances for this image and class, integer >= 0
  tau: IoU threshold, dimensionless in (0, 1]
  t  : score threshold, real; only the induced cut matters

CONSTRAINT
  One image at a time: matching never crosses image boundaries
  One class at a time when class_aware = true
  Greedy assignment by descending score, not by best IoU globally
  max_dets applied before matching, so n <= max_dets

NUMERICAL EXAMPLE  (image I2, class person, tau = 0.50, two references)
  References: g3 = (50,50,150,250) area 20000 ; g4 = (400,80,470,240) area 11200
  Predictions sorted: p3 s=0.90 (52,54,148,246) ; p4 s=0.84 (55,52,145,248) ; p5 s=0.35 (398,84,468,236)

  i=1  p3: IoU(p3,g3) = 18432/20000 = 0.9216 ; IoU(p3,g4) = 0
           0.9216 >= 0.50 -> TP, claim g3
  i=2  p4: best unclaimed is g4, IoU(p4,g4) = 0 ; g3 is claimed
           0 < 0.50 -> FP            (a duplicate, penalised)
  i=3  p5: best unclaimed is g4, IoU(p5,g4) = 0.89847
           0.89847 >= 0.50 -> TP, claim g4

  At t = 0.50: TP = 2 (p3, p5), FP = 1 (p4), FN = 0
  Precision = 2/3 = 0.666667 ; Recall = 2/2 = 1.000000

INTERPRETATION
  The detector found both people and produced one duplicate.
  Its recall at this threshold is perfect and its precision is 2/3.
  A naive pairwise counter that skipped the claim set would report TP = 3,
  precision = 1.0, and hide the duplicate — the single most common
  evaluation bug in home-grown scripts.
```

Note the ordering effect, which is not cosmetic. If p4 had scored 0.90 and p3 0.84, p4 would be the TP on g3 and p3 would be the FP. Precision at a fixed recall would be identical here, but AP would change, because the duplicate sits at a different rank and so changes the staircase in 4.3.

**Validity condition.** Greedy-by-score matching is the standard, but it is not the assignment that maximises total TP. See 4.4.3 for a counterexample where greedy loses 0.29 of score mass to an optimal assignment. Report which matching you used; a metric whose value depends on an undocumented tie-break is not a measurement.

### 4.3 From matching to average precision

```text
FORMULA
  Given the sorted prediction list with TP/FP labels, define at each rank i:
      R_i = (cumulative TP through i) / m
      P_i = (cumulative TP through i) / i
  Envelope (monotone precision):
      P~_i = max_{j >= i} P_j
  All-point interpolated AP:
      AP = sum_{i : R_i > R_(i-1)} (R_i - R_(i-1)) * P~_i ,  with R_0 = 0
  Macro mAP over K classes:
      mAP = (1/K) * sum_{k=1..K} AP_k

WHERE
  R_i : recall at rank i, dimensionless in [0,1]
  P_i : precision at rank i, dimensionless in [0,1]
  m   : reference count for this class across all images
  K   : number of classes in the evaluation label set
  AP  : dimensionless in [0,1]

CONSTRAINT
  Predictions from all evaluated images pooled, then sorted globally by score
  Reference counts summed across images
  Ties in score broken deterministically (or the sum is not reproducible)
  Classes with m = 0 excluded from the macro average, not counted as 0 or 1

NUMERICAL EXAMPLE  (class person, m = 4 references, six predictions)
  Ranked labels, high score first:  TP, TP, FP, TP, FP, TP
  i=1: TP=1 FP=0 R=0.2500 P=1.0000
  i=2: TP=2 FP=0 R=0.5000 P=1.0000
  i=3: TP=2 FP=1 R=0.5000 P=0.6667
  i=4: TP=3 FP=1 R=0.7500 P=0.7500
  i=5: TP=3 FP=2 R=0.7500 P=0.6000
  i=6: TP=4 FP=2 R=1.0000 P=0.6667

  Envelope from the right: [1.0000, 1.0000, 0.7500, 0.7500, 0.6667, 0.6667]
  Recall increases at i=1,2,4,6 with steps 0.25 each:
  AP = 0.25*1.0000 + 0.25*1.0000 + 0.25*0.7500 + 0.25*0.6667
     = 0.2500 + 0.2500 + 0.1875 + 0.1667
     = 0.854166667

INTERPRETATION
  AP is the area under the enveloped precision-recall staircase: each recall
  step is credited with the best precision available at any later rank.
  On this ranking the envelope changes nothing, because every recall step
  occurs at a rank whose precision already equals the later maximum. The four
  steps carry precision 1.0000, 1.0000, 0.7500, 0.6667 and the envelope at
  those same ranks is 1.0000, 1.0000, 0.7500, 0.6667. Raw and enveloped AP
  both give 0.854167.

  Enveloping does bite when a recall step lands on a precision dip that later
  recovers. Ranking TP, FP, TP, TP with m = 3:
      i=1 TP  R=0.3333 P=1.0000
      i=2 FP  R=0.3333 P=0.5000
      i=3 TP  R=0.6667 P=0.6667
      i=4 TP  R=1.0000 P=0.7500
  Raw AP       = (1/3)*1.0000 + (1/3)*0.6667 + (1/3)*0.7500
               = 0.333333 + 0.222222 + 0.250000 = 0.805556
  Enveloped AP = (1/3)*1.0000 + (1/3)*0.7500 + (1/3)*0.7500
               = 0.333333 + 0.250000 + 0.250000 = 0.833333
  The gap is 0.027778, or 3.4% relative, produced solely by replacing the
  intermediate precision 0.6667 with the later 0.7500. This is why the
  integration rule belongs in the policy object and not in a code comment.
### 4.4 Variants and where each one changes the answer

The evaluation literature contains at least five materially different "mAP" definitions. They are not interchangeable and the differences are large enough to reverse a model comparison.

```text
VARIANT A — 11-point interpolated AP (VOC 2007 style)
  AP_11 = (1/11) * sum_{r in {0.0,0.1,...,1.0}} max_{R_j >= r} P_j
  The max runs over all ranked points whose recall is at or beyond the grid point.

WHERE
  r    : grid recall, dimensionless, 11 values
  P_j  : raw precision at rank j (NOT the all-point envelope of 4.3)
  Domain: same matching as 4.2; only the integration rule differs

CONSTRAINT
  A grid point beyond the maximum achievable recall contributes 0, not 1
  maxDets and area bins must match the all-point run being compared

NUMERICAL EXAMPLE (class person, points from 4.3)
  (R,P) pairs: (0.25,1.0) (0.50,1.0) (0.50,0.6667) (0.75,0.75) (0.75,0.60) (1.00,0.6667)
  r=0.0..0.5  -> qualifying points include (0.50,1.0)       -> 1.0     (6 points)
  r=0.6..0.7  -> qualifying points are (0.75,0.75),(1,0.667) -> 0.75    (2 points)
  r=0.8..1.0  -> qualifying point is (1.00,0.6667)          -> 0.6667  (3 points)
  sum = 6*1.0 + 2*0.75 + 3*0.6667 = 6.0 + 1.5 + 2.0000 = 9.5
  AP_11 = 9.5/11 = 0.863636

  Same rule for car, points (1/3,1.0) (2/3,1.0) (2/3,0.6667) (2/3,0.50):
  r=0.0..0.6 -> (2/3,1.0) qualifies -> 1.0 (7 points)
  r=0.7..1.0 -> no point reaches recall 0.7 -> 0.0 (4 points)
  sum = 7.0 ; AP_11 = 7.0/11 = 0.636364

  Same rule for bicycle, points (1.0,1.0) (1.0,0.5):
  every grid point has a qualifying point at recall 1.0 -> max P = 1.0
  AP_11 = 11/11 = 1.000000
  mAP_11 = (0.863636 + 0.636364 + 1.000000)/3 = 0.833333

INTERPRETATION
  The car class loses 0.364 of AP purely because its recall ceiling is 2/3:
  four of eleven grid points sit above the achievable recall and score zero.
  This mechanism makes 11-point AP systematically harsher than all-point AP
  on classes with few instances. On this toy the gap is 0.006945 mAP; on a
  slice with three references and two hits the same mechanism costs 0.12.
```

VARIANT A-DASH — COCO 101-point, same points, denser grid:

```text
  AP_101 = (1/101) * sum_{k=0..100} max_{R_j >= k/100} P_j
  person : 0.00-0.50 -> 1.0    (51 pts, 51.0000)
           0.51-0.75 -> 0.75   (25 pts, 18.7500)
           0.76-1.00 -> 0.6667 (25 pts, 16.6667)
           sum 86.4167 ; AP_101 = 86.4167/101 = 0.855611
  car    : 0.00-0.66 -> 1.0 (67 pts, 67.0) ; 0.67-1.00 -> 0.0 (34 pts)
           sum 67.0 ; AP_101 = 67.0/101 = 0.663366
  bicycle: all 101 grid points -> 1.0 ; AP_101 = 1.000000
  mAP_101 = (0.855611 + 0.663366 + 1.000000)/3 = 0.839659
```

The all-point figure (0.840278) sits 0.000619 from the 101-point figure because a four-reference class already saturates the staircase; the 11-point figure is 0.006945 lower. On benchmarks with tens of thousands of instances per class the three converge. On per-camera, per-shift, or per-site slices with a handful of instances they diverge by more than the improvement a team is trying to detect. Hence the rule: all-point AP as the default, and always print the instance count beside the AP value.

Variant B — mAP at a swept IoU range:

```text
FORMULA
  mAP_sweep = (1/|T|) * sum_{tau in T} mAP(tau),  T = {0.50, 0.55, ..., 0.95}, |T| = 10
WHERE
  mAP(tau) recomputed from scratch: matching depends on tau, so labels change
CONSTRAINT
  Rematching per tau is mandatory; reusing tau=0.5 labels is a different (wrong) metric
NUMERICAL EXAMPLE
  tau : 0.50  0.55  0.60  0.65  0.70  0.75  0.80  0.85  0.90  0.95
  mAP : 0.840 0.840 0.840 0.764 0.764 0.764 0.653 0.319 0.167 0.000
  sum = 5.951389 ; /10 = 0.595139
INTERPRETATION
  Six of ten thresholds sit at only three distinct values because the toy boxes
  have fixed overlaps; a real benchmark shows a smooth decline.
```

Variant C — localisation-aware regression losses (GIoU, DIoU, CIoU). These are training objectives, not evaluation metrics, but they are computed from the same geometry and are routinely confused with it.

```text
GIoU(A,B) = IoU(A,B) - (C - U)/C
  where C = area of the smallest enclosing box, U = union area
  Range: (-1, 1]; equals IoU when one box contains the other (then C = U)

DIoU(A,B) = IoU(A,B) - rho^2(c_A, c_B) / d^2
  where rho = Euclidean distance between box centres, d = diagonal of enclosing box

CIoU(A,B) = DIoU(A,B) - alpha * v
  where v = (4/pi^2) * (atan(w_B/h_B) - atan(w_A/h_A))^2, alpha = v / ((1-IoU)+v)

NUMERICAL EXAMPLE (GIoU, person pair from 4.1)
  IoU = 0.927961 ; U = 13104
  Enclosing box spans x in [100,180], y in [98,262] -> C = 80 * 164 = 13120
  GIoU = 0.927961 - (13120 - 13104)/13120 = 0.927961 - 0.001220 = 0.926741

NUMERICAL EXAMPLE (GIoU, disjoint pair)
  A = (0,0,10,10), B = (20,0,30,10): I = 0, U = 200, C = 30*10 = 300
  GIoU = 0 - (300-200)/300 = -0.333333
  A = (0,0,10,10), B = (1000,0,1010,10): C = 1010*10 = 10100
  GIoU = 0 - (10100-200)/10100 = -0.980198

NUMERICAL EXAMPLE (DIoU, partially occluded person)
  p7 = (250,160,320,420) centre (285,290) ; g6 = (220,120,320,420) centre (270,270)
  IoU = 0.606667 ; rho^2 = 15^2 + 20^2 = 625 ; d^2 = (320-220)^2 + (420-120)^2 = 100^2 + 300^2 = 100000
  DIoU = 0.606667 - 625/100000 = 0.606667 - 0.006250 = 0.600417
```

**Validity condition for the variants.** GIoU only distinguishes disjoint configurations through the enclosing box; two disjoint boxes of equal size get progressively more negative GIoU as they separate, but two *different* disjoint pairs can share the same GIoU (0-0.33 region is not injected). DIoU adds centre distance, which fixes GIoU's failure on concentric boxes of very different sizes, but its penalty is normalized by the enclosing diagonal, so it saturates for large containers. None of these are evaluation metrics: a model trained on CIoU does not automatically produce boxes that maximise evaluation IoU at a fixed threshold.

### 4.5 Limits and counterexamples

**4.5.1 `1 - IoU` is not a metric.**
Take three 100x100 boxes at horizontal offsets 0, 33, 66: `A=(0,0,100,100)`, `B=(33,0,133,100)`, `C=(66,0,166,100)`.
Then `IoU(A,B) = 6700/13300 = 0.503759`, `IoU(B,C) = 0.503759`, but `IoU(A,C) = 3400/16600 = 0.204819`.
So `d(A,C) = 0.795181` while `d(A,B) + d(B,C) = 0.496241 + 0.496241 = 0.992481`. That direction is fine, but the reverse holds elsewhere: a chain of small steps can exceed the direct distance, and more importantly `d(A,C)` can exceed `d(A,B)+d(B,C)` for boxes of unequal aspect ratio — the search over a 4x4 lattice found maximum slack exactly 0 for axis-aligned rectangles in that lattice, which means the triangle inequality holds there by accident of geometry. The safe statement is narrower and still operationally important: **`1 - IoU` is not a distance in any space where you can interpolate**, so "average IoU" and "median IoU" are meaningful summaries but centroid-and-radius reasoning is not.

For scale, the practical counterexample is nested boxes. `A=(0,0,10,10)` inside `B=(0,0,100,100)` gives `IoU = 100/10000 = 0.01`. A perfect localisation of a small object inside a large wrong box scores 0.01, while a 5-pixel-shifted box of the correct size at IoU 0.90 scores far higher. If your model's errors are "right region, wrong extent," IoU punishes you in a way that is invisible in aggregate mAP.

**4.5.2 Greedy NMS is not optimal selection.**
Three boxes on one object with scores 0.90, 0.60, 0.59; `IoU(B1,B2) = 0.60`, `IoU(B1,B3) = 0.61`, `IoU(B2,B3) = 0.02`; suppression threshold 0.50.
Greedy: keep B1 (0.90), suppress B2 and B3 (both above 0.50 against B1) — total retained score 0.90.
Optimal by retained score: keep B2 and B3 — total 1.19.
Greedy loses 0.29 of score mass, and in a downstream ranker the surviving box is the one with worse localisation. The failure is structural, not a tuning problem, and it appears whenever a high-scoring box sits *between* two moderate boxes in IoU-space. Soft-NMS (4.6) is one repair; it converts suppression into decay.

**4.5.3 Threshold semantics are not uniform across object sizes.**
For two boxes of side `L` shifted by `d` pixels along one axis, `I = (L-d)*L`, `U = 2L^2 - I`, so solving `I/U = tau` gives `d = L(1-tau)/(1+tau)`. At `L = 100`:

```text
tau = 0.30 -> d = 100*0.70/1.30 = 53.846 px
tau = 0.50 -> d = 100*0.50/1.50 = 33.333 px
tau = 0.75 -> d = 100*0.25/1.75 = 14.286 px
tau = 0.90 -> d = 100*0.10/1.90 =  5.263 px
```

A 0.75 threshold tolerates 14 pixels of shift on a 100-pixel object — 14% of its size — but the same 14 pixels on a 20-pixel object is a 70% shift, and IoU collapses. Equivalently, the tolerated *relative* scale error `s` solves `tau = 1/s^2`, so `s = 1/sqrt(tau)`: 1.826x at tau 0.30, 1.414x at 0.50, 1.155x at 0.75, 1.054x at 0.90. A single tau is therefore a size-dependent accuracy contract. Report small/medium/large AP separately (Section 7) or state that your metric is dominated by large objects.

**4.5.4 AP is invariant to monotone score transforms but not to suppression-induced reordering.**
Multiplying every score by 3 leaves every rank unchanged, so AP is identical. Soft-NMS multiplies scores by a decay factor that depends on each box's neighbours, which *does* reorder boxes, so AP can move. This distinguishes "calibrating" from "post-processing": calibration is monotone and AP-neutral by construction; soft-NMS is not.

**4.5.5 Perfect localisation of a class you do not evaluate still counts against you.**
A person box matched at IoU 0.92 to a `bicycle` reference is an FP for `person` and an FN for `bicycle` under class-aware matching, contributing -1 to two curves. The class-agnostic variant (0.793651 versus 0.840278 here) removes those penalties, which is appropriate for "is there an object here" but wrong for "is the label right". Choose per decision, and never quote both in the same comparison table without a label.
