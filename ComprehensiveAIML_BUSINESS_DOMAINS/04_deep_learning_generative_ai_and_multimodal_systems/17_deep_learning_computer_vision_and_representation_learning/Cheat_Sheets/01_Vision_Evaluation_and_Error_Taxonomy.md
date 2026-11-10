---
collection: "module-17-vision"
file_id: "01"
topic: "Vision evaluation and error taxonomy"
level: 1
status: "draft"
version: "1.2"
created: "2026-11-08"
reviewed: null
review_trigger: "Task contract, label policy, matching rule, capture device, operating threshold, or review capacity changes"
notation_style: "plain-text"
prerequisites: ["02_CNNs_Attention_and_Transfer_Learning.md", "probability basics, pixel/box coordinate conventions"]
pattern_tags: ["vision", "evaluation", "error-taxonomy"]
source_status: "OPEN — visible inventory names only; original source bodies unavailable"
owner_role: "ML evaluation owner"
source_index: "INDEX.md"
---

# Vision Evaluation and Error Taxonomy

> **Purpose:** Convert a visual model score into a versioned decision contract — unit, cohort, deadline, matching rule, threshold, slices — then attribute each failure to a mechanism before anyone retrains anything.
> **Prerequisites:** [CNN geometry and transfer baseline](02_CNNs_Attention_and_Transfer_Learning.md), probability basics, box/mask coordinate conventions.
> **Key outcome:** Given a visual task and a workflow, you can write the evaluation contract, compute threshold-dependent counts, decompose detection errors at a fixed operating point, size the review queue, and defend a bounded intervention.
> **This file does NOT cover:** clinical validation, safety certification, detector architecture selection (see [04](04_Detection_and_Segmentation.md)), or any measured benchmark claim.
> **Level:** 1 Foundations.
> **Version / reviewed:** 1.2 / not reviewed.
> **Evidence:** DERIVED metric algebra; VERIFIED illustrative arithmetic; SOURCED metric conventions (VOC/COCO/PQ); no local measurement.
> **Retrieval cue:** If a score improved but the workflow worsened, start at Section 8 and use the F-06/F-08/F-11 chain.

## 1. File Header

Evaluation is the act of fixing what "correct" means at the moment a person or a downstream system must act. It is not the last dashboard before launch. A classifier can show excellent image-level accuracy and still be unusable: its false positives exceed review capacity, its misses concentrate in one camera model, or its test set contains frames from recordings that also appear in training. None of those failures is visible in an aggregate accuracy number, because the aggregate was never the thing the workflow needed.

This file teaches four artifacts, and every later section serves one of them.

| Artifact | Question it freezes | Fails when |
|---|---|---|
| Cohort manifest | Which records are eligible, and by which dependency group are they split? | Eligibility is reconstructed after the score is known |
| Metric contract | Which unit, matching rule, threshold, aggregation, and support define the number? | Any of those six drifts between two reports |
| Slice board | Which mechanism-linked cohorts are blocking gates rather than decoration? | Slices are chosen after seeing where the model looks bad |
| Error register | Which mechanism produced each failure, and which test discriminates it? | The register lists examples without categories |

Ownership is deliberately split. The evaluation owner owns the metric contract and the promotion gate. The data owner owns label provenance, maturity, and split eligibility. The workflow owner owns what happens after accept, reject, and abstain. A role may not replace another role's contract with a scalar; that substitution is the most common way an evaluation becomes unfalsifiable.

**Contract version vector.** Treat an evaluation as a tuple, not a number:

```text
V = (unit, cohort, deadline, label_policy, matching, threshold, aggregation, slices)
```

Two reports are comparable only when their tuples agree on every component that can move the estimator. "Model B beats model A by 3 points" is not a claim about models if the two tuples differ in `matching` or `aggregation`. Write the tuple into the report header, and diff the tuples before diffing the numbers.

**Where the numbers in this file come from.** All workloads, costs, prevalences, and reviewer rates are ILLUSTRATIVE constructions with arithmetic the reader can recompute. Metric definitions such as intersection over union, average precision interpolation, COCO size bands, and panoptic quality are SOURCED conventions from the named primary papers; they are conventions, not laws, and a workflow may legally choose others as long as it says so. No value in this file is a benchmark of any product, dataset, or device, and the source inventory behind this collection records visible titles only.

**What the reader must be able to do afterwards.** Define the decision unit for four task shapes; compute IoU, precision, recall, AP, ECE, and PQ from counts; explain why AP is not additive across slices; choose a threshold under a cost model and a capacity model simultaneously; distinguish six competing causes of a recall drop; and reject a promotion whose cohort or matching rule cannot be reconstructed.

**Takeaway:** Name the tuple before naming the number, because an unnamed protocol makes every later comparison ambiguous.

## 2. Concept Map / Mental Model

The whole discipline is one pipeline with three feedback loops. Read the map top to bottom; every arrow that points left is a place where an evaluation can become invalid without any model changing.

```text
capture event  --->  eligibility filter  --->  cohort manifest (frozen IDs, groups)
   |                        |                          |
   | device, exposure,      | exclusions must be       | group key: patient, device,
   | firmware, timestamp    | pre-declared             | scene, recording, site, day
   v                        v                          v
label policy  -------->  ground truth  ----------> dependency-aware split
(policy, maturity,        (with ignore/             (train | val | test; never
 annotator, tool)          ambiguity marks)          re-split after scoring)
                                |
                                v
                    prediction at declared deadline
                                |
                                v
              +-----------------+------------------+
              |                                    |
      matching rule (class, IoU,          score -> action rule
      one-to-one, ignore, cap)            (threshold, abstain band)
              |                                    |
              v                                    v
      TP / FP / FN counts  ------------>  operating point on PR curve
              |                                    |
      +-------+--------+                           |
      |                |                           v
 aggregate metric   slice table  ----------> review queue + backlog
      |                |                           |
      v                v                           v
  decision gate  <---  error register  <---  reviewer dispositions
      |                |
      |                +--> taxonomy share (with sampling weights)
      v
  promote | hold | roll back | collect evidence
```

Three loops matter. The **label loop** returns from reviewer dispositions to the label policy: when reviewers disagree with the model, the disagreement may be about the policy, not the model. The **threshold loop** returns from queue behaviour to the operating point: an alert rate that the team cannot process is a gate failure even if recall improved. The **taxonomy loop** returns from error categories to the next experiment: a localization-dominated register argues for a different intervention than a classification-dominated one, and treating both as "the model is bad" wastes a training cycle.

**The central invariant.** Every reported quantity must be reconstructible from immutable cohort IDs plus the frozen tuple. If a reader cannot rebuild the numerator and denominator, the report is an anecdote. This invariant is stronger than reproducibility of code: identical code over a re-filtered population produces a different estimand.

**What changes between variants.** Classification versus detection changes the unit and the matching rule; detection versus segmentation changes the cardinality from objects to pixels; single-site versus multi-site changes the split key and the number of slices worth testing; offline batch review versus real-time gating changes the deadline and therefore which features are even legal. The algebra of counts stays the same across all of them — that is exactly why mixing units silently is so easy and so damaging.

**The most important boundary.** A metric is an estimator of a decision-relevant quantity under a protocol. Changing the protocol changes the estimand, so a "better" number under a changed protocol is not evidence of a better model. Section 4.4 makes this precise with a detection cap that moves AP from 0.729 to 0.549 while every prediction, label, and weight is untouched.

**Why the map runs top to bottom in that order.** Each stage can invalidate everything below it, and nothing below can repair a stage above. If the cohort is contaminated, no matching rule makes the score honest. If the matching rule is undisclosed, no slice table localises anything. If the threshold is copied from another site, no taxonomy entry explains the resulting alert rate. The practical consequence is a rule about where to spend diagnostic effort: always start from the stage furthest up the map that could have changed, and only descend when it is cleared. Most incident reviews descend immediately to "the model", which is the second-to-last stage, and spend their time on the one component that a protocol failure makes irrelevant.

A worked illustration of stage ordering. A site reports that recall "dropped from 0.78 to 0.61 overnight". Walking the map: capture unchanged (same firmware digest), eligibility changed (43 images excluded instead of 12, because a metadata field went null and the join dropped them), cohort therefore changed, and the excluded 31 images were disproportionately positive. The matching rule never changed, the model never changed, and the threshold never changed. The correct action is to repair the join and recompute, not to investigate the representation. The map's ordering is what makes that conclusion reachable in ten minutes rather than ten days.

**The map is also a communication device.** When a stakeholder asks "is the model good?", the useful reply walks the map: "for these 1,000 eligible records, split by machine, at this matching rule and this threshold, recall is 0.76 with an interval of 0.73–0.79, and on the small-defect band that your action depends on it is 0.40 with an interval of 0.35–0.46, and the review queue at this threshold runs at 96% utilisation." That answer is long, but every clause is a decision someone can act on. The one-word answer is short because it has discarded all of them.

**Takeaway:** Treat the tuple as the object under version control; the score is only its projection.

## 3. Foundational Definitions

The definitions below are ordered so that each one depends only on earlier rows. The rightmost column names the confusion each term is usually mistaken for.

| Term | Formal meaning | Units / type | Valid domain | Common confusion |
|---|---|---|---|---|
| Capture event | One immutable acquisition with device, time, and settings | event | must exist before labelling | a re-export of the same event |
| Unit of decision | The entity receiving exactly one action | event | exactly one per row of the metric table | the batch, the folder, the request |
| Eligible population | Records satisfying pre-declared inclusion rules at a cut-off | count | fixed before scoring | whatever rows survived cleaning |
| Dependency group | Set of records that must not straddle a split boundary | set | key chosen from serving reality | an arbitrary hash of file path |
| Ground truth | Label produced under a stated policy, tool, and maturity date | categorical / geometry | policy and date recorded | an operational disposition made later |
| Prediction deadline | Latest instant at which inputs may be known | timestamp | precedes the action | adjudication time |
| Matching | Rule assigning predictions to labels one-to-one under a class and overlap constraint | procedure | declared, deterministic | "closest box" applied by eye |
| IoU | Intersection cardinality divided by union cardinality | dimensionless, [0,1] | union cardinality positive, or convention stated | Dice, pixel accuracy |
| TP / FP / FN | Matched accepted prediction / unmatched accepted prediction / unmatched eligible label | counts | non-negative integers | percentages, or per-image flags |
| Ignore region | Area or label excluded from both numerator and denominator | mask / flag | policy declared | a false negative |
| Support | Number of eligible labels or events in a slice | count | reported beside every rate | statistical power |
| Slice | Pre-specified subset tied to a mechanism or obligation | subset | declared before scoring | post-hoc subgroup chosen for a story |
| Operating threshold | Rule mapping score to accept / abstain / reject | score or probability | selected against harm and capacity | a universal 0.5 |
| Calibration | Agreement between predicted probability and observed frequency in a stated population | dimensionless | population and regime recorded | ranking quality |
| Label maturity | Whether the outcome has had time to become observable | duration | recorded per record | label correctness |
| Error taxonomy | Mutually usable categories, each mapped to a discriminating test and a bounded fix | register | categories exhaustive enough to close | a gallery of screenshots |

Four of these definitions do most of the work, and each deserves a paragraph because the failure it prevents is expensive.

**Unit of decision.** For image classification the unit is the image; for detection it is the object instance; for segmentation it is the pixel or region; for a triage workflow it may be the study or the case, which can contain several images. A model may predict per image while the workflow acts per case. In that situation the metric must be computed at the *action* grain, which usually requires a case-level aggregation rule ("alert if any image exceeds the threshold"). Reporting image-level recall for a case-level action overstates how many cases the workflow catches, because a case with five images has five chances to trigger and the aggregation rule decides the consequence.

**Ground truth versus operational disposition.** A disposition is what a person did at the time, possibly under time pressure, incomplete information, and an incentive to close the queue. Ground truth is a label produced under an explicit policy with a named reviewer role and a maturity date. Confusing the two produces a model that learns to imitate queue behaviour and an evaluation that rewards it.

**Ignore regions.** Detectors and segmentation models are commonly evaluated with crowd regions, unlabelled borders, and "don't care" zones removed from both counts. Omitting the policy turns every prediction inside a crowd region into a false positive and every unlabelled object into a false negative, which can dominate the metric in dense scenes. The check is mechanical: list the ignore masks, count pixels or objects removed, and confirm the count is nonzero in the slices you care about.

**Label maturity.** In workflows where the outcome arrives later (a defect confirmed at a downstream station, a lesion confirmed by follow-up), a record labelled "negative" today may be positive in three weeks. An evaluation run before maturity measures agreement with an incomplete outcome. The remedy is a maturity cut-off in the cohort manifest, not an after-the-fact correction.

Two further distinctions prevent arithmetic errors. First, a **count** and a **rate** are different objects: counts add across slices, rates do not. Second, **conditional** quantities such as precision (conditioned on prediction) and recall (conditioned on label) answer different operational questions and cannot be traded without a cost model, which Section 4.3 supplies.

**Disambiguation drill.** Six statements that sound alike and mean different things. For each, the right-hand column gives the definition that must be applied before any arithmetic.

| Statement | What it actually asserts | Definition to apply |
|---|---|---|
| "The model is 95% accurate" | 95% of units received the correct label at one threshold | Accuracy; check prevalence and the constant-predictor baseline first |
| "The model finds 95% of defects" | 95% of eligible labelled defects were matched at one threshold and IoU | Recall; check label completeness, cap, and maturity |
| "95% of alerts are real" | 95% of accepted predictions matched a label | Precision; check the threshold and the accepted count |
| "The model is 95% confident" | The mean predicted probability is 0.95 on some population | Nothing yet — confidence is not accuracy and not calibration |
| "The model is well calibrated" | Predicted probability tracks observed frequency in stated bins on a stated population | Calibration; check binning, support, and population match |
| "The model agrees with the reviewer 95% of the time" | 95% of units match an operational disposition | Agreement with a disposition, not ground truth; check reviewer independence and policy |

The last row is the most dangerous because it sounds like the strongest. Agreement with a disposition can be 0.95 while agreement with adjudicated ground truth is 0.80, if the reviewer was rushed or the policy was applied inconsistently. Agreement is a measurement of the reviewer-plus-model system, and it is only a model metric when the disposition itself was produced under the label policy.

**A definition test that fails loudly.** Take any report and try to answer three questions from it alone: how many units were eligible, how many were accepted, and which rule assigned an accepted prediction to a label. If any answer requires opening a notebook, the report has not defined its own terms. This test is cheap, mechanical, and catches the majority of unpublishable evaluations before they reach a decision meeting.

**Takeaway:** Define the action grain and the label policy first, because every later count inherits both.

## 4. Core Formulas with Full Derivation

Every block below states validity before use, derives the expression from a definition, then gives the five-part record. Equation IDs are stable and are referenced from Section 11 answers.

### 4.1 Base case: one prediction against one label

The base case is a single prediction compared with a single label under one class and a declared coordinate frame. Everything later in the file is this case plus bookkeeping.

**EV-01 — Intersection over union.**

Validity before use: both regions are expressed in the same coordinate frame, the same class, and the same annotation policy; the union has positive cardinality or an explicit convention is declared for the empty case.

Derivation: inclusion–exclusion gives `|P ∪ G| = |P| + |G| − |P ∩ G|`. Substituting the intersection `I` and dividing gives the ratio. If the regions are raster sets, cardinality is a pixel count and the decomposition `|P| = TP + FP`, `|G| = TP + FN`, `I = TP` yields the count form, which is the same algebra with object counts.

```text
[1] FORMULA: IoU = |P ∩ G| / |P ∪ G| = I / (P + G - I) = TP / (TP + FP + FN)
[2] WHERE: P predicted region (pixels or objects); G ground-truth region;
           I = |P ∩ G|; TP matched region; all counts in one frame and class.
[3] CONSTRAINT: P + G - I > 0; identical coordinate convention on both sides;
           ignore regions already removed. Empty-on-both-sides needs a protocol.
[4] NUMERICAL EXAMPLE: box A = [10,10,110,110], box B = [60,60,160,160].
           widths = 100, heights = 100, areas = 10,000 each.
           x overlap = 110 - 60 = 50; y overlap = 110 - 60 = 50; I = 2,500.
           union = 10,000 + 10,000 - 2,500 = 17,500; IoU = 2,500/17,500 = 0.142857.
[5] INTERPRETATION: at a 0.50 matching threshold this pair is a localization failure;
           the number itself does not say whether the box is shifted, scaled, or wrong.
```

Boundary: two identical 10×10 boxes drawn in inclusive pixel coordinates cover 121 pixels, and the same pair in half-open coordinates covers 100. Identical conventions cancel; mixed conventions do not. If the prediction uses inclusive edges and the label uses half-open edges, the intersection is the smaller region, 100, the union is 121, and IoU falls to `100/121 = 0.826` for an object that a human would call perfectly localized. For a 100×100 object the same one-pixel disagreement costs only `10,000/10,201 = 0.980`. A convention error is therefore a *size-dependent* recall loss that looks like a model weakness on small objects — one of the most common misdiagnoses in detection review.

**EV-02 — Dice and IoU are the same measurement in different clothes.**

Validity before use: the same region pair, with `U = P + G − I > 0`.

Derivation: Dice is defined as `2I/(P + G)`. Write `P + G = U + I` by the definition of `U`. Then `Dice = 2I/(U + I)`. Divide numerator and denominator by `U` and substitute `IoU = I/U`, giving `2·IoU/(1 + IoU)`. The map is strictly increasing on `[0,1]`, so it preserves the ordering of pairs but not the spacing and not the threshold meaning.

```text
[1] FORMULA: Dice = 2*IoU / (1 + IoU), with inverse IoU = Dice / (2 - Dice)
[2] WHERE: IoU and Dice both dimensionless in [0,1]; same region pair and frame.
[3] CONSTRAINT: U > 0; the identity holds per pair, not for averaged scores.
[4] NUMERICAL EXAMPLE: IoU = 1/7 = 0.142857 gives Dice = (2/7)/(8/7) = 0.25.
           Inverse check: IoU = 0.25/(2 - 0.25) = 0.25/1.75 = 0.142857. Consistent.
[5] INTERPRETATION: a team that reports "Dice 0.25" and a team that reports
           "IoU 0.143" may be describing identical geometry; compare only after
           converting, and never compare an averaged Dice with an averaged IoU.
```

Boundary: the *mean* of Dice is not the Dice of the mean IoU, because the map is nonlinear. Averaging first and converting second produces a different number than converting each pair and averaging; state which you did.

**EV-03 — Counts to precision, recall, and their Fβ family.**

Validity before use: positives, negatives, ignore policy, unit, and threshold are frozen, and every eligible label is accounted for exactly once as TP or FN.

Derivation: accepted predictions partition into matched and unmatched, so `accepted = TP + FP`. Similarly eligible labels partition into matched and unmatched, so `positives = TP + FN`. Each rate is the matched count divided by its conditioning total. `Fβ` is defined as the weighted harmonic mean with `β²` weighting recall, and simplifying the harmonic mean gives the standard closed form.

```text
[1] FORMULA: precision = TP/(TP+FP); recall = TP/(TP+FN)
             F_beta = (1+beta^2)*P*R / (beta^2*P + R)
[2] WHERE: TP, FP, FN are counts at one threshold under one matching rule;
           P and R dimensionless; beta > 0 dimensionless.
[3] CONSTRAINT: TP+FP > 0 and TP+FN > 0; else the rate is undefined and the
           protocol must say "undefined", not "zero". beta=1 gives F1.
[4] NUMERICAL EXAMPLE: TP=150, FP=50, FN=90.
           precision = 150/200 = 0.75; recall = 150/240 = 0.625.
           F1 = 2*0.75*0.625/(0.75+0.625) = 0.9375/1.375 = 0.681818...
           F2 = 5*0.75*0.625/(4*0.75+0.625) = 2.34375/3.625 = 0.646552.
[5] INTERPRETATION: with beta=2 the same counts score lower than F1, because F2
           weights the weaker recall more heavily; the choice of beta is a
           statement about which error the workflow punishes, not a preference.
```

Boundary: `TP+FP = 0` at a very high threshold. Reporting precision as 0 makes a model that abstains look maximally wrong; reporting it as 1 makes it look perfect. Both are inventions. The protocol must declare the convention and the support.

**Takeaway:** One pair, one overlap, one count decomposition — every later metric is bookkeeping over this base case.

### 4.2 General case: many predictions, many labels, one protocol

The general case replaces "one pair" with "a set of predictions and a set of labels" and adds four decisions: which pairs are admissible, how pairs are chosen, how detections are ordered, and how scores are swept. Everything else is arithmetic over the resulting counts.

**Admissibility.** A pair `(p, g)` is admissible when the class matches under the declared class map and `IoU(p,g) ≥ τ`. The class map matters more than people expect: a fine-grained label set (twelve fastener types) mapped to a coarse one (present/absent) changes the admissible set, and evaluating a coarse model against fine labels without a map manufactures both false positives and false negatives from pure ontology mismatch. Record the map as data, not as code.

**Pairing rule.** Two conventions are common and they disagree:

| Rule | Definition | Deterministic | Gives |
|---|---|---|---|
| Greedy by score | Sort detections by descending score; assign each to the admissible unmatched label with highest IoU | Yes, given a tie-break | Lower bound on matches under most definitions, cheap |
| Optimal cardinality | Maximize the number of admissible pairs (bipartite matching) | Yes | Maximum TP count |
| Optimal total overlap | Maximize total IoU over matched pairs | Yes | Higher localization reward, possibly fewer matches |

Greedy and optimal cardinality can differ. Constructed case: labels G1, G2; predictions P1 (score 0.9) overlapping both G1 and G2 at IoU 0.55 and 0.52; P2 (score 0.8) overlapping only G1 at IoU 0.80. Greedy assigns P1→G1 (highest IoU), then P2 finds G1 taken and G2 inadmissible, so TP = 1. Optimal cardinality assigns P1→G2 and P2→G1, so TP = 2. Same model, same labels, one TP different, purely from the pairing rule. Any comparison across papers or vendors must state the rule.

**EV-04 — Average precision as a sum over recall increments.**

Validity before use: a single class, a fixed IoU threshold, a fixed detection cap, a complete label set for the evaluated images, and detections sorted by descending score with ties broken consistently.

Derivation. Let the sorted detections have cumulative true-positive counts `TP_k` and precision `P_k = TP_k/k` at rank `k`. Recall after rank `k` is `R_k = TP_k/N_pos`, a non-decreasing step function with jumps only at true positives. The precision–recall curve is the graph `(R_k, P_k)`. Average precision is the area under the *monotone envelope* of that curve, which we obtain by enforcing `P*_k = max_{j ≥ k} P_j`. Writing the area as a Riemann–Stieltjes integral over the step function gives `AP = ∫ P* dR`, and because `R` jumps by `1/N_pos` exactly at each true positive, the integral collapses to a sum over true-positive ranks:

```text
AP = (1/N_pos) * sum over true positives k of P*_k
```

**EV-04 block.**

```text
[1] FORMULA: AP = (1/N_pos) * SUM_{k in TP ranks} max_{j >= k} P_j
                 = SUM_k (R_k - R_{k-1}) * P*_k
[2] WHERE: N_pos eligible labels for the class after ignore removal and cap;
           P_j = TP_j / j; R_k in [0,1]; AP dimensionless in [0,1].
[3] CONSTRAINT: single class, single IoU threshold, single cap, complete labels;
           ties in score must be broken by a declared rule (e.g. by image id).
[4] NUMERICAL EXAMPLE: N_pos = 5; 8 detections with flags TP,FP,TP,TP,FP,TP,FP,FP.
           k=1 P=1.000 R=0.2 | k=2 P=0.500 R=0.2 | k=3 P=0.667 R=0.4
           k=4 P=0.750 R=0.6 | k=5 P=0.600 R=0.6 | k=6 P=0.667 R=0.8
           k=7 P=0.571 R=0.8 | k=8 P=0.500 R=0.8
           envelope at TP ranks: 1.000, 0.750, 0.750, 0.667
           AP = (1.000+0.750+0.750+0.667)/5 = 3.167/5 = 0.6333
           cross-check via increments: 0.2*(1.000) + 0.2*(0.750)
           + 0.2*(0.750) + 0.2*(0.667) = 0.2*3.167 = 0.6333. Agrees.
[5] INTERPRETATION: AP rewards ranking quality across the whole score range, so it
           can rise while the chosen operating point's recall is unchanged. It is
           a summary of a curve, not a description of the deployed threshold.
```

The same eight detections with the older 11-point interpolation give `(3.0 + 3.0 + 1.3334)/11 = 0.667`, about 0.033 higher. That gap is not model improvement; it is a different interpolation rule. VOC2007 used 11 points, VOC2010 and COCO use all-point or dense sampling, and a vendor that "improved" from 0.633 to 0.667 may have changed nothing but the script. Record the interpolation in the metric contract.

**EV-05 — Mean average precision and why macro is the default.**

Validity before use: each class has an independently computed AP with the same IoU threshold, cap, and ignore policy; classes with zero eligible labels are excluded and counted.

Derivation: mAP is the unweighted arithmetic mean of per-class AP. The unweighted choice is deliberate: it makes each class contribute equally to the summary so that a rare class is not drowned by a frequent one. Support-weighted aggregation is a different estimand and must be labelled as such.

```text
[1] FORMULA: mAP = (1/C) * SUM_{c=1..C} AP_c     (macro)
             mAP_w = SUM_c (n_c * AP_c) / SUM_c n_c   (support-weighted)
[2] WHERE: AP_c per class; n_c eligible labels of class c; C classes with n_c > 0.
[3] CONSTRAINT: identical protocol per class; report C and the excluded empty classes.
[4] NUMERICAL EXAMPLE: AP = 0.633, 0.420, 0.880 with supports 900, 100, 1000.
           macro mAP = (0.633+0.420+0.880)/3 = 1.933/3 = 0.6443
           weighted  = (900*0.633 + 100*0.420 + 1000*0.880)/2000
                     = (569.7 + 42.0 + 880.0)/2000 = 1491.7/2000 = 0.74585
[5] INTERPRETATION: the two aggregations differ by 0.102 on identical per-class
           results, so "mAP 0.746" and "mAP 0.644" can describe the same model.
```

**EV-06 — Panoptic quality as a product of recognition and segmentation quality.**

Validity before use: instance masks with a one-to-one matching rule at `IoU > 0.5`, where a matched pair counts once even if a prediction overlaps several labels, and unmatched predictions and labels are counted as errors.

Derivation: let `TP` be matched pairs, `FP` unmatched predicted segments, `FN` unmatched labelled segments, and `SQ` the mean IoU over matched pairs. Recognition quality is the F1-style ratio `RQ = TP/(TP + ½FP + ½FN)`. Because each unmatched segment contributes half an error, `TP + ½FP + ½FN` is the effective error mass. Multiplying mean overlap by this ratio distributes the overlap reward only over recognized segments:

```text
[1] FORMULA: PQ = SQ * RQ where SQ = (1/|TP|) SUM_{(p,g) in TP} IoU(p,g)
                         and RQ = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
[2] WHERE: |TP| matched segment pairs; |FP|,|FN| unmatched segments; SQ,RQ,PQ in [0,1].
[3] CONSTRAINT: IoU > 0.5 matching with one-to-one assignment; void/non-void policy
           declared; empty-image handling declared.
[4] NUMERICAL EXAMPLE: |TP|=40, |FP|=10, |FN|=6, mean IoU over matches = 0.82.
           RQ = 40/(40 + 5 + 3) = 40/48 = 0.83333
           PQ = 0.82 * 0.83333 = 0.68333
           direct: (40*0.82)/48 = 32.8/48 = 0.68333. Agrees.
[5] INTERPRETATION: PQ punishes a model that segments well but invents segments,
           because SQ alone would stay at 0.82 while PQ falls to 0.683.
```

Boundary: PQ is not comparable across datasets with different class granularity, because merging two classes changes `TP`, `FP`, and `FN` simultaneously. And unlike AP, PQ is defined at a single operating threshold; a PQ without a stated confidence threshold describes an unfiltered output set.

**Takeaway:** Pairing rule, interpolation, and aggregation are three independent switches; state all three or the number is not a measurement.

### 4.3 Variants: calibration, cost, capacity, and grain

Each variant below changes exactly one part of the tuple. That is the point of presenting them together: a team often changes several at once and then argues about which one moved the score.

**EV-07 — Expected calibration error and its bin dependence.**

Validity before use: probabilities are on the same population and regime as the decision, and bins are declared. ECE is a weighted average of absolute bin gaps; it is not a proper scoring rule and is not monotone in usefulness.

Derivation: partition `[0,1]` into `B` bins. In bin `b`, let `n_b` be the count, `acc_b` the observed positive frequency, and `conf_b` the mean predicted probability. The gap `|acc_b − conf_b|` measures local miscalibration. Weighting by `n_b/N` gives the population average of that gap.

```text
[1] FORMULA: ECE = SUM_b (n_b/N) * |acc_b - conf_b|
[2] WHERE: n_b counts per bin; N total; acc_b, conf_b dimensionless in [0,1].
[3] CONSTRAINT: bins declared in advance; N > 0; a sparse bin's gap is noisy and
           must be shown with its support. ECE is threshold-free.
[4] NUMERICAL EXAMPLE: N=1000. bin [0.0,0.3): n=500, conf=0.20, acc=0.12
           bin [0.3,0.7): n=350, conf=0.50, acc=0.55
           bin [0.7,1.0]: n=150, conf=0.85, acc=0.80
           ECE = 0.5*0.08 + 0.35*0.05 + 0.15*0.05
               = 0.040 + 0.0175 + 0.0075 = 0.065
[5] INTERPRETATION: at an 0.85-score band the workflow expects 85 positives per
           100 alerts and this population delivered 80; the gap is 5 points, and
           the low-score band is the most overconfident in relative terms.
```

Demonstration of bin dependence with four records: scores `0.60` (y=1) and `0.40` (y=0). With one bin, mean confidence is 0.50, accuracy is 0.50, and ECE is 0.00 — the model looks perfectly calibrated. With two bins, the 0.40 record sits at accuracy 0 against confidence 0.40 (gap 0.40) and the 0.60 record at 1.00 against 0.60 (gap 0.40), so ECE is 0.40. The scores did not change. Report the binning, the support per bin, and never compare ECE values produced with different bin counts.

**EV-08 — Brier score as a proper alternative.**

Validity before use: binary outcome per record, probability per record, same population as the decision.

Derivation: the Brier score is the mean squared error between the predicted probability and the realized binary outcome. Squaring is what makes it proper: it is minimized in expectation by reporting the true conditional probability, which ECE does not guarantee.

```text
[1] FORMULA: Brier = (1/N) * SUM_i (p_i - y_i)^2
[2] WHERE: p_i predicted probability in [0,1]; y_i in {0,1}; N records.
[3] CONSTRAINT: outcome definition and population fixed; not a ranking metric;
           not comparable across populations with different base rates.
[4] NUMERICAL EXAMPLE: (0.90,1), (0.70,0), (0.60,1), (0.20,0)
           terms: 0.01, 0.49, 0.16, 0.04; sum = 0.70; Brier = 0.70/4 = 0.175
[5] INTERPRETATION: 0.175 is the mean squared probability error per record; a
           reference model that always predicts the base rate 0.5 scores
           (1/4)*(4*0.25) = 0.25, so this model beats the constant baseline here.
```

**EV-09 — Cost-optimal threshold from calibrated probabilities.**

Validity before use: scores are calibrated probabilities for the *decision population*; costs per false positive and per false negative are known in the same units; decisions are independent. If any of those fails, the resulting threshold is a heuristic, not an optimum.

Derivation: consider one candidate with probability `p`. Accepting it incurs expected cost `(1−p)·c_FP` (it is a negative, so it becomes a false positive). Rejecting it incurs expected cost `p·c_FN` (it is a positive, so it becomes a false negative). Accept when the first is smaller:

```text
(1 - p) * c_FP < p * c_FN
c_FP < p * (c_FP + c_FN)
p > c_FP / (c_FP + c_FN)
```

```text
[1] FORMULA: p* = c_FP / (c_FP + c_FN); accept iff calibrated p >= p*
[2] WHERE: c_FP cost of one false positive; c_FN cost of one false negative;
           both in the same currency/unit (money, minutes, harm points).
[3] CONSTRAINT: calibrated probabilities on the decision population; costs constant
           across cases; independence of decisions; no capacity limit.
[4] NUMERICAL EXAMPLE: c_FP = EUR 4 (4 review minutes at EUR 60/h);
           c_FN = EUR 180 (illustrative rework cost).
           p* = 4/(4+180) = 4/184 = 0.021739
[5] INTERPRETATION: with a cheap false positive and an expensive miss, the cost
           model wants an aggressively low threshold; the capacity rule in
           EV-10 is what stops it in practice.
```

Boundary: the cost optimum is *unconstrained*. A threshold of 0.022 can flood a queue, as Section 5 shows, so the deployed threshold is the solution of a constrained problem, not of this equation. Also, if `p` is uncalibrated but monotone, the ordering of candidates is preserved and the threshold can still be chosen by ranking, but the numeric value `p*` loses its meaning and must be re-derived empirically as an alert-rate target.

**EV-10 — Review capacity as a queue constraint.**

Validity before use: arrival rate and review duration are measured or explicitly illustrative for the *same* policy; the analysis is a mean-rate balance, not a latency guarantee.

Derivation: every accepted alert consumes review time. The arrival rate of alerts is the arrival rate of items multiplied by the acceptance fraction `a`. Required reviewer-hours per unit time is that alert rate times the mean service time; dividing by productive hours per reviewer gives the headcount. Utilization is the ratio of required to available service.

```text
[1] FORMULA: lambda_a = lambda * a;  H = lambda_a * s;  rho = lambda_a * s / (m * h_r)
[2] WHERE: lambda items/hour; a acceptance fraction in [0,1]; s mean review time
           in hours/alert; m reviewers; h_r productive hours per reviewer per hour.
[3] CONSTRAINT: mean rates; no bursts, no absenteeism, no rework loops. rho < 1 is
           necessary, not sufficient: queueing delay grows as rho -> 1.
[4] NUMERICAL EXAMPLE: lambda = 41.67 items/h (1000/day), a = 0.355, s = 0.05 h
           lambda_a = 14.78 alerts/h; H = 0.739 h/h
           with m = 3 and h_r = 1.0: rho = 0.739/3 = 0.246
[5] INTERPRETATION: three reviewers absorb 1000 images/day at 24.6% utilisation,
           leaving room for breaks and variability; seven would be needed at the
           low threshold of Section 5.
```

Boundary: utilisation is not a service-level statement. At `rho = 0.85` with variable service times, queue length and wait time can grow much faster than the mean-rate balance suggests; the correct instrument is a queueing model with measured service-time variability, and the correct operational check is an observed 95th-percentile wait, not the mean.

**EV-11 — Case-level aggregation and the independence trap.**

Validity before use: the action grain is the case; the aggregation rule is declared; the dependence among images of one case is acknowledged.

Derivation: with rule "alert if any image of the case exceeds the threshold", the probability of catching a positive case is one minus the probability that every image misses. If images were independent with per-image sensitivity `q`, the miss probability is `(1−q)^m` for `m` images. Dependence inflates that miss probability, and a standard design-effect correction replaces `m` by an effective count.

```text
[1] FORMULA: q_case = 1 - (1-q)^m       (independence)
             n_eff = m / (1 + (m-1)*rho)  (design-effect approximation)
             q_case ~ 1 - (1-q)^n_eff
[2] WHERE: q per-image sensitivity; m images per case; rho intra-case correlation
           of miss indicators in [0,1]; n_eff effective independent images.
[3] CONSTRAINT: independence version assumes independent misses; design-effect
           version assumes a common intraclass correlation; both are models.
[4] NUMERICAL EXAMPLE: q=0.80, m=4.
           independent: 1 - 0.2^4 = 1 - 0.0016 = 0.9984
           rho=0.5: n_eff = 4/(1+1.5) = 1.6; 1 - 0.2^1.6 = 1 - 0.0762 = 0.9238
           rho=1.0: n_eff = 1; q_case = 0.80
[5] INTERPRETATION: the same per-image sensitivity implies 0.998, 0.924, or 0.800
           at case grain depending only on within-case dependence; the rule must
           be measured, never assumed, before a case-level claim is made.
```

**EV-12 — Test-time augmentation as variance reduction.**

Validity before use: the augmentation family is label-preserving for the task, and the aggregation rule is fixed in advance.

Derivation: averaging `K` predictions over label-preserving transforms reduces the variance of the averaged score if the individual errors are not perfectly correlated; the bias changes only if the transforms distort class evidence. With correlation `rho` between two transformed views, the variance of the mean of `K` views is `sigma²(1 + (K−1)rho)/K`, which tends to `sigma²·rho` as `K` grows — a floor set by correlation, not by `K`.

```text
[1] FORMULA: Var(mean_K) = sigma^2 * (1 + (K-1)*rho) / K
[2] WHERE: sigma^2 per-view score variance; rho inter-view correlation; K views.
[3] CONSTRAINT: label-preserving transforms; fixed aggregation (mean or vote);
           rho >= 0; the floor sigma^2*rho remains as K -> infinity.
[4] NUMERICAL EXAMPLE: sigma^2=0.04, rho=0.6, K=4:
           Var = 0.04*(1 + 3*0.6)/4 = 0.04*2.8/4 = 0.028
           K=8: 0.04*(1+4.2)/8 = 0.04*5.2/8 = 0.026; floor = 0.024.
[5] INTERPRETATION: quadrupling K from 4 to 16 buys only 0.004 of variance
           reduction here; the cost is 16 forward passes per image.
```

Boundary: horizontal flip is label-preserving for many classes but not for text, left/right anatomy, or directional defects. Applying flip-TTA to those tasks averages a correct prediction with a systematically wrong one and lowers accuracy while looking more sophisticated.

**Takeaway:** Calibration, cost, capacity, and grain each move the answer; change one at a time and record which.

### 4.4 Limiting behaviour and counterexamples

**Limit 1 — the detection cap sets a recall ceiling.** With at most `K` detections retained per image, recall on an image with `n` positives cannot exceed `K/n` when `K < n`, regardless of model quality. In a crowded image with 25 positives and `K = 10`, the ceiling is `10/25 = 0.40`. Every miss beyond that ceiling is an artifact of the evaluation script. COCO's default of 100 detections per image per class exists to bound this effect; a team that lowers it to speed up scoring, then reports a recall drop, has measured its own cap.

Worked version, so the size of the artifact is visible. Twelve detections in descending-score order, flags `TP, FP, TP, TP, FP, TP, FP, TP, FP, TP, TP, TP`, with `N_pos = 8` labels in the image.

```text
Cap = 12 (no truncation): 8 TP retained of 8 labels -> recall = 1.000
  cumulative precision at each TP rank: 1.0000, 0.6667, 0.7500, 0.6667,
                                         0.6250, 0.6000, 0.6364, 0.6667
  envelope (max precision at or after the rank): 1.0000, 0.7500, 0.7500,
                                        0.6667, 0.6667, 0.6667, 0.6667, 0.6667
  AP = (1.0000+0.7500+0.7500+0.6667+0.6667+0.6667+0.6667+0.6667)/8
     = 5.8335/8 = 0.7292

Cap = 10 (drop the two lowest-score detections, both TP): recall = 6/8 = 0.750
  TP ranks available: 1, 3, 4, 6, 8, 10
  envelope: 1.0000, 0.7500, 0.7500, 0.6667, 0.6250, 0.6000
  AP = (1.0000+0.7500+0.7500+0.6667+0.6250+0.6000)/8 = 4.3917/8 = 0.5490
```

AP fell by 0.180 and recall by 0.250 while nothing about the model or the labels changed. The reason the loss is larger than the two dropped detections suggests is that truncation removes the *tail* of the ranking, where the envelope was still being read; the dropped true positives also reduce `TP_k` for every later rank, which lowers precision below the cut. This is why a cap change must be treated as a protocol change and reported in the tuple diff, and why "we sped up evaluation" is a legitimate but must-be-disclosed reason for a metric drop.

**Limit 2 — macro and micro diverge as prevalence skews.** With per-class APs 0.90 and 0.50 at supports 900 and 100, macro mAP is 0.70 while the support-weighted value is `(900·0.90 + 100·0.50)/1000 = 0.86`. The 0.16 gap is not error; it is two different questions ("how well does the model handle each class?" versus "how often is a random positive handled well?"). Report the aggregation with the number, and report per-class AP beside it.

**Limit 3 — precision is undefined, not zero, at zero alerts.** A threshold above every score produces no predictions. `precision = TP/(TP+FP)` is `0/0`. Any convention — 0, 1, or "undefined" — changes the reported value by a full point. The honest convention is "undefined, support 0", and precision-recall curves must mark the endpoint rather than inventing it.

**Limit 4 — IoU with two empty masks.** If both prediction and label are empty, `0/0` appears. Conventions differ: score 1 (perfect agreement on emptiness), score 0, or exclude the pair. Excluding is usually right for object metrics and wrong for dense prediction where emptiness is informative. Declare it; then check how many pairs fall in the case, because a dataset that is 90% empty images will move by tens of points depending on the convention.

**Limit 5 — a label-error floor on measured recall.** Suppose a fraction `q` of positive labels are wrong (an object exists but is not labelled, or a labelled object does not exist). A correct prediction on an unlabelled object is scored as a false positive; a prediction on a wrongly labelled object is scored as a false positive too. Measured recall is then bounded above by roughly `1 − q` when model and label errors are independent, because at least the `q` fraction of positives that are labelled incorrectly cannot be matched correctly. With `q = 0.05`, a measured recall of 0.95 may already be the ceiling, and the remaining gap is annotation, not representation. This is a bound under stated independence, not a law: correlated errors can exceed it in either direction.

**Limit 6 — ECE is not invariant to the score transform.** Applying a strictly increasing map to scores (a monotone recalibration that changes no ranking) can change ECE dramatically, because bin membership and the confidence values both move. Accuracy, recall, AP, and PQ are invariant to monotone score maps at a fixed rank threshold; ECE and Brier are not, and cost-optimal thresholds require the probability scale itself. Choosing a metric therefore chooses which invariances you are buying.

**Limit 7 — a small slice cannot resolve a small difference.** With 30 positives in a slice and observed recall 0.40, the Wilson 95% interval is about `(0.246, 0.577)`; a claim that the slice moved from 0.40 to 0.45 is not supported. Hoeffding requires `n ≥ ln(2/α)/(2ε²)` samples for an `ε`-accurate estimate with confidence `1−α`: for `ε = 0.05` and `α = 0.05`, `n ≥ ln(40)/0.005 = 3.689/0.005 = 737.8`, so about 738 positives. Slices with 30 labels are diagnostic instruments; they are not comparison instruments.

```text
Result/invariant            | Required assumption          | Diagnostic            | Violating example                    | Correct alternative
IoU = TP/(TP+FP+FN)         | one frame, one class         | coordinate audit      | inclusive vs half-open boxes         | single convention project-wide
AP monotone under envelope  | complete labels, fixed cap   | recount N_pos         | cap 10 on 25-object images           | raise cap, restate
macro mAP = micro mAP       | equal support per class      | compare supports      | 900 vs 100 positives                 | report both, or weight explicitly
precision in [0,1]          | TP+FP > 0                    | count accepted        | threshold above all scores           | report "undefined, n=0"
ECE meaningful              | declared bins, dense support | support per bin       | 4 records, one bin                    | report bins + support, use Brier
slice comparison valid      | adequate support             | Wilson interval       | 12/30 vs 13/30                       | enlarge slice or pool evidence
case recall from image q    | independent misses           | within-case correlation| 4 images of one session              | measure q_case directly
```

**Takeaway:** Every metric has a floor, a ceiling, and an undefined region; a report that does not name them can be moved without touching the model.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE scenario with VERIFIED arithmetic. Nothing here is a measurement of any product, camera, or dataset. The point is to show every decision that a single accuracy number hides, with each number recomputable from the ones before it.

**Workflow and decision.** Four injection-moulding lines produce one part per image. A defect detector marks boxes; the workflow then either passes the part or routes it to a human rework review. The action grain is the *image* (one part), while the model's natural output grain is the *object* (one or more defects). Both grains must be reported, because the object metric describes the model and the image metric describes the decision.

**Cohort.** 1,000 test images captured over 14 days on lines L1–L4, drawn from 40 machines. Split is by machine: 28 machines train, 6 validation, 6 test. Machines were assigned to splits before any scoring, and no image from a test machine appears in training. Eligibility: images with a successful capture timestamp and a completed label; 1,000 of 1,043 captured images qualify, and 43 are excluded for a documented capture fault (a sensor-readout error recorded in the acquisition log), not for looking unusual.

**Labels.** 900 defect instances across 800 positive images: 700 images carry one defect, 100 carry two, and 200 images carry none. Labels were produced under a written policy, with 5% of images double-read; the double-read disagreement rate is recorded as part of the contract (see F-02). All arithmetic below uses 900 objects and 1,000 images as the frozen denominators.

### 5.1 Object-grain results across a threshold sweep

Matching is greedy by descending score, class-exact, `IoU ≥ 0.50`, ignore regions removed (crowd-labelled shadows), detection cap 100 per image. Only the score threshold changes between rows; the matching rule is untouched, so the rows are comparable.

| Threshold τ | TP | FP | FN | precision | recall | accepted |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 810 | 490 | 90 | 0.623077 | 0.900000 | 1,300 |
| 0.50 | 684 | 216 | 216 | 0.760000 | 0.760000 | 900 |
| 0.70 | 558 | 90 | 342 | 0.861111 | 0.620000 | 648 |
| 0.90 | 360 | 27 | 540 | 0.930233 | 0.400000 | 387 |

Arithmetic for the 0.50 row, spelled out because every later section reuses it: `TP+FN = 684+216 = 900` (the label count, so no object was lost), `TP+FP = 684+216 = 900` (the accepted count), `precision = 684/900 = 0.76`, `recall = 684/900 = 0.76`. The equality of the two 900s is a coincidence of this row being chosen on the diagonal; it is not an identity and must never be assumed.

Arithmetic for the 0.90 row: `precision = 360/(360+27) = 360/387 = 0.9302326`, `recall = 360/900 = 0.40`. The high threshold makes almost every alert correct and finds fewer than half the defects. A dashboard showing only precision would describe this row as the best model.

### 5.2 Ranking quality: a hand-checkable AP trace

The sweep gives four operating points; AP describes the whole ranking. To keep it checkable, take 12 detections in descending score order with `N_pos = 5` labels present in those images.

| Rank k | Outcome | TP_k | Precision TP_k/k | Recall TP_k/5 |
|---:|---|---:|---:|---:|
| 1 | TP | 1 | 1.0000 | 0.2 |
| 2 | FP | 1 | 0.5000 | 0.2 |
| 3 | TP | 2 | 0.6667 | 0.4 |
| 4 | FP | 2 | 0.5000 | 0.4 |
| 5 | FP | 2 | 0.4000 | 0.4 |
| 6 | TP | 3 | 0.5000 | 0.6 |
| 7 | TP | 4 | 0.5714 | 0.8 |
| 8 | FP | 4 | 0.5000 | 0.8 |
| 9 | TP | 5 | 0.5556 | 1.0 |
| 10 | FP | 5 | 0.5000 | 1.0 |
| 11 | FP | 5 | 0.4545 | 1.0 |
| 12 | FP | 5 | 0.4167 | 1.0 |

Monotone envelope at the five TP ranks (`max` of precision over all ranks at or after the TP rank): rank 1 → `max(1.0000, 0.5000, 0.6667, 0.5000, 0.4000, 0.5000, 0.5714, 0.5000, 0.5556, 0.5000, 0.4545, 0.4167) = 1.0000`; rank 3 → `max(0.6667, 0.5000, 0.4000, 0.5000, 0.5714, 0.5000, 0.5556, …) = 0.6667`; rank 6 → `max(0.5000, 0.5714, …) = 0.5714`; rank 7 → `0.5714`; rank 9 → `0.5556`.

```text
AP = (1.0000 + 0.6667 + 0.5714 + 0.5714 + 0.5556) / 5
   = 3.3651 / 5
   = 0.673020
```

Independent check by recall increments: each TP contributes `ΔR = 1/5 = 0.2`, so `0.2 × 3.3651 = 0.67302`. The two constructions agree to five decimals, which is the point of the derivation in EV-04.

Now the interpolation check that trips up vendor comparisons. The 11-point rule samples `max precision at recall ≥ r` for `r ∈ {0.0, 0.1, …, 1.0}`: values are `1.0000, 1.0000, 1.0000, 0.6667, 0.6667, 0.5714, 0.5714, 0.5714, 0.5714, 0.5556, 0.5556`. Sum `= 7.7302`, divided by 11 gives `0.70275`. The same 12 detections therefore score **0.673** under dense/all-point interpolation and **0.703** under VOC2007 11-point. A 0.030 "improvement" is available for free by changing the script.

### 5.3 Slice decomposition at the selected operating point

At `τ = 0.50`, split the 900 objects by native area, because the physical action (rework) is driven by whether small defects are caught.

| Slice | Objects | TP | FN | Recall | Wilson 95% interval |
|---|---:|---:|---:|---:|---|
| small (< 1,024 px²) | 300 | 120 | 180 | 0.400 | (0.346, 0.456) |
| medium (1,024–9,216 px²) | 400 | 368 | 32 | 0.920 | (0.890, 0.943) |
| large (> 9,216 px²) | 200 | 196 | 4 | 0.980 | (0.949, 0.993) |
| **all** | **900** | **684** | **216** | **0.760** | (0.732, 0.786) |

Check the decomposition closes: `120+368+196 = 684` = total TP, `180+32+4 = 216` = total FN. A slice table that does not close has a join bug, and a join bug is the second most common cause of a "slice effect".

Wilson interval for the small slice, with `z = 1.96`, `n = 300`, `p = 0.4`: centre `= (p + z²/2n)/(1 + z²/n) = (0.4 + 3.8416/600)/(1 + 3.8416/300) = 0.406403/1.012805 = 0.40127`; half-width `= (1.96/1.012805)·sqrt(0.24/300 + 3.8416/360000) = 1.93523 × 0.028471 = 0.05510`. Interval `(0.3462, 0.4564)`. The large slice's interval is much tighter at the same support because `p` is near 1, which is why "n is enough" is not a rule without `p`.

Note what the aggregate hides: object recall 0.76 looks like a middling model, while the small-defect band — the band the rework action targets — sits at 0.40 with an interval that excludes anything above 0.46. If the workflow decision is "catch small surface defects", the aggregate is the wrong selection metric and 0.40 is the number that matters.

### 5.4 Image-grain aggregation and review load

Object counts do not tell the queue what arrives. The workflow alerts on an image if any accepted detection survives on that image.

| Image-level outcome at τ=0.50 | Count |
|---|---:|
| Positive images, alerted (image TP) | 640 |
| Positive images, silent (image FN) | 160 |
| Negative images, alerted (image FP) | 140 |
| Negative images, silent (image TN) | 60 |
| Total alerts | 780 |

Consistency check with the object table: 800 positive images, 640 alerted, 160 silent. The 216 missed objects sit inside those 160 silent images, so some silent image must contain more than one missed defect — the 100 two-defect images are where that happens. Stating this explicitly catches the common error of assuming one missed object per missed image.

Derived rates: image recall `640/800 = 0.80`; image specificity `60/200 = 0.30`, so the false-alarm rate on clean parts is `140/200 = 0.70`; image precision `640/780 = 0.820513`; alert fraction `a = 780/1000 = 0.78`. Object recall was 0.76 and image recall is 0.80 — the gap is the aggregation rule doing its job (more chances to alert), and it is exactly why a case-grain claim cannot be inferred from an object metric.

**Queue sizing at 1,000 images/day.** Reviewers inspect an alert for 2 minutes on average (`s = 0.0333 h`). Arrival `λ = 1,000/8 = 125` images/h over an 8-hour shift; alerts `λ_a = 125 × 0.78 = 97.5/h`. Required service per hour `= 97.5 × 0.0333 = 3.25` reviewer-hours per clock hour. With 4 reviewers at 0.85 productive utilisation, available service is `4 × 0.85 = 3.40` reviewer-equivalents, so utilisation `ρ = 3.25/3.40 = 0.956`.

```text
H     = lambda_a * s        = 97.5 * 0.03333 = 3.2500 reviewer-hours/clock-hour
rho   = H / (m * eta)       = 3.2500 / (4 * 0.85) = 0.9559
```

`ρ = 0.956` is technically below 1 and operationally alarming: with any service-time variability the queue will drift upward over a shift, because utilisation near one converts small variability into long waits. The number is a *feasibility screen*, not a service-level guarantee.

### 5.5 Choosing the threshold under two constraints at once

Unit costs, illustrative and stated in one currency: `c_FP = EUR 2` (two review minutes at EUR 60/h), `c_FN = EUR 240` (rework, scrap, and line-stop share for an escaped defect). Unconstrained optimum from EV-09: `p* = 2/(2+240) = 0.008264`, which would accept nearly everything and is useless as an operating point. Total expected cost per 1,000 images at each row:

| τ | FP | FN | FP cost | FN cost | total | alerts/day | reviewers needed at ρ≤0.85 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 490 | 90 | 980 | 21,600 | 22,580 | 940 | 4.60 → 5 |
| 0.50 | 216 | 216 | 432 | 51,840 | 52,272 | 780 | 3.82 → 4 |
| 0.70 | 90 | 342 | 180 | 82,080 | 82,260 | 520 | 2.55 → 3 |
| 0.90 | 27 | 540 | 54 | 129,600 | 129,654 | 300 | 1.47 → 2 |

Arithmetic for one row: at `τ = 0.30`, `490 × 2 = 980` and `90 × 240 = 21,600`, total `22,580`. Reviewer count `= (940/8) × 0.03333 / 0.85 = 117.5 × 0.03333 / 0.85 = 3.9167/0.85 = 4.608`, so five reviewers are required to hold utilisation at or below 0.85.

Cost alone prefers `τ = 0.30` by a wide margin (22,580 versus 52,272). Capacity at the approved headcount of four forbids it: four reviewers at `ρ ≤ 0.85` provide `4 × 0.85 = 3.40` reviewer-hours per clock hour, which buys `3.40/0.03333 = 102` alerts per hour, or `816` per 8-hour day. The feasible set is `{τ : alerts ≤ 816}`, and the cheapest feasible row is `τ = 0.50` with 780 alerts and utilisation 0.956 — feasible on the mean but outside the stated 0.85 design target.

**The actual decision.** Deploy `τ = 0.50` with three riders, all of which are part of the promotion record rather than footnotes: (i) add one reviewer or move to a 3-minute target only after measuring actual review time, since 780 alerts at 3 minutes would need `780 × 0.05 = 39` reviewer-hours per day against `4 × 6.8 = 27.2` available; (ii) introduce an abstain band `[0.40, 0.50)` routed to a lighter queue so that the utilisation target can be met without raising the accept threshold; (iii) block promotion of the small-object slice until a capture or model change lifts small recall above the pre-declared 0.60 gate, because the cost model is dominated by misses and small defects are where the misses are.

Sensitivity of the decision. If `c_FN` falls to EUR 40 (defects are caught downstream anyway), `p* = 2/42 = 0.0476` and the cost curve flattens: totals become `τ=0.30: 980+3,600 = 4,580`, `τ=0.50: 432+8,640 = 9,072`, `τ=0.70: 180+13,680 = 13,860`. The cost-optimal row is still 0.30, but the penalty for choosing 0.50 shrinks from EUR 29,692 to EUR 4,492 — a 6.6-fold reduction that weakens the capacity argument from "the cost model forbids 0.50" to "0.50 costs EUR 4,492/day more than an infeasible alternative, and capacity is what decides". Note the direction: the capacity constraint is doing more of the work in the decision than the harm estimate is, which is the opposite of what most reviews assume before they compute both.

### 5.6 Production-scale interpretation

At 50,000 images/day on the same policy (`a = 0.78`, `s = 2 min`), alerts are 39,000/day and reviewer-hours are `39,000 × 0.03333 = 1,300` per day. At 6.8 productive hours per reviewer-day that is `1,300/6.8 = 191.2`, so 192 reviewers. The policy that is merely expensive at 1,000 images/day is structurally impossible at 50,000. The remedies are not "a bigger queue": they are a higher accept threshold with a cheaper second-stage model, a narrower defect ontology, or an automation step for the high-prevalence obvious cases. Each remedy changes the tuple, so each must be evaluated under its own contract.

```text
Scale    images/day   alerts/day   reviewer-hours/day   reviewers (6.8 h)
small         1,000          780             26.0            4
medium       10,000        7,800            260.0           39
large        50,000       39,000          1,300.0          192
```

### 5.7 Independent checks and the wrong path

Checks performed on the numbers above: `TP+FN = 900` in every row of the sweep; `TP+FP` equals the stated accepted count; precision and recall recomputed from raw counts rather than copied from the previous row; slice counts sum to the aggregate; image-level contingency table sums to 1,000 and its marginals agree with the object table's silent-image claim; Wilson intervals computed from the formula rather than quoted; AP verified by two independent constructions; cost rows recomputed from unit costs; reviewer counts derived from hours rather than asserted.

The plausible wrong path, in the order it usually happens. A team takes the `τ = 0.90` row because precision 0.93 is the highest; the dashboard looks excellent. Rework misses 540 defects a day-equivalent, the escape is eventually traced to the detector, and the response is "retrain with more data". Retraining cannot fix a threshold that was chosen on the wrong axis: the same model at `τ = 0.50` finds 324 more defects at the cost of 189 more false alerts. The correction is to select on the action-weighted objective under the capacity constraint, which is a decision change, not a model change.

The second wrong path: reading the 0.76 aggregate as uniform and shipping. The slice table shows small defects at 0.40, so the team's stated goal (catch surface defects on small features) fails while the aggregate passes. The discriminating check is the slice closure table plus a pre-declared blocking gate on the action-critical band.

**Takeaway:** The threshold is the solution to a constrained problem whose two constraints — harm and capacity — live in different units and must both be computed.

## 6. Visual / Geometric Intuition

### 6.1 What an IoU value cannot tell you

A single overlap number collapses three independent geometric facts — shift, scale, and identity — into one scalar. The diagrams below are schematic (not to scale) but the pixel arithmetic is exact.

```text
Case A: translated, same size          Case B: same centre, doubled size
label      [ 0 ... 99 ]  area 100      label      [ 0 ... 99 ]   area 100
pred          [20 ... 119] area 100    pred    [-50 ... 149]    area 200
I = 80   U = 120   IoU = 0.667          I = 100   U = 200   IoU = 0.500
shift only                             scale only, label fully covered

Case C: wrong object                   Case D: correct object, coarse boundary
label  [0 ... 99]    area 100          label  [0 ... 99]
pred   [200 ... 299] area 100          pred   [0 ... 104]  (annotation tolerance)
I = 0   U = 200   IoU = 0.000          I = 100  U = 104  IoU = 0.962
identity error                         boundary-policy effect
```

Four mechanically different failures produce IoU 0.667, 0.500, 0.000, and 0.962. The intervention for each is different: A needs localization refinement, B needs scale regression, C needs a representation or class error investigation, D needs an annotation-policy decision. A single "improve the detector" response treats all four as one problem and therefore cannot work on more than one of them.

### 6.2 The precision–recall plane as an operating-point map

```text
recall
 1.0 |                                        * (0.90, 0.62) tau=0.30
     |                              *
 0.8 |                    *  (0.76, 0.76) tau=0.50
     |              *
 0.6 |        *
     |   *
 0.4 |  *  (0.40, 0.93) tau=0.90
     |
 0.2 |
     |
 0.0 +----+----+----+----+----+----+----+----+----+----> precision
      0.0  0.2  0.4  0.6  0.8  1.0

  curve = same model, different thresholds. Points are NOT models.
  The review-capacity constraint is a horizontal line on the *negative* axis:
  it bounds recall from above because recall x positives = missed + caught.
```

The essential reading: moving along the curve is free (one threshold), moving the curve is expensive (a new model or new data). A team that reports a single point has hidden which of these happened. Always plot the curve and mark the deployed point, plus the capacity-feasible region if the workflow has one.

### 6.3 Calibration: three populations, one model

```text
predicted
prob conf
 1.0 |            ideal
     |          /
 0.8 |        /  o  population R (rare, over-confident)
     |      /
 0.6 |    / x     population D (base rate 0.50, on diagonal)
     |  / o
 0.4 |/ x
     |o
 0.0 +------------------ observed frequency
      0.0   0.4   0.6   1.0

  o = same model scored on a rare population: the high bins overstate.
  x = same model scored on a balanced population: nearly on-diagonal.
```

Calibration is not a property of weights alone. It is a property of the weights *and* the population *and* the regime (preprocessing, sensor, exposure). The same checkpoint can be calibrated in the lab and miscalibrated at a site whose base rate is 3% instead of 30%. This is why the promotion record must name the population on which the calibration was checked.

### 6.4 Error taxonomy as a partition you can close

```text
900 labelled objects
 |
 +-- 684 matched (TP)
 |     |
 |     +-- 640 with IoU >= 0.75 ....... localization acceptable
 |     +--  44 with 0.50 <= IoU < 0.75 . weak localization, counted as hit
 |
 +-- 216 unmatched labels (FN)
       |
       +-- 150 no overlapping prediction at all ... detection miss
       +--  40 overlapping prediction, wrong class ... classification error
       +--  20 overlapping prediction, IoU < 0.50 .... localization error
       +--   6 overlapping prediction, suppressed by NMS ... post-processing
```

The partition closes when `150 + 40 + 20 + 6 = 216`. Every error register must close like this against the frozen denominator; an unclassified remainder is a finding in itself, usually pointing at a join or ignore-region bug. Note that the four FN categories imply different next actions: detection misses argue for capacity/data/resolution, class errors for ontology or features, localization for regression targets or resolution, NMS suppression for post-processing policy.

### 6.5 The leakage geometry

```text
time --->
recording R1:  f1 f2 f3 f4 f5 f6 f7 f8
random split:  [tr tr va tr | tr tr te tr]     <-- frames of R1 on both sides
group split:   [tr tr tr tr | va va | te te]   <-- R1 entirely in train

apparent random-split accuracy: high (memorised scene, subject, exposure)
apparent group-split accuracy:  lower (generalisation measured honestly)
```

The horizontal bars are the same data. Only the partition differs. When the random split scores much higher, the difference is a lower bound on how much of the score came from within-group memorisation, not a measurement of it (the model may also have learned something genuinely transferable).

### 6.6 The matching geometry that produces the counts

The partition in 6.4 is the *result* of matching. The geometry below shows how the four FN categories arise from one image's layout, which is the mental picture that makes taxonomy assignment fast in practice.

```text
image with 5 labelled objects (G1..G5) and 6 predictions (P1..P6)

  +--------------------------------------------------+
  |  G1[########]           G2[###]                  |
  |   P1[########]  IoU .92 -> TP (accepted)         |
  |                     P2[##########] IoU .18 -> FP |
  |  (P2 is a large false blob spanning G1/G2)       |
  |                                                  |
  |  G3[######]          G4[###]                     |
  |        P3[######] IoU .61 -> TP                  |
  |                     P4[###] IoU .55 class=other  |
  |                     -> FP + G4 becomes FN-class  |
  |                                                  |
  |  G5[####]                                        |
  |   (no prediction overlaps)      -> FN-detect     |
  |                                                  |
  |  P5[####] on bare background   -> FP             |
  |  P6[####] duplicate of P1      -> FP (suppressed  |
  |                                        at NMS .45)|
  +--------------------------------------------------+
  outcome: TP=2 (G1,G3)  FP=4 (P2,P4,P5,P6)  FN=3 (G2,G4,G5)
  precision = 2/6 = 0.333   recall = 2/5 = 0.400
```

Read the failure mechanisms off the diagram, one per false negative: G2 is a *detection miss* caused by a large false blob stealing capacity and by no admissible overlap (the blob covers G2 at IoU 0.18, below τ); G4 is a *classification error* — the geometry is adequate at IoU 0.55 but the class is wrong, so the pair is inadmissible and both the prediction and the label become errors; G5 is a plain *detection miss* with no overlapping prediction at all. The four FP also have distinct mechanisms: P2 is a scale error, P4 is a class error, P5 is a background false positive, P6 is a duplicate that NMS at 0.45 failed to suppress because its IoU with P1 is below 0.45.

The operational consequence: a taxonomy that reports "3 misses" and "4 false alarms" gives no next step. The same counts, attributed as above, point at four different levers: proposal capacity and scale regression (P2), class features or ontology (G4/P4), background hard-negative mining (P5), and post-processing policy (P6). This is the entire argument for a taxonomy over a confusion matrix: the confusion matrix tells you *how many*, and the attributed diagram tells you *which lever*.

### 6.7 Scale-space: why resolution changes the answer

```text
native object:        16 x 16 px   (256 px^2)
after 0.5x resize:     8 x 8  px   ( 64 px^2)   -> 4x fewer pixels
after 0.25x resize:    4 x 4  px   ( 16 px^2)   -> 16x fewer pixels

localization penalty for a 1-px boundary error, as IoU:
  16x16 object: I=240, U=272  -> IoU 0.882   (a shift is survivable)
   8x8  object: I= 56, U= 72  -> IoU 0.778
   4x4  object: I= 12, U= 20  -> IoU 0.600   (a 1-px error nearly fails tau=0.5)

same physical error, three different metric verdicts
```

The arithmetic for the `4×4` case: the object covers 16 px; a one-pixel shift leaves `(4−1) × 4 = 12` px of intersection and `16 + 16 − 12 = 20` px of union, so `IoU = 0.600`. At `16×16` the same one-pixel shift leaves `15 × 16 = 240` of intersection and `256 + 256 − 240 = 272` of union, giving `0.882`. The *physical* localization error is one pixel in both cases; the *metric* verdict changes from comfortable to marginal purely because the object is smaller relative to the sampling grid. This is the geometric reason a resize pipeline silently converts localization quality into recall loss on the small band, and it is why the slice check in F-03 compares native and post-resize area rather than native area alone.

### 6.8 The confusion between "confident" and "correct"

```text
score distribution, positives vs negatives (overlapping, illustrative)

  density
    |      negatives            positives
    |        /\                  /\
    |       /  \      overlap   /  \
    |      /    \       |      /    \
    |     /      \______|_____/      \
    |    /              |             \
    +---+-------+-------+-------+------+----> score
        0.2     0.4     0.5     0.7    0.9

  the overlap band is where the threshold decides everything:
  move tau right -> fewer false alarms, more escapes
  move tau left  -> more false alarms, fewer escapes
  no threshold removes the overlap; only a better score function does
```

The diagram's teaching point is that threshold selection is a *position* choice inside an overlap that the model has already fixed. A team that spends a quarter tuning the threshold is moving the vertical line within a band whose width was set by the representation. Once the threshold is at the cost-optimal or capacity-feasible position, further gains require the distributions to separate — which is a data, capacity, or target question, not a threshold question. This is the geometric form of the "threshold movement is not model improvement" rule in F-06.

**Takeaway:** Draw the partition and the curve, not just the score, because the score alone cannot distinguish a better model from a different question.

## 7. Comparison / Selection Table

The tables below answer "which evaluation instrument should I use here". The first is by decision condition — the situation the workflow imposes. The second compares metric families on assumptions and failure behaviour. The third resolves conflicting constraints, which is the case real reviews spend their time on.

### 7.1 By decision condition

| Decision condition | Preferred evaluation | Why | Disqualifier | Fallback |
|---|---|---|---|---|
| Adjacent video frames or bursts | Group split by recording/scene, plus per-scene recall | Random split leaks scene identity and exposure | Online history is genuinely available at serving, and it is in the contract | Document the online-state protocol and evaluate with it |
| Several sites or camera models | Site/device holdout plus slice board | Measures capture shift directly | A site has fewer than ~30 positives | Abstain on that site or pool with a documented weight |
| Object-level action (count, route, reject) | One-to-one matching, IoU ≥ declared τ, size slices | Preserves duplicate and miss costs at the action grain | Only image-level labels exist | Change the task contract or relabel |
| Human review bottleneck | Threshold plus queue model (`ρ`, wait) | Binds the score to capacity | Service time unmeasured | Shadow mode for two weeks; measure `s` |
| Uncertain or ambiguous labels | Adjudicated sample, ambiguity/ignore class | Separates label error from model error | No label owner exists | Restrict the claim to operational agreement |
| Rare, high-cost positive | Recall at fixed capacity, missed-case audit | Aligns the metric to harm | Positives immature | Defer promotion; record the maturity date |
| Probability consumed downstream (risk score) | Reliability + Brier on the decision population | Both the level and the ranking are used | Population differs from decision population | Recalibrate on a representative sample |
| Ranking/discovery (retrieve then review) | AP or recall@K with a declared K | The consumer truncates at K | K not tied to a workflow | Report the full curve and the truncation point |
| Dense scene understanding | PQ plus per-class SQ | Separates recognition from segmentation | Class granularity differs between compared systems | Report both granularities |
| Case-level action from many images | Case-grain contingency table | Image grain overstates coverage | Within-case dependence unmeasured | Measure `q_case` directly on held-out cases |

### 7.2 Metric families compared

| Metric | Preconditions | Guarantee / meaning | Cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Accuracy | Balanced cost, single label per unit | Fraction correct at one threshold | O(n) | First sanity check | Class imbalance, cost asymmetry | DERIVED |
| Precision / recall | Frozen matching; denominators > 0 | Conditional error rates | O(n) | Operational gating | Ranking comparison across thresholds | DERIVED |
| Fβ | Chosen β reflects harm ratio | Single-number trade-off | O(n) | Contractual summary | Reporting to different stakeholders with different costs | DERIVED |
| AP | Complete labels, fixed τ, cap, interpolation | Threshold-free ranking summary | O(n log n) sort | Model comparison | Deployed-point decisions | SOURCED convention |
| mAP | Uniform protocol per class | Cross-class summary | Σ per-class | Benchmarking | Mixed class granularity | SOURCED convention |
| IoU / Dice | Aligned frames, U > 0 | Localization quality | O(HW) | Localization diagnostics | Whole-image verdicts alone | DERIVED |
| PQ | One-to-one at IoU > 0.5 | Recognition × segmentation | O(n·H·W) matching | Panoptic tasks | Cross-ontology comparison | SOURCED convention |
| ECE | Declared bins, dense support | Local probability agreement | O(n) | Probability consumers | Sparse bins, model selection | SOURCED method, HEURISTIC bins |
| Brier | Binary outcome per record | Proper scoring rule | O(n) | Probability model selection | Ranking-only use | DERIVED (properness) |
| Cost curve | Unit costs + calibrated p | Expected currency per unit time | O(n) | Threshold choice | Uncalibrated scores, non-constant costs | DERIVED |
| Queue model | Arrival + service distributions | Capacity feasibility | O(1) closed form | Alert-rate limits | Tail-latency guarantees | DERIVED (mean rate only) |

Two entries deserve a warning. **AP is not a substitute for the operating point**: a model can gain 0.02 AP from better ranking among low-scoring detections that the workflow never shows to anyone. **Accuracy is not a substitute for anything on this list** when prevalence is below about 10%, because a constant-negative predictor reaches 0.90 accuracy at 10% prevalence while catching nothing; compute the trivial baseline and report it beside every accuracy value.

### 7.3 Selection procedure under conflicting constraints

```text
IF the cohort or split cannot be reconstructed        -> stop; no metric is admissible.
ELSE IF labels are immature or the policy is disputed -> collect evidence; publish nothing.
ELSE IF a hard validity constraint fails
     (frame mismatch, missing ignore policy, unit mix)
                                                     -> disqualify the evaluation itself.
ELSE IF the action is high-harm and misses dominate   -> maximise recall subject to
                                                        capacity; audit missed cases.
ELSE IF the action is high-volume and false alarms
     dominate                                         -> maximise precision subject to a
                                                        recall floor; watch utilisation.
ELSE IF probabilities feed a downstream decision      -> require calibration + Brier on the
                                                        decision population.
ELSE IF the question is "which model ranks better"    -> AP/mAP under one frozen protocol,
                                                        reported with per-class and per-slice AP.
ELSE                                                   -> publish the simple baseline
                                                        (constant predictor + threshold-only
                                                        policy) and collect the evidence that
                                                        would change the decision.

If two conditions conflict (harm says low threshold, capacity says high threshold):
    escalate with both numbers and the cheapest structural option
    (abstain band, second-stage model, ontology narrowing), not a silent compromise.
```

The ordering is deliberate. Validity comes before optimisation because an invalid evaluation cannot be improved by a better model. Evidence comes before validity disputes because a metric computed over disputed labels measures the dispute. And the final `ELSE` is important: "publish the baseline" is a legitimate outcome, and it is far better than a promotion justified by an unreconstructible 0.03.

**Takeaway:** Select the instrument from the failure it must prevent, then check that its denominator survives the constraint you are most likely to hit.

## 8. Failure Modes & Edge Cases

The register is the operational core of this sheet. Each entry has a symptom you can observe, at least two competing causes so that the first plausible story is not automatically accepted, a check that discriminates between them, a bounded fix, and a verification that the fix worked. Provenance is marked `HYPOTHETICAL` where the mechanism is a teaching construction with derived arithmetic, and `DERIVED` where the arithmetic alone forces the conclusion.

### 8.1 How to read and maintain the register

An error taxonomy is not a list of bugs. It is a partition whose categories each map to (a) a test that separates them and (b) an intervention class. When a new failure arrives, the first job is to place it: is it a *metric/protocol* failure (the number is wrong), an *input* failure (the pixels or labels are wrong), a *model* failure (the function is wrong), or a *decision* failure (the number is right and the action is wrong)? These four superclasses have almost disjoint fixes, and most bad incident reviews are caused by jumping from a model-shaped symptom straight to a model-shaped fix.

| Superclass | Distinguishing question | Typical fix | Typical wasted fix if misclassified |
|---|---|---|---|
| Protocol | Does the tuple reconstruct the reported number? | Restate and recompute | Retraining, new data |
| Input | Do the bytes and labels match the declared contract? | Restore pipeline, relabel, re-capture | Architecture change |
| Model | Does the representation fail on the mechanism-linked slice? | Data, capacity, target redesign | Threshold tuning |
| Decision | Is the action mapped correctly from the score? | Threshold, abstain band, queue design | Any model work |

### F-01 — Frame and scene leakage

**Symptom.** A random image split reports 0.94 accuracy; a recording-level split on the same data reports 0.71. The gap appears immediately and survives retraining.

**Competing causes.** (a) Genuine scene memorisation: near-identical frames on both sides. (b) A legitimate temporal feature that really is available at serving time, in which case the random split is measuring the deployed system. (c) Duplicate files or re-exports with different names. (d) Test-set contamination from a previous run whose predictions leaked into a cache.

**Discriminating check.** Hash perceptual embeddings of all images and list near-duplicate clusters with their split membership. Then list recording IDs and scene IDs per partition. If clusters straddle splits, cause (a) or (c) is present; if recording IDs straddle splits while embeddings do not cluster, look for (d). Cause (b) is only credible if the serving contract literally provides earlier frames, which is a documentation question, not a modelling one.

**Bounded fix.** Re-split by the dependency group, keep one untouched group holdout, and re-report every metric from scratch. Do not patch the number by averaging the two splits.

**Mitigation risk.** Group splitting reduces the effective sample size and widens intervals; the lower number is not evidence that the model degraded. Communicate the estimand change explicitly, or the organisation will read the drop as a regression and demand a rollback.

**Verification.** Recompute the tuple-diff report showing `cohort` changed and nothing else; confirm zero duplicate clusters across partitions by running the hash check as a release gate rather than once.

### F-02 — Label disagreement masquerading as model error

**Symptom.** Errors concentrate at object boundaries, at low contrast, and on one annotator's batch. A blinded second read reproduces the model's "mistake" on a meaningful share of cases.

**Competing causes.** (a) The model is wrong. (b) The label policy is genuinely ambiguous in that region. (c) A stale label from an old policy version survived the migration. (d) A coordinate or tool offset shifts every annotation by a pixel or two. (e) Selection bias: only errors were re-read.

**Discriminating check.** Draw a *random* sample stratified by class and slice — not by error — blind the second reviewer to the model output, and compute the disagreement rate separately for model-correct and model-wrong cases. If disagreement is high in both, the policy is the problem. Then audit coordinates: overlay ten annotations against raw pixels and check whether the same constant offset appears across annotators (tool bug) or randomly (human variance).

**Bounded fix.** Freeze a written policy with an ambiguity class and ignore regions; adjudicate the sample; version the labels. Re-score against the adjudicated set and report both the original and adjudicated numbers so the size of the label effect is visible.

**Mitigation risk.** Relabelling only the model's errors biases the evaluation optimistically — it removes the cases the model fails and leaves the easy ones. The sample must be random within strata, and the strata must be declared before the re-read.

**Verification.** Compare disagreement rate and model error rate by category after the policy freeze; the taxonomy shares should shift in the predicted direction, and the ambiguity share should shrink.

### F-03 — Small-object collapse under a stable aggregate

**Symptom.** Aggregate recall holds near 0.76 while the small-object band sits at 0.40 and its interval excludes the gate value. Adding data improves the aggregate without moving the small band.

**Competing causes.** (a) Insufficient pixels after the resize, so information is destroyed before the model sees it. (b) Unlabelled small objects, so the model is penalized for finding real defects. (c) IoU sensitivity: a fixed pixel offset costs a small object much more overlap than a large one. (d) A size/site confound, where small objects come from one camera whose optics differ. (e) Annotation-tool minimum-size limits, so no small objects were ever labelled in one batch.

**Discriminating check.** Report recall as a function of both native area and post-resize area; if the collapse tracks *post-resize* area while native area is fine, cause (a) dominates. Check the label density per image against a visual count on a random sample to test (b). Recompute with a size-normalized localization tolerance to test (c). Cross-tabulate size against device to test (d). Inspect the annotation tool's minimum box for (e).

**Bounded fix.** Choose among capture (higher resolution), tiling with overlap, a size-aware assignment of anchors/proposals, or a narrower claim ("the model is validated for defects above `X` px²"). Each has a different cost, so the fix follows the diagnosed cause, not the symptom.

**Mitigation risk.** Tiling multiplies inference cost and can duplicate detections at tile seams, adding false positives; higher resolution increases both memory and latency. Neither is free, and neither may be adopted without re-running the whole tuple.

**Verification.** Block release on the pre-declared size gate (for instance, small-band recall ≥ 0.60 with a lower interval bound above 0.55), and confirm the gate is a *promotion* condition rather than a report line.

### F-04 — Calibration shift after a silent pipeline change

**Symptom.** Scores drift upward on one line while accuracy is unchanged. Operators report "the model became more confident" and the alert volume rises by 12%.

**Competing causes.** (a) A preprocessing change (resize kernel, colour order, normalization constants) altered the input distribution. (b) A firmware or lens change altered the sensor response. (c) Prevalence shifted, so the same conditional probabilities now appear against a different base rate. (d) Genuine quality improvement. (e) The calibration set was reused so many times it no longer represents the population.

**Discriminating check.** Replay an immutable fixture set (fixed bytes, fixed expected outputs) through the new pipeline. A fixture mismatch localizes the change to code or preprocessing within minutes. If fixtures pass, compare input statistics (channel means, histogram distance) and then prevalence by slice. Cause (e) is detected by governance, not by statistics: check the calibration-set access log.

**Bounded fix.** Restore the compatible pipeline digest, or recalibrate on a representative fresh sample under the freeze-and-version policy. Never recalibrate against the same records used to select the previous calibrator.

**Mitigation risk.** A fresh calibrator applied to a broken pipeline hides a representation failure behind a monotone correction: the probabilities look right while the features are wrong. Calibration must follow, not precede, the pipeline check.

**Verification.** Reliability diagram with support per bin, Brier score, and alert volume by slice, before and after, all on the decision population.

### F-05 — Denominator drift in a headline metric

**Symptom.** A report states recall improved from 0.62 to 0.70. No eligibility table exists, and the two runs disagree on how many positives were eligible.

**Competing causes.** (a) A legitimate new quality exclusion was added. (b) A silent filter dropped hard images, for example those with a missing metadata field used in a join. (c) The unit changed from object to image. (d) Labels matured and some records flipped. (e) Duplicate records were de-duplicated in one run only.

**Discriminating check.** Rebuild TP, FN, exclusions, and candidate population from immutable cohort IDs, and diff the cohort manifests by record ID rather than by count. A count-only diff cannot distinguish (a) from (b); an ID diff can.

**Bounded fix.** Publish the cohort manifest as an artifact, freeze exclusion rules in the contract, and require an ID-level diff for any comparison. Recompute both numbers under the *old* manifest before claiming an improvement.

**Mitigation risk.** Restoring the old manifest may lower the metric while making it honest; expect the conversation about "why did the number get worse" and pre-empt it with the tuple diff.

**Verification.** An independent reviewer recomputes numerator and denominator from the manifest alone, without access to the model or the original script.

### F-06 — Recall improvement that is pure threshold movement

**Symptom.** "Recall rose from 0.62 to 0.76." The threshold moved from 0.70 to 0.50; no retraining happened.

**Competing causes.** (a) Threshold movement. (b) A genuinely better ranking. (c) A changed matching rule (IoU 0.75 → 0.50). (d) A changed detection cap. (e) A changed ignore policy.

**Discriminating check.** Compare the full PR curves, not points. If the curves coincide, the gain is a threshold effect (a). If the new curve dominates, the ranking improved (b). Then read the tuple: any change in `matching`, `cap`, or `ignore` explains (c)–(e) without a model change.

**Bounded fix.** Report AP (threshold-free) alongside the operating point, and report both at a fixed, identical threshold for the comparison. State the threshold in the same sentence as the recall, always.

**Mitigation risk.** Reporting only AP hides the deployment question: a better ranking can be irrelevant if the operating point is chosen for other reasons. Report both, always, and never substitute one for the other in a promotion.

**Verification.** Recompute both models' curves on the same frozen cohort and confirm the claimed curve dominance with the per-slice curves as well as the aggregate.

### F-07 — Duplicate and near-duplicate detections inflating true positives

**Symptom.** Object recall rises above what visual inspection suggests. The model appears to find more defects than a human can see in the same images.

**Competing causes.** (a) Missing NMS or a per-class NMS that does not suppress across classes. (b) A lenient matching rule that allows several predictions to match one label. (c) Genuinely more defects than the label captured. (d) Test-time augmentation whose views are treated as independent detections and not merged.

**Discriminating check.** Count predictions per label per image; the maximum must be 1 under a correct one-to-one rule. If it exceeds 1, the matching implementation is at fault (b), regardless of what NMS does. Then inspect the top-10 overlapping prediction pairs by IoU to see whether NMS is functioning.

**Bounded fix.** Enforce one-to-one matching in the evaluator (never rely on the model's NMS for correctness of the metric), and merge TTA views before matching rather than after.

**Mitigation risk.** Strict one-to-one matching can turn genuine multi-instance clusters into false positives when labels are merged; verify the label policy for touching objects before tightening.

**Verification.** Assert in the evaluator that `matches_per_label ≤ 1` and `matches_per_prediction ≤ 1`, and fail the run rather than reporting a number when the assertion breaks.

### F-08 — Aggregate improvement with action-slice regression

**Symptom.** mAP improves by 2 points; the action-critical slice, or one site, degrades. The aggregate hides it because the slice is small.

**Competing causes.** (a) Genuine trade-off from added capacity on the dominant slice. (b) Prevalence shift in the test cohort between runs. (c) A class-map change that moved objects between classes. (d) A slice-definition change (band edges moved). (e) Random variation in a small slice.

**Discriminating check.** Freeze the slice definitions and recompute per-slice AP for both models on the same cohort. Then check the class map and the band edges by diffing the configuration. Use the Wilson interval on the slice to test whether the degradation exceeds noise.

**Bounded fix.** Promote on a Pareto rule: no blocking slice may regress beyond its interval, and the aggregate must improve. If the trade-off is genuine and acceptable, record the acceptance decision with the harm it implies.

**Mitigation risk.** A Pareto gate can block genuinely better models when slices are underpowered; the correct response is more labels in the slice, not a lower gate.

**Verification.** Re-run the gate script with both checkpoint digests and the same cohort; the gate must be executable, not a prose commitment.

### F-09 — Operating threshold copied across sites

**Symptom.** A threshold tuned at one site produces an alert rate of 5% there and 22% at a second site with the same model.

**Competing causes.** (a) Different prevalence. (b) Different capture characteristics, shifting the score distribution. (c) Different label policy between sites, so what counts as a defect differs. (d) A different reviewer population with different tolerance, which is a workflow issue rather than a model issue.

**Discriminating check.** Plot score histograms by site on the same axes and compare the positive/negative conditional distributions separately from the marginal. If the conditionals are similar and only the marginal differs, (a) dominates and the threshold is doing the right thing on a different mix. If the conditionals differ, (b) or (c) dominates.

**Bounded fix.** Set thresholds per site against local prevalence and local capacity, and record each site's threshold in the contract. Alternatively recalibrate per site with enough local data; with few local labels, prefer abstention to extrapolation.

**Mitigation risk.** Per-site thresholds multiply the number of decisions to monitor and can encode local bias; audit the *outcome* rates per site, not just the thresholds.

**Verification.** Compare alerts/day, precision, and recall per site against that site's pre-declared gates for a full week before and after the change.

### F-10 — Metric computed on the wrong population

**Symptom.** A published precision of 0.92 cannot be reproduced operationally; the live precision is nearer 0.70.

**Competing causes.** (a) The evaluation cohort excluded hard cases (for example, low-quality captures). (b) The evaluation used a curated benchmark rather than the deployment stream. (c) The operational population includes an image type the model was never trained or tested on. (d) Label maturity differs. (e) The live system applies a different threshold.

**Discriminating check.** Compare the covariate distributions (resolution, exposure, device, time of day) between the evaluation cohort and a week of live traffic. A large discrepancy localizes (a) or (b) or (c). Check the deployed configuration digest for (e) first — it is the cheapest check and the most common cause.

**Bounded fix.** Rebuild the evaluation cohort from a random sample of live traffic, label it under the same policy, and re-report. Where live traffic contains an unsupported type, add an explicit abstain path rather than silently scoring it.

**Mitigation risk.** A live-traffic sample includes the label maturity problem; schedule the evaluation after the maturity cut-off or accept a bounded, stated optimism.

**Verification.** Reproduce the reported number from the new cohort manifest, then compare the live alert outcomes for the same week.

### F-11 — Review-queue collapse from a "better" model

**Symptom.** Recall rises 8 points; the review backlog triples within four days; reviewers begin rubber-stamping, and the effective review precision falls well below the model's measured precision.

**Competing causes.** (a) Alert volume exceeded capacity (utilisation near 1). (b) Service time increased because alerts are harder to adjudicate. (c) Reviewer fatigue, a human-factor effect that degrades label quality and thus future ground truth. (d) The queue's triage order changed, front-loading hard cases. (e) A second workflow (rework) has an independent capacity limit that was not modelled.

**Discriminating check.** Plot alerts/day against reviewer-hours/day, and utilisation against measured wait time. If utilisation exceeds ~0.85 and wait time grows superlinearly, (a) dominates. Measure adjudication time per alert before and after to test (b). Sample reviewed alerts to estimate agreement between disposition and a blind re-read, which tests (c).

**Bounded fix.** Introduce or widen an abstain band, add a cheap second-stage filter for the abstain band, or return to the higher threshold with an explicit statement of the recall forgone — stated in cases per day, not in points.

**Mitigation risk.** An abstain band can become a hidden threshold if the abstain queue is drained without review; instrument the abstain disposition rate, not just its size.

**Verification.** Track backlog, 95th-percentile wait, and blind re-read agreement weekly; the gate is "backlog stable or falling at the target recall", not "recall achieved".

### F-12 — Unclassifiable remainder in the error register

**Symptom.** The FN partition closes to 190 of 216; 26 labels cannot be assigned to any category after review.

**Competing causes.** (a) A join bug between predictions and labels (frame, image id, or version mismatch). (b) Ignore-region removal applied to one side only. (c) Missing or corrupt prediction records for a subset of images. (d) Genuinely novel error types outside the taxonomy. (e) Duplicate label IDs that overwrite each other in a dictionary.

**Discriminating check.** For ten unclassified cases, print the raw prediction record, the raw label record, and the join key. If either side is absent, it is (a) or (c). If both are present but the overlap computation returned a sentinel, it is (b) or (e). Only after those are excluded is (d) a legitimate finding.

**Bounded fix.** Add invariants: every eligible label appears exactly once; every prediction record belongs to an evaluated image; the taxonomy partition sums to the FN count. Fail the run when an invariant breaks.

**Mitigation risk.** Forcing a remainder into the nearest category destroys the signal that something structural is wrong; an "unclassified" category is safer than a wrong classification, provided it is small and investigated.

**Verification.** Re-run the partition with the invariant asserts enabled; the remainder must be zero or explicitly explained case by case.

### 8.2 Cross-cutting edge cases

| Edge case | What breaks | Correct handling |
|---|---|---|
| Zero positives in a slice | Recall undefined; some tools print 0 | Report `undefined, n_pos = 0`; exclude from macro with a count |
| Zero predictions at a threshold | Precision undefined | Report `undefined, n_pred = 0`; never 0 or 1 |
| Both masks empty | IoU 0/0 | Declare the convention; report the number of such pairs |
| Single-pixel object | IoU highly sensitive to any offset | Use a tolerance-based or center-distance metric for that class |
| Many objects per image (> cap) | Recall ceiling from the cap | Raise the cap; report the cap; count truncated images |
| Ties in score | Ordering changes matching and AP | Break ties by a stable key such as image id then box index |
| Mixed resolutions in one cohort | Size bands incomparable | Band by native area; report post-resize area too |
| Multi-label image | "Accuracy" ambiguous | Report per-label precision/recall and exact-match ratio separately |
| Label flips after maturity | Cohort changes silently | Freeze labels by maturity date in the manifest |
| Class with 3 examples | AP on 3 labels is unstable | Report support; do not rank models on it |
| Ignore region covering the object | Object neither TP nor FN | Ensure the ignore policy is symmetric for prediction and label |
| Evaluator version change | Numbers move with no model change | Pin the evaluator version in the tuple; diff before comparing |

### 8.3 Escalation rules

Stop the comparison and escalate when: the cohort manifest cannot be produced; two runs disagree on the eligible count and no ID diff exists; the taxonomy remainder exceeds 5% of FN; the evaluator version changed between the compared runs; or a blocking slice has an interval that spans both the gate value and the previous value. In each case the correct action is to restore comparability first. Every one of these is a *protocol* failure, which means no amount of model work will fix it, and any model work done during the investigation will be wasted because it will be judged against an invalid baseline.

**Takeaway:** Diagnose the superclass first — protocol, input, model, or decision — because the fix class follows the superclass and nothing else.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic complexity of the evaluation itself

Define the sizes first, because evaluation cost is usually misattributed to the model: `n` evaluated units (images), `p` predictions per image (after the cap), `g` labels per image, `H×W` mask dimensions, `s` declared slices, and `C` classes.

| Operation | Input sizes | Time | Space | Dominant cost / assumption |
|---|---|---|---|---|
| Confusion counts | n, C | O(n·C) | O(C²) | One pass; trivially cheap |
| Precision/recall at one threshold | p, g | O(n·(p+g)) | O(n) | Counting after matching |
| IoU per pair | H, W | O(H·W) | O(1) extra | Pixel-cardinality intersection |
| Greedy matching | p, g | O(p·g·log(p·g)) | O(p·g) | Sort by score dominates |
| Optimal (Hungarian) matching | p, g | O(max(p,g)³) | O(max(p,g)²) | Fine for p ≤ ~100; not for dense scenes |
| AP per class | N_pos, D detections | O(D log D) | O(D) | Sort once, sweep once |
| mAP | C, D_c | O(Σ D_c log D_c) | O(Σ D_c) | Linear in classes after sorting |
| PQ | n, segments | O(Σ (P_i·G_i)) matching + O(H·W) masks | O(H·W) per image | Matching dominates in dense images |
| Slice aggregation | n, s | O(n·s) naive | O(s) | Predeclare s to bound the scan |
| Calibration/bins | n, B | O(n) | O(B) | One pass given bin edges |
| Queue model | — | O(1) | O(1) | Closed form; inputs must be measured |

Two practical consequences follow. First, the optimal-matching choice is a cost decision: at `p = g = 100`, `max(p,g)³ = 10⁶` operations per image against roughly `10⁴·log` for greedy — two orders of magnitude — and the cardinality difference is usually small. Evaluate greedy for the production metric and report optimal-matching on a sample to bound the discrepancy. Second, mask IoU at high resolution is memory-bound, not compute-bound: a 2,048×2,048 mask is 4.19 M pixels, and evaluating 50,000 images with pairwise comparisons over 10 segments each is `50,000 × 90 × 4.19×10⁶ ≈ 1.89×10¹³` pixel comparisons. Downsampling masks for *matching* and using full resolution only for the reported quality changes the cost by the area ratio, which is why mask-evaluation scripts are often slower than the model that produced the masks.

### 9b. Worked sizing: the cost of evaluating, not just training

**Workload (ILLUSTRATIVE).** 50,000 test images/day, `p = g = 8` on average with a cap of 100, 4 classes, `s = 12` slices, masks not used (boxes only), one threshold sweep over 20 candidate thresholds.

```text
Step 1  IoU pair computations per image: p * g = 8 * 8 = 64 pairs.
        Total pairs        = 50,000 * 64            = 3,200,000
        Pair cost (box)    = ~10 operations        -> 3.2e7 operations: negligible.

Step 2  Sorting for greedy matching: p log p = 8 * 3 = 24 comparisons/image.
        Total              = 50,000 * 24            = 1,200,000 comparisons.

Step 3  Threshold sweep: matching is monotone in the threshold *only* if the
        matching rule is recomputed from scratch per threshold. Doing so:
        total work         = 20 sweeps * 3.2e6 pairs = 64,000,000 pairs.
        This is the step that makes "sweep and choose later" expensive at scale.

Step 4  Slice aggregation: n * s = 50,000 * 12 = 600,000 row updates.
        Space: 12 rows of counters, constant.

Step 5  Human review at a = 0.78, s_review = 2 min:
        alerts             = 50,000 * 0.78 = 39,000/day
        reviewer-hours     = 39,000 * (2/60) = 1,300 h/day
        reviewers @6.8 h   = 1,300 / 6.8 = 191.2 -> 192 people
```

The asymmetry is the lesson: the machine evaluation is cheap (tens of millions of operations, seconds on a laptop) while the human consequence is 192 full-time reviewers. Any evaluation design that spends machine effort to reduce human effort — an abstain band, a second-stage model, a better triage order — is worth far more than a marginally better metric. This is the correct frame for "evaluation cost": the dominant term is usually the human loop the metric authorises, not the script.

### 9c. Cost per decision, and where it hides

The costing convention must match the derivation, or the totals will not reconcile with Section 5. `c_FP` is the *incremental* cost of a false alarm, because an alert is reviewed under either policy; `c_FN` is the cost of an escape. Charging review time to every alert as well would double-count the review of true positives, which is a fixed cost of operating the detector at all.

```text
Incremental cost per 1,000 images at tau=0.50 (matches Section 5.5):
  false-alarm review = 216 FP * EUR 2                    = EUR    432
  escape             = 216 FN * EUR 240                   = EUR 51,840
  total                                                    = EUR 52,272
  per image                                                  = EUR 52.27

Same model, tau=0.30:
  false-alarm review = 490 * 2                            = EUR    980
  escape             =  90 * 240                          = EUR 21,600
  total                                                    = EUR 22,580
  per image                                                  = EUR 22.58
```

A 2.3× difference in cost per image sits inside one model and one evaluator, controlled entirely by a threshold. Yet the capacity constraint of four reviewers makes the cheaper row infeasible without structural change. That tension — cheap policy, infeasible policy — is the normal state of an evaluation review, and it is resolved by changing structure (abstain band, second stage, ontology), not by choosing a point on the curve and hoping.

Note also what the total is dominated by: 99.2% of the `τ=0.50` cost is escapes (51,840 of 52,272). The review cost is not the expensive part; the review cost is what *constrains* the decision, while the escape cost is what the decision is *about*. Those are two different roles, and conflating them produces the classic error of cutting review budget to "save money" when review is the mechanism that bounds escapes.

### 9d. Sensitivity of cost to the two inputs that are usually guessed

Both unit costs are estimates, and the decision's robustness depends on how far they can move. In Section 5, a 15× reduction in `c_FN` (240 → 40) shrank the cost gap between `τ=0.30` and `τ=0.50` from EUR 29,692 to EUR 4,492 while leaving the ranking unchanged. A 3× increase in review time (2 → 6 min) changes the reviewer requirement at `τ=0.50` from 3.82 to 11.46, making the policy infeasible at any plausible headcount. So the decision is robust to error in the harm estimate and fragile to error in the service-time estimate. That asymmetry tells you which measurement to make first: time ten real reviews before debating the cost of an escape.

**Takeaway:** Bound the human loop first; the machine evaluation is almost never the bottleneck.

## 10. Key Theorems / Results Reference

The results below are the load-bearing statements behind the rest of the sheet. Each is stated precisely enough to use, with the essential hypotheses and the overclaim it forbids. Proofs are sketched or cited; none is reproduced in full.

| Result | Correct statement | Essential hypotheses | Practical implication | Overclaim it forbids |
|---|---|---|---|---|
| Decomposition of counts | For a frozen matching at one threshold, `accepted = TP + FP` and `eligible positives = TP + FN`; hence precision and recall are determined by the partition | One-to-one matching; complete label set for evaluated images; ignore policy applied symmetrically | Every rate is reconstructible from the partition, so a report without the partition is unverifiable | That two rates with the same value came from the same counts |
| AP as a Riemann–Stieltjes integral | For a step recall function with jumps `1/N_pos` at TP ranks, `AP = ∫ P* dR = (1/N_pos) Σ_{TP} P*_k` | Complete labels; sorted scores; monotone envelope; fixed cap and IoU | AP is threshold-free but not cap-free or label-free | That AP is a property of the model alone |
| Interpolation dependence (VOC2007 vs VOC2010/COCO) | 11-point interpolation is a lower-resolution quadrature of the same integral and generally differs from all-point | Same detection list; same N_pos | Comparing AP across interpolation schemes compares schemes | That a 0.03 AP gain from a script change is a model gain |
| IoU–Dice bijection | `Dice = 2·IoU/(1+IoU)` is a strictly increasing bijection on `[0,1]` | Same region pair; `U > 0` | Thresholds do not transfer: Dice 0.5 corresponds to IoU 1/3 | That mean Dice equals the Dice of mean IoU |
| Macro ≠ micro aggregation | Macro mAP weights classes equally; support-weighted mAP weights by `n_c`; they coincide only when supports or APs are equal | Per-class protocol identical | Report the aggregation and the supports | That a higher mAP is better for the workflow without the weighting rule |
| PQ factorization | `PQ = SQ × RQ` with `RQ = TP/(TP + ½FP + ½FN)` and one-to-one matching at IoU > 0.5 | Void policy declared; instance segmentation task | Separates "found it" from "drew it well" | That PQ is comparable across ontologies |
| Properness of the Brier score | The expected Brier score is minimized by reporting the true conditional probability | Binary outcome; the same population and information set | Brier rewards honest probabilities, unlike accuracy or ECE | That a lower Brier implies better decisions without a cost model |
| Threshold optimality under binary cost | With calibrated `p` and constant costs, expected cost is minimized at `p* = c_FP/(c_FP + c_FN)` | Calibration on the decision population; constant costs; independent decisions; no capacity limit | Gives the unconstrained benchmark against which a constrained choice is argued | That the optimum is deployable |
| Little's law | `L = λW` for a stable system in steady state | Long-run averages; conserved flow; stationarity | Backlog `= arrival rate × mean time in system`; a capacity screen | That it bounds tail latency or burst behaviour |
| Hoeffding bound | `P(|p̂ − p| ≥ ε) ≤ 2exp(−2nε²)`, giving `n ≥ ln(2/α)/(2ε²)` | Independent bounded observations in `[0,1]` | Sizes a slice before claiming a comparison | That correlated (clustered) records satisfy the bound |
| Wilson interval | Score interval for a binomial proportion, valid at small `n` and extreme `p` | Independent trials; fixed `n` | Reporting uncertainty beside every slice rate | That overlapping intervals prove equality |
| No-free-lunch for metric choice | No scalar summary is optimal for all decision problems | Broad class of problems, uniform prior over them | Metric choice is a modelling decision that must be justified by the cost structure | That a standard metric is automatically appropriate |

Three of these deserve expansion, because they are the ones most often misquoted.

**Little's law in review queues.** `L = λW` says the average number in the system equals the arrival rate times the average time in the system, for any stable system with conserved flow. It is remarkable because it assumes almost nothing about distributions. Its weakness is exactly what people ask of it: it says nothing about the 95th percentile. For a review queue, use it as a consistency check — if alerts arrive at 39,000/day and the average alert sits 0.5 days in the system, then 19,500 alerts are open, which either matches the dashboard or reveals a measurement error.

**Hoeffding and clustering.** The bound requires independent, bounded observations. Images from the same machine, patient, or recording are not independent, so the effective sample size is smaller and the bound is optimistic. The honest move is to compute the bound at the group level (clusters, not images) or to report a cluster-robust interval. A team that sizes a slice with the image-level bound and then evaluates it on six machines has sized the wrong thing.

**Wilson versus normal approximation.** The normal approximation `p̂ ± 1.96√(p̂(1−p̂)/n)` produces intervals that extend below 0 and above 1 at small `n` and collapse to zero width at `p̂ = 0` or `1`. The Wilson interval does not. Since slice rates are routinely at 0.98 or 0.40 with modest `n`, the normal approximation is not a conservative simplification; it is wrong in exactly the regime the slices occupy.

**What this section does not establish.** None of these results says which metric a workflow should use, and none gives a threshold. They constrain the arithmetic and the uncertainty statement; the decision still requires costs, capacity, and a stated harm model, all of which are judgement inputs labelled elsewhere as ILLUSTRATIVE or OPEN.

**Takeaway:** The theorems constrain what a number can mean; they never choose the number for you.

## 11. Interview / Exam Questions

Fourteen questions across three levels. Each carries the prompt, a checked answer with reasoning or arithmetic, a follow-up that changes a constraint, a red flag with the mechanism that makes it wrong, and a scoring note. Answers reference equation IDs so they can be checked against Section 4.

### Level 1 — Definition and recall

**Q-01 — Why does a random image split overstate performance for video?**

*Prompt.* You have 20,000 frames from 400 recordings. A random 80/20 split gives 0.95 accuracy; a recording-level split gives 0.72. Which number do you report, and why?

*Answer.* Report the recording-level number for any generalisation claim, because the random split evaluates a population the deployment will not have: at serving time the frames come from recordings the model has never seen. Adjacent frames share scene, subject, lighting, exposure, and compression history, so a random split places near-duplicates on both sides. The gap of 0.23 is a lower bound on the memorisation share, because the model may also have learned something genuinely transferable across recordings. Keep the random-split number only as a diagnostic labelled "within-recording performance", never as a promotion metric.

A valid alternative exists: if the serving contract genuinely provides earlier frames of the same recording (online tracking), then a scene-stratified split is the right estimand — but the contract must be documented in the tuple, not assumed from the presence of frames.

*Follow-up.* If the recordings come from 12 cameras and the deployment adds a 13th, which split do you need? *Answer:* a camera-level holdout, because the dependency that will be absent at serving time is the camera, not merely the recording.

*Red flag.* "Random splitting is standard practice, so it is fine." The mechanism is dependence between samples; the standard practice is wrong whenever the samples are not exchangeable, and frames of one recording are not.

*Scoring.* 0 = says random is fine. 1 = says group split without stating why. 2 = states the dependency and the estimand change. 3 = also distinguishes the online-history case and names the correct split for a new camera.

**Q-02 — What exactly does IoU measure, and what does it not measure?**

*Prompt.* Define IoU for boxes and for masks, then name one decision it cannot support alone.

*Answer.* IoU is the cardinality of the intersection divided by the cardinality of the union of a predicted and a labelled region, in the same coordinate frame and class (`EV-01`). For boxes the cardinality is an area in pixels²; for masks it is a pixel count. It measures geometric agreement and nothing else: it does not identify whether the disagreement is translation, scale, a boundary-policy difference, or a different object. Identical IoU values arise from mechanically different failures (Section 6.1), so an IoU-driven decision needs a second view — an overlay, a size band, or a centre-distance check — before an intervention is chosen. IoU also does not measure class correctness: a perfectly localized box of the wrong class has IoU 1.0 and is a classification error.

*Follow-up.* Both masks are empty. *Answer:* `0/0`; the protocol must declare a convention (1, 0, or excluded) and report the number of such pairs, because the choice can move a dense-prediction metric by tens of points.

*Red flag.* "IoU above 0.5 means the detection is correct." At IoU 0.5 a box can be twice the true area or displaced by half its width; the threshold is a convention, not a correctness criterion.

*Scoring.* 0 = confuses IoU with accuracy. 1 = gives the formula only. 2 = gives the formula and the empty-mask caveat. 3 = also explains that IoU 0.5 admits substantial scale and shift error, and that class is separate.

**Q-03 — Why is precision undefined rather than zero when there are no alerts?**

*Prompt.* A threshold above every score produces no predictions. What is the precision, and what should the report say?

*Answer.* `precision = TP/(TP+FP) = 0/0`, which is undefined. Reporting 0 says the model was wrong on every alert it made when it made none; reporting 1 says it was perfect on that same empty set. Neither statement has a referent. The report must say "undefined at this threshold, alerts = 0" and the precision–recall curve must mark the endpoint rather than interpolate through an invented value. The same discipline applies to recall when a slice has zero positives: `0/0`, undefined, and the slice is excluded from a macro average with its exclusion counted.

*Follow-up.* A dashboard averages precision across 300 operating points, ten of which have zero alerts. *Answer:* the average is corrupted by ten invented values; exclude them and state the count, or plot the curve instead of averaging it.

*Red flag.* "Precision is 0 there, so the model is useless at high thresholds." The model is abstaining, which may be exactly the desired behaviour when the queue is full.

*Scoring.* 0 = says 0 or 1. 1 = says undefined without consequence. 2 = states the convention and the reporting rule. 3 = also generalises to zero-positive slices and to averaging over degenerate points.

### Level 2 — Derivation and application

**Q-04 — Derive average precision from the precision–recall curve.**

*Prompt.* Given sorted detections with cumulative true positives, derive `AP = (1/N_pos) Σ_{TP ranks} P*_k` and use it on the sequence TP, FP, TP, TP, FP, TP, FP, FP with `N_pos = 5`.

*Answer.* Cumulative `TP_k` gives `P_k = TP_k/k` and `R_k = TP_k/N_pos`. `R` is a non-decreasing step function whose jumps occur only at true positives, each by `1/N_pos`. AP is the area under the monotone envelope `P*_k = max_{j≥k} P_j`, which is `∫P* dR`; a step function integrates by summing value times jump, so the integral collapses to the average of the envelope at the jump points. Numerically: `P` and `R` at each rank are `(1.000,0.2), (0.500,0.2), (0.667,0.4), (0.750,0.6), (0.600,0.6), (0.667,0.8), (0.571,0.8), (0.500,0.8)`. The envelope at the four TP ranks is `1.000, 0.750, 0.750, 0.667`, whose sum is `3.1667`, so `AP = 3.1667/5 = 0.6333`. Note recall reaches only 0.8 because one label is unmatched; AP does not penalize the curve for that beyond the missing jump, which is why `N_pos` must be complete and correct.

*Follow-up.* The same eight detections under 11-point interpolation score higher. *Answer:* 11-point samples `max P` at eleven recall levels, which is a coarser quadrature of the same integral; it generally differs and is not a model improvement.

*Red flag.* "AP is the average of precision values." That averages over all ranks including false positives and ignores the recall weighting, producing a different and non-comparable number.

*Scoring.* 0 = cannot state the formula. 1 = states it without derivation. 2 = derives from the step function and computes correctly. 3 = also explains the incomplete-recall case and the interpolation dependence.

**Q-05 — Compute PQ and explain why it is not just mean IoU.**

*Prompt.* 40 matched segments with mean IoU 0.82, 10 unmatched predictions, 6 unmatched labels. Compute PQ and state what the two factors each represent.

*Answer.* `RQ = 40/(40 + 0.5·10 + 0.5·6) = 40/48 = 0.8333`; `PQ = 0.82 × 0.8333 = 0.6833`. The factorization separates recognition (`RQ`: did the system find and merge segments correctly?) from segmentation (`SQ`: how well do matched segments overlap?). Mean IoU alone would report 0.82 and conceal that the system invented 10 segments and missed 6 — behaviour that matters operationally because invented segments become false alarms and missed ones become escapes. Because `RQ` uses half-weights on FP and FN, it is an F1-style measure of segment-level recognition, so a system that over-segments is penalized on `RQ` while keeping `SQ` high.

*Follow-up.* Compare two systems with equal PQ on datasets with different class granularity. *Answer:* the comparison is invalid; merging classes changes TP, FP, and FN simultaneously, so PQ is not portable across ontologies.

*Red flag.* "PQ is just mean IoU with a threshold." The product structure is the definition; dropping `RQ` changes the estimand.

*Scoring.* 0 = computes mean IoU only. 1 = computes PQ without the RQ interpretation. 2 = computes both and interprets. 3 = also states the ontology non-portability and the threshold dependence.

**Q-06 — Turn a cost model and a capacity limit into one threshold decision.**

*Prompt.* `c_FP = EUR 2`, `c_FN = EUR 240`, arrival 1,000 images/day, 8-hour shift, 2 minutes per review, 4 reviewers at 0.85 utilisation. The cost optimum is `p* = 0.0083`. What do you deploy?

*Answer.* Compute both constraints and take the intersection. Capacity: `4 × 0.85 = 3.40` reviewer-hours per clock hour; at `s = 2 min = 0.0333 h` that is `102` alerts per hour, `816` per 8-hour day, or `81.6%` of images. From the sweep, `τ = 0.30` produces 940 alerts/day (infeasible), `τ = 0.50` produces 780 (feasible on the mean, utilisation 0.956), `τ = 0.70` produces 520. The cost-optimal threshold is infeasible, so the deployed point is the cheapest feasible one: `τ = 0.50`, with the utilisation shortfall and the recall forgone stated in cases per day. The correct framing is that `p*` is the unconstrained benchmark that justifies arguing about capacity — it is what you would do if reviewers were free.

*Follow-up.* `c_FN` falls to EUR 40. *Answer:* totals become 4,580 / 9,072 / 13,860; the ranking is unchanged, but the penalty for choosing `τ=0.50` over the cheaper infeasible row shrinks from EUR 29,692 to EUR 4,492, so the capacity constraint — not the harm estimate — is now carrying the decision.

*Red flag.* "Use `p*` because it is optimal." It is optimal only within an unconstrained model whose capacity assumption is false; deploying it collapses the queue.

*Scoring.* 0 = uses `p*`. 1 = computes `p*` and ignores capacity. 2 = computes both and intersects. 3 = also quantifies the forgone recall in cases/day and notes the robustness asymmetry between `c_FN` and service-time error.

**Q-07 — Show that ECE depends on the binning.**

*Prompt.* Four records produce one ECE value with one bin and another with two. Demonstrate, and say what must be reported.

*Answer.* With one bin covering `[0,1]`, mean confidence is 0.50 and observed accuracy 0.50, so ECE is 0. With two bins, the 0.40-scored negative sits in a bin of mean confidence 0.40 and accuracy 0, a gap of 0.40, and the 0.60-scored positive sits at 0.60 against 1.00, another gap of 0.40; ECE is 0.40. The records and the model are identical. ECE is a weighted average of absolute bin gaps (`EV-07`), so it is a function of the partition as much as of the predictions. Report the bin edges, the count per bin, and the population; compare ECE values only under identical binning. Where a single calibrated summary is needed for model selection, prefer a proper scoring rule such as Brier (`EV-08`), which does not depend on bins.

*Follow-up.* One bin has 3 records. *Answer:* its gap is noise, and weighting it by `n_b/N` understates it while reporting it alone overstates it; show support per bin and refuse to interpret sparse bins.

*Red flag.* "ECE 0.02 means the model is well calibrated." It means the model is well calibrated *under that binning on that population*; a different binning or a different base rate can move it.

*Scoring.* 0 = cannot state the formula. 1 = states it without the binning issue. 2 = demonstrates and names the reporting requirement. 3 = also argues for a proper scoring rule and identifies the sparse-bin failure.

**Q-08 — Decompose an aggregate recall drop into slice and protocol components.**

*Prompt.* Recall fell from 0.78 to 0.72 after a "minor" checkpoint update. Give the diagnostic order.

*Answer.* Start with the protocol, because a protocol change can produce any delta for free: diff the tuple (unit, cohort, matching, threshold, cap, ignore, aggregation, slices) and the evaluator version. If unchanged, recompute per-slice recall on the frozen cohort and check the decomposition closes (`Σ TP` and `Σ FN` equal the aggregate). Then check the class map and band edges. Only if all of that is clean is the drop a model property. Decompose the delta by slice: if one slice fell 0.30 while the rest moved 0.01, the mechanism is localised and the next test is slice-specific; if every slice fell by a similar amount, suspect the threshold or the preprocessing digest. Finally check whether the drop exceeds the slice's Wilson interval — at 30 labels a 0.06 drop is well inside noise.

*Follow-up.* The evaluator version changed. *Answer:* the comparison is void; re-score both checkpoints with one evaluator version before interpreting anything.

*Red flag.* "Retrain with more data." Nothing in the diagnostic has yet established that the representation is the cause; retraining before the protocol is cleared produces a fourth number with an unknown tuple.

*Scoring.* 0 = retrains. 1 = checks slices only. 2 = protocol first, then slices, with decomposition closure. 3 = also uses intervals to separate noise from signal and names the threshold/preprocessing alternative.

### Level 3 — Judgement, design, and debug

**Q-09 — A vendor reports mAP 0.68 against your baseline 0.65. What do you require before believing it?**

*Prompt.* The vendor's number comes from their own report; your number comes from your harness.

*Answer.* Require the tuple: IoU threshold, detection cap, interpolation rule, ignore policy, class map, aggregation (macro or weighted), dataset version and split, and evaluator version. Then require a re-score of *their* checkpoint in *your* harness on the *same frozen cohort*, because cross-harness comparisons confound model and protocol. The 0.03 gap is smaller than the spread this sheet has already demonstrated from interpolation alone (0.673 versus 0.703 on 12 detections) and from aggregation (0.644 versus 0.746 on identical per-class results), so the reported gap is not yet evidence of anything. If the re-score stands up, then check the blocking slices before the aggregate.

*Follow-up.* They cannot ship the checkpoint. *Answer:* the claim is not evaluable; require a protocol-matched re-run on your cohort or treat the number as marketing.

*Red flag.* "Their mAP is higher, so their model is better." Cross-harness mAP comparisons measure harnesses as much as models, in both directions.

*Scoring.* 0 = accepts the number. 1 = asks for the dataset. 2 = demands a common harness and cohort. 3 = also names the two mechanisms (interpolation, aggregation) that can move the number by more than the claimed gain.

**Q-10 — Recall is 0.40 on the slice your action depends on, and 0.76 overall. What do you do?**

*Prompt.* Small-defect recall 0.40 (300 objects), medium 0.92, large 0.98. The approved headcount is four reviewers.

*Answer.* Treat the aggregate as irrelevant to this decision and the slice as the decision. Establish the cause before choosing the fix: compare native versus post-resize area (destroys information before the model sees it), check label density for unlabelled small objects, test a size-normalised localization tolerance, and cross-tabulate size against device. Then choose the intervention from the cause: capture resolution or tiling for the information case, labels for the annotation case, tolerance or assignment policy for the metric case. Capacity constrains the *threshold*, not this choice; a model that cannot see small defects cannot be rescued by lowering a threshold, it can only be made to alert more often on the ones it half-sees. Set the promotion gate on the slice (for example lower bound above 0.55) before running the next experiment, not after.

*Follow-up.* Tiling fixes small recall but doubles inference cost and duplicates detections at seams. *Answer:* measure the duplicate rate at seams and the latency budget; treat it as a new tuple requiring its own evaluation, not as a patch.

*Red flag.* "Overall 0.76 is acceptable, ship it." The stated goal is the action-critical band; the aggregate answers a different question and happens to pass.

*Scoring.* 0 = ships on the aggregate. 1 = notices the slice but picks a fix at random. 2 = diagnoses before fixing and gates on the slice. 3 = also separates the threshold lever from the representational lever and pre-declares the gate.

**Q-11 — Alert volume tripled after a recall improvement. Diagnose and bound the fix.**

*Prompt.* Reviewers now see 2,340 alerts/day against 780 before; the backlog grows; blind re-read agreement of dispositions is falling.

*Answer.* Compute utilisation first: alerts per hour times service hours against available reviewer-hours. If `ρ` exceeds roughly 0.85 and the measured wait grows superlinearly, the queue is the binding constraint, not the model. The falling blind re-read agreement is the second, worse signal: it means the ground truth for future evaluation is degrading, so the evaluation loop itself is now at risk. Bound the fix to structural changes — widen an abstain band with a cheap second-stage filter, or restore the previous threshold and state the recall forgone in cases/day. Do not simply add reviewers unless the budget is genuinely available and service time is measured; do not silently raise the threshold and report the old recall.

*Follow-up.* The second-stage filter accepts 15% of the abstain band. *Answer:* recompute alerts as `a_accept + 0.15·a_abstain`, recompute `ρ`, and verify that the filter's own errors are counted in the taxonomy rather than excluded.

*Red flag.* "Recall went up, so the system improved." The system's output is reviewed decisions; an unprocessable queue produces rubber-stamping, which is a quality regression dressed as a coverage gain.

*Scoring.* 0 = celebrates recall. 1 = adds reviewers. 2 = computes utilisation and bounds the fix structurally. 3 = also recognises the feedback contamination of future labels and instruments abstain dispositions.

**Q-12 — The taxonomy remainder is 26 of 216 false negatives. Is that a model problem?**

*Prompt.* After review, 190 of 216 FN are categorised; 26 are not.

*Answer.* Probably not a model problem, and almost certainly a protocol problem until proven otherwise. Print the raw prediction record, raw label record, and join key for ten remainders: if a side is absent, it is a join or completeness bug; if both are present but the overlap rule returned a sentinel, it is an asymmetric ignore policy or duplicate IDs overwriting in a dictionary. Only after those are excluded is a novel error type a legitimate finding. Add invariants — every eligible label exactly once, every prediction belongs to an evaluated image, the partition sums to `FN` — and fail the run when they break, so the remainder cannot silently persist.

*Follow-up.* The remainder persists with all invariants passing. *Answer:* it is a genuine taxonomy gap; add a category with a discriminating test rather than forcing cases into the nearest class, because a wrong category sends the next intervention in the wrong direction.

*Red flag.* "The unmatched ones are probably the hard cases." That is a causal story with no test; hard cases would still be classifiable as localization or detection misses.

*Scoring.* 0 = treats them as model failures. 1 = inspects a few without a rule. 2 = checks join/ignore/completeness first. 3 = also adds executable invariants and distinguishes a real taxonomy gap.

**Q-13 — Design the promotion gate for a vision model in four lines or fewer.**

*Prompt.* What must be in it, and what must it refuse?

*Answer.* It must contain: the frozen cohort manifest with group split and maturity cut-off; the tuple including evaluator version and matching rule; the aggregate metric with support; the blocking slices with executable thresholds and interval rules; the operating threshold with its utilisation and wait gates; the abstain route and its disposition tracking; and the rollback owner with a trigger. It must refuse: any comparison whose cohort or evaluator version differs; any aggregate-only claim for an action that is slice-specific; any threshold justified by cost alone without capacity; any slice verdict built on fewer labels than the interval requires; and any promotion where the label policy is under dispute. A gate that cannot fail is not a gate.

*Follow-up.* A blocking slice has 30 labels and the new model scores 0.45 versus 0.40. *Answer:* the interval spans both values, so the gate cannot resolve it; the correct action is more labels in that slice or a documented acceptance of uncertainty, not a verdict.

*Red flag.* "Promote if mAP improves." That criterion passes on aggregate gains that regress the action slice, and it has no capacity, no cohort, and no rollback.

*Scoring.* 0 = one metric. 1 = metric plus dataset. 2 = cohort, slices, threshold, capacity. 3 = also interval discipline, abstain disposition tracking, and rollback ownership.

**Q-14 — Change the constraint: the model must run on-device with a 40 ms budget. What changes in the evaluation?**

*Prompt.* The same task moves from a server batch to an embedded device with a hard deadline.

*Answer.* The deadline becomes part of the tuple, because it changes which inputs are legal: no multi-frame aggregation that requires future frames, no TTA that multiplies latency, no second-stage model unless it fits. The evaluation gains three gates: measured latency at the specified percentile on the target device (not theoretical throughput), behaviour under dropped or late frames, and accuracy at the *deployed* precision and resolution rather than the full-precision server configuration. Quantisation changes the score distribution, so the threshold and calibration must be re-derived on the device configuration, and the slice board must be re-run because quantisation error is not uniform across classes or scales. Capacity analysis changes too: review happens downstream, but the alert stream may now be bursty because store-and-forward batching changes arrival shape.

*Follow-up.* The device cannot meet 40 ms at the required recall. *Answer:* the decision becomes which requirement to relax and who owns the consequence — a narrower ontology, a smaller input, or a higher latency with a stated queue effect; the evaluation must be re-run under whichever tuple results.

*Red flag.* "The model is the same, so the accuracy is the same." Different precision, resolution, and latency envelope mean a different function; accuracy is a property of a configuration, not of a checkpoint name.

*Scoring.* 0 = ignores the constraint. 1 = mentions latency only. 2 = adds latency and re-derives the threshold. 3 = also covers quantisation's non-uniform slice effects, bursty arrivals, and the ownership of the relaxed requirement.

**Takeaway:** Every answer above is a protocol plus a number plus a boundary; an answer missing any of the three is a preference.

## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Topic | Vision evaluation and error taxonomy: freezing a decision contract (unit, cohort, deadline, matching, threshold, aggregation, slices) and attributing a failure to a mechanism before intervening |
| Core formula | `IoU = I/(P+G−I)`; `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`; `AP = (1/N_pos)Σ_{TP} max_{j≥k}P_j`; `p* = c_FP/(c_FP+c_FN)` subject to `ρ = λas / (m·h_r) < 1`. Critical validity cue: one unit, one frame, one class, complete labels, fixed cap and interpolation |
| Key assumption | The reported number is reconstructible from immutable cohort IDs plus the frozen tuple. The assumption most often violated is exchangeability: frames, machines, patients, or scenes straddle the split, and the score measures memorisation |
| Complexity | Matching `O(p·g log(p·g))` greedy versus `O(max(p,g)³)` optimal per image; AP `O(D log D)` per class; mask IoU `O(HW)` per pair. The dominant real cost is the human review loop, not the evaluator |
| Use when | You must promote, compare, or diagnose a visual model, and the action grain, cohort, and label policy can be written down and frozen |
| Avoid when | The cohort cannot be reconstructed, labels are immature or disputed, or two runs used different evaluator versions — in each case no metric is admissible, whatever its value |
| Go deeper | [02 CNNs, attention and transfer](02_CNNs_Attention_and_Transfer_Learning.md) for representation and preprocessing digests; [04 Detection and segmentation](04_Detection_and_Segmentation.md) for matching traces and output grain; [03 ViT](03_Vision_Transformers.md) for resolution and token budgets that move the small-object slice |
| Diagnostic | Classify first: protocol, input, model, or decision. A protocol failure is fixed by recomputing, never by retraining; a decision failure is fixed by threshold or queue design, never by a new checkpoint |
| Guardrail | Every slice rate ships with support and a Wilson interval; every threshold ships with its alert rate and utilisation; every comparison ships with a tuple diff |
| Evidence | DERIVED metric algebra, VERIFIED illustrative arithmetic, SOURCED convention definitions (IoU, AP interpolation, PQ); all workloads and costs are ILLUSTRATIVE. No benchmark, device, or deployment result is claimed; technical review is OPEN |

<!--END-->


