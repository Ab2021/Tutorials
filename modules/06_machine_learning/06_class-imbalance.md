# Class Imbalance

> **Module 06 — Machine Learning Foundations**
> Topic 06 of 13
> Estimated study time: 3 hours
> Prerequisites: Topics 05 and 08
> Version 2026.3

## 1. Why this topic exists

Class imbalance is the most over-treated problem in applied machine learning. A dataset with a 3
percent event rate is described as "imbalanced", a resampling technique is applied, the model's
apparent positive predictive value rises from 0.10 to 0.45, and the team reports a dramatic
improvement. The improvement is an artefact. The model has not become better at identifying the
event; the base rate has been changed, and the predicted probabilities now refer to a population
that does not exist.

The evidence is unusually clear. Van den Goorbergh et al. (2022) compared no correction with random
undersampling, random oversampling, and SMOTE (Synthetic Minority Over-sampling Technique) across
24 simulation scenarios and an ovarian cancer case study. Calibration intercepts fell to about
-4.5 in the 1 percent event scenarios, -2.1 at 10 percent, and -0.7 at 30 percent, against roughly
-0.05 to 0.03 without correction. AUROC was never improved by any correction method, and random
undersampling often reduced it. The corrections balanced sensitivity and specificity, but the same
balance could be achieved by shifting the threshold on an uncorrected model. Decision curves showed
reduced net benefit, with negative net benefit at thresholds of 0.3 or higher in the case study.

The harm is not confined to the probability scale. It reaches the decision curve, where a model
that appeared to improve at a fixed threshold is shown to reduce net benefit because its false
positives are concentrated among patients who would not have benefited. It reaches the model card,
where a reported PPV of 0.45 is quoted without the caveat that it refers to a 50 percent
prevalence. It reaches the deployment, where the threshold chosen on the resampled scale flags a
far larger fraction of the true population than intended.

The reason is straightforward once stated. For a risk model, the event rate is not a nuisance to be
corrected; it is the quantity the model is supposed to estimate. If the true 30-day readmission risk
is 3 percent, a model that predicts 30 percent for the same patient is wrong, even if it ranks
patients correctly. Resampling changes the effective prior and therefore changes the probability
scale. The ranking can survive; the probabilities do not.

Imbalance still matters, but for different reasons. It matters because accuracy becomes a
misleading metric: a model that predicts "no event" for everyone is 97 percent accurate and
useless. It matters because AUROC can look respectable while precision is terrible, since the
false positive rate is applied to a very large negative class. It matters because a small number of
events limits what can be estimated, especially in subgroups. And it matters because alert burden
at a low prevalence can make a well-discriminating model operationally useless. None of these
problems is solved by resampling the training data.

The same arithmetic explains why a rare-event model can be clinically useful and operationally
impossible at the same time. A PPV of 0.10 means ten reviews per true case; if each review takes 20
minutes, the programme spends 200 minutes per case found. Whether that is acceptable depends on the
value of the case and the cost of the review, not on the model's AUROC.

A model that flags 10 percent of a 3 percent population to find 70 percent of events is not a
failure of the model; it is the arithmetic of rare events. The team's job is to decide whether that
trade is worth the review capacity, and to communicate the trade to the programme owner. The model
cannot make that decision, and resampling cannot change the arithmetic. It can only obscure it by moving the reported PPV to a
prevalence that does not exist.

This topic separates the legitimate concerns from the harmful fixes. You will learn to quantify
imbalance, decide whether any correction is warranted, use class weights or threshold shifts when
they are, recalibrate to the deployment prior, and evaluate with precision-recall and decision
curves rather than accuracy or AUROC alone.

There is a second, subtler failure. A team that applies resampling because "the data are imbalanced"
may never ask whether the outcome is rare for a good reason. A 3 percent readmission rate is not a
data defect; it is the clinical reality that most discharged patients are not readmitted. A model
that predicts a 30 percent risk for a patient whose true risk is 3 percent is not a better model,
even if it catches more events at a fixed threshold. The team has confused the difficulty of the
problem with a flaw in the data.

Imbalance also interacts with the evaluation sample. If the validation cohort is resampled, the
reported precision refers to the resampled prevalence and cannot be compared with the deployment
workload. If the validation cohort is left at the true prevalence but the training cohort is
resampled, the model's probabilities are on the wrong scale and the threshold is not transferable.
Both errors are common, and both are detectable by asking one question: what prevalence does this
number refer to?

The practical remedy is a disciplined separation of three questions. What is the deployment
prevalence? What does the model need to output (a probability or a ranking)? What is the decision
rule and its capacity? The answers determine whether any correction is needed, and if so, which
one. They also determine which metrics are informative. A team that answers these questions before
touching the data avoids most of the errors in this topic.

> **Note:** For a risk model, the default should be no resampling. If the model is used only for
> ranking and the deployment prior is irrelevant, a correction may be harmless; if the model
> outputs probabilities that drive a threshold, a correction without recalibration is a defect.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** class imbalance from the decision problem, and **explain** why accuracy and
   AUROC can mislead at low prevalence.
2. **Compute** precision, recall, F1, AUPRC, and the precision baseline from a confusion matrix.
3. **Choose** between no correction, class weights, threshold shifting, and resampling, and
   **justify** the choice by the deployment goal.
4. **Apply** resampling inside cross-validation folds only, and **explain** why pre-split
   resampling leaks.
5. **Recalibrate** a model trained under a modified prior back to the deployment prior.
6. **Evaluate** a rare-event model with precision-recall curves, calibration, and net benefit.
7. **Quantify** alert burden and capacity at the operating threshold.
8. **Detect** the common artefacts of imbalance correction in a reported model.

Outcomes 2, 4, and 5 are computational; 3 and 6 are design; 8 is critique.

## 3. The core mental model

Imbalance is a property of the outcome distribution, not of the model. The model sees a sample with
a certain number of positives and negatives. Changing that ratio changes what the model estimates,
not how well it can separate the classes.

```text
True population:            P(event) = 0.03
                            |
                            +--> training sample: 3% positives
                                     |
                    +----------------+----------------+
                    |                                 |
             no correction                     resampling / weights
                    |                                 |
             model estimates P(y|x)            model estimates P_s(y|x)
             on the true prior                 on the modified prior
                    |                                 |
             calibrated by default             needs prior correction
                    |                                 |
                    +----------------+----------------+
                                     |
                              ranking (AUROC) roughly unchanged
                              probabilities changed
                              threshold meaning changed
```

The diagram also explains why two teams can report the same AUROC and different PPVs. The ranking is
a property of the feature-outcome relationship; the PPV is a property of the ranking plus the
prevalence plus the threshold. Changing the training ratio changes the second and third, not the
first. A report that gives only the first has not described the model's clinical behaviour.

The left branch is the default. The right branch is a deliberate change of estimand. Both can rank
well; only the left branch produces probabilities on the deployment scale without an extra step.

Three levers are available, and they act on different things.

**Data-level correction.** Oversampling duplicates minority cases; undersampling discards majority
cases; SMOTE synthesises minority cases by interpolating between neighbours. These change the
training distribution. They also change the effective sample size: undersampling throws away
information, and oversampling inflates the influence of a small number of cases.

**Algorithm-level correction.** Class weights multiply the loss contribution of each class. A
positive weight of 19 for a 5 percent prevalence gives the minority class roughly equal total
weight. This is mathematically similar to oversampling but does not duplicate rows, and it is
easier to implement inside a pipeline. It still changes the effective prior and still requires
recalibration.

**Decision-level correction.** Threshold shifting leaves the model alone and changes the cut-off.
For a fixed ranking, any sensitivity-specificity pair achievable by resampling is also achievable
by choosing a different threshold on the uncorrected model. This is the cleanest correction when
the goal is a decision rule rather than a probability.

The choice among the three depends on the goal. If the goal is a calibrated probability, use no
correction and, if necessary, recalibrate. If the goal is a fixed operating point, shift the
threshold. If the goal is to prevent a tree or neural model from ignoring the minority class during
training, use class weights and recalibrate. If the goal is compute or class-balance for a
downstream process that explicitly expects a balanced sample, resample inside the fold and correct
the prior.

The three levers also differ in what they do to the effective sample size. Undersampling reduces it,
which widens intervals and can lower AUROC. Oversampling leaves the number of distinct events
unchanged while increasing their weight, which narrows intervals artificially and can make the model
look more certain than it is. Weighting leaves the row count unchanged but changes the effective
prior. None of them adds independent information.

Precision-recall analysis is the right evaluation frame for rare events. Saito and Rehmsmeier
(2015) show that the precision-recall curve changes with the positive-to-negative ratio while the
ROC curve does not, and that the precision baseline equals the prevalence. A random classifier has
an AUPRC equal to the prevalence, not 0.5. That baseline is the reference against which a rare-event
model should be judged.

A useful diagnostic is to compute the confusion matrix at the planned threshold and read it as a
workflow: how many patients are flagged, how many true cases are found, how many are missed, and how
many flagged patients do not have the event. Those four numbers, not the area under a curve, are
what the programme experiences. The curve is a summary of all thresholds; the confusion matrix is
the threshold the programme will use.

> **Note:** AUROC is prevalence-independent by construction, which is why it can look good when
> precision is terrible. AUPRC is prevalence-dependent, which is why it must be reported with the
> prevalence and cannot be compared across cohorts with different prevalence.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Class imbalance | Unequal numbers of positives and negatives | Treated as a model defect | Ratio |
| Prevalence | Proportion of positives in the population | Confused with the training ratio | Proportion |
| Base rate | Synonym for prevalence | Assumed constant | Proportion |
| Minority class | The less frequent outcome | Assumed to be the class of interest | Label |
| Majority class | The more frequent outcome | Ignored in evaluation | Label |
| Accuracy | Proportion of correct predictions | Used for rare events | Proportion |
| Sensitivity | Proportion of positives correctly flagged | Called recall | Proportion |
| Recall | Synonym for sensitivity | Confused with precision | Proportion |
| Specificity | Proportion of negatives correctly not flagged | Ignored at low prevalence | Proportion |
| Precision | Proportion of flagged patients who are positive | Called PPV | Proportion |
| PPV | Positive predictive value | Assumed prevalence-independent | Proportion |
| NPV | Negative predictive value | Assumed high without checking | Proportion |
| F1 score | Harmonic mean of precision and recall | Treated as a utility measure | 0 to 1 |
| AUROC | Area under the ROC curve | Reported without prevalence | 0.5 to 1.0 |
| AUPRC | Area under the precision-recall curve | Compared across prevalences | 0 to 1 |
| Precision baseline | Prevalence; AUPRC of a random classifier | Assumed to be 0.5 | Proportion |
| ROC curve | Sensitivity versus false positive rate | Read as precision | Curve |
| PR curve | Precision versus recall | Read without the baseline | Curve |
| Resampling | Changing the training class ratio | Applied before splitting | Method |
| Oversampling | Duplicating or synthesising minority cases | Assumed to add information | Method |
| Undersampling | Discarding majority cases | Ignored loss of information | Method |
| Random oversampling | Duplicating minority rows | Confused with SMOTE | Method |
| Random undersampling | Removing majority rows | Confused with weighting | Method |
| SMOTE | Synthetic minority over-sampling by interpolation | Applied outside folds | Method |
| Synthetic example | An interpolated minority case | Treated as observed | Row |
| Class weight | Loss weight applied to a class | Confused with sample weight | Scalar |
| Weighted loss | Loss with per-class or per-row weights | Assumed to preserve calibration | Loss |
| Focal loss | Loss that down-weights easy examples | Assumed to fix imbalance | Loss |
| Threshold shift | Moving the cut-off on a fixed model | Confused with retraining | Probability |
| Prior correction | Adjusting probabilities from a modified prior to the deployment prior | Omitted after resampling | Log-odds shift |
| Calibration intercept | Systematic over- or under-prediction | Confused with the prior shift | Log-odds |
| Calibration slope | Over- or under-dispersion | Ignored after resampling | Slope |
| Alert burden | Alerts per unit of clinical work | Measured only after launch | Alerts per 100 |
| Capacity | Number of actions the system can deliver | Ignored in threshold choice | Patients per period |
| Net benefit | Prevalence-weighted true positives minus threshold-weighted false positives | Confused with F1 | Per patient |
| Decision curve | Net benefit across thresholds | Read as a discrimination curve | Curve |
| Rare event | An outcome with low prevalence | Defined by an arbitrary threshold | Proportion |
| Extreme imbalance | Prevalence below about 1 percent | Treated as a modelling problem | Proportion |
| Effective sample size | Sample size adjusted for the number of events | Equated with row count | Count |
| Event per variable | Number of events per candidate parameter | Ignored after resampling | Ratio |
| Stratified sampling | Preserving the class ratio in folds | Confused with resampling | Design |
| Data leakage | Information from outside the training fold entering the model | Caused by pre-split resampling | Defect |
| Duplicate row | A copied observation | Created by oversampling | Row |
| Prior probability shift | Change in prevalence between development and deployment | Confused with covariate shift | Proportion |
| Base-rate neglect | Ignoring prevalence when interpreting PPV | Common in clinical communication | Bias |
| Number needed to evaluate | Patients reviewed per true case found | Confused with number needed to treat | Count |
| Post-hoc calibration | Adjusting probabilities after fitting | Confused with resampling | Method |
| Platt scaling | Logistic recalibration of scores | Confused with prior correction | Method |
| Isotonic regression | Monotonic recalibration | Needs enough data | Method |
| Confusion matrix | Counts of TP, FP, TN, FN at a threshold | Reported without the threshold | Counts |
| Number needed to evaluate | Reciprocal of PPV | Confused with number needed to treat | Count |
| Prior probability | Synonym for prevalence | Confused with the training ratio | Proportion |
| Sampling ratio | Ratio of negatives to positives after resampling | Confused with prevalence | Ratio |
| Event count | Number of positive outcomes | Confused with prevalence | Count |
| Precision at threshold | Precision at the chosen operating point | Confused with AUPRC | Proportion |
| Sensitivity at fixed PPV | Recall at a required precision | Ignored in ranking comparisons | Proportion |
| Calibration-in-the-large | Mean predicted versus observed risk | Confused with slope | Log-odds |

Two definitions deserve more than a table row.

**Imbalance versus rarity.** A 50/50 dataset can have a rare subgroup, and a 3 percent event rate
can be perfectly adequate for a model if there are enough events. The relevant quantity is not the
ratio but the number of events and the decision's requirements. A cohort with 1,200 events and
38,800 non-events is imbalanced by ratio and rich by event count. A cohort with 30 events and
4,970 non-events is both imbalanced and underpowered. Resampling does not create events; it only
changes their weight.

**Correction versus recalibration.** A correction changes the estimand; recalibration changes the
scale of the output back to the deployment prior. They are not interchangeable. Resampling and then
recalibrating can recover approximately the uncorrected probabilities, but the recovery is
imperfect and adds a step that must be validated. If the goal is a calibrated probability, the
simplest path is not to change the prior in the first place.

> **Note:** Every metric in this topic is reported with the prevalence and the threshold. An AUPRC
> of 0.21 at 3 percent prevalence is meaningfully better than the 0.03 baseline; an AUPRC of 0.21
> at 20 percent prevalence is worse than the baseline.

## 5. The mechanics, step by step

The procedure has thirteen steps. Steps 1 through 3 decide whether there is a problem to solve.
Steps 4 through 7 choose and apply a lever. Steps 8 through 13 evaluate and document the result.

### Step 1. Quantify prevalence and event count

**Input:** the cohort and the outcome.

**Output:** prevalence, number of events, number of non-events, events per candidate parameter, and
the prevalence by site, year, and subgroup.

**Rationale:** The ratio alone is not informative. A 1 percent prevalence in a cohort of 500,000 is
5,000 events, which is ample for a moderately complex model. A 20 percent prevalence in a cohort of
300 is 60 events, which is not. The event count determines what can be estimated, and the
stratified prevalence determines whether the model will behave consistently across sites.

The event count should be computed after exclusions and after the label definition is fixed. A
cohort that appears to have 1,200 events may have 800 after excluding prevalent cases and
incomplete follow-up. The difference matters for every downstream calculation, including the
precision of AUPRC and the stability of subgroup estimates.

**Branch point:** if the event count is below about 100, the priority is to increase the sample or
simplify the model, not to resample. Resampling 60 events into 6,000 synthetic rows does not create
information; it creates 6,000 correlated copies of 60 events, and the effective sample size remains
60.

### Step 2. Define the decision and the error costs

**Input:** the prediction task specification.

**Output:** the action, the harm ratio, the threshold rule, and the capacity.

**Rationale:** The decision determines whether a probability or a ranking is needed. A calibrated
probability is required for shared decision-making and for a threshold derived from a harm ratio.
A ranking is sufficient for allocating a fixed capacity. The error costs determine which operating
point is acceptable, and capacity determines whether that point is feasible.

**Branch point:** if the decision requires a probability and the team plans to resample, the plan
must include a recalibration step. If the decision only requires a ranking, resampling is
unnecessary but not necessarily harmful, provided the ranking is preserved.

### Step 3. Decide whether correction is needed

**Input:** the prevalence, the event count, and the model family.

**Output:** a decision, with a rationale.

**Rationale:** For most risk models, no correction is needed. The model can learn from an imbalanced
sample if the events are numerous enough and the loss is appropriate. Correction is indicated when
the model family systematically ignores the minority class (some neural networks with unweighted
loss), when the downstream process explicitly expects a balanced sample, or when compute constraints
make undersampling attractive. It is not indicated merely because the ratio is low.

A useful test is to compare the model's ranking before and after the planned correction. If the
ranking is unchanged, the correction has not added information; it has only changed the scale. In
that case, threshold shifting is the simpler intervention. If the ranking improves, the correction
may be acting as a form of regularisation or as a way to prevent the model from ignoring the
minority class, and it may be worth the recalibration cost.

**Branch point:** if the team cannot state what the correction is expected to change, do not apply
it. "It is standard practice" is not a rationale. A correction should have a named target: a
ranking improvement, a training-stability improvement, or a compute saving.

### Step 4. Choose the lever

**Input:** the decision requirement and the model family.

**Output:** one of no correction, class weights, threshold shift, or resampling, with the
recalibration plan.

**Rationale:** The levers act on different objects. Threshold shifting is the least invasive and is
sufficient when the ranking is good. Class weights are appropriate when the training objective
needs to attend to the minority class. Resampling is appropriate when the training procedure
requires a balanced sample or when compute is constrained. The table below maps goals to levers.

| Goal | Lever | Recalibration needed | Notes |
|---|---|---|---|
| Calibrated probabilities for decision-making | None | No | Default for risk models |
| Fixed operating point on a good ranking | Threshold shift | No | Equivalent to weighting for ranking |
| Minority class ignored during training | Class weights | Yes, prior correction | Common for neural networks |
| Compute-constrained training | Undersampling inside folds | Yes, prior correction | Discards majority information |
| Downstream process expects balanced classes | Resampling inside folds | Yes | Document the effective prior |
| Extreme rarity with few events | None; collect more data | Not applicable | Resampling cannot create events |

**Branch point:** if two levers are plausible, prefer the one that leaves the probability scale
intact. The fewer transformations between the model and the decision, the fewer places for a
calibration error to enter.

### Step 5. If resampling, do it inside the folds

**Input:** the training fold and the resampling method.

**Output:** a resampled training fold and an untouched validation fold.

**Rationale:** Resampling before splitting leaks information. Duplicated minority rows can appear
in both training and validation, and synthetic rows generated from a validation patient can carry
that patient's characteristics into training. The validation fold must reflect the deployment
distribution, including its prevalence, or the performance estimate refers to a population that
will never be seen.

The validation fold should also be left at the true prevalence. A common mistake is to resample
both the training and the validation folds, which makes the validation estimate refer to the
resampled population and inflates precision. The validation fold is a small sample of the
deployment population; changing its prevalence changes the question it answers.

**Branch point:** if the resampling method requires the full dataset (for example, a nearest-
neighbour synthesis that uses all minorities), restrict it to the training fold. A method that
cannot be implemented inside the fold cannot be validated.

### Step 6. If weighting, set the weights from the goal

**Input:** the class prevalence and the loss function.

**Output:** class weights and the effective prior they imply.

**Rationale:** A common choice is `w_pos = N_neg / N_pos` and `w_neg = 1`, which gives the two
classes equal total weight. That is equivalent to a 50/50 prior, not to the deployment prior. Other
weights correspond to other effective priors. The team should record the implied prior, because it
determines the recalibration shift.

**Branch point:** if the weights are chosen to optimise a metric rather than to reflect a prior,
the resulting probabilities have no clean interpretation. Use the weights for ranking and shift the
threshold, or recalibrate explicitly.

### Step 7. Train and record the effective prior

**Input:** the weighted or resampled training data.

**Output:** a fitted model and a record of the prior it was trained under.

**Rationale:** The effective prior is the prevalence implied by the weights or the resampling. It
is needed for the prior correction in Step 8. Without it, the correction cannot be computed. The
record should state the original prevalence, the method, the weights or sampling ratio, and the
resulting prior.

The record should also state the software implementation and the random seed, because resampling
and SMOTE are stochastic. Two runs with different seeds can produce different synthetic rows and
slightly different models. Reproducibility requires the seed, not only the method name.

**Branch point:** if the effective prior is not stated, the model's probabilities cannot be
corrected and should not be used as probabilities. Use them only for ranking.

### Step 8. Recalibrate to the deployment prior

**Input:** the fitted model, the effective prior, and the deployment prevalence.

**Output:** a prior-corrected model or a recalibrated model, with the correction validated on
held-out data.

**Rationale:** For a logistic model trained under a modified prior, the correction is a shift in
the intercept:

```text
shift = logit(deployment prevalence) - logit(training prevalence)
```

The shift is added to the log-odds before converting to a probability. For a non-linear model, a
post-hoc calibrator (Platt scaling or isotonic regression) fitted on held-out data is more reliable
than an analytic shift. In both cases, the correction must be validated on data that were not used
to fit the model or the calibrator.

For a non-linear model, the analytic shift is only approximate. Tree ensembles and neural networks
do not have a single intercept that can be shifted. A post-hoc calibrator fitted on a held-out set
at the true prevalence is the safer approach. The calibrator should be fitted after the model is
frozen and before the threshold is chosen, and it should be evaluated on a separate set if
possible.

**Branch point:** if the deployment prevalence is unknown, the correction cannot be computed.
Report the model's performance across a range of plausible prevalences rather than assuming one.

### Step 9. Choose the threshold on calibrated probabilities

**Input:** the calibrated model and the harm ratio.

**Output:** a threshold and the operating characteristics at that threshold.

**Rationale:** The threshold is meaningful only on the deployment probability scale. A threshold of
0.5 on a model trained at a 50/50 prior corresponds to a much lower risk on the true prior. Choosing
the threshold before recalibration is a common source of over-alerting.

The threshold should be chosen on the deployment prior, after any correction. A threshold selected
on a resampled validation set is on the resampled scale and will not transfer. If the team must
choose a threshold before recalibration, it should choose a quantile (for example, the top 20
percent of scores) rather than a probability, and convert the quantile to a probability after
recalibration.

**Branch point:** if the threshold is constrained by capacity, use the capacity-constrained
threshold and report the sensitivity loss. If the capacity is unknown, report the threshold implied
by the harm ratio and the flagged count it produces.

### Step 10. Evaluate with precision-recall and net benefit

**Input:** the held-out predictions and outcomes.

**Output:** AUPRC with the prevalence baseline, precision and recall at the operating threshold, and
a decision curve.

**Rationale:** AUROC is prevalence-independent and can hide poor precision. AUPRC makes the
rare-event performance visible, and its baseline is the prevalence. Net benefit translates the
confusion matrix into the decision's terms. Reporting all three prevents the common pattern of a
respectable AUROC with an unusable alert burden.

The precision-recall curve should be plotted with the baseline as a horizontal line. A curve that
sits barely above the baseline at the operating recall is not useful, even if its area is several
times the baseline. The area is an average over thresholds; the decision uses one threshold. Report
both the area and the operating point.

**Branch point:** if AUPRC is close to the baseline, the model is not adding value at that
prevalence, regardless of AUROC. If net benefit is negative at the planned threshold, the decision
rule is harmful and must change.

### Step 11. Compute alert burden and capacity

**Input:** the threshold, the score distribution, and the workflow capacity.

**Output:** alerts per period, true positives per period, number needed to evaluate, and the
capacity gap.

**Rationale:** A model that flags 20 percent of admissions to find 70 percent of events creates a
review burden that may be impossible. The number needed to evaluate is the reciprocal of PPV: at a
PPV of 0.10, ten patients are reviewed for every true case. That number belongs in the decision
conversation, not only in the appendix.

Alert burden should be reported per unit that the workflow understands: alerts per 100 admissions,
per bed-day, per clinic session, or per shift. A model that generates 20 alerts a day sounds
manageable until the team learns that the ward has four nurses and each alert requires a 15-minute
review. The arithmetic of staff time is part of the decision analysis.

**Branch point:** if the flagged count exceeds capacity, either raise the threshold or expand
capacity. Report both options and their net benefit.

### Step 12. Check subgroups and event counts

**Input:** the held-out predictions and subgroup identifiers.

**Output:** prevalence, AUPRC, sensitivity, PPV, and event counts by subgroup.

**Rationale:** A subgroup with few events will have unstable precision and may appear to benefit or
suffer disproportionately. The check separates a real disparity from small-sample noise. It also
reveals whether the correction was applied evenly: a resampling method can over-represent one
subgroup's events if the minority class is concentrated there.

Subgroup precision is especially sensitive to event count because it depends on both sensitivity
and the subgroup prevalence. A subgroup with 30 events and a 1 percent prevalence can have a PPV
near 0.05 even when the model's overall PPV is 0.20. That is not necessarily a fairness failure; it
may be arithmetic. The report should distinguish the two by comparing the subgroup's prevalence and
the model's subgroup sensitivity before interpreting the PPV difference. A subgroup with the same
sensitivity and specificity as the overall cohort but a lower prevalence will have a lower PPV by
arithmetic, and that is not evidence of a model failure. It is evidence that the subgroup needs a different
threshold or a different decision rule, which is a separate question with its own arithmetic.

**Branch point:** if a subgroup has fewer than about 20 events, report the count and avoid a
subgroup-specific conclusion.

### Step 13. Document the correction and the prior

**Input:** the full analysis.

**Output:** a methods paragraph and a model card section describing the prevalence, any correction,
the effective prior, the recalibration, the threshold, and the alert burden.

**Rationale:** A reader cannot interpret a reported PPV without knowing whether the model was
resampled and whether the probabilities were corrected. The documentation should be sufficient for
another analyst to reproduce the probability scale.

**Branch point:** if the correction cannot be described in one paragraph, it is too complex for the
deployment. Simplify it.

### Imbalance edge cases

**One positive in a fold.** A fold with a single event has an undefined or unstable AUPRC. Use
stratified folds, merge the fold, or report the pooled metric only.

**Zero positives in a fold.** AUROC and AUPRC are undefined. Some libraries return 0.5 or raise an
error. Report the fold as not evaluable.

**Prevalence shift.** A model corrected to a 3 percent prior and deployed at 1 percent will
over-predict. The prior correction must be recomputed for the deployment prevalence, and the
threshold re-derived.

**Ties in scores.** Resampled models can produce many tied scores, especially with few synthetic
rows. Ties affect precision at a fixed threshold and capacity-based selection. Report the tie rate.

**Empty folds after resampling.** Oversampling can create a fold with no majority examples if the
sampling is done incorrectly. Check the class composition of every fold.

**Perfect separation.** A resampling method that duplicates rare positives can create a feature
combination that separates the classes in training. The resulting coefficients are unstable and the
validation performance is poor.

**Subgroup with no events.** A subgroup with zero events cannot support a subgroup PPV. Report the
count and the overall model's behaviour for that subgroup descriptively.

**A correction applied to the wrong class.** Weighting the negative class instead of the positive
class inverts the intended effect. Check the weight vector against the label encoding before
training; a label encoded as {-1, 1} rather than {0, 1} is a common source of sign errors.

**Capacity below one alert per period.** If the capacity is very small, the threshold must be very
high and the sensitivity will be low. The decision may be to use the model for a different action
or not at all.

## 6. Worked example

### The setting

A hospital wants a model for 30-day severe deterioration after discharge. The cohort is 40,000
discharges, with 1,200 events, a prevalence of 3.0 percent. The intervention is a nurse-led check
call; the team can make 20 calls per day, or about 7,300 per year. The harm ratio implies a
threshold of 0.05, but the team is also constrained by capacity.

A gradient-boosted model is trained on the full cohort with no resampling. On a held-out year, its
AUROC is 0.82 and its AUPRC is 0.21, against a prevalence baseline of 0.03. At a threshold of 0.05,
sensitivity is 0.70 and specificity is 0.80.

### The operating characteristics

The flagged fraction is:

```text
0.70 * 0.03 + 0.20 * 0.97 = 0.021 + 0.194 = 0.215
```

At 40,000 discharges, that is 8,600 alerts per year, against a capacity of 7,300. PPV is:

```text
0.70 * 0.03 / (0.70 * 0.03 + 0.20 * 0.97)
= 0.021 / 0.215
= 0.0977
```

About one in ten flagged patients has the event. The number needed to evaluate is
`1 / 0.0977 = 10.2`. At a threshold of 0.10, sensitivity is 0.55 and specificity is 0.92:

```text
flagged fraction = 0.55 * 0.03 + 0.08 * 0.97 = 0.0165 + 0.0776 = 0.0941
alerts = 3,764 per year
PPV = 0.0165 / 0.0941 = 0.175
```

The higher threshold fits within capacity and doubles the PPV, at the cost of missing 45 percent of
events instead of 30 percent. The team must choose between a larger programme and a higher
threshold; the model cannot resolve that trade-off. What the model can do is make the trade-off
explicit: each additional percentage point of sensitivity costs a known number of alerts and a
known amount of nursing time.

### What resampling would do

Suppose the team oversamples the minority class to a 50/50 ratio and retrains. The apparent PPV on
the resampled validation set is about 0.45, and the model looks dramatically better. But the
validation set now has a 50 percent prevalence, which is not the deployment population. The
calibration intercept on the true prior is about -4.5, consistent with the simulation results of
van den Goorbergh et al. The model's probabilities are on the wrong scale.

If the team applies the prior correction:

```text
shift = logit(0.03) - logit(0.50)
      = -3.476 - 0
      = -3.476
```

the probabilities return to approximately the true prevalence, and the PPV returns to about 0.10.
The AUROC is unchanged at 0.82. The entire apparent improvement was the changed base rate.

### The F1 trap

The team also considers reporting F1, the harmonic mean of precision and recall. At the 0.05
threshold, precision is 0.098 and recall is 0.70:

```text
F1 = 2 * (0.098 * 0.70) / (0.098 + 0.70)
   = 0.137 / 0.798
   = 0.172
```

An F1 of 0.17 looks poor, and the team might reject the model. But F1 treats precision and recall
as equally important, which is a strong utility assumption that the programme has not made. The
intervention's harm ratio implies that a true positive is worth much more than a false positive.
F1 also ignores the true negative count, which is 97 percent of the cohort, and it has no
interpretation in terms of the decision. It is a summary of the confusion matrix, not a measure of
value. Report it only alongside the components and never as the primary metric.

### What SMOTE adds and what it does not

SMOTE creates synthetic minority cases by interpolating between a minority case and its nearest
minority neighbours. The interpolation can produce feature combinations that are clinically
implausible, especially when features are categorical or constrained (a synthetic patient with a
fractional number of prior admissions, or a combination of values that never co-occur). The method
also assumes that the region between two minority cases is also minority, which is not guaranteed
in a clinical feature space. For these reasons, SMOTE is better suited to continuous, smooth feature
spaces than to clinical tabular data with strong constraints.

If SMOTE is used, it should be applied inside the training fold, with a seed, and the synthetic
rows should be checked against clinical plausibility ranges. The model should be recalibrated to
the true prior, and the evaluation should use the untouched validation fold. The method does not
create information, and it does not improve AUROC in the simulation evidence.

### Two diagnostics that would reveal a broken analysis

**Diagnostic 1: the calibration intercept after correction.** A model trained on a 50/50 sample and
evaluated on the true 3 percent cohort has a calibration intercept near -4.5. If the team reports a
mean predicted risk of 0.45 against an observed prevalence of 0.03, the miscalibration is obvious.
A calibration plot with the identity line makes it visible; a table of AUROC and F1 does not.

**Diagnostic 2: precision at the operating threshold versus the baseline.** The AUPRC of 0.21 is
seven times the 0.03 baseline, which is a real signal. But at the threshold of 0.05, precision is
0.098, only about three times the baseline. A report that gives AUPRC without precision at the
threshold hides the operational reality. The diagnostic is a precision-recall curve with the
operating point marked and the baseline drawn.

### The capacity arithmetic in staff time

The programme has 20 calls per day, and each call takes 25 minutes including documentation. That is
500 minutes, or 8.3 hours, of nursing time per day. At 8,600 alerts per year, the programme would
need 3,583 hours per year, or about 1.8 full-time equivalents, against the 1.0 FTE currently
funded. At the higher threshold of 0.10, the requirement is 1,568 hours, or about 0.8 FTE, which
fits. The capacity constraint is therefore not an abstract number; it translates into a staffing
decision.

The team should present both options with their consequences: the 0.05 threshold reaches 70 percent
of events but requires 1.8 FTE and exceeds the funded capacity; the 0.10 threshold reaches 55
percent of events, fits the funded capacity, and doubles the PPV. A decision to fund the additional
0.8 FTE is a programme decision, not a modelling decision, and it should be made by the programme
owner with the net benefit arithmetic in hand.

### Varying one assumption: prevalence falls to 1 percent

Suppose the deployed population has a 1 percent event rate, while sensitivity and specificity at
the 0.05 threshold remain 0.70 and 0.80. PPV becomes:

```text
0.70 * 0.01 / (0.70 * 0.01 + 0.20 * 0.99)
= 0.007 / (0.007 + 0.198)
= 0.034
```

One in 29 flagged patients has the event. Net benefit at 0.05 is:

```text
0.70 * 0.01 - 0.20 * 0.99 * (0.05 / 0.95)
= 0.0070 - 0.1980 * 0.0526
= 0.0070 - 0.0104
= -0.0034
```

Net benefit is negative: the alert programme harms more than it helps at this prevalence and
threshold. The conclusion changes from "deploy with a capacity adjustment" to "do not deploy at
this prevalence without a different threshold or a different intervention". The model did not
change; the base rate did.

### What to report

The primary figure is the precision-recall curve with the prevalence baseline and the operating
point marked. The secondary figure is the calibration plot on the deployment prior, with the
intercept and slope annotated. The decision curve at the deployment prevalence belongs in the
appendix, with the capacity line. The model card should state the prevalence, the resampling or
weighting method (if any), the recalibration, the threshold, and the alert burden. If the model was
not corrected, the card should say so explicitly, because the absence of a correction is also a
design decision that a reader needs to know.

## 7. Code, query, or calculation

The script fits three logistic models on the same synthetic cohort: unweighted, class-weighted, and
oversampled. It shows that AUROC is essentially unchanged while the mean predicted risk moves far
from the observed prevalence. The prior correction at the end returns the oversampled model's
probabilities toward the deployment scale.

```python
"""Class weighting, oversampling, and prior correction. Standard library only."""
import math
import random


def fit(X, y, w, l2=0.01, iterations=400, learning_rate=0.3):
    n, d = len(X), len(X[0])
    beta, b = [0.0] * d, 0.0
    weight_sum = sum(w)
    for _ in range(iterations):
        g, gb = [0.0] * d, 0.0
        for xi, yi, wi in zip(X, y, w):
            z = b + sum(bj * xj for bj, xj in zip(beta, xi))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            error = (p - yi) * wi
            for j in range(d):
                g[j] += error * xi[j]
            gb += error
        for j in range(d):
            beta[j] -= learning_rate * (g[j] / weight_sum + l2 * beta[j])
        b -= learning_rate * gb / weight_sum
    return beta, b


def predict(X, beta, b):
    return [1 / (1 + math.exp(-max(-30, min(30, b + sum(bj * xj
            for bj, xj in zip(beta, x)))))) for x in X]


def auroc(y, p):
    pairs = sorted(zip(p, y))
    positives, negatives = sum(y), len(y) - sum(y)
    wins, seen_negatives = 0.0, 0
    for _, label in pairs:
        if label == 1:
            wins += seen_negatives
        else:
            seen_negatives += 1
    return wins / (positives * negatives)


rng = random.Random(9)
X, y = [], []
for _ in range(5000):
    x = [rng.gauss(0, 1) for _ in range(4)]
    z = -4.55 + 1.2 * x[0] - 0.8 * x[1] + 0.5 * x[2]
    X.append(x)
    y.append(1 if rng.random() < 1 / (1 + math.exp(-z)) else 0)

print(f"prevalence={sum(y) / len(y):.3f}")
for name, weights in (("unweighted", [1.0] * len(y)),
                      ("weighted", [19.0 if v else 1.0 for v in y])):
    beta, b = fit(X, y, weights)
    p = predict(X, beta, b)
    print(f"{name}: AUROC={auroc(y, p):.3f} mean_pred={sum(p) / len(p):.3f}")

positives = [i for i, v in enumerate(y) if v]
negatives = [i for i, v in enumerate(y) if not v]
keep = negatives + positives * 19
beta, b = fit([X[i] for i in keep], [y[i] for i in keep], [1.0] * len(keep))
p = predict([X[i] for i in keep], beta, b)
print(f"oversampled: AUROC={auroc([y[i] for i in keep], p):.3f} "
      f"mean_pred={sum(p) / len(p):.3f}")

shift = math.log(0.03 / 0.97) - math.log(0.5 / 0.5)
corrected = [1 / (1 + math.exp(-(math.log(pi / (1 - pi)) + shift))) for pi in p]
print(f"prior-corrected mean_pred={sum(corrected) / len(corrected):.3f} shift={shift:.3f}")
```

Expected output:

```text
prevalence=0.028
unweighted: AUROC=0.842 mean_pred=0.031
weighted: AUROC=0.843 mean_pred=0.249
oversampled: AUROC=0.843 mean_pred=0.354
prior-corrected mean_pred=0.033 shift=-3.476
```

Reading the output:

- The AUROC is 0.842 to 0.843 in all three models. The ranking is essentially unchanged. The
  correction did not make the model better at separating events from non-events.
- The mean predicted risk rises from 0.031 in the unweighted model to 0.249 with weights and 0.354
  with oversampling. The observed prevalence is 0.028. The weighted and oversampled models are
  predicting risks for a population in which roughly a quarter to a third of patients have the
  event, which is not the deployment population.
- The prior-corrected mean is 0.033, close to the observed 0.028. The residual difference comes
  from the fact that the oversampled model's intercept is not exactly calibrated to a 0.5 prior and
  from the finite sample. The correction improves the average but does not guarantee per-patient
  calibration; a calibration slope check is still needed.
- The shift of -3.476 is `logit(0.03) - logit(0.5)`. If the deployment prevalence were 1 percent,
  the shift would be `logit(0.01) - logit(0.5) = -4.595`, a larger correction.

The reusable function is `fit` with a weight vector, plus the prior-correction arithmetic. The
weight vector can implement class weights, and the prior correction can be applied to any model
with a probability output, provided the effective prior is known.

## 8. Data, standards, and regulatory context

Class imbalance is addressed explicitly in reporting guidance and implicitly in regulatory
expectations about representative data. The table distinguishes binding requirements from
conventions.

| Instrument | Status | What it says about imbalance | What it does not do |
|---|---|---|---|
| TRIPOD+AI item on class imbalance | Reporting convention | Requires the approach to imbalance to be reported, including any resampling or weighting | Does not prescribe a method |
| PROBAST+AI analysis domain | Appraisal tool | Assesses whether the analysis handled imbalance appropriately | Does not validate the model |
| FDA GMLP principles | Guidance, not binding | Calls for data representative of the intended population, including rare outcomes | Does not specify a correction |
| FDA PCCP guidance | Guidance, not binding | Requires planned changes to training data or loss to be described | Does not cover initial imbalance handling |
| CONSORT-AI | Reporting convention | Requires trial reports to describe the AI intervention and its evaluation | Does not address resampling |
| EU AI Act | Binding in the EU, phased | Requires data governance and testing for high-risk systems | Does not specify a method |
| ISO 14971 | Voluntary standard | Risk management across the lifecycle | Does not address statistics |
| NIST AI RMF | Voluntary framework | Measure and manage risks, including performance in subgroups | Not sector-specific |

Three points matter for practice.

**TRIPOD+AI made imbalance a reporting item.** The 2024 statement added an item on class imbalance,
which means a report that resamples without saying so is incomplete. The item does not say
resampling is required or forbidden; it requires transparency. A methods section that says "the
dataset was imbalanced, so SMOTE was applied" without stating the ratio, the fit scope, and the
recalibration is not sufficient.

**PROBAST+AI treats imbalance as an analysis issue.** The tool asks whether the analysis accounted
for the outcome distribution and whether the performance evaluation was appropriate. A study that
reports accuracy on a 3 percent prevalence cohort is at high risk of bias because accuracy is
uninformative. A study that resamples and reports only the resampled performance is also at high
risk.

**Regulators care about the deployment population.** The FDA's GMLP principles and the EU AI Act's
data governance requirements both emphasise that the training and test data should represent the
population in which the device will be used. A model trained on a 50/50 resampled cohort and
validated on a 50/50 cohort does not represent a 3 percent deployment population. The validation
must include the true prevalence, even if the training set is resampled.

### Why the metric must be pre-specified

The choice between AUROC, AUPRC, F1, and net benefit is a choice about what the model is for. If the
metric is chosen after seeing results, the team will choose the one that makes the model look best,
and the choice will be difficult to challenge. Pre-specifying the metric in the protocol, with the
prevalence and threshold, prevents that. It also forces the team to decide, before modelling,
whether the decision needs a probability, a ranking, or a classification.

### Validation data and the deployment population

The regulatory expectation is not only that the model be tested, but that the test data represent
the intended population. For a rare-event model, that means the validation cohort must include the
true prevalence, the full spectrum of severity, and the sites where the model will be used. A
validation cohort assembled by resampling the positives does not represent the deployment
population, even if it is large. A validation cohort from a single tertiary centre may not represent
a community hospital. The class-imbalance choice is therefore a data-governance choice as well as a
statistical one.

### Reporting the correction in a model card

A model card should answer five questions about imbalance. What was the prevalence in the
development cohort and in the validation cohort? Was any correction applied, and with what
parameters? What prior does the model's probability output refer to? How was the model
recalibrated, and on what data? What is the alert burden at the intended threshold? If any of these
questions cannot be answered, the model's probabilities cannot be interpreted by a reader, and the
card is incomplete.

The questions are also a useful internal checklist. A team that cannot answer them before
deployment is unlikely to be able to answer them during an incident, when the score distribution
shifts and someone asks whether the model is still calibrated.

> **Note:** No regulation requires a specific imbalance correction. The requirement is that the
> correction, if any, be described and that the performance be evaluated on a population
> representative of deployment.

## 9. Industry practice and case evidence

**The harm of class imbalance corrections.** Van den Goorbergh et al. (2022) compared no correction
with random undersampling, random oversampling, and SMOTE across 24 simulation scenarios and an
ovarian cancer case study. Calibration intercepts fell to about -4.5 at 1 percent event rates,
-2.1 at 10 percent, and -0.7 at 30 percent, against roughly -0.05 to 0.03 without correction.
AUROC was never improved, random undersampling often reduced it, and decision curves showed reduced
net benefit. The authors conclude that outcome imbalance is not a problem in itself and that
correction may worsen performance.

**Precision-recall versus ROC.** Saito and Rehmsmeier (2015) show that the ROC curve is unchanged
when the class ratio changes, while the precision-recall curve changes and its baseline equals the
prevalence. In their miRNA example, a classifier with an ROC AUC of 0.886 had a precision-recall
AUC of 0.054, close to the baseline, exposing near-random precision that the ROC curve concealed.
The lesson is that AUROC alone can make a rare-event model look useful when it is not.

**Alert burden in sepsis.** The external validation of the Epic Sepsis Model reported that a score
threshold of 6 flagged 18 percent of all hospitalisations, with a PPV of 12 percent and sensitivity
of 33 percent. At a 6.6 percent prevalence, the alert burden was incompatible with the review
capacity of the wards. The case is not about resampling, but it shows the same underlying issue: a
model's usefulness at low prevalence depends on precision and capacity, not on discrimination
alone.

**SMOTE in practice.** SMOTE was introduced by Chawla et al. (2002) and has become a default in many
toolkits. The original paper evaluates synthetic minority over-sampling on several datasets and
reports improved classifier performance on some metrics. Later work, including the simulation study
above, shows that the improvement often comes from the changed decision threshold rather than from
better ranking, and that calibration suffers. The method is not wrong; it is misapplied when the
goal is a calibrated risk.

**Rare events in logistic regression.** King and Zeng (2001) showed that logistic regression in rare
events data can have biased coefficients and underestimated probabilities, and proposed a
correction. The bias is small when the event count is large, even if the prevalence is low. The
practical implication is that the event count, not the ratio, is the binding constraint.

**What the simulation evidence means for a team.** The van den Goorbergh simulations varied sample
size (2,500 or 5,000), the number of predictors (3 to 24), and the event fraction (0.30, 0.10,
0.01). The calibration harm was largest at the lowest prevalence and with the most predictors,
which is exactly the setting where clinical risk models are often built. The practical reading is
that the harm is not a corner case; it is the typical case for rare-event clinical models.

**A note on the ovarian cancer case study.** The case study in van den Goorbergh et al. used 3,369
patients with a 20 percent prevalence and compared the corrections on a real clinical dataset. The
decision curves showed negative net benefit at thresholds of 0.3 or higher, which is the range a
clinician might use for a high-harm intervention. The case is a useful counterexample to the claim
that resampling is harmless if the AUROC is unchanged: the harm appears in the decision, not in the
ranking.

**Industry defaults.** Many automated machine-learning platforms apply class weighting by default
when the minority class is below a threshold. The default is convenient but rarely documented in
the model card. A reviewer who sees a reported PPV should ask whether the model was weighted and
whether the probabilities were corrected. The question is not optional for a clinical model.

**The cost of a false positive in a low-prevalence programme.** At a 3 percent prevalence and a PPV
of 0.10, a programme that reviews every alert spends nine reviews on patients who will not have the
event for every one who will. If a review costs USD 40 in staff time, the cost per true case found
is USD 400, before any intervention cost. That arithmetic, not the AUROC, determines whether the
programme is affordable. The model's contribution is to raise the PPV above the prevalence; the
programme's contribution is to make the review cheap enough or the intervention effective enough.

**A note on metric choice.** The choice between AUROC and AUPRC is not a matter of taste. If the
decision is a ranking with fixed capacity, AUROC is a reasonable summary of the ranking. If the
decision is a threshold with a precision requirement, AUPRC and precision at the threshold are the
relevant metrics. If the decision has explicit harms and benefits, net benefit is the relevant
metric. Reporting all three, with the prevalence, is the safest default.

> **Note:** The cases are documented. The inference that resampling is overused in clinical
> prediction is supported by the simulation evidence, but the prevalence of the practice in
> industry is not quantified here.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Resampling before splitting | Duplicated rows in train and test | Leakage of minority cases | Inflated performance | Resample inside folds |
| Resampling the validation fold | Validation prevalence differs from deployment | Changed base rate | Misleading precision | Leave validation at true prevalence |
| No recalibration after weighting | Mean predicted risk far above prevalence | Changed effective prior | Wrong probabilities | Prior correction or post-hoc calibration |
| Reporting accuracy | Accuracy above 95 percent at 3 percent prevalence | Majority-class prediction | Useless model looks good | Report precision, recall, AUPRC |
| AUROC only | Good AUROC, unknown precision | Prevalence-independent metric | Hidden alert burden | Add AUPRC and PPV at threshold |
| F1 as primary metric | F1 reported without components | Implicit equal weighting | Wrong utility assumption | Report components and net benefit |
| SMOTE on categorical features | Implausible synthetic rows | Interpolation of categories | Invalid cases | Use appropriate method or avoid |
| Undersampling without correction | Small effective sample | Information loss | Wider intervals, lower AUROC | Weight instead |
| Threshold chosen on resampled scale | Threshold not transferable | Probability scale changed | Wrong operating point | Choose after recalibration |
| Ignoring capacity | Alert count exceeds staff | Operational mismatch | Alert fatigue | Compute alerts per period |
| Subgroup event counts ignored | PPV differences read as bias | Small-sample noise | Misattributed disparity | Report counts and prevalence |
| Prior shift after deployment | Prevalence changes | Base rate changed | PPV falls | Monitor and recalibrate |
| Duplicate synthetic rows | Same synthetic case repeated | Oversampling without diversity | Overfitting | Check synthesis method |
| Correction undocumented | Methods section omits it | Irreproducible | Cannot interpret PPV | Document method and prior |
| Weight applied to the wrong class | Weights inverted | Label encoding mismatch | Opposite effect | Check weight vector against labels |
| Synthetic rows clinically implausible | Values outside ranges | Interpolation of constrained features | Invalid cases | Range check and clinical review |
| Precision reported without baseline | AUPRC looks impressive | Baseline omitted | Overstated value | Draw the prevalence baseline |
| Sampling ratio mistaken for prevalence | Methods state 50/50 as the "prevalence" | Training composition confused with population | Wrong PPV interpretation | Report both separately |
| Correction applied to a subgroup only | Subgroup prevalence differs from overall | Inconsistent pipeline | Unfair or unstable performance | Apply consistently or justify |
| Threshold tuned to maximise F1 | F1-maximising threshold treated as clinical | Implicit equal weighting | Wrong operating point | Derive from harms and capacity |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The event count is adequate | Precision of estimates depends on events | Fewer than about 100 events | Report intervals |
| The deployment prevalence is known | Prior correction and PPV depend on it | Prevalence shift | Monitor prevalence |
| The correction preserves ranking | Threshold shifting assumes it | Resampling changes the ranking | Compare AUROC before and after |
| The validation fold is untouched | Performance refers to deployment | Resampling the validation fold | Audit fold composition |
| The effective prior is recorded | Correction requires it | Weights or ratio undocumented | Document |
| The recalibration is validated | Correction may be imperfect | Calibrator fitted in-sample | Held-out calibration check |
| The threshold is on the deployment scale | Decision depends on it | Threshold chosen pre-correction | Re-derive after correction |
| The model is used for the intended action | Utility depends on the action | Off-label use | Confirm workflow |
| Subgroup event counts are adequate | Subgroup precision is stable | Fewer than 20 events | Report counts |
| The correction is reproducible | Audit requires it | No seed or method detail | Re-run |
| The model output is on the deployment scale | Threshold and PPV depend on it | Uncorrected resampling | Calibration check |
| The evaluation cohort has the deployment prevalence | PPV is comparable | Resampled validation | Audit validation composition |
| The alert burden is computed on the deployment cohort | Workload estimate is valid | Alerts computed on resampled data | Recompute on true prevalence |
| The metric matches the decision | Metric is informative | AUROC for a precision requirement | Map metric to decision |
| The threshold is on the deployment scale | Operating point transfers | Threshold from resampled data | Re-derive after calibration |
| The correction is applied consistently | Fair comparison | Different pipelines across groups | Audit per group |

The deepest limit is that no correction can create events. A cohort with 60 events supports a
limited model, and resampling changes the weights, not the information. The report should state the
event count and the effective sample size, not only the resampled row count. A reader who sees
50,000 rows and 30 events should be able to tell that the model is estimated from 30 events,
regardless of how many synthetic rows were created.

## 12. Exercises

The exercises are ordered from arithmetic to design, and each one should be answered with the
prevalence written next to the metric. Every exercise asks for the prevalence to be
stated alongside the metric, because a metric without prevalence is not interpretable. Exercises 1 to 3 can be done with a calculator;
4 to 7 need a dataset or the predictions from a model you have built; 8 produces a document you can
reuse in the module assignment.

**Exercise 1. Confusion matrix arithmetic.** At 2 percent prevalence, a model has sensitivity 0.65
and specificity 0.90. Compute PPV, NPV, the flagged fraction, and the number needed to evaluate.
Deliverable: the arithmetic. Acceptance criteria: all four values correct; interpretation states
the alert burden. Time: 30 minutes.

**Exercise 2. AUPRC baseline.** Explain why the AUPRC of a random classifier equals the prevalence,
and compute the baseline for prevalences of 1, 5, and 20 percent. Deliverable: a paragraph and three
values. Acceptance criteria: explanation correct; values correct. Time: 20 minutes.

**Exercise 3. Prior correction.** A model is trained on a 50/50 sample and deployed at 4 percent
prevalence. Compute the logit shift and the corrected probability for a patient with a raw
prediction of 0.70. Deliverable: the arithmetic. Acceptance criteria: shift and corrected
probability correct to three decimals. Time: 30 minutes.

**Exercise 4. Compare corrections.** On a dataset with 3 percent prevalence, compare no correction,
class weights, and oversampling on AUROC, mean predicted risk, and calibration slope. Deliverable:
a results table and a 300-word interpretation. Acceptance criteria: all three metrics reported;
recommendation justified. Time: 75 minutes.

**Exercise 5. Threshold and capacity.** A cohort has 50,000 admissions, 2 percent prevalence, and a
programme capacity of 12 patients per day. At a proposed threshold, the model flags 15 percent of
admissions. Compute the annual alerts, compare with capacity, and propose two options. Deliverable:
a memo. Acceptance criteria: arithmetic correct; both options described. Time: 45 minutes.

**Exercise 6. Precision-recall critique.** Find a rare-event model paper and assess whether it
reports AUPRC with the baseline, precision at the threshold, and calibration. Deliverable: a
500-word critique. Acceptance criteria: at least three specific gaps; claims cited. Time: 60
minutes.

**Exercise 7. Subgroup precision.** Given subgroup prevalences of 4, 2, and 1 percent and a model
with sensitivity 0.60 and specificity 0.90, compute PPV for each subgroup. Deliverable: a table and
an interpretation. Acceptance criteria: values correct; interpretation distinguishes arithmetic from
bias. Time: 30 minutes.

**Exercise 8. Documentation.** Write the class-imbalance section of a model card for a model you
have built or specified. Include the development prevalence, validation prevalence, correction
method and parameters, effective prior, recalibration, threshold derivation, PPV and sensitivity,
alert burden, and the prevalence range validated. Deliverable: one page. Acceptance criteria: all
eight elements present; numbers traceable to the analysis. Time: 45 minutes.

## 13. Self-check answers

**Answer to Exercise 1.** At 2 percent prevalence, per 10,000 patients: 200 events, 9,800
non-events. True positives = 0.65 * 200 = 130. False negatives = 70. False positives = 0.10 * 9,800
= 980. True negatives = 8,820. PPV = 130 / (130 + 980) = 130 / 1,110 = 0.117. NPV = 8,820 /
(8,820 + 70) = 0.992. Flagged fraction = 1,110 / 10,000 = 0.111. Number needed to evaluate = 1 /
0.117 = 8.5. The model flags 11 percent of patients to find 65 percent of events, and about one in
nine flagged patients has the event.

**Answer to Exercise 3.** The shift is `logit(0.04) - logit(0.50) = -3.178 - 0 = -3.178`. For a
raw prediction of 0.70, the logit is 0.847. The corrected logit is `0.847 - 3.178 = -2.331`. The
corrected probability is `1 / (1 + exp(2.331)) = 0.0886`. The raw 0.70 becomes about 0.089 on the
deployment scale.

**Answer to Exercise 7.** At 4 percent prevalence, PPV = `0.60 * 0.04 / (0.60 * 0.04 + 0.10 *
0.96) = 0.024 / 0.120 = 0.200`. At 2 percent: `0.012 / (0.012 + 0.098) = 0.109`. At 1 percent:
`0.006 / (0.006 + 0.099) = 0.057`. The PPV falls as prevalence falls even though sensitivity and
specificity are constant. The difference across subgroups may be arithmetic rather than a model
failure, and the report should say so before attributing it to bias.

**Answer to Exercise 2.** A random classifier assigns the same expected precision at every recall,
and that precision is the proportion of positives in the evaluation set, `P / (P + N)`. At 1
percent, the baseline is 0.01; at 5 percent, 0.05; at 20 percent, 0.20. An AUPRC equal to the
baseline means the model is no better than random. An AUPRC of 0.21 at 3 percent prevalence is seven
times the baseline; an AUPRC of 0.21 at 20 percent prevalence is barely above it.

**Answer to Exercise 5.** At 2 percent prevalence and 15 percent flagged, the annual alerts are
`0.15 * 50,000 = 7,500`, or about 20.5 per day. Capacity is 12 per day, so the threshold flags
about 8.5 more patients per day than the programme can review. Option one is to raise the threshold
until the flagged count is 12 per day, which reduces sensitivity. Option two is to fund 8.5
additional reviews per day, about 0.7 FTE if each review takes 25 minutes. The recommendation should
state the sensitivity loss of option one and the cost of option two.

**Answer to Exercise 4 (what a strong comparison shows).** A strong comparison reports AUROC, mean
predicted risk, and calibration slope for each approach on the same untouched validation cohort. It
shows that AUROC is approximately unchanged, that mean predicted risk rises sharply with weighting
and oversampling, and that the calibration slope falls below 1. It then applies a prior correction
and shows that the mean returns toward the observed prevalence, while noting that per-patient
calibration still needs a calibration plot. The recommendation is usually no correction for a
probability model, or weights plus recalibration if the training procedure requires them.

**Answer to Exercise 6 (what a strong critique checks).** A strong critique checks whether the
paper reports the prevalence, the event count, and the AUPRC baseline; whether it reports precision
at the operating threshold; whether it states any resampling or weighting and the recalibration;
whether the validation cohort has the deployment prevalence; and whether the decision analysis is
present. It distinguishes a genuine ranking improvement from a changed base rate. It ends with two
or three specific corrections, not a general statement.

**Answer to Exercise 8 (what a complete model-card section contains).** A complete section states the
development prevalence and event count, the validation prevalence, any correction and its
parameters, the effective prior of the model's output, the recalibration method and data, the
threshold and its derivation, the PPV and sensitivity at that threshold, and the alert burden per
unit of workflow. It also states the prevalence range over which the model has been validated. A
section that says only "the data were imbalanced and SMOTE was applied" is not complete.

**Answer to Exercise 6 (what a weak critique looks like).** A weak critique says the paper "should
have used AUPRC" without checking whether the prevalence was high enough for AUROC to be
informative, or says the model "is biased" without separating arithmetic from model behaviour. It
does not check the validation prevalence or the recalibration, and it does not distinguish a
reporting gap from an analysis error. The distinction matters because the remedies differ.

**Rubric for Exercises 4, 5, 6, and 8.** A strong submission (1) reports prevalence and event
count, (2) states whether any correction was applied and why, (3) recalibrates to the deployment
prior, (4) reports AUPRC with the baseline, (5) computes alert burden and capacity, and (6)
documents the method. A competent submission meets four of the six. A weak submission reports
accuracy or F1 as the primary metric and resamples before splitting.

## 14. Interview questions

**1. Why is accuracy a bad metric for rare events?** A strong answer explains that predicting the
majority class gives high accuracy and zero sensitivity, and gives a numeric example. A weak answer
says accuracy is "misleading" without a mechanism.

**2. What is the harm of oversampling for a risk model?** A strong answer explains the changed
prior, the calibration intercept, and the need for recalibration, citing the simulation evidence. A
weak answer says oversampling improves the minority class.

**3. When would you use class weights?** A strong answer says when the training objective ignores
the minority class, and notes that weights change the effective prior and require recalibration. A
weak answer says always.

**4. What is the difference between resampling and threshold shifting?** A strong answer explains
that both can achieve the same sensitivity-specificity pair, but resampling changes the probability
scale and threshold shifting does not. A weak answer treats them as equivalent.

**5. How do you choose a threshold at low prevalence?** A strong answer uses the harm ratio,
capacity, and net benefit on calibrated probabilities. A weak answer uses 0.5 or the F1-maximising
threshold.

**6. What is AUPRC and why is its baseline the prevalence?** A strong answer defines precision and
recall and explains that a random classifier's precision equals the prevalence at every recall. A
weak answer says the baseline is 0.5.

**7. How do you recalibrate after resampling?** A strong answer gives the logit shift for a logistic
model and describes post-hoc calibration for non-linear models, validated on held-out data. A weak
answer says "recalibrate" without a method.

**8. What is the number needed to evaluate?** A strong answer defines it as the reciprocal of PPV
and relates it to workload. A weak answer confuses it with number needed to treat.

**9. How does prevalence shift affect PPV?** A strong answer gives the arithmetic and shows that PPV
falls with prevalence at fixed sensitivity and specificity. A weak answer says PPV is a property of
the model.

**10. What does the precision baseline tell you?** A strong answer says it is the prevalence and that AUPRC must be compared with it. A weak answer assumes the baseline is 0.5.

**11. How do you handle imbalance in a subgroup with very few events?** A strong answer says do not resample; report the event count, pool the subgroup if possible, and avoid a subgroup-specific claim. A weak answer applies SMOTE to the subgroup.

**12. How does the choice of metric change the optimal threshold?** A strong answer explains that F1, Youden's J, and net benefit can select different thresholds because they encode different utility assumptions. A weak answer says the optimal threshold is always the one that maximises F1.

**13. What is the difference between a sampling ratio and a prevalence?** A strong answer distinguishes the training composition from the population base rate and explains why the two must be recorded separately. A weak answer uses them interchangeably.

**14. How would you explain the base-rate fallacy to a clinician?** A strong answer uses the PPV arithmetic at two prevalences and shows why the same test means different things in different populations. A weak answer says the prevalence matters without an example.

**15. What would you monitor after deployment of a rare-event model?** A strong answer monitors prevalence, PPV, alert volume, calibration, and subgroup performance. A weak answer monitors AUROC, which cannot be computed without labels and is insensitive to prevalence.

**16. Why might a model with AUROC 0.82 be useless?** A strong answer points to low precision, high
alert burden, poor calibration, or a threshold mismatch. A weak answer says AUROC 0.82 is always
good.

## 15. Key terms

- **Accuracy** — proportion of correct predictions.
- **Alert burden** — alerts per unit of clinical work.
- **AUPRC** — area under the precision-recall curve.
- **AUROC** — area under the receiver operating characteristic curve.
- **Base rate** — prevalence of the outcome.
- **Capacity** — number of actions deliverable per period.
- **Class weight** — loss weight applied to a class.
- **Data leakage** — information from outside the training fold entering the model.
- **Decision curve** — net benefit across thresholds.
- **Effective prior** — prevalence implied by weights or resampling.
- **Effective sample size** — sample size adjusted for events or clustering.
- **Event per variable** — events per candidate parameter.
- **Extreme imbalance** — prevalence below about 1 percent.
- **F1 score** — harmonic mean of precision and recall.
- **False positive rate** — proportion of negatives flagged.
- **Focal loss** — loss that down-weights easy examples.
- **Majority class** — the more frequent outcome.
- **Minority class** — the less frequent outcome.
- **Net benefit** — prevalence-weighted true positives minus threshold-weighted false positives.
- **Number needed to evaluate** — patients reviewed per true case found.
- **Oversampling** — increasing the minority class.
- **PPV** — positive predictive value.
- **Precision** — proportion of flagged patients who are positive.
- **Precision baseline** — prevalence; AUPRC of a random classifier.
- **Prior correction** — adjusting probabilities to a different prior.
- **PR curve** — precision versus recall.
- **Prevalence** — proportion of positives in the population.
- **Recall** — proportion of positives correctly flagged.
- **Resampling** — changing the training class ratio.
- **ROC curve** — sensitivity versus false positive rate.
- **Sensitivity** — proportion of positives correctly flagged.
- **SMOTE** — synthetic minority over-sampling by interpolation.
- **Specificity** — proportion of negatives correctly not flagged.
- **Stratified sampling** — preserving class ratios in folds.
- **Threshold shift** — moving the cut-off on a fixed model.
- **Undersampling** — reducing the majority class.

## 16. Sources

All URLs accessed 25 October 2026.

1. van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. The harm of class imbalance
   corrections for risk prediction models: illustration and simulation using logistic regression.
   *Journal of the American Medical Informatics Association* 2022;29(9):1525-1534.
   https://doi.org/10.1093/jamia/ocac093
2. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when
   evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 2015;10(3):e0118432.
   https://doi.org/10.1371/journal.pone.0118432
3. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: synthetic minority over-sampling technique.
   *Journal of Artificial Intelligence Research* 2002;16:321-357.
   https://doi.org/10.1613/jair.953
4. King G, Zeng L. Logistic regression in rare events data. *Political Analysis*
   2001;9(2):137-163. https://doi.org/10.1093/pan/9.2.137
5. He H, Garcia EA. Learning from imbalanced data. *IEEE Transactions on Knowledge and Data
   Engineering* 2009;21(9):1263-1284. https://doi.org/10.1109/TKDE.2008.239
6. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
7. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
   https://doi.org/10.1136/bmj-2024-082505
8. Vickers AJ, Elkin EB. Decision curve analysis. *Medical Decision Making* 2006;26(6):565-574.
   https://doi.org/10.1177/0272989X06295361
9. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive
   analytics. *BMC Medicine* 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
10. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
    sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
    https://doi.org/10.1001/jamainternmed.2021.2626
11. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
    Guiding Principles. 2021.
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
12. Liu X, Cruz Rivera S, Moher D, et al. CONSORT-AI extension. *Nature Medicine*
    2020;26:1364-1374. https://doi.org/10.1038/s41591-020-1034-x
13. European Union. Regulation (EU) 2024/1689 (Artificial Intelligence Act). 2024.
    https://eur-lex.europa.eu/eli/reg/2024/1689/oj
14. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
15. scikit-learn developers. Preprocessing and imbalanced data.
    https://scikit-learn.org/stable/modules/preprocessing.html
16. imbalanced-learn developers. User guide.
    https://imbalanced-learn.org/stable/user_guide.html
17. Davis J, Goadrich M. The relationship between precision-recall and ROC curves. *Proceedings of
    the 23rd International Conference on Machine Learning*, 2006.
    https://doi.org/10.1145/1143844.1143874
18. Steyerberg EW. *Clinical Prediction Models*. 2nd edition. Springer, 2019.
    https://doi.org/10.1007/978-3-030-16399-0

A final test for any reported rare-event metric: ask what prevalence it refers to. If the answer is
not in the sentence, the metric is incomplete. The same test applies to a PPV quoted in a
presentation, an AUPRC in a paper, and a threshold in a deployment plan. The prevalence is not a
footnote; it is half of the number.

A final check before correcting imbalance: state the deployment prevalence, the model output, and the
decision rule. If the correction does not change any of the three, it is not needed.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_model-families.md`](05_model-families.md)
- Next topic: [`07_metrics.md`](07_metrics.md)
- Calibration: [`08_calibration.md`](08_calibration.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../course/06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
