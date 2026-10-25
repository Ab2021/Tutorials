# Thresholds and Decision Curves

> **Module 06 — Machine Learning Foundations**
> Topic 09 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 07 and 08
> Version 2026.3

## 1. Why this topic exists

A model outputs a probability. A programme acts at a threshold. The threshold is the point where
the model stops being a description and becomes a decision, and it is the point most often chosen
by convention rather than by analysis. The default of 0.5 is a decision to treat only patients whose
risk exceeds one half, which is almost never the right rule for a clinical programme. A readmission
programme with a cheap intervention and a costly outcome should act at a much lower risk; a
high-harm intervention with limited capacity should act at a much higher one.

The threshold has two independent determinants. The first is the relative harm of the two errors.
If a false positive costs little and a false negative costs a great deal, the threshold should be
low. The second is capacity. If the programme can only review 12 patients a day, the threshold must
be high enough that the flagged count fits, regardless of what the harm ratio alone would imply.
The threshold that maximises net benefit and the threshold a clinic can staff are often different,
and the gap between them is a management decision, not a modelling one.

Decision curve analysis (DCA), introduced by Vickers and Elkin (2006), makes the trade-off visible.
It plots net benefit against the threshold probability, with the strategies "treat all" and "treat
none" as references. Net benefit is:

```text
NB(t) = sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * t / (1 - t)
```

The term `t / (1 - t)` is the odds of the threshold, which is the harm ratio implied by the
threshold. A threshold of 0.20 means the decision-maker is willing to accept four false positives
for every true positive, because the odds are 0.20/0.80 = 1:4. A threshold of 0.05 means the
decision-maker is willing to accept nineteen false positives per true positive. The threshold is
therefore a statement about values, not a property of the model.

The most important property of DCA is that it can reverse an AUROC ranking. A model with a higher
AUROC can have lower net benefit in the threshold range where the programme operates, because its
errors are concentrated where the decision is made. A model with a lower AUROC can be the better
decision aid at the staffable threshold. A report that chooses the model by AUROC and then chooses
the threshold by convenience can miss both the better model and the better decision.

DCA also exposes a common error: a model can have an AUROC above 0.5 and still be harmful at the
threshold the programme uses. If the model's net benefit is below the net benefit of treat-all or
below zero, using it is worse than the default strategy. That result is not visible in the AUROC,
and it is the reason the decision curve belongs in the primary evidence.

This topic teaches the threshold as a decision, the net benefit as its metric, and the decision
curve as the comparison. The same tools apply to a rule-out decision, a triage decision, and a
resource-allocation decision; the action changes, but the arithmetic of harms and capacity does
not. What changes is which error is costly and which capacity binds, and those are the first
questions to ask. You will learn to derive the harm-ratio threshold, compute net benefit,
plot and interpret the curve, reconcile the unconstrained optimum with capacity, and report the
operating point with its uncertainty.

The threshold also determines the model's apparent performance in deployment. A model with a
modest AUROC can have a high PPV at a high threshold and a low PPV at a low threshold; the
programme experiences the PPV at the threshold it uses, not the AUROC. That is why the operating
point must be part of the performance claim.

The threshold is also the point where the model's output becomes an action, and actions have
consequences that the model does not see. A false positive may lead to an unnecessary admission, a
medication side effect, or a patient's anxiety. A false negative may lead to a missed deterioration.
The harm ratio should include those consequences, not only the direct cost of the intervention.

The threshold is also where equity enters the decision. A single threshold applied to groups with
different prevalence produces different PPVs and different numbers needed to evaluate, even when the
model's sensitivity and specificity are identical. A programme that does not examine subgroup net
benefit can adopt a threshold that is efficient overall and inequitable in practice. The decision
curve is a population-average summary; the subgroup analysis is the check on who bears the errors.

A practical example makes the stakes clear. A 30-day readmission programme with a nurse call costs
about USD 600 per enrolled patient and prevents a readmission worth about USD 16,200 with a 25
percent relative reduction. The net benefit of a true positive is USD 3,410, and the net cost of a
false positive is USD 600. The threshold is `600 / (3,410 + 600) = 0.158`. A programme that uses 0.5
instead will enrol only the highest-risk patients, miss most of the preventable readmissions, and
report a model that "works" while the intervention underperforms. The arithmetic is not optional;
it is the difference between a model and a programme.

The same logic applies to a model that is used only for ranking. If the programme takes the top
1,200 patients, the threshold is implicit in the ranking and the capacity, and it can be derived
after the fact as the score of the 1,200th patient. That is a legitimate operating point, but it
should be reported as a capacity-constrained threshold, with the sensitivity and net benefit it
implies, rather than presented as if it were a harm-ratio threshold.

The same arithmetic applies to rule-out decisions. A test that avoids an admission costs less than
the admission, and a missed case costs more. The threshold for acting (avoiding the admission) may
be low or high depending on the relative costs, and the decision curve makes the trade-off visible.
The direction is not always "lower threshold for rare events"; it depends on which error is costly.

A final framing point: the threshold is a property of the decision, not of the model. Two
programmes using the same model with different interventions should use different thresholds. A
model card that reports a single threshold without stating the intervention and the capacity has
not described a decision; it has described one programme's choice.

> **Note:** The threshold is not chosen to maximise a metric. It is chosen to reflect the relative
> harms and the operational capacity, and then the metrics at that threshold are reported. A
> threshold chosen after seeing the curve is a form of selection.

## 2. Learning outcomes

By the end of this topic you can:

1. **Derive** the harm-ratio threshold from the benefits and costs of the two errors.
2. **Compute** net benefit at a threshold and compare it with treat-all and treat-none.
3. **Plot** and **interpret** a decision curve, including its crossing points and its operating
   range.
4. **Reconcile** the threshold that maximises net benefit with the threshold that capacity allows.
5. **Explain** how a decision curve can reverse an AUROC ranking, and **identify** when the reversal
   matters.
6. **Check** calibration before trusting a threshold, and **adjust** the operating point when the
   probability scale is wrong.
7. **Report** a threshold with its derivation, its capacity, its net benefit, and its uncertainty.
8. **Critique** a model report that uses 0.5 by default or reports net benefit without a threshold.

Outcomes 1, 2, and 4 are computational; 5 and 8 are critique; 7 is a writing outcome.

## 3. The core mental model

A decision has two actions and two errors. The threshold is the risk at which the expected value of
acting exceeds the expected value of not acting.

```text
Act if:   p * B  >  (1 - p) * C
Rearrange: p > C / (B + C)

B = net benefit of a true positive
C = net cost of a false positive
t* = C / (B + C) = harm-ratio threshold
odds(t*) = C / B
```

The algebra is simple; the discipline is stating `B` and `C`. If `B` is the avoided cost of the
outcome minus the cost of the intervention, and `C` is the cost of the intervention plus any
downstream harm, then the threshold follows. If the team cannot state `B` and `C`, it cannot defend
a threshold, and any threshold it uses is arbitrary.

The decision curve adds the population. At a threshold `t`, net benefit is the true positive rate
weighted by prevalence minus the false positive rate weighted by the odds of the threshold:

```text
NB(t) = TPR(t) * prev - FPR(t) * (1 - prev) * odds(t)
```

The units are net true positives per patient. A net benefit of 0.03 means three net true positives
per 100 patients. Multiplying by the cohort size gives the net true positives per year, which is the
quantity a programme manager can compare with cost.

Three reference strategies anchor the curve.

```text
Treat none:  NB = 0
Treat all:   NB = prev - (1 - prev) * odds(t)
Model:       NB = TPR * prev - FPR * (1 - prev) * odds(t)
```

A model is useful at a threshold if its net benefit is above both references. If it is below
treat-all, the model is worse than acting on everyone; if it is below zero, it is worse than acting
on no one. Both are possible at different thresholds, and both are visible on the curve.

Capacity enters as a horizontal constraint on the flagged fraction, not as a term in net benefit.
The flagged fraction at a threshold is:

```text
flagged(t) = TPR(t) * prev + FPR(t) * (1 - prev)
```

If `flagged(t)` exceeds the programme's capacity, the operating point is not feasible. The feasible
threshold is the lowest threshold whose flagged fraction is within capacity, or the highest-risk
`k` patients if the programme allocates by rank. The net benefit at the feasible threshold is the
relevant number, not the maximum of the curve.

The net benefit formula assumes that the benefit of a true positive and the cost of a false
positive are the same for every patient, and that the decision has only two actions. A programme
that offers a graded response (a call for moderate risk, a visit for high risk) needs a separate
threshold for each grade and a separate net benefit analysis for each action. The single-threshold
formula is a special case, not a universal model. That is a simplification. A young patient with a
preventable readmission may benefit more than an elderly patient with a poor prognosis; a false
positive in a patient with limited reserve may cause more harm. The threshold is an average over
those differences, and the subgroup analysis is where the differences are examined.

The curve is drawn on the probability scale, so it inherits any miscalibration. If the model
over-predicts, the curve's apparent advantage may come from flagging patients whose true risk is
below the nominal threshold. The calibration check in Step 11 is not optional for a decision-curve
analysis; it is what makes the horizontal axis meaningful.

The net benefit formula has a useful property: it is linear in the confusion matrix and in the
prevalence, so the contribution of each patient can be written down. A true positive contributes
`1 - odds(t) * (1 - prev)/prev`? The exact decomposition is less important than the intuition: at a
given threshold, a true positive adds a fixed amount and a false positive subtracts a fixed amount,
and the amounts are set by the odds. The curve is the sum of those contributions across thresholds.

The curve can reverse the AUROC ranking because AUROC integrates over all thresholds, while net
benefit at a threshold weights errors by the odds. A model that is better at ruling out (high
sensitivity at low thresholds) can have a higher AUROC, while a model that is better at ruling in
(high specificity at high thresholds) can have higher net benefit at the staffable threshold. The
reversal is not a paradox; it is the consequence of asking a different question.

> **Note:** Net benefit depends on calibration. If the model's probabilities are miscalibrated, the
> nominal threshold does not correspond to the risk the decision-maker intended, and the curve is
> drawn on the wrong scale. Calibration is a prerequisite for threshold analysis, not a parallel
> activity.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Threshold | Predicted risk above which the action is taken | Set to 0.5 by default | Probability |
| Operating point | The threshold and the resulting confusion matrix | Confused with the threshold alone | Pair |
| Harm ratio | Cost of a false positive relative to benefit of a true positive | Assumed to be 1 | Ratio |
| Benefit of a true positive (`B`) | Net value of catching a case | Ignored in threshold choice | Currency or utility |
| Cost of a false positive (`C`) | Net cost of flagging a non-case | Ignored in threshold choice | Currency or utility |
| Odds of the threshold | `t / (1 - t)` | Confused with the threshold | Ratio |
| Net benefit | TPR-weighted prevalence minus FPR-weighted odds | Confused with profit | Per patient |
| Net true positives | Net benefit times the cohort size | Confused with true positives | Count |
| Treat-all | The strategy of acting on everyone | Ignored as a reference | Strategy |
| Treat-none | The strategy of acting on no one | Ignored as a reference | Strategy |
| Decision curve | Net benefit across thresholds | Read as a discrimination curve | Curve |
| Decision curve analysis | The method of computing and plotting net benefit | Confused with cost-effectiveness | Method |
| Standardised net benefit | Net benefit divided by prevalence | Used to compare across prevalences | Ratio |
| Test tradeoff | The number of additional tests per additional true positive | Confused with number needed to treat | Count |
| Capacity | Number of actions deliverable per period | Ignored until deployment | Patients per period |
| Capacity-constrained threshold | The threshold that flags exactly the deliverable number | Chosen without disclosure | Probability or rank |
| Flagged fraction | Proportion of patients above the threshold | Confused with prevalence | Proportion |
| Alert burden | Alerts per unit of clinical work | Measured only after launch | Alerts per 100 |
| Number needed to evaluate | Patients reviewed per true case found | Confused with number needed to treat | Count |
| Prevalence | Proportion of positives | Confused with flagged fraction | Proportion |
| Sensitivity | TPR at the threshold | Reported without the threshold | Proportion |
| Specificity | 1 minus FPR at the threshold | Reported without the threshold | Proportion |
| False positive rate | FP / (FP + TN) | Confused with false discovery rate | Proportion |
| PPV | TP / (TP + FP) | Assumed prevalence-independent | Proportion |
| Calibration | Agreement between predicted and observed risk | Ignored before thresholding | Property |
| Miscalibration | Systematic over- or under-prediction | Confused with poor discrimination | Property |
| Threshold drift | Change in the optimal threshold over time | Confused with model drift | Probability |
| Prevalence shift | Change in the outcome rate | Confused with threshold drift | Proportion |
| Harm-ratio sensitivity analysis | Repeating the analysis over a range of harm ratios | Omitted | Analysis |
| Decision analysis | Comparing strategies by expected utility | Confused with DCA alone | Method |
| Expected utility | Probability-weighted value of a strategy | Confused with net benefit | Utility |
| Quality-adjusted life year | A utility measure combining length and quality of life | Used without local weights | QALY |
| Number needed to treat | Patients treated for one additional benefit | Confused with number needed to evaluate | Count |
| Opportunity cost | Value forgone when capacity is used on one patient | Omitted from harm calculations | Currency |
| Rule-in | A decision that confirms and treats | Confused with any positive prediction | Decision type |
| Rule-out | A decision that avoids a test or admission | Confused with any negative prediction | Decision type |
| Threshold probability | Synonym for the threshold | Confused with the predicted probability | Probability |
| Preference | The value judgement encoded by the threshold | Treated as a statistical quantity | Value |
| Net harm | Negative net benefit | Reported as a positive number | Per patient |
| Clinical utility | The degree to which a model improves decisions | Confused with accuracy | Property |
| Decision rule | The threshold plus the action | Confused with the model | Rule |
| Default strategy | Treat-all or treat-none | Omitted from comparisons | Strategy |
| Cost-effectiveness | Cost per unit of health gain | Confused with net benefit | Ratio |
| Budget impact | Total cost of a strategy | Confused with cost-effectiveness | Currency |
| Equity constraint | A requirement that the rule not disadvantage a group | Ignored in threshold choice | Constraint |
| Threshold rule | The formula that maps risk to action | Confused with a single number | Rule |
| Post-hoc threshold | A threshold chosen after seeing the curve | Called exploratory | Bias |
| Pre-specified threshold | A threshold fixed before analysis | Required for confirmatory claims | Protocol |
| Net benefit curve | Synonym for decision curve | Read without references | Curve |
| Clinical range | The range of thresholds a reasonable clinician would use | Chosen arbitrarily | Interval |
| Net benefit difference | The change in net benefit between strategies | Confused with a p-value | Per patient |
| Decision threshold curve | Synonym for the decision curve | Read without references | Curve |
| Capacity gap | Flagged count minus capacity | Ignored in threshold choice | Count |
| Threshold sensitivity analysis | Repeating the choice over assumptions | Omitted | Analysis |
| Rule-out threshold | The threshold for avoiding a test or admission | Confused with the rule-in threshold | Probability |
| Net benefit per 1,000 | Net benefit times 1,000 | Confused with net true positives | Count |
| Capacity-constrained net benefit | Net benefit at the staffable threshold | Confused with the maximum | Per patient |
| Threshold rule version | The versioned decision rule | Not recorded | Version |

Three definitions deserve more than a table row.

**Net benefit versus expected utility.** Net benefit is a specific utility model in which the
benefit of a true positive and the cost of a false positive are constant and the harm ratio is
encoded by the threshold. Expected utility is broader and can include costs that accrue after the
decision, quality-of-life weights, and budget constraints. Net benefit is useful because it needs
only one assumption and can be plotted; it is not a full economic evaluation, and the report should
say so.

**The threshold as a preference.** The threshold is not a statistical estimate; it is a statement
about how much a false positive is worth relative to a true positive. Two clinicians can reasonably
disagree about the threshold for the same model, because they disagree about the harms. The
analysis should present the curve across a clinically plausible range and let the decision-maker
choose within it, rather than asserting a single threshold as objectively correct.

**Capacity as a constraint, not a preference.** Capacity is different from preference. A programme
that can review 12 patients a day cannot act on 20 percent of admissions, regardless of how much it
values the outcome. The capacity constraint is a fact about the system, and it should be reported
alongside the preference-based threshold. When the two conflict, the decision is either to expand
capacity or to accept a higher threshold and lower sensitivity; the model cannot resolve it.

> **Note:** The decision curve is a comparison of strategies, not a property of the model. The same
> model can be useful in one capacity and harmful in another, and the curve should be read with the
> capacity in view.

## 5. The mechanics, step by step

Threshold analysis has fourteen steps. Steps 1 through 4 derive the threshold. Steps 5 through 8
compute and plot net benefit. Steps 9 through 11 reconcile capacity and check calibration. Steps 12
through 14 report and defend the result.

### Step 1. Define the action and the capacity

**Input:** the prediction task specification.

**Output:** the action, the actor, the capacity per period, and the eligible population.

**Rationale:** The action determines the errors. If the action is a phone call, a false positive
costs a call; if it is a biopsy, it costs a procedure. The capacity determines the feasible
operating range. Both must be stated before the threshold, because both constrain it.

The capacity should be expressed in the same unit as the action, not in patients per year if the
action is a visit that takes an hour of a nurse's time. A capacity of 1,200 visits per year is
meaningful only if the staffing model supports 1,200 visits; a capacity of 1,200 patients per year
may hide a constraint on visit slots. Use the binding resource.

**Branch point:** if the action has a graded intensity (for example, a call, a visit, and an
admission), define the decision for each level separately. A single threshold cannot represent
three different actions with three different harm profiles.

### Step 2. Derive the harm ratio

**Input:** the benefit of a true positive, the cost of a false positive, and the evidence for each.

**Output:** `B`, `C`, and the ratio `C / B`, with a range for sensitivity analysis.

**Rationale:** The threshold is `C / (B + C)`. The benefit of a true positive is the absolute risk
reduction of the action times the value of the avoided outcome, minus the cost of the action. The
cost of a false positive is the cost of the action plus any downstream harm. The arithmetic is only
as good as the inputs, so both should carry a source and a range.

The benefit of a true positive should be net of the action's cost, and the cost of a false positive
should include the action's cost plus any downstream harm from unnecessary investigation. If the
action is a phone call, `C` is the call time; if the action is an admission, `C` includes the bed,
the risks of hospital-acquired complications, and the opportunity cost of the bed. The asymmetry
between `B` and `C` is what makes the threshold different from 0.5.

**Branch point:** if `B` or `C` is unknown, present the curve across a range of harm ratios and
report which decisions are stable across the range. Do not invent a point estimate. Do not invent a point estimate.

### Step 3. Compute the threshold and its odds

**Input:** `B` and `C`.

**Output:** the threshold `t* = C / (B + C)` and its odds `t* / (1 - t*) = C / B`.

**Rationale:** The odds form is often more intuitive: a threshold of 0.20 means the decision-maker
accepts four false positives per true positive. The odds also appear directly in the net benefit
formula, so computing both makes the arithmetic transparent.

**Branch point:** if the threshold is above 0.5, the intervention is high-harm relative to the
outcome, and the programme should act only on the highest-risk patients. If it is below 0.1, the
intervention is low-harm and the programme should cast a wide net, subject to capacity.

### Step 4. State the threshold rule before seeing the curve

**Input:** the harm ratio and the capacity.

**Output:** a pre-specified threshold or a pre-specified rule for choosing one (for example, "the
lowest threshold that fits capacity, within the harm-ratio range").

**Rationale:** A threshold chosen after seeing the curve is selected on the data. The selection
inflates the apparent net benefit and makes the result unreproducible. Pre-specifying the rule is
the threshold analogue of pre-registering the primary metric.

The rule can be conditional: "use the harm-ratio threshold if it fits capacity; otherwise use the
capacity-constrained threshold and report the sensitivity loss." A conditional rule is still
pre-specified, and it is more honest than choosing a single number and pretending it satisfies both
constraints.

**Branch point:** if the analysis is exploratory, label it exploratory and do not present the
chosen threshold as a validated operating point.

### Step 5. Compute the confusion matrix at candidate thresholds

**Input:** the held-out predictions, outcomes, and a grid of thresholds.

**Output:** sensitivity, specificity, and flagged fraction at each threshold.

**Rationale:** The confusion matrix is the input to net benefit. Computing it on a grid makes the
curve reproducible and lets the reader see the operating characteristics at the thresholds of
interest. The grid should include the harm-ratio threshold, the capacity threshold, and a
clinically plausible range around them.

The grid should be dense enough that the curve's shape is visible, but not so dense that every
point is noise. A grid of 0.01 from 0.01 to 0.60 is usually adequate for a clinical model; a
coarser grid can miss a crossing. The grid is a reporting choice and should be stated.

**Branch point:** if the score has many ties, the confusion matrix can change with the comparison
rule. Use a consistent rule and report the tie count at the operating point.

### Step 6. Compute net benefit and the reference strategies

**Input:** sensitivity, specificity, prevalence, and the threshold.

**Output:** net benefit for the model, treat-all, and treat-none at each threshold.

**Rationale:** The references are what make the curve interpretable. A model's net benefit of 0.03
means little without knowing that treat-all is 0.01 and treat-none is 0. The comparison is the
decision-relevant quantity.

Net benefit can be negative. A negative value means the strategy is worse than doing nothing, not
merely less good than another strategy. The treat-all curve is negative above the prevalence
threshold, because the false positive penalty exceeds the true positive gain. A model curve that
crosses below zero is a signal that the decision rule is harmful at that threshold, and the report
should say so plainly.

**Branch point:** if the model is below treat-all across the entire clinical range, the model is
not useful for this decision, regardless of its AUROC. Report that result directly.

### Step 7. Plot the decision curve

**Input:** the net benefit table.

**Output:** a plot with net benefit on the y-axis, threshold on the x-axis, and curves for the
model, treat-all, and treat-none.

**Rationale:** The plot shows the range where the model is useful, the crossing points, and the
shape of the trade-off. It is the primary communication device for the threshold decision, because
it lets a clinician see the consequence of their own preference.

**Branch point:** if the model curve crosses treat-all within the clinical range, the model is
useful only on one side of the crossing. Report the crossing point and the side that matters.

### Step 8. Mark the clinical range and the operating point

**Input:** the plot and the pre-specified threshold.

**Output:** a shaded clinical range and a marked operating point with its sensitivity, specificity,
PPV, and flagged fraction.

**Rationale:** A curve without a marked operating point invites the reader to choose the most
flattering point. Marking the pre-specified point makes the decision explicit and shows whether the
chosen threshold is in a region where the model is actually useful.

**Branch point:** if the operating point is outside the range where the model beats the references,
the threshold and the model are mismatched. Revisit the threshold or the model.

### Step 9. Compute the capacity-constrained threshold

**Input:** the flagged fraction at each threshold and the capacity.

**Output:** the lowest threshold whose flagged fraction is within capacity, and the net benefit at
that threshold.

**Rationale:** The capacity-constrained threshold is the best the programme can do with its current
resources. It is often higher than the harm-ratio threshold, which means lower sensitivity and
higher PPV. The net benefit at the constrained threshold is the number the programme will actually
experience.

The capacity-constrained threshold is not necessarily the threshold that maximises net benefit
subject to capacity. The true constrained optimum is the threshold that maximises net benefit among
those whose flagged fraction is within capacity. Because net benefit usually declines with the
threshold once the model is reasonably good, the constrained optimum is often the lowest feasible
threshold, but not always. Compute the net benefit at each feasible threshold and choose the
maximum rather than assuming the lowest.

**Branch point:** if no threshold within the clinical range fits capacity, the programme cannot
implement the decision as designed. Report the gap and the options: expand capacity, narrow the
population, or change the action.

### Step 10. Compare the unconstrained and constrained optima

**Input:** the net benefit curve, the harm-ratio threshold, and the capacity threshold.

**Output:** a table of the unconstrained maximum, the harm-ratio threshold, and the
capacity-constrained threshold, with net benefit and operating characteristics for each.

**Rationale:** The comparison makes the cost of the capacity constraint explicit. The difference in
net benefit between the unconstrained optimum and the constrained point is the value of additional
capacity, which is a budget argument the programme owner can use.

The value of additional capacity is the difference between the unconstrained net benefit and the
constrained net benefit, multiplied by the cohort size. That number can be compared with the cost
of the additional staff or slots. If the value is below the cost, expanding capacity is not
justified, and the constrained threshold is the right operating point.

**Branch point:** if the constrained threshold has negative net benefit, the programme should not
deploy at its current capacity. The model may be useful only if capacity expands.

### Step 11. Check calibration at the operating point

**Input:** the calibration slope, intercept, and plot.

**Output:** a statement of whether the nominal threshold corresponds to the intended risk, and a
recalibrated threshold if it does not.

**Rationale:** The threshold is a risk, and a risk is only meaningful on a calibrated scale. If the
model over-predicts, a nominal threshold of 0.20 may correspond to a true risk of 0.15, which means
the programme is acting on patients below its harm-ratio threshold. The decision curve drawn on the
nominal scale is then misleading.

The calibration check should be done at the operating threshold, not only on average. A model can
have a slope near 1 overall and be miscalibrated in the region of the threshold. Plot observed
versus predicted risk in the neighborhood of the threshold, with counts, before trusting the
nominal value.

**Branch point:** if the calibration slope is far from 1, recalibrate before choosing the threshold,
and recompute the curve on the recalibrated scale.

### Step 12. Assess subgroups and equity

**Input:** the curve and the subgroup identifiers.

**Output:** subgroup net benefit at the same threshold, with counts.

**Rationale:** A threshold that is optimal overall can disadvantage a subgroup with a different
prevalence or a different harm profile. The subgroup analysis shows whether the same threshold
produces acceptable net benefit across groups, or whether a group-specific threshold is needed.

A subgroup-specific threshold is sometimes justified when the prevalence or the harm profile
differs materially. It is also a source of inequity if the difference is not clinically defensible.
The report should state the reason for any subgroup-specific threshold and the evidence that
supports it, rather than adjusting thresholds until each group's metrics look similar.

**Branch point:** if a subgroup has fewer than about 20 events, report the count and avoid a
subgroup-specific threshold. A threshold difference based on a small subgroup is likely noise.

### Step 13. Run a sensitivity analysis over the harm ratio and prevalence

**Input:** the curve and ranges for `B`, `C`, and prevalence.

**Output:** a table of net benefit and the preferred strategy across the ranges.

**Rationale:** The harm ratio and prevalence are estimates, not certainties. A decision that is
stable across the plausible range is more defensible than one that depends on a single point
estimate. The sensitivity analysis also shows which assumption the decision is most sensitive to.

The sensitivity analysis should vary one assumption at a time, so that the effect of each is
visible. A joint scenario analysis (for example, low benefit and low prevalence together) is useful
for stress-testing, but it does not replace the one-at-a-time analysis that shows which assumption
drives the result.

**Branch point:** if the preferred strategy changes within the plausible range, the decision is
fragile. Report that and propose the evidence needed to resolve it.

### Step 14. Report the threshold decision

**Input:** the full analysis.

**Output:** a threshold section with the harm ratio, the capacity, the curve, the operating point,
the calibration check, and the sensitivity analysis.

**Rationale:** The report should let a reader see why the threshold was chosen, what it costs, and
how the decision changes under different assumptions. A threshold reported without its derivation
is a convention, not a decision.

**Branch point:** if the threshold was chosen by capacity rather than by the harm ratio, say so.
The reader needs to know which constraint bound.

### Threshold edge cases

**Perfect separation at the threshold.** A model with a threshold that perfectly separates the
validation sample is usually leaking. Check the split and the features before trusting the curve.

**Zero events in a threshold bin.** Sensitivity or specificity may be undefined. Use the pooled
confusion matrix or a smoother, and report the bin counts.

**Ties at the capacity boundary.** More patients may share the boundary score than there are slots.
Pre-specify a tie-break rule and report the number affected.

**Prevalence shift.** A curve at 12 percent prevalence does not apply at 6 percent. Recompute the
curve at the deployment prevalence before choosing the threshold.

**Calibration drift.** A threshold chosen on a calibrated model may be wrong after drift.
Recompute the curve on the current calibration.

**Capacity varying by day.** An annual average can hide daily peaks. Use the upper end of the daily
volume for the capacity constraint.

**Multiple actions from one score.** A score that triggers a call and a visit has two thresholds and
two capacities. Analyse them separately.

**A model with no useful threshold.** Some models are below the references at every threshold in
the clinical range. Report that, with the curve, rather than searching for a threshold where the
model happens to look good.

**A decision curve computed on the training data.** The curve will be optimistic. Use held-out
predictions, and if the model was tuned, use nested predictions.

**A curve with no confidence band.** The crossing points and the net benefit differences are
estimates, and a small difference can reverse with a different sample. Bootstrap the curve over patients and report the band, especially near the operating
point.

**A threshold that no patient exceeds.** If the highest predicted risk is below the threshold, the
model flags no one and the net benefit is zero. Report the maximum predicted risk and the
threshold.

**A threshold at the edge of the observed range.** A threshold at the 99th percentile of predicted
risk may flag only a handful of patients, and the net benefit estimate will be unstable. Report the
count and the interval.

**A threshold outside the observed risk range.** If the threshold is above the highest predicted
risk, the model flags no one. Report the range and the impossibility.

## 6. Worked example

### The setting

A heart failure programme compares two readmission models. Model A has an AUROC of 0.74; model B
has an AUROC of 0.69. The deployment prevalence is 12 percent, and the programme can staff 1,200
enrolments per year out of 10,000 discharges, or 12 percent of the cohort. The harm ratio implies a
threshold of 0.20, but the team also wants to see the full curve.

### The operating characteristics

| Threshold | A sensitivity | A specificity | B sensitivity | B specificity |
|---:|---:|---:|---:|---:|
| 0.05 | 0.88 | 0.45 | 0.80 | 0.55 |
| 0.08 | 0.82 | 0.58 | 0.74 | 0.65 |
| 0.12 | 0.72 | 0.68 | 0.66 | 0.78 |
| 0.18 | 0.55 | 0.85 | 0.62 | 0.84 |
| 0.25 | 0.42 | 0.93 | 0.58 | 0.88 |
| 0.30 | 0.35 | 0.95 | 0.48 | 0.93 |
| 0.35 | 0.30 | 0.96 | 0.40 | 0.94 |

### The net benefit arithmetic

At 12 percent prevalence and a threshold of 0.25:

```text
Model A: 0.42 * 0.12 - 0.07 * 0.88 * (0.25 / 0.75)
       = 0.0504 - 0.0616 * 0.3333
       = 0.0504 - 0.0205
       = 0.0299

Model B: 0.58 * 0.12 - 0.12 * 0.88 * 0.3333
       = 0.0696 - 0.0352
       = 0.0344
```

Model B has higher net benefit at 0.25, despite a lower AUROC. At a threshold of 0.05, the ranking
reverses:

```text
Model A: 0.88 * 0.12 - 0.55 * 0.88 * (0.05 / 0.95)
       = 0.1056 - 0.4840 * 0.0526
       = 0.1056 - 0.0255
       = 0.0801

Model B: 0.80 * 0.12 - 0.45 * 0.88 * 0.0526
       = 0.0960 - 0.0208
       = 0.0752
```

Model A is better at the low threshold, where high sensitivity matters; model B is better at the
high threshold, where specificity matters. The AUROC integrates over both regions and ranks A
higher, but the decision at the staffable threshold favours B.

The full curve, computed at each threshold, is monotonic in this table because the sensitivity and
specificity pairs decline smoothly. In a real model, the curve often has a shoulder where the
model's advantage over treat-all is largest, and the choice within that shoulder is a matter of
preference rather than a single optimum. The shoulder is the clinically relevant range; the exact
maximum is less important than the range.

### The capacity constraint

The flagged fraction at 0.25 is `0.42 * 0.12 + 0.07 * 0.88 = 0.112` for A and `0.58 * 0.12 + 0.12 *
0.88 = 0.175` for B. Model A fits within the 12 percent capacity at 0.25; model B does not.

The lowest threshold at which B fits is 0.30, where the flagged fraction is `0.48 * 0.12 + 0.07 *
0.88 = 0.119`, or 1,190 patients. Its net benefit at 0.30 is:

```text
0.48 * 0.12 - 0.07 * 0.88 * (0.30 / 0.70)
= 0.0576 - 0.0616 * 0.4286
= 0.0576 - 0.0264
= 0.0312
```

Model A at 0.25 has net benefit 0.0299; model B at its capacity-constrained threshold has 0.0312.
Model B still wins, but by a much smaller margin. The capacity constraint cost B about 0.0032 net
true positives per patient, or 3.2 per 1,000, relative to its net benefit at 0.25.

### The unconstrained optimum versus the staffable threshold

Both models have their highest net benefit at the lowest threshold in the table, 0.05, because the
table does not extend below it. At 0.05, model A has net benefit 0.0801 and flags 59 percent of the
cohort; model B has 0.0752 and flags 49 percent. Neither is staffable. The unconstrained optimum is
a decision to act on half the population, which the programme cannot do.

The staffable thresholds are 0.25 for A and 0.30 for B. The gap between the unconstrained optimum
and the staffable threshold is the value of capacity: at 0.05, A would gain `0.0801 - 0.0299 =
0.0502` net true positives per patient, or 502 per 10,000, if it could staff the additional
enrolments. That is the number the programme owner needs for the budget conversation.

### Two diagnostics that would reveal a broken analysis

**Diagnostic 1: calibration at the threshold.** Model A's calibration slope is 0.72 with an
intercept of 0.10; model B's is 0.91 with an intercept of 0.01. Model A's nominal 0.25 corresponds
to a higher true risk, so its threshold is more conservative than intended. Recalibrating A changes
its operating point: at a true risk of 0.25, A's sensitivity rises to about 0.48 and its specificity
falls to about 0.90, giving a net benefit of `0.0576 - 0.0672 * 0.3333 = 0.0352`. The reversal
disappears, and A becomes the better model. A decision-curve report that ignores calibration can
choose the wrong model.

**Diagnostic 2: prevalence at the deployment site.** The curves above are at 12 percent prevalence.
At 6 percent, the net benefit at 0.25 for A is `0.42 * 0.06 - 0.07 * 0.94 * 0.3333 = 0.0033`, and
for B it is `0.58 * 0.06 - 0.12 * 0.94 * 0.3333 = -0.0028`. Model B becomes harmful while A remains
barely positive. The ranking reverses again, and the decision changes from "deploy B" to "deploy A
or do not deploy". The prevalence is not a detail; it is half of the decision.

### The uncertainty in the difference

The net benefit difference between A and B at 0.25 is 0.0045 per patient. Whether that difference
is real depends on its interval. A bootstrap over patients, recomputing both models' curves on each
resample, would show whether the interval includes zero. With 10,000 patients and 1,200 events, the
interval is likely narrow enough to exclude zero, but the report should show it. A difference
smaller than its interval is not a basis for a model choice.

### The test tradeoff

The difference in net benefit between two models can be converted into a test tradeoff: the number
of additional patients who must be evaluated to gain one additional true positive. At 0.25, model B
has a net benefit 0.0045 higher than A, or 4.5 net true positives per 1,000. The reciprocal of the
difference, `1 / 0.0045 = 222`, is the number of patients who must be evaluated with the better
model to gain one net true positive, under the assumptions of the comparison. That number is useful
for a programme deciding whether the additional evaluation burden is worth the gain.

The test tradeoff is a communication device, not a substitute for the decision curve. It assumes
that the difference in net benefit is stable and that the cost of evaluation is the only cost. It
should be reported with those caveats.

### Varying one assumption: prevalence falls to 6 percent

The arithmetic above shows the reversal. At 6 percent, model B's false positives cost more than its
true positives gain at the 0.25 threshold, and its net benefit is negative. Model A's net benefit is
small but positive. The conclusion changes from "B is the better decision aid" to "B is harmful at
this prevalence, and A is marginally useful". A report that fixed prevalence at 12 percent would
have missed the change.

### The decision at a lower threshold

If the programme could staff 30 percent of the cohort, the feasible threshold for both models would
fall to about 0.12, where A has net benefit 0.0480 and B has 0.0528. B still wins, but the margin
is larger, and the sensitivity is higher. The capacity constraint, not the model, is the binding
constraint on the programme's reach.

### What would change the recommendation

The recommendation would change if the deployment prevalence were below about 8 percent, where
model B's net benefit at the staffable threshold approaches zero. It would change if model A were
recalibrated, which raises its net benefit at 0.25 to about 0.035 and makes it the better model. It
would change if the programme's capacity were expanded to 20 percent, where B's lower threshold
becomes feasible and its net benefit advantage grows. Each of these is a specific, testable
condition, and the report should state them rather than presenting the recommendation as universal.

### What to report

The primary figure is the decision curve at the deployment prevalence, with the model curves, the
treat-all and treat-none references, the clinical range shaded, and the operating point marked. The
secondary table is the unconstrained optimum, the harm-ratio threshold, and the capacity-constrained
threshold, with net benefit and operating characteristics for each. The calibration check belongs
next to the operating point, because a threshold on a miscalibrated scale is not the threshold the
decision-maker intended.

## 7. Code, query, or calculation

The script computes net benefit, the references, the unflagged and flagged fractions, the
unconstrained maximum, and the capacity-constrained operating point from a table of sensitivity and
specificity at candidate thresholds. The table is the output of a model evaluated on held-out data;
the arithmetic is the reusable part.

```python
"""Net benefit, decision curves, and capacity constraints. Standard library only."""

PREVALENCE = 0.12
CAPACITY = 0.12

# threshold: (sens_A, spec_A, sens_B, spec_B)
POINTS = {
    0.05: (0.88, 0.45, 0.80, 0.55),
    0.08: (0.82, 0.58, 0.74, 0.65),
    0.12: (0.72, 0.68, 0.66, 0.78),
    0.18: (0.55, 0.85, 0.62, 0.84),
    0.25: (0.42, 0.93, 0.58, 0.88),
    0.30: (0.35, 0.95, 0.48, 0.93),
    0.35: (0.30, 0.96, 0.40, 0.94),
}


def net_benefit(sensitivity, specificity, threshold, prevalence=PREVALENCE):
    odds = threshold / (1 - threshold)
    return sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * odds


def flagged_fraction(sensitivity, specificity, prevalence=PREVALENCE):
    return sensitivity * prevalence + (1 - specificity) * (1 - prevalence)


def treat_all(threshold, prevalence=PREVALENCE):
    return prevalence - (1 - prevalence) * threshold / (1 - threshold)


print("t     NB_A    NB_B    NB_all  flagA  flagB")
for t, (sa, pa, sb, pb) in POINTS.items():
    print(f"{t:.2f}  {net_benefit(sa, pa, t):.4f}  {net_benefit(sb, pb, t):.4f}  "
          f"{treat_all(t):.4f}  {flagged_fraction(sa, pa):.3f}  "
          f"{flagged_fraction(sb, pb):.3f}")

for name, index in (("A", 0), ("B", 2)):
    best = max((net_benefit(v[index], v[index + 1], t), t)
               for t, v in POINTS.items())
    print(f"max NB model {name}: {best[0]:.4f} at t={best[1]:.2f}")

for name, index in (("A", 0), ("B", 2)):
    feasible = [(t, net_benefit(v[index], v[index + 1], t))
                for t, v in POINTS.items()
                if flagged_fraction(v[index], v[index + 1]) <= CAPACITY]
    if feasible:
        t, nb = max(feasible, key=lambda pair: pair[1])
        print(f"capacity-constrained model {name}: t={t:.2f} NB={nb:.4f}")
    else:
        print(f"capacity-constrained model {name}: no feasible threshold")
```

Expected output:

```text
t     NB_A    NB_B    NB_all  flagA  flagB
0.05  0.0801  0.0752  0.0737  0.590  0.492
0.08  0.0663  0.0620  0.0435  0.468  0.397
0.12  0.0480  0.0528  0.0000  0.368  0.273
0.18  0.0370  0.0435  -0.0732  0.198  0.215
0.25  0.0299  0.0344  -0.1733  0.112  0.175
0.30  0.0231  0.0312  -0.2571  0.086  0.119
0.35  0.0170  0.0196  -0.3538  0.071  0.101
max NB model A: 0.0801 at t=0.05
max NB model B: 0.0752 at t=0.05
capacity-constrained model A: t=0.25 NB=0.0299
capacity-constrained model B: t=0.30 NB=0.0312
```

Reading the output:

- At low thresholds, A has the higher net benefit, consistent with its higher AUROC and its
  sensitivity advantage. At 0.12 and above, B has the higher net benefit, despite the lower AUROC.
- Treat-all is better than both models below about 0.10 and much worse above 0.12. The crossing
  points are where the model's advantage over the default changes sign.
- The unconstrained maximum is at 0.05 for both models, where 59 percent or 49 percent of the
  cohort is flagged. Those operating points are not staffable at 12 percent capacity.
- At capacity, A's feasible threshold is 0.25 and B's is 0.30. B has the higher net benefit at its
  feasible threshold, by 0.0013 per patient, or 1.3 net true positives per 1,000. The margin is
  small enough that its interval matters.
- The capacity constraint costs B `0.0344 - 0.0312 = 0.0032` per patient relative to its net
  benefit at 0.25. That is the value of additional capacity at the margin.

The table is a discretised curve. A real analysis should compute the curve on a finer grid and
bootstrap the difference. The code is designed to make the arithmetic visible, not to replace a
decision-curve library.

## 8. Data, standards, and regulatory context

Thresholds and net benefit are addressed by reporting guidelines and health-technology assessment
practice, not by a single binding standard. The table distinguishes requirements from conventions.

| Instrument | Status | What it says about thresholds | What it does not do |
|---|---|---|---|
| TRIPOD+AI | Reporting convention | Requires the threshold and how it was chosen to be reported | Does not prescribe a threshold |
| PROBAST+AI | Appraisal tool | Assesses whether the evaluation reflects the intended use, including thresholds | Does not compute net benefit |
| FDA GMLP principles | Guidance, not binding | Calls for performance testing at the intended operating point | Does not specify a threshold method |
| FDA PCCP guidance | Guidance, not binding | Requires acceptance criteria for planned changes, which may include thresholds | Does not define net benefit |
| NICE diagnostic guidance | HTA convention | Expects clinical and cost effectiveness, often with decision modelling | Not a statistical standard |
| EU AI Act | Binding in the EU, phased | Requires technical documentation and testing for high-risk systems | Does not specify thresholds |
| ISO 14971 | Voluntary standard | Risk management across the lifecycle | Does not address thresholds |
| NIST AI RMF | Voluntary framework | Measure and manage performance in context | Not sector-specific |

Three points matter for practice.

**TRIPOD+AI expects the threshold to be reported.** A model report that gives AUROC and calibration
but not the operating threshold has not described the decision. The threshold is part of the
performance claim, because the same model has different sensitivity and PPV at different
thresholds. The report should state the threshold, its derivation, and the capacity it implies.

**The threshold and the capacity plan.** A threshold that fits an annual average may not fit a
Monday in January. The capacity plan should use the peak period, and the threshold should be
reviewed when the peak changes. A model that is safe on average and unsafe on the busiest day is
not safe.

**The threshold and the model card.** A model card should state the threshold, the capacity it
assumes, the harm ratio it encodes, and the net benefit at that point. A reader who does not know
the threshold cannot interpret the reported sensitivity and PPV, because those metrics change with
the threshold. The card should also state the range of thresholds over which the model has been
evaluated.

**The threshold and the label.** A threshold rule is only as valid as the outcome it predicts. If
the label changes, the threshold's meaning changes, even if the model's probabilities are unchanged.
A coding update that broadens the outcome definition will lower the prevalence and shift the
optimal threshold. The threshold report should reference the label version, not only the model
version.

**The FDA's GMLP principles emphasise the intended use.** A device's performance claim should be
evaluated at the operating point the user will use. A submission that reports performance across all
thresholds but not at the labelled threshold leaves the user without the information they need. The
guidance does not prescribe net benefit, but it does expect the operating point to be justified.

**HTA bodies expect cost-effectiveness, not only net benefit.** Net benefit is a decision-analytic
measure with a single harm ratio; a health-technology assessment typically requires a full
cost-effectiveness model with quality-adjusted life years, resource use, and budget impact. Net
benefit is a useful bridge, not a substitute. The report should state that the economic evaluation
is outside its scope if it is.

### A threshold reporting checklist

A threshold section that can be audited contains the following.

```text
action, actor, and eligible population
capacity per period and the binding resource
benefit of a true positive, with source and range
cost of a false positive, with source and range
harm-ratio threshold and odds
pre-specified threshold rule
decision curve with treat-all and treat-none
clinical range and operating point
confusion matrix at the operating point
calibration check at the operating point
capacity-constrained threshold and sensitivity loss
subgroup net benefit with counts
sensitivity analysis over harm ratio and prevalence
```

The checklist maps onto the TRIPOD+AI threshold item and the PROBAST+AI analysis domain. A section
that gives a threshold without the harm ratio and capacity has not explained the decision.

> **Note:** No regulation requires a decision curve. The requirement is that the operating point be
> justified and that the performance at that point be reported. DCA is the clearest way to do both.

## 9. Industry practice and case evidence

**The original method.** Vickers and Elkin (2006) introduced decision curve analysis as a way to
evaluate prediction models by their clinical utility rather than their discrimination. The method
computes net benefit across thresholds and compares it with treat-all and treat-none. The original
paper includes a step-by-step calculation and an extension for test harm.

**The interpretation guide.** Vickers, van Calster, and Steyerberg (2019) provide a step-by-step
guide to interpreting DCA. They recommend relabelling the axes as "benefit" and "preference",
explain that the threshold probability is the odds of the harm ratio, and caution against using DCA
to choose a threshold. They also note that a model can be harmful despite an AUROC above 0.5 if it
is miscalibrated.

**Reporting guidance.** Van Calster et al. provide a guide for investigators that introduces the
test tradeoff, the number of additional tests per additional true positive, as a way to judge
whether a net-benefit gain justifies the cost of measuring predictors. The guide also recommends
reporting the clinical range and the uncertainty in the curve.

**Sepsis alert burden.** The external validation of the Epic Sepsis Model reported an AUROC of 0.63,
a sensitivity of 33 percent, a PPV of 12 percent, and 18 percent of hospitalisations flagged at the
vendor's threshold. The case is a threshold failure as much as a discrimination failure: the
operating point produced an alert volume incompatible with review capacity, and the low PPV meant
most reviewed patients did not have sepsis. A decision curve at the local prevalence would have
shown the model's net benefit at the staffable threshold.

**Readmission programmes.** The readmission literature repeatedly finds modest discrimination, with
c-statistics of 0.55 to 0.65 for administrative models. A model in that range can still have
positive net benefit at a low threshold if the intervention is cheap and effective, and negative
net benefit at a high threshold if it is not. The threshold and the intervention's effect are as
important as the c-statistic.

**The threshold in a regulated claim.** A device's labelling may state a threshold and a
corresponding sensitivity and specificity. If the threshold is changed, the claim changes, and the
change may require a new submission unless it is covered by a predetermined change control plan.
The threshold is part of the regulated artefact, not a local configuration.

**The net benefit of a rule-out strategy.** A rule-out decision, such as avoiding an admission or a
test, has the opposite error structure: a false negative is the costly error, and the threshold is
the risk below which the rule-out is safe. The DCA formula is the same, but the action is attached
to the negative prediction, and the curve should be read accordingly. A report that applies a
rule-in curve to a rule-out decision has reversed the decision.

**A note on the evidence for DCA.** The method is analytic: net benefit follows from expected
utility under stated assumptions. The empirical evidence that DCA improves clinical decisions is
thinner, because few trials have randomised clinicians to a DCA-derived threshold. The report
should distinguish the method's mathematical validity from the evidence that using it changes
outcomes.

**What the cases do not show.** None of the public cases reports a full decision curve with a
capacity constraint at the deployment site. That is a gap in the literature, not evidence that the
analysis is unnecessary. The methods papers describe how to do it; the empirical papers rarely
report it. A team that reports the curve and the capacity is ahead of the common practice.

**The capacity conversation.** In practice, the threshold is often set by capacity rather than by a
harm ratio: the programme takes the top 1,200 patients because that is what it can staff. That is a
legitimate decision, but it should be reported as a capacity-constrained threshold, with the
sensitivity loss relative to the harm-ratio threshold stated. Presenting a capacity-driven threshold
as if it were derived from harms misrepresents the decision.

> **Note:** The cases are documented. The claim that a specific programme's threshold was set by
> capacity rather than by harm is not made here; the cases show the mechanism.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Threshold at 0.5 by default | No derivation | Convention substitutes for decision analysis | Wrong operating point | Derive from harms |
| Threshold chosen after the curve | The chosen point is the maximum | Selection on the data | Optimistic net benefit | Pre-specify the rule |
| Capacity ignored | Flagged count exceeds staff | Operational mismatch | Alert fatigue | Compute flagged fraction |
| Net benefit without references | Model curve only | No comparison | Uninterpretable | Add treat-all and treat-none |
| AUROC used to choose the threshold | The model with the highest AUROC is selected | Ranking confused with utility | Wrong model at the operating point | Compare net benefit at the threshold |
| Calibration ignored | Nominal threshold treated as true risk | Miscalibration | Wrong patients flagged | Check calibration first |
| Prevalence assumed | Curve at development prevalence only | Base rate changes | Net benefit changes sign | Recompute at deployment prevalence |
| Harm ratio invented | No source for B or C | False precision | Wrong threshold | Sensitivity analysis |
| No confidence band | Single curve | Sampling noise | Overconfident crossing | Bootstrap |
| Subgroup net benefit omitted | Overall curve only | Heterogeneity hidden | Inequity | Subgroup curves |
| Threshold not implementable | Score not shown at decision time | Workflow mismatch | No action | Workflow test |
| Test tradeoff ignored | Net benefit gain without cost | Cost of measurement omitted | Impractical recommendation | Report tradeoff |
| Multiple actions, one threshold | One cut-off for call and visit | Different harms collapsed | One decision mis-set | Separate analyses |
| Negative net benefit not reported | Only positive results shown | Selective reporting | Harmful deployment | Report the sign |
| Capacity gap not quantified | "We will manage" | Operational optimism | Unreviewed alerts | Report the gap |
| Curve computed at one prevalence | No prevalence range | Base-rate neglect | Wrong decision | Recompute |
| No subgroup curve | Only overall | Heterogeneity hidden | Inequity | Subgroup analysis |
| Threshold not versioned | Changed without record | Drift in practice | Unreproducible | Version the rule |
| Label version omitted | The outcome definition changed | Threshold meaning shifts | Wrong threshold | Reference the label version |
| Capacity assumed constant | No seasonal variation | Daily peaks | Unreviewed alerts | Use peak capacity |
| Threshold chosen by a vendor | No local derivation | Commercial default | Mismatch with local harms | Local elicitation |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The harm ratio is constant across patients | A single threshold summarises the decision | Harms vary by severity or group | Subgroup analysis |
| The action's effect is constant | Net benefit assumes a fixed benefit per true positive | Effect heterogeneity | Stratify the effect |
| Capacity is fixed and known | The constraint is well defined | Staffing varies | Use a range |
| The prevalence is known | Net benefit depends on it | Prevalence shift | Recompute |
| The model is calibrated | The threshold is a risk | Miscalibration | Calibration plot |
| The validation sample represents deployment | The curve transports | Case-mix or site shift | External validation |
| The score is available at decision time | The threshold can be applied | Late-arriving features | Workflow audit |
| The comparison uses the same sample | Differences reflect models | Different validation sets | Reuse the split |
| The curve is estimated on held-out data | Net benefit is honest | In-sample evaluation | Nested predictions |
| The decision has two actions | Net benefit formula applies | Graded actions | Multi-action analysis |
| The threshold is stable over time | The rule remains valid | Threshold drift | Monitor net benefit |
| The capacity is the binding resource | The constraint is real | Other bottlenecks | Workflow study |
| The harm ratio applies to the deployment population | The threshold transports | Different values | Local elicitation |
| The threshold is recorded in the model artefact | The deployed rule matches the validated rule | Configuration drift | Version audit |
| The capacity is measured in the binding resource | The constraint is real | Wrong unit | Workflow study |
| The net benefit is computed on held-out predictions | The estimate is honest | In-sample curve | Audit fit scope |
| The decision rule is implemented as specified | The threshold is used | Software default | Configuration test |
| The threshold is within the validated risk range | The rule is supported | Extrapolation | Range check |
| The operating point is reviewed after drift | The rule remains valid | Silent drift | Monitoring plan |

The deepest limit is that net benefit is a one-dimensional summary of a multi-dimensional decision.
It assumes a constant harm ratio, a constant effect, and a binary action. A real programme may have
graded actions, heterogeneous effects, and equity constraints that a single curve cannot represent.
The curve is a starting point for the conversation, not the end of it. The decision also depends
on the intervention's effect, the programme's fidelity, and the patient's preferences, none of
which the curve measures. Those belong in the implementation plan, not in the model report.

A second limit is that net benefit is not a budget. It counts net true positives, not dollars, and
a programme with a fixed budget must compare the cost per net true positive across strategies. The
test tradeoff and the cost per case found are the bridge between net benefit and the budget, and
they belong in the report when the programme has a fixed budget.

## 12. Exercises

A learner who completes the arithmetic exercises should be able to reproduce the worked example's
table and explain the reversal. The exercises are deliberately numerical: a threshold argument that
cannot be reproduced from the counts is not an argument. Where a dataset is unavailable, use the
worked example's table and vary one assumption at a time. A learner who completes the design exercises should be able to
write the threshold section of a model card.

The exercises build the threshold section from the harm ratio to the report. Exercises 1 and 2 are
arithmetic, 3 through 6 require predictions, 7 is a sensitivity analysis, and 8 is the written
section for the module assignment.

**Exercise 1. Harm-ratio threshold.** For `B = USD 8,000` and `C = USD 500`, compute the threshold
and the odds. Repeat for `C = USD 3,000`. Deliverable: the arithmetic and a sentence on the
direction. Acceptance criteria: values correct; interpretation states the false positives accepted
per true positive. Time: 30 minutes.

**Exercise 2. Net benefit.** At 10 percent prevalence, sensitivity 0.70, specificity 0.80, compute
net benefit at thresholds 0.10, 0.20, and 0.30, and compare with treat-all. Deliverable: a table.
Acceptance criteria: values correct; the threshold where the model stops beating treat-all
identified. Time: 45 minutes.

**Exercise 3. Decision curve.** Using predictions and outcomes, plot the decision curve with the
references and mark the operating point. Deliverable: the plot and a 300-word interpretation.
Acceptance criteria: references present; clinical range shaded; crossing points stated. Time: 60
minutes.

**Exercise 4. Capacity reconciliation.** A cohort has 20,000 patients, 8 percent prevalence, and a
capacity of 1,200. At the harm-ratio threshold, 18 percent are flagged. Compute the flagged count,
the capacity gap, and two options. Deliverable: a memo. Acceptance criteria: arithmetic correct;
options described with their sensitivity consequences. Time: 45 minutes.

**Exercise 5. Reversal.** Construct or find a case where a model with a higher AUROC has lower net
benefit at the staffable threshold. Deliverable: a table and a 400-word explanation. Acceptance
criteria: reversal demonstrated with arithmetic; mechanism explained. Time: 60 minutes.

**Exercise 6. Calibration and threshold.** For a model with slope 0.7 and intercept 0.1, compute the
true risk at a nominal threshold of 0.25 and state how the operating point changes. Deliverable:
the arithmetic and a recommendation. Acceptance criteria: recalibrated risk correct; recommendation
addresses the threshold. Time: 45 minutes.

**Exercise 7. Sensitivity analysis.** For a range of harm ratios and prevalences, compute the
preferred strategy and identify the assumptions that change it. Deliverable: a table. Acceptance
criteria: at least three values per assumption; the fragile assumption identified. Time: 60
minutes.

**A note on the exercises.** The arithmetic in Exercises 1, 2, and 4 is the same arithmetic used in
the worked example; a learner who can reproduce those three can reproduce the threshold section of
any model report.

**Exercise 8. Threshold section.** Write the threshold section of a model development report.
Deliverable: two pages. Acceptance criteria: harm ratio, capacity, curve, operating point,
calibration check, and sensitivity analysis all present. Time: 60 minutes.

## 13. Self-check answers

**Answer to Exercise 1.** With `B = 8,000`, `C = 500`: threshold = `500 / 8,500 = 0.0588`, odds =
`500 / 8,000 = 1:16`. With `C = 3,000`: threshold = `3,000 / 11,000 = 0.2727`, odds = `3,000 /
8,000 = 1:2.67`. The higher false positive cost raises the threshold and reduces the number of
false positives the decision-maker is willing to accept per true positive.

**Answer to Exercise 2.** At 10 percent prevalence: NB(0.10) = `0.70 * 0.10 - 0.20 * 0.90 * (0.10 /
0.90) = 0.070 - 0.020 = 0.050`. NB(0.20) = `0.070 - 0.180 * 0.25 = 0.025`. NB(0.30) = `0.070 -
0.180 * 0.4286 = -0.0071`. Treat-all is `0.10 - 0.90 * odds`. At 0.10, treat-all = `0.10 - 0.90 *
0.1111 = 0.0`; at 0.20, `0.10 - 0.225 = -0.125`; at 0.30, `0.10 - 0.3857 = -0.2857`. The model beats
treat-all at all three, but its own net benefit turns negative at 0.30, so it is worse than doing
nothing there.

**Answer to Exercise 4.** At 8 percent prevalence, 18 percent flagged = 3,600 patients. Capacity is
1,200, so the gap is 2,400 patients, or 200 percent of capacity. Option one is to raise the
threshold until 1,200 are flagged, which will reduce sensitivity and may raise PPV. Option two is
to expand capacity by 2,400 evaluations, which requires additional staff. The recommendation should
state the sensitivity loss of option one and the cost of option two.

**Answer to Exercise 6.** `logit(0.25) = -1.0986`. Recalibrated logit = `0.1 + 0.7 * (-1.0986) =
-0.6690`. Recalibrated risk = `1 / (1 + exp(0.6690)) = 0.339`. A nominal 25 percent threshold
corresponds to a true risk of about 34 percent, so the programme is acting on patients well above
its intended risk. The threshold should be lowered on the nominal scale, or the model recalibrated
and the curve recomputed.

**Answer to Exercise 3 (what a strong curve shows).** A strong curve has net benefit on the y-axis,
threshold on the x-axis, the model curve, treat-all, treat-none, a shaded clinical range, and the
operating point marked with its sensitivity, specificity, PPV, and flagged fraction. It states the
prevalence and the capacity, and it reports the thresholds where the model crosses each reference.
A curve without the references or the operating point is a line, not a decision analysis.

**Answer to Exercise 5 (what a strong reversal example shows).** A strong example reports both
models' AUROCs, their sensitivity and specificity at the staffable threshold, and their net benefit
at that threshold. It shows that the higher-AUROC model has lower net benefit and explains why: the
AUROC integrates over thresholds where the programme does not operate. It also checks whether
calibration explains the reversal, because a miscalibrated model can appear worse than it is.

**Answer to Exercise 7 (what a strong sensitivity analysis shows).** A strong analysis varies the
harm ratio, the prevalence, and the capacity one at a time and reports the preferred strategy at
each value. It identifies the assumption to which the decision is most sensitive, and it states
whether the preferred strategy changes within the plausible range. If the decision is stable across
the range, that stability is the finding; if it flips, the flip is the finding.

**Rubric for Exercises 3, 4, 5, 7, and 8.** A strong submission (1) states the harm ratio and
capacity, (2) plots the references, (3) marks the operating point, (4) reconciles capacity, (5)
checks calibration, and (6) reports the sensitivity analysis. A competent submission meets four of
the six. A weak submission uses 0.5 by default or reports net benefit without a threshold.

## 14. Interview questions

**1. How do you choose a threshold?** A strong answer derives `C/(B+C)`, checks capacity, and
compares net benefit. A weak answer says 0.5 or the Youden index.

**2. What is net benefit?** A strong answer gives the formula and the interpretation as net true
positives per patient. A weak answer confuses it with profit or accuracy.

**3. Why can a model with a higher AUROC have lower net benefit?** A strong answer explains that
AUROC integrates over thresholds while net benefit weights errors at one threshold. A weak answer
says it cannot happen.

**4. What is the treat-all reference for?** A strong answer says it is the default strategy and the
model must beat it to be useful. A weak answer ignores it.

**5. How does capacity affect the threshold?** A strong answer computes the flagged fraction and
raises the threshold if it exceeds capacity. A weak answer says capacity is an implementation
detail.

**6. Why is the threshold a preference?** A strong answer explains the harm ratio and the odds
interpretation. A weak answer says the threshold is a statistical estimate.

**7. What is the test tradeoff?** A strong answer defines it as the number of additional tests per
additional true positive. A weak answer has not heard of it.

**8. Why does calibration matter for a threshold?** A strong answer says the threshold is a risk,
and a risk is only meaningful on a calibrated scale. A weak answer says calibration is separate.

**9. Should DCA be used to choose a threshold?** A strong answer says no, it should be used to
compare strategies across a range of preferences, with the threshold pre-specified. A weak answer
says yes, choose the maximum.

**10. What is the difference between the harm-ratio threshold and the capacity threshold?** A strong answer says the first encodes values and the second encodes resources, and that the report should state which one bound. A weak answer uses them interchangeably.

**11. When does a decision curve reverse an AUROC ranking?** A strong answer explains that the models have different error profiles across thresholds and the decision operates in a region where the lower-AUROC model has better net benefit. A weak answer says reversal is impossible.

**12. What is the difference between net benefit and a cost-effectiveness analysis?** A strong answer says net benefit uses a single harm ratio and counts net true positives, while cost-effectiveness uses costs and health outcomes and produces a ratio. A weak answer treats them as the same.

**13. Why might a threshold be set by a vendor rather than by the clinic?** A strong answer says a vendor may have derived it in a different population or for a different intervention, and the clinic should re-derive it locally. A weak answer accepts the vendor threshold.

**14. How would you report a capacity-constrained threshold?** A strong answer reports the
constraint, the threshold, the sensitivity loss, and the net benefit. A weak answer reports the
threshold without saying it was capacity-driven.

## 15. Key terms

- **Alert burden** — alerts per unit of clinical work.
- **Benefit of a true positive** — net value of catching a case.
- **Capacity** — actions deliverable per period.
- **Capacity-constrained threshold** — threshold that fits the deliverable number.
- **Clinical range** — thresholds a reasonable clinician would consider.
- **Cost of a false positive** — net cost of flagging a non-case.
- **Decision curve** — net benefit across thresholds.
- **Decision curve analysis** — the method of computing and plotting net benefit.
- **Decision rule** — threshold plus action.
- **Default strategy** — treat-all or treat-none.
- **Expected utility** — probability-weighted value of a strategy.
- **False positive rate** — FP / (FP + TN).
- **Flagged fraction** — proportion above the threshold.
- **Harm ratio** — cost of a false positive relative to benefit of a true positive.
- **Net benefit** — TPR-weighted prevalence minus FPR-weighted odds.
- **Net harm** — negative net benefit.
- **Net true positives** — net benefit times cohort size.
- **Number needed to evaluate** — patients reviewed per true case.
- **Number needed to treat** — patients treated for one benefit.
- **Odds of the threshold** — `t / (1 - t)`.
- **Operating point** — threshold and its confusion matrix.
- **Opportunity cost** — value forgone when capacity is used elsewhere.
- **Post-hoc threshold** — threshold chosen after seeing the curve.
- **Pre-specified threshold** — threshold fixed before analysis.
- **Prevalence** — proportion of positives.
- **Quality-adjusted life year** — utility combining length and quality of life.
- **Rule-in** — a decision that confirms and treats.
- **Rule-out** — a decision that avoids a test or admission.
- **Sensitivity** — TPR at the threshold.
- **Specificity** — 1 minus FPR at the threshold.
- **Standardised net benefit** — net benefit divided by prevalence.
- **Test tradeoff** — additional tests per additional true positive.
- **Threshold** — risk above which the action is taken.
- **Threshold drift** — change in the optimal threshold over time.
- **Treat-all** — acting on everyone.
- **Treat-none** — acting on no one.

## 16. Sources

All URLs accessed 25 October 2026.

1. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models.
   *Medical Decision Making* 2006;26(6):565-574. https://doi.org/10.1177/0272989X06295361
2. Vickers AJ, van Calster B, Steyerberg EW. A simple, step-by-step guide to interpreting decision
   curve analysis. *Diagnostic and Prognostic Research* 2019;3:18.
   https://doi.org/10.1186/s41512-019-0064-7
3. Van Calster B, Wynants L, Verbeek JFM, et al. Reporting and interpreting decision curve
   analysis: a guide for investigators. *European Urology* 2018;74(6):796-804.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC6261531/
4. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
5. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
   https://doi.org/10.1136/bmj-2024-082505
6. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive
   analytics. *BMC Medicine* 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
7. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
   sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
   https://doi.org/10.1001/jamainternmed.2021.2626
8. Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission.
   *JAMA* 2011;306(15):1688-1698. https://doi.org/10.1001/jama.2011.1515
9. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
   Guiding Principles. 2021.
   https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
10. US Food and Drug Administration. Marketing Submission Recommendations for a Predetermined Change
    Control Plan for Artificial Intelligence-Enabled Device Software Functions. 2024.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
11. European Union. Regulation (EU) 2024/1689 (Artificial Intelligence Act). 2024.
    https://eur-lex.europa.eu/eli/reg/2024/1689/oj
12. National Institute for Health and Care Excellence. Evidence standards framework for digital
    health technologies. https://www.nice.org.uk/about/what-we-do/our-programmes/evidence-standards-framework-for-digital-health-technologies
13. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
14. Steyerberg EW. *Clinical Prediction Models*. 2nd edition. Springer, 2019.
    https://doi.org/10.1007/978-3-030-16399-0
15. Vickers AJ, Van Calster B, Steyerberg EW. Net benefit approaches to the evaluation of prediction
    models, molecular markers, and diagnostic tests. *BMJ* 2016;352:i6.
    https://doi.org/10.1136/bmj.i6
16. decisioncurveanalysis.org. Tutorials and software.
    https://www.decisioncurveanalysis.org/
17. scikit-learn developers. Probability calibration and decision thresholds.
    https://scikit-learn.org/stable/modules/calibration.html
18. Hunink MGM, Weinstein MC, Wittenberg E, et al. *Decision Making in Health and Medicine*.
    Cambridge University Press, 2014.

The threshold is a decision, and every decision has an owner. Name the owner, the review
date, and the evidence that would trigger a revision, so that the threshold is governed rather
than inherited.

The next topic examines how to explain the model that produces the risk.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_calibration.md`](08_calibration.md)
- Next topic: [`10_interpretability.md`](10_interpretability.md)
- Metrics: [`07_metrics.md`](07_metrics.md)
- Calibration: [`08_calibration.md`](08_calibration.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../course/06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)


