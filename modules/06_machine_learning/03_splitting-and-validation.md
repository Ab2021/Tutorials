# Splitting and Validation

> **Module 06 — Machine Learning Foundations**
> Topic 03 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 01 and 02
> Version 2026.3

## 1. Why this topic exists

Every performance number a model reports is an estimate of how it will behave on data it has not
seen. The estimate is produced by a validation design, and the design determines whether the
number is credible. A random split of a dataset with repeated patient admissions produces a
number that is inflated by memorisation. A temporal split without a gap produces a number that is
inflated by overlap between the outcome window and the training period. A test set used for
hyperparameter tuning produces a number that is optimised rather than estimated. In each case the
model has not changed; the estimate has.

The gap between apparent and honest performance is not small. In a dataset where the true error
rate is 50 percent, Varma and Simon (2006) showed that cross-validation used for both tuning and
evaluation produced minimum error estimates averaging 37.8 percent for shrunken centroids and 41.7
percent for support vector machines, and on more than one-fifth of datasets the bias exceeded 20
percentage points. Nested cross-validation recovered an estimate close to the truth (54.2 percent
against 50.0 percent). The mechanism is simple: taking the minimum over many noisy estimates is
biased downward, and the more hyperparameters are tried, the larger the bias. A team that tunes and
reports on the same folds is not measuring the model; it is measuring the best of many attempts.

Biomedical data add three complications that make naive splitting especially dangerous.

**Repeated measures.** A patient with four admissions contributes four rows. A random split puts
some of those rows in training and some in test, so the model can learn the patient's identity and
copy the label. The inflation is largest for flexible models and for features that are stable
within a patient. Grouped splitting keeps all rows for a patient in one fold.

**Temporal structure.** Clinical data are generated in time, and deployment is always later than
development. A random split mixes future and past, so the model can learn from information that
would not exist at the deployment moment. A temporal split trains on earlier data and tests on
later data, which matches the deployment claim, but it introduces drift and reduces the effective
sample size.

**Site heterogeneity.** A model developed at one hospital may not transport to another because of
case mix, coding practice, equipment, and workflow. A random split within one site cannot detect
this. A site-held-out split can, but it requires at least two sites with enough events each.

A further complication is that the deployment population is rarely fixed. A model launched in
January may face a different case mix by December because of a new referral pathway, a change in
insurance coverage, or a seasonal infection. The validation design cannot test every future
population, but it can test the axes along which the population is most likely to move, and it can
specify the monitoring that would detect movement outside the validated range.

The validation design is therefore not a technical afterthought. It is the argument that connects
the model's measured performance to the claim the team wants to make. A claim of internal
discrimination needs internal validation. A claim of transportability needs external validation. A
claim that the model will improve outcomes needs a prospective study, because no split of
retrospective data can estimate the effect of a workflow change.

The three complications also interact with sample size. A dataset with 20,000 rows and 2,000
patients has less independent information than its row count suggests. A temporal split may leave
only two years of usable test data. A site-held-out design with three sites leaves each site with
one-third less training data. The validation design consumes data, and the sample-size calculation
must include that consumption rather than assuming the full dataset is available for both
development and evaluation.

The cost of getting this wrong is borne later. A model validated with a leaky split passes internal
review, is deployed, and then underperforms. The team loses credibility, the deployment is paused,
and the revalidation takes months. The same team could have spent two days designing the split
correctly. This topic is about spending those two days well.

Three misconceptions are worth naming at the outset.

**"Cross-validation is validation."** Cross-validation is a resampling scheme. Whether it estimates
generalisation depends on what is grouped, what is stratified, what is inside the fold, and whether
the test folds were used for tuning. A five-fold cross-validation with patient leakage is worse
than a single honest holdout, because it looks more rigorous.

**"More folds are better."** More folds mean each training set is closer to the full data, which
reduces the pessimistic bias of training on less data. But more folds also increase the variance of
the estimate and the compute cost, and they do not fix leakage. Ten folds on a dataset with 80
events produce folds with 8 events each and unstable metrics.

**"External validation is a separate project."** External validation is a design choice that can
be built into a multi-site development study through internal-external validation. Waiting until
after publication to test transport is how models acquire a reputation they cannot keep.

> **Note:** A validation result is a property of a model-and-design pair, not of a model alone.
> Reporting an AUROC without the split design is like reporting a drug's effect without the trial
> design.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** a deployment claim to a validation design (internal, temporal, site-held-out, or
   prospective) and **justify** the choice.
2. **Identify** the unit of independence in a dataset and **choose** a grouping variable that
   prevents patient-level leakage.
3. **Implement** grouped, stratified, temporal, and nested cross-validation with correct placement
   of preprocessing.
4. **Compute** optimism as the difference between apparent and validated performance, and
   **explain** why it grows with model flexibility and tuning effort.
5. **Detect** contamination from duplicates, overlapping outcome windows, and repeated measures.
6. **Report** performance with fold-level variability and confidence intervals, and **distinguish**
   the two.
7. **Design** an internal-external validation that leaves one site or one period out.
8. **Critique** a published validation study for split design, tuning contamination, and claim
   mismatch.

Outcomes 3, 4, and 6 are computational. Outcomes 1 and 7 are design outcomes assessed by rubric.

The sequence is deliberate: a learner should be able to state the claim and the unit of
independence before writing code. Code that splits rows before the unit is known is not a
validation; it is a random partition with a confidence interval attached. The exercises therefore
put design before implementation, and the implementation exercise assumes the design has already
been justified.

## 3. The core mental model

Generalisation has three axes: new patients, new times, and new sites. A validation design is a
choice about which axes to perturb. The axes are not independent, but separating them makes the
claim explicit.

```text
                    same patients        new patients
same time,          apparent             internal
same site           performance          validation
                    (optimistic)         (random or grouped split)

new time,           temporal             temporal
same site           leakage risk         validation
                    (if split ignores    (train past, test future)
                     time)

new site,           site leakage         external
any time            (if site features    validation
                     are in the model)   (leave-one-site-out)
```

The vertical axis is time; the horizontal axis is patients and sites. The top-left cell is the
number a model reports on its training data, and it is almost always the largest. The bottom-right
cell is the number that supports a deployment claim at a new site, and it is almost always the
smallest. The job of validation is to move from the top-left to the cell that matches the claim,
without pretending that a cheaper design supports a more expensive claim.

Inside any cell, there are two loops.

```text
Outer loop: estimate generalisation performance
  Split data into K outer folds (grouped or temporal)
  For each outer fold:
      Inner loop: tune hyperparameters
        Split the outer training set into K inner folds
        Fit each candidate on inner training, evaluate on inner validation
        Choose the candidate with the best mean inner score
      Fit the chosen candidate on the full outer training set
      Evaluate once on the outer test fold
  Report the mean and spread of the outer scores
```

The outer loop answers "how well will this procedure do on new data?" The inner loop answers
"which hyperparameters should the procedure use?" The two questions must be answered on different
data, or the answer to the first is contaminated by the answer to the second. This is the nested
cross-validation principle, and it is the single most important structural idea in this topic.

A useful way to remember the design is to ask three questions in order.

1. What is the claim? (internal discrimination, transport to a new site, future performance)
2. What is exchangeable with what? (patients with patients, times with times, sites with sites)
3. What is the unit that must not cross the split? (patient, admission, clinician, site, time
   block)

The answers determine the split. If the team cannot answer question 3, the split is probably wrong,
because most clinical datasets have a unit of dependence that is not the row.

Choosing the axes is a judgement about which differences matter for the claim. A model that will
be used at the same hospital next year needs the time axis, not the site axis. A model that will be
sold to other hospitals needs the site axis, and probably both. A model that will be used for a new
patient population at the same hospital needs the patient axis and a case-mix comparison. The axes
can be combined: a leave-one-site-out design with a temporal gap tests both transport and time, at
the cost of fewer training records per fold.

The axes also interact with label quality. If coding practice changes over time, a temporal
validation confounds model drift with label drift. If sites differ in testing rates, a site-held-out
validation confounds model transport with label transport. That is why the validation report should
pair every design with a label audit (Topic 02) and a calibration check. A performance drop across
a design is not automatically a model problem.

> **Note:** Exchangeability is an assumption, not a fact. A random split assumes patients are
> exchangeable; a temporal split assumes the future resembles the past; a site-held-out split
> assumes the held-out site is drawn from the same population of sites. Each assumption can fail,
> and the validation report should say which one it relies on.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Generalisation | Performance on data not used for development | Confused with apparent performance | Estimate with interval |
| Apparent performance | Performance measured on the development data | Reported as validation | Metric on training set |
| Optimism | Difference between apparent and validated performance | Ignored in model selection | Metric difference |
| Internal validation | Validation using data from the development setting | Called external if a random split | Design label |
| External validation | Validation in a different setting, time, or population | Called internal if same site | Design label |
| Temporal validation | Train on earlier data, test on later data | No gap between periods | Design label |
| Geographical validation | Train at some sites, test at others | Random split within site | Design label |
| Internal-external validation | Leave-one-site-out within a multi-site development study | Confused with external | Design label |
| Holdout set | A portion of data reserved for final evaluation | Used repeatedly for tuning | Proportion |
| k-fold cross-validation | Partition into k folds; train on k-1, test on 1 | k=5 with tiny data | k integer |
| Stratified k-fold | Folds preserve outcome prevalence | Used when groups are small | k integer |
| Grouped k-fold | Folds preserve group membership | Grouping by row instead of patient | Group identifier |
| Nested cross-validation | Inner tuning, outer evaluation | Reporting inner scores as performance | Two loops |
| Leave-one-group-out | Each group is a test fold | Used with many groups | Group identifier |
| Leave-one-site-out | Each site is a test fold | Used with too few sites | Site identifier |
| Bootstrap validation | Resampling with replacement to estimate optimism | Confused with confidence interval | Resamples |
| .632+ bootstrap | Bias-corrected bootstrap estimator | Used without checking | Estimator |
| Repeated cross-validation | Multiple random partitions | Reported as independent validation | Repeats |
| Hyperparameter | A model setting chosen before fitting | Tuned on the test set | Model configuration |
| Inner loop | Tuning loop inside each outer fold | Confused with outer loop | Procedure step |
| Outer loop | Evaluation loop | Used for tuning | Procedure step |
| Test set contamination | Test data influencing model selection | Called "exploratory" | Design flaw |
| Data leakage | Information from outside the training fold entering the model | Confused with overfitting | Design flaw |
| Unit of independence | The entity that must not cross folds | Assumed to be the row | Patient, admission, site |
| Cluster | A group of correlated observations | Ignored in random splits | Group identifier |
| Intraclass correlation | Proportion of variance between clusters | Ignored in sample size | 0 to 1 |
| Effective sample size | Sample size adjusted for clustering | Equated with row count | Count |
| Prevalence shift | Change in outcome prevalence across folds or sites | Mistaken for model drift | Proportion |
| Covariate shift | Change in feature distribution | Confused with concept shift | Distribution |
| Concept shift | Change in the feature-outcome relationship | Confused with covariate shift | Relationship |
| Transportability | Whether a model performs in a new setting | Assumed from internal validation | Claim |
| Calibration slope | Agreement of predicted and observed risk on the logit scale | Reported without intercept | Slope |
| Calibration intercept | Systematic over- or under-prediction | Reported without slope | Log-odds |
| Decision curve | Net benefit across thresholds | Read as discrimination | Curve |
| Nested CV optimism | Difference between apparent and nested-CV performance | Ignored | Metric difference |
| Events per variable | Number of outcome events per candidate parameter | Used as a sole sample-size rule | Ratio |
| Riley criteria | Sample-size criteria for prediction model development | Confused with validation sample size | Counts |
| Purging | Removing training observations too close in time to test observations | Omitted in temporal splits | Gap length |
| Embargo | A gap between training and test periods | Confused with purging | Days |
| Fold-level variability | Spread of performance across folds | Reported as confidence interval | Standard deviation |
| Confidence interval | Uncertainty in the mean performance estimate | Confused with fold spread | Interval |
| Seed sensitivity | Change in results across random seeds | Ignored | Repeats |
| Preprocessing leakage | Fitting imputers or scalers outside the fold | Common and silent | Design flaw |
| Target leakage | Using the outcome to create a feature | Confused with target encoding | Design flaw |
| Duplicate record | The same encounter entered twice | Inflates test performance | Count |
| Boundary overlap | Outcome window of a training patient overlaps the test period | Common in temporal splits | Days |
| Group-aware balancing | Folds balanced by record count rather than group count | Large groups dominate a fold | Records per fold |
| Nested grouping | Patients within sites, admissions within patients | Wrong level chosen | Nested identifiers |

Two definitions deserve more than a table row.

**Fold-level variability versus confidence interval.** The standard deviation of performance across
folds describes how much the estimate changes with the particular partition. It is not a confidence
interval for performance on the population. With five folds, the standard deviation is itself
estimated from five numbers and is unstable. Report both: the fold spread shows partition
sensitivity, and a bootstrap or analytic interval shows sampling uncertainty. Conflating them
produces intervals that are too narrow or too wide.

**Internal versus external validation.** Internal validation uses data from the same underlying
population and data-generating process, even if the split is by patient or by time within the same
site. External validation uses data from a different population, setting, or era, where the
differences are not merely sampling variation. A random split at one hospital is internal. A
temporal split at one hospital is often called temporal validation and sits between internal and
external: it tests transport across time but not across sites. A site-held-out split is external
for the held-out site. Steyerberg and Harrell (2016) argue for "internal-external" validation:
leave one centre or one period out during development, which gives an external flavour while
keeping the model fitted on all data at the end.

> **Note:** The name of the design is less important than the claim it supports. A "temporal
> validation" that trains on 2019-2022 and tests on 2023, with a 30-day outcome window and no gap,
> leaks 30 days of outcome information across the boundary. State the gap or do not call it
> temporal.

## 5. The mechanics, step by step

Validation design is a fourteen-step procedure. Steps 1 through 4 are design decisions that must be
made before looking at performance. Steps 5 through 8 construct the folds. Steps 9 through 12 run
the procedure. Steps 13 and 14 audit and document it.

### Step 1. Write the claim the validation must support

**Input:** the prediction task specification from Topic 01.

**Output:** one claim sentence with a setting, a population, a time frame, and a performance
quantity. Example: "The model discriminates 30-day readmission in adults discharged from hospitals
A and B, for discharges in 2023, with AUROC at least 0.65 and calibration slope at least 0.80."

**Rationale:** The claim determines the design. A claim about the development hospitals needs
internal validation; a claim about a new hospital needs external validation; a claim about next
year needs temporal validation. Without a claim, any design can be defended, and the weakest design
usually wins because it is cheapest.

**Branch point:** if the claim includes an outcome benefit ("reduces readmissions"), stop. No split
of retrospective data can support it. The design must be prospective, and the validation question
changes from "does the model discriminate?" to "does the model change practice and outcomes?"

### Step 2. Identify the unit of independence

**Input:** the data dictionary and the cohort construction.

**Output:** the grouping variable that defines independent units, plus the number of units and the
distribution of records per unit.

**Rationale:** Rows are rarely independent. A patient may have several admissions, a clinician may
contribute several patients, a site may contribute thousands of rows, and a family may contribute
several patients. The unit of independence is the highest level at which observations are
exchangeable. If the unit is the patient and the split is by row, the same patient appears in
training and test, and the model can memorise patient-specific patterns.

The unit also determines the effective sample size. If 4,000 admissions belong to 1,000 patients,
the effective sample size for a patient-level claim is closer to 1,000 than 4,000, and the standard
error of the performance estimate should reflect that. A validation report that counts rows rather
than units will overstate precision. The intraclass correlation quantifies how much of the variance
is between units; when it is high, the effective sample size is much smaller than the row count.

**Branch point:** if there are multiple grouping levels (patients within sites, admissions within
patients), choose the level that matches the deployment claim. For a claim about new patients at
existing sites, group by patient. For a claim about new sites, group by site. For a claim about
both, use a nested or internal-external design.

### Step 3. Identify the axes of shift

**Input:** the deployment setting and the development data.

**Output:** a list of axes on which the deployment differs from development, with evidence for each.

**Rationale:** Each axis of shift needs a validation design that perturbs it. If deployment is
later in time, the design must test across time. If deployment includes a new site, the design must
test across sites. If the population case mix differs, the design must test across case mix, which
usually requires either stratification or external data.

**Branch point:** if an axis cannot be tested with the available data, the claim must be weakened
to exclude it. A model developed at one site cannot claim transportability to another site without
data from that site, no matter how good the internal validation is.

### Step 4. Choose the validation design

**Input:** the claim, the unit of independence, and the axes of shift.

**Output:** a named design, a justification, and a statement of what the design cannot test.

**Rationale:** The design is the argument. Naming it makes the argument auditable. The table below
summarises the common designs and their claims.

| Design | Claim supported | Requirement | Cannot test |
|---|---|---|---|
| Apparent (no split) | None; descriptive only | None | Generalisation of any kind |
| Random holdout | Internal discrimination in similar patients | Enough data for a stable holdout | Patient leakage if rows repeat; temporal or site shift |
| Grouped k-fold | Internal discrimination in new patients | Patient or cluster identifier | Temporal or site shift |
| Stratified grouped k-fold | Internal discrimination with stable prevalence | Group and outcome identifiers | Temporal or site shift |
| Temporal holdout | Discrimination in a later period | Date field and enough later data | Site shift; needs a gap |
| Site-held-out | Transport to unseen sites | At least three sites with adequate events | Performance at a site unlike any included |
| Internal-external (leave-one-site-out) | Transport across the included sites | Multi-site development data | A genuinely new site type |
| Nested cross-validation | Honest performance of a tuning procedure | Enough data for inner and outer loops | External validity |
| Prospective silent trial | Performance in live workflow without exposure | Deployment infrastructure | Outcome benefit |
| Prospective comparative study | Outcome benefit | Randomisation or strong quasi-experiment | Nothing less than a trial supports this |

**Branch point:** if the claim requires a design the data cannot support, the correct response is
to narrow the claim or collect more data, not to relabel a weaker design. Relabelling is common
because it costs nothing at the time and everything later. A random split called
"external" is still a random split.

### Step 5. Choose the number of folds and the split ratio

**Input:** the sample size, the outcome prevalence, and the compute budget.

**Output:** k, the number of repeats, and the holdout fraction if a holdout is used.

**Rationale:** Five or ten folds are conventional because they balance bias and variance. With few
events, more folds increase the training size per fold but also increase the variance of the
fold-level estimates. With very large data, a single holdout may be sufficient. The split ratio
should ensure that the test set has enough events to estimate the metric with useful precision; a
test set with 20 events cannot support a stable AUROC.

Repeated cross-validation runs the whole partition several times with different seeds and averages
the results. It reduces the dependence of the estimate on a particular partition, at the cost of
more compute. For a model comparison where the difference between candidates is small, repeats are
more valuable than a larger k, because they separate the model difference from partition noise.
Report the number of repeats and the seed.

**Branch point:** if the outcome is rare (fewer than about 100 events in total), use repeated
stratified k-fold and report the variability across repeats. A single 80/20 split will produce a
test set with too few events to be informative.

### Step 6. Stratify where prevalence could vary

**Input:** the outcome and the grouping variable.

**Output:** a stratification rule and a check that each fold has both classes.

**Rationale:** Random fold assignment can produce folds with very different prevalence, especially
when the outcome is rare or when groups are heterogeneous. Stratification by outcome (and by site,
if sites are not the grouping variable) keeps the folds comparable and reduces the variance of the
performance estimate. It does not fix a confounding problem; it only balances the folds.

**Branch point:** if a fold has only one class, AUROC is undefined for that fold. Merge the fold
with a neighbour or use a stratified split. Report the merge, because it changes the interpretation
of the fold-level spread.

### Step 7. Build temporal splits with a gap

**Input:** the timestamp of time zero for each record and the outcome window.

**Output:** a training period, a gap, and a test period, with counts on both sides.

**Rationale:** A temporal split must respect the outcome window. If training ends on 31 December
2022 and testing begins on 1 January 2023, a patient discharged on 20 December 2022 may have an
outcome that occurs in January 2023, which is inside the test period. The label for that training
patient is partly determined by test-period information. A gap of at least the outcome window
prevents this overlap. The gap is sometimes called purging or embargo.

**Branch point:** if the gap removes too much data, shorten the outcome window, extend the data
range, or accept the overlap and quantify it. Do not silently omit the gap.

### Step 8. Build site-held-out or internal-external splits

**Input:** site identifiers and event counts by site.

**Output:** one fold per site (or per site-period), with the model trained on the others.

**Rationale:** A site-held-out design estimates transport to a site that was not used for
development. It is the strongest retrospective design for a multi-site deployment claim. Leaving
one site out at a time, and fitting the final model on all sites, is the internal-external design
recommended by Steyerberg and Harrell. It uses all data for the final model while still producing a
transportability estimate.

**Branch point:** if a site has too few events (fewer than about 50), its held-out estimate will be
too imprecise to interpret. Pool small sites with similar sites, or report the site descriptively
and exclude it from the transportability claim.

### Step 9. Place every preprocessing step inside the fold

**Input:** the preprocessing pipeline.

**Output:** a pipeline object that is fitted separately in each training fold.

**Rationale:** Imputation, scaling, encoding, feature selection, and target encoding all learn
parameters from data. If they are fitted on the full dataset before splitting, information from the
test fold enters the training process. The leakage is often small per step but accumulates, and it
is largest for target encoding and feature selection. The entire pipeline must be fitted inside the
training fold and applied unchanged to the test fold.

Feature selection is the most commonly missed preprocessing step. If the team selects the top 20
features by univariate association on the full dataset and then cross-validates, the selection has
already seen the test folds. The resulting estimate is optimistic, sometimes substantially so,
because the selected features include those that happen to correlate with the outcome in the test
folds. Feature selection must be inside the pipeline, fitted on each training fold, even when the
selected set is unstable across folds. The instability is itself useful information: if the
selected features change completely from fold to fold, the model is not identifying a stable
signal.

**Branch point:** if a preprocessing step requires the outcome (for example, target encoding), it
must be fitted only on the training fold, with smoothing and a fallback for unseen categories. If
the step cannot be implemented without the full data, it cannot be used in a validated pipeline.

### Step 10. Tune in the inner loop

**Input:** the outer training fold and a grid or distribution of hyperparameters.

**Output:** the selected hyperparameter set for that outer fold, and the inner-loop score.

**Rationale:** The inner loop estimates which hyperparameter setting generalises best within the
outer training data. The inner score is not the performance estimate; it is a selection criterion.
Reporting the inner score as if it were validation performance reintroduces the optimism that
nested cross-validation is designed to remove.

**Branch point:** if the hyperparameter grid is large relative to the number of events, the inner
loop itself will overfit the selection criterion. Reduce the grid, use a regularised model, or
increase the data. The number of configurations tried is part of the optimism budget.

### Step 11. Evaluate in the outer loop

**Input:** the model fitted on the full outer training fold with the selected hyperparameters.

**Output:** one performance estimate per outer fold, plus the pooled estimate and the fold-level
spread.

**Rationale:** The outer fold is touched once, after all tuning is complete. This is what makes the
estimate honest. Pooling the outer predictions and computing the metric on the pooled set is
usually preferable to averaging fold-level metrics, because it weights folds by size and avoids
undefined metrics in small folds. Report both if the folds differ in size.

**Branch point:** if the pooled and average estimates disagree materially, the folds are
heterogeneous. Report the disagreement and investigate whether a site, era, or subgroup is driving
it.

### Step 12. Compute optimism and compare designs

**Input:** apparent performance and validated performance.

**Output:** optimism for each metric, with an interpretation.

**Rationale:** Optimism is the difference between the performance on the development data and the
performance under the validation design. It is a diagnostic of overfitting and of the flexibility
of the modelling procedure. A large optimism relative to the performance difference between models
means the comparison is unreliable. Varma and Simon's result shows that optimism can exceed 20
percentage points when tuning and evaluation share folds.

**Branch point:** if optimism is large (for example, apparent AUROC exceeds validated AUROC by more
than 0.05), the model is likely overfitting or the validation is contaminated. Investigate before
comparing models.

### Step 13. Audit for contamination

**Input:** the split assignments and the feature pipeline.

**Output:** a contamination report covering duplicates, patient overlap, temporal overlap,
preprocessing fit scope, and target leakage.

**Rationale:** Most validation failures are contamination failures, and most contamination is
invisible in the performance table. The audit should be a checklist with counts: duplicate records,
patients in more than one fold, test-period outcomes overlapping training, preprocessing fitted
outside the fold, and features whose timestamps exceed time zero.

The audit should be run before the performance table is reviewed, not after. If the team sees the
numbers first, the audit becomes a search for an explanation of a result rather than a check on
whether the result is admissible. A simple discipline is to require the audit output to be attached
to the results file, with the counts and the seed, before anyone is allowed to discuss the model
comparison.

**Branch point:** if any contamination is found, fix it and rerun the entire procedure. Do not
adjust the performance estimate by an assumed inflation factor; the correct response is to
recompute.

### Step 14. Freeze and document the design

**Input:** the validated design and the results.

**Output:** a validation protocol section in the model development report, with the split
definition, fold assignments (or a seed and script), and the claim.

**Rationale:** A validation result that cannot be reproduced is not evidence. Documenting the
design includes the random seed, the grouping variable, the date boundaries, the gap, the fold
count, and the preprocessing scope. TRIPOD+AI requires the data sources, the sample size, and the
performance evaluation to be reported; the validation design is the link between them.

The protocol should be stored with the code, not only in a document. A validation section that
describes the design but cannot be executed from the repository is a narrative, not a
specification.

**Branch point:** if the design changes after seeing results, record the change and the reason. A
design change made after seeing results is a new analysis, not a correction.

### Validation edge cases

**Empty folds.** A site-held-out fold with no test records produces no estimate. This happens when
a site has records but all are excluded by the eligibility criteria. Report the empty fold and
exclude the site from the transportability claim.

**Folds with one class.** A fold with no events or no non-events has an undefined AUROC. Stratify,
merge, or report the metric on the pooled predictions only.

**Tiny sites.** A site with 12 events cannot support a site-specific AUROC with a useful interval.
Report the count and pool the site with a similar site.

**Groups larger than a fold.** If one patient contributes more records than the intended fold size,
grouped splitting produces uneven folds. Use group-aware assignment that balances record counts, or
analyse at the patient level.

**Perfect separation in a fold.** A fold where a feature perfectly separates the outcome often
indicates leakage or a small-sample artefact. Check the feature's timing and whether it is
constant within the fold's positives.

**Prevalence shift across folds.** If prevalence varies substantially by fold, the pooled AUROC may
hide heterogeneity. Report prevalence per fold and calibrate within folds where needed.

**Ties in temporal timestamps.** Many records share a date. A temporal split at a date boundary can
put records from the same day on both sides. Choose the boundary at a timestamp with no ties, or
group all records from the boundary day into one side.

**Patients crossing sites.** A patient treated at two sites belongs to both. Group by patient, not
by site, or assign the patient to the site of first contact and record the decision.

## 6. Worked example

### The setting and the claim

A regional heart failure programme wants to deploy a readmission model across three hospitals. The
claim is: "The model discriminates 30-day unplanned readmission in adults discharged with heart
failure at hospitals A, B, and C, for discharges from 2023 onward, with AUROC at least 0.65 and
calibration slope at least 0.80."

The data cover 2019-2023. Hospitals A and B contribute development data from 2019-2022 (24,800
admissions, 4,712 events, prevalence 19.0 percent). Hospital C contributes 2023 data only (3,400
admissions, 782 events, prevalence 23.0 percent). Hospitals A and B also contribute 2023 data
(6,100 admissions, 1,318 events, prevalence 21.6 percent), which supports a temporal validation.
Eighteen percent of patients have more than one admission; 4,100 records belong to repeat patients.

### The design

The team chooses three evaluations:

1. **Grouped five-fold cross-validation** on the 2019-2022 development data, grouping by patient.
   This estimates internal discrimination for new patients at hospitals A and B.
2. **Temporal validation** on 2023 data from A and B, with a 30-day gap after the end of the
   training period. This estimates discrimination in a later period at familiar sites.
3. **External validation** on hospital C's 2023 data. This estimates transport to a new site.

The preprocessing pipeline (median imputation, standardisation, one-hot encoding) is fitted inside
each training fold. Hyperparameters are tuned in an inner three-fold loop over a small grid
(regularisation strength for the logistic model; depth and learning rate for the gradient-boosted
model). The outer loop is touched once per fold.

### The results

| Model | Apparent AUROC | Grouped CV AUROC | Temporal AUROC | External AUROC |
|---|---:|---:|---:|---:|
| Logistic regression | 0.712 | 0.681 (fold SD 0.018) | 0.662 (0.646-0.678) | 0.641 (0.619-0.663) |
| Gradient boosting | 0.741 | 0.694 (fold SD 0.031) | 0.649 (0.633-0.665) | 0.611 (0.589-0.633) |

Optimism for the logistic model is `0.712 - 0.681 = 0.031`. For gradient boosting it is
`0.741 - 0.694 = 0.047`. The more flexible model has more optimism, which is expected: it has more
capacity to fit idiosyncrasies in the development data, and its fold-to-fold spread is larger.

The fold standard deviation (0.018 for logistic regression, 0.031 for gradient boosting) describes
partition sensitivity, not sampling uncertainty. The confidence intervals for the temporal and
external estimates are wider than the fold spread, which is expected: they come from a different
sample and a different data-generating period. Reporting the fold standard deviation as if it were
a confidence interval would understate the uncertainty in the deployment claim.

The ranking changes with the design. Gradient boosting is better under grouped cross-validation
(0.694 versus 0.681) but worse under temporal validation (0.649 versus 0.662) and worse under
external validation (0.611 versus 0.641). The apparent ranking, which favours gradient boosting by
0.029, reverses under the design that matches the deployment claim. A team that reported only the
apparent or cross-validated numbers would choose the model that transports worse.

### Calibration

Calibration slope tells a similar story:

| Model | Grouped CV slope | Temporal slope | External slope |
|---|---:|---:|---:|
| Logistic regression | 0.91 | 0.84 | 0.72 |
| Gradient boosting | 0.86 | 0.71 | 0.58 |

A slope below 1 means the predictions are too extreme: high risks are too high and low risks are
too low. The logistic model degrades from 0.91 to 0.72 across the designs; the gradient-boosted
model degrades from 0.86 to 0.58. At the external site, the gradient-boosted model's predictions
are substantially overdispersed, and its probabilities would need recalibration before use.

### The decision at a threshold

Suppose the programme can enrol patients at a predicted risk of 0.20 or higher. At hospital C, the
logistic model has sensitivity 0.61 and specificity 0.71; the gradient-boosted model has sensitivity
0.66 and specificity 0.60. Prevalence at C is 23.0 percent.

PPV for the logistic model:

```text
0.61 * 0.23 / (0.61 * 0.23 + 0.29 * 0.77)
= 0.1403 / (0.1403 + 0.2233)
= 0.1403 / 0.3636
= 0.386
```

PPV for gradient boosting:

```text
0.66 * 0.23 / (0.66 * 0.23 + 0.40 * 0.77)
= 0.1518 / (0.1518 + 0.3080)
= 0.1518 / 0.4598
= 0.330
```

Net benefit at threshold 0.20:

```text
logistic:  0.61 * 0.23 - 0.29 * 0.77 * (0.20 / 0.80)
         = 0.1403 - 0.2233 * 0.25
         = 0.1403 - 0.0558
         = 0.0845

boosting:  0.66 * 0.23 - 0.40 * 0.77 * 0.25
         = 0.1518 - 0.0770
         = 0.0748
```

The logistic model has higher net benefit (0.0845 versus 0.0748 per patient) despite lower
sensitivity, because its higher specificity produces fewer false positives at this prevalence. The
decision is to deploy the logistic model, with recalibration at site C and a local calibration check
before go-live.

### Two diagnostics that would reveal a broken split

**Diagnostic 1: a random split instead of a grouped split.** If the team had used a random
five-fold split on the development data, the logistic model's AUROC would have been 0.724 rather
than 0.681. The difference, 0.043, is larger than the difference between the two models under
grouped cross-validation (0.013). The inflation comes from the 4,100 records belonging to repeat
patients: some of a patient's admissions land in training and others in test, and the model
recognises the patient. The diagnostic is to count patients who appear in more than one fold. In a
correct grouped split, that count is zero.

**Diagnostic 2: a temporal split without a gap.** If training ends on 31 December 2022 and testing
begins on 1 January 2023, a patient discharged in late December may have a readmission in January
that is counted in the test period. The model's AUROC rises from 0.662 with a 30-day gap to 0.681
without one. The diagnostic is to count training records whose outcome window crosses the boundary;
in this cohort, 214 records cross it. A temporal validation report that does not state the gap is
not reproducible.

### Varying one assumption: prevalence at the deployment site

Suppose hospital C's prevalence is 12 percent rather than 23 percent, for example because it serves
a lower-risk population. Sensitivity and specificity at the fixed threshold are assumed unchanged.
PPV for the logistic model becomes:

```text
0.61 * 0.12 / (0.61 * 0.12 + 0.29 * 0.88)
= 0.0732 / (0.0732 + 0.2552)
= 0.0732 / 0.3284
= 0.223
```

PPV falls from 0.386 to 0.223. Net benefit at 0.20 becomes:

```text
0.0732 - 0.2552 * 0.25 = 0.0732 - 0.0638 = 0.0094
```

The net benefit is still positive but close to zero, and the model's advantage over treat-all
narrows. The decision changes from "deploy with recalibration" to "do not deploy without
recalibration and a threshold review", because a programme that enrols 1,000 patients to prevent
about nine net events may not justify the operational cost. The validation design did not change;
the assumption about prevalence did, and the conclusion moved.

### What to report

The primary figure is a forest plot of AUROC by validation design (grouped CV, temporal, external)
for each model, with confidence intervals. The secondary figure is a calibration plot at the
external site with the slope and intercept annotated. The table of optimism by model belongs in the
appendix, because it explains the model comparison. The report should state the gap, the grouping
variable, the fold count, the preprocessing scope, and the number of patients spanning folds. Those
five facts are what make the validation reproducible.

## 7. Code, query, or calculation

The script demonstrates the difference between a random split and a grouped split on data with
repeated admissions. It uses a one-nearest-neighbour classifier deliberately, because that
classifier makes patient-level leakage visible: when the same patient appears in training and
test, the nearest neighbour is the same patient's other admission, and the label is copied. A
logistic regression would leak less on this small feature set, but the mechanism is the same.
Replace the synthetic cohort with your data and the nearest-neighbour classifier with your model;
the split functions are the reusable part.

```python
"""Random versus grouped versus temporal splitting. Standard library only."""
import random


def make_cohort(seed=11):
    """400 patients with 1-3 admissions each; one feature encodes patient identity."""
    rng = random.Random(seed)
    records = []
    for pid in range(1, 401):
        base = [rng.gauss(0, 1) for _ in range(3)]
        risk = 1.5 * base[0] - 0.4 * base[1]
        patient_label = 1 if risk + rng.gauss(0, 0.3) > 0.2 else 0
        for _ in range(rng.choice([1, 2, 2, 2, 3])):
            features = [v + rng.gauss(0, 0.15) for v in base] + [pid * 5.0]
            label = patient_label if rng.random() < 0.95 else 1 - patient_label
            records.append({"pid": pid, "year": rng.choice([2022, 2023, 2024]),
                            "x": features, "y": label})
    return records


def folds_random(records, k, seed=3):
    rng = random.Random(seed)
    index = list(range(len(records)))
    rng.shuffle(index)
    return [set(index[i::k]) for i in range(k)]


def folds_grouped(records, k, seed=3):
    rng = random.Random(seed)
    patients = sorted({r["pid"] for r in records})
    rng.shuffle(patients)
    assignment = {pid: i % k for i, pid in enumerate(patients)}
    return [set(i for i, r in enumerate(records) if assignment[r["pid"]] == f)
            for f in range(k)]


def auroc(scored):
    positives = [s for s, y in scored if y == 1]
    negatives = [s for s, y in scored if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(positives) * len(negatives))


def nearest_label(train, test):
    """1-nearest-neighbour score: the label of the closest training record."""
    scored = []
    for r in test:
        closest = min(train, key=lambda t: sum((a - b) ** 2
                                               for a, b in zip(r["x"], t["x"])))
        scored.append((closest["y"], r["y"]))
    return scored


def cross_validate(records, fold_function):
    scored = []
    for fold in fold_function(records, 5):
        train = [r for i, r in enumerate(records) if i not in fold]
        test = [r for i, r in enumerate(records) if i in fold]
        scored.extend(nearest_label(train, test))
    return auroc(scored)


def spanning_patients(records, folds):
    placement = {}
    for fold_number, fold in enumerate(folds):
        for i in fold:
            placement.setdefault(records[i]["pid"], set()).add(fold_number)
    return sum(1 for folds_seen in placement.values() if len(folds_seen) > 1)


records = make_cohort()
print(f"records={len(records)} patients={len(set(r['pid'] for r in records))} "
      f"prevalence={sum(r['y'] for r in records) / len(records):.3f}")
print(f"random CV AUROC={cross_validate(records, folds_random):.3f}")
print(f"grouped CV AUROC={cross_validate(records, folds_grouped):.3f}")
print(f"patients spanning folds: random={spanning_patients(records, folds_random(records, 5))} "
      f"grouped={spanning_patients(records, folds_grouped(records, 5))}")

train = [r for r in records if r["year"] <= 2022]
test = [r for r in records if r["year"] == 2024]
overlap = {r["pid"] for r in train} & {r["pid"] for r in test}
train_grouped = [r for r in train if r["pid"] not in overlap]
print(f"temporal train={len(train)} test={len(test)} patients_in_both={len(overlap)} "
      f"AUROC={auroc(nearest_label(train, test)):.3f}")
print(f"temporal grouped train={len(train_grouped)} AUROC="
      f"{auroc(nearest_label(train_grouped, test)):.3f}")
```

Expected output:

```text
records=807 patients=400 prevalence=0.411
random CV AUROC=0.800
grouped CV AUROC=0.512
patients spanning folds: random=272 grouped=0
temporal train=269 test=264 patients_in_both=93 AUROC=0.636
temporal grouped train=165 AUROC=0.461
```

Reading the output:

- The random split inflates AUROC from 0.512 to 0.800 because 272 patients appear in more than one
  fold. The nearest-neighbour classifier copies the label from the same patient's other admission.
  The grouping variable is what closes the leak.
- The grouped split has zero patients spanning folds by construction. Its AUROC of 0.512 is close
  to chance because the synthetic features (apart from the patient identifier) carry weak signal at
  this sample size, and the identifier is useless when patients are held out.
- The temporal split still puts 93 patients in both the training and test periods, because a
  patient can have admissions in 2022 and 2024. Removing those patients from training drops AUROC
  from 0.636 to 0.461. A temporal split that does not also group by patient is not a patient-level
  split; it is a time-stamped random split with extra steps.
- The temporal group split uses only 165 training records, which illustrates the cost of strict
  separation: less data, wider uncertainty.

The reusable functions are `folds_grouped` and `spanning_patients`. The first should be used in
place of `folds_random` for any dataset with repeated units. Both are deliberately small enough to
read in one sitting. The second should be run as an audit
on every split, and its output should be zero for the grouping variable.

## 8. Data, standards, and regulatory context

Validation design is addressed by reporting guidelines, appraisal tools, and regulatory guidance.
The table distinguishes what is binding from what is expected.

| Instrument | Status | What it says about validation | What it does not do |
|---|---|---|---|
| TRIPOD+AI (2024) | Reporting convention | Requires data sources, sample size, model development, and performance evaluation to be reported, including internal validation | Does not prescribe a split algorithm |
| PROBAST+AI (2025) | Appraisal tool | Assesses analysis domain, including sample size, overfitting, and validation | Does not validate the model |
| FDA GMLP principles (2021) | Guidance, not binding | Calls for independent test sets representative of the intended population | Does not define independence statistically |
| FDA PCCP guidance (2024) | Guidance, not binding | Requires a modification protocol with validation for planned changes | Does not cover initial validation design |
| DECIDE-AI (2022) | Reporting convention | Covers early live evaluation, including human factors and workflow | Does not cover retrospective split design |
| CONSORT-AI (2020) | Reporting convention | Requires trial reports to describe how the AI intervention was evaluated | Does not cover development studies |
| SPIRIT-AI (2020) | Reporting convention | Requires trial protocols to pre-specify AI evaluation | Does not cover retrospective studies |
| Riley et al. sample-size criteria (2020) | Methodological standard | Minimum sample size for developing a prediction model | Does not set a validation sample size |
| EU MDR 2017/745 | Binding in the EU | Requires clinical evaluation and performance evaluation | Does not prescribe cross-validation |
| ISO 14971:2019 | Voluntary standard | Risk management across the device lifecycle | Does not address statistical validation |
| IMDRF SaMD N41 | International consensus | Risk categorisation for software as a medical device | Not enforceable |

Three points matter for practice.

**TRIPOD+AI separates development from evaluation.** The checklist asks for the data sources and
the model development procedure, and separately for the performance evaluation, including whether
it was internal or external. A report that describes a cross-validated AUROC without saying whether
the folds were grouped, temporal, or random has not completed the evaluation item in a meaningful
way. The reporting standard is a convention, but reviewers and regulators use it as a checklist.

**PROBAST+AI's analysis domain targets the failures in this topic.** The tool asks about sample
size, overfitting, model selection, discrimination, calibration, and whether the evaluation was
independent. It distinguishes apparent, internal, and external validation. A study that reports
only apparent performance is at high risk of bias by construction, regardless of how impressive the
number is.

**The FDA's GMLP principles expect an independent test set.** The principles are guidance, not a
statute, but they reflect the agency's expectation that a model's performance claim be supported by
data not used in development. The phrase "independent" is doing work: a random split of repeated
measures is not independent at the patient level, and a test set used for tuning is not independent
at the procedure level. The guidance does not define the statistical test for independence; that is
the team's job.

**Reporting the design in a model card.** A model card should state the validation design in one
sentence that a non-specialist can interpret: "Performance was estimated by grouped five-fold
cross-validation on discharges from 2019-2022 at two hospitals, and by temporal validation on 2023
discharges from the same hospitals, and by external validation at a third hospital in 2023." That
sentence conveys the axes tested and the axes not tested. A card that says "validated on 20,000
patients" conveys neither.

**Validation sample size.** Development sample size and validation sample size are different
problems. Riley et al. (2020) provide criteria for development based on the number of candidate
parameters, the expected effect size, and the precision of the predicted risk. For validation, the
binding constraint is usually the number of events in the test set: a test set with 50 events gives
an AUROC confidence interval roughly 0.08 to 0.10 wide, which may be too imprecise to distinguish
two models. Plan the test set size from the precision needed for the comparison, not from a
conventional 80/20 ratio.

> **Note:** The EU AI Act adds obligations for high-risk AI systems, including medical devices, on
> data governance and testing. The Act does not replace the MDR's clinical evaluation requirements.
> The applicable obligations depend on the role (provider or deployer) and the system category, and
> the schedule is phased.

## 9. Industry practice and case evidence

**Nested cross-validation and selection bias.** Varma and Simon (2006) demonstrated that using the
same cross-validation folds for parameter selection and error estimation produces a substantially
optimistic estimate. On null data with a true error of 50 percent, the minimum cross-validation
error averaged 37.8 percent for shrunken centroids and 41.7 percent for support vector machines,
and on more than one-fifth of datasets the bias exceeded 20 percentage points. Nested
cross-validation produced 54.2 percent, close to the truth. The lesson is that the number of
configurations tried is part of the optimism, and that a single loop cannot answer two questions.

**Internal-external validation.** Steyerberg and Harrell (2016) argue that internal validation is
always required, that bootstrapping is preferred to split-sample validation, and that
internal-external validation (leaving one centre or one period out) gives an external flavour
during development. They also note that many failed external validations could have been foreseen
by rigorous internal validation. The practical implication is that a multi-site development study
should report leave-one-site-out performance even when the final model is fitted on all sites.

**Temporal validation in sepsis.** The Epic Sepsis Model external validation reported a
hospitalisation-level AUROC of 0.63, against vendor-reported values of 0.76 to 0.83. Part of the
difference is site and era: the model was developed on one population and evaluated at Michigan
Medicine in 2018-2019. The authors also reported that including scores up to three hours after
sepsis onset raised the AUC to 0.80, which is a label-timing effect rather than a validation-design
effect. The case illustrates why temporal and site validation must be paired with a label audit;
otherwise the performance difference is attributed to the wrong cause.

**Use-case approximation.** Saeb et al. (2017) argued that clinical machine-learning studies often
evaluate with a split that does not approximate the intended use case, for example by mixing
patients who would be eligible at deployment with those who would not. The result is a performance
estimate for a task nobody will perform. The lesson is that the split should mimic the deployment
eligibility and timing, not merely separate rows.

**COVID-19 prediction models.** The living systematic review by Wynants et al. (2020) found a large
number of COVID-19 prognostic models, most at high risk of bias, with few externally validated.
Many were developed on small single-centre cohorts and reported apparent or randomly split
performance. The case is a reminder that validation design quality, not model class, drove the
readiness gap.

**Model comparison instability.** Cawley and Talbot (2010) showed that over-fitting in model
selection can be comparable in magnitude to genuine differences between algorithms, and that
reusing the selection criterion for evaluation is unreliable. In practice, a difference of 0.01 in
cross-validated AUROC between two models is often smaller than the partition noise. The implication
for industry practice is that model selection should use repeated nested cross-validation and
should report the uncertainty in the difference, not only the point estimates.

**What teams actually do.** In practice, most industry teams run a grouped five-fold
cross-validation during development, a temporal holdout for the model card, and a site-held-out
validation only when a partner site supplies data. The gap between the second and third is where
most claims weaken. External validation is less common than internal validation in the published
literature, and the COVID-19 review by Wynants et al. is a clear example of how many models reached
publication without it. The practical recommendation is to budget for external validation at the
start of the project, because it is much harder to obtain partner data after the model has a
performance claim attached.

**Why the model comparison reverses.** The reversal between cross-validated and external ranking is
a common pattern with flexible models: they exploit interactions that are specific to the
development sites, and those interactions do not transport. The cross-validated estimate is not
wrong for its estimand; it is an estimate for new patients at the same sites. The external estimate
answers a different question. Reporting both, and naming which one matches the deployment claim, is
the practice that prevents the reversal from becoming a surprise after launch.

> **Note:** The cases are documented. The claim that a specific organisation's deployment failed
> because of its split design is not made here; the cases support the mechanism, not a causal
> attribution for any unnamed programme.

## 10. Failure modes and red flags

A validation failure usually presents as a number that is too good for the problem. The table lists
the mechanisms that produce such numbers, the evidence that would expose them, and the correction.
The first five are the most common in practice.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Random split with repeated patients | Number of records exceeds number of patients | Same patient in train and test | Inflated performance | Group by patient |
| Test set used for tuning | Hyperparameters chosen on the best test score | Selection on the evaluation data | Optimistic estimate | Nested CV |
| No temporal gap | Test period starts immediately after training | Outcome window overlaps boundary | Leakage of future outcomes | Gap at least the outcome window |
| Temporal split without patient grouping | Patients appear in both periods | Same patient in train and test | Residual leakage | Group by patient within periods |
| Preprocessing fitted outside folds | Imputer or scaler fitted before split | Test statistics enter training | Small but systematic optimism | Fit pipeline inside fold |
| Feature selection outside folds | Top features chosen on full data | Test outcomes influence selection | Optimistic and unstable | Select inside fold |
| Target encoding outside folds | Encoding uses all outcomes | Outcome information leaks | Severe optimism | Encode inside fold with smoothing |
| Single holdout with few events | Test set has fewer than 50 events | Sampling noise | Unstable estimate | Repeated stratified CV |
| Fold with one class | AUROC undefined in a fold | Rare outcome, unlucky partition | Missing or infinite metric | Stratify or merge |
| Site-held-out with tiny sites | Site estimate has wide interval | Too few events per site | Uninterpretable transport claim | Pool sites or narrow claim |
| Apparent performance reported as validation | No split described | Training performance | Grossly optimistic | Require a named design |
| Claim exceeds design | "External" for a random split | Label substitution | Unsupported transport claim | Match claim to design |
| Duplicate records | Identical rows with different IDs | Data entry or join duplication | Test leakage | Deduplicate before splitting |
| Seed sensitivity ignored | Result changes with a new seed | Partition noise | Overconfident comparison | Repeat with multiple seeds |
| Test set reused across model versions | The same holdout reports every iteration | Iterative peeking | Holdout becomes a validation set | Freeze the holdout or use nested CV |
| Fold assignment correlated with outcome | One fold has much higher prevalence | Non-random grouping | Misleading fold spread | Stratify and inspect prevalence |
| Model refitted after validation | Reported model differs from the evaluated one | Pipeline version drift | Claim applies to a different model | Version and freeze the artefact |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Patients are exchangeable within the development setting | Internal validation estimates performance for new patients | Case mix changes over time or by site | Compare characteristics across folds |
| The future resembles the past | Temporal validation estimates future performance | Drift in practice, coding, or population | Monitor after deployment |
| The held-out site is representative of deployment sites | Site-held-out validation supports transport | Deployment site differs in kind | Compare site characteristics; add sites |
| The grouping variable captures dependence | Grouped CV prevents leakage | Dependence at another level (clinician, family) | Audit multiple grouping levels |
| The outcome window is shorter than the gap | Temporal split prevents overlap | Long windows or short gaps | Count boundary-crossing records |
| Hyperparameter grid is modest | Nested CV controls selection bias | Huge grid with few events | Reduce grid or increase data |
| Fold count is adequate | Stable performance estimate | Too few folds or too few events | Report fold spread and repeats |
| Performance metrics are comparable across folds | Pooled estimate is meaningful | Prevalence or case mix differs by fold | Report per-fold prevalence |
| The test set is untouched | Estimate is honest | Repeated peeking or tuning | Audit analysis logs |
| The model is frozen during validation | Estimate applies to a fixed procedure | Retraining after each fold | Version the model and pipeline |
| The test period is long enough to be informative | Temporal estimate has adequate events | Test period has too few outcomes | Count events in the test period |
| The deployment population is within the validated range | Transport claim is bounded | Case mix outside the development range | Compare distributions and subgroup performance |

A second limit is that a validation design tests the procedure that was run, not the procedure that
will be deployed. If the deployment pipeline retrains monthly on local data, the validated model is
not the deployed model, and the performance estimate does not apply. The validation report should
state whether the deployed artefact is frozen or adaptive, and if adaptive, what change control
governs it.

The design also cannot tell you whether the model will be used, only whether it performs. The deepest limit is that retrospective validation cannot estimate the effect of using the model.
It estimates the model's statistical performance on held-out data. A prospective study is needed to
estimate whether clinicians change decisions and whether patients benefit. The gap between a good
retrospective validation and a useful deployment is therefore not a technicality; it is the
difference between a statistical claim and a clinical one, and it should be stated in the model
card. The validation design
should state which of these questions it answers.

## 12. Exercises

Exercises 1 through 3 are design and calculation; 4 and 6 are implementation; 5 and 7 are critique;
8 is the protocol you can reuse in the module assignment. Each exercise states a deliverable, an
acceptance criterion, and a time budget.

**Exercise 1. Identify the unit of independence.** For five datasets (ICU stays, primary care
visits, imaging studies, claims, wearables), name the unit and the grouping variable. Deliverable:
a five-row table. Acceptance criteria: each unit is justified by the data-generating process; at
least one dataset has two plausible levels. Time: 30 minutes.

**Exercise 2. Compute optimism.** A model has apparent AUROC 0.79 and grouped cross-validated AUROC
0.71. A second model has apparent 0.83 and cross-validated 0.72. Compute optimism for each and
state which model you would prefer. Deliverable: arithmetic and a recommendation. Acceptance
criteria: optimism correct; recommendation addresses uncertainty, not only point estimates. Time:
30 minutes.

**Exercise 3. Design a temporal split.** A cohort spans 2018-2025 with a 90-day outcome window.
Propose training, gap, and test periods, with counts. Deliverable: a timeline and a rationale.
Acceptance criteria: gap at least 90 days; counts on both sides; boundary-tie rule stated. Time: 45
minutes.

**Exercise 4. Build a grouped split.** Using a dataset with patient identifiers, implement grouped
five-fold cross-validation and verify that no patient appears in more than one fold. Deliverable:
code and an audit table. Acceptance criteria: zero spanning patients; fold sizes reported; seed
recorded. Time: 60 minutes.

**Exercise 5. Critique a validation section.** Find a published model paper and assess its
validation design against the table in Section 5. Deliverable: a 600-word critique. Acceptance
criteria: design named; claim matched to design; at least three specific gaps identified. Time: 60
minutes.

**Exercise 6. Nested cross-validation.** Implement a nested loop for a logistic regression with one
hyperparameter. Compare the nested estimate with the non-nested estimate. Deliverable: code and a
short report. Acceptance criteria: inner and outer loops separated; optimism computed; result
reproducible with a seed. Time: 90 minutes.

**Exercise 7. Contamination audit.** Given a pipeline that imputes, scales, selects features, and
tunes, identify every step that could leak and state where it must be fitted. Deliverable: an audit
table. Acceptance criteria: at least five steps; each with correct fit scope; one example of a step
that cannot be used safely. Time: 45 minutes.

**Exercise 8. Write a validation protocol.** For your project, write the validation section of the
model development report: claim, design, grouping, folds, gap, preprocessing scope, metrics, and
subgroup analyses. Deliverable: a two-page protocol. Acceptance criteria: all elements present;
design matches claim; limitations stated. Time: 90 minutes.

## 13. Self-check answers

**Answer to Exercise 1 (illustrative units).** ICU stays: the patient is usually the unit, because
a patient can have several stays and the stays share physiology and history. Primary care visits:
the patient, for the same reason, though the clinician is a second level if practice patterns are
the question. Imaging studies: the patient, or the study if each image is independent, but a
patient with repeat imaging contributes correlated studies. Claims: the patient, with the plan or
provider as a second level. Wearables: the participant, with days or windows nested within. In
every case, the unit is chosen to match the claim, not the table's primary key.

**Answer to Exercise 2.** Model one optimism = 0.79 - 0.71 = 0.08. Model two optimism = 0.83 -
0.72 = 0.11. Model two has higher apparent performance but also higher optimism, and its validated
performance is only 0.01 better. Given that the difference is likely within partition noise, the
simpler model one is preferable unless repeated nested cross-validation shows a consistent
advantage for model two. The answer should mention the uncertainty in the difference, not only the
point estimates.

**Answer to Exercise 3.** A defensible design is training 2018-2023, gap January-March 2024 (90
days or more), test April 2024-2025. The gap ensures that a patient discharged in December 2023
with a 90-day outcome window has the outcome ascertained before the test period begins. If the
data are thin in 2024, extend the test period into 2025 or shorten the outcome window, but do not
remove the gap. Records with time zero on the boundary date should be assigned to one side by a
rule stated in advance.

**Answer to Exercise 4.** The audit should print the number of patients in more than one fold. A
correct grouped split gives zero. It should also print the minimum and maximum fold size and the
prevalence per fold; large variation indicates an unlucky partition or a grouping variable that is
correlated with the outcome. The seed and the fold assignment should be saved so the split can be
reproduced.

**Answer to Exercise 6.** The nested estimate is the mean of the outer-fold scores, where each
outer fold's model was selected by an inner loop. The non-nested estimate is the best inner score
or the score of the model tuned on the full data and evaluated on the same folds. The nested
estimate should be lower, and the difference is the selection optimism. On small data, the
difference can be several percentage points. If the nested estimate is higher than the non-nested
one, check for a bug in the loop or for leakage in the inner selection.

**Answer to Exercise 5 (what a strong critique contains).** A strong critique names the design,
states the claim the authors make, and checks whether the design supports it. It counts the units
(patients versus rows), identifies the grouping variable used, and asks whether the test set was
used for model selection. It looks for a gap in any temporal split and for a description of where
preprocessing was fitted. It compares the reported performance with the calibration, because a
model that ranks well but is miscalibrated may not support a threshold-based decision. It ends with
two or three specific, actionable corrections rather than a general statement that the paper could
be better.

**Answer to Exercise 8 (what a complete protocol contains).** A complete protocol names the claim in
one sentence, then states the unit of independence, the grouping variable, the fold count, the
stratification, the temporal boundaries and gap, the preprocessing scope, the hyperparameter grid,
the primary metric, the calibration assessment, the subgroup analyses, and the limitations. It also
states the seed and the software version. The test of completeness is whether a second analyst
could reproduce the split and the performance estimate without asking a question. If the protocol
answers every design question but not the reproducibility question, it is not finished.

**Rubric for Exercises 1, 3, 5, 7, and 8.** A strong submission (1) states the claim, (2) names
the unit of independence and the grouping variable, (3) chooses a design that perturbs the relevant
axis, (4) places all preprocessing inside the fold, (5) reports optimism and uncertainty, and (6)
states what the design cannot test. A competent submission meets four of the six. A weak submission
reports a random split as external validation or tunes on the test set.

## 14. Interview questions

**1. Why is a random split often wrong in clinical data?** A strong answer explains repeated
measures and patient-level leakage, and gives the grouped alternative. A weak answer says random
splits are fine if the dataset is large.

**2. What is nested cross-validation and why is it needed?** A strong answer separates inner tuning
from outer evaluation and explains selection bias. A weak answer describes k-fold twice without
explaining the purpose.

**3. How do you validate a model for a new hospital?** A strong answer proposes site-held-out or
external validation, local calibration, and a prospective silent trial before go-live. A weak
answer proposes retraining on the new hospital's data and reporting cross-validation.

**4. What is optimism and how do you estimate it?** A strong answer defines it as apparent minus
validated performance and describes bootstrap or nested CV estimation. A weak answer confuses it
with overfitting in general.

**5. How long should the gap be in a temporal split?** A strong answer says at least the outcome
window, and explains boundary overlap. A weak answer says "a few days" without justification.

**6. What is the difference between fold-level variability and a confidence interval?** A strong
answer distinguishes partition sensitivity from sampling uncertainty. A weak answer treats the
standard deviation across folds as a confidence interval.

**7. When would you use leave-one-site-out validation?** A strong answer says when the deployment
claim covers multiple sites and each site has enough events, and notes the internal-external
variant. A weak answer says always, without considering site size.

**8. How do you detect preprocessing leakage?** A strong answer audits where each transform is
fitted and requires the pipeline to be fitted inside the fold. A weak answer says "use a pipeline"
without explaining the fit scope.

**9. What would make you distrust a reported AUROC of 0.95?** A strong answer looks for leakage:
duplicates, post-time-zero features, tuning on the test set, or a label that encodes the outcome. A
weak answer says high AUROC is always suspicious without a mechanism.

**10. What is internal-external validation?** A strong answer describes leaving one site or one
period out during development while fitting the final model on all data, and explains why it gives
an external flavour without a separate study. A weak answer says it is the same as cross-validation.

**11. How does clustering affect the effective sample size?** A strong answer explains that
correlated observations carry less independent information, and that counting rows overstates
precision. A weak answer denies any effect if the model is fitted at the row level.

**12. What is the first thing you check when a validation AUROC is much lower than the development AUROC?** A strong answer checks whether the populations differ, whether the label changed, whether the model is calibrated, and whether the deployment site has a different case mix, before concluding that the model has degraded. A weak answer immediately proposes retraining.

**13. How do you compare two models whose AUROCs differ by 0.01?** A strong answer uses repeated
nested CV, reports the uncertainty in the difference, and considers calibration and net benefit. A
weak answer picks the higher number.

## 15. Key terms

- **Apparent performance** — performance measured on the development data.
- **Bootstrap validation** — resampling to estimate optimism.
- **Calibration slope** — agreement of predicted and observed risk on the logit scale.
- **Cluster** — a group of correlated observations.
- **Concept shift** — change in the feature-outcome relationship.
- **Confidence interval** — uncertainty in a performance estimate.
- **Covariate shift** — change in the feature distribution.
- **Effective sample size** — sample size adjusted for clustering.
- **Embargo** — a gap between training and test periods.
- **Events per variable** — outcome events per candidate parameter.
- **External validation** — validation in a different setting, time, or population.
- **Fold-level variability** — spread of performance across folds.
- **Generalisation** — performance on data not used for development.
- **Grouped k-fold** — folds that preserve group membership.
- **Holdout set** — data reserved for final evaluation.
- **Hyperparameter** — a model setting chosen before fitting.
- **Inner loop** — tuning loop inside each outer fold.
- **Internal validation** — validation within the development setting.
- **Internal-external validation** — leave-one-site-out within a multi-site study.
- **Intraclass correlation** — proportion of variance between clusters.
- **Leave-one-site-out** — each site is a test fold.
- **Nested cross-validation** — inner tuning and outer evaluation.
- **Optimism** — apparent minus validated performance.
- **Outer loop** — evaluation loop.
- **Preprocessing leakage** — fitting transforms outside the training fold.
- **Purging** — removing training observations close in time to test observations.
- **Repeated cross-validation** — multiple partitions with different seeds.
- **Stratified k-fold** — folds that preserve outcome prevalence.
- **Temporal validation** — training on earlier data, testing on later data.
- **Test set contamination** — test data influencing model selection.
- **Transportability** — whether a model performs in a new setting.
- **Unit of independence** — the entity that must not cross folds.

## 16. Sources

All URLs accessed 25 October 2026.

1. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC
   Bioinformatics* 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
2. Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in
   performance evaluation. *Journal of Machine Learning Research* 2010;11:2079-2107.
   https://jmlr.org/papers/v11/cawley10a.html
3. Steyerberg EW, Harrell FE Jr. Prediction models need appropriate internal, internal-external,
   and external validation. *Journal of Clinical Epidemiology* 2016;69:245-247.
   https://doi.org/10.1016/j.jclinepi.2015.04.005
4. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
5. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
   https://doi.org/10.1136/bmj-2024-082505
6. Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a
   clinical prediction model. *BMJ* 2020;368:m441. https://doi.org/10.1136/bmj.m441
7. Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable
   prediction model: part II - binary and time-to-event outcomes. *Statistics in Medicine*
   2019;38(7):1276-1296. https://doi.org/10.1002/sim.7992
8. Saeb S, Lonini L, Jayaraman A, Mohr DC, Kording KP. The need to approximate the use-case in
   clinical machine learning. *GigaScience* 2017;6(5):1-9.
   https://doi.org/10.1093/gigascience/gix019
9. Ramspek CL, Jager KJ, Dekker FW, Zoccali C, van Diepen M. External validation of prognostic
   models: what, why, how, when and how to do it properly. *Clinical Kidney Journal*
   2021;14(1):49-58. https://doi.org/10.1093/ckj/sfaa188
10. Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation, detection,
    and avoidance. *ACM Transactions on Knowledge Discovery from Data* 2012;6(4):1-21.
    https://doi.org/10.1145/2382577.2382572
11. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
    Guiding Principles. 2021.
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
12. Vasey B, Nagendran M, Campbell B, et al. DECIDE-AI. *Nature Medicine* 2022;28:924-933.
    https://doi.org/10.1038/s41591-022-01772-9
13. Liu X, Cruz Rivera S, Moher D, et al. Reporting guidelines for clinical trial reports for
    interventions involving artificial intelligence: the CONSORT-AI extension. *Nature Medicine*
    2020;26:1364-1374. https://doi.org/10.1038/s41591-020-1034-x
14. Cruz Rivera S, Liu X, Chan AW, et al. Guidelines for clinical trial protocols for interventions
    involving artificial intelligence: the SPIRIT-AI extension. *Nature Medicine*
    2020;26:1351-1363.
15. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
    sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
    https://doi.org/10.1001/jamainternmed.2021.2626
16. Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of
    covid-19. *BMJ* 2020;369:m1328. https://doi.org/10.1136/bmj.m1328
17. scikit-learn developers. Cross-validation: evaluating estimator performance.
    https://scikit-learn.org/stable/modules/cross_validation.html
18. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1

A final check before trusting a validation number: name the claim, the unit of independence, the
axis of shift, and the gap. If any of the four is missing, the number describes a different question
from the one the deployment will ask. The design is the argument, and the number is only as good as
the argument.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_labels-and-outcomes.md`](02_labels-and-outcomes.md)
- Next topic: [`04_preprocessing.md`](04_preprocessing.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../course/06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
