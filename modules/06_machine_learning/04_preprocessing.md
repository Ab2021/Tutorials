# Preprocessing

> **Module 06 — Machine Learning Foundations**
> Topic 04 of 13
> Estimated study time: 3.5 hours
> Prerequisites: Topics 02 and 03
> Version 2026.3

## 1. Why this topic exists

Preprocessing is where most leakage enters a clinical machine-learning pipeline, and it is where
most teams spend the least design effort. The steps look mechanical: fill missing values, scale the
columns, encode the categories. Each step, however, estimates something from data. A median is
estimated. A mean and standard deviation are estimated. A category's outcome rate is estimated. If
those estimates are computed on the full dataset before the split, information from the test fold
enters the training process, and the validation estimate becomes optimistic.

The effect is usually smaller than patient-level leakage, but it is systematic and it compounds.
A median imputation fitted on all data leaks the distribution of the test set. A scaler fitted on
all data leaks the test set's mean and variance. A target encoder fitted on all data leaks the test
set's labels directly, which is the most severe form. A feature-selection step fitted on all data
leaks the association between features and outcomes in the test set. None of these show up as an
error message; they show up as a validation number that is too good.

Preprocessing also determines whether the model can be deployed at all. A pipeline that imputes a
laboratory value with a population median assumes the laboratory is missing at random with respect
to the patient's risk. If the laboratory is missing because the patient was too unstable to be
tested, the imputation assigns a normal value to the sickest patients, and the model under-predicts
their risk. A pipeline that one-hot encodes a category with 4,000 levels creates a sparse matrix
that is expensive to store and impossible to transport to a site with different categories. A
pipeline that scales a variable using the development mean will mis-scale at a site with a
different assay or unit.

The industrial consequence is that preprocessing choices are part of the model's identity. Two
teams with the same algorithm and the same features but different imputation and encoding
strategies do not have the same model. Their performance numbers are not comparable, and their
deployment behaviour can differ materially. A model card that reports the algorithm but not the
preprocessing pipeline is incomplete.

The commercial consequence is subtler. A preprocessing pipeline that depends on a local laboratory
feed, a local coding dictionary, or a local missingness pattern is not a portable product. Vendors
who sell a model to multiple health systems discover this during implementation, when the pipeline
that worked at the development site produces implausible scores at the first customer. The cost of
re-engineering the pipeline often exceeds the cost of the model. Teams that treat preprocessing as
part of the product, with versioning and site-specific validation, avoid that surprise.

Preprocessing is also where clinical judgement enters the model. A decision to impute a missing
ejection fraction with the population median is a clinical statement that the median is a
reasonable stand-in for a patient whose heart function was not measured. A decision to clip a
creatinine value at 10 mg/dL is a statement that values above 10 are data errors rather than
genuine renal failure. Those statements should be made by someone with clinical knowledge, not by
the default settings of a library.

This topic treats preprocessing as a set of fitted transformations with a strict fit/transform
boundary. You will learn to inventory missingness, choose an imputation strategy, encode
categorical variables without leaking, handle degenerate features, scale only when the model
requires it, enforce point-in-time correctness, and audit the pipeline. The output is a pipeline
specification that can be fitted inside a cross-validation fold and transported to a deployment
site.

A final reason to care: preprocessing is the part of the pipeline most likely to change after
deployment. A new laboratory assay, a new EHR module, or a new coding dictionary can alter a
feature without altering the model file. If the pipeline is not monitored, the first sign of change
is a shift in the score distribution, which may be attributed to the model rather than to the
data. Monitoring the preprocessing inputs, not only the outputs, shortens the time to diagnosis.

> **Note:** The rule is simple and absolute: every transformation that learns a parameter is fitted
> on the training fold only, and applied unchanged to the validation fold. Everything else in this
> topic follows from that rule.

## 2. Learning outcomes

By the end of this topic you can:

1. **Inventory** missingness by feature, subgroup, site, and time, and **distinguish** structural
   from incidental missingness.
2. **Choose** an imputation strategy that matches the missingness mechanism and the deployment
   process, and **justify** the choice.
3. **Implement** median, model-based, and multiple imputation inside a cross-validation fold.
4. **Encode** categorical variables by cardinality, using one-hot, grouped rare levels, hashing, or
   fold-safe target encoding with smoothing.
5. **Detect** and remove zero-variance and single-level features, and **handle** new levels at
   deployment.
6. **Scale** and transform features only where the model family requires it, and **explain** the
   effect on interpretability.
7. **Enforce** point-in-time correctness and unit harmonisation across source systems.
8. **Audit** a preprocessing pipeline for leakage, reproducibility, and transportability.

Outcomes 3, 4, and 5 are computational and are assessed in the code exercises. Outcomes 2 and 8
are design outcomes assessed by rubric.

## 3. The core mental model

A preprocessing pipeline is a sequence of fitted objects. Each object has a `fit` step that learns
parameters from data and a `transform` step that applies them. The fit step is the leak. The
transform step is safe once the parameters are fixed.

```text
Full dataset
    |
    +--> split by patient / time / site
            |
            +--> training fold ----------------------------+
            |       fit imputer (medians)                  |
            |       fit scaler (means, SDs)                |
            |       fit encoder (category statistics)      |
            |       fit selector (feature associations)    |
            |       fit model                              |
            |                                              |
            +--> validation fold                           |
                    transform with training parameters ----+
                    predict
                    evaluate
```

The diagram shows the only correct order. The validation fold never contributes to a fitted
parameter. When the pipeline is deployed, the parameters learned on the full development set are
frozen into the artefact, and new patients are transformed with those parameters.

There are four kinds of preprocessing decision, and they fail differently.

**Missing-data decisions** determine what the model sees when a value is absent. The choices are
deletion, single imputation, multiple imputation, model-native handling, and missing indicators.
The right choice depends on why the value is missing and on whether the same mechanism will operate
at deployment.

**Encoding decisions** determine how categorical information enters a numeric model. One-hot
encoding is safe but expands dimensionality. Ordinal encoding imposes an order that may be false.
Target encoding is powerful but leaks if not fitted inside the fold and smoothed. Hashing is
scalable but loses interpretability.

**Scaling decisions** determine whether features are on comparable scales. Linear models with
regularisation, support vector machines, k-nearest neighbours, and neural networks need scaling.
Tree-based models do not, and scaling them adds a transformation that must be maintained without
changing predictions.

**Temporal decisions** determine whether a feature was knowable at time zero. A laboratory value
drawn after the prediction time, a medication order placed because of the outcome, or a diagnosis
code entered at discharge are all preprocessing problems in the broad sense: the feature exists in
the table, but it is not available at the moment of prediction.

Two properties make preprocessing harder than it looks. First, the transformation is often
invisible in the final model: a logistic regression reports coefficients for the transformed
features, not the raw ones, and a reader cannot tell from the coefficient table whether creatinine
was standardised, log-transformed, or imputed. Second, the transformation is learned from a
particular population. A scaler fitted on a cohort with a mean creatinine of 1.1 mg/dL will
mis-scale a cohort with a mean of 1.8, and nothing in the model will raise an error. Both
properties mean that preprocessing decisions must be documented at the level of the individual
feature, not summarised as "data were preprocessed".

A useful discipline is to write the pipeline as a list of steps with, for each step, the
parameters it learns, the data it is fitted on, and the fallback for unseen values. If a step
cannot state those three things, it is not ready to be in the pipeline.

> **Note:** "Fit on train, transform test" applies within cross-validation, not only to the final
> holdout. Fitting an imputer once on the full development set and then cross-validating is still
> leakage, because each fold's validation patients contributed to the imputer.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Preprocessing | Transformations applied to features before model fitting | Treated as cosmetic | Pipeline step |
| Pipeline | A sequence of fitted transforms and a final estimator | Fitted outside cross-validation | Object |
| Fit | Estimate transformation parameters from data | Called on the full dataset | Method |
| Transform | Apply fixed parameters to data | Refitted on test data | Method |
| Fit-transform | Fit and transform the training data in one step | Used on test data | Method |
| Missing completely at random (MCAR) | Missingness independent of observed and unobserved values | Assumed without evidence | Mechanism |
| Missing at random (MAR) | Missingness depends on observed variables but not on the missing value itself | Confused with MCAR | Mechanism |
| Missing not at random (MNAR) | Missingness depends on the unobserved value | Ignored | Mechanism |
| Complete-case analysis | Analysing only records with no missing values | Called unbiased without checking | Design |
| Single imputation | Replacing a missing value with one estimate | Treated as observed | Method |
| Multiple imputation | Generating several plausible values and pooling results | Used with one imputation | Method |
| MICE | Multiple imputation by chained equations | Treated as assumption-free | Method |
| Missing indicator | A binary feature marking whether a value was missing | Added without justification | Feature |
| Structural missingness | A value is absent because the feature does not apply | Treated as random | Mechanism |
| Incidental missingness | A value is absent because it was not measured | Treated as structural | Mechanism |
| Imputation model | The model used to predict missing values | Omits auxiliary variables | Model |
| Auxiliary variable | A variable that helps predict missingness or the missing value | Ignored | Feature |
| Deletion | Removing records or features with missing values | Used to reach a complete case set | Method |
| Listwise deletion | Removing any record with at least one missing value | Confused with pairwise deletion | Method |
| Pairwise deletion | Using available pairs for each calculation | Produces inconsistent samples | Method |
| Mean imputation | Replacing missing values with the mean | Distorts variance | Method |
| Median imputation | Replacing missing values with the median | Distorts variance less for skewed data | Method |
| Mode imputation | Replacing missing categories with the most frequent | Understates uncertainty | Method |
| Regression imputation | Predicting missing values from other variables | Single imputation; underestimates variance | Method |
| K-nearest-neighbour imputation | Using similar records to fill values | Sensitive to scaling and distance | Method |
| Native missing handling | A model that handles missingness internally | Assumed equivalent to imputation | Model feature |
| One-hot encoding | One binary column per category | Dummy trap if all columns kept | Encoding |
| Dummy variable trap | Perfect collinearity from including all one-hot columns with an intercept | Ignored in regularised models | Encoding |
| Ordinal encoding | Integer codes for ordered categories | Applied to unordered categories | Encoding |
| Label encoding | Integer codes for categories | Confused with ordinal encoding | Encoding |
| Target encoding | Replacing a category with its outcome mean | Fitted outside folds | Encoding |
| Mean encoding | Synonym for target encoding | Treated as safe | Encoding |
| Smoothing | Shrinking a category estimate toward the global mean | Omitted for rare categories | Parameter |
| Shrinkage | Reducing an estimate toward a prior or global value | Confused with regularisation | Parameter |
| Leave-one-out encoding | Encoding a row using other rows in its category | Still leaks without care | Encoding |
| Cross-fitting | Encoding each fold using the other folds | Confused with fitting on all data | Encoding |
| Hashing | Mapping categories to a fixed number of columns | Loses reversibility | Encoding |
| Rare-level grouping | Combining infrequent categories into "other" | Threshold chosen arbitrarily | Encoding |
| Cardinality | Number of distinct values in a categorical feature | Ignored in encoding choice | Count |
| High cardinality | Many distinct values relative to sample size | One-hot encoded | Count |
| Zero-variance feature | A feature with the same value for every record | Left in the model | Feature |
| Near-zero-variance feature | A feature with almost no variation | Confused with zero variance | Feature |
| Standardisation | Centring and scaling to mean 0, SD 1 | Applied to tree models unnecessarily | Transform |
| Normalisation | Scaling to a fixed range, often 0 to 1 | Confused with standardisation | Transform |
| Robust scaling | Scaling by median and interquartile range | Confused with standardisation | Transform |
| Log transform | Compressing skewed positive values | Applied to zero or negative values | Transform |
| Box-Cox transform | Power transform to approximate normality | Requires positive values | Transform |
| Winsorisation | Capping extreme values at percentiles | Confused with trimming | Transform |
| Clipping | Limiting values to a range | Hides data errors | Transform |
| Binning | Converting a continuous variable into intervals | Loses information | Transform |
| Discretisation | Synonym for binning | Treated as a modelling choice | Transform |
| Collinearity | Highly correlated features | Ignored in linear models | Diagnostic |
| Variance inflation factor | Quantifies collinearity | Reported without context | Ratio |
| Point-in-time correctness | A feature is available at or before time zero | Assumed from the table | Audit |
| Unit harmonisation | Converting measurements to a common unit | Assumed across sites | Unit |
| UCUM | Unified Code for Units of Measure | Ignored in data integration | Standard |
| LOINC | Codes for laboratory observations | Used without unit checks | Standard |
| Transportability | Whether the pipeline works at a new site | Assumed from internal validation | Claim |
| New level | A category at deployment not seen in development | Causes encoder failure | Encoding |
| Unseen category fallback | The value assigned to a new category | Omitted | Encoding |
| Schema validation | Checking that expected columns and types are present | Assumed | Audit |
| Range check | Flagging values outside a clinical range | Omitted | Audit |
| Distribution comparison | Comparing feature distributions across sites | Assumed | Audit |
| Pipeline artefact | The serialised fitted transforms | Not versioned | Artefact |
| Imputation variance | Uncertainty from imputed values | Ignored in single imputation | Variance |

Two definitions deserve more than a table row.

**Structural versus incidental missingness.** A male patient has no value for "pregnancy status",
and that absence is structural: the feature does not apply. A patient without a creatinine value
may have been too well to test, too sick to survive a test, or treated at a facility without a
laboratory. Those are different mechanisms, and they imply different imputations. Treating
structural missingness as incidental produces nonsense imputations (a pregnancy probability for a
male patient). Treating incidental missingness as structural discards information. The inventory in
Step 1 should classify each feature's missingness before any imputation is chosen.

**Cross-fitting versus fitting on all data.** Cross-fitting means that each training fold's
encoding is estimated from the other folds within that training set, so that no row's own label
contributes to its own encoding. Fitting the encoder on all data, including the test fold, is
leakage. Fitting the encoder on the training fold and applying it to the test fold is correct but
can still overfit rare categories; smoothing and cross-fitting address that. The scikit-learn
documentation distinguishes `fit_transform` (cross-fitting for training data) from `fit` plus
`transform` (no cross-fitting), and the distinction matters.

The pipeline also has a contract with the deployment environment. It must accept the columns the
site can provide, tolerate the missingness the site actually has, and produce a value in the units
the model expects. A pipeline that satisfies that contract at one site and not at another is a
local artefact, not a product. The contract should be written down and tested at each site before
go-live.

> **Note:** Every preprocessing choice is a modelling assumption. Median imputation assumes the
> median is a reasonable stand-in; target encoding assumes the category mean is stable; scaling
> assumes the distance metric is meaningful. State the assumption, not only the method.

## 5. The mechanics, step by step

Preprocessing has fourteen steps. Steps 1 through 4 are diagnosis. Steps 5 through 12 are the
transformations. Step 13 is the temporal and unit audit. Step 14 freezes the pipeline.

### Step 1. Inventory features, types, and provenance

**Input:** the feature list from the prediction task specification.

**Output:** a table with name, type (continuous, ordinal, nominal, binary, count, date), source
table, unit, lookback window, and expected availability at time zero.

**Rationale:** The type determines the legal transformations. Applying a log transform to a
negative value, ordinal encoding to an unordered category, or arithmetic to a date produces errors
that are hard to trace. The provenance determines whether the feature is point-in-time correct. The
unit determines whether values from different sites are comparable.

**Branch point:** if a feature has no unit and no source, it cannot be validated. Remove it or
trace it before proceeding. A feature whose provenance is unknown is a leakage suspect.

### Step 2. Quantify missingness

**Input:** the inventory and the cohort.

**Output:** missingness rates overall, by feature, by site, by year, and by outcome; a missingness
matrix or heatmap; and the complete-case fraction.

**Rationale:** Overall missingness hides structure. A feature that is 10 percent missing overall
may be 2 percent missing at one site and 40 percent at another, or 5 percent missing in survivors
and 45 percent in patients who died. Those patterns are evidence about the mechanism, and the
mechanism determines the imputation. The complete-case fraction is also a sample-size decision: if
only 45 percent of records are complete, complete-case analysis discards more than half the data
and changes the population.

**Branch point:** if missingness exceeds roughly 40 percent for a feature, ask whether the feature
is measuring the same construct for everyone. High missingness often means the feature is
conditional on a test or a workflow, not that it is randomly absent.

### Step 3. Classify the missingness mechanism

**Input:** the missingness patterns, clinical knowledge, and, where possible, a validation sample
with the missing values recovered from the chart.

**Output:** a per-feature classification as structural, MCAR, MAR, or MNAR, with the evidence and a
confidence level.

**Rationale:** The mechanism determines which methods are valid. Complete-case analysis is unbiased
when missingness depends on observed covariates and not on the outcome, but biased when it depends
on the outcome. Multiple imputation is valid under MAR and can be biased under MNAR. No method is
valid under MNAR without an assumption about the missing values, which should be stated and
sensitivity-tested.

**Branch point:** if the mechanism cannot be determined, use a method that is robust across
plausible mechanisms and report a sensitivity analysis. Do not label the mechanism as MCAR merely
because the data cannot reject it.

### Step 4. Choose the imputation strategy

**Input:** the mechanism classification, the deployment process, and the model family.

**Output:** a strategy per feature, with the rationale and the fallback for values outside the
observed range.

**Rationale:** Different mechanisms and different deployment processes call for different methods.
A laboratory value that will also be missing at deployment needs a strategy that works with the
same missingness pattern in production. A value that is missing only in the development extract
because of a data migration needs a different strategy from one that is missing because the test
was not ordered.

| Situation | Recommended strategy | Alternative | Avoid |
|---|---|---|---|
| Structural missingness | Encode "not applicable" as its own category | Separate model for the applicable subgroup | Imputing a clinical value |
| MCAR, low missingness, simple baseline | Median or mode imputation | Complete case with a sensitivity analysis | Complex model-based imputation with no benefit |
| MAR, auxiliary variables available | Multiple imputation with chained equations | Model-based single imputation | Mean imputation without auxiliaries |
| MNAR suspected | Sensitivity analysis over plausible shifts | Pattern-mixture model | Claiming MAR without evidence |
| Missingness informative at deployment | Missing indicator plus imputation | Model-native missing handling | Dropping the indicator |

Multiple imputation adds a pooling step. Each of the M completed datasets is analysed separately,
and the results are combined using Rubin's rules, which account for both within-imputation and
between-imputation variance. For a prediction model, the pooling is usually applied to the
performance estimates rather than to the coefficients, because the model is refitted on each
completed dataset and the performance is averaged with an interval that includes the imputation
variance. Reporting only one completed dataset understates the uncertainty.

**Branch point:** if the same feature will be missing in deployment, the imputation model must be
estimable from deployment-time information only. An imputation model that uses a post-time-zero
variable is leakage even if it improves development performance.

### Step 5. Fit imputation inside the fold

**Input:** the training fold and the chosen strategy.

**Output:** imputation parameters (medians, models, category modes) fitted on the training fold,
and transformed training and validation folds.

**Rationale:** The imputation parameters are estimates. Fitting them on the full dataset lets the
validation fold influence the training data. The effect is small for a median but can be
substantial for a regression imputation or a k-nearest-neighbour imputation, which can memorize
validation records. The rule is the same for every transform: fit on train, apply to test.

**Branch point:** if the imputation model uses the outcome, it must be fitted only on the training
fold, and its predictions for the validation fold must not use validation outcomes. For multiple
imputation, generate the imputations separately within each fold and pool the performance estimates
across imputations.

### Step 6. Decide on missing indicators

**Input:** the missingness mechanism and the deployment process.

**Output:** a decision per feature: include an indicator, exclude it, or include it only in a
sensitivity analysis.

**Rationale:** A missing indicator tells the model that a value was absent. If absence is
informative (for example, a test not ordered because the patient was low risk), the indicator
carries signal. If absence is a proxy for a protected attribute or a site, the indicator can encode
bias. Indicators also change the meaning of the imputed value: the model can learn a separate
effect for "missing" and for "value equals the median".

**Branch point:** if the indicator is highly correlated with site, payer, or race, do not include
it without a fairness review. The indicator may improve internal performance while encoding a
structural inequity that fails at deployment.

### Step 7. Encode categorical variables

**Input:** the categorical features and their cardinality.

**Output:** an encoding per feature, with the number of resulting columns and the handling of
unseen levels.

**Rationale:** The encoding determines dimensionality, interpretability, and leakage risk. One-hot
encoding is safe for low cardinality but creates a column per level. Ordinal encoding is appropriate
only when the categories have a genuine order. Target encoding is powerful for high cardinality but
must be fitted inside the fold and smoothed. Hashing scales but loses the mapping.

| Cardinality | Recommended encoding | Reason | Watch for |
|---|---|---|---|
| 2 to 10 levels | One-hot | Stable, interpretable | Dummy trap with linear models |
| 11 to 50 levels | One-hot with rare-level grouping | Controls dimensionality | Grouping threshold |
| 51 to 500 levels | Fold-safe target encoding with smoothing | Captures signal without explosion | Leakage, rare levels |
| More than 500 levels | Hashing or target encoding with a hierarchical model | Scalability | Loss of interpretability |
| Ordered categories | Ordinal encoding | Preserves order | Assuming equal spacing |

**Branch point:** if a category has fewer than about 20 observations, its target encoding is
unstable. Smooth it heavily, group it with similar categories, or use one-hot with an "other"
level.

### Step 8. Implement high-cardinality target encoding safely

**Input:** a high-cardinality categorical feature and the training fold.

**Output:** a mapping from category to a smoothed outcome rate, fitted with cross-fitting inside the
training fold, and a fallback for unseen categories.

**Rationale:** Target encoding replaces a category with its outcome rate. Without smoothing, a
category with three patients and two events gets an encoding of 0.67, which is noise. Without
cross-fitting, each training row's own outcome contributes to its own encoding, which is a form of
leakage even inside the training fold. The smoothed estimate is
`(n * category_mean + m * global_mean) / (n + m)`, where `m` controls shrinkage.

**Branch point:** if the category is an identifier (patient, physician, facility with unique
codes), do not target-encode it. An identifier with one observation per level encodes the outcome
perfectly in training and is useless out of sample. Drop it or treat it as a grouping variable, not
a feature.

### Step 9. Handle rare levels and single-level features

**Input:** the category counts.

**Output:** a grouping rule for rare levels and a list of features to drop.

**Rationale:** A single-level categorical feature has no variance and cannot contribute to a model;
some encoders fail on it. A rare level with five observations produces unstable estimates and can
cause the encoder to fail when the level is absent from a fold. Grouping rare levels into "other"
preserves the information that the level exists without estimating a separate effect.

**Branch point:** if a level is rare in development but common at deployment, grouping it as
"other" loses the signal that matters most. Check the deployment site's category distribution
before fixing the grouping threshold.

### Step 10. Remove zero-variance and near-zero-variance features

**Input:** the training fold.

**Output:** a list of features removed, with the variance and the reason.

**Rationale:** A constant feature carries no information and can break scalers or encoders. A
near-constant feature can produce unstable coefficients and misleading importance. Removal should
be based on the training fold only; a feature that is constant in training but varies in validation
is a transportability warning, not a feature to keep.

**Branch point:** if a feature is constant in training because of a data extraction bug, removing
it hides the bug. Check the source before removing.

### Step 11. Scale and transform where the model requires it

**Input:** the model family and the feature distributions.

**Output:** a scaling decision per feature, with the parameters fitted on the training fold.

**Rationale:** Regularised linear models, support vector machines, k-nearest neighbours, and neural
networks are sensitive to feature scale. Tree-based models are not. Scaling a tree model adds a
transformation with no benefit and a maintenance cost. For linear models, standardisation also
makes coefficients comparable, which matters for interpretability.

**Branch point:** if the feature is skewed, consider a log or Box-Cox transform before scaling. If
the feature has outliers, consider robust scaling. Do not apply a transform that requires positive
values to a feature with zeros or negatives.

### Step 12. Handle outliers and implausible values

**Input:** the feature distributions and clinical plausibility ranges.

**Output:** a rule for clipping, winsorisation, or setting implausible values to missing, with the
count affected.

**Rationale:** Clinical data contain values that are physiologically impossible (a heart rate of
900) or unit errors (a creatinine in micromoles reported as milligrams). These values distort
scaling and can dominate a linear model. The correct treatment depends on whether the value is a
data error or a genuine extreme: errors should be corrected or set to missing, genuine extremes
should be retained or winsorised with a sensitivity analysis.

**Branch point:** if more than a small fraction of values fall outside the plausibility range, the
unit or the extraction is wrong. Fix the source rather than clipping the symptom.

### Step 13. Enforce point-in-time correctness and unit harmonisation

**Input:** the feature timestamps and units.

**Output:** a point-in-time audit and a unit conversion table.

**Rationale:** A feature is only valid if it is available at time zero. A laboratory value with a
timestamp after the prediction time, a medication order placed because of the outcome, or a
diagnosis code entered at discharge is leakage regardless of how it is preprocessed. Units must be
harmonised because a model trained on milligrams cannot consume micromoles, and because a site that
reports a different unit will silently produce implausible predictions.

**Branch point:** if a feature is available only after time zero for some patients and before for
others, the feature is not usable in its current form. Either restrict to the patients with
pre-time-zero values or drop the feature.

### Step 14. Freeze, version, and audit the pipeline

**Input:** the fitted pipeline and the audit results.

**Output:** a serialised pipeline, a version identifier, and an audit report covering fit scope,
missingness handling, encoding, scaling, and point-in-time correctness.

**Rationale:** A pipeline that is not versioned cannot be reproduced, and a pipeline that is not
audited cannot be defended. The audit should be run as part of the validation, not after it. The
serialised artefact should include the fitted parameters, the software versions, and the fallback
rules for unseen categories.

**Branch point:** if the pipeline changes after validation, the validated performance no longer
applies. Re-run the validation or version the change as a new model.

### Preprocessing edge cases

**A feature that is 100 percent missing in a subgroup.** Imputing the global median for that
subgroup produces a constant within the subgroup and can distort subgroup performance. Report the
subgroup missingness and consider a subgroup-specific model or exclusion.

**A new category at deployment.** The encoder must have a fallback. For one-hot, all columns are
zero. For target encoding, use the global mean. For ordinal encoding, use a reserved "unknown"
code. Test the fallback before deployment.

**A single-level category in a fold.** Some encoders raise an error. Drop the feature within that
fold or use an encoder that handles constant columns, and record the decision.

**Zero variance in a fold but not in the full data.** This is a small-sample artefact, not a reason
to drop the feature globally. Fit the variance check inside the fold but decide feature inclusion
from the full training set with a documented rule.

**Perfect separation from a missing indicator.** If the indicator alone separates the outcome, the
missingness is the outcome in disguise. Check whether the value is missing because of the outcome,
which is label leakage.

**Tiny subgroup with imputation.** A subgroup with 30 patients and 45 percent missingness cannot
support a subgroup-specific imputation model. Use the global model, report the subgroup
missingness, and avoid subgroup claims.

**Unit change at a site.** A new assay or a different unit changes the feature distribution. The
pipeline's scaler will mis-scale. Detect by comparing feature distributions by site before scoring.

**Prevalence shift.** Imputation and encoding parameters are prevalence-dependent when they use the
outcome. Re-fit target encodings on local data only if the local prevalence is stable and the
governance allows it; otherwise use development parameters and recalibrate the model.

**A category present in training but absent in a fold.** Grouped or temporal folds can exclude an
entire category. The encoder must tolerate a missing level at fit time; if it cannot, the fold
fails. Test the encoder on each fold before running the full loop.

**An indicator that is constant in a fold.** A missing indicator for a feature that is never
missing in a particular fold is constant and will be dropped by a variance filter. That is
correct within the fold, but it means the model's feature set differs across folds. Record the
difference; a feature set that changes across folds is a stability warning.

**Empty fold after preprocessing.** A fold can lose all events after listwise deletion. The
performance metric is undefined. Impute or merge the fold; do not report an infinite metric.

## 6. Worked example

### The setting

A heart failure readmission model uses 22 features from three hospitals. The features include
demographics, vitals, laboratory values, ejection fraction, medication counts, prior utilisation,
and discharge disposition. The cohort is 24,800 admissions from 2019-2022, with 4,712 events
(prevalence 19.0 percent).

### The missingness inventory

| Feature | Missing overall | Missing at site A | Missing at site C | Notes |
|---|---:|---:|---:|---|
| Creatinine | 18% | 14% | 29% | Not tested in some short stays |
| Ejection fraction | 41% | 33% | 58% | Echocardiogram not always ordered |
| Sodium | 6% | 5% | 9% | Routine lab |
| Medication count | 3% | 2% | 6% | Pharmacy feed gap at C |
| Discharge disposition | 0% | 0% | 0% | Always coded |

Complete cases across all 22 features number 11,200, or 45.2 percent of the cohort. Their
readmission prevalence is 22.4 percent (2,509 events), against 19.0 percent in the full cohort. The
complete-case subset is sicker, because sicker patients receive more tests and therefore have fewer
missing values. Complete-case analysis would train on a different population from the one the model
will serve.

### The mechanism assessment

Creatinine missingness is mostly MAR: it depends on length of stay and on whether a chemistry panel
was ordered, both observed. Ejection fraction missingness is closer to MNAR: clinicians order an
echocardiogram when they suspect cardiac dysfunction, so the absence of a value is associated with
lower risk, but the association is not fully captured by observed variables. Sodium is close to
MCAR. Medication count missingness is a site-specific data feed issue, which is structural at site
C and incidental elsewhere.

### The strategies

The team chooses:

- Creatinine: multiple imputation with chained equations, using age, length of stay, sodium, and
  prior admission count as auxiliary variables.
- Ejection fraction: multiple imputation plus a missing indicator, with a sensitivity analysis that
  shifts imputed values downward by 10 percentage points to represent the MNAR possibility.
- Sodium: median imputation.
- Medication count: median imputation plus a site indicator, flagged for fairness review because
  the indicator is a site proxy.
- Discharge disposition: no imputation.

All imputations are fitted inside each training fold. The multiple imputation model does not use the
outcome, following evidence that outcome-inclusive imputation can distort predictions when
missingness also occurs at deployment.

### The results

| Approach | n | AUROC (95% CI) | Calibration slope | Prevalence |
|---|---:|---|---:|---:|
| Complete case | 11,200 | 0.671 (0.657-0.685) | 0.97 | 22.4% |
| Median imputation | 24,800 | 0.681 (0.667-0.695) | 0.88 | 19.0% |
| Multiple imputation (M = 20) | 24,800 | 0.686 (0.672-0.700) | 0.94 | 19.0% |
| Multiple imputation plus fold-safe target encoding | 24,800 | 0.686 (0.672-0.700) | 0.93 | 19.0% |
| Multiple imputation plus leaky target encoding | 24,800 | 0.731 (0.718-0.744) | 0.81 | 19.0% |

The complete-case model has the best calibration slope but the worst AUROC, and it is estimated on
a population with a 3.4 percentage point higher prevalence. Its apparent advantage in calibration
is partly a consequence of evaluating on the same selected population. The multiple imputation
model recovers most of the discrimination and keeps calibration close to 1.

### The target encoding arithmetic

The team considers target-encoding the discharging ward, which has 180 levels. With smoothing
`m = 20` and a global event rate of 0.19, a ward with 12 patients and an observed rate of 0.42 is
encoded as:

```text
(12 * 0.42 + 20 * 0.19) / (12 + 20)
= (5.04 + 3.80) / 32
= 8.84 / 32
= 0.276
```

A ward with 3 patients and an observed rate of 1.00 is encoded as:

```text
(3 * 1.00 + 20 * 0.19) / (3 + 20)
= (3.00 + 3.80) / 23
= 6.80 / 23
= 0.296
```

The smoothing pulls both estimates toward the global rate. Without smoothing, the 3-patient ward
would be encoded at 1.00, which is almost certainly noise. With smoothing, it is close to the
12-patient ward, which is appropriate given the uncertainty.

### The leakage arithmetic

If the encoding is fitted on all 24,800 admissions, including the validation fold, each validation
patient's own outcome contributes to the encoding of their ward. The model's AUROC rises from 0.686
to 0.731. The inflation of 0.045 is larger than the difference between the complete-case and
multiple-imputation models. A team that compared encodings on the leaky estimate would conclude
that target encoding is clearly better, when most of the improvement is the test labels leaking
into the feature.

### Two diagnostics that would reveal a broken pipeline

**Diagnostic 1: complete-case prevalence.** The complete-case prevalence is 22.4 percent against
19.0 percent in the full cohort. That difference is the first sign that complete-case analysis has
changed the population. A pipeline report that shows only the model metrics and not the cohort
characteristics by missingness pattern misses it. The fix is to report prevalence and key
covariates for complete cases, imputed cases, and the full cohort.

**Diagnostic 2: calibration by missingness pattern.** The overall calibration slope for median
imputation is 0.88, which looks acceptable. Stratified by ejection-fraction missingness, the slope
is 1.02 for patients with a measured ejection fraction and 0.71 for patients without one. The
model under-predicts risk in the missing group, because median imputation assigns an average
ejection fraction to patients whose ejection fraction was not measured and who are, on average,
lower risk in the observed data but higher risk in the unobserved tail. The diagnostic is a
calibration plot by missingness pattern, not a single overall slope.

### Varying one assumption: MNAR instead of MAR

Suppose ejection fraction is missing because clinicians do not order an echocardiogram for
patients they judge to be at low risk, but a subset of those patients have occult dysfunction. The
MAR imputation assigns the average ejection fraction conditional on observed variables, which
overstates function for that subset. The team runs a sensitivity analysis that subtracts 10
percentage points from the imputed ejection fraction for missing patients. Under that assumption,
the predicted risk for 12 percent of the missing group rises above the 0.20 threshold, and the
model's sensitivity at the threshold increases from 0.62 to 0.68 while PPV falls from 0.34 to 0.31.
The conclusion changes from "the model is adequately calibrated" to "the model is calibrated on
average but under-predicts a clinically important subgroup". Reporting the MAR result alone would
have hidden the assumption.

### The decision at the threshold

At a threshold of 0.20 and a prevalence of 19.0 percent, the multiple-imputation model has
sensitivity 0.62 and specificity 0.72. PPV is:

```text
0.62 * 0.19 / (0.62 * 0.19 + 0.28 * 0.81)
= 0.1178 / (0.1178 + 0.2268)
= 0.1178 / 0.3446
= 0.342
```

Net benefit is:

```text
0.62 * 0.19 - 0.28 * 0.81 * (0.20 / 0.80)
= 0.1178 - 0.2268 * 0.25
= 0.1178 - 0.0567
= 0.0611
```

The complete-case model, evaluated on its own selected cohort, has sensitivity 0.60, specificity
0.70, and prevalence 22.4 percent. Its PPV is:

```text
0.60 * 0.224 / (0.60 * 0.224 + 0.30 * 0.776)
= 0.1344 / (0.1344 + 0.2328)
= 0.366
```

The complete-case model looks better on PPV, but the comparison is invalid because the populations
differ. Applied to the full cohort, its performance would be lower, and its calibration slope would
move away from 0.97. This is the practical reason to prefer the imputed model even when the
complete-case number looks more favourable.

### What to report

The primary preprocessing figure is the missingness heatmap by feature and site, which makes the
mechanism visible. The secondary figure is the calibration plot stratified by missingness pattern.
The table of imputation strategies belongs in the appendix, with the auxiliary variables and the
fit scope. The model card should state the imputation method, the encoding method, the scaling
method, and the fallback for unseen categories, because those four choices define the deployed
pipeline as much as the algorithm does.

## 7. Code, query, or calculation

The script demonstrates three preprocessing operations with the standard library: fold-safe target
encoding, leaky target encoding, and median imputation fitted on the training fold. It uses a
synthetic cohort with 200 categories and a missing laboratory value. The score is a fixed linear
combination, so the difference in AUROC between the two encodings is attributable to leakage, not
to model fitting.

```python
"""Preprocessing arithmetic: fit on train, transform test. Standard library only."""
import math
import random
import statistics


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


def target_means(records, smoothing=20.0):
    """Smoothed category means; fit this on the training fold only."""
    global_mean = sum(r["y"] for r in records) / len(records)
    groups = {}
    for r in records:
        groups.setdefault(r["cat"], []).append(r["y"])
    mapping = {}
    for category, labels in groups.items():
        n = len(labels)
        mapping[category] = (n * (sum(labels) / n) + smoothing * global_mean) / (n + smoothing)
    return mapping, global_mean


def encode(records, mapping, global_mean, weight=4.0):
    return [(weight * mapping.get(r["cat"], global_mean) + 0.5 * r["x"], r["y"])
            for r in records]


def median_impute(train_values, test_values):
    observed = [v for v in train_values if v is not None]
    fill = statistics.median(observed)
    return fill, [fill if v is None else v for v in test_values]


rng = random.Random(5)
records = []
for i in range(2000):
    category = i % 200
    effect = 0.6 if category % 5 == 0 else -0.1
    x = rng.gauss(0, 1)
    probability = 1 / (1 + math.exp(-(-1.2 + effect + 0.4 * x)))
    label = 1 if rng.random() < probability else 0
    lab = x + 1.0 if label == 1 else x
    if rng.random() < (0.35 if label == 1 else 0.15):
        lab = None
    records.append({"cat": category, "x": x, "y": label, "lab": lab})

train, test = records[:1500], records[1500:]
leaky_map, global_mean = target_means(records)
safe_map, _ = target_means(train)
print(f"prevalence={sum(r['y'] for r in records) / len(records):.3f}")
print(f"leaky target encoding AUROC={auroc(encode(test, leaky_map, global_mean)):.3f}")
print(f"fold-safe target encoding AUROC={auroc(encode(test, safe_map, global_mean)):.3f}")
fill, imputed = median_impute([r["lab"] for r in train], [r["lab"] for r in test])
full_median = statistics.median([r["lab"] for r in records if r["lab"] is not None])
print(f"median fit on train={fill:.3f} median on all data={full_median:.3f}")
print(f"imputed test values={sum(1 for r in test if r['lab'] is None)} of {len(test)}")
```

Expected output:

```text
prevalence=0.251
leaky target encoding AUROC=0.656
fold-safe target encoding AUROC=0.588
median fit on train=0.096 median on all data=0.104
imputed test values=103 of 500
```

Reading the output:

- The leaky encoding gives an AUROC of 0.656; the fold-safe encoding gives 0.588. The difference of
  0.068 is leakage, because the only change is whether the test patients' labels contributed to the
  category means.
- The gap would grow if the smoothing were removed. With `smoothing=0` and 200 categories, the
  leaky encoding can approach perfect separation on the test set, because each category's mean is
  dominated by a handful of labels including the test patient's own.
- The median fitted on the training fold is 0.096; the median on all data is 0.104. The difference
  is small here because missingness is only weakly related to the value. In a cohort where
  missingness depends strongly on severity, the difference can be large enough to shift predictions.
- The test set has 103 missing values out of 500, which are imputed with the training median. That
  is the correct order: the median is a training parameter, and the test values are transformed
  with it.

The reusable functions are `target_means` and `median_impute`. Both take training data as the
source of parameters and return a transform that can be applied to test data. Neither function
should ever be called with the full dataset before splitting.

## 8. Data, standards, and regulatory context

Preprocessing is constrained by data standards and by reporting and regulatory expectations. The
table distinguishes binding requirements from conventions.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| TRIPOD+AI items on predictors and model building | Reporting convention | Requires predictor definitions, measurement timing, and preprocessing to be reported | Does not prescribe a method |
| PROBAST+AI predictors domain | Appraisal tool | Assesses predictor availability, measurement, and handling | Does not validate the pipeline |
| FDA GMLP principles | Guidance, not binding | Calls for representative data, appropriate feature engineering, and testing | Does not specify imputation |
| FDA PCCP guidance | Guidance, not binding | Requires planned changes to preprocessing to be described and validated | Does not cover initial pipeline design |
| ISO 13485 | Voluntary standard, often harmonised | Design controls and configuration management | Does not specify statistical methods |
| IEC 62304 | Voluntary standard, often harmonised | Software lifecycle and change control | Does not cover data quality |
| LOINC | Terminology standard | Laboratory observation identity | Does not enforce units |
| UCUM | Unit standard | Machine-readable units and conversions | Does not validate conversion factors |
| FHIR | Interoperability standard | Resource structure and timestamps | Does not validate completeness |
| OMOP CDM | Common data model | Table structure and concept identifiers | Does not prescribe preprocessing |
| CDISC SDTM and ADaM | Binding for many regulatory submissions | Data structure and analysis flags | Does not cover EHR extraction |
| HIPAA Privacy Rule | Binding in the US | Use and disclosure of protected health information | Does not govern preprocessing validity |

Three points matter for practice.

**TRIPOD+AI expects the pipeline to be describable.** The checklist asks for predictor definitions,
measurement timing, and the model-building procedure, which includes preprocessing. A report that
says "missing values were imputed" without saying how, fitted on what, and with what auxiliary
variables does not meet the spirit of the item. The same applies to encoding and scaling.

**Units are a patient-safety issue, not a formatting issue.** A model trained on creatinine in
mg/dL and deployed where the laboratory reports micromoles will produce implausible predictions
unless the pipeline converts units. LOINC identifies the observation, and UCUM provides the
machine-readable unit, but neither enforces the conversion. The pipeline must include an explicit
conversion table and a validation check against the expected range.

**A pipeline is a medical device component when the model is one.** If the model is a regulated
device, the preprocessing that feeds it is part of the device software. A change to an imputation
method or an encoding vocabulary is a software change, and it should follow the same change-control
process as a change to the model coefficients. Treating preprocessing as an unregulated utility is
a common source of inspection findings.

**Configuration management applies to the pipeline, not only the model.** ISO 13485 and IEC 62304
expect design controls and change control for the software artefact. A pipeline whose imputation
parameters are recomputed at deployment is a different artefact from the one that was validated,
unless the change is covered by a predetermined change control plan. The version identifier should
cover the preprocessing as well as the model weights.

### Unit harmonisation in practice

Unit errors are among the few preprocessing failures that can directly harm patients, because a
model can produce a plausible-looking but wrong risk from a value that is off by a known factor.
The safeguards are mechanical: store the unit alongside every measurement, convert to a canonical
unit at ingestion, validate the converted value against a clinical range, and log the count of
values that fail the range check by source. A pipeline that does not do this will eventually
encounter a laboratory that reports haemoglobin in grams per litre rather than grams per decilitre,
or a creatinine in micromoles rather than milligrams.

The canonical unit should be chosen once per feature and recorded in the data dictionary. The
conversion table should be versioned, because conversion factors are not always exact and because
a laboratory may change its reporting unit. A range check is not a substitute for conversion, but
it catches the failure mode where conversion was forgotten.

### A preprocessing specification checklist

A pipeline that can be validated and transported states the following for each step. The list is a
convention, not a regulatory requirement, but it maps onto the TRIPOD+AI predictor items and the
PROBAST+AI predictor domain.

```text
feature name and type
source table, code system, and unit
lookback window and time-zero rule
missingness rate overall and by site
missingness mechanism and evidence
imputation method, fit data, auxiliary variables
missing indicator decision and rationale
encoding method, cardinality, fallback
scaling method and fit data
outlier rule and count affected
unit conversion and range check
version identifier and change log
```

> **Note:** The EU AI Act's data governance obligations for high-risk systems include training,
> validation, and testing data quality. For medical devices, those obligations interact with the
> MDR's clinical evaluation requirements. Neither instrument prescribes a specific imputation
> method.

## 9. Industry practice and case evidence

**Missing data handling in machine-learning prediction studies.** Nijman et al. (2021) reviewed 152
machine-learning prediction studies and found that, among the 96 that reported a method, 43 used
deletion or complete-case analysis and only 8 used multiple imputation. Complete-case analysis is
valid only under stringent conditions, and the review notes that it can cause bias and loss of
power. The practical implication is that many published models are estimated on a selected
population, and their reported prevalence and performance may not transport.

**Complete case versus multiple imputation.** Hughes et al. (2019) show that complete-case analysis
is unbiased for many regression models when the probability of being complete does not depend on
the outcome after conditioning on covariates, including some MNAR situations. Multiple imputation
assuming MAR is biased in those same situations. The lesson is that multiple imputation is not
universally superior; the choice depends on the mechanism, the missingness pattern, and the
auxiliary data.

**Missing indicators in prediction models.** Sisk et al. (2023) found that, for prediction rather
than causal inference, the best-performing imputation specification sometimes omitted the outcome
from the imputation model, and that missing indicators helped under MNAR but harmed under
outcome-dependent missingness. The finding complicates the common advice to "always add a missing
indicator". The indicator is a modelling choice with its own assumptions.

**Target encoding leakage.** The scikit-learn documentation demonstrates the problem with a
near-unique categorical feature: using `fit` and `transform` produces a training R-squared of about
0.86 and a test R-squared of about 0.63, while `fit_transform` with internal cross-fitting produces
0.80 and 0.79. The example is a clean demonstration that the encoding, not the model, caused the
apparent overfitting.

**Regularised target encoding.** Pargent et al. (2022) benchmarked seven encoders across five
algorithms and 24 datasets and found that regularised target encoding, using a mixed model with
cross-validation, consistently ranked best. Simple target encoding overfits rare levels, and in the
extreme unique-identifier case it can perfectly predict training rows while being useless out of
sample. The practical recommendation is smoothing plus cross-fitting, not smoothing alone.

**Imputation in deployed sepsis models.** A sepsis model that uses lactate, creatinine, and white
cell count faces a deployment population in which those values are measured only when ordered. A
median imputation fitted on the development cohort assigns a normal value to a patient who has not
yet been tested, which can lower the predicted risk at the moment the clinician is deciding whether
to test. The pipeline therefore interacts with the workflow: the act of ordering a test changes the
input to the model. This is a form of informative missingness that no purely statistical
imputation can resolve, and it is one reason some deployed models recompute as new results arrive.

**Missingness in critical care databases.** MIMIC and eICU contain substantial missingness because
tests are ordered for clinical reasons. A laboratory value is more likely to be measured in sicker
patients, which makes the missingness informative. Models trained on these databases without
accounting for the ordering process can learn "was tested" as a proxy for severity. This is an
inference from the data-generating process, not a finding of a specific paper, but it is consistent
with the documented use of these databases.

**Encoding choices and fairness.** A one-hot encoding of a category that is a proxy for a protected
attribute allows the model to learn group-specific effects directly. Target encoding can do the
same more subtly, because the category mean reflects the outcome distribution of the group. Neither
is inherently unfair, but both require a subgroup performance review. The preprocessing choice
determines which group effects are representable, and therefore which fairness questions can be
asked of the model.

**Why pipelines fail at the second site.** A pipeline developed at one site encodes the local
laboratory feed, the local formulary, and the local missingness pattern. At a second site, the same
feature name may come from a different source table, the units may differ, and the missingness may
be structurally different. The model does not fail because the algorithm is wrong; it fails because
the pipeline's assumptions were local. The remedy is a site-readiness checklist that compares the
feature distributions and missingness patterns before go-live, not after the first implausible
score.

> **Note:** The cases are documented. The general claim that preprocessing leakage is common is
> supported by the scikit-learn example and the review by Nijman et al.; the claim that a specific
> deployed model failed because of preprocessing is not made here.

## 10. Failure modes and red flags

Preprocessing failures are usually silent: the pipeline runs, the model trains, and the metrics
look plausible. The table pairs each failure with the evidence that would expose it, so that the
audit in Step 14 can be specific rather than a general review.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Imputation fitted on all data | Median or model fitted before split | Test distribution enters training | Optimistic validation | Fit inside each fold |
| Target encoding fitted on all data | Encoding map built before split | Test labels enter the feature | Severe optimism | Fold-safe encoding |
| Missing indicator encodes site | Indicator correlates with site or race | Proxy for structural difference | Bias and transport failure | Fairness review; drop or stratify |
| Complete-case analysis changes population | Complete-case prevalence differs from full | Selection on testing or data availability | Model trained on sicker patients | Multiple imputation or weighting |
| Mean imputation for skewed data | Imputed values outside plausible range | Mean pulled by outliers | Distorted predictions | Median or model-based imputation |
| Structural missingness imputed clinically | Male patients get pregnancy risk | Feature does not apply | Nonsense values | Encode "not applicable" |
| High-cardinality one-hot | Thousands of sparse columns | Dimensionality explosion | Overfitting and slow scoring | Target encoding or hashing |
| Rare level encoded without smoothing | A 3-patient category has rate 1.0 | Noise treated as signal | Unstable predictions | Smoothing and grouping |
| Zero-variance feature retained | Scaler or encoder error | No information | Numerical instability | Remove within fold |
| Scaling fitted on test | Mean and SD include test | Distribution leakage | Small optimism | Fit scaler on train |
| Unit mismatch at deployment | Predictions implausible at one site | mg/dL versus micromoles | Patient-safety risk | Unit conversion and range checks |
| Post-time-zero feature retained | A lab after time zero is predictive | Temporal leakage | Inflated performance | Point-in-time audit |
| New category at deployment | Encoder returns missing or error | No fallback | Scoring failure | Explicit fallback |
| Pipeline not versioned | Model artefact lacks preprocessing version | Change not tracked | Unreproducible results | Version the full pipeline |
| Schema change at deployment | Missing column or changed type | Pipeline assumes development schema | Silent default or crash | Schema validation with hard failure |
| Distribution shift in a feature | Scores shift without model change | New assay, unit, or case mix | Miscalibrated risk | Monitor inputs and recalibrate |
| Imputation model uses the outcome | Imputed values correlate too strongly with label | Outcome leakage into features | Inflated performance | Exclude outcome from imputation or fit inside fold |
| Encoding cardinality mismatch | Column count differs between train and deploy | New levels expand the matrix | Scoring error | Fixed vocabulary with fallback |
| Outlier rule applied inconsistently | Different clipping at train and deploy | Rule not versioned | Distribution shift | Store the rule with the pipeline |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The missingness mechanism is correctly classified | Determines the valid method | MNAR mistaken for MAR | Sensitivity analysis; chart review |
| The imputation model uses deployment-available variables | Ensures the pipeline works in production | A post-time-zero auxiliary variable | Audit auxiliary variables |
| Category means are stable across folds | Target encoding assumes transport | Rare or drifting categories | Cross-fitting; smoothing; fold checks |
| The training distribution matches deployment | Scalers and imputers transport | Different assay, unit, or case mix | Compare distributions by site |
| Missingness at deployment matches development | Imputation parameters apply | Development-only missingness | Prospective missingness monitoring |
| Structural missingness is identified | Prevents nonsense imputation | Conditional features treated as random | Clinical review of each feature |
| Scaling is required by the model | Avoids unnecessary transforms | Tree model scaled for no reason | Model-family check |
| Outlier handling preserves genuine extremes | Avoids discarding signal | Clipping removes real high-risk values | Sensitivity analysis with and without |
| The pipeline is deterministic | Reproducibility | Random imputation without a seed | Re-run with the same seed |
| The versioned artefact includes preprocessing | Validation applies to the deployed object | Model updated without pipeline | Configuration audit |
| The deployment schema matches development | The pipeline can find its inputs | Renamed or removed columns | Schema validation |
| The outlier rule is stable | Clipping does not change between sites | Different thresholds applied | Versioned rule and range check |

A second limit is that the pipeline assumes the deployment data arrive in the same schema. A new
column name, a changed unit, a reordered code set, or a missing source table will break the
pipeline before the model sees a single patient. Schema validation is therefore part of
preprocessing, not part of integration engineering. The pipeline should fail loudly when an
expected column is absent or a value falls outside the expected range, rather than silently
imputing the default.

The pipeline is also a record of what the team knew at development time, and of what it chose to
ignore. When a feature is added,
removed, or redefined, the pipeline version is the only reliable way to reconstruct which model saw
which inputs. Without that record, a performance comparison across model versions is not
interpretable, because the inputs may have changed as much as the algorithm.

The deepest limit is that imputation creates data that were never observed. Every imputed value is
a model-based prediction, and the downstream model treats it as if it were measured. That is
acceptable when the imputation is good and the uncertainty is propagated; it is not acceptable when
the imputed values are treated as ground truth. Multiple imputation propagates the uncertainty;
single imputation does not. The report should say which was used.

## 12. Exercises

The exercises are designed to be done with a real extract. If none is available, generate a
synthetic cohort with the missingness pattern and cardinalities stated in each exercise, and treat
the resulting numbers as illustrative rather than as evidence.

Exercises 1 and 2 are diagnosis, 3 through 5 are encoding and imputation arithmetic, 6 and 7 are
audits, and 8 is the specification you can reuse in the module assignment. The exercises assume
access to a dataset; where none is available, a synthetic cohort with the stated missingness
pattern works.

**Exercise 1. Missingness inventory.** For a dataset you can access, compute missingness by feature,
site, and outcome, and classify each feature's mechanism. Deliverable: a table and a heatmap.
Acceptance criteria: at least 10 features; mechanism justified; structural missingness identified.
Time: 60 minutes.

**Exercise 2. Imputation comparison.** Compare complete-case, median, and multiple imputation on
the same cohort. Report cohort prevalence, AUROC, and calibration slope for each. Deliverable: a
results table and a 300-word interpretation. Acceptance criteria: prevalence differences reported;
fit scope stated; recommendation justified. Time: 90 minutes.

**Exercise 3. Smoothing arithmetic.** A category has 8 observations, a mean outcome of 0.50, and the
global mean is 0.12. Compute the smoothed encoding for m = 5, 20, and 100. Deliverable: three
values and a sentence on the effect of m. Acceptance criteria: values correct to three decimals;
interpretation states the direction of shrinkage. Time: 30 minutes.

**Exercise 4. Target encoding audit.** Given a pipeline that target-encodes a physician identifier
with 300 levels, identify the leakage and propose a safe alternative. Deliverable: a 400-word
audit. Acceptance criteria: leakage mechanism named; alternative is implementable; fallback for
unseen levels stated. Time: 45 minutes.

**Exercise 5. Encoding decision.** For five categorical features with cardinalities 2, 8, 30, 200,
and 5,000, choose an encoding and justify it. Deliverable: a five-row table. Acceptance criteria:
each choice matches cardinality; dimensionality and interpretability discussed. Time: 30 minutes.

**Exercise 6. Point-in-time audit.** For a feature list, identify which features could be recorded
after time zero and propose a test for each. Deliverable: an audit table. Acceptance criteria: at
least eight features; tests are specific; leakage direction stated. Time: 60 minutes.

**Exercise 7. Unit harmonisation.** Two sites report creatinine in mg/dL and micromoles per litre.
Write the conversion and a validation rule. Deliverable: the conversion factor, a range check, and
a test case. Acceptance criteria: factor correct (88.4); range check catches unit errors; test case
documented. Time: 30 minutes.

**Exercise 8. Pipeline specification.** Write the preprocessing section of a model development
report: missingness, imputation, encoding, scaling, outlier handling, units, and versioning.
Deliverable: a two-page specification. Acceptance criteria: every step states fit data and
fallback; leakage controls explicit; reproducibility addressed. Time: 90 minutes.

## 13. Self-check answers

**Answer to Exercise 3.** For m = 5: `(8 * 0.50 + 5 * 0.12) / 13 = (4.0 + 0.6) / 13 = 0.354`. For
m = 20: `(4.0 + 2.4) / 28 = 6.4 / 28 = 0.229`. For m = 100: `(4.0 + 12.0) / 108 = 16.0 / 108 =
0.148`. As m increases, the encoding shrinks toward the global mean of 0.12. The choice of m is a
bias-variance trade-off: small m preserves category signal but is noisy, large m is stable but
ignores the category.

**Answer to Exercise 1 (what a strong inventory shows).** A strong inventory reports missingness
for every feature, not only the ones with obvious gaps, and breaks it down by site, year, and
outcome. It distinguishes structural missingness (a feature that does not apply) from incidental
missingness (a value that was not measured), and it states the evidence for each classification.
It also reports the complete-case fraction, because that number determines whether complete-case
analysis is even feasible.

**Answer to Exercise 7.** Creatinine in mg/dL is converted to micromoles per litre by multiplying
by 88.4. A validation rule should flag values outside a plausible range, for example below 10 or
above 2,000 micromoles per litre, and should compare the distribution at each site against the
development distribution. A test case is a patient with 1.2 mg/dL, which should become 106.1
micromoles per litre. The conversion must be applied before imputation, because imputing in the
wrong unit produces a value that is wrong by a factor of 88.4.

**Answer to Exercise 2 (what a strong comparison shows).** A strong comparison reports the cohort
size and prevalence for each approach, because complete-case analysis changes the population. It
reports AUROC and calibration with intervals, states that all imputations were fitted inside the
cross-validation folds, and interprets the difference in light of the selection. It recommends the
approach that matches the deployment missingness pattern, not necessarily the one with the highest
AUROC.

**Answer to Exercise 4.** The physician identifier has 300 levels, and each level has few patients.
Target-encoding it on the full dataset gives each physician an outcome rate that includes the
validation patients, so the encoding carries test labels. Even inside the training fold, a
physician with three patients and two events is encoded near 0.67, which is noise. The safe
alternative is to drop the identifier as a feature and use it only as a grouping variable for
cross-validation; if the physician's practice pattern is genuinely predictive, replace the
identifier with interpretable aggregates such as panel size, years since training, or
case-mix-adjusted rate, computed without the outcome. If a target encoding is retained, it must be
fitted with cross-fitting inside the training fold, smoothed heavily, and given a global-mean
fallback for unseen physicians.

**Answer to Exercise 5 (illustrative choices).** Cardinality 2: one-hot or a single binary column,
since either is equivalent. Cardinality 8: one-hot, with a check for rare levels. Cardinality 30:
one-hot with rare-level grouping, or ordinal encoding if the categories are ordered. Cardinality
200: fold-safe target encoding with smoothing, or hashing if interpretability is not required.
Cardinality 5,000: hashing or target encoding with a hierarchical model; one-hot would create
5,000 columns and almost certainly overfit. In every case, the choice should be recorded with the
fallback for unseen levels.

**Rubric for Exercises 1, 4, 5, 6, and 8.** A strong submission (1) classifies the mechanism, (2)
fits every transform inside the fold, (3) chooses an encoding that matches cardinality, (4)
provides a fallback for unseen values, (5) enforces units and point-in-time correctness, and (6)
versions the pipeline. A competent submission meets four of the six. A weak submission imputes
before splitting or one-hot encodes a high-cardinality identifier.

## 14. Interview questions

**1. Why is imputing before splitting a problem?** A strong answer explains that the imputation
parameters are estimated from the test data, so the validation estimate is optimistic, and gives the
median example. A weak answer says it is "bad practice" without a mechanism.

**2. How do you choose between complete-case and multiple imputation?** A strong answer discusses
the missingness mechanism, the deployment pattern, and the auxiliary variables, and notes that
complete case can be unbiased in some settings. A weak answer says multiple imputation is always
better.

**3. What is target encoding and when does it leak?** A strong answer defines it, explains
fold-safety and smoothing, and gives the identifier example. A weak answer says it is a way to
encode categories without saying how it leaks.

**4. How do you handle a categorical feature with 5,000 levels?** A strong answer considers target
encoding with smoothing, hashing, rare-level grouping, and dropping identifier-like features. A
weak answer says one-hot encode it.

**5. What is a missing indicator and when would you not use one?** A strong answer explains that it
can encode informative missingness but can also proxy for site or protected attributes. A weak
answer says always add one.

**6. How do you handle structural missingness?** A strong answer encodes "not applicable" and
avoids imputing a clinical value. A weak answer imputes the median.

**7. What is point-in-time correctness?** A strong answer defines time zero and explains that a
feature recorded after it is leakage, with an example. A weak answer says it means using recent
data.

**8. Why does scaling matter for some models and not others?** A strong answer names the
distance-based and regularised models that need it and the tree models that do not. A weak answer
says scaling always improves performance.

**9. How do you handle a new category at deployment?** A strong answer specifies a fallback for
each encoding and a test before go-live. A weak answer says retrain the model.

**10. How do you handle a laboratory value that is missing because the test was not ordered?** A strong answer distinguishes informative missingness from random absence, considers a missing indicator and a sensitivity analysis, and notes that the ordering behaviour may differ by site. A weak answer imputes the median without discussion.

**11. What is the difference between standardisation and normalisation?** A strong answer defines the target scale for each and names the model families that require them. A weak answer uses the terms interchangeably.

**12. How do you detect a unit error before it reaches a patient?** A strong answer uses range checks, cross-site distribution comparisons, and a conversion table with a test case. A weak answer says to check the data manually.

**13. What is the risk of target-encoding a facility identifier?** A strong answer explains that the identifier encodes local practice and outcome rates, leaks test labels if not fold-safe, and fails for unseen facilities. A weak answer says it improves performance.

**14. What would you audit in a preprocessing pipeline?** A strong answer lists fit scope,
missingness mechanism, encoding safety, scaling, units, point-in-time correctness, and versioning.
A weak answer lists only missing values.

## 15. Key terms

- **Auxiliary variable** — a variable that helps predict missingness or a missing value.
- **Cardinality** — number of distinct values in a categorical feature.
- **Complete-case analysis** — analysis restricted to records with no missing values.
- **Cross-fitting** — encoding each fold using the other folds.
- **Dummy variable trap** — perfect collinearity from keeping all one-hot columns.
- **Fit** — estimate transformation parameters from data.
- **Hashing** — mapping categories to a fixed number of columns.
- **High cardinality** — many distinct values relative to sample size.
- **Imputation model** — the model used to predict missing values.
- **Incidental missingness** — absence because a value was not measured.
- **Listwise deletion** — removing any record with a missing value.
- **MAR** — missing at random; missingness depends on observed variables.
- **MCAR** — missing completely at random.
- **Mean imputation** — replacing missing values with the mean.
- **Median imputation** — replacing missing values with the median.
- **MICE** — multiple imputation by chained equations.
- **Missing indicator** — a binary feature marking missingness.
- **MNAR** — missing not at random; missingness depends on the unobserved value.
- **Multiple imputation** — generating several plausible values and pooling results.
- **Native missing handling** — model-internal handling of missingness.
- **Near-zero-variance feature** — a feature with almost no variation.
- **New level** — a category at deployment not seen in development.
- **One-hot encoding** — one binary column per category.
- **Ordinal encoding** — integer codes for ordered categories.
- **Pipeline** — a sequence of fitted transforms and a final estimator.
- **Point-in-time correctness** — a feature is available at or before time zero.
- **Rare-level grouping** — combining infrequent categories.
- **Smoothing** — shrinking a category estimate toward the global mean.
- **Standardisation** — centring and scaling to mean 0, SD 1.
- **Structural missingness** — absence because a feature does not apply.
- **Target encoding** — replacing a category with its outcome mean.
- **Transform** — apply fixed parameters to data.
- **UCUM** — Unified Code for Units of Measure.
- **Unseen category fallback** — the value assigned to a new category.
- **Winsorisation** — capping extreme values at percentiles.
- **Zero-variance feature** — a feature with the same value for every record.

## 16. Sources

All URLs accessed 25 October 2026.

1. Nijman SWJ, Leeuwenberg AM, Beekers I, et al. Missing data is poorly handled and reported in
   prediction model studies using machine learning: a literature review. *Journal of Clinical
   Epidemiology* 2021. https://www.jclinepi.com/article/S0895-4356(21)00375-9/fulltext
2. Hughes RA, Heron J, Sterne JAC, Tilling K. Accounting for missing data in statistical analyses:
   multiple imputation is not always the answer. *International Journal of Epidemiology*
   2019;48(4):1294-1304. https://doi.org/10.1093/ije/dyz032
3. Sisk R, Lin L, Sperrin M, et al. Imputation and missing indicators for handling missing data in
   the development and deployment of clinical prediction models: a simulation study. *Statistical
   Methods in Medical Research* 2023;32(8):1461-1477. https://doi.org/10.1177/09622802231165001
4. Pargent F, Pfisterer F, Thomas J, Bischl B. Regularized target encoding outperforms traditional
   methods in supervised machine learning with high cardinality features. *Computational
   Statistics* 2022. https://arxiv.org/abs/2104.00629
5. scikit-learn developers. Target encoder's internal cross fitting.
   https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html
6. scikit-learn developers. Common pitfalls and recommended practices.
   https://scikit-learn.org/stable/common_pitfalls.html
7. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive
   analytics. *BMC Medicine* 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
8. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
9. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
   https://doi.org/10.1136/bmj-2024-082505
10. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
    Guiding Principles. 2021.
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
11. US Food and Drug Administration. Marketing Submission Recommendations for a Predetermined Change
    Control Plan for Artificial Intelligence-Enabled Device Software Functions. 2024.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
12. HL7. FHIR Release 4 specification. https://hl7.org/fhir/R4/
13. Observational Health Data Sciences and Informatics. OMOP Common Data Model.
    https://ohdsi.github.io/CommonDataModel/
14. Regenstrief Institute. LOINC. https://loinc.org/
15. National Library of Medicine. Unified Code for Units of Measure (UCUM).
    https://ucum.nlm.nih.gov/
16. Clinical Data Interchange Standards Consortium. SDTM and ADaM implementation guides.
    https://www.cdisc.org/standards
17. US Department of Health and Human Services. HIPAA Privacy Rule.
    https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
18. Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database.
    *Scientific Data* 2016;3:160035. https://doi.org/10.1038/sdata.2016.35

A final check before freezing a pipeline: list every step that learns a parameter and confirm that it
is fitted inside the training fold. If a step cannot state its fit data and its fallback, it is not
ready for deployment.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_splitting-and-validation.md`](03_splitting-and-validation.md)
- Next topic: [`05_model-families.md`](05_model-families.md)
- Metrics and prevalence: [`07_metrics.md`](07_metrics.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../course/06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
