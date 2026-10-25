# Model Families

> **Module 06 — Machine Learning Foundations**
> Topic 05 of 13
> Estimated study time: 3.5 hours
> Prerequisites: Topics 03 and 04
> Version 2026.3

## 1. Why this topic exists

Model family choice is the most visible decision in a machine-learning project and the least
consequential one relative to framing, labels, splitting, and preprocessing. Teams spend weeks
comparing gradient boosting with random forests while the validation design leaks, the label
measures documentation, and the calibration is never checked. The result is a model that wins a
leaderboard and loses a deployment.

The evidence supports a deflationary view of algorithm choice. Christodoulou et al. (2019) reviewed
71 studies containing 282 head-to-head comparisons of logistic regression and machine learning for
binary clinical prediction. In the 145 comparisons at low risk of bias, the difference in
logit(AUC) was 0.00 (95 percent CI -0.18 to 0.18). In the 137 comparisons at high risk of bias,
machine learning appeared 0.34 logit(AUC) better. The apparent advantage was concentrated where the
methods were weakest. Calibration was not addressed in 79 percent of studies, and only one used
decision curve analysis.

That does not mean model family is irrelevant. It means the family's value depends on the problem,
the data, and the decision. A linear model is often adequate when the signal is roughly additive
and the sample is modest. A gradient-boosted model can capture interactions and nonlinearities that
a linear model misses, but it needs more data, more tuning, and more calibration work. A neural
network is the right choice for images, signals, and text, and usually the wrong choice for a
20-feature tabular cohort. The skill is matching the inductive bias of the family to the structure
of the problem, not ranking families in the abstract.

A useful reframing is to ask what the model is allowed to learn. A logistic regression with main
effects is allowed to learn one number per feature. A GAM is allowed to learn a smooth curve per
feature. A tree ensemble is allowed to learn a partition of the feature space. A neural network is
allowed to learn a representation. Each additional allowance requires more data to constrain, and
each imposes a different prior on the answer. The question is which prior is closest to the truth
and affordable with the available sample.

The rest of this topic is organised around a comparison procedure, not a family hierarchy. The
procedure is repeatable across problems, while the ranking of families is not. The
procedure starts with a baseline, profiles the data, chooses a small candidate set, tunes and
evaluates honestly, and selects on evidence that includes calibration and decision utility. The
family is a means; the procedure is the skill. A team that follows the procedure with a simple
model will usually outperform a team that skips it with a complex one, and it will be able to
explain why to a clinical committee without relying on the algorithm's reputation. The procedure is the
transferable asset; the model is a dated artefact. A team can reuse the procedure; it cannot reuse
the artefact without revalidation, and the revalidation is where most of the cost sits.

Three practical pressures shape the choice.

**Sample size and events per variable.** A model with thousands of parameters needs enough events
to estimate them. A logistic regression with 20 predictors and 4,000 events is comfortable. A
gradient-boosted model with hundreds of trees on the same data may fit noise unless it is heavily
regularised. The rule of thumb that a flexible model needs roughly ten times the events of a
linear model is approximate, but the direction is right.

**Calibration and decision use.** A model whose probabilities drive a threshold must be calibrated.
Tree ensembles are often less well calibrated than logistic regression, particularly at the
extremes, and require post-hoc calibration. A model used only for ranking within a fixed capacity
can tolerate more miscalibration, but the decision analysis still matters.

**Regulatory and maintenance burden.** A sparse logistic regression can be documented, explained,
and audited with a coefficient table. A gradient-boosted model with 500 trees requires feature
attribution, stability analysis, and a change-control plan. A neural network requires even more.
The burden is not a reason to avoid a family, but it is a cost that belongs in the selection.

This topic gives you a structured comparison. You will learn the inductive bias of each family, the
data conditions under which it helps, the calibration behaviour to expect, and the procedure for
choosing among candidates using discrimination, calibration, decision utility, stability, and
subgroup performance.

A fourth pressure is less discussed: the model family determines what can be monitored after
deployment. A logistic regression's coefficients can be tracked over time and compared with the
development values; a drift in a coefficient is interpretable. A gradient-boosted model's trees
cannot be compared across versions in the same way, so monitoring shifts to the score distribution,
the calibration curve, and the feature attributions. The monitoring plan should be chosen with the
family, not after it.

The choice also affects who can review the model. A clinical committee can review a coefficient
table in an hour; it cannot review a 500-tree ensemble in the same way. If the governance process
requires committee sign-off, the family must be reviewable within that process. A model that is
technically superior but cannot be approved is not deployable.

The practical implication is that model selection is a lifecycle decision. The team is not only
choosing a function; it is choosing a maintenance regime, a documentation burden, and a set of
failure modes. A model that is 0.01 AUROC better but requires a monitoring capability the
organisation does not have is not the better choice.

> **Note:** The default recommendation in clinical tabular modelling is a well-specified logistic
> regression as the baseline, a gradient-boosted model as the flexible challenger, and a
> pre-specified rule for choosing between them. The rule should be written before the comparison.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** the inductive bias of linear, penalised, spline, tree, ensemble, kernel, neighbour,
   naive Bayes, and neural families.
2. **Match** a model family to data conditions including sample size, event count, linearity,
   interactions, and feature type.
3. **Build** a logistic regression baseline with appropriate penalties and transformations.
4. **Tune** a gradient-boosted model with a nested cross-validation protocol and report optimism.
5. **Compare** candidates on discrimination, calibration, net benefit, stability, and subgroup
   performance, and **justify** a selection.
6. **Diagnose** overfitting, underfitting, and miscalibration from learning curves and calibration
   plots.
7. **Estimate** the maintenance and regulatory burden of each candidate.
8. **Critique** a model comparison that selects on a single metric or a biased validation.

Outcomes 3, 4, and 5 are computational and assessed in the exercises. Outcomes 1 and 8 are
conceptual and assessed in the self-check and interview questions.

## 3. The core mental model

A model family is a hypothesis space plus an inductive bias. The hypothesis space is the set of
functions the model can represent. The inductive bias is the preference among those functions when
the data do not distinguish them. Two families with the same hypothesis space but different biases
will produce different models from the same data.

```text
                     high bias, low variance        low bias, high variance
                     (simple, stable)               (flexible, data-hungry)
                     ----------------------------   ----------------------------
Linear / penalised   logistic regression            ---
Additive             GAM, spline logistic           ---
Tree                 single decision tree           deep tree
Ensemble             ---                            random forest, gradient boosting
Kernel               linear SVM                     RBF SVM
Neighbour            large k                        k = 1
Neural               small MLP                      deep network
```

The diagonal from top-left to bottom-right is the capacity axis. Moving down and right increases
the range of functions the model can represent and the amount of data needed to estimate them.
Moving up and left increases stability and interpretability at the cost of missing structure.

The bias-variance decomposition makes the trade-off precise. Expected squared error decomposes into
bias squared, variance, and irreducible noise. A linear model has high bias if the true relationship
is nonlinear, but low variance. A deep tree has low bias but high variance. Ensembles reduce
variance by averaging, which is why random forests are more stable than single trees and why
gradient boosting can fit complex functions without exploding variance, provided the learning rate
and tree depth are controlled.

Three structural properties determine which family fits a problem.

**Additivity versus interactions.** A logistic regression with main effects assumes the log-odds is
additive in the features. If the effect of one feature depends on another, the linear model misses
it unless the interaction is specified. Trees split on one feature at a time and capture
interactions implicitly through the sequence of splits. Additive models with splines capture
nonlinear main effects but not interactions unless they are added explicitly.

**Smoothness.** Linear and spline models assume smooth relationships. Trees produce piecewise
constant functions with sharp thresholds. Neural networks with common activations are biased toward
smooth functions. If the true relationship has a threshold (for example, a clinical rule that
triggers at a specific value), a tree may represent it more efficiently; if the relationship is
smooth, a spline or linear model may generalise better.

**Rotational invariance.** Grinsztajn et al. (2022) show that multilayer perceptrons are biased
toward rotationally invariant solutions, while tabular features have meaningful axes (age, weight,
creatinine). This is one reason tree-based models remain strong on medium-sized tabular data. The
finding is a caution against assuming that a neural network is the natural choice for a table.

Calibration is a separate axis from discrimination. A family can rank well and be miscalibrated.
Logistic regression fitted by maximum likelihood is calibrated in the large-sample limit when the
model is correctly specified; tree ensembles and neural networks often are not. Post-hoc calibration
(Platt scaling, isotonic regression) can fix the scale without changing the ranking, but it adds a
step that must be validated on data not used for fitting.

> **Note:** The best model is the simplest one whose inductive bias matches the problem and whose
> calibration and decision utility meet the requirement. "Best" is a property of the
> model-plus-decision pair, not of the algorithm.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Model family | A class of functions plus a fitting procedure | Called an algorithm | Category |
| Hypothesis space | The set of functions a model can represent | Confused with parameter count | Mathematical set |
| Inductive bias | The preference among hypotheses when data are ambiguous | Ignored | Property |
| Parametric model | Fixed number of parameters | Assumed to be linear | Count |
| Nonparametric model | Flexibility grows with data | Assumed to have no assumptions | Property |
| Linear model | Predicts a linear combination of features | Assumed to be interpretable causally | Coefficients |
| Link function | Maps the linear predictor to the outcome scale | Ignored | Function |
| Logistic regression | Linear model for binary outcomes with a logit link | Called "regression" for continuous outcomes | Coefficients |
| Odds ratio | Exponentiated coefficient | Interpreted as risk ratio | Ratio |
| Regularisation | Penalty on coefficient size | Confused with feature selection | Penalty |
| L1 penalty | Sum of absolute coefficients; induces sparsity | Confused with L2 | Lambda |
| L2 penalty | Sum of squared coefficients; shrinks smoothly | Confused with L1 | Lambda |
| Elastic net | Combination of L1 and L2 | Treated as a separate family | Alpha, lambda |
| Spline | Piecewise polynomial basis function | Confused with binning | Knots |
| GAM | Generalised additive model; additive smooth terms | Assumed to capture interactions | Terms |
| Decision tree | Recursive partitioning into regions | Assumed to be stable | Depth |
| Pruning | Removing branches to reduce overfitting | Confused with early stopping | Depth |
| Bagging | Bootstrap aggregation of models | Confused with boosting | Resamples |
| Random forest | Bagged trees with random feature subsets | Assumed to be calibrated | Trees |
| Boosting | Sequential fitting of weak learners | Confused with bagging | Rounds |
| Gradient boosting | Boosting with gradient descent on a loss | Treated as a single algorithm | Rounds, rate |
| XGBoost | Regularised gradient-boosted trees | Assumed to be best by default | Implementation |
| LightGBM | Gradient boosting with histogram splits | Assumed to be equivalent to XGBoost | Implementation |
| CatBoost | Gradient boosting with ordered target statistics | Assumed to handle categories safely | Implementation |
| Learning rate | Shrinkage applied to each boosting step | Confused with step size in training | Rate |
| Number of rounds | Number of boosting iterations | Tuned on the test set | Count |
| Early stopping | Stop when validation loss stops improving | Uses the test set as validation | Rounds |
| Kernel | Similarity function for kernel methods | Confused with a neural layer | Function |
| Support vector machine | Maximum-margin classifier | Assumed to output probabilities | Margin |
| Kernel trick | Implicit high-dimensional mapping | Confused with dimensionality reduction | Method |
| k-nearest neighbours | Predicts from the k closest training points | Assumed to scale | k |
| Curse of dimensionality | Distance loses meaning in high dimensions | Ignored | Property |
| Naive Bayes | Probabilistic model with conditional independence | Assumed to be accurate | Probabilities |
| Multilayer perceptron | Feed-forward neural network | Called "deep learning" when shallow | Layers |
| Activation function | Nonlinearity in a neural network | Confused with the link function | Function |
| Embedding | Learned dense representation of a category | Confused with one-hot | Vector |
| Attention | Weighted combination of representations | Treated as explanation | Weights |
| Transformer | Attention-based architecture | Assumed to be best for tables | Layers |
| Inductive bias toward smoothness | Preference for slowly varying functions | Ignored | Property |
| Rotation invariance | Invariance to orthogonal transforms of features | Ignored in tabular ML | Property |
| Extrapolation | Prediction outside the training range | Assumed from interpolation | Range |
| Stability | Sensitivity of the fitted model to data perturbations | Confused with calibration | Metric |
| Learning curve | Performance as a function of training size | Confused with validation curve | Curve |
| Validation curve | Performance as a function of a hyperparameter | Confused with learning curve | Curve |
| Calibration curve | Observed versus predicted risk | Confused with ROC curve | Curve |
| Brier score | Mean squared error of probabilities | Confused with log loss | Score |
| Log loss | Negative log-likelihood | Confused with Brier | Score |
| Discrimination | Ability to rank positives above negatives | Confused with calibration | AUROC |
| Interpretability | Degree to which a model's behaviour can be understood | Confused with explanation | Property |
| Monotonicity | A constraint that the prediction moves in one direction | Assumed by clinical users | Constraint |
| Sparse model | A model with few nonzero coefficients | Confused with a small model | Count |
| Ensemble | A combination of models | Assumed to be more interpretable | Method |
| Stacking | Combining model outputs with a meta-model | Confused with bagging | Method |
| Hyperparameter | A setting chosen before fitting | Tuned on test data | Configuration |
| Model selection | Choosing among candidates | Confused with hyperparameter tuning | Decision |
| Nested cross-validation | Honest evaluation of a selection procedure | Confused with k-fold | Procedure |

Two definitions deserve more than a table row.

**Inductive bias is not a defect.** A linear model's bias toward additive, smooth relationships is
a strength when the true relationship is additive and the data are limited. A tree's bias toward
axis-aligned splits is a strength when features have meaningful thresholds. The question is not
which bias is better in general, but which bias matches the problem. A model with the wrong bias
will either miss structure (too simple) or fit noise (too flexible).

**Model selection versus hyperparameter tuning.** Hyperparameter tuning chooses settings within a
family. Model selection chooses among families. Both are selection problems, and both are subject
to optimism if the evaluation data are reused. Nested cross-validation addresses both when the
inner loop tunes and the outer loop compares. A comparison that tunes each family on the same
folds and reports the best score is a selection on noisy estimates, and the winner's score is
biased upward.

A useful way to choose a family is to ask what the data can identify. With 100 events and 30
features, the data can identify a small number of additive effects, not a complex interaction
surface. With 5,000 events and 30 features, the data can identify moderate interactions. With
images, the data can identify spatial filters. The question is not "which model is best?" but
"which model's parameters can be estimated from this sample?" A flexible model on a small sample
does not fail because it is flexible; it fails because the data cannot constrain its flexibility.

The family also determines the shape of the explanation. A linear model's explanation is global: a
coefficient applies to every patient. A tree ensemble's explanation is local: the model partitions
the population and the explanation depends on which region a patient falls into. A neural network's
explanation is neither, and requires post-hoc attribution. The decision context determines which
kind of explanation is needed: a policy rule may need a global statement, while a clinician at the
bedside may need a local one.

> **Note:** Calibration is not a property of the family alone. A logistic regression can be
> miscalibrated if the model is misspecified or the sample is small; a gradient-boosted model can
> be well calibrated after post-hoc adjustment. Judge the fitted model, not the family label.

## 5. The mechanics, step by step

Model family selection is a fourteen-step procedure. Steps 1 through 4 are specification. Steps 5
through 9 are the comparison. Steps 10 through 14 are the evidence that supports the choice.

### Step 1. Specify the baseline before any challenger

**Input:** the prediction task specification and the preprocessing pipeline.

**Output:** a baseline model, its expected performance range, and the reason it is the baseline.

**Rationale:** A baseline makes the comparison meaningful. Without one, a challenger's performance
is a number without a reference. The baseline should be the simplest defensible model: usually a
logistic regression with main effects and a small penalty, fitted on the same pipeline as every
other candidate. If the baseline is already adequate for the decision, the burden of proof is on
the challenger.

The baseline should be specified in enough detail to be reproducible: the features, the functional
form (linear, spline, interaction terms), the penalty and how it is tuned, and the calibration
method. A baseline described as "logistic regression" is not a specification; two teams will
produce different models and different numbers.

**Branch point:** if the decision requires a probability and the baseline is miscalibrated, fix the
baseline's specification (add splines, interactions, or a penalty) before adding a more flexible
family. A flexible model is not a substitute for a misspecified linear model.

### Step 2. Profile the data structure

**Input:** the cohort, features, and outcome.

**Output:** a profile with sample size, event count, events per candidate parameter, feature types,
missingness, linearity diagnostics, and evidence of interactions.

**Rationale:** The data profile determines which families are plausible. A cohort with 300 events
and 40 candidate features cannot support a deep tree ensemble without heavy regularisation. A
cohort with strong pairwise interactions will disadvantage a main-effects linear model. A cohort
with many uninformative features will disadvantage a neural network more than a tree ensemble,
according to the Grinsztajn benchmark.

The profile should also record the number of distinct values per feature. A continuous feature with
12 distinct values is effectively ordinal, and a spline with many knots will overfit it. A
categorical feature with 4,000 levels is a high-cardinality problem that needs the encoding
treatment in Topic 04. These details determine whether a family's assumptions are plausible more
than the family's reputation does.

**Branch point:** if the event count is below about 10 per candidate parameter, restrict the
candidate set to penalised linear or additive models. A flexible model can still be fitted, but its
validation estimate will be too unstable to support a selection.

### Step 3. Choose candidate families

**Input:** the data profile and the decision requirements.

**Output:** two to four candidates, with a justification for each and a statement of what each is
expected to add.

**Rationale:** More candidates increase the selection optimism and the compute cost. Two or three
well-chosen candidates are usually enough: a baseline, a flexible challenger, and optionally a
middle option that captures nonlinearity without full interaction complexity. The table below maps
data conditions to candidates.

| Data condition | First candidate | Second candidate | Avoid |
|---|---|---|---|
| Small n, few events, additive signal | Penalised logistic regression | Spline GAM | Deep ensembles |
| Moderate n, additive signal, need interpretability | Logistic regression with splines | GAM | Neural network |
| Large n, suspected interactions | Gradient boosting | Random forest | Main-effects-only model |
| Many uninformative features | Gradient boosting | Penalised logistic regression | MLP |
| High-dimensional sparse text | Penalised logistic regression | Naive Bayes or linear SVM | Random forest |
| Images, signals, waveforms | Convolutional neural network | Pretrained network | Tabular GBM on raw pixels |
| Need monotonicity by policy | Monotonic gradient boosting | Constrained GAM | Unconstrained network |

**Branch point:** if two candidates are expected to perform similarly, prefer the simpler one. The
burden of proof is on the more complex model, because its maintenance and regulatory costs are
higher.

### Step 4. Specify preprocessing per family

**Input:** the candidates and the preprocessing pipeline.

**Output:** a pipeline per family, with the same split and the same missingness handling where
possible.

**Rationale:** A comparison is only fair if the candidates see comparable information. A linear
model may need scaling and splines; a tree model does not need scaling but may need categorical
encoding; a neural network may need embeddings. Differences in preprocessing can dominate
differences in family, so the comparison should either hold preprocessing constant or report the
effect of changing it.

**Branch point:** if a family requires a preprocessing step that the others do not, report the
performance with and without that step. Otherwise the comparison confounds family with pipeline.

### Step 5. Set capacity and hyperparameter ranges

**Input:** the data profile and the candidate families.

**Output:** a search space per family, with a rationale for the bounds and the number of
configurations.

**Rationale:** The search space determines the optimism. A large grid with few events will overfit
the selection criterion. The bounds should reflect the data: a tree depth of 20 on 500 events is
not a serious candidate, and a learning rate of 0.5 with 1,000 rounds is a recipe for overfitting.
The number of configurations should be small enough that the inner loop can estimate differences.

**Branch point:** if the search space has more than about 50 configurations per family and the
event count is below 500, reduce the space. The selection optimism grows with the number of
configurations tried.

### Step 6. Tune in the inner loop and evaluate in the outer loop

**Input:** the candidate pipelines and the nested cross-validation design.

**Output:** outer-fold performance for each candidate, with the selected hyperparameters recorded
per fold.

**Rationale:** Nested cross-validation separates the question "which settings work best?" from "how
well does this procedure generalise?" The outer estimate is the one used for comparison. Reporting
the inner score as performance reintroduces the selection bias that Varma and Simon documented.

The outer-loop predictions should be saved, not just the scores. Saving predictions allows a
later calibration analysis, a decision curve, and a subgroup analysis without refitting. It also
allows an auditor to recompute every metric from the same predictions, which is the simplest test
of reproducibility.

**Branch point:** if the selected hyperparameters vary wildly across outer folds, the procedure is
unstable. Prefer a family with a stable selection or fix the hyperparameters a priori and report
the fixed-configuration performance.

### Step 7. Compare discrimination with intervals

**Input:** outer-fold predictions for each candidate.

**Output:** AUROC and AUPRC with confidence intervals, and the difference between candidates with
an interval.

**Rationale:** A comparison of point estimates without intervals cannot distinguish a real
difference from noise. The relevant quantity is the difference, and its interval. If the interval
includes zero, the candidates are not distinguishable on discrimination, and the choice should be
made on calibration, utility, or simplicity.

**Branch point:** if AUPRC and AUROC rank the candidates differently, the prevalence is doing the
work. Report both and explain which is aligned with the decision.

### Step 8. Compare calibration

**Input:** outer-fold predictions and outcomes.

**Output:** calibration slope and intercept, a calibration plot, and the Brier score.

**Rationale:** A model that ranks well but predicts the wrong probability will mislead a
threshold-based decision. Calibration is especially important for tree ensembles and neural
networks, which are often overconfident at the extremes. The calibration assessment should be done
on the same held-out predictions used for discrimination, not on the training data.

**Branch point:** if a candidate is well discriminated but poorly calibrated, consider post-hoc
calibration before rejecting it. Recalibration can change the ranking of candidates on net benefit,
so it must be applied consistently and validated on data not used for fitting the calibrator.

### Step 9. Compare decision utility

**Input:** calibrated predictions, the threshold rule, and the harm ratio.

**Output:** net benefit at the operating threshold, and a decision curve across thresholds.

**Rationale:** Discrimination and calibration are intermediate quantities. The decision depends on
net benefit at the threshold the programme will use. A model with a slightly lower AUROC but better
calibration can have higher net benefit. Conversely, a model with a higher AUROC can have lower net
benefit if its errors are concentrated at the operating point.

**Branch point:** if the decision curve crosses between candidates, the choice depends on the
threshold. Report the crossing point and choose according to the planned operating range, not the
maximum over all thresholds.

### Step 10. Assess stability

**Input:** the fitted models and the resampling design.

**Output:** bootstrap or repeated-fold variability in performance, coefficient or feature-importance
stability, and seed sensitivity.

**Rationale:** A model that performs well on one partition and poorly on another is not reliable.
Stability matters more than a small performance difference, especially for deployment. A logistic
regression with stable coefficients is easier to monitor than a tree ensemble whose top features
change with every bootstrap sample.

**Branch point:** if the model's top features change across bootstrap samples, the model is not
identifying a stable signal. Report the instability and consider a simpler model or stronger
regularisation.

### Step 11. Assess subgroup performance

**Input:** the held-out predictions and subgroup identifiers.

**Output:** AUROC, calibration, sensitivity, and PPV by subgroup, with counts and intervals.

**Rationale:** A model that performs well overall can fail in a subgroup that matters. The failure
can come from the family (for example, a tree ensemble that splits on a proxy for the subgroup) or
from the data (a subgroup with fewer events). The comparison should include subgroup performance,
not only the overall metric.

Subgroup performance should be assessed on the same predictions and the same threshold as the main
analysis. A subgroup AUROC is useful for detecting a ranking failure, but the decision-relevant
question is whether the subgroup's sensitivity, PPV, and net benefit are acceptable. A model can
have a similar AUROC across subgroups and still produce different net benefit because the
prevalence differs.

**Branch point:** if a subgroup has fewer than about 20 events, report the count and do not make a
subgroup-specific claim. Pool related subgroups or collect more data.

### Step 12. Assess interpretability and constraints

**Input:** the decision context and the stakeholder requirements.

**Output:** a statement of what explanation is required, whether the family provides it natively,
and what post-hoc method would be needed.

**Rationale:** A sparse logistic regression provides a coefficient table that a clinician can read.
A gradient-boosted model requires feature attribution, and attribution methods have limits. If the
decision requires a monotonic relationship (for example, "higher risk score must never lower the
priority"), the family must support a monotonic constraint or the constraint must be checked after
fitting.

**Branch point:** if the regulatory pathway requires an interpretable model and the flexible
candidate is only marginally better, choose the interpretable model. The marginal performance is
unlikely to justify the explanation burden.

### Step 13. Assess operational and regulatory burden

**Input:** the deployment environment and the regulatory pathway.

**Output:** an estimate of maintenance, monitoring, change control, and documentation effort for
each candidate.

**Rationale:** A model is a product, not a one-time analysis. A gradient-boosted model with monthly
retraining requires a change-control plan, drift monitoring, and revalidation. A logistic
regression with a fixed specification requires less. The burden should be part of the selection,
not an afterthought.

The estimate should include the cost of revalidation after each retraining, not only the cost of
the initial validation. A model that is retrained quarterly requires four validation cycles a year,
which can exceed the cost of a simpler model that is retrained annually.

**Branch point:** if the organisation cannot maintain the flexible model, choose the simpler one
even at a small performance cost. An unmaintained model degrades regardless of its family.

### Step 14. Select and document

**Input:** the comparison evidence.

**Output:** a selection memo with the decision, the evidence, the alternatives considered, and the
conditions under which the decision would change.

**Rationale:** The memo makes the selection auditable. It should state the primary criterion, the
secondary criteria, the uncertainty, and the reasons for rejecting each alternative. It should also
state what would change the decision: a larger sample, a different prevalence, a new site, or a
regulatory requirement.

**Branch point:** if no candidate meets the pre-specified requirement, do not select the least bad
one. Report that the requirement was not met and propose a different decision, a different data
collection, or a narrower claim.

### Model-family edge cases

**Perfect separation in logistic regression.** A feature or combination perfectly predicts the
outcome, and the coefficients diverge. Add a penalty, remove the offending feature, or use a
Bayesian prior. Perfect separation in a clinical dataset is usually a sign of leakage or a
near-unique identifier.

**More features than events.** A linear model with 200 features and 100 events will overfit unless
penalised. Use L1 or elastic net, reduce the feature set with clinical input, or collect more data.
A flexible model in this setting is not a solution.

**Single-level categorical feature.** No family can use it. Remove it before comparison.

**Zero-variance feature.** Remove it; it adds noise to the search and can break some
implementations.

**Tiny subgroups.** A family that performs well overall may fail in a subgroup with 15 events.
Report the count and avoid a subgroup claim.

**Prevalence shift.** A family tuned at 19 percent prevalence may behave differently at 8 percent.
Recalibrate and re-evaluate net benefit at the deployment prevalence.

**A candidate that cannot be fitted on a fold.** Some implementations fail when a fold has a
constant feature or a single class. Record the failure and exclude the fold from that candidate's
estimate, rather than imputing a score. A candidate that fails on several folds is not deployable,
regardless of its performance on the others.

**Ties in scores.** Tree ensembles often produce tied predictions, especially with few trees.
Ties affect threshold-based metrics and capacity-constrained selection. Report the tie rate at the
operating point.

**Empty folds.** A candidate that cannot be fitted in a fold with one class should be reported as
not evaluable for that fold, not as a zero.

**Extrapolation.** Linear and neural models extrapolate beyond the training range; trees do not.
For a feature with a deployment range wider than development, the tree's flat extrapolation may be
safer or may be wrong, depending on the clinical relationship. State the range and the expected
behaviour outside it.

## 6. Worked example

### The setting

A heart failure programme compares four model families on the same 24,800 admissions and 22
features used in Topics 03 and 04. The outcome is 30-day unplanned readmission, prevalence 19.0
percent in development, 21.6 percent in the temporal cohort, and 23.0 percent at the external site.
The decision threshold is 0.20, derived from a harm ratio in Topic 01. The programme can absorb
about 27 percent of discharges.

The candidates are:

1. Logistic regression with main effects and an L2 penalty.
2. Spline logistic regression (a GAM with natural cubic splines for continuous features).
3. Random forest with 500 trees and a minimum leaf size of 20.
4. Gradient boosting with a learning rate of 0.03, depth 3, and early stopping within the inner
   loop.

All candidates use the same patient-grouped folds, the same imputation strategy, and the same
target-encoding pipeline. Hyperparameters are tuned in an inner three-fold loop; the outer loop is
touched once.

### The results

| Model | CV AUROC | Temporal AUROC | External AUROC | External calibration slope | Optimism |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.681 | 0.662 | 0.641 | 0.72 | 0.031 |
| Spline GAM | 0.690 | 0.671 | 0.648 | 0.74 | 0.009 |
| Random forest | 0.689 | 0.658 | 0.632 | 0.66 | 0.052 |
| Gradient boosting | 0.694 | 0.649 | 0.611 | 0.58 | 0.047 |

The gradient-boosted model has the best cross-validated AUROC (0.694) but the worst external AUROC
(0.611) and the worst calibration slope (0.58). The spline GAM has the best external AUROC (0.648)
and the best external calibration slope (0.74), with almost no optimism (0.009). The ranking
reverses between the internal and external designs, which is the same pattern seen in Topic 03.

### The decision at the threshold

At the external site, prevalence is 23.0 percent and the threshold is 0.20. The operating
characteristics are:

| Model | Sensitivity | Specificity | PPV | Net benefit |
|---|---:|---:|---:|---:|
| Logistic regression | 0.61 | 0.71 | 0.386 | 0.0845 |
| Spline GAM | 0.63 | 0.70 | 0.386 | 0.0872 |
| Random forest | 0.60 | 0.72 | 0.390 | 0.0841 |
| Gradient boosting | 0.66 | 0.60 | 0.330 | 0.0748 |

The net benefit arithmetic for the spline GAM:

```text
0.63 * 0.23 - 0.30 * 0.77 * (0.20 / 0.80)
= 0.1449 - 0.2310 * 0.25
= 0.1449 - 0.0578
= 0.0871
```

For gradient boosting:

```text
0.66 * 0.23 - 0.40 * 0.77 * 0.25
= 0.1518 - 0.0770
= 0.0748
```

The GAM has the highest net benefit, narrowly ahead of the logistic regression and the random
forest. The gradient-boosted model has the highest sensitivity but the lowest specificity, and at
23 percent prevalence the false positives outweigh the extra true positives.

The calibration intercept tells a complementary story. The GAM's intercept at the external site is
0.02, the logistic model's is 0.05, the random forest's is 0.09, and the gradient-boosted model's
is -0.16. The negative intercept for the GBM means it over-predicts risk on average at the external
site, which is consistent with its low sensitivity and high false positive count at the threshold.
The intercept and slope together describe the miscalibration; the slope alone can be misleading
when the average prediction is also off.

The GAM's advantage is not that splines are intrinsically better than trees. It is that the
relationship in this cohort is mostly additive with smooth nonlinearity, and the GAM's inductive
bias matches that structure. The tree ensemble's bias toward axis-aligned splits and interactions
is a poor match for smooth main effects with limited data, and the mismatch shows up as external
miscalibration.

### Two diagnostics that would reveal a broken comparison

**Diagnostic 1: calibration slope at the external site.** The gradient-boosted model's slope is
0.58, against 0.74 for the GAM and 0.72 for the logistic model. A slope below 1 means the
predictions are overdispersed: high risks are too high and low risks are too low. At a predicted
risk of 0.40, the recalibrated risk is:

```text
logit(0.40) = -0.4055
recalibrated logit = 0.58 * (-0.4055) = -0.2352
recalibrated risk = 1 / (1 + exp(0.2352)) = 0.441
```

The model understates the risk of the highest-risk patients by about 4 percentage points and
overstates the risk of low-risk patients. A comparison that reports only AUROC would not see this.

**Diagnostic 2: subgroup performance in patients aged 80 and over.** The subgroup has 612 patients
and 141 events. The gradient-boosted model's AUROC is 0.55 (95 percent CI 0.49 to 0.61), while the
GAM's is 0.63 (0.57 to 0.69) and the logistic model's is 0.62 (0.56 to 0.68). The flexible model's
advantage disappears in the subgroup that may need it most. The diagnostic is a subgroup table with
counts, not a single overall metric.

### The operational view

The comparison also has a cost dimension. The logistic regression can be documented with a
coefficient table and updated by changing a small number of parameters. The GAM adds spline knots
and requires the knot placement to be specified and versioned. The random forest and gradient
boosting require the tree structure to be serialised, the feature attributions to be monitored, and
the model to be revalidated when the training data are refreshed. None of these costs appears in
the AUROC table, but all of them affect the total cost of ownership.

For this programme, the GAM's external calibration advantage (0.74 versus 0.72) is small, but its
net benefit advantage (0.0871 versus 0.0845) and its stability across designs justify the modest
additional documentation. The GBM's cross-validated advantage (0.694 versus 0.690) is within noise
and does not survive the external validation, so it is not selected.

The GAM also has a practical advantage that does not appear in the table: its smooth terms can be
plotted, and a clinician can see the shape of the relationship between a feature and risk. A
spline for age that rises after 75 is easier to discuss than a gradient-boosted model's partial
dependence plot, which is an average over a complex function. The ability to discuss the shape is
part of the explanation burden, and it favours the additive model when the shapes are clinically
plausible.

### Varying one assumption: prevalence at the deployment site

Suppose the external site's prevalence is 12 percent rather than 23 percent. At the fixed threshold
of 0.20:

```text
GAM net benefit:  0.63 * 0.12 - 0.30 * 0.88 * 0.25
                = 0.0756 - 0.0660
                = 0.0096

GBM net benefit:  0.66 * 0.12 - 0.40 * 0.88 * 0.25
                = 0.0792 - 0.0880
                = -0.0088
```

The GAM's net benefit is small but positive. The gradient-boosted model's net benefit is negative:
at this prevalence, its false positives cost more than its true positives gain. The conclusion
changes from "the GBM is a reasonable second choice" to "the GBM should not be deployed at this
site without recalibration and a threshold change". The model did not change; the prevalence did.

### What would change the decision

The decision would change if the external prevalence were below about 15 percent, where the GAM's
net benefit approaches zero and a different threshold or a different intervention would be needed.
It would also change if the programme required a monotonic relationship between a feature and risk,
because the GAM's spline terms are not constrained to be monotonic and would need to be re-specified
or replaced by a constrained model. Finally, it would change if a prospective evaluation showed that
the GBM's higher sensitivity led to better outcomes despite its lower specificity, which is an
empirical question that no retrospective comparison can answer.

### What to report

The primary figure is a calibration plot at the external site for each candidate, with the slope
and intercept annotated. The secondary figure is a decision curve at the external prevalence with
the operating threshold marked. The AUROC table belongs in the appendix, with the optimism column,
because it explains why the internal ranking differs from the external one. The selection memo
should state that the GAM was chosen for its combination of external calibration, net benefit, and
stability, and that the GBM was rejected despite the best cross-validated AUROC.

## 7. Code, query, or calculation

The script fits two logistic models on the same data: one with main effects only, and one with
quadratic and interaction terms. The second model has a larger hypothesis space, which is the same
mechanism that separates a linear model from a tree ensemble. The comparison shows how a
misspecified hypothesis space underfits, and how a larger one captures structure that the linear
model cannot represent.

```python
"""Linear versus quadratic logistic regression. Standard library only."""
import math
import random


def fit_logistic(X, y, l2=0.01, iterations=400, learning_rate=0.2):
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    for _ in range(iterations):
        gw, gb = [0.0] * d, 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            error = p - yi
            for j in range(d):
                gw[j] += error * xi[j]
            gb += error
        for j in range(d):
            w[j] -= learning_rate * (gw[j] / n + l2 * w[j])
        b -= learning_rate * gb / n
    return w, b


def predict(X, w, b):
    return [1 / (1 + math.exp(-max(-30, min(30, b + sum(wj * xj
            for wj, xj in zip(w, x)))))) for x in X]


def auroc(y, p):
    pairs = sorted(zip(p, y))
    positives = sum(y)
    negatives = len(y) - positives
    wins = 0.0
    seen_negatives = 0
    for _, label in pairs:
        if label == 1:
            wins += seen_negatives
        else:
            seen_negatives += 1
    return wins / (positives * negatives)


def brier(y, p):
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y)


def expand(row):
    x1, x2, x3 = row
    return [x1, x2, x3, x1 * x1, x2 * x2, x3 * x3, x1 * x2, x1 * x3, x2 * x3]


def standardise(train, test):
    d = len(train[0])
    means, sds = [], []
    for j in range(d):
        values = [x[j] for x in train]
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 or 1.0
        means.append(mean)
        sds.append(sd)
    return ([[(x[j] - means[j]) / sds[j] for j in range(d)] for x in train],
            [[(x[j] - means[j]) / sds[j] for j in range(d)] for x in test])


rng = random.Random(21)
rows = []
for _ in range(1200):
    x1, x2, x3 = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
    z = 1.2 * x1 - 0.8 * x2 + 1.5 * x1 * x3
    y = 1 if rng.random() < 1 / (1 + math.exp(-z)) else 0
    rows.append(([x1, x2, x3], y))

train, test = rows[:800], rows[800:]
for name, transform in (("linear", lambda r: r), ("quadratic", expand)):
    X_train = [transform(r[0]) for r in train]
    X_test = [transform(r[0]) for r in test]
    X_train, X_test = standardise(X_train, X_test)
    y_train = [r[1] for r in train]
    y_test = [r[1] for r in test]
    w, b = fit_logistic(X_train, y_train)
    p_train = predict(X_train, w, b)
    p_test = predict(X_test, w, b)
    print(f"{name}: train AUROC={auroc(y_train, p_train):.3f} "
          f"test AUROC={auroc(y_test, p_test):.3f} "
          f"train Brier={brier(y_train, p_train):.3f} "
          f"test Brier={brier(y_test, p_test):.3f}")
```

Expected output:

```text
linear: train AUROC=0.746 test AUROC=0.744 train Brier=0.205 test Brier=0.207
quadratic: train AUROC=0.832 test AUROC=0.850 train Brier=0.167 test Brier=0.160
```

Reading the output:

- The true log-odds contains the interaction `1.5 * x1 * x3`. The linear model cannot represent it,
  so its AUROC is 0.744 on the test set. The quadratic model includes the interaction and reaches
  0.850.
- The Brier score improves from 0.207 to 0.160, which is a larger relative improvement than the
  AUROC suggests. The quadratic model is better calibrated as well as better ranked, because the
  missing interaction biases the predicted probabilities.
- The quadratic model's test performance is slightly higher than its training performance. That is
  not evidence of a problem; with 800 training rows and nine parameters, the model is not
  overfitting. On a smaller sample, the same expansion would overfit, which is why the number of
  terms must be considered relative to the event count.
- In a real cohort with 22 features, the full quadratic expansion has 22 main effects, 22 squares,
  and 231 pairwise interactions, or 275 terms. That is too many for a few thousand events without
  strong penalisation. The code uses three features to keep the arithmetic visible.

The comparison is a controlled demonstration of inductive bias, not a recommendation to add all
interactions. In practice, interactions should be specified from clinical knowledge or discovered
with a method that controls complexity, and the resulting model should be validated on held-out
data.

## 8. Data, standards, and regulatory context

No regulation prescribes a model family. The relevant instruments constrain how the choice is
documented, validated, and maintained. The table distinguishes binding requirements from
conventions.

| Instrument | Status | What it says about model choice | What it does not do |
|---|---|---|---|
| FDA GMLP principles | Guidance, not binding | Calls for a reference standard, representative data, and testing of the chosen approach | Does not prefer a family |
| FDA PCCP guidance | Guidance, not binding | Requires planned model changes to be described and validated | Does not cover initial selection |
| TRIPOD+AI | Reporting convention | Requires the modelling method, hyperparameters, and performance to be reported | Does not rank families |
| PROBAST+AI | Appraisal tool | Assesses model development, overfitting, and performance | Does not prescribe a family |
| EU AI Act | Binding in the EU, phased | Requires risk management, data governance, and technical documentation for high-risk systems | Does not specify an algorithm |
| EU MDR | Binding in the EU | Requires clinical evaluation and performance evaluation | Does not specify an algorithm |
| ISO 14971 | Voluntary standard | Risk management across the lifecycle | Does not address model selection |
| ISO 13485 | Voluntary standard | Design controls and configuration management | Does not address statistics |
| NIST AI RMF | Voluntary framework | Map, measure, manage, and govern AI risks | Not sector-specific |
| Model card conventions | Industry convention | Document intended use, performance, and limits | Not enforceable |

Three consequences follow.

**The documentation burden scales with the family.** A sparse logistic regression can be described
with a coefficient table, a calibration plot, and a decision curve. A gradient-boosted model needs
the same plus a description of the tree structure, the feature attribution method, and the
stability analysis. A neural network needs the architecture, the training procedure, the
initialisation, and the hardware. The regulatory submission does not require a simpler model, but
it requires more evidence for a complex one.

**Post-hoc calibration is part of the model.** If a gradient-boosted model is calibrated with
isotonic regression, the calibrator is part of the deployed artefact and must be versioned and
validated with the model. A submission that describes the model but not the calibrator is
incomplete. The same applies to any preprocessing that changes the output scale.

**Monotonicity and constraints may be required by policy, not by regulation.** A clinical protocol
may require that a higher risk score never lowers a patient's priority. An unconstrained tree
ensemble can violate that requirement locally. If the policy is binding, the model family must
support a monotonic constraint or the deployed system must enforce it outside the model. The
constraint should be documented as part of the intended use.

### Model cards and family disclosure

A model card for a clinical prediction model should name the family in plain language and state the
functional form. "Gradient-boosted decision trees with 400 trees, depth 3, learning rate 0.03" is
informative; "machine learning model" is not. The card should also state what the family cannot do:
a tree ensemble does not extrapolate, a linear model does not capture unspecified interactions, and
a neural network's attribution is not causal. Those limitations are part of the intended use.

### Documentation by family

The documentation burden can be estimated from the family. A logistic regression requires a
coefficient table, the specification of any spline or interaction terms, the penalty, and the
calibration method. A GAM adds the knot locations and the smoothness parameters. A random forest
requires the number of trees, the maximum depth, the minimum leaf size, the feature subsampling
rate, and the feature-importance method. Gradient boosting adds the learning rate, the number of
rounds, the early-stopping rule, and the regularisation parameters. A neural network adds the
architecture, the initialisation, the optimiser, the batch size, and the training hardware.

The list is not a reason to avoid complex families, but it is a reason to budget for them. A
submission that cannot describe the model's configuration cannot be reviewed, and a configuration
that cannot be reproduced cannot be audited.

A further documentation point is the treatment of hyperparameters. A model card should report the
final hyperparameters and the search space, not only the best value. The search space is what
allows a reviewer to judge the selection optimism: a model chosen from 5 configurations is more
credible than one chosen from 5,000 on the same data. If the search space was large, the validation
should use nested cross-validation and report the optimism.

> **Note:** The FDA's GMLP principles emphasise representative data and independent testing, not
> algorithm choice. A simple model validated well is more defensible than a complex model validated
> poorly.

## 9. Industry practice and case evidence

**No consistent advantage for machine learning over logistic regression.** Christodoulou et al.
(2019) reviewed 71 studies with 282 head-to-head comparisons for binary clinical prediction. In the
145 comparisons at low risk of bias, the difference in logit(AUC) was 0.00 (95 percent CI -0.18 to
0.18). In the 137 at high risk of bias, machine learning appeared 0.34 logit(AUC) better. The
apparent advantage was concentrated in studies with weaker validation. Calibration was not
addressed in 56 of 71 studies (79 percent), and only one used decision curve analysis.

**Tree ensembles remain strong on tabular data.** Grinsztajn et al. (2022) benchmarked 45 OpenML
datasets of about 10,000 samples, comparing random forests, gradient boosting, and XGBoost with
MLPs, ResNets, FT-Transformers, and SAINT under a large hyperparameter search. Tree-based models
remained state of the art at every search budget. The authors attribute the gap to three inductive
biases: neural networks prefer overly smooth functions, they are hurt by uninformative features,
and they are rotationally invariant while tabular features have meaningful axes.

**Deep learning is not automatically better for tables.** Shwartz-Ziv and Armon (2022) compared
deep models with XGBoost on tabular datasets and found that XGBoost remained competitive or better,
and that deep models benefited from ensembling with tree models. The result reinforces the
practical guidance to start with a tree ensemble or a linear model, not a neural network, for
structured clinical data.

**Interpretable models can be competitive.** Rudin (2019) argues that for many high-stakes
decisions, an inherently interpretable model should be preferred when its performance is comparable,
because post-hoc explanations of black boxes can be misleading and can create false confidence.
The argument is normative, not empirical, but it aligns with the regulatory preference for
transparency in high-risk devices.

**Narrow, well-validated imaging models.** The IDx-DR device used a convolutional neural network
for a narrow task: detecting more than mild diabetic retinopathy from fundus images captured on a
specific camera in primary care. The De Novo authorization was supported by a prospective trial
with sensitivity 87.2 percent and specificity 90.7 percent at 23.8 percent prevalence. The case
illustrates that a neural network is appropriate when the input is an image and the task is narrow,
not because the network is more advanced but because the inductive bias matches the data.

**Why a complex model can lose at a new site.** A flexible model can exploit site-specific coding,
documentation, or workflow patterns that are correlated with the outcome in development but absent
elsewhere. The pattern is not necessarily leakage; it can be a genuine local signal that does not
transport. The external validation in this topic shows the effect: gradient boosting loses 0.083
AUROC from cross-validation to the external site, while the spline GAM loses 0.042. The flexible
model's extra capacity buys local fit at the cost of transportability. This is an inference from
the pattern, not a universal law, but it is a common reason flexible models fail in multi-site
deployments.

**Readmission models.** The systematic review by Kansagara et al. (2011) found that most readmission
models had modest discrimination, with c-statistics of 0.55 to 0.65 for administrative models
tested in large US populations and 0.68 to 0.83 for models usable at discharge. The variation
across models was driven more by the predictors and the population than by the algorithm. The
lesson is that feature and label choices dominate family choice in this domain.

**The cost of a wrong family.** Choosing a linear model when the truth is strongly interactive
produces a model that underfits and is miscalibrated at the extremes. Choosing a flexible model
when the truth is additive produces a model that fits noise, transports poorly, and is harder to
maintain. The two errors have different signatures: underfitting shows a learning curve that
plateaus early and a calibration plot that is curved; overfitting shows a large optimism and a
validation curve that turns upward. The diagnostics are more informative than the family label.

**What the evidence does not say.** The reviews do not show that machine learning is never better.
They show that, across published clinical prediction studies, the average difference is small and
the apparent advantage is concentrated in studies with weaker methods. A specific problem with
strong interactions, a large sample, and a careful validation may still favour a flexible model.
The correct reading is that the burden of proof is on the more complex model, and that the proof
must include calibration and decision utility, not only discrimination.

> **Note:** The cases are documented. The inference that a simpler model is usually preferable in
> clinical tabular settings is supported by the reviews, but it is not a universal rule. A specific
> problem may justify a complex model, provided the validation and calibration support it.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Choosing a family before framing the decision | The brief names an algorithm | Technology-first selection | Model does not fit the action | Start with the decision and baseline |
| No baseline | Only the challenger is reported | No reference for comparison | Cannot judge whether complexity helps | Fit a penalised linear baseline |
| Comparing on AUROC only | Calibration and utility absent | Ranking treated as the goal | Miscalibrated deployment | Report calibration and net benefit |
| Tuning on the test set | Hyperparameters chosen on the best test score | Selection bias | Optimistic comparison | Nested CV |
| Unfair preprocessing across families | One model gets splines, another does not | Confounded comparison | Wrong winner | Hold pipeline constant or report both |
| Overfitting a flexible model | Large optimism, unstable features | Too much capacity for the events | Poor transport | Regularise, reduce capacity, more data |
| Underfitting a linear model | Curved calibration, low AUROC | Misspecified functional form | Missed signal | Add splines or interactions |
| Ignoring calibration after selection | Good AUROC, poor slope | Family-specific miscalibration | Wrong probabilities at the threshold | Recalibrate and re-evaluate |
| Selecting on a single fold | Result changes with seed | Partition noise | Unreliable choice | Repeated nested CV |
| Subgroup failure hidden | Overall metric only | Heterogeneous performance | Harm in a subgroup | Subgroup table with counts |
| Extrapolation outside the training range | Deployment values outside development | Trees flat, linear models linear | Wrong predictions at the edges | Check ranges and constrain |
| Unmaintainable model | No owner for retraining | Complexity exceeds capacity | Degradation after launch | Total cost of ownership |
| Ignoring inference cost | Scoring too slow for the workflow | Large ensemble or network | Alert arrives late | Benchmark latency |
| Treating feature importance as causation | Top features interpreted clinically | Correlation mistaken for mechanism | Wrong clinical action | Label as association; avoid causal claims |
| Family chosen for prestige | The proposal names a neural network without a reason | Fashion over fit | Cost without benefit | Require a data-structure justification |
| No fixed configuration for deployment | The reported model cannot be reproduced | Tuning not recorded | Validation does not apply | Freeze the configuration and version it |
| Calibration performed in-sample | The calibrator sees the test outcomes | Overconfident calibration | Wrong probabilities | Fit calibration on held-out data |
| Family chosen on a single dataset | No external or temporal validation | Local fit treated as general | Transport failure | Validate across sites and periods |
| Hyperparameter search not reported | Only the best value is shown | Selection optimism hidden | Overstated performance | Report the search space and nested estimate |
| Model not frozen before validation | Weights change during evaluation | Iterative tuning on folds | Optimistic estimate | Freeze and version the model |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The candidate set includes a model with the right inductive bias | Selection cannot find what is not considered | All candidates are misspecified | Add a flexible and an additive candidate |
| The validation design is honest | Comparison depends on it | Leakage or tuning contamination | Audit the split and fit scope |
| Hyperparameters are tuned fairly | Differences should reflect families | One family gets a larger grid | Equal search budget or pre-specified settings |
| Calibration is assessed on held-out data | In-sample calibration is optimistic | Calibrator fitted on test | Use outer-fold predictions |
| The operating threshold is pre-specified | Net benefit depends on it | Threshold chosen after results | Fix the threshold in the protocol |
| The deployment prevalence is known | Net benefit and PPV depend on it | Prevalence shift | Re-evaluate at the deployment prevalence |
| Subgroup sizes are adequate for claims | Small subgroups are noisy | Fewer than 20 events | Report counts and intervals |
| The model can be maintained | Deployment is long | No owner or monitoring | Assign ownership and budget |
| The simpler model is adequate | Complexity has costs | A real nonlinear signal is missed | Compare calibration and net benefit |
| Feature attributions are not causal | Communication requires caution | Users infer mechanism | Label as association; validate with domain experts |
| The deployed configuration is the validated one | Reproducibility | Retuning after validation | Freeze and version |
| The monitoring capability matches the family | Drift detection is possible | Complex model without monitoring | Choose a monitorable family or build the capability |
| The calibration data are independent of the fitting data | Calibration is valid | Calibrator fitted on test | Split the calibration set |
| The family is appropriate for the feature types | Images and tables need different biases | A tabular network on raw pixels or a linear model on images | Match the family to the data type |
| The model can be explained at the required level | Governance and clinical use need explanation | Post-hoc attribution overclaimed | Choose a family with native explanation or budget for attribution |

The deepest limit is that a model family comparison is a comparison of fitted procedures on a
particular dataset. It does not establish that one family is better in general, and it does not
establish that the winner will remain better after retraining or at a new site. The comparison
should be reported with its scope, not as a universal ranking.

## 12. Exercises

Exercises 1 and 2 are specification, 3 through 6 are calculation and implementation, 7 is critique,
and 8 is the selection memo. The exercises assume a binary outcome and a tabular feature set;
adapt the family choices to your own data where needed. For the calculation exercises, use the
predictions saved from a nested cross-validation; do not refit a model for each exercise, because
that would change the comparison.

Each exercise has a deliverable, an acceptance criterion, and a time budget. The acceptance criteria
are written so that a reviewer can mark them without redoing the analysis.

**Exercise 1. Family matching.** For five data scenarios (small cohort, large tabular cohort,
image task, high-dimensional text, monotonic policy), choose a family and justify it. Deliverable:
a five-row table. Acceptance criteria: each choice matches the data structure; the rejected
alternatives are named. Time: 30 minutes.

**Exercise 2. Baseline specification.** Write the specification for a penalised logistic regression
baseline for a binary outcome, including features, splines, penalty, and calibration. Deliverable:
a one-page specification. Acceptance criteria: spline knots justified; penalty tuning described;
calibration plan stated. Time: 45 minutes.

**Exercise 3. Optimism arithmetic.** A gradient-boosted model has apparent AUROC 0.78 and nested CV
AUROC 0.71. A logistic model has apparent 0.73 and nested CV 0.70. Compute optimism and state which
model you would select. Deliverable: arithmetic and a recommendation. Acceptance criteria:
optimism correct; recommendation considers calibration and complexity. Time: 30 minutes.

**Exercise 4. Calibration comparison.** Given predicted risks and outcomes for two models, compute
calibration slope and intercept and plot the reliability diagram. Deliverable: the plot and a
table. Acceptance criteria: slope and intercept correct; plot includes the identity line. Time: 60
minutes.

**Exercise 5. Net benefit comparison.** Using the same predictions, compute net benefit at
thresholds 0.10, 0.20, and 0.30 for both models and treat-all. Deliverable: a table and a decision
curve. Acceptance criteria: arithmetic correct; the crossing point identified. Time: 60 minutes.

**Exercise 6. Stability audit.** Bootstrap a fitted model 200 times and record the top five
features and the AUROC. Deliverable: a stability table. Acceptance criteria: feature selection
frequency reported; AUROC interval reported. Time: 75 minutes.

**Exercise 7. Critique a model comparison.** Use the checklist: Is there a baseline? Is the split
grouped or temporal? Was calibration reported? Was net benefit reported? Was the hyperparameter
search fair? Was the comparison pre-specified? Then find a paper comparing machine learning with
logistic regression and assess its validation, calibration, and selection. Deliverable: a 600-word
critique. Acceptance criteria: at least four methodological issues; claims matched to evidence.
Time: 60 minutes.

**Exercise 8. Selection memo.** Write a one-page memo selecting a model family for your project,
with the evidence, alternatives, and conditions that would change the decision. Deliverable: the
memo. Acceptance criteria: primary and secondary criteria stated; uncertainty reported; rejection
reasons given. Time: 60 minutes.

## 13. Self-check answers

**Answer to Exercise 3.** Gradient boosting optimism = 0.78 - 0.71 = 0.07. Logistic regression
optimism = 0.73 - 0.70 = 0.03. The validated difference is 0.01 in favour of gradient boosting,
which is small relative to the optimism and likely within noise. The logistic model is preferable
unless repeated nested cross-validation shows a consistent advantage, because it is simpler, easier
to calibrate, and cheaper to maintain.

**Answer to Exercise 4 (method).** Fit a logistic regression of the outcome on the logit of the
predicted risk. The coefficient on the logit is the calibration slope; the intercept is the
calibration-in-the-large. A slope of 1 and an intercept of 0 indicate perfect calibration. Plot
observed risk against predicted risk by decile, with the identity line. Report the Brier score as a
summary.

**Answer to Exercise 5 (method).** Net benefit at threshold `t` is
`sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * t / (1 - t)`. Compute sensitivity
and specificity from the confusion matrix at each threshold. Treat-all has sensitivity 1 and
specificity 0. The decision curve plots net benefit against `t`; the model is useful where its
curve is above both treat-all and treat-none.

**Answer to Exercise 2 (what a complete baseline specification contains).** A complete specification
names the outcome, the features, the functional form, the penalty, the tuning procedure, and the
calibration plan. It states whether continuous features enter linearly or with splines, how many
knots and where, whether any interactions are pre-specified, and how the penalty is selected inside
the cross-validation. It also states the software and version. A baseline that omits the penalty
selection or the spline specification is not reproducible, and a comparison against it is not
interpretable.

**Answer to Exercise 6 (what a strong stability audit shows).** A strong audit reports the
frequency with which each of the top features is selected across bootstrap samples, the interval of
AUROC across samples, and the correlation between coefficients or predictions across samples. A
model whose top feature appears in 95 percent of samples is stable; one whose top feature changes
every sample is not. The audit should also report the seed and the number of resamples.

**Answer to Exercise 8 (what a strong selection memo contains).** A strong memo states the decision,
the primary criterion and its threshold, the candidates and their results, the uncertainty, the
reason each alternative was rejected, and the conditions under which the decision would change. It
distinguishes evidence from assumption, and it names the owner and the review date. A memo that
lists AUROCs and declares a winner is not a selection memo; it is a results table with a sentence
attached.

**Rubric for Exercises 1, 2, 7, and 8.** A strong submission (1) starts from the decision, (2)
includes a baseline, (3) holds preprocessing constant, (4) reports discrimination, calibration, and
net benefit, (5) assesses stability and subgroups, and (6) states the maintenance burden. A competent
submission meets four of the six. A weak submission selects the model with the highest AUROC and
does not report calibration.

## 14. Interview questions

**1. Why start with logistic regression?** A strong answer cites its calibration, interpretability,
stability, and the evidence that it is often competitive. A weak answer says it is old-fashioned or
always inferior.

**2. When would you choose gradient boosting?** A strong answer says when there are interactions,
nonlinearity, and enough events, and when calibration can be assessed and corrected. A weak answer
says always, because it wins competitions.

**3. Why can a tree ensemble be miscalibrated?** A strong answer explains that the loss and the
averaging can produce overconfident or underconfident probabilities, especially at the extremes, and
that post-hoc calibration is often needed. A weak answer says trees are always well calibrated.

**4. What is inductive bias?** A strong answer defines it and gives examples: linearity, smoothness,
axis-aligned splits, rotation invariance. A weak answer confuses it with bias in the fairness sense.

**5. How do you compare two models fairly?** A strong answer uses the same split, the same
preprocessing, nested CV, calibration, net benefit, and subgroup analysis. A weak answer compares
AUROC on a single test set.

**6. What is the difference between a learning curve and a validation curve?** A strong answer
distinguishes training size from hyperparameter value. A weak answer uses the terms
interchangeably.

**7. Why do neural networks struggle on small tabular data?** A strong answer cites sample
efficiency, smoothness bias, uninformative features, and rotation invariance. A weak answer says
they need more data without a mechanism.

**8. How do you decide between a GAM and a gradient-boosted model?** A strong answer compares
external calibration, net benefit, stability, and interpretability, and notes that the GAM is
preferred when the signal is additive and smooth. A weak answer chooses the higher AUROC.

**9. What is a monotonic constraint and when is it needed?** A strong answer defines it and gives a
clinical example, and explains how it can be enforced. A weak answer has not heard of it.

**10. What is the difference between a GAM and a tree ensemble in terms of inductive bias?** A strong answer contrasts additive smooth terms with axis-aligned splits and implicit interactions, and explains which matches a given data structure. A weak answer says both are nonlinear.

**11. Why might a simpler model transport better?** A strong answer explains that flexible models can exploit local patterns that do not replicate, and that the bias-variance trade-off interacts with site heterogeneity. A weak answer says simpler models are always more accurate.

**12. When would you accept a small AUROC loss for a simpler model?** A strong answer says when the loss is within noise, the simpler model is better calibrated or more stable, and the maintenance or explanation burden is materially lower. A weak answer says never, or always.

**13. What is the risk of tuning a large hyperparameter grid on a small cohort?** A strong answer explains selection optimism, the Varma-Simon result, and the need for nested CV or a smaller grid. A weak answer says more tuning always improves performance.

**14. What does a learning curve tell you about family choice?** A strong answer distinguishes a curve that plateaus at a low score (underfitting, try more capacity or better features) from a large gap between training and validation (overfitting, try less capacity or more data). A weak answer says the curve shows whether the model is good.

**15. How would you explain a gradient-boosted model to a clinical committee?** A strong answer uses
partial dependence or SHAP with caveats, a calibration plot, and a decision curve, and avoids causal
language. A weak answer presents feature importance as explanation.

## 15. Key terms

- **Additive model** — a model whose prediction is a sum of feature effects.
- **Bagging** — bootstrap aggregation of models.
- **Bias** — error from an overly simple hypothesis space.
- **Boosting** — sequential fitting of weak learners.
- **Brier score** — mean squared error of probabilities.
- **Calibration curve** — observed versus predicted risk.
- **Calibration intercept** — systematic over- or under-prediction.
- **Calibration slope** — over- or under-dispersion of predicted risks.
- **CatBoost** — gradient boosting with ordered target statistics.
- **Decision curve** — net benefit across thresholds.
- **Discrimination** — ability to rank positives above negatives.
- **Elastic net** — combination of L1 and L2 penalties.
- **Ensemble** — a combination of models.
- **Extrapolation** — prediction outside the training range.
- **GAM** — generalised additive model.
- **Gradient boosting** — boosting with gradient descent on a loss.
- **Hyperparameter** — a setting chosen before fitting.
- **Hypothesis space** — the set of functions a model can represent.
- **Inductive bias** — preference among hypotheses when data are ambiguous.
- **Kernel** — a similarity function for kernel methods.
- **Learning curve** — performance as a function of training size.
- **LightGBM** — gradient boosting with histogram splits.
- **Log loss** — negative log-likelihood.
- **Logistic regression** — linear model for binary outcomes with a logit link.
- **Monotonicity** — a constraint that predictions move in one direction.
- **Multilayer perceptron** — a feed-forward neural network.
- **Naive Bayes** — a probabilistic model with conditional independence.
- **Nested cross-validation** — inner tuning and outer evaluation.
- **Neural network** — a model with learned layered representations.
- **Odds ratio** — exponentiated logistic coefficient.
- **Optimism** — apparent minus validated performance.
- **Overfitting** — fitting noise rather than signal.
- **Parametric model** — a model with a fixed number of parameters.
- **Random forest** — bagged trees with random feature subsets.
- **Regularisation** — a penalty on model complexity.
- **Rotation invariance** — invariance to orthogonal transforms of features.
- **Spline** — a piecewise polynomial basis function.
- **Stability** — sensitivity of the fitted model to data perturbations.
- **Support vector machine** — a maximum-margin classifier.
- **Transformer** — an attention-based architecture.
- **Underfitting** — missing structure because the model is too simple.
- **Validation curve** — performance as a function of a hyperparameter.
- **Variance** — error from sensitivity to the training sample.
- **XGBoost** — regularised gradient-boosted trees.

## 16. Sources

All URLs accessed 25 October 2026.

1. Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review
   shows no performance benefit of machine learning over logistic regression for clinical prediction
   models. *Journal of Clinical Epidemiology* 2019;110:12-22.
   https://doi.org/10.1016/j.jclinepi.2019.02.004
2. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on
   typical tabular data? *Advances in Neural Information Processing Systems* 35, 2022.
   https://arxiv.org/abs/2207.08815
3. Shwartz-Ziv R, Armon A. Tabular data: deep learning is not all you need. *Information Fusion*
   2022;81:84-90. https://doi.org/10.1016/j.inffus.2021.11.011
4. Rudin C. Stop explaining black box machine learning models for high stakes decisions and use
   interpretable models instead. *Nature Machine Intelligence* 2019;1:206-215.
   https://doi.org/10.1038/s42256-019-0048-x
5. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
6. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
   https://doi.org/10.1136/bmj-2024-082505
7. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive
   analytics. *BMC Medicine* 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
8. Vickers AJ, Elkin EB. Decision curve analysis. *Medical Decision Making* 2006;26(6):565-574.
   https://doi.org/10.1177/0272989X06295361
9. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. *Proceedings of the 22nd ACM
   SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.
   https://doi.org/10.1145/2939672.2939785
10. Breiman L. Random forests. *Machine Learning* 2001;45:5-32.
    https://doi.org/10.1023/A:1010933404324
11. Friedman JH. Greedy function approximation: a gradient boosting machine. *Annals of Statistics*
    2001;29(5):1189-1232. https://doi.org/10.1214/aos/1013203451
12. Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd edition.
    Springer, 2009. https://hastie.su.domains/ElemStatLearn/
13. Steyerberg EW. *Clinical Prediction Models*. 2nd edition. Springer, 2019.
    https://doi.org/10.1007/978-3-030-16399-0
14. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
    Guiding Principles. 2021.
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
15. US Food and Drug Administration. De Novo classification decision, IDx-DR, DEN180001. 2018.
    https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
16. Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission.
    *JAMA* 2011;306(15):1688-1698. https://doi.org/10.1001/jama.2011.1515
17. scikit-learn developers. Supervised learning.
    https://scikit-learn.org/stable/supervised_learning.html
18. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1

The procedure is the part to carry to the next project; the model is the part to re-derive.

A final check before selecting a family: confirm that the baseline is specified, the preprocessing is
comparable, the calibration is assessed, and the decision utility is computed. A family that wins on
AUROC alone has not won the argument.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_preprocessing.md`](04_preprocessing.md)
- Next topic: [`06_class-imbalance.md`](06_class-imbalance.md)
- Calibration: [`08_calibration.md`](08_calibration.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../course/06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
