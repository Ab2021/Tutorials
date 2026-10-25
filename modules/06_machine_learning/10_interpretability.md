# Interpretability

> **Module 06 — Machine Learning Foundations**
> Topic 10 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 05, 07, and 08
> Version 2026.3

## 1. Why this topic exists

A clinical model that cannot be explained is difficult to trust, difficult to audit, and difficult
to improve. Clinicians want to know why a patient was flagged. Regulators want to know what the
model relies on. Data scientists want to know whether the model learned a real signal or a
shortcut. Interpretability is the set of methods that answer those questions, and it is also the
set of methods most often overclaimed.

The central distinction is between explaining the model and explaining the world. A coefficient, a
partial dependence curve, a permutation importance, or a SHAP value describes how the fitted model
uses its inputs. None of them describes what would happen to a patient if an input were changed.
A model may rely on age because age is correlated with comorbidity, frailty, and social support;
the explanation "age is important" is true about the model and false as a causal statement about
aging. Confusing the two produces clinical recommendations that the data cannot support.

A second distinction is between global and local explanations. A global explanation describes the
model's behaviour across the population: which features matter, in what direction, and with what
shape. A local explanation describes one prediction: why this patient was flagged. Both are useful,
and they can disagree. A feature can be important globally and irrelevant for a particular patient,
or vice versa.

A third distinction is between inherently interpretable models and post-hoc explanations. A
logistic regression with a small number of pre-specified terms is interpretable by construction. A
gradient-boosted model with hundreds of trees is not, and post-hoc methods approximate its
behaviour. Rudin (2019) argues that for high-stakes decisions, an inherently interpretable model
should be preferred when its performance is comparable, because post-hoc explanations can be
unstable, can disagree with each other, and can create false confidence.

A fourth distinction is between explanation and justification. An explanation describes how the
model produced a score; a justification argues that the score is appropriate for the decision. A
model can be well explained and poorly justified, for example if it relies on a proxy that encodes
an inequity. The interpretation section should not let a clear explanation stand in for a
justification. An explanation describes how the
model produced a score; a justification argues that the score is appropriate for the decision. A
model can be well explained and poorly justified, for example if it relies on a proxy that encodes
an inequity. The interpretation section should not let a clear explanation stand in for a
justification. A
logistic regression with a small number of pre-specified terms is interpretable by construction. A
gradient-boosted model with hundreds of trees is not, and post-hoc methods approximate its
behaviour. Rudin (2019) argues that for high-stakes decisions, an inherently interpretable model
should be preferred when its performance is comparable, because post-hoc explanations can be
unstable, can disagree with each other, and can create false confidence. Ghassemi, Oakden-Rayner,
and Beam (2021) make a related argument for health care: current explanation methods often do not
provide the kind of understanding clinicians need.

The technical failure modes are well documented. Partial dependence plots marginalise over the
marginal distribution of the other features, so with correlated features they evaluate the model in
regions where no data exist (Apley and Zhu, 2020). Permutation importance breaks feature
dependencies and forces the model to extrapolate, which inflates the importance of correlated
features (Hooker, Mentch, and Zhou, 2021). SHAP's additive guarantee holds by construction, but the
attribution depends on the reference distribution and on how dependent features are handled; with
correlated features, the credit can be split arbitrarily between redundant variables (Kumar et al.,
2020; Aas et al., 2021). None of these methods is wrong; each answers a slightly different question,
and the answer changes with the question.

The organisational consequence is that interpretability needs an owner and a review. It also needs
a defined audience: an explanation written for a data scientist is not the same as one written for a
clinical committee, and a report that tries to serve both usually serves neither. A model
explanation that is produced once, at development, and never revisited will describe a model that no
longer exists. The explanation should be versioned with the model and rechecked when the model, the
population, or the coding changes. That is a governance requirement, not only a technical one.

This topic teaches interpretability as a set of claims with scope conditions. You will learn to
compute coefficients and marginal effects, partial dependence and ICE curves, permutation
importance, and SHAP values; to diagnose correlated-feature failures; to check stability; and to
write explanations that do not overclaim causality. The output is an interpretation section that
states what the method shows, what it does not show, and how the result was checked.

> **Note:** An explanation is a claim about the model, not about the patient. Every explanation in
> a clinical report should be labelled as model behaviour, and causal language should be used only
> when a causal design supports it.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** coefficients from marginal effects and **compute** both for a logistic model.
2. **Construct** and **interpret** partial dependence and ICE curves, and **identify** extrapolation
   into unsupported feature combinations.
3. **Compute** permutation importance, including grouped permutation for correlated features.
4. **Explain** SHAP's additive guarantee and **diagnose** its failure under correlated features.
5. **Compare** explanation methods and **interpret** their disagreement as evidence about the model.
6. **Check** the stability of an explanation across resamples and subgroups.
7. **Write** an explanation that states scope, method, and limits, without causal language.
8. **Choose** between an inherently interpretable model and a post-hoc explanation for a given
   decision.

Outcomes 1 through 4 are computational; 5 through 8 are critique and design. The order matters:
a learner should be able to compute a marginal effect before critiquing a causal claim, because the
critique depends on knowing what the number means. The computational
outcomes are assessed in the code exercises and the self-check items, and the design outcomes in the
interpretation section and the rubric. A learner who can compute the quantities but cannot state
their scope has not met the design outcomes. The exercises are weighted accordingly, and the
interpretation section exercise is the one that assesses the full set.

## 3. The core mental model

An explanation method is a map from a fitted model to a human-readable statement. Different methods
map different aspects of the model, and the map is not the model.

```text
Fitted model f(x)
   |
   +-- model-specific: coefficients, odds ratios, spline terms
   |        answers: how does the model use x_j on its own scale?
   |
   +-- global, model-agnostic: partial dependence, ALE, permutation importance
   |        answers: how does f change when x_j changes, averaging over others?
   |
   +-- local, model-agnostic: ICE, SHAP, LIME
   |        answers: why did f produce this prediction for this patient?
   |
   +-- causal: requires a design, not an explanation method
            answers: what would happen to the outcome if x_j changed?
```

The vertical axis is the scope: global versus local. The horizontal axis is the claim: model
behaviour versus causal effect. The explanatory methods live in the left column; causal inference
lives in the right column and requires a different toolkit.

The same logic applies to a tree ensemble. A feature can be important globally because the model
splits on it often, and unimportant locally because this patient's path did not use it. A local
explanation that contradicts the global one is not necessarily an error; it is a reminder that the
two answer different questions.

For a logistic model, the coefficient is on the log-odds scale. The marginal effect on the
probability scale depends on the baseline risk:

```text
marginal effect of x_j = beta_j * p * (1 - p)
```

At a predicted risk of 0.20, a coefficient of 0.035 per year implies a marginal effect of
`0.035 * 0.20 * 0.80 = 0.0056`, or 0.56 percentage points per year. At a predicted risk of 0.05,
the same coefficient implies `0.035 * 0.05 * 0.95 = 0.00166`, or 0.17 percentage points per year.
The coefficient is constant; the marginal effect is not. A report that gives only the coefficient
and describes it as "a 3.5 percent increase in risk per year" is wrong: the coefficient is a change
in log-odds, not a change in risk.

Partial dependence describes the average prediction as a function of one feature, marginalising over
the others:

```text
PD_j(v) = average over i of f(v, x_i,-j)
```

The average is over the observed values of the other features. If the feature is correlated with
the others, the average includes combinations that never occur, and the model is evaluated outside
its data envelope. The curve can be flat, steep, or nonsensical in those regions, and the
nonsense is an artefact of the method, not a property of the model.

Permutation importance measures the loss in performance when a feature's values are randomly
shuffled:

```text
PI_j = performance(model, data) - performance(model, data with x_j permuted)
```

Shuffling breaks the relationship between `x_j` and the other features. If `x_j` is correlated with
`x_k`, the model can still use `x_k`, so the measured importance is lower than the feature's true
contribution; if the model relies on the joint pattern, shuffling one feature creates impossible
combinations and the importance is inflated. Grouped permutation, which shuffles a set of correlated
features together, gives a more interpretable answer for the group.

SHAP values decompose a prediction into contributions from each feature:

```text
f(x) = phi_0 + sum_j phi_j
```

The guarantee is additive and local: the contributions sum to the prediction minus the baseline.
The baseline `phi_0` is the expected prediction under a reference distribution, and the choice of
reference changes the attributions. With independent features, the decomposition is relatively
stable; with correlated features, the credit can be split between redundant features in ways that
depend on the reference and on the algorithm. The additive guarantee is real, but it does not make
the attribution unique or causal.

> **Note:** Agreement between methods is reassuring; disagreement is informative. If SHAP and
> permutation importance rank the same features differently, the model likely relies on correlated
> features, and the "importance" of any single one is not well defined.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Interpretability | The degree to which a model's behaviour can be understood | Confused with explanation quality | Property |
| Explainability | The ability to produce an explanation for a prediction | Confused with causality | Property |
| Global explanation | A statement about the model across the population | Confused with a local explanation | Scope |
| Local explanation | A statement about one prediction | Confused with a global one | Scope |
| Coefficient | A parameter on the model's linear scale | Interpreted as a risk difference | Log-odds |
| Odds ratio | Exponentiated coefficient | Interpreted as a risk ratio | Ratio |
| Marginal effect | Change in predicted probability per unit change in a feature | Confused with the coefficient | Probability |
| Average marginal effect | Mean marginal effect over the sample | Confused with the effect at the mean | Probability |
| Partial dependence | Average prediction as a function of one feature | Interpreted as a causal effect | Probability |
| ICE curve | Individual conditional expectation for one patient | Confused with partial dependence | Curve |
| ALE | Accumulated local effects; local differences within bins | Confused with partial dependence | Curve |
| Permutation importance | Performance loss when a feature is shuffled | Interpreted as a causal contribution | Metric difference |
| Grouped permutation | Shuffling a set of correlated features together | Confused with individual permutation | Metric difference |
| SHAP | Shapley additive explanations | Interpreted as causal | Log-odds or probability |
| Shapley value | A game-theoretic attribution satisfying efficiency and symmetry | Assumed unique | Value |
| Additive guarantee | The contributions sum to the prediction minus the baseline | Confused with uniqueness | Property |
| Baseline | The reference prediction in a SHAP decomposition | Chosen arbitrarily | Probability |
| Background distribution | The distribution used to compute the baseline | Confused with the training data | Distribution |
| TreeSHAP | An efficient SHAP algorithm for tree ensembles | Assumed exact for correlated features | Algorithm |
| KernelSHAP | A model-agnostic SHAP approximation | Assumed exact | Algorithm |
| LIME | Local interpretable model-agnostic explanation | Assumed stable | Algorithm |
| Surrogate model | A simple model fitted to approximate a complex one | Confused with the original | Model |
| Feature attribution | A numerical contribution assigned to a feature | Confused with importance | Value |
| Feature importance | A ranking of features by contribution | Confused with effect size | Ranking |
| Interaction | A feature's effect depends on another feature | Confused with correlation | Property |
| Correlation | Statistical association between features | Confused with dependence in the model | Coefficient |
| Collinearity | High linear correlation among features | Ignored in attribution | Diagnostic |
| Extrapolation | Evaluating the model outside the data envelope | Common in PDP and permutation | Problem |
| Data envelope | The region of feature space where data exist | Ignored in PDP | Region |
| Counterfactual | A hypothetical input with a changed feature | Confused with an explanation | Input |
| Causal effect | The effect of an intervention on an outcome | Confused with a model attribution | Effect |
| Confounding | A common cause of feature and outcome | Ignored in attributions | Structure |
| Mediation | A feature affects the outcome through another | Confused with interaction | Structure |
| Proxy feature | A feature that stands in for an unmeasured construct | Treated as the construct | Feature |
| Stability | Consistency of an explanation across resamples | Ignored | Property |
| Robustness | Consistency of an explanation under small input changes | Confused with stability | Property |
| Faithfulness | Whether an explanation reflects the model's actual computation | Assumed | Property |
| Plausibility | Whether an explanation matches domain knowledge | Confused with faithfulness | Property |
| Monotonicity | A constraint that the prediction moves in one direction | Assumed by users | Constraint |
| Decision path | The sequence of splits for a tree prediction | Confused with a causal path | Path |
| Attention weight | A neural network's weighting of inputs | Interpreted as explanation | Weight |
| Saliency map | A gradient-based input attribution | Interpreted as explanation | Map |
| Model card | Documentation of intended use, performance, and limits | Confused with an explanation | Document |
| Explanation audit | A review of explanation stability and scope | Omitted | Process |
| Group attribution | A contribution assigned to a set of correlated features | Confused with individual attribution | Value |
| Conditional importance | Importance computed within strata of other features | Confused with marginal importance | Metric |
| Model behaviour | How the fitted model maps inputs to outputs | Confused with causal effect | Property |
| Group importance | Joint importance of correlated features | Confused with individual importance | Metric |
| Model-agnostic | A method that works for any model | Assumed to be model-free | Property |
| Model-specific | A method that uses the model's form | Confused with model-agnostic | Property |
| Explanation stability | Consistency of an explanation across resamples | Confused with model stability | Property |
| Reference distribution | Distribution used to compute a baseline | Confused with the training set | Distribution |
| Interaction effect | A feature's effect depends on another | Confused with a main effect | Effect |

Three definitions deserve more than a table row.

**Faithfulness versus plausibility.** An explanation is faithful if it reflects the model's actual
computation; it is plausible if it matches what a clinician expects. The two can diverge. A
plausible explanation that is not faithful misleads; a faithful explanation that is not plausible
may be rejected by users even though it is correct. The report should state which property the
method provides, and should not use plausibility as evidence of faithfulness.

**SHAP's additive guarantee.** The guarantee is that the attributions sum to the prediction minus
the baseline. It is a property of the decomposition, not a claim that the attribution is the
feature's true effect. With correlated features, many decompositions satisfy the same additive
guarantee, and the choice among them depends on the reference and the algorithm. The guarantee
makes the explanation internally consistent; it does not make it unique or causal.

**Explanation versus causal inference.** A causal claim requires a design that identifies the
effect of an intervention: randomisation, a natural experiment, or a structural model with
defensible assumptions. An explanation method applied to observational data does not identify a
causal effect, because the feature is correlated with unmeasured causes and the model has learned
the correlation. The explanation can be useful for debugging and communication, but it cannot
substitute for a causal analysis.

> **Note:** Every explanation in this topic is a statement about the fitted model. The phrase
> "the model relies on" is accurate; the phrase "X causes" is not, unless a causal design supports
> it.

## 5. The mechanics, step by step

Interpretation has fourteen steps. Steps 1 through 4 define the question and the model-specific
view. Steps 5 through 9 compute the model-agnostic views. Steps 10 through 14 check, compare, and
report them.

### Step 1. Define the explanation user and question

**Input:** the decision context and the audience.

**Output:** a specific question, such as "which features drive the score for patients near the
threshold?" or "why was this patient flagged?" or "is the model relying on a post-discharge
variable?"

**Rationale:** Different questions require different methods. A clinician asking about one patient
needs a local explanation; a committee asking about the model needs a global one; an auditor asking
whether the model uses a prohibited variable needs a targeted ablation or permutation test. A
generic "explain the model" request produces a generic and usually unhelpful answer.

The question should be written down before the method is chosen. "Which features are important?" is
too vague to answer, because importance depends on the method, the population, and the performance
metric. "Which features drive the score for patients within 0.05 of the threshold?" is answerable,
and it tells the analyst which local method and which subgroup to use.

**Branch point:** if the question is causal ("does treating hypertension reduce readmission?"), stop.
No explanation method answers it. The question requires a causal design, and the interpretation
should say so.

### Step 2. Choose the scope: global or local

**Input:** the question.

**Output:** a scope decision and the population or patient to which it applies.

**Rationale:** Global explanations summarise the model across the population; local explanations
describe one prediction. A global statement can be false for a particular patient, and a local
statement can be atypical of the population. Choosing the scope prevents the common error of
presenting a local explanation as a global one, or vice versa.

A global explanation should state the population over which it was computed. "Age is important" is
not a global statement unless the population is specified; it may be true in one cohort and false in
another. The scope is part of the claim.

**Branch point:** if the question is about a subgroup, the scope is the subgroup, not the whole
cohort. Compute the explanation within the subgroup and report the counts.

### Step 3. Check the model form and the link

**Input:** the fitted model and its output scale.

**Output:** a statement of whether the model is linear, additive, tree-based, or neural, and
whether the output is a probability or a score.

**Rationale:** Coefficients are interpretable directly only for linear or additive models on the
model's scale. For a logistic model, the coefficient is on the log-odds scale; for a tree or neural
network, there is no coefficient to interpret. The output scale determines whether a marginal
effect is a probability difference or a log-odds difference.

The output scale also determines the baseline: a logistic model's coefficients are additive on the
log-odds scale, and a probability-scale explanation requires a marginal effect. A model that outputs
a score without a probability has no calibrated scale, and its explanation is a ranking explanation.

**Branch point:** if the model is a tree ensemble or neural network, skip the coefficient step and
use model-agnostic methods. A tree ensemble can still be described by the distribution of its
predictions and by the feature groups it uses, even without coefficients. Do not present feature importance as if it were a coefficient.

### Step 4. Compute coefficients and marginal effects

**Input:** a linear or logistic model.

**Output:** coefficients, odds ratios, and marginal effects at representative risks (for example,
0.05, 0.20, and 0.50), with intervals.

**Rationale:** The coefficient is constant on the model scale; the marginal effect changes with the
baseline risk. Reporting only the coefficient invites the reader to interpret it as a risk
difference, which is wrong for a nonlinear link. Reporting the marginal effect at a representative
risk gives the reader a quantity on the probability scale.

The marginal effect should be reported at the risk range where the decision operates, not at the
mean by default. A model used at a 0.10 threshold needs the marginal effect near 0.10, not near the
population mean of 0.25.

**Branch point:** if the model includes interactions, the marginal effect of a feature depends on
the values of the interacting features. Report the effect at representative combinations, not a
single number.

### Step 5. Compute partial dependence with the data density

**Input:** the fitted model and a feature of interest.

**Output:** a partial dependence curve and a histogram or rug of the feature's observed values.

**Rationale:** The curve shows the average prediction as the feature varies, holding the other
features at their observed values. The density shows where the curve is supported by data. Without
the density, the curve's extrapolations look as credible as its interpolations.

The partial dependence curve is an average, and an average can be flat even when every individual
curve is steep, if the individual curves have opposite signs. That is why the ICE step follows the
partial dependence step: the average is a summary, and the individual curves are the evidence that
the summary is representative.

**Branch point:** if the feature is strongly correlated with another, the partial dependence may
evaluate impossible combinations. Use ALE or a conditional method instead, and state the
correlation.

### Step 6. Compute ICE curves for heterogeneity

**Input:** the fitted model and a sample of patients.

**Output:** one curve per patient, plus the average (the partial dependence).

**Rationale:** The average can hide heterogeneity. If some patients' risk rises with a feature and
others' falls, the average may be flat, and the flat average is misleading. ICE curves show the
individual shapes and whether the average is representative.

**Branch point:** if the ICE curves cross, the average is not representative and the partial
dependence should not be reported alone. Report the distribution of individual slopes or a
stratified summary.

### Step 7. Use ALE for correlated features

**Input:** the fitted model, a feature, and the correlation structure.

**Output:** an accumulated local effects curve.

**Rationale:** ALE computes local differences within bins of the feature, using the conditional
distribution of the other features, so it does not extrapolate across the data envelope. It is
preferred to partial dependence when the feature is correlated with others, at the cost of being
less intuitive to explain.

ALE is not a perfect solution. It estimates local effects within bins, and the bin width is a
choice that affects the curve. It can be harder to explain to a clinical audience than partial
dependence. The report should state the binning rule and the reason for choosing ALE over partial
dependence.

**Branch point:** if the feature is independent of the others, partial dependence and ALE will be
similar, and either can be reported. If they differ materially, the correlation is affecting the
partial dependence.

### Step 8. Compute permutation importance, individually and by group

**Input:** the fitted model, the validation data, and a performance metric.

**Output:** importance for each feature and for pre-specified groups of correlated features.

**Rationale:** Individual permutation measures the loss when one feature is shuffled. With
correlated features, the model can use the partner, so the individual importance understates the
joint contribution and the ranking can be misleading. Grouped permutation, which shuffles the group
together, gives the joint importance and is the more defensible quantity when the group is
clinically coherent.

The permutation should be done on the validation data, not the training data, because the training
performance is already optimistic and the importance inherits the optimism. The metric should be the
one used for the decision, not accuracy by default; a feature that improves AUROC but not net
benefit may not be important for the decision.

**Branch point:** if individual and grouped importances differ materially, report both and explain
that the individual values are not additive. Do not present the individual values as a partition of
the model's performance.

### Step 9. Compute SHAP values with a stated reference and algorithm

**Input:** the fitted model, a background sample, and a SHAP algorithm.

**Output:** local contributions, a global summary (mean absolute value), and the additive check.

**Rationale:** SHAP decomposes a prediction into contributions that sum to the prediction minus the
baseline. The baseline is the expected prediction under the background distribution, and the
algorithm determines how dependent features are handled. Both should be stated, because both change
the attributions.

The background sample should be representative of the population the explanation is about. A
background drawn from the training set will produce a baseline that reflects the training
distribution; a background drawn from the deployment site will produce a different baseline and
different attributions. The choice should be stated and justified.

**Branch point:** if the features are correlated, use a conditional or grouped approach, or state
that the attributions are interventional and may split credit between redundant features. Do not
present a single feature's SHAP value as its unique contribution.

### Step 10. Diagnose correlated-feature failures

**Input:** the correlation matrix, the explanation results, and the model.

**Output:** a list of feature groups with material correlation and a statement of how each
explanation method was affected.

**Rationale:** Correlation is the most common cause of disagreement between methods. A feature that
is a proxy for another will receive credit in one method and not in another, depending on how the
method handles the joint distribution. Diagnosing the correlation prevents the team from
over-interpreting a ranking.

Correlation should be measured on the deployment population, not only on the development sample. A
pair that is correlated at development may be less correlated at a new site, which changes the
attribution. The interpretation is population-specific, like calibration.

**Branch point:** if two features are nearly collinear (correlation above about 0.9), the
attribution between them is not identifiable from the data. Report the group, not the individual
features.

### Step 11. Check stability across resamples and subgroups

**Input:** the explanation and a resampling or bootstrap design.

**Output:** the frequency with which each feature appears in the top ranks, the variability of the
SHAP values, and the subgroup differences.

**Rationale:** An explanation that changes with the bootstrap sample is not a stable description of
the model. Stability matters more than a small difference in importance, especially when the
explanation will be used to justify a decision or to guide a clinical action.

Stability should be assessed on the quantity the report uses: if the report ranks features, the
stability of the rank; if it reports SHAP values, the stability of the value; if it reports partial
dependence, the stability of the curve's shape. A method can be stable in one quantity and unstable
in another.

**Branch point:** if a feature's rank changes in a large fraction of resamples, do not present it as
a key driver. Report the instability and the correlated group that may explain it.

### Step 12. Compare methods and interpret disagreement

**Input:** coefficients, partial dependence, permutation importance, and SHAP.

**Output:** a comparison table and a written interpretation of where the methods agree and differ.

**Rationale:** Agreement across methods is evidence that the signal is robust to the method.
Disagreement is evidence that the model relies on correlated features, interactions, or a region of
the data where the methods behave differently. Both are informative, and neither should be
suppressed.

A comparison table should include the method, the quantity it produces, the population, and the
correlation handling. A table that lists four rankings without those columns invites the reader to
treat them as four measurements of the same thing, which they are not.

**Branch point:** if the methods disagree about the top feature, investigate the correlation and the
model's interactions before choosing one method's answer. The disagreement is the finding.

### Step 13. Avoid causal language and state the scope

**Input:** the interpretation results.

**Output:** a written explanation with explicit scope: model behaviour, population, method, and
limits.

**Rationale:** The most common interpretability failure is not a numerical error; it is a causal
claim inferred from a model attribution. A coefficient of 0.035 for age does not mean that aging
causes readmission; it means the model associates age with readmission after adjusting for the
other features in the model. The distinction matters for clinical action.

**Branch point:** if the team wants a causal claim, propose a causal design (target trial,
instrumental variable, difference-in-differences) rather than a stronger interpretation of the
model.

### Step 14. Document and monitor

**Input:** the interpretation and the model.

**Output:** an interpretation section in the model report and a monitoring plan for explanation
stability.

**Rationale:** The interpretation should be reproducible: the method, the background sample, the
correlation handling, the resampling seed, and the software version. After deployment, the
explanation can change as the model is updated or the population shifts, and the monitoring plan
should detect that.

The documentation should also state which explanation is the primary one and which are supporting.
A report with four rankings and no hierarchy leaves the reader to choose, and the reader will choose
the most flattering.

**Branch point:** if the model is updated, recompute the explanations and compare them with the
previous version. A change in the top features is a signal that the model's behaviour has changed,
even if the AUROC has not.

### Interpretability edge cases

**Perfect collinearity.** Two features carry the same information; the attribution between them is
arbitrary. Report the group and the combined effect.

**Constant feature.** It has no effect and should be removed before interpretation.

**High-cardinality categorical feature.** One-hot encoding creates many columns, and the
importance is spread across them. Group the levels or use a target-encoded version, and interpret
the group.

**Nonlinear link.** A coefficient on the log-odds scale is not a risk difference. Report marginal
effects at representative risks.

**Interactions.** The effect of a feature depends on another; a single partial dependence curve
averages over the interaction and can be flat. Use ICE or a two-way analysis.

**Extrapolation.** A partial dependence curve at values outside the data envelope is unreliable.
Report the density and the range.

**Tiny subgroup.** An explanation computed on 40 patients is unstable. Report the count and avoid a
subgroup-specific claim.

**Unstable SHAP values.** If the values change materially across background samples, the
attribution is not identified. Report the range and the correlation.

**A feature that is a proxy for a protected attribute.** The model may rely on the proxy even if the
attribute is excluded. Check the correlation and the subgroup performance.

## 6. Worked example

### The setting

A heart failure readmission model is a logistic regression with 18 features. The team is asked why
the model flags patients and whether it relies on anything it should not. The explanation is
computed on a held-out sample of 8,000 discharges with 1,520 events, a prevalence of 19 percent.

### Coefficients and marginal effects

| Feature | Coefficient | Odds ratio | Marginal effect at p = 0.20 | Marginal effect at p = 0.05 |
|---|---:|---:|---:|---:|
| Age (per year) | 0.035 | 1.036 | 0.0056 | 0.0017 |
| Creatinine (per mg/dL) | 0.220 | 1.246 | 0.0352 | 0.0105 |
| Prior admissions (per admission) | 0.310 | 1.363 | 0.0496 | 0.0147 |

The coefficients are on the log-odds scale. The marginal effect at a predicted risk of 0.20 is
`beta * 0.20 * 0.80`; at 0.05 it is `beta * 0.05 * 0.95`. The age coefficient of 0.035 implies a
0.56 percentage point increase in risk per year at a baseline risk of 20 percent, and a 0.17
percentage point increase at a baseline risk of 5 percent. A report that says "age increases risk by
3.5 percent per year" has confused the log-odds coefficient with a risk difference.

Per decade, the age effect at p = 0.20 is `10 * 0.0056 = 0.056`, or 5.6 percentage points. That is
larger than the per-unit effect of prior admissions, but smaller than the effect of a 2 mg/dL
creatinine rise. The ranking of features depends on the unit and on the baseline risk, which is why
coefficient tables alone are not enough.

### Partial dependence and the data density

The partial dependence curve for age rises gradually from 60 to 80 and then flattens. The curve for
creatinine rises steeply above 1.5 mg/dL. Both curves are supported by data in the range where most
patients lie. At the extremes, the density is thin: fewer than 2 percent of patients are over 90,
and fewer than 1 percent have creatinine above 3.0 mg/dL.

The age curve at 90 is estimated from 140 patients, and the creatinine curve at 3.5 from 60
patients. Those regions are reported with a rug and a note about the counts. A curve without the
counts would make the extremes look as reliable as the middle.

### ICE curves and heterogeneity

The ICE curves for age cross. For patients with a high comorbidity burden, risk rises steeply with
age; for patients with a low burden, risk is nearly flat. The average partial dependence curve is a
compromise that describes neither group well. The team reports the ICE curves for three
representative comorbidity profiles instead of a single average.

### Permutation importance and grouped permutation

| Feature or group | Individual permutation | Grouped permutation |
|---|---:|---:|
| Prior admissions | 0.081 | 0.081 |
| Age | 0.052 | 0.124 (with comorbidity) |
| Creatinine | 0.028 | 0.031 |
| Comorbidity index | 0.044 | included with age |

Age and the comorbidity index are correlated at r = 0.68. Individually, age appears less important
than prior admissions; grouped with comorbidity, the pair is the most important group. The
individual values are not a partition of the model's performance, and the report says so.

### SHAP values and the additive guarantee

For one patient, the baseline logit is -0.90, and the patient's logit is -1.20. The SHAP
contributions are age -0.18, creatinine -0.08, and prior admissions -0.04, with the remaining
features contributing 0.00. The additive check is:

```text
-0.90 + (-0.18) + (-0.08) + (-0.04) = -1.20
```

The contributions sum to the prediction minus the baseline, as the guarantee requires. The
guarantee is satisfied, but the attribution of age and comorbidity is not unique: because the two
are correlated, the -0.18 assigned to age could be partly assigned to comorbidity under a different
reference distribution. The report states that the attribution is interventional and that the
age-comorbidity group is the identifiable unit.

Globally, the mean absolute SHAP values rank age first (0.091), prior admissions second (0.067), and
creatinine third (0.039). Permutation importance ranks prior admissions first (0.081), age second
(0.052), and creatinine third (0.028). The rankings disagree on the top feature.

### Two diagnostics that would reveal a broken interpretation

**Diagnostic 1: the SHAP-permutation rank disagreement.** The disagreement is the signal that age
and comorbidity are correlated. A report that presents one method's ranking without the other would
not reveal the ambiguity. The diagnostic is a side-by-side table, plus the correlation matrix and
the grouped permutation.

**Diagnostic 2: partial dependence at an impossible combination.** The partial dependence for
creatinine at 4.0 mg/dL is computed by averaging over the observed values of the other features,
including age. For a 40-year-old with no prior admissions, a creatinine of 4.0 is rare but possible;
for a 90-year-old with six prior admissions, it is more common. The curve at 4.0 mixes these
patients and can produce a risk estimate that no subgroup experiences. The diagnostic is to plot the
feature's joint distribution with age, or to use ALE, which does not extrapolate.

### Why this is not a causal analysis

The coefficient for age is 0.035, and the marginal effect at p = 0.20 is 0.56 percentage points per
year. Neither number means that aging one year causes a 0.56 percentage point increase in
readmission. Age is correlated with comorbidity, frailty, social support, and healthcare access,
and the model adjusts only for the features included in the table. The coefficient is the model's
association conditional on those features, not the effect of an intervention on age.

The same caution applies to the SHAP value of -0.18 for age in the example. The value describes how
the model's prediction changes when age is shifted in the reference distribution, not what would
happen to the patient if they aged a year. A causal claim would require a design that identifies the
effect, and no amount of attribution can substitute for that design.

### A proxy check

The team also checks whether the model relies on a feature that proxies for a protected attribute.
The comorbidity index is correlated with race at r = 0.31 in this cohort, because of differences in
access and diagnosis. The model does not include race, but the comorbidity index carries some of
its association. The subgroup performance analysis shows a lower PPV in the group with the highest
comorbidity-index values, which is partly a prevalence effect and partly a model effect. The report
states both and does not attribute the difference to a single cause.

### Stability across resamples

Across 200 bootstrap resamples, prior admissions appears in the top three features in 94 percent of
samples, age in 88 percent, and creatinine in 61 percent. The age-comorbidity group appears in the
top two in 97 percent. The individual creatinine rank is unstable, which is consistent with its
correlation with age and with its lower individual importance. The report presents the group
ranking as the stable result and the individual creatinine rank as exploratory.

### Varying one assumption: the subgroup aged 80 and over

The subgroup has 612 patients and 141 events. The SHAP values for creatinine are unstable: across
200 bootstrap resamples, the mean absolute value ranges from 0.018 to 0.061, and the rank of
creatinine moves between third and sixth in 35 percent of resamples. The overall ranking is not
reproducible in this subgroup. The conclusion changes from "creatinine is a consistent driver" to
"creatinine's importance is not established in the oldest patients", and the report says so.

The report also includes a note on the background sample: it was drawn from the held-out 2025
cohort, not from the 2019-2021 development data, because the explanation is about the deployed
population. That choice changes the baseline and therefore the attributions.

### The interpretation the team writes

The report says: "The model's strongest stable signal is the prior-admission count, followed by the
age-comorbidity group. Creatinine is associated with risk but its individual ranking is unstable and
correlated with age. The model does not use race, but the comorbidity index carries some of the
same association, and the subgroup PPV is lower in the highest-comorbidity group. These are
statements about the model, not about the causal effect of any feature." That paragraph is the
interpretation: it states the method, the stable result, the instability, the proxy concern, and the
scope.

### What would change the interpretation

The interpretation would change if the correlation between age and comorbidity fell at a new site,
because the group attribution would no longer be the right unit. It would change if a prospective
audit found that the model's age coefficient was driven by a documentation artefact. It would change
if a causal analysis found that the age association was confounded by a variable not in the model.
Each of these is a specific condition that the report should name.

### What to report

The primary figure is a partial dependence or ALE curve for each of the top features, with a rug
showing the data density and the supported range. The secondary figure is a SHAP beeswarm for the
top features, annotated with the correlation caveat. The table of coefficients, marginal effects,
individual and grouped permutation importance, and mean absolute SHAP values belongs in the
appendix. The report should state the method, the background sample, the correlation handling, the
resampling seed, and the scope of the claims.

## 7. Code, query, or calculation

The script fits a logistic model with two correlated features, computes coefficients and an average
marginal effect, measures individual and grouped permutation importance, and checks the additive
decomposition that SHAP generalises. The second feature is generated from the first, so its
coefficient and importance come entirely from the correlation.

```python
"""Marginal effects, permutation importance, and additive attribution. Standard library only."""
import math
import random


def fit(X, y, iterations=600, learning_rate=0.3, l2=0.01):
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    for _ in range(iterations):
        g, gb = [0.0] * d, 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            error = p - yi
            for j in range(d):
                g[j] += error * xi[j]
            gb += error
        for j in range(d):
            w[j] -= learning_rate * (g[j] / n + l2 * w[j])
        b -= learning_rate * gb / n
    return w, b


def predict(X, w, b):
    return [1 / (1 + math.exp(-max(-30, min(30, b + sum(wj * xj
            for wj, xj in zip(w, x)))))) for x in X]


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


rng = random.Random(12)
X, y = [], []
for _ in range(3000):
    x1 = rng.gauss(0, 1)
    x2 = 0.95 * x1 + rng.gauss(0, 0.25)
    z = -1.0 + 1.2 * x1
    X.append([x1, x2])
    y.append(1 if rng.random() < 1 / (1 + math.exp(-z)) else 0)

w, b = fit(X, y)
base = auroc(y, predict(X, w, b))
print("coef", [round(v, 3) for v in w], "intercept", round(b, 3), "AUROC", round(base, 3))

mean = [sum(x[j] for x in X) / len(X) for j in range(2)]
p_bar = 1 / (1 + math.exp(-(b + sum(wj * m for wj, m in zip(w, mean)))))
print("pbar", round(p_bar, 3), "AME x1", round(w[0] * p_bar * (1 - p_bar), 4))

for name, index in (("x1", [0]), ("x2", [1]), ("both", [0, 1])):
    permuted = [row[:] for row in X]
    values = [row[j] for row in X for j in index]
    random.Random(7).shuffle(values)
    for i, row in enumerate(permuted):
        for k, j in enumerate(index):
            row[j] = values[i * len(index) + k]
    print(name, "drop", round(base - auroc(y, predict(permuted, w, b)), 4))

x = X[0]
linear = b + sum(wj * xj for wj, xj in zip(w, x))
linear_bar = b + sum(wj * m for wj, m in zip(w, mean))
contributions = [wj * (xj - m) for wj, xj, m in zip(w, x, mean)]
print("additive check", round(linear - linear_bar, 4), round(sum(contributions), 4),
      "pred", round(predict([x], w, b)[0], 3))
```

Expected output:

```text
coef [0.777, 0.47] intercept -1.093 AUROC 0.787
pbar 0.255 AME x1 0.1474
x1 drop 0.1394
x2 drop 0.0376
both drop 0.2642
additive check -1.7593 -1.7593 pred 0.056
```

Reading the output:

- The coefficient for `x2` is 0.47, even though `x2` has no direct effect on the outcome. The model
  uses `x2` because it is correlated with `x1`. A coefficient table alone would suggest that `x2`
  matters, and a causal reading would be wrong.
- The average marginal effect of `x1` is 0.1474 at the mean predicted risk of 0.255. That is the
  probability-scale effect, not the coefficient. It is much larger than the coefficient because the
  link is nonlinear.
- Individual permutation gives `x1` 0.1394 and `x2` 0.0376. The sum is 0.177, less than the joint
  drop of 0.2642 when both are permuted. The individual values are not additive, because permuting
  one feature leaves the other available.
- The additive check is exact: the linear predictor difference equals the sum of the contributions.
  This is the property SHAP generalises; it is a decomposition, not a causal claim.
- The `x2` importance is entirely a correlation artefact. A grouped permutation of `x1` and `x2`
  together would give the joint importance, which is the defensible quantity.

The reusable functions are `fit`, `predict`, and the permutation loop. In a real analysis, replace
the synthetic data with the validation set and use a SHAP library for the attributions, but keep the
grouped permutation as a check.

## 8. Data, standards, and regulatory context

Interpretability is addressed by transparency requirements and reporting conventions, not by a
single binding technical standard. The table distinguishes requirements from conventions.

| Instrument | Status | What it says about explanation | What it does not do |
|---|---|---|---|
| FDA GMLP principles | Guidance, not binding | Calls for transparency about the model's inputs, outputs, and limitations | Does not prescribe an explanation method |
| FDA PCCP guidance | Guidance, not binding | Requires a description of planned changes and their impact | Does not require SHAP or PDP |
| TRIPOD+AI | Reporting convention | Requires the model's predictors and rationale to be reported | Does not require post-hoc explanation |
| PROBAST+AI | Appraisal tool | Assesses predictor handling and model development | Does not assess explanation quality |
| EU AI Act | Binding in the EU, phased | Requires transparency and instructions for use for high-risk systems | Does not prescribe a method |
| GDPR Article 22 | Binding in the EU | Provides rights around solely automated decisions | Does not define a technical explanation |
| ISO 14971 | Voluntary standard | Risk management across the lifecycle | Does not address explanation |
| NIST AI RMF | Voluntary framework | Calls for transparency and accountability | Not sector-specific |

Three points matter for practice.

**The explanation and the model card.** A model card should state which explanation method was
used, on which population, with which reference, and what the stable findings were. It should also
state what the explanation does not show: causal effects, individual-level mechanisms, or the
behaviour of the model outside the data envelope. That paragraph is often more useful to a clinician
than a beeswarm.

**Transparency is not the same as an explanation.** A model card that lists the features and the
performance is transparent, but it does not explain an individual prediction. A SHAP value explains
an individual prediction, but it does not by itself satisfy a transparency requirement. The two are
complementary, and the report should provide both.

**The GDPR right is narrower than it is often described.** Article 22 concerns solely automated
decisions with legal or similarly significant effects, and the right is to meaningful information
about the logic involved, not necessarily to a full technical explanation of a specific prediction.
The interpretation should not overstate the legal requirement, and legal advice is outside this
module.

**The EU AI Act adds transparency obligations for high-risk systems.** The obligations include
instructions for use and information about the system's characteristics, but they do not prescribe
SHAP, LIME, or partial dependence. A provider should describe the explanation method and its limits,
not claim that the method makes the model transparent in every sense.

### An interpretation reporting checklist

An interpretation section that can be audited contains the following.

```text
question and audience
scope: global or local; population or patient
model form and output scale
coefficients and marginal effects at representative risks
partial dependence or ALE with data density
ICE curves and heterogeneity
permutation importance, individual and grouped
SHAP method, background sample, and additive check
correlation diagnosis and feature groups
stability across resamples
comparison of methods and interpretation of disagreement
causal-language check
software and version
```

The checklist maps onto the transparency expectations of the FDA GMLP principles and the
predictor-handling items of TRIPOD+AI. A section that gives a SHAP plot without the method, the
background, or the correlation caveat is not reproducible.

### An interpretation reporting checklist

An interpretation section that can be audited contains the following.

```text
question and audience
scope: global or local; population or patient
model form and output scale
coefficients and marginal effects at representative risks
partial dependence or ALE with data density
ICE curves and heterogeneity
permutation importance, individual and grouped
SHAP method, background sample, and additive check
correlation diagnosis and feature groups
stability across resamples
comparison of methods and interpretation of disagreement
causal-language check
software and version
```

The checklist maps onto the transparency expectations of the FDA GMLP principles and the
predictor-handling items of TRIPOD+AI. A section that gives a SHAP plot without the method, the
background, or the correlation caveat is not reproducible.

> **Note:** No regulation requires a particular explanation method. The requirement is that the
> model's behaviour be describable and that the limits of the description be stated.

## 9. Industry practice and case evidence

**The case for inherently interpretable models.** Rudin (2019) argues that for high-stakes
decisions, an inherently interpretable model should be preferred when its performance is comparable,
because post-hoc explanations can be unstable, can disagree, and can create false confidence. The
argument is normative, but it is supported by the observation that explanation methods are
approximations and that the approximation can fail in ways the user cannot see.

**The limits of explanation in health care.** Ghassemi, Oakden-Rayner, and Beam (2021) argue that
current explainable AI methods often do not give clinicians the understanding they need, and that
the field should focus on evaluating whether explanations improve decisions rather than assuming
they do. The paper is a caution against treating an explanation as evidence of safety.

**SHAP and correlated features.** Kumar et al. (2020) show that Shapley-value explanations depend on
arbitrary choices about which features are treated as players, and that correlated or redundant
features can change the attribution even when the model is effectively unchanged. Aas et al. (2021)
show that KernelSHAP's independence assumption produces wrong answers under correlation, and that
TreeSHAP does not reliably fix it. The practical lesson is to report correlated features as a group.

**Partial dependence and extrapolation.** Apley and Zhu (2020) show that partial dependence
marginalises over the marginal distribution of the other features, which requires extrapolation when
features are correlated. They propose ALE, which averages local differences and avoids the
extrapolation. Hooker, Mentch, and Zhou (2021) make a related argument for permutation importance:
unrestricted permutation forces extrapolation and can over-emphasise correlated features.

**Permutation importance under correlation.** The simulation literature, including Nicodemus et al.
(2010), shows that permutation importance is affected by predictor correlation and that conditional
permutation schemes can change the ranking. The practical recommendation is to report individual and
grouped importance and to avoid treating the individual values as a partition.

**What the industry actually reports.** Many industry model cards report a SHAP summary plot and a
short list of top features, without the background sample, the correlation handling, or the
stability check. That is better than nothing, but it is not reproducible and it invites
over-interpretation. A card that reports the grouped importance, the stability, and the limits is
more useful even if the plot is less colourful.

**The distinction between explanation and prediction.** An explanation method does not improve
the model's predictions; it changes how the model is understood. A team that spends weeks on SHAP
and no time on validation has optimised the wrong artefact. The interpretation should follow the
validation, not replace it.

**The clinical use of explanations.** In practice, clinicians often want a small number of
plausible drivers and a sense of whether the model is relying on something implausible. A
well-designed report gives them the top stable group, the direction, the data density, and the
caveat about correlation. A beeswarm with 50 features is less useful than a table with five groups
and a clear statement of what was checked.

> **Note:** The cases are documented. The claim that a specific clinical explanation changed a
> decision is not made here; the evidence that explanations improve decisions is limited.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Coefficient read as a risk difference | "3.5 percent per year" | Log-odds scale confused with probability | Wrong clinical message | Report marginal effects |
| Partial dependence with correlated features | Curve extends into empty space | Marginalisation over impossible combinations | Misleading shape | Use ALE or conditional PDP |
| Permutation importance with correlated features | Ranking disagrees with SHAP | Permutation breaks dependence | Inflated or deflated importance | Grouped permutation |
| SHAP treated as causal | "Age causes readmission" | Attribution confused with effect | Wrong action | State model-behaviour scope |
| SHAP reference unstated | Baseline not described | Attribution depends on reference | Irreproducible | State background sample |
| Single local explanation generalised | One patient's reasons presented as the model's | Local/global confusion | Overgeneralisation | Report scope |
| Unstable feature rank | Rank changes across bootstrap | Correlated or weak signal | False confidence | Stability analysis |
| Explanation on training data | Values look too clean | In-sample fit | Optimistic attribution | Use held-out data |
| Interactions ignored | PDP flat while ICE curves vary | Average hides heterogeneity | Wrong interpretation | ICE or two-way analysis |
| High-cardinality one-hot spread | Importance split across many columns | Encoding artefact | Misleading ranking | Group levels |
| Proxy feature unflagged | Feature correlates with protected attribute | Indirect use of a protected attribute | Inequity | Proxy audit |
| Explanation not reproducible | Method or seed missing | Undocumented choices | Cannot audit | Document method |
| Explanation used as safety evidence | "We can explain it, so it is safe" | Explanation confused with validation | Unsupported claim | Separate validation |
| Model updated without re-explanation | Top features change silently | Behaviour drift | Stale interpretation | Recompute after updates |
| Group attribution omitted | Correlated features ranked individually | Non-identifiable split | Misleading driver | Report groups |
| Background sample biased | SHAP baseline differs from deployment | Reference mismatch | Wrong attribution | Compare distributions |
| Local explanation over-generalised | One patient's reasons in a policy document | Scope confusion | Wrong policy | State the scope |
| Explanation method mismatch | A global method used for a local question | Method does not answer the question | Misleading answer | Map question to method |
| No confidence in the attribution | A single SHAP value quoted | Sampling and reference uncertainty | False precision | Report range |
| Explanation used to justify a proxy | The proxy is a top feature | Attribution legitimises a shortcut | Inequity | Proxy audit |
| Method presented without limits | The report says SHAP "explains" | Overclaim | False trust | State scope |
| Explanation on a different population | Development explanation used at a new site | Distribution shift | Wrong drivers | Recompute locally |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The explanation population is specified | The claim is bounded | Population shift | Compare cohorts |
| The model is the one being explained | Attribution applies to the deployed model | Model updated | Version the model |
| The features are measured as in deployment | Attribution reflects real inputs | Different units or coding | Data audit |
| The correlation structure is known | Correlated features need grouping | Unmeasured correlation | Correlation matrix |
| The background sample is representative | SHAP baseline is meaningful | Biased background | Compare distributions |
| The explanation is stable | The ranking is reliable | Bootstrap instability | Resample |
| The method answers the question | The explanation is relevant | Method mismatch | Map question to method |
| The explanation is not causal | Scope is stated | Causal language | Review |
| The data density supports the curve | PDP is estimated where data exist | Extrapolation | Rug and counts |
| The explanation is reproducible | Audit is possible | Missing method or seed | Re-run |
| The explanation is stable across resamples | The ranking is reliable | Bootstrap instability | Resample |
| The correlation is measured in deployment | Grouping is valid | Different correlation structure | Site comparison |
| The explanation is labelled as model behaviour | Scope is clear | Causal language | Review |
| The method matches the question | The answer is relevant | Wrong method | Review |
| The group definition is clinical | Grouping is meaningful | Arbitrary groups | Clinical input |
| The explanation is recomputed after deployment | It describes the current model | Stale explanation | Version check |
| The explanation is not used as safety evidence | Validation remains separate | Explanation confused with validation | Review |

The deepest limit is that an explanation is a model of a model. It approximates the fitted function,
and the approximation can fail in ways that are invisible without a stability check. The report
should present the explanation as evidence about the model's behaviour, with its method and its
limits, not as a window into the patient's biology.

## 12. Exercises

The exercises move from the model-specific view to the model-agnostic views and then to the written
interpretation. Each asks for the scope, the method, and the correlation handling to be stated,
because an explanation without those is not reproducible.

**Exercise 1. Coefficient versus marginal effect.** For a logistic coefficient of 0.18 and predicted
risks of 0.05, 0.20, and 0.50, compute the marginal effect. Deliverable: three values and a sentence.
Acceptance criteria: values correct; interpretation states the dependence on baseline risk. Time: 30
minutes.

**Exercise 2. Partial dependence and density.** Compute a partial dependence curve for a feature and
overlay the data density. Deliverable: the plot and a 300-word interpretation. Acceptance criteria:
density shown; supported range stated; extrapolation flagged. Time: 60 minutes.

**Exercise 3. ICE curves.** Compute ICE curves for 50 patients for one feature and compare with the
average. Deliverable: the plot and a paragraph. Acceptance criteria: crossing curves identified;
average's representativeness assessed. Time: 45 minutes.

**Exercise 4. Permutation importance.** Compute individual and grouped permutation importance for a
model with correlated features. Deliverable: a table. Acceptance criteria: group defined; individual
values not presented as additive; difference explained. Time: 60 minutes.

**Exercise 5. SHAP additive check.** For a small model, verify that the SHAP values sum to the
prediction minus the baseline. Deliverable: the arithmetic. Acceptance criteria: check exact;
reference stated. Time: 30 minutes.

**Exercise 6. Correlation diagnosis.** Given two features correlated at 0.85, explain why their
individual attributions are not identifiable and propose a grouped analysis. Deliverable: a 400-word
memo. Acceptance criteria: mechanism explained; group analysis proposed. Time: 45 minutes.

**Exercise 7. Causal language audit.** Find a model report that uses causal language for a feature
attribution and rewrite the claims as model-behaviour statements. Deliverable: the rewritten
paragraph and a note on what causal evidence would be needed. Acceptance criteria: at least three
claims corrected; causal design named. Time: 45 minutes.

**A note on the exercises.** The exercises assume that the learner has access to a fitted model and
a held-out sample. Where that is not possible, the worked example provides the arithmetic, and the
learner can critique the published SHAP plots instead. The interpretation exercises are designed to be done on a model the
learner has already built in the module labs. Where no model is available, the worked example's
coefficient table can be used for the arithmetic, but the correlation and stability exercises need
predictions.

**Exercise 8. Interpretation section.** Write the interpretation section of a model development
report. Include the question, the scope, the method, the stable results, the correlation handling,
the stability check, and the limits. Deliverable: two pages. Acceptance criteria: method, scope,
stability, correlation handling, and limits all present. Time: 75 minutes. Write the interpretation section of a model development
report. Deliverable: two pages. Acceptance criteria: method, scope, stability, correlation handling,
and limits all present. Time: 75 minutes.

## 13. Self-check answers

**Answer to Exercise 1 (interpretation note).** The three values show that the same coefficient can
imply a small or a large probability effect depending on the baseline. A clinical report should state
the baseline risk at which the marginal effect was computed.

**Answer to Exercise 2 (what a strong PDP shows).** A strong PDP shows the curve, the rug or
histogram, the supported range, and the count of patients at each extreme. It states that the curve
is an average and that ICE curves are needed if the individual shapes differ. It does not present
the curve as a causal effect.

**Answer to Exercise 1.** At 0.05: `0.18 * 0.05 * 0.95 = 0.00855`. At 0.20: `0.18 * 0.20 * 0.80 =
0.0288`. At 0.50: `0.18 * 0.50 * 0.50 = 0.045`. The marginal effect rises with the baseline risk
until 0.5 and then falls. The coefficient is constant; the probability-scale effect is not.

**Answer to Exercise 5.** If the baseline prediction is 0.20 and the patient's prediction is 0.35,
the contributions must sum to `0.35 - 0.20 = 0.15` on the probability scale, or to the corresponding
difference on the log-odds scale if the SHAP values are on that scale. The check is arithmetic: sum
the contributions and compare with the difference. A discrepancy indicates a different baseline, a
different scale, or an implementation error.

**Answer to Exercise 4 (what a strong permutation table shows).** A strong table reports the
metric, the validation sample, the individual importance, and the grouped importance for
pre-specified correlated groups. It states that individual values are not additive and that the
group value is the defensible quantity. It also reports the stability of the ranking across
resamples.

**Answer to Exercise 6.** Two features correlated at 0.85 carry largely the same information. The
model's prediction changes little when one is changed while the other is held fixed, because the
model can use the other. The individual attribution depends on the reference and the algorithm, and
the credit can be split arbitrarily. The defensible analysis groups the two features and reports the
group's joint importance, with the individual values labelled as non-identifiable.

**Answer to Exercise 7 (what a strong rewrite looks like).** A strong rewrite changes "age causes
readmission" to "the model associates older age with higher predicted risk, conditional on the
other features", and "creatinine drives the score" to "creatinine is one of the features the model
uses, and its individual contribution is not separable from age because the two are correlated". It
then names the causal design that would be needed for the stronger claim: a target trial, an
instrumental variable, or a longitudinal design with time-varying confounders.

**Answer to Exercise 8 (what a complete section contains).** A complete section states the
question, the scope, the method, the model version, the result, the stability check, the correlation
handling, the comparison with at least one other method, and the limits. It includes one figure of
the model's behaviour with the data density and one table of stable groups. It does not include a
causal claim unless a causal design is cited.

**Rubric for Exercises 2, 4, 6, 7, and 8.** A strong submission (1) states the question and scope,
(2) uses a method that matches the question, (3) reports data density and correlation, (4) checks
stability, (5) avoids causal language, and (6) states the limits. A competent submission meets four
of the six. A weak submission presents a single SHAP ranking as the explanation without caveats.

## 14. Interview questions

**1. What is the difference between a coefficient and a marginal effect?** A strong answer explains
the log-odds scale and the dependence on baseline risk. A weak answer treats them as the same.

**2. Why can partial dependence be misleading?** A strong answer explains marginalisation and
extrapolation under correlation. A weak answer says it is always reliable.

**3. What does SHAP guarantee?** A strong answer states the additive decomposition and the baseline,
and denies uniqueness or causality. A weak answer says SHAP explains the true reason for a
prediction.

**4. Why do SHAP and permutation importance disagree?** A strong answer explains correlation and
the different questions the methods ask. A weak answer says one of them is wrong.

**5. When would you prefer an inherently interpretable model?** A strong answer says when
performance is comparable and the explanation burden or regulatory context favours transparency. A
weak answer says always or never.

**6. How do you check explanation stability?** A strong answer bootstraps and reports rank or value
stability. A weak answer says to look at the plot.

**7. What is the difference between a local and a global explanation?** A strong answer defines each
and notes that they can disagree. A weak answer uses the terms interchangeably.

**8. Can an explanation be causal?** A strong answer says no without a causal design, and names what
would be needed. A weak answer says a strong model attribution is causal evidence.

**9. How do you handle correlated features in an explanation?** A strong answer groups them, uses
conditional methods, and reports the group. A weak answer picks the feature with the larger SHAP
value.

**10. When is a feature's SHAP value not identifiable?** A strong answer says when it is collinear with another feature and the reference does not distinguish them. A weak answer says SHAP values are always identifiable.

**11. What is the difference between a proxy and a cause?** A strong answer says a proxy is correlated with an unmeasured construct and the model may use it for that reason, while a cause is identified by a design. A weak answer treats a strong association as causal.

**12. How do you explain a model to a patient?** A strong answer uses a small number of plain-language drivers, states that they describe the model rather than the person, and avoids false precision. A weak answer reads SHAP values aloud.

**13. What is the risk of explaining a model with a proxy feature?** A strong answer says the explanation may legitimise an inequitable proxy, and a fairness review is needed. A weak answer says the model is transparent so it is fair.

**14. What is the difference between stability and robustness of an explanation?** A strong answer says stability is consistency across resamples and robustness is consistency under small input changes. A weak answer uses the terms interchangeably.

**15. Why is a beeswarm plot not a causal diagram?** A strong answer says it summarises model attributions, not intervention effects, and the ordering of features does not identify causes. A weak answer treats the plot as a mechanism.

**16. What would you put in a model card's interpretability section?** A strong answer includes the
method, the top stable features or groups, the correlation handling, the stability check, and the
limits. A weak answer includes a SHAP plot without context.

## 15. Key terms

- **Additive guarantee** — contributions sum to the prediction minus the baseline.
- **ALE** — accumulated local effects.
- **Average marginal effect** — mean marginal effect over the sample.
- **Baseline** — reference prediction in a SHAP decomposition.
- **Causal effect** — effect of an intervention on an outcome.
- **Coefficient** — parameter on the model's linear scale.
- **Collinearity** — high linear correlation among features.
- **Correlation** — statistical association between features.
- **Counterfactual** — hypothetical input with a changed feature.
- **Data envelope** — region of feature space where data exist.
- **Explanation audit** — review of explanation stability and scope.
- **Extrapolation** — evaluating outside the data envelope.
- **Faithfulness** — whether an explanation reflects the model's computation.
- **Feature attribution** — numerical contribution assigned to a feature.
- **Feature importance** — ranking of features by contribution.
- **Global explanation** — statement about the model across the population.
- **Grouped permutation** — shuffling correlated features together.
- **ICE curve** — individual conditional expectation.
- **Interaction** — a feature's effect depends on another.
- **Interpretability** — degree to which behaviour can be understood.
- **KernelSHAP** — model-agnostic SHAP approximation.
- **LIME** — local interpretable model-agnostic explanation.
- **Local explanation** — statement about one prediction.
- **Marginal effect** — change in probability per unit change in a feature.
- **Model card** — documentation of intended use, performance, and limits.
- **Monotonicity** — constraint that predictions move in one direction.
- **Odds ratio** — exponentiated coefficient.
- **Partial dependence** — average prediction as a function of one feature.
- **Permutation importance** — performance loss when a feature is shuffled.
- **Plausibility** — whether an explanation matches domain knowledge.
- **Proxy feature** — feature standing in for an unmeasured construct.
- **SHAP** — Shapley additive explanations.
- **Shapley value** — game-theoretic attribution.
- **Stability** — consistency across resamples.
- **Surrogate model** — simple model approximating a complex one.
- **TreeSHAP** — SHAP algorithm for tree ensembles.

## 16. Sources

All URLs accessed 25 October 2026.

1. Rudin C. Stop explaining black box machine learning models for high stakes decisions and use
   interpretable models instead. *Nature Machine Intelligence* 2019;1:206-215.
   https://doi.org/10.1038/s42256-019-0048-x
2. Ghassemi M, Oakden-Rayner L, Beam AL. The false hope of current approaches to explainable
   artificial intelligence in health care. *Lancet Digital Health* 2021;3(11):e745-e750.
   https://doi.org/10.1016/S2589-7500(21)00208-9
3. Apley DW, Zhu J. Visualizing the effects of predictor variables in black box supervised learning
   models. *Journal of the Royal Statistical Society: Series B* 2020;82(4):1059-1086.
   https://doi.org/10.1111/rssb.12377
4. Hooker G, Mentch L, Zhou S. Unrestricted permutation forces extrapolation: variable importance
   requires at least one more model, or there is no free variable importance. *Statistics and
   Computing* 2021;31:82. https://doi.org/10.1007/s11222-021-10057-z
5. Kumar IE, Venkatasubramanian S, Scheidegger C, Friedler S. Problems with Shapley-value-based
   explanations as feature importance measures. *Proceedings of the 37th International Conference on
   Machine Learning*, 2020. https://proceedings.mlr.press/v119/kumar20e.html
6. Aas K, Jullum M, Løland A. Explaining individual predictions when features are dependent: more
   accurate approximations to Shapley values. *Artificial Intelligence* 2021;298:103502.
   https://arxiv.org/abs/1903.10464
7. Molnar C, König G, Herbinger J, et al. General pitfalls of model-agnostic interpretation methods
   for machine learning models. *Lecture Notes in Computer Science*, 2022.
   https://doi.org/10.1007/978-3-031-04083-2_4
8. Nicodemus KK, Malley JD, Strobl C, Ziegler A. The behaviour of random forest permutation-based
   variable importance measures under predictor correlation. *BMC Bioinformatics* 2010;11:110.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC2848005/
9. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *Advances in Neural
   Information Processing Systems* 30, 2017. https://arxiv.org/abs/1705.07874
10. Ribeiro MT, Singh S, Guestrin C. "Why should I trust you?": explaining the predictions of any
    classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery
    and Data Mining*, 2016. https://doi.org/10.1145/2939672.2939778
11. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
    https://doi.org/10.1136/bmj-2023-078378
12. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
    https://doi.org/10.1136/bmj-2024-082505
13. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
    Guiding Principles. 2021.
    https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
14. European Union. Regulation (EU) 2024/1689 (Artificial Intelligence Act). 2024.
    https://eur-lex.europa.eu/eli/reg/2024/1689/oj
15. European Union. Regulation (EU) 2016/679 (General Data Protection Regulation), Article 22.
    https://eur-lex.europa.eu/eli/reg/2016/679/oj
16. National Institute of Standards and Technology. Artificial Intelligence Risk Management
    Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
17. SHAP documentation. https://shap.readthedocs.io/
18. Molnar C. *Interpretable Machine Learning*. 2nd edition, 2022.
    https://christophm.github.io/interpretable-ml-book/

An explanation is a claim with a scope, a method, and a limit. A report that states all three is
more useful than one that states none, even if the figure is less impressive.

The interpretation is not the last word; it is the evidence that the model behaves as the team
claims. The next topic turns to the uncertainty in that behaviour and in the model's predictions. An
explanation without an uncertainty statement is a point estimate presented as a fact.

Scope, method, and limits: those three words are the whole discipline.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_thresholds-and-decision-curves.md`](09_thresholds-and-decision-curves.md)
- Next topic: [`11_uncertainty.md`](11_uncertainty.md)
- Model families: [`05_model-families.md`](05_model-families.md)
- Fairness and subgroups: [`12_fairness-and-subgroups.md`](12_fairness-and-subgroups.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)


