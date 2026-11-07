---
collection: "fraud_modelling_playbook"
file_id: "02"
topic: "Calibration and probability scoring"
level: 3
status: "draft"
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Changes to outcome definition, label maturity, sampling, population, or calibration method"
notation_style: "plain-text"
prerequisites: ["Binary outcomes, conditional probability, logarithms, and temporal validation"]
pattern_tags: ["calibration", "proper-scoring", "fraud", "distribution-shift"]
source_status: "OPEN"
owner_role: "Fraud analytics lead"
source_index: "INDEX.md"
---
# Calibration and Probability Scoring

## 1. File Header

**Purpose:** Decide whether a fraud-risk probability is meaningful for a defined population and action.
**Reader:** A practitioner evaluating, calibrating, or monitoring a binary event predictor.
**Outcome:** Compute proper scores, diagnose reliability, and design a leakage-safe calibration evaluation.
**Scope:** Binary probabilities and their operational interpretation in fraud and transferable data-science applications.
**Prerequisites:** Conditional probability, binary classification, logarithms, and held-out evaluation.
**Level:** Advanced / production.
**Status:** Draft; not independently reviewed or published.
**Evidence:** DERIVED identities; ILLUSTRATIVE constructed cohorts; empirical performance NOT MEASURED.
**VERIFIED arithmetic:** Independent Python arithmetic checked the Section 5 Brier loss (0.16), log loss (0.500402), AUC (0.8), Section 9 byte conversion (8.583 MiB), and C8 target probability (0.1); the inputs remain illustrative.
**Source scope:** The [fraud methods source plan](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-methods-02) and [directory](../../../../../Templates/directory.txt) establish topic scope, not deployment outcomes.
**Related draft:** [Fraud evaluation and capacity](01_Fraud_Evaluation_Calibration_and_Capacity.md) develops action selection.
**Boundary:** A probability is not proof of fraud and does not by itself justify blocking or denial.
**Notation:** All logarithms are natural unless stated otherwise; rates and probabilities are dimensionless.
**Takeaway:** First define the event and population; only then interpret a numerical probability.

## 2. Concept Map / Mental Model

```text
Decision-time eligible event -> score -> calibration map -> predicted probability
            |                    |                              |
            |                    +-> ranking evaluation        +-> probability evaluation
            |                                                   |
            +-> intervention -> label observation and delay <--+
                                    |
                                    v
                 mature representative audit -> reliability and proper scores
                                    |
                                    v
               decision policy -> monitoring -> governed recalibration
```
A score orders cases only to the extent that it separates positives from negatives.
A probability claims a frequency interpretation for a specified target event and population.
Calibration is a conditional statement about observed frequency given predicted probability.
Ranking is a relative ordering statement, not a probability statement.
A constant base-rate forecast can be calibrated while offering no individual discrimination.
A perfectly ordered score can have systematically incorrect probability values.
A proper score combines calibration and discriminatory resolution; it does not isolate either one.
The quality of a decision also depends on action effects, costs, capacity and fairness constraints.
These are additional assumptions, not consequences of score calibration.
A workflow intervention may change whether fraud occurs or whether it becomes observable.
Therefore a post-intervention label need not equal the no-intervention target.
The analysis should name which potential outcome is intended.
Fraud outcomes commonly arrive late, are contested, or are observable only for reviewed cases.
An unbiased-looking mature subset can still be selectively labeled.
Monitoring must preserve both the original prediction and the policy that acted on it.
The scoring contract includes eligibility, horizon, target, model version and calibration version.
It also includes the observation process, sample selection, and available decision-time information.
An operational alert is a prompt for investigation, not a finding of wrongdoing.
**Takeaway:** Separate ordering, probability accuracy, observation, and intervention value.

## 3. Foundational Definitions

The following objects refer to a fixed decision-time eligible population unless a subscript says otherwise.
A record is one eligible transaction, not an account or customer unless explicitly declared.
Let `i` identify one transaction and `n` the number of evaluated transactions.
Let `Y_i` be the binary outcome: 1 for the specified event within the fixed horizon, 0 otherwise.
The event definition must state whether an intervention is included in the outcome.
Let `S_i` be a real-valued model score; larger values conventionally indicate higher risk.
Let `P_i` be the forecast probability in the closed interval from zero to one.
A calibration map transforms `S_i` into `P_i`; an arbitrary score is not automatically a probability.
Let `pi` denote the prevalence, the population mean of `Y`.
Let `B` denote a score bin and `n_B` its count of evaluated records.
Let `bar_p_B` denote the average forecast in bin B.
Let `bar_y_B` denote the observed positive fraction in bin B.
Let `R_i` indicate that a trustworthy label is observed and mature.
Let `q_i` be the known probability of audit inclusion when a randomized audit is used.
Let `C_FP` and `C_FN` be nonnegative false-positive and false-negative costs in currency per event.
Let `t` be a probability threshold, inclusive of ties according to a declared rule.
Let `a` and `b` be the fitted logistic calibration intercept and slope, respectively.
Let `logit(p)` mean the log odds for a probability strictly between zero and one.
Let `sigma(z)` be the logistic function, one divided by one plus exp(-z).
Population calibration means the conditional mean outcome given the forecast equals the forecast almost surely.
In a finite sample, only noisy estimates of conditional frequency are available.
Perfect marginal calibration does not imply calibration in every subgroup.
It also does not imply that all useful information is encoded in the forecast.
Calibration-in-the-large compares average predicted probability with the event rate.
Calibration slope measures a particular logistic recalibration relationship, not all calibration defects.
Reliability diagrams compare predicted and observed frequencies in bins.
Expected calibration error (ECE) is a chosen weighted summary of bin-level absolute gaps.
ECE varies with bin choices and has sampling bias; small ECE is not proof of exact calibration.
A proper scoring rule encourages honest probability reporting in expectation under its assumptions.
Brier loss is squared error between a binary outcome and a probability.
Log loss penalizes improbable realized outcomes especially strongly.
Resolution describes differences in outcome frequency among forecast groups in a Brier decomposition.
AUC is a pairwise ranking summary with a stated convention for ties.
A cohort is the eligible set selected by decision time and outcome horizon.
A mature cohort has had the declared observation window and label-processing lag.
Maturity does not guarantee complete labels or eliminate selective observation.
Covariate shift changes feature distribution while preserving the conditional outcome law given features.
Label shift changes class prevalence while preserving the class-conditional feature or score distribution.
Concept shift changes the conditional outcome relationship; simple prior correction generally fails.
These assumptions concern distributions, not a single drift statistic.
**Takeaway:** Calibration is conditional frequency agreement for a precisely defined event and population.

## 4. Core Formulas with Full Derivation

### 4.1 Base: calibration and reliability

Validity: Binary outcomes with a defined forecast and target population; the conditional expectation is a population quantity.
Derivation: The expectation of an indicator equals its conditional event probability.
Thus a forecast is calibrated if that conditional probability equals the forecast value almost surely.
```text
FORMULA [C1]: E[Y | P] = P almost surely.
WHERE: Y binary outcome; P forecast probability, both dimensionless random variables.
CONSTRAINT: Same eligible population, outcome definition and horizon for forecasts and outcomes.
NUMERICAL EXAMPLE: For 100 equally forecast 0.20, 20 positives yield empirical frequency 0.20.
INTERPRETATION: This single group is consistent with calibration there; it does not prove population calibration.
```
Boundary: A finite group of 100 is noisy, and a group of mixed forecasts tests only its average.
Independent check: 20 divided by 100 is 0.20; the outcome count is in events.

Validity: Every record in a nonempty bin has a forecast and a valid label.
Derivation: Replace the two population expectations in C1 with arithmetic means over the bin.
```text
FORMULA [C2]: gap_B = bar_y_B - bar_p_B.
WHERE: bar_y_B = sum(Y_i in B)/n_B; bar_p_B = sum(P_i in B)/n_B.
CONSTRAINT: n_B > 0; binning and inclusion rules are fixed and reported.
NUMERICAL EXAMPLE: n_B=100, 20 positives, sum of probabilities=15; gap_B=0.20-0.15=0.05.
INTERPRETATION: Observed event rate exceeds the average forecast by 5 percentage points in this sample.
```
Boundary: Bin averaging can hide compensating errors inside a bin and cannot prove individual calibration.
Independent check: The expected count is 15, compared with 20 observed, a difference of five events.

### 4.2 General: proper scoring and logistic recalibration

Validity: Binary Y, probability p in [0,1], finite mean and an identical outcome population for comparisons.
Derivation: Under true event probability r, expected squared loss is r(1-p)^2+(1-r)p^2.
Expanding and completing the square gives r(1-r)+(p-r)^2.
The minimum is attained uniquely at p=r, so this is a strictly proper loss.
```text
FORMULA [C3]: BS = (1/n) * sum_i (P_i-Y_i)^2.
WHERE: n positive integer; P_i in [0,1]; Y_i in {0,1}; BS dimensionless.
CONSTRAINT: Evaluate the same defined cohort and target; lower is better.
NUMERICAL EXAMPLE: p=(0.2,0.8), y=(0,1); BS=(0.04+0.04)/2=0.04.
INTERPRETATION: Average squared probability error is 0.04; it combines reliability and resolution.
```
Boundary: A constant forecast at the base rate can beat a bad discriminator on Brier loss.
Independent check: Both errors have magnitude 0.2; squaring yields 0.04 each.

Validity: Binary Y; probabilities assigned to realized outcomes must be positive for finite log loss.
Derivation: The likelihood for one Bernoulli observation is p^y(1-p)^(1-y).
Taking the negative natural logarithm gives the per-record log loss.
Expected loss is minimized at the true probability, including boundary limits.
```text
FORMULA [C4]: LL = -(1/n) * sum_i [Y_i*ln(P_i)+(1-Y_i)*ln(1-P_i)].
WHERE: n positive integer; P_i in [0,1]; Y_i binary; LL in nats/event.
CONSTRAINT: Treat zero times log zero by continuity; an impossible realized outcome has infinite loss.
NUMERICAL EXAMPLE: p=(0.2,0.8), y=(0,1); LL=-ln(0.8) approximately 0.223144 nats/event.
INTERPRETATION: Both realized outcomes were assigned probability 0.8.
```
Boundary: Clipping changes the evaluated forecast and must be disclosed; it is not a cure for bad certainty.
Independent check: The two realized-outcome probabilities are both 0.8.

Validity: A fixed score is mapped using coefficients estimated on disjoint calibration data.
Derivation: Logistic regression makes log odds affine in a scalar score and inverts log odds via sigma.
```text
FORMULA [C5]: p_cal = sigma(a+b*s) = 1/(1+exp(-(a+b*s))).
WHERE: s scalar score; a intercept and b slope; p_cal dimensionless in (0,1).
CONSTRAINT: Fit a and b without using final test labels; score direction and model form are declared.
NUMERICAL EXAMPLE: a=0, b=1, s=0; p_cal=1/(1+1)=0.5.
INTERPRETATION: The fitted model maps score zero to one-half for these illustrative parameters.
```
Boundary: A negative fitted slope reverses ranking; a zero slope collapses it.
A positive slope preserves strict ordering, but neither sign ensures out-of-sample calibration.
Independent check: The log odds of 0.5 are zero.

### 4.3 Variants: bins, baseline and prior correction

Validity: Nonempty disjoint bins cover a labeled cohort, with fixed bin boundaries or reported construction.
Derivation: Take a weighted average of the absolute empirical calibration gaps C2.
```text
FORMULA [C6]: ECE = sum_B (n_B/n) * abs(bar_y_B-bar_p_B).
WHERE: B indexes bins; n=sum_B n_B; all gaps and weights dimensionless.
CONSTRAINT: n>0; chosen binning, sample selection and uncertainty must be reported.
NUMERICAL EXAMPLE: Two equally sized bins have absolute gaps 0.02 and 0.04; ECE=0.03.
INTERPRETATION: The bin-weighted absolute gap is 3 percentage points for this binning.
```
Boundary: If within-bin errors cancel, ECE may be zero for a miscalibrated predictor.
Independent check: Equal weights produce the simple average of 0.02 and 0.04.

Validity: Constant forecast p0 is chosen before evaluation for a binary population with prevalence pi.
Derivation: Substitute the constant forecast into the expected Brier loss from C3 and differentiate.
The optimal constant is p0=pi and its expected Brier loss is pi(1-pi).
```text
FORMULA [C7]: BS_constant = pi*(1-pi), for p0=pi.
WHERE: pi is population prevalence; BS_constant is dimensionless expected loss.
CONSTRAINT: The stated prevalence is correct for the evaluated population; forecast is constant.
NUMERICAL EXAMPLE: pi=0.01 gives BS_constant=0.01*0.99=0.0099.
INTERPRETATION: A low raw Brier score can occur simply because fraud is rare.
```
Boundary: A prevalence estimated from test labels is an optimistic oracle benchmark, not a deployable forecast.
Independent check: 1% positives contribute 0.01*(0.99)^2 and 99% negatives contribute 0.99*(0.01)^2.

Validity: Pure label shift; class-conditional score distributions are unchanged and both priors are strictly between zero and one.
Derivation: Bayes odds equal likelihood ratio times prior odds.
Replace the source prior odds with the target prior odds while retaining the score likelihood ratio.
```text
FORMULA [C8]: odds_target(s) = odds_source(s) * [pi_target/(1-pi_target)] / [pi_source/(1-pi_source)].
WHERE: odds_source(s)=p_source(s)/(1-p_source(s)); pi denotes class prevalence.
CONSTRAINT: Source p_source(s) is calibrated for its population; pure label shift and support hold.
NUMERICAL EXAMPLE: pi_source=0.5, pi_target=0.1, p_source(s)=0.5; target odds=1/9, target p=0.1.
INTERPRETATION: A score with no source evidence beyond prior odds receives the target base rate.
```
Boundary: Changing class-conditional score distributions or policy-dependent labels invalidates the simple correction.
Independent check: Target odds 1/9 convert to probability (1/9)/(1+1/9)=0.1.
Target prevalence must be estimated independently or with a validated label-shift estimator and uncertainty.
Case-control sampling correction is a special use of the same odds principle when selection depends only on class.

### 4.4 Limits, counterexamples and decision thresholds

Validity: Two available actions, calibrated no-action risk, known constant error costs, and no capacity constraint.
Derivation: Expected false-positive cost of acting is (1-p)*C_FP.
Expected false-negative cost of not acting is p*C_FN.
Act if the first cost is strictly smaller than the second; define a tie policy.
```text
FORMULA [C9]: act if p > C_FP/(C_FP+C_FN).
WHERE: p event probability; C_FP and C_FN currency/event, both nonnegative.
CONSTRAINT: Positive cost sum; binary action removes the stated false-negative loss without other effects.
NUMERICAL EXAMPLE: C_FP=1, C_FN=9; threshold=1/(1+9)=0.10.
INTERPRETATION: Above 10% risk, acting has smaller modeled expected error cost.
```
Boundary: A review may not prevent fraud and blocking may change customer behavior; model actual incremental consequences.
Independent check: At p=0.1, both modeled costs equal 0.9 currency/event.
Capacity limits require joint allocation rather than independent thresholding.

Counterexample to ranking implying calibration: predictions 0.9 for every positive and 0.8 for every negative rank perfectly.
When all positive forecasts are 0.9, their observed positive frequency is 1.0, not 0.9.
When all negative forecasts are 0.8, their observed positive frequency is 0.0, not 0.8.
Counterexample to calibration implying ranking: a constant correct base-rate probability is calibrated in population.
That constant predictor ties all pairs and has AUC 0.5 under the half-credit tie convention.
Counterexample to aggregate subgroup reliability: equal-sized groups can have opposing errors that cancel.
Suppose both groups receive forecast 0.5, but true group event rates are 0.2 and 0.8.
The overall rate is 0.5, yet neither group is calibrated.
The subgroup rates are constructed assumptions, not measured results.
**Takeaway:** Proper scoring, reliability, ranking and cost-sensitive action are different mathematical questions.

## 5. Worked Numerical Example

### Cohort and prediction contract

**ILLUSTRATIVE:** A fraud team evaluates ten fully labeled, mature transactions.
The target is confirmed fraud within a fixed horizon under the observed historical policy.
The events are equally weighted and each row is one transaction.
The small data set is pedagogical and not representative of an operational prevalence.
The probability is the model's forecast for that specific target.
The five low forecasts are 0.2 and the five high forecasts are 0.8.
The low group has one positive; the high group has four positives.
A within-group order is irrelevant to these computations.

| Forecast group | Count | Positives | Negatives | Forecast sum | Observed rate |
|---|---:|---:|---:|---:|---:|
| Low | 5 | 1 | 4 | 1.0 | 0.2 |
| High | 5 | 4 | 1 | 4.0 | 0.8 |
| Total | 10 | 5 | 5 | 5.0 | 0.5 |

The mean probability is 5/10 = 0.5.
The observed event rate is also 5/10 = 0.5.
The high and low groups each match their forecast exactly in this constructed sample.
That exact fit is not evidence that the population predictor is calibrated.

### Compute Brier loss

Apply C3 to the four forecast/outcome cells.
The low positive contributes one times 0.8 squared, or 0.64.
The four low negatives contribute four times 0.2 squared, or 0.16.
The four high positives contribute four times 0.2 squared, or 0.16.
The high negative contributes one times 0.8 squared, or 0.64.
Total squared loss is 1.60; dividing by ten gives 0.16 per event.
The constant 0.5 forecast scores 0.25 on every row and has Brier loss 0.25.
The difference of 0.09 reflects improved resolution in this specific cohort.
The empirical group frequencies happen to match forecasts, so the grouped reliability term is zero.

### Compute log loss

Eight observations have realized-outcome probability 0.8.
Two observations have realized-outcome probability 0.2.
By C4, the sum of losses is -8 ln(0.8)-2 ln(0.2).
This is approximately 5.004024 nats for ten events.
The average is approximately 0.500402 nats per event.
The constant 0.5 forecast has average log loss approximately 0.693147 nats per event.
The two measures agree on the preferred forecast here but may rank models differently elsewhere.

### Check pairwise ranking

There are five positives and five negatives, hence 25 positive-negative pairs.
Four high positives outrank four low negatives, yielding 16 strictly correct pairs.
Within high and low groups there are four ties each, for eight tied pairs.
The low positive loses to the high negative, yielding one incorrect pair.
Giving each tie half credit gives AUC (16+4)/25 = 0.8.
The pair counts sum to 16+8+1 = 25.
A constant forecast has AUC 0.5 by the same tie convention.

### Reliability, uncertainty and sensitivity

With the two stated bins, both C2 gaps are zero and C6 ECE is zero.
The two bins contain only five observations each; uncertainty is substantial.
The event rates are discrete in increments of 0.2 for these bin sizes.
A narrow-bin frequency cannot establish a smooth population reliability curve here.
If one low-group negative becomes positive, low-group rate becomes 0.4 and its gap becomes +0.2.
Holding forecasts and all other outcomes fixed gives ECE 0.5*0.2+0.5*0 = 0.1.
The total event prevalence then becomes 0.6, not 0.5.
This is a sensitivity thought experiment, not a representative drift estimate.
If fraud labels were only obtained on high-score reviews, the low-group quantities would be unidentified without additional assumptions or audits.
The same apparent scores cannot justify a no-intervention risk if actions affected outcomes.

### Decision implication

If C9's strong assumptions and costs of 1 and 9 currency/event held, the threshold would be 0.1.
Both forecast groups exceed that threshold, so an unconstrained binary decision would act on all ten.
This result is a consequence of the assumed costs, not a recommendation to block transactions.
A real review action needs a measured or justified effect on losses and customer friction.
Capacity, treatment effects and policy constraints can change the optimum.
**Takeaway:** On the same ten events, calibration gaps, Brier, log loss and AUC answer distinct questions.

## 6. Visual / Geometric Intuition

A reliability diagram puts the average forecast on the horizontal axis.
It puts the observed fraction of positives on the vertical axis.
Each plotted point represents a reported bin, including its count.
The diagonal corresponds to equal predicted and observed frequency.
A point above the diagonal has more observed events than predicted in that bin.
A point below the diagonal has fewer observed events than predicted.
A point on the diagonal is compatible with calibration for the bin average only.
Plot uncertainty intervals and the score histogram beside the diagram.
Avoid smoothing away regions with little data or no observed positives.
Use identical bin definitions when making directly comparable time-window plots.
Quantile bins may be useful for count balance, but their probability boundaries change by cohort.
Fixed-width bins retain comparable probability ranges, but can be sparse in rare-event problems.
Neither binning convention is universally best; disclose it and inspect alternatives.
The histogram explains where the forecast distribution has support.
Extrapolating a calibration curve into an empty score region is unsupported.
AUC instead compares the ordering of positive-negative pairs.
A monotone increasing transformation preserves strict pair order and existing ties.
Thus a drastic probability change can leave AUC unchanged.
The Brier loss is squared distance from each binary outcome on the probability line.
Squaring makes a confident error larger than a modest error.
Log loss has a steep boundary near the probability assigned to a realized event of zero.
In a forecast-outcome scatter, forecasts near zero for true positives are particularly costly under log loss.
Calibration only measures one aspect of the joint distribution of scores and outcomes.
A constant calibrated forecast occupies a single point on a reliability diagram.
A more resolved calibrated forecast can occupy several points near the diagonal.
The latter can separate risk groups while retaining meaningful probabilities.
A decision threshold is a vertical line on the probability axis under its cost assumptions.
Changing the cost ratio changes that line without changing model calibration.
Changing review capacity can instead require ranking by estimated incremental value.
A favorable diagram cannot prove that intervention reduces fraud.
**Takeaway:** Read the reliability curve, support, uncertainty and decision context together.

## 7. Comparison / Selection Table

| Method | Main question | Preconditions | Useful when | Avoid / boundary |
|---|---|---|---|---|
| AUC | Are positives ranked above negatives? | Comparable outcomes and tie convention | Ordering candidates | Treating it as probability accuracy |
| Brier | What is mean squared probability error? | Same target and cohort | Overall probabilistic comparison | Reading low loss as calibration alone |
| Log loss | How much probability was assigned to realized outcomes? | Same target; boundary handling | Penalizing confident mistakes | Hiding clipping or comparing different labels |
| Reliability plot | Do forecasts match rates by bin? | Representative mature labels | Local diagnostic | Claiming bins prove population calibration |
| ECE | How large are chosen bin gaps? | Disclosed bins and coverage | Compact supplemental diagnostic | Comparing incompatible binning as ground truth |
| Logistic map | Can a simple monotone map correct score scale? | Separate fit data, adequate support | Limited calibration sample and plausible sigmoid | Assuming a sigmoid fixes arbitrary shape |
| Isotonic map | Can a flexible nondecreasing map fit rates? | Separate fit data and sufficient support | Large calibration sample | Small samples, unsupported extrapolation, ties |
| Prior correction | Does a changed class prior explain shift? | Pure label shift and source calibration | Known selection by class | Concept drift or changing class-conditionals |
| Constant base rate | What does a simple probabilistic baseline achieve? | Representative prior estimated separately | Baseline and fallback | Treating it as useful individual ranking |
| Random audit | Are observed labels selected by policy? | Known positive inclusion probabilities | Estimating full-population outcomes | Assuming audited cases represent unaudited without design |

Choose metrics jointly rather than optimizing a single number.
For probability decisions, compare proper scores with a fixed historical or estimated-prior baseline.
For investigator capacity, also evaluate ranking and realized utility at the available capacity.
For fairness, examine defined subgroup calibration and action errors alongside sample uncertainty.
Subgroup equality goals can conflict when base rates or distributions differ.
Specify which fairness objective matters and obtain domain and legal review.
A smaller proper score does not establish better treatment effects.
A stable AUC does not show stable fraud prevalence or calibration.
A better reliability curve can coexist with poorer resolution.
An isotonic map is nondecreasing rather than strictly increasing, so it can introduce ties.
This may change tie-adjusted AUC even though it never reverses strict ordering.
A logistic map with positive slope preserves order and AUC absent numerical ties.
A parametric map may underfit a nonlinear calibration pattern.
A flexible map may overfit rare positives and produce overconfident plateaus.
Compare maps on a later untouched cohort, not on their fitting data.
If evidence is too sparse, report uncertainty and consider a conservative predeclared fallback.
A fallback's behavior, coverage and harm need their own validation.
**Takeaway:** Select methods by the population, decision and evidence, not by a favored metric.

## 8. Failure Modes & Edge Cases

The following are constructed diagnostic scenarios, not observed incidents.
Each scenario distinguishes at least two mechanisms and a way to separate them.
No diagnostic result alone proves a cause; document the evidence before changing policy.

### F01. Score shift

- **Symptom:** Probability histogram moves upward.
- **Hypothesis A:** genuine event mix changed.
- **Hypothesis B:** a feature serialization change altered scores.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** replay frozen events with old and new feature artifacts.
- **Expected signal:** identical replay but changed live features.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** freeze risky rollout and repair the observed data contract.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** replay holdout and inspect live feature parity.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F02. Calibration intercept

- **Symptom:** Mean predicted risk is too low.
- **Hypothesis A:** prevalence increased under label shift.
- **Hypothesis B:** outcomes matured differently across periods.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare mature random-audit prevalence and class-conditional score distributions.
- **Expected signal:** higher prevalence with stable class-conditional score distributions.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** re-estimate the prior only if the label-shift assumptions hold.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check later mature cohort and uncertainty.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F03. Calibration slope

- **Symptom:** Extreme predictions are overconfident.
- **Hypothesis A:** calibration model overfit.
- **Hypothesis B:** distribution or label quality changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare slope on independent temporal cohorts and audit labels.
- **Expected signal:** overconfidence persists on independently adjudicated labels.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** refit a constrained map on representative data.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check slope and proper scores on untouched later data.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F04. Perfect ECE

- **Symptom:** ECE is zero in a broad bin.
- **Hypothesis A:** opposing local errors cancel.
- **Hypothesis B:** sample happened to align with forecasts.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** inspect finer fixed bins with counts and uncertainty.
- **Expected signal:** opposing positive and negative local gaps.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** report local gaps rather than hiding cancellation.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** repeat on later held-out cohort.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F05. Sparse tail

- **Symptom:** Top-risk bin appears perfectly reliable.
- **Hypothesis A:** few high-risk cases were observed.
- **Hypothesis B:** genuine strong calibration in that region.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** show bin counts and interval and examine multiple mature cohorts.
- **Expected signal:** wide interval around the observed rate.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** avoid tail-specific claims or pool with justified assumptions.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** measure future tail coverage and outcome counts.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F06. Zero events

- **Symptom:** A bin contains no observed fraud.
- **Hypothesis A:** true risk is low.
- **Hypothesis B:** small sample or immature labels obscure positives.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare exposure and label age against the maturity contract.
- **Expected signal:** few mature events or unresolved labels.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** report an upper interval rather than asserting zero risk.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** reassess after maturity and independent audit.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F07. AUC stable

- **Symptom:** AUC stays stable while fraud loss rises.
- **Hypothesis A:** base rate or loss severity rose.
- **Hypothesis B:** ranker drift masked by unchanged overall AUC.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare prevalence, severity and capacity-specific ranking on mature cohorts.
- **Expected signal:** higher prevalence with comparable pairwise ranking.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** reassess decision economics and capacity without claiming AUC caused loss.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** monitor realized loss and matched-cohort ranking.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F08. AUC falls after map

- **Symptom:** AUC falls after recalibration.
- **Hypothesis A:** isotonic plateaus introduced ties.
- **Hypothesis B:** the learned mapping reverses scores.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** count new ties and inversions on the same records.
- **Expected signal:** more ties without order inversions.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** use tie-aware analysis or a validated strictly monotone map.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** recompute pair counts on untouched data.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F09. Low Brier

- **Symptom:** A low Brier score is interpreted as excellent.
- **Hypothesis A:** rare prevalence makes a constant forecast strong.
- **Hypothesis B:** probabilities genuinely improve over baseline.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare against a pre-estimated prior and report log loss and reliability.
- **Expected signal:** model gains vanish relative to the constant baseline.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** retain an explicit baseline and measure resolution.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** repeat comparison on a later common cohort.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F10. Infinite log loss

- **Symptom:** Log loss becomes infinite.
- **Hypothesis A:** forecast assigned zero probability to a realized event.
- **Hypothesis B:** numerical underflow turned a small positive value into zero.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** inspect pre-rounding logits and stored probability precision.
- **Expected signal:** finite logits but stored probability rounded to zero.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** use stable log-domain evaluation and disclose any clipping.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** compare stable loss with high-precision reference.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F11. Brier disagreement

- **Symptom:** Brier improves while log loss worsens.
- **Hypothesis A:** a few confident errors dominate log loss.
- **Hypothesis B:** different cohorts were used for the two metrics.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** join record IDs and inspect per-record loss differences.
- **Expected signal:** same cohort with a small set of extreme log losses.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** investigate tail probabilities instead of choosing the flattering metric.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** recompute both scores on the matched frozen cohort.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F12. Threshold drift

- **Symptom:** The same threshold creates too many alerts.
- **Hypothesis A:** risk distribution shifted.
- **Hypothesis B:** calibration changed without much rank change.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare score histogram, audit prevalence and reliability by bin.
- **Expected signal:** observed bin rates differ at the same predicted probabilities.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** reassess mapping and capacity policy separately.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** monitor mature labels and alert volume after controlled rollout.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F13. Training leakage

- **Symptom:** Holdout reliability looks implausibly perfect.
- **Hypothesis A:** calibrator was fitted on the final test.
- **Hypothesis B:** duplicated entities crossed splits.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** trace artifact lineage and entity membership across partitions.
- **Expected signal:** test labels or entity duplicates appear in fitting data.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** rebuild chronological disjoint splits and refit.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** evaluate once on a new untouched cohort.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F14. Temporal leakage

- **Symptom:** Early forecasts appear unrealistically accurate.
- **Hypothesis A:** post-decision outcome information entered features.
- **Hypothesis B:** timestamps were parsed in the wrong time zone.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** audit feature availability timestamps and raw time-zone conversion.
- **Expected signal:** features first become available after decision time.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** remove unavailable features and correct timestamp contracts.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** replay historical decisions point in time.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F15. Case-control sampling

- **Symptom:** Forecasts are systematically high after training.
- **Hypothesis A:** positive cases were oversampled.
- **Hypothesis B:** label shift occurred after deployment.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** reconstruct training inclusion odds and compare current mature prevalence.
- **Expected signal:** sampling depended only on class and recorded odds explain the offset.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** apply justified prior-odds correction or recalibrate.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** test on a representative later sample.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F16. Unrecorded sampling

- **Symptom:** Prior correction gives implausible probabilities.
- **Hypothesis A:** selection also depended on features.
- **Hypothesis B:** sampling inclusion probabilities were misrecorded.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare selection rates within outcome and score strata.
- **Expected signal:** selection varies by score even after conditioning on outcome.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** replace prior-only correction with a defensible sampling model.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** validate on a randomized audit sample.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F17. Selective labels

- **Symptom:** Reviewed alerts show different calibration from all events.
- **Hypothesis A:** review policy selected high-risk records.
- **Hypothesis B:** actual population relationship changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare randomized audit cases with policy-selected cases.
- **Expected signal:** audit reveals a different rate within the same score band.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** use design-aware audit estimates where inclusion probabilities are positive.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** track audit coverage and uncertainty by score.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F18. No overlap

- **Symptom:** An unreviewed score region has no labels.
- **Hypothesis A:** policy never sampled it.
- **Hypothesis B:** labels are delayed but will eventually arrive.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** inspect inclusion probabilities and follow-up by score band.
- **Expected signal:** inclusion probability remains zero after maturity.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** declare the region unidentified and seek permitted random audit.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** verify positive coverage before estimating its rate.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F19. Audit weights

- **Symptom:** Weighted rates fluctuate wildly.
- **Hypothesis A:** a few inclusion probabilities are tiny.
- **Hypothesis B:** duplicate joins inflate weighted contributions.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** inspect weight distribution and event-key uniqueness.
- **Expected signal:** a few legitimate high weights dominate effective sample size.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** redesign sampling or report wide uncertainty; do not silently trim.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check design and sensitivity to declared truncation.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F20. Label delay

- **Symptom:** Recent cohorts seem to have lower fraud.
- **Hypothesis A:** fraud reports are immature.
- **Hypothesis B:** true fraud prevalence declined.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** plot cumulative resolution by cohort age and score band.
- **Expected signal:** recent labels are much less complete at the same calendar date.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** compare equal-age cohorts and report unresolved outcomes.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** track delayed labels through the agreed horizon.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F21. Maturity bias

- **Symptom:** Mature labels disagree with later adjudications.
- **Hypothesis A:** early closure is correlated with fraud difficulty.
- **Hypothesis B:** the adjudication policy changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare fixed-age and fully resolved outcomes within audit strata.
- **Expected signal:** resolution speed differs systematically by outcome.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** treat unresolved outcomes explicitly and perform sensitivity bounds.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** reconcile later adjudications against frozen predictions.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F22. Policy feedback

- **Symptom:** Post-deployment observed fraud drops.
- **Hypothesis A:** the action prevented events.
- **Hypothesis B:** the action prevented label observation.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare randomized permitted policy arms and observation rates.
- **Expected signal:** observation changes without corresponding independent loss evidence.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** separate effect on outcome from effect on measurement.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** evaluate incremental effects on consistently measured outcomes.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F23. Counterfactual target

- **Symptom:** A post-block label is used as no-block risk.
- **Hypothesis A:** blocking changed the event.
- **Hypothesis B:** the label definition is invariant to intervention.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** trace how the outcome can occur and be recorded under each action.
- **Expected signal:** blocked cases cannot produce the same observed transaction outcome.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** define potential outcomes and obtain defensible counterfactual evidence.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** validate the target against an approved experimental or causal design.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F24. Subgroup cancellation

- **Symptom:** Overall reliability looks good but one region complains.
- **Hypothesis A:** group errors cancel.
- **Hypothesis B:** small subgroup samples fluctuate.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare group-by-score rates with counts and intervals.
- **Expected signal:** persistent opposite-sign deviations in sufficiently supported groups.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** investigate data and calibration by group with governance.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check future group reliability and action harms.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F25. Group composition

- **Symptom:** Overall calibration shifts despite stable group curves.
- **Hypothesis A:** mix of groups changed.
- **Hypothesis B:** within-group risk relationships changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** standardize aggregate rates to a fixed group mix.
- **Expected signal:** standardized rates remain stable while mix changes.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** report both mix-adjusted and raw performance.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** monitor group proportions and within-group outcomes.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F26. Simpson reversal

- **Symptom:** A group-level curve differs from the overall curve.
- **Hypothesis A:** score composition differs by group.
- **Hypothesis B:** a group-specific predictor relationship changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare overlapping fine score ranges with adequate support.
- **Expected signal:** group gaps shrink after score stratification.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** avoid unsupported aggregate comparisons.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** replicate on a separate mature cohort.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F27. Covariate shift

- **Symptom:** Input distributions change while reliability appears stable.
- **Hypothesis A:** covariate shift preserves the outcome mechanism.
- **Hypothesis B:** opposing concept changes happen to cancel.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare conditional outcomes within supported feature and score strata.
- **Expected signal:** stable conditional rates on representative audit support.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** monitor support and reweight only under defensible assumptions.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** test forward calibration on the target population.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F28. Concept shift

- **Symptom:** Reweighting old data fails to restore reliability.
- **Hypothesis A:** the conditional outcome mechanism changed.
- **Hypothesis B:** importance weights were estimated incorrectly.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** evaluate target outcomes in overlapping covariate regions and inspect weights.
- **Expected signal:** persistent conditional outcome differences with stable weight estimates.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** refit or redesign using current representative labels.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** evaluate on later independent target data.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F29. Label shift violation

- **Symptom:** Prior correction worsens target loss.
- **Hypothesis A:** class-conditional score distributions changed.
- **Hypothesis B:** target prevalence estimate is noisy.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare class-conditional score distributions using mature audits.
- **Expected signal:** distributions differ beyond sampling uncertainty.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** stop relying solely on prior correction.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** compare a refitted map on a new target cohort.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F30. New merchant

- **Symptom:** A new merchant has no calibration history.
- **Hypothesis A:** score support is absent.
- **Hypothesis B:** global calibration transfers adequately.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** check merchant feature and score support against training and audit data.
- **Expected signal:** new score range falls outside historical support.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** route unsupported cases to governed fallback review.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** monitor labeled support before allowing calibrated automation.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F31. Score range

- **Symptom:** Production scores exceed calibration training range.
- **Hypothesis A:** model version changed.
- **Hypothesis B:** normal seasonal variation generated extremes.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare model artifacts and training versus live score ranges.
- **Expected signal:** artifact version changed or live range exceeds fit support.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** reject unsupported extrapolation or use a validated fallback.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check boundary behavior in replay and live shadow mode.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F32. Stale map

- **Symptom:** Calibration degrades after model retraining.
- **Hypothesis A:** mapping was trained for the prior model.
- **Hypothesis B:** only population prevalence changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** cross-evaluate old and new models with their own maps on a common cohort.
- **Expected signal:** new raw score distribution mismatches the map input contract.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** pair each model with its fitted calibration artifact.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** verify model-map version compatibility in deployment.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F33. Feature skew

- **Symptom:** Offline calibration is good but live reliability is poor.
- **Hypothesis A:** online feature values differ.
- **Hypothesis B:** labels are selected differently live.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** replay identical event snapshots through online and offline paths.
- **Expected signal:** feature or score differences occur before labels are used.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** repair feature parity and backfill only with valid timestamps.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** run shadow parity and later outcome checks.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F34. Rounding

- **Symptom:** A calibration plot has unexpected vertical bands.
- **Hypothesis A:** stored probabilities were rounded.
- **Hypothesis B:** model genuinely emits discrete probabilities.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare original logits with serialized outputs.
- **Expected signal:** distinct logits collapse to the same stored value.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** preserve sufficient numerical precision end to end.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check tie counts and loss before and after serialization.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F35. Out-of-range

- **Symptom:** Stored probabilities exceed one.
- **Hypothesis A:** a raw score was mislabeled as a probability.
- **Hypothesis B:** serialization corrupted a probability.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** inspect model output schema and raw model interface.
- **Expected signal:** raw output is an unbounded decision function.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** enforce typed output and map score to probability.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** verify contract on representative and boundary inputs.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F36. Wrong class

- **Symptom:** Forecasts appear inversely calibrated.
- **Hypothesis A:** positive class probability column was reversed.
- **Hypothesis B:** target encoding changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** trace class order in model artifact, training labels and serving schema.
- **Expected signal:** column order does not match fraud-positive target.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** correct class mapping with versioned schema.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check known examples and mature reliability after deployment.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F37. Duplicate joins

- **Symptom:** The estimated event rate changes after a label join.
- **Hypothesis A:** one event has multiple label revisions.
- **Hypothesis B:** the population actually changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare distinct event counts before and after the join.
- **Expected signal:** multiple joined labels per event.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** select versioned as-of labels with a declared adjudication rule.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** reconcile event counts and totals.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F38. Wrong horizon

- **Symptom:** Calibration drops after a reporting change.
- **Hypothesis A:** label horizon changed.
- **Hypothesis B:** true event probability changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare event definitions and maturity windows.
- **Expected signal:** rates diverge when horizons differ but align at common horizon.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** restore the contract or train a horizon-specific model.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** verify cohort definitions and aligned outcomes.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F39. Timezone boundary

- **Symptom:** Daily reliability spikes around midnight.
- **Hypothesis A:** events were assigned to the wrong day.
- **Hypothesis B:** real operational behavior changes at midnight.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare UTC and local timestamp conversions around transition periods.
- **Expected signal:** spikes align with conversion or daylight-saving boundaries.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** normalize instants and preserve decision-time zone.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** recompute fixed windows and inspect boundary events.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F40. Missing values

- **Symptom:** A subgroup risk forecast jumps.
- **Hypothesis A:** missing features use a new default.
- **Hypothesis B:** the genuine group event rate rose.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare missingness, online defaults and audit outcomes.
- **Expected signal:** forecast jump tracks default changes before outcome maturity.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** restore the feature contract or retrain with explicit missingness.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** verify feature distributions and later outcomes.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F41. Target leakage

- **Symptom:** A feature strongly predicts fraud.
- **Hypothesis A:** it is a post-adjudication marker.
- **Hypothesis B:** it is a legitimate pre-decision signal.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** trace data lineage and first availability per feature.
- **Expected signal:** feature arrives after fraud determination.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** remove the leaked field and rebuild training snapshots.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** evaluate on fresh time-split records.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F42. Threshold selection

- **Symptom:** Reported reliability is good only above the chosen cutoff.
- **Hypothesis A:** threshold was tuned on the evaluation set.
- **Hypothesis B:** the upper score region is genuinely better calibrated.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** inspect selection history and test the fixed cutoff on a later cohort.
- **Expected signal:** large deterioration on the untouched later cohort.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** separate calibration fit, threshold selection and final evaluation.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** report full-range and threshold-specific results.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F43. Multi-model pooling

- **Symptom:** A pooled plot hides a model-specific defect.
- **Hypothesis A:** different model versions are mixed.
- **Hypothesis B:** both versions share the same calibration function.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** stratify by model and map version at decision time.
- **Expected signal:** a single version drives the deviation.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** roll back or refit the affected version rather than the pooled curve.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check version-specific mature cohorts.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F44. Adaptive bins

- **Symptom:** A week-to-week ECE change is hard to explain.
- **Hypothesis A:** quantile bin boundaries moved.
- **Hypothesis B:** probability quality genuinely changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** recompute on common fixed bins and show score support.
- **Expected signal:** the apparent change disappears with common bins.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** report fixed-bin and quantile diagnostics together.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** retain versioned bin definitions.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F45. Confidence interval

- **Symptom:** A narrow bin interval suggests high certainty.
- **Hypothesis A:** events are correlated within accounts.
- **Hypothesis B:** the analyst treated independent outcomes correctly.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare record-level and account-clustered intervals.
- **Expected signal:** clustered uncertainty is materially wider.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** use a justified cluster-level resampling or model.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** assess coverage in a suitable simulation or design review.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F46. Multiple comparisons

- **Symptom:** One of many slices shows a severe gap.
- **Hypothesis A:** random noise from many slice inspections.
- **Hypothesis B:** a genuine subgroup-specific defect.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** predefine critical slices and compare independent follow-up data.
- **Expected signal:** the gap persists in held-out time and related evidence.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** investigate with multiplicity and practical importance in mind.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** monitor the prespecified slice with uncertainty.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F47. Calibration fit

- **Symptom:** In-sample isotonic reliability looks perfect.
- **Hypothesis A:** the step function memorized sparse labels.
- **Hypothesis B:** there is enough data for a stable monotone map.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare fit and disjoint validation losses and map variability.
- **Expected signal:** validation loss worsens and bootstrap maps vary sharply.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** simplify or pool with justified constraints.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** evaluate on a later untouched cohort.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F48. Flat logistic map

- **Symptom:** Every forecast becomes nearly identical.
- **Hypothesis A:** estimated slope is near zero.
- **Hypothesis B:** serving dropped or standardized scores incorrectly.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare fitted slope and served input distribution with training.
- **Expected signal:** served inputs are constant despite variable raw scores.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** repair preprocessing or reject the incompatible map.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check output spread and versioned inputs in shadow traffic.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F49. Negative slope

- **Symptom:** Calibration reverses risk ranking.
- **Hypothesis A:** score direction was inverted.
- **Hypothesis B:** training data shows a true inverse association.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare score orientation and label encoding on independent data.
- **Expected signal:** schema direction conflicts with calibration training.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** fix orientation and refit with correct labels.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check pairwise order and reliability on held-out events.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F50. Overfitted intercept

- **Symptom:** An intercept adjustment looks beneficial.
- **Hypothesis A:** the current window was used both to fit and score.
- **Hypothesis B:** prevalence really shifted.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** evaluate a fixed adjustment on a later independent cohort.
- **Expected signal:** gain vanishes out of sample.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** separate fitting and evaluation and report uncertainty.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** use rolling prospective evaluation.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F51. Population mismatch

- **Symptom:** A merchant-level forecast is evaluated per transaction.
- **Hypothesis A:** unit of analysis differs.
- **Hypothesis B:** merchant risk truly transfers to transactions.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** trace keys, weighting and aggregation before comparison.
- **Expected signal:** high-volume merchants receive unintended extra weight.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** define the intended estimand and aggregate accordingly.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** reconcile units, counts and weights.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F52. Repeated customers

- **Symptom:** Intervals are too narrow.
- **Hypothesis A:** transactions from the same customer are dependent.
- **Hypothesis B:** the assumed sampling unit was correct.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare customer-clustered and transaction-level estimates.
- **Expected signal:** clustered interval is wider.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** resample or model the actual dependence structure.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** document sampling unit and sensitivity.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F53. Unresolved disputes

- **Symptom:** Chargeback labels contradict later reversals.
- **Hypothesis A:** early labels are provisional.
- **Hypothesis B:** different event definitions are being mixed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare label revisions and contractual target definitions.
- **Expected signal:** a substantial fraction reverses after the first snapshot.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** use a fixed adjudication cutoff and record revisions.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** monitor revision rates by cohort age.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F54. Missing calibration version

- **Symptom:** Historical predictions cannot be reproduced.
- **Hypothesis A:** map artifact was not retained.
- **Hypothesis B:** a newer artifact was incorrectly joined.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare registry entries with decision logs.
- **Expected signal:** missing or mismatched model-map identifiers.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** retain immutable artifacts and log identifiers per decision.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** replay sampled decisions using retained versions.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F55. Action costs

- **Symptom:** A calibrated model yields harmful decisions.
- **Hypothesis A:** costs or treatment effects were wrong.
- **Hypothesis B:** risk probabilities were wrong.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** audit outcome calibration separately from intervention impact.
- **Expected signal:** risk is reliable but action effect is weak or negative.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** revise the decision policy using measured incremental outcomes.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** measure action-level harms and benefits in a governed evaluation.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F56. Capacity constraint

- **Symptom:** A cost threshold overloads investigators.
- **Hypothesis A:** capacity is binding.
- **Hypothesis B:** the score distribution was miscalculated.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare incoming eligible volume with available review slots.
- **Expected signal:** expected alerts exceed feasible slots.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** optimize feasible incremental value rather than apply independent threshold blindly.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** check actual queue load and outcome quality.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F57. Intervention heterogeneity

- **Symptom:** Highest-risk cases do not yield highest benefit.
- **Hypothesis A:** review effectiveness differs across cases.
- **Hypothesis B:** probability ranking is wrong.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** estimate risk and treatment effect separately where identification permits.
- **Expected signal:** some high-risk cases have little preventable loss.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** allocate by defensible incremental value with capacity constraints.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** evaluate treatment effects and customer harms.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F58. Weight mismatch

- **Symptom:** Offline reliability differs from live reporting.
- **Hypothesis A:** sampling weights were omitted.
- **Hypothesis B:** traffic genuinely changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** reconstruct inclusion design and compare weighted versus unweighted estimates.
- **Expected signal:** weighted audit aligns while unweighted selected data differs.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** use known design weights and report variance.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** verify coverage and inclusion probabilities each period.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F59. Unknown drift

- **Symptom:** A drift alarm fires without mature labels.
- **Hypothesis A:** feature mix shifted harmlessly.
- **Hypothesis B:** conditional risk changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** inspect support and missingness now and schedule mature audit comparison.
- **Expected signal:** features move but no outcome evidence is yet available.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** bound automatic changes and flag outcome claim as unresolved.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** compare matured representative labels when available.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F60. Data outage

- **Symptom:** All probabilities fall back to a constant.
- **Hypothesis A:** model serving failed.
- **Hypothesis B:** the actual risk distribution became homogeneous.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** check model service health, fallback flags and score entropy.
- **Expected signal:** fallback flag coincides with constant scores.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** restore service or apply approved safe fallback.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** test recovery and measure fallback coverage.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F61. Calibration delay

- **Symptom:** A recalibration job uses a recent cohort.
- **Hypothesis A:** recent outcomes are immature.
- **Hypothesis B:** the label pipeline is unusually fast and complete.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** inspect resolution by age and historical revision curves.
- **Expected signal:** material outcome revisions arrive after the fit cutoff.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** fit on mature representative cohorts and disclose lag.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** re-evaluate after the full observation window.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F62. Metric denominator

- **Symptom:** Reported average loss improves abruptly.
- **Hypothesis A:** unlabeled rows were silently dropped.
- **Hypothesis B:** the model genuinely improved.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** reconcile eligible, scored, mature and evaluated row counts.
- **Expected signal:** coverage dropped in a difficult score region.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** report coverage and compare matched cohorts.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** assert row-count reconciliation in monitoring.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F63. Adversarial adaptation

- **Symptom:** Calibration changes after policy launch.
- **Hypothesis A:** fraud behavior adapted to controls.
- **Hypothesis B:** observation or traffic mix changed.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** compare randomized audit outcomes, channel mix and time of policy change.
- **Expected signal:** changes remain within comparable audited strata.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** update threat model and retrain only with representative evidence.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** track independent audit rates and policy changes.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

### F64. Uncertainty communication

- **Symptom:** A single calibration estimate is treated as exact.
- **Hypothesis A:** sampling variability was omitted.
- **Hypothesis B:** the estimated population is a complete fixed census.
- **Diagnostic input:** Retain the original cohort definition, model and calibration versions.
- **Discriminating check:** record sampling design and event dependence.
- **Expected signal:** estimate comes from a small selected sample.
- **Interpretation:** The signal supports the corresponding explanation but does not alone prove causality.
- **Alternative path:** If the signal is absent, investigate the other hypothesis and measurement lineage.
- **Bounded mitigation:** show a design-appropriate interval and selection limitations.
- **Mitigation risk:** A change made from selected or immature labels can worsen full-population decisions.
- **Prevention:** Version the cohort, source artifacts, selection policy and label maturity contract.
- **Verification:** review interval method against the actual sampling unit.
- **Evidence:** ILLUSTRATIVE scenario; no production result is asserted.

**Takeaway:** Diagnose population, labels, artifacts and mechanics before changing a calibration map.

## 9. Complexity & Cost Analysis

Let n denote the number of evaluated records and k the number of bins.
Let m denote the number of calibration records and I the optimizer iterations.
A streaming Brier or log-loss pass takes O(n) time and O(1) aggregate state.
Per-slice or per-bin aggregates require state proportional to the number of groups or bins.
Fixed-bin reliability with direct bin assignment takes O(n+k) time and O(k) state.
This excludes label joins, storage scans and reporting infrastructure.
Exact quantile bins generally require sorting, O(n log n) time and O(n) stored values.
An approximate streaming quantile sketch trades exact boundaries for lower memory.
Its error and implementation-specific cost must be documented rather than assumed.
A scalar logistic fit typically costs O(I*m) arithmetic for fixed-dimensional iterative passes.
Solver behavior, line search and convergence affect actual runtime.
Sorting-based isotonic regression costs O(m log m) for ordering plus a linear pooling pass.
If scores are already sorted, the ordering cost is avoided.
The fitted isotonic map stores a number of breakpoints at most proportional to m.
Lookup can use binary search over breakpoints, O(log m) per score, with implementation-dependent constants.
AUC calculated by sorting scores takes O(n log n); explicit pair enumeration takes O(n_positive*n_negative).
Ties need correct group accounting in either implementation.
Memory estimates omit dataframe overhead, indexes, identifiers, object headers and replication.

**ILLUSTRATIVE sizing:** 1,000,000 rows each storing one 64-bit score and one 8-bit label use 9,000,000 raw bytes.
This is approximately 8.583 MiB using 1 MiB = 1,048,576 bytes.
A second 64-bit score array adds 8,000,000 raw bytes, or approximately 7.629 MiB.
The result excludes alignment, keys, version metadata and working space.
Storage bandwidth, joins, shuffles, and delayed labels may dominate arithmetic cost.
A distributed implementation must specify partitioning, merge semantics and consistent label snapshots.
A calibration job also has human costs: adjudication, audit design, governance and deployment review.
Reducing audit sampling cost can increase uncertainty or eliminate positive support.
Measure total cost against decision value, not only CPU time.
**Takeaway:** Complexity describes an algorithmic model; real cost requires measured workload and observation design.

## 10. Key Theorems / Results Reference

| Result | Use | Assumptions and boundary | Evidence |
|---|---|---|---|
| Conditional calibration | Forecast equals conditional event frequency | Defined target and population; does not imply individual truth | DERIVED C1 |
| Brier strict propriety | Honest probability minimizes expected squared loss | Binary outcome and expectation under true distribution | DERIVED C3; [Brier 1950](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2) |
| Log loss strict propriety | Honest probability minimizes expected negative log likelihood | Correct event and probability domain; boundary limits | DERIVED C4 |
| Bayes prior-odds update | Adjust for changed class prior | Stable class-conditional score distribution and support | DERIVED C8 |
| Increasing transform | Preserves strict score order | Strict monotonicity; finite-precision ties can differ | DERIVED from order definition |
| Isotonic nondecreasing fit | Flexible monotone calibration | Sufficient calibration support; ties may alter AUC | See [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) |
| Cost threshold | Compare two expected action costs | C9 assumptions, including no capacity coupling | DERIVED C9 |

**Source ledger:** The Brier article is a primary publication dated 1950; its scope is probabilistic forecast verification.
The linked library guide describes calibration methods; API-specific details are not a deployment claim.
The guide is a living reference, and its current version and content require review before publication.
The course source plan identifies topic scope but does not substantiate operational effects.
Source availability, link behavior and substantive review are OPEN, not independently verified here.
Recheck source content when the method, event definition, library version or decision context changes.
All numerical cohorts in this sheet are constructed and do not establish field performance.
**Takeaway:** A theorem's assumptions and the evidence provenance travel with the result.

## 11. Interview / Exam Questions

### Q01. What does a calibrated forecast of 0.2 mean?

**Answer:** Among the relevant population of cases forecast 0.2, the event frequency is 0.2 in the population calibration statement. A single case is still uncertain; finite-sample frequency need not be exactly 0.2.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q02. Can a model have perfect ranking and poor calibration?

**Answer:** Yes. Assign 0.9 to every positive and 0.8 to every negative: all pairs rank correctly, but neither forecast matches the constructed group event frequency.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q03. Can a calibrated model be useless for ranking?

**Answer:** Yes. A constant forecast equal to the true prevalence is calibrated but ties all pairs, giving AUC 0.5 under half-credit ties.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q04. Compute Brier loss for forecasts 0.2 and 0.8 with outcomes 0 and 1.

**Answer:** Both squared errors are 0.04, so their average is 0.04. The result is dimensionless and is specific to these two observations.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q05. Compute log loss for the same two records.

**Answer:** Both realized outcomes have probability 0.8; the mean is -ln(0.8), approximately 0.223144 nats per event.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q06. Why can a low ECE be misleading?

**Answer:** Within-bin over- and underprediction can cancel. The result also depends on binning and sampling, so inspect bin counts, uncertainty and local curves.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q07. What happens to AUC under a strictly increasing recalibration?

**Answer:** Strict pairwise order and existing ties are preserved, hence AUC is unchanged in exact arithmetic. Isotonic plateaus can introduce ties and alter tie-adjusted AUC.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q08. How do you split data for calibration?

**Answer:** Use earlier data for model fitting, a later disjoint cohort for fitting the map, and a still later untouched cohort for evaluation. Respect entity leakage, label maturity and policy selection.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q09. How do you correct class-prior shift from 0.5 to 0.1 when the source probability is 0.5?

**Answer:** Under pure label shift, multiply source odds 1 by target/source prior-odds ratio 1/9. Convert odds 1/9 to probability 0.1; conditional score stability and source calibration are essential.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q10. Why are reviewed fraud labels insufficient for a population reliability plot?

**Answer:** Review selection depends on score and action. Use a permitted random audit with known positive inclusion probabilities, account for maturity, and report uncertainty and unsupported regions.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q11. A forecast bin has 15 expected positives and 20 observed among 100 records. What is the gap?

**Answer:** Observed frequency is 0.20, mean forecast is 0.15, and observed minus predicted is +0.05, or five percentage points. This sample gap is uncertain.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q12. With false-positive cost 1 and false-negative cost 9, what threshold follows?

**Answer:** Under the two-action C9 assumptions, the threshold is 1/(1+9)=0.10. Act above it and explicitly resolve ties; this ignores capacity and heterogeneous action effects.

**Reasoning check:** State the target population and the assumptions before applying this answer.

### Q13. If overall calibration is good, are all demographic slices calibrated?

**Answer:** No. Two equal groups both forecast 0.5 can have event rates 0.2 and 0.8; aggregate calibration hides opposite subgroup errors. Inspect supported slices with uncertainty and governance.

**Reasoning check:** State the target population and the assumptions before applying this answer.

**Takeaway:** Strong answers define the estimand, calculate correctly, and name the failure boundary.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Probability calibration, proper scoring and fraud decision boundaries |
| Core formula | Brier score is the mean squared probability error `(p − y)²` for the same binary target and evaluation cohort |
| Key assumption | Forecasts and mature outcomes measure the declared target in the population where the decision will be applied |
| Complexity | One scoring pass is O(N); calibration fitting and storage depend on the selected model and validation process |
| Use when | Probabilities support cost-sensitive decisions or must be compared across cohorts |
| Avoid when | Missing outcomes or policy-selected labels are treated as representative without a justified sampling design |
| Go deeper | [Fraud evaluation and capacity](01_Fraud_Evaluation_Calibration_and_Capacity.md); source and publication review remain open |
| Diagnostic | Report reliability by supported slice, score distribution, proper scores and label coverage alongside ranking |
| Evidence | Worked examples are illustrative and do not establish production performance |
