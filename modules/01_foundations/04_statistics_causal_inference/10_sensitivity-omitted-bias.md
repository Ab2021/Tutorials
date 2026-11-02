> Module 04 — Statistics and Causal Inference
> Topic 10 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.06, 04.08, 04.09

# Sensitivity and Omitted-Variable Bias

---

## 1. Why this topic exists

A supplier quality team reports that its joint-improvement programme cut the receipt rejection
rate by 1.2 percentage points. The regression has 9,120 supplier-months, standard errors
clustered on 380 suppliers, and a p-value below 0.01. The category director asks one question:
"Could something else explain it?"

The software has already answered a narrower question: if the model were correct and the errors
behaved, a coefficient this large would be unlikely under a null of zero. It has not answered
the director's. The director is asking about the model, not about the noise.

That gap is this topic. Every estimate in supply chain analytics is a difference between an
observed outcome and an assumed counterfactual, and the assumption lives outside the data. The
question is never whether the assumption is true. It is how much of the assumption the
conclusion needs. That question has an arithmetic, and this topic gives it.

Three questions organise everything that follows.

1. **How large is the bias?** The omitted-variable bias (OVB) formula turns "there might be a
   confounder" into a product of two quantities, each of which has a sign, a unit, and a
   plausible range.
2. **How strong would a confounder have to be to overturn the conclusion?** This is the
   inversion that makes the analysis useful to a decision-maker. It produces a threshold, not
   a verdict.
3. **What empirical probe would reveal the failure?** Placebo treatments, negative-control
   outcomes, and specification curves test the design rather than the coefficient.

Supply chain is unusually exposed to the first question and unusually poor at the second.
Three structural reasons:

- **Treatment is chosen by managers, not by chance.** A supplier is audited because it is
  large, a lane moves to rail because it is long, a distribution centre automates because its
  volume justifies the capital. Each of those selection rules is a confounder with a name, and
  the name is usually written in a procurement policy nobody gave the analyst.
- **The network changes while you measure it.** Suppliers are acquired, lanes are
  renegotiated, products are substituted, sourcing regions shift. The unit of analysis is not
  stable, and the instability is correlated with the treatment, because both respond to the
  same commercial pressures.
- **The outcome is measured by the organisation that chose the treatment.** Rejection rates,
  on-time rates, and cost per unit are produced by the operation being evaluated. Measurement
  error is therefore not classical, and it does not average out.

The methods in this topic come mostly from economics, epidemiology, and social science. Their
supply chain application is thinner in the public record, which is a reason for care rather
than a reason to ignore them. The arithmetic does not care about the industry. A covariance is
a covariance whether the units are students, patients, or pallets.

**What this topic is not.** It is not a way to rescue a bad design. Sensitivity analysis cannot
manufacture identification. A study with no credible comparison, no time structure, and no
exclusion argument does not become credible by adding an E-value. What sensitivity analysis
does is honest bookkeeping: it states what the estimate depends on, how much, and what
evidence would settle it.

**Why it closes the module.** Each of the previous nine topics named an assumption and taught a
design that leans on it. This topic asks what each of those assumptions is worth when it is
wrong, and it closes by assembling them into a single argument.

| Topic | What it establishes | What topic 10 asks of it |
|---|---|---|
| 01 Estimands | The quantity being estimated, and the decision it serves | Is this the quantity the decision needs? |
| 02 Probability and bootstrap | Sampling uncertainty that survives dependence | Does the interval account for model uncertainty? |
| 03 Hypothesis tests | Effect size separated from noise | Does significance survive a bias of plausible size? |
| 04 Regression and time | Seasonality, holidays, distributed lags | Could a seasonal confounder move the sign? |
| 05 Robust errors | Correct uncertainty under dependence | Does clustering change the threshold, not only the standard error? |
| 06 DAGs | Back-door paths, colliders, bad controls | Which open path is unmeasured, and how strong is it? |
| 07 Experiments and quasi-experiments | What randomisation buys, and what replaces it | What if assignment was imperfect in a specific way? |
| 08 Difference-in-differences | Parallel trends and event time | How much divergence would overturn the estimate? |
| 09 Elasticity and endogeneity | Exclusion restriction and instruments | How strong must a direct path be to break it? |
| 10 Sensitivity | Fragility as a quantity | This is the question, stated once, for every design above |

The module's argument, in one sentence: **a causal claim in supply chain is a chain of explicit
assumptions, each testable in the weak sense, and each worth naming.** This topic supplies the
last link, which is the discipline of asking what would break the chain.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** the OVB identity for a short and a long regression, and **decompose** the bias into
   the confounder's link to the outcome and its link to the treatment, with units on each.
2. **Compute** the bias for one and for two omitted variables from reported coefficients, and
   **state** the sign and direction of the distortion without estimating anything new.
3. **Invert** a result into a robustness question: compute an impact threshold, a robustness
   value, or an Oster delta, and **say** in domain terms what the threshold means.
4. **Compute** an E-value from a risk ratio, **state** the scale requirements it imposes, and
   **explain** why a small effect has a small E-value even when its p-value is tiny.
5. **Distinguish** coefficient stability from identifying evidence, and **explain** why a stable
   coefficient with a small R-squared movement is not reassurance.
6. **Design** a placebo treatment and a negative-control outcome, each with an explicit
   shared-bias argument and a stated power condition.
7. **Pre-register** a robustness ladder and **read** a specification curve without treating the
   median specification as the answer.
8. **State**, for any causal claim in a supply chain setting, the single assumption whose
   failure would reverse the decision, and the cheapest evidence that would test it.

Each outcome maps to an exercise in section 12 and an answer in section 13.

## 3. The core mental model

Hold one picture: a triangle with the omitted variable at the apex and the treatment and
outcome at the base. The two sides of the triangle are the only two things that determine the
bias. Draw this by hand once and it will stay with you.

```text
                          +---------------------------+
                          |  W    unobserved          |
                          |  supplier capability,     |
                          |  plant quality culture    |
                          +---------------------------+
                              /                   \
                    a = Cov(W,X) / Var(X)     b = effect of W
                    "how strongly W            on Y, holding X
                     moves with treatment"     "how strongly W
                                                moves the outcome"
                            /                       \
                           v                         v
                 +-------------------+     +-------------------+
                 |  X   treatment    |---->|  Y   outcome      |
                 |  programme        |  ?  |  rejects per 100  |
                 |  coverage, 0 to 1 |     |  lots, pp         |
                 +-------------------+     +-------------------+

        bias in the short estimate   =   a  x  b
        beta_short  =  beta_long  +  a  x  b

        units:  a  in sd(W) per unit of X
                b  in pp per sd(W)
                a x b  in pp, the same units as the estimate
```

The picture has five properties worth memorising.

**Property 1 — The bias is a product, so either factor can null it.** If `W` does not affect the
outcome (`b = 0`), or is unrelated to treatment (`a = 0`), the omission is harmless. Most
arguments about "an omitted variable" are really arguments about one of the two factors, and
naming which one is the first useful move. This is also why the useful question about a
confounder is never "is it important?" but "is it important for both sides?".

**Property 2 — The formula is an identity, not a model.** It follows from the algebra of least
squares, specifically from partitioned regression, and requires no assumption about the
distribution of anything. That is why it is trustworthy as arithmetic and untrustworthy as a
conclusion: it tells you exactly how much bias a given confounder would produce, not whether
such a confounder exists.

**Property 3 — Direction is free; magnitude is not.** You can sign the bias from the two signs
without any data. You cannot size it without knowing both factors. An analyst who says "the
bias could go either way" has usually not looked at the signs; an analyst who says "the bias is
small" has usually not looked at the magnitudes.

**Property 4 — Inverting the question produces a threshold.** Instead of asking "is the effect
zero?", ask "how strong must `W` be for the effect to be zero?" The answer is a number in the
same units as the confounder's two links. A decision-maker can compare that number with what
they know about the business. This inversion is the single most useful move in the topic.

**Property 5 — Different tools answer the inversion in different currencies.** OVB works in
coefficient units. Oster works in a ratio of selection strengths. E-values work in risk ratios.
Robustness values work in partial R-squared. The currencies are not interchangeable, and
reporting one while the reader is thinking in another is a common source of false comfort.

### 3.1 The inversion, stated

The estimate is a point on a line from its own value toward zero, and the distance it must travel
before the decision changes is the fragility. A short distance means a modest confounder
overturns the result; a long distance means the result survives plausible confounding. Neither
tells you whether the design is valid; both tell you what the claim costs.

### 3.2 Three currencies, one question

| Currency | Question it answers | Strength | Weakness |
|---|---|---|---|
| Coefficient units (pp, EUR, units) | How much would the estimate move? | Directly interpretable to the business | Requires knowing the confounder's scale |
| Partial R-squared | How much of the residual variation must the confounder explain? | Scale-free; comparable across studies | Not intuitive to non-analysts |
| Risk ratio / E-value | How strong must the confounder's associations be? | Comparable to observed risk factors | Requires a ratio scale and a well-defined baseline risk |
| Selection ratio (Oster delta) | How much more selected must unobservables be than observables? | Anchored to the controls you already have | Depends on `R_max`, a convention |

The practical rule: compute in at least two currencies, and report the one the decision-maker
already uses. A plant manager thinks in percentage points of rejects; a medical reviewer thinks in
risk ratios; a model risk committee thinks in R-squared. A report that speaks only one language
will be misread by the other two.

### 3.3 Coefficient stability is not evidence

A coefficient that barely moves when controls are added is often presented as reassurance. It is
not, on its own. Oster's central point is that bias is proportional to the coefficient movement
only when the movement is scaled by the change in R-squared. A coefficient can be stable because
the new controls are irrelevant, because they are collinear with the treatment, or because their
inclusion moves the coefficient and the R-squared in compensating ways.

| What you observe | Naive reading | Correct reading |
|---|---|---|
| Coefficient stable, R-squared barely moves | "The result is robust" | The controls explained nothing; the test had no power |
| Coefficient stable, R-squared rises a lot | "The result is robust" | Possibly; but the controls may be collinear with treatment |
| Coefficient moves, R-squared rises a lot | "The result is fragile" | Not necessarily; the movement may be the removal of bias |
| Coefficient moves, R-squared barely moves | "Something is wrong" | The controls are proxies for the confounder and absorb its bias |

### 3.4 Robustness is not validity

A design can pass every sensitivity check and still be wrong, because the checks share the
design's blind spot. A placebo test with low power passes trivially. A negative control chosen
because its data were convenient may not share the confounding structure. A specification curve
built from specifications that all omit the same variable will show a tight, confident, wrong
distribution.

The correct epistemic posture is asymmetric. A failed check is informative: it refutes. A passed
check is weak evidence: it fails to refute. This is the same asymmetry that Topic 06 established
for DAGs and Topic 08 for pre-trend tests. Sensitivity analysis inherits it.

> **Note:** The most dangerous output of this topic is a table of robustness checks that all
> "pass". A table of checks with no stated failure threshold, no power calculation, and no
> pre-registration is decoration. The value is in the threshold you named before you saw the
> result, not in the number of columns you filled afterwards.

## 4. Definitions and vocabulary

The fourth column states the unit or coding convention. A sensitivity claim without a currency
is not a claim; it is a mood.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Omitted-variable bias (OVB) | The difference between a short-regression coefficient and the long-regression coefficient that includes the omitted confounder | Called a "missing data" problem rather than a specification problem | Outcome units per treatment unit |
| Confounder | A pre-treatment common cause of treatment and outcome | Used for any correlate of the outcome | Node in a directed acyclic graph (DAG) |
| Unmeasured confounder | A confounder absent from the data | Assumed absent because it is unmeasured | Latent variable; no direct unit |
| Sensitivity analysis | Asking how much an assumption would have to fail to change a conclusion | Reported as a check that the result "passed" | A threshold, not a p-value |
| Bias formula | `bias = gamma x delta`, where `gamma` is the omitted variable's effect on the outcome and `delta` its regression on the treatment | Applied without stating which variable is omitted | Coefficient units |
| Short regression | The model that omits the confounder | Treated as the naive or "unadjusted" model | Coefficient per treatment unit |
| Long regression | The model that includes the confounder | Treated as the causal model by default | Coefficient per treatment unit |
| Auxiliary regression | The regression of the omitted variable on the treatment and the other omitted variables, used to obtain `delta` | Replaced by a simple correlation, or run univariately with correlated omitted variables | sd of `W` per treatment unit |
| Partial R-squared | The share of residual variance in one variable explained by another, after partialling out controls | Confused with the raw R-squared | Proportion, 0 to 1 |
| Robustness value (RV) | The partial R-squared a confounder would need with both treatment and outcome to move the estimate by a stated amount | Reported without stating the target (`q = 1` for zero, or the significance threshold) | Proportion, 0 to 1 |
| Impact threshold for a confounding variable (ITCV) | The product of the confounder's partial correlations with treatment and outcome needed to make a coefficient non-significant | Reported without the benchmark impact it is compared against | Correlation product, -1 to 1 |
| E-value | The minimum strength of association, on the risk ratio scale, that an unmeasured confounder would need with both treatment and outcome to explain away the effect | Applied to an odds ratio or a mean difference without conversion | Risk ratio, dimensionless, at least 1 |
| Bounding factor | The multiplicative factor `(RR_EU x RR_UD) / (RR_EU + RR_UD - 1)` by which confounding can move a risk ratio | Treated as an estimate of the bias rather than a bound | Ratio, at least 1 |
| Oster's delta (`delta_O`) | The ratio of selection on unobservables to selection on observables implied by a bias-adjusted estimate | Confused with the OVB auxiliary coefficient | Ratio, dimensionless |
| `R_max` | The hypothetical R-squared from a regression including all unobservables | Set to 1 by default, which makes the bound uninformative | Proportion, 0 to 1 |
| Proportional selection | The assumption that unobservables are related to treatment and outcome in the same proportion as observables | Left unstated | Ratio, usually benchmarked at 1 |
| Bias-adjusted estimate | An estimate corrected under a stated selection assumption | Reported as if it were identified | Outcome units per treatment unit |
| Benchmark covariate | An observed control used to calibrate how strong an unobserved confounder might be | Chosen because it is convenient, not because it is comparable | Partial R-squared or correlation |
| Placebo outcome | An outcome the treatment cannot plausibly affect, used to detect residual bias | Chosen because data were available | Outcome units |
| Placebo treatment | A fake treatment date or assignment used to detect pre-existing differences | Placed where the pre-period is shortest | Date or indicator |
| Negative control outcome | An outcome with no causal link to treatment but the same confounding structure | Assumed to validate the design rather than probe it | Outcome units |
| U-comparable | Sharing, as closely as possible, the same unobserved common causes with the target variable | Asserted rather than argued | Qualitative statement |
| Shared bias structure | The condition that a negative control is confounded in the same way as the target association | Assumed rather than argued | Qualitative, must be stated |
| Specification curve | A display of estimates across a pre-defined set of defensible specifications, with a dashboard of choices | Used to select the specification with the largest estimate | Distribution of estimates |
| Pre-registration | A record of the analysis plan made before results are seen | Applied only to experiments | Registry entry or timestamped document |
| Robustness ladder | An ordered list of checks with thresholds, fixed in advance | Assembled after the results, from the checks that passed | Ordered table |
| Breakdown value | The value of a sensitivity parameter at which the conclusion changes | Reported without the conclusion it breaks | Parameter units |
| Relative-magnitudes restriction | A bound on post-treatment violations of parallel trends as a multiple of observed pre-treatment violations | Applied without stating the multiple | Multiple `M-bar` |
| Honest confidence interval | An interval valid uniformly over a class of assumption violations | Read as a point estimate with a wider standard error | Interval in outcome units |
| Selection on observables | The assumption that treatment is independent of potential outcomes given the controls | Asserted from the covariate list | Conditional independence statement |
| Selection on unobservables | The failure of selection on observables, and the object sensitivity analysis studies | Treated as a binary property of the study | Magnitude, not a yes or no |
| Stable unit treatment value assumption (SUTVA) | No interference between units, and one version of treatment | Assumed silently in every regression | A statement about the exposure mapping |

Three warrant expansion, because the damage they cause is disproportionate.

**E-value versus p-value.** A p-value asks how surprising the estimate is if the effect were zero and
the model correct. An E-value asks how strong a confounder would have to be to make it zero. A tiny
effect measured precisely has a tiny p-value and a tiny E-value, so reporting only the first says
nothing about fragility.

**Oster's delta versus the OVB delta.** The OVB formula uses `delta` for the auxiliary coefficient;
Oster uses it for a ratio of selection strengths. This topic writes `delta_j` and `delta_O` to keep
them apart.

**Partial R-squared versus raw R-squared.** A raw R-squared of 0.36 sounds substantial; a partial
R-squared of 0.02 sounds trivial. The robustness value lives on the partial scale, so comparing it
with a raw R-squared overstates a confounder's strength by roughly an order of magnitude.

## 5. The mechanics, step by step

The sequence below is ordered so that each step constrains the next. Most bad sensitivity work
is not a computational error. It is a step skipped, usually Step 1 or Step 4, and discovered when
someone asks what the threshold means for the decision.

### Phase A — Fix the claim and the number

**Step 1. State the estimand and the decision rule.**

An estimand has six slots: outcome, treatment contrast, population, time window, unit, and summary
measure (Topic 01). Add a seventh for sensitivity work: **the decision rule**, the value of the
estimate that would change what someone does. "Expand if the effect exceeds 0.5 percentage points
of rejects" is a decision rule. "Expand if the effect is significant" is not, because
significance is a property of the sample size as much as of the effect.

**Step 2. State the estimate, its units, and its interval.**

Quote the coefficient with its units, standard error, and clustering level. A sensitivity analysis
on a coefficient whose units are ambiguous is uninterpretable, because the bias cannot be compared
with the estimate. If the coefficient is in standard deviations, say so; if in percentage points,
say of what base.

**Step 3. Write the identifying assumption in one sentence.**

One sentence, in the language of the business. For a regression with controls: "After adjusting for
supplier capability and order complexity, programme coverage is unrelated to any other determinant
of the rejection rate." For difference-in-differences: "Absent the programme, treated and
comparison suppliers would have followed parallel rejection trends." If you cannot write the
sentence, you do not yet have a design.

### Phase B — Enumerate what the design cannot rule out

**Step 4. List confounders by mechanism, not by data availability.**

Walk the graph from Topic 06 and write down every variable with an arrow into both treatment and
outcome. Remove the ones you measured; what remains is the candidate set. Do this before looking at
the regression output, because the output will bias which candidates seem plausible. Name the
**selection rule**: why did this supplier get the programme? The answer is usually a managerial
criterion, and the criterion is the confounder. "Because it was a strategic supplier" is a name for
a latent variable.

**Step 5. Sign each candidate.**

For each candidate `j`, write the sign of `gamma_j` (its effect on the outcome, holding treatment)
and of `delta_j` (its regression on the treatment). The bias sign is their product. Ten minutes of
this eliminates half the candidates, because many plausible confounders bias toward zero.

**Step 6. Keep the two or three that could plausibly be large.**

The criterion is not significance in the long regression. It is whether the product could exceed a
meaningful fraction of the estimate. A large `gamma` with no relation to treatment is irrelevant; a
large `delta` with a tiny `gamma` is irrelevant. Only the product matters.

### Phase C — Quantify the bias

**Step 7. Estimate the short and the long regression.**

The short model holds the treatment and the non-negotiable controls; the long model adds the
candidates. Report both on the same sample, with the same clustering and outcome definition.

**Step 8. Decompose the bias.**

For one omitted variable, `bias = beta_short - beta_long = gamma x delta`. For several, the
decomposition is exactly additive:

```text
   beta_short = beta_long + gamma_1*delta_1 + gamma_2*delta_2 + ... + gamma_k*delta_k
```

Here `delta_j` comes from the auxiliary regression of `W_j` on `X` **and the other omitted
variables**, and `gamma_j` is the coefficient on `W_j` in the long regression. Additivity is exact
because covariance is linear.

> **Note:** Each `delta_j` from a separate *univariate* regression makes the decomposition exact
> only when the omitted variables are uncorrelated. With correlated confounders the shortcut
> double-counts shared variation, and the terms will not sum to the observed movement.

**Step 9. Convert to partial R-squared.**

The confounder's partial R-squared with the outcome is `R2_Y.W|X`; with the treatment it is
`R2_X.W`. The standardized bias bound is proportional to the product of their square roots:

```text
   |standardized bias|  <=  sqrt( R2_Y.W|X * R2_X.W / (1 - R2_X.W) )
```

This is the Cinelli–Hazlett bound, useful because partial R-squared is comparable across studies
with different units and because it leads directly to the robustness value.

### Phase D — Invert the question

**Step 10. Oster's delta.**

```text
   beta_star  =  beta_long  -  delta_O * (beta_short - beta_long) * C
   where  C   =  (R_max - R_long) / (R_long - R_short)

   delta_O*   =  beta_long / ( (beta_short - beta_long) * C )
```

`delta_O = 1` means unobservables are assumed to be selected in the same proportion as
observables, and `delta_O*` is the ratio that drives the estimate to zero. Below 1, a confounder
weaker than the observed controls overturns the result; above 1, the result survives the
equal-selection benchmark. Oster's `R_max = 1.3 x R_long` comes from comparing randomised and
non-randomised estimates in the literature she studied. It is a convention, not a law, and the
common implementation is `R_max = min(1.3 x R_long, 1)`.

**Step 11. The E-value.**

For a risk ratio above 1:

```text
   E  =  RR  +  sqrt( RR * (RR - 1) )
```

For a risk ratio below 1, invert it first and apply the same formula. The E-value is the minimum
strength of association, on the risk ratio scale, that an unmeasured confounder would need with
**both** treatment and outcome to explain away the observed association. Report it for the point
estimate and for the confidence limit closest to the null.

The underlying bound is worth knowing, because it explains where the formula comes from. If
`RR_EU` is the confounder's risk ratio with the exposure and `RR_UD` its risk ratio with the
outcome, the bias factor is:

```text
   BF  =  (RR_EU * RR_UD) / (RR_EU + RR_UD - 1)
   RR_true  >=  RR_observed / BF
```

Setting `BF = RR_observed` and solving for equal strengths gives the E-value formula above. The
E-value is therefore not an arbitrary heuristic; it is the exact solution to a worst-case
bounding problem under a specified class of confounders.

**Step 12. The robustness value.**

The Cinelli–Hazlett robustness value is the partial R-squared a confounder needs with both
treatment and outcome to move the estimate by a factor `q`:

```text
   RV_q  =  0.5 * ( sqrt( f_q^4 + 4*f_q^2 ) - f_q^2 )
   f_q   =  q * |t| / sqrt(df)
```

For `q = 1` the target is zero. For the significance target, replace `q|t|` with
`q|t| - t_critical`. The formula looks forbidding and behaves simply: for small `f`, the
robustness value is approximately `f`.

**Step 13. Translate the threshold into domain language.**

"The result would be overturned by a confounder with a correlation of 0.09 with both coverage and
rejects" is a sentence a quality director can evaluate. "The robustness value is 0.008" is not.
The translation is the deliverable. Do it in every report.

### Phase E — Probe the design empirically

**Step 14. Placebo treatment.**

Re-run the design with a fake treatment date in the pre-period, on units that were eventually
treated. A large "effect" before the programme began is evidence of a pre-existing trend. State
the power of the placebo: a null from a test that could not have detected anything is not
evidence.

**Step 15. Negative-control outcome.**

Choose an outcome the treatment cannot plausibly affect but which shares the confounding
structure. Estimate the same model. A null is weakly supportive. A non-null is a direct warning
that the design is picking up something other than treatment. The critical assumption is
**U-comparability** — the control outcome must share the unobserved causes of the target
outcome. It must be argued, not assumed. An outcome chosen because its data were convenient
usually fails this test.

**Step 16. Specification curve.**

Enumerate the defensible analytic choices before seeing results: outcome definition, treatment
coding, control set, fixed effects, time window, trimming rule, clustering level. Estimate every
combination. Display the sorted estimates with a dashboard of the choices. Conduct joint
inference on the whole set, not on the favourite member of it. The median is not the answer, and
the maximum is not the answer. The distribution is the answer, together with the share of
specifications that clear the decision rule.

**Step 17. Honest bounds for design-based estimates.**

For difference-in-differences, replace the parallel-trends assumption with a bound on how far
trends may diverge. The relative-magnitudes restriction says that the largest post-treatment
violation is at most `M-bar` times the largest pre-treatment violation, with `M-bar = 1` as the
natural benchmark. Report the breakdown value: the `M-bar` at which the conclusion changes. This
is the sensitivity analysis that matches the design, and it replaces the pre-trend test, which
Topic 08 showed is underpowered.

### Phase F — Report

**Step 18. Pre-register the ladder.**

Write the checks and their thresholds before running them. A ladder has five to eight rungs, each
with a question, a test, a pass condition, and a consequence of failure. Pre-registration does
not require a public registry for internal work; a timestamped document circulated to the
decision-maker is enough. What matters is that the choice of checks is not made after seeing
which ones pass.

**Step 19. Report the ladder, not the winner.**

A report that says "the result survives all robustness checks" without showing the checks is
unfalsifiable. Show the table, including the checks that failed. State the one check whose
failure would change the recommendation. That sentence is the most valuable line in the memo.

### 5.1 The sign rules, in one table

| `gamma` (W to Y) | `delta` (W to X) | Bias on the short estimate | Reading |
|---|---|---|---|
| Positive | Positive | Positive | Short estimate too large |
| Positive | Negative | Negative | Short estimate too small |
| Negative | Positive | Negative | Short estimate too small |
| Negative | Negative | Positive | Short estimate too large |
| Zero | Anything | Zero | W is not a confounder for this outcome |
| Anything | Zero | Zero | W does not predict treatment; not a confounder |

The last two rows are the ones to check first. A variable that affects the outcome but is
unrelated to treatment is a precision variable, not a confounder. Including it changes the
standard error, not the bias.

### 5.2 The robustness ladder

| Rung | Question | Test | Pass condition | If it fails |
|---:|---|---|---|---|
| 0 | What is the unadjusted association? | Short regression | Reported with units | You have no baseline |
| 1 | What happens with observed controls? | Long regression | Coefficient and R-squared both reported | You cannot assess selection |
| 2 | How much bias would an omitted variable produce? | OVB decomposition | Bias smaller than the decision threshold | Downgrade the claim |
| 3 | How strong must selection be to nullify it? | Oster `delta_O*`, robustness value | `delta_O* > 1`; RV above benchmark | Report as fragile |
| 4 | How strong must a confounder be on the risk ratio scale? | E-value | E-value above plausible confounder RR | Report the threshold |
| 5 | Is there evidence of a pre-existing difference? | Placebo treatment | Placebo near zero, with power stated | Redesign or downgrade |
| 6 | Does an unaffected outcome move? | Negative control | Null, with a shared-bias argument | Investigate the mechanism |
| 7 | How much do analytic choices matter? | Specification curve | Distribution centred away from zero | Report the spread |
| 8 | How much assumption violation does the design tolerate? | Honest bounds, partial identification | Decision unchanged across the bound | Report the identified set |

The ladder is ordered by cost, not by importance. Rungs 0 to 4 are arithmetic and can be done in
an afternoon. Rungs 5 to 8 require data, time, or a different design. A team that does only the
cheap rungs and calls the result robust has done the arithmetic and skipped the evidence.

> **Note:** Rung 8 applies mainly to design-based estimates such as difference-in-differences.
> For a cross-sectional regression with controls, the analogous rung is a partial identification
> bound under a stated confounder strength, not a parallel-trends restriction. Do not transplant
> the HonestDiD machinery to a design it was not built for.

## 6. Worked example

All figures below are illustrative and internally consistent. They are constructed to make the
arithmetic visible, not to represent a real programme.

### 6.1 Setting and decision rule

Halden Components is a European industrial-parts manufacturer. It buys from 380 suppliers and
receives about 9,120 supplier-months of receipts over 24 months. In month 7 it began a joint
quality programme. Coverage is voluntary and partial: a supplier can have 0 to 100% of its
receipts under the programme, and coverage changes month to month as audits are scheduled.

- **Treatment `X`:** programme coverage, share of receipts under the programme, 0 to 1.
  Mean 0.42, standard deviation 0.31.
- **Outcome `Y`:** rejected lots per 100 received lots, in percentage points. Mean 4.6,
  standard deviation 2.9.
- **Confounder `W1`:** supplier capability index, standardized, mean 0, standard deviation 1.
  Measured from a pre-period quality audit, so it precedes treatment.
- **Confounder `W2`:** order complexity index, standardized. Measured from bills of material and
  order mix in the pre-period.
- **Clustering:** 380 suppliers. Residual degrees of freedom used below: 379.

The decision rule agreed in advance: expand the programme to all suppliers only if the adjusted
effect exceeds 0.50 percentage points of rejects.

### 6.2 The two regressions

| Model | Coverage coefficient | Std. error | t | R-squared | Controls |
|---|---:|---:|---:|---:|---|
| Short | -1.20 pp | 0.24 | -5.00 | 0.03 | None |
| Long | -0.40 pp | 0.19 | -2.11 | 0.36 | `W1`, `W2` |

The coefficient falls by two thirds when the two confounders enter, and the R-squared rises from
0.03 to 0.36. On the naive stability reading, the result looks fragile. The arithmetic below
shows what actually happened, and it is not what the coefficient movement alone suggests.

Intercepts, for completeness: the short model is `reject = 5.10 - 1.20 cover`. The long model is
`reject = 4.77 - 0.40 cover - 2.00 capability + 0.25 complexity`. Both reproduce the sample mean
of 4.6 at the sample means of the regressors: `5.10 - 1.20(0.42) = 4.60` and
`4.77 - 0.40(0.42) - 2.00(0) + 0.25(0) = 4.60`.

### 6.3 The OVB decomposition, by hand

The auxiliary regressions give the two `delta` values. Capability is 0.45 standard deviations
higher per unit of coverage; complexity is 0.40 standard deviations higher. Both are plausible:
capable suppliers adopt the programme, and complex suppliers are prioritised for support.

| Component | `gamma_j` (pp per sd) | `delta_j` (sd per unit coverage) | Product (pp) |
|---|---:|---:|---:|
| Capability `W1` | -2.00 | +0.45 | -0.90 |
| Complexity `W2` | +0.25 | +0.40 | +0.10 |
| **Total bias** | | | **-0.80** |

Check the identity:

```text
   beta_short  =  beta_long + bias
   -1.20       =  -0.40 + (-0.80)
   -1.20       =  -1.20
```

The naive estimate overstates the programme's benefit by 0.80 percentage points, which is two
thirds of its own size. The larger contributor is capability, and the direction is what the
selection story predicts: capable suppliers both adopt the programme and have lower reject rates,
so the short regression credits the programme with capability's effect.

Complexity works the other way and partly offsets it. Complex orders are more likely to receive
support and more likely to be rejected, so omitting complexity biases the estimate toward zero.
The net is still strongly negative. This is why Step 5 matters: a confounder that biases toward
zero does not need to be "controlled away" to protect the conclusion, but it does need to be
named, because it changes the size of the bias you attribute to the others.

### 6.4 Oster's delta

Take `R_short = 0.03`, `R_long = 0.36`, and the conventional `R_max = 1.3 x 0.36 = 0.468`.

```text
   C  =  (0.468 - 0.36) / (0.36 - 0.03)
      =  0.108 / 0.33
      =  0.3273

   beta_star  =  -0.40 - 1.00 * (-1.20 - (-0.40)) * 0.3273
              =  -0.40 - 1.00 * (-0.80) * 0.3273
              =  -0.40 + 0.2618
              =  -0.138 pp

   delta_O*   =  -0.40 / ( (-0.80) * 0.3273 )
              =  -0.40 / (-0.2618)
              =  1.53
```

Read the two numbers together, because they say different things.

- `beta_star = -0.14 pp` is Oster's bias-adjusted estimate at equal selection. It is a third of
  the long-regression estimate and well below the 0.50 pp decision threshold. If you take the
  equal-selection assumption seriously, the programme does not clear the bar.
- `delta_O* = 1.53` says unobservables would have to be about 1.5 times as strongly selected as
  the observed controls to drive the estimate to zero. That is above Oster's benchmark of 1, so
  the result is not trivially fragile on this metric.

The apparent contradiction is the lesson. A result can be stable in the `delta` sense and still
be too small to justify the decision. Sensitivity analysis answers "could it be zero?", not "is
it big enough?" Those are different questions, and the decision rule is about the second.

> **Note:** `delta_O* > 1` is often reported as "the result is robust". It is not. It says only
> that a confounder must be stronger than the observed controls to make the estimate *exactly*
> zero. If the decision threshold is larger than the long estimate, the estimate already fails
> the decision, and the delta answers a question nobody asked.

### 6.5 The E-value

The E-value needs a risk ratio. Convert the linear probability coefficients to predicted
rejection rates at coverage 0 and coverage 1, holding the confounders at their means.

| Model | Rate at `cover = 0` | Rate at `cover = 1` | Risk ratio | E-value |
|---|---:|---:|---:|---:|
| Short | 5.10% | 3.90% | 0.765 | 1.94 |
| Long | 4.77% | 4.37% | 0.916 | 1.41 |
| Long, upper 95% limit | 4.77% | 4.74% | 0.994 | 1.09 |

The arithmetic for the long model:

```text
   RR      =  4.37 / 4.77  =  0.9161
   1 / RR  =  1.0916
   E       =  1.0916 + sqrt( 1.0916 * 0.0916 )
           =  1.0916 + sqrt( 0.09999 )
           =  1.0916 + 0.3162
           =  1.41
```

Interpretation: an unmeasured confounder would need a risk ratio of at least 1.41 with **both**
programme coverage and rejection to explain away the adjusted association. For comparison, the
observed capability variable is associated with rejection at a considerably larger risk ratio,
so a confounder of that strength is not implausible. The upper confidence limit has an E-value of
1.09, which means a confounder barely stronger than chance could move the interval to include the
null.

The naive model's E-value of 1.94 is the most misleading number in the table. It looks more
robust precisely because the effect looks bigger, and the effect looks bigger because it is
confounded. Sensitivity analysis applied to the wrong estimate produces confident nonsense.

> **Note:** The conversion from a percentage-point coefficient to a risk ratio fixes the baseline
> risk at the model's predicted value. If the baseline varies systematically across suppliers,
> the E-value is an approximation and should be labelled as one; the alternative is to estimate
> the effect on a ratio scale, which changes the estimand.

### 6.6 The robustness value and the impact threshold

Using the long regression, `t = 2.11` and `df = 379`:

```text
   f  =  2.11 / sqrt(379)  =  2.11 / 19.47  =  0.1084
   f^2 = 0.01175
   f^4 = 0.000138

   RV_1  =  0.5 * ( sqrt(0.000138 + 4*0.01175) - 0.01175 )
         =  0.5 * ( sqrt(0.04714) - 0.01175 )
         =  0.5 * ( 0.2171 - 0.01175 )
         =  0.5 * 0.2054
         =  0.103
```

So a confounder explaining about 10.3% of the residual variance of **both** coverage and rejects
would be enough to drive the point estimate to zero. For the significance target, subtract the
critical value in the same metric, `1.966 / 19.47 = 0.1010`, leaving `f = 0.0074` and a robustness
value of about 0.7%. That is the knife-edge: the estimate is significant, and a confounder
explaining less than one percent of residual variance on each side would make it not.

The impact threshold says the same thing in a different currency. From `t = 2.11` and `df = 379`,
the partial correlation between coverage and rejects is:

```text
   r  =  2.11 / sqrt(2.11^2 + 379)  =  2.11 / 19.58  =  0.1078
   r# =  1.966 / sqrt(379 + 1.966^2)  =  1.966 / 19.57  =  0.1005
   ITCV = (0.1078 - 0.1005) / (1 - 0.1005)  =  0.0073 / 0.8995  =  0.0081
```

An impact of 0.008 — a confounder whose partial correlations with coverage and rejects multiply
to 0.008, which equal-strength correlations of about 0.09 would produce — is enough to make the
coefficient non-significant. For calibration, the observed capability control has a partial
R-squared of about 0.30 with rejects and 0.10 with coverage, giving an impact of
`sqrt(0.30 x 0.10) = 0.173`, more than twenty times the threshold. The long estimate sits one
standard error from the significance boundary, so every threshold computed against significance
will be tiny. That is not a scandal; it is arithmetic.

The two robustness values are not contradictory: one targets zero, the other targets
significance, and the estimate sits close to the significance boundary. Reporting only the first
overstates robustness; reporting only the second makes every borderline result sound equally
fragile.

### 6.7 The empirical probes

Three probes were run, and two of them are uncomfortable.

| Probe | Construction | Result | Reading |
|---|---|---|---|
| Placebo treatment | Fake start 12 months before the real one, on eventually-treated suppliers | -0.52 pp (SE 0.26), p = 0.045 | A pre-existing trend of about half the naive effect |
| Negative-control outcome | Invoice price variance, which the programme cannot plausibly affect | +0.02 pp (95% CI -0.09 to +0.13) | Passes; no evidence of a generic reporting shift |
| Specification curve | 72 pre-registered specifications across coverage coding, window, controls, fixed effects, trimming | Median -0.61 pp; 61% negative and significant; range -1.15 to +0.22 | Fails the pre-registered 80% criterion |

The placebo is the most damaging result. A coefficient of -0.52 pp before the programme started
means that suppliers entering the programme were already improving. The negative control passes,
which rules out a simple reporting artifact but not selection on improvement trajectories. The
specification curve shows a distribution centred on a real negative association, but with a tail
that crosses zero and a share of significant results below the threshold fixed in advance.

### 6.8 The decision, and what it should have been

Put the four pieces of evidence together:

| Evidence | Value | What it supports |
|---|---|---|
| Short estimate | -1.20 pp | An association, not an effect |
| Long estimate | -0.40 pp (95% CI -0.77 to -0.03) | A small negative effect, barely distinguishable from zero |
| OVB decomposition | -0.80 pp of the naive estimate is selection | The naive number is not usable for the decision |
| Oster `delta_O*` | 1.53 | Not fragile in the selection-ratio sense |
| E-value (adjusted) | 1.41; 1.09 at the CI limit | Fragile on the risk ratio scale |
| Robustness value | 10.3% to zero; 0.7% to non-significance | Fragile on the significance margin |
| Placebo | -0.52 pp, p = 0.045 | A pre-existing trend is present |
| Specification curve | 61% significant, below the 80% criterion | The result is not robust to analytic choices |

The honest conclusion is not "the programme works" and not "the programme does nothing". It is:
**the association is real, roughly a third of it survives adjustment, the adjusted effect is
below the decision threshold, and a pre-existing trend of half the naive size cannot be ruled
out.** The recommendation is to continue the programme only with a design that can separate it
from the trend: a staggered rollout with a comparison group, or a matched comparison within
supplier capability strata. That is a design recommendation, which is the correct output of a
failed sensitivity analysis.

### 6.9 What an analyst should have asked

Coverage may be triggered by rejections, which reverses the design; the capability index may
include the programme period, which makes it a mediator; the placebo may reflect anticipation, a
trend, or a seasonal artifact, which have different remedies; and the 72 specifications may not be
equally defensible. Each is a design question, and none requires a better estimator.

## 7. Code, query, or calculation

The four sensitivity quantities in this topic are short enough to compute from a coefficient, a
t-statistic, and two R-squared values. The block below reproduces every number in section 6 and
asserts the OVB identity, so it doubles as a check on the hand arithmetic.

```python
import math
b_short, b_long = -1.20, -0.40            # pp per unit coverage
g1, d1 = -2.00, 0.45                      # capability: pp/sd, sd per unit coverage
g2, d2 = 0.25, 0.40                       # complexity: pp/sd, sd per unit coverage
R2s, R2l = 0.03, 0.36
Rmax = min(1.3 * R2l, 1.0)
bias = g1 * d1 + g2 * d2                  # -0.80
assert abs(b_short - (b_long + bias)) < 1e-9
C = (Rmax - R2l) / (R2l - R2s)            # 0.3273
delta_star = b_long / ((b_short - b_long) * C)   # 1.53
rr = 4.37 / 4.77                          # protective, so invert first
e_value = (1 / rr) + math.sqrt((1 / rr) * (1 / rr - 1))
t, df = 2.11, 379
f = t / math.sqrt(df)
rv_zero = 0.5 * (math.sqrt(f**4 + 4 * f**2) - f**2)
r = t / math.sqrt(t**2 + df)
r_crit = 1.966 / math.sqrt(df + 1.966**2)
itcv = (r - r_crit) / (1 - r_crit)
print(f"bias={bias:.2f} C={C:.4f} delta*={delta_star:.2f}")
print(f"E={e_value:.2f} RV={rv_zero:.3f} ITCV={itcv:.4f}")
```

Two implementation notes. First, `Rmax` is capped at 1, since `1.3 x R_long` exceeds one whenever
`R_long > 0.77`. Second, `1.966` is the two-sided 5% normal quantile, close enough to the `t`
critical value at 379 degrees of freedom for a teaching example; in production use the exact
quantile and the degrees of freedom that match your clustering.

## 8. Data, standards, and regulatory context

Sensitivity analysis is not a reporting requirement anywhere, and no regulator prescribes an
E-value. What the surrounding standards do is determine whether the data needed for the analysis
exists at all. Three artifacts matter more than the rest.

**The selection record.** Every confounder argument begins with the question "why did this unit
get treated?" In supply chain that answer is usually recorded somewhere operational, not
analytically: an audit schedule, a category strategy, a supplier segmentation, a lane
optimisation, a capital committee minute. If the selection rule is not recoverable, Step 4
becomes speculation. The single highest-value data request in a sensitivity analysis is often a
process document rather than a table.

**Pre-period covariates.** Confounders must precede treatment. Quality-management systems under
ISO 9001 record audit findings and non-conformities; automotive suppliers under IATF 16949
record process audits and corrective actions; medical-device manufacturers under ISO 13485 and
21 CFR 820 keep device history records and supplier evaluation files. These are the records that
let you build a capability covariate that is genuinely pre-treatment. A capability index computed
from the programme period is a mediator and will remove part of the effect you are trying to
measure.

**Traceability events.** GS1 standards and EPCIS 2.0 give lot-level movement and status events.
They are excellent for constructing outcomes (receipt, inspection, disposition) and poor for
constructing selection, because they record what happened, not why the unit was chosen.

| Data artifact | What it supplies | Typical defect | Consequence for sensitivity |
|---|---|---|---|
| Audit schedule and scope | The selection rule for coverage | Stored in a planner's spreadsheet, not a system | Candidate confounders cannot be enumerated |
| Supplier segmentation | Strategic versus transactional tier | Rebuilt annually, so the pre-period tier is lost | Tier is measured post-treatment |
| Pre-period audit score | Capability covariate | Scored on a different scale each year | `delta` is mis-scaled, bias mis-sized |
| Non-conformance records | Outcome and covariate | Reason codes default to "other" | Outcome measurement error is non-classical |
| Receipt and inspection logs | Rejection rate | Inspection sampling rate varies by supplier | Outcome severity is confounded with inspection |
| Contract and price files | Negative-control outcome | Discounts netted into list price | Control variable is itself affected by treatment |
| Lane and mode history | Difference-in-differences panel | Reroutes recorded at shipment, not at decision | Treatment timing is misdated |

The regulatory context is indirect but real. Model risk management guidance in banking asks for
conceptual soundness, ongoing monitoring, and outcome analysis, which is the vocabulary of this
topic applied to credit and market models rather than suppliers. AI governance regimes
increasingly require documented assumptions and validation evidence for high-risk systems, and a
sensitivity table is the cheapest artifact that satisfies the spirit of that requirement. Neither
constrains a supply chain analyst directly. Both signal where an auditor will look.

> **Note:** Records kept for quality certification are designed to demonstrate conformity, not
> to support causal inference. An audit file that shows a supplier passed says nothing about why
> the supplier was audited. Treat certification records as a source of covariates, never as a
> source of the selection rule.

## 9. Industry practice and case evidence

The public record on formal sensitivity analysis in supply chain is thin, and it is worth saying
so plainly before describing what does exist.

**The methodological gate has moved.** The editors of the *Journal of Operations Management*
announced in 2015 that manuscripts which uncritically interpret cross-sectional survey
correlations as causal would no longer be sent for review, and that authors must address
endogeneity through theory, design, or estimation (Guide and Ketokivi, 2015). This is editorial
policy, not evidence about practice, but it marks the point at which "we controlled for a few
variables" stopped being an acceptable answer in the field's leading journal.

**The common industry pattern is a pre/post comparison with no comparison group.** A programme is
launched, a metric improves, and the improvement is attributed to the programme in a business case
rather than a regression, so no standard error and no sensitivity analysis is requested. This is
observable in procurement transformation reporting; it is not a measured base rate.

**Vendor claims are the weakest evidence in the area.** Procurement analytics vendors report
savings attributable to their recommendations by comparing realised prices with a baseline the
vendor also constructed; because that baseline is the counterfactual, the vendor controls both
sides. This is Grade D evidence: dated, unaudited, usually unfalsifiable. The correct response is
to ask which assumption the baseline encodes, and what a different baseline would produce.

**A useful analogue from marketing measurement.** Spend and sales respond to the same demand
shocks, so a marketing mix model attributes correlation to incrementality. Industry guidance has
documented that model-based and attribution-based estimates routinely disagree, and that vendor
methodology opacity is a principal complaint (Interactive Advertising Bureau, 2019). Vendor
documentation shows the mechanism directly: a simulated unobserved confounder driving both spend
and sales inflates a channel's return on ad spend and can invert the ranking of channels, and
experimental calibration pulls the estimate back while making out-of-sample fit *worse*
(PyMC-Marketing documentation, 2025–2026, Grade D). That last clause is the transferable lesson:
fit quality is not evidence of causal validity, and a model can become more accurate about the
past while less correct about the counterfactual.

| Claim type | Evidence typically offered | Sensitivity typically reported | Grade |
|---|---|---|---|
| Peer-reviewed operations study | Regression with controls, robustness table | Sometimes; OVB or alternative specifications | A or B |
| Journal editorial policy | Stated requirement for endogeneity treatment | Not applicable | A |
| Consultancy value-realisation report | Pre/post metric, no comparison group | None | D or E |
| Procurement software savings claim | Realised versus vendor-built baseline | None; baseline not disclosed | D |
| Marketing mix calibration case | Lift tests versus model estimate | Implied by the calibration, not tabulated | C or D |
| Supplier scorecard vendor | Ranking accuracy, not causal effect | None | D |

## 10. Failure modes and red flags

| # | Failure mode | What it looks like | Why it misleads | Fix |
|---:|---|---|---|---|
| 1 | Sensitivity run on the naive estimate | E-value computed for the unadjusted coefficient | A larger confounded effect produces a larger E-value, so the naive model looks more robust | Run every check on the adjusted estimate, and report the naive one only as a contrast |
| 2 | Passing checks with no power | Placebo returns a null with a wide interval | A test that could not detect an effect is reported as evidence of no effect | State the minimum detectable effect for every probe |
| 3 | `R_max = 1` | The bound is reported as the extreme scenario | With `R_max = 1` the bound is usually vacuous and always uninformative | Report `R_max` explicitly; use 1.3 x `R_long` or a stated alternative |
| 4 | E-value from a risk difference | A percentage-point coefficient fed into the formula | The formula is defined on the ratio scale; the result is not an E-value | Convert with a stated baseline risk, or estimate on a ratio scale |
| 5 | E-value for the point estimate only | A single number, no interval version | The interval limit is usually the binding constraint | Report both, and lead with the limit nearest the null |
| 6 | Post-hoc specification curve | The specification set includes only variants that were run | The distribution is selected, so its spread is understated | Pre-register the set, including specifications you expect to fail |
| 7 | Negative control that is not U-comparable | Any unrelated outcome is used | A null proves nothing if the control does not share the confounding structure | Write the shared-bias argument before estimating |
| 8 | `delta_O* > 1` read as robust | The number is quoted without the decision threshold | It answers "could it be zero", not "is it big enough" | Report the bias-adjusted estimate against the decision rule |
| 9 | Sign error in the OVB decomposition | Bias added with the wrong sign | The conclusion reverses while the arithmetic looks fine | Check the identity `beta_short = beta_long + bias` numerically |
| 10 | Post-treatment control | Capability measured during the programme | The control absorbs part of the effect and biases the estimate | Verify each control's measurement date precedes treatment |
| 11 | Sensitivity theatre | Thirty checks, no thresholds, all "passed" | Volume substitutes for a falsifiable claim | Fix five to eight rungs with pass conditions in advance |
| 12 | OVB formula applied to a nonlinear model | Logit marginal effects decomposed with the linear identity | The identity is exact only for linear least squares | Use the linear probability model, or a simulation-based analysis |
| 13 | HonestDiD transplanted to a cross-section | Relative-magnitudes bound applied to a design with no pre-period | The restriction is defined on event-time violations | Match the sensitivity tool to the design |

The pattern behind most of these is a check that answers a different question from the decision.
Failures 1 and 8 are the same mistake in different clothing: a correct number, irrelevant to the
choice.

## 11. Assumptions and limits

| Assumption | What it says | What breaks it | Diagnostic | If violated |
|---|---|---|---|---|
| Linear specification | The short and long models are linear in the treatment and the confounder | Threshold effects, interactions, or a bounded outcome near its floor | Residual plots, specification curve | The decomposition is an approximation; say so |
| Correct long regression | The long model includes the confounders without omitted variable bias of its own | Another unmeasured confounder remains | The whole exercise is recursive; report the residual | Interpret the long estimate as an upper bound on adjustment, not a truth |
| Measurement without classical error | Variables are measured well enough that attenuation is second-order | Inspection sampling varies with supplier | Compare inspection rates across groups | Bias direction changes; magnitude is unknown |
| Pre-treatment covariates | Controls are measured before treatment | Running averages, post-period audits | Check measurement dates against treatment dates | The control is a mediator; drop it and report |
| No interference | One supplier's treatment does not affect another's outcome (SUTVA) | Shared auditors, capacity shifting between suppliers | Map the exposure pathway | Standard errors and estimands both change |
| `R_max` is a defensible ceiling | 1.3 x `R_long` approximates the explanatory power of unobservables | The outcome has a large unpredictable component | Report the result across `R_max` values | The delta conclusion is sensitive to a convention |
| Proportional selection | Unobservables are selected in the same proportion as observables | Selection is qualitative, as with a strategic-supplier designation | Benchmark against a plausibly comparable control | The delta benchmark is not informative |
| Risk ratio scale | The E-value's outcome is a ratio with a meaningful baseline | Very common outcomes, or continuous outcomes with no natural risk | Check baseline risk and outcome definition | Convert, or use a different sensitivity tool |
| Parallel trends under the bound | Pre-treatment violations bound post-treatment violations | Anticipation, or a shock that arrives with the treatment | Event-study plot before and after | Widen the bound and report the breakdown value |

Two limits are not fixable by better execution. **The recursion does not terminate.** Every
sensitivity analysis is conditional on a model of the confounder, and that model can itself be
wrong: the decomposition tells you what a *particular* omitted variable would do, not that you
have enumerated the right set, which is why Step 4 is a mechanism exercise. **A bound is not a
prior.** An E-value of 1.41 means a confounder of that strength would suffice; it says nothing
about the probability of confounding. Turning a bound into a probability requires beliefs about
the distribution of confounders in your domain; if you report one, label it as a belief.

## 12. Exercises

### Exercise 1 — Decompose a two-confounder bias (calculation, 40 minutes)

You are given `beta_short = 0.85`, `beta_long = 0.40`, `gamma_1 = 0.40`, `delta_1 = 1.20`,
`gamma_2 = -0.05`, `delta_2 = 0.60`, where the outcome is cost per order in euros.
**Deliverable:** a table as in 6.3, the identity check, and one sentence naming the dominant confounder.
**Acceptance:** products sum to `beta_short - beta_long`; units on every term; signs stated in words.
**Time:** 40 minutes.

### Exercise 2 — Compute an E-value from a 2x2 table (calculation, 30 minutes)

A supplier audit programme is associated with a defect rate of 24 in 400 lots, against 40 in 400
among unaudited lots.
**Deliverable:** the risk ratio, the E-value, and the E-value for the CI limit nearest the null.
**Acceptance:** the protective-ratio inversion is shown step by step; the interpretation names both
associations the confounder would need.
**Time:** 30 minutes.

### Exercise 3 — Sign the bias before you estimate (analysis, 25 minutes)

For a proposed study of rail conversion on damage claims, list five candidate confounders and
complete the six-row sign table from section 5.1 for each.
**Deliverable:** five rows with `gamma` sign, `delta` sign, bias sign, and a mechanism.
**Acceptance:** one candidate biases toward zero and one away; each mechanism names a process, not a field.
**Time:** 25 minutes.

### Exercise 4 — Write the identifying assumption in one sentence (writing, 20 minutes)

Take the study you described in Exercise 3 and write its identifying assumption twice: once for
a regression with controls, once as if it were a difference-in-differences design.
**Deliverable:** two sentences under 30 words each, plus a note on which is more plausible.
**Acceptance:** neither uses "control for", "endogeneity", or "bias"; both name the comparison.
**Time:** 20 minutes.

### Exercise 5 — Pre-register a robustness ladder (design, 45 minutes)

For the Halden example, write a five-rung ladder before reading section 6.8 again.
**Deliverable:** a table with rung, question, test, pass condition, and failure consequence.
**Acceptance:** every pass condition is numeric; at least one failing rung does not change the
recommendation, and you say which.
**Time:** 45 minutes.

### Exercise 6 — Design a placebo and a negative control (design, 40 minutes)

For a study of a warehouse automation rollout on pick error rates, specify one placebo treatment
and one negative-control outcome.
**Deliverable:** a half-page memo covering construction, the shared-bias argument, the minimum
detectable effect, and what would change your mind.
**Acceptance:** the control is argued U-comparable, not merely unrelated; power is a number.
**Time:** 40 minutes.

### Exercise 7 — Read a specification curve (interpretation, 30 minutes)

A curve of 96 specifications has a median of -0.22, a range from -0.51 to +0.09, and 58% of
specifications significant in the predicted direction.
**Deliverable:** a 150-word memo stating what the curve supports and what it does not.
**Acceptance:** the median is not reported as the estimate; the decision rule is named and the
distribution tested against it; one choice driving the spread is identified.
**Time:** 30 minutes.

### Exercise 8 — Stress an Oster conclusion (calculation, 35 minutes)

Using `beta_short = -1.20`, `beta_long = -0.40`, `R_short = 0.03`, `R_long = 0.36`, recompute
`delta_O*` for `R_max` values of 0.40, 0.468, 0.60, and 0.80.
**Deliverable:** four rows with `C`, `delta_O*`, and the bias-adjusted estimate at `delta_O = 1`.
**Acceptance:** state which conclusion is invariant to `R_max` and which is not, in one sentence.
**Time:** 35 minutes.

### Exercise 9 — Find the assumption that would reverse the decision (analysis, 30 minutes)

Take a causal claim from your own organisation, or from a published supply chain case.
**Deliverable:** one page naming the single decisive assumption, the cheapest test, and its cost.
**Acceptance:** exactly one assumption is decisive; the test is feasible within a quarter; cost is
stated in person-weeks or currency.
**Time:** 30 minutes.

## 13. Self-check answers

### Answer 1 — Two-confounder decomposition (Exercise 1)

Write each term as a product before adding anything.

| Component | `gamma_j` (EUR per sd) | `delta_j` (sd per unit) | Product (EUR) |
|---|---:|---:|---:|
| `W1` | +0.40 | +1.20 | +0.480 |
| `W2` | -0.05 | +0.60 | -0.030 |
| **Total** | | | **+0.450** |

The arithmetic is `0.40 x 1.20 = 0.48` and `-0.05 x 0.60 = -0.03`, giving a bias of `+0.45`. The
identity check is `0.40 + 0.45 = 0.85`, which reproduces the short estimate exactly. Always run
this check: if it fails, one of the inputs is wrong, and the decomposition is worthless until you
find which.

`W1` is positively associated with both treatment and outcome, so omitting it inflates the short
estimate by 0.48 EUR. `W2` has a negative outcome link and a positive treatment link, so it
deflates the estimate by 0.03 EUR. The net is dominated by `W1`, and the short estimate overstates
the effect by more than half its own size.

### Answer 2 — E-value from a 2x2 table (Exercise 2)

| Group | Lots | Defects | Risk |
|---|---:|---:|---:|
| Audited | 400 | 24 | 0.060 |
| Not audited | 400 | 40 | 0.100 |

```text
   RR        =  0.060 / 0.100  =  0.600
   protective, so invert:  1 / 0.600  =  1.6667
   E         =  1.6667 + sqrt( 1.6667 x 0.6667 )
             =  1.6667 + sqrt( 1.1111 )
             =  1.6667 + 1.0541
             =  2.72
```

For the interval, work on the log scale:

```text
   ln(0.600)      =  -0.5108
   SE             =  sqrt( 1/24 - 1/400 + 1/40 - 1/400 )
                  =  sqrt( 0.041667 - 0.0025 + 0.025 - 0.0025 )
                  =  sqrt( 0.061667 )  =  0.2483
   95% CI log     =  -0.5108 +/- 1.96 x 0.2483  =  (-0.9976, -0.0241)
   95% CI ratio   =  ( 0.369, 0.976 )
```

The limit nearest the null is 0.976. Invert it to 1.0244 and apply the formula:

```text
   E_CI  =  1.0244 + sqrt( 1.0244 x 0.0244 )
         =  1.0244 + sqrt( 0.02500 )
         =  1.0244 + 0.1581
         =  1.18
```

Interpretation: a confounder would need a risk ratio of at least 2.72 with both audit status and
defect status to explain away the point estimate, but only 1.18 to move the interval to include
the null. The second number is the one that matters, and it says the result is fragile.

### Answer 3 — Signing the bias (Exercise 3)

Five rows, each with a mechanism rather than a sign alone. Two of the required form: "Lane
congestion is positively associated with conversion (busy lanes are prioritised) and with damage
(longer handling), so the bias is positive." "Contract age is higher for new contracts, which
convert more and carry less legacy damage, so the bias is negative." The mechanism sentence is
the graded element.

### Answer 4 — One-sentence assumptions (Exercise 4)

Regression form: "After adjusting for lane distance, product family, and pre-period damage rate,
rail conversion is unrelated to any other determinant of damage claims." Difference-in-differences
form: "Absent conversion, converted and non-converted lanes would have followed parallel damage
claim trends." The second is the stronger claim, because it is testable in the pre-period; the
first is unfalsifiable without a model of what "any other determinant" includes.

### Answers 5 and 6 — Design rubric for the ladder and the probes

| Criterion | 0–1 point | 2–3 points | 4–5 points |
|---|---|---|---|
| Thresholds | Qualitative ("should be small") | One numeric threshold | Every rung has a numeric pass condition |
| Ordering | Arbitrary | Roughly cheapest first | Order states which assumption is most likely to fail |
| Failure consequence | Not stated | "Downgrade the claim" | Named alternative decision or design |
| Shared-bias argument | "It is unrelated" | Names one shared cause | Names the confounding structure and why the control shares it |
| Power | Not mentioned | MDE stated | MDE stated and compared with the effect size of interest |
| Falsifiability | Any result is "fine" | One result would worry the analyst | A pre-stated result that would change the design |

Score each design out of 30. Below 15, the artifact is a table rather than a test.

### Answer 7 — Reading the curve (Exercise 7)

The curve supports a negative association present in most defensible specifications (58%
significant in the predicted direction), median -0.22, range crossing zero. It does not support a
point estimate of -0.22: the median summarises analytic choices, not a parameter. If the decision
rule requires an effect of at least 0.30 in magnitude, the distribution does not clear it — the
median is below the threshold and the upper tail reaches +0.09. The honest memo says the sign is
reasonably stable and the magnitude is not, and names the choice driving the spread.

### Answer 8 — Stress an Oster conclusion (Exercise 8)

With `C = (R_max - 0.36) / 0.33` and `beta_star = -0.40 + 0.80 x C`:

| `R_max` | `C` | `beta_star` at `delta_O = 1` | `delta_O*` |
|---:|---:|---:|---:|
| 0.40 | 0.1212 | -0.303 | 4.13 |
| 0.468 | 0.3273 | -0.138 | 1.53 |
| 0.60 | 0.7273 | +0.182 | 0.69 |
| 0.80 | 1.3333 | +0.667 | 0.38 |

The conclusion is not invariant to `R_max`. At a ceiling just above the long R-squared the result
looks robust (`delta_O* = 4.13`); at a ceiling of 0.60 the equal-selection estimate changes sign
and `delta_O*` falls below 1, which is the conventional definition of fragile. `R_max` is doing
enormous work here because the observed controls moved the R-squared by 0.33, which makes the
denominator of `C` large. Any report that quotes a single `delta_O*` without the `R_max` behind it
has hidden the most consequential choice in the calculation.

### Answer 9 — The decisive assumption (Exercise 9)

A strong answer names one assumption, not three. For the Halden case it is not "no confounding" in
general but "suppliers were not already on differential improvement trajectories when the
programme began". The cheapest test is the placebo date — one analyst-week, existing data — and
it is the test that failed. Naming the assumption at that specificity makes the test obvious.

## 14. Interview questions

**1. "What is omitted-variable bias, in one sentence?"**
**Strong:** the difference between the short and long coefficients, equal to the omitted variable's
effect on the outcome times its regression on the treatment; an identity, not a modelling
assumption. **Weak:** "bias from variables you forgot to include", which describes a data problem
rather than a specification one.

**2. "Your coefficient falls by half when you add controls. Is the result fragile?"**
**Strong:** not necessarily; ask what happened to the R-squared, because bias is proportional to
the coefficient movement scaled by the R-squared movement, and a large movement with a large
R-squared rise may be the removal of bias. **Weak:** "yes, it is fragile", or "no, it is stable",
without the second moments.

**3. "What does an E-value of 1.4 mean?"**
**Strong:** an unmeasured confounder would need a risk ratio of at least 1.4 with both treatment
and outcome to explain away the association; it is a bound, not a probability, and the interval
version is usually smaller. **Weak:** "there is a 1.4 times chance of confounding", which
misreads a bound as a likelihood.

**4. "Why is the E-value for a tiny but precise effect also tiny?"**
**Strong:** the E-value is a function of the effect size, not the standard error; precision makes
an estimate significant, not robust, and a small effect is easy to explain away. **Weak:** "small
effects are less important", which is true but not the mechanism.

**5. "What is Oster's delta, and what does delta greater than 1 tell you?"**
**Strong:** the ratio of unobserved to observed selection needed to produce the bias-adjusted
estimate; above 1 means unobservables must be more strongly selected than observables to drive
the estimate to zero, and it says nothing about whether the estimate clears the decision
threshold. **Weak:** "the result is robust", full stop.

**6. "What makes a good negative control?"**
**Strong:** an outcome the treatment cannot affect but which shares the unobserved causes of the
target outcome — U-comparable — with a stated shared-bias argument and enough power to detect a
meaningful violation. **Weak:** "an unrelated outcome", which is necessary but not sufficient.

**7. "Your placebo test is null. Are you reassured?"**
**Strong:** only if the placebo had power; ask for the minimum detectable effect and compare it
with the size of the pre-trend you were worried about. **Weak:** "yes, the placebo passed", which
treats absence of evidence as evidence of absence.

**8. "What is a specification curve, and what is the answer it gives?"**
**Strong:** estimates across a pre-registered set of defensible specifications, with joint
inference over the set; the distribution and the share clearing the decision rule are the answer,
not the median and not the maximum. **Weak:** "we ran many models and picked the robust one".

**9. "How would you sensitivity-test a difference-in-differences estimate?"**
**Strong:** bound the post-treatment violation of parallel trends relative to the largest
pre-treatment violation, report the breakdown value of the bound at which the conclusion changes,
and note that the pre-trend test itself is underpowered. **Weak:** "run an event study and check
the pre-trends", which is necessary but not a sensitivity analysis.

**10. "A vendor says its tool cut procurement costs by 12%. What is your first question?"**
**Strong:** what is the counterfactual, who constructed it, and what would the number be under a
different baseline; a realised-versus-baseline comparison where the vendor owns both sides is not
an estimate of the tool's effect. **Weak:** "what is the sample size?", which is the right
question for a different problem.

## 15. Key terms

- **Omitted-variable bias** — the gap between short and long coefficients, equal to `gamma x delta`.
- **Short regression** — the model without the confounder; the naive estimate.
- **Long regression** — the model with the confounder; still not necessarily causal.
- **Auxiliary regression** — the regression of the omitted variable on treatment and the other omitted variables.
- **`gamma`** — the omitted variable's effect on the outcome, holding treatment.
- **`delta`** — the omitted variable's regression on the treatment.
- **Partial R-squared** — residual variance in one variable explained by another, after controls.
- **Robustness value** — the equal partial R-squared needed to move the estimate by a stated amount.
- **Impact threshold (ITCV)** — the correlation product needed to make a coefficient non-significant.
- **E-value** — the minimum risk-ratio strength with both treatment and outcome needed to nullify an association.
- **Bounding factor** — the multiplicative bias bound `RR_EU x RR_UD / (RR_EU + RR_UD - 1)`.
- **Risk ratio** — outcome risk in the treated divided by risk in the untreated.
- **Risk difference** — absolute difference in risk, in percentage points.
- **Oster's delta** — the ratio of unobserved to observed selection implied by a bias-adjusted estimate.
- **`R_max`** — the hypothetical R-squared including all unobservables; a convention, not a fact.
- **Proportional selection** — the assumption that unobservables are selected like observables.
- **Bias-adjusted estimate** — the estimate under a stated selection assumption, not an identified quantity.
- **Placebo treatment** — a fake treatment date or assignment used to detect pre-existing differences.
- **Negative control outcome** — an outcome unaffected by treatment but sharing its confounding structure.
- **U-comparable** — sharing the unobserved causes of the target variable.
- **Specification curve** — estimates across a pre-registered set of defensible specifications.
- **Robustness ladder** — an ordered set of checks with thresholds fixed in advance.
- **Breakdown value** — the sensitivity parameter value at which the conclusion changes.
- **Relative-magnitudes restriction** — a bound on post-treatment trend violations relative to pre-treatment ones.

## 16. Sources

**Peer-reviewed methods (Grade A)**

- Oster, E. (2019). "Unobservable Selection and Coefficient Stability: Theory and Evidence."
  *Journal of Business & Economic Statistics*, 37(2), 187–204. DOI: 10.1080/07350015.2016.1227711.
  Source of the coefficient-stability bound and the `R_max = 1.3 x R_long` convention.
- VanderWeele, T. J., and Ding, P. (2017). "Sensitivity Analysis in Observational Research:
  Introducing the E-Value." *Annals of Internal Medicine*, 167(4), 268–274.
  DOI: 10.7326/M16-2607. Defines the E-value and recommends reporting it for the estimate and the
  confidence limit nearest the null.
- Ding, P., and VanderWeele, T. J. (2016). "Sensitivity Analysis Without Assumptions."
  *Epidemiology*, 27(3), 368–377. DOI: 10.1097/EDE.0000000000000457. Source of the bounding-factor
  inequality from which the E-value is derived.
- Frank, K. A. (2000). "Impact of a Confounding Variable on a Regression Coefficient."
  *Sociological Methods & Research*, 29(2), 147–194. DOI: 10.1177/0049124100029002001. Source of
  the impact threshold and the equal-correlation worst case.
- Cinelli, C., and Hazlett, C. (2020). "Making Sense of Sensitivity: Extending Omitted Variable
  Bias." *Journal of the Royal Statistical Society: Series B*, 82(1), 39–67.
  DOI: 10.1111/rssb.12348. Source of the partial-R-squared reparameterisation, the robustness
  value, and the `sensemakr` implementation.
- Simonsohn, U., Simmons, J. P., and Nelson, L. D. (2020). "Specification Curve Analysis."
  *Nature Human Behaviour*. DOI: 10.1038/s41562-020-0912-z. Defines the specification curve and
  joint inference over the specification set.
- Rambachan, A., and Roth, J. (2023). "A More Credible Approach to Parallel Trends." *Review of
  Economic Studies*, 90(5), 2555–2591. DOI: 10.1093/restud/rdad018. Relative-magnitudes and
  smoothness restrictions, honest confidence sets, and the `HonestDiD` implementation.
- Lipsitch, M., Tchetgen Tchetgen, E., and Cohen, T. (2010). "Negative Controls: A Tool for
  Detecting Confounding and Bias in Observational Studies." *Epidemiology*, 21(3), 383–388.
  DOI: 10.1097/EDE.0b013e3181d61eeb. Defines negative-control outcomes and exposures and the
  U-comparability condition.
- Guide, V. D. R., and Ketokivi, M. (2015). "Notes from the Editors: Redefining Some
  Methodological Criteria for the Journal." *Journal of Operations Management*, 37, v–viii.
  DOI: 10.1016/S0272-6963(15)00056-X. Editorial policy requiring authors to address endogeneity.

**Textbooks (Grade A)**

- Angrist, J. D., and Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An Empiricist's
  Companion*. Princeton University Press. Section 3.2.2 gives the short-equals-long-plus-bias
  identity and the auxiliary-regression derivation used in sections 5 and 6.

**Official standards (Grade B)**

- ISO 9001:2015, *Quality management systems — Requirements*.
- ISO 13485:2016, *Medical devices — Quality management systems — Requirements for regulatory purposes*.
- IATF 16949:2016, *Quality management system requirements for automotive production and relevant service parts organizations*.
- 21 CFR Part 820, *Quality System Regulation*, United States Food and Drug Administration.
- GS1 General Specifications, including EPCIS 2.0 event data.

**Industry and practitioner evidence (Grade C)**

- Interactive Advertising Bureau (2019). *The Essential Guide to Marketing Mix Modeling and
  Multi-Touch Attribution*. Documents disagreement between model-based and attribution-based
  estimates and vendor methodology opacity. Used only as an analogue, not as supply chain evidence.

**Vendor documentation (Grade D, dated)**

- PyMC-Marketing (2025–2026). "Mitigating Unobserved Confounders in MMMs with Lift Test
  Likelihoods." Documentation notebook. Reports a simulation in which an unobserved confounder
  inflates a channel's return on ad spend and inverts channel ranking, and in which experimental
  calibration improves the estimate while worsening out-of-sample fit. Vendor documentation, not
  peer reviewed.

**Press and anecdote (Grade E)**

- No press source is used for a quantitative claim. The description of pre/post value-realisation
  reporting in section 9 is a practitioner pattern, assembled from observation rather than from a
  survey, and must not be quoted as a base rate.

**Note on citations.** Details are as returned by the searches. Two claims were deliberately
softened: Oster's `R_max` calibration is described as coming from comparing randomised and
non-randomised estimates in the literature she studied, without quoting a specific ratio of
biases; and the marketing-measurement evidence is labelled vendor-documented rather than general.
The Halden Components figures are constructed for exposition, not an empirical result.

## 17. Links

This topic closes the module. Read it after the designs it stress-tests, then return to the
module map and reread the argument with the fragility of each link in mind.

- Module map and the ten-topic sequence: [`_index.md`](_index.md)
- Topic 01, the estimand this topic stress-tests: [`01_estimands-question-types.md`](01_estimands-question-types.md)
- Topic 06, the back-door paths that generate the candidate confounders: [`06_dags.md`](06_dags.md)
- Topic 08, the parallel-trends assumption bounded in section 5: [`08_difference-in-differences.md`](08_difference-in-differences.md)
- Topic 09, the exclusion restriction this topic asks about in threshold terms: [`09_elasticity-endogeneity.md`](09_elasticity-endogeneity.md)
- Next module, where these ideas become experimental design: [`../05_experimentation_pilots_evidence/_index.md`](../05_experimentation_pilots_evidence/_index.md)
- Survey file for the whole module: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
