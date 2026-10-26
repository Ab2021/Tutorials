# Drift

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 06 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 05
> Version 2026.3

## 1. Why this topic exists

Drift is the most discussed and least understood concept in production machine learning. It is
discussed because models do degrade after deployment, and the degradation is often invisible
without a deliberate monitoring effort. It is misunderstood because the word is used for four
different phenomena with different causes and different responses.

The four phenomena are covariate shift, label shift, concept drift, and upstream data change. A
covariate shift is a change in the distribution of inputs. A label shift is a change in the
distribution of outcomes. A concept drift is a change in the relationship between inputs and
outcomes. An upstream data change is a change in how a value is measured, coded, or delivered. The
first three are statistical; the fourth is operational and usually the most common.

Treating them as one thing produces two expensive errors. The first is retraining in response to a
measurement change, which teaches the model to compensate for a defect and embeds the defect in the
artifact. The second is ignoring a concept drift because the input distributions look stable, which
leaves a degrading model in service because the dashboard is green.

### 1.1 The decisions this topic unblocks

| Decision | Who is blocked | Consequence of treating drift as one thing |
|---|---|---|
| Whether to retrain, recalibrate, or fix the pipeline | Data science, data platform | Retraining churn or a missed degradation |
| When a drift alert is actionable | On-call, clinical owner | Alert fatigue or ignored warnings |
| How to set a threshold that works at the current sample size | Monitoring owner | Constant false alarms or no detection |
| Whether a reference distribution is still valid | Data science, quality | Comparisons against a stale baseline |
| Whether a change requires regulatory review | Regulatory | An unauthorized model change |
| How to distinguish drift from harm | Clinical owner | A statistical signal is mistaken for a clinical problem |

### 1.2 Drift is not harm

A change in the input distribution can be benign, expected, or even desirable. A hospital that
admits older patients in winter will see an age shift every winter. A laboratory that changes its
assay will see a creatinine shift that has nothing to do with the patient population. A model can
drift and remain perfectly calibrated.

Harm is a change in the model's usefulness for the decision. It is measured by calibration,
discrimination, subgroup performance, and outcomes. Drift is a leading indicator that harm may be
occurring. The two must be monitored separately, and the response must be chosen from the evidence,
not from the drift statistic alone.

### 1.3 Why upstream changes dominate

In biomedical deployments, the most common cause of a sudden input shift is not a change in the
patient population. It is a change in the measurement or delivery system: a new laboratory analyzer,
a coding update, a unit change, a new interface, a firmware update, a shift in the time at which a
value becomes available.

These changes are operational, and they are often invisible to the data science team. The
laboratory knows about the analyzer change; the model team sees a population stability index (PSI) alert three weeks later. The
monitoring plan must connect the statistical signal to the operational change log, or it will
diagnose every shift as a population change.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** covariate shift, label shift, concept drift, and upstream data change by cause
   and by evidence.
2. **Compute** PSI and the two-sample Kolmogorov-Smirnov statistic, and explain what each does and
   does not measure.
3. **Adjust** a drift threshold for sample size, and explain why a fixed threshold fails at small
   and large cohorts.
4. **Diagnose** a drift alert using feature distributions, missingness, calibration, and the
   upstream change log.
5. **Choose** a response from evidence: monitor, fix the pipeline, recalibrate, reweight, retrain,
   or retire.
6. **Design** a reference-window policy and a drift review cadence.
7. **Explain** why drift detection is not performance monitoring.

## 3. The core mental model

Drift is a change in one of three distributions. The taxonomy is the mental model.

```text
             ┌─────────────────────────────────────────────┐
             │              p(X, y) = p(y|X) p(X)          │
             └─────────────────────────────────────────────┘
                    │                 │                │
                    ▼                 ▼                ▼
              p(X) changes      p(y) changes     p(y|X) changes
             COVARIATE SHIFT    LABEL SHIFT      CONCEPT DRIFT
                    │                 │                │
                    └────────┬────────┴────────┬───────┘
                             ▼                 ▼
                    performance may      performance
                    or may not change    usually changes
                             │
                             ▼
                  ┌──────────────────────┐
                  │  UPSTREAM CHANGE     │
                  │  measurement, coding,│
                  │  units, timing, feed │
                  └──────────────────────┘
```

The diagram is deliberately causal. Covariate shift changes what the model sees. Label shift
changes the base rate against which its probabilities are interpreted. Concept drift changes the
relationship it learned. Upstream change can produce any of the three, and is the first hypothesis
to test when a shift is sudden.

### 3.1 The diagnosis tree

A drift alert is a symptom. The diagnosis follows a tree.

```text
drift alert
   │
   ├── Is there an upstream change in the same window?
   │        │ yes ──▶ upstream change: fix data, backfill, re-verify
   │        │ no
   │        ▼
   ├── Is the change in p(X) only?
   │        │ yes ──▶ covariate shift: check calibration and performance
   │        │          ├── performance stable ──▶ monitor, document
   │        │          └── performance changed ──▶ recalibrate or retrain
   │        │ no
   │        ▼
   ├── Is the change in p(y) with p(X|y) stable?
   │        │ yes ──▶ label shift: recalibrate or reweight
   │        │ no
   │        ▼
   └── Is p(y|X) changed on labelled data?
            │ yes ──▶ concept drift: retrain and revalidate
            └── no ──▶ no drift; check the statistic and the reference
```

The tree is ordered by cost. Checking the upstream change log is cheap and often conclusive.
Retraining is expensive and should be the last branch, not the first.

### 3.2 Detection methods and what they measure

| Method | Measures | Sensitive to | Blind to |
|---|---|---|---|
| Population stability index (PSI) | Binned divergence between two distributions | Location and shape changes in binned data | Changes within bins; small samples |
| KS statistic | Maximum distance between empirical CDFs | Location and shape, especially in the body | Tail changes; multivariate structure |
| Jensen-Shannon | Symmetric divergence between distributions | Broad distributional change | Direction; calibration |
| Wasserstein | Earth-mover distance | Location and scale changes | Shape changes without mass movement |
| MMD | Distance between kernel mean embeddings | Multivariate and nonlinear changes | Interpretability; kernel choice |
| Prediction distribution | Change in model output | Anything that changes the output | Whether the output is right |
| Label distribution | Change in outcome prevalence | Label shift | Concept drift with stable prevalence |
| Performance on labels | Change in calibration or discrimination | Harm, whenever labels exist | Anything before labels arrive |

No single method detects all four drift types. A monitoring plan needs at least one input method,
one output method, and one performance method when labels permit.

### 3.3 Sample size changes the threshold

PSI and KS are sample-size dependent. A fixed threshold that works for a cohort of 50,000 will
fire constantly on a cohort of 200, and a threshold that is quiet on 200 will miss meaningful
changes on 50,000.

For PSI with B bins, under the null hypothesis that the two distributions are equal,
Yurdakul and Naranjo show that the expected value is approximately:

```text
E[PSI] ≈ (1/n + 1/m) × (B − 1)
```

With 10 bins and 5,000 observations in each sample, the expected null PSI is 0.0036. With 500
observations in each, it is 0.036. With 200, it is 0.09. A threshold of 0.1 is therefore very
conservative at 5,000 per sample and almost guaranteed to fire at 200.

The practical consequence is that a drift threshold must be stated with the sample size it was
calibrated for, or computed from the null distribution. A threshold without a sample size is not
a threshold; it is a number.

### 3.4 Drift patterns

The temporal pattern matters for the response.

- **Sudden**: a step change, often an upstream change or a policy change.
- **Gradual**: a slow trend, often a population or practice change.
- **Incremental**: many small changes that accumulate.
- **Recurring**: a seasonal pattern that returns, such as winter admissions.
- **Transient**: a spike that resolves, often a data outage or a coding error.

A sudden change should be investigated operationally first. A gradual change should be reviewed
against the retraining cadence. A recurring change should be modeled, not retrained away. A
transient change should be confirmed as resolved before any model action.

### 3.5 Why drift monitoring is not performance monitoring

Performance monitoring measures whether the model is right. Drift monitoring measures whether the
inputs or outputs have changed. The first requires labels; the second does not. The second is
faster and cheaper; the first is the only direct evidence of harm.

A plan that monitors only drift will detect some failures early and will miss others entirely. A
plan that monitors only performance will detect harm late and will not know why. The two are
complements, not substitutes.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Drift | A change over time in inputs, labels, or relationships | Any change in a metric | Unitless statistic |
| Covariate shift | A change in p(X) with p(y|X) unchanged | Any input change | PSI, KS, or distance |
| Label shift | A change in p(y) with p(X|y) unchanged | Class imbalance in training | Prevalence delta |
| Concept drift | A change in p(y|X) | Any performance change | Performance delta |
| Upstream change | A change in measurement, coding, units, timing, or delivery | A synonym for covariate shift | Change record |
| Virtual drift | A change in p(X) that does not affect the decision boundary | Harmless drift | Statistic only |
| Real drift | A change that affects the decision boundary | Any drift | Performance delta |
| Sudden drift | A step change at a point in time | Any large statistic | Change point |
| Gradual drift | A slow trend over time | A seasonal pattern | Slope |
| Recurring drift | A pattern that returns periodically | A new drift each season | Period |
| PSI | Population stability index, a binned divergence | A universal threshold | Unitless |
| KS statistic | Maximum distance between two empirical CDFs | A p-value | Unitless, 0 to 1 |
| Jensen-Shannon divergence | A symmetric, bounded divergence between distributions | A distance metric | Bits or nats |
| Wasserstein distance | The minimum cost of moving one distribution to another | A generic distance | Unit of the variable |
| MMD | Maximum mean discrepancy, a kernel two-sample statistic | A classifier accuracy | Unitless |
| Reference window | The baseline distribution used for comparison | The training set, always | Window identifier |
| Detection delay | Time from drift onset to detection | The monitoring interval | Days |
| False alarm rate | Fraction of alerts with no real change | Alert precision | Ratio |
| Effect size | Magnitude of the change, independent of sample size | The p-value | Standardized difference |
| p-value | Probability of the observed statistic under the null | The probability of drift | Probability |
| Importance weighting | Reweighting samples to correct for a distribution change | A universal fix | Weight |
| BBSE | Black box shift estimation of label shift | A model retrain | Weight per class |
| Recalibration | Adjusting the probability mapping | Retraining | Slope, intercept |
| Retraining | Fitting a new model on new data | Any model update | Run identifier |
| Backfill | Replacing missing or wrong historical values | Any data fix | Rows affected |
| Change point | A time at which the distribution changes | Any alert time | Timestamp |
| Concept | The relationship between inputs and outcome | The model | Function |
| Domain adaptation | Methods for transferring a model across distributions | Fine-tuning | Method family |

Two definitions are the source of most confusion.

**Covariate shift is not concept drift.** Covariate shift changes the inputs. Concept drift changes
the input-output relationship. A model can experience a large covariate shift and remain well
calibrated, because the relationship it learned still holds in the new region of input space. A
model can experience no covariate shift and still degrade, because the relationship changed while
the inputs stayed the same.

**A p-value is not an effect size.** With a large enough sample, any tiny difference is
statistically significant. A KS test with 100,000 observations will reject equality for a
difference that has no clinical meaning. The effect size, and its consequence for calibration or
performance, is what matters.

## 5. The mechanics, step by step

The procedure treats drift as a diagnostic process, not a single alert. The output is a diagnosis
and a response, not a statistic.

### Step 1 — Define and version the reference distribution

**Input.** The training cohort, the deployment population, and the model version.

**Output.** A reference dataset with a version, a date range, a sample size, and a stored binning
or kernel configuration.

**Rationale.** Every drift statistic compares a current window to a reference. If the reference is
not versioned, the comparison cannot be reproduced. If the bin edges are recomputed on the current
data, the statistic changes meaning between runs.

The reference should represent the population the model was validated on, not the entire history.
A reference that mixes pre-deployment and post-deployment data will hide the change the model
experienced at deployment.

Branch point: if the model is retrained, the reference must be updated to the new validation cohort
and the old reference archived. Comparing a new model to an old reference produces a drift alert
that is an artifact of the model change.

### Step 2 — Choose the comparison window

**Input.** The monitoring cadence, the event rate, and the label latency.

**Output.** A window length and a comparison schedule.

**Rationale.** The window length determines the sample size, which determines the null distribution
of the statistic. A daily window on a low-volume feature produces noisy statistics; a quarterly
window produces stable statistics but slow detection.

A workable default is a rolling window sized to contain at least 500 observations per feature, with
a minimum of one week. For rare features or small sites, aggregate across sites or lengthen the
window.

Branch point: if the window cannot reach the minimum sample, the statistic is not interpretable.
Report the sample size with the statistic and use a method appropriate to small samples, or wait.

### Step 3 — Select features to monitor

**Input.** Feature importance, clinical relevance, and data-quality risk.

**Output.** A prioritized list of features, with a rationale for each.

**Rationale.** Monitoring every feature produces many alerts and no prioritization. Monitoring only
the most important features can miss a data-quality failure in a feature the model uses weakly.
The list should include the top features by importance, the features with known upstream risk, and
the features that are required for safe operation.

A feature that is missing or wrong can change the output even if its coefficient is small, because
imputation or a default value can shift many predictions at once.

Branch point: if a feature is not monitored, the reason must be recorded. "We did not get to it" is
a residual risk, not a decision.

### Step 4 — Choose a statistic per feature type

**Input.** The feature types and the monitoring list.

**Output.** A statistic and a configuration for each feature.

**Rationale.** The statistic must match the data. PSI and KS are for continuous or ordinal
features. Chi-square or Jensen-Shannon is for categorical features. For counts, a rate ratio may be
more interpretable than a distribution distance.

The configuration matters as much as the statistic. PSI depends on the number and placement of
bins. KS depends on the sample size. A statistic reported without its configuration is not
reproducible.

Branch point: if a feature is multivariate or a text embedding, a univariate statistic is not
sufficient. Use a multivariate method or monitor the model's output on that input instead.

### Step 5 — Set thresholds from the null distribution

**Input.** The statistic, the sample size, and the acceptable false alarm rate.

**Output.** A threshold with a stated false alarm rate at the reference sample size.

**Rationale.** A threshold chosen by convention has an unknown false alarm rate. A threshold
computed from the null distribution has a stated one. For PSI, the expected null value is
approximately (1/n + 1/m)(B − 1); for KS, the critical value at α is approximately
1.36 × sqrt((n + m)/(n × m)) for a two-sided test at 0.05.

The threshold should also include an effect-size floor. A statistically significant but tiny
change is not worth an investigation. The floor should be tied to the change that would matter for
calibration or performance.

Branch point: if the sample size varies by window, the threshold must vary or the statistic must be
standardized. A fixed PSI threshold applied to windows of different sizes will produce a varying
false alarm rate.

### Step 6 — Monitor the output distribution

**Input.** The prediction log with model version and threshold.

**Output.** Score distribution, decision rate, and entropy, segmented by version and site.

**Rationale.** Output monitoring requires no labels and detects changes that input monitoring
misses. A model can have stable inputs and a changed output if the model version changed, the
threshold changed, or the feature computation changed.

The output distribution is also a leading indicator of workload. A shift in the share of `review`
decisions changes the number of alerts the ward receives, even if accuracy is unchanged.

Branch point: if the output distribution changes but inputs are stable, the first hypothesis is a
model or configuration change, not drift.

### Step 7 — Monitor labels and prevalence when available

**Input.** The outcome data and its latency.

**Output.** Prevalence over time, and label distribution by subgroup and site.

**Rationale.** Label shift is common in clinical settings because case mix and seasonality change.
A change in prevalence changes the positive predictive value even when sensitivity and specificity
are unchanged. Monitoring prevalence is cheap and interpretable.

The label definition must be stable. If the coding or the outcome definition changes, the
prevalence change is an artifact of the definition, not a change in the population.

Branch point: if labels are unavailable, prevalence cannot be monitored directly. Use the
predicted prevalence as a proxy and state its limitation.

### Step 8 — Triage: is this an upstream change?

**Input.** The drift alert, the upstream change log, and the data-quality metrics.

**Output.** A classification: upstream change, statistical drift, or false alarm.

**Rationale.** The upstream hypothesis is cheap to test and often correct. A laboratory analyzer
change, an interface upgrade, a coding update, or a firmware release will produce a distribution
shift that looks like drift. Fixing the pipeline is the correct response; retraining is not.

The change log must be available to the monitoring team. If the laboratory does not tell the model
team about an analyzer change, the triage will fail.

Branch point: if there is no upstream change and the data-quality metrics are normal, the alert is
a candidate statistical drift. Proceed to diagnosis.

### Step 9 — Diagnose the drift type

**Input.** Feature distributions, label distributions, and performance on labelled data.

**Output.** One of: covariate shift, label shift, concept drift, mixed, or undetermined.

**Rationale.** The type determines the response. Covariate shift with stable calibration needs
monitoring or recalibration. Label shift needs recalibration or reweighting. Concept drift needs
retraining. A mixed pattern needs a staged response.

The diagnosis must use more than one signal. A PSI alert alone cannot distinguish a population
change from a measurement change. A performance drop alone cannot distinguish concept drift from
label shift.

Branch point: if the evidence is insufficient, the diagnosis is "undetermined" and the response is
to collect more evidence, not to retrain.

### Step 10 — Quantify the impact

**Input.** The diagnosis and the labelled evaluation data.

**Output.** A change in calibration, discrimination, subgroup performance, and expected clinical
consequence.

**Rationale.** A drift statistic measures the size of a distributional change, not its effect on
the decision. A large input shift can leave the model well calibrated; a small shift in a critical
feature can move many decisions.

The impact should be expressed in the units the decision owner uses: additional alerts per day,
missed events per month, or a change in positive predictive value at the operating threshold.

Branch point: if the impact is below the pre-specified harm threshold, the response may be to
monitor and document. If it is above, the response must be proportional.

### Step 11 — Choose the response

**Input.** The diagnosis and the impact.

**Output.** One of: monitor, fix the pipeline, recalibrate, reweight, retrain, retire.

**Rationale.** The response must follow the diagnosis. The table below is the core of this step.
It has more than two outcomes because the evidence pattern, not the size of the statistic,
determines the action.

| Evidence pattern | Diagnosis | Response | Owner | Change-control assessment |
|---|---|---|---|---|
| Upstream change in the same window | Operational | Fix the pipeline, backfill, re-verify | Data platform | Incident record; no model change |
| p(X) shifted, calibration and performance stable | Covariate, no harm | Monitor, document, update reference only if the new distribution is the new normal | Data science | Post-market surveillance note |
| p(X) shifted, calibration slope changed, discrimination stable | Covariate plus calibration | Recalibrate on recent labelled data | Data science | Model change; assess predetermined change control plan (PCCP) coverage |
| p(y) changed, p(X|y) stable | Label shift | Recalibrate intercept or reweight | Data science | Model change; assess PCCP coverage |
| p(y|X) changed on labelled data | Concept drift | Retrain and revalidate | Data science and clinical | New validation evidence; change control |
| Subgroup gap widened, aggregate stable | Population or measurement bias | Stratified investigation, local validation | Clinical and data science | Equity review; updated labelling |
| No measurable change, statistic above threshold | False alarm or sample-size artifact | Review the statistic, adjust the threshold, no model action | Monitoring owner | None |

Branch point: if the response is retraining, the retraining must be validated on a future window,
not a random split. Retraining on the drifted data and validating on the same data proves nothing.

### Step 12 — Execute, validate, and record

**Input.** The chosen response.

**Output.** A change record, a validation report, and an updated monitoring configuration.

**Rationale.** Every response except "monitor" is a change to the system. It needs a record, a
validation, and a rollback path. A recalibration that is not recorded is an unaudited model change.

The validation must show that the response fixed the problem. A recalibration that does not restore
calibration is a failed response, not a completed one.

Branch point: if the response does not restore performance, escalate to the next response in the
table. Do not repeat the same response with a larger parameter.

### Step 13 — Update the reference and close the loop

**Input.** The validated model and the new distribution.

**Output.** A new reference version and a closed monitoring ticket.

**Rationale.** If the new distribution is the new normal, the reference must be updated, or the
monitoring will continue to alert on a change that has already been addressed. If the distribution
is expected to revert, the reference should not be updated.

The decision to update the reference is a judgment about whether the change is permanent. It must
be recorded, because it changes the meaning of future alerts.

Branch point: if the change is seasonal, do not update the reference to the seasonal distribution.
Use a seasonal reference or a longer comparison window.

### 5.14 Edge cases the procedure must handle

**Schema drift.** A field changes type, unit, or name. This is an upstream change, not a
statistical drift. The data check must fail before the statistic is computed.

**Late data.** A window is incomplete because a source delivered late. The statistic will show a
shift that is an artifact of missing data. Compute on complete windows only, or weight for
incompleteness.

**Partial rollout.** Two model versions are live. Compute drift per version. An aggregate can show
a shift caused entirely by the mix of versions.

**Model and feature skew.** Training and serving compute a feature differently. The reference and
the current window are not comparable. Fix the parity before interpreting the statistic.

**Clock skew.** Event time and processing time disagree. A window defined on processing time will
include events from the wrong period. Use event time and monitor the offset.

**Quota exhaustion.** The monitoring job cannot fetch the full window. A partial fetch produces a
false drift signal. Fail the job rather than compute on partial data.

**Dependency rotation.** A credential expires and the feature pipeline serves cached data. The
distribution looks stable because it is the old data. Monitor freshness alongside drift.

**Regional outage.** A region is missing from the window. The aggregate distribution shifts
because the case mix changed, not because the model's population changed. Segment by region.

### 5.15 The drift investigation ticket

Every drift alert that is not dismissed as noise should produce a ticket with the same structure.
The structure is what prevents the investigation from becoming a fishing expedition.

| Field | Content | Example |
|---|---|---|
| Alert | Statistic, feature, window, reference version | PSI creatinine 0.23, current 1–31 March, reference v3 |
| Sample | Observations in reference and current windows | 5,000 and 4,820 |
| Change log check | Upstream changes in or before the window | Laboratory analyzer replacement, 12 March |
| Data-quality check | Missingness, freshness, schema, range | Missingness 0.4 percent, freshness p95 22 min |
| Distribution | Location, scale, shape, and the largest bin change | Mean −0.12 mg/dL, shape unchanged |
| Output | Score distribution and decision rate | Mean risk 0.205 to 0.168; review rate 12.4 to 8.6 percent |
| Performance | Calibration and discrimination on labelled data | Slope 0.88, AUROC 0.79 unchanged |
| Diagnosis | One of the five categories | Upstream measurement change, not population drift |
| Impact | Effect in decision units | About 190 fewer reviews per month at unchanged prevalence |
| Response | Chosen action and owner | Fix interface, backfill, recalibrate; data platform and data science |
| Change control | Whether the response is a model change | Recalibration is a model change; assess PCCP coverage |
| Closure | Validation evidence and reference update decision | Slope 0.99 after recalibration; reference not updated |

The ticket is also the record that an auditor or a reviewer will read. A ticket that says "PSI
high, retrained" and nothing else is not evidence of a controlled process.

A useful discipline is to require the diagnosis field to be filled before the response field. It
forces the investigator to separate observation from action, which is the difference between
monitoring and reflex.

### 5.16 What the ticket reveals about thresholds

The example ticket contains a detail that is easy to miss: the sample sizes are 5,000 and 4,820,
and the PSI is 0.23. Under the null, the expected PSI is about (1/5,000 + 1/4,820) × 9 = 0.0037.
The observed value is roughly sixty times the null expectation, which is far beyond sampling
noise. The alert is real by any standard.

Now suppose the same PSI of 0.23 had been computed on 200 observations in each window. The null
expectation would be about 0.09, and the observed value would be only two and a half times the
null. The alert would still be interesting, but it would not carry the same weight, and a single
outlier or a small bin could produce it.

The same lesson applies to the KS statistic. A D of 0.08 is significant at 5,000 per sample and
not significant at 200 per sample. The statistic is identical; the evidence is not. Recording the
sample size and the critical value turns a number into an argument.

The ticket must therefore record the sample size next to the statistic. A reviewer who sees only
the statistic cannot judge whether it is signal or noise.

## 6. Worked example

A health system monitors a deterioration model with four features. On 1 April, the PSI for
creatinine crosses the review threshold. The reference window is the validation cohort from
September, with 5,000 observations. The current window is March, with 4,820 observations. Both
windows use the same ten bin edges, frozen at validation time.

### 6.1 The bin proportions and the PSI

| Bin | Reference proportion | Current proportion | Difference | Contribution |
|---:|---:|---:|---:|---:|
| 1 | 0.100 | 0.205 | +0.105 | 0.0754 |
| 2 | 0.100 | 0.165 | +0.065 | 0.0326 |
| 3 | 0.100 | 0.128 | +0.028 | 0.0069 |
| 4 | 0.100 | 0.100 | 0.000 | 0.0000 |
| 5 | 0.100 | 0.082 | −0.018 | 0.0036 |
| 6 | 0.100 | 0.068 | −0.032 | 0.0123 |
| 7 | 0.100 | 0.058 | −0.042 | 0.0229 |
| 8 | 0.100 | 0.052 | −0.048 | 0.0314 |
| 9 | 0.100 | 0.047 | −0.053 | 0.0400 |
| 10 | 0.100 | 0.095 | −0.005 | 0.0003 |
| **Total** | **1.000** | **1.000** | **0.000** | **0.2254** |

The PSI is 0.225. The distribution has shifted toward the lower bins, which is consistent with a
reduction in reported creatinine values. The shape is otherwise similar; the change is mostly a
location shift, not a change in dispersion.

### 6.2 The null expectation

With ten bins and these sample sizes, the expected PSI under the null is:

```text
E[PSI] ≈ (1/5,000 + 1/4,820) × (10 − 1)
       = (0.000200 + 0.000207) × 9
       = 0.00367
```

The observed 0.225 is about 61 times the null expectation. The alert is not a sampling artifact.
That conclusion is robust to the exact null distribution because the ratio is so large.

### 6.3 The sample-size trap

Suppose the same statistic had been computed on 200 observations in each window. The null
expectation would be:

```text
E[PSI] ≈ (1/200 + 1/200) × 9 = 0.09
```

A threshold of 0.1 would sit barely above the null expectation. Under the chi-square approximation
with nine degrees of freedom, a PSI of 0.1 at this sample size corresponds to a chi-square value of
0.1 / 0.01 = 10, which has a tail probability of about 0.35. A simulation with these bins gives 0.39. In other
words, a fixed 0.1 threshold would fire on roughly two-fifths of windows in which nothing had
changed.

This is the reason a threshold must be stated with its sample size. The same number, 0.1, is
conservative at 5,000 per window and nearly useless at 200.

### 6.4 The KS cross-check

The two-sample KS statistic for creatinine is 0.08. The critical value at a two-sided 5 percent
level is approximately:

```text
D_critical = 1.36 × sqrt((n + m) / (n × m))
           = 1.36 × sqrt(9,820 / 24,100,000)
           = 1.36 × 0.0202
           = 0.0275
```

The observed 0.08 exceeds the critical value, so the KS test also rejects equality. If the sample
were 200 per window, the critical value would be 0.136, and the same 0.08 would not be
significant. The statistic is the same; the evidence is not.

### 6.5 The impact on decisions

The shift is small in absolute units, about 0.12 mg/dL, but it moves the model's output
substantially because creatinine has a large coefficient.

| Quantity | Before | After |
|---|---:|---:|
| Mean predicted risk | 0.205 | 0.168 |
| Observed prevalence | 0.210 | 0.210 |
| Calibration slope | 1.00 | 0.88 |
| Review rate | 12.4 percent | 8.6 percent |
| Alerts per day | 143 | 99 |
| Sensitivity at the threshold | 0.63 | 0.60 |
| Missed events per month | 453 | 490 |

The model is now under-confident. It flags fewer patients, misses about 37 more events per month,
and produces 44 fewer reviews per day. The review reduction looks like a workload benefit and is
actually a loss of sensitivity.

### 6.6 Diagnostic one: the upstream change

The first diagnostic is the laboratory change log. A chemistry analyzer was replaced on 12 March,
and the new analyzer reports creatinine with a mean bias of −0.12 mg/dL against the old assay. The
distribution shape is unchanged, and the missingness rate is stable at 0.4 percent.

The second diagnostic is a comparison of paired samples measured on both analyzers during the
validation period. The mean difference is −0.118 mg/dL with a standard deviation of 0.04, which
matches the observed shift.

The diagnosis is an upstream measurement change, not a population change. The correct response is
to fix the data contract (record the analyzer and assay version), decide whether to harmonize the
values or to recalibrate the model, and backfill where possible. Retraining on the new analyzer
values without harmonizing would make the model correct for the new assay and wrong for any site
still using the old one.

### 6.7 Diagnostic two: quiet PSI, falling performance

In a different quarter, all feature PSIs are below 0.07, but sensitivity on the monthly audit falls
from 0.63 to 0.55. The input distributions are stable. The output distribution has shifted only
slightly.

The first diagnostic is calibration by month, which shows a slope of 0.93 and an intercept shift.
The second is a stratified analysis by discharge pathway. A new early-discharge protocol was
introduced, and the patients discharged under it have a different relationship between the
features and the outcome. The model's inputs are the same; the outcome process changed.

The diagnosis is concept drift. The response is retraining on data that includes the new pathway,
with a validation that holds out the post-protocol period. Recalibration alone would not fix a
change in the relationship.

### 6.8 Response economics

The two responses have different costs:

| Response | Cost | Time | Evidence required |
|---|---:|---|---|
| Recalibration | 2,930 USD | 3 days | Calibration restored on a recent labelled cohort |
| Retrain and revalidate | 28,500 USD | 6 weeks | New validation on a future window |

Recalibration is about one-tenth the cost and one-fourteenth the time. It is the correct response
to a measurement shift that changes the intercept but not the ranking. Retraining is the correct
response to a relationship change. Choosing retraining for every drift alert multiplies cost by ten
and delays the fix by six weeks.

If the team retrains on 12 drift alerts per year and 40 percent of them are measurement shifts that
recalibration would have fixed, the unnecessary retraining cost is:

```text
0.40 × 12 × 28,500 = 136,800 USD per year
```

That is the price of treating drift as one phenomenon.

### 6.9 Vary one assumption: the reference window

Suppose the reference had been built from 200 observations rather than 5,000. The null PSI would
rise to 0.09, and the 0.1 threshold would fire on about a third of unchanged windows. The team
would see frequent alerts, investigate them, find nothing, and gradually learn to ignore the
signal.

The response changes in two ways. First, the threshold must be derived from the null distribution
at the actual sample size, not copied. Second, the reference should be enlarged by pooling
historical windows, provided the pooling does not mix pre- and post-change populations. A reference
of 5,000 is not inherently better than 200; it is better because it makes the null distribution
tight enough to distinguish signal from noise.

### 6.10 What the example establishes

The same statistic, 0.225, is strong evidence at 5,000 observations and weak evidence at 200. The
diagnosis, not the statistic, determined the response. The upstream change was found in a log, not
in the data. The concept drift was found in the labels, not in the inputs.

A drift programme that monitors only input statistics would have caught the first case and missed
the second. A programme that monitors only performance would have caught the second and been slow
on the first. The two layers are complementary, and the diagnosis is the work.

## 7. Code, query, or calculation

The script implements PSI from proportions and from counts, the null expectation, the KS critical
value, and a simulation of the false alarm rate at a small sample size. It uses only the standard
library, so it can run wherever the monitoring job runs.

```python
"""PSI, null expectation, and KS critical values. Standard library only."""

import math
import random

REF_PROP = [0.10] * 10
CUR_PROP = [0.205, 0.165, 0.128, 0.100, 0.082, 0.068, 0.058, 0.052, 0.047, 0.095]


def psi_from_props(p, q):
    return sum((qi - pi) * math.log(qi / pi) for pi, qi in zip(p, q))


def psi_from_counts(ref, cur):
    n, m = sum(ref), sum(cur)
    p = [c / n for c in ref]
    q = [c / m for c in cur]
    return psi_from_props(p, q)


def null_expected_psi(n, m, bins):
    return (1 / n + 1 / m) * (bins - 1)


def ks_critical(n, m, alpha_factor=1.36):
    return alpha_factor * math.sqrt((n + m) / (n * m))


def false_alarm_rate(n, bins=10, threshold=0.1, reps=400, seed=11):
    rng = random.Random(seed)
    weights = [1 / bins] * bins
    hits = 0
    for _ in range(reps):
        ref = [0] * bins
        cur = [0] * bins
        for _ in range(n):
            ref[rng.choices(range(bins), weights)[0]] += 1
            cur[rng.choices(range(bins), weights)[0]] += 1
        if psi_from_counts(ref, cur) > threshold:
            hits += 1
    return hits / reps


big_n, big_m = 5_000, 4_820
psi = psi_from_props(REF_PROP, CUR_PROP)
null_big = null_expected_psi(big_n, big_m, 10)
print(f"psi                     {psi:.4f}")
print(f"null_expected_big       {null_big:.4f}")
print(f"ratio_to_null           {psi / null_big:.1f}")
print(f"ks_critical_5000_4820   {ks_critical(big_n, big_m):.4f}")
print(f"ks_critical_200_200     {ks_critical(200, 200):.4f}")
print(f"null_expected_200       {null_expected_psi(200, 200, 10):.4f}")

chi2_95_df9 = 16.92
psi_crit_200 = chi2_95_df9 * (1 / 200 + 1 / 200)
print(f"psi_crit_5pct_200       {psi_crit_200:.4f}")
print(f"false_alarm_rate_200    {false_alarm_rate(200):.2f}")

assert abs(psi - 0.2254) < 0.001
assert abs(null_big - 0.00367) < 0.0001
assert abs(ks_critical(big_n, big_m) - 0.0275) < 0.0005
assert abs(ks_critical(200, 200) - 0.1360) < 0.0005
```

Expected output, confirmed on Python 3.11:

```text
psi                     0.2253
null_expected_big       0.0037
ratio_to_null           61.4
ks_critical_5000_4820   0.0275
ks_critical_200_200     0.1360
null_expected_200       0.0900
psi_crit_5pct_200       0.1692
false_alarm_rate_200    0.39
```

The PSI computed from proportions is 0.2253, which matches the table in Section 6.1 to rounding.
The ratio to the null expectation is 61.4, which is the strongest evidence in the output.

The KS critical values show the sample-size effect directly. At 5,000 and 4,820 observations, a D
of 0.0275 is enough to reject equality at 5 percent. At 200 and 200, the threshold rises to
0.1360. A D of 0.08 would be significant in the first case and not in the second.

The `psi_crit_5pct_200` value is the PSI threshold that would give a 5 percent false alarm rate at
200 observations per window, using the chi-square approximation with nine degrees of freedom. It is
0.1692, well above the conventional 0.1. The simulation gives a false alarm rate of 0.39 at the
conventional threshold, which is close to the chi-square tail estimate of about 0.35 and different
because of the discrete bins and the finite simulation.

The practical rule is: compute the threshold from the null distribution at the actual sample size,
or report the false alarm rate alongside the threshold. A threshold of 0.1 without a sample size is
a number, not a control.

The lab's `monitoring/drift.py` computes PSI for one numeric feature with ten quantile bins and
prints `review` when the value is at least 0.2. It does not record the sample size, subtract the
null expectation, or compute a KS cross-check. Those omissions are acceptable for a teaching script
and would be defects in a production monitor.

## 8. Data, standards, and regulatory context

Drift is a statistical concept with regulatory consequences, not a regulated quantity. No binding
instrument specifies a drift statistic or a threshold. The instruments specify that change must be
detected, assessed, and controlled.

### 8.1 The instruments

| Instrument | Status | What it requires about drift | What it does not specify |
|---|---|---|---|
| FDA GMLP principle 10, 2021 | Non-binding guidance | Monitor deployed models for performance and manage retraining risks, including dataset drift and staleness | A statistic, threshold, or cadence |
| FDA AI-enabled device software functions guidance, draft January 2025 | Draft | A postmarket performance monitoring plan that addresses drift and staleness | The statistical method |
| FDA PCCP guidance, 2024, reissued 2025 | Final guidance implementing FD&C 515C | The retraining and update protocol, including data management and performance evaluation | Drift detection methods |
| ISO 14971:2019 clause 10 | Binding when invoked | Monitor production and post-production information for changes affecting risk | Statistical thresholds |
| IEC 62304:2006/Amd 1:2015 | Binding when invoked | Change control and risk management for software changes | Drift monitoring design |
| ISO/IEC 42001:2023 | Voluntary, certifiable | Operational control and performance evaluation of AI systems | Metrics |
| NIST AI RMF 1.0, MEASURE and MANAGE | Voluntary | Measure AI risks and manage responses, including distribution shift | Thresholds |
| HIPAA Security Rule | Binding for covered entities | Protect electronic protected health information used in monitoring | De-identification method |

### 8.2 What the regulator actually asks

The regulatory question is not "what is your PSI threshold." It is "how do you know the device
still performs as intended, and what do you do when it does not." A drift statistic is one input to
that answer. The rest is the monitoring plan, the review cadence, the response procedure, the
change record, and the evidence that the response restored performance.

A team that answers with a PSI threshold and no response procedure has answered a different
question. A team that answers with a documented diagnosis tree, a response table, and records of
past investigations has answered the regulatory question, even if its statistic is not the one a
vendor would recommend.

### 8.3 Retraining under a PCCP

If retraining is automated or routine, the PCCP must describe the data management, the retraining
practice, the performance evaluation, and the update procedure. A drift-triggered retrain is within
the protocol only if the protocol anticipated drift triggers and specified the acceptance criteria.

A team that retrains whenever PSI exceeds 0.2, without a protocol that describes that trigger and
its evaluation, has an unauthorized change process. The pipeline may be technically sound and the
regulatory position unsound.

### 8.4 Privacy in drift monitoring

Drift monitoring uses feature values, which may be protected health information. The monitoring
dataset should be de-identified or aggregated before it leaves the clinical environment. Bin counts
and summary statistics are usually sufficient; individual records are not.

The exception is an audit sample used to recompute performance, which requires labels and
identifiers to link predictions to outcomes. That sample should be handled under the same controls
as the clinical data, with access limited and retention defined.

### 8.5 What is convention

- PSI thresholds of 0.1 and 0.25. They come from credit scoring practice and have no universal
  statistical justification.
- Ten bins. The number of bins affects the statistic and the null distribution.
- A monthly drift review. The cadence should follow the data volume and the label latency.
- A single statistic per feature. Different feature types need different statistics.
- A vendor's default method. Defaults are starting points, not validated thresholds.

> **Note:** No regulator requires PSI. A plan that monitors calibration and performance directly,
with a documented response, may be stronger than one that monitors PSI and hopes it correlates
with harm.

## 9. Industry practice and case evidence

### 9.1 The concept drift taxonomy

Gama, Žliobaitė, Bifet, Pechenizkiy, and Bouchachia published a survey of concept drift adaptation
in ACM Computing Surveys in 2014. It defines concept drift as a change in p(y|X), distinguishes real
drift from virtual drift, characterizes patterns as sudden, incremental, gradual, and recurring,
and describes detection methods including sequential analysis and two-window distribution
comparison.

The documented taxonomy is the basis for the diagnosis tree in Section 3. The inference for
clinical monitoring is that the pattern matters as much as the magnitude: a recurring seasonal
shift should be modeled, not retrained away, and a sudden shift should be investigated
operationally before it is treated as concept drift.

### 9.2 Label shift and black box shift estimation

Lipton, Wang, and Smola published a method for detecting and correcting label shift in 2018. The
method estimates the test-time label distribution from the predictor's outputs and a confusion
matrix estimated on held-out data, without retraining the predictor. It also provides a detection
test based on the predictor's output distribution.

The documented contribution is a way to separate label shift from concept drift using only a
black-box predictor. The inference for clinical monitoring is that a change in prevalence can be
estimated and corrected without retraining, which is cheaper and faster than a full model update.
The method assumes that p(X|y) is stable, which is exactly the assumption that must be checked
before applying it.

### 9.3 The statistical properties of PSI

Yurdakul and Naranjo's 2020 paper in the Journal of Risk Model Validation shows that PSI's
distribution depends on sample size and bin count, and that the conventional thresholds have no
known Type I or Type II error properties. They derive the null distribution and propose critical
values that depend on sample size.

The documented result is the basis for Section 3.3 and Section 7. The inference is that a PSI
threshold copied from a vendor or a blog is not a statistical control. It becomes one only when it
is paired with a sample size and a false alarm rate.

### 9.4 Data validation at scale

Breck and colleagues' 2019 paper describes production data validation with schema checks,
distribution checks, and training-serving skew detection. They note that distribution distance
metrics can produce false positives at large sample sizes and choose L-infinity and L1 distances
over chi-squared or KL divergence for their setting.

The documented lesson is that the choice of statistic is an engineering decision with statistical
consequences. The inference for clinical drift monitoring is that the statistic should be chosen
for the sample size and the failure mode, not for familiarity.

### 9.5 Vendor defaults

Evidently documents PSI with a default threshold of 0.1, Wasserstein distance with 0.1, and KS with
a p-value threshold of 0.05. It also documents automatic method selection based on the reference
sample size. These are documented defaults, not clinical evidence.

The inference is that a team adopting the library should record the defaults it uses, measure the
false alarm rate on its own data, and adjust. A default that produces frequent false alarms in a
clinical service is not a control.

### 9.6 What the evidence does not show

There is no public study that estimates the clinical harm avoided by drift monitoring, or the rate
of unnecessary retraining triggered by drift alerts. The statistical literature establishes
properties of the statistics; the operational literature establishes that drift happens. The link
between the two is local.

A team should record every drift alert, its diagnosis, its response, and its outcome. Over a year,
that record is the evidence for the programme, and it is more useful than a benchmark from another
industry.

The same gap applies to thresholds. A team can find published advice to use 0.1, 0.2, or 0.25,
and no published evidence that any of those values produces an acceptable false alarm rate in a
clinical cohort of its size. The null-distribution calculation is the closest thing to evidence,
and it is available to anyone with the sample size and the bin count.

A threshold is a decision rule, and a decision rule without a stated error rate is incomplete.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Upstream change treated as concept drift | A sudden shift with an unchanged shape | The change log was not checked | Retraining embeds a measurement defect | Triage upstream first; require the change-log field in the ticket |
| Fixed threshold at small n | Frequent alerts with no performance change | Null PSI rises as sample size falls | Alert fatigue and unnecessary work | Compute the threshold from the null distribution |
| p-value without effect size | A significant test with a trivial difference | Large samples make small differences significant | Investigations with no clinical consequence | Report the effect size and the decision impact |
| Reference updated silently | Trends reset without a record | A well-meaning analyst refreshes the baseline | Historical comparisons become invalid | Version the reference; record every update |
| Reference mixes pre- and post-change data | The drift signal weakens over time | Pooling across a known change | A real change is hidden | Build references from a defined cohort |
| Every feature monitored | Dozens of alerts per week | No prioritization by importance or risk | Noise and ignored signals | Prioritize by importance, risk, and required features |
| Auto-retrain on drift | Model versions change without review | The pipeline treats PSI as a trigger | Unauthorized changes and churn | Human decision gate; PCCP coverage check |
| Input-only monitoring | PSI stable while sensitivity falls | Concept drift is invisible in p(X) | Late detection of harm | Add performance monitoring when labels permit |
| Performance-only monitoring | No labels, no detection | Labels arrive late or never | Months of undetected degradation | Add proxies and input monitoring |
| Label definition change | Prevalence shifts abruptly | Coding or criteria changed | Misdiagnosis as label shift | Version the outcome definition |
| Seasonal drift retrained away | A recurring pattern each winter | The model learns the season, not the relationship | Churn and no benefit | Use seasonal references or calendar features |
| Aggregate hides subgroup drift | Overall PSI stable, one group shifts | No stratified monitoring | Equity harm | Stratify by site and subgroup |
| Missingness treated as drift | A feature distribution shifts because values are missing | Imputation changes the observed distribution | Wrong diagnosis | Check missingness before the statistic |
| Retrain validated on the same drift | Validation metric looks good | Random split of the drifted cohort | No evidence the fix generalizes | Validate on a future window |

Red flags: a drift dashboard with no diagnosis field; a retrain job triggered directly by a
statistic; a reference distribution with no version; a PSI threshold quoted without a sample size;
a drift review that never records a "no action" decision.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The reference represents the validated population | Drift is measured against it | The reference mixes eras or sites | Compare reference cohort to the validation report |
| Bins are frozen at reference time | PSI is comparable across windows | Bins recomputed on current data | Store bin edges with the reference version |
| The current window is complete | Statistics reflect the population | Late data, partial fetch | Completeness check before computing |
| Sample sizes are sufficient | The null distribution is tight | Small sites, rare features | Compute the null expectation and report it |
| The outcome definition is stable | Label shift is interpretable | Coding or criteria change | Version the outcome definition |
| Upstream changes are visible to the monitoring team | Triage depends on it | No change log, no notification | Contract clause and a shared calendar |
| Calibration and performance are measurable | Harm requires them | Labels absent or delayed | State the proxy and the audit cadence |
| The drift statistic is appropriate to the feature type | Different statistics measure different changes | PSI on a categorical feature with many rare levels | Validate the statistic on known synthetic shifts |
| The response is proportional to the impact | Cost and risk are balanced | Reflexive retraining | Review response cost against benefit |

## 12. Exercises

### Exercise 12.1 — Classify four scenarios

Classify each as covariate shift, label shift, concept drift, or upstream change, and name the first
diagnostic: (a) a laboratory switches analyzers and creatinine falls by 0.1 mg/dL; (b) winter
admissions increase the prevalence of respiratory deterioration; (c) a new discharge protocol
changes the relationship between vital signs and readmission; (d) the age distribution shifts
older after a new nursing home contract.

**Deliverable.** A four-row table with classification and first diagnostic.

**Acceptance criteria.** (a) is distinguished from (d); (b) is distinguished from (c); the first
diagnostic is an action, not a statistic.

**Time.** 25 minutes.

### Exercise 12.2 — Compute PSI

A feature has ten equal reference bins. The current proportions are 0.18, 0.15, 0.12, 0.10, 0.09,
0.08, 0.07, 0.06, 0.05, 0.10. Compute the PSI to three decimal places and state which bins
contribute most.

**Deliverable.** The arithmetic and the top two contributing bins.

**Acceptance criteria.** Contributions sum to the total; the sign convention is consistent; the
interpretation names the direction of the shift.

**Time.** 30 minutes.

### Exercise 12.3 — Compute the null and the critical value

For PSI with ten bins and samples of 1,000 and 800, compute the expected null PSI. For a two-sample
KS test at a two-sided 5 percent level, compute the critical value.

**Deliverable.** Two numbers with the formulas.

**Acceptance criteria.** Null expectation uses both sample sizes; KS critical uses the correct
form; units are clear.

**Time.** 25 minutes.

### Exercise 12.4 — Diagnose a drift alert

PSI for a key feature is 0.19 with 4,000 observations per window. Missingness is stable, the shape
is unchanged, and the mean has shifted by 0.4 standard deviations. The laboratory change log shows
a reagent lot change two weeks before the window. Calibration slope is 0.91 and AUROC is unchanged.
Diagnose and choose a response.

**Deliverable.** A 350-word diagnosis with a response and a change-control note.

**Acceptance criteria.** Identifies the upstream change; distinguishes calibration from
discrimination; does not recommend retraining; states whether recalibration is a model change.

**Time.** 35 minutes.

### Exercise 12.5 — Design a reference policy

Write the policy for when a reference distribution is created, updated, archived, and compared.
Include the retraining case, the seasonal case, and the site-specific case.

**Deliverable.** A one-page policy with a decision table.

**Acceptance criteria.** Every update has an owner and a record; seasonal references are handled
explicitly; a retrain triggers a new reference.

**Time.** 40 minutes.

### Exercise 12.6 — Set a threshold from a false alarm target

Using the chi-square approximation with nine degrees of freedom, compute the PSI threshold that
gives a 5 percent false alarm rate at 300 observations per window. Compare it with the conventional
0.1 and state the implication.

**Deliverable.** The threshold and a two-sentence implication.

**Acceptance criteria.** Uses the correct scaling; compares with 0.1; states which threshold is
more conservative and why.

**Time.** 25 minutes.

### Exercise 12.7 — Write a drift ticket

Using the template in Section 5.15, write the ticket for the scenario in Exercise 12.4. Fill every
field, including the diagnosis, impact, response, and closure.

**Deliverable.** A completed ticket.

**Acceptance criteria.** Diagnosis precedes response; impact is in decision units; the change
control field is completed; the closure field states what evidence would close the ticket.

**Time.** 35 minutes.

### Exercise 12.8 — Design a response table

Write a response table for a model with four monitored features. Include at least six evidence
patterns, the diagnosis, the response, the owner, and the change-control assessment.

**Deliverable.** A six-row table.

**Acceptance criteria.** Responses are proportional to evidence; at least one row is "monitor and
document"; at least one row is "fix the pipeline"; retraining is not the default.

**Time.** 45 minutes.

## 13. Self-check answers

### 13.1 Exercise 12.1

(a) Upstream change. First diagnostic: the laboratory change log and a paired-sample comparison
across analyzers.

(b) Label shift. First diagnostic: prevalence over time, with the outcome definition held constant.

(c) Concept drift. First diagnostic: calibration and performance by pathway on labelled data.

(d) Covariate shift. First diagnostic: the age distribution by site and contract, with calibration
and performance checked for change.

### 13.2 Exercise 12.2

```text
bin 1: 0.08 × ln(1.8) = 0.0470
bin 2: 0.05 × ln(1.5) = 0.0203
bin 3: 0.02 × ln(1.2) = 0.0036
bin 4: 0.00
bin 5: −0.01 × ln(0.9) = 0.0011
bin 6: −0.02 × ln(0.8) = 0.0045
bin 7: −0.03 × ln(0.7) = 0.0107
bin 8: −0.04 × ln(0.6) = 0.0204
bin 9: −0.05 × ln(0.5) = 0.0347
bin 10: 0.00
total = 0.1423
```

The PSI is 0.142. The largest contributions are bin 1 (0.0470) and bin 9 (0.0347). The
distribution has shifted toward the lower bins, with a compensating loss in the upper-middle bins.

### 13.3 Exercise 12.3

```text
E[PSI] = (1/1,000 + 1/800) × 9 = (0.001 + 0.00125) × 9 = 0.02025
D_crit = 1.36 × sqrt((1,000 + 800) / (1,000 × 800))
       = 1.36 × sqrt(1,800 / 800,000)
       = 1.36 × 0.04743
       = 0.0645
```

The expected null PSI is 0.020. A PSI of 0.1 at this sample size is about five times the null
expectation, which is meaningful but not overwhelming. The KS critical value is 0.065.

### 13.4 Exercise 12.4

The pattern is an upstream measurement change: a reagent lot change, a location shift with
unchanged shape, stable missingness, and a calibration slope of 0.91 with unchanged AUROC. The
model's ranking is intact; its probability scale is shifted.

The response is to confirm the lot effect with a paired comparison, harmonize the values if
possible, and recalibrate on a recent labelled cohort if harmonization is not possible. Recalibration
is a model change and requires a change record and a PCCP coverage assessment.

Retraining is not appropriate because the relationship has not changed. Retraining on the new lot
would make the model correct for that lot and wrong for any site still using the old one.

### 13.5 Exercise 12.6

```text
threshold = 16.92 × (1/300 + 1/300) = 16.92 × 0.006667 = 0.1128
```

The conventional 0.1 is slightly more conservative than the 5 percent threshold at this sample
size. At 200 observations, the 5 percent threshold rises to 0.169, and 0.1 becomes too loose. At
5,000 observations, the 5 percent threshold falls to 0.0068, and 0.1 becomes very conservative.

### 13.6 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Diagnosis | 25% | Names a statistic | Distinguishes the four types | Uses the change log and multiple signals |
| Threshold | 25% | Copies a number | States the sample size | Derives it from the null and the cost |
| Response | 25% | Retrains | Chooses from evidence | Quantifies impact and cost, records the change |
| Record | 15% | Notes in a chat | Ticket with fields | Auditable record with closure evidence |
| Proportionality | 10% | Overreacts or ignores | Matches response to evidence | States the residual risk of "monitor" |

### 13.7 Exercise 12.7

A complete ticket states the statistic and sample sizes, the change-log finding, the distribution
and calibration evidence, the diagnosis (upstream reagent change), the impact (calibration slope
0.91, no AUROC change, small review-rate effect), the response (harmonize or recalibrate), the
change-control assessment (recalibration is a model change), and the closure criterion (slope
between 0.95 and 1.05 on a recent cohort).

## 14. Interview questions

**1. What is the difference between covariate shift and concept drift?**
Strong answers define p(X) versus p(y|X) and give an example where one occurs without the other.
Weak answers use "drift" for both.

**2. Why is PSI threshold of 0.1 not universal?**
Strong answers explain the sample-size and bin dependence and the null expectation. Weak answers
say it is a rule of thumb and move on.

**3. How do you distinguish an upstream change from population drift?**
Strong answers check the change log, the shape of the distribution, and paired measurements. Weak
answers retrain and see if it helps.

**4. When would you recalibrate instead of retrain?**
Strong answers cite stable discrimination with a changed calibration slope, often from prevalence
or measurement change. Weak answers say recalibration is cheaper.

**5. What does a p-value tell you about drift?**
Strong answers say it measures evidence against equality, not the size or consequence of the
change. Weak answers treat it as the probability of drift.

**6. How do you set a drift threshold?**
Strong answers derive it from the null distribution at the sample size, with an effect-size floor.
Weak answers copy a vendor default.

**7. What is label shift and how do you detect it?**
Strong answers define p(y) change with p(X|y) stable and describe prevalence monitoring or BBSE.
Weak answers confuse it with class imbalance.

**8. Why can a model drift without harm?**
Strong answers say the relationship may be unchanged in the new input region, and calibration may
hold. Weak answers equate drift with degradation.

**9. What is the first thing you check after a drift alert?**
Strong answers check data quality and the upstream change log before statistics. Weak answers
retrain.

**10. How do you validate a response to drift?**
Strong answers validate on a future window and compare calibration and subgroup performance. Weak
answers check the same data that triggered the alert.

## 15. Key terms

- **Backfill** — replacing missing or wrong historical values.
- **BBSE** — black box shift estimation of label shift from predictor outputs.
- **Change point** — a time at which a distribution changes.
- **Concept drift** — a change in the relationship between inputs and outcome.
- **Covariate shift** — a change in the input distribution with the relationship unchanged.
- **Detection delay** — time from drift onset to detection.
- **Domain adaptation** — methods for transferring a model across distributions.
- **Effect size** — the magnitude of a change, independent of sample size.
- **False alarm rate** — the fraction of alerts with no real change.
- **Gradual drift** — a slow trend over time.
- **Importance weighting** — reweighting samples to correct for a distribution change.
- **Incremental drift** — many small changes that accumulate.
- **Jensen-Shannon divergence** — a symmetric, bounded divergence between distributions.
- **KS statistic** — the maximum distance between two empirical CDFs.
- **Label shift** — a change in the outcome distribution with p(X|y) unchanged.
- **MMD** — maximum mean discrepancy, a kernel two-sample statistic.
- **Null distribution** — the distribution of a statistic when there is no change.
- **p-value** — the probability of the observed statistic under the null.
- **Population stability index** — a binned divergence between reference and current distributions.
- **Real drift** — a change that affects the decision boundary.
- **Recalibration** — adjusting the probability mapping without refitting.
- **Recurring drift** — a pattern that returns periodically.
- **Reference window** — the baseline distribution used for comparison.
- **Retraining** — fitting a new model on new or additional data.
- **Sudden drift** — a step change at a point in time.
- **Transient drift** — a spike that resolves.
- **Upstream change** — a change in measurement, coding, units, timing, or delivery.
- **Virtual drift** — a change in p(X) that does not affect the decision boundary.
- **Wasserstein distance** — the minimum cost of moving one distribution to another.

## 16. Sources

All URLs accessed 26 October 2026.

- Gama J, Žliobaitė I, Bifet A, Pechenizkiy M, Bouchachia A. A survey on concept drift adaptation. *ACM Computing Surveys*. 2014;46(4):44. https://doi.org/10.1145/2523813
- Lipton ZC, Wang Y-X, Smola A. Detecting and correcting for label shift with black box predictors. *Proceedings of ICML 2018*, PMLR 80:3122-3130. https://proceedings.mlr.press/v80/lipton18a.html
- Yurdakul B, Naranjo J. Statistical properties of the population stability index. *Journal of Risk Model Validation*. 2020;14(4):89-100. https://doi.org/10.21314/JRMV.2020.227
- Breck E, Polyzotis N, Roy S, Whang SE, Zinkevich M. Data validation for machine learning. *Proceedings of MLSys 2019*. https://proceedings.mlsys.org/paper_files/paper/2019/file/928f1160e52192e3e0017fb63ab65391-Paper.pdf
- SciPy. *scipy.stats.ks_2samp documentation*. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- Evidently AI. *Data drift documentation*. https://docs.evidentlyai.com/metrics/explainer_drift
- US Food and Drug Administration, Health Canada, and UK MHRA. *Good Machine Learning Practice for Medical Device Development: Guiding Principles*, 2021. https://www.fda.gov/media/153486/download
- US Food and Drug Administration. *Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations*, draft guidance, January 2025. https://www.fda.gov/media/184856/download
- US Food and Drug Administration. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, 2024, reissued 2025. https://www.fda.gov/media/166704/download
- ISO 14971:2019. *Medical devices — Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
- IEC 62304:2006/AMD1:2015. *Medical device software — Software life cycle processes*. https://webstore.iec.ch/en/publication/22794
- ISO/IEC 42001:2023. *Information technology — Artificial intelligence — Management system*. https://www.iso.org/standard/42001
- NIST. *AI Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, 2023. https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- 45 CFR 164.312, Technical safeguards. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Previous topic: [`05_monitoring.md`](05_monitoring.md)
- Next topic: [`07_incident-response.md`](07_incident-response.md)
- Related topic: [`01_the-production-loop.md`](01_the-production-loop.md)
- Related topic: [`09_model-governance.md`](09_model-governance.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- Drift script: [`../../labs/mlops/monitoring/drift.py`](../../labs/mlops/monitoring/drift.py)
- Prometheus rules: [`../../labs/mlops/monitoring/prometheus-rules.yaml`](../../labs/mlops/monitoring/prometheus-rules.yaml)
