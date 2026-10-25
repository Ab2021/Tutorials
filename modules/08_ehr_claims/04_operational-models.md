# Operational Models

> **Module 08 — EHR, Claims, and Health System Operations**
> Topic 04 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Topics 02 and 03
> Version 2026.3

## 1. Why this topic exists

Operational models predict events that happen often enough to measure and soon enough to act on: a missed appointment, a long length of stay, a readmission, a cancelled case, a shift that runs short of nurses. They are the workhorse models of health-system analytics, and they fail in ways that are different from clinical prediction models. The clinical model is judged on discrimination and calibration. The operational model is judged on whether the action it triggers changes the outcome within the capacity available.

Three failure patterns recur.

**The label does not match the operational event.** A no-show model trained on "appointment not attended" may include cancellations made in advance, patients rescheduled by the clinic, and appointments cancelled because the provider was unavailable. Those are different operational failures with different interventions. A model that predicts a mixture of them cannot be acted on: a reminder call may reduce forgetfulness but not a clinic-initiated reschedule. The label must be defined in the same language as the workflow, and every component must be separable.

**The model has no capacity.** A readmission model that flags 30 percent of discharges will overwhelm a transitional-care team that can manage 8 percent. The threshold must be derived from capacity, not from a statistical criterion. The operating point is a joint decision of the model, the intervention, and the staffing. A model with excellent discrimination and an impossible threshold is not deployable, and the failure is visible before go-live if the team computes alerts per day.

**The evaluation measures the model, not the decision.** An AUROC of 0.80 says the model ranks risk well. It does not say that the intervention reduced readmissions, that the reduction was worth the cost, or that the programme reached the patients who needed it. The evaluation must measure the outcome the programme intends to change, compared with a credible counterfactual, and it must include the operational metrics that determine whether the workflow worked: alerts reviewed, patients contacted, time to contact, and capacity used.

The evidence base is sobering. Kansagara et al. reviewed readmission prediction models in 2011 and found that most had modest discrimination and that few had been validated prospectively; the LACE index, one of the most widely used, had a derivation c-statistic of 0.711 and an external validation c-statistic of 0.684 in a million Ontario discharges, with later validations ranging from about 0.55 to 0.81 depending on the population. The model is useful for ranking, not for deciding. The intervention and the workflow produce the outcome.

The financial context makes the operational model consequential. The Hospital Readmissions Reduction Program (HRRP) reduces Medicare payments by up to 3 percent for hospitals with excess 30-day readmissions for six conditions: acute myocardial infarction, chronic obstructive pulmonary disease, heart failure, pneumonia, coronary artery bypass grafting, and hip or knee replacement. In FY2025, about 7 percent of hospitals faced a penalty of 1 percent or more, 71.6 percent faced a penalty under 1 percent, and 21.4 percent faced no penalty, according to preliminary CMS data reported by the Advisory Board. A model that shifts a hospital's readmission rate by a fraction of a percentage point can change a penalty that applies to all Medicare fee-for-service base operating DRG payments.

This topic teaches the operational model as a decision system: label, cohort, time zero, features, model, calibration, threshold, capacity, workflow, evaluation, and monitoring. The output is a model specification with an operating point and an evaluation plan, not a notebook with an AUROC.

> **Note:** Operational models predict a process outcome, not a clinical truth. A predicted no-show is a probability that a patient will miss an appointment, not a judgement that the patient is unreliable. A predicted readmission is a probability, not a preventable event. The language used to describe the output shapes how clinicians act on it.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** an operational label (no-show, length of stay, readmission, cancellation) with the same precision as the workflow that acts on it.
2. **Specify** time zero, the prediction horizon, and the feature window for an operational model, and **detect** temporal leakage.
3. **Choose** a model family (classification, regression, survival, forecasting, optimisation) for the decision and the outcome distribution.
4. **Derive** an operating threshold from intervention capacity and harm ratio, and **compute** alerts per day at that threshold.
5. **Evaluate** calibration and decision utility, not only discrimination, and **explain** why AUROC is insufficient.
6. **Design** an evaluation that separates the model's effect from the intervention's effect.
7. **Compute** cost per avoided event, breakeven effect, and the capacity required to deliver the programme.
8. **Identify** the operational edge cases that invalidate a model: missing appointments, late cancellations, transfers, planned readmissions, and mid-year workflow changes.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcomes 4, 5, and 7 are the ones that separate an operational analyst from a model builder.

## 3. The core mental model

An operational model is a loop, not a prediction. The loop starts with a decision, uses a prediction to rank patients, applies an action within capacity, and measures the outcome against a counterfactual.

```text
  DECISION        PREDICTION         ACTION
  who acts, on whom, -> probability at time  ->  contact, schedule,
  how often, with     zero for a horizon     reserve, escalate
  what capacity        |              |
    ^           v              v
    |         CALIBRATION         CAPACITY
    |         does p mean what      slots per day,
    |         it says?          staff per shift
    |           |              |
    |           v              v
  MONITORING <- EVALUATION <- OUTCOME <- WORKFLOW FIT
  drift, harm   counterfactual  measured   did the action
          comparison    at horizon  happen as designed
```

The loop has four properties that determine whether it works.

**Time zero is the decision point, not the event.** For a no-show model, time zero is the moment the reminder list is generated, typically 48 to 72 hours before the appointment. For a readmission model, time zero is the moment the discharge order is signed or the discharge planning huddle occurs. For a length-of-stay model, time zero is admission or the first post-operative day, depending on when the discharge team acts. If time zero is vague, the features are not frozen and the model cannot be deployed.

**The horizon determines the label and the action.** A no-show model predicts a single appointment. A readmission model predicts a 30-day window. A length-of-stay model predicts a discharge date or a probability of discharge by a given day. A staffing model forecasts census over a shift. The horizon must match the action: a 30-day readmission prediction shown at discharge is actionable; the same prediction shown on day 25 is not.

**The threshold is a capacity decision.** The number of patients the programme can manage per period, divided by the number of eligible patients, gives the flag rate. The threshold is the score at which the flag rate equals capacity. A threshold chosen without capacity is a preference, not an operating point. The model's calibration determines how many of the flagged patients actually have the event; the intervention's effect determines how many events are avoided.

**The counterfactual is the evaluation.** The outcome after the programme is not evidence that the programme worked. Patients who are flagged and enrolled may be sicker, more motivated, or more connected to care than those who are not. The evaluation needs a comparison group, a pre-specified outcome, and a design that handles selection: randomisation, a stepped wedge, a difference-in-differences with a credible control, or a regression discontinuity at the threshold.

A model family is chosen after the decision and the outcome distribution are known.

| Outcome | Distribution | Model family | Output | Evaluation |
|---|---|---|---|---|
| No-show | Binary per appointment | Logistic regression, gradient boosting | Probability | AUROC, calibration, show rate |
| Length of stay | Skewed positive, censored | Quantile regression, survival | Expected days, P(discharge by day) | MAE, quantile loss, bed-days |
| Readmission | Binary within 30 days | Logistic regression, gradient boosting | Probability | AUROC, calibration, net benefit |
| ED arrivals | Count time series | SARIMA, Prophet, gradient boosting | Hourly or daily count | MAE, MAPE, coverage |
| OR duration | Skewed positive | Quantile regression, ML | Minutes, quantiles | Pinball loss, schedule reliability |
| Staffing | Count with constraints | Forecasting plus optimisation | Required staff per shift | Fill rate, overtime, cost |

> **Note:** The model is the smallest part of the loop. In most operational programmes, the label definition, the capacity, and the workflow determine the result more than the choice of algorithm. A logistic regression with a well-defined label and a staffed workflow beats a gradient-boosted model with no capacity.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| No-show | A missed appointment without advance cancellation | Includes cancellations and clinic reschedules | Binary per appointment |
| Cancellation | An appointment cancelled before the scheduled time | Confused with no-show | Binary plus timing |
| Lead time | Days between booking and appointment | Ignored as a predictor | Days |
| Overbooking | Scheduling more appointments than slots | Applied without a safety limit | Extra slots per session |
| Show rate | Attended appointments divided by scheduled | Computed without cancellations | Proportion |
| Length of stay | Days from admission to discharge | Includes observation | Days |
| ALOS | Average length of stay | Sensitive to outliers | Days per discharge |
| Expected discharge date | Predicted date of discharge | Treated as a commitment | Date |
| Discharge barrier | A non-clinical reason for delay | Ignored in LOS models | Category |
| Readmission | Inpatient return within a window | Includes planned returns | Binary per index |
| Planned readmission | A readmission for a scheduled procedure | Counted as a failure | Binary flag |
| Index admission | The admission that starts the readmission window | Confused with the readmission | Admission ID |
| 30-day readmission | Readmission within 30 days of discharge | Window starts at admission | Days |
| HRRP | Hospital Readmissions Reduction Program | Treated as a quality score | Payment programme |
| Excess readmission ratio | Predicted divided by expected readmission rate | Confused with observed rate | Ratio |
| LACE | Length of stay, acuity, comorbidity, ED visits index | Treated as a complete model | Score |
| HOSPITAL score | A readmission risk score | Confused with LACE | Score |
| AUROC | Area under the receiver operating characteristic curve | Treated as decision utility | 0.5 to 1.0 |
| AUPRC | Area under the precision-recall curve | Compared across prevalences | 0 to 1 |
| Calibration | Agreement between predicted and observed rates | Ignored in operational models | Slope, intercept |
| Brier score | Mean squared error of predicted probabilities | Confused with discrimination | 0 to 1 |
| Threshold | Score above which the action is taken | Set at 0.5 by default | Probability |
| Capacity | Number of actions deliverable per period | Ignored until launch | Patients per day |
| Alert burden | Alerts per clinician per shift | Measured only after launch | Alerts per shift |
| Positive predictive value | Proportion of flagged patients with the event | Assumed constant across sites | Proportion |
| Number needed to treat | Patients who must receive the action for one to benefit | Reported without effect size | Count |
| Cost per avoided event | Programme cost divided by events avoided | Reported without the denominator | Currency per event |
| Breakeven effect | Relative effect at which benefits equal costs | Ignored in business cases | Percent |
| Counterfactual | The outcome that would have occurred without the action | Assumed to be the pre-period | Comparison group |
| Difference-in-differences | A design comparing change in treated and control groups | Used without parallel trends | Effect estimate |
| Interrupted time series | A design comparing pre and post trends | Used with too few points | Effect estimate |
| Stepped wedge | A design with staggered rollout | Confused with before-after | Effect estimate |
| Randomised quality improvement | Random assignment of the intervention | Confused with random sampling | Effect estimate |
| Drift | Change in the relationship between features and outcome | Detected only after harm | Monitoring metric |
| Census | Number of inpatients at a point in time | Confused with admissions | Patients |
| Occupancy | Occupied bed-days divided by staffed bed-days | Computed with licensed beds | Percent |
| Staffed bed | A bed that can be occupied with available staff | Confused with licensed bed | Count |
| Throughput | Rate of patient flow through a process | Confused with volume | Patients per time |
| Bottleneck | The constraint that limits throughput | Assumed to be the OR or ED | Process step |
| Queue | Patients waiting for a service | Ignored in capacity models | Count and wait time |
| Boarding | Holding admitted patients in the ED | Confused with ED crowding | Hours |
| OR utilisation | Used OR time divided by available OR time | Computed without turnover | Percent |
| Turnover time | Time between cases in an OR | Ignored in scheduling | Minutes |
| First-case on-time start | Proportion of first cases starting on time | Confused with OR utilisation | Percent |
| Case duration | Time from incision to closure or wheels-in to wheels-out | Confused with scheduled duration | Minutes |
| Block time | Reserved OR time for a surgeon or service | Treated as owned capacity | Hours |
| PACU | Post-anaesthesia care unit | Ignored in OR scheduling | Bed and staff |
| Forecast | A prediction of future demand | Confused with a target | Count per period |
| MAE | Mean absolute error | Reported without scale | Units |
| MAPE | Mean absolute percentage error | Undefined when actual is zero | Percent |
| Service level | Probability that demand is met | Confused with utilisation | Percent |
| Simulation | A model of system behaviour under scenarios | Confused with prediction | Scenario output |
| Discrete event simulation | A simulation of queues and resources | Treated as a forecast | Scenario output |

## 5. The mechanics, step by step

The procedure below builds an operational model as a decision system. It applies to no-show, length of stay, readmission, cancellation, and staffing models, with the label and the action changed. Each step names its input, output, and rationale.

### Step 1. State the decision, the actor, and the cadence

**Input:** a request such as "predict no-shows" or "reduce readmissions".

**Output:** one sentence: "At [cadence], [role] uses [prediction] to [action] for [population] within [capacity]."

**Rationale:** the cadence determines the feature freshness and the workflow. A daily no-show list at 48 hours requires features available two days before the appointment. A discharge huddle at 8 a.m. requires features available overnight. A weekly readmission review requires a different cadence and a different action. The actor determines whether the prediction will be used: a scheduler with 20 calls per shift can act on 20 flags, not 200.

Write the sentence before modelling. If the actor cannot be named, the model has no user, and the project will produce a report rather than a change.

**Branch point:** if the request is for a dashboard rather than an action, the deliverable is a monitoring product. Do not build a prediction model to support a dashboard that nobody acts on.

### Step 2. Define the operational label

**Input:** the workflow and the event to be predicted.

**Output:** a label definition with inclusion, exclusion, timing, and data source.

**Rationale:** the label is the most consequential choice. A no-show label must separate no-shows from cancellations, clinic reschedules, and provider cancellations. A readmission label must exclude planned readmissions and transfers. A length-of-stay label must define whether observation days count and whether the clock starts at admission or at the inpatient order. A cancellation label must distinguish patient cancellation from clinic cancellation.

A useful test is to ask whether the action could change the label. A reminder call can reduce a forgetful no-show; it cannot reduce a clinic-initiated reschedule. If the label mixes both, the model's effect will be diluted and the evaluation will be uninterpretable.

**Branch point:** if the data cannot separate the components, the label is not ready. Either find a source that separates them or narrow the label to the component the action can influence.

### Step 3. Define the cohort and the eligible population

**Input:** the label and the workflow.

**Output:** the eligible population with inclusion, exclusion, and a flow diagram.

**Rationale:** the cohort determines who can be flagged. A no-show model is built on scheduled appointments that are still active at the time of the reminder. Appointments already cancelled, rescheduled, or completed before the reminder are not eligible. A readmission model is built on index discharges that survive to discharge, excluding deaths, transfers, and hospice. A length-of-stay model is built on admissions, with planned admissions and same-day discharges handled explicitly.

The eligible population at scoring time may differ from the cohort used for training. A model trained on all scheduled appointments but deployed only on appointments with a phone number has a different case mix. Report both populations and their event rates.

**Branch point:** if the eligible population changes at deployment, recalibrate on the deployment population. A model calibrated on a broader population will misstate probabilities for the narrower one.

### Step 4. Set time zero and the horizon

**Input:** the decision cadence and the action.

**Output:** a timestamp for time zero and a horizon in days or hours.

**Rationale:** time zero is the instant the prediction is made and the features are frozen. For a no-show model, it is the reminder-list generation time. For a readmission model, it is the discharge order or the discharge huddle. For a length-of-stay model, it is admission or the first post-operative day. The horizon is the period over which the outcome is ascertained: the appointment date for no-show, 30 days for readmission, the discharge date for length of stay.

Every feature must be available at or before time zero. A feature recorded after time zero is leakage. A discharge disposition recorded after discharge is not available at the discharge huddle; a follow-up appointment scheduled after discharge is not available at discharge. Test the feature set by asking, for each feature, whether a user standing at time zero could have seen it.

**Branch point:** if a feature is clinically important but recorded late, replace it with a proxy available at time zero or exclude it. A proxy may reduce performance but preserve deployability.

### Step 5. Specify features and their availability

**Input:** the data sources and the clinical workflow.

**Output:** a feature dictionary with name, source, window, transformation, and availability timestamp.

**Rationale:** operational models rely on features that are stable and available: prior utilisation, appointment history, lead time, insurance, distance, discharge disposition, comorbidity, and recent labs. The feature window must be anchored to time zero. "Prior admissions in the last 12 months" means the 12 months before time zero, not before the outcome. "Last HbA1c" means the most recent result before time zero, not the result during the follow-up.

Feature availability is a deployment constraint. A feature from a claims feed that lags 60 days is not available at a daily cadence. A feature from a health information exchange that is incomplete for 30 percent of patients will produce missing values at scoring time that were not present in training if the training data were curated differently.

**Branch point:** if a feature is missing for more than 20 percent of the scoring population, either impute with a documented method or exclude it. Missingness that is related to the outcome is a source of bias, not a nuisance.

### Step 6. Choose the training and validation split

**Input:** the cohort, time zero, and the deployment period.

**Output:** training, validation, and test sets with a temporal or site-based split.

**Rationale:** operational models are deployed into the future, so the validation must mimic the future. A random split overestimates performance when the data have temporal drift, seasonal patterns, or workflow changes. Use a temporal split: train on the earliest period, validate on the next, test on the most recent. For multi-site deployment, use a site-based split: train on some sites, test on others.

The split must respect the unit of analysis. For appointment-level models, split by appointment date, not by patient, unless the same patient appears multiple times and the model is intended for repeat use. For readmission, split by discharge date. For length of stay, split by admission date.

**Branch point:** if the data are too small for a temporal split, use cross-validation with time-series folds and state the limitation. Do not use a random split and call it validation.

### Step 7. Fit and calibrate the model

**Input:** the training data and the model family.

**Output:** a fitted model, calibration plot, and performance metrics on the validation set.

**Rationale:** discrimination and calibration answer different questions. AUROC says how well the model ranks; calibration says whether a predicted 30 percent means 30 percent. Operational decisions depend on calibration because the threshold is a probability. A model with AUROC 0.80 and poor calibration can flag the right patients in the wrong order at the threshold.

Fit a simple baseline first: logistic regression with a small feature set, or a rule based on prior utilisation. A complex model that does not beat the baseline is not worth the deployment cost. Report AUROC, AUPRC, Brier score, calibration slope and intercept, and the calibration plot. For length of stay, report MAE and quantile loss; for forecasting, report MAE and coverage of the prediction interval.

**Branch point:** if calibration is poor, recalibrate with isotonic regression or Platt scaling on the validation set, and re-evaluate on the test set. Do not calibrate on the test set.

### Step 8. Set the threshold from capacity

**Input:** the score distribution, the eligible population per period, and the intervention capacity.

**Output:** a threshold, a flag rate, and the number of flags per period.

**Rationale:** capacity is the number of actions the system can deliver. If the programme can manage 1,200 patients per year and 9,000 discharges are eligible, the flag rate is 13.3 percent. The threshold is the score at which 13.3 percent of discharges are flagged. The threshold is not 0.5, and it is not chosen to maximise F1.

Compute the alert burden at the threshold: flags per day, flags per clinician per shift, and the time required to review each flag. If the burden exceeds capacity, lower the flag rate or increase the staff. A threshold that produces 40 alerts per clinician per shift will be ignored.

**Branch point:** if the score distribution is concentrated and the threshold cannot achieve the desired flag rate without including low-risk patients, the model is not useful at this capacity. Either improve the model, broaden the intervention to a lower-risk group with a cheaper action, or accept a lower reach.

### Step 9. Compute the workflow load

**Input:** the flag rate, the action, and the staff time per action.

**Output:** staff hours per period, cost per period, and the gap against available capacity.

**Rationale:** a flag is not an action. Each flag requires review, contact, documentation, and follow-up. A nurse navigator may need 20 minutes for a review and 15 minutes for a call. At 1,200 flags per year, that is 700 hours, or about 0.4 FTE, before accounting for documentation and no-answer attempts. The workflow load is often the binding constraint, not the model.

Compute the load explicitly and compare it with the staff available. If the load exceeds capacity, the programme will either miss flags or work unpaid overtime. Both erode the effect.

**Branch point:** if the workflow load exceeds capacity, reduce the flag rate, simplify the action, or add staff. Do not deploy a model that the workflow cannot absorb.

### Step 10. Design the intervention and the comparison

**Input:** the action and the evaluation question.

**Output:** an intervention protocol and a comparison design.

**Rationale:** the model identifies; the intervention changes. The intervention must be specified: who calls, what they say, how many attempts, what resources they can offer, and what happens if the patient declines. The comparison must be credible: randomisation, a stepped wedge, a difference-in-differences with a control site, or a regression discontinuity at the threshold.

A pre-post comparison without a control is the weakest design. It confounds the intervention with secular trends, seasonality, and regression to the mean. If a randomised pilot is not feasible, use a comparison group and test the parallel-trends assumption.

**Branch point:** if the intervention is delivered to everyone above the threshold and no one below, a regression discontinuity at the threshold can estimate the effect, provided the score is continuous and the threshold is enforced.

### Step 11. Evaluate the outcome and equity

**Input:** the intervention data, the comparison data, and the outcome definition.

**Output:** an effect estimate with a confidence interval, a cost per avoided event, and subgroup results.

**Rationale:** the primary outcome should be the event the programme intends to change, not the model's AUROC. For a no-show programme, it is the show rate; for a readmission programme, it is the unplanned readmission rate; for a length-of-stay programme, it is bed-days or discharge by a target day. The effect estimate should be adjusted for the design and reported with uncertainty.

Equity is part of the evaluation. A programme that improves the average but widens the gap between groups has not succeeded. Report the outcome by race, ethnicity, language, payer, and area deprivation, and check whether the model's performance or the intervention's reach differs across groups.

**Branch point:** if the effect is concentrated in a subgroup, the programme may need a different action for the others. A single threshold and a single intervention may not fit a heterogeneous population.

### Step 12. Monitor drift, harm, and workflow

**Input:** the deployed model and the operational data.

**Output:** a monitoring plan with metrics, thresholds, and owners.

**Rationale:** operational models degrade when the population, the workflow, or the data change. Monitor the score distribution, the calibration, the flag rate, the action rate, and the outcome. A drift in the score distribution without a change in the outcome may be a data issue; a drift in calibration is a model issue; a drop in the action rate is a workflow issue.

Monitor harm as well as benefit. A no-show model that flags patients with access barriers and then restricts scheduling is harmful. A readmission model that flags patients who cannot be reached wastes capacity and may widen disparities. Define the harm metrics before launch and review them monthly.

**Branch point:** if calibration drifts beyond a pre-specified threshold, recalibrate or retrain. If the outcome worsens in a subgroup, pause the programme and investigate.

### Decision table: label definitions

| Outcome | Include | Exclude | Time zero | Horizon |
|---|---|---|---|---|
| No-show | Appointment not attended, no advance cancellation | Cancellations, clinic reschedules, provider cancellations | Reminder-list generation | Appointment date |
| Readmission | Unplanned inpatient return to any hospital | Planned returns, transfers, deaths | Discharge order | 30 days |
| Length of stay | Inpatient days from admission to discharge | Observation days, transfers out | Admission | Discharge |
| OR cancellation | Case cancelled after scheduling | Cancellations before the scheduling lock | Scheduling lock | Day of surgery |
| ED arrival | Treat-and-release and admitted visits | Transfers from other EDs | Arrival | Hour or day |

### Decision table: threshold and capacity

| Capacity per period | Eligible per period | Flag rate | Threshold rule | Alert burden |
|---|---:|---:|---|---|
| 1,200 per year | 9,000 per year | 13.3% | 86.7th percentile of score | About 4.8 per day |
| 3,000 per year | 9,000 per year | 33.3% | 66.7th percentile | About 11.5 per day |
| 20 per shift | 40 per shift | 50% | Median score | 20 per clinician |
| 100 per week | 2,000 per week | 5% | 95th percentile | 20 per weekday |

### Decision table: evaluation design

| Design | Counterfactual | Strength | Weakness | When to use |
|---|---|---|---|---|
| Pre-post, no control | The prior period | Simple | Confounded by trends | Never for a causal claim |
| Difference-in-differences | A control site or group | Handles fixed differences | Requires parallel trends | Natural rollout |
| Interrupted time series | The pre-intervention trend | Uses many periods | Requires 12+ points per segment | System-wide change |
| Stepped wedge | Randomised rollout order | Handles logistics | Requires many clusters | Multi-site rollout |
| Regression discontinuity | Patients just below the threshold | Strong local identification | Only at the threshold | Score-based eligibility |
| Randomised pilot | Randomised control | Strongest | Expensive, limited scale | High-stakes decision |

### Edge cases that invalidate an operational model

**Missing appointment data.** A patient who books, cancels, and rebooks may appear as one appointment or three, depending on the scheduling system. Duplicate or linked appointments distort the no-show denominator. Detect by counting appointments per patient and comparing to the scheduling log.

**Late cancellations.** A cancellation made two hours before the appointment is operationally a no-show for staffing purposes but is recorded as a cancellation. The label must define the cutoff. Detect by plotting cancellation timing and comparing the no-show rate under different cutoffs.

**Transfers and planned readmissions.** A transfer to another acute hospital is not a discharge, and a planned readmission is not a failure. Both distort the readmission label. Detect by applying a transfer flag and a planned-readmission algorithm, and report both the all-cause and unplanned rates.

**Mid-year workflow changes.** A new scheduling template, a reminder policy, or a discharge process changes the outcome and the features. A model trained before the change may be miscalibrated after it. Detect by marking workflow changes on the time series and revalidating after each change.

**Case-mix shifts.** A new service line, a closure, or a payer change alters the population. A model trained on the old population may rank the new one poorly. Detect by monitoring the score distribution and the calibration by site and service.

**Capacity changes.** A staffing reduction or an expansion changes the threshold and the workflow. Recompute the flag rate and the alert burden whenever capacity changes. A model that was well calibrated for 1,200 patients may be unusable at 600.

### Feature families for operational models

| Family | Examples | Availability at time zero | Leakage risk |
|---|---|---|---|
| Prior utilisation | Admissions, ED visits, no-shows | Yes, from claims or EHR | Low if windowed correctly |
| Appointment history | Lead time, prior cancellations, booking channel | Yes | Low |
| Demographics | Age, sex, language, distance | Yes | Low; check fairness |
| Insurance | Payer, plan, cost-sharing | Yes, from eligibility | Retroactive changes |
| Clinical | Comorbidity, recent labs, medications | Yes, if resulted | Results pending at time zero |
| Social | Area deprivation, transport, housing | Yes, if recorded | Missingness bias |
| Workflow | Provider, clinic, day of week, slot type | Yes | Low |
| Discharge | Disposition, services ordered, follow-up scheduled | Only if documented before time zero | High for post-discharge fields |

The leakage column is the one to review with the clinical team. A discharge disposition is available at discharge; a follow-up appointment scheduled after discharge is not. A medication list at discharge is available; a medication filled after discharge is not.

### Calibration and decision metrics

| Metric | What it answers | Target | When it misleads |
|---|---|---|---|
| AUROC | How well the model ranks | 0.70-0.85 typical | High AUROC with poor calibration |
| AUPRC | Precision-recall at low prevalence | Compare to baseline | Across different prevalences |
| Brier score | Probability accuracy | Lower is better | Scale-dependent |
| Calibration slope | Are predictions too extreme or too flat | 1.0 | Small samples |
| Calibration intercept | Are predictions systematically high or low | 0.0 | Case-mix shift |
| Sensitivity at threshold | Share of events flagged | Depends on capacity | Ignores false positives |
| PPV at threshold | Share of flags with the event | Depends on prevalence | Assumed constant across sites |
| Net benefit | Value of the decision rule | Positive vs treat-all | Requires a threshold range |

A model with AUROC 0.82 and a calibration slope of 1.4 overestimates risk at the high end, which is exactly where the threshold sits. The flag rate will be higher than planned and the PPV lower than expected. The calibration plot, not the AUROC, predicts the operational experience.

### Operating point arithmetic

```text
Eligible patients per period = E
Intervention capacity per period = C
Flag rate = C / E
Threshold = quantile of the score at 1 - (C / E)
Flags per day = C / days in the period
Flags per clinician per shift = C / (clinicians x shifts)
Expected events flagged = flags x PPV at threshold
Events avoided = expected events flagged x relative effect
Cost per avoided event = programme cost / events avoided
Breakeven effect = programme cost / (expected events x value per event)
```

The formulas are simple, and they expose the two numbers that decide the programme: the PPV at the threshold and the relative effect. A high PPV with a small effect can be worse than a lower PPV with a larger effect, because the cost per avoided event depends on both.

### Alert burden example

| Setting | Eligible per day | Capacity per day | Flag rate | Flags per clinician | Review minutes per flag | Hours per day |
|---|---:|---:|---:|---:|---:|---:|
| No-show calls | 180 | 40 | 22% | 20 | 6 | 4.0 |
| Readmission huddle | 30 | 8 | 27% | 8 | 15 | 2.0 |
| LOS review | 60 | 20 | 33% | 10 | 10 | 3.3 |
| Sepsis alert | 25 | 25 | 100% | 25 | 4 | 1.7 |

The table shows why a model that flags 100 percent of patients is not a model; it is a workflow. The sepsis row is included because alert fatigue is the dominant failure mode when the flag rate is not constrained.

### Model card fields

A deployable operational model has a model card with the following fields:

1. Intended use and decision.
2. Eligible population and exclusion criteria.
3. Time zero and horizon.
4. Label definition and data source.
5. Feature list with windows and availability.
6. Training period and validation design.
7. Discrimination and calibration at the operating threshold.
8. Threshold, flag rate, and capacity.
9. Subgroup performance.
10. Monitoring metrics and retraining triggers.

A model without fields 3, 4, and 8 cannot be deployed safely, because the user does not know when the prediction applies, what it predicts, or how many patients it will flag.

### Workflow mapping

| Step | Owner | Time | System | Failure mode |
|---|---|---|---|---|
| Score generation | Analytics | Overnight | Data warehouse | Late or missing data |
| List review | Care team | 15 minutes | EHR or spreadsheet | List not reviewed |
| Patient contact | Navigator | 10-20 minutes | Phone or portal | No answer, wrong number |
| Documentation | Navigator | 5 minutes | EHR | Free-text only |
| Escalation | Physician or social work | Variable | EHR or pager | No escalation path |
| Outcome capture | Analytics | Monthly | Registry | Loss to follow-up |

The workflow table is the intervention. A model that is not mapped to these steps is a score without an action.

### No-show specific considerations

No-show models have three properties that differ from readmission models. First, the outcome is known quickly, so the feedback loop is short and retraining can be frequent. Second, the intervention is cheap and scalable (a call, a text, a transport voucher), so a lower PPV may be acceptable. Third, the equity risk is high: the strongest predictors include prior no-shows, insurance, distance, and language, which are markers of access barriers. An intervention that restricts scheduling for flagged patients penalises the patients with the least access. The action should reduce barriers, not deny care.

A no-show model should report the show rate by subgroup and the reach of the intervention by subgroup. If the model flags a group that the intervention cannot reach, the programme will widen the gap it intended to close.

A final operational check is the threshold drift test. Compute the threshold on the validation period, then apply it to the deployment period without recalibration. If the flag rate changes by more than 20 percent, the score distribution has shifted and the operating point is no longer the one that was planned. The test is cheap, it uses data the team already has, and it catches the most common cause of alert fatigue after launch.

The operating point also has a political dimension. A threshold that flags 5 percent of patients will be accepted by clinicians; a threshold that flags 40 percent will be resisted, even if the model is accurate. The flag rate should be negotiated with the clinical owner before launch, documented in the model card, and revisited when capacity changes. A model that is technically deployable but clinically unacceptable is not deployed.

| Capacity scenario | Flag rate | Threshold | Expected PPV |
|---|---:|---|---:|
| Constrained | 5% | 95th percentile | High |
| Moderate | 15% | 85th percentile | Medium |
| Broad | 30% | 70th percentile | Lower |

The PPV falls as the flag rate rises because the marginal flagged patient is lower risk. That is arithmetic, not model failure, and it is why the programme's cost per avoided event rises at broader thresholds. The threshold table is therefore a budget table as much as a model table.

A threshold that cannot be staffed is a threshold that will be ignored.

## 6. Worked example

An accountable care organisation (ACO) with 60,000 attributed members is deciding whether to expand a transitional-care programme for high-risk discharges. The programme uses a 30-day readmission model and a nurse navigator team. The analytics team must compute the population rate, compare three clinics after risk adjustment, size the high-risk cohort against capacity, and estimate the cost per avoided readmission.

### Part A: population rate and risk adjustment

All 60,000 members are enrolled for 12 months, so member-months are 60,000 x 12 = 720,000. There were 9,000 index admissions and 1,260 readmissions within 30 days.

```text
Admissions per 1,000 member-years = 9,000 / 720,000 x 12,000 = 150.0
30-day readmission rate = 1,260 / 9,000 = 14.0%
Readmissions per 1,000 member-years = 1,260 / 720,000 x 12,000 = 21.0
```

Three primary care clinics serve the population. Crude admission rates would rank clinic A worst. Risk adjustment with the HCC score reverses the ranking.

| Clinic | Members | Member-months | Admissions | Crude per 1,000 | Average HCC risk | Expected per 1,000 | O/E |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 20,000 | 240,000 | 3,300 | 165.0 | 1.10 | 165.0 | 1.000 |
| B | 20,000 | 240,000 | 2,700 | 135.0 | 0.92 | 138.0 | 0.978 |
| C | 20,000 | 240,000 | 3,000 | 150.0 | 1.08 | 162.0 | 0.926 |
| Total | 60,000 | 720,000 | 9,000 | 150.0 | 1.033 | 155.0 | 0.968 |

The expected rate is the network average of 150.0 per 1,000 multiplied by the clinic risk score. Clinic C has the best O/E (0.926) despite a crude rate equal to the network average, because its patients are sicker than average. Clinic A is exactly at expectation. A programme that allocated navigators by crude rate would send the most resources to clinic A, which is not the worst performer after adjustment.

### Part B: high-risk cohort and capacity

The model scores every index discharge at the time the discharge order is signed. At a threshold of 0.25, 1,800 discharges per year are flagged, or 20 percent of the 9,000 index admissions. The observed readmission rate in the flagged group is 30 percent, so the expected readmissions without the programme are 1,800 x 0.30 = 540.

The navigator team can manage 1,200 patients per year. That is 66.7 percent of the flagged group (1,200 / 1,800). The programme enrols 1,200 patients, and their baseline readmission count is 1,200 x 0.30 = 360.

```text
Flag rate = 1,800 / 9,000 = 20.0%
Capacity = 1,200 per year
Reach = 1,200 / 1,800 = 66.7%
Baseline readmissions among enrolled = 1,200 x 0.30 = 360
```

The 600 flagged patients who cannot be enrolled are a source of inequity if selection is not random. The programme should report how the enrolled and unenrolled flagged patients differ, and whether the unenrolled group has a higher readmission rate.

### Part C: programme economics

The programme costs $480,000 per year: 4.0 FTE navigators at $105,000 fully loaded ($420,000), information technology and training ($40,000), and transportation and pharmacy vouchers ($20,000). The assumed relative reduction is 15 percent among enrolled patients.

```text
Avoided readmissions = 360 x 0.15 = 54
Cost per avoided readmission = $480,000 / 54 = $8,889
Allowed cost per readmission = $16,800
Shared-savings rate = 50%
Savings to the ACO = 54 x $16,800 x 0.50 = $453,600
Net result = $453,600 - $480,000 = -$26,400
Breakeven relative reduction = $480,000 / (360 x $16,800 x 0.50) = 15.87%
```

At a 15 percent effect, the programme loses $26,400 per year, and the breakeven effect is 15.87 percent, just above the assumed effect. The decision is close: a 1 percentage point improvement in the effect, or a 6 percent reduction in programme cost, would make the programme break even.

### Diagnostic 1: planned readmissions are counted

Suppose 168 of the 1,260 readmissions were planned, for example a staged procedure or a scheduled chemotherapy admission. The unplanned readmission rate is (1,260 - 168) / 9,000 = 1,092 / 9,000 = 12.13 percent, not 14.0 percent.

If the flagged group's 30 percent baseline includes a similar 10 percent planned share, the unplanned baseline is 27 percent. The expected unplanned readmissions among 1,200 enrolled patients are 324, not 360. At a 15 percent effect, avoided readmissions fall to 48.6, and the cost per avoided readmission rises to $9,877. Savings fall to $408,240, the net result becomes -$71,760, and the breakeven effect rises to 17.64 percent. The programme now looks clearly unfavourable.

The diagnostic is to apply a planned-readmission algorithm to the numerator and to report both all-cause and unplanned rates. An analysis that uses all-cause readmissions overstates the opportunity and understates the cost per avoided event.

### Diagnostic 2: the model is miscalibrated at deployment

The model was trained on a period when the flagged group's readmission rate was 30 percent. In the deployment year, the observed rate in the flagged group is 24 percent, because a prior discharge programme and a new follow-up clinic have already reduced readmissions. The score distribution is unchanged, but the calibration has drifted.

```text
Baseline readmissions among 1,200 enrolled = 1,200 x 0.24 = 288
Avoided at 15% = 43.2
Cost per avoided = $480,000 / 43.2 = $11,111
Savings = 43.2 x $16,800 x 0.50 = $362,880
Net = $362,880 - $480,000 = -$117,120
Breakeven = $480,000 / (288 x $16,800 x 0.50) = 19.84%
```

The programme's value depends on the baseline it faces, not on the baseline in the training data. A model that was well calibrated in development can be badly calibrated at deployment, and the error changes the decision. The diagnostic is to plot calibration by month and to recompute the expected events from the deployment period before the business case is finalized.

### Varying one assumption

The decision is sensitive to the effect size. If the navigator programme achieves a 20 percent relative reduction rather than 15 percent, the economics change:

```text
Avoided = 360 x 0.20 = 72
Savings = 72 x $16,800 x 0.50 = $604,800
Net = $604,800 - $480,000 = $124,800
Breakeven effect remains 15.87%
```

At a 20 percent effect, the programme returns $124,800 per year and the decision reverses from reject to fund. The breakeven effect does not change because it is a property of the cost and the baseline, not of the assumed effect. The question for the pilot is whether the effect is closer to 15 or 20 percent, and the only way to answer it is a prospective evaluation with a comparison group.

| Scenario | Baseline rate | Effect | Avoided | Savings | Net | Decision |
|---|---:|---:|---:|---:|---:|---|
| Base | 30% | 15% | 54.0 | $453,600 | -$26,400 | Do not fund |
| Planned excluded | 27% | 15% | 48.6 | $408,240 | -$71,760 | Do not fund |
| Calibration drift | 24% | 15% | 43.2 | $362,880 | -$117,120 | Do not fund |
| Stronger effect | 30% | 20% | 72.0 | $604,800 | $124,800 | Fund |
| Full risk | 30% | 15% | 54.0 | $907,200 | $427,200 | Fund |

The table is the deliverable. It shows that the programme is viable under a stronger effect or a fuller risk share, and not viable under the base case or the two diagnostics. The recommendation is a pilot with a pre-specified effect estimate, a comparison group, and a decision rule at 12 months.

### What the model does not decide

The model ranks discharges by risk. It does not decide who is enrolled, what the navigator does, or whether the readmission was preventable. Those are clinical and operational decisions. A high score is an invitation to review, not a diagnosis of preventable harm. The programme's effect depends on the intervention, and the evaluation must measure the intervention, not the score.

### Part D: capacity and alert burden

The annual numbers become operational only when they are converted to a daily workload. Assume 250 operating days per year.

```text
Flags per day = 1,800 / 250 = 7.2
Enrolled per day = 1,200 / 250 = 4.8
Navigator capacity = 4.0 FTE x 25 active patients x 12 episodes per year = 1,200
```

The navigator capacity is exactly the programme capacity, which is why the reach is 66.7 percent of the flagged group. Each navigator carries 25 active patients, and each episode lasts about 30 days, so each navigator completes 25 x 12 = 300 episodes per year. Four navigators complete 1,200. A fifth navigator would raise capacity to 1,500 and reach to 83.3 percent, at an additional $105,000; the marginal cost per additional enrolled patient would be $350, and the marginal cost per avoided readmission at a 15 percent effect would be $2,333. That is a better marginal buy than the first four navigators, because the fixed programme cost is already sunk.

The daily workload is 4.8 new enrollments and about 25 active patients across the team. A navigator who also carries a hospital caseload cannot absorb this without a reduction in other duties. The workflow load is a staffing decision, not a model parameter.

### Part E: model performance at the threshold

The model's validation results at the 0.25 threshold are:

| Metric | Value | Interpretation |
|---|---:|---|
| AUROC | 0.78 | Moderate discrimination |
| Calibration slope | 1.10 | Slightly overconfident at high risk |
| PPV at threshold | 0.30 | 30% of flagged patients are readmitted |
| Sensitivity at threshold | 0.429 | 42.9% of readmissions are flagged |
| Specificity at threshold | 0.837 | 83.7% of non-readmissions are not flagged |
| NPV at threshold | 0.900 | 90% of unflagged patients are not readmitted |

The sensitivity is 540 / 1,260 = 42.9 percent. The specificity is (7,200 - 720) / 7,740 = 83.7 percent. The NPV is 6,480 / 7,200 = 90.0 percent. The model misses 57.1 percent of readmissions, which is expected at a threshold set by capacity. A programme that wanted to catch more readmissions would need a lower threshold, more navigators, or a different intervention for the unflagged group.

### Part F: evaluation design

A randomised pilot is feasible if the navigator team has spare capacity, but it is not feasible at full capacity because randomising away a service that is already staffed is ethically and operationally difficult. A stepped wedge across clinics is a practical alternative: clinics are randomised to the start date of the programme, and the outcome is compared within clinics over time. A difference-in-differences with a matched comparison clinic is weaker but easier to run.

The primary outcome should be the unplanned 30-day readmission rate among enrolled patients, measured at 30 days after discharge. Secondary outcomes are the all-cause rate, the ED visit rate, patient-reported follow-up, and cost per avoided readmission. The analysis should be intention-to-treat, with the enrolled group defined by the flag and the offer, not by whether the patient accepted.

### Part G: what would change the recommendation

Three changes would move the recommendation from reject to fund without a stronger clinical effect. First, a fifth navigator would raise reach from 66.7 to 83.3 percent and increase avoided readmissions by 13.5 at the base case, adding about $113,400 of savings for $105,000 of cost, a marginal net of about $8,400. Second, a full-risk contract would double the ACO's share and produce the $427,200 net shown in the scenario table. Third, a reduction in the readmission cost estimate would not help; a higher allowed cost per readmission, such as $24,000, would raise savings to $648,000 at the base case and produce a net of $168,000.

The three levers are different in kind. The fifth navigator is an operational decision, the full-risk contract is a contracting decision, and the readmission cost is a measurement assumption. The analyst should present them separately, because the operating committee can act on the first two and can only validate the third.

The recommendation is to run a 12-month pilot with the fifth navigator, a pre-specified effect estimate of 15 percent, a stepped-wedge comparison, and a decision rule: continue if the observed effect is at least 15 percent and the net result is positive under the contract's risk share; stop if the effect is below 10 percent or the calibration drifts by more than 20 percent.

| Lever | Change | Effect on net | Owner |
|---|---|---:|---|
| Fifth navigator | Reach 66.7% to 83.3% | +$8,400 | Operations |
| Full-risk contract | Risk share 50% to 100% | +$453,600 | Contracting |
| Higher readmission cost | $16,800 to $24,000 | +$194,400 | Finance |

The levers are not additive in practice: a full-risk contract changes the value of the navigator and the readmission cost assumption, and the combination should be modelled as a contract scenario rather than as a sum of independent effects.

The scenario table is the basis for the pilot decision, not the base case alone.
It should be reviewed when the pilot data arrive.
The decision rule is the control.

## 7. Code, query, or calculation

The SQLite script below reproduces the population rate, clinic O/E, unplanned readmission rate, and scenario economics from Section 6.

Assumptions:

- All 60,000 members are enrolled for 12 months.
- Clinic counts are index admissions and 30-day readmissions.
- The unplanned rate uses 168 planned readmissions out of 1,260.
- Programme capacity is 1,200 enrolled patients per year, and the baseline rate is applied to the enrolled group.

```python
import sqlite3

con = sqlite3.connect(":memory:")
cur = con.cursor()
cur.executescript("""
CREATE TABLE clinics (
 clinic TEXT PRIMARY KEY, members INTEGER, member_months INTEGER,
 admissions INTEGER, readmissions INTEGER, risk REAL
);
INSERT INTO clinics VALUES
 ('A',20000,240000,3300,462,1.10),
 ('B',20000,240000,2700,378,0.92),
 ('C',20000,240000,3000,420,1.08);

CREATE TABLE planned (readmissions INTEGER, planned INTEGER);
INSERT INTO planned VALUES (1260,168);

CREATE TABLE program (
 capacity INTEGER, baseline_rate REAL, effect REAL,
 program_cost REAL, readmit_cost REAL, risk_share REAL
);
INSERT INTO program VALUES (1200,0.30,0.15,480000,16800,0.50);

CREATE TABLE scenarios (
 name TEXT, baseline_rate REAL, effect REAL, risk_share REAL
);
INSERT INTO scenarios VALUES
 ('Base',0.30,0.15,0.50),
 ('Planned excluded',0.27,0.15,0.50),
 ('Calibration drift',0.24,0.15,0.50),
 ('Stronger effect',0.30,0.20,0.50),
 ('Full risk',0.30,0.15,1.00);
""")

print("POPULATION RATE")
print(cur.execute("""
SELECT SUM(member_months), SUM(admissions), SUM(readmissions),
    ROUND(SUM(admissions)*12000.0/SUM(member_months),1),
    ROUND(100.0*SUM(readmissions)/SUM(admissions),2)
FROM clinics;
""").fetchone())

print("\nCLINIC OBSERVED/EXPECTED")
for row in cur.execute("""
SELECT clinic, ROUND(admissions*1000.0/members,1) AS crude,
    risk, ROUND(150.0*risk,1) AS expected,
    ROUND((admissions*1000.0/members)/(150.0*risk),3) AS oe
FROM clinics ORDER BY clinic;
"""):
  print(row)

print("\nUNPLANNED READMISSION RATE")
print(cur.execute("""
SELECT readmissions, planned, readmissions-planned AS unplanned,
    ROUND(100.0*(readmissions-planned)/9000,2)
FROM planned;
""").fetchone())

print("\nSCENARIO ECONOMICS")
for row in cur.execute("""
SELECT s.name, ROUND(1200*s.baseline_rate,1) AS baseline_readmits,
    ROUND(1200*s.baseline_rate*s.effect,1) AS avoided,
    ROUND(p.program_cost/(1200*s.baseline_rate*s.effect),0) AS cost_per_avoided,
    ROUND(1200*s.baseline_rate*s.effect*p.readmit_cost*s.risk_share,0) AS savings,
    ROUND(1200*s.baseline_rate*s.effect*p.readmit_cost*s.risk_share
       - p.program_cost,0) AS net,
    ROUND(p.program_cost/
       (1200*s.baseline_rate*p.readmit_cost*s.risk_share),4) AS breakeven
FROM scenarios s, program p;
"""):
  print(row)
con.close()
```

Expected output, produced by Python 3.11 with SQLite 3.50.4:

```text
POPULATION RATE
(720000, 9000, 1260, 150.0, 14.0)

CLINIC OBSERVED/EXPECTED
('A', 165.0, 1.1, 165.0, 1.0)
('B', 135.0, 0.92, 138.0, 0.978)
('C', 150.0, 1.08, 162.0, 0.926)

UNPLANNED READMISSION RATE
(1260, 168, 1092, 12.13)

SCENARIO ECONOMICS
('Base', 360.0, 54.0, 8889.0, 453600.0, -26400.0, 0.1587)
('Planned excluded', 324.0, 48.6, 9877.0, 408240.0, -71760.0, 0.1764)
('Calibration drift', 288.0, 43.2, 11111.0, 362880.0, -117120.0, 0.1984)
('Stronger effect', 360.0, 72.0, 6667.0, 604800.0, 124800.0, 0.1587)
('Full risk', 360.0, 54.0, 8889.0, 907200.0, 427200.0, 0.0794)
```

Read the output in three passes. The population row gives 720,000 member-months, 9,000 admissions, 1,260 readmissions, 150.0 admissions per 1,000 member-years, and a 14.0 percent readmission rate. The clinic rows show that clinic C has the best O/E (0.926) despite a crude rate equal to the network average. The scenario rows show that the base case loses $26,400, the planned-exclusion and calibration-drift scenarios lose more, and a stronger effect or a full-risk contract produces a positive net.

> **Note:** The scenario table applies the baseline rate to 1,200 enrolled patients in every row. That is an intention-to-treat framing: the programme offers the service to 1,200 patients and the effect applies to the offered group. A per-protocol analysis would use the patients who accepted, which may be a different and healthier group.

### Modelling the score distribution

The programme economics depends on the PPV at the threshold, which depends on the score distribution. A simple way to model the distribution is a beta distribution calibrated to the observed mean and the flag rate.

```python
import math
from statistics import NormalDist

def threshold_for_flag_rate(mean, sd, flag_rate):
  # Normal approximation; use the empirical quantile in production.
  return NormalDist(mean, sd).inv_cdf(1 - flag_rate)

mean_risk, sd_risk = 0.14, 0.12
for rate in (0.05, 0.10, 0.20, 0.30):
  t = threshold_for_flag_rate(mean_risk, sd_risk, rate)
  print(f"flag_rate={rate:.2f} threshold={t:.3f}")
```

The output is a set of thresholds for different flag rates:

```text
flag_rate=0.05 threshold=0.337
flag_rate=0.10 threshold=0.294
flag_rate=0.20 threshold=0.241
flag_rate=0.30 threshold=0.203
```

The normal approximation is a teaching device, not a production method. Readmission scores are skewed and bounded, so use the empirical quantile from the validation set. The point is that the threshold moves with capacity, and the PPV at each threshold must be computed from the validation data, not assumed.

### PostgreSQL dialect: computing the flag rate by day

The following block is **PostgreSQL** syntax. It computes the number of flags per day at a fixed threshold from a scoring table.

```sql
-- PostgreSQL 15+
SELECT score_date,
    COUNT(*) FILTER (WHERE score >= 0.25) AS flags,
    COUNT(*) AS eligible,
    ROUND(100.0 * COUNT(*) FILTER (WHERE score >= 0.25) / COUNT(*), 2) AS flag_rate
FROM readmission_scores
WHERE score_date >= DATE '2026-01-01'
GROUP BY score_date
ORDER BY score_date;
```

The `FILTER` clause is PostgreSQL-specific. In SQLite, use `SUM(CASE WHEN score >= 0.25 THEN 1 ELSE 0 END)`. The flag rate by day is the first operational metric to review after launch, because it shows whether the threshold is producing the workload that was planned.

## 8. Data, standards, and regulatory context

Operational models are governed less by transaction standards than by measure specifications and payment programmes. A readmission model that does not align with the HRRP definition will produce a number that cannot be compared with the penalty. A no-show model that does not align with the scheduling system's definition will produce a rate that cannot be reproduced.

### Measure and payment specifications

| Specification | What it defines | Binding status | What it does not define |
|---|---|---|---|
| HRRP | 30-day readmission for six conditions, excess readmission ratio, 3% penalty cap | Binding for eligible hospitals | The intervention or the model |
| CMS 30-day readmission measure | Index admission, planned readmission algorithm, cohort exclusions | Binding for CMS programmes | Local care processes |
| HEDIS Plan All-Cause Readmissions | 30-day readmission for commercial, Medicaid, Medicare | Binding for NCQA reporting | Clinical preventability |
| CMS Star Ratings | Measure set, weights, cut points | Binding for MA quality bonus | Local improvement strategy |
| LACE index | Length of stay, acuity, comorbidity, ED visits | A validated score, not a regulation | Intervention effect |
| HOSPITAL score | Seven predictors of readmission | A validated score | Local calibration |
| No-show definition | Local scheduling policy | Local | Cross-site comparability |

The HRRP uses an excess readmission ratio: predicted readmissions divided by expected readmissions, where expected is derived from a risk model and peer group. The penalty applies to Medicare fee-for-service base operating DRG payments and is capped at 3 percent. The six conditions are acute myocardial infarction, chronic obstructive pulmonary disease, heart failure, pneumonia, coronary artery bypass grafting, and hip or knee replacement. A hospital's readmission model may use a different cohort and a different outcome, but the penalty calculation uses the CMS specification. The two should be reconciled, not conflated.

The planned readmission algorithm is part of the CMS measure. It identifies readmissions that are planned or related to a scheduled procedure and excludes them from the numerator. A local analysis that counts all readmissions will overstate the rate and the opportunity, as the worked example shows. The algorithm is updated periodically; the version used must be recorded.

### Clinical and operational data standards

FHIR R4 and US Core provide the exchange resources for encounters, conditions, observations, and care plans. They are relevant to operational models because the features and the outcomes may be assembled from multiple systems. A readmission may be recorded in the EHR, in a health information exchange, or in claims; a no-show may be recorded in the scheduling system and the EHR. The model's label must specify which source is authoritative.

HL7 FHIR also supports Clinical Decision Support (CDS) Hooks, which allow a prediction to be delivered inside the clinician's workflow. The technical standard does not determine whether the prediction is a regulated device. The FDA's guidance on clinical decision support software distinguishes software that supports a clinician's independent judgement from software that directs a specific diagnosis or treatment. An operational model that ranks patients for outreach is usually not a device; a model that recommends a specific treatment is more likely to be regulated. The boundary is use-specific and should be reviewed with regulatory affairs.

### Privacy and equity

Operational models use identifiable data and must comply with HIPAA. A model that scores patients for outreach is a use for treatment or health care operations, which is permitted without authorization, but the minimum necessary standard applies. A model that is used for research requires IRB review.

Equity is a regulatory and ethical requirement in several contexts. CMS quality programmes stratify some measures by race, ethnicity, and dual-eligible status. The Joint Commission and NCQA accreditation standards address health equity. A no-show model that restricts access for patients with transport barriers may violate the organisation's equity policy even if it is legal. The model card should include a subgroup performance section and an equity impact statement.

> **Note:** Regulatory citations are dated to 25 October 2026. The HRRP parameters and the CMS measure specifications change with annual rules. Verify the current specification before using a measure definition in a payment calculation.


### Binding versus convention

| Element | Binding? | Source | Local choice |
|---|---|---|---|
| HRRP cohort | Yes, for eligible hospitals | CMS | None for the penalty calculation |
| Planned readmission algorithm | Yes, for CMS measures | CMS | Version used |
| No-show definition | No | Local policy | Cutoff and categories |
| Risk threshold | No | Local | Capacity and harm |
| Evaluation design | No | Local or funder | Comparison and outcome |
| Equity stratification | Programme-specific | CMS, NCQA, Joint Commission | Subgroups and actions |

### What the standards do not settle

1. **The intervention.** No measure specification says what to do with a flagged patient. That is the programme design.
2. **The threshold.** No standard sets the operating point. Capacity and harm determine it.
3. **The comparison.** No standard defines the counterfactual. The evaluation design does.
4. **The local label.** A no-show, a cancellation, and a reschedule are local definitions. The scheduling policy governs.
5. **The equity action.** A subgroup difference is a finding; what to do about it is a policy decision.

## 9. Industry practice and case evidence

### Documented evidence on readmission models

Kansagara et al. reviewed the readmission prediction literature in *JAMA* in 2011 and found that most models had modest discrimination and that few had been prospectively validated. The review is dated but its conclusion has held: the model ranks risk, and the intervention determines the outcome. DOI 10.1001/jama.2011.1515.

The LACE index is the most widely implemented readmission score. Van Walraven et al. reported a derivation c-statistic of 0.711 and an external validation c-statistic of 0.684 in about one million Ontario discharges, with good calibration. Later validations have ranged from about 0.55 to 0.81 depending on the population: a Danish validation of 19,277 patients reported 0.648 with poor calibration, a US community hospital study of 432 discharges reported 0.58, and a surgical cohort of 192,670 admissions reported 0.77 to 0.82. The variation is the point: a score validated in one population is a hypothesis in another.

The Kaiser Permanente Northern California Transitions Program, described in Topic 01, is the best-documented example of an analytics-targeted readmission programme at scale. Across 21 hospitals and 1.54 million discharges, patients with an EHR-based 30-day readmission or death risk of 25 percent or more were auto-enrolled for telephone case management. The difference-in-differences estimate was an adjusted odds ratio of 0.91 (95 percent CI 0.89 to 0.93), an absolute risk reduction of about 2.5 percentage points, and a number needed to treat of about 40. The programme also found that higher-risk patients benefited less, which is a warning against assuming that the model's top-ranked patients are the ones who benefit most.

The Geisinger post-discharge monitoring programme, also described in Topic 01, reported a 44 percent relative reduction in 30-day readmissions in a pre-post matched-control design, at a programme cost of about $25 per patient. The result combines monitoring and case management; it is not a model effect.

### Documented evidence on no-show models

Dantas et al. reviewed 105 studies of appointment no-shows and reported an average no-show rate of 23.0 percent, with a range of 4.0 to 79.2 percent. The most consistent predictors were long lead time and prior no-show history. The review supports the use of these features, and it also supports the equity warning: the predictors are markers of access barriers, not of patient character. DOI 10.1016/j.healthpol.2018.02.002.

Machine-learning no-show models have been reviewed in several systematic reviews. Carreras-García et al. reviewed 50 studies and found that prior no-shows and the number of prior appointments were the most significant predictors, that logistic regression was the most common method, and that only two studies beat the show-rate baseline on accuracy. The last finding is a caution: a model that predicts the majority class is not useful, and many published models do not clear that bar.

### Documented evidence on operating room scheduling

Bartek et al. evaluated machine-learning prediction of surgical case duration using 46,986 operations at the University of Washington. A surgeon-specific model improved the proportion of cases predicted within plus or minus 10 percent from 32 percent under the institutional standard to 39 percent. The improvement is modest and operationally meaningful: better duration estimates reduce overruns and improve block utilization.

A real-time surgical activity regulation model at Pitié-Salpêtrière, reported in *JAMA Surgery* in 2024, used a control-office nurse, live OR data, and weekly multidisciplinary planning across 12 ORs. Turnover time fell by 11.4 minutes, off-hours surgeries fell from 8.9 to 5.8 percent, non-operative time fell from 7.7 to 5.5 percent, case-duration accuracy within 15 minutes rose from 77 to 89 percent, and first-case on-time starts remained around 96 percent. The intervention was a workflow and governance change supported by data, not a model alone.

### Documented evidence on alert burden

Wong et al. externally validated a widely implemented proprietary sepsis prediction model in *JAMA Internal Medicine* in 2021. The model's discrimination was lower in external validation than in the vendor's reports, and the alert burden was substantial. The case is a reminder that a model's performance in the developer's data is not a deployment guarantee, and that the number of alerts per patient-day is a first-class operational metric. DOI 10.1001/jamainternmed.2021.2626.

### Inference

> **Inference:** The readmission and no-show effect sizes in the literature are not transferable to a local programme without a local evaluation. The Kaiser result (OR 0.91) and the Geisinger result (44 percent relative reduction) differ by an order of magnitude in relative effect, and the difference is explained by the population, the intervention, and the design, not by the model. A local business case should use a range and a pilot, not a single published number.

### Evidence table

| Programme | Model or trigger | Population | Reported effect | What it supports |
|---|---|---|---|---|
| Kaiser Transitions | EHR risk >= 25% | 21 hospitals, 1.54M discharges | Adjusted OR 0.91; ARR about 2.5 points | Analytics-targeted outreach can produce a modest effect |
| Geisinger IVR | Case management plus monitoring | Medicare Advantage, 875 enrolled | 44% relative reduction | Monitoring plus case management can reduce readmissions |
| Sepsis model external validation | Proprietary sepsis model | External health system | Lower discrimination than vendor reports | Validation must be external and prospective |
| No-show review | 105 studies | Varied | Mean no-show 23.0% | Lead time and prior no-show are the dominant predictors |
| OR duration ML | Surgeon-specific model | 46,986 operations | Within +/-10% improved from 32% to 39% | Duration prediction improves scheduling at the margin |

The table is a range, not a benchmark. The Kaiser and Geisinger effects differ by a factor of about five in relative terms, and the difference is explained by the intervention, the population, and the design. A local business case should use a range and a pilot.

### What mature organisations do

They separate the model from the programme. The model has an owner in analytics; the programme has an owner in operations; the evaluation has an owner in quality or research. The three meet monthly to review calibration, reach, effect, and equity. The separation prevents the analytics team from being blamed for a workflow failure and the operations team from being blamed for a model drift.

They also maintain a model register with the intended use, the label, the time zero, the threshold, the capacity, the validation results, and the review date. The register is the source of truth for what is deployed and why. A model that is not in the register is not in production, even if it is running in a spreadsheet.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Label mixes no-shows and cancellations | No-show rate is implausible or variable | Different events have different causes | Intervention is diluted or misdirected | Separate labels and define the cutoff |
| Threshold set at 0.5 | Flag rate is arbitrary | Default convention, not capacity | Too few or too many alerts | Derive from capacity |
| Capacity ignored | Alerts per shift exceed staff | Model and workflow designed separately | Alerts ignored; programme fails | Compute workload before launch |
| Temporal leakage | Validation performance is excellent | Post-time-zero features included | Performance collapses in deployment | Freeze features at time zero |
| Random train-test split | Test performance is stable across time | Future information leaks into training | Overstated performance | Temporal or site split |
| AUROC only | No calibration or decision curve | Discrimination is not utility | Threshold behaves unexpectedly | Calibration and net benefit |
| No comparison group | Pre-post change is attributed to the model | Secular trends and regression to the mean | Effect overstated | Difference-in-differences or trial |
| Alert fatigue | Clinicians dismiss most alerts | Flag rate exceeds review capacity | Harm to patients and trust | Cap the flag rate; tier alerts |
| Model drift | Calibration slope moves after launch | Population or workflow changed | Wrong patients flagged | Monitor calibration monthly |
| Case-mix shift | Score distribution changes by site | New service line or payer | Performance differs by site | Stratify and recalibrate |
| Equity harm | Flags concentrate in disadvantaged groups | Predictors are access barriers | Access is restricted for those who need it | Action reduces barriers; audit subgroups |
| Overbooking without limit | Wait times rise after overbooking | No safety constraint | Access worsens for everyone | Set an overbooking cap and monitor |
| Mean LOS model | Outliers dominate the prediction | Mean is sensitive to the tail | Bed planning is wrong | Quantile or survival model |
| OR schedule ignores turnover | Cases start late despite accurate durations | Turnover and PACU are not modelled | Utilization gains are not realized | Model the full flow, not only case time |
| Dashboard with no owner | Nobody can name the action | Product is a report | No behaviour change | Require decision sentence and owner |

Three failures deserve emphasis.

**The label that mixes events.** A no-show programme that includes cancellations will report a high no-show rate and will target patients who already told the clinic they cannot attend. The intervention cannot change their behaviour because the behaviour is not a no-show. The fix is a label taxonomy: attended, cancelled in advance, cancelled late, no-show, clinic-cancelled, rescheduled. Each has a different action and a different denominator.

**The threshold without capacity.** A model with a 0.5 threshold on a population with 14 percent prevalence flags far more patients than the programme can manage. The team then either ignores the model or works unpaid overtime. Both outcomes erode the programme. The threshold must be computed backward from capacity, and the workload must be simulated before launch.

**The evaluation without a counterfactual.** A readmission rate that falls after a programme launches may fall because of seasonality, a new discharge policy, or regression to the mean. Without a comparison group, the programme gets credit for changes it did not cause, and the organisation scales an intervention that does not work. The evaluation design is part of the model specification, not an afterthought.


### Pre-mortem for an operational model

1. Can the actor name the list they will receive and the action they will take?
2. Is the threshold derived from capacity, and does the workload fit the staff?
3. Is every feature available at time zero?
4. Is the validation split temporal or site-based?
5. Is there a comparison group for the evaluation?
6. What will be monitored, by whom, and how often?

A model that cannot answer all six is a prototype, not a deployment.

### Red flags in a model review

- The threshold is 0.5 and nobody can explain why.
- The validation split is random and the data span a workflow change.
- The reported metric is AUROC only.
- The capacity is described as "the team will manage".
- The outcome is measured only after launch, with no baseline.
- The subgroup analysis is missing.
> **Note:** A high AUROC is not evidence that the programme works. It is evidence that the model ranks. The programme's effect is measured by the outcome in a comparison, and the model's value is the incremental outcome per unit of capacity.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The label matches the operational event | The model must predict what the action changes | Mixed definitions, data entry lag | Label audit against the workflow |
| Features are available at time zero | Deployment must reproduce training | Late lab, post-discharge field | Feature availability audit |
| The training period resembles deployment | Calibration and ranking must transfer | Secular trend, new workflow | Temporal validation and calibration plot |
| Capacity is known and stable | Threshold depends on it | Staffing changes, leave | Monthly capacity review |
| The intervention has a plausible effect | Business case depends on it | No evidence, poor fidelity | Pilot with a comparison |
| The comparison group is credible | Effect estimate depends on it | Selection, secular trend | Parallel trends or randomisation |
| The outcome is measured consistently | Evaluation depends on it | Definition change, data lag | Version the outcome definition |
| Subgroup performance is adequate | Equity depends on it | Small samples, access barriers | Stratified validation |
| The model is monitored | Drift is detected | No owner, no dashboard | Monthly calibration review |
| The programme is staffed to deliver | Reach and fidelity depend on it | Vacancies, competing duties | Workload and vacancy tracking |


### Assumption owners

| Assumption | Owner | Review cadence |
|---|---|---|
| Label definition | Operations | Quarterly |
| Feature availability | Analytics and informatics | Each release |
| Threshold and capacity | Operations | Monthly |
| Calibration | Analytics | Monthly |
| Evaluation design | Quality or research | At protocol |
| Equity | Health equity office | Quarterly |

The limits follow from the assumptions. An operational model is a ranking tool, not a causal agent. It cannot make a patient attend, a discharge happen, or a bed appear. It can identify who is most likely to benefit from an action, and it can do so with a stated error rate. The action and the capacity produce the outcome.

A second limit is that the model is built on historical data, and the historical data reflect the old workflow. A new workflow changes the relationship between features and outcome. The model must be revalidated after a material workflow change, even if the population is stable.

A further limit is that operational data are recorded for operations, not for measurement. A no-show may be recorded in the scheduling system, the EHR, or both; a discharge may be recorded at the time of the order or the time the patient leaves. The definition must specify the source, and the source must be stable across the evaluation period.

A third limit is that the evaluation is expensive and slow. A randomised pilot may take a year, and a difference-in-differences design requires a credible control. Organisations often skip the evaluation and rely on the model's validation metrics. That is the choice that produces programmes with no evidence of benefit.

## 12. Exercises

### Exercise 1. Build a label taxonomy (recall and application)

A clinic's scheduling system records appointments as attended, cancelled, no-show, rescheduled, or clinic-cancelled. Write a label definition for a reminder programme that targets patients likely to miss an appointment. State which categories are in the label, which are excluded, and how a late cancellation is handled.

**Deliverable:** a one-page taxonomy with inclusion, exclusion, and a decision rule for late cancellations.

**Acceptance criteria:** the label matches an action the programme can deliver; late cancellations have an explicit cutoff; the denominator is stated.

**Time:** 30 minutes.

### Exercise 2. Derive a threshold from capacity (application)

A readmission programme has 12,000 eligible discharges per year and capacity for 1,500 patients. The programme operates 250 days per year. At the chosen threshold, the PPV is 0.32. Compute the flag rate, flags per day, expected events among flagged patients, and avoided events at an 18 percent relative effect. If the programme costs $300,000, compute the cost per avoided event.

**Deliverable:** a calculation table with each step.

**Acceptance criteria:** the flag rate is capacity divided by eligible; the daily workload is computed; the avoided events use the PPV and the effect; the cost per avoided event uses the programme cost.

**Time:** 45 minutes.

### Exercise 3. Compute alert burden (application)

A daily list has 40 eligible patients and capacity for 8. Each review takes 15 minutes, and each patient contact takes 10 minutes. Compute the daily review hours, the weekly hours, and the FTE required at 0.5 productive hours per paid hour. State whether the programme is feasible with one 0.5 FTE nurse.

**Deliverable:** a workload calculation and a feasibility statement.

**Acceptance criteria:** review and contact time are separated; the FTE calculation uses productive hours; the feasibility statement is explicit.

**Time:** 30 minutes.

### Exercise 4. Audit temporal leakage (critique)

A length-of-stay model uses admission source, diagnosis, comorbidity, first-day labs, discharge destination, and the number of consults completed by day 3 to predict discharge by day 4. Identify the leakage, explain the mechanism, and propose a corrected feature set. State how the corrected model should be validated.

**Deliverable:** a leakage audit with the affected features and a corrected specification.

**Acceptance criteria:** the post-time-zero features are identified; the mechanism is explained; the validation uses a temporal split.

**Time:** 45 minutes.

### Exercise 5. Implement the programme economics in SQL (application)

Using the schema from Section 7, add a scenario with a baseline rate of 0.22, an effect of 0.25, a capacity of 1,500, and a programme cost of $540,000. Write a query that returns the avoided events, cost per avoided event, savings at a 60 percent risk share, net result, and breakeven effect. Run it and paste the output.

**Deliverable:** the query, the output, and a one-paragraph recommendation.

**Acceptance criteria:** the query executes; the capacity and baseline are applied correctly; the recommendation states the condition under which the programme is viable.

**Time:** 90 minutes.

### Exercise 6. Critique an evaluation (critique)

A programme reports that readmissions fell from 16 to 13 percent after launch and concludes that the model reduced readmissions by 3 percentage points. Identify at least four threats to the conclusion and propose a stronger design.

**Deliverable:** a numbered critique and a design recommendation.

**Acceptance criteria:** at least one threat is secular trend and one is regression to the mean; the design names a comparison group and an outcome definition.

**Time:** 45 minutes.

### Exercise 7. Build a business case (application)

A no-show programme costs $90,000 per year, targets 6,000 appointments, has a baseline no-show rate of 20 percent, and assumes a 15 percent relative reduction. The contribution margin per attended visit is $110. Compute the avoided no-shows, the value of recovered visits, the net result, and the breakeven effect. State whether the programme should be funded.

**Deliverable:** a business case with the arithmetic and a recommendation.

**Acceptance criteria:** the denominator is appointments, not patients; the value uses contribution margin; the breakeven is computed; the recommendation is conditional on the assumptions.

**Time:** 60 minutes.

### Exercise 8. Design an equity audit (design)

A no-show model flags patients with prior no-shows, long lead times, Medicaid, and limited English proficiency. Design an equity audit for the programme: the subgroups, the metrics, the comparison, and the actions if a disparity is found.

**Deliverable:** a two-page audit plan.

**Acceptance criteria:** at least four subgroups; metrics include reach, effect, and access; the actions distinguish model change from programme change; the plan states who reviews the results.

**Time:** 60 minutes.


### Assessment notes

Exercises 1 to 3 can be completed individually and checked against Section 13. Exercises 4 to 8 are assessed with the rubric. Exercise 5 requires SQL; the schema in Section 7 can be extended with the new scenario. Exercise 8 benefits from a real scheduling extract; if none is available, use the categories in the exercise.

Add 20 minutes to Exercise 5 for learners who need to install SQLite.

## 13. Self-check answers

### Exercise 1

A strong taxonomy includes attended, cancelled in advance, cancelled late (within 24 hours), no-show, clinic-cancelled, and rescheduled. The reminder programme label includes no-shows and late cancellations, because both leave a slot empty and both may respond to a reminder. It excludes advance cancellations, clinic cancellations, and reschedules, because the slot is not lost in the same way and the action is different. The denominator is appointments that are still active at the time the reminder list is generated.

A weak taxonomy uses "not attended" for everything and cannot distinguish the action.

### Exercise 2

```text
Flag rate = 1,500 / 12,000 = 12.5%
Flags per day = 1,500 / 250 = 6.0
Expected events among flagged = 1,500 x 0.32 = 480
Avoided events = 480 x 0.18 = 86.4
Cost per avoided event = $300,000 / 86.4 = $3,472.22
```

The programme would need to prevent 86.4 readmissions to break even on a cost basis. Whether that is worthwhile depends on the value per avoided readmission, which is a contract and finance question. The calculation shows the scale required: the effect must be real and the PPV must hold at the deployed threshold.

### Exercise 3

```text
Review time = 8 x 15 minutes = 120 minutes
Contact time = 8 x 10 minutes = 80 minutes
Total = 200 minutes = 3.33 hours per day
Weekly = 3.33 x 5 = 16.67 hours
Paid hours at 0.5 productive = 16.67 / 0.5 = 33.3 hours per week
FTE = 33.3 / 40 = 0.83
```

One 0.5 FTE nurse is not sufficient. The programme needs about 0.83 FTE, or a reduction in the list size, or a shorter contact protocol. The calculation assumes that every contact is attempted and completed; no-answer attempts and documentation add time.

### Exercise 4

The discharge destination and the number of consults completed by day 3 are post-time-zero if the prediction is made on day 1. The discharge destination is recorded near discharge and is a consequence of the discharge process; the consult count by day 3 is future information if the prediction is made at admission. A model that uses them will predict discharge by day 4 using information from day 3, which is not available at the decision point.

A corrected feature set includes admission source, diagnosis, comorbidity, first-day labs, age, and prior utilisation. The corrected model should be validated on a later period with the same time-zero definition, and the performance should be compared with the leaky model to quantify the inflation.

### Exercise 5

The scenario has 1,500 capacity, a 0.22 baseline, and a 0.25 effect:

```text
Baseline events = 1,500 x 0.22 = 330
Avoided = 330 x 0.25 = 82.5
Cost per avoided = $540,000 / 82.5 = $6,545
Savings at 60% risk share = 82.5 x $16,800 x 0.60 = $831,600
Net = $831,600 - $540,000 = $291,600
Breakeven effect = $540,000 / (330 x $16,800 x 0.60) = 16.2%
```

The programme is viable at a 25 percent effect and a 60 percent risk share. It would not be viable at a 15 percent effect, where avoided events are 49.5, savings are $498,960, and the net is -$41,040.

### Exercise 6

A strong critique includes at least four of the following:

1. **Secular trend.** Readmissions may have been falling for reasons unrelated to the programme.
2. **Regression to the mean.** A high baseline year is likely to be followed by a lower year even without an intervention.
3. **Selection.** The programme may have enrolled patients who were more motivated or less sick.
4. **Definition change.** The readmission definition or the planned-readmission algorithm may have changed.
5. **Case-mix change.** The population may have become healthier or sicker.
6. **Concurrent interventions.** Another discharge or follow-up programme may have launched at the same time.

A stronger design is a difference-in-differences with a matched comparison hospital, or a stepped-wedge rollout across units. The outcome should be the unplanned 30-day readmission rate, defined consistently across periods.

### Exercise 7

```text
Baseline no-shows = 6,000 x 0.20 = 1,200
Avoided no-shows = 1,200 x 0.15 = 180
Value of recovered visits = 180 x $110 = $19,800
Net = $19,800 - $90,000 = -$70,200
Breakeven effect = $90,000 / (1,200 x $110) = 68.2%
```

The programme is not viable at a 15 percent effect. The breakeven effect of 68.2 percent is far above any plausible reminder effect. The programme would need a much lower cost, a much higher visit value, or a different intervention (for example, overbooking or transport support) to justify itself.

### Exercise 8 rubric

A strong equity audit specifies subgroups by race, ethnicity, language, payer, area deprivation, and disability; metrics for model performance (PPV, sensitivity), programme reach (contact rate, enrollment rate), effect (show rate or readmission rate by subgroup), and access (wait time, scheduling restrictions); a comparison to the overall population and to a pre-programme baseline; and actions that distinguish model changes (recalibration, feature removal) from programme changes (transport, interpreter services, flexible scheduling). It names the review body and the frequency.

A weak audit reports only the overall effect and calls it equitable.


A note on rounding: the worked example rounds to whole dollars and one decimal place. A production calculation should carry full precision and round only in the final table. At 1,200 patients, a $1 rounding difference in cost per avoided event is immaterial; at 100,000 patients, it is not.

## 14. Interview questions

**1. How do you define a no-show?**
A strong answer separates no-shows from cancellations, late cancellations, clinic cancellations, and reschedules, and states the cutoff and the denominator. A weak answer says "a patient who does not come".

**2. How do you choose a threshold?**
A strong answer derives it from capacity and harm, computes the flag rate and alert burden, and checks calibration at the threshold. A weak answer says 0.5 or "the Youden point".

**3. Why is AUROC not enough?**
A strong answer explains that AUROC is a ranking metric and that the decision depends on calibration, prevalence, threshold, and capacity. A weak answer says "AUROC is a good summary".

**4. What is temporal leakage in an operational model?**
A strong answer gives an example (discharge destination, follow-up appointment) and explains that the feature is not available at time zero. A weak answer confuses it with overfitting.

**5. How do you evaluate a readmission programme?**
A strong answer defines the outcome, the comparison, the design, the effect estimate, and the cost per avoided event, and notes the limits of pre-post. A weak answer compares the rate before and after.

**6. What is alert burden and how do you measure it?**
A strong answer defines alerts per clinician per shift, review time per alert, and the share of alerts acted on, and computes the FTE required. A weak answer counts total alerts.

**7. How do you handle a model that is well calibrated in development but not in deployment?**
A strong answer checks the population, the workflow, and the data pipeline, recalibrates on recent data, and monitors by subgroup. A weak answer retrains on the same data.

**8. What is the difference between a prediction and an intervention?**
A strong answer says the prediction ranks, the intervention changes the outcome, and the evaluation must measure the combination. A weak answer treats the model as the intervention.

**9. How would you model length of stay?**
A strong answer uses a skewed distribution, quantile regression or survival, and reports bed-days and discharge-by-day probabilities, not only the mean. A weak answer uses linear regression on LOS.

**10. How do you avoid penalising patients in a no-show model?**
A strong answer uses the model to target support (transport, reminders, flexible scheduling), audits subgroup reach, and avoids restricting access. A weak answer says the model is neutral because it uses data.

**11. What would you monitor after launch?**
A strong answer monitors calibration, flag rate, action rate, outcome, subgroup performance, and drift, with thresholds and owners. A weak answer monitors AUROC once.

**12. How do you size the programme to the model?**
A strong answer computes eligible patients, capacity, flag rate, PPV, effect, and cost per avoided event, and iterates until the programme is feasible. A weak answer sets the threshold first and staffs later.


**13. What is the difference between a threshold and a target?**
A strong answer says the threshold is the score at which the action is taken, derived from capacity and harm, while the target is the outcome the programme aims to achieve; conflating them leads to a threshold chosen for a desired result rather than for a feasible workflow. A weak answer uses the terms interchangeably.

**14. How would you detect that a deployed model is harming patients?**
A strong answer monitors subgroup outcomes, action rates, and unintended consequences (for example, scheduling restrictions), with a pre-specified harm metric and a stop rule. A weak answer says the model is monitored for accuracy.

## 15. Key terms

- **No-show:** a missed appointment without advance cancellation.
- **Cancellation:** an appointment cancelled before the scheduled time.
- **Late cancellation:** a cancellation within a defined window before the appointment.
- **Lead time:** days between booking and appointment.
- **Overbooking:** scheduling more appointments than available slots.
- **Show rate:** attended appointments divided by scheduled appointments.
- **Length of stay:** days from admission to discharge.
- **ALOS:** average length of stay.
- **Expected discharge date:** the predicted date of discharge.
- **Discharge barrier:** a non-clinical reason for a delayed discharge.
- **Readmission:** an inpatient return within a defined window.
- **Planned readmission:** a readmission for a scheduled procedure.
- **Index admission:** the admission that starts the readmission window.
- **HRRP:** Hospital Readmissions Reduction Program.
- **Excess readmission ratio:** predicted divided by expected readmission rate.
- **LACE:** a readmission score based on length of stay, acuity, comorbidity, and ED visits.
- **HOSPITAL score:** a seven-predictor readmission score.
- **AUROC:** area under the receiver operating characteristic curve.
- **AUPRC:** area under the precision-recall curve.
- **Calibration:** agreement between predicted and observed rates.
- **Brier score:** mean squared error of predicted probabilities.
- **Threshold:** the score above which the action is taken.
- **Capacity:** the number of actions deliverable per period.
- **Alert burden:** alerts per clinician per shift.
- **PPV:** positive predictive value.
- **NNT:** number needed to treat.
- **Cost per avoided event:** programme cost divided by events avoided.
- **Breakeven effect:** the relative effect at which benefits equal costs.
- **Counterfactual:** the outcome that would have occurred without the action.
- **Difference-in-differences:** a design comparing change in treated and control groups.
- **Interrupted time series:** a design comparing pre and post trends.
- **Stepped wedge:** a design with staggered rollout.
- **Randomised quality improvement:** random assignment of an intervention.
- **Drift:** change in the feature-outcome relationship over time.
- **Census:** the number of inpatients at a point in time.
- **Occupancy:** occupied bed-days divided by staffed bed-days.
- **Staffed bed:** a bed that can be occupied with available staff.
- **Bottleneck:** the constraint that limits throughput.
- **Boarding:** holding admitted patients in the emergency department.
- **OR utilisation:** used operating room time divided by available time.
- **Turnover time:** time between cases in an operating room.
- **First-case on-time start:** the proportion of first cases starting on time.
- **Block time:** reserved operating room time for a service.
- **Service level:** the probability that demand is met.

## 16. Sources

All URLs were accessed on 25 October 2026.

### Readmission models and policy

1. Kansagara, D. et al. Risk prediction models for hospital readmission: a systematic review. *JAMA*, 2011. https://doi.org/10.1001/jama.2011.1515
2. van Walraven, C. et al. Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community. *CMAJ*, 2010. https://pmc.ncbi.nlm.nih.gov/articles/PMC2845681/
3. Rajaguru, V. et al. LACE index to predict the high risk of 30-day readmission: a systematic review and meta-analysis. *Journal of Personalized Medicine*, 2022. https://doi.org/10.3390/jpm12040545
4. Cooksley, T. et al. Readmissions of medical patients: an external validation of two existing prediction scores. *QJM*, 2016. https://academic.oup.com/qjmed/article/109/4/245/1752723
5. CMS. Hospital Readmissions Reduction Program. https://qualitynet.cms.gov/inpatient/hrrp
6. CMS. FY2025 IPPS Final Rule Fact Sheet. https://www.cms.gov/newsroom/fact-sheets/fy-2025-hospital-inpatient-prospective-payment-system-ipps-long-term-care-hospital-prospective-0
7. Advisory Board. Charted: Hospital readmission penalties for 2025. https://www.advisory.com/daily-briefing/2024/09/24/readmisson-penalties

### Programme evidence

8. Marafino, B. J. et al. Evaluation of an intervention targeted with predictive analytics to prevent readmissions in an integrated health system: observational study. *BMJ* 2021;374:n1747. https://www.bmj.com/content/374/bmj.n1747
9. Graham, J. et al. Postdischarge monitoring using interactive voice response system reduces 30-day readmission rates in a case-managed Medicare population. *Medical Care*, 2011. DOI 10.1097/MLR.0b013e318229433e
10. Wong, A. et al. External validation of a widely implemented proprietary sepsis prediction model. *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626
11. Dantas, L. F. et al. No-shows in appointment scheduling: a systematic literature review. *Health Policy*, 2018. https://doi.org/10.1016/j.healthpol.2018.02.002
12. Carreras-García, D. et al. Patient no-show prediction: a systematic literature review. *Entropy*, 2020. https://www.mdpi.com/1099-4300/22/6/675

### Operating room and flow

13. Bartek, M. A. et al. Improving operating room efficiency: machine learning approach to predict case-time duration. *Journal of the American College of Surgeons*, 2019. https://journals.lww.com/journalacs/abstract/2019/10000/improving_operating_room_efficiency__machine.2.aspx
14. Mathon, B., Kletz, F. Operating room organization and methodology of surgical activity regulation. *JAMA Surgery*, 2024. https://jamanetwork.com/journals/jamasurgery/fullarticle/2817237
15. A comprehensive review on operating room scheduling and optimization. *Operational Research*, 2024. https://link.springer.com/article/10.1007/s12351-024-00884-z

### Methods and standards

16. Lopez Bernal, J., Cummins, S., Gasparrini, A. Interrupted time series regression for the evaluation of public health interventions: a tutorial. *International Journal of Epidemiology*, 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5407170/
17. Li, L. et al. Three statistical approaches for assessment of intervention effects: a primer for practitioners. https://pmc.ncbi.nlm.nih.gov/articles/PMC7910529/
18. HL7. FHIR Release 4. https://hl7.org/fhir/
19. CMS. Interoperability and Prior Authorization Final Rule (CMS-0057-F). https://www.cms.gov/initiatives/burden-reduction/overview/interoperability/policies-regulations/cms-interoperability-prior-authorization-final-rule-cms-0057-f
20. NCQA. HEDIS Measures and Technical Resources. https://www.ncqa.org/hedis/measures/
21. CMS. Quality Measures. https://www.cms.gov/medicare/quality/measures
22. AHRQ. Healthcare Cost and Utilization Project. https://hcup-us.ahrq.gov/

> **Note:** The LACE and HOSPITAL scores are validated tools, not regulations. Their performance varies by population, and a local validation is required before using either as a deployment threshold.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_claims-analytics.md`](03_claims-analytics.md) — the claims and risk-adjustment basis for operational outcomes.
- Next topic: [`05_capacity-and-patient-flow.md`](05_capacity-and-patient-flow.md) — queues, bottlenecks, boarding, and simulation.
- Related topic: [`06_quality-measures.md`](06_quality-measures.md) — the measure specifications that operational models must align with.
- Root survey file: [`../../08_ehr_claims_operations.md`](../../08_ehr_claims_operations.md)
- Labs: [`../../labs/`](../../labs/)

### Reading path

| If you need to | Read | Then |
|---|---|---|
| Build a no-show model | Sections 5 and 6 | Topic 05 for scheduling capacity |
| Build a readmission model | Sections 5, 6, and 9 | Topic 08 for the evaluation design |
| Set an operating threshold | Sections 5 and 11 | The capacity model in Topic 05 |
| Evaluate a programme | Sections 9 and 10 | Topic 08 |
| Report to an operating committee | Sections 6 and 12 | The value case in Topic 07 |

Operational models sit between the population definition in Topic 02, the cost basis in Topic 03, and the capacity model in Topic 05. A model that is accurate but cannot be staffed is not an operational model; it is a score. The test of the topic is whether the reader can compute the capacity, the threshold, and the cost per avoided event before building the model, and then decide whether the model is worth building at all.

The operational model is the bridge from prediction to operations. Topic 05 takes the capacity constraint further, into queues and flow; Topic 08 takes the evaluation further, into interrupted time series and difference-in-differences. A reader who can size the programme before building the model has learned the main lesson of this topic.

| Limit | Consequence | Workaround |
|---|---|---|
| Label ambiguity | Effect is diluted | Label audit |
| Workflow change | Calibration drifts | Revalidate after change |
| Capacity change | Threshold is stale | Recompute flag rate |
| Small subgroups | Equity estimates imprecise | Pool periods or sites |
| Short follow-up | Outcome incomplete | Extend the window |

| Small samples | Unstable estimates | Report confidence intervals |

The table is a reminder that every operational model has a measurement limit, and the limit should be stated before the model is deployed.

A model that is accurate, calibrated, staffed, and evaluated is still a model. The clinical judgement remains with the clinician.

That division of labour is the subject of the next topic.
