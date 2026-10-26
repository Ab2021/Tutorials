# Post-Market Evaluation

> Module 18 — Quality, Evaluation, and Reporting
> Topic 07 of 09
> Estimated study time: 3.5 hours
> Prerequisites: 03, 06
> Version 2026.3

## 1. Why this topic exists

A model that passes validation has been shown to work on a sample collected before deployment. The
deployment changes the sample. Clinicians learn to react to the alert, documentation practices
shift, a new laboratory assay is introduced, the case mix changes with a new service line, and the
model's inputs are quietly different from the ones it was trained on. None of these events is a
failure of the model. They are the normal life of a deployed system, and they are the reason
post-market evaluation exists.

The evidence that deployment goes wrong is not anecdotal. A 2025 analysis of 950 FDA-cleared
AI-enabled devices found that 60 devices (6.3 percent) were linked to 182 recall events, that 79
recalls (43.4 percent) occurred within 12 months of clearance, and that diagnostic or measurement
errors caused 109 of the recalls. Lack of clinical validation was associated with recall (odds
ratio 2.8, 95 percent confidence interval 1.6 to 4.7). The pattern is early failure, and early
failure is exactly the window in which a monitoring programme would have to operate.

The regulatory expectation has converged on the same point. The FDA's Good Machine Learning
Practice principles, published jointly with Health Canada and the MHRA in October 2021, include a
principle that deployed models be monitored for performance and that retraining risks be managed.
The EU Medical Device Regulation requires post-market surveillance and, for certain devices,
post-market clinical follow-up. The FDA's predetermined change control plan guidance, finalised in
2025, allows planned modifications within an approved envelope, which only works if the
manufacturer can measure the effect of each change.

The commercial expectation has converged too. A hospital buying a model now asks what happens when
it drifts, who detects it, in what timeframe, and with what authority to act. A vendor that cannot
answer is asking the buyer to absorb a risk the vendor has not measured.

The failure pattern is consistent enough to name. A model is validated retrospectively, deployed
with an announcement, and monitored through the vendor's quarterly report. The report shows stable
aggregate performance because it is computed on the vendor's customer base, not the local
population. A clinician mentions that the alerts seem less useful, the department raises it at a
governance meeting, and a local review is commissioned. The review finds a material gap. The
sequence takes twelve to eighteen months, which is longer than the recall window documented in the
literature and long enough for the model to have caused harm that no one attributed to it.

The alternative is not a large programme. It is a small one, designed before go-live, with five to
nine metrics and a monthly meeting that can act. The cost is a fraction of the cost of the
retrospective review, and the difference in detection time is measured in months.

This topic teaches how to design and run a post-market evaluation programme: what to measure, how
often, with what statistical rule, with what thresholds, and with what governance. It builds on
the appraisal in topic 03 (what performance means) and the quality system in topic 06 (how a
finding becomes a CAPA). It does not teach a specific dashboard product; the principles are
tool-independent.

The organisational obstacle is rarely technical. A monitoring programme needs three things that
sit in different departments: data access from information technology, clinical interpretation from
the service line, and authority to act from quality or medical leadership. Each department can
approve its part and none can deliver the whole. The programme therefore needs an owner with a
mandate that crosses the three, and the most common reason a programme never launches is that no
such owner exists. The technical design in this topic is straightforward by comparison.

A note on scope. Post-market evaluation is not the same as post-market surveillance in the
regulatory sense. The regulatory term covers complaints, literature, and field data as well as
performance; this topic focuses on the performance measurement component, which is the part an
analytics team owns.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** covariate shift, label shift, and concept shift, and predict which metrics each
   affects.
2. **Design** a monitoring plan with metrics, data sources, cadence, statistical rules, thresholds,
   owners, and escalation paths.
3. **Compute** and interpret a CUSUM or a control-chart signal for a performance metric.
4. **Separate** leading indicators (inputs, alerts, overrides) from lagging indicators (outcomes).
5. **Set** a threshold that balances detection delay against false alarms, with a stated rationale.
6. **Investigate** a drift signal to a cause: data pipeline, population, workflow, or model.
7. **Link** a monitoring finding to change control and to a CAPA.
8. **Report** post-market performance to a regulator, a customer, or a quality committee without
   overstating what the data show.

## 3. The core mental model

Post-market evaluation is three concentric loops, each with a different latency.

```text
   outer loop    clinical outcomes and safety        months to years
        ^
   middle loop   model performance (AUROC, calibration, PPV)   weeks to months
        ^
   inner loop    inputs, alerts, overrides, data quality       days to weeks
```

The inner loop is fast and cheap. It measures what the system is doing: how many alerts fire, how
often they are overridden, whether input variables are missing or shifted, whether the data
pipeline is delivering the fields the model expects. The middle loop measures whether the model is
still accurate, which requires labels and therefore lags. The outer loop measures whether patients
are better or worse off, which requires outcomes and a design that can attribute change, and
therefore lags further.

The loops are nested because a signal in an inner loop is a hypothesis about the outer loops. An
increase in override rate is a hypothesis that the model's alerts have become less relevant. An
increase in missing input data is a hypothesis that performance has degraded. The programme's job
is to test the hypothesis cheaply before the outcome loop can.

A second model is the monitoring triangle. Every metric needs three things, and a metric without
all three is a number on a dashboard rather than a control.

```text
        signal  --------  threshold  --------  action
          |                                        |
     what is measured                      who does what,
     how often                             by when, with
     from where                            what authority
```

A signal without a threshold never triggers. A threshold without an action produces an alert that
no one owns. An action without a signal is a procedure waiting for a trigger. The triangle is the
minimum unit of a monitoring plan, and a plan that lists twenty metrics and no actions is a
reporting exercise.

A third model is the drift taxonomy, because the type of shift determines which metric moves first.

| Shift type | What changes | First metric to move | Detection approach |
|---|---|---|---|
| Covariate shift | Distribution of inputs | Input distributions, missingness | Feature-based tests |
| Label shift | Prevalence of the outcome | PPV, alert rate, calibration intercept | Prevalence estimation |
| Concept shift | Relationship between inputs and outcome | AUROC, calibration slope | Performance monitoring with labels |
| Data pipeline change | Field definitions, units, coding | Missingness, range checks, schema | Data quality checks |
| Workflow change | Who uses it, how they respond | Override rate, time to action | Usage telemetry |

The taxonomy matters because a monitoring programme that watches only AUROC will miss covariate
shift until labels arrive, which may be months. A programme that watches inputs and usage can
detect the same problem in days. That is the argument for the inner loop.

A fourth property of the loops is that each has a different data requirement, and the data
requirement determines the feasible cadence. The inner loop uses data the system already generates:
logs, timestamps, override events, field completeness. The middle loop uses labels, which exist only
if the outcome is captured for the same patients the model scored, and only after enough time has
passed for the outcome to occur. The outer loop uses outcomes and, to attribute them, a design. A
programme that tries to run the outer loop monthly will either use an inadequate design or produce
noise; a programme that runs only the inner loop will detect operational problems but never confirm
whether the model is still accurate.

A fifth property is that the loops can disagree, and the disagreement is informative. An inner loop
that shows a rising alert rate with a stable middle-loop AUROC suggests a prevalence change or a
threshold change, not a loss of discrimination. A middle loop that shows falling AUROC with stable
inputs suggests a concept change, which is the hardest to diagnose because nothing in the data
distribution looks wrong. The review should read the loops together, which is why a single dashboard
with separate panels is more useful than a single composite score.

A fourth model is the detection trade-off. Every statistical rule has a false-alarm rate and a
detection delay, and the two trade against each other. A sensitive rule fires early and often; a
specific rule fires late and rarely. The plan should state which error is more costly for each
metric, because the answer differs. For a safety signal, a false alarm is cheap and a missed
signal is expensive. For an alert-burden metric, a false alarm consumes operational attention and
should be rarer.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| Post-market surveillance | Ongoing collection of data about a marketed device | Confused with performance monitoring only | EU MDR, FDA |
| Post-market clinical follow-up | Prospective clinical data collection after CE marking | Treated as optional for all devices | EU MDR |
| Drift | Change in data or performance after deployment | Used for any change | Generic |
| Covariate shift | Change in the distribution of inputs | Confused with concept shift | Distribution |
| Label shift | Change in outcome prevalence | Missed when AUROC is stable | Prevalence |
| Concept shift | Change in the input-outcome relationship | Assumed absent if inputs look stable | Relationship |
| Dataset shift | Umbrella term for the three shifts | Used loosely | Generic |
| Calibration drift | Change in agreement between predicted and observed risk | Confused with discrimination change | Slope, intercept |
| Discrimination drift | Change in ranking performance | The only drift many teams measure | AUROC |
| O:E ratio | Observed to expected events | Computed without an interval | Ratio |
| Alert burden | Alerts per unit of clinical work | Reported as a count without a denominator | Alerts per 100 admissions |
| Override rate | Share of alerts dismissed without action | Treated as a satisfaction metric | Proportion |
| Leading indicator | A signal that moves before the outcome | Confused with a proxy outcome | Metric |
| Lagging indicator | A signal that moves after the outcome | Treated as the only evidence | Metric |
| Control chart | Time series with limits for common-cause variation | Used without a baseline | Chart |
| CUSUM | Cumulative sum of deviations from a target | Used without specifying the shift to detect | Statistic |
| SPRT | Sequential probability ratio test | Confused with CUSUM | Test |
| Sensitivity of monitoring | Probability of detecting a true shift | Not computed | Proportion |
| False-alarm rate | Probability of a signal when no shift | Not computed | Rate per period |
| Detection delay | Time from shift to signal | Not measured | Days or periods |
| Trigger | A threshold that initiates an investigation | Confused with an alert to clinicians | Rule |
| Escalation | Defined path from signal to authority | Informal | Procedure |
| Silent mode | Running a model without clinical action | Confused with shadow deployment of a full workflow | Mode |
| Registry | Structured collection of outcomes for a population | Confused with a biobank | Dataset |
| Real-world evidence | Evidence from routine care data | Treated as automatically generalisable | Evidence type |

Four distinctions carry the topic.

**Drift versus degradation.** Drift is a change in the data or the relationship. Degradation is a
change in performance. Drift can occur without degradation (the inputs shift but the model remains
accurate) and degradation can occur without measurable drift (a subtle concept change that does not
show in input distributions). Monitoring needs both, and the plan should say which is being
measured.

**Monitoring versus evaluation.** Monitoring is continuous, automated, and cheap per observation.
Evaluation is periodic, designed, and expensive. A monitoring signal is a reason to evaluate, not
an evaluation. Confusing the two produces either endless dashboards with no conclusions or
expensive evaluations that are too late.

**Leading versus lagging.** An alert rate is leading because it changes when the model or the
workflow changes and is available immediately. A mortality rate is lagging because it requires
outcomes and a design. A programme that monitors only lagging indicators learns about problems
after they have harmed patients.

**Alert burden versus alert usefulness.** Alert burden is the volume of alerts per unit of clinical
work. Alert usefulness is the share of alerts that lead to a correct action. A model can have a
stable burden and falling usefulness, which is the signature of concept drift: the same number of
alerts, fewer of them right. Monitoring both separates a volume problem (too many alerts) from an
accuracy problem (wrong alerts), and the remedies differ: threshold or workflow for volume,
recalibration or retraining for accuracy.

**Signal versus action.** A signal is a statistical event. An action is a decision with an owner.
The gap between them is where most programmes fail: the dashboard shows a red cell, no one is
accountable, and the cell stays red for a year.

## 5. The mechanics, step by step

This procedure designs and operates a post-market evaluation programme for one deployed model. It
takes four to eight weeks to design and then runs continuously with a monthly review. The steps are
written for an analytics team that owns the model, with interfaces to clinical operations and
quality.

### Step 0. Secure data access and the right to monitor

**Input:** the vendor contract, the data governance arrangements, and the deployment agreement.

**Output:** a documented right to receive the data needed to compute the metrics, and a defined
data flow.

**Rationale:** Monitoring is impossible without data. A buyer that cannot obtain alert logs,
overrides, and outcomes cannot compute the metrics, and a contract that does not grant those rights
leaves the buyer dependent on the vendor's own reports. The access question is negotiated before
deployment, not after a problem appears.

**Branch point:** if the vendor will not provide record-level monitoring data, the buyer can still
monitor aggregate alert rates and overrides from its own systems, and should record the limitation.
A programme with local data only is weaker but not absent.

### Step 1. State the deployment and the monitoring question

**Input:** the intended use, the deployment sites, the workflow, and the risk classification.

**Output:** a monitoring charter: what the programme is for, which decisions it informs, and which
risks it covers.

**Rationale:** Monitoring without a purpose accumulates metrics. The charter should name the
decisions: continue, recalibrate, retrain, restrict, or withdraw. Each decision needs different
evidence and a different cadence.

**Branch point:** three deployment types, each with a different emphasis.

| Deployment | Primary monitoring focus | Cadence |
|---|---|---|
| Silent mode (no clinical action) | Data quality, alert rate, performance with labels | Weekly |
| Assistive (clinician decides) | Override rate, alert burden, calibration, subgroup performance | Weekly to monthly |
| Autonomous (model acts) | Safety events, performance, drift, fail-safe behaviour | Continuous |

### Step 2. Define the metrics

**Input:** the charter and the risk file.

**Output:** a metric set covering inputs, outputs, performance, safety, and equity, with a
definition and a data source for each.

**Rationale:** The metric set should cover each loop of the model in section 3. A set that covers
only performance will detect problems late; a set that covers only usage will never confirm
whether the model is still accurate.

**Branch point:** if a metric cannot be computed from available data, either build the data source
or record the gap. Do not include a metric that will never be populated; it creates the appearance
of monitoring without the substance.

A workable minimum set:

| Layer | Metric | Definition | Source |
|---|---|---|---|
| Input | Missingness | Share of required fields null at scoring time | Pipeline log |
| Input | Distribution shift | Distance between current and reference input distributions | Feature store |
| Output | Alert rate | Alerts per 100 eligible patients | Audit log |
| Output | Override rate | Alerts dismissed without documented action | Audit log |
| Performance | AUROC | Discrimination on labelled recent cases | Outcomes data |
| Performance | Calibration slope | Observed vs predicted on the logit scale | Outcomes data |
| Performance | PPV at operating point | Share of alerts that are true cases | Outcomes data |
| Safety | Serious safety events | Events with a possible model contribution | Safety reporting |
| Equity | Subgroup performance | AUROC and PPV by prespecified groups | Outcomes data |

**What good metric definitions have in common.** Four properties, and a metric missing any of them
will be argued about at the first review.

- A numerator and a denominator that are both computable from a named source.
- A time window, because "override rate" without a window is ambiguous.
- An exclusion rule, because patients who were never eligible should not be in the denominator.
- A version, because a metric definition that changes silently breaks the trend.

The last property is the one teams omit. A metric that is redefined between periods produces a
step change that looks like drift and is not. The monitoring plan should treat the metric definition
as a controlled document, with the same discipline as the model.

### Step 3. Establish the baseline

**Input:** the validation results and the first deployment period.

**Output:** a baseline value, an interval, and a reference period for each metric.

**Rationale:** A threshold without a baseline is arbitrary. The baseline should be the validated
performance in the deployment population, or the first stable period if the validation population
differs. The interval matters because a threshold set at the point estimate will fire on normal
variation.

**Branch point:** if the baseline period is unstable (a launch effect, a new site), extend it or
exclude the launch period and record the exclusion.

### Step 4. Choose the cadence

**Input:** the metric, the risk, and the data latency.

**Output:** a monitoring frequency and a reporting frequency for each metric.

**Rationale:** Cadence is a function of how fast the metric can change and how fast the consequence
arrives. Input data quality can change overnight and should be checked continuously. AUROC requires
labels that may take 30 to 90 days to accrue, so a monthly computation with a rolling window is
realistic. Mortality may require a year.

**Branch point:** if the cadence is slower than the consequence, the monitoring is retrospective.
Record the mismatch and add a leading indicator that moves faster.

### Step 5. Set thresholds and statistical rules

**Input:** the baseline, the interval, and the cost of each error.

**Output:** a rule per metric: a fixed threshold, a control-chart limit, or a sequential test, with
the rationale.

**Rationale:** The rule determines the false-alarm rate and the detection delay. A fixed threshold
of "AUROC below 0.80" is simple and slow. A CUSUM tuned to detect a 0.03 drop is faster and fires
more often. The choice should be justified by the consequence, not by convention.

**Branch point:** three rule types.

| Rule | Use when | Strength | Weakness |
|---|---|---|---|
| Fixed threshold | The decision has a hard boundary | Simple, auditable | Ignores trend and variance |
| Control chart | The metric is stable and periodic | Separates common and special cause | Needs a stable baseline |
| CUSUM or SPRT | Small persistent shifts matter | Fast detection of small shifts | Requires a target and a drift size |

**Edge case, incomplete reporting:** if labels arrive late or incompletely, the performance metric
is biased toward the patients who were followed up. Record the completeness and compute the metric
only for the labelled subset, with the subset size. A metric computed on 60 percent of cases with no
statement of the 60 percent is a number that will be over-read at the review.

### Step 6. Build the data pipeline for monitoring

**Input:** the metric definitions and the source systems.

**Output:** an automated pipeline that computes the metrics on schedule and stores the results with
a timestamp and a version.

**Rationale:** A monitoring metric computed by hand is not a monitoring metric. The pipeline must
run without the analyst, log its inputs, and fail visibly. The most common failure is a silent
pipeline that stops updating and a dashboard that shows the last value indefinitely.

**Branch point:** if the pipeline cannot be automated, the metric should be marked manual and
scheduled, with an owner and a fallback. A manual metric that no one owns is not monitored.

### Step 7. Set the review and escalation process

**Input:** the thresholds and the organisation chart.

**Output:** a review calendar, a named reviewer, an escalation path, and decision authority.

**Rationale:** The signal is useless without a decision process. The review should have a standing
agenda, a record, and the authority to order an investigation, a recalibration, or a suspension.
The escalation path should name the role, not the person, so that it survives turnover.

**Branch point:** if no role has the authority to suspend the model, the programme cannot act on
its most important signal. Escalate the governance gap before launch.

### Step 8. Run the review meeting

**Input:** the dashboard and the metric history.

**Output:** minutes with decisions, owners, and dates.

**Rationale:** The meeting is the control. It should review exceptions, not every metric, and it
should record decisions rather than observations. "AUROC declined" is an observation. "Investigate
the input pipeline by 15 April; owner data lead; if confirmed, recalibrate and revalidate by 30
May" is a decision.

**Branch point:** if the meeting has no authority to decide, it becomes a reporting forum. Record
the limitation and escalate.

### Step 9. Investigate a signal

**Input:** the triggered metric and the surrounding data.

**Output:** a cause attribution and an investigation report.

**Rationale:** A signal is a hypothesis. The investigation tests it against the possible causes:
data pipeline, population change, workflow change, model change, or measurement change. The order
matters, because a pipeline change is cheap to check and can mimic any other cause.

**Branch point:** four causes, each with a different remedy.

| Cause | Evidence | Remedy |
|---|---|---|
| Data pipeline | Missingness, schema change, unit change | Fix pipeline; backfill; re-run |
| Population | Case mix, prevalence, new service | Recalibrate; stratify; restrict use |
| Workflow | Override rate, usage pattern | Training; interface change; threshold |
| Model | Version change, drift in weights | Roll back; retrain; revalidate |

### Step 10. Decide and act

**Input:** the investigation and the change-control procedure.

**Output:** a decision: continue, recalibrate, retrain, restrict, or withdraw; with a change record.

**Rationale:** Every action on a deployed model is a change and must go through change control. A
recalibration is a change with a revalidation requirement; a retraining is a larger one; a
withdrawal is a recall-like action with its own communication.

**Branch point:** if the evidence is insufficient to decide, the correct action is a time-boxed
investigation with a decision date, not an indefinite "monitor".

### Step 11. Report

**Input:** the review record and the investigation.

**Output:** periodic reports to quality, clinical leadership, and where required the regulator;
and customer communications when performance changes materially.

**Rationale:** Reporting is a legal obligation in some regimes and a commercial necessity in all.
The report should state what was measured, what changed, what was done, and what remains uncertain.

**Branch point:** if a safety signal is identified, the reporting timeline is governed by the
applicable vigilance or medical device reporting rules, not by the internal calendar. The programme
must have a named handoff to the regulatory function, tested before the first incident rather than
during it. A handoff that exists only as an org chart line is not a handoff; it should be exercised
once, with a tabletop scenario, before the programme goes live.

### Step 12. Update the baseline and retire

**Input:** the monitoring history.

**Output:** a revised baseline after a validated change, and a retirement plan for the model.

**Rationale:** After a recalibration, the old baseline is obsolete. The new baseline should be
established from the post-change period, with the change documented. Retirement is a planned event
with a migration path, not a surprise.

**Branch point:** if the model is replaced, the monitoring continues for the replacement and the
old model's history is retained for the record.

### Recording the plan

The monitoring plan is a controlled document, not a slide. It has six parts.

| Part | Content | Owner |
|---|---|---|
| Scope | Model, version, sites, population, period | Product |
| Metric set | Names, definitions, sources, cadence | Data lead |
| Baselines | Value, interval, reference period | Data lead |
| Rules | Threshold, statistical method, rationale | Quality |
| Governance | Reviewer, meeting, escalation, authority | Clinical lead |
| Change log | Metric or threshold changes with rationale | Quality |

The change log is what makes the plan auditable. A threshold that moved from 0.78 to 0.75 without a
record is indistinguishable from a threshold that was always 0.75, and the second is a claim the
organisation cannot support.

### Edge cases

**Incomplete reporting.** Labels are the binding constraint. If only 60 percent of cases have
outcomes, the performance metric describes that 60 percent. Report the completeness and the
selection mechanism.

**Selective outcome reporting.** If the monitoring report shows only the metrics that improved,
the report is selecting. Publish the full metric set with the exceptions highlighted.

**Subgroup claims.** A subgroup signal needs the subgroup definition, the n, and the interval. A
subgroup with 40 cases cannot support a claim of inequity or of safety.

**Protocol deviations.** If the monitoring plan specified monthly review and the reviews happened
quarterly, the deviation is a finding, because the detection delay was longer than designed.

**Post-hoc changes.** Changing a threshold after seeing the data is a post-hoc change. It requires
a documented rationale and a re-baselining, not a quiet edit.

**Conflicting reference standards.** If the outcome can be ascertained by a registry and by codes,
report both. A performance drop under one standard and not the other is informative about the
measurement, not only the model.

**Retracted studies.** If the model's validation study is retracted, the monitoring programme
should reassess the baseline and the claim. A retraction does not change the deployed model, but it
changes what can be claimed about it.

**What the plan does not cover.** Every monitoring plan has a boundary, and stating it is part of the
plan. The common omissions are: performance in populations excluded from the metric set, the effect
of co-interventions introduced at the same time, the behaviour of the model when inputs are missing
at scoring time, and the performance of the fallback pathway when the model is unavailable. A plan
that does not name these leaves them to be discovered during an incident, which is the worst time
to discover them.

### Decision table for a triggered signal

| Signal | Confirmed cause | Recommended action | Authority |
|---|---|---|---|
| Input missingness spike | Pipeline | Fix and backfill; re-run | Data lead |
| Alert rate rise, stable AUROC | Threshold or prevalence | Recalibrate threshold; review workflow | Clinical lead |
| AUROC decline with labels | Concept or population shift | Recalibrate; revalidate; consider retraining | Model owner |
| Calibration drift, stable AUROC | Population prevalence | Recalibrate | Model owner |
| Subgroup gap | Population or measurement | Investigate; restrict or mitigate | Clinical governance |
| Safety event | Any | Suspend; investigate; report | Medical safety officer |

## 6. Worked example

The case is a post-market appraisal of the Epic Sepsis Model (ESM) deployment described by Wong and
colleagues in *JAMA Internal Medicine* 2021. The study is a retrospective external validation at
Michigan Medicine, not a monitoring programme, and that is the point: it provides the performance
evidence that a monitoring programme would have produced, and it allows an appraisal of what
monitoring would have been needed to detect the gap before a retrospective study did.

The published facts used here: 38,455 hospitalisations, 27,697 patients, AUROC 0.63 (95 percent
confidence interval 0.62 to 0.64) against a vendor-reported 0.76 to 0.83, sensitivity 33 percent
and PPV 12 percent at the operational threshold, alerts in 18 percent of hospitalisations, and
1,709 of 2,552 sepsis cases not identified.

### The deployment context

The model was in routine use at the site, scoring hospitalised patients every 15 minutes and
surfacing a score to ward staff. The study period ran from December 2018 to October 2019. The model
had been implemented as part of a sepsis pathway, which means the alerts were embedded in a
workflow with a response protocol, not presented as a standalone score.

That context matters for the monitoring question. In a workflow-embedded deployment, three things
can change without any change to the model: the population (who is admitted), the workflow (how
staff respond), and the data (which fields are available and how they are coded). A monitoring
programme that watches only the model's weights will miss all three.

### What the study can and cannot tell you

The study is a retrospective external validation with a carefully specified outcome. It can tell
you the model's discrimination and operating characteristics in the deployment population over ten
months. It cannot tell you what the site knew during those months, because it was not designed as a
monitoring evaluation and does not report the monitoring that existed.

That limitation is why the appraisal in this section is an appraisal of the monitoring implied by
the evidence, not of a documented programme. The score is therefore a judgement about adequacy, and
the memo should say so. A reviewer who reads the score as a measurement of a programme has
over-read the evidence.

### The appraisal question

Was the monitoring in place at the deploying site adequate to detect, within 90 days, that the
model's performance in this population was materially below the vendor's reported range?

This is a post-market question, not a validation question. The validation tells us the performance;
the monitoring question is whether the organisation would have known without commissioning the
study.

### Scoring the monitoring programme

Six domains, each 0 to 2, scored against what the study implies about the site's monitoring.

| Domain | Question | Score | Reasoning |
|---|---|---:|---|
| Metric coverage | Were discrimination, calibration, and alert burden measured? | 1 | The study measured all three retrospectively; no evidence they were monitored prospectively |
| Baseline | Was a local baseline established before or at go-live? | 0 | The vendor range was the only baseline; no local pre-deployment measurement is reported |
| Cadence | Was performance reviewed at an interval shorter than the harm horizon? | 0 | No prospective review is described; the study was a retrospective response |
| Threshold and rule | Was there a pre-specified rule for investigating a drop? | 0 | No rule is reported |
| Action and authority | Was there a defined action and an owner? | 1 | The study implies a decision to evaluate; the governance path is not described |
| Reporting | Was performance reported to clinical leadership and quality? | 1 | The study is a public report; routine reporting is not described |

Total: 3 of 12, or 25 percent. The profile is the finding: the site could measure the right things
but had no baseline, no cadence, no rule, and no evident action path. The programme was, in effect,
a retrospective evaluation triggered by concern rather than a monitoring system.

### The three loops applied to this deployment

The inner loop would have tracked missingness in the input variables, the alert rate per 100
admissions, and the override rate. The study reports an alert rate of 18 percent, which is an inner
loop metric computed retrospectively. At that rate, a ward with 30 admissions per day receives about
five alerts per day, which is high enough to produce fatigue and low enough to be dismissed as
normal if no baseline exists.

The middle loop would have tracked AUROC and calibration on labelled cases. The study computes
AUROC at several horizons and reports poor calibration at all of them. A monthly middle-loop review
with labels would have detected the calibration problem before discrimination, because calibration
responds to prevalence and case mix and does not require as many events to estimate.

The outer loop would have tracked sepsis mortality, time to antibiotics, and safety events. The
study does not measure these, and it could not attribute them without a design. The outer loop is
the reason the model was deployed, and it is the loop with the least evidence.

### The baseline arithmetic

A local baseline should have been established before go-live. Suppose the site had run the model in
silent mode for 60 days, accruing 2,000 admissions with 130 sepsis cases. The standard error of
AUROC with 130 events and 1,870 non-events is approximately 0.027. A baseline of 0.63 with an
interval of 0.58 to 0.68 would have been immediately and visibly different from the vendor's 0.76
to 0.83, and the gap would have been detected before any clinical use.

The arithmetic is deliberately simple because the point is simple: the information needed to detect
the problem was available at a sample size that a two-month silent period would provide. The
failure was not statistical power. It was the absence of a baseline, a rule, and a review.

### Translating to the detection question

With a local baseline AUROC of 0.63 and a 95 percent interval of 0.62 to 0.64, the gap from the
vendor's lower bound of 0.76 is 0.13, or about eight standard errors of the estimate. A monitoring
rule with a baseline of 0.76 and a threshold at 0.72 would have fired at the first adequately
powered measurement. With 38,455 hospitalisations over ten months, the monthly volume is about
3,800, which is enough to estimate AUROC with a standard error of about 0.01. A monthly review
would have detected the gap in the first month in which labels were complete, not after a
retrospective study.

The counterfactual is therefore uncomfortable: the information needed to detect the problem was
available cheaply and early. The missing element was not data or statistics; it was a baseline, a
rule, and a meeting.

### Recommendation

For the site: establish a local baseline before deployment, set a monitoring rule with a stated
detection target, and review monthly with an owner who can suspend the model. For a vendor: supply
a monitoring specification with the product, including the metrics, the expected ranges, and the
recommended thresholds, because a customer without a specification will not build one.

For the buyer of a similar product: treat the absence of a monitoring specification as a
procurement finding, and require one before go-live.

### What would make this a failing programme rather than an absent one

The distinction matters for the recommendation. An absent programme is a design task; a failing
programme is a governance task, and the remedies differ.

If the site had been reviewing monthly and had attributed the low AUROC to "a difficult case mix"
without investigation, the programme would be failing rather than absent. The metric coverage would
score 2, the cadence 2, and the action 0, because the review produced an explanation rather than a
test. The recommendation would be to fix the investigation discipline, not to build a dashboard.

If the site had investigated and found the calibration problem, then recalibrated without
revalidating, the programme would be failing in a different way: the action would exist but the
change control would be absent. The pattern is common, and it produces a model whose probabilities
have been adjusted without evidence that the adjustment works.

The recommendation in each case names the specific failing control, because a generic
"improve monitoring" is not actionable.

### Two diagnostics that would change the verdict

**Diagnostic 1: a documented pre-deployment local validation with AUROC 0.72 and a monitoring rule
at 0.68.** The baseline domain would move from 0 to 2 and the threshold domain from 0 to 2, raising
the score to 7 of 12. The recommendation would change from "no monitoring system" to "monitoring
system with a baseline that was too optimistic and a rule that was set too low." The detection
would have occurred later than optimal but earlier than the retrospective study. The verdict moves
from absence to mis-calibration of the monitoring, which is a different and less severe finding.

**Diagnostic 2: monthly performance reports to the quality committee with no action taken.** The
reporting domain would move to 2 and the action domain would remain 1. The total would be 5 of 12,
and the finding would become "monitoring without a decision process." This is the most common
failure in practice and the most corrosive, because it produces evidence that the organisation knew
and did not act. The recommendation would shift from building a programme to fixing governance.

### Vary one assumption: cadence

Suppose the site had reviewed performance annually rather than monthly. The metrics and the rule
would be unchanged, and the detection delay would be up to twelve months. Given that 43.4 percent
of AI device recalls occur within twelve months of clearance, an annual cadence is aligned with the
period in which the risk is highest, which means the first review could arrive after the harm. The
cadence domain would score 0, and the recommendation would add a requirement for a leading
indicator, such as alert rate and override rate, reviewed weekly, to compensate for the slow
performance loop.

The arithmetic is simple and worth stating. If the model is degrading at a rate that produces a
0.01 AUROC loss per month, a monthly review detects it after one or two months, and an annual
review detects it after twelve. The difference is eleven months of exposure, which is the quantity
the cadence decision is actually about.

## 7. Code, query, or calculation

The script implements a CUSUM monitor for monthly AUROC, computes the first alarm, and estimates
the sample size needed to detect a given drop. It uses only the standard library.

Three caveats on the implementation. The standard error function uses the Hanley-McNeil
approximation, which assumes a binormal model and is adequate for planning but not for inference.
The CUSUM parameters are chosen to illustrate the trade-off; a production rule should be calibrated
on the metric's actual variance, for example by simulation on historical data. And the sample-size
formula answers "how many cases to detect a drop at 80 percent power in a single comparison"; a
sequential monitor accumulates evidence differently, so the two numbers are not interchangeable.

```python
"""CUSUM monitoring for a performance metric.

The CUSUM accumulates deviations from a target. k is the reference value
(half the shift to detect); h is the decision interval. Both are in the
units of the standardised deviation.
"""

import math

# Monthly AUROC after deployment, with the number of labelled cases per month.
monthly = [
    (1, 0.821, 640), (2, 0.819, 655), (3, 0.815, 702), (4, 0.808, 688),
    (5, 0.799, 731), (6, 0.785, 745), (7, 0.772, 719), (8, 0.761, 760),
    (9, 0.748, 742), (10, 0.741, 755), (11, 0.738, 748), (12, 0.735, 770),
]

TARGET = 0.820      # validated performance
SHIFT = 0.030       # smallest drop we care to detect quickly
K = SHIFT / 2       # reference value
H = 5 * 0.012       # decision interval, in AUROC units (five sigma)


def cusum(values, target, k, h):
    s_hi = 0.0
    out = []
    for i, (month, x, n) in enumerate(values, start=1):
        s_hi = max(0.0, s_hi + (target - x) - k)
        out.append((month, x, s_hi, s_hi > h))
    return out


print(f"target {TARGET:.3f}, shift {SHIFT:.3f}, k {K:.3f}, h {H:.3f}")
print(f"{'month':>6}{'auroc':>8}{'cusum':>8}  alarm")
print("-" * 32)
alarm_month = None
for month, x, s, alarm in cusum(monthly, TARGET, K, H):
    print(f"{month:>6}{x:>8.3f}{s:>8.3f}  {'YES' if alarm else ''}")
    if alarm and alarm_month is None:
        alarm_month = month

print()
print("first alarm month:", alarm_month)

# Expected detection delay for a sustained shift of SHIFT, in months.
# For a one-sided CUSUM, the average run length after a shift is roughly h / (shift - k).
arl = H / (SHIFT - K)
print(f"approximate ARL after shift: {arl:.1f} months")

# Sample size for detecting a drop of 0.05 in AUROC at 80% power (normal approximation).
# SE(AUROC) ~ 0.63 * (1 - 0.63) / sqrt(n_pos) is not exact; use the Hanley-McNeil form.
def se_auroc(a, n_pos, n_neg):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a)
                      + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))


a = 0.82
n_pos, n_neg = 150, 850          # 15% prevalence, 1,000 cases
se = se_auroc(a, n_pos, n_neg)
z_alpha, z_beta = 1.96, 0.84
print()
print(f"SE(AUROC) at n=1000, 15% prevalence: {se:.4f}")
for delta in (0.05, 0.02):
    multiplier = ((z_alpha + z_beta) * se / delta) ** 2
    print(f"  detect a {delta:.2f} drop: {multiplier:.2f}x the sample, "
          f"about {multiplier*1000:.0f} labelled cases")
```

Expected output:

```text
target 0.820, shift 0.030, k 0.015, h 0.060
 month   auroc   cusum  alarm
--------------------------------
     1   0.821   0.000
     2   0.819   0.000
     3   0.815   0.000
     4   0.808   0.000
     5   0.799   0.006
     6   0.785   0.026
     7   0.772   0.059
     8   0.761   0.103  YES
     9   0.748   0.160  YES
    10   0.741   0.224  YES
    11   0.738   0.291  YES
    12   0.735   0.361  YES

first alarm month: 8
approximate ARL after shift: 4.0 months

SE(AUROC) at n=1000, 15% prevalence: 0.0217
  detect a 0.05 drop: 1.47x the sample, about 1475 labelled cases
  detect a 0.02 drop: 9.22x the sample, about 9218 labelled cases
```

Reading the output. The CUSUM stays at zero for the first four months because the deviations are
smaller than k, then accumulates and crosses the decision interval in month 8. The true degradation
begins around month 5, so the detection delay is about three months. That is the trade-off the
parameters encode: a smaller h would alarm earlier and fire more often on stable data; a larger k
would ignore small shifts and detect later.

The approximate average run length of 4.0 months after the shift is consistent with the observed
three-month delay. The formula is a rough guide, not a guarantee; the actual run length depends on
the variance of the metric, which the script does not model.

The sample-size calculation is the more important output for design. With 1,000 labelled cases and
15 percent prevalence, the standard error of AUROC is about 0.022. Detecting a 0.05 drop at 80
percent power requires about 1,475 labelled cases, which at 700 per month is roughly two months.
Detecting a 0.02 drop requires about 9,218 cases, or roughly thirteen months.

That difference is the design decision. A monitoring programme that wants to catch a large
degradation quickly can do so with monthly AUROC. A programme that wants to catch a small
degradation cannot, because the labels required exceed the period in which most recalls occur. For
small shifts, the programme must rely on the inner loop, on calibration metrics that use prevalence
rather than labels, or on a longer accrual window with a stated detection target. A programme that
claims to detect a 0.02 drop monthly with 700 cases per month is claiming a power it does not have.

## 8. Data, standards, and regulatory context

**FDA Good Machine Learning Practice.** The ten guiding principles published in October 2021 by the
FDA, Health Canada, and the MHRA include principle 10: deployed models should be monitored for
performance, and retraining risks should be managed. The principles are non-binding, and they do
not specify metrics, cadence, or thresholds. They establish the expectation that monitoring exists.

**Predetermined change control plans.** The FDA's final guidance on predetermined change control
plans for AI-enabled device software functions was issued in 2025. A PCCP describes planned
modifications, the methodology to develop and validate them, and the impact assessment, within a
marketing submission. The monitoring programme supplies the evidence that a modification performed
as predicted, and it is the mechanism by which a manufacturer knows whether the change stayed
inside the envelope.

**EU Medical Device Regulation.** The MDR requires a post-market surveillance system proportionate
to the risk class, a post-market surveillance report or periodic safety update report, and
post-market clinical follow-up for certain devices. The regulation specifies the data sources to
consider, including complaints, literature, and registries. It does not prescribe statistical
rules; those are for the manufacturer to justify.

**ISO 13485 and feedback.** Clause 8.2 requires monitoring and measurement of feedback, including
complaints, and clause 8.4 requires analysis of data to demonstrate the suitability and
effectiveness of the quality management system. A monitoring programme is the operational form of
these clauses for a model.

**IMDRF and software as a medical device.** The International Medical Device Regulators Forum
framework for software as a medical device includes post-market surveillance as part of the total
product life cycle, and it recognises that software changes frequently and that the risk of a
change must be assessed before release. The framework is guidance, not law, but it is the reference
point for many regulators.

**Data standards.** Monitoring depends on data being comparable over time. A change in a laboratory
assay, a coding system update (for example, a new ICD-10 release), or a FHIR version change can
create the appearance of drift when the underlying patients are unchanged. The monitoring pipeline
should record the vocabulary and schema versions alongside the metrics, so that a signal can be
attributed to a data change rather than a clinical one.

**Vigilance and medical device reporting.** A performance signal that suggests a risk to patients
may trigger a reporting obligation independent of the internal monitoring calendar. In the United
States, the Medical Device Reporting regulation requires manufacturers and importers to report
deaths, serious injuries, and certain malfunctions; in the EU, the vigilance provisions of the MDR
apply. The monitoring programme should have a defined handoff to the regulatory function, because
the reporting clock does not wait for the next monthly review. This is a legal determination and
requires professional advice.

**Contracts and data rights.** Monitoring rights are contractual. A deployment agreement should
specify which data the buyer receives, at what grain, at what frequency, and for what purpose, and
whether the vendor will supply model updates and their validation evidence. A contract that grants
only aggregate quarterly reports limits the programme to the outer loop and leaves the buyer unable
to compute local performance.

**What the standards do not require.** They do not specify a metric set, a cadence, a statistical
process control method, or a threshold. They require that the manufacturer define these and be able
to justify them. That is why the design rationale in Step 5 matters as much as the numbers.

## 9. Industry practice and case evidence

**Documented fact: early recalls.** Lee and colleagues (*JAMA Health Forum* 2025,
doi:10.1001/jamahealthforum.2025.3172) found 60 of 950 cleared AI-enabled devices (6.3 percent)
linked to 182 recalls, with 79 (43.4 percent) within 12 months of clearance and diagnostic or
measurement error the most common cause (109 recalls). Lack of clinical validation was associated
with recall (odds ratio 2.8, 95 percent CI 1.6 to 4.7). The inference for monitoring design: the
highest-risk window is the first year, so the cadence should be highest then.

**Documented fact: performance gaps at deployment.** Wong et al. (2021) found an AUROC of 0.63 for
a widely deployed proprietary sepsis model against a vendor range of 0.76 to 0.83, with an alert
rate of 18 percent and a PPV of 12 percent. The inference: a deployed model can perform materially
below its reported range in a new setting, and the gap is detectable with local data.

**Documented fact: drift detection methods.** Kore et al. (*Nature Communications* 2024) tested
label-free drift detectors on 239,000 chest radiographs and found that aggregate performance
metrics were a poor proxy for data drift: performance stayed stable while drift was present. Koch
et al. (*npj Digital Medicine* 2024) evaluated shift-detection methods on 130,000 retinal images
and found classifier-based tests performed best overall, detecting some shifts with as few as 500
images and others requiring more than 30,000. The inference: input-based detection is complementary
to performance monitoring and can be much faster, but its sensitivity depends on sample size and
on which feature shifted.

**Documented fact: label-free detection limits.** Kore et al. found that detectors were more
sensitive to some patient features than others and that individual-class metrics outperformed
aggregated metrics. Koch et al. found that the best method detected a patient-sex shift only with
more than 30,000 images, while easy out-of-distribution shifts were detected with 300. The
inference for a monitoring plan: a detector that works on a large academic dataset may be
underpowered in a smaller hospital, and the plan should estimate the sample size required for the
shift it cares about rather than importing a method and assuming it works.

**Inference: why monitoring programmes fail.** Three patterns. First, the programme measures
everything and decides nothing, because no metric has an owner or a threshold. Second, the
programme measures only AUROC, which requires labels and therefore arrives after the harm. Third,
the programme is built by the vendor and operated by no one, because the contract did not assign
responsibility.

**Inference: what a workable programme looks like.** Five to nine metrics, each with a baseline, a
rule, a cadence, an owner, and an action. A monthly meeting with authority to act. A weekly check of
input quality and alert burden. An annual review of the metric set itself, because the risks change.
The programme is small, and its value is in the decisions it produces, not the charts it displays.

**Inference: the commercial angle.** Vendors that supply a monitoring specification and a
performance dashboard shorten procurement and reduce the risk of a public performance gap. Buyers
that require one get a contractual basis for action when performance changes. The specification is
cheap to produce and is increasingly expected in diligence.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No baseline | Thresholds set from the vendor's paper | No local reference | Alarms are arbitrary | Measure before go-live |
| AUROC-only monitoring | Dashboard has one chart | Labels lag | Late detection | Add input and usage metrics |
| No owner | Red cell persists for months | Diffuse responsibility | No action | Named owner per metric |
| No authority | Review meeting cannot decide | Governance gap | Signals escalate nowhere | Assign suspension authority |
| Threshold without rationale | "We chose 0.80" | Convention | Wrong alarm rate | Justify by consequence |
| Cadence slower than harm | Annual review | Convenience | Detection after harm | Match cadence to risk |
| Silent pipeline failure | Dashboard shows last value | No freshness check | Monitoring stops silently | Freshness alarm |
| Labels biased | Only followed-up cases labelled | Selective ascertainment | Performance estimate wrong | Report completeness |
| Prevalence change unnoticed | Alert rate rises, AUROC stable | Label shift | PPV falls | Monitor alert rate and calibration |
| Override rate ignored | Clinicians dismiss alerts | Automation bias or irrelevance | Model ineffective | Track and investigate |
| Subgroup not monitored | Aggregate stable | Masking | Inequity undetected | Prespecified subgroup metrics |
| Change without re-baseline | Recalibration deployed | Old baseline retained | False alarms or missed shift | Re-baseline after change |
| Vendor-only monitoring | No local data access | Contract gap | Buyer cannot verify | Contract data rights |
| Report selects metrics | Only improvements shown | Reporting bias | False assurance | Publish full metric set |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Labels arrive with acceptable latency | Performance monitoring needs outcomes | Claims lag or are incomplete | Measure label completeness and delay |
| The baseline is stable | Thresholds depend on it | Launch effects, new sites | Exclude launch period; test stability |
| Inputs are comparable over time | Drift detection assumes a reference | Coding or assay change | Record schema and vocabulary versions |
| The metric distribution is known | Control limits need variance | Small samples | Use intervals; widen limits |
| Actions are available | Monitoring informs decisions | No recalibration path | Confirm change-control route |
| Clinicians use the model as intended | Usage metrics interpret behaviour | Workarounds | Observe workflow; audit overrides |
| Subgroup sizes are adequate | Equity monitoring needs power | Rare groups | Report n; pool where justified |
| The monitoring pipeline is reliable | Silent failure defeats the programme | No freshness check | Alert on data staleness |
| Governance survives turnover | Owners are roles, not people | Reorganisation | Name roles in the plan |
| A signal implies a cause | Investigation is feasible | Multiple concurrent changes | Test causes in order |

## 12. Exercises

**Exercise 1. Classify the shift (application, 20 minutes).**
Deliverable: five scenarios classified as covariate, label, concept, pipeline, or workflow shift,
with the first metric expected to move. Acceptance criteria: each classification is justified by
what changed; no scenario is left unclassified. Time: 20 minutes.

**Exercise 2. Design a metric set (design, 30 minutes).**
Deliverable: a nine-metric table with layer, definition, source, cadence, and owner. Acceptance
criteria: at least two leading indicators; every metric has a computable source; no metric lacks an
owner. Time: 30 minutes.

**Exercise 3. Set a monitoring rule (application, 30 minutes).**
Deliverable: a CUSUM or control-chart rule for one metric, with the target, the shift to detect, and
the resulting false-alarm and delay expectations. Acceptance criteria: parameters are justified;
the trade-off is stated. Time: 30 minutes.

**Exercise 4. Compute a CUSUM by hand (application, 25 minutes).**
Deliverable: a hand calculation of a CUSUM for six monthly values, with the alarm point. Acceptance
criteria: arithmetic shown; the comparison with the script output matches. Time: 25 minutes.

**Exercise 5. Write the escalation path (design, 25 minutes).**
Deliverable: a role-based escalation path from signal to suspension, with time limits. Acceptance
criteria: each step names a role and a deadline; the suspension authority is explicit. Time: 25
minutes.

**Exercise 6. Investigate a signal (critique, 40 minutes).**
Deliverable: an investigation report for a simulated alert-rate rise, testing pipeline, population,
workflow, and model causes in order. Acceptance criteria: each cause is tested with evidence; the
attribution is stated with residual uncertainty. Time: 40 minutes.

**Exercise 7. Appraise a monitoring plan (application, 45 minutes).**
Deliverable: a six-domain score for a real or constructed monitoring plan, with findings and a
recommendation. Acceptance criteria: scores cite evidence; at least one domain is zero; the
recommendation names the missing control. Time: 45 minutes.

**Exercise 8. Build the dashboard specification (design, 35 minutes).**
Deliverable: a one-page specification listing panels, metrics, reference lines, thresholds, and
ownership, plus a freshness indicator for each data feed. Acceptance criteria: every panel maps to
a decision; each metric shows a baseline and a rule; staleness is visible. Time: 35 minutes.

**Exercise 9. Design the annual review (design, 30 minutes).**
Deliverable: an agenda and a decision log template for the annual review of the monitoring
programme itself. Acceptance criteria: the review covers metric relevance, thresholds, incidents,
and governance; decisions are recorded with owners. Time: 30 minutes.

## 13. Self-check answers

**A1. What are the three types of dataset shift?** Covariate shift (inputs), label shift
(prevalence), and concept shift (input-outcome relationship).

**A2. Why is AUROC a lagging indicator?** Because it requires labels, which arrive after the
outcome, often weeks or months later.

**A3. What is the monitoring triangle?** Signal, threshold, and action. A metric without all three
is a dashboard number, not a control.

**A4. What does a CUSUM accumulate?** The cumulative sum of deviations from a target, in units
related to the shift to be detected. It alarms when the sum crosses a decision interval.

**A5. Why does a smaller decision interval increase false alarms?** Because the cumulative sum
crosses it more easily under normal variation, so stable periods also trigger.

**A6. A model's AUROC is stable but the alert rate has doubled. What is the likely cause?** Label
shift (higher prevalence) or a threshold change, not a loss of discrimination. Check calibration and
case mix.

**A7. Why must a baseline be local?** Because performance depends on the population, prevalence, and
workflow. A vendor's baseline describes a different deployment.

**A8. What is the difference between monitoring and evaluation?** Monitoring is continuous and cheap
per observation; evaluation is periodic, designed, and expensive. A signal is a reason to evaluate.

**A9. What should happen after a recalibration?** A change record, revalidation as required, and a
new baseline established from the post-change period.

**A10. Why is the first year the highest-risk period?** Because 43.4 percent of AI device recalls
occur within 12 months of clearance, and deployment exposes the model to a new population and
workflow immediately.

**Rubric for Exercise 7.** Full credit: six domains scored with evidence; at least one zero; the
recommendation names the missing control and its owner. Partial: scores without evidence.

**Rubric for Exercise 6.** Full credit: causes tested in a sensible order; each test described;
attribution stated with uncertainty; a decision proposed. Deduction for jumping to "the model
drifted" without testing the pipeline.

## 14. Interview questions

**Q1. How would you monitor a deployed clinical model?**
Strong answer: three loops; input and usage metrics weekly, performance monthly with labels,
outcomes periodically; each metric with a baseline, rule, owner, and action. Weak answer: says
"track AUROC."

**Q2. What is the difference between covariate and concept shift?**
Strong answer: inputs versus input-outcome relationship; covariate shift can be detected without
labels, concept shift usually requires them; the remedies differ. Weak answer: treats all drift as
the same.

**Q3. Why not just monitor accuracy?**
Strong answer: labels lag, so accuracy monitoring detects late; leading indicators catch pipeline
and workflow problems sooner. Weak answer: says accuracy is sufficient.

**Q4. How do you set a threshold?**
Strong answer: from the baseline distribution, the cost of a false alarm versus a missed shift, and
a stated detection target; use a control chart or CUSUM rather than a round number. Weak answer:
picks a round number.

**Q5. A dashboard shows a red metric for six months. What went wrong?**
Strong answer: no owner, no authority, or no action defined; the signal was not connected to a
decision. Weak answer: blames the data.

**Q6. How do you detect a problem before labels arrive?**
Strong answer: input distribution tests, missingness, schema checks, alert rate, override rate, and
calibration proxies; notes sensitivity depends on sample size. Weak answer: says it is impossible.

**Q7. What do you do when a subgroup signal appears?**
Strong answer: checks n and interval, confirms the definition, investigates measurement and
population causes, and considers restriction or mitigation; does not over-read a small cell. Weak
answer: immediately announces inequity or dismisses it.

**Q8. How does a monitoring finding become a CAPA?**
Strong answer: the signal triggers an investigation, the cause is established, the action goes
through change control, and the effectiveness check measures the metric after the change. Weak
answer: says the model is retrained.

**Q9. What is a PCCP and why does it need monitoring?**
Strong answer: a pre-approved envelope for planned changes; monitoring verifies that each change
behaves as predicted and stays inside the envelope. Weak answer: confuses it with a software
update.

**Q10. How often should a model be reviewed?**
Strong answer: cadence follows the risk and the latency of the metric; weekly for leading
indicators, monthly for performance, annually for the programme design. Weak answer: gives a fixed
answer for all models.

**Q11. What would you put in a vendor monitoring specification?**
Strong answer: metrics and definitions, expected ranges, recommended thresholds, data requirements,
reporting cadence, and escalation contacts. Weak answer: says a dashboard.

**Q12. When would you withdraw a model?**
Strong answer: when a safety signal or a confirmed material performance gap cannot be mitigated
quickly, with a communication and migration plan. Weak answer: says when accuracy drops.

## 15. Key terms

- **Post-market surveillance** — ongoing data collection after market entry.
- **Post-market clinical follow-up** — prospective clinical data after CE marking.
- **Drift** — change in data or performance after deployment.
- **Covariate shift** — change in input distribution.
- **Label shift** — change in outcome prevalence.
- **Concept shift** — change in the input-outcome relationship.
- **Calibration drift** — change in risk accuracy.
- **Alert burden** — alerts per unit of clinical work.
- **Override rate** — share of alerts dismissed.
- **Leading indicator** — signal that moves before the outcome.
- **Control chart** — time series with limits.
- **CUSUM** — cumulative sum change detector.
- **Detection delay** — time from shift to signal.
- **False-alarm rate** — signals during stable periods.
- **Silent mode** — running without clinical action.
- **Real-world evidence** — evidence from routine care.
- **PCCP** — predetermined change control plan.
- **Baseline** — reference performance for monitoring.
- **Escalation** — defined path from signal to authority.
- **Re-baselining** — resetting the reference after a change.

## 16. Sources

- Lee B, Kramer D, Sandri F, et al. Early recalls and clinical validation gaps in artificial
  intelligence-enabled medical devices. *JAMA Health Forum* 2025.
  doi:10.1001/jamahealthforum.2025.3172. Accessed 26 October 2026.
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
  sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*
  2021;181(8):1065-1070. doi:10.1001/jamainternmed.2021.2626. Accessed 26 October 2026.
- Kore A, et al. Empirical data drift detection experiments on real-world medical imaging data.
  *Nature Communications* 2024. doi:10.1038/s41467-024-46142-w. Accessed 26 October 2026.
- Koch S, Baumgartner M, Berens P. Distribution shift detection for the postmarket surveillance of
  medical AI algorithms: a retrospective simulation study. *npj Digital Medicine* 2024.
  doi:10.1038/s41746-024-01085-w. Accessed 26 October 2026.
- US Food and Drug Administration, Health Canada, Medicines and Healthcare products Regulatory
  Agency. Good Machine Learning Practice for Medical Device Development: Guiding Principles.
  October 2021. https://www.fda.gov/media/153486/download. Accessed 26 October 2026.
- US Food and Drug Administration. Marketing Submission Recommendations for a Predetermined Change
  Control Plan for Artificial Intelligence-Enabled Device Software Functions. Final guidance, 2025.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents. Accessed 26 October
  2026.
- ISO 13485:2016, clauses 8.2 and 8.4. Geneva: ISO; 2016.
- International Medical Device Regulators Forum. Software as a Medical Device: Possible Framework
  for Risk Categorization and Corresponding Considerations. 2014. Accessed 26 October 2026.

> **Note:** Regulatory content is educational and is not legal advice. Reporting timelines for
> safety events are governed by the applicable jurisdiction.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_quality-management.md`](06_quality-management.md)
- Next topic: [`08_audit.md`](08_audit.md)
- Appraisal: [`03_appraising-clinical-ai-evidence.md`](03_appraising-clinical-ai-evidence.md)
- Module survey: [`../../course/18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md)
- Labs: [`../../labs/`](../../labs/)
