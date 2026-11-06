# Ongoing Monitoring and Revalidation Triggers

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 09 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 07, and 25; see the bridge below  
> Version 2026.5

## 1. Why this topic exists

A model approved in January does not become permanently safe in February.
Its input feeds, customers, suppliers, product mix, constraints, decision process, and operators can change while the model version remains unchanged.
A validation report establishes what evidence supported a particular use at a particular time.
Monitoring tests whether the conditions for that use remain credible.
Revalidation provides an independent, appropriately scoped challenge when a scheduled review or a material change warrants one.

Consider a weekly demand forecast used to propose component purchases.
An apparently stable aggregate error can hide a new underforecast at the constrained supplier that determines whether a factory can ship.
A delayed actuals feed can make a dashboard appear unusually quiet.
A new promotion policy can change the meaning of the inputs without changing their column names.
A supplier closure may make the old forecast-to-order conversion unsafe even if the forecast remains accurate.
A monitoring program must detect these different problems and route them to people who can act.

The business question is not simply whether a statistic has crossed a threshold.
It is whether an authorized decision can still rely on this particular model, for this population, horizon, and action, given current evidence.
That decision may be to continue, restrict use, switch to a controlled fallback, or stop.
It must identify the owner, the rationale, and the date at which the decision expires or will be reconsidered.

Monitoring is a recurring operational control.
Incident response handles a suspected or realized harmful event.
Revalidation reassesses model suitability with independent challenge.
The three interact, but none is a substitute for the others.
A model can need revalidation without a detected incident; an incident can require containment before the revalidation report is ready.

### Prerequisite bridge

Module 06 is deferred in this edition.
Here, a model is a quantitative method that transforms inputs into estimates or recommendations for a decision.
Training selects model parameters or a method from data; evaluation tests a claim using appropriately separated evidence.
A model version identifies the method, parameters, input contract, implementation, and approved use at a specific time.
A holdout is data reserved from model selection; a forecast holdout must also respect what was known at each forecast origin.
An assumption is a condition supporting a conclusion; a limitation is a known boundary of that conclusion.
Module 07 supplies forecasting and temporal evaluation; Module 25 supplies business value and counterfactual reasoning.
This lesson explains the monitoring concepts it needs without assuming that the deferred module has been completed.

A forecast estimates an uncertain outcome.
A plan describes intended action, a target describes a desired outcome, and a commitment is an authorized obligation.
A forecast can inform a purchase plan but does not itself authorize a purchase order.
The examples use fictional organizations, illustrative thresholds, and hypothetical data unless otherwise stated.
They are educational rather than legal, regulatory, or procurement advice.

## 2. Learning outcomes

After completing this lesson, you can:

1. Map an approved model use to a monitor, evidence owner, cadence, and decision authority; complete Exercise 1.
2. Distinguish input, prediction, outcome, and concept drift from data-pipeline failure; complete Exercises 2 and 8.
3. Calculate a drift statistic, error, bias, and an action threshold using explicit denominators; complete Exercises 3 and 4.
4. Choose proportionate thresholds and handle small samples, delayed labels, and repeated testing; complete Exercises 4 and 5.
5. Triage a market change or incident and select containment, escalation, and restoration criteria; complete Exercise 6.
6. Define scheduled and event-triggered revalidation, including independent challenge and scope; complete Exercises 6 and 7.
7. Write an auditable action record that separates a model assessment from business risk acceptance; complete Exercise 7.
8. Explain the limits of an observed metric, a public case, and an apparently successful fallback; complete Exercises 5 and 8.

A complete answer identifies what is measured, which decision population it covers, when the information becomes available, and what action follows.
It also explains what would make the measurement misleading.

## 3. The core mental model

```text
Approved use + model/version + baseline + limits + accountable owner
                              |
                              v
          Observe inputs, outputs, outcomes, process, and environment
                              |
                              v
           Data quality and coverage gates; labels may arrive later
                              |
                              v
     Compare like with like; review threshold, persistence, and impact
                              |
               +--------------+---------------+
               |                              |
          Within limits                Breach, incident, or change
               |                              |
       Record evidence                    Triage and contain
               |                              |
               |                    Owner decides safe use now
               |                              |
               |               Independent review when trigger met
               |                              |
               |                   Findings and remediation
               |                              |
               +--------<--- Controlled restoration and closure
                              |
                              v
                 Scheduled revalidation also enters review
```

The monitoring loop starts with the approved use, not with whichever metrics happen to be available.
A forecast for a one-week replenishment decision and a forecast for a twelve-week capacity decision can share a model yet need different thresholds and different owners.
The first may be sensitive to daily feed latency; the second may depend more on supplier capacity and structural demand changes.

A useful monitoring system has five linked views.
The input view asks whether data arrive on time, are complete, and resemble the intended population.
The prediction view asks whether forecast distributions, recommendations, and constraint activations behave as expected.
The outcome view compares predictions with subsequently observed outcomes using a documented lag and denominator.
The operational view tracks overrides, incidents, fallback use, and missed control execution.
The environment view records market, supplier, product, legal, and policy changes that may undermine assumptions before labels arrive.

None of these views alone establishes model fitness.
Stable inputs can coexist with changing relationships between inputs and outcomes.
High aggregate accuracy can coexist with severe harm in a small critical segment.
A distribution shift can be harmless if the model remains well calibrated and the decision is robust.
The appropriate response depends on evidence, impact, and uncertainty rather than on a single score.

Separate detection from authorization.
An automated job may raise an alert and open an incident, but an accountable person must decide what to do within delegated authority.
An independent reviewer challenges model suitability and remediation; the business owner accepts or rejects the residual operational risk.
Internal audit may later assess the control process, but it is not the everyday model owner or a substitute for independent validation.

The loop is closed only when the organization verifies that an action was completed and effective.
Changing a dashboard color or retraining a model is not, by itself, evidence of restored fitness.
Keep the original alert, the investigation, the decision, any exception, and the closure evidence linked to the exact model version and affected decisions.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Approved use | Authorized decision, population, horizon, and constraints | Treating approval as permission for every use |
| Baseline | Frozen comparison population and period, with version and selection rationale | Replacing it silently after performance worsens |
| Reference window | Historical observations used for comparison, in dates and decision units | Assuming any convenient history is representative |
| Current window | Explicitly bounded production observations, in the same decision units | Comparing an incomplete week with a full historical week |
| Input drift | Change in an input distribution, such as lead time in days | Assuming drift proves that predictive quality declined |
| Prediction drift | Change in model output distribution, such as units per forecast | Treating output change as proof of failure |
| Outcome drift | Change in actual outcomes, such as demand units per week | Assuming observed sales equal unconstrained demand |
| Concept drift | Change in the relationship between inputs and outcome | Calling any input distribution change concept drift |
| Data-quality incident | Defect in timeliness, completeness, validity, lineage, or semantics | Labeling a broken feed as a market change |
| Performance degradation | Deterioration in a specified performance measure on comparable cases | Using an unqualified aggregate score |
| Bias | Signed error under an explicitly declared sign convention | Confusing bias with absolute error |
| Weighted absolute percentage error (WAPE) | Sum absolute errors divided by sum actuals, dimensionless or percent | Averaging item-level percentage errors and calling it WAPE |
| Population stability index (PSI) | Sum of bin-share differences times log share ratios, dimensionless | Treating conventional cutoffs as universal significance tests |
| Segment | Defined subset such as supplier, product class, or horizon | Selecting only the segment that confirms a preferred story |
| Materiality | Decision impact and exposure relevant to the approved use | Equating materiality solely with observation count |
| Threshold | Predefined condition for a response, including units and window | Assuming a threshold is an automatic causal diagnosis |
| Persistence | Number or duration of consecutive qualifying observations | Requiring persistence even for an immediate safety incident |
| Alert | Signal requiring the specified investigation or action | Treating delivery of an email as completed control |
| Incident | Suspected or realized event requiring managed response | Restricting incidents to proven model defects |
| Near miss | Event that could have caused harm but did not, perhaps due to another control | Excluding it because realized loss was zero |
| Escalation | Routing to a person with appropriate authority and deadline | Forwarding a message without confirming ownership |
| Containment | Immediate action limiting exposure while facts remain uncertain | Waiting for root cause before preventing further harm |
| Fallback | Preapproved alternative decision process with operating conditions | Assuming an untested manual process is automatically safe |
| Revalidation | Independent reassessment of suitability for the specified use | Calling a developer's routine dashboard review independent validation |
| Scheduled review | Review due at a risk-based interval regardless of alerts | Postponing indefinitely because no alert fired |
| Event trigger | Condition that starts a review or assessment outside the schedule | Assuming every event requires the same review scope |
| Change control | Authorization and evidence process for a version or use change | Treating a vendor's silent release as no change |
| Independence | Sufficient distance, authority, expertise, and freedom from relevant conflicts | Equating independence with an external job title |
| Finding | Documented issue, evidence, impact, owner, and closure criteria | Recording a concern without assigning action |
| Residual risk | Remaining exposure after controls and remediation | Treating it as zero because a finding is closed |
| Exception | Time-limited authorized deviation with compensating controls | An undocumented permanent bypass |
| Label maturity | Point when the outcome is sufficiently complete for a specified comparison | Treating late or censored outcomes as final |
| Censoring | Outcome not fully observable, for example sales capped by stockouts | Treating observed sales as true demand without qualification |
| Override | Authorized human modification of a model output or recommendation | Assuming every override improves outcomes |
| Challenger | Alternative method evaluated on the same relevant evidence | Comparing against a retrospectively chosen perfect alternative |
| Audit trail | Linked, immutable or controlled records of evidence and decisions | Keeping only the current dashboard state |

## 5. The mechanics, step by step

### Step 1. Define the governed decision

**Input:** inventory record, approval, decision workflow, and current model card.
**Output:** a use-specific monitoring mandate.

Record the model identifier, version, owner, business decision owner, deployment date, and approved population.
Specify the prediction horizon, frequency, unit, geography, decision cutoff, and downstream action.
Identify whether outputs are advisory, automatically acted on, or used within a human approval process.
Record the actual authority that can place an order, modify a plan, or suspend automated use.
The same numerical error can have different consequences depending on lead time, inventory, supplier constraints, and decision reversibility.

List material exposures.
Examples include stockout risk, excess and obsolete inventory, expedited freight, contractual obligations, service-level harm, and disproportionate impact on a critical customer.
Separate a causal claim of realized loss from an exposure estimate or a scenario.
If financial impacts overlap, do not add them as if they were independent losses.

Define an accountable owner for model performance, an operational owner for each feed, an incident commander, an independent validator, and the business authority for residual-risk decisions.
One person may hold several compatible roles in a small organization, but the reviewer must not validate their own material development or remediation work.
Conflicts and compensating arrangements require explicit documentation and escalation.

**Branch:** if the use, owner, or action authority is unknown, do not assume monitoring makes the use acceptable.
Escalate the governance gap and apply the organization's approved restrictions while the gap is resolved.

### Step 2. Establish reference evidence

**Input:** prior validation, historical data, seasonal calendar, and model change history.
**Output:** frozen baseline and a comparison plan.

Choose a reference period representative of the approved use, including relevant peak and non-peak periods when feasible.
Record its dates, version, data availability, exclusions, segment definitions, and the decision population.
A baseline from an earlier model version can be informative, but document changes that make comparison imperfect.
Avoid using a period selected after seeing the current alert merely because it makes the comparison favorable.

Where demand is seasonal, compare like weeks, matching events, or a documented seasonal expectation.
A rolling reference adapts to gradual changes but can normalize a persistent deterioration.
Keep a fixed approved reference alongside any rolling comparator, and explain why each exists.
If the business genuinely changes its population, explicitly approve a new baseline rather than overwriting history.
Retain the old baseline to reconstruct earlier decisions.

For every measure define numerator, denominator, units, grain, treatment of missing values, minimum sample size, and data latency.
Record the formula and code version where relevant.
Freeze bin boundaries for distribution measures before seeing the comparison period.
A calculated value is only comparable if inputs and definitions have not silently changed.

### Step 3. Instrument the input path

**Input:** data contracts and production records at the time decisions were made.
**Output:** timely data-quality and coverage evidence.

Monitor feed arrival time against the decision deadline, not only against an overnight batch schedule.
Measure missingness by field and by critical segment.
Track unexpected units, values outside plausible bounds, duplicate keys, schema changes, and broken joins.
Record whether a missing value is genuinely missing, imputed, or carried forward.
A complete table can still contain stale or semantically incorrect values.

Measure population coverage: expected eligible item-location-weeks compared with received item-location-weeks.
A low aggregate missing rate can conceal the absence of every record from one critical supplier.
Check whether the population used for the denominator itself is current and complete.
Alert on both rate and absolute affected exposure where possible.
For example, 1 missing record out of 100 may be material if that record represents the only source of a constrained part.

Compare distributions for major drivers, using business-relevant bins or suitable statistics.
Track both total volume and shares; a stable share can conceal a large absolute volume change.
Preserve enough information to determine whether a shift reflects real behavior, a feed defect, a unit conversion, a new population, or a changed feature definition.
A schema-conformant feed may still encode a different business meaning after a system migration.

**Branch:** if a data gate fails, distinguish invalid data from a valid but changed distribution before interpreting the model metric.
Contain the affected scope promptly; do not silently impute or discard affected cases to make a dashboard green.

### Step 4. Monitor outputs and decision behavior

**Input:** timestamped forecasts, recommendations, overrides, constraints, and actual actions.
**Output:** evidence about prediction and workflow stability.

Track forecast level, spread, zero rate, unusual jumps, and segment coverage at the approved horizon.
For probabilistic outputs, check interval width and calibration once outcomes mature.
A narrower interval is not automatically better if it misses outcomes more frequently.
A jump in forecasts can be a correct response to a promotion or an artifact of duplicated inputs.
Investigate rather than classify it from the output alone.

Measure recommendation adoption and override rates by reason and segment.
Capture the pre-override value, the revised value, who changed it, when, and why.
Compare subsequent outcomes with both values when the comparison is meaningful.
Overrides are selected by humans and are not randomized; a simple comparison cannot establish the causal value of the human or model.
A falling override rate may indicate improved confidence, or it may reflect reduced staffing and rubber-stamping.

Monitor downstream constraints separately from predictive metrics.
An accurate demand forecast can produce an infeasible plan when supplier capacity, minimum order quantity, shelf life, or transport lead time changes.
Track feasibility violations, emergency orders, and use outside the authorized population.
Do not treat a model's prediction as a committed order or assign responsibility for an unauthorized order to the algorithm.

### Step 5. Monitor mature outcomes

**Input:** frozen forecasts at origin and appropriately mature observed outcomes.
**Output:** comparable outcome evidence by horizon and segment.

Join each forecast to the corresponding outcome by item, location, target period, horizon, and version.
Retain the forecast as it existed before the decision; a later revised forecast must not replace it.
Define the availability lag and a watermark beyond which labels are considered mature for the specified measure.
Publish the share of expected labels that have matured.
A metric from a small, nonrepresentative subset of early outcomes is not a complete performance result.

Use complementary metrics.
For forecasts \(f_i\) and actuals \(a_i\), WAPE is \(\sum_i |f_i-a_i|/\sum_i a_i\), provided the denominator is positive.
Under the convention used here, signed bias is \(\sum_i(f_i-a_i)/\sum_i a_i\); positive means overforecast and negative means underforecast.
Absolute and signed measures answer different questions.
Evaluate the planning cost of errors where an agreed cost model and counterfactual are available, without presenting a scenario as an observed causal loss.

Report counts, actual volume, and missing-label share with every metric.
Assess aggregate performance alongside prespecified critical segments, decision horizons, and relevant seasons.
Where actual sales were constrained by stockouts, observed sales may understate demand; label the metric as error against sales, not unconstrained demand.
Do not manufacture unavailable demand labels.
Use inventory and lost-sales evidence to flag the interpretation limit and define a separate estimation exercise if needed.

Compare against a feasible benchmark, such as a seasonal-naive forecast, using the same forecast origins and mature cases.
A model can worsen because the environment is harder while still outperforming a feasible benchmark.
A model can also look stable while a simpler benchmark improves markedly.
Neither comparison alone authorizes continued use without considering decision cost and exposure.

### Step 6. Set thresholds and response rules

**Input:** historic variation, materiality, response capacity, and validation findings.
**Output:** a versioned threshold and action matrix.

Thresholds should specify the measure, population, horizon, comparison window, minimum count, calculation time, severity, owner, and action deadline.
Set them before routine monitoring and document the rationale.
Use historical variability and backtesting to understand how often a proposed rule would alert, but do not treat historical alert frequency as a guarantee.
Choose tolerances that reflect harm and operating capacity, not merely the desire for a quiet dashboard.

A warning can request investigation without immediately stopping use.
A critical data-integrity failure or a severe realized harm may require immediate containment.
Persistence can reduce noise for gradual statistical changes, but waiting for two periods is inappropriate when a single event can create irreversible exposure.
Specify whether equality at a boundary is a breach, whether the rule uses absolute or relative change, and which period closes the measurement window.

Use minimum sample sizes and report uncertainty.
For a small critical segment, suppressing alerts until a large sample accumulates can conceal a serious issue.
Use an exposure-based or qualitative escalation as well as the statistical rule.
Multiple segments and repeated weekly tests increase the chance of apparently exceptional results.
Group related signals, evaluate the history and business context, and avoid interpreting each unadjusted alert as a stand-alone hypothesis test.
Where formal inference is required, predefine the method and consider multiplicity and serial dependence.

A dashboard with no current metric because data failed to arrive must show unknown or failed, not green.
Define a separate control for monitor execution, freshness, and alert delivery.
A human must acknowledge unowned or overdue alerts, and escalation must continue if the first recipient is unavailable.
Thresholds are illustrative policy choices, not universal regulatory requirements.

### Step 7. Detect market and operating changes

**Input:** market intelligence, procurement events, operations changes, and model-change notices.
**Output:** logged change assessment and decision on event-triggered review.

Monitor known external and internal events even if the model is statistically stable.
Examples include a supplier bankruptcy, port closure, tariff change, sudden competitor exit, promotion redesign, new product introduction, or change in the order allocation policy.
Record source, date learned, affected products, expected duration, uncertainty, and owner.
Separate a confirmed event from a rumor and a scenario from an observed outcome.

Ask whether the event changes the model's input availability, causal relationships, decision costs, feasible actions, or approved population.
A tariff may leave demand prediction unchanged while changing the economics of the purchase recommendation.
A new supplier can change lead-time distributions and the meaning of a supplier identifier.
A new promotion mechanism can invalidate historical price-response relationships even before a measurable output shift appears.

A qualitative trigger can require an immediate impact assessment; that assessment then determines whether independent revalidation is necessary.
For a predefined material event, policy may require independent revalidation directly.
Record the rationale if a potentially material change is assessed as not requiring it, including the evidence and reassessment date.
Do not wait for lagged labels when the decision horizon makes the exposure immediate.

### Step 8. Triage and contain incidents

**Input:** alert, human report, market event, or near miss.
**Output:** incident record and temporary operating decision.

Capture detection time, event time if known, affected version, scope, decisions at risk, reporter, evidence, and initial severity.
Assign an incident commander and an accountable business decision owner.
Acknowledge within the service level defined by policy and escalate if no owner responds.
Preserve raw records, feature snapshots, forecasts, recommendations, overrides, and action logs before correction.

First prevent additional harm within delegated authority.
Possible actions include suspending automated order release, restricting a segment, imposing a manual approval gate, holding new commitments, or switching to a preapproved fallback.
A fallback must have a named owner, inputs, capacity, authorization, and monitoring.
If the fallback is unavailable or infeasible, escalate for an explicit safe operating decision rather than silently treating it as available.
Communicate the operational impact to procurement, operations, and affected partners under the organization's incident policy.

Investigate competing explanations: data defect, deployment change, new environment, incorrect assumptions, decision integration, and human process.
Do not assume the model caused every downstream loss.
Where an outcome is still developing, distinguish known facts from estimates and schedule an update.
A severe incident may require review of similar models or locations using the same data feed.

### Step 9. Decide on revalidation

**Input:** incident facts, change assessment, validation schedule, and risk tier.
**Output:** documented scope, timing, reviewer assignment, and interim use conditions.

A scheduled revalidation checks whether approval remains supported even when no alerts have fired.
The interval should be risk-based, approved in policy, and tracked from the relevant approval or last completed review.
The example below uses an annual interval for its fictional high-impact model; it is not a universal requirement.
An overdue review is a governance exception, not evidence that the model is necessarily defective.
The business owner must decide whether continued use is permitted under a time-limited exception and compensating controls.

Event triggers can include a material input or methodology change, a move to a new population, sustained performance degradation, severe incident, significant market change, loss of a key control, or a vendor release with material effects.
A single breach may trigger investigation, while a severe event or sustained breach triggers independent revalidation under the local policy.
Explicitly identify which events are automatic triggers and which require an impact assessment.
A broad model redesign may require a full pre-deployment validation rather than a narrow review.

The review scope should address the reason for the trigger and other assumptions plausibly affected by it.
For a data-feed defect, test the corrected feed, historical impact, reconciliation, and controls, not just a new accuracy score.
For market change, consider representativeness, conceptual assumptions, outcomes, and the decision process.
For a new use, assess the new population and consequences rather than transferring the old approval unchanged.
A narrow scope is justified only when the reviewer records why excluded areas are unaffected.

### Step 10. Conduct independent challenge

**Input:** frozen evidence, trigger, prior findings, current version, and proposed scope.
**Output:** independent opinion, findings, and limitations.

Select reviewers with relevant technical and business expertise who did not materially develop the component or fix they review.
Identify organizational, financial, prior-work, and reporting conflicts.
Independence requires authority to obtain evidence and escalate disagreement, not simply a different title.
If adequate independence or competence cannot be obtained internally, use a qualified independent resource or escalate the resulting limitation.
An external reviewer must also disclose conflicts and must have sufficient access.

Review the approved use, assumptions, monitoring history, incidents, overrides, data lineage, version changes, and previous findings.
Reproduce selected calculations using decision-time data where feasible.
Challenge the chosen metrics, benchmarks, segments, thresholds, handling of missing labels, and treatment of market events.
Assess conceptual soundness and decision consequences where relevant to the trigger.
The reviewer records evidence not available and what that omission prevents them from concluding.

Record findings with severity, evidence, impact, owner, due date, and measurable closure criteria.
Distinguish the independent validation opinion from the business owner's decision about whether to accept residual risk.
A reviewer can recommend restriction or rejection, but the authorized governance process makes the operating decision.
A disagreement must be escalated without editing away the independent opinion.

### Step 11. Remediate, restore, and close

**Input:** findings, incident root-cause evidence, and approved remediation plan.
**Output:** verified controls and controlled return to approved use, or retirement.

For each action assign an owner, deadline, dependency, test, and independent verifier where appropriate.
A remediation may repair data, change a model, alter a threshold, change the workflow, or restrict the approved use.
A model change follows change control and may itself require validation before release.
Record the version and effective time; do not overwrite the evidence for the incident version.

Define restoration criteria before releasing the fix.
They may include reconciled data, successful replay on affected cases, comparison with a benchmark, documented approval, and monitoring under heightened surveillance.
A short post-fix period may not contain enough mature labels to prove future quality.
Use provisional controls until adequate outcomes arrive and document this limitation.
Confirm that the fallback can be disengaged without losing traceability of decisions made during containment.

Independent verification assesses whether the agreed evidence demonstrates remediation.
The business authority then accepts residual risk or keeps restrictions in place.
Close the incident and findings only when their respective criteria are met; an incident can be operationally contained while a longer-term finding remains open.
Keep any exception time-limited, with compensating controls, explicit authority, and a review date.
If suitability cannot be restored or evidence is unavailable, retire or continue restricting the model through the authorized process.

### Step 12. Learn and maintain the control

**Input:** alerts, missed events, revalidation results, and operating decisions.
**Output:** improved monitoring plan and preserved history.

Review alert volume, acknowledgement time, false alarms, missed incidents, unresolved findings, and fallback performance.
Investigate why an important event escaped detection rather than merely lowering every threshold.
Update the metric definition or control design through change control.
Changing a threshold after an alert must not erase the original breach.

Reassess whether owners, contacts, data contracts, and approval scopes remain current after organizational change.
Test escalation paths and fallback capacity periodically in a manner appropriate to the risk.
Link lessons to model cards, inventory, validation reports, and business continuity procedures.
The aim is a traceable decision loop, not a dashboard that is permanently green.

## 6. Worked example

### Case and decision

Fictional Northstar Components forecasts weekly demand for a specialized pump seal.
The forecast supports a purchase plan for two factories; only procurement can authorize a purchase order.
The approved model covers established seals at a four-week horizon.
A forecast is an estimate, the purchase plan is intended action, the service-level target is an objective, and an approved purchase order is a commitment.

The model owner is the planning analytics lead.
Procurement owns the purchase decision and can pause automated order release.
A separate risk analyst, who did not develop or remediate the model, performs independent review.
A data engineer owns the supplier-feed contract.
The model has annual scheduled revalidation and event-triggered review under Northstar's fictional policy.

The model version is M-17.4, approved on 1 January.
The latest review is therefore due by 1 January of the following year unless an earlier trigger occurs.
The four-week forecast uses demand history, promotion flags, and supplier lead-time bands.
Labels are considered mature two weeks after the end of the target week, subject to completeness checks.

### Baseline and monitored window

The approved reference is 20 comparable non-promotion weeks from the prior year.
It contains 1,000 eligible item-location-weeks and uses frozen lead-time bands.
The current window contains 200 eligible item-location-weeks over four completed decision weeks.
Counts below are observations, not distinct suppliers.
For the lead-time distribution, all records are present and bins use the same definition in both windows.

| Lead-time band | Reference count | Reference share | Current count | Current share |
|---|---:|---:|---:|---:|
| At most 7 days | 500 | 0.50 | 60 | 0.30 |
| More than 7 to 14 days | 300 | 0.30 | 60 | 0.30 |
| More than 14 days | 200 | 0.20 | 80 | 0.40 |
| Total | 1,000 | 1.00 | 200 | 1.00 |

Northstar uses population stability index (PSI) as a descriptive distribution-shift signal:

\[
\mathrm{PSI}=\sum_b(q_b-p_b)\ln(q_b/p_b),
\]

where \(p_b\) is the reference share and \(q_b\) is the current share in frozen bin \(b\).
For this example all shares are positive, so no zero-share adjustment is needed.
The first contribution is \((0.30-0.50)\ln(0.30/0.50)\approx0.1022\).
The second contribution is \((0.30-0.30)\ln(0.30/0.30)=0\).
The third contribution is \((0.40-0.20)\ln(0.40/0.20)\approx0.1386\).
The total is approximately \(0.2408\), dimensionless.

The example policy calls PSI at least 0.20 a warning for investigation, not proof of a harmful model defect.
It requires a minimum of 100 current observations for this rate-based rule.
A separate critical-supplier event rule has no 100-observation minimum.
PSI depends on bins and the reference period; 0.20 is a fictional local threshold, not a universal significance level.

### Mature forecast outcomes

For simplicity, the following four rows summarize non-overlapping mature target-week groups from the monitored use.
The actual and forecast units are pump seals, and the forecast is frozen at the four-week origin.
All expected labels for these four rows are mature and available.
These are actual sales, and there were no recorded stockouts in these four groups; this does not prove absence of every demand constraint.

| Group | Actual seals | Forecast seals | Signed error, forecast minus actual | Absolute error |
|---|---:|---:|---:|---:|
| Factory A, first group | 120 | 108 | -12 | 12 |
| Factory A, second group | 80 | 92 | +12 | 12 |
| Factory B, first group | 150 | 120 | -30 | 30 |
| Factory B, second group | 50 | 40 | -10 | 10 |
| Total | 400 | 360 | -40 | 64 |

WAPE is \(64/400=0.16\), or 16%.
Signed bias is \(-40/400=-0.10\), or -10%; under the declared convention this is underforecast.
For Factory B, WAPE is \((30+10)/(150+50)=40/200=20%\).
Factory B signed bias is \((-30-10)/200=-20%\).
The aggregate signed error does not show the full Factory B concentration.
The outcome rows summarize units, whereas the 200 lead-time observations are item-location-weeks; these denominators must not be combined.

A feasible seasonal-naive benchmark has 72 seals of absolute error on the same 400 actual seals.
Its WAPE is \(72/400=18%\), so M-17.4 has 2 percentage points lower WAPE in this small sample.
This comparison does not establish that the model is safe for Factory B, prove statistical significance, or demonstrate lower decision cost.
A benchmark can be worse overall while the model still violates a critical-segment risk limit.

### Threshold matrix and event

| Signal | Fictional rule | Response |
|---|---|---|
| Lead-time PSI | At least 0.20 with at least 100 observations | Investigate within two business days |
| Overall WAPE | Above 20% for two consecutive mature windows | Independent revalidation trigger |
| Factory B bias | At or below -15% for one mature window with at least 150 actual seals | Restrict Factory B and request independent review |
| Critical supplier interruption | Confirmed interruption affecting the sole source | Immediate containment and independent revalidation |
| Missing feed | No valid feed by purchase cutoff | Hold affected automated release immediately |
| Schedule | One year since approval or completed review | Start scheduled independent revalidation |

The actual PSI warning is 0.2408, and Factory B bias is -20% on 200 actual seals.
Both relevant thresholds are met.
Overall WAPE is 16%, so the overall WAPE rule is not met.
Not meeting that aggregate rule does not cancel the segment rule.

At 09:00 on Monday, procurement confirms that the sole supplier's heat-treatment line is unavailable for an estimated three weeks.
The estimate is a supplier statement, not a guaranteed recovery date.
This confirmed interruption independently meets the critical-supplier event trigger.
The supplier capacity change may affect fulfillment and decision feasibility even if lead-time inputs and forecast accuracy remain correct.
The event does not prove that it caused the previously observed forecast error.

### Immediate action and exposure

Procurement pauses automated release for Factory B at 09:30 and uses an approved manual review gate.
The gate uses current on-hand inventory, confirmed supplier capacity, and the last authorized forecast; procurement still approves any purchase commitment.
A planner has capacity to review all 18 affected orders per day; the capacity estimate and assignment are recorded.
If manual capacity falls short, the incident commander escalates rather than releasing unreviewed orders.
Factory A continues under the existing approval, subject to the owner's documented assessment that its supply path is unaffected.

A scenario estimates that 40 potentially missing seals could cost USD 35 per seal in incremental expedited handling if each could be recovered that way.
The scenario exposure is \(40\times35=\text{USD }1{,}400\).
It is not a realized loss, does not include service penalties, and assumes all 40 seals are recoverable at the same incremental cost.
No additional scenario is summed without checking overlap.

### Investigation and revalidation

The data engineer confirms that the supplier feed arrived and the unit conversion did not change.
The observed lead-time shift is therefore not explained by those two checked defects; other explanations remain possible.
The analyst preserves the original feed, forecast origins, order recommendations, and thresholds.
The business owner records the pause, affected decisions, expected review date, and communication to factory planners.

An independent reviewer assesses the four-week horizon, the Factory B segment, the supplier feed, and the feasibility of the order workflow.
The reviewer checks the baseline, model version, dates, missing labels, signed-error convention, and benchmark alignment.
The reviewer also examines whether the interruption changes assumptions used by the planning process.
Because the model is not being approved for a new population or method, the reviewer may justify a targeted scope, but must document why other areas are unaffected.
The confirmed sole-source interruption and bias trigger cannot be closed merely by reporting the overall 16% WAPE.

Suppose the reviewer finds that the forecast omits a known recurring customer maintenance shutdown that is represented in the current planning calendar but not the historical feature set.
This is a fictional finding, not a claim about a real firm.
The development team proposes a calendar feature and replay; change control treats the resulting model as a new version.
The reviewer independently checks decision-time availability of that calendar and assesses replay against the same eligible origins and baseline.
The new feature must not use dates learned only after a forecast was issued.

### Restoration decision

Before release, Northstar requires a reconciled feed, documented feature availability, independent review of the new version, procurement approval, and a controlled deployment record.
It also requires a tested manual fallback and daily monitoring during the supplier interruption.
If the independent review supports only a restricted use, the business owner cannot describe the model as fully approved.
A two-week post-release period is not sufficient to evaluate mature outcomes for a four-week horizon plus a two-week label lag.
The team therefore tracks leading data and workflow controls immediately, then reviews the first mature outcomes when available.

The incident closes only after the operational containment criteria are met.
The model finding closes separately after the independent reviewer verifies its documented criteria.
Procurement records the residual supplier risk and the authority for resuming automated releases.
All values, communications, and decisions remain linked to the incident and versions M-17.4 and its successor.

## 7. Code, query, or calculation

The following standalone Python 3 program uses only the standard library.
It reproduces the distribution and forecast calculations above and evaluates the fictional rules.
Save it as `monitor_example.py` and run `python monitor_example.py`.
The lists are assumed to be already aligned to the frozen windows and mature outcomes described in Section 6.
The program does not retrieve production data, determine causality, handle every operational failure, or authorize a decision.

```python
from math import log

# Frozen bin counts: same bins and complete populations in both windows.
reference = [500, 300, 200]
current = [60, 60, 80]

# Rows are (segment, actual units, forecast units), already mature.
rows = [
    ("A", 120, 108),
    ("A", 80, 92),
    ("B", 150, 120),
    ("B", 50, 40),
]
benchmark_absolute_error = 72


def psi(reference_counts, current_counts):
    """Return PSI for positive, aligned bin shares; do not smooth zero bins."""
    if len(reference_counts) != len(current_counts) or not reference_counts:
        raise ValueError("bins must be nonempty and aligned")
    if any(x <= 0 for x in reference_counts + current_counts):
        raise ValueError("zero/negative bins require an explicit policy")
    ref_total = sum(reference_counts)
    cur_total = sum(current_counts)
    return sum(
        (c / cur_total - r / ref_total)
        * log((c / cur_total) / (r / ref_total))
        for r, c in zip(reference_counts, current_counts)
    )


def error_metrics(observations):
    """WAPE and signed bias; positive signed bias means overforecast."""
    actual_total = sum(actual for _, actual, _ in observations)
    if actual_total <= 0:
        raise ValueError("positive actual total required")
    absolute = sum(abs(forecast - actual) for _, actual, forecast in observations)
    signed = sum(forecast - actual for _, actual, forecast in observations)
    return actual_total, absolute / actual_total, signed / actual_total


shift = psi(reference, current)
total, wape, bias = error_metrics(rows)
b_total, b_wape, b_bias = error_metrics([row for row in rows if row[0] == "B"])
benchmark_wape = benchmark_absolute_error / total

psi_warning = sum(current) >= 100 and shift >= 0.20
b_trigger = b_total >= 150 and b_bias <= -0.15
supplier_interruption_confirmed = True
revalidation = b_trigger or supplier_interruption_confirmed

print(f"PSI: {shift:.4f}")
print(f"Overall WAPE: {wape:.1%}; bias: {bias:.1%}")
print(f"Factory B WAPE: {b_wape:.1%}; bias: {b_bias:.1%}")
print(f"Benchmark WAPE: {benchmark_wape:.1%}")
print(f"PSI warning: {psi_warning}")
print(f"Factory B trigger: {b_trigger}")
print(f"Independent revalidation: {revalidation}")
print(f"Illustrative exposure: USD {40 * 35:,}")
```

Expected output:

```text
PSI: 0.2408
Overall WAPE: 16.0%; bias: -10.0%
Factory B WAPE: 20.0%; bias: -20.0%
Benchmark WAPE: 18.0%
PSI warning: True
Factory B trigger: True
Independent revalidation: True
Illustrative exposure: USD 1,400
```

The program deliberately rejects zero-count bins rather than silently choosing a smoothing constant.
In production, the approved methodology must specify whether to merge bins, apply a documented smoothing rule, or mark the statistic unavailable.
It must also check population completeness, data provenance, label maturity, and the actual version and time keys before computing these measures.
It should record the calculation version, evidence snapshot, alert timestamp, recipient, and acknowledgement.
A failed calculation should create an operational control failure, not produce a zero or a green status.

The code illustrates a single mature window.
It does not implement the two-consecutive-window overall WAPE rule because no prior-window observation is supplied.
It also does not infer whether the Factory B condition caused the supplier event.
The event flag is a confirmed input to the demonstration, not an automatically verified external fact.

## 8. Data, standards, and regulatory context

### Required evidence and data contracts

A monitoring dataset needs stable identifiers for model, version, decision, item, location, horizon, origin, and target period.
It needs both event time and processing time where late arrivals or corrections are possible.
Record what was known at the decision cutoff, not only the latest corrected value.
Retain the original and corrected records with lineage and access controls appropriate to their sensitivity.

Forecasts, actuals, and planning actions often live in different systems.
A join must specify key uniqueness, time-zone conventions, unit conversion, cancellation handling, and revision policy.
Check unmatched and duplicate records before calculating performance.
If an item moves between categories, define whether segment membership is as-of-origin or as-of-outcome and maintain consistency.
A denominator should be derived from the expected population, not merely from the records that survived the join.

Lead time can mean quoted, planned, confirmed, or realized transit plus processing time.
Define which is used and how partially received orders are treated.
Observed sales can be constrained by inventory and allocation.
A product introduction or discontinuation changes the eligible population and may make a historical baseline unsuitable.
A revised accounting value is not necessarily the value known when the model was used.

If labels are delayed, monitor their arrival and distinguish provisional from mature metrics.
A correction policy should identify when a closed period can be restated and retain prior published values.
Privacy, retention, access, and contractual restrictions may limit the evidence that can be stored or shared.
Involve the appropriate legal, security, and data-governance owners for the actual deployment.

### Standards and guidance

The National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 describes a voluntary approach organized around Govern, Map, Measure, and Manage.
Its monitoring and management ideas are useful for assigning responsibility, measuring risk, and responding to changing conditions.
The framework itself does not prescribe Northstar's PSI threshold, annual interval, or specific supply-chain operating decision.
NIST's AI RMF Playbook provides suggested actions rather than a universal checklist.
See the official sources in Section 16.

The Federal Reserve's SR 11-7, *Guidance on Model Risk Management*, discusses ongoing monitoring, outcomes analysis, and effective challenge in the context of supervised banking organizations.
Its applicability depends on the organization and its regulatory circumstances; a supply-chain company should not represent the guidance as automatically binding merely because it uses a model.
The example's governance practices are educational adaptations, not a claim of SR 11-7 compliance.
Consult the applicable regulator and current guidance before making a legal or supervisory claim.

ISO 31000:2018 provides general risk-management guidelines; it is not a model-specific certification of a forecast.
ISO/IEC 23894:2023 provides guidance on risk management for organizations developing, producing, deploying, or using artificial intelligence.
Standards may require purchase for full text and their contractual or legal effect depends on adoption and context.
A standard reference does not establish that a particular implementation conforms to it.

The EU Artificial Intelligence Act, Regulation (EU) 2024/1689, contains risk-based obligations and phased application dates.
Whether a particular supply-chain system is in scope, prohibited, high-risk, subject to other obligations, or outside a particular requirement depends on its actual characteristics, role, and applicable dates.
Do not infer that an ordinary demand forecasting application is high-risk solely because it affects procurement.
Check the current consolidated law, any amendments, and professional advice before claiming an obligation or deadline.
The monitoring design here is not a legal compliance determination.

### Control design implications

Regulatory or contractual retention requirements may exceed or differ from local business preferences.
A minimum sample rule must not override a legal or contractual incident-reporting obligation.
A documented independent review is not automatically equivalent to an external audit, certification, or regulatory approval.
Vendor-supplied metrics are evidence to assess, not a transfer of the deployer's accountability.
When the vendor cannot disclose full internals, document access limits and use available independent outcome, data, and operational tests.

## 9. Industry practice and case evidence

### Public evidence and the lesson it supports

The 2011 Tōhoku earthquake and tsunami disrupted Japanese manufacturing and global supply chains.
The World Bank and other public accounts document the scale of the disaster and economic disruption.
Toyota's public sustainability reporting describes supply-chain continuity and supplier risk management measures after major disruptions.
These sources support the general point that physical events can alter supplier availability and require supply-chain response.
They do not establish that Toyota's particular forecasting model breached a monitoring threshold or that a specific model revalidation occurred.

**Inference:** a firm using a demand model to plan purchases from an affected sole-source supplier should assess whether the decision workflow and availability assumptions remain valid even before forecast labels mature.
This is an application of the monitoring framework, not a reported Toyota model-control practice.
The distinction matters: a documented disruption is not evidence for an invented model incident.

The U.S. Food and Drug Administration's 2021 *Good Machine Learning Practice for Medical Device Development: Guiding Principles*, developed with partner regulators, identifies monitoring deployed models and managing re-training risks among its principles.
Medical devices are not supply-chain forecasts and have different harm and regulatory contexts.
The general lifecycle lesson is that a deployed model needs monitoring and controlled changes; the document does not prescribe the thresholds in this lesson.

NIST's AI Risk Management Framework describes continuous and context-sensitive risk management for AI systems.
It is a framework rather than a public claim that every organization has successfully implemented a particular drift detector.
A practitioner should translate the framework into a specific use, owner, metric, action, and evidence record.

### What practitioners must adapt

A retailer may care about product-location service and seasonal promotions.
An industrial manufacturer may focus on a critical component with long lead time and few comparable outcomes.
A logistics provider may need immediate response to route closures, carrier capacity, and weather.
In each case, an aggregate model score is insufficient if the operational bottleneck lies in a small segment.

Some organizations monitor a fixed historical baseline and a rolling baseline together.
The fixed baseline preserves a stable point of comparison; the rolling baseline highlights recent changes.
The rolling baseline can gradually absorb deterioration, while the fixed baseline can become less representative after a legitimate structural change.
A governed baseline update records both the rationale and the old values.

Teams often distinguish leading indicators from lagging outcomes.
Data freshness, supplier interruption, and output anomalies can be observed quickly.
Forecast error and actual decision cost may take weeks or months to mature.
Leading indicators provide opportunities to contain exposure but are not proof of degraded predictive performance.
Lagging evidence can assess outcomes but may arrive after a costly commitment.

Independent review is most useful when it has access to the same decision-time evidence and authority to challenge the operating decision.
Reviewers should ask whether an apparent improvement results from excluding hard cases or from a changed denominator.
They should also assess whether the intended fallback can operate at required volume and whether decisions during fallback remain traceable.

### Limits of public case transfer

A public incident report rarely reveals the complete internal model inventory, threshold design, data lineage, or confidential decision history.
Absence of such detail is not evidence that a control was absent.
Likewise, a company statement about improved resilience is not independent proof of a particular model's accuracy.
Treat public examples as context and test the proposed control against the actual deployment's evidence.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Missing monitor treated as green | No new metric after a feed delay | Missing output defaults to pass | Exposure continues unobserved | Separate freshness and execution alarms; show unknown |
| Silent baseline replacement | Historical dashboard values change | Current period becomes its own reference | Gradual degradation disappears | Version baselines and retain prior comparisons |
| Aggregate masks critical segment | Overall WAPE stable while a critical site complains | Large low-risk volumes dominate | Critical shortage persists | Prespecify material segment and exposure rules |
| Changed units | Lead times suddenly divide by 24 | Hours ingested as days or vice versa | Erroneous inputs and plans | Contract checks, reconciliation, and containment |
| Label leakage | Replay accuracy implausibly improves | Future information enters historical features | False validation confidence | Reconstruct as-of-origin inputs and independently review |
| Immature labels | Metric improves while many actuals are missing | Only early-reporting cases enter denominator | Misleading score | Label watermark, completeness, and provisional status |
| Stockout-censored sales | Forecast exceeds sales during empty shelves | Sales cannot reveal unmet demand | True demand underestimation | Report sales-based error and inventory limitation |
| Alert fatigue | Growing unacknowledged alerts | Noisy rules exceed response capacity | Important signal missed | Tune with evidence, ownership, routing, and escalation |
| Excess persistence | Severe event waits for a second week | Generic persistence applied to urgent incident | Irreversible commitments | Immediate event rules and preapproved containment |
| No fallback capacity | Manual queue grows after automation stops | Fallback was not tested at peak volume | Delayed or uncontrolled decisions | Test staffing, inputs, authority, and escalation |
| Conflicted review | Fix author signs off own change | Reviewer cannot independently challenge work | Residual defect accepted | Assign independent qualified reviewer; disclose conflicts |
| Revalidation scope too narrow | Only accuracy checked after supplier change | Decision feasibility and assumptions omitted | Accurate forecast supports infeasible plan | Scope review to affected decision and assumptions |
| Vendor silent update | Output distribution changes without local release | Hosted implementation changes outside local control | Evidence no longer matches version | Contract for notices, versioning, access, and change assessment |
| Override opacity | Many recommendations replaced without reasons | Human intervention is not recorded | Cannot assess control or harm | Capture pre/post values, reason, actor, and timing |
| False causal attribution | All losses assigned to model after disruption | Other drivers and counterfactual ignored | Misleading remediation priorities | Separate observed facts, scenarios, and causal claims |
| Threshold gaming | Limit raised immediately after breach | Governance bypasses original rule | Event disappears from reports | Preserve original alert and require approved change |
| Unbounded exception | Temporary approval passes expiry | No owner or reassessment | Residual risk becomes permanent | Named authority, expiry, controls, and escalation |
| Comparator mismatch | Benchmark uses different items or horizons | Unequal case selection | False relative performance | Join on same eligible mature origins and disclose exclusions |
| Correction erases history | Only latest forecast is stored | Revised values replace decision-time evidence | Incident cannot be reproduced | Immutable snapshots and versioned corrections |
| Market event ignored | Procurement reports supplier closure but dashboard is green | Monitoring covers only numerical drift | Unsafe plan continues | Qualitative event channel and impact assessment |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Reference period is relevant | Comparison has a meaningful context | Structural change, promotion mix, new products | Review seasonality and population; retain fixed and contextual baselines |
| Bins have stable meaning | PSI compares the same concept | Unit or feature-definition change | Validate contracts and frozen bin definitions |
| Positive PSI shares | Log ratio is defined in the example | Empty reference or current bin | Apply an approved merge or smoothing policy, or mark unavailable |
| Forecast and actual keys align | Error refers to the intended prediction | Time-zone, item mapping, or horizon mismatch | Reconcile joins and inspect unmatched cases |
| Labels are sufficiently mature | Outcome metric reflects eligible cases | Late reporting and corrections | Measure label completeness and use a watermark |
| Actual denominator is positive | WAPE and signed ratio are defined | Zero actual units in a window | Report absolute error and mark ratio undefined |
| Observed sales represent target | Error has its stated interpretation | Stockout or allocation censoring | Add inventory context and name the target as sales |
| Critical segments are identified | Small important risks are visible | New supplier or customer dependency | Reassess segmentation when business exposure changes |
| Thresholds fit the decision | Response is proportionate to risk | Changed costs, volumes, or capacity | Reassess under governance using historical and scenario evidence |
| Owner can act in time | An alert can reduce exposure | No authority, absent contact, or expired delegation | Test escalation, backup, and action deadlines |
| Fallback is feasible | Suspension does not create uncontrolled decisions | Insufficient people, inputs, or approval capacity | Exercise fallback and monitor its queue and outcomes |
| Reviewer is independent and competent | Challenge can identify material defects | Prior development, incentives, or insufficient expertise | Declare conflicts and use a qualified alternative |
| Historical replay is valid | Comparison estimates behavior on past decisions | Leakage, changed data semantics, or unavailable counterfactual | Reconstruct decision-time data and report limits |
| Market event is correctly classified | Trigger corresponds to an actual exposure | Rumor or incomplete supplier information | Record source, confidence, and reassessment time |
| Cost scenario is additive | Sum reflects distinct exposures | Double-counted expediting and service loss | Map overlaps and label unverified estimates |
| Stable metrics imply neither safety nor failure | The governance decision considers more than one signal | Unobserved concept drift or decision change | Combine inputs, outcomes, incidents, and qualitative review |

Statistical uncertainty is particularly important when event rates are low or observations are correlated over time.
A threshold crossing is a signal for a specified response, not a confidence interval or an estimate of causality.
Frequent testing and many segments can produce apparent anomalies even when the process is stable.
Conversely, a non-significant result can reflect inadequate power, delayed labels, or a harmful segment hidden by aggregation.

The worked example is deliberately small enough to calculate by hand.
Its threshold values, cadence, supplier event, and cost are invented teaching assumptions.
They should not be deployed without calibration, stakeholder approval, and appropriate validation for the actual use.
The example does not quantify the probability of shortage, the cost of all possible outcomes, or a causal effect of the model on procurement.

## 12. Exercises

### Exercise 1 — Build the monitoring mandate

**Time:** 20 minutes.
A weekly demand forecast supports replenishment for 200 stores and one distribution center.
A separate purchase system converts the forecast into suggested orders, and procurement approves orders above USD 50,000.
One chilled product expires after seven days and is supplied by a single plant.

**Deliverable:** a one-page monitoring mandate with decision, population, horizon, owners, and action authority.
**Acceptance criteria:** distinguish forecast, plan, target, and commitment; identify the critical product; name a monitor owner and an independent reviewer; specify at least one permitted restriction and one missing fact requiring discovery.
Do not assume the procurement approval threshold is a model-quality threshold.

### Exercise 2 — Classify a signal

**Time:** 15 minutes.
Classify each observation and give one alternative explanation.
(a) A supplier field switches from days to hours without a column-name change.
(b) Average predicted volume rises 30% during a documented promotion.
(c) Actual sales are flat but stockout hours double.
(d) The relationship between price and demand changes after a competitor exits.

**Deliverable:** four rows stating signal type, plausible mechanism, immediate evidence, and response.
**Acceptance criteria:** distinguish data defect, output shift, censored outcome, and potential concept change; do not treat any signal alone as proof of model harm.

### Exercise 3 — Calculate distribution shift

**Time:** 20 minutes.
A frozen reference has bin shares 0.50 and 0.50; the current window has shares 0.25 and 0.75.
Use PSI \(\sum_b(q_b-p_b)\ln(q_b/p_b)\).
The local warning threshold is at least 0.20, and both windows have enough observations under the local policy.

**Deliverable:** both contributions, total PSI, and the resulting warning decision.
**Acceptance criteria:** use natural logarithms, report a dimensionless result to four decimals, and explain why a warning does not prove degraded accuracy.

### Exercise 4 — Calculate error and segment exposure

**Time:** 20 minutes.
For two mature groups, actuals are 100 and 300 units, and forecasts are 80 and 330 units.
Use signed error forecast minus actual.
A segment escalation applies when absolute signed bias is at least 10% and actual volume is at least 100 units.

**Deliverable:** overall WAPE and signed bias; each group's signed bias; triggered group or groups.
**Acceptance criteria:** use sums before division for overall metrics, state the sign interpretation, and apply the inclusive threshold and volume rule.

### Exercise 5 — Design thresholds for incomplete labels

**Time:** 25 minutes.
A model forecasts a 12-week planning horizon; outcome labels mature four weeks after the target week.
A weekly dashboard currently reports WAPE on whatever labels have arrived, without completeness or population information.
A supplier disruption can create an irreversible commitment within two days.

**Deliverable:** a short threshold and response table with at least one leading indicator, one mature-outcome indicator, and one qualitative trigger.
**Acceptance criteria:** give timing, denominator, owner, unknown-data treatment, and action; explain why the mature WAPE cannot protect every immediate decision; include a rule for small critical segments.

### Exercise 6 — Triage an incident

**Time:** 25 minutes.
A sole-source supplier announces a confirmed shutdown.
The model's latest aggregate WAPE is below its warning threshold, but the lead-time feed has not arrived by the ordering cutoff.
The business asks to release all automated purchase orders while the team investigates.

**Deliverable:** an incident timeline for the first business day and a revalidation decision.
**Acceptance criteria:** preserve evidence, assign an incident commander and business authority, contain affected automated releases, identify an operable fallback or escalate its absence, and distinguish the shutdown trigger from evidence of model error.

### Exercise 7 — Write revalidation and closure terms

**Time:** 30 minutes.
A developer fixes the late feed and proposes to validate the fix personally.
The business owner requests an indefinite exception until the next annual review.

**Deliverable:** a finding, independent review scope, and bounded operating decision.
**Acceptance criteria:** name the issue, evidence, impact, owner, deadline, independent verifier, measurable closure criteria, exception expiry, compensating controls, and escalation path; separate validation opinion from risk acceptance.

### Exercise 8 — Critique a public-case claim

**Time:** 20 minutes.
A presentation says, “The 2011 Japan earthquake proved Toyota's forecast drift monitor failed, so all manufacturers must use PSI 0.20.”
The presentation cites a general disaster report but no model-specific evidence.

**Deliverable:** a corrected two-paragraph claim and evidence request.
**Acceptance criteria:** separate documented disruption from unsupported model assertions; describe PSI as a local descriptive measure rather than a universal legal requirement; identify what evidence would be needed to assess an actual model control.

## 13. Self-check answers

### Exercise 1

A strong mandate identifies item-store-week or item-distribution-center-week as the decision grain, subject to confirmation.
It records the actual horizon, cutoff, model version, eligible stores, and downstream order conversion rather than inventing them.
The forecast estimates demand; a replenishment plan proposes action; a service-level target is a desired result; an approved purchase order is a commitment.

The chilled sole-source item warrants explicit monitoring because expiry and supplier concentration can make both underordering and overordering costly.
A monitor owner should own feed and metric execution, while procurement has the delegated authority over affected commitments.
An independent reviewer must not have materially developed the model or the fix being assessed.
A restriction could require manual procurement review for that item; its feasibility and authority must be confirmed.
The exact decision cutoff, horizon, fallback capacity, and approved use are facts to obtain from the operating record.

### Exercise 2

(a) A unit change is a data-contract defect, even if the column still has the same name.
Compare source-system definitions, value ranges, and conversion logs; hold affected automated decisions until the data are corrected or a safe alternative is authorized.
A real change in supplier lead times is an alternative to investigate, not a reason to ignore the contract check.

(b) The rise is prediction drift; it may be a correct response to the promotion.
Compare like promotional periods, input values, output distribution, and eventually mature outcomes.
Do not automatically roll back the model solely because the mean changed.

(c) Flat sales with increased stockouts indicate a possible censored outcome and operational incident.
Sales may understate unconstrained demand; check inventory, lost-sales evidence, and affected products.
The data do not prove how much unmet demand existed.

(d) A changed price-demand relationship is a candidate concept shift.
Confirm the event, examine whether the model depends on the old relationship, and gather comparable evidence as it matures.
A competitor exit alone does not measure the magnitude or prove a model defect.

### Exercise 3

The first contribution is \((0.25-0.50)\ln(0.25/0.50)\approx0.1733\).
The second contribution is \((0.75-0.50)\ln(0.75/0.50)\approx0.1014\).
The sum is approximately 0.2747, dimensionless.
Because 0.2747 is at least the local 0.20 threshold and the count condition is met, the warning requires investigation.
PSI compares distributions in the chosen bins; it does not determine whether predictions or decisions became worse.

### Exercise 4

Absolute errors are 20 and 30 units, so WAPE is \((20+30)/(100+300)=12.5%\).
Signed errors are -20 and +30 units, so overall signed bias is \((-20+30)/400=+2.5%\), an aggregate overforecast.
The first group has \(-20/100=-20%\) bias; the second has \(+30/300=+10%\) bias.
Both meet the absolute 10% inclusive threshold and the minimum of 100 actual units.
The small aggregate bias results from cancellation and does not make the group alerts disappear.

### Exercise 5 rubric

A strong answer monitors feed freshness and confirmed supplier events at or before the two-day decision deadline.
It separately reports mature WAPE on eligible forecast-origin/target pairs with an explicit label-completeness denominator.
The 12-week forecast horizon plus four-week post-target maturity lag means a decision-origin cohort can take at least 16 weeks to mature.
The exact availability date depends on period boundaries and data delivery.

A missing feed or missing outcome metric is unknown or failed, not green.
The answer assigns operational and decision owners and identifies immediate hold or manual review authority.
A small critical segment receives an exposure-based or qualitative event rule rather than waiting indefinitely for the aggregate sample threshold.
Threshold levels must be locally justified rather than copied from the worked example.

### Exercise 6 rubric

A strong first action is to restrict affected automated releases and invoke the preapproved manual gate if it is staffed and authorized.
If it is not operable, the incident commander escalates to the authority that can decide a safe alternative.
The team records the confirmed shutdown, missing feed, timestamps, affected orders, forecast version, and all decisions.

The aggregate WAPE does not override the missing-feed control or the supplier event trigger.
Independent revalidation should assess affected assumptions and workflow under the fictional local policy; interim containment does not wait for the review.
The shutdown is an operational fact, not proof that the forecast model caused it or that the model's accuracy already deteriorated.

### Exercise 7 rubric

A complete finding states that a late feed violated a specified cutoff and identifies affected decisions and supporting logs.
The developer may implement the correction but must not independently verify their own material fix.
The reviewer tests reconciliation, historical impact, cutoff controls, monitoring, and evidence of the corrected version.
Closure criteria should include the exact tests and approvals required, not merely “feed fixed.”

The business owner may request a time-limited exception only under delegated policy, with a clear expiry, manual control, monitoring, and escalation.
An indefinite exception is not an acceptable substitute for a completed review.
The reviewer records an independent opinion; the authorized business owner separately accepts or rejects remaining risk.

### Exercise 8 rubric

The 2011 disaster and supply-chain disruption are publicly documented.
The cited general report does not establish a Toyota forecast-monitor failure or the design of Toyota's internal drift controls.
A defensible inference is that such a disruption can invalidate availability assumptions and should prompt an impact assessment.

PSI is a descriptive statistic whose result depends on binning, comparison population, and time window.
A 0.20 threshold in this lesson is illustrative and is not established as a universal requirement by the disaster report.
Evidence for the original assertion would include a reliable model-specific incident report, baseline and threshold definitions, timestamps, affected decisions, and the organization's documented investigation.

## 14. Interview questions

1. **What should you monitor after approving a model?**
   A strong answer connects data, outputs, mature outcomes, decision workflow, market events, and control health to an approved use.
   A weak answer lists only a single aggregate accuracy score.
2. **How do input drift and concept drift differ?**
   A strong answer distinguishes a change in input distribution from a change in the input-outcome relationship and notes that one does not prove the other.
   A weak answer calls every distribution shift model failure.
3. **How would you handle delayed actuals?**
   A strong answer defines a maturity watermark, completeness denominator, provisional status, and leading controls.
   A weak answer treats absent outcomes as zero error.
4. **When can a single event require immediate action?**
   A strong answer identifies severity, irreversibility, authority, and preapproved containment.
   A weak answer insists on two statistical breaches before responding to a confirmed sole-source outage.
5. **What does a PSI threshold mean?**
   A strong answer explains fixed bins, reference and current shares, local calibration, zero-bin treatment, and limits.
   A weak answer presents a universal threshold as a causal or legal finding.
6. **What is the difference between monitoring and revalidation?**
   A strong answer describes ongoing detection versus independent reassessment, with scheduled and event-triggered entry points.
   A weak answer calls the developer's daily dashboard check independent validation.
7. **Who decides whether to continue use after a finding?**
   A strong answer separates independent opinion from the authorized business owner's documented residual-risk decision and escalation.
   A weak answer assumes an analyst can waive all risk without delegated authority.
8. **How do you validate a remediation?**
   A strong answer specifies versioned evidence, independent verification, closure criteria, and controlled restoration.
   A weak answer closes the incident when the developer says the code is fixed.
9. **Why might a stable aggregate WAPE be misleading?**
   A strong answer covers critical segments, changing mix, label completeness, benchmark alignment, and decision costs.
   A weak answer treats aggregate stability as proof all decisions are safe.
10. **What makes a fallback credible?**
    A strong answer identifies tested capacity, data, authority, operating procedure, monitoring, and traceability.
    A weak answer says “use manual judgment” without naming who can perform it.
11. **How should a vendor update affect monitoring?**
    A strong answer requires version and change evidence, impact assessment, access limitations, and proportionate review.
    A weak answer assumes the vendor's release note transfers the deployer's accountability.
12. **What would you do if a critical reviewer conflict cannot be resolved?**
    A strong answer obtains a qualified alternative or escalates the limitation and operating decision.
    A weak answer relabels the conflicted developer as independent.

## 15. Key terms

- [Approved use](#4-definitions-and-vocabulary) — authorized decision, population, horizon, and constraints.
- [Baseline](#4-definitions-and-vocabulary) — frozen reference with a recorded selection rationale.
- [Bias](#4-definitions-and-vocabulary) — signed forecast error under a stated convention.
- [Challenger](#4-definitions-and-vocabulary) — alternative evaluated on comparable cases.
- [Concept drift](#4-definitions-and-vocabulary) — change in the input-outcome relationship.
- [Containment](#4-definitions-and-vocabulary) — immediate limitation of exposure.
- [Current window](#4-definitions-and-vocabulary) — bounded production comparison period.
- [Data-quality incident](#4-definitions-and-vocabulary) — defect in a data control or meaning.
- [Decision authority](#step-1-define-the-governed-decision) — person or role authorized to change operations.
- [Event trigger](#4-definitions-and-vocabulary) — condition initiating an out-of-cycle assessment or review.
- [Exception](#4-definitions-and-vocabulary) — authorized, time-limited deviation with controls.
- [Fallback](#4-definitions-and-vocabulary) — approved alternative operating process.
- [Finding](#4-definitions-and-vocabulary) — evidenced issue with ownership and closure criteria.
- [Forecast](#1-why-this-topic-exists) — estimate of an uncertain outcome.
- [Incident](#4-definitions-and-vocabulary) — suspected or realized event requiring response.
- [Independence](#4-definitions-and-vocabulary) — reviewer distance, competence, access, and authority.
- [Input drift](#4-definitions-and-vocabulary) — change in an input distribution.
- [Label maturity](#4-definitions-and-vocabulary) — sufficient outcome completeness for a stated measure.
- [Market change](#step-7-detect-market-and-operating-changes) — external or internal event affecting assumptions or decisions.
- [Materiality](#4-definitions-and-vocabulary) — significance of impact and exposure.
- [Outcome drift](#4-definitions-and-vocabulary) — change in actual outcome distribution.
- [Override](#4-definitions-and-vocabulary) — authorized human change to an output or recommendation.
- [Persistence](#4-definitions-and-vocabulary) — repeated or sustained threshold condition.
- [Population stability index](#6-worked-example) — descriptive comparison of bin shares.
- [Prediction drift](#4-definitions-and-vocabulary) — change in model output distribution.
- [Residual risk](#4-definitions-and-vocabulary) — exposure remaining after controls.
- [Revalidation](#4-definitions-and-vocabulary) — independent reassessment of suitability for use.
- [Segment](#4-definitions-and-vocabulary) — defined subset of the population.
- [Signed error](#6-worked-example) — forecast minus actual in this lesson.
- [Threshold](#4-definitions-and-vocabulary) — predefined condition and associated response.
- [Weighted absolute percentage error](#step-5-monitor-mature-outcomes) — sum absolute error divided by sum actuals.

## 16. Sources

1. National Institute of Standards and Technology, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, 2023. [Official publication](https://doi.org/10.6028/NIST.AI.100-1). Voluntary framework; see its lifecycle risk-management functions.
2. NIST, [AI Risk Management Framework Playbook](https://airc.nist.gov/airmf-resources/playbook/). Suggested actions accompanying the framework, not a universal certification checklist.
3. Board of Governors of the Federal Reserve System and Office of the Comptroller of the Currency, [SR 11-7, *Guidance on Model Risk Management*](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm), 2011. Banking supervisory guidance; applicability depends on the organization and current supervisory context.
4. International Organization for Standardization, [ISO 31000:2018, *Risk management — Guidelines*](https://www.iso.org/standard/65694.html). General risk-management guidance.
5. ISO and International Electrotechnical Commission, [ISO/IEC 23894:2023, *Artificial intelligence — Guidance on risk management*](https://www.iso.org/standard/77304.html). AI risk-management guidance.
6. European Union, [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), official journal text, 2024. Check current consolidated law and applicable dates for a particular system.
7. U.S. Food and Drug Administration, Health Canada, and UK Medicines and Healthcare products Regulatory Agency, [*Good Machine Learning Practice for Medical Device Development: Guiding Principles*](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles), 2021. Medical-device context, used here only for a general lifecycle principle.
8. World Bank, [*The Great East Japan Earthquake: Learning from Megadisasters*](https://openknowledge.worldbank.org/entities/publication/5e4c9e46-d4a4-5ea8-9702-17f067840f4c), 2012. Disaster and resilience context; not evidence of a particular model-control failure.
9. Toyota Motor Corporation, [Sustainability Data Book](https://global.toyota/en/sustainability/report/). Corporate reporting source; verify the relevant edition and distinguish company statements from independent evidence.
10. Gama, João, et al., “A Survey on Concept Drift Adaptation,” *ACM Computing Surveys* 46(4), 2014. [DOI: 10.1145/2523813](https://doi.org/10.1145/2523813). Concept-drift definitions and adaptation context.
11. Hyndman, Rob J., and George Athanasopoulos, [*Forecasting: Principles and Practice*, 3rd edition](https://otexts.com/fpp3/), 2021. Forecast evaluation, accuracy, and benchmark context.

## 17. Links

- [Module 26 index](_index.md).
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md).
- [Data quality and representativeness](03_data-representativeness.md).
- [Outcomes analysis and backtesting](04_outcomes-backtesting.md).
- [Metrics that matter for planning models](05_planning-model-metrics.md).
- [Model documentation and the model card](06_model-documentation.md).
- [Reproducibility and audit trail](07_reproducibility-audit.md).
- [Independence and validation](08_validation-independence.md).
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).
