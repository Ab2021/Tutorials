# Data Quality and Representativeness

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 03 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 07, and 25; use the bridge below where needed  
> Version 2026.5

## 1. Why this topic exists

A demand model can have excellent aggregate accuracy and still be unsuitable for a particular warehouse, product family, or customer group.
The problem may be missing records, records that are present but incorrect, a population absent from training, or a market that has changed since validation.
These are different failure mechanisms.
They require different evidence and different responses.

A planner deciding whether to use a recommendation needs to know where the evidence applies.
A procurement leader needs to know whether a vendor's stated accuracy includes the locations in the contract.
A risk owner needs to know whether exclusions create an unacceptable operational or social consequence.
A data steward needs to know which source, transformation, and owner can correct the defect.
A model validator needs to distinguish a genuine absence of demand from an absence of recorded sales.

A single average metric hides all of these questions.
A data-quality dashboard can be green while a strategically important rural region has no usable history.
A training set can cover every region but contain only normal weeks, leaving promotions and disruptions untested.
A current feed can contain valid values while their distribution no longer resembles the historical evidence.
None of these facts alone proves that the model will fail, but each limits what the organization may reasonably claim.

### Prerequisite bridge

Module 06 is deferred; this topic supplies the minimum machine-learning vocabulary it needs.
A model is a rule or statistical procedure that turns inputs into an estimate or recommendation.
Training uses historical examples to choose model parameters.
Validation examines evidence independent of the fitting process, ideally including a later period and relevant operating groups.
A feature is an input known at prediction time, and a label is the observed outcome used to learn or assess the model.
A benchmark is an alternative such as last week's demand or the current planner's process.
Module 07 gives forecasting context, and Module 25 explains counterfactual business value.

A **forecast** estimates an uncertain future quantity.
A **plan** is a proposed action based on that estimate and other constraints.
A **target** is a desired outcome, not an observation or forecast.
A **commitment** is an authorized obligation, such as an accepted purchase order.
This distinction matters when a planned shipment is mistakenly used as if it were realized customer demand.

The practical question is not whether a dataset is universally representative.
It is representative of a named decision population, time horizon, operating regime, and intended use to a degree justified by evidence.
The answer may differ across decisions even when the same table and model are used.

## 2. Learning outcomes

After this unit you can:

1. Define a decision population, observation unit, horizon, and authoritative denominator; apply them in Exercise 1.
2. Compute record coverage, exposure-weighted coverage, missingness, and segment-level validation coverage; apply them in Exercises 2 and 3.
3. Diagnose whether a gap reflects data absence, invalidity, selection, censoring, or a delayed label; apply this in Exercise 4.
4. Compare current and reference distributions, interpret a drift statistic, and propose a proportionate investigation; apply this in Exercise 5.
5. Explain why an excluded population cannot inherit an aggregate performance claim; apply this in Exercise 6.
6. Estimate decision value against a specified fallback, including cost, uncertainty, and excluded exposure; apply this in Exercise 7.
7. Design an approval, monitoring, incident, and exception workflow with clear decision rights; apply this in Exercise 8.
8. Separate standards guidance, legal obligations, public case evidence, and illustrative policy; apply this in Exercises 6 and 8.

A strong answer shows numerators and denominators.
It states the time window and grain for each measure.
It does not confuse the availability of records with the validity of the underlying information.
It identifies what evidence would change the decision and who must collect it.

## 3. The core mental model

```text
Decision and permissible use
          |
          v
Define target population, exposure, time and observation grain
          |
          v
Independent population frame -----> enumerate missing and excluded groups
          |                                   |
          v                                   v
Source -> ingestion -> transformations -> usable training/evaluation data
          |                 |                    |
          v                 v                    v
   quality controls   lineage/point-in-time   segment and regime tests
          \_________________|___________________/
                            |
                            v
              compare evidence with intended use
                            |
                  +---------+----------+
                  |                    |
                  v                    v
        approve bounded use     restrict / fallback / remediate
                  |                    |
                  +---------+----------+
                            |
                            v
          monitor data, population, outcomes, and action costs
                            |
                            v
          investigate, reassess, approve changes, retain evidence
```

The decision defines what needs representation.
A model that predicts weekly warehouse demand needs evidence about relevant warehouse-product-weeks, not just a large number of transaction rows.
If the intended decision is a replenishment order, demand volume and shortage consequences may matter more than the count of low-volume products.
The correct weighting is a business and governance choice, not an intrinsic property of the data.

The independent population frame gives the denominator.
It may be an effective-dated product-location master joined to a calendar, contracted lanes, active suppliers, or an orderable-customer registry.
The modeling table cannot prove its own completeness: missing members do not appear in it.
The frame itself can be wrong, so reconcile its owners, effective dates, exclusions, and material totals.

Quality is fitness for the intended use.
Completeness asks whether required values exist; validity asks whether values meet rules; accuracy asks whether they correspond to reality.
Consistency, uniqueness, timeliness, and lineage address other ways a present value can mislead.
A valid zero sale is not automatically zero demand during a stockout.
A non-null lead time of 999 days may pass a completeness check but still be unusable.

Representativeness concerns the relationship between observed examples and the decision population.
Coverage is necessary for many claims but is not sufficient: a group with a single row is technically present but provides little evidence.
Assess sample size, temporal range, rare conditions, dependence, and label maturity within each material segment.
A group may also be represented in training but absent from independent evaluation.

Drift is a change between distributions, processes, or relationships over time.
It is not synonymous with degradation.
A feature distribution can change while decision performance remains acceptable, and performance can worsen with little marginal feature drift.
The response is investigation and evidence gathering rather than automatic retraining.

Finally, a model forecast does not create value by itself.
The forecast changes a plan only through a decision process; the plan generates outcomes subject to constraints, execution, and uncertainty.
Compare outcomes with a clearly stated alternative and count the cost of operation and remediation.
A decision to exclude a population must specify the fallback and assess what happens to that population under the fallback.

## 4. Definitions and vocabulary

| Term | Definition and unit or coding | Common misuse |
|---|---|---|
| Intended use | Named decision, user, horizon, and environment | Describing only the algorithm |
| Target population | All eligible decision units in a specified period | Equating it with observed rows |
| Population frame | Independently maintained list or rule for eligible units | Assuming a stale master is complete |
| Observation grain | Meaning of one row, such as SKU-location-week | Mixing daily and weekly denominators |
| Exposure | Decision volume or consequence, e.g. units/year or USD/year | Calling record count business exposure |
| Coverage | Eligible units with qualifying data divided by eligible units | Counting any record as sufficient evidence |
| Exposure-weighted coverage | Covered exposure divided by total eligible exposure | Treating the result as accuracy |
| Training coverage | Eligible units or exposure represented in fitting data | Confusing fitting with independent validation |
| Evaluation coverage | Eligible units or exposure with valid independent tests | Counting training observations as test evidence |
| Excluded population | Eligible group outside the evidence or approved use | Assuming exclusion means no operational harm |
| Selection bias | Difference arising from which units enter observed data | Describing all differences as random noise |
| Missingness | Required value or record not observed under a defined rule | Assuming all missing values mean zero |
| Completeness | Proportion of required fields or records present | Calling complete data accurate |
| Validity | Conformance to a stated domain or rule | Treating plausible values as verified truth |
| Accuracy of data | Agreement of a value with an appropriate reference | Confusing data accuracy with model accuracy |
| Uniqueness | One intended record per defined key and period | Removing legitimate repeat transactions |
| Timeliness | Availability within the decision's permitted delay | Measuring source timestamp instead of arrival |
| Consistency | Agreement between related fields, systems, or units | Forcing disagreements to match without investigation |
| Lineage | Traceable path from source through transformation to use | Listing a source without transformation versions |
| Point-in-time correctness | Inputs were actually available at the decision timestamp | Using a corrected later value in a historical test |
| Leakage | Information unavailable at prediction time used in fitting or testing | Assuming every correlated input is legitimate |
| Censoring | Outcome partly unobserved due to a limit or process | Using constrained sales as unconstrained demand |
| Label maturity | Time until an outcome is sufficiently complete for use | Treating recent incomplete outcomes as final |
| Feature drift | Change in the distribution of model inputs | Assuming it proves decision loss |
| Label drift | Change in the distribution of outcomes | Measuring before outcomes mature |
| Concept drift | Change in the relationship between inputs and outcomes | Inferring it solely from feature drift |
| Data pipeline shift | Change caused by collection, coding, or transformation | Calling every shift a market change |
| Reference window | Frozen, documented period used for comparison | Quietly changing it to make a metric green |
| PSI | Population stability index, a dimensionless distribution comparison | Treating a universal threshold as law |
| Segment | Defined subset, such as region or product class | Choosing groups only after seeing favorable results |
| Slice | A measurable subset used for analysis | Reporting a small slice without its denominator |
| Regime | Operating condition such as promotion or disruption | Assuming a calendar month captures every regime |
| Benchmark | Named alternative process or simple model | Comparing with an unrealistic zero-skill baseline |
| Fallback | Authorized action when the model is unavailable or out of scope | Assuming manual intervention is costless or unbiased |
| Backtest | Historical simulation using information available at each decision | Reusing tuned test data as an untouched test |
| Outcome metric | Measure of realized prediction or decision results | Treating feature drift as an outcome metric |
| Uncertainty interval | Range expressing sampling or model uncertainty under assumptions | Claiming it captures all structural uncertainty |
| Reweighting | Altering sample contributions to resemble a target population | Giving infinite weight to absent groups |
| Effective sample size | Information-equivalent sample size under stated assumptions | Treating dependent weeks as independent observations |
| Exception | Time-limited, owned deviation from an approved control | Permanent undocumented waiver |
| Incident | Event that triggers documented response and investigation | Requiring proven loss before escalation |

## 5. The mechanics, step by step

### Step 1 — Define the decision and its boundaries

**Input:** proposed decision, operating workflow, users, model output, and planning horizon.
**Action:** write one sentence stating what is estimated, for whom, when, and what action may follow.
**Output:** a versioned intended-use statement and an explicit list of prohibited uses.

For example, predict next-week demand in units for active product-location pairs to inform replenishment proposals.
The model does not authorize an order, set a service target, or predict demand during an unobserved emergency.
Name whether the planner may override a recommendation and who makes the final commitment.
Specify the unit of time, currency where applicable, decision cadence, and any product or location restrictions.

**Branch:** if one model serves materially different decisions, document separate populations and evidence requirements.
A store replenishment use cannot automatically justify an allocation decision during a shortage.
If the organization cannot identify the decision owner or stop authority, do not infer approval from a successful data pipeline.

### Step 2 — Establish the population frame

**Input:** active master data, contract scope, effective dates, calendar, and relevant decision records.
**Action:** enumerate eligible units independently of the training and test tables.
**Output:** a reconciled frame with inclusion rules, exclusions, owner, snapshot date, and known defects.

For a weekly forecast, construct eligible SKU-location-week combinations from effective-dated assortment and open-location records.
Do not count a closed site before opening or a discontinued item after its last eligible week.
Check whether temporary closures, direct shipments, new items, and external partners belong in scope.
Use an independent control total such as shipped units, active contract value, or planner portfolio counts to challenge the frame.

A missing master record remains invisible to a coverage calculation based on that master.
Ask a second source owner to reconcile high-risk sites and categories.
Record unresolved discrepancies and possible direction of bias.
A denominator is an estimate with a lineage, not an oracle.

**Branch:** if no defensible frame exists, label coverage unknown.
A count of records received is not a substitute for the count that should have arrived.
Restrict use or authorize an explicit, time-limited exception while the owner builds the frame.

### Step 3 — Define material segments and regimes

**Input:** frame, business impact analysis, and domain expertise.
**Action:** define segments before looking for favorable model results.
**Output:** a segment dictionary and a list of intersections and operational regimes that need review.

Segment by geography, channel, product family, product lifecycle, customer class, supplier, and facility where relevant.
Define non-overlapping primary categories for aggregation; a site can also belong to overlapping risk slices.
State whether segments refer to present classification or classification as known at the historical decision time.
Record regime indicators for promotions, stockouts, closures, holidays, shortages, and demand spikes.

Consider intersections such as rural locations with cold-chain products.
A geography and a product category can each look adequate in isolation while their intersection is absent.
Avoid presenting every small cell as a reliable estimate; disclose counts and aggregate only when the decision permits it.
For people-related outcomes, involve appropriate privacy, legal, and affected-community expertise before collecting or reporting sensitive attributes.

**Branch:** when a segment has low volume but severe potential harm, require a separate qualitative or scenario assessment.
A volume-weighted average cannot override an independently defined safety or legal boundary.

### Step 4 — Define qualification rules for observed data

**Input:** decision grain, source contracts, feature list, labels, timestamps, and quality requirements.
**Action:** specify what makes a unit usable for training and separately for evaluation.
**Output:** executable rules and counts for eligible, received, usable, and excluded units.

A rule might require at least 26 observed weeks, a valid unit of measure, a known stockout indicator, and source arrival before the weekly decision cutoff.
That rule is illustrative; the right requirement depends on the model and business use.
Specify whether partial weeks, returns, transfers, negative quantities, and stock corrections are legitimate.
Define permissible values and a process for genuine exceptional values rather than silently clipping them.

Use separate flags for missing, invalid, late, duplicate, and unavailable at the historical decision timestamp.
Preserve raw values and reason codes when creating a cleaned version.
Distinguish a record missing entirely from a present row with a missing field.
Maintain a quarantine record so rejected rows do not disappear from the denominator.

**Branch:** a field may be optional for one model version and required for another.
Do not reuse a quality score without reviewing its dependency on the actual feature contract.

### Step 5 — Measure coverage using compatible denominators

**Input:** eligible frame, qualifying flags, and independent exposure weights.
**Action:** join at the decision grain and calculate counts and weights by segment.
**Output:** a coverage table with numerators, denominators, dates, and reasons for exclusion.

Record coverage = qualifying eligible units / all eligible units.
Exposure-weighted coverage = exposure of qualifying eligible units / exposure of all eligible units.
Compute each separately for training and evaluation.
Use annual unit volume, revenue, or another appropriate exposure only if it is defined consistently for covered and uncovered units.
If a unit has no known exposure, report the unknown amount or use a separately justified proxy; never quietly assign zero.

In the worked example, the unit is an active product-location pair, not a transaction or a week.
Each pair is counted once in the denominator for the stated scope.
In a live deployment, use pair-weeks when eligibility and coverage change week to week.
Avoid joining one-to-many transaction tables directly into the frame before aggregating; duplication can inflate both coverage and exposure.

**Branch:** record and exposure coverage can disagree substantially.
Report both and examine high-consequence low-volume groups separately.
Neither measure establishes that the observed sample is a random draw from its segment.

### Step 6 — Test quality at each boundary

**Input:** raw extracts, transformations, reference systems, and prior-period controls.
**Action:** evaluate field, record, cross-system, and temporal checks at source, ingestion, transformation, and feature-serving boundaries.
**Output:** a defect register with severity, affected units, age, owner, and root-cause status.

Completeness: compare required non-null fields and expected files with the agreed delivery schedule.
Validity: test controlled codes, units, sign conventions, and physically plausible ranges.
Uniqueness: test the business key and source revision semantics before deduplicating.
Consistency: reconcile quantities across order, shipment, receipt, and inventory records using explicit timing and conversion rules.
Timeliness: compare actual availability with the decision cutoff, not merely with a source-created timestamp.
Accuracy: sample against an independently verified reference, such as receiving documents or cycle counts, while recording reference uncertainty.

A failed check is a signal, not an automatic instruction to delete the record.
Determine whether the rule is wrong, the data is wrong, or a real business event is exceptional.
Track defect rates by segment: a global 99% complete field can conceal a 60% complete high-risk site.
Do not average percentages without compatible denominators.

**Branch:** if a critical field is unavailable before the decision cutoff, remove it or restrict the use unless a separately justified substitute is available.
A later correction cannot be used in a historical replay as if it had been known earlier.

### Step 7 — Investigate missingness and censoring

**Input:** exclusion reason codes, source process maps, stock status, and available follow-up evidence.
**Action:** compare observed and unobserved units and investigate why observations are absent.
**Output:** a missingness assessment with plausible mechanisms, impacts, and limitations.

Missing completely at random means missingness is unrelated to observed or unobserved values under a strong statistical assumption.
Missing at random means missingness can be explained by observed variables included in the analysis.
Missing not at random means missingness still depends on unobserved values after conditioning on observed information.
These are modeling assumptions, not labels that a missing-value chart can prove.

A small supplier without electronic integration may systematically lack data.
A store may show zero sales because it stocked out, not because no customers wanted the item.
A closed warehouse may report no shipments during the very disruption the model should support.
Late returns may make a recent demand label appear higher than its mature value.
Each mechanism can distort a different feature or label.

Compare availability by site, age, channel, regime, and consequence.
Interview source owners and planners to determine whether a missing value means zero, not applicable, late, or unknown.
Quantify how conclusions change under credible alternative assumptions.
Do not claim unbiased imputation where the underlying missingness mechanism cannot be verified.

### Step 8 — Evaluate representation across time

**Input:** training window, evaluation windows, segment history, and event calendar.
**Action:** measure how much each segment and regime contributes over time and whether labels are mature.
**Output:** a coverage matrix by segment, period, regime, and evaluation status.

Use a later evaluation period where appropriate for a forecast, and preserve the actual cutoff for every historical decision.
Include recurring and rare events when the available history supports them.
If the data includes no disruption comparable to the decision at hand, say so rather than calling the model disruption-tested.
Ensure a long history from one site does not mask newly opened sites with no observations.

Record the number of independent sites or events, not just the number of correlated weeks.
One long disruption affecting every site may not constitute many independent disruption examples.
A single promotional campaign may dominate an apparently large transaction sample.
Where clusters matter, use site- or event-level resampling or other methods appropriate to the dependence.

**Branch:** if evaluation labels are not mature, use leading indicators and hold final performance claims until labels settle.
Do not fill the gap with outcomes computed from incomplete returns or delayed inventory adjustments.

### Step 9 — Evaluate outcomes within supported slices

**Input:** point-in-time predictions, mature outcomes, baseline predictions, decisions, and eligible frame.
**Action:** compare the model and benchmark using segment-level error and decision measures.
**Output:** a table of sample sizes, uncertainty, performance, and uncovered slices.

For a demand forecast, examine signed bias, absolute error, and operational costs in units and currency where measurable.
Define the weighting and denominator of every metric.
Compare against a benchmark available at the same decision time and subject to the same scope.
Check that excluded observations are not silently dropped from both model and baseline.

Aggregate error is not a substitute for segment results.
A high-volume segment can hide large errors in a smaller critical category.
Conversely, a noisy result from three sites does not establish a systematic difference without further evidence.
State confidence intervals or other uncertainty where their assumptions are defensible, and report sample sizes regardless.

**Branch:** if a segment has no usable evaluation data, report performance as unknown.
If it has too little evidence, report the estimate as uncertain and use a proportionate restricted deployment or fallback.
Do not assign the overall error rate to an untested segment.

### Step 10 — Quantify selection and transport limits

**Input:** frame, sample membership, observed predictors, and performance by segment.
**Action:** compare training and evaluation composition with intended use, and assess whether evidence can transport to the target population.
**Output:** documented mismatch, assumptions, and sensitivity analyses.

Reweighting can adjust observed groups when their relative prevalence differs and their outcomes are sufficiently observed.
It requires support: relevant target cases must have a nonzero chance of appearing in the data.
Large weights can make the estimate unstable even when raw sample size is large.
Inspect the weight distribution and effective sample size, and show results with and without defensible weight caps.
Weight caps trade variance for bias and must not be presented as neutral.

No weighting scheme manufactures independent examples from a population that is completely absent.
Collect representative data, use a justified proxy with explicit uncertainty, conduct a controlled pilot, or keep the group out of scope.
Check whether the proxy differs in service constraints, customer behavior, or operating costs.
Ask whether a group is excluded because it is harder to measure, not because it is unimportant.

### Step 11 — Monitor drift with context

**Input:** frozen reference data, current production data, shared transformations, and monitoring calendar.
**Action:** compare population composition, input distributions, missingness, predictions, and mature outcomes.
**Output:** alerts with context, thresholds, owner, and investigation records.

Compare like with like: same grain, unit, segment definition, feature version, and season or operational context where possible.
Track absolute counts with rates; a 50% change from two to three observations differs from a change across thousands.
Separate a change in segment mix from a change within a segment.
Use both a fixed reference for long-term change and a recent reference for short-term changes when justified.

For a simple population stability index (PSI), define bins from a reference sample and hold them fixed.
For each bin, compute `(current share - reference share) × ln(current share / reference share)` and sum across bins.
This is a dimensionless descriptive statistic, not a probability of failure.
Zero shares need an explicitly documented smoothing rule; smoothing affects the result.
The reference and current sample sizes and bin definitions are part of the evidence.

Thresholds must be calibrated to the use, sample size, seasonality, and cost of missed incidents.
There is no universally binding PSI threshold.
A threshold crossing starts triage: check data quality, coding changes, business events, prediction behavior, and eventually outcomes.
A stable aggregate distribution does not prove each segment is stable.

#### Monitoring design: separate the signals

Define a monitoring record for each signal before production.
Its fields should include the signal name, owner, source, calculation, segment, unit, cadence, reference, threshold, and response.
The record should also identify how missing monitoring data is reported.
An unavailable monitoring feed is not the same thing as a passing quality check.

**Population signal:** count eligible and covered product-location pairs by segment.
Compare newly eligible pairs with the model's approved-use registry.
Investigate any pair receiving predictions before its segment is approved.
Report both the number of pairs and the proportion of relevant exposure.
If the population frame itself is late, report coverage as unavailable rather than reusing an unmarked old denominator.

**Quality signal:** track missingness, invalid values, late arrivals, duplicates, and failed reconciliations.
Separate a source's expected non-delivery from a genuine zero.
Record the oldest unresolved critical defect and affected decisions.
Use absolute counts alongside rates to make small denominators visible.
For a critical input, also report the number of decisions made with fallback inputs.

**Input distribution signal:** compare production features with the documented reference.
Check numerical variables and categorical codes separately.
For a new category absent from the reference, report its share explicitly.
Do not hide a novel category by mapping it to a generic category without recording that transformation.
Use segment-specific comparisons when aggregate mix can obscure a local change.

**Output signal:** monitor forecasts and recommendations for impossible values, extreme changes, and concentration.
Examples include negative demand where negative values are not meaningful, or a sudden jump in proposed orders for one facility.
Check output behavior against model version and input availability.
An output shift can reflect genuine demand change, a pipeline defect, or a model defect.
The output alone does not distinguish these explanations.

**Outcome signal:** once labels mature, compare forecasts with actual outcomes and a feasible baseline.
Monitor bias as well as absolute error.
Where possible, connect forecast errors to service, inventory, and other decision consequences.
Interpret these outcomes in light of stockouts, interventions, and the fact that the model may influence what is observed.
Delayed outcomes should be labeled pending rather than filled with provisional values presented as final.

#### Drift triage example

Suppose the current PSI crosses the organization's chosen trigger on Monday.
The operator first verifies that the monitoring calculation completed and used the expected reference version.
The operator checks whether the number of eligible observations and the missing-value rate changed.
The data steward checks deployment logs and source changes around the alert time.
The business owner checks the promotion calendar, site openings, and known disruptions.
The model owner compares the aggregate with A, B, and C segment distributions.

If a unit conversion changed, contain the affected feed and correct the transformation.
If the feed is correct but a promotion changed the business mix, check whether that regime was represented in evaluation.
If the promotion is untested, restrict the relevant decisions or apply the agreed fallback while evidence is collected.
If mature outcomes deteriorate, investigate the benchmark, inventory constraints, and decision overrides as well as the model.
An approved model change requires the organization's normal change and validation process.
Record the incident closure rationale even when the final decision is that no model change is needed.

#### Sampling and uncertainty

The number of rows is not necessarily the number of independent observations.
Fifty product-weeks from one facility during one promotion share local conditions.
A thousand daily observations from the same supplier may share one operational shock.
An interval calculated as if all observations were independent can therefore be too narrow.

Start by describing the sampling unit and the likely sources of dependence.
If facilities are the decision units and facilities are reasonably independent, resampling facilities may be more appropriate than resampling individual daily rows.
If a network-wide disruption affects all facilities together, facility resampling still may not capture uncertainty about future disruptions.
Report the number of sites and distinct events supporting the analysis.
Explain what the interval measures and which uncertainty remains outside it.

Rare but important groups create a further tradeoff.
Pooling groups can reduce noise but can hide systematic differences.
Keeping groups separate can preserve visibility but result in estimates too imprecise for a confident claim.
Use a policy-based minimum evidence requirement and show the actual sample sizes rather than silently suppressing weak cells.
Where authorized reporting of a small group risks disclosure, protect the data while ensuring the accountable reviewer can assess the risk.

#### Diagnose a composition shift

Imagine a network with urban and remote facilities.
Suppose the reference period contains 900 urban decisions and 100 remote decisions.
The current period contains 700 urban decisions and 300 remote decisions.
The remote share has risen from 100 / 1,000 = 10% to 300 / 1,000 = 30%.
Even if each facility type's internal demand distribution remains unchanged, the network-wide demand distribution can move.

That composition shift has two distinct implications.
First, the aggregate model input distribution may look different without any within-group change.
Second, the network now has more decisions in a group that may have limited independent evaluation.
The right action is not to declare the shift harmless because its cause is understood.
Compare the approved scope, group-level coverage, expected decision consequences, and available fallback.
If the remote facilities were never evaluated, their performance remains unknown regardless of their new population share.

To distinguish composition and within-group changes, calculate the monitored distribution separately for each group and period.
Keep the same feature definitions, bins, and data-availability rules.
Then calculate the group shares in each period.
Use a common weighting scheme as a diagnostic comparison, while preserving the real production-weighted result.
This standardization describes how composition influences the aggregate; it does not by itself estimate causal impact.

#### Define alert thresholds without false certainty

Thresholds should reflect the decision cadence and plausible cost of missed detection.
They can be estimated using historical variability, expert judgment, simulation, or a combination.
Document the method, review interval, and who may change the threshold.
Test how often the proposed rule would have alerted in known normal and abnormal historical periods when those periods exist.
Past alert frequency does not guarantee future sensitivity, particularly for previously unseen events.

Consider minimum sample requirements separately from the alert threshold.
Below the minimum, report insufficient evidence rather than automatically passing the signal.
For small but severe populations, supplement statistical monitoring with case review and process controls.
Use escalation levels that distinguish investigation, containment, and suspension.
Do not create an automatic suspension policy without confirming that its fallback is safer and operationally feasible.

Repeated testing creates many opportunities for random alerts.
Correlated signals can also trigger several alerts for one root cause.
Group related alerts into an incident without losing the underlying evidence.
Review missed incidents as well as false alarms when improving the policy.
Maintain a change history so a reviewer can see whether thresholds were relaxed after a failure.

### Step 12 — Assign decision rights and response

**Input:** quality failures, coverage limits, outcome evidence, and approved risk appetite.
**Action:** map findings to approved use, restrictions, remediation, escalation, and fallback.
**Output:** a signed decision record and a monitoring and incident plan.

The business owner is accountable for intended use and the operational fallback.
The data steward owns source definitions and defect remediation.
The model owner owns model-specific monitoring and technical investigation.
An independent validator challenges whether the evidence supports the proposed use.
A risk or designated approval authority accepts material residual risk; legal and privacy specialists advise on applicable obligations.
Actual authority follows the organization's policy, not a universal job title.

Record the trigger, detection time, affected scope, immediate containment, decision-maker, and re-entry criteria.
If a feed is late, the fallback might be last approved inputs only where their age is safe, or a manual plan with a named reviewer.
If a population is not validated, the fallback must be operationally available for that population.
If fallback quality is also unknown, document that risk rather than describing manual action as inherently safe.

### Step 13 — Preserve reproducible evidence

**Input:** source snapshots, code, model versions, decisions, and approval artifacts.
**Action:** retain a governed evidence package subject to access, security, and retention rules.
**Output:** a reviewer can reconstruct the denominator, exclusions, metrics, and approval as of the decision date.

Capture source extract identifiers, source availability times, transformation versions, and quality rule versions.
Store the segment dictionary, frame snapshot, exposure weights, exclusion reason codes, and reference period.
Record the model version, prediction timestamp, benchmark version, label maturity cutoff, and assessment code.
Preserve decisions and overrides with rationale, approver, expiry, and conditions for reconsideration.

A dashboard screenshot alone rarely shows how the denominator was built.
A reproducible package need not retain personal data indefinitely; design retention and access to meet applicable policy and law.
If the vendor controls raw data or code, negotiate appropriate audit evidence and document remaining limitations.

### Step 14 — Reassess after change

**Input:** changes to scope, suppliers, source schemas, business process, population, or model.
**Action:** determine whether existing evidence remains relevant and obtain additional approval where required.
**Output:** a change assessment linked to the inventory and previous validation.

A new region, acquisition, product family, or supplier can add a previously unseen population.
A unit conversion, source replacement, or feature definition change can invalidate earlier quality results without changing the model binary.
An automated ordering rollout can increase consequence even if forecast accuracy stays constant.
An exception expiry or repeated alert should trigger a decision rather than a silent renewal.

Revalidation scope should match the change and potential consequence.
More data may help, but collecting data without correcting selection, label quality, or decision context can make the wrong answer more confident.
A retrain is a controlled model change, not a universal response to every shift.

## 6. Worked example

### Decision and assumptions

A fictional distributor is considering a weekly demand forecast for replenishment planning.
All figures below are illustrative; they are not a claim about a named company.
The proposed model produces a forecast in units per week.
A planner uses the forecast to construct a plan; only an authorized buyer creates a purchase-order commitment.
The service-level target is a management objective, not a prediction of realized service.

The target population is 1,000 active product-location pairs in the coming year.
The independently reconciled frame is assumed complete for this illustration.
Pairs are mutually exclusive across three segments.
Annual exposure means expected demand units per year from the same planning baseline for every segment.
This is a proxy for operational exposure, not a measure of margin or harm.

| Segment | Decision context | Eligible pairs | Annual exposure (units) |
|---|---|---:|---:|
| A | Established urban locations | 600 | 720,000 |
| B | Newer regional locations | 250 | 210,000 |
| C | Remote locations with limited feeds | 150 | 70,000 |
| Total | All in-scope locations | 1,000 | 1,000,000 |

The historical training and evaluation coverage are measured against the same eligible frame.
A pair is covered only if it satisfies the predefined usable-data rule for that purpose.
Training and evaluation counts are not added together: the same pair can appear in both at different times.
Evaluation is later in time and uses mature outcomes.
A pair can be evaluated only when its point-in-time inputs and labels pass the specified checks.

| Segment | Eligible pairs | Training-covered pairs | Evaluation-covered pairs | Evaluation-covered exposure (units/year) |
|---|---:|---:|---:|---:|
| A | 600 | 540 | 180 | 216,000 |
| B | 250 | 160 | 40 | 33,600 |
| C | 150 | 0 | 0 | 0 |
| Total | 1,000 | 700 | 220 | 249,600 |

For clarity, the example assumes equal annual exposure per pair within each segment.
For A, 720,000 / 600 = 1,200 units per pair per year.
For B, 210,000 / 250 = 840 units per pair per year.
For C, 70,000 / 150 is approximately 466.67 units per pair per year.
The assumption is convenient, not generally realistic; real data should sum pair-level weights.

### Coverage arithmetic

Overall training record coverage is 700 / 1,000 = 70% of eligible pairs.
A's training coverage is 540 / 600 = 90%.
B's training coverage is 160 / 250 = 64%.
C's training coverage is 0 / 150 = 0%.
These percentages describe data meeting the rule, not the probability of an accurate forecast.

Training-covered annual exposure is:

- A: 540 × 1,200 = 648,000 units/year.
- B: 160 × 840 = 134,400 units/year.
- C: 0 units/year.
- Total: 782,400 units/year.

Training exposure coverage is 782,400 / 1,000,000 = 78.24%.
That exceeds the 70% record coverage because the higher-volume segment has better coverage.
It does not establish suitability for the remaining 21.76% of exposure.

Overall evaluation record coverage is 220 / 1,000 = 22%.
A's evaluation coverage is 180 / 600 = 30%.
B's evaluation coverage is 40 / 250 = 16%.
C's evaluation coverage is 0 / 150 = 0%.
Overall evaluation exposure coverage is 249,600 / 1,000,000 = 24.96%.
The evaluation sample is not the whole deployment population even though A and B are represented.

C represents 150 / 1,000 = 15% of pairs and 70,000 / 1,000,000 = 7% of annual unit exposure.
The lower exposure share does not settle the risk: C's remote locations may have higher shortage consequences or fewer fallback options.
That potential severity is a hypothesis to investigate, not an observed result in this fictional case.

### A quality defect and its consequences

Among B's 250 eligible pairs, 50 pairs have a late inventory feed.
Of those 50, 20 otherwise have an evaluation record and are excluded from the usable count.
The remaining 30 do not have an otherwise qualifying evaluation record.
The 40 B pairs in the table are already net of this exclusion; do not subtract the 20 again.

The late-feed rate is 50 / 250 = 20% of B's eligible pairs.
The affected otherwise-evaluable share is 20 / 250 = 8% of B's eligible pairs.
If those 20 pairs were repaired and all other criteria remained satisfied, B's evaluation count would rise from 40 to 60.
Its resulting evaluation coverage would be 60 / 250 = 24%, an 8 percentage-point improvement.
It would still leave 190 pairs without qualifying evaluation evidence.

The steward traces the late feed to an overnight batch that sometimes arrives after the ordering cutoff.
That diagnosis is an assumed case fact, not something the coverage percentage alone proves.
Using the corrected next-day inventory value in a historical test would leak information not available at decision time.
A repair requires earlier delivery or a genuinely available alternative, followed by a fresh point-in-time evaluation.

### Distribution change

For a monitored feature, the reference distribution has 60% low values and 40% high values.
The current distribution has 40% low values and 60% high values.
Bins are fixed in advance, counts are assumed large enough for this illustration, and neither share is zero.

PSI = (0.40 − 0.60) × ln(0.40 / 0.60)
      + (0.60 − 0.40) × ln(0.60 / 0.40).

The first contribution is approximately 0.081093.
The second contribution is approximately 0.081093.
The total is approximately 0.162186, a dimensionless statistic.
The symmetry occurs because the two shares swap; it is not a general property of every two-bin comparison.

Suppose the organization has an illustrative investigation trigger of PSI ≥ 0.15 for this feature.
The statistic crosses that local trigger, so the owner investigates.
It does not demonstrate that the model is inaccurate, nor does it mandate an automatic retrain.
A change in mix between A and B might explain the aggregate change; a within-segment comparison can distinguish that hypothesis.
Check seasonal comparability, source coding, missingness, and mature outcome metrics before deciding on a model change.

### Business decision and counterfactual

Assume a later controlled assessment estimates the following incremental annual contribution versus the current planner process.
These values are scenario estimates for the covered decision scope, not measured company results.
Positive values mean benefit relative to the fallback; negative values mean loss relative to it.
Both options incur USD 24,000 per year in fixed run and governance cost.
No incremental model benefit is credited for C, which remains on the existing process.
The counterfactual includes the existing planner process in every segment, including its existing costs and risks.

| Scope | A incremental contribution (USD/year) | B incremental contribution (USD/year) | Model run cost (USD/year) | Net incremental value (USD/year) |
|---|---:|---:|---:|---:|
| A only | 30,000 | 0 | 24,000 | 6,000 |
| A and B | 30,000 | −12,000 | 24,000 | −6,000 |

For A only, 30,000 − 24,000 = USD 6,000 per year.
For A and B, 30,000 − 12,000 − 24,000 = −USD 6,000 per year.
On these point estimates, adding B decreases annual value by USD 12,000 relative to A-only use.
This comparison assumes the same fixed operating cost in both options and no incremental transition cost.
The estimates cannot be extrapolated to all A or B pairs merely from the coverage table.
They require a causal or otherwise defensible estimate for the actual proposed rollout scope.

If the annual A contribution were USD 20,000 rather than USD 30,000, A-only net value would be −USD 4,000.
If the cost were USD 30,000 rather than USD 24,000, A-only net value would be USD 0.
The narrow margin warrants uncertainty analysis and a decision on whether nonfinancial consequences justify the program.
Do not add the USD 24,000 cost again to each segment if it is a shared fixed cost already counted once.

### Proposed disposition

Approve no automatic expansion to B or C on the evidence shown.
For A, consider a bounded planner-assist pilot only for the subpopulation supported by the evaluation, subject to the organization's approval policy and its assessment of uncertainty.
The business owner must specify how those supported pairs are identified in production.
Retain the existing planner process elsewhere until evidence supports a change.
This is an illustrative recommendation, not a universal threshold-based rule.

The data steward owns the B feed repair and must demonstrate on-time delivery before re-evaluation.
The model owner specifies an independent later evaluation and a benchmark for repaired B pairs.
The business owner assesses the C fallback, including service and shortage risk, and funds data collection if warranted.
An independent validator reviews the revised scope, selection mechanism, and outcome evidence.
The approver records the decision, expiration date, incident triggers, and authority to suspend use.

### What is not established

The example does not prove that A's evaluated pairs are representative of every A pair.
It does not establish causality for the illustrative USD estimates or quantify their confidence intervals.
It does not estimate lost demand during stockouts, impacts on individuals, or the cost of remediation.
It does not assume C is safe because its incremental model benefit is set to zero.
A complete business case would investigate these gaps before extending the use.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the segment coverage, exposure calculation, two-bin PSI, and business-case arithmetic in Section 6.
Counts and exposures are illustrative aggregated inputs; the program is not a production data-quality pipeline.
It assumes equal exposure per pair within a segment, nonzero denominators, nonzero PSI shares, and a common fixed annual cost.
It does not infer segment performance or generate evidence for C.

```python
from math import log

segments = [
    {"name": "A", "eligible": 600, "train": 540,
     "evaluation": 180, "exposure": 720_000},
    {"name": "B", "eligible": 250, "train": 160,
     "evaluation": 40, "exposure": 210_000},
    {"name": "C", "eligible": 150, "train": 0,
     "evaluation": 0, "exposure": 70_000},
]

eligible = sum(s["eligible"] for s in segments)
exposure = sum(s["exposure"] for s in segments)
train_count = sum(s["train"] for s in segments)
eval_count = sum(s["evaluation"] for s in segments)

train_exposure = sum(
    s["exposure"] * s["train"] / s["eligible"]
    for s in segments
)
eval_exposure = sum(
    s["exposure"] * s["evaluation"] / s["eligible"]
    for s in segments
)

for s in segments:
    train_pct = 100 * s["train"] / s["eligible"]
    eval_pct = 100 * s["evaluation"] / s["eligible"]
    print(f"{s['name']}: train={train_pct:.2f}% eval={eval_pct:.2f}%")

print(f"All: train records={100 * train_count / eligible:.2f}%")
print(f"All: train exposure={100 * train_exposure / exposure:.2f}%")
print(f"All: eval records={100 * eval_count / eligible:.2f}%")
print(f"All: eval exposure={100 * eval_exposure / exposure:.2f}%")

reference = [0.60, 0.40]
current = [0.40, 0.60]
psi = sum((c - r) * log(c / r) for r, c in zip(reference, current))
print(f"PSI={psi:.6f}")

annual_cost = 24_000
contribution_a = 30_000
contribution_b = -12_000
print(f"A only net USD/year={contribution_a - annual_cost:,}")
print(f"A and B net USD/year={contribution_a + contribution_b - annual_cost:,}")
```

Expected output:

```text
A: train=90.00% eval=30.00%
B: train=64.00% eval=16.00%
C: train=0.00% eval=0.00%
All: train records=70.00%
All: train exposure=78.24%
All: eval records=22.00%
All: eval exposure=24.96%
PSI=0.162186
A only net USD/year=6,000
A and B net USD/year=-6,000
```

In production, calculate exposure from pair-level weights rather than assuming equal weights.
Check eligibility and weights against an independently controlled frame before aggregating.
Require valid input types, unique keys, reconciliation totals, timestamp controls, and documented treatment of unknown exposure.
A PSI implementation must also define stable bins, zero-share handling, minimum sample sizes, and versioned references.
The sample intentionally avoids smoothing because all four shares are nonzero.

## 8. Data, standards, and regulatory context

### Data contracts and supply-chain identifiers

A useful data contract states the business meaning, owner, grain, identifier, unit, event time, arrival time, and expected update frequency of each field.
It also identifies mandatory fields, allowed values, revision behavior, quality thresholds, and the escalation contact.
A contract does not guarantee that every source conforms; monitoring and reconciliation provide evidence of actual behavior.

Global Standards 1 (GS1) publishes identification and data-sharing standards used across supply chains.
A Global Trade Item Number (GTIN), for example, can identify a trade item within the GS1 system.
A valid identifier does not prove the item is in scope, that a transaction is complete, or that a quantity is accurate.
Master-data changes, packaging hierarchies, and unit conversions still need effective dates and governance.
Use the standard and implementation guidance relevant to the actual trading relationship.

Electronic Data Interchange (EDI) communicates structured business documents between parties.
A compliant message format does not guarantee on-time delivery, correct business content, or comprehensive participation by small suppliers.
Map each source's document events to the model's decision-time availability and explain what happens when a partner does not transmit.

### Data quality standards

ISO/IEC 25012 describes a data quality model with characteristics relevant to data within computer systems.
It provides a vocabulary for quality attributes; it does not select a business-specific coverage threshold or certify a model's fitness for a decision.
An organization should translate general attributes into testable rules, accountable owners, and operational consequences.
The applicable edition and contractual requirements should be verified rather than assumed.

### AI and model-risk guidance

The US National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 is voluntary guidance.
Its Govern, Map, Measure, and Manage functions provide a useful structure for documenting context, measurement, and response.
The framework does not make a dataset representative merely because a checklist has been completed.
It also does not create a universal PSI threshold or replace a domain-specific risk assessment.

The US Federal Reserve and Office of the Comptroller of the Currency's 2011 SR 11-7 / OCC 2011-12 guidance addresses model risk management for supervised banking organizations.
For Federal Reserve supervision, [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm) superseded SR 11-7 in April 2026; consult the relevant agency's current scope.
It discusses robust development, effective challenge, outcomes analysis, and governance.
A nonfinancial supply-chain organization may adapt the principles voluntarily; applicability to an entity depends on its regulatory context.
Consult current supervisory guidance and counsel before making a compliance claim.

### Privacy, fairness, and law

Where personal data is involved, the European Union General Data Protection Regulation (GDPR) can impose obligations depending on territorial scope and processing context.
Relevant topics may include data minimization, accuracy, transparency, access, and lawful processing.
The legal basis and the treatment of sensitive attributes depend on facts and jurisdiction; this unit is not legal advice.
A segment analysis involving people should be designed with privacy and legal specialists, appropriate safeguards, and a clear purpose.

The EU AI Act uses a risk-based structure; whether any particular supply-chain use is covered and what obligations apply depend on its use and applicable implementation dates.
Do not infer that every forecast is high-risk, or that a general governance checklist establishes compliance.
Consult authoritative current law and qualified counsel for the actual system.

Excluded groups can be locations, suppliers, customers, or workers, and impact may be indirect.
A geographic proxy may track sensitive characteristics even when a protected attribute is not collected.
Where lawful and appropriate, evaluate meaningful consequences and engage affected stakeholders rather than assuming that absence of explicit demographic fields eliminates risk.

### Evidence and retention

Document consent or another lawful basis where required, access controls, purpose, retention schedule, and correction procedures.
Versioned source snapshots can support reproducibility but should not be retained indefinitely without a lawful and policy-compliant reason.
When individual records cannot be shared, an independent reviewer may need controlled access, aggregate evidence, or a privacy-preserving alternative.
Aggregate evidence can still reveal small groups; apply disclosure controls without suppressing material risk from authorized decision-makers.

## 9. Industry practice and case evidence

### M5 forecasting competition

The M5 forecasting competition used Walmart unit-sales data and evaluated hierarchical forecasts across products, stores, and aggregation levels.
Its public competition and results illustrate the importance of defining the evaluation population, hierarchy, horizon, and metric.
The published data and challenge do not establish that a winning approach generalizes to every retailer, disruption, or replenishment decision.
Sales observations are also not automatically unconstrained demand when stock availability limits purchases.
That last point is a methodological interpretation, not a claim that the competition authors measured all lost demand.

**Decision lesson:** report performance at operationally meaningful levels as well as the overall score.
Ask whether the real deployment includes new products, channels, and regimes absent from the benchmark dataset.
Evaluate the decision process and its costs separately from a public forecasting leaderboard.

### COVID-19 supply-chain disruption

The US National Academies report *Building Resilience into the Nation's Medical Product Supply Chains* describes vulnerabilities and resilience needs in medical product supply chains exposed during the COVID-19 pandemic.
This is evidence that supply-chain disruptions can change availability and operating context; it is not evidence about the performance of the fictional model in Section 6.
A historical dataset dominated by normal periods may not contain comparable disruption examples.
A model owner should identify that limitation and define human oversight and fallback for conditions outside observed support.

**Decision lesson:** a rare but material regime needs explicit treatment even when it contributes little to average transaction volume.
A continuity plan should identify who can override a forecast, where alternative data comes from, and which decisions cannot wait for mature labels.
Scenario analysis supplements but does not replace observed validation evidence.

### Retail stockouts and observed sales

The peer-reviewed work of Anupindi, Dada, and Gupta on estimating consumer demand with stock-out-based substitution examines how observed sales are affected when products become unavailable.
The methodological point is directly relevant to retail demand modeling: observed purchase quantities can differ from latent demand.
The work does not provide a universal correction factor for every category or supply chain.
A practical application needs availability records, substitution assumptions, and sensitivity analysis suited to its own setting.

**Decision lesson:** join sales with availability and identify whether zero sales coincide with a stocked-out period.
Do not label an unobserved lost sale as a verified zero.
Where stock status is missing, document what cannot be identified and test plausible alternative demand assumptions.

### A vendor evidence request

The following is a proposed practice, not a documented claim about a specific vendor.
Ask the supplier to provide target population definitions, training and evaluation windows, segment counts, exclusions, benchmark results, and known quality issues.
Require evidence that the customer can reproduce or independently inspect critical results under reasonable security constraints.
Ask how the vendor detects population shifts, not merely how often it retrains.

If the supplier cannot disclose raw training data, request controlled access, independently reviewed summaries, or contractually agreed audit artifacts.
These alternatives may reduce but not eliminate uncertainty.
A claim that a model was trained on millions of records does not show representation of the buyer's specific lanes, products, or disruption conditions.
A contract should identify the operational fallback and responsibility for notifying the buyer of material changes.

### Interpret case evidence carefully

Public cases often report a dataset, methodology, or organization-wide outcome but omit the exact denominator needed for a particular buyer's decision.
Separate documented observations from causal claims and from transfer assumptions.
An external result can motivate a hypothesis and a measurement design; it is not a substitute for local evaluation.
When an external case has different incentives, data collection, product mix, or decision constraints, state those differences explicitly.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Self-referential denominator | Coverage computed only from training table | Missing units never appear | Inflated coverage | Independent effective-dated frame and reconciliation |
| Hidden excluded region | No evaluation rows for a region | Source onboarding incomplete | Unsupported deployment | Restrict scope; collect and evaluate data |
| Exposure masking | High volume-weighted coverage but low site coverage | Large sites dominate weights | Small sites overlooked | Report counts, exposure, and severity |
| Count masking | Many records from a few sites | Repeated correlated observations | False confidence | Report independent sites and events |
| Late feature leakage | Backtest uses final corrected value | Historical availability ignored | Inflated test results | Point-in-time joins and arrival timestamps |
| Stockout censoring | Sales zero during unavailable inventory | Demand is constrained by supply | Underforecast and shortages | Availability data and sensitivity analysis |
| Wrong unit conversion | Sudden quantity jump after package change | Case and each units confused | Bad orders | Versioned units and reconciliation |
| Stale master frame | Active site missing from master | Frame maintenance lags reality | Unmeasured exclusion | Compare with contracts and operational systems |
| Silent duplicate | Total volume exceeds finance control | One-to-many join multiplies rows | Wrong weights and metrics | Unique-key and join-cardinality checks |
| Global green metric | Aggregate missingness low, one segment high | Unequal defects concealed | Disparate failure | Segment-level thresholds and ownership |
| Missing labels | Recent outcomes look unusually good or bad | Returns and adjustments not mature | Misleading performance | Maturity lag and separate leading indicators |
| Drift overreaction | Retrain after every alert | Shift treated as proven failure | Instability and wasted cost | Triage pipeline, context, and outcomes |
| Drift underreaction | Stable aggregate but worsening slice | Mix offsets within-group changes | Local harm remains hidden | Monitor both aggregate and material segments |
| Unsupported weighting | Extremely large weights or absent group | Weak or nonexistent support | Unstable or impossible estimate | Diagnose support, collect data, report limits |
| Invalid fallback | Manual process has no owner or capacity | Exclusion treated as risk transfer | Delayed or unsafe decisions | Test fallback and assign operational authority |
| Benchmark mismatch | Model and baseline use different sites or dates | Comparison selects different cases | False business value | Matched point-in-time evaluation scope |
| Vendor opacity | Only a global accuracy figure supplied | No evidence for local population | Unbounded residual risk | Contract evidence, independent challenge, restrict use |
| Uncontrolled exception | Waiver has no expiry or approver | Temporary gap becomes permanent | Persistent unowned exposure | Expiry, remediation, review and escalation |
| Privacy overcollection | Sensitive attributes gathered without purpose | Fairness analysis lacks safeguards | Legal and privacy risk | Purpose, lawful basis, minimization and review |
| Segment cherry-picking | Groups defined after seeing results | Selective reporting | Misleading approval | Predefine material slices and disclose changes |

Treat an alert as an invitation to investigate the mechanism.
A late feed, a source schema change, and a genuine market shift can all move the same summary metric.
The appropriate containment differs for each.
Record the difference between a confirmed defect, a hypothesis, and an unresolved concern.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or bound it |
|---|---|---|---|
| Frame includes eligible units | Gives coverage denominator | Missing sites or contracts | Reconcile independent operational sources |
| Eligibility dates are correct | Aligns units with decision period | Late master updates | Sample against opening and delisting records |
| Exposure weights are comparable | Supports weighted coverage | Missing weights or inconsistent units | Reconcile totals and report unknown weights |
| Within-segment weights are equal in the example | Allows aggregate arithmetic | Large differences between pairs | Sum actual pair-level weights in practice |
| Data rules match decision requirements | Defines usable observations | Model changes or hidden dependency | Review feature contract with model owner |
| Timestamps reflect availability | Prevents leakage | Backfilled or overwritten timestamps | Inspect ingestion logs and immutable snapshots |
| Labels are mature | Makes outcome comparisons meaningful | Delayed returns and corrections | Track revisions by label age |
| Benchmark is feasible | Defines relevant counterfactual | Baseline unavailable at decision time | Replay current process with same cutoff |
| Tested pairs resemble proposed rollout | Supports cautious transport | Selection by connectivity or performance | Compare characteristics and run sensitivity analyses |
| Segment definitions are meaningful | Exposes relevant risks | Important intersection omitted | Consult operators and review incidents |
| Sample has sufficient independent information | Supports uncertainty estimates | Correlated sites, weeks, or events | Count clusters and use suitable resampling |
| Reference period remains relevant | Makes drift comparison interpretable | Seasonality or business redesign | Compare seasonal and fixed references |
| Missingness mechanism is understood | Supports correction or imputation | Unobserved outcomes drive missingness | Compare sources and bound plausible alternatives |
| Fallback can actually operate | Contains out-of-scope risk | Insufficient staff or obsolete procedure | Exercise fallback and measure capacity |
| Monetary estimates are incremental | Avoids crediting unrelated changes | Confounding, double counting, scope mismatch | Document counterfactual and causal design |
| Costs are complete for the scenario | Makes value comparison interpretable | Remediation, transition, or support omitted | Itemize costs and test sensitivity |
| Legal interpretation is current | Avoids inappropriate compliance claims | New rules or different jurisdiction | Obtain current specialist review |

The worked example assumes facts for teaching and is not a recommendation for a particular organization.
When an assumption fails, state the direction and scale of possible impact where known.
Where the impact cannot be bounded, document that uncertainty and restrict the claim.
A precise percentage can still be materially wrong if its denominator is wrong.

## 12. Exercises

### Exercise 1 — Define the target frame

A distributor plans to forecast weekly demand for 200 active sites and 3,000 products.
Some products are not sold at every site, and some sites open midyear.
Transfers and customer sales appear in the same transaction table.

**Deliverable:** write the decision statement, observation grain, eligibility rule, and two independent frame checks.
**Acceptance criteria:** distinguish active product-site-weeks from the full Cartesian product; specify effective dates and how transfers are treated.
**Time:** 15 minutes.

### Exercise 2 — Calculate coverage

A segment has 400 eligible pairs and 300 training-covered pairs.
The eligible exposure is 500,000 units/year, of which 420,000 units/year belongs to covered pairs.
A separate test covers 80 pairs and 100,000 units/year.

**Deliverable:** calculate training and evaluation coverage by count and by exposure.
**Acceptance criteria:** give four percentages with denominators and explain why the two training percentages differ.
**Time:** 10 minutes.

### Exercise 3 — Identify an excluded population

The total frame contains 1,200 pairs across three segments.
A has 700 eligible pairs and 350 evaluated pairs; B has 300 eligible and 120 evaluated; C has 200 eligible and no evaluated pairs.
Annual unit exposure is 700,000 for A, 240,000 for B, and 60,000 for C.

**Deliverable:** compute count-based evaluation coverage for each group and overall; state C's exposure share.
**Acceptance criteria:** identify the untested group and do not infer its model accuracy from the overall value.
**Time:** 10 minutes.

### Exercise 4 — Diagnose data defects

A store has zero recorded sales during a week with no available stock.
A warehouse receives a product file one day after the forecast cutoff.
A supplier sends two versions of a shipment with the same business key and different revision timestamps.

**Deliverable:** classify each issue and propose one evidence-gathering action and one control.
**Acceptance criteria:** distinguish censoring, point-in-time availability, and revision-aware uniqueness; do not assume every duplicate is an error.
**Time:** 15 minutes.

### Exercise 5 — Compute and interpret drift

A feature has reference shares 0.50 and 0.50 and current shares 0.25 and 0.75 across two fixed bins.
All shares are nonzero and the organization has chosen an investigation trigger of 0.20 for this use.

**Deliverable:** calculate PSI to three decimal places and describe two next investigation steps.
**Acceptance criteria:** use the specified formula and do not equate the trigger with proven loss or a legal requirement.
**Time:** 15 minutes.

### Exercise 6 — Critique a deployment claim

A vendor says: “Our training set has ten million rows and our overall forecast error is 8%; therefore this model is validated for your remote facilities.”
The vendor supplies no population definitions or per-facility counts.

**Deliverable:** write a short challenge and list four evidence requests and an interim decision.
**Acceptance criteria:** distinguish training from evaluation and aggregate performance from the untested facility population; specify a fallback.
**Time:** 15 minutes.

### Exercise 7 — Compare business cases

Against the same current process, A is estimated to contribute USD 45,000/year and B is estimated to lose USD 8,000/year.
Fixed model and governance costs are USD 25,000/year for either A-only or A-and-B use.
There are no other assumed incremental costs or benefits.

**Deliverable:** calculate both net values and the incremental value of adding B; name two important uncertainties.
**Acceptance criteria:** count fixed cost once per option, state the counterfactual, and do not extrapolate the estimates beyond their scope.
**Time:** 15 minutes.

### Exercise 8 — Design governance and monitoring

A model is approved for a subset of sites, while a new supplier feed is scheduled for onboarding.
A recent drift alert coincides with a source-code change and a seasonal demand peak.

**Deliverable:** write an approval and incident plan with owners, containment, investigation, evidence, and re-entry criteria.
**Acceptance criteria:** include an independently challenged scope change, a usable fallback, data and outcome monitoring, and an expiry for any exception.
**Time:** 25 minutes.

## 13. Self-check answers

### Exercise 1

One defensible grain is one eligible product-site-week at the time a forecast is needed.
Define eligibility using effective-dated active assortment and site opening and closure dates, not all 600,000 possible product-site combinations every week.
Use customer sales as the demand-related observation only after establishing how stockouts, returns, and timing are handled.
Transfers are separate inventory movements unless the documented decision calls for them as inputs.
Reconcile the frame against site opening records and an independently controlled assortment or orderable-item list.
A strong response identifies who owns unresolved mismatches and how historical corrections are handled.

### Exercise 2

Training count coverage is 300 / 400 = 75%.
Training exposure coverage is 420,000 / 500,000 = 84%.
Evaluation count coverage is 80 / 400 = 20%.
Evaluation exposure coverage is 100,000 / 500,000 = 20%.
The training exposure percentage is higher because the covered pairs represent a larger share of exposure than of pair count.
These measures do not establish sample representativeness within the covered pairs.

### Exercise 3

A evaluation coverage is 350 / 700 = 50%.
B evaluation coverage is 120 / 300 = 40%.
C evaluation coverage is 0 / 200 = 0%.
Overall evaluation count coverage is (350 + 120 + 0) / 1,200 = 470 / 1,200, or approximately 39.17%.
C's exposure share is 60,000 / (700,000 + 240,000 + 60,000) = 6%.
C has no evaluation evidence in this scenario; its error is unknown.
Exposure-weighted evaluation coverage cannot be calculated from the information supplied because exposure of evaluated pairs is not given.

### Exercise 4

The zero sales may be censored by stock availability; check stock status and demand evidence, and flag the period rather than equating zero sales with zero demand.
The late warehouse file is unavailable at prediction time; inspect arrival logs and use only data available at cutoff in the replay.
The two shipment records may be valid revisions; check the source contract and revision sequence before choosing the version known at the decision time.
A control should retain the original records and the reason for selecting or excluding each revision.

### Exercise 5

PSI = (0.25 − 0.50) × ln(0.25 / 0.50) + (0.75 − 0.50) × ln(0.75 / 0.50).
The first contribution is approximately 0.173287 and the second is approximately 0.101366.
The sum is approximately 0.274653, or 0.275 to three decimal places.
It exceeds the stated local investigation trigger of 0.20.
Check that the feature definition and feed did not change, and compare seasonally matched segment distributions and mature outcome metrics.
No part of this calculation alone proves degradation or requires retraining.

### Exercise 6 rubric

A strong answer rejects the inference from row count and global error to local validity.
It requests a target frame, training and evaluation segment counts, independent time-based performance, and data quality and exclusions.
It asks for the metric definition and benchmark as well as evidence of remote-facility operating conditions.
Until evidence is sufficient, it limits use and names a tested, owned alternative for those facilities.
It treats a vendor's confidentiality constraints as a reason to negotiate evidence, not as proof of validation.

### Exercise 7

A-only net value is 45,000 − 25,000 = USD 20,000/year relative to the current process.
A-and-B net value is 45,000 − 8,000 − 25,000 = USD 12,000/year relative to that same process.
Adding B changes net value by 12,000 − 20,000 = −USD 8,000/year under these assumptions.
Possible uncertainties include whether the estimates are causal and whether the evaluated pairs represent the proposed rollout.
Transition costs, changes in fallback cost, and nonfinancial impacts may also change the conclusion.

### Exercise 8 rubric

A strong plan names a business owner, data steward, model owner, independent reviewer, and approval authority with explicit responsibilities.
It contains the affected scope, a safe operational fallback, alert triage, source-code and seasonal checks, and a record of the incident.
It defines re-entry evidence such as corrected point-in-time feeds, sufficient local evaluation, and documented authorization.
It monitors both current data and outcomes after labels mature.
A time-limited exception includes the rationale, residual risk, owner, expiration, and escalation if remediation is late.

## 14. Interview questions

1. **What is the population denominator for a forecast?**
   A strong answer defines eligible decision units and an independent frame with effective dates.
   A weak answer uses the training table row count as the denominator.
2. **Why can exposure-weighted coverage exceed count coverage?**
   A strong answer explains uneven exposure and reports both measures.
   A weak answer calls the higher figure proof of accuracy.
3. **What would you conclude about an untested region?**
   A strong answer says its performance is unknown and proposes bounded use or fallback.
   A weak answer assigns the global error to the region without evidence.
4. **How do you distinguish completeness from accuracy?**
   A strong answer gives an example of a present but incorrect value and a reference check.
   A weak answer treats non-null as ground truth.
5. **How can stockouts bias a demand forecast?**
   A strong answer distinguishes constrained sales from latent demand and proposes a sensitivity analysis.
   A weak answer assumes all zero sales reveal zero demand.
6. **What is point-in-time correctness?**
   A strong answer uses actual availability timestamps and historical revisions.
   A weak answer uses a final corrected extract in every historical decision.
7. **Does PSI above a threshold mean a model must be retrained?**
   A strong answer treats the threshold as a local investigation trigger and checks data, context, and outcomes.
   A weak answer treats a convention as a universal law.
8. **When does reweighting fail?**
   A strong answer identifies no support, extreme weights, dependence, and unverified transport assumptions.
   A weak answer asserts that weights can create observations for an absent population.
9. **How do you compare business value?**
   A strong answer specifies a feasible counterfactual, common scope, incremental benefits and costs, and uncertainty.
   A weak answer credits all revenue in the modeled segment to the model.
10. **Who may approve an exception?**
    A strong answer uses organizational decision rights, independent challenge, expiry, and accountable remediation.
    A weak answer allows the model developer to waive their own control indefinitely.

## 15. Key terms

- [Accuracy of data](#4-definitions-and-vocabulary): agreement with an appropriate reference.
- [Benchmark](#4-definitions-and-vocabulary): feasible alternative used for comparison.
- [Censoring](#5-the-mechanics-step-by-step): partial observation caused by a limit or process.
- [Completeness](#5-the-mechanics-step-by-step): required data present under a stated rule.
- [Concept drift](#4-definitions-and-vocabulary): change in the input-outcome relationship.
- [Coverage](#5-the-mechanics-step-by-step): qualifying eligible units divided by eligible units.
- [Data contract](#data-contracts-and-supply-chain-identifiers): agreement on meaning, delivery, and controls.
- [Decision population](#5-the-mechanics-step-by-step): eligible units for a named use and period.
- [Drift](#5-the-mechanics-step-by-step): change that warrants interpretation in context.
- [Evaluation coverage](#coverage-arithmetic): proportion with qualifying independent test evidence.
- [Exception](#5-the-mechanics-step-by-step): owned, time-limited control deviation.
- [Excluded population](#3-the-core-mental-model): eligible group outside evidence or authorized scope.
- [Exposure](#5-the-mechanics-step-by-step): decision volume or consequence associated with units.
- [Fallback](#5-the-mechanics-step-by-step): authorized alternative action.
- [Feature](#prerequisite-bridge): input known at prediction time.
- [Frame](#5-the-mechanics-step-by-step): independently maintained list or rule for eligibility.
- [Grain](#4-definitions-and-vocabulary): meaning of one observation row.
- [Label maturity](#5-the-mechanics-step-by-step): sufficient completion of an outcome.
- [Leakage](#4-definitions-and-vocabulary): use of unavailable information in fitting or evaluation.
- [Lineage](#5-the-mechanics-step-by-step): trace of source and transformations.
- [Missingness](#5-the-mechanics-step-by-step): absence of a required observation or value.
- [Point-in-time correctness](#5-the-mechanics-step-by-step): use of information actually available at the decision.
- [PSI](#5-the-mechanics-step-by-step): dimensionless distribution-change statistic.
- [Reference window](#5-the-mechanics-step-by-step): documented comparison period.
- [Regime](#5-the-mechanics-step-by-step): operating condition relevant to model behavior.
- [Reweighting](#5-the-mechanics-step-by-step): adjustment of observed sample contributions.
- [Segment](#5-the-mechanics-step-by-step): defined subset of the target population.
- [Selection bias](#5-the-mechanics-step-by-step): distortion from how observations are selected.
- [Timeliness](#5-the-mechanics-step-by-step): data availability within the allowed decision delay.
- [Validity](#5-the-mechanics-step-by-step): conformity to a stated rule or domain.

## 16. Sources

1. [NIST AI Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1), National Institute of Standards and Technology, 2023. Voluntary risk-management framework.
2. [SR 11-7: Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm), Board of Governors of the Federal Reserve System, 2011. Banking supervisory guidance; assess current applicability.
3. [ISO/IEC 25012: Data quality model](https://iso25000.com/index.php/en/iso-25000-standards/iso-25012). Overview of the data quality model; consult the applicable official standard for normative text.
4. [GS1 Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin), GS1. Official identifier description.
5. [General Data Protection Regulation, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj), EUR-Lex. Official legal text; seek current legal advice.
6. [Artificial Intelligence Act, Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), EUR-Lex. Official legal text; applicability and dates depend on the use and provisions.
7. [M5 Forecasting: Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy), Kaggle. Public dataset, rules, and evaluation context.
8. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. “The M5 accuracy competition: Results, findings, and conclusions.” *International Journal of Forecasting*, 2022. [DOI: 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013).
9. Anupindi, R., Dada, M., and Gupta, S. “Estimation of Consumer Demand with Stock-Out Based Substitution: An Application to Vending Machine Products.” *Marketing Science*, 1998. [DOI: 10.1287/mksc.17.4.406](https://doi.org/10.1287/mksc.17.4.406).
10. [Building Resilience into the Nation's Medical Product Supply Chains](https://doi.org/10.17226/26420), National Academies of Sciences, Engineering, and Medicine, 2022. Public report on supply-chain resilience.
11. Sculley, D., et al. “Hidden Technical Debt in Machine Learning Systems.” *Advances in Neural Information Processing Systems*, 2015. [Conference paper](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). Discusses dependencies and monitoring risks in deployed systems.
12. [Dataset Shift in Machine Learning](https://doi.org/10.7551/mitpress/9780262170055.001.0001), edited by Quiñonero-Candela, Sugiyama, Schwaighofer, and Lawrence, MIT Press, 2009. Background on changes between training and application distributions.

Sources describe particular methods, contexts, or legal texts; they do not validate the fictional case or its local thresholds.
Confirm current editions, effective dates, and legal applicability before operational reliance.

## 17. Links

- [Module 26 index](_index.md): topic sequence and scope.
- [Model inventory and tiering](01_model-inventory-tiering.md): decision exposure and control burden.
- [Conceptual soundness](02_conceptual-soundness.md): whether a method fits the intended decision and evidence.
- [Outcomes analysis and backtesting](04_outcomes-backtesting.md): temporal evaluation and benchmarks.
- [Ongoing monitoring and revalidation](09_monitoring-revalidation.md): production triggers and review.
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md): broader model-risk context.
- [Forecasting survey](../../../course/07_time_series_forecasting.md): forecasting methods and evaluation context.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): operational model exercises.

Sibling topics may be listed as planned in the module index until their files are published.
