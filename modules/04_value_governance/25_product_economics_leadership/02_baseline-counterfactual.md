# Baseline and Counterfactual

> Module 25 — Product, Economics, and Leadership
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: stakeholder and value taxonomy; basic descriptive statistics
> Version 2026.5

## 1. Why this topic exists

A supply-chain team proposes a new replenishment recommendation.
The sponsor reports that stockouts fell after launch.
The finance partner asks what would have happened without the recommendation.
That question determines whether the improvement is investable evidence or merely a coincident change.

The decision might be to expand to another distribution center (DC).
It might be to stop a pilot that imposes planner workload.
It might be to fund a controlled comparison before committing to a multi-year contract.
Each decision requires a comparison with an explicit alternative, not just a green dashboard.

The observed past is useful, but it is not automatically the future without the project.
Demand, supplier reliability, price, mix, season, and other programs can all change.
A warehouse can improve even while a tool has no effect.
A tool can help even while a severe external shock makes the warehouse look worse than before.
Both outcomes are possible because the missing comparison is not directly observable for the same unit at the same time.

A baseline describes a starting reference with a scope, definition, and time period.
A counterfactual describes the outcome under a specified alternative intervention.
The two may coincide as a forecast under stable conditions, but they are not synonyms.
For a pilot, the counterfactual is usually the current process operating in the same future conditions.
For a procurement decision, it might instead be a competing solution or a delayed implementation.

Incorrect comparisons are expensive in both directions.
An inflated benefit can lead to an unnecessary rollout and recurring license cost.
An understated benefit can prevent an effective process from reaching customers.
A weak comparison can also cause two projects to claim the same reduction in expedite spend.
The operational owner, finance, and the evaluation team need a shared claim boundary before that occurs.

Consider a control group whose suppliers improve at the same time as the pilot's suppliers.
A pilot-only before-and-after comparison credits the tool for both changes.
A credible control comparison can remove the common movement if its assumptions hold.
A control group is not automatically credible: selection, spillovers, and differential shocks can still bias it.
The evaluation design must specify how those threats will be tested or bounded.

This topic gives you a practical sequence from decision to evidence to finance handoff.
It separates causal effect, physical activity, cash realization, and a funding forecast.
It also teaches when the evidence is insufficient for a causal or monetary claim.
The worked case and executable calculation use synthetic data, not a claimed company result.

### Decisions this topic supports

- An operations leader chooses whether to deploy, modify, or stop a pilot.
- A product owner chooses which workflow and users to study.
- A finance partner accepts, rejects, or reserves a quantified benefit.
- A data owner verifies the measurement series and its lineage.
- A governance board decides whether evidence supports the scope of a rollout.
- A vendor manager distinguishes a vendor's claim from an independently measured effect.
- An analyst chooses a defensible comparison before outcomes are known.

A defensible answer can be that the effect is not yet identified.
That answer is useful when paired with a feasible next measurement or decision threshold.

## 2. Learning outcomes

After this topic, you can perform the following tasks without treating a target as an observed result.

1. Define the intervention, comparator, unit, outcome, horizon, and target population in an estimand.
2. Construct and critique a baseline from a reproducible source and a frozen metric definition.
3. Choose between randomization, phased rollout, matched comparison, interrupted time series, and a forecast-based alternative.
4. Calculate a difference-in-differences estimate and interpret its sign and units.
5. Diagnose selection, contamination, time-varying confounding, and missing outcome data.
6. Convert an incremental operational outcome to a finance-reviewed benefit without double-counting.
7. Report uncertainty, limitations, ownership, and a decision threshold in an executive-ready claim.
8. Design a measurement plan that remains reproducible after a metric or source changes.

Exercise 1 checks outcome 1.
Exercises 2 and 3 check outcomes 2 and 3.
Exercises 4 and 5 check outcomes 4 and 5.
Exercises 6 and 7 check outcomes 6 and 7.
Exercise 8 checks outcome 8.
Section 13 supplies calculation answers and design rubrics.

### Evidence of competence

- You can explain why a 10% reduction against last year is not automatically a 10% causal effect.
- You can identify the unit that was assigned, even when outcomes are recorded at order-line level.
- You can specify what happens to non-users in an assigned treatment group.
- You can show a calculation both per exposure and in the decision period's total units.
- You can distinguish a cash saving from a balance-sheet release or a productivity opportunity.
- You can state a condition under which the conclusion would change.

## 3. The core mental model

### 3.1 Two possible outcomes, one observed

For a defined unit and period, imagine the same supply chain under two alternatives.
One alternative receives the intervention; the other continues the comparator process.
Only one of those outcomes is observed for that unit in that period.
The missing outcome must be estimated from a credible design and stated assumptions.

```text
                           common target population
                                    |
                      specified decision and period
                                    |
             +----------------------+----------------------+
             |                                             |
      actual intervention                         comparator process
             |                                             |
      observed outcome Y(1)                     missing outcome Y(0)
             |                                             |
             +------------- credible design ---------------+
                                    |
                    estimate incremental physical effect
                                    |
            validate exposure, adoption, quality and uncertainty
                                    |
                translate only eligible effects to money
                                    |
               finance reconciliation and decision record
```

The diagram is a comparison of alternatives, not a chronological pipeline of two measured realities.
Random assignment or another design supplies information about the missing branch.
A pre-period alone describes history; it does not identify the missing post-period outcome.
The price of a weak comparison is not merely statistical imprecision.
It can reverse which project a leadership team chooses.

### 3.2 Four different numbers

A baseline is the documented reference level: for example, 80 expedited orders per 1,000 orders before launch.
A forecast is a prediction, such as 75 per 1,000 next quarter without a tool.
A target is a desired result, such as 65 per 1,000 next quarter.
A commitment is an authorized promise made by a named owner under specified conditions.
None is automatically an observed post-launch outcome or a causal effect.

A forecast may serve as a counterfactual estimate if its construction and uncertainty are credible.
A target does not serve as evidence merely because leadership approved it.
A commitment may be revised if dependencies fail; a measured effect still requires a comparator.
Label all four separately in the decision pack.

### 3.3 Chain from causal effect to economic value

The causal estimate is not the same as a recognized financial saving.
Start with incremental units, such as avoided expedites.
Map those units to an avoidable cost per unit, excluding unaffected fixed costs.
Subtract incremental operating and transition costs in the appropriate period.
Reconcile the estimate to the budget or ledger and name the person who can realize it.
Keep working-capital changes and recurring operating savings in separate categories.

An improvement in a leading metric is not a guarantee of downstream value.
A better forecast may change an order recommendation.
The planner may override the recommendation.
The supplier may be unable to execute it.
A service improvement may then be smaller than the forecast improvement suggests.
The effect on the endpoint captures the combined operating pathway, not an isolated algorithm score.

### 3.4 What the comparison identifies

An intention-to-treat effect compares units according to their assigned offer, whether they use it or not.
It answers the operational question: what happens if this eligible group is offered the intervention?
It includes uptake and implementation friction observed in that setting.
A comparison of users to non-users instead mixes adoption with the characteristics that caused adoption.
That contrast is generally not a causal effect of use.

A local pilot result need not apply to every supplier, DC, or season.
An effect can differ where the process, unit economics, or user behavior differs.
The decision pack should name the population to which a result is being transported.
It should also state which features are unsupported by the pilot evidence.

## 4. Definitions and vocabulary

The definitions below fix the meaning used throughout this lesson.

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Intervention | Change whose consequences are compared | Treating a bundle as a single model-only change | Version, dates, exposure |
| Comparator | Explicit alternative course of action | Calling a target the alternative | Process/version and period |
| Baseline | Documented starting reference for a defined metric and scope | Assuming the past is the missing future | Metric unit and pre-period |
| Counterfactual | Outcome that would occur under the alternative for the same target population and period | Equating it with any historical value | Same unit and horizon as outcome |
| Potential outcome | Outcome under one specified treatment state | Claiming both states were observed for one unit | Unit-period outcome |
| Estimand | Precise causal quantity of interest | Reporting a metric without population or contrast | Difference, ratio, or total |
| Experimental unit | Entity assigned to a condition | Treating each line as independently randomized | DC, route, supplier, or cluster ID |
| Observation unit | Entity on which an outcome is recorded | Confusing order lines with independent clusters | Order, line, DC-week |
| Exposure | Opportunity for the intervention to act | Counting ineligible orders in the denominator | Eligible orders or unit-time |
| Adoption | Actual use of offered capability | Assuming access means use | Active eligible users or decisions |
| Intention to treat | Comparison by original assignment | Dropping noncompliant units | Outcome per assigned group |
| Treatment-on-treated | Effect among units actually treated, under additional identification assumptions | Using the naive user/non-user gap | Outcome among treated units |
| Selection bias | Distortion from how units enter groups or analysis | Attributing every difference to the intervention | Difference in outcome units |
| Confounder | Common cause of assignment and outcome | Controlling for a post-treatment mediator by default | Variable and measurement time |
| Mediator | Variable on the pathway from treatment to outcome | Adjusting it away when estimating the total effect | Variable measured after assignment |
| Effect modifier | Pre-treatment characteristic associated with different treatment effects | Assuming a pilot effect is uniform | Subgroup and outcome units |
| Spillover | One unit's assignment affects another unit's outcome | Assuming a control group is untreated because it lacks a license | Exposure path and cluster |
| Contamination | Comparator deviates from the intended control condition | Ignoring shared planners or suppliers | Protocol deviation |
| Seasonality | Systematic pattern across a calendar cycle | Calling any fluctuation seasonal | Week, month, or year |
| Trend | Persistent directional change over time | Assuming one pre-period level is sufficient | Outcome units per period |
| Difference in differences | Difference in outcome changes between treated and comparison groups | Calling an unadjusted pre/post change causal | Outcome units per exposure |
| Parallel trends | Untreated group outcomes would have changed in parallel on the chosen scale | Claiming pre-trends prove the assumption | Difference in changes |
| Randomization | Allocation using a known chance mechanism | Assuming volunteers are randomly selected | Assignment probability and seed |
| Cluster randomization | Random assignment of groups of related observations | Using order-level sample size as independent sample size | Assigned cluster ID |
| Regression to the mean | Extreme observations tend to be less extreme on later measurement | Treating a selected bad month's improvement as proof | Outcome over repeated periods |
| Common support | Overlap in covariate patterns across comparison groups | Extrapolating to treated units with no comparable controls | Covariate range or score |
| Attrition | Missing units or outcomes after selection or assignment | Analyzing only completed records without checking why | Missing fraction by arm |
| Rate | Numerator divided by an explicitly defined denominator | Averaging unequal-exposure rates as if equal | Events per 1,000 eligible orders |
| Weighted average | Average that uses weights representing a target population | Equal weighting when the question is per order | Numerator and exposure |
| Absolute effect | Arithmetic difference between outcomes | Reporting a relative percentage without a reference | Events per 1,000 or currency |
| Relative effect | Absolute difference divided by the specified reference outcome | Omitting whether the denominator is baseline or counterfactual | Percent |
| Gross benefit | Monetized eligible causal improvement before relevant initiative costs | Presenting it as net cash benefit | Currency per period |
| Net benefit | Eligible benefit less specified incremental costs | Omitting implementation or recurring costs | Currency per period |
| Avoidable cost | Cost that actually changes when activity changes | Multiplying all activity by fully allocated cost | Currency per avoided event |
| Realization | Conversion of an estimated opportunity into evidenced financial or operational outcome | Booking an unapproved estimate as audited savings | Ledger, cash, or activity |
| Sensitivity analysis | Recalculation under alternative explicit assumptions | Presenting a scenario as a statistical confidence interval | Range with assumption labels |
| Confidence interval | Procedure-based uncertainty range for an estimand | Claiming it captures unmeasured bias automatically | Outcome units and confidence level |
| Placebo test | Test where the design predicts no effect, such as a pre-launch pseudo-date | Treating a passed test as proof of identification | Specified pseudo-outcome or date |
| Versioned metric | Definition and transformation tied to a reproducible release | Silently changing the numerator after launch | Version, timestamp, owner |

For all rates, write the numerator, denominator, time zone, event date, and eligibility rules.
For all currency figures, specify currency, price basis, and period.
For all populations, write both inclusion and exclusion rules.

## 5. The mechanics, step by step

### Step 1 — Name the decision and the alternatives

**Input:** the sponsor's proposed decision, budget, operational constraints, and decision date.
**Action:** write a specific choice between an intervention and a feasible alternative.
**Output:** a signed decision statement with a date and accountable owner.

For example: decide by the end of June whether to offer a recommendation workflow to ten additional DCs.
The alternative is to continue the existing planner workflow for the same period.
A second possible alternative is a competing vendor, which is a different comparison.
Do not combine the two into a single unnamed “do nothing” baseline.

Document whether the comparator includes routine maintenance and existing process improvements.
A genuine business-as-usual comparator may include changes already funded before the pilot.
Freezing all normal operational improvement can artificially inflate the intervention's apparent incremental value.
Conversely, giving the comparator a new unplanned investment changes the decision question.

Branch: if both alternatives are feasible and the decision can wait, design a comparative test.
If an urgent safety or service requirement prevents withholding, document the constraint and use the strongest feasible observational design.
If neither outcome nor decision owner can be defined, do not convert a model metric into a benefit claim.

### Step 2 — Specify the intervention as an implementable package

**Input:** workflow map, deployment plan, and change log.
**Action:** define eligibility, access, training, version, user authority, and operational actions.
**Output:** an intervention specification that another site can implement and audit.

A replenishment model is not the whole intervention if planners also receive training and a new escalation process.
A comparison of the full package to current practice estimates the package effect.
It does not separately identify the effect of the algorithm, training, and process change.
A factorial experiment can separate components when operationally feasible and adequately powered.
Otherwise, describe the bundle honestly.

Specify go-live by site and record actual availability.
A user receiving access halfway through a week creates a different exposure from a full week of use.
Predefine the handling of partial weeks and outage periods.
For intention to treat, retain assigned units despite an outage; report outage as implementation context.

Document control-side changes too.
Shared training materials, supplier negotiations, and revised service policies may affect both arms.
If shared changes are part of normal operations, the treatment contrast is the additional package on top of them.
If they leak from treatment to control, the contrast may be diluted or otherwise distorted.

### Step 3 — Define the estimand before inspecting post outcomes

**Input:** decision, intervention, comparator, and eligible population.
**Action:** write the estimand in a sentence and an equation.
**Output:** an analysis specification approved before outcome review.

One useful sentence is: “For eligible orders at assigned DCs during the next 12 weeks, estimate the change in expedite events per 1,000 eligible orders caused by offering the workflow instead of current practice.”
Here the intervention is an offer and implementation package, not mandatory use.
The population is eligible orders at assigned DCs, not all orders in the enterprise.
The horizon is 12 weeks, not an implicit annualized period.

Let `Y(1)` be the outcome under assignment to the package and `Y(0)` the outcome under assignment to current practice.
The average treatment effect for the defined population is `E[Y(1) - Y(0)]`.
For an undesirable outcome such as expedited orders, a negative effect indicates improvement.
Express a positive avoided-event rate as `-E[Y(1) - Y(0)]` only when the direction is clearly labeled.

Decide whether the estimand is order-weighted or DC-weighted.
An order-weighted effect answers the average eligible order question.
A DC-weighted effect gives a small DC the same weight as a large DC.
Both may be useful, but they answer different questions.
Changing weights after seeing the result is a source of selection.

Specify a primary outcome and a small number of operational guardrails.
For an expedite outcome, guardrails might include service level, backlog age, and inventory.
A decrease in expedites caused by simply refusing urgent shipments is not necessarily value creation.
State decision thresholds before examining the outcome, including acceptable harm on guardrails.

### Step 4 — Freeze the measurement baseline

**Input:** source extracts, metric definition, and a pre-intervention period.
**Action:** create a versioned baseline using data available before deployment.
**Output:** a reproducible baseline table, source snapshot, and signed metric dictionary.

Select a period that covers relevant operating cycles when data permit.
A full prior year can expose seasonal patterns, but it may also include an exceptional disruption.
A shorter recent period may better reflect the current process but fail to represent the seasonal mix.
Document the trade-off rather than choosing the window that creates the largest improvement.

Freeze the eligibility logic and record exclusions with reasons.
For example, exclude canceled orders only if cancellation is consistently defined and the rule is justified before analysis.
If the intervention itself can change cancellation, excluding canceled orders after treatment can introduce selection bias.
In that case consider an outcome that includes cancellations or separately report them as a guardrail.

Record source system, table or report, extract timestamp, timezone, and data owner.
Record whether the metric is based on order creation, shipment, invoice, or accounting posting date.
A shipment-date series and an order-date series are not interchangeable near period boundaries.
Keep immutable snapshots or reproducible point-in-time queries where policy permits.

Use a correction protocol rather than silently editing the frozen baseline.
A genuine source defect can be corrected with a version number, rationale, approver, and both old and new values.
Apply the same corrected definition to all groups and periods when possible.
Record whether the correction was proposed before or after outcome visibility.

### Step 5 — Understand the assignment and selection mechanism

**Input:** site selection notes, rollout criteria, historical performance, and operational constraints.
**Action:** reconstruct who was eligible, invited, assigned, accepted, and analyzed.
**Output:** a flow table and a list of plausible selection paths.

Volunteer sites may be more motivated, better staffed, or more digitally mature.
Sites selected because their latest expedite rate was extreme may improve naturally through regression to the mean.
Sites selected for urgent problems may also face shocks that continue into the pilot.
These mechanisms can push an unadjusted estimate in either direction.

Draw the causal paths before choosing covariates.
If a manager's capability affects both adoption and outcomes, it is a plausible pre-treatment confounder.
If adoption changes because the intervention is useful, adoption is on the post-assignment pathway.
Conditioning on adoption can change the estimand and introduce selection bias.

Compare treatment and control distributions of pre-period level, trend, size, mix, and constraints.
Balance on observed fields is informative but cannot verify balance on unobserved causes.
A model score with apparent overlap is not evidence that every relevant confounder was measured.
If comparable untreated units do not exist, restrict the target or state that the estimate is extrapolative.

Keep units in their originally assigned groups for the primary intention-to-treat analysis.
Record protocol deviations, missingness, and adoption separately.
A per-protocol estimate requires stronger assumptions and should be labeled secondary.
An instrumental-variable analysis is not a free correction for noncompliance: its relevance, exclusion, and other assumptions must be defensible.

### Step 6 — Select the strongest feasible design

**Input:** operational constraints, units available for allocation, and outcome data history.
**Action:** select a design and record the reason stronger alternatives are infeasible.
**Output:** a design decision with identification assumptions and planned diagnostics.

#### Randomized offer or rollout

Randomize eligible sites or other independent clusters when intervention spillovers make order-level assignment unsafe.
Use a documented random seed, allocation procedure, and assignment list stored before outcomes.
Stratify or pair on important pre-treatment characteristics when justified in advance.
The unit of analysis and inference must respect the assignment and dependence structure.

A randomized offer estimates the effect of being offered the package.
It does not automatically estimate the effect of perfect adoption.
Check assignment implementation, outcomes for every assigned unit, and spillovers.
With few clusters, conventional large-sample confidence intervals may be unreliable; consider design-based methods appropriate to the actual randomization.

#### Phased rollout

If rollout capacity requires a sequence, randomly assign eligible clusters to early and later waves when feasible.
Before later sites receive the intervention, they can serve as a contemporaneous comparator.
Account for calendar time and differences in exposure duration.
Do not treat later sites as untreated controls after their own intervention begins.

When timing is chosen by readiness, the schedule is not random merely because it is phased.
Readiness may predict both early deployment and improving outcomes.
With staggered adoption and heterogeneous effects, naive two-way fixed-effects comparisons can use problematic comparisons of already-treated and newly treated units.
Choose an estimator suited to the timing and target estimand; document valid control periods and weighting.

#### Matched or adjusted observational comparison

Find eligible controls using pre-treatment characteristics and history.
Check common support, balance, trends, and reasons for nonparticipation.
Matching on the last extreme outcome alone does not eliminate regression to the mean.
Matching cannot control an unmeasured factor that affects both treatment and outcomes.

Do not match on outcomes measured after treatment begins.
If site selection depended on a manager's unrecorded readiness, measured covariates may not remove selection bias.
Report the retained population after any trimming.
A result for matched sites is not automatically a result for excluded sites.

#### Interrupted time series

Use a sufficiently long and consistently measured sequence before and after a clearly dated intervention.
Model baseline level, slope, seasonality, and plausible serial dependence.
A contemporaneous comparison series can strengthen the design when it shares relevant external shocks.
A coincident policy change or supplier contract makes attribution difficult even if the series changes sharply.

#### Model-based counterfactual

Forecast the untreated outcome using only information available at the prediction origin.
Backtest on earlier periods and report error by horizon and regime.
Avoid using predictors that the intervention itself changes unless their counterfactual values are separately modeled.
A model can predict well in ordinary periods and fail during a structural break.
Report prediction uncertainty and scenario sensitivity, not only the point forecast.

### Step 7 — Specify the difference-in-differences calculation

**Input:** treatment and comparison outcomes in pre- and post-periods.
**Action:** calculate each change and subtract the comparison change from the treatment change.
**Output:** an effect estimate on the stated measurement scale.

Define `T_pre` and `T_post` as treatment rates and `C_pre` and `C_post` as comparison rates.
The estimate is `(T_post - T_pre) - (C_post - C_pre)`.
The implied treatment counterfactual is `T_pre + (C_post - C_pre)`.
The estimate is also `T_post - implied treatment counterfactual`.

The key assumption is that the treatment group's untreated outcome would have changed by the same amount as the comparison group's outcome on the selected scale.
This is parallel trends for the specified population and period.
It permits different starting levels; it does not require identical pre-period outcomes.
It does require no unaccounted differential change that would have occurred without treatment.

If the metric is a rate, define whether the parallel-trends assumption concerns absolute rate points or proportional changes.
Parallel changes on the additive scale and parallel changes on the logarithmic scale are different assumptions.
Choose the scale on substantive grounds before comparing favorable estimates.
When composition changes, aggregate rates can move even if every subgroup's rate is unchanged.

Use multiple pre-periods to inspect level and trend where possible.
A visibly different pre-trend challenges the design.
A statistically insignificant pre-trend test does not prove parallel trends, especially with few clusters.
A placebo date can reveal some problems, but cannot establish that no post-period differential shock occurred.

### Step 8 — Guard against selection and denominator changes

**Input:** assignment roster, eligibility rules, outcome records, and exposure counts by period.
**Action:** reconcile eligible, observed, missing, and excluded units in each arm.
**Output:** a population flow and missingness assessment.

Do not use only orders with complete delivery scans if treatment changes scanning behavior.
More complete scanning can make one group appear to have more events even when actual events are unchanged.
Compare source completeness and latency by arm and week.
Use a stable denominator defined independently of the observed expedite event.

Avoid dividing by delivered orders when the intervention can change delivery completion.
An intervention that reduces deliveries can then look better on a per-delivery metric for the wrong reason.
Consider all eligible order commitments as the denominator, plus service and cancellation guardrails.
If eligibility itself is changed by treatment, state how the target population is defined and consider a fixed pre-treatment cohort.

When order volume varies, aggregate events over aggregate exposure for an order-weighted rate.
Do not average weekly rates equally unless the estimand is the average week.
A high-volume week deserves more weight in a per-order estimate.
For inference, retain cluster and time information rather than treating all pooled orders as independent.

### Step 9 — Diagnose interference and concurrent changes

**Input:** planner rosters, shared supplier and carrier networks, deployment logs, and change calendar.
**Action:** map how an intervention can affect comparison units and identify simultaneous changes.
**Output:** an interference and co-intervention register.

A planner serving both arms can carry lessons from treated sites to control sites.
A carrier contract negotiated for the whole network can reduce expedites everywhere.
A supplier can prioritize treated sites by delaying control sites, increasing the apparent contrast.
Shared capacity means that interference can either dilute or exaggerate an effect.

Use assignment clusters that contain the most important spillover channels where feasible.
Alternatively define geographic buffers or choose non-overlapping supplier networks.
Record the cost of excluding units and the resulting loss of generalizability.
If interference cannot be prevented, specify the network-level intervention and estimand rather than pretending units are independent.

Ask operational owners about policy changes, promotions, holidays, disruptions, and staffing changes.
A control group exposed to the same shock can help remove common movement.
A shock that affects treated and control sites differently remains a threat.
An observed event calendar helps explain risk but does not prove the absent counterfactual.

### Step 10 — Quantify uncertainty and robustness

**Input:** unit-level outcomes, assignment mechanism, sample size, and design assumptions.
**Action:** estimate uncertainty appropriate to the assignment and dependence structure.
**Output:** interval estimate, diagnostic results, and labeled sensitivity scenarios.

For cluster assignment, correlations among orders within the same cluster reduce effective independent information.
Do not calculate a narrow interval as if every order were separately randomized.
Use cluster-aware or randomization-based inference consistent with the allocation and sample size.
Where only two clusters exist, do not claim a reliable conventional cluster-robust interval from those two clusters.

Separate sampling uncertainty from uncertainty due to assumptions.
A confidence interval conditional on the design does not quantify an unmeasured differential shock.
A scenario for a possible shock is a sensitivity analysis, not a confidence interval.
Show the size of bias that would move the estimate across the decision threshold.

Prespecify reasonable alternatives such as a different eligible window or fixed product-mix weights.
State whether each analysis changes the estimand or only tests robustness.
Do not try many specifications and present only the favorable one.
Disclose exploratory subgroup results as exploratory and avoid overinterpreting small cells.

### Step 11 — Translate the effect into finance-owned value

**Input:** causal physical effect, relevant exposure, unit-cost evidence, and implementation cost schedule.
**Action:** multiply only comparable incremental quantities and reconcile to the ledger.
**Output:** gross and net benefit with uncertainty, timing, category, and owner.

For fewer expedited orders, identify the incremental carrier premium avoided per event.
Exclude costs that remain unchanged, such as a fixed contracted minimum, unless contract terms show they can be avoided.
If an expedite prevents a lost sale, removing it might have a service cost; check the guardrail and account for material consequences.
If several projects reduce the same invoices, establish an allocation rule and a single ledger owner.

Multiply the avoided rate per 1,000 by the eligible post-period exposure divided by 1,000.
Use the actual exposure for a realized pilot estimate.
For a future rollout, forecast exposure and effect separately and label both as forecasts.
Do not simply multiply a 12-week estimate by 52/12 without a defensible seasonality and adoption assumption.

Separate one-time implementation cost from recurring operating cost.
Separate estimated economic opportunity from budget reduction and cash recorded.
If staff hours are freed but headcount and overtime do not change, the result may be capacity rather than cash saving.
A finance owner decides recognition under applicable accounting policy.

### Step 12 — Decide, archive, and monitor

**Input:** analysis, finance reconciliation, operational guardrails, and decision thresholds.
**Action:** compare evidence with the decision rule and document the outcome.
**Output:** a decision record and a post-decision measurement plan.

A decision can be expand, extend the test, change the workflow, or stop.
A small point estimate with wide uncertainty may justify a larger test rather than immediate scale-up.
A favorable economic estimate with a service guardrail failure may justify redesign.
A null or adverse result is still a useful learning outcome when the design is credible.

Archive assignment, analysis plan, baseline version, code, data lineage, exclusions, approvals, and results.
Restrict access according to data sensitivity.
Track changes after the pilot because cost, adoption, product mix, and vendor behavior can change.
Monitor both benefit realization and harms; neither a pilot result nor a target guarantees future performance.

## 6. Worked example

### 6.1 Decision and setting

A fictional distributor is considering a replenishment recommendation package.
The package combines a model, planner training, and a revised exception queue.
The decision is whether to expand the package after a 12-week pilot.
The comparator is the existing planner workflow plus routine operations.
All figures in this case are illustrative, not a named company's results.

The primary outcome is expedited orders per 1,000 eligible orders.
An expedited order is an order with at least one paid carrier expedite premium.
Multiple premium charges on the same order count as one event for the primary rate.
The source records each paid premium, so the order-level event requires deduplication.
The denominator is all eligible order commitments created in the period.
Canceled orders remain in the denominator; cancellation is also a separate guardrail.

The target population is eligible orders in two treatment and two comparison DCs.
Sites were selected for operational comparability, not randomly assigned.
The primary estimate is therefore an observational difference in differences, conditional on its assumptions.
The pilot cannot separately identify the model's effect from training or queue redesign.

### 6.2 Frozen baseline and outcomes

The pre-period and post-period each contain 12 weeks.
Both groups have 100,000 eligible orders in each period in this example.
Rates are pooled events divided by pooled orders, multiplied by 1,000.
The table uses actual event counts rather than rounded intermediate calculations.

| Group | Period | Eligible orders | Expedited orders | Rate per 1,000 orders |
|---|---|---:|---:|---:|
| Treatment | Pre | 100,000 | 8,000 | 80 |
| Treatment | Post | 100,000 | 6,000 | 60 |
| Comparison | Pre | 100,000 | 7,000 | 70 |
| Comparison | Post | 100,000 | 6,000 | 60 |

The treatment baseline is 8,000 / 100,000 × 1,000 = 80 events per 1,000 orders.
The comparison baseline is 7,000 / 100,000 × 1,000 = 70 events per 1,000 orders.
The groups do not begin at the same level.
Different levels do not invalidate difference in differences by themselves.
Different untreated changes would invalidate its key assumption.

The treatment rate falls by 60 − 80 = −20 events per 1,000 orders.
The comparison rate falls by 60 − 70 = −10 events per 1,000 orders.
An unadjusted treatment before-and-after calculation credits all 20 events per 1,000 to the package.
That calculation ignores the concurrent improvement in the comparison group.

### 6.3 Estimate the counterfactual

Assume the comparison group's 10-event decline is the best estimate of the treatment group's untreated decline.
The treatment counterfactual is therefore 80 + (60 − 70) = 70 events per 1,000 orders.
The observed treatment post rate is 60 events per 1,000 orders.
The estimated effect is 60 − 70 = −10 events per 1,000 orders.
Equivalently, `(60 − 80) − (60 − 70) = −10` events per 1,000 orders.
The estimated improvement is 10 avoided expedited orders per 1,000 eligible orders.

The absolute improvement is 10 events per 1,000 orders.
The relative improvement against the estimated counterfactual is 10 / 70 = 14.29%.
Using the pre-period baseline as denominator gives 10 / 80 = 12.5% instead.
Both ratios can be calculated, but they answer different questions.
The case reports 14.29% relative to the estimated post-period counterfactual.

The model's missing treatment-group outcome is not observed.
A comparison rate is evidence about that missing outcome only under the documented assumptions.
If treatment DCs would have improved by 15 events per 1,000 without the package, the true improvement would be 5, not 10.
If they would have improved by only 5, the implied improvement would be 15.
These are sensitivity scenarios, not statistical confidence limits.

### 6.4 Convert to period benefit

The treated post-period contains 100,000 eligible orders.
Estimated avoided events are 10 / 1,000 × 100,000 = 1,000 expedited orders.
The finance team estimates an avoidable carrier premium of USD 48 per avoided event.
This rate is a synthetic assumed incremental premium, not a fully allocated shipping cost.
The estimated gross pilot-period opportunity is 1,000 × USD 48 = USD 48,000.

Assume USD 13,000 of incremental operating expense in the 12-week pilot period.
Assume a one-time implementation expense of USD 22,000 incurred in that same period.
Estimated net pilot-period benefit is USD 48,000 − USD 13,000 − USD 22,000 = USD 13,000.
The calculation is undiscounted because all amounts are treated as occurring in the same short period.
It is an estimate, not an audited cash result.
A separate financial analysis should use actual cash dates and the organization's discount policy.

The effect of the package is not USD 48,000 of model-only value.
The pilot includes training and queue redesign.
The estimate does not prove the supplier invoice will fall by USD 48,000.
Finance must reconcile premium charges and contract minimums before recognizing savings.

### 6.5 Why a simple before-and-after case overstates value

A naive calculation uses 20 fewer events per 1,000 orders.
At 100,000 post-period orders, this implies 2,000 avoided events.
At USD 48 per event, it suggests USD 96,000 gross benefit.
The difference-in-differences estimate is USD 48,000 gross.
The naive estimate is USD 48,000 higher because it credits the common 10-event improvement to the package.
This comparison is illustrative: the adjusted estimate still depends on parallel untreated trends.

### 6.6 Sensitivity and operational checks

Suppose the comparison group's untreated improvement understates the treatment group's true untreated improvement by 5 events per 1,000.
The true treatment counterfactual would then be 65 events per 1,000, not 70.
The implied avoided rate would be 65 − 60 = 5 events per 1,000.
That is 500 avoided orders and USD 24,000 gross benefit.
After the same USD 35,000 of costs, the pilot-period net would be −USD 11,000.

Suppose instead that the treatment group's untreated improvement was 5 events per 1,000 smaller than the comparison group's.
The counterfactual would be 75 events per 1,000.
The implied avoided rate would be 15 events per 1,000.
That is 1,500 avoided orders and USD 72,000 gross benefit.
After USD 35,000 of costs, the pilot-period net would be USD 37,000.

The no-loss threshold on the assumed period costs is USD 35,000 / USD 48 = 729.17 avoided orders.
With 100,000 eligible orders, this is 7.29 avoided events per 1,000 orders.
This is a deterministic break-even calculation, not an inference interval.
The point estimate of 10 exceeds the threshold, but the 5-event sensitivity scenario does not.
A decision maker should consider the evidence for both scenarios and the cost of further measurement.

Check the percentage of eligible orders with a valid carrier invoice in both arms.
Check whether a carrier contract changed for one group but not the other.
Review pre-period weekly rates, site mix, and whether a shared planner learned from the package.
Compare service, cancellation, backlog, and inventory guardrails.
No numerical guardrail data are provided here, so no guardrail is asserted to have passed.

### 6.7 Interpretation and decision boundary

The estimate supports a conditional claim: under parallel untreated trends and the other design assumptions, the package is associated with an estimated 10 fewer expedited orders per 1,000 eligible orders in the 12-week pilot.
The calculation yields an estimated USD 13,000 net after the specified pilot costs.
It does not establish a statistically significant effect or a generalizable annual saving.
There are only four DCs in the illustration, and no cluster-level data or confidence interval are supplied.
The appropriate action may be a larger randomized rollout if the decision requires stronger evidence.
A rollout forecast should account for site mix, adoption, seasonality, cost, and uncertainty separately.

## 7. Code, query, or calculation

### 7.1 Runnable Python implementation

The following program requires Python 3 and only the standard library.
Save it as `counterfactual.py` and run `python counterfactual.py`.
It uses the four aggregate rows from the worked example.
It computes pooled rates, the difference-in-differences effect, monetary translation, and two sensitivity scenarios.
It does not calculate a confidence interval because aggregate rows do not provide independent cluster-level observations.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Period:
    eligible_orders: int
    expedited_orders: int

    def rate_per_1000(self) -> float:
        if self.eligible_orders <= 0:
            raise ValueError("eligible_orders must be positive")
        if not 0 <= self.expedited_orders <= self.eligible_orders:
            raise ValueError("expedited_orders must be within eligible orders")
        return 1000.0 * self.expedited_orders / self.eligible_orders


def evaluate(
    treatment_pre: Period,
    treatment_post: Period,
    control_pre: Period,
    control_post: Period,
    premium_usd: float,
    operating_usd: float,
    implementation_usd: float,
) -> dict[str, float]:
    t_pre = treatment_pre.rate_per_1000()
    t_post = treatment_post.rate_per_1000()
    c_pre = control_pre.rate_per_1000()
    c_post = control_post.rate_per_1000()

    control_change = c_post - c_pre
    counterfactual_rate = t_pre + control_change
    effect_rate = t_post - counterfactual_rate
    avoided_rate = -effect_rate
    avoided_orders = avoided_rate * treatment_post.eligible_orders / 1000.0
    gross_usd = avoided_orders * premium_usd
    cost_usd = operating_usd + implementation_usd
    net_usd = gross_usd - cost_usd

    return {
        "treatment_pre": t_pre,
        "treatment_post": t_post,
        "control_pre": c_pre,
        "control_post": c_post,
        "counterfactual_rate": counterfactual_rate,
        "effect_rate": effect_rate,
        "avoided_rate": avoided_rate,
        "avoided_orders": avoided_orders,
        "gross_usd": gross_usd,
        "cost_usd": cost_usd,
        "net_usd": net_usd,
    }


result = evaluate(
    treatment_pre=Period(100_000, 8_000),
    treatment_post=Period(100_000, 6_000),
    control_pre=Period(100_000, 7_000),
    control_post=Period(100_000, 6_000),
    premium_usd=48.0,
    operating_usd=13_000.0,
    implementation_usd=22_000.0,
)

print(f"Treatment pre rate: {result['treatment_pre']:.2f} per 1,000")
print(f"Treatment post rate: {result['treatment_post']:.2f} per 1,000")
print(f"Control pre rate: {result['control_pre']:.2f} per 1,000")
print(f"Control post rate: {result['control_post']:.2f} per 1,000")
print(f"Counterfactual: {result['counterfactual_rate']:.2f} per 1,000")
print(f"Treatment effect: {result['effect_rate']:.2f} per 1,000")
print(f"Avoided events: {result['avoided_orders']:.0f}")
print(f"Gross benefit: USD {result['gross_usd']:,.2f}")
print(f"Pilot costs: USD {result['cost_usd']:,.2f}")
print(f"Net benefit: USD {result['net_usd']:,.2f}")

for adjustment in (-5.0, 0.0, 5.0):
    adjusted_counterfactual = result["counterfactual_rate"] + adjustment
    avoided_rate = adjusted_counterfactual - result["treatment_post"]
    avoided_orders = avoided_rate * 100_000 / 1000.0
    net_usd = avoided_orders * 48.0 - result["cost_usd"]
    print(
        f"Counterfactual adjustment {adjustment:+.0f}: "
        f"avoided {avoided_rate:.2f} per 1,000; "
        f"net USD {net_usd:,.2f}"
    )
```

Expected output:

```text
Treatment pre rate: 80.00 per 1,000
Treatment post rate: 60.00 per 1,000
Control pre rate: 70.00 per 1,000
Control post rate: 60.00 per 1,000
Counterfactual: 70.00 per 1,000
Treatment effect: -10.00 per 1,000
Avoided events: 1000
Gross benefit: USD 48,000.00
Pilot costs: USD 35,000.00
Net benefit: USD 13,000.00
Counterfactual adjustment -5: avoided 5.00 per 1,000; net USD -11,000.00
Counterfactual adjustment +0: avoided 10.00 per 1,000; net USD 13,000.00
Counterfactual adjustment +5: avoided 15.00 per 1,000; net USD 37,000.00
```

### 7.2 Interpretation and production adaptation

A negative treatment effect means a reduction in an undesirable outcome.
The program prints avoided events as a positive number for ease of finance communication.
It permits an adverse result: a positive effect produces a negative avoided count and negative gross benefit.
The sensitivity adjustment changes the assumed counterfactual directly; it is not an observed shock.

The input rows are already pooled counts.
For real data, deduplicate by eligible order before aggregation and retain cluster and time identifiers.
A production analysis must validate date boundaries, historical source versions, completeness, and assignment.
If clusters differ in size, explicitly decide whether pooled or equally weighted cluster effects answer the decision.
Use the actual randomization and repeated-measure structure for any uncertainty interval.

The deterministic arithmetic does not prove parallel trends.
It does not correct for co-interventions, changes in eligibility, or leakage.
It cannot distinguish an algorithm effect from the package effect.
It assumes each avoided event saves the same USD 48 premium and all listed costs are incremental.
Those assumptions belong in the results, not just in code comments.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

Create a written contract among operations, data engineering, finance, and the evaluation owner.
The contract describes a reproducible extract rather than a screenshot of a dashboard.
Define the unit of observation, key, owner, source, refresh, retention, and correction policy.
For each field, record whether it is known before assignment or can be affected by treatment.

| Field | Purpose | Required interpretation |
|---|---|---|
| Site identifier | Assignment and clustering | Stable mapping across periods |
| Order identifier | Event deduplication | One eligible order per key |
| Assignment arm | Intention-to-treat grouping | Frozen original allocation |
| Assignment date | Exposure timing | Local and canonical timestamps |
| Eligibility flag | Target population | Versioned rule, including exclusions |
| Order creation date | Denominator period | Time zone and late-arrival policy |
| Expedite event | Outcome numerator | One or more paid premiums per order |
| Carrier charge | Financial reconciliation | Currency, contract, and invoice status |
| Source update time | Point-in-time reproducibility | Original versus corrected records |
| Product and customer class | Mix diagnostics | Predefined categories and changes |
| Adoption event | Implementation context | Not a post-treatment exclusion rule |
| Guardrail outcomes | Harm detection | Defined independently of primary result |

### 8.2 Metric and event lineage

Source systems may disagree about when an order exists or when an expedite occurs.
Order creation, warehouse release, shipment, invoice, and ledger posting are distinct events.
Choose the event that corresponds to the decision and state its latency.
Track late invoices and backfill windows consistently across treatment and control.
A cutoff that captures 95% of one group's invoices and 70% of the other's is not a fair comparison.

A reproducible pipeline should retain raw event identifiers and transformation versions.
Document joins that can multiply rows, such as orders joined to multiple carrier invoices.
Reconcile order counts and currency totals before calculating rates.
A missing join key is a data-quality problem, not automatically a zero-cost order.
Log and review data corrections rather than replacing historical numbers without trace.

Record calendar conventions and fiscal periods.
A 4-4-5 retail calendar and calendar months have different period lengths.
A holiday can move between fiscal periods from year to year.
If sites operate in different time zones, define the local business day before aligning events.

### 8.3 Identifiers and standards

GS1 identifiers can help align products and locations across systems when trading partners use them consistently.
For example, a Global Trade Item Number (GTIN) identifies a trade item, while a Global Location Number (GLN) identifies a location or party.
Identifier use does not guarantee correct mapping, event completeness, causal identification, or comparable costs.
Check mapping changes and the unit level, such as case versus each, before interpreting product mix.

GS1 Electronic Product Code Information Services (EPCIS) defines a standard for sharing visibility event data.
An EPCIS event can help describe what, when, where, why, and how an event occurred.
EPCIS does not specify the counterfactual, assign treatment, or determine whether an event caused a cost.
Its use is a convention or contractual requirement where adopted, not a universal legal mandate for this analysis.
Consult the relevant GS1 specifications and partner agreements for actual implementation requirements.

### 8.4 Financial and statistical governance

Finance policy determines whether an operational opportunity is recognized in a budget, forecast, or financial statement.
A causal estimate is not, by itself, an accounting entry.
Inventory reductions can release working capital once, while related recurring carrying costs require separate analysis.
Do not add a capital charge to a cash-flow model that already discounts the same invested capital without checking for double-counting.
Do not assume an allocated overhead rate is avoidable when volume falls.

For reporting statistical evidence, the American Statistical Association's statement on p-values is a useful professional reference.
It does not mandate a single threshold for a business decision or establish causal identification.
An uncertainty interval should reflect the actual design and what was sampled or randomized.
Sensitivity analysis should be distinguished from the stochastic interval.

### 8.5 Privacy and contractual constraints

Order, customer, employee, and supplier data can include confidential or personal information.
Apply the organization's access control, retention, and data minimization requirements.
Where the EU General Data Protection Regulation applies, the responsible organization must determine the lawful basis and applicable obligations for its processing.
This lesson does not determine whether any specific deployment falls within that regulation or provide legal advice.
Consult counsel and the data-protection team before combining or sharing identifiable data.

Vendor agreements can restrict data use, benchmarking, audit rights, and cross-client disclosure.
A license to use a tool does not necessarily grant permission to share order-level records externally.
Agree on data ownership, reproducibility, and exit access before the pilot.
De-identification lowers some risks but does not automatically make a dataset anonymous under every applicable law.

## 9. Industry practice and case evidence

### 9.1 Public evidence versus teaching scenarios

Public operational cases often reveal a result without revealing assignment, exclusions, or raw data.
A named organization and a large reported improvement do not establish a causal estimate.
The source may be a peer-reviewed study, company disclosure, vendor case study, or news account.
Those sources support different claims and have different risks of selective reporting.
The cases below distinguish documented facts from practical inference.

### 9.2 UPS ORION and routing decisions

**Documented fact:** UPS publicly describes its On-Road Integrated Optimization and Navigation (ORION) system as a route optimization capability.
Its corporate materials discuss route efficiency and operational use.
The company has also publicly reported estimated operational benefits in its own communications.
Those reports are company claims, not independently reproduced randomized treatment effects in this lesson.
See the UPS ORION source in section 16 for the company's description.

**Interpretation:** Route optimization changes a bundle of routing recommendations, driver workflow, and operational processes.
A valid value case would compare that bundle with the feasible routing alternative over a defined period.
The relevant denominator might be stops, routes, miles, or packages depending on the decision.
Miles per route can change with customer density and stop mix even without the tool.
An analyst should therefore record route characteristics and any changes in service obligations.

**Inference:** A comparison of routes offered the tool with routes not offered it may be biased if dispatchers select easier routes first.
Shared drivers can transfer knowledge across routes, creating spillovers.
A route-level experiment may be unsuitable if routes compete for the same vehicles or driver capacity.
A depot-level randomized rollout could reduce some interference but would have fewer independent units.
These are methodological implications, not claims about the company's actual experimental design.

**Economic boundary:** Fuel, driver time, vehicle wear, and service can change together.
Savings in driver time are not necessarily payroll savings if hours or staffing are unchanged.
The business case should avoid counting an already-included fuel component again inside a fully allocated mileage rate.
The source does not establish an independent causal effect for this lesson's hypothetical calculation.

### 9.3 Retail availability and consumer response

**Documented fact:** Corsten and Gruen examine retail out-of-stocks and consumer responses in peer-reviewed research.
The paper documents that shoppers can respond in multiple ways, including switching or leaving without the desired product.
It does not establish that any particular replenishment model produced a specific revenue uplift.
The study's context and outcomes should be read before transferring its findings to another retailer.

**Interpretation:** An availability improvement does not convert one-for-one into incremental company revenue.
Some demand moves to another size, brand, store, or time period.
Some demand may be lost entirely, but the share is context dependent.
A value case needs a comparator and a contribution-margin model consistent with actual substitution.
Treating every unavailable unit as a lost sale can substantially overstate value.

**Inference:** A store-level rollout can compare availability and sales while preserving a holdout group.
The analysis should consider customers who substitute across nearby stores and whether promotion calendars differ.
If the treatment affects replenishment and store execution together, the effect belongs to the package.
These recommendations are not a claim that the cited study evaluated this proposed intervention.

### 9.4 Randomized online experiments at Microsoft

**Documented fact:** Kohavi and coauthors describe controlled experiments at large online organizations, including practical difficulties in running trustworthy tests.
Their work emphasizes randomization, metrics, and experiment integrity.
The cited article is about online controlled experiments, not a supply-chain replenishment case.
It supplies a transferable measurement principle rather than a measured logistics benefit.

**Inference:** Supply-chain experiments must adapt those principles to physical interference and slower feedback.
A warehouse cannot always isolate one order's treatment from inventory shared by other orders.
Assignment at a larger cluster can prevent some leakage while reducing independent sample size.
A credible operational test needs both assignment integrity and a feasible operational workflow.
A statistically clean test of an impractical workflow is not a useful deployment decision.

### 9.5 National accounts of an observational method

**Documented fact:** The difference-in-differences literature describes designs that compare changes across treated and comparison groups.
Bertrand, Duflo, and Mullainathan show that serial correlation can invalidate conventional standard errors in common applications.
Their analysis is not a direct validation of any particular logistics pilot.
It supports the general warning that repeated observations are not automatically independent evidence.

**Inference:** Hundreds of thousands of orders can still contain little independent information when only a few DCs are assigned.
Finance should not accept a narrow interval solely because order count is large.
The analyst must identify the level of assignment and dependence before choosing inference.
Where the number of clusters is too small, seek additional independent clusters or report the limitation.

### 9.6 How to evaluate a vendor case study

1. Ask for the untreated alternative and exact population.
2. Ask whether the reported improvement is historical, forecast, target, or realized effect.
3. Ask for assignment, selection criteria, and adoption by group.
4. Ask whether a comparison group experienced the same calendar and supplier conditions.
5. Ask which costs are incremental and which were allocated.
6. Ask whether service, inventory, and labor guardrails were monitored.
7. Ask for the uncertainty, exclusions, and duration of observation.
8. Ask whether finance reconciled the claimed savings to invoices or budgets.
9. Ask whether the result is attributable to the vendor alone or to a broader change package.
10. Ask whether a signed reference is permitted to discuss unfavorable outcomes as well as successes.

A missing answer does not prove a claim is false.
It limits what the case study can support in your decision.
Use the published evidence to design a test, not to replace local validation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Historical baseline treated as future | No comparator after launch | External conditions also change | Coincident improvement credited to project | Use contemporaneous control or forecast with error |
| Target used as baseline | Slide labels a desired rate as actual | Goal substitutes for measurement | Inflated benefit gap | Separate target, forecast, baseline, and outcome |
| Volunteer-only pilot | Early sites are unusually capable | Adoption and performance share causes | Upward selection bias | Randomize offer or adjust with explicit limits |
| Worst-site selection | Sites chosen after unusually bad month | Regression to mean | Natural recovery appears causal | Use longer prehistory and eligible comparison sites |
| User-only analysis | Non-users disappear from denominator | Adoption depends on motivation or difficulty | Overstated intervention effect | Analyze assigned units; report adoption separately |
| Treatment changes eligibility | Exclusions rise after deployment | Post-treatment selection alters population | Incomparable rates | Define fixed cohort or include affected outcomes |
| Control leakage | Same planner uses insights in both arms | Comparison receives partial treatment | Contrast may be diluted | Cluster on workflow or measure spillover |
| Resource displacement | Treated sites receive scarce capacity | Control sites lose resources | Contrast exaggerates network effect | Measure network outcome and capacity transfers |
| Differential shock | One arm has a supplier outage | Untreated paths diverge | Biased difference in differences | Record shocks, redesign, or bound bias |
| Bad rate denominator | Orders and event counts use different date fields | Exposure is misaligned | Artificial rate movement | Align event window and reconcile counts |
| Unequal weighting | Mean of weekly rates ignores volume | Low-volume weeks dominate | Wrong target estimand | Pool numerator and denominator for per-order rate |
| Join multiplication | Multiple invoices per order | Duplicate numerator events | Inflated rate and costs | Deduplicate order events and reconcile ledger |
| Late-arriving data | Recent arm has incomplete invoices | Observation probability differs | False post improvement | Fixed maturity lag and arm-level completeness |
| Changed metric definition | A new dashboard filter appears post-launch | Numerator or denominator shifts | Artificial improvement | Version definition and recompute consistently |
| Inappropriate inference | Millions of orders, two assigned sites | Correlated observations treated as independent | Misleadingly narrow interval | Cluster-aware design and more independent sites |
| Unmeasured co-intervention | Contract and tool change same day | Alternative cause affects outcome | Package attribution unclear | Stage changes or identify bundle explicitly |
| Post-treatment adjustment | Adoption or inventory used as routine control | Mediator or collider conditioned on | Total effect biased or changed | Specify causal graph and estimand first |
| Cost double-count | Same carrier premium in two benefits | Shared ledger line claimed twice | Inflated portfolio value | Single owner and reconciliation bridge |
| Fixed cost treated as variable | Standard cost multiplied by saved events | Contract minimum or staffing persists | Phantom cash savings | Verify avoidability and contract terms |
| Service harm hidden | Expedites fall while backlog grows | Suppressed activity displaces harm | Net business value overstated | Predefine service and backlog guardrails |
| Annualization without basis | Twelve-week rate multiplied by 52/12 | Season and rollout differ | Unsupported annual benefit | Model exposure and adoption by season |
| Overgeneralized pilot | One geography assumed representative | Effect modification and new constraints | Rollout misses target | Compare target mix and validate new contexts |
| Selective specifications | Only favorable window reported | Analyst searches across outcomes | Optimistic result | Freeze analysis and disclose alternatives |
| False precision | Dollar result shown to the cent | Assumptions dominate calculation | Misleading confidence | Show meaningful precision and sensitivity |

A failure table is a planning tool, not evidence that every threat has been eliminated.
For every material red flag, assign an owner and a concrete diagnostic before interpreting the result.
Some threats cannot be resolved with available data and must remain in the limitation statement.

## 11. Assumptions and limits

The worked estimate is conditional on the following assumptions.
Diagnostics can challenge them; most cannot prove them from the observed data alone.

| Assumption | Why needed | What breaks it | Diagnostic or response |
|---|---|---|---|
| Well-defined intervention | Potential outcomes require a meaningful treatment contrast | Different sites receive materially different packages | Version and audit deployment; redefine bundle |
| Comparable untreated changes | Difference in differences uses the comparison trend | Site-specific supplier improvement | Inspect pre-trends and events; run sensitivity |
| No anticipatory effect | Pre-period represents untreated behavior | Sites change orders before formal launch | Log training and announcements; adjust window |
| Stable eligibility | Rates refer to a comparable population | Treatment changes inclusion or cancellation | Fix cohort; report eligibility and cancellation |
| Consistent measurement | Rates are comparable across groups and time | New scan process only in treatment | Audit field definitions and completeness |
| Limited interference | Comparator represents its assigned process | Planners carry treatment practices to control | Cluster or measure exposure; redefine estimand |
| No differential co-intervention | Outcome change is not caused by another arm-specific change | New carrier contract only in treatment | Review change calendar and contract terms |
| Adequate overlap | Comparison is relevant to treatment population | No controls for large urban DCs | Restrict scope or collect comparable sites |
| No selective missing outcomes | Observed data represent assigned units | Failed orders lack invoices in one arm | Compare missingness; bound or investigate |
| Appropriate scale | Parallel change applies to selected rate scale | Multiplicative trends with additive model | Justify scale and examine alternatives |
| Correct exposure weighting | Aggregate effect matches the intended order population | Large site shifts product mix | Report site/mix weights and standardized rates |
| Stable unit cost | Avoided events save the assumed amount | Contract minimum or rate tier | Inspect invoices, tiers, and marginal charges |
| Costs fully specified | Net result covers incremental resources | Training or integration omitted | Reconcile procurement, payroll, and operating records |
| Guardrails capture relevant harm | Monetary result is not offset by hidden losses | Backlog or customer loss not measured | Add and review service and quality outcomes |
| Transportability | Pilot evidence informs rollout sites | New sites have different suppliers or processes | Compare target characteristics; phased validation |
| Suitable uncertainty method | Interval reflects dependence and allocation | Treating orders as independent random units | Use assignment-aware inference and enough clusters |

A favorable pre-trend is supportive but not proof of the unobserved post-period trend.
A randomized assignment protects against systematic pre-assignment differences in expectation, not against missing outcomes or interference.
A precise estimate can be precisely biased if a design assumption fails.
A large sensitivity range can coexist with a narrow sampling interval.

The example's USD 13,000 net figure depends on assumed cost and event savings.
The four aggregate rows do not allow a defensible estimate of sampling uncertainty.
The example should not be presented as a statistically verified deployment recommendation.

## 12. Exercises

Work independently before consulting section 13.
Each exercise names a deliverable, acceptance criteria, and suggested time.
Use the definitions and sign conventions in this topic.

### Exercise 1 — Write the estimand

**Scenario:** A retailer offers a replenishment tool to randomly selected stores for eight weeks.
The existing planner process remains available to all stores.
The primary outcome is stockout days per 1,000 eligible SKU-days.
Some assigned stores never log in.

**Deliverable:** One sentence specifying the intention-to-treat estimand and its population, comparator, outcome, and horizon.
**Acceptance criteria:** Retain non-users in the assigned group; state the eight-week horizon and the rate denominator.
**Time:** 10 minutes.

### Exercise 2 — Select a credible baseline

**Scenario:** A DC reports 12% fewer late orders than the previous month after implementing a dashboard.
The previous month contained a major port disruption; a network-wide carrier contract changed at launch.
The DC processed 20% fewer orders in the current month.

**Deliverable:** A short baseline and data plan plus two reasons the 12% figure is not an identified effect.
**Acceptance criteria:** Define a rate denominator, address the disruption and contract change, and distinguish reference from counterfactual.
**Time:** 15 minutes.

### Exercise 3 — Choose a design

**Scenario:** Twenty comparable warehouses will adopt an exception queue over six months.
Training capacity permits five warehouses per month for four months.
Some planners serve warehouses in more than one proposed wave.

**Deliverable:** A rollout evaluation design and a diagram of its assignment unit and comparison periods.
**Acceptance criteria:** Address random timing, planner spillover, valid untreated control periods, and cluster-aware inference.
**Time:** 20 minutes.

### Exercise 4 — Calculate the effect

**Scenario:** Treatment late-delivery rates change from 90 to 72 per 1,000 eligible orders.
Comparison rates change from 80 to 74 per 1,000 eligible orders.
There are 50,000 eligible treatment orders in the post-period.

**Deliverable:** The implied treatment counterfactual, difference-in-differences effect, and avoided late orders.
**Acceptance criteria:** Show units, the sign of the effect, and all arithmetic.
**Time:** 10 minutes.

### Exercise 5 — Diagnose selection and leakage

**Scenario:** The ten warehouses with the worst previous month's rates are invited first.
Five volunteer, and the report compares their active users with non-users at the remaining sites.
A regional planner serves both groups and shares the new playbook.

**Deliverable:** A table of at least four threats with direction where knowable and a mitigation.
**Acceptance criteria:** Include regression to the mean, volunteer selection, user conditioning, and spillover; do not assert an unknown bias direction as fact.
**Time:** 20 minutes.

### Exercise 6 — Value an operational effect

**Scenario:** A credible estimate is 600 fewer premium shipments in one quarter.
The avoidable carrier premium is USD 35 per shipment.
The team spent USD 5,000 on incremental operating costs and USD 8,000 on implementation in the quarter.
A separate project claims USD 4,000 of the same carrier savings.

**Deliverable:** Gross and net quarter benefit and an overlap reconciliation proposal.
**Acceptance criteria:** Show arithmetic, avoid adding the overlapping claim, and distinguish estimated from recognized cash.
**Time:** 15 minutes.

### Exercise 7 — Report uncertainty for a decision

**Scenario:** An estimated effect is 8 avoided events per 1,000 orders.
The decision threshold is 6 avoided events per 1,000 orders.
A design-based 95% confidence interval is 2 to 14 avoided events per 1,000 orders.
A sensitivity scenario for an unmeasured shock changes the point estimate to 4.

**Deliverable:** A three-sentence decision brief that separates sampling uncertainty from the shock scenario.
**Acceptance criteria:** Mention the threshold, the interval, the scenario, and a defensible next action without claiming certainty.
**Time:** 15 minutes.

### Exercise 8 — Build a measurement contract

**Scenario:** A carrier invoice table can contain several charge lines per order and invoices arrive up to six weeks late.
A new scanner is installed only at treatment sites during the pilot.
The finance owner wants a monthly cash-savings dashboard immediately.

**Deliverable:** A one-page contract covering identifiers, definition, latency, data quality, ownership, and correction policy.
**Acceptance criteria:** Prevent duplicate events, state a maturity cutoff, address differential measurement, and separate estimated benefit from ledger realization.
**Time:** 25 minutes.

## 13. Self-check answers

### Exercise 1 — Answer

Estimate the average eight-week difference in stockout days per 1,000 eligible SKU-days for all eligible stores assigned to receive the offer, compared with their outcomes under the existing planner process.
The assignment, not login, determines the primary comparison.
The result is an offer-and-implementation effect, not necessarily the effect of actual usage.
A correct answer states the relevant eligible store population and keeps the same outcome definition in both arms.

### Exercise 2 — Rubric

A strong response calculates late orders divided by eligible orders, not simply the change in late-order counts.
It documents the pre-period disruption and checks multiple prior periods, seasonality, and product mix.
It records the carrier contract as a co-intervention and searches for a comparison with similar exposure.
A network-wide contract could create a common shock, but heterogeneous site effects remain possible.
The chosen pre-period is a baseline; a credible estimate of current outcomes without the dashboard is the counterfactual.
A strong response proposes a frozen metric, owner, source, and date rather than selecting a favorable month after results.

### Exercise 3 — Rubric

Randomly assign eligible warehouse clusters to rollout waves before looking at post outcomes.
Keep later-wave sites as controls only until they receive the queue.
Group warehouses sharing a planner where feasible, or measure and explicitly address the leakage.
Specify the effect by exposure duration and calendar period.
Use inference appropriate to the randomized cluster and recognize that only 20 warehouses may limit precision.
A strong response acknowledges that warehouse readiness or staffing may constrain randomization and documents deviations.

### Exercise 4 — Answer

Treatment change is 72 − 90 = −18 late deliveries per 1,000 orders.
Comparison change is 74 − 80 = −6 late deliveries per 1,000 orders.
The treatment counterfactual is 90 − 6 = 84 late deliveries per 1,000 orders.
The effect is 72 − 84 = −12 late deliveries per 1,000 orders.
Equivalently, `(72 − 90) − (74 − 80) = −12` per 1,000 orders.
The estimated improvement is 12 avoided late deliveries per 1,000 eligible orders.
At 50,000 eligible post-period orders, the estimate is 12 × 50 = 600 avoided late deliveries.
The causal interpretation depends on the design assumptions, including parallel untreated trends.

### Exercise 5 — Rubric

Selecting a particularly bad month creates regression-to-the-mean risk.
Volunteers may have unusually capable managers or unusually urgent problems, so bias direction is not guaranteed.
Restricting to active users conditions on post-offer behavior and changes the estimand.
The shared planner can carry practices into comparison sites, potentially diluting the contrast.
Capacity transfers or other network effects can instead exaggerate or reverse the contrast.
A better design randomizes an offer across eligible clusters, retains assigned non-users, and documents pre-period history.
Use a cluster boundary aligned with shared planners where practical.
Do not claim that a single adjustment removes unmeasured readiness differences.

### Exercise 6 — Answer

Gross estimated benefit is 600 × USD 35 = USD 21,000 for the quarter.
Specified incremental costs are USD 5,000 + USD 8,000 = USD 13,000.
Estimated net benefit is USD 21,000 − USD 13,000 = USD 8,000 for the quarter.
The USD 4,000 overlapping claim is part of the same USD 21,000, not an extra portfolio benefit.
Assign the shared invoices to one benefit owner or agree an allocation that sums to no more than the reconciled total.
Finance must verify actual contract avoidability and ledger realization before calling the estimate recognized cash.

### Exercise 7 — Example brief

The point estimate of 8 avoided events per 1,000 exceeds the 6-event decision threshold, but the 95% interval of 2 to 14 crosses it.
A separate unmeasured-shock sensitivity scenario lowers the point estimate to 4, below the threshold; it is not a confidence interval.
If the rollout is difficult to reverse, extend or randomize the test and investigate the shock before committing, subject to the cost of delay.
A different decision can be defensible if decision costs and reversibility are explicitly considered.

### Exercise 8 — Rubric

Define a stable order key and deduplicate charge lines to one expedited-order event.
Reconcile invoice totals separately from the order event count.
Set a six-week maturity lag or quantify and disclose remaining incomplete invoices using a consistent rule.
Audit scanner coverage by arm and time; do not assume a post-launch increase in scans is a true increase in events.
Assign a source owner, finance reconciler, metric owner, and correction approver.
Version the eligibility and metric rules and retain the old and corrected results.
Label the monthly dashboard's preliminary estimate separately from mature invoice-based and ledger-realized amounts.

## 14. Interview questions

### 1. What is the difference between a baseline and a counterfactual?

A strong answer defines a baseline as a documented reference and a counterfactual as the unobserved outcome under a specific alternative in the relevant period.
A weak answer treats last year's value as automatically equal to this year's untreated outcome.

### 2. What does an intention-to-treat effect answer?

A strong answer compares all eligible units according to original assignment and interprets the effect of offering the package, including uptake.
A weak answer removes non-users and calls the remaining contrast randomized.

### 3. When can a before-and-after comparison be misleading?

A strong answer identifies seasonality, trend, shocks, regression to the mean, measurement changes, and other simultaneous interventions.
A weak answer assumes an improvement following launch must have been caused by launch.

### 4. Explain parallel trends without requiring equal baseline levels.

A strong answer states that untreated outcomes would have changed in parallel on the chosen scale, even if pre-period levels differ.
A weak answer says the groups must have identical starting rates or that a passed pre-trend test proves the assumption.

### 5. Why randomize by warehouse rather than by order?

A strong answer discusses shared inventory, planner decisions, supplier capacity, and interference, then notes the loss of independent units.
A weak answer ignores the assignment unit and counts every order as an independent experiment.

### 6. What does regression to the mean imply for a worst-site pilot?

A strong answer explains that selection on an extreme noisy period can produce subsequent improvement without treatment and proposes a longer history and comparable controls.
A weak answer assumes the improvement proves that the selected sites benefited.

### 7. How can a better metric produce a worse business outcome?

A strong answer gives a mechanism such as fewer expedites caused by delaying shipments, and names service, backlog, and inventory guardrails.
A weak answer assumes every reduction in an operational activity is a saving.

### 8. How would you value avoided expedites?

A strong answer multiplies incremental avoided events by an evidenced avoidable premium, subtracts incremental costs, and reconciles contract minimums and overlapping claims.
A weak answer applies fully allocated shipping cost and calls the result audited cash.

### 9. What is the distinction between an interval and a sensitivity scenario?

A strong answer describes design-based sampling uncertainty separately from assumptions about unmeasured bias.
A weak answer presents arbitrary high and low cases as a 95% confidence interval.

### 10. When would you refuse to claim a causal effect?

A strong answer names unaddressed differential changes, missing comparable controls, compromised assignment, or measurement changes and proposes the strongest feasible next test.
A weak answer replaces evidence with executive confidence or a vendor's headline percentage.

### 11. Can an effect estimated in one pilot be scaled to the network?

A strong answer compares population, adoption, constraints, season, and economics, and separates the forecast from the observed result.
A weak answer multiplies a pilot percentage by annual enterprise spend without checking eligibility or overlap.

### 12. How do you handle a baseline data correction after results are visible?

A strong answer requires a documented defect, versioned correction, consistent application, approval, and disclosure of timing and impact.
A weak answer silently overwrites the baseline to improve the reported outcome.

## 15. Key terms

- **Adoption:** actual use of a capability; see [Step 5](#step-5--understand-the-assignment-and-selection-mechanism).
- **Assignment:** allocation to an intervention or comparator; see [Step 6](#step-6--select-the-strongest-feasible-design).
- **Attribution:** causal allocation of an outcome difference to an intervention; see [Section 3](#3-the-core-mental-model).
- **Avoidable cost:** cost that changes when activity changes; see [Step 11](#step-11--translate-the-effect-into-finance-owned-value).
- **Baseline:** versioned reference for a defined metric and period; see [Step 4](#step-4--freeze-the-measurement-baseline).
- **Cluster:** group assigned or exposed together; see [Step 6](#step-6--select-the-strongest-feasible-design).
- **Common support:** overlap between treated and comparison covariate patterns; see [Step 5](#step-5--understand-the-assignment-and-selection-mechanism).
- **Comparator:** feasible alternative to the intervention; see [Step 1](#step-1--name-the-decision-and-the-alternatives).
- **Confounder:** common cause of assignment and outcome; see [Step 5](#step-5--understand-the-assignment-and-selection-mechanism).
- **Confidence interval:** design-based uncertainty interval; see [Step 10](#step-10--quantify-uncertainty-and-robustness).
- **Contamination:** unintended deviation from the comparator; see [Step 9](#step-9--diagnose-interference-and-concurrent-changes).
- **Counterfactual:** outcome under the alternative that is not observed for the same unit and period; see [Section 3](#3-the-core-mental-model).
- **Difference in differences:** treatment change minus comparison change; see [Step 7](#step-7--specify-the-difference-in-differences-calculation).
- **Eligible exposure:** units that can contribute to the defined outcome; see [Step 8](#step-8--guard-against-selection-and-denominator-changes).
- **Estimand:** population, treatment contrast, outcome, and horizon of a causal question; see [Step 3](#step-3--define-the-estimand-before-inspecting-post-outcomes).
- **Forecast:** prediction of an outcome; see [Section 3](#32-four-different-numbers).
- **Guardrail:** secondary metric that detects unacceptable harm; see [Step 3](#step-3--define-the-estimand-before-inspecting-post-outcomes).
- **Intention to treat:** comparison by original assignment; see [Step 5](#step-5--understand-the-assignment-and-selection-mechanism).
- **Interference:** an assigned unit affects another unit's outcome; see [Step 9](#step-9--diagnose-interference-and-concurrent-changes).
- **Mediator:** variable on the treatment-to-outcome pathway; see [Section 4](#4-definitions-and-vocabulary).
- **Net benefit:** eligible benefit less specified incremental costs; see [Section 6](#64-convert-to-period-benefit).
- **Parallel trends:** assumed equal untreated changes on the stated scale; see [Step 7](#step-7--specify-the-difference-in-differences-calculation).
- **Potential outcome:** outcome under a defined intervention state; see [Section 3](#31-two-possible-outcomes-one-observed).
- **Randomization:** allocation by a known chance mechanism; see [Step 6](#step-6--select-the-strongest-feasible-design).
- **Regression to the mean:** tendency for extreme noisy observations to become less extreme on repeat measurement; see [Step 5](#step-5--understand-the-assignment-and-selection-mechanism).
- **Selection bias:** distortion caused by entry into groups or analysis; see [Step 5](#step-5--understand-the-assignment-and-selection-mechanism).
- **Sensitivity analysis:** recalculation under explicit alternative assumptions; see [Step 10](#step-10--quantify-uncertainty-and-robustness).
- **Spillover:** effect of one unit's treatment on another unit; see [Step 9](#step-9--diagnose-interference-and-concurrent-changes).
- **Target:** desired outcome, distinct from observation; see [Section 3](#32-four-different-numbers).
- **Transportability:** relevance of an estimated effect to a different population; see [Section 11](#11-assumptions-and-limits).

## 16. Sources

The following sources support methods and context; they do not validate the synthetic numerical example.
Publication and access details should be checked for the specific edition used in an organizational decision.

1. Hernán, Miguel A., and James M. Robins. *Causal Inference: What If*. Chapman & Hall/CRC, 2020. [Author-hosted book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/). Potential outcomes, observational causal inference, and study design.
2. Bertrand, Marianne, Esther Duflo, and Sendhil Mullainathan. “How Much Should We Trust Differences-in-Differences Estimates?” *Quarterly Journal of Economics* 119, no. 1 (2004): 249–275. [DOI: 10.1162/003355304772839588](https://doi.org/10.1162/003355304772839588). Serial correlation and inference.
3. Callaway, Brantly, and Pedro H. C. Sant’Anna. “Difference-in-Differences with Multiple Time Periods.” *Journal of Econometrics* 225, no. 2 (2021): 200–230. [DOI: 10.1016/j.jeconom.2020.12.001](https://doi.org/10.1016/j.jeconom.2020.12.001). Multiple periods and staggered treatment timing.
4. Goodman-Bacon, Andrew. “Difference-in-Differences with Variation in Treatment Timing.” *Journal of Econometrics* 225, no. 2 (2021): 254–277. [DOI: 10.1016/j.jeconom.2021.03.014](https://doi.org/10.1016/j.jeconom.2021.03.014). Interpretation of two-way fixed-effects comparisons.
5. Imbens, Guido W., and Jeffrey M. Wooldridge. “Recent Developments in the Econometrics of Program Evaluation.” *Journal of Economic Literature* 47, no. 1 (2009): 5–86. [DOI: 10.1257/jel.47.1.5](https://doi.org/10.1257/jel.47.1.5). Evaluation estimands and identification.
6. Kohavi, Ron, Randal M. Henne, and Dan Sommerfield. “Practical Guide to Controlled Experiments on the Web: Listen to Your Customers Not to the HiPPO.” *Proceedings of KDD* (2007). [DOI: 10.1145/1281192.1281295](https://doi.org/10.1145/1281192.1281295). Experiment design and practical interpretation.
7. Corsten, Daniel, and Thomas Gruen. “Desperately Seeking Shelf Availability: An Examination of the Extent, the Causes, and the Efforts to Address Retail Out-of-Stocks.” *International Journal of Retail & Distribution Management* 31, no. 12 (2003): 605–617. [DOI: 10.1108/09590550310507731](https://doi.org/10.1108/09590550310507731). Retail availability context; not evidence of a particular analytics tool's effect.
8. Wasserstein, Ronald L., and Nicole A. Lazar. “The ASA Statement on p-Values: Context, Process, and Purpose.” *The American Statistician* 70, no. 2 (2016): 129–133. [DOI: 10.1080/00031305.2016.1154108](https://doi.org/10.1080/00031305.2016.1154108). Interpretation of statistical evidence.
9. GS1. [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin) and [Global Location Number (GLN)](https://www.gs1.org/standards/id-keys/gln). Identifier definitions and applications.
10. GS1. [EPCIS and CBV standards](https://www.gs1.org/standards/epcis). Visibility event standards and specifications.
11. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Primary legal text; applicability depends on the specific processing.
12. UPS. [ORION route optimization](https://about.ups.com/us/en/our-stories/innovation-driven/orion.html). Company description of route optimization; company claims should be distinguished from independent causal evidence.

## 17. Links

- [Module 25 index](./_index.md).
- [Stakeholders and the value taxonomy](./01_stakeholders-value-taxonomy.md) — stakeholder and benefit ownership.
- [ROI, NPV, and payback](./03_roi-npv-payback.md) — convert period-level evidence into an investment decision.
- [Benefits realization](./09_benefits-realization.md) — monitor and reconcile benefits after deployment.
- [Module 25 survey](../../../course/25_product_economics_leadership.md) — module-level business case.
- [Simulation, optimization, and production labs](../../../course/30_simulation_optimization_production_labs.md) — use simulation and operational experiments as supporting practice.
- [Experiment design](../../02_methods/20_digital_twins_simulation_optimization_uncertainty/05_experiment-design.md) — experimental design in simulation and operational settings.
