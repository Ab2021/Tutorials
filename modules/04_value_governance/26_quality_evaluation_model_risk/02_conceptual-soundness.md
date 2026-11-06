# Conceptual Soundness

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 02 of 10  
> Estimated study time: 4 hours  
> Prerequisites: decision framing, basic probability and forecasting, and value measurement; Modules 06, 07, and 25 provide useful context.  
> Version 2026.5

## 1. Why this topic exists

A planning team can produce a statistically impressive model that makes a bad business decision.
The disconnect usually appears at the boundary between a modeled quantity and an operational action.
A forecast of shipments, for example, is not necessarily a forecast of unconstrained customer demand.
When stock is unavailable, shipments may fall even while demand rises.
A model trained on those shipments can learn that low availability means low demand.
An apparently accurate prediction may then reinforce a shortage.

Conceptual soundness is the reasoned assessment of whether a model's design is appropriate for its intended use.
It asks whether the proposed representation of the world is defensible before relying on performance claims.
It also asks whether the assumptions remain plausible where, when, and by whom the model will be used.
The question is not whether a method is fashionable, complex, or familiar.
The question is whether its output, data, and failure behavior support a specific decision at an acceptable risk.

Consider a buyer deciding how much material to order before a long supplier lead time.
The forecast must cover demand during that lead time, not merely tomorrow's sales.
The order quantity must reflect shortage and excess costs as well as feasible ordering rules.
A single average error metric may hide the expensive tail of the decision.
A model with lower average error can recommend too little stock when shortages matter more than leftovers.

An operations leader needs to know which decisions may use the model and when to stop.
A model developer needs a clear target, information cutoff, and comparison baseline.
A data owner needs to know which records represent true demand and which represent constrained fulfillment.
An independent challenger needs enough evidence to identify a plausible alternative explanation.
An accountable approver must decide whether the remaining uncertainty is acceptable for a bounded use.
None of these roles can substitute a dashboard score for the decision rationale.

The topic covers model design, not a complete model validation program.
Data representativeness, backtesting, monitoring, documentation, and independence require additional work.
The immediate deliverable is a decision-specific argument connecting evidence to permitted use.
The argument may conclude that the model is suitable, suitable only with restrictions, or not yet suitable.

### Prerequisite bridge

Module 06 is deferred in this edition; do not assume its expanded topics are available.
For this lesson, a **feature** is an input available to the model when the decision is made.
A **target** is the quantity the model is trained or evaluated to predict.
A **baseline** is a simple, reproducible comparator available under the same information constraints.
A **holdout** is data withheld from model fitting and method selection for an honest evaluation.
A **loss function** assigns a numerical penalty to an outcome or decision error.
Probability is a weight from zero to one over mutually exclusive outcomes that sum to one.
A conditional probability describes uncertainty given the information available at a stated time.

You also need the forecasting distinction between a prediction and a decision.
A forecast is an estimate of an uncertain future quantity.
A plan is a proposed action or coordinated set of actions.
A target is a desired performance level, not necessarily the expected outcome.
A commitment is an authorized promise to supply, buy, or deliver under stated conditions.
The numerical example uses all four terms separately.

## 2. Learning outcomes

After this lesson you can produce the following observable artifacts.

1. **Frame** a decision with its action, owner, timing, horizon, and fallback.
   Exercise 1 checks whether the decision can actually be executed.
2. **Distinguish** the modeled target from the business quantity and identify censoring.
   Exercise 2 checks a demand-versus-shipment distinction.
3. **Compute** expected asymmetric decision cost for competing actions.
   Exercise 3 checks the arithmetic and the chosen action.
4. **Critique** information availability and a leakage mechanism.
   Exercise 4 checks whether the proposed feature is available at the decision cutoff.
5. **Choose** a simple comparator and an evidence design matched to the use case.
   Exercise 5 checks the fairness of the comparison.
6. **Write** a testable assumption, limitation, trigger, and fallback.
   Exercise 6 checks whether a control can be operated.
7. **Design** an independent challenge and record a proportionate approval decision.
   Exercises 7 and 8 check the challenge and decision record.

The outcomes deliberately separate conceptual reasoning from proof of realized value.
A good design hypothesis must still survive representative, prospective evidence.
A strong retrospective score does not automatically establish causal impact.

## 3. The core mental model

```text
Business purpose and owner
        |
        v
Decision, action set, cutoff, horizon, constraints
        |
        v
Decision quantity and asymmetric consequences
        |
        +-----> Approved fallback and human authority
        |
        v
Target and inputs as known at the cutoff
        |
        v
Method + assumptions + proposed output
        |
        v
Independent challenge: alternative explanations,
leakage, edge cases, limitations, and comparator
        |
        v
Evidence against decision-relevant acceptance criteria
        |
        v
Approve with bounds / restrict / reject / gather evidence
        |
        v
Monitor conditions and outcomes; revalidate on change
```

Read the arrows as obligations, not as a guarantee of correctness.
The decision determines what information is needed and when it must exist.
The model is one component of a larger decision system including data, users, policies, and overrides.
A model may have a defensible mathematical formulation and still be unsound in that system.
For example, a correct forecast arriving after a purchase-order cutoff cannot inform that order.
A demand distribution that ignores substitution may be adequate for one stable item and inadequate for a promotion involving nearby alternatives.

### Four fit questions

**Fit to decision:** Does the output support the actual action at the relevant time and granularity?
A monthly regional forecast cannot directly determine a daily store order without a defensible allocation rule.
That rule is part of the decision system and needs its own evidence.

**Fit to data:** Do the historical records represent the quantity and population in scope?
Are the data available at the point of use rather than reconstructed with future information?
The right algorithm cannot repair an unobserved target without additional assumptions or information.

**Fit to consequences:** Does the evaluation reflect who bears shortages, excess, delay, safety, and service costs?
A symmetric forecast metric may be useful diagnostically while being insufficient for approval.
High-consequence failures may require hard constraints even when their average cost is small.

**Fit to use:** Can users understand the output's limits, intervene, and apply an alternative safely?
A valid output becomes unsafe if a user treats a probabilistic estimate as a guaranteed supply commitment.
The approval boundary must travel with the output.

### Argument, evidence, and residual uncertainty

A conceptual soundness argument has a claim, supporting evidence, a counterargument, and a disposition.
A claim might say that lead-time demand is adequately represented for a particular product family.
Supporting evidence might include timestamped data lineage and a documented stockout adjustment.
A counterargument might note that the adjustment cannot recover lost demand for a new product.
The disposition might exclude new products and require a separate planner review.

Evidence is not interchangeable.
A design document shows what developers intended.
A data extract shows what was recorded under a particular extraction rule.
A backtest shows performance on selected historical periods under a stated simulation.
A controlled operational comparison can estimate effects under its own design and assumptions.
None alone proves that future deployments will behave identically.

### Model, policy, and environment

Separate a model that predicts demand from a policy that converts that prediction into an order.
The forecast can be sound while the ordering policy ignores minimum order quantities.
The ordering policy can be reasonable while its lead-time input is stale.
The environment can change while both components remain unchanged.
Conceptual review therefore traces the whole chain but identifies which component owns each issue.

## 4. Definitions and vocabulary

| Term | Definition and relevant unit | Common misuse |
|---|---|---|
| Decision | A choice among feasible actions at a stated time. | Treating a forecast as the action itself. |
| Decision owner | Person accountable for the operational choice. | Assuming the model developer owns business risk. |
| Action | An executable choice, such as order quantity in units. | Treating an infeasible fractional quantity as executable. |
| Decision cutoff | Latest instant at which information can influence the action. | Using values published after the cutoff. |
| Horizon | Time span over which uncertainty affects the decision, in days or weeks. | Evaluating one day when replenishment takes four weeks. |
| Grain | Level of a record or decision, such as item-location-week. | Joining monthly and daily records without a mapping. |
| Population | Entities and periods eligible for the intended use. | Assuming historical active items represent launches. |
| Target | Quantity a model is trained or assessed to predict, with units. | Calling observed shipments unconstrained demand without evidence. |
| Label | Recorded target value used for supervised learning. | Assuming every label is error-free ground truth. |
| Feature | Information supplied to a model at inference. | Using an eventual outcome as an input. |
| Availability timestamp | Time a value became usable, including reporting delay. | Substituting transaction date for availability date. |
| Event timestamp | Time the underlying event occurred. | Assuming the event was immediately known. |
| Leakage | Use of information unavailable or improperly derived at the decision cutoff. | Limiting leakage checks to explicit future columns. |
| Censoring | Partial observation of a quantity because measurement or capacity is bounded. | Treating stock-limited sales as complete demand. |
| Selection bias | Distortion caused by how records enter the observed sample. | Assuming unrecorded customers behaved like recorded ones. |
| Distribution shift | Difference between development and deployment populations or mechanisms. | Calling every seasonal change unexpected drift. |
| Concept drift | Change in the relationship between inputs and outcomes. | Equating input drift with proof of model failure. |
| Baseline | Simple comparator under the same information and operational constraints. | Comparing against an obsolete or advantaged process selectively. |
| Challenger | Alternative method or explanation used to test a model claim. | Requiring the challenger to be more complex. |
| Backtest | Historical simulation that respects information available at each past cutoff. | Randomly splitting time series and calling it prospective evidence. |
| Validation | Evidence-based assessment of fitness and limitations for a use. | Equating validation with a single accuracy calculation. |
| Conceptual soundness | Defensibility of the model's design and assumptions for intended use. | Treating mathematical elegance as business suitability. |
| Assumption | A proposition needed for an inference or design to be useful. | Hiding a consequential premise as an implementation detail. |
| Limitation | A known boundary or weakness affecting interpretation or use. | Calling an untested population supported. |
| Restriction | An enforceable boundary on permitted model use. | Writing a caveat no user can observe or enforce. |
| Trigger | Observable condition that requires an action or review. | Setting a threshold without an owner or response. |
| Fallback | Predefined alternative when the model cannot be relied upon. | Assuming a manual process exists without capacity or training. |
| Underage cost | Incremental cost per unit of unmet demand under an action. | Including the same lost margin twice. |
| Overage cost | Incremental cost per unit of excess stock under an action. | Using purchase price as cost when inventory is fully recoverable. |
| Expected cost | Probability-weighted average cost over specified outcomes. | Describing a long-run average as a guaranteed single-period cost. |
| Calibration | Agreement between stated probabilities and observed frequencies over comparable cases. | Claiming a single forecast is calibrated. |
| Quantile | Value at or below which a stated fraction of a distribution lies. | Treating the 80th percentile as an 80% guarantee for each instance. |
| Aleatoric uncertainty | Variation in outcomes under a modeled process. | Assuming more training data removes all operational randomness. |
| Epistemic uncertainty | Uncertainty about the model, parameters, or missing knowledge. | Treating sparse evidence as a narrow, certain estimate. |
| Residual risk | Risk remaining after the stated controls. | Treating residual risk as zero after approval. |
| Independence | Sufficient separation of challenge from development and incentives, with authority to escalate. | Treating a different job title as proof of objectivity. |

## 5. The mechanics, step by step

The procedure below produces a traceable argument rather than an automatic certification.
Adjust the depth to impact, autonomy, uncertainty, and reversibility.
A safety-critical allocation or a high-value automated purchase needs stronger challenge than a low-impact advisory report.

### Step 1 — State the real decision

**Input:** A business request, workflow, owner, and proposed model output.
**Action:** Write a sentence naming the actor, decision, frequency, cutoff, action set, and objective.
**Output:** A decision statement that an operator can confirm or correct.

For replenishment, write: “At 09:00 each Monday, the buyer selects an integer order quantity for item A at site B to cover the next four weeks, subject to supplier and storage constraints.”
Avoid “predict demand accurately” as the entire statement.
Prediction accuracy is an intermediate property; purchasing is the decision.

Ask whether the actor has authority to execute the recommendation.
Ask whether the recommendation is advisory, automatically applied, or automatically applied unless overridden.
Document the latest moment a correction can be made and the cost of reversal.
If there are multiple decisions, identify each separately.
An allocation decision and a purchase decision can have different objectives and failure costs even when they use the same forecast.

**Branch:** If no owner can name an action, the proposed model may be exploratory rather than decision-ready.
Do not label exploratory insight as an approved automated decision.
If the decision can be deferred, specify the cost of waiting and who makes the deferral.
If no feasible action can change the outcome during the horizon, choose a different decision or horizon.

### Step 2 — Map the consequences of being wrong

**Input:** Decision statement, stakeholders, cost data, service obligations, and risk policy.
**Action:** List plausible outcomes for each type of error and identify affected parties.
**Output:** A consequence map and a decision-relevant objective or set of constraints.

Include financial costs such as lost contribution, expedite charges, disposal, and holding cost where applicable.
Include service consequences such as missed deliveries and the duration of unavailability.
Include safety, contractual, environmental, or equity impacts where relevant to the use.
Identify consequences that cannot be adequately represented by a simple currency value.
A hard safety or legal constraint should not be quietly traded away by a low expected-cost estimate.

Distinguish underprediction from overprediction only after mapping each to an action.
An underforecast does not always cause a shortage if an existing safety-stock policy already covers it.
Likewise, an overforecast does not automatically create excess if capacity constraints prevent ordering.
Trace the actual policy rather than assuming the forecast error directly equals the business loss.

Avoid double counting.
If lost contribution already includes variable purchase cost, do not add that cost again as lost revenue.
If leftover inventory can be sold later, use incremental holding and markdown cost rather than the entire acquisition price unless justified.
Specify the currency, time period, quantity basis, and the uncertainty in each cost estimate.

**Branch:** If reliable monetary costs are unavailable, use scenarios, ranges, or explicit service constraints.
Do not invent a precise cost merely to make a dashboard look quantitative.
Record which stakeholder accepts the residual uncertainty.

### Step 3 — Identify the decision quantity and target

**Input:** Decision statement, causal process, operational records, and outcome definitions.
**Action:** Trace the quantity needed for the decision back to what the data actually observe.
**Output:** A target specification with unit, grain, horizon, population, and observation limits.

Demand during supplier lead time can differ from sales, shipments, orders, and forecasts.
Sales may be constrained by available inventory.
Orders may be canceled, duplicated, backordered, or diverted to another product.
Shipments may reflect warehouse throughput or allocation policy rather than customer need.
Forecasts may contain human judgment or commercial targets rather than observed outcomes.

A target definition should identify returns and substitutions.
It should specify whether demand is gross, net of cancellations, or net of returns.
It should distinguish a customer request date from a fulfillment date.
It should define how a stockout is detected and how partial availability is represented.
A proxy target can be acceptable, but its assumptions and likely bias must be explicit.

For example, suppose observed shipments are the smaller of demand and available stock.
When stock equals zero, a shipment count of zero does not identify the number of customers who would have purchased.
Correcting that label requires external data, a model of lost demand, or a bounded assumption.
The reviewer should ask which evidence distinguishes those possibilities.

**Branch:** If the decision requires unconstrained demand but only censored sales exist, do not claim the unobserved quantity is directly measured.
Consider a restricted population with high availability, lost-sales signals, or a sensitivity interval.
Each option changes the population and may introduce its own bias.
If no adequate identification is possible, limit the claim and require human review or additional data.

### Step 4 — Define timing and information availability

**Input:** Source-system timestamps, publication delays, revisions, and operational cutoff.
**Action:** Create an as-of-time data contract for every feature and label.
**Output:** A feature availability table with a reproducible point-in-time join rule.

Record both event time and the time the value became available to the decision process.
The event can occur before a decision while its corrected record arrives afterward.
Historical warehouse extracts often contain final revised values, not the values known at the time.
Using final values in a backtest can make a model look much better than a deployable system.

For each feature, record source, grain, unit, update cadence, availability latency, revision policy, and missing-value behavior.
For each past simulated decision, use the latest version actually available by that cutoff.
Use publication time for external statistics and data-feed receipt time for operational feeds.
Specify timezone and daylight-saving handling when cutoffs are intraday.

A feature based on actual end-of-week sales is unavailable for a Monday order at the start of that week.
A planned promotion might be available, but its eventual realized discount is not.
A supplier's quoted lead time is available if received before cutoff; the actual lead time is not.
A future value estimated by another model can be used only as that model's as-of prediction, not as the later realized value.

**Branch:** If a historically revised series cannot be reconstructed, label the backtest as using an unavailable information set.
Obtain archived snapshots or redesign the feature.
Do not call the resulting score a faithful production simulation.

### Step 5 — Match horizon and grain

**Input:** Decision cadence, lead time, reporting grain, and aggregation rules.
**Action:** Translate the decision into the forecast horizon and aggregation level required by the action.
**Output:** A documented mapping from prediction to action.

A four-week replenishment decision needs a view of cumulative demand over the exposed period.
Four independent one-week intervals are not automatically a valid four-week interval.
Correlation between weeks changes the distribution of their sum.
If demand rises across all weeks during a promotion, independent simulations may understate the tail.

A store-level action cannot be justified solely by a national total without an allocation rule.
Aggregate forecasts can be more stable while hiding local stockouts.
A product hierarchy can change when items are substituted, bundled, or reclassified.
Document how the forecast is reconciled across locations and products and who owns exceptions.

Check calendar alignment, partial weeks, lead-time variability, and weekends.
If lead time is uncertain, a fixed-horizon forecast alone may miss variation in exposure.
If orders are placed in case packs, round or optimize in feasible case-pack increments.
Evaluate the resulting action, not an infeasible intermediate quantity.

**Branch:** If the required grain has insufficient data, aggregate, pool, or use a defensible fallback.
Explain how a higher-level estimate is allocated and where that allocation may fail.
Do not claim item-level precision merely because an output contains item-level decimal places.

### Step 6 — Select the model family and output

**Input:** Decision quantity, available data, operating constraints, and consequences.
**Action:** Compare candidate representations with the simplest credible alternative.
**Output:** A method rationale tied to the decision and explicit assumptions.

A point forecast can support a decision when costs and constraints make its implied action appropriate.
A distribution or selected quantile is often more useful when shortages and excess have different costs.
An optimization model may be needed when items compete for a constrained resource.
A simulation may represent lead-time uncertainty and nonlinear operational rules.
A simple rule may outperform a complex model when data are sparse or the process is stable.

Do not select complexity solely because a model can express more relationships.
Additional parameters can increase estimation error and make explanations or operations harder.
A black-box model can still be assessed through data, controls, behavior, and outcomes, but unavailable internal evidence limits the claim.
If vendor documentation cannot establish a necessary assumption, record that gap rather than inferring compliance.

Ask whether the output is a mean, median, quantile, class, score, or recommended action.
Ask whether downstream software interprets it the same way.
A forecast mean passed to software expecting a service-level quantile can systematically understock.
A score without calibration should not automatically be interpreted as a probability.

**Branch:** If a method cannot represent a decision-critical feature, either add a defensible component or restrict its use.
For example, a model that ignores an impending shutdown should not silently extrapolate business-as-usual demand through the shutdown.

### Step 7 — Build an assumption ledger

**Input:** Method documentation, data lineage, workflow, and interviews with developers and users.
**Action:** Translate implicit premises into falsifiable statements and identify consequences when they fail.
**Output:** An owned ledger linking assumption, evidence, test, trigger, and response.

Some assumptions concern the world: supplier behavior, product substitution, and demand stability.
Some concern measurement: inventory accuracy, event timestamps, and missing records.
Some concern the method: independent residuals, aggregation, or the form of a loss function.
Some concern operation: planner review, system latency, and fallback capacity.

Write “the order cutoff feed contains all accepted orders received before 08:45 local time” rather than “data are timely.”
Name how completeness is measured and what happens if the feed misses its service-level threshold.
Where a condition cannot be directly observed, define a proxy and explain its limitations.
Assign an owner who can act, not merely a team mailbox.

Do not confuse a convenient assumption with a known fact.
Do not write “no drift” when the intended use is a changing market.
Instead, specify the changes tested and the boundaries within which evidence is adequate.

**Branch:** If a critical assumption lacks evidence, determine whether it can be tested before use.
If not, either narrow the approved use, add a compensating control, or decline approval.
A control should be practical, observable, and assigned to someone with authority.

### Step 8 — Design fair alternatives and tests

**Input:** Candidate model, decision objective, available history, and feasible baseline.
**Action:** Compare against alternatives using the same information, cutoff, population, and action constraints.
**Output:** A preregistered or time-stamped evaluation plan with acceptance criteria.

A useful baseline might be the current approved planner process, last comparable period, seasonal naive forecast, or a simple reorder rule.
The right baseline is one the organization could actually use under the same constraints.
A baseline using perfect future knowledge is not a deployable competitor.
A baseline deprived of information given to the candidate is not a fair comparator.

For historical evaluation, use rolling decision origins and train only on information available at each origin.
Separate method selection from final evaluation to reduce optimistic selection bias.
When many candidates are tuned on the same holdout, it is no longer an untouched holdout.
Keep records of all material design choices and the dates they were made.

Measure forecast quality and decision outcomes separately.
For a replenishment policy, evaluate simulated shortage, excess, service, and cost under documented assumptions.
Include segment results for high-value items, intermittent demand, promotions, and disruptions where sample size supports interpretation.
Report counts and uncertainty; a small subgroup with no observed failures does not establish zero risk.

**Branch:** If historical policy influenced observations, simulate cautiously.
Observed sales under a stockout do not reveal all counterfactual sales under a different order.
An offline replay may require an explicit lost-demand assumption or additional experiment.
Do not claim measured incremental profit from an unidentifiable counterfactual.

### Step 9 — Conduct independent challenge

**Input:** Decision specification, data and model artifacts, development evidence, and known limitations.
**Action:** Assign a qualified reviewer with sufficient separation and authority to challenge the claim.
**Output:** A challenge record with findings, developer responses, unresolved issues, and escalation.

Independence is a matter of incentives, capability, access, and authority, not only reporting lines.
A reviewer who selected the method or whose compensation depends on approval has a conflict that must be managed.
In a small organization, an external reviewer or a qualified person outside the delivery chain can provide separation.
The approver remains accountable even when review work is outsourced.

Give the reviewer access to data definitions, as-of extracts, code or vendor explanations, assumptions, comparison results, and incident history as appropriate.
Record any inaccessible evidence and how that limits the conclusion.
The reviewer should be able to reproduce key calculations independently and ask for alternative specifications.
They should challenge the business framing as well as the implementation.

Ask: What would make this model appear accurate while being unsuitable?
Ask: Which population is absent and which outcome is unobservable?
Ask: What plausible alternative mechanism explains the apparent association?
Ask: How would the recommendation change if the largest assumption were wrong?
Ask: What should happen if the model cannot produce an output on time?

Classify findings by decision impact, not by the number of documents affected.
Record issue owner, severity, due date, interim control, and closure evidence.
If an issue threatens the validity of a high-impact use, escalate rather than silently accepting a remediation promise.
An unresolved issue may be accepted only by someone with explicit risk authority, subject to organizational policy.

### Step 10 — Set the permitted-use boundary

**Input:** Conceptual argument, empirical evidence, independent findings, and stakeholder risk appetite.
**Action:** Decide whether the evidence supports the exact intended use.
**Output:** An approval decision and an operationally enforceable scope.

The boundary should name product, site, decision type, time horizon, user, and automation level.
It should state excluded populations, known failure regimes, and required human review.
It should distinguish approved use from an experiment or shadow mode.
A shadow model can generate comparisons without changing decisions; it does not prove the effect of acting on its outputs.

Possible dispositions are approval, approval with conditions, restricted or pilot use, deferral, and rejection.
These are organizational choices, not universal legal categories.
For a conditional decision, name the condition, owner, deadline, and what happens if the deadline is missed.
Avoid a permanent “temporary” exception with no renewal or expiry.

A model card or equivalent record should identify the version and environment covered.
If downstream users can access the output without the restriction, the restriction may not be effective.
Apply controls in workflow, permissions, user interface, or service integration where practical.

### Step 11 — Specify monitoring and fallback

**Input:** Assumptions, boundaries, measurable leading indicators, and response capabilities.
**Action:** Define what to monitor, who responds, and what operational action follows.
**Output:** A runbook and revalidation triggers linked to the risks.

Monitor input freshness, completeness, schema, distribution, output behavior, and decision outcomes as relevant.
A threshold is useful only if someone can investigate and act on it.
For example, late inventory snapshots might suspend automatic ordering and route the case to a buyer.
The runbook should state how to detect late data, how to notify the buyer, and how to resume.

Monitor actual lead time against the horizon assumed by the model.
Monitor the frequency and direction of overrides, but do not interpret every override as an error.
A planner may possess information missing from the model; repeated overrides can reveal a missing input or a process failure.
Record the reason at the time rather than reconstructing it later.

Define fallback capacity, not merely a fallback name.
A manual process may fail if thousands of recommendations require attention at once.
A prior rule may be unsafe after a supplier change.
Exercise the fallback with users before relying on it for a material use.

**Branch:** On material change in target, data source, method, population, workflow, or cost structure, assess whether the approval still applies.
A change does not automatically imply failure, but it invalidates unexamined assumptions of sameness.
Use a defined change process and document whether targeted testing or full revalidation is appropriate.

### Step 12 — Preserve the argument and decision

**Input:** All outputs above, independent review, and approver disposition.
**Action:** Assemble a versioned, reproducible record.
**Output:** A concise decision dossier with references to detailed evidence.

Include the decision statement, target specification, information cutoff, assumptions, baseline, and cost model.
Include test design, segment counts, limitations, issue log, and disposition.
Identify model version, code or vendor version, data snapshot, parameters, and output interface.
State who developed, challenged, approved, and operates the model.

Record disagreement without rewriting history after an adverse outcome.
A dissenting reviewer should be able to identify unresolved concerns in the approval record.
Capture expiry or review dates, triggers, and the fallback.
Protect sensitive data while preserving enough lineage for authorized reproduction.

The dossier is not evidence merely because it is long.
Each claim should point to an artifact or a clearly identified judgment.
An approver should be able to trace a material decision from claim to evidence to limitation.

## 6. Worked example

### Situation and scope

A distributor orders a replacement component for one regional warehouse every four weeks.
The supplier requires the order at 09:00 on Monday and delivers before the covered four-week selling period.
Assume delivery is certain and on time for this illustrative calculation.
The buyer must choose 80, 100, 110, or 120 units.
All four quantities are feasible under the stated capacity and supplier constraints.
There is no starting inventory, no replenishment during the period, and no substitution.
The example is synthetic; it is not evidence of a real distributor's performance.

The decision is the order quantity in units for one four-week period.
The decision owner is the buyer; the proposed model is advisory.
The forecast is a probability distribution over demand, not the order quantity.
The plan is the buyer's selected order after considering that distribution and constraints.
The target is a desired service level, if management specifies one; none is imposed in this calculation.
The commitment is the accepted purchase order, not the forecast or an unapproved recommendation.

### Demand forecast

Assume the model produces the following three mutually exclusive demand scenarios.
The probabilities are assumed for the example, not estimated from an actual dataset.

| Demand in four weeks | Probability | Probability-weighted demand |
|---:|---:|---:|
| 80 units | 0.20 | 16 units |
| 100 units | 0.50 | 50 units |
| 120 units | 0.30 | 36 units |
| **Total** | **1.00** | **102 units** |

The expected demand is 16 + 50 + 36 = 102 units per four-week period.
The median demand is 100 units because cumulative probability reaches 0.70 at 100 units.
Neither number by itself determines the best feasible order when shortage and excess costs differ.

### Incremental costs

Assume an unmet unit costs USD 8 in lost contribution and service recovery.
Assume an excess unit costs USD 2 in holding and markdown over the decision horizon.
Both amounts are incremental costs per unit for this four-week decision.
Assume no salvage or later sales beyond what is already reflected in the USD 2 excess cost.
Assume these costs do not change with the number of units.
These are simplified decision inputs, not universal estimates.

For order quantity `q` and demand `d`, define:

```text
shortage units = max(d - q, 0)
excess units   = max(q - d, 0)
cost in USD    = 8 × shortage units + 2 × excess units
```

If q = 100 units and d = 80 units, excess is 20 units and cost is USD 40.
If q = 100 units and d = 100 units, both shortage and excess are zero.
If q = 100 units and d = 120 units, shortage is 20 units and cost is USD 160.
The expected cost is 0.20 × 40 + 0.50 × 0 + 0.30 × 160 = USD 56 per decision.

### Compare feasible actions

| Order | Cost if demand 80 | Cost if demand 100 | Cost if demand 120 | Expected cost |
|---:|---:|---:|---:|---:|
| 80 units | USD 0 | USD 160 | USD 320 | USD 176 |
| 100 units | USD 40 | USD 0 | USD 160 | USD 56 |
| 110 units | USD 60 | USD 20 | USD 80 | USD 46 |
| 120 units | USD 80 | USD 40 | USD 0 | USD 36 |

For 80 units: 0.20 × 0 + 0.50 × 160 + 0.30 × 320 = USD 176.
For 110 units: 0.20 × 60 + 0.50 × 20 + 0.30 × 80 = USD 46.
For 120 units: 0.20 × 80 + 0.50 × 40 + 0.30 × 0 = USD 36.
Under these assumptions, 120 units is the lowest-cost action among the four feasible choices.
It has USD 20 lower expected cost than ordering 100 units.
That USD 20 is a model-based expected difference per decision, not realized savings or a guaranteed outcome.

The expected shortage at 100 units is 0.30 × 20 = 6 units per decision.
The expected excess at 100 units is 0.20 × 20 = 4 units per decision.
Their costs are USD 48 and USD 8, respectively, totaling USD 56.
The expected shortage at 120 units is zero in these three scenarios.
The expected excess at 120 units is 0.20 × 40 + 0.50 × 20 = 18 units.
At USD 2 per unit, the expected excess cost is USD 36.
Zero shortage here follows from the assumed upper scenario; it is not a real-world guarantee.

### Why the median can mislead

The median minimizes expected absolute error when all unit errors have equal cost.
For an order of 100 units, expected absolute deviation is 0.20 × 20 + 0.30 × 20 = 10 units.
For an order of 120 units, expected absolute deviation is 0.20 × 40 + 0.50 × 20 = 18 units.
The 100-unit action therefore looks better under symmetric absolute error.
But the assumed shortage cost is four times the excess cost, so symmetric absolute error is not the decision objective.

Under the simple single-period newsvendor assumptions, the critical fractile is underage cost divided by underage plus overage cost.
Here it is 8 / (8 + 2) = 0.80, or 80%.
The smallest demand value whose cumulative probability is at least 0.80 is 120 units.
The discrete result agrees with the explicit feasible-action comparison.
The formula does not automatically handle fixed order charges, multiple periods, shared capacity, or uncertain costs.

### Sensitivity to a cost assumption

Now hold the demand distribution and shortage cost fixed, but raise excess cost to USD 5 per unit.
For 100 units, expected cost becomes 0.20 × 100 + 0.30 × 160 = USD 68.
For 110 units, expected cost becomes 0.20 × 150 + 0.50 × 50 + 0.30 × 80 = USD 79.
For 120 units, expected cost becomes 0.20 × 200 + 0.50 × 100 = USD 90.
For 80 units, expected cost remains USD 176 because this action has no excess in the assumed scenarios.
The preferred action changes to 100 units among the four feasible actions.
The critical fractile becomes 8 / 13, approximately 0.615; the corresponding discrete quantile is 100 units.
The change illustrates decision sensitivity, not proof that either cost estimate is correct.

### Challenge of the data and method

Suppose the forecast was fitted to shipments during periods when inventory sometimes ran out.
An independent reviewer asks whether 80 shipped units means only 80 units were demanded.
If stockout sales were censored, the upper demand tail may be too small.
The reviewer asks for inventory availability, backorder, lost-sales, and substitution evidence where available.
If those records are incomplete, the demand probabilities need a sensitivity range or a restricted-use decision.

The reviewer also asks whether the three probabilities were fixed before seeing the evaluation periods.
If they were chosen after inspecting outcomes, the apparent performance may be optimistic.
The reviewer requests rolling-origin results and a simple historical baseline using the same as-of data.
The reviewer checks that four-week cumulative demand, not a one-week forecast, is evaluated.
The reviewer checks whether actual lead times and purchase cutoffs match the example's simplifications.

The model developer states that the distribution excludes new products and promotional periods.
The reviewer asks how those exclusions will be enforced in the purchasing interface.
A written limitation alone will not prevent a user from applying the recommendation to an excluded item.
The buyer confirms a manual ordering procedure exists for those cases.
That procedure still requires capacity and effectiveness evidence before production reliance.

### Illustrative disposition

On the fictional facts given, the arithmetic supports an advisory order of 120 units for this scenario.
It does not establish that the demand probabilities are calibrated or that the model is production-ready.
A responsible disposition is to require evidence on censored demand, availability timestamps, and out-of-sample decision cost before broader approval.
If deployed as a limited pilot, the scope, reviewer, fallback, trigger, and expiry must be recorded.
No actual approval or validated cost saving is asserted by this example.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Copy it into a file and run it with `python filename.py`.
It reproduces the expected-cost and sensitivity calculations above.
It compares only the four stated feasible actions and three stated scenarios.
It does not estimate probabilities, validate the model, or simulate real operational outcomes.

```python
from decimal import Decimal

# Each demand is units per four-week decision; probabilities are dimensionless.
SCENARIOS = (
    (80, Decimal("0.20")),
    (100, Decimal("0.50")),
    (120, Decimal("0.30")),
)
ACTIONS = (80, 100, 110, 120)  # Feasible order quantities, in units.


def expected_cost(order, shortage_cost, excess_cost):
    """Return expected incremental cost in USD per decision."""
    total = Decimal("0")
    for demand, probability in SCENARIOS:
        shortage = max(demand - order, 0)
        excess = max(order - demand, 0)
        scenario_cost = shortage_cost * shortage + excess_cost * excess
        total += probability * scenario_cost
    return total


def expected_units(order):
    """Return expected shortage and excess, in units per decision."""
    shortage = Decimal("0")
    excess = Decimal("0")
    for demand, probability in SCENARIOS:
        shortage += probability * max(demand - order, 0)
        excess += probability * max(order - demand, 0)
    return shortage, excess


def report(shortage_cost, excess_cost):
    print(f"Shortage USD {shortage_cost}/unit; excess USD {excess_cost}/unit")
    costs = {}
    for order in ACTIONS:
        costs[order] = expected_cost(order, shortage_cost, excess_cost)
        shortage, excess = expected_units(order)
        print(
            f"order={order:3d} units  "
            f"expected_shortage={shortage:.2f} units  "
            f"expected_excess={excess:.2f} units  "
            f"expected_cost=USD {costs[order]:.2f}"
        )
    best = min(costs, key=costs.get)
    print(f"lowest-cost feasible order={best} units\n")


if __name__ == "__main__":
    if sum(probability for _, probability in SCENARIOS) != Decimal("1"):
        raise ValueError("Scenario probabilities must sum to one")
    report(Decimal("8"), Decimal("2"))
    report(Decimal("8"), Decimal("5"))
```

Expected output:

```text
Shortage USD 8/unit; excess USD 2/unit
order= 80 units  expected_shortage=22.00 units  expected_excess=0.00 units  expected_cost=USD 176.00
order=100 units  expected_shortage=6.00 units  expected_excess=4.00 units  expected_cost=USD 56.00
order=110 units  expected_shortage=3.00 units  expected_excess=11.00 units  expected_cost=USD 46.00
order=120 units  expected_shortage=0.00 units  expected_excess=18.00 units  expected_cost=USD 36.00
lowest-cost feasible order=120 units

Shortage USD 8/unit; excess USD 5/unit
order= 80 units  expected_shortage=22.00 units  expected_excess=0.00 units  expected_cost=USD 176.00
order=100 units  expected_shortage=6.00 units  expected_excess=4.00 units  expected_cost=USD 68.00
order=110 units  expected_shortage=3.00 units  expected_excess=11.00 units  expected_cost=USD 79.00
order=120 units  expected_shortage=0.00 units  expected_excess=18.00 units  expected_cost=USD 90.00
lowest-cost feasible order=100 units
```

The 80-unit action has expected shortage 0.50 × 20 + 0.30 × 40 = 22 units.
Its expected cost is USD 176 at either excess cost.
The program calculates these quantities directly from the scenarios.

### Interpreting the artifact

The program is an arithmetic check on the assumed distribution and costs.
It does not tell you whether the demand probabilities reflect stockouts, promotions, or future supplier conditions.
It uses exact decimal arithmetic for the stated decimal probabilities and unit costs.
It breaks a tie by selecting the first action in `ACTIONS`; a production policy should explicitly specify tie handling.
The code checks that probabilities sum to one; its fixed scenarios are nonnegative and mutually exclusive by construction.
A production input pipeline would also need appropriate validation and lineage, but this example does not implement one.

The comparison assumes every action has the same fixed ordering and transport costs.
If those costs differ, include the incremental difference in the scenario cost.
If backorders have different consequences from lost sales, model them separately.
If a supply constraint prevents 120 units, remove that action before comparing expected costs.
If a minimum service requirement is binding, filter out actions that fail it before selecting the lowest-cost remaining action.

Expected cost is a long-run weighted average under the assumed distribution.
A single decision can experience any scenario-specific cost in the table.
A rare but severe outcome outside the three scenarios is absent from the calculation.
An approver should therefore examine tail coverage and any non-negotiable constraints separately.

## 8. Data, standards, and regulatory context

### Data requirements and ownership

A soundness review needs an inventory of sources and a definition of each field's business meaning.
For replenishment this can include sales, orders, cancellations, stock on hand, stockouts, purchase orders, receipts, lead times, promotions, and item-location master data.
The need for a source depends on the decision; collecting all fields does not itself make the model sound.
A data owner should identify how records are created, corrected, archived, and made available.

Record item and location identifiers and their validity dates.
A product identifier reused for a reformulation can join incompatible histories.
A store closure or warehouse move can change the operational population without changing a dashboard label.
A unit-of-measure conversion error can turn cases into individual units or vice versa.
A correct algorithm fitted to inconsistent units can produce systematically incorrect recommendations.

Use point-in-time master data for historical reconstruction.
A current hierarchy may assign past transactions to categories that did not exist at the time.
Backfilled promotion classifications may expose future knowledge.
Maintain a distinction between what happened, what was recorded, and what was knowable at the decision time.

### Missingness and target observation

Missing values have different mechanisms.
A missing sales record might mean no sale, a closed store, a failed feed, or an unavailable item.
Replacing all four with zero creates a false uniform interpretation.
Document the operational meaning of zero separately from null.

Check data coverage by product, location, season, channel, and availability state.
A representative average across all rows can hide a complete absence of high-impact products.
Report denominators and excluded records alongside quality rates.
When demand is censored, accuracy against shipments may reward a model that predicts the stock constraint rather than demand.

Data protection and contractual restrictions may limit access to customer or supplier information.
A model review should use the minimum information necessary and follow applicable organizational controls.
Privacy compliance is not established by omitting a name if records remain linkable to a person.
Consult the relevant legal and privacy teams for the applicable jurisdiction and processing purpose.

### Supervisory guidance: applicability matters

The US Federal Reserve and Office of the Comptroller of the Currency published SR 11-7/OCC 2011-12 model risk management guidance in 2011.
For Federal Reserve supervision, [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm) superseded SR 11-7 in April 2026; consult the relevant agency's current scope.
The guidance discusses conceptual soundness, ongoing monitoring, outcomes analysis, and effective challenge for relevant banking organizations.
Its applicability depends on the institution, supervisory context, and current regulatory position.
A supply-chain company does not become a regulated bank merely because it uses a forecast.
Use the guidance as a risk-management reference where appropriate, not as a blanket legal requirement for all supply chains.
Verify current supervisory requirements and any superseding guidance with qualified counsel or compliance staff.

The guidance's effective-challenge idea is broader than a checklist.
It calls for critical analysis by objective, informed parties who can identify limitations and produce appropriate change.
The organizational implementation should be proportionate to model risk and governance requirements.
A vendor model still needs institution-specific understanding of its intended use and limitations where applicable.

### NIST and ISO frameworks

The US National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0, published in 2023, is a voluntary framework for managing risks of artificial intelligence (AI) systems.
Its Govern, Map, Measure, and Manage functions provide a useful lifecycle structure.
Using its terminology does not automatically demonstrate compliance with any law or prove a model is safe.
NIST's AI RMF Playbook gives suggested actions, not a universal pass/fail test.

ISO/IEC 23894:2023 provides guidance on AI risk management.
ISO/IEC 42001:2023 specifies requirements for an AI management system.
Whether a standard is contractually or legally required depends on the applicable circumstances.
An ISO management-system certification, if obtained, does not certify that every individual model is accurate or suitable for every decision.
Consult the actual standards and competent assessors for their requirements.

### EU AI Act and sector rules

The European Union AI Act is Regulation (EU) 2024/1689.
Its obligations depend on system classification, role, jurisdiction, and applicable dates and transitional provisions.
Not every supply-chain forecast is a high-risk AI system merely because it is used in a supply chain.
A particular use may trigger other obligations or sector-specific rules; determine applicability case by case.
As of this lesson's version, consult the current consolidated legal text and qualified counsel rather than assuming all provisions apply identically or simultaneously.
This educational lesson is not legal advice.

### Standards are not evidence of fit

A standard can structure governance, terminology, or management-system controls.
It does not supply the correct shortage cost, uncover missing demand, or prove a specific forecast is timely.
A documented validation can satisfy an internal process while still overlooking a faulty target.
Connect every external framework control to the actual decision and evidence.
Record the framework edition, applicability assessment, owner, and date of review.

## 9. Industry practice and case evidence

### What a public case can and cannot establish

Public cases illustrate failure mechanisms; they rarely expose every model artifact and decision counterfactual.
A news report or inquiry does not by itself establish that a particular forecasting algorithm was conceptually unsound.
Separate a documented operational outcome from an inference about how model governance might have helped.
Do not use a company name as proof that a specific control would have prevented its losses.

### The M5 forecasting competition

The M5 forecasting competition used retail unit sales data supplied by Walmart and made available through the competition organizers.
Its published work describes hierarchical retail forecasting and evaluation across products and aggregation levels.
The data and competition demonstrate that business forecasting can involve large hierarchies, intermittent sales, promotions, and asymmetric importance across series.
The competition's evaluation setup is not identical to any individual company's purchasing decision or loss function.
A strong competition score is therefore evidence of performance under that competition's rules, not proof of fitness for a specific replenishment workflow.

The M5 paper by Makridakis, Spiliotis, and Assimakopoulos describes the competition and its results; see the source list.
A practitioner should inspect the evaluation period, hierarchy, data availability, and metric before transferring a method.
**Inference:** A decision-specific review would additionally ask whether the model's outputs and timing align with the buyer's order policy and shortage cost.
The competition does not establish the result of that review for a particular warehouse.

### FDA medical-device change control as an analogy

The US Food and Drug Administration (FDA) has issued guidance on predetermined change control plans for artificial intelligence-enabled device software functions.
The guidance addresses a regulated medical-device context, not ordinary inventory forecasting.
It illustrates why intended use, planned changes, and evidence about effects of modifications matter over a system lifecycle.
It should not be presented as imposing medical-device requirements on a warehouse forecast.
**Inference:** A supply-chain organization can borrow the discipline of documenting which model changes require renewed review, without claiming regulatory equivalence.

### Forecast value and inventory decisions

Classical inventory theory connects decision quantities to demand uncertainty and underage/overage consequences.
The single-period newsvendor model illustrates why the cost ratio determines the relevant demand quantile under specific assumptions.
That theory does not establish actual cost inputs or prove that a product has a single-period lifecycle.
A practitioner must check replenishment timing, salvage, constraints, and the distribution before applying the formula.

Forecast accuracy and inventory value need not rank models identically.
The worked example demonstrates this mathematically under explicitly assumed costs and probabilities.
Real operations may involve multi-period dynamics and interactions between items.
The correct evaluation may require a policy simulation or operational experiment rather than a point-forecast contest.

### Operational evidence hierarchy

A conceptual rationale is necessary but is not a substitute for data evidence.
A historical replay can challenge the rationale if it preserves the information set and operational rules.
A prospective shadow comparison can test data latency and output behavior without changing the live decision.
A controlled pilot can provide stronger evidence about operational effects if allocation, contamination, and measurement are designed appropriately.
An observational before-and-after comparison can be informative but may be confounded by simultaneous changes.

For each case, ask which populations and periods were actually observed.
Ask whether the comparison had access to equal information.
Ask whether actions, costs, and outcomes were measured or modeled.
Ask whether the cost of running the comparison and its effect on operators were included.
Do not generalize a result beyond its evidence boundary without additional justification.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong target | Shipments fall when stock is zero. | Censored sales are treated as demand. | Shortages reinforce low forecasts. | Flag stockouts, investigate lost demand, and restrict unsupported use. |
| Wrong horizon | One-day accuracy is reported for a four-week order. | Evaluation misses cumulative exposure. | Order quantity ignores lead-time demand. | Evaluate the full decision horizon and lead-time variability. |
| Future leakage | A feature uses revised or realized values. | Historical inputs contain later information. | Backtest is optimistically biased. | Reconstruct point-in-time snapshots and publication latency. |
| Grain mismatch | National accuracy is good but local service deteriorates. | Aggregation hides site-level errors. | Local stockouts and excess coexist. | Test the action grain and the allocation rule. |
| Symmetric objective | Only mean absolute error is shown. | Underage and overage have different consequences. | A low-error policy has high decision cost. | Evaluate policy cost and binding service constraints. |
| Incorrect cost | Purchase price is counted as both lost margin and excess cost. | Cost categories overlap or ignore salvage. | Wrong action is selected. | Reconcile incremental costs with finance and sensitivity ranges. |
| Missing population | New products are absent from training and testing. | Historical coverage excludes launches. | Unsupported recommendations appear precise. | Exclude launches or validate a dedicated approach. |
| Unavailable output | Recommendations arrive after order cutoff. | Pipeline latency is ignored in design. | Users act without current information. | Monitor availability and define a timely fallback. |
| Unchecked assumption | Lead time is assumed fixed despite frequent delays. | Horizon and tail exposure are understated. | Stockouts increase during disruption. | Measure lead times and test variable-lead-time scenarios. |
| Misread probability | A user treats the mean as a service quantile. | Output semantics are not communicated. | Systematic under-ordering or over-ordering. | Label output, document policy mapping, and train users. |
| Baseline disadvantage | Candidate uses promotions but comparator does not. | Information sets differ. | Claimed improvement is misleading. | Apply identical as-of information and constraints. |
| Selection on holdout | Many variants are chosen using final test results. | Test data become part of tuning. | Reported performance is optimistic. | Preserve a later untouched evaluation or prospective test. |
| Counterfactual error | Offline savings assume observed sales equal unconstrained demand. | Alternative-policy outcomes are unobserved. | Benefits are overstated. | State identification assumptions and use bounds or experiment. |
| Unenforced restriction | Interface permits excluded products. | Caveat is not connected to workflow. | Unsupported use becomes routine. | Enforce eligibility rules and log exceptions. |
| Weak challenge | Validator reports to the release owner and cannot escalate. | Incentives and authority impair objectivity. | Material issues are accepted silently. | Provide independent expertise, access, escalation, and documented dissent. |
| Unusable fallback | Manual review is required for thousands of alerts. | Staffing and timing are not tested. | Controls fail at scale. | Capacity-test fallback and prioritize by impact. |
| Silent change | Vendor updates the model without notice. | Approved version differs from deployed version. | Prior evidence no longer maps to use. | Contract for change notification and version-aware review. |
| Hidden automation | A user interface auto-accepts recommendations by default. | Actual autonomy exceeds stated use. | Errors propagate without review. | Verify workflow, permissions, override logs, and accountability. |
| Misleading precision | Sparse segments receive narrow intervals. | Parameter uncertainty or heterogeneity is ignored. | Users over-trust uncertain recommendations. | Report coverage, uncertainty, and segment limits. |
| Monitor without action | Drift alerts have no owner or threshold response. | Detection is disconnected from control. | Known degradation persists. | Assign runbook, decision authority, and escalation. |

An issue can have several causes.
A late data feed can create both missingness and leakage if the backtest substitutes a later corrected value.
A control should address the causal mechanism, not merely suppress an alert.
The issue log should distinguish observed evidence, plausible mechanism, consequence, and unresolved uncertainty.

## 11. Assumptions and limits

The following ledger is a starting point, not a claim that these assumptions hold in every supply chain.

| Assumption | Why it is needed | What breaks it | How to test or respond |
|---|---|---|---|
| The decision and owner are stable. | Evidence is tied to a specific action and authority. | Workflow or automation changes. | Reconfirm decision mapping at each material change. |
| Forecast horizon covers exposure. | Demand during replenishment drives the order. | Lead time exceeds the assumed period. | Compare actual lead times with modeled horizon and stress delays. |
| Demand labels are informative. | The model needs evidence about the target. | Stockouts censor sales. | Compare availability, backorders, and lost-sales signals; use bounds. |
| Features are known at cutoff. | Production cannot use future information. | Revised or delayed data enter a replay. | Reconstruct as-of snapshots and audit timestamps. |
| Units and identifiers are consistent. | Data must represent comparable entities and quantities. | Pack changes or item code reuse. | Reconcile master data and conversions over time. |
| Historical cases represent use. | Performance evidence must transfer to the eligible population. | Launches, closures, or channel shifts. | Report segment coverage and restrict unsupported populations. |
| Scenario probabilities are credible. | Expected cost depends on their weights. | Structural change or sparse tail observations. | Assess calibration where feasible and vary probabilities. |
| Cost inputs are incremental. | The objective must not double-count consequences. | Overlapping finance definitions or omitted recovery. | Reconcile categories and run cost sensitivity. |
| Actions are feasible. | An optimal recommendation must be executable. | Capacity, pack size, or supplier minimum changes. | Validate constraints at decision time and evaluate rounded actions. |
| A single-period approximation is adequate. | The worked formula ignores future decisions. | Replenishment, returns, or carryover matter. | Compare against a multi-period policy model or restrict interpretation. |
| Outcomes are sufficiently independent for the calculation used. | Some interval and sample-size formulas rely on dependence assumptions. | Common promotions, correlated sites, or serial dependence. | Examine dependence and use a suitable joint or block-based analysis. |
| Planner review is effective. | Advisory use relies on human intervention. | Alert fatigue, insufficient time, or misunderstood output. | Observe workflow, measure workload, and inspect override reasons. |
| Fallback can operate at scale. | A failure response must be executable. | Simultaneous feed outages overwhelm staff. | Exercise the runbook at realistic volume. |
| Monitoring detects relevant change. | Controls rely on actionable signals. | Unobserved demand changes before labels arrive. | Combine leading indicators with delayed outcome review; acknowledge blind spots. |
| Independence is substantive. | Effective challenge requires objectivity and authority. | Reviewer shares delivery incentives or lacks access. | Record conflicts, access gaps, dissent, and escalation path. |

A model can be conditionally sound even when not every assumption is literally true.
The relevant question is whether deviations are material to the intended decision and whether the evidence bounds their effect.
Conversely, a model can satisfy its mathematical assumptions and still be inappropriate for the business objective.
The ledger should evolve as evidence, data sources, and operating conditions change.

### Limits of the worked example

The three-point distribution is hypothetical and deliberately small enough for hand calculation.
It omits uncertain lead time, multiple products, dynamic pricing, storage interactions, and future replenishment.
Its costs are constant per unit and do not include fixed ordering costs.
Its action set is constrained by assumption rather than derived from supplier records.
The result should not be copied as a policy for an actual operation without evidence for those premises.

Expected cost is not the only possible criterion.
An organization might require a minimum service probability or place a hard limit on a safety consequence.
A model can minimize expected cost while producing unacceptable harm in a low-probability scenario.
Those constraints must be defined by accountable stakeholders and assessed alongside cost.

## 12. Exercises

Each exercise refers to the definitions and example in this lesson.
Use explicit units, document assumptions, and distinguish evidence from inference.

### Exercise 1 — Frame the decision

A warehouse manager receives a monthly dashboard predicting regional shipments.
A buyer places item-location orders each Monday for a supplier with a variable three-to-five-week lead time.
The buyer can choose order quantities in multiples of ten units.

**Deliverable:** Write a decision statement and identify the owner, action, cutoff, horizon, and two constraints.
**Acceptance criteria:** Your statement distinguishes forecast from order; it addresses variable lead time and feasible case-pack quantities.
**Time:** 15 minutes.

### Exercise 2 — Diagnose the target

An item has 70 units available and ships 70 units before going out of stock.
Customer demand is not otherwise recorded; three customers later request a substitute.
The developer labels the period as 70 units of demand.

**Deliverable:** Explain what is observed, what is not identified, and propose two additional evidence sources.
**Acceptance criteria:** You do not claim the true demand is exactly 73 units; you explain why substitution signals may be incomplete or duplicated.
**Time:** 15 minutes.

### Exercise 3 — Calculate decision cost

Use the three demand scenarios and USD 8 shortage/USD 2 excess costs from Section 6.
Compare orders of 100 and 110 units.

**Deliverable:** Show shortage and excess for each scenario, expected costs, and the lower-cost action.
**Acceptance criteria:** Your totals are USD 56 and USD 46 per decision, and you state the model assumptions behind the comparison.
**Time:** 20 minutes.

### Exercise 4 — Find leakage

A Monday 09:00 model uses the final promotion discount for that week.
The final discount is confirmed on Wednesday; a provisional plan is available on Sunday.
The retrospective dataset contains only the final discount.

**Deliverable:** Describe the leakage and propose an as-of feature and evaluation approach.
**Acceptance criteria:** You distinguish planned from realized discount and explain the limitation if historical plans were not archived.
**Time:** 15 minutes.

### Exercise 5 — Design a fair comparison

A new model uses archived promotion plans and supplier quotes.
The current baseline is a seasonal naive forecast that uses neither, although planners can see both before the cutoff.
The developer reports that the new model has lower error and attributes the entire difference to its architecture.

**Deliverable:** Design a fair comparison and name two metrics or decision outcomes.
**Acceptance criteria:** Your design aligns information timing, action constraints, and evaluation periods; it does not attribute an information advantage to model architecture.
**Time:** 25 minutes.

### Exercise 6 — Write an operable assumption and control

A model assumes inventory snapshots arrive before every 09:00 order cutoff.
In practice, the feed sometimes arrives late.

**Deliverable:** Write an assumption, measurable trigger, response, owner, and evidence of fallback capacity.
**Acceptance criteria:** The trigger is observable before the cutoff; the response names who acts and what they do when the feed is late.
**Time:** 20 minutes.

### Exercise 7 — Plan independent challenge

The developer also chairs the release committee and receives a bonus for on-time rollout.
A vendor will provide output samples but not source code or training data.
The model recommends high-value automated orders.

**Deliverable:** Design the review roles, access request, three challenge questions, and escalation route.
**Acceptance criteria:** The plan addresses conflicts, restricted vendor evidence, potential material harm, and authority to defer release.
**Time:** 25 minutes.

### Exercise 8 — Decide and record a bounded use

A model passes an as-of retrospective comparison for mature items at three sites.
It has no evidence for new products or promotional periods.
The workflow can block new products, but cannot yet identify every promotion reliably.
A manual fallback exists for 50 decisions per day; the peak excluded volume is unknown.

**Deliverable:** Write an approval disposition and a short list of conditions, owners, and triggers.
**Acceptance criteria:** Your disposition does not claim untested populations are approved; it addresses unreliable promotion detection and fallback capacity before broader use.
**Time:** 30 minutes.

## 13. Self-check answers

### Exercise 1

A strong statement is: “Before 09:00 each Monday, the buyer chooses an item-location order in multiples of ten units, accounting for demand over the supplier's three-to-five-week lead time and available capacity.”
The buyer owns the purchase decision, while the dashboard is an information source.
A monthly regional shipment forecast does not directly answer the item-location variable-horizon demand question.
A strong answer identifies at least two real constraints, such as order multiples and storage or supplier capacity.
If the exercise does not provide a cutoff, a proposed time must be labeled as an assumption.
The statement does not imply that the forecast is a purchase commitment.

### Exercise 2

Seventy units shipped and inventory exhaustion are observed.
Unconstrained demand is not identified from those records alone.
The three substitute requests show possible unmet demand but might duplicate earlier requests or represent customers who would not otherwise have purchased the item.
Useful evidence can include timestamped customer requests, backorders, lost-sale logs, website availability views, and inventory records.
Each source has its own coverage and bias; no source should be assumed complete without testing.
A reviewer could examine periods with high availability, while acknowledging that those periods may differ from stockout periods.

### Exercise 3

At 100 units, the 80-unit demand scenario creates 20 excess units and USD 40 cost.
The 100-unit scenario costs USD 0; the 120-unit scenario creates 20 shortage units and USD 160 cost.
Expected cost is 0.20 × 40 + 0.50 × 0 + 0.30 × 160 = USD 56.
At 110 units, costs are USD 60, USD 20, and USD 80 for the three scenarios.
Expected cost is 0.20 × 60 + 0.50 × 20 + 0.30 × 80 = USD 46.
The 110-unit action has USD 10 lower expected cost than 100 units under the stated scenarios.
This comparison does not establish that 110 is globally optimal; 120 units has lower expected cost in Section 6.

### Exercise 4

The final discount was not available at Monday 09:00 and leaks information from Wednesday.
A legitimate candidate feature is the promotion plan actually available on Sunday, as received before the cutoff.
A historical evaluation needs archived snapshots or another reliable reconstruction of that plan.
If only final discounts exist, a backtest using them does not reproduce the production information set.
The team may omit the feature, obtain archives, or report the analysis as an optimistic non-deployable upper-information comparison.
A planned discount can still be wrong; that uncertainty belongs in evaluation.

### Exercise 5 rubric

A strong design gives both approaches the same as-of data or explicitly separates the effect of new information from the effect of architecture.
It evaluates the same decision origins, items, locations, constraints, and lead-time horizon.
It includes an operationally credible current-process comparator, with the same planner information where reconstructable.
Suitable outcomes include decision cost, shortage units, service, excess units, bias, and a forecast accuracy measure.
It preserves a final evaluation period or prospective comparison after method selection.
It reports segment counts and acknowledges counterfactual limits when stockouts censor demand.
A weak answer compares an information-rich candidate with an intentionally blind baseline and calls the difference architectural value.

### Exercise 6 rubric

A strong assumption is: “A complete, valid inventory snapshot for each eligible item-location is available by 08:45 local time on each ordering day.”
A trigger could be an absent or incomplete snapshot at 08:45, with the completeness definition and clock source specified.
The runbook could suspend automated recommendations and alert the named buyer or operations role before 09:00.
The response must specify the approved fallback, the maximum volume it can handle, and escalation when capacity is exceeded.
Evidence should include a realistic fallback exercise or observed operating history, not merely a policy statement.
The threshold and timing are illustrative and must be agreed with the real operation.

### Exercise 7 rubric

A strong answer separates the reviewer and approver from the developer's release incentive and gives them access to the committee or risk authority.
It requests vendor documentation of intended use, data lineage, output meaning, version changes, known limitations, and available independent test evidence.
It challenges timing, representativeness, failure cost, and sensitivity to unavailable internal evidence.
Black-box testing can assess behavior but cannot establish every internal property.
The release decision should explicitly state what cannot be reviewed and whether compensating controls are sufficient.
For high-value automated orders, an unresolved material evidence gap can justify deferral or a bounded advisory pilot.

### Exercise 8 rubric

A defensible disposition is defer broader approval or allow a tightly bounded pilot only where eligibility and manual capacity are demonstrably controlled.
New products remain excluded and the existing technical block should be tested operationally.
Because promotion detection is unreliable, the team must establish another reliable exclusion or restrict the population and period to one where promotion exposure is known.
The operations owner must measure peak excluded volume and demonstrate fallback capacity before depending on it.
The model owner should document the three sites, version, decision horizon, and eligible mature items.
Triggers include expansion to new sites, a promotion-data change, late inputs, and fallback overload.
The approver and independent reviewer should record residual risks and a review or expiry date.

## 14. Interview questions

1. **What makes a model conceptually sound?**
   A strong answer connects method, decision, data, assumptions, consequences, and evidence.
   A weak answer equates soundness with mathematical sophistication or one accuracy score.
2. **Why is a shipment forecast not always a demand forecast?**
   A strong answer explains stockout censoring, fulfillment constraints, and unobserved lost demand.
   A weak answer assumes every missing sale is a zero-demand observation.
3. **How would you choose a forecast horizon for replenishment?**
   A strong answer starts with decision timing and exposure over variable lead time.
   A weak answer chooses the shortest convenient prediction interval.
4. **What is point-in-time leakage?**
   A strong answer distinguishes event time from availability and revision time.
   A weak answer says leakage only occurs when a column is explicitly named “future.”
5. **When is mean absolute error insufficient?**
   A strong answer discusses asymmetric losses, action constraints, and decision-level outcomes.
   A weak answer asserts that the lowest forecast error always maximizes value.
6. **How do you compare a complex model with a baseline?**
   A strong answer aligns information, cutoff, population, horizon, and operational constraints.
   A weak answer compares a new information-rich system with an outdated blind rule.
7. **What is effective independent challenge?**
   A strong answer includes competence, objectivity, access, authority, dissent, and escalation.
   A weak answer treats a separate title or a signed checklist as sufficient.
8. **What can you conclude from a vendor's accuracy claim?**
   A strong answer asks about population, target, cutoff, baseline, metric, and reproducibility.
   A weak answer transfers the claim to the buyer's decision without evidence.
9. **How do you handle an untestable critical assumption?**
   A strong answer considers sensitivity, restrictions, controls, additional evidence, and possible rejection.
   A weak answer declares the assumption true because the method requires it.
10. **What does approval with conditions require?**
    A strong answer names scope, owner, deadline, interim control, escalation, and expiry.
    A weak answer treats an open issue as harmless because it is documented.
11. **How can an accurate model cause harm?**
    A strong answer traces data, output semantics, policy, human workflow, and failure costs.
    A weak answer assumes any improvement in aggregate accuracy guarantees better decisions.
12. **When should conceptual review be revisited?**
    A strong answer mentions changes in use, data, model, environment, costs, and observed incidents.
    A weak answer relies only on a fixed annual calendar regardless of material change.

## 15. Key terms

- [Action](#4-definitions-and-vocabulary): an executable choice made by a decision owner.
- [Assumption](#5-the-mechanics-step-by-step): a proposition needed by the design or interpretation.
- [Availability timestamp](#5-the-mechanics-step-by-step): when a value could actually be used.
- [Baseline](#5-the-mechanics-step-by-step): a feasible comparator under a fair information set.
- [Calibration](#4-definitions-and-vocabulary): agreement between predicted probabilities and frequencies over comparable cases.
- [Censoring](#6-worked-example): partial observation caused here by limited availability.
- [Challenger](#5-the-mechanics-step-by-step): an alternative method or explanation used to test a claim.
- [Concept drift](#4-definitions-and-vocabulary): change in the input-outcome relationship.
- [Conceptual soundness](#3-the-core-mental-model): defensibility of the design for the intended decision.
- [Counterfactual](#9-industry-practice-and-case-evidence): an outcome under an action not actually taken.
- [Critical fractile](#6-worked-example): underage cost divided by total underage and overage cost in the simple model.
- [Decision cutoff](#5-the-mechanics-step-by-step): the last time information can affect an action.
- [Decision horizon](#5-the-mechanics-step-by-step): the period over which uncertainty affects the choice.
- [Expected cost](#6-worked-example): probability-weighted cost under stated scenarios.
- [Fallback](#5-the-mechanics-step-by-step): a prepared alternative when model use is suspended.
- [Feature](#1-why-this-topic-exists): a model input available at the required time.
- [Forecast](#1-why-this-topic-exists): an estimate of an uncertain future quantity.
- [Grain](#5-the-mechanics-step-by-step): the entity and time level of a record or decision.
- [Holdout](#1-why-this-topic-exists): data withheld from fitting and method selection.
- [Independent challenge](#5-the-mechanics-step-by-step): informed, objective scrutiny with escalation authority.
- [Leakage](#10-failure-modes-and-red-flags): improper access to information unavailable at the cutoff.
- [Limitation](#11-assumptions-and-limits): a known boundary on inference or use.
- [Overage cost](#6-worked-example): incremental cost of one excess unit.
- [Plan](#1-why-this-topic-exists): a proposed action, distinct from a forecast.
- [Population](#4-definitions-and-vocabulary): entities and periods within the intended-use scope.
- [Quantile](#6-worked-example): a threshold associated with a cumulative probability.
- [Residual risk](#11-assumptions-and-limits): risk remaining after the stated controls.
- [Restriction](#5-the-mechanics-step-by-step): an enforceable boundary on use.
- [Target](#5-the-mechanics-step-by-step): the quantity a model predicts or evaluates.
- [Trigger](#5-the-mechanics-step-by-step): an observable condition requiring a response.
- [Underage cost](#6-worked-example): incremental cost of one unmet unit.
- [Validation](#3-the-core-mental-model): evidence-based assessment of fitness for a specific use.

## 16. Sources

1. Board of Governors of the Federal Reserve System, **SR 11-7: Guidance on Model Risk Management**, 2011. [Official guidance and attachment](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm). Applicability and current supervisory status require checking.
2. Office of the Comptroller of the Currency, **OCC 2011-12: Supervisory Guidance on Model Risk Management**, 2011. [Official bulletin](https://www.occ.gov/news-issuances/bulletins/2011/bulletin-2011-12.html).
3. National Institute of Standards and Technology, **AI Risk Management Framework (AI RMF 1.0)**, 2023. [NIST AI 100-1](https://doi.org/10.6028/NIST.AI.100-1).
4. National Institute of Standards and Technology, **AI RMF Playbook**. [Official resource](https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook). The playbook is voluntary guidance.
5. International Organization for Standardization, **ISO/IEC 23894:2023, Information technology — Artificial intelligence — Guidance on risk management**. [ISO standard page](https://www.iso.org/standard/77304.html).
6. International Organization for Standardization, **ISO/IEC 42001:2023, Information technology — Artificial intelligence — Management system**. [ISO standard page](https://www.iso.org/standard/81230.html).
7. European Parliament and Council, **Regulation (EU) 2024/1689**, Artificial Intelligence Act. [EUR-Lex official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj). Consult current consolidated text and applicable dates.
8. Makridakis, S., Spiliotis, E., and Assimakopoulos, V., **The M5 accuracy competition: Results, findings, and conclusions**, *International Journal of Forecasting*, 2022. [DOI: 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013).
9. Gneiting, T., **Making and Evaluating Point Forecasts**, *Journal of the American Statistical Association*, 2011. [DOI: 10.1198/jasa.2011.r10138](https://doi.org/10.1198/jasa.2011.r10138). Explains how point forecasts depend on the chosen loss.
10. U.S. Food and Drug Administration, **Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions**. [FDA guidance page](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence). Medical-device context only.
11. Silver, E. A., Pyke, D. F., and Thomas, D. J., **Inventory and Production Management in Supply Chains**, fourth edition, CRC Press, 2016. Inventory decisions and their assumptions.

Sources establish frameworks and published methods; they do not validate the synthetic case or create automatic legal obligations.

## 17. Links

- [Module 26 index](./_index.md): topic order, prerequisites, and scope.
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md): lifecycle, inventory, and validation overview.
- [Model inventory and tiering](01_model-inventory-tiering.md): inventory scope and risk-based control burden; see the index for availability.
- [Data quality and representativeness](03_data-representativeness.md): population coverage and data limitations; see the index for availability.
- [Outcomes analysis and backtesting](04_outcomes-backtesting.md): time-respecting comparisons and challengers; see the index for availability.
- [Independence and validation](08_validation-independence.md): reviewer roles and conflicts; see the index for availability.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): apply decision and constraint reasoning in a lab context.
