# The Transfer Test

> Module 28 — Domain Case Library and Sector Playbooks  
> Topic 03 of 10  
> Estimated study time: 3–4 hours  
> Prerequisites: Module 28 topics 01–02; basic forecasting and value measurement  
> Version 2026.5

## 1. Why this topic exists

A published case can answer a useful question without answering **your** question.
A retailer's successful replenishment pilot may inspire a distributor, but it does not establish that the distributor should buy the same software, use the same policy, or expect the same savings.
The distributor may place orders less often, sell items with a shorter shelf life, or have less reliable inventory records.
These differences affect both the possible value and the harm from a wrong decision.

A transfer test turns a story into a decision about reuse.
It asks whether the decision, economics, data environment, operating constraints, and evidence are sufficiently similar for the specific action proposed.
It also records what is unknown and who will resolve it.
The output is not a universal similarity score.
It is a conditional recommendation: use as a plan, use as an analogy while testing, or do not use as support for the proposed action.

The audience includes an operations sponsor choosing a pilot, an analyst preparing a business case, a risk owner authorizing spend, and a frontline owner who must implement the recommendation.
A strong memo lets each person trace a proposed action to its source and to local evidence.
It also makes visible the differences that may reverse the value calculation.

**Plan** in this lesson means an action proposal supported by local facts, a feasible implementation design, a measured baseline, and authorization gates.
It does not mean a guarantee, a budget approval, or a claim that the source outcome will repeat.
An **analogy** is a useful comparison that suggests a mechanism or question but is not itself sufficient authorization to copy an intervention.
A case can be a plan for a small diagnostic while remaining only an analogy for a full rollout.

For example, a published case about route optimization may justify examining delivery density and driver adoption.
It does not justify promising the source company's savings per route unless the economics, constraints, and evidence have been independently checked for the destination operation.
The decision is made at the level of a particular proposed action, not the company name or the broad method.

The survey's rule is deliberately conservative: if **three answers are no**, the case is an analogy, not a plan.
It is not a voting rule that makes two serious failures acceptable.
One hard safety, legal, capacity, or evidence failure can block a plan even when four other answers are yes.
Unknown is not yes; unresolved material unknowns require a bounded investigation or test before the affected commitment.

## 2. Learning outcomes

After working through this topic, you can:

1. **Frame** a destination decision, owner, alternatives, time horizon, and proposed commitment before comparing cases.
   Exercise 1 checks this skill.
2. **Apply** the five exact transfer questions and classify each answer as yes, no, or unknown with a reason.
   Exercises 2 and 3 check classification and the three-no rule.
3. **Calculate** comparable unit economics, including costs that are absent from the source case.
   Exercise 4 checks the arithmetic and units.
4. **Evaluate** data and operational differences that can change an intervention's feasibility or effect.
   Exercise 5 asks you to design local checks.
5. **Judge** whether evidence is strong enough for a specified risk and distinguish a claim from an attributed effect.
   Exercise 6 checks evidence and counterfactual reasoning.
6. **Design** a bounded local evaluation with decision thresholds, owners, stop rules, and uncertainty.
   Exercise 7 asks for an evaluation design.
7. **Write** a transfer memo that separates the source, inference, local forecast, plan, target, and commitment.
   Exercise 8 supplies an approval scenario.

These outcomes are intentionally decision-specific.
A correct comparison is not a promise of the same result in a different population or period.

## 3. The core mental model

```text
 SOURCE CASE                                  DESTINATION DECISION
 Claim + evidence + limits                    Owner + alternatives + stakes
             \                                     /
              +-------- five transfer questions ---+
                       1  Same decision?
                       2  Similar product economics?
                       3  Similar data environment?
                       4  Similar operating constraint?
                       5  Evidence sufficient for risk?
                                  |
                 facts / differences / unknowns / hard gates
                                  |
              +-------------------+--------------------+
              |                   |                    |
       Eligible for plan    Analogy: learn/test    Do not use as support
       local gates passed   material gap remains   unsafe or irrelevant
              |                   |                    |
       controlled action     bounded evidence work    choose another basis
              +-------------------+--------------------+
                                  |
                    measure, update, and re-decide
```

The left side is not a standardized product to be copied.
It contains a claim made by someone about an intervention in a particular setting.
Trace the claim to its source, population, dates, metric, and comparison.
A case may document an outcome without documenting whether the intervention caused it.

The right side is a decision in a different setting.
Specify the action, alternative, population, time bucket, risk exposure, and owner before rating similarity.
Changing the action changes the test: a low-cost data audit and a network-wide purchase have different evidence requirements.

The five questions are dependent but should not be collapsed.
For example, a different data environment may make the same method impossible to run, while different economics may make it unprofitable even if it runs correctly.
A source with strong internal causal evidence can still be a poor fit for the destination.
Conversely, an apparently similar operation does not turn a vendor claim into proof.

Record each question as **yes**, **no**, or **unknown** for a stated action.
Use yes only when relevant support is documented.
Use no when a relevant difference is established.
Use unknown when the information needed to decide is missing or unreliable.
A no can sometimes be mitigated by redesign; record both the original gap and the evidence that the redesigned action resolves it.
Do not silently convert unknown to yes because an assumption is convenient.

Three or more no answers trigger the survey's analogy rule.
Fewer than three no answers do not establish readiness: assess material unknowns and hard constraints separately.
Plan readiness is a positive finding that local evidence, design, controls, and approvals fit the intended commitment.
A low-stakes learning action may be approved with unresolved questions if it has a defined scope and does not cross a hard gate.

## 4. Definitions and vocabulary

| Term | Working definition | Common misuse or unit |
|---|---|---|
| Source case | Documented account of an intervention and its setting | Treating the source organization as a representative sample. |
| Destination | Operation considering reuse | Treating all sites within a firm as identical. |
| Transfer | Applying a claim, mechanism, or intervention in the destination | Assuming that a result automatically generalizes. |
| Decision | Choice among specified alternatives at a specified time | Naming a technology rather than a choice. |
| Decision owner | Person accountable for the choice and its consequences | Confusing model developer with operational authority. |
| Analogy | Comparison that generates a hypothesis or diagnostic | Treating a resemblance as evidence of realized benefit. |
| Plan | Locally supported, feasible action proposal with gates | Treating the source case as the whole plan. |
| Forecast | Estimate of what is likely under explicit assumptions | Confusing an estimate with an approved target. |
| Target | Desired performance level | Calling a desired value a measured outcome. |
| Commitment | Authorized obligation to act or spend | Calling a model output an order. |
| Baseline | Measured current or comparison condition | Mixing a pre-period with a concurrent control without explanation. |
| Counterfactual | Outcome that would have occurred without the intervention | Assuming the observed pre-period is always the counterfactual. |
| Intervention | Specific change in policy, process, tool, or behavior | Attributing a bundle's effect to one component without evidence. |
| Outcome | Measured result for a defined population and period | Reporting an unverified expectation as an outcome. |
| Product economics | Revenue, variable costs, inventory costs, and loss structure | Comparing revenue per unit while ignoring margin and waste. |
| Contribution margin | Revenue less specified variable costs | Using gross margin without identifying included costs; currency per unit. |
| Avoidable cost | Cost that changes under a decision | Counting a fixed expense as cash savings without a release mechanism. |
| Stockout | Inability to satisfy demand from available inventory | Equating all missed demand with permanently lost sales. |
| Spoilage | Units becoming unsaleable through expiry or damage | Treating selling price as the only loss component. |
| Data grain | Entity and time resolution represented by a record | Combining daily sales with weekly decisions without aggregation rules. |
| Availability | Whether required information arrives by decision time | Equating an accessible historical extract with a usable live feed. |
| Latency | Elapsed time between an event and data availability | Reporting timestamp precision instead of delivery delay; hours or days. |
| Data lineage | Documented route from source record to analytic input | Assuming identical labels imply identical definitions. |
| Constraint | Limit that restricts feasible actions | Treating a hard legal or physical limit as a preference. |
| Hard gate | Requirement that must pass before the proposed action | Averaging a failure away through other favorable scores. |
| Local validation | Test of relevant assumptions in the destination | Calling a one-time demonstration a causal evaluation. |
| Evidence grade | Course shorthand for source type and typical strength | Treating the letter as an automatic guarantee of validity. |
| External validity | Applicability of an effect to another setting | Confusing it with whether the original estimate is internally credible. |
| Internal validity | Credibility of the causal comparison within the study | Assuming randomization alone proves local applicability. |
| Selection bias | Distortion arising from how cases or observations are chosen | Generalizing from a marketed success alone. |
| Heterogeneity | Variation in effects across populations or conditions | Applying an average effect to every site. |
| Incremental value | Difference in value between a proposal and its alternative | Adding gross benefit without subtracting incremental costs. |
| Sensitivity analysis | Recalculation under alternative assumptions | Presenting a single favorable scenario as uncertainty analysis. |
| Stop rule | Predefined condition for pausing or terminating a test | Inventing an adverse-event threshold after observing results. |

## 5. The mechanics, step by step

### Step 1 — State the decision, not the technology

**Input:** a source case and a request such as “do what they did.”
**Output:** a one-sentence destination decision and a bounded action.

Name the owner, alternatives, decision date, location, product or lane population, time horizon, and unit of consequence.
For example: “The replenishment manager must choose by 1 May whether to test a daily order recommendation for 80 chilled items at four stores for eight weeks, against the existing weekly review.”
That statement identifies an experiment, not an automatic enterprise-wide deployment.

Write the status quo as a real alternative, including any pending process change.
An alternative might be improving data quality first, changing order frequency without a model, or continuing the current policy.
If the owner can approve only an investigation, do not describe a rollout as already approved.
Separate reversible decisions from commitments with exit costs, safety consequences, or contractual obligations.

The scope determines how much evidence is required.
A desk review may need only a plausible mechanism and permission to access data.
A decision that places customers, employees, regulated goods, or material capital at risk requires substantially more local evidence and controls.
The same source can support the former but not the latter.

### Step 2 — Reconstruct the source claim

**Input:** source documents and a source-access date.
**Output:** a claim ledger with provenance and explicit gaps.

Record the organization, geography, period, population, intervention, outcome metric, baseline or comparison, and source type.
Quote or paraphrase only what the document actually supports.
Record whether the reported figure is a forecast, a target, a company claim, an observed result, or an estimated causal effect.
Do not attribute a company's total improvement to one component of a multi-part program unless the evidence supports that attribution.

Identify other changes during the period: seasonality, demand trend, promotions, network redesign, workforce levels, and measurement changes.
Ask whether there was a control group, random assignment, pre-specified analysis, or independent audit.
These features affect confidence in the source effect; none by itself proves that the destination will see the same effect.

Use the course source ladder as a starting point: A peer-reviewed or audited, B regulatory filing or official statistic, C company case study or investor claim, D vendor marketing, and E press summary or anecdote.
Grade individual claims, not the entire organization.
A regulatory filing can reliably document a disclosed loss while not establishing a model's contribution to that loss.
A peer-reviewed study can be poorly matched to the destination or have an imprecise effect estimate.

If source documents omit a critical value, write “not reported.”
Do not fill it with an industry average and then call the result source evidence.
An assumption may be used for scenario analysis if it is labelled and stress-tested.

### Step 3 — Ask question 1: Is the decision the same?

**Input:** source decision, destination decision, objectives, decision cadence, and decision rights.
**Output:** yes, no, or unknown with the material decision difference.

Compare the actual decision variable: order quantity, route sequence, allocation, price, inventory location, or capacity reservation.
Compare when the choice is made and when it becomes irreversible.
Check whether the objective is cost, margin, service, safety, waste, or a constrained combination.
Compare alternatives and who can override the recommendation.

A source that forecasts demand at a monthly national level may inform a demand-planning process.
It is not automatically a plan for hourly store-level replenishment.
Similarly, a source using route sequencing after orders are accepted differs from a destination deciding whether to accept orders at all.
A change in objective can reverse the preferred policy even with identical demand forecasts.

Mark **yes** if the consequential decision, timing, and objective are aligned or a documented redesign bridges the difference.
Mark **no** if the proposed action depends on treating a materially different decision as equivalent.
Mark **unknown** if the case does not report enough detail.
Capture the exact gap, not just “different sector.”

Branch: if the source cannot support the proposed decision, ask whether it can still support a narrower question.
A forecasting case may provide a feature hypothesis while leaving the reorder rule and economic decision to local design.
Do not relabel the broader decision as the narrower one just to pass the test.

### Step 4 — Ask question 2: Is the product economics similar?

**Input:** comparable unit economics, demand characteristics, and cost consequences.
**Output:** a local incremental-value model and a fit rating.

Compare net revenue and contribution per fulfilled unit, purchase or production cost, holding cost, salvage value, waste, and lost-demand consequences.
For a service, compare value and cost per order, stop, kilogram, pallet, or hour as appropriate.
Use the same currency, period, population, and accounting treatment across alternatives.
Translate percentages into absolute units before comparing cases of different scale.

Look for asymmetry.
An extra unit of a durable spare part may tie up working capital but remain saleable later.
An extra unit of short-life food may expire, incur disposal cost, and lose nearly all acquisition cost.
A stockout might lose one contribution-margin opportunity, trigger an expensive expedited shipment, or cause a contractual penalty.
These cases can have different optimal actions even if their demand forecast error is identical.

Distinguish a financial transfer from a behavioral transfer.
The idea that better decisions can reduce waste may transfer while the source's dollar savings do not.
Build destination scenarios for volume, response, benefit per unit, implementation cost, and recurring operating cost.
Include adoption and any capacity needed to realize the benefit.

Mark **yes** when material drivers are close enough for the proposed decision or when local calculations demonstrate that differences do not change the decision.
Mark **no** when known economics reverse or invalidate the proposed action.
Mark **unknown** if a decisive driver such as salvage value or incremental labor is missing.
The analyst should not use a source percentage as a local dollar value without a defensible denominator and cost model.

### Step 5 — Ask question 3: Is the data environment similar?

**Input:** data inventory, schemas, sample extracts, access rules, and decision-time delivery requirements.
**Output:** a data-fit assessment and a remediation or no-go list.

For every required feature and outcome, identify source system, owner, grain, history, latency, completeness, definition, and known error modes.
Check whether the information was actually available at the time of the historical decision.
A corrected month-end stock ledger is not necessarily usable for an 08:00 daily ordering decision.
Data leakage can make a historical backtest look substantially better than the deployed process.

Compare demand observations with true demand.
Sales during a stockout are censored: observed sales may be lower than the demand that would have arrived with inventory available.
If the source case used a stockout adjustment and the destination cannot estimate availability, its reported forecast comparison may not transfer.
Promotions, returns, substitutions, and product introductions also change the meaning of sales history.

Inspect identifiers and joins.
If store-item codes change, a seemingly complete time series may splice different products.
If delivery receipts arrive after ordering, a model may act on stale on-hand stock.
If data access is restricted, possession of historical data does not establish authorization for a live feed.

Mark **yes** only for the features, grain, freshness, quality, and permissions needed by the proposed action.
Mark **no** when a known non-remediated gap prevents the proposed action.
Mark **unknown** when the extract or authorization has not been checked.
A data-quality assessment may be a reasonable first action even where deployment is not.

### Step 6 — Ask question 4: Is the operating constraint similar?

**Input:** workflow map, capacity, physical limits, service rules, and approvals.
**Output:** feasible action space, hard gates, and owners.

Walk the recommendation from generation to execution.
Who reviews it, when does the order or route lock, and what happens on an exception?
Can the supplier deliver at the recommended frequency?
Can the warehouse pick the proposed quantities and can the receiving team handle them?
A technically valid recommendation has no operational value if the decision cannot be acted on.

Check minimum order quantities, case packs, supplier lead-time variation, vehicle capacity, labor availability, cold-chain requirements, shelf life, and contractual cutoffs.
Identify binding service or safety constraints separately from soft preferences.
A method may be transferable after redesigning its optimization constraints; the source configuration itself is not thereby transferable.

Interview the people who execute the work.
Compare the planned cadence with actual shift schedules, weekend coverage, exception queues, and local override authority.
Document workload and escalation paths.
If an intervention requires drivers, planners, or suppliers to behave differently, measure adoption rather than assuming it.

Mark **yes** when local feasibility has been established at the proposed scope.
Mark **no** when a known constraint rules out the proposed action.
Mark **unknown** when the operating owner has not verified feasibility.
A hard gate blocks the associated action regardless of how many other answers are yes.

### Step 7 — Ask question 5: Is the evidence strong enough to justify the risk?

**Input:** claim ledger, proposed commitment, plausible downside, and available local evidence.
**Output:** evidence-sufficiency judgment tied to a decision.

Separate three questions: did the source outcome occur, did the intervention cause it, and would the effect occur here?
An official filing can establish that a company reported a loss without isolating a causal mechanism.
A controlled study may support a causal effect in its own setting without settling transport to another market.
A vendor testimonial can suggest an operational hypothesis but is weak support for an irreversible investment.

Assess the size and uncertainty of the effect, quality of comparison, selection of sites, missing outcomes, conflicts of interest, and relevance to the destination.
Ask what would have happened without the intervention.
If the source provides only before-and-after results, list plausible alternative explanations rather than treating the change as causal.
If only successful pilots are published, recognize that the evidence base may be selected.

Risk is action-specific.
A small, reversible observation-only diagnostic can be justified by weaker evidence than an automated ordering policy with material waste exposure.
At high stakes, require reliable local validation, explicit limits, accountable owners, and any required legal or safety approvals.
Even strong evidence does not waive an independently binding requirement.

Mark **yes** when the available evidence is sufficient for this action and risk, not when it is perfect.
Mark **no** when evidence is known to be inadequate for the commitment.
Mark **unknown** when the nature of the evidence or downside has not been established.
A yes for a pilot does not become a yes for rollout without a new assessment.

### Step 8 — Synthesize without averaging away risk

**Input:** five reasoned ratings, missing facts, materiality, and hard gates.
**Output:** a disposition and next decision.

Count explicit no answers.
If there are three or more, label the case an analogy rather than a plan under the module rule.
This count is a screening condition, not proof of suitability when the count is smaller.
In particular, one disqualifying safety or legal gap can prevent a plan.

Treat each material unknown as a pending issue with an owner, evidence request, deadline, and consequence if unresolved.
Do not count unknown as no for the literal three-no rule, but do not promote the case to a plan merely because the no count is below three.
When a key input is not available before a commitment, either narrow the action to a safe evidence-gathering step or do not use the case to justify it.

Use **plan eligible** only when the source is relevant, the destination proposal is feasible, local evidence is sufficient for the commitment, all hard gates pass, and the responsible owner has a decision-ready plan.
Eligibility is not approval: governance may still reject the plan.
Use **analogy** when the mechanism is worth studying but material transfer gaps remain, or when three or more answers are no.
Use **do not use as support** when the source is irrelevant, misleading for the proposal, or incompatible with a non-negotiable condition.

Keep the distinction between the source case and the locally supported proposal.
A locally designed evaluation can be plan eligible even if the external deployment remains an analogy.
Re-run the five questions at each escalation from observation to pilot to rollout.

### Step 9 — Design a local test

**Input:** prioritized gaps and an authorized, bounded test opportunity.
**Output:** protocol, measurement plan, and stop/scale decision.

Start with a measurable hypothesis and a decision rule agreed before results are inspected.
Specify the eligible units, assignment method, duration, baseline, outcome definitions, and analysis population.
Where practical, compare treatment and control units over the same period; randomize eligible units if feasible and safe.
If randomization is not feasible, document selection, confounding, and the resulting limits on attribution.

Include operational outcomes as well as the model metric: contribution, waste, service, workload, overrides, and adverse events.
Specify how to value inventory changes and lost sales without double counting.
Define how missing data, stockouts, returns, and substitutions will be handled before analysis.
If the test is too small to rule out a consequential downside, report the uncertainty and do not equate a positive point estimate with proof.

Set an authority and response for each stop rule.
For example, the operations owner may pause a pilot when a pre-specified safety or service threshold is breached.
Legal and safety review remain separate from a statistical test.
Set the date and owner for the next decision, and identify what result would support a wider test, continued observation, or termination.

### Step 10 — Write the decision memo

**Input:** claim ledger, five ratings, local economics, risks, and evaluation plan.
**Output:** one concise memo plus traceable calculation and evidence appendices.

Start with the decision requested and the recommendation.
State whether the source is being used as a plan, analogy, or not as support for this action.
List source facts separately from local facts and analyst inference.
Give each rating and its evidence, including unknowns and disqualifying constraints.

Present the local incremental value relative to the actual alternative, not the source outcome.
Label forecast, target, plan, and commitment separately.
State costs and benefits in currency per period and units per period, and provide a downside scenario.
Name the decision owner, budget authority, safety owner, data owner, timeline, and next gate.

End with what would change the recommendation.
A memo that cannot name a disconfirming observation is a sales pitch, not a transfer assessment.
Archive source versions and analysis inputs so that a reviewer can reproduce the reasoning.

## 6. Worked example

### 6.1 Decision and source

**Synthetic teaching case:** All organizations, figures, and effects in this section are invented for calculation practice.
They are not evidence about any real company, supplier, or product.

A food distributor, Harbor, considers a daily replenishment recommendation for 120 chilled items across six stores.
The decision owner can approve an eight-week, supervised pilot, not an automatic chain-wide rollout.
The existing policy is a twice-weekly manual order review.
The source case is an invented account of a dry-goods wholesaler, Cedar, that introduced daily recommendations.
Cedar reports a 12% reduction in expired units relative to its own prior quarter.
The hypothetical account is company-authored, does not describe a control group, and does not quantify adoption or other concurrent changes.
The 12% is therefore a **source claim**, not a causal effect or a Harbor forecast.

The destination decision is whether to authorize a bounded test against the current process.
It is not whether to promise a 12% reduction or buy an enterprise license.
A separate future decision would address rollout.

### 6.2 The five answers for the proposed pilot

| Question | Answer | Rationale and remaining evidence |
|---|---|---|
| Same decision? | Yes, for the proposed pilot | Both choose reorder quantities; Harbor reviews daily with human approval. Confirm item-level cutoffs with the replenishment owner. |
| Similar product economics? | No | Chilled goods expire quickly and have little salvage value; Cedar sells durable goods. Estimate Harbor-specific waste and stockout consequences. |
| Similar data environment? | Unknown | Cedar describes item-level sales but not latency or inventory accuracy. Harbor must check decision-time stock and stockout flags. |
| Similar operating constraint? | No | Harbor has supplier case-pack and chilled receiving limits not documented for Cedar. The pilot must obey them. |
| Evidence strong enough for this risk? | Unknown | The source has no causal comparison; pilot controls and loss limits have not yet been approved. |

There are two no answers, not three.
That count alone does not authorize a pilot.
The data and evidence questions are material unknowns, and supplier and safety constraints are hard gates.
At this point Cedar is an analogy, not a rollout plan or an approved pilot.

The stock controller will check eight weeks of timestamped stock and sales records against physical counts.
The supplier manager will confirm case-pack rules and lead times in writing.
The food-safety owner will confirm that receiving capacity and temperature-control procedures remain unchanged.
The finance owner will confirm the valuation of waste and lost sales before results are interpreted.
Only after these gates pass can the proposed supervised pilot be presented for authorization.

### 6.3 Local economics

Assume Harbor's measured baseline for the 120 items is 18,400 units sold per eight weeks.
The purchase cost is USD 2.40 per unit; disposal costs USD 0.15 per expired unit.
The contribution lost when one otherwise profitable sale is permanently missed is USD 1.10 per unit.
These are destination assumptions for this example, not facts about Cedar.
They exclude other effects, such as substitution and goodwill, which require separate estimation.

Suppose a planning scenario forecasts 310 fewer expired units per eight weeks and 90 fewer permanently missed sales per eight weeks.
The forecast is hypothetical; no pilot has yet measured either outcome.
The value of avoided expiry is:

```text
310 units / 8 weeks × (USD 2.40 + USD 0.15) / unit
= USD 790.50 / 8 weeks
```

The value of avoided permanently missed sales is:

```text
90 units / 8 weeks × USD 1.10 / unit
= USD 99.00 / 8 weeks
```

Expected gross incremental benefit under these assumptions is USD 889.50 per eight weeks.
Suppose incremental data support costs USD 300 per eight weeks and planner review costs USD 240 per eight weeks.
Incremental benefit after those recurring costs is:

```text
USD 790.50 + USD 99.00 - USD 300.00 - USD 240.00
= USD 349.50 / 8 weeks
```

An additional one-time setup cost of USD 1,100 makes the first eight-week pilot net value USD -750.50 if the scenario is realized and all stated costs are included.
This simplified financial result excludes taxes, financing, inventory holding changes, and any costs not listed.
A pilot can still be justified as a learning investment, but this arithmetic alone does not prove that it should be funded.
Avoided expiry and avoided lost sales are separate effects in this example; verify actual unit accounting to avoid counting the same demand twice.
The avoided-expiry value assumes the spoiled units can be avoided without a compensating purchase or other cost.

### 6.4 Downside and decision boundary

In a downside scenario, assume only 80 units of expiry are avoided per eight weeks and 40 additional sales are permanently missed per eight weeks.
With the same recurring costs:

```text
80 × USD 2.55 - 40 × USD 1.10 - USD 540
= USD -380 / 8 weeks
```

After the one-time USD 1,100 setup cost, the first-period downside is USD -1,480.
This is a scenario, not a statistical confidence bound or a maximum loss.
It excludes any unquantified safety, customer, and operational effects.
The finance owner must decide whether the authorized risk budget covers plausible downside before the pilot starts.

Holding the assumed 90 avoided missed sales fixed, recurring break-even expiry reduction is:

```text
(USD 540 - 90 × USD 1.10) / USD 2.55 per unit
= 172.94 units / 8 weeks
```

Because units are discrete, at least 173 fewer expired units per eight weeks are needed for nonnegative recurring value under those assumptions.
First-period break-even including setup is 605 units, rounded up from 604.31 units.
These thresholds shift if service, costs, adoption, or volume changes.

### 6.5 Test and memo recommendation

A possible design randomizes eligible, comparable store-item groups into recommendation and control arms, subject to operational and safety approval.
Both arms run during the same eight-week period.
The treatment arm retains human approval and respects case packs and receiving limits.
The control arm continues the twice-weekly review.
The analysis measures net expiry units, permanently lost sales where estimable, contribution, labor, overrides, and adverse events.
The group is the assignment unit and clustering and spillovers must be considered in uncertainty estimates.

The 310-unit and 90-unit numbers refer to the full 120-item population per eight weeks.
Do not compare them directly with the raw treatment-arm outcome if only half the population receives treatment.
Estimate the treatment-control difference at a common exposure and state any scaling assumption.
The small population and eight-week duration may not resolve commercially important effects; determine detectable differences before committing to a definitive rollout decision.

An illustrative guardrail is to pause the affected group if a food-safety breach occurs, with the food-safety owner applying the existing incident process.
A service guardrail should be defined using a measurable rate, minimum denominator, threshold, review window, and owner before the pilot; no threshold is supplied by the external source.
An adverse service signal should trigger review without waiting for the final financial analysis.

**Recommendation:** Treat Cedar as an analogy for examining daily replenishment.
Prepare the local pilot protocol and gate evidence; do not authorize automated rollout or claim Cedar's 12% as a Harbor benefit.
The illustrative forecast is USD 349.50 per eight weeks after recurring costs and before setup, not an observed result.
A possible target may be set by the owner after reviewing baseline variability; no target is yet set.
The plan is to obtain data and operational approvals and seek pilot authorization.
There is no commitment to purchase a license or scale the intervention.

### 6.6 Second worked example: transferring a routing method

**Synthetic teaching case:** North Parcel and Valley Medical are invented organizations.
Neither their operating data nor the calculations below describe a real fleet.

North Parcel reports that a route-sequencing project reduced planned driving distance by 8% in a dense urban network.
The source is a hypothetical vendor case study with no independent comparison.
The claim concerns planned kilometers, not observed driver hours, fuel costs, or customer outcomes.
Valley Medical has ten vehicles serving rural clinics and must decide whether to test new route sequences on two vehicles for four weeks.
Deliveries have appointment windows and some products require temperature control.

The decision owner compares a supervised sequencing pilot with the existing dispatcher-created routes.
The destination may not delay a temperature-sensitive delivery outside its validated handling limits.
The dispatch supervisor can reject a route, and the safety owner can stop the pilot.
The source percentage is not an estimate of Valley's savings.

| Question | Initial answer | Reason |
|---|---|---|
| Same decision? | Yes | Both sequence accepted stops, although the service requirements differ. |
| Similar economics? | No | Dense parcel delivery and rural clinical delivery have different distance, labor, and failure costs. |
| Similar data environment? | Unknown | Valley must verify appointment windows, travel times, and location accuracy at dispatch time. |
| Similar operating constraint? | No | Validated cold-chain limits and clinic windows are not documented in the source. |
| Evidence strong enough for this risk? | Unknown | The source reports planned distance without a causal comparison; the proposed pilot needs defined safeguards. |

There are two no answers and two unknowns.
The source is an analogy while Valley establishes the feasibility and safety of the proposed supervised pilot.
The hard cold-chain gate must pass independently of the numerical assessment.
Even a well-designed pilot cannot authorize a route that breaches that gate.

Assume Valley's two-vehicle pilot baseline is 2,300 km per week and that the destination forecasts a 4% distance reduction.
The 4% is a local scenario assumption, not an observed improvement and not half of the source effect by a proven adjustment.
At USD 0.42 of avoidable vehicle cost per kilometer, forecast gross vehicle savings are:

```text
2,300 km/week × 0.04 × USD 0.42/km = USD 38.64/week
```

Assume dispatch review adds 1.5 hours per week at an incremental cost of USD 24 per hour.
The resulting forecast net is:

```text
USD 38.64/week − 1.5 hours/week × USD 24/hour = USD 2.64/week
```

This deliberately narrow calculation excludes setup cost, changes in driver hours, customer service effects, and safety costs.
It does not assume that a shorter route releases a paid driver hour.
If driver shifts are fixed, an apparent time saving may have no immediate labor cash value.
If fewer kilometers reduce fuel but not maintenance contracts, USD 0.42 per kilometer may be an overestimate.
The finance owner must decompose the rate into costs that really change.

If the achieved reduction is only 2%, gross savings fall to USD 19.32 per week and net value becomes USD -16.68 per week before setup.
If review time rises to two hours per week while the reduction is 4%, net value becomes USD -9.36 per week.
These two sensitivities show why a positive gross distance result does not automatically support deployment.
They are scenarios, not confidence intervals.

The operational test should log actual departure, arrival, stop sequence, kilometers, delivery-window compliance, temperature excursions, and dispatcher overrides.
Compare treatment routes with a concurrent, appropriately matched alternative if assignment can be arranged without compromising service.
Account for differences in number of stops and geography.
The comparison should use actual completed routes, not merely planned route distance.

The target, if management chooses one, must be stated before evaluation and must not override safety requirements.
The current plan is to validate data and handling procedures, then seek approval for a bounded test.
No vehicle, contract, or network-wide software purchase is committed by this example.
A failed safety gate means no pilot under the proposed design, regardless of the forecast net value.

### 6.7 Third worked example: a transferable mechanism without a transferable policy

**Synthetic teaching case:** a spare-parts distributor reads a fictional case in which a grocery retailer improved availability by ordering every day.
The distributor stocks a component costing USD 180 per unit and sells it for USD 250 per unit.
The supplier charges USD 45 per order and the annual carrying charge is 20% of inventory acquisition value.
The source case reports an availability improvement but omits its ordering cost and supplier terms.

The business decision is whether to move from one replenishment order per week to five orders per week for this component.
Assume 52 operating weeks per year and that the ordering fee applies to each order.
The extra order fees would be:

```text
(5 − 1) orders/week × 52 weeks/year × USD 45/order
= USD 9,360/year
```

Assume, for illustration only, that more frequent ordering reduces average cycle stock by 12 units.
At USD 180 per unit and a 20% annual carrying charge, forecast annual carrying-cost reduction is:

```text
12 units × USD 180/unit × 0.20/year = USD 432/year
```

On these two terms alone, the change is USD 8,928 per year more expensive.
The USD 70 per unit selling-price difference is not automatically the full incremental contribution: other variable costs may apply.
If USD 70 per unit were the verified incremental contribution and no other changes occurred, at least 128 additional permanently retained sales per year would be needed to offset the USD 8,928 difference.
That threshold is a scenario calculation, not a prediction that these sales exist.

The grocery case may prompt investigation of replenishment cadence.
Its policy does not transfer automatically because the decision economics differ.
Data must also show whether the distributor actually has preventable stockouts and whether a five-order cadence changes them.
If the supplier offers consolidated ordering or a different fee, rerun the calculation rather than assuming the original conclusion remains true.

For the proposed policy, mark the decision question yes only if both cases choose order frequency for comparable decisions.
Mark economics no because the source provides no support for the distributor's material order fee and the local scenario makes the proposed cadence expensive.
The data, constraints, and evidence answers require their own documentation.
Do not infer a three-no result merely from the adverse economic calculation.
One economically decisive no is enough to reject this particular policy on the stated assumptions.

The broader idea of matching replenishment cadence to service and cost remains useful as an analogy.
The next local decision could be to measure stockout causes and obtain a supplier quote for consolidated orders.
That evidence-gathering action is not a commitment to place five orders per week.

### 6.8 How to compare evidence across the three synthetic decisions

The three cases have different action units.
Harbor chooses item-level reorder quantities.
Valley chooses the sequence of stops on already accepted routes.
The spare-parts distributor chooses how often to place orders.
Calling all three “optimization” would hide the different decisions and consequences.

The appropriate denominator also differs.
Harbor's example values units of expiry and missed sales per eight weeks.
Valley values kilometers and dispatcher hours per week.
The distributor values order fees and carrying costs per year.
Before comparing proposals across a portfolio, translate each into a common currency and decision horizon without losing its operational denominator.
For example, multiply a stable weekly estimate by 52 only if the weekly conditions and activity apply throughout the year.
An eight-week pilot observed during a holiday peak is not automatically representative of 52 weeks.

The comparison groups are also different.
A replenishment intervention may spill over across stores sharing the same warehouse stock.
A routing intervention may affect multiple vehicles when stops are reassigned.
An order-frequency intervention may alter a supplier's batch economics across several items.
Define the unit of assignment and identify shared resources before claiming that observations are independent.
If spillovers are likely, use a design and analysis that acknowledge them.

Each case also has a different possible source of downside.
For Harbor, extra waste and missed sales matter alongside food safety.
For Valley, a missed appointment or temperature excursion can be more consequential than a few extra kilometers.
For the distributor, extra order fees can exceed the value of reduced cycle stock.
An average financial benefit does not cancel a non-negotiable safety failure.

### 6.9 Decision record for reassessment

Before a second approval meeting, update the record rather than overwriting the first assessment.
For each question, keep the initial answer, the new evidence, the revised answer, the evidence owner, and the decision date.
This makes it possible to distinguish a genuine change in facts from a change in risk appetite.

For Harbor, a decision record could contain:

| Item | Initial record | Evidence needed before pilot decision |
|---|---|---|
| Decision | Supervised eight-week pilot | Sponsor confirms eligible sites and alternatives. |
| Product economics | No: short-life inventory differs from the source | Finance validates incremental waste and lost-sales values. |
| Data | Unknown | Stock controller validates decision-time accuracy and access. |
| Operations | No: different constraints | Supplier and safety owners approve feasible case-pack and receiving rules. |
| Evidence for risk | Unknown | Sponsor accepts protocol, comparison, stop rules, and loss budget. |
| Approval | Not requested yet | Accountable authority makes a recorded decision after gates. |

An operating difference can remain factually true even when a redesigned pilot is feasible.
For example, case packs do not disappear because an optimization model enforces them.
Describe the original mismatch and the local control rather than claiming the source and destination have become identical.
Rate the redesigned action explicitly; do not retroactively change the rating for the original unconstrained action.

If a gate fails, record the date, owner, and reason.
The next action may be a data repair, a revised scope, or rejection.
If the proposed design changes materially, repeat the five questions for that design.
A documented rejection can be a successful transfer test because it prevents an unsupported commitment.

### 6.10 What the sponsor should receive

The executive summary should fit on one page, but it must not conceal a material failure.
List the action requested, the comparison alternative, the recommendation, the main downside, and the next gate.
Attach the five-question record and calculation separately so a reviewer can follow the result.

The forecast belongs beside its assumptions and time bucket.
The target belongs beside its owner and the date on which it was set.
The plan belongs beside its dependencies and approval gates.
The commitment belongs beside the authorized amount, responsible authority, and duration.
Where one of these does not yet exist, state that explicitly rather than manufacturing a number.

The source claim belongs in the evidence ledger with its link and publication date.
The analyst's causal or transfer inference belongs in a separate field.
The local test result, when available, belongs with its comparison and uncertainty.
This separation lets the sponsor revise a forecast without misrepresenting the original evidence.

End the memo with a specific decision request.
For example: “Authorize the data-quality assessment and return with a pilot protocol; no automated orders are authorized.”
That sentence sets a boundary on both expenditure and operational authority.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the worked example's economic scenarios and applies the three-no screening rule.
It intentionally requires an explicit `hard_gate_passed` and `material_unknowns_resolved` input for plan eligibility.
These booleans are governance inputs, not findings established by the program.
The code does not estimate a causal effect, prove regulatory compliance, or authorize a decision.

Run the fenced block as a Python file, for example `python transfer_test.py`.
Currency is USD and each scenario covers eight weeks.
Inputs are incremental relative to the stated current process.
A negative `avoided_lost_sales` means additional permanently missed sales.

```python
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING


@dataclass(frozen=True)
class Scenario:
    avoided_expiry_units: Decimal
    avoided_lost_sales: Decimal
    acquisition_cost_per_unit: Decimal
    disposal_cost_per_unit: Decimal
    lost_sale_contribution: Decimal
    recurring_cost: Decimal
    setup_cost: Decimal


def economics(scenario: Scenario) -> tuple[Decimal, Decimal]:
    expiry_value = scenario.avoided_expiry_units * (
        scenario.acquisition_cost_per_unit + scenario.disposal_cost_per_unit
    )
    sales_value = scenario.avoided_lost_sales * scenario.lost_sale_contribution
    recurring_net = expiry_value + sales_value - scenario.recurring_cost
    first_period_net = recurring_net - scenario.setup_cost
    return recurring_net, first_period_net


def expiry_break_even(scenario: Scenario, include_setup: bool) -> int:
    unit_value = scenario.acquisition_cost_per_unit + scenario.disposal_cost_per_unit
    if unit_value <= 0:
        raise ValueError("Avoided expiry value must be positive")
    cost = scenario.recurring_cost + (scenario.setup_cost if include_setup else 0)
    required = (cost - scenario.avoided_lost_sales * scenario.lost_sale_contribution) / unit_value
    return int(max(Decimal(0), required).to_integral_value(rounding=ROUND_CEILING))


def disposition(
    answers: tuple[str, ...],
    *,
    hard_gate_passed: bool,
    material_unknowns_resolved: bool,
    locally_supported: bool,
) -> str:
    if len(answers) != 5 or any(x not in {"yes", "no", "unknown"} for x in answers):
        raise ValueError("Provide five answers: yes, no, or unknown")
    if answers.count("no") >= 3:
        return "analogy"
    if not hard_gate_passed or not material_unknowns_resolved:
        return "analogy or do not use, depending on the gap"
    if "unknown" in answers or not locally_supported:
        return "analogy"
    return "plan eligible, subject to approval"


base = Scenario(
    avoided_expiry_units=Decimal("310"),
    avoided_lost_sales=Decimal("90"),
    acquisition_cost_per_unit=Decimal("2.40"),
    disposal_cost_per_unit=Decimal("0.15"),
    lost_sale_contribution=Decimal("1.10"),
    recurring_cost=Decimal("540"),
    setup_cost=Decimal("1100"),
)
downside = Scenario(
    avoided_expiry_units=Decimal("80"),
    avoided_lost_sales=Decimal("-40"),
    acquisition_cost_per_unit=base.acquisition_cost_per_unit,
    disposal_cost_per_unit=base.disposal_cost_per_unit,
    lost_sale_contribution=base.lost_sale_contribution,
    recurring_cost=base.recurring_cost,
    setup_cost=base.setup_cost,
)

for label, scenario in (("base", base), ("downside", downside)):
    recurring, first_period = economics(scenario)
    print(f"{label}: recurring USD {recurring:.2f}; first period USD {first_period:.2f}")
print(f"base break-even expiry units, recurring: {expiry_break_even(base, False)}")
print(f"base break-even expiry units, including setup: {expiry_break_even(base, True)}")
print("pilot disposition:", disposition(
    ("yes", "no", "unknown", "no", "unknown"),
    hard_gate_passed=False,
    material_unknowns_resolved=False,
    locally_supported=False,
))
```

Expected output:

```text
base: recurring USD 349.50; first period USD -750.50
downside: recurring USD -380.00; first period USD -1480.00
base break-even expiry units, recurring: 173
base break-even expiry units, including setup: 605
pilot disposition: analogy or do not use, depending on the gap
```

The classifier deliberately cannot decide whether a source is irrelevant or whether a hard gate is remediable.
Those are professional judgments documented in the memo.
A `no` may remain in the record even after a local redesign has demonstrated that the resulting plan is feasible.
Set `locally_supported` only after that redesign and its evidence are documented.
The implementation also conservatively leaves any unresolved `unknown` as an analogy, even when a user has marked material unknowns resolved.

## 8. Data, standards, and regulatory context

### Data contract

A transfer memo needs both source-case metadata and destination operational records.
Source metadata should include publication date, population, intervention, reported outcomes, comparison, original wording, and source location.
Destination records should include item and location identifiers, time zone, event time, availability time, transaction type, unit of measure, and version.
An extract alone is not a data contract: define meaning, owner, refresh schedule, permitted use, and quality checks.

At a minimum, document:

| Data element | Question for transfer | Example check |
|---|---|---|
| Sales | Does the metric reflect unconstrained demand? | Compare zero-sales periods with in-stock flags. |
| Inventory | Was the stock record available before the decision? | Compare event and ingestion timestamps. |
| Orders | Are quantities recommendations, commitments, or receipts? | Reconcile order lines to confirmed receipts. |
| Products | Do item codes and units remain stable over time? | Inspect identifier changes and pack conversions. |
| Costs | Are avoided costs actually avoidable? | Trace costs to finance and operations owners. |
| Outcomes | Are losses, substitutions, and returns measured consistently? | Reconcile numerator and denominator to the source systems. |
| Restrictions | Is the proposed use authorized? | Obtain appropriate privacy, security, and contractual review. |

Use an as-of snapshot where available: the information that the planner could actually have seen at the historical decision time.
Record revisions rather than replacing history silently.
Preserve the transformation from transaction to outcome so that a reviewer can reproduce the comparison.
Data access and retention should follow the destination's policy and applicable obligations; an external case does not grant permission to reuse its records or infer permission to process local records.

### Standards and obligations

[GS1's Global Trade Item Number guidance](https://www.gs1.org/standards/id-keys/gtin) describes a globally used identification key for trade items.
A GTIN can help distinguish trade items across trading partners, but possession of a code does not prove inventory accuracy, product equivalence, or permission to exchange data.
Local mappings and packaging levels still need verification.
Adoption is a business and trading-partner convention unless a particular contract or rule makes it mandatory in context.

The [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) is voluntary guidance for managing AI risk, not a general authorization to deploy a model.
Its govern, map, measure, and manage functions can structure accountability, context, measurement, and response in a local test.
It does not supply a universal pass threshold or substitute for applicable law.

For food operations, the [U.S. Food and Drug Administration's Food Safety Modernization Act page](https://www.fda.gov/food/food-safety-modernization-act-fsma) describes the statutory and regulatory program.
Whether a specific facility, product, activity, or jurisdiction is subject to a particular rule depends on its facts and current law.
As of November 2026, this lesson does not determine the applicability of any particular requirement; obtain current qualified legal and food-safety advice.
Do not treat a benchmark case as evidence that temperature controls, traceability, or a specific workflow are compliant.

Where personal information is involved, privacy, access, retention, and security requirements depend on jurisdiction and processing context.
For example, the [European Commission's data protection overview](https://commission.europa.eu/law/law-topic/data-protection_en) explains the EU framework.
A route-optimization case may include employee location or performance data even if the published result reports only route cost.
Review the actual data use rather than assuming a supply-chain purpose removes privacy obligations.

Standards are useful for naming and organizing data, but conformity with a standard is not proof of causal benefit or transferability.
A voluntary framework, a contractual obligation, and a legal requirement are different things.
For each proposed action, document the applicable instrument, its owner, and the approval evidence rather than copying the source's compliance label.

## 9. Industry practice and case evidence

### Public case: M5 forecasting competition

The [M5 forecasting competition paper](https://doi.org/10.1016/j.ijforecast.2021.11.013) describes the competition and its retail forecasting setting.
Its data and evaluation design are public and therefore suitable for examining forecasting methods and the relationship between item-level and aggregated performance.
The published competition concerns a particular dataset and scoring problem; it is not a randomized deployment of a replenishment policy.

**Documented context:** the competition used retail sales series with hierarchical structure and evaluated forecasts under competition-defined rules.
Consult the paper for the precise dataset, horizon, and metric.
**Inference:** an analyst designing forecasts for a multi-level retail network can use the competition to formulate questions about hierarchy, evaluation, and cross-series learning.
**Not established:** that the winning approach produces a particular inventory saving in a different retailer.

For transfer, the decision question matters first.
Forecasting sales for a competition is not the same as deciding order quantities under supplier and shelf-life constraints.
The economic objective may be contribution or service rather than the competition metric.
Data latency, availability, assortment, and censoring may differ.
A benchmark result may justify a local backtest, while an operational deployment requires local process and financial evidence.

### Public case: Zillow Offers

[Zillow Group's 2021 annual report](https://investors.zillowgroup.com/financials/sec-filings/default.aspx) documents the company's decision to wind down Zillow Offers and provides the company's discussion of the business and associated financial results.
An annual report is an official company filing; the company supplies the disclosures and explanations.
The filing should be read for the specific reported period and exact amounts rather than reduced to a single model-failure slogan.

**Documented fact:** the company disclosed the wind-down of its home-buying business.
**Inference:** a decision process that buys inventory on the basis of uncertain estimates can expose capital to forecast error and market changes.
**Not established:** that any specific forecasting model alone caused the reported outcome, or that a food distributor faces the same housing-market dynamics.

The case can transfer as a question about downside exposure, concentration, limits, and exit paths.
It does not transfer as a numeric loss rate or as proof that a particular local algorithm is unsafe.
A prudent destination team would stress-test positions and decision limits using its own economics and governance.

### Public case: UPS routing

[UPS's announcement about UPSNav](https://about.ups.com/us/en/newsroom/press-releases/innovation-driven/ups-deploys-purpose-built-navigation-for-ups-service-personnel.html) describes a navigation capability built on the company's route-optimization initiative and its stated benefits.
Treat reported benefits in a company announcement as company claims, not independent causal estimates.
Read the publication for its time period and exact wording; later operating conditions may differ.

**Inference:** route recommendations need compatible data, operating rules, and driver or dispatcher workflows to create value.
**Not established:** that the same savings per route, mile, or year would appear in a destination fleet.
The transfer test asks whether the destination is sequencing stops, selecting customers, changing fleet capacity, or making another decision entirely.
Density, time windows, vehicle characteristics, labor arrangements, and service obligations may materially alter the solution.

### Reading the three cases together

These sources support different kinds of statements.
A competition paper documents a benchmark setting, a filing documents company disclosures, and a company announcement describes a corporate initiative and claim.
None alone establishes destination value.
For each source, cite the relevant passage and date, preserve the original denominator, and state whether the local recommendation is based on a source fact or on an analyst's inference.
A source that fails to support a rollout may still motivate a question worth testing.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Technology-first framing | The request names a tool but no decision. | Similar products solve different problems. | Procurement without a usable workflow. | Name owner, alternative, cadence, and decision variable. |
| False decision match | Both cases say “forecasting” but one makes orders and the other predicts sales. | Prediction and action have different objectives. | Wrong metric and policy. | Map prediction to the operational decision and counterfactual. |
| Margin mismatch | Benefits use revenue rather than contribution. | Revenue includes costs that continue after a sale. | Overstated financial value. | Agree incremental cost and contribution definitions with finance. |
| Unpriced spoilage | The source sells durable goods and the destination sells chilled goods. | Unsold units have different salvage and disposal costs. | Excess ordering and waste. | Model local cost asymmetry and shelf life. |
| Hidden data latency | A historical table is complete only after month end. | Backtest accesses information unavailable at order time. | Deployment misses backtest performance. | Reconstruct decision-time snapshots. |
| Censored demand | Zero sales are interpreted as zero demand during stockouts. | Availability limits observed sales. | Underestimated demand and repeated stockouts. | Join stock availability and examine censoring. |
| Constraint omission | Recommendations violate case packs or delivery windows. | Optimization searches infeasible actions. | Overrides, workarounds, or service failure. | Encode constraints and validate with operators. |
| Evidence inflation | A vendor quote becomes “proven savings.” | A claim is mistaken for a causal estimate. | Unjustified commitment. | Trace claim, comparison, source type, and uncertainty. |
| Source selection | Only successful deployments are presented. | Unpublished failures are missing. | Optimistic effect expectations. | Seek counterexamples and document selection limitations. |
| Unknown treated as yes | Missing supplier information is called “similar.” | Absence of contrary evidence is confused with evidence of fit. | Premature approval. | Assign unknown, owner, deadline, and gate. |
| Counting away a hard gate | Four yes answers are used to override an unsafe process. | A tally obscures a non-negotiable constraint. | Safety or compliance breach. | Treat hard gates separately from the three-no screen. |
| Pre/post confounding | A gain follows implementation during a promotion change. | Other changes also affect the outcome. | False attribution. | Use a contemporaneous comparison where feasible. |
| Pilot extrapolation | Treatment-arm savings are multiplied without checking exposure. | Different population sizes and spillovers bias scaling. | Inflated rollout case. | Normalize to common exposure and show uncertainty. |
| Adoption ignored | The model is accurate but planners override it frequently. | Recommendations do not change decisions. | No realized operational benefit. | Measure adoption and reasons for overrides. |
| Safety metric omitted | Financial results are reviewed before incidents. | Economic average hides unacceptable adverse events. | Harm during evaluation. | Define stop rules and independent safety authority. |
| Stale comparison | The source is from a different market regime. | Demand, rates, or regulations change. | Invalid assumptions. | Date evidence and refresh decisive local inputs. |
| Double-counted benefit | Fewer expired units and higher sales claim the same units twice. | Overlapping causal pathways are summed. | Overstated value. | Reconcile unit flows and define distinct outcomes. |
| Pilot becomes rollout | A small diagnostic receives a permanent budget without a new gate. | Scope expands while evidence remains unchanged. | Uncontrolled exposure. | Reassess all five questions at escalation. |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test or respond |
|---|---|---|---|
| Decision is specified | Fit depends on the action. | Different stakeholders mean different choices. | Obtain a signed decision statement and alternatives. |
| Source claim is traceable | Comparison requires knowing what was actually reported. | Secondary retellings omit denominators or qualifiers. | Locate primary wording and archive the relevant version. |
| Source population is described | Applicability depends on context. | Selection and exclusions are hidden. | Record gaps and avoid numerical transfer. |
| Local cost per expired unit is avoidable | The economic model values reduced expiry. | Purchases are committed or salvage value is material. | Trace purchase, disposal, and salvage flows with finance. |
| Avoided lost sales are incremental | Contribution value assumes a sale would otherwise be lost. | Customers substitute or purchase later. | Estimate substitution and compare customer-level outcomes where permitted. |
| Benefits are not double counted | The sum assumes distinct outcomes. | One reduced expiry unit also appears as a separately valued sale. | Reconcile inventory and demand flows. |
| Data reflects decision-time availability | Local backtest must mimic live operation. | Corrections or receipts arrive after the order cutoff. | Compare event and ingestion times using historical snapshots. |
| Control represents the alternative | A comparison estimates an incremental effect. | Unequal selection, spillovers, or concurrent changes. | Randomize where feasible; report design and residual limitations. |
| Pilot conditions resemble intended scale | Extrapolation assumes stable effects. | A wider rollout changes capacity, supplier behavior, or adoption. | Test scale-sensitive constraints before escalation. |
| Eight-week exposure is representative | The example uses a single period. | Seasonality, holidays, and promotions alter outcomes. | Repeat or stratify across relevant conditions. |
| Overrides are observable | Adoption affects realized value. | Decisions occur outside the system. | Reconcile recommendations, orders, and exception logs. |
| Hard gates remain effective | Safety and legal conditions must hold throughout the test. | Workflow changes or incident-response capacity is exceeded. | Obtain accountable sign-off and monitor continuously. |
| Value estimates have suitable precision | The decision may depend on a narrow margin. | Small samples or rare adverse events leave uncertainty wide. | Evaluate detectable effect and interval before a rollout decision. |

A simple three-no screen cannot represent every dependency or uncertainty.
The labels in this lesson support consistent discussion, not a validated statistical classifier.
The financial model is a partial decision model, not a full profit-and-loss forecast.
The example excludes unlisted costs and benefits and does not claim real-world evidence.
A decision owner may reasonably fund a negative-expected-value pilot for information if the learning objective and downside are explicit.
That is a separate decision from approving permanent operations on the basis of a projected positive recurring value.

## 12. Exercises

### Exercise 1 — Decision framing

A vice president says: “Our competitor uses artificial intelligence, so we should copy its replenishment platform.”
You know only that the competitor reports fewer stockouts.

**Deliverable:** Write a destination decision statement and two alternatives.
**Acceptance criteria:** Name an owner, decision variable, population, cadence, time horizon, and one constraint; distinguish a pilot from a rollout.
**Time:** 15 minutes.

### Exercise 2 — Five-question classification

A source case uses daily delivery routing in a dense city.
Your destination chooses weekly inventory allocation for rural depots.
The destination has reliable sales records, but the source's data, route constraints, and causal comparison are not described.
A marketing article claims 20% savings.

**Deliverable:** Give five yes/no/unknown answers and a one-paragraph recommendation for copying the route policy into allocation.
**Acceptance criteria:** Justify each rating; distinguish unknown from no; do not assume the claim is a local forecast.
**Time:** 20 minutes.

### Exercise 3 — Screening and hard gates

Consider these answers for a proposal: yes, no, no, no, yes.
A second proposal has answers yes, yes, yes, no, yes, but its no is a known violation of a non-negotiable safety constraint.

**Deliverable:** Classify each proposal under the three-no rule and explain whether either is plan eligible.
**Acceptance criteria:** Apply the literal no count to the first and explain why the second does not pass solely because it has fewer than three no answers.
**Time:** 10 minutes.

### Exercise 4 — Calculate incremental value

A synthetic 12-week scenario forecasts 220 fewer expired units, valued at USD 3.20 per unit including disposal, and 40 fewer permanently missed sales worth USD 1.50 contribution each.
Recurring incremental labor and data cost USD 500 per 12 weeks; one-time setup costs USD 450.

**Deliverable:** Calculate gross benefit, recurring net, first-period net, and whole-unit recurring break-even expiry reduction, holding the sales assumption constant.
**Acceptance criteria:** Show units and arithmetic; round the break-even threshold up; label the result as a forecast rather than an observed benefit.
**Time:** 15 minutes.

### Exercise 5 — Data and operations

A local inventory feed is corrected nightly, but orders lock at 15:00.
The source used a real-time feed.
The destination supplier also requires orders in packs of 12 units.

**Deliverable:** List two data checks and two workflow checks needed before a supervised pilot.
**Acceptance criteria:** Address decision-time availability, stock accuracy, pack feasibility, and an operational owner or escalation.
**Time:** 20 minutes.

### Exercise 6 — Evidence and risk

A vendor's case study says that a customer reduced costs by 15% after installing a tool.
No baseline period, counterfactual, or excluded costs are given.
A director wants to commit USD 2 million to a three-year rollout.

**Deliverable:** Write an evidence assessment and name three further evidence requests.
**Acceptance criteria:** Separate occurrence, causal attribution, and local transfer; connect the evidence threshold to commitment size and reversibility.
**Time:** 20 minutes.

### Exercise 7 — Evaluation design

Design an eight-week comparison for 40 stores where a recommendation changes reorder quantities.
Assume store managers can approve or override each recommendation.

**Deliverable:** Produce a one-page protocol.
**Acceptance criteria:** Include eligibility, assignment or comparison, outcomes, baseline, exposure, adoption, missing data, stop rule, authority, and a scale decision; acknowledge uncertainty and spillovers.
**Time:** 35 minutes.

### Exercise 8 — Decision memo

Use the synthetic Harbor case in section 6.
The sponsor asks for immediate enterprise deployment and calls the source's 12% a guaranteed local outcome.

**Deliverable:** Write a memo of at most 350 words.
**Acceptance criteria:** State a recommendation, five ratings, key financial values, a material unknown, a hard gate, a proposed owner, and the next decision; separate source claim, forecast, target, plan, and commitment.
**Time:** 25 minutes.

## 13. Self-check answers

### Exercise 1

A strong decision statement could say: “The replenishment director will decide this quarter whether to authorize an eight-week supervised recommendation pilot for selected store-item pairs, subject to supplier pack and safety constraints.”
Possible alternatives are to retain the current ordering policy or first improve inventory data quality.
The owner, decision, population, period, constraint, and alternatives are explicit.
Neither the competitor's identity nor the phrase “artificial intelligence” defines the decision.
A full rollout would be a separate higher-risk decision.

### Exercise 2

A defensible classification for copying the route policy directly into allocation is:

| Question | Answer | Reason |
|---|---|---|
| Same decision? | No | Routing and inventory allocation choose different actions. |
| Similar economics? | Unknown | No unit economics for either operation are supplied. |
| Similar data environment? | Unknown | Destination reliability alone does not establish source comparability. |
| Similar operating constraint? | Unknown | Density differs, but the source constraint set is not described; rurality alone is not enough to prove the decisive constraint. |
| Evidence sufficient for risk? | No | An unsupported marketing claim is insufficient for the proposed direct copy. |

There are two no answers and three unknowns.
The case is an analogy at most; the no count does not make it a plan.
The article may prompt a local question about constraints and operational adoption, but its 20% is not a local forecast.
Alternative defensible ratings must explain which known facts establish a material difference and must not turn missing information into a yes.

### Exercise 3

The first proposal has three no answers, so the module rule makes the case an analogy, not a plan.
The team may investigate a narrower action if appropriate; it should not call the original proposal ready.
The second has only one no, so the three-no trigger does not apply.
Nevertheless the safety failure blocks the proposed action until the requirement is satisfied and independently verified.
Passing a count is not passing a hard gate.

### Exercise 4

Gross forecast benefit is `(220 × USD 3.20) + (40 × USD 1.50) = USD 764` per 12 weeks.
Recurring net is `USD 764 − USD 500 = USD 264` per 12 weeks.
First-period net including setup is `USD 264 − USD 450 = USD -186`.
The recurring break-even condition is `3.20 × units + 60 ≥ 500`, so units must be at least `137.5`.
Whole units require 138 fewer expired units per 12 weeks.
These values rely on distinct, avoidable effects and are not measured outcomes.

### Exercise 5

Check whether the feed available at 15:00 includes all relevant receipts and transactions that occurred before the order lock.
Compare as-of inventory to physical counts and investigate correction frequency and direction.
Check that proposed quantities are multiples of 12 and that the supplier can meet the intended cadence and lead time.
Have the replenishment and supplier owners confirm exception handling and who can stop or override orders.
A nightly corrected feed cannot be assumed to reflect the information available at 15:00.

### Exercise 6

The article makes a vendor claim; it does not by itself establish the measured cost reduction, the effect caused by the tool, or applicability to the destination.
For a USD 2 million, three-year commitment, ask for the underlying outcome definitions and data, a credible comparison or evaluation design, and the destination's own cost, feasibility, and downside analysis.
Also request the contract's exit terms and an independent reference if available.
A smaller diagnostic could be considered under a separate risk and authorization decision.
A company or vendor statement may still identify useful hypotheses; it does not justify treating 15% as a guaranteed saving.

### Exercise 7 rubric

A strong protocol defines a common eligibility population and proposes randomized store assignment if operationally feasible.
It specifies pre-intervention measurements and a concurrent control, with outcomes including financial value, service, waste, and overrides.
It states the assignment unit, outcome denominator, measurement period, handling of missing data, and how cross-store spillovers could dilute the contrast.
It sets an actionable adverse-event stop rule and owner before the test, estimates whether the sample can detect a meaningful effect, and explains how uncertainty affects the scale decision.
A weak protocol reports only forecast accuracy or declares success from a treatment-only before-and-after comparison.
A well-explained quasi-experimental alternative can be acceptable when randomization is infeasible, provided its confounding limits are explicit.

### Exercise 8 rubric

A strong memo rejects immediate deployment and correctly calls 12% a source claim.
It reports the two no answers and two unknowns in section 6, notes that fewer than three no answers do not authorize a plan, and identifies the local hard gates.
It distinguishes USD 349.50 recurring forecast net per eight weeks from USD -750.50 first-period forecast net including setup.
It names a decision owner and proposes a gated pilot or evidence-gathering decision rather than an enterprise commitment.
It explicitly states that no local target has been set and that the plan to seek approval is not a purchase commitment.

## 14. Interview questions

1. **When is a case an analogy rather than a plan?**
   A strong answer distinguishes hypothesis generation from locally supported action and mentions the three-no rule, unknowns, and hard gates.
   A weak answer equates a matching sector with permission to copy.
2. **What exactly must match in the decision question?**
   A strong answer names decision variable, objective, cadence, authority, and alternative.
   A weak answer says only that both organizations use forecasting.
3. **How can a good forecast harm value?**
   A strong answer connects a prediction to asymmetric overage and underage costs, constraints, and policy.
   A weak answer assumes lower error always means lower cost.
4. **Why is a source's percentage saving not a local forecast?**
   A strong answer examines denominator, counterfactual, economics, adoption, and uncertainty.
   A weak answer multiplies the percentage by local spend without checking scope.
5. **What is decision-time leakage?**
   A strong answer gives an example of a corrected inventory record arriving after the ordering cutoff.
   A weak answer describes only missing values.
6. **How do you treat three no answers?**
   A strong answer applies the analogy rule and still considers narrower evidence-gathering actions.
   A weak answer averages the five answers into a favorable score.
7. **What if only one answer is no?**
   A strong answer checks whether the no is a hard safety, legal, or feasibility blocker and considers material unknowns.
   A weak answer declares the proposal safe because four answers are yes.
8. **How do you distinguish an outcome from an effect?**
   A strong answer asks what would have happened without the intervention and identifies credible comparison options.
   A weak answer treats temporal sequence as causation.
9. **How would you design a local pilot?**
   A strong answer names population, comparison, outcomes, assignment, duration, stop rules, and limitations.
   A weak answer measures only whether the software ran.
10. **What changes between pilot and rollout?**
    A strong answer discusses exposure, adoption, capacity, economics, evidence precision, governance, and a renewed decision.
    A weak answer simply scales the pilot point estimate.

## 15. Key terms

- [Analogy](#3-the-core-mental-model): comparison used to generate a question, not authorize a copy.
- [Plan eligibility](#3-the-core-mental-model): locally supported readiness for a specified action, subject to approval.
- [Decision owner](#5-the-mechanics-step-by-step): accountable person for the choice and consequences.
- [Decision variable](#5-the-mechanics-step-by-step): controllable quantity or action under consideration.
- [Alternative](#5-the-mechanics-step-by-step): feasible option against which the proposal is compared.
- [Source case](#4-definitions-and-vocabulary): account of an intervention in its original context.
- [Destination](#4-definitions-and-vocabulary): operation considering an action informed by the source.
- [Product economics](#5-the-mechanics-step-by-step): local revenue and cost consequences of decisions.
- [Contribution margin](#4-definitions-and-vocabulary): revenue less specified variable costs.
- [Avoidable cost](#4-definitions-and-vocabulary): cost that changes under the decision.
- [Stockout](#4-definitions-and-vocabulary): inability to satisfy demand from available inventory.
- [Spoilage](#4-definitions-and-vocabulary): inventory made unsaleable through expiry or damage.
- [Data grain](#4-definitions-and-vocabulary): entity and time resolution of a record.
- [Latency](#4-definitions-and-vocabulary): delay between an event and data availability.
- [Decision-time snapshot](#8-data-standards-and-regulatory-context): information available at the actual decision time.
- [Censored demand](#5-the-mechanics-step-by-step): observed sales limited by availability.
- [Constraint](#5-the-mechanics-step-by-step): limit on the feasible action set.
- [Hard gate](#3-the-core-mental-model): condition that cannot be averaged away.
- [Evidence grade](#5-the-mechanics-step-by-step): course shorthand for the type and typical strength of a source.
- [Internal validity](#4-definitions-and-vocabulary): credibility of attribution in the original setting.
- [External validity](#4-definitions-and-vocabulary): applicability to a different setting.
- [Counterfactual](#5-the-mechanics-step-by-step): outcome without the intervention.
- [Selection bias](#10-failure-modes-and-red-flags): distortion from which cases or observations are available.
- [Incremental value](#6-worked-example): difference in value relative to the specified alternative.
- [Sensitivity analysis](#6-worked-example): recalculation under alternative assumptions.
- [Forecast](#6-worked-example): estimated future outcome under stated assumptions.
- [Target](#6-worked-example): desired result, distinct from a forecast.
- [Commitment](#6-worked-example): authorized obligation, distinct from a proposed plan.
- [Stop rule](#5-the-mechanics-step-by-step): pre-specified condition and response for pausing a test.
- [Adoption](#10-failure-modes-and-red-flags): extent to which recommendations change operational decisions.

## 16. Sources

The following are traceable starting points, not a claim that every source proves destination outcomes.
Access the current original document and preserve its date and relevant passage when using it in a memo.

1. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. “The M5 accuracy competition: Results, findings, and conclusions.” *International Journal of Forecasting*, 2022. [https://doi.org/10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013). Benchmark context; not proof of a local inventory outcome.
2. Zillow Group. *2021 Annual Report*, available through [SEC filings](https://investors.zillowgroup.com/financials/sec-filings/default.aspx). Company disclosure; distinguish reported financial facts from causal inference.
3. UPS. “UPS Deploys Purpose-Built Navigation for UPS Service Personnel.” [Company announcement](https://about.ups.com/us/en/newsroom/press-releases/innovation-driven/ups-deploys-purpose-built-navigation-for-ups-service-personnel.html). Treat outcome figures as company claims.
4. GS1. [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin). Identification standard; does not establish local data quality.
5. National Institute of Standards and Technology. [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework). Voluntary risk-management guidance, not a legal clearance.
6. U.S. Food and Drug Administration. [Food Safety Modernization Act](https://www.fda.gov/food/food-safety-modernization-act-fsma). Regulatory program overview; check current rules and applicability.
7. European Commission. [Data protection](https://commission.europa.eu/law/law-topic/data-protection_en). Overview of the EU data-protection framework; applicability requires context-specific review.
8. Hernán, M. A., and Robins, J. M. *Causal Inference: What If*. Chapman & Hall/CRC, 2020. [Author-hosted book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/). Reference for counterfactuals and causal study design.
9. Rothwell, P. M. “External validity of randomised controlled trials: ‘to whom do the results of this trial apply?’” *The Lancet*, 2005. [https://doi.org/10.1016/S0140-6736(04)17670-8](https://doi.org/10.1016/S0140-6736(04)17670-8). General methodological discussion; medical examples are not supply-chain evidence.

## 17. Links

- [Module 28 topic index](_index.md).
- [Case anatomy](01_case-anatomy.md): structure the source before testing transfer.
- [Evidence grades A–E](02_evidence-grades.md): grade individual claims, not brand names.
- [Case selection bias](04_selection-bias.md): examine which cases are absent.
- [Building a sector playbook](05_sector-playbooks.md): aggregate multiple cases without erasing their limits.
- [Module 28 survey and Lab 28.2](../../../course/28_domain_case_library.md): apply the transfer test and submit a memo.
