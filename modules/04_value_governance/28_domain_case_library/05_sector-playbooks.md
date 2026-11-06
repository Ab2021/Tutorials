# Building a Sector Playbook

> Module 28 — Domain Case Library and Sector Playbooks  
> Topic 05 of 10  
> Estimated study time: 3–4 hours  
> Prerequisites: Module 28 topics 01–03; forecasting and value measurement basics  
> Version 2026.5

## 1. Why this topic exists

A sector playbook converts a collection of stories into a disciplined way to make a local operating decision.
A grocery replenishment team, a manufacturer scheduling components, and a vaccine distributor allocating scarce doses all face uncertainty.
Their objectives and constraints are not interchangeable.
A method that reduces grocery waste may be unacceptable for an item whose temperature history is missing.
A recommendation that raises utilization in a factory may compromise service if it consumes the capacity needed for urgent orders.

The playbook makes the differences explicit before a team selects an algorithm or cites a successful case.
It names the decision and its owner, alternatives, data available at decision time, operating rules, evidence, economics, risks, and measurement design.
It is not a catalogue of vendors or a guarantee that a published outcome will transfer.
It is a working contract between decision makers, analysts, risk owners, and frontline operators.

A useful sector label narrows the search; it does not determine the answer.
A chilled-food distributor and a grocery store share perishability but differ in order authority and customer visibility.
A pharmaceutical distributor and a food distributor share lot tracking but face different legal obligations and release procedures.
The unit of analysis is therefore a named decision in a defined operating context, not an entire industry.

The person who owns the decision must be able to accept, override, or reject a recommendation.
A data scientist may calculate quantities but cannot silently authorize a purchase, release a product, or change a safety constraint.
The owner needs a feasible alternative, a record of exceptions, and a way to stop a harmful policy.
Finance needs a measurable counterfactual and non-overlapping costs.
Legal and quality specialists need a jurisdiction- and product-specific applicability assessment.

A sector playbook changes as evidence changes.
Version it when a product, market, system, policy, regulation, or evaluation result changes materially.
A stale playbook that says only “use machine learning for forecasting” is not an operating control.

## 2. Learning outcomes

After this topic you can:

1. **Frame** a decision with an owner, scope, cadence, alternatives, and decision deadline; exercise 1 checks the framing.
2. **Map** sector-specific data and point-in-time availability to the decision; exercise 2 checks the data contract.
3. **Select** a baseline and a method consistent with the operational constraint; exercises 3 and 4 check the choice.
4. **Calculate** incremental value and distinguish a forecast, plan, target, and commitment; exercises 4 and 5 check the calculation.
5. **Evaluate** case evidence and transferability without treating correlation or a vendor claim as causal proof; exercise 6 checks the judgment.
6. **Design** measurement, guardrails, ownership, and stop rules for a bounded pilot; exercise 7 checks the design.
7. **Explain** a regulatory applicability question and escalation path without presenting the lesson as legal advice; exercise 8 checks the distinction.
8. **Maintain** a playbook with decision logs, version triggers, and review responsibilities; exercise 9 checks the lifecycle.

The outcomes are observable because each requires a concrete artifact or calculation.
A persuasive description of an industry is not a substitute for specifying who acts and how success will be measured.

## 3. The core mental model

```text
Sector context and actual decision
        |
        v
Owner, objective, alternatives, cadence, safety and legal gates
        |
        v
Point-in-time data -> baseline -> feasible candidate policy
        |                              |
        |                       override and fallback
        +---------------+--------------+
                        v
          local comparison and guardrails
                        |
             measured incremental value
                        |
       adopt / change / stop / investigate
                        |
          versioned evidence and ownership
```

A playbook is a decision system rather than a prediction recipe.
The sector context explains what can go wrong and which constraints are likely to matter.
The decision statement specifies what can actually be changed.
A decision owner approves the action; data and model owners support it.
Safety, legal, quality, and contractual requirements are gates, not terms that can automatically be traded away for profit.

Data describe the state of the world only imperfectly.
A sales record can reflect unavailable stock rather than true demand.
An arrival timestamp can describe a recorded event rather than physical arrival.
A model trained on later-corrected data can look more accurate than one that actually ran at the time.
The playbook therefore records event time, availability time, unit, coverage, and known missingness.

A baseline is the real feasible alternative, including the existing human process.
A candidate has to fit the same constraints and be executable at the same decision time.
A model may improve prediction but not action when pack sizes, minimum orders, production changeovers, or delivery cutoffs prevent the decision from changing.
Measure both intermediate model performance and the final decision outcome.

The loop ends with an explicit decision, not merely a dashboard.
If the guardrail fails, the owner stops or changes the policy according to a pre-agreed rule.
If evidence is inconclusive, the owner can extend a bounded test rather than declare success.
The playbook records the outcome and the next review date.

### Four different statements

- **Forecast:** an estimate of future demand or another uncertain quantity, with a time horizon and uncertainty.
- **Plan:** a proposed set of actions subject to capacity, inventory, and other constraints.
- **Target:** a desired result used for management, not an estimate that it will occur.
- **Commitment:** an authorized promise or obligation to act, spend, or deliver.

A forecast of 980 units next week is not a commitment to sell 980 units.
A plan to order 900 units can be rejected by an authorized buyer.
A target of 98% fill rate is not evidence that the operation will achieve it.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Sector playbook | Versioned decision and evidence guide for an operating context | Treating a sector label as a causal model |
| Decision | Choice among feasible actions at a stated time | Treating a forecast as an action |
| Decision owner | Person accountable for authorizing the choice | Assuming the model builder owns the business risk |
| Operator | Person who executes or overrides an action | Excluding the operator from design |
| Escalation owner | Person authorized to resolve an exception | Escalating without a deadline or fallback |
| Decision cadence | Frequency and deadline, such as daily at 14:00 local time | Training at one cadence and acting at another |
| Scope | Products, locations, customers, dates, and exclusions | Generalizing from a favorable subset |
| Baseline | Feasible comparator available at the same decision time | Comparing with an obsolete or impossible process |
| Policy | Rule that maps information to an action | Equating a fitted prediction with a policy |
| Constraint | Condition a feasible action must satisfy | Treating an absolute safety gate as a soft preference |
| Guardrail | Predefined metric or event limiting pilot harm | Choosing a threshold after seeing outcomes |
| Forecast | Estimate of an uncertain future quantity per horizon | Treating it as a target or purchase order |
| Plan | Proposed actions and timing | Treating an unapproved plan as a commitment |
| Target | Desired outcome in specified units and period | Calling the target an unbiased estimate |
| Commitment | Authorized obligation or action | Treating a scenario as approved spending |
| Observation window | Dates from which evidence is collected | Choosing only dates that flatter the candidate |
| Counterfactual | Outcome that would occur under the alternative | Claiming the unobserved outcome is known |
| Incremental effect | Difference attributable to candidate versus comparator | Subtracting unrelated before-and-after totals |
| Unit economics | Revenue, cost, or contribution per relevant unit | Mixing gross revenue and profit |
| Contribution margin | Revenue less specified variable costs, currency/unit | Counting a sale's full revenue as incremental profit |
| Holding cost | Cost of carrying inventory, currency/unit/time | Counting the purchase cost again as holding cost |
| Waste | Units or value rendered unusable | Counting returns as waste without disposition |
| Fill rate | Fulfilled demand divided by eligible demand | Mixing order fill with unit fill |
| Stockout | Unavailability at the decision-relevant location and time | Treating zero recorded sales as zero demand |
| Lead time | Duration from defined release event to defined receipt event | Mixing supplier processing and transport without boundaries |
| Shelf life | Permitted usable duration under stated conditions | Assuming all received lots have equal remaining life |
| Lot | Defined quantity linked to a production or handling history | Treating a product identifier as a lot identifier |
| Traceability | Ability to link relevant product and movement records | Assuming a barcode alone proves complete traceability |
| Point-in-time data | Information available to the decision maker at the decision timestamp | Using retrospectively corrected values without noting leakage |
| Leakage | Use of information unavailable at decision time | Calling a hindsight feature a forecast input |
| Censoring | Observation restricted by availability or measurement | Treating stockout sales as unconstrained demand |
| Granularity | Level such as item-location-day | Joining daily totals to hourly events without a rule |
| Evidence grade | Label describing source type and credibility | Assuming source type alone proves causal identification |
| Transfer test | Assessment of decision, economics, data, constraints, and evidence fit | Transferring a source's savings without local evidence |
| Randomization | Assignment by chance according to a defined design | Calling selection by convenience random |
| Spillover | Effect of treatment on a comparison unit | Assuming shared capacity creates no interference |
| Override | Authorized departure from a proposed action with a reason | Treating all overrides as model failure |
| Fallback | Executable alternative when inputs, system, or controls fail | Assuming a disabled model restores a safe process automatically |
| Data contract | Agreed definitions, delivery, quality, and ownership of data | Treating a schema as proof of accurate data |
| Traceability event | Recorded event linking product, location, time, and relevant identifiers | Assuming event capture equals verified physical custody |
| Applicability | Determination whether a rule governs a product, entity, and activity | Assuming all firms in a sector have identical obligations |
| Version | Identifiable snapshot of policy, assumptions, and evidence | Changing thresholds without recording when or why |

## 5. The mechanics, step by step

### Step 1 — Specify the decision

**Input:** a business problem, a sponsor, and a list of candidate operations.
**Output:** a short decision charter approved by the accountable owner.

Write the decision as a verb, object, location, and deadline.
For example: “Choose each store's next-day order quantity for chilled salads by 14:00 local time.”
Include the person or role that can authorize the resulting order.
Name the systems that carry the approved action into execution.

Record the feasible alternatives, including the current process.
An alternative must be operable with the information available at the same deadline.
A post-hoc rule that uses tomorrow's realized demand is not an executable alternative.
Identify the financial, service, safety, and operational consequences of a wrong action.

**Branch:** if nobody owns the action, stop before model selection.
A sponsor who controls the budget but cannot change a purchase order is not necessarily the operational owner.
If the decision is an occasional strategic design rather than a daily action, change the horizon and evidence requirements accordingly.

### Step 2 — Define the sector boundary

**Input:** product, market, actors, flow, and customers.
**Output:** a context card and explicit exclusions.

Draw the physical flow separately from the information flow.
A purchase order, physical shipment, financial invoice, and traceability event need not occur at the same time.
Record relevant temperature, shelf-life, substitution, capacity, and contractual properties.
Specify which legal entities own, transport, store, and release the product.

A sector's average characteristics are only starting hypotheses.
For example, a retailer with make-to-order products does not necessarily share the inventory dynamics of a grocery chain.
A producer making a single continuous product does not necessarily resemble a high-mix assembly plant.

**Branch:** split the playbook when two populations require materially different constraints, decision rights, or economic units.
Do not split solely because two items have different names if the same decision rules and control framework apply.

### Step 3 — Map stakeholders and authority

**Input:** decision charter, workflow, and risk inventory.
**Output:** named roles, rights, escalation deadlines, and fallback.

Use one accountable operational owner for the decision.
Assign a data owner for source definitions and delivery.
Assign an analytics owner for model and evaluation quality.
Assign a quality or safety owner for applicable product release and hold decisions.
Assign a finance owner for costs, margin, and benefit realization.
Assign a legal or compliance reviewer where applicable.

The operator needs authority to override within a documented scope.
An override reason should distinguish bad data, a capacity constraint, a safety hold, a local event, and an informed business judgment.
Repeated overrides can indicate a missing feature, incorrect constraint, or a rational exception.
They are not automatically evidence of resistance.

**Branch:** if a legal or safety gate is unresolved, do not infer approval from a profitable scenario.
Route the issue to the authorized specialist, record the deadline, and use the safe fallback until resolved.

### Step 4 — Translate objectives into measures

**Input:** owner objectives, operating constraints, and accounting definitions.
**Output:** one primary outcome, diagnostic measures, guardrails, and decision thresholds.

Choose a primary outcome aligned to the action.
A replenishment policy might be evaluated on incremental contribution after waste and incremental operating cost.
Report fill rate and spoilage separately so the mechanism and trade-off remain visible.
Define the numerator, denominator, unit, reporting window, eligible population, and source for every measure.

A unit fill rate is filled units divided by eligible requested units.
An order fill rate is fully filled orders divided by eligible orders.
These are different metrics and should not share an unlabeled percentage.
Decide how cancellations, returns, substitutions, stockouts, and late records enter each measure.

Set guardrails before assigning pilot units.
Examples include a maximum product-quality incident rate, an inventory accuracy minimum, and a service floor.
For a rare safety event, a zero count in a small pilot does not establish zero risk.
The quality owner may require independent controls rather than a statistical threshold alone.

**Branch:** where hard safety or legal conditions are violated, stop regardless of the aggregate value estimate.
Where only a business target is missed, examine uncertainty and a pre-specified decision rule before deciding whether to extend or stop.

### Step 5 — Inventory the data and its timing

**Input:** data sources, system access, event definitions, and decision deadline.
**Output:** a point-in-time data contract and data-quality assessment.

For every source record the owner, coverage, grain, identifiers, timezone, refresh cadence, and availability delay.
Separate event time from the time the decision system can observe the event.
A delivery scan from 13:30 that reaches the system at 15:00 cannot support a 14:00 decision on that day.
Record corrections and late-arriving events so historical reconstruction does not gain hindsight.

Join demand, inventory, orders, receipts, and item attributes at a defensible grain.
Check whether identifiers are stable across supplier, warehouse, store, and finance systems.
Document how a product replacement or changed pack size affects history.
Check duplicates, missing records, negative inventory, impossible time order, and unexpected unit conversions.

Inventory records should be compared with physical counts on a representative sample.
Sales are not necessarily demand when the product was unavailable.
A sold-out shelf can generate a low sales record precisely when customers wanted more.
Treat imputed demand as an assumption and test sensitivity rather than silently relabeling it as observed demand.

**Branch:** if critical inputs are unavailable at the deadline, use an authorized fallback or change the decision cadence.
Do not train on revised values and claim the result is deployable without a point-in-time replay.

### Step 6 — Establish an executable baseline

**Input:** current decision logs and operating rules.
**Output:** documented baseline actions and historical performance.

Capture the actual current workflow, including human adjustments and exceptions.
Where current decisions were not logged, interview operators and reconstruct only what the records support.
Separate a historical descriptive baseline from an experimentally assigned control policy.
A historical average may be useful for sizing but is not automatically a causal counterfactual.

A simple baseline can be a recent average, seasonal naive forecast, fixed reorder point, or existing planning rule.
Choose a baseline that a competent operator could execute under the same information and constraints.
Record how the baseline handles missing data, promotions, pack sizes, and unavailable suppliers.

**Branch:** if the baseline is not feasible at the candidate's decision cadence, repair the comparison.
Do not claim improvement against a deliberately weak straw man.

### Step 7 — Select a method for the decision

**Input:** baseline, data maturity, constraints, cost of error, and execution capacity.
**Output:** a method card with inputs, outputs, assumptions, and fallback.

Forecasting estimates uncertain future quantities; optimization chooses among feasible actions against an objective.
Simulation can examine performance under uncertain demand and lead time.
Rules can be preferable where data are sparse, explanations are essential, or constraints dominate.
A sophisticated predictor cannot repair missing decision rights or an unsafe release process.

For retail replenishment, compare a seasonal baseline with a constrained order policy.
For manufacturing, account for sequence-dependent changeovers, material availability, and bottleneck capacity.
For logistics, route plans must respect vehicle capacity, time windows, driver rules, and travel-time uncertainty.
For pharmaceutical allocation, approved quality status and traceability are gates before any prioritization rule.
For humanitarian allocation, explicit need, equity, access, and consent considerations may dominate commercial margin.

The method card should state when a human may override and when a recommendation is suppressed.
Treat an infeasible optimization result as an exception requiring action, not as permission to relax a safety constraint.

**Branch:** use a simpler method if the candidate cannot demonstrate reliable input availability or actionable improvement.
Escalate any objective that implicitly prices a non-negotiable legal or safety requirement.

### Step 8 — Evaluate sources and transfer

**Input:** candidate cases, source material, and local context card.
**Output:** evidence ledger with source, claim, grade, transfer judgment, and open questions.

Record who made each claim, when, about which population, against what comparator, and with which outcome measure.
The module's evidence taxonomy uses A for peer-reviewed or audited evidence, B for regulatory or official statistics, C for company claims, D for vendor marketing, and E for press or anecdote.
These labels describe source type, not a universal ranking of causal validity.
An official regulation can establish a legal text without establishing a company's realized savings.
An audited financial figure can be accurate without proving which intervention caused it.

Apply the five transfer questions: same decision, similar product economics, similar data environment, similar operating constraint, and evidence sufficient for the proposed risk.
For each record yes, no, or unknown and the reason.
Three no answers make the case an analogy rather than a plan under the module rule.
A single hard safety, legal, capacity, or evidence failure can still block action.
Unknown is not yes.

**Branch:** use a source as an analogy when it suggests a mechanism but does not establish a local effect.
Use a local plan only after feasible operation, material assumptions, evidence, and authorization gates have been addressed.

### Step 9 — Design the local comparison

**Input:** eligible population, baseline, candidate policy, risk controls, and measurement definitions.
**Output:** a pre-specified test protocol and analysis plan.

Prefer randomized assignment where feasible and ethical.
Choose the unit of assignment to limit contamination: stores rather than individual orders when staff or inventory are shared, for example.
Document stratification, assignment, exclusions, treatment start, and analysis population.
Measure compliance and overrides, but preserve an intention-to-treat comparison when assignment is randomized.

If randomization is not feasible, consider matched groups, phased rollout, or an interrupted time series with explicit assumptions.
Record contemporaneous promotions, supplier disruptions, seasonality, and changes in customer mix.
A before-and-after comparison without an appropriate counterfactual is descriptive and vulnerable to confounding.
The analysis should report uncertainty and distinguish a null result from insufficient precision.

**Branch:** if interference across units is material, redesign clusters or explicitly estimate the shared-capacity effect.
If the intervention creates unacceptable risk, use a simulation, retrospective replay, or a narrower supervised trial instead of live randomization.

### Step 10 — Convert outcomes into economics

**Input:** comparable outcome counts, agreed unit values, and incremental implementation costs.
**Output:** a benefit bridge with units, period, assumptions, uncertainty, and finance sign-off.

Start with changes in outcome units, such as additional fulfilled units or avoided waste units.
Value incremental sales at contribution margin, not gross revenue, when variable costs are excluded from the margin.
Subtract new operating costs, technology costs, and other incremental costs not already included.
Avoid counting a reduction in waste both as recovered purchase cost and as an additional saving unless they represent distinct economic flows.

Separate one-time setup costs from recurring costs and specify the time horizon.
Do not annualize a short or seasonal test without stating the representativeness assumption.
If increased sales displace another product, account for the net category effect where relevant.
Include downside scenarios and a decision rule for insufficient benefit.

**Branch:** if the effect estimate is not causally identified, call the result an observed difference or scenario, not realized attributable savings.
If finance cannot validate unit costs, report the units and a range pending reconciliation.

### Step 11 — Operationalize the decision

**Input:** approved candidate, operating procedures, controls, and named owners.
**Output:** runbook with action deadline, escalation, fallback, and monitoring.

Specify the data arrival deadline, recommendation time, approval deadline, and execution confirmation.
Show operators the recommendation, the baseline, the material constraints, and the reason for an exception.
Log proposed, approved, overridden, and executed actions separately.
Record the version of data, model, policy, and constraints used for each decision.

A fallback must be executable even when the candidate system is unavailable.
Specify who activates it, which data it needs, and how the organization returns to the normal process.
A quality hold must remain in force during fallback.
A system outage is not permission to bypass traceability or release controls.

**Branch:** if the team cannot recover the baseline process within the required time, do not launch.
If execution differs materially from recommendations, investigate implementation before attributing outcomes to the model.

### Step 12 — Review and retire

**Input:** monitoring reports, incidents, changed context, and evaluation results.
**Output:** versioned decision to continue, change, stop, or re-evaluate.

Review data quality and operational exceptions at the cadence of the decision.
Review benefit, uncertainty, and representativeness over an appropriate business cycle.
Trigger review when product mix, supplier lead time, regulation, model, owner, or data feed changes materially.
Document who approved changes and the effective dates.

Retire a policy that no longer has an owner or a feasible fallback.
Preserve the evidence and decision record under the organization's retention rules.
The playbook is complete only when its continuing ownership is explicit.

### A compact playbook record

| Field | Minimum record |
|---|---|
| Decision | Verb, product, location, deadline, and choice |
| Accountable owner | Role and named delegate |
| Baseline | Executable alternative and information cutoff |
| Candidate | Method, constraints, actions, and fallback |
| Data | Grain, availability, quality, lineage, and owner |
| Evidence | Source, claim, comparator, grade, and limits |
| Transfer | Five questions, hard gates, and unresolved unknowns |
| Regulation | Jurisdiction, product, activity, reviewer, and date |
| Measurement | Primary metric, guardrails, assignment, and analysis window |
| Economics | Unit values, costs, horizon, and finance approval |
| Operations | Override rights, escalation, execution, and audit trail |
| Lifecycle | Version, effective date, review trigger, and retirement owner |

### Sector decision cards

The following cards are starting templates, not evidence of results.
Each is deliberately bounded to a particular decision and should be rewritten for the local operation.
The arithmetic examples are synthetic and do not represent named organizations.

#### Retail and e-commerce: replenishment

**Decision:** Choose item-store order quantities before the supplier cutoff.
**Accountable owner:** The buyer or replenishment manager approves the purchase order.
**Data:** Point-in-time sales, available inventory, on-order stock, receipt timing, product attributes, and planned promotions.
**Method:** Compare an executable seasonal or recent-sales baseline with a constrained order policy.
**Constraints:** Pack sizes, supplier minimums, lead times, shelf life, store capacity, and quality holds.
**Outcome:** Incremental contribution after waste and operating cost, reported alongside unit fill rate.
**Evidence:** A public forecasting benchmark informs baseline design; it does not establish local value.
**Failure:** A zero-sale day during a stockout is incorrectly treated as zero customer demand.
**Fallback:** The buyer uses the approved manual ordering rule and retains quality holds.

Suppose a store expects 83 units of demand over its replenishment horizon and holds 27 usable units.
If its policy adds 16 units of safety stock, its unconstrained net requirement is 83 + 16 − 27 = 72 units.
With 12 units per case and no other constraints, the proposed order is 6 cases.
This is a proposed plan, not a purchase commitment.
If 9 of the 27 units are on quality hold, the usable quantity is only 18 units.
The revised requirement is 81 units, rounded up to 7 cases under the assumed pack rule.
The quality owner, not the model, decides whether the held units may be released.

#### Retail and e-commerce: returns

**Decision:** Route each returned item to restock, repair, resale, recycling, or disposal.
**Accountable owner:** The reverse-logistics manager owns disposition within quality and legal controls.
**Data:** Return reason, item condition, elapsed time, location, handling cost, resale value, and inspection results.
**Method:** Use a transparent disposition rule or a validated probability model to estimate recoverable value.
**Constraints:** Product safety, hygiene, customer rights, inventory accuracy, and facility capacity.
**Outcome:** Net recovery value per eligible return and time to disposition.
**Evidence:** Company descriptions can suggest workflows; they do not prove a particular recovery rate for another retailer.
**Failure:** The system treats a customer-selected return reason as a verified physical condition.
**Fallback:** Route uncertain or safety-sensitive items to inspection.

Suppose inspection and restocking cost USD 3.20 per unit and an inspected item is expected to recover USD 11.50.
The simple expected net value is USD 8.30 per unit before transport and other omitted costs.
If the recoverable value is uncertain, evaluate the distribution rather than treating USD 11.50 as guaranteed.
If a product cannot lawfully or safely be resold, the estimated positive value does not authorize resale.
The returns owner checks the disposition; the quality owner controls relevant safety exceptions.

#### Consumer goods: promotion and replenishment

**Decision:** Set a promotion-period supply plan and replenishment order.
**Accountable owner:** The supply planner owns the proposed allocation; commercial and finance owners approve relevant assumptions.
**Data:** Baseline demand, promotion calendar, prices, shipment history, customer orders, inventory, and capacity.
**Method:** Estimate demand under the promotion and compare feasible supply plans with a no-change baseline.
**Constraints:** Production slots, retailer commitments, minimum order quantities, and promotion lead times.
**Outcome:** Incremental category contribution, service, and post-promotion inventory.
**Evidence:** A reported uplift from another promotion is a hypothesis, not an estimate of this campaign's causal effect.
**Failure:** Shipments into a retailer are mistaken for incremental sales to consumers.
**Fallback:** Apply the current approved planning process when the promotion feed is incomplete.

Suppose a promotion is forecast to add 1,800 units, with USD 1.10 incremental contribution per unit.
The gross contribution scenario is USD 1,980 before the promotion's incremental fixed and operating costs.
If incremental campaign and handling costs are USD 1,250, the scenario net is USD 730.
If half of the additional units displace equally profitable products, the contribution scenario falls to USD 990 before those costs.
The revised net is a loss of USD 260 under that displacement assumption.
These are scenarios, not proof that the campaign caused the stated volume.

#### Manufacturing: production scheduling

**Decision:** Choose the sequence and timing of work orders on a constrained line.
**Accountable owner:** The production scheduler issues the schedule; quality controls release and holds.
**Data:** Work orders, available materials, bill of materials, machine availability, processing rates, and cleaning times.
**Method:** Compare the current sequencing rule with constrained optimization or simulation.
**Constraints:** Maintenance, labor, material availability, changeovers, quality status, and due dates.
**Outcome:** On-time eligible units, overtime cost, changeover time, and schedule stability.
**Evidence:** A similar plant's published result is not transferable without comparing process and capacity.
**Failure:** The algorithm schedules material that is physically present but not quality-released.
**Fallback:** The scheduler uses the approved manual sequence while preserving the hold.

Suppose product A requires 3 hours and product B requires 4 hours of processing.
Changing from A to B requires 1.5 hours of cleaning; changing from B to A requires 0.5 hours.
Ignoring all other constraints, A then B takes 8.5 hours and B then A takes 7.5 hours.
The one-hour difference does not by itself determine the preferred sequence.
An earlier customer due date, material release, or cleaning validation may make the shorter sequence infeasible.
Record both the planned and actual sequence so that replanning and overruns are visible.

#### Industrial spare parts: stocking

**Decision:** Set a stocking level for a critical replacement component at a service depot.
**Accountable owner:** The maintenance or service-parts manager approves the stocking policy.
**Data:** Failure events, installed base, repair lead times, repairable returns, inventory state, and supplier reliability.
**Method:** Compare an intermittent-demand baseline with a risk-based stocking rule or simulation.
**Constraints:** Long lead time, obsolescence, compatibility, repair capacity, and downtime consequences.
**Outcome:** Equipment downtime, urgent shipment cost, availability, and total inventory carrying cost.
**Evidence:** A benchmark for high-volume consumables may not transfer to a low-frequency critical part.
**Failure:** A long history of zero requests is taken as evidence that a critical failure is impossible.
**Fallback:** Escalate a potential shortage to the service owner and follow the agreed emergency sourcing process.

Suppose holding one spare costs USD 900 per year.
The business estimates that one eligible shortage would cost USD 12,000 in downtime and emergency handling.
If the spare prevents an expected 0.10 shortages per year, the illustrative avoided cost is USD 1,200 per year.
The simple net scenario is USD 300 per year before purchase, obsolescence, and other omitted costs.
The 0.10 value is an assumption about an uncertain event rate, not an observed guarantee.
Review whether a common spare can serve several assets and whether failures are correlated.

#### Logistics: daily dispatch

**Decision:** Assign deliveries to vehicles and sequence stops for the next shift.
**Accountable owner:** The dispatcher approves the route; safety and labor owners govern relevant constraints.
**Data:** Orders, locations, service windows, driver availability, vehicle capacity, travel times, and road restrictions.
**Method:** Compare the current dispatch process with a feasible vehicle-routing policy.
**Constraints:** Driver working time, vehicle capacity, customer windows, vehicle-product compatibility, and safety.
**Outcome:** Cost per completed eligible stop, on-time delivery, overtime, and failed-delivery rate.
**Evidence:** A carrier's published description can motivate a pilot, not establish the destination's savings.
**Failure:** Historical travel times omit service time or recurring congestion.
**Fallback:** The dispatcher applies the approved manual route process and communicates changes to drivers.

Suppose a candidate saves 24 vehicle-kilometers per day across a pilot.
At an assumed avoidable variable cost of USD 0.70 per vehicle-kilometer, the distance-related scenario is USD 16.80 per day.
This does not include driver time, fixed vehicle cost, failed deliveries, or software expense.
If the shorter route creates 30 minutes of overtime costing USD 22, the net of these two items is a USD 5.20 loss per day.
Do not count fixed vehicle depreciation as avoided unless the decision actually changes that expense.
Verify the route is executable and safe before interpreting a cost difference.

#### Pharmaceuticals: allocation and release

**Decision:** Allocate eligible, released stock among authorized destinations during a shortage.
**Accountable owner:** The allocation lead decides quantities; the quality function controls release and investigation.
**Data:** Lot or serial identifiers where applicable, status, expiry, verified trading partners, demand, and available transport.
**Method:** Apply a documented allocation rule after legal, quality, and product eligibility gates.
**Constraints:** Product-specific storage, traceability, verified status, expiry, contracts, and jurisdiction-specific obligations.
**Outcome:** Eligible orders filled on time, shortage exposure, traceability completeness, and exception resolution time.
**Evidence:** Official guidance establishes questions about applicable requirements, not vendor effectiveness.
**Failure:** Stock on hold is treated as available because a quantity field is positive.
**Fallback:** Quarantine the affected record or stock and escalate to authorized quality personnel.

Suppose 1,000 doses are physically recorded, of which 120 doses are on hold and 80 are reserved.
If the held and reserved quantities are disjoint, the maximum available before other constraints is 800 doses.
If the categories overlap, subtracting both without reconciliation understates availability.
No dose should be allocated merely because it is in the arithmetic remainder.
Check release, expiry, trading-partner eligibility, and the applicable product-specific handling rules first.
An allocation recommendation is not an authorization to release a held lot.

#### Humanitarian response: scarce supplies

**Decision:** Allocate available relief kits across locations over a stated response window.
**Accountable owner:** The response lead owns allocation with input from protection and local community representatives.
**Data:** Estimated need, accessibility, stock, transport capacity, population coverage, and uncertainty.
**Method:** Compare transparent needs-based alternatives and test sensitivity to uncertain access and population estimates.
**Constraints:** Safety, impartiality, funding conditions, physical access, and protection of personal information.
**Outcome:** Coverage of eligible need, time to reach people, unmet need, and distribution complaints.
**Evidence:** Prior responses provide context but cannot establish current needs or local acceptance.
**Failure:** Historical shipment volume is substituted for need, reproducing past access gaps.
**Fallback:** Escalate inaccessible or unsafe locations and document the alternative response.

Suppose two locations have estimated eligible needs of 600 and 400 kits and 500 kits are available.
A proportional starting proposal is 300 kits and 200 kits, respectively.
If only 150 kits can reach the second location in the response window, the proposal is not feasible as written.
The remaining 50 kits require an explicit decision about storage, alternate delivery, or reallocation.
Record that population estimates, access constraints, and equity impacts may be uncertain.
The allocation owner consults affected communities and records the rationale; the calculation does not decide alone.

## 6. Worked example

### Context and decision

This entire example is **synthetic**; the organization, numbers, and outcomes are invented for instruction.
They are not evidence about a real retailer or a claim about expected performance.

A regional grocer orders chilled prepared meals for 24 stores.
The buyer decides next-day order quantities at 14:00 local time every day.
The product has short remaining shelf life, and unsold units can become waste.
The buyer owns purchase orders; store managers can report local exceptions.
The quality manager controls product holds and may stop a sale regardless of a recommendation.

The existing policy uses a recent sales average adjusted manually for local events.
The candidate uses a demand forecast plus a constrained ordering rule.
Both policies use the same supplier minimum, pack size, approved shelf-life rule, and data cutoff.
The candidate cannot order an item on quality hold.

The primary outcome is incremental contribution after waste and incremental operating costs over a four-week pilot.
The service guardrail is unit fill rate of at least 95% in the candidate group, evaluated using the defined eligible demand denominator.
The quality guardrail is no bypass of a hold; any bypass stops the pilot immediately.
This service threshold is a local hypothetical decision rule, not a legal requirement.

### Forecast, plan, target, and commitment

For one store and one day the candidate forecasts 116 units of demand.
The proposed plan is an order of 108 units after considering usable stock, pack size, and delivery constraints.
The management target is at least 95% unit fill rate across the pilot group.
The commitment is the buyer-approved purchase order actually submitted to the supplier.
None of these four values is interchangeable with another.

### Design and data

The grocer pairs 24 stores on size and recent demand, then randomly assigns one store per pair to each policy.
Each group contains 12 stores for four weeks.
Staff continue to record approved overrides in both groups.
The analysis uses assigned stores, not only stores that followed the candidate recommendation.
Shared distribution capacity is monitored because interference between stores could distort the comparison.

The primary data are requested units, fulfilled units, waste units, inventory snapshots, approved orders, and receipts.
Records are captured at item-store-day grain.
The team freezes the 14:00 data snapshot so that later corrections do not leak into the policy.
A physical count sample checks inventory reliability before launch.
The demand denominator is independently captured customer requests; if it were inferred solely from sales during stockouts, fill rate would be biased.

### Observed synthetic results

| Measure, four-week pilot | Baseline group | Candidate group |
|---|---:|---:|
| Assigned stores | 12 stores | 12 stores |
| Eligible requested units | 24,000 units | 24,000 units |
| Fulfilled units | 22,800 units | 23,200 units |
| Waste units | 1,200 units | 1,000 units |
| Incremental candidate operating cost | USD 0 | USD 600 |
| Hold bypasses | 0 events | 0 events |

Baseline unit fill rate = 22,800 / 24,000 = 95.00%.
Candidate unit fill rate = 23,200 / 24,000 = 96.67%, rounded to two decimal places.
The observed difference is 1.67 percentage points, not 1.67% relative improvement.
The candidate group meets the hypothetical 95% service floor.
The zero observed hold bypasses satisfy the stated event rule but do not establish that future bypass risk is zero.

Assume each additional fulfilled unit earns USD 4.20 of incremental contribution after its variable selling cost.
Assume each avoided waste unit saves USD 2.40 in disposal and product cost not already included in that contribution.
These values are scenario assumptions and require finance validation; there is no real company data here.
Assume the two group totals can be compared directly because their eligible demand totals and store counts are equal.
Even then, random variation and store heterogeneity remain.

Additional fulfilled units = 23,200 − 22,800 = 400 units per four weeks.
Avoided waste = 1,200 − 1,000 = 200 units per four weeks.
Incremental contribution from fulfilled units = 400 × USD 4.20 = USD 1,680 per four weeks.
Avoided waste value = 200 × USD 2.40 = USD 480 per four weeks.
Incremental operating cost = USD 600 per four weeks.
Observed difference in economic value = USD 1,680 + USD 480 − USD 600 = **USD 1,560 per four weeks**.

The calculation assumes no overlap between contribution and waste value and no material displacement of other products.
It also assumes the groups face comparable demand, and that the candidate's operating cost is incremental and complete.
If additional meals displace an equally profitable substitute, the sales contribution may be overstated.
If the USD 2.40 waste value already appears in the contribution calculation, add only the non-overlapping portion.
If the candidate consumed scarce distribution capacity needed by baseline stores, the group difference may overstate system-wide value.

### Uncertainty and decision

The table does not provide store-level outcomes or assignment variance.
Therefore it cannot support a confidence interval or a precise causal effect estimate.
A randomized design supports causal interpretation only if assignment, adherence analysis, missingness, and interference are handled correctly.
The aggregate USD 1,560 is an observed pilot-group difference under stated unit-value assumptions, not a verified annual saving.

The buyer and finance owner should review store-pair differences and cost records.
The data owner should reconcile late records and stockout demand capture.
The quality manager should review the hold audit.
The business owner can continue the controlled pilot if the pre-agreed protocol allows it; otherwise the owner returns to the baseline pending more evidence.
A full rollout or annualized benefit commitment is not justified solely by this synthetic table.

### Transfer notes

A comparable grocery case could suggest that perishable replenishment deserves investigation.
Before reuse, compare the order decision, margin and waste economics, data available at 14:00, supplier constraints, and strength of source evidence.
A case involving shelf-stable products or a vendor-reported sales uplift may be useful as an analogy but does not prove this policy's effect.
An unknown quality requirement or a missing lot hold cannot be resolved by a favorable value calculation.

## 7. Code, query, or calculation

The following standard-library Python reproduces the synthetic arithmetic.
It is a transparent calculation, not a statistical test or a production ordering system.
Run it with Python 3; no external packages or files are needed.
Inputs are group-level counts for the same four-week window.
The code checks basic count consistency and applies the unit values stated in section 6.
It does not estimate demand, model uncertainty, validate accounting, or authorize a purchase.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    requested: int
    fulfilled: int
    waste: int

    def validate(self) -> None:
        if self.requested <= 0:
            raise ValueError("requested units must be positive")
        if not 0 <= self.fulfilled <= self.requested:
            raise ValueError("fulfilled units must be within requested units")
        if self.waste < 0:
            raise ValueError("waste units cannot be negative")

    @property
    def fill_rate(self) -> float:
        return self.fulfilled / self.requested


def evaluate(
    baseline: Group,
    candidate: Group,
    contribution_per_unit: float,
    waste_cost_per_unit: float,
    incremental_operating_cost: float,
) -> dict[str, float]:
    baseline.validate()
    candidate.validate()
    if baseline.requested != candidate.requested:
        raise ValueError("this simple comparison requires equal requested units")
    if min(contribution_per_unit, waste_cost_per_unit, incremental_operating_cost) < 0:
        raise ValueError("unit values and operating cost must be nonnegative")

    extra_fulfilled = candidate.fulfilled - baseline.fulfilled
    avoided_waste = baseline.waste - candidate.waste
    value = (
        extra_fulfilled * contribution_per_unit
        + avoided_waste * waste_cost_per_unit
        - incremental_operating_cost
    )
    return {
        "baseline_fill": baseline.fill_rate,
        "candidate_fill": candidate.fill_rate,
        "fill_difference_pp": 100 * (candidate.fill_rate - baseline.fill_rate),
        "extra_fulfilled": extra_fulfilled,
        "avoided_waste": avoided_waste,
        "value_usd": value,
    }


baseline = Group(requested=24000, fulfilled=22800, waste=1200)
candidate = Group(requested=24000, fulfilled=23200, waste=1000)
result = evaluate(
    baseline,
    candidate,
    contribution_per_unit=4.20,
    waste_cost_per_unit=2.40,
    incremental_operating_cost=600.00,
)
print(f"Baseline fill: {result['baseline_fill']:.2%}")
print(f"Candidate fill: {result['candidate_fill']:.2%}")
print(f"Fill difference: {result['fill_difference_pp']:.2f} percentage points")
print(f"Extra fulfilled: {result['extra_fulfilled']:.0f} units")
print(f"Avoided waste: {result['avoided_waste']:.0f} units")
print(f"Observed value difference: USD {result['value_usd']:,.2f} per 4 weeks")
```

Expected output:

```text
Baseline fill: 95.00%
Candidate fill: 96.67%
Fill difference: 1.67 percentage points
Extra fulfilled: 400 units
Avoided waste: 200 units
Observed value difference: USD 1,560.00 per 4 weeks
```

The equal-request check prevents interpreting unequal group volumes as directly comparable in this simplified calculation.
It does not make the groups exchangeable or remove shared-capacity interference.
For real studies, retain store-level assignment, cluster structure, missingness, and uncertainty.
Use decimal currency arithmetic and controlled rounding when reconciling financial postings.
A real deployment also needs independently validated data pipelines, access controls, logging, and operational approval.

## 8. Data, standards, and regulatory context

This section is educational, not legal, trade-compliance, or procurement advice.
Legal and quality specialists must determine applicability for the entity, product, activity, jurisdiction, and date.
Rules can change; consult the linked official sources and current implementing material before acting.
A standard can help implement a process without itself being a law.

### Cross-sector data controls

| Control | Practical question | Owner |
|---|---|---|
| Identity | Do item, location, supplier, and customer identifiers join consistently? | Master-data owner |
| Time | Was each input actually available at the decision cutoff? | Data engineering owner |
| Unit | Are cases, packs, eaches, kilograms, and currency converted consistently? | Data steward |
| State | Does inventory distinguish available, reserved, damaged, and held stock? | Inventory owner |
| Lineage | Can a decision be traced to source records and policy version? | Analytics owner |
| Quality | Are missingness, duplication, latency, and reconciliation monitored? | Data owner |
| Access | Are personnel and systems permitted to use the data? | Security and privacy owners |
| Retention | Are records kept and disposed of under applicable rules and policy? | Records owner |
| Execution | Did the approved recommendation become the actual action? | Operations owner |
| Exception | Are quality holds, overrides, and system outages logged? | Control owner |

Global Standards 1 (GS1) publishes identification and event-sharing standards.
A Global Trade Item Number (GTIN) identifies a trade item; lot or serial information may be needed to distinguish particular product instances.
Electronic Product Code Information Services (EPCIS) is a GS1 standard for sharing visibility-event information.
Its event framework can support traceability, but using EPCIS or a barcode does not by itself demonstrate regulatory compliance or accurate physical records.
See the GS1 General Specifications and EPCIS standard in section 16.

### Food

In the United States, the Food and Drug Administration (FDA) Food Safety Modernization Act (FSMA) food traceability rule establishes additional traceability recordkeeping requirements for entities and foods within its scope, subject to exemptions and other conditions.
The relevant regulation is 21 CFR Part 1, Subpart S; consult current FDA guidance and the regulation for applicability and compliance dates.
A sector playbook should ask whether the food appears on the Food Traceability List, what covered activities the entity performs, and what records must be available.
Do not infer that every food, firm, transaction, or location is covered.

For chilled foods, capture temperature excursions, timestamps, receipt and dispatch events, lot identity, and disposition where relevant to the control plan.
A forecast can recommend an order, but it cannot establish that an individual lot is safe to release.
The quality owner decides hold and release under the organization's validated procedures and applicable rules.

### Pharmaceuticals

The U.S. Drug Supply Chain Security Act (DSCSA) establishes requirements for tracing certain prescription drugs through the supply chain.
FDA guidance describes enhanced drug distribution security requirements and implementation policies; scope and current deadlines must be checked against current FDA materials.
A playbook should map product identifiers, trading partners, transaction information, verification, exceptions, and the entity's role.
A prediction of future demand does not replace statutory verification or product investigation obligations.
Do not assume that every medicine, transaction, or jurisdiction is covered by the same U.S. requirements.

Temperature-controlled medicines also require appropriate handling controls for the specific product and jurisdiction.
A temperature reading is an observation, not an automatic release decision.
An authorized quality function should assess an excursion against product-specific procedures.

### Transport and manufacturing

Transportation rules depend on country, vehicle, driver, product, and operation.
For example, U.S. Federal Motor Carrier Safety Administration hours-of-service rules regulate covered drivers and contain exceptions; applicability must be assessed for the actual operation.
A route optimizer should enforce applicable working-time and safety constraints rather than trade them away against route cost.
The dataset must capture driver availability, vehicle capacity, service times, and planned versus actual travel.

Manufacturing systems may use ISA-95 concepts to describe enterprise-control integration.
ISA-95 is a standards family, not a universal legal mandate to implement a specific software architecture.
For scheduling, define work orders, bill of materials, yield, machine states, changeovers, and quality status.
Where a regulated product is involved, validate the relevant process and systems under the applicable requirements rather than relying on a generic manufacturing playbook.

### Personal data and competition

Retail and logistics records can contain customer, employee, location, and delivery data.
Privacy obligations depend on jurisdiction, purpose, and the data involved.
The European Union General Data Protection Regulation (GDPR), for example, sets rules for processing personal data within its scope.
A lawful-basis assessment, minimization, retention, access control, and data-subject rights need specialist review.
Pseudonymization can reduce exposure but does not automatically make data anonymous.

Sharing detailed prices, capacity, or customer data with competitors can create legal risk.
Route data-sharing proposals through legal and security review.
A useful analytical feature is not by itself permission to collect or share it.

### Record the applicability decision

For each potential requirement, record the official source, version or access date, entity, product, geography, activity, exceptions considered, conclusion, and reviewer.
Record unresolved questions separately from approved conclusions.
Escalate a conflict between a model recommendation and a required control; never silently override the control.
Revisit the assessment when the product, market, trading partner, regulation, or operating process changes.

## 9. Industry practice and case evidence

A sector playbook should carry both a public evidence ledger and a local implementation ledger.
The public ledger describes what a source actually says.
The local ledger records the destination decision, transfer assessment, assumptions, and any local test.
A named company is not a substitute for a comparator or reproducible outcome.

### Public case 1 — M5 forecasting competition

**Evidence:** Makridakis and colleagues describe the M5 competition and its evaluation using Walmart unit sales data in the *International Journal of Forecasting*.
The competition provides a public forecasting benchmark, including a hierarchical retail context and evaluation design.
Consult the cited paper for its precise data, methods, and outcomes.

**What it supports:** a reproducible question about forecasting accuracy under the competition's rules.
**What it does not establish:** that a particular retailer will reduce waste or increase profit by deploying a winning method.
**Inference:** a retailer may use the competition to identify baseline forecasting methods worth testing on its own point-in-time data.
**Transfer questions:** Is the decision actually replenishment? Are stockouts and lost demand observable? Are item mix, lead times, and decision costs comparable?
**Owner:** the replenishment lead decides whether a local trial is worthwhile; analytics owns benchmark replication.

### Public case 2 — UPS route optimization

**Evidence:** UPS describes its On-Road Integrated Optimization and Navigation (ORION) system and reports operational benefits in its own public material.
This is a company account, not an independently randomized comparison of all outcomes.
The source documents UPS's description of the initiative and its reported benefits; it does not prove those benefits for another carrier.

**What it supports:** route planning can be organized around a decision about stop sequence subject to operational inputs.
**What it does not establish:** a transferable savings rate for a different geography, fleet, service promise, or workforce.
**Inference:** a carrier can use the case to frame a local test of route feasibility and driver adoption.
**Transfer questions:** Are stops dense, travel times stable, delivery windows comparable, and driver constraints encoded?
**Owner:** the transport operations lead owns dispatch; safety and labor specialists review constraints.

### Public case 3 — pharmaceutical traceability

**Evidence:** FDA materials describe DSCSA requirements and implementation for applicable trading partners and products.
These are regulatory and official sources, not a case study proving operational savings from a particular technology.

**What it supports:** the need to assess applicable tracing and verification obligations for the relevant role and product.
**What it does not establish:** that a specific identifier, vendor platform, or forecast model makes a distributor compliant.
**Inference:** a distributor should integrate product-status and traceability exceptions into its allocation workflow.
**Transfer questions:** Are the product, legal entity, transaction, jurisdiction, and current compliance policy the same?
**Owner:** the quality and compliance functions assess requirements; the allocation owner acts only on eligible stock.

### Synthetic case — factory bottleneck

A plant has one constrained packaging line and two products with different changeover times.
A forecast-only model suggests producing the high-demand product first.
The scheduler instead evaluates feasible sequences, available materials, cleaning time, and customer due dates.
The example is synthetic and establishes no empirical result.
It illustrates why a forecast must be translated into a constrained schedule before the plant can act.

### Synthetic case — humanitarian allocation

A relief organization has limited transport capacity and uncertain accessibility after a flood.
A proposed allocation based only on historical shipment volume may reproduce historical access inequities.
The accountable response lead consults affected communities, combines need and accessibility evidence, and records an explicit allocation rule.
The example is synthetic; no location, beneficiary outcome, or organization is asserted.
The playbook should measure coverage, timeliness, protection risks, and distribution omissions, not merely shipped tonnage.

### Evidence ledger example

| Source | Claim type | Suitable use | Important limit |
|---|---|---|---|
| M5 paper | Peer-reviewed benchmark report, grade A | Compare forecasting designs and metrics | Forecast benchmark is not a local business-value trial |
| UPS material | Company description, grade C | Understand a reported operational application | Company-reported results may lack an independently observed counterfactual |
| FDA DSCSA material | Regulatory or official source, grade B | Determine questions for legal applicability review | Does not establish a vendor's savings or complete compliance |
| Synthetic grocer | Teaching assumption, not graded field evidence | Reproduce arithmetic and critique a test | Cannot establish any real-world outcome |

A source can be strong for one question and weak for another.
Record grade and relevance separately from the evaluation design.
When a source reports an effect without defining a comparator, retain that uncertainty in the local playbook.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No decision owner | Meetings produce models but no approved action | Authority is unclear | Recommendations are unused | Name accountable owner and delegate before build |
| Wrong decision grain | Weekly forecasts feed hourly dispatch | Information and action cadences differ | Late or unusable recommendations | Align grain and cutoff to action |
| Hindsight leakage | Backtest uses corrected inventory | Later information enters training or replay | Inflated apparent accuracy | Preserve point-in-time snapshots |
| Censored demand | Sales fall when shelves are empty | Unavailable sales are mistaken for no demand | Under-ordering persists | Measure availability and estimate lost demand with sensitivity bounds |
| Inventory state ignored | Held stock appears available | Quantity is not equivalent to usable stock | Invalid allocation or quality breach | Separate available, reserved, damaged, and held states |
| Weak comparator | Candidate beats a retired spreadsheet | Baseline is not the real alternative | False value claim | Compare against executable current process |
| Metrics mismatch | Accuracy improves but waste rises | Prediction loss differs from business cost | Model success without operating benefit | Measure decision outcomes and guardrails |
| Cost double count | Waste savings appear in two finance lines | Same avoided expense is valued twice | Inflated business case | Reconcile a non-overlapping benefit bridge |
| Unequal exposure | One group has more stores or requests | Group totals reflect volume | Spurious incremental value | Normalize or use design-consistent unit-level analysis |
| Shared capacity spillover | Pilot stores receive priority at warehouse | Treatment changes control conditions | Biased group comparison | Cluster assignment or model interference |
| Unsafe optimization | Solver relaxes a quality hold | Hard constraint treated as a cost | Unsafe or unlawful action | Encode immutable gates and escalate infeasibility |
| Uncontrolled overrides | Operators change orders without reasons | Execution differs from assigned policy | Uninterpretable result | Log action, reason, role, and timestamp |
| Blind automation | Operators cannot pause a recommendation | System lacks exception handling | Errors propagate at scale | Give authorized override and tested fallback |
| Unverifiable source | Savings claim lacks dates or denominator | Marketing replaces evidence | Unsupported transfer | Record source, population, comparator, and unknowns |
| Selection bias | Only successful deployments are catalogued | Failures and withdrawals are invisible | Overestimated success probability | Seek negative cases and document search limits |
| Regulatory overreach | Team asserts all products are covered | Scope and exemptions ignored | Wrong controls or false assurance | Obtain product- and role-specific legal review |
| Incomplete traceability | Product identifier has no lot linkage | Item identity is confused with physical history | Slow or incomplete investigation | Validate event linkage and reconcile physical flows |
| Privacy drift | Detailed location data reused for a new purpose | Purpose and access controls are not reviewed | Privacy or legal exposure | Review lawful basis, minimization, and access |
| Seasonality confounding | Trial coincides with a promotion | Time and intervention effects are mixed | Misattributed benefit | Use contemporaneous comparison and log events |
| Stale playbook | Supplier lead time changes without review | Assumptions no longer describe operations | Persistent wrong orders | Trigger versioned review on material change |
| Rare-event overconfidence | No incidents in a tiny sample | Absence of observed harm is treated as proof | Unsafe rollout | Use independent controls and appropriate risk evidence |
| Benefit annualization | Four-week gain multiplied by 13 without qualification | Seasonal and scale effects are assumed away | Unsupported budget commitment | Report pilot period and scenario assumptions separately |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Decision owner can act | A recommendation must become an authorized action | Ownership resides in another team | Trace approval and execution rights before launch |
| Inputs arrive by cutoff | Candidate must operate without hindsight | Late receipts or corrected sales | Replay actual availability timestamps |
| Product identifiers are stable | History and actions must refer to the same item | Repack, substitution, or reused identifiers | Reconcile master-data changes and units |
| Inventory is usable | Order rule depends on available stock | Holds, reservations, damage, or stale counts | Reconcile inventory states with physical counts |
| Eligible demand is measured | Fill rate needs a valid denominator | Requests disappear during stockouts | Compare request capture with independent demand signals |
| Comparison groups are comparable | Observed differences should not merely reflect mix | Biased assignment or differential attrition | Inspect assignment, baseline balance, and missingness |
| No material interference | One group's treatment should not alter the other | Shared stock, staff, fleet, or promotion | Map shared resources; cluster or redesign |
| Unit values do not overlap | Economic bridge should count each benefit once | Waste recovery already included in margin | Finance reconciles account definitions |
| Cost values apply to the period | Value depends on current economics | Prices, labor, and product mix shift | Refresh values and run sensitivity scenarios |
| Operating constraints remain stable | Feasible action depends on current limits | New supplier cutoff, driver rule, or quality hold | Monitor constraints and trigger review |
| Safety controls remain independent | A profitable policy cannot waive a hard gate | Solver or fallback bypasses a hold | Test authorization and exception workflows |
| Public case is relevant | Source may inform local hypotheses | Different decision, data, economics, or risk | Apply five-question transfer test |
| Pilot represents rollout | A larger population may behave similarly | Different stores, seasons, or capacity | Replicate across strata and seasons before scaling |
| Outcome records are complete | Measured changes must reflect all eligible units | Late or missing events differ by group | Audit completeness by assignment and date |
| Legal assessment stays current | Requirements depend on jurisdiction and time | Rule, product, or entity changes | Assign a reviewer and dated refresh trigger |

These are assumptions to investigate, not claims that they hold in a particular organization.
A positive calculation is conditional on them.
When a material assumption is unknown, preserve the unknown and choose a bounded way to resolve it.

## 12. Exercises

### Exercise 1 — Frame the decision

**Scenario:** A grocery sponsor asks for “an AI demand solution.”
The team currently orders chilled meals daily, and the buyer can approve orders before a supplier cutoff.

**Deliverable:** A six-line decision charter naming action, owner, product, location, deadline, baseline, and risk.
**Acceptance criteria:** Distinguish forecast from order; include at least one safety or service constraint and a fallback.
**Time:** 12 minutes.

### Exercise 2 — Build a data contract

**Scenario:** Receipts have physical event time and system availability time; a corrected inventory balance is posted next morning.
The order deadline is 14:00 local time.

**Deliverable:** A table for receipts, inventory, demand, and quality status with grain, cutoff, owner, and quality check.
**Acceptance criteria:** Exclude information unavailable at 14:00 from the historical decision replay; distinguish held from available stock.
**Time:** 18 minutes.

### Exercise 3 — Choose a method

**Scenario:** A plant has a single bottleneck line, product-specific cleaning times, and uncertain demand.
A proposed solution produces the most accurate weekly forecast but does not model line availability.

**Deliverable:** A method card with baseline, candidate, constraints, inputs, outputs, and fallback.
**Acceptance criteria:** Explain why forecasting alone does not create a feasible schedule; identify the scheduler as decision owner.
**Time:** 15 minutes.

### Exercise 4 — Calculate service and value

**Scenario:** In comparable groups each has 10,000 requested units over four weeks.
Baseline fulfills 9,400 units and wastes 450 units; candidate fulfills 9,600 units and wastes 350 units.
Incremental contribution is USD 3.00 per extra fulfilled unit, avoided waste costs USD 1.50 per unit, and candidate operating cost is USD 250.

**Deliverable:** Both fill rates, difference in percentage points, and observed value difference in USD per four weeks.
**Acceptance criteria:** Show arithmetic, avoid counting revenue as profit, and state at least one identification limitation.
**Time:** 15 minutes.

### Exercise 5 — Separate economic concepts

**Scenario:** A manager calls 1,000 forecast units an approved order, calls a 98% target a forecast, and counts both gross sales and contribution margin as separate benefits.

**Deliverable:** Correct each statement and write a non-overlapping benefit bridge.
**Acceptance criteria:** Define forecast, plan, target, and commitment; identify at least one double count.
**Time:** 12 minutes.

### Exercise 6 — Critique a source

**Scenario:** A vendor reports a 20% improvement for a named logistics customer but provides no comparator, population, or dates.
Your company has a different delivery density and driver schedule.

**Deliverable:** A source ledger row and five-question transfer assessment.
**Acceptance criteria:** Attribute the claim, classify missing evidence as unknown, avoid promising 20%, and identify one local test.
**Time:** 18 minutes.

### Exercise 7 — Design a controlled pilot

**Scenario:** Twenty stores share one distribution center and may compete for limited stock.
The business wants to compare an ordering policy with its current workflow.

**Deliverable:** A one-page pilot protocol covering assignment, comparison, primary outcome, guardrails, override, uncertainty, and stop rule.
**Acceptance criteria:** Address shared-capacity interference, analyze assigned units, identify owners, and state what would stop the test.
**Time:** 25 minutes.

### Exercise 8 — Assess regulatory applicability

**Scenario:** A distributor handles both food and prescription drugs in the United States.
The analytics team proposes one generic barcode rule for both flows.

**Deliverable:** An applicability checklist and escalation record, not a legal opinion.
**Acceptance criteria:** Separate product, entity, activity, jurisdiction, source, exceptions, and reviewer; explain why a barcode alone does not prove compliance.
**Time:** 18 minutes.

### Exercise 9 — Maintain the playbook

**Scenario:** A supplier changes its cutoff time and a store moves to a new inventory system.
The candidate policy continues producing orders with no alert.

**Deliverable:** A version-change and incident-response record.
**Acceptance criteria:** Name the owner, effective dates, fallback, affected assumptions, revalidation, and approval route.
**Time:** 15 minutes.

## 13. Self-check answers

### Exercise 1

A strong charter says that the buyer chooses the next-day chilled-meal order quantity by the supplier cutoff, using available inventory and a forecast, with the current manual ordering process as baseline.
It names the stores and products, the supplier deadline, the quality hold rule, and the person who can approve and override the order.
The forecast is an input; the buyer-approved purchase order is the commitment.
A missing feed triggers the documented manual fallback rather than an invented quantity.

### Exercise 2

A receipt that occurred at 13:30 but became available at 15:00 is unavailable for the 14:00 decision.
The next-morning inventory correction cannot be inserted retrospectively into the decision snapshot.
The contract should retain both event and availability timestamps, item-store grain, source owner, timezone, and late-arrival rate.
Inventory needs separate available, reserved, damaged, and held states.
Demand capture should be checked independently during stockouts.

### Exercise 3

The scheduler owns the sequence and start times; the forecast is one input.
A feasible schedule also requires line capacity, cleaning time, materials, due dates, and applicable quality controls.
The baseline is the current executable scheduling procedure.
A candidate may combine a forecast with constrained scheduling or simulation.
If inputs or the solver fail, the authorized scheduler uses the documented baseline while retaining quality gates.

### Exercise 4

Baseline fill rate = 9,400 / 10,000 = 94%.
Candidate fill rate = 9,600 / 10,000 = 96%.
Difference = 96% − 94% = 2 percentage points.
Extra fulfilled units = 9,600 − 9,400 = 200 units.
Avoided waste = 450 − 350 = 100 units.
Value difference = 200 × USD 3.00 + 100 × USD 1.50 − USD 250 = USD 500 per four weeks.
The result is an observed difference under the stated assumptions, not necessarily a causal effect.
Random variation, non-comparable groups, missing demand, or shared capacity can alter the interpretation.

### Exercise 5

The forecast estimates uncertain demand; a plan proposes an action; the target expresses the desired result; the commitment is an authorized obligation.
A 1,000-unit forecast is not itself a purchase order.
A 98% target is not necessarily a forecast of 98% performance.
If contribution margin is used to value extra sales, gross sales cannot also be added as a separate benefit.
Add avoided waste only to the extent it is not already included in the contribution calculation.
Subtract incremental operating and implementation costs once.

### Exercise 6

A vendor claim is grade D under the module taxonomy and should be attributed to that vendor.
The claimed 20% has unknown denominator, comparison, population, period, and independent verification.
The destination's decision, economics, data, operating constraint, and evidence sufficiency must each be assessed as yes, no, or unknown.
Different delivery density and driver schedules are material transfer questions.
A controlled local route test can compare feasible actual routes with the existing dispatch policy while monitoring safety, service, and adoption.
The claim is an analogy, not a promised local improvement.

### Exercise 7 rubric

A strong protocol defines the store or cluster assignment, eligibility, timing, baseline, candidate, and primary outcome before launch.
It explains how the shared distribution center could transfer stock or capacity between groups and either designs clusters to contain that effect or states how it will be assessed.
It retains assigned units in the primary analysis and reports overrides and missingness separately.
It specifies safety and service guardrails, an accountable stop authority, and a fallback that does not bypass holds.
It describes uncertainty, escalation, and what decision follows an inconclusive result.

### Exercise 8 rubric

A strong checklist separates food and drug product scope, the distributor's role, covered activities, jurisdiction, exemptions, and current official sources.
It records the date and the legal and quality reviewers responsible for an applicability decision.
It distinguishes identifier standards from legally required records, verification, and processes.
A barcode can identify an item but does not prove complete, accurate, timely records or compliance.
An unresolved question remains open until an authorized reviewer resolves it.

### Exercise 9 rubric

A strong record identifies the old and new cutoff, inventory-source change, effective dates, and affected decisions.
The operational owner activates a safe fallback; the data owner checks availability and mapping; the analytics owner replays with the new cutoff.
The quality owner confirms that holds remain enforced.
The change is approved, versioned, monitored, and accompanied by a review of the previous evaluation's relevance.

## 14. Interview questions

1. **How do you turn a sector case into a playbook?**
   A strong answer starts with a local decision and owner, then covers data, baseline, constraints, evidence, transfer, measurement, and controls.
   A weak answer lists technologies without an action or accountable owner.
2. **Why is a forecast not an order?**
   A strong answer distinguishes uncertainty, feasible planning, authorization, and execution.
   A weak answer assumes the predicted number can be sent directly to a supplier.
3. **What is the right baseline?**
   A strong answer uses a feasible current alternative with the same information cutoff and constraints.
   A weak answer compares against an obsolete or hindsight-based rule.
4. **How do stockouts affect demand data?**
   A strong answer explains censored sales and asks for availability and independent demand evidence.
   A weak answer treats all zero sales as zero demand.
5. **What makes a case transferable?**
   A strong answer checks decision, economics, data, constraints, and evidence, while retaining unknowns and hard gates.
   A weak answer copies a reported savings percentage because the industry label matches.
6. **How would you evaluate a store pilot?**
   A strong answer addresses assignment, comparability, interference, missingness, uncertainty, and guardrails.
   A weak answer attributes a before-and-after change entirely to the policy.
7. **How do you value service improvement?**
   A strong answer converts incremental units to contribution, accounts for displacement, and reconciles non-overlapping costs.
   A weak answer counts all gross sales as profit and adds the same savings twice.
8. **What happens if the optimizer cannot satisfy a quality hold?**
   A strong answer treats the hold as a hard gate, stops the affected recommendation, and escalates.
   A weak answer relaxes the hold to produce a profitable schedule.
9. **What does a GS1 identifier establish?**
   A strong answer distinguishes item identity, lot or serial linkage, event capture, physical reconciliation, and legal applicability.
   A weak answer asserts that scanning any barcode proves compliance.
10. **Who owns an override?**
    A strong answer names authorized operators, logs the reason, and preserves accountability for the executed action.
    A weak answer forbids human intervention or assumes every override proves the model is wrong.
11. **When should the playbook be reviewed?**
    A strong answer names owner, cadence, material-change triggers, and versioned approvals.
    A weak answer treats the initial model validation as permanent.
12. **How do you describe an inconclusive pilot?**
    A strong answer reports the observed effect, uncertainty, limits, guardrails, and pre-agreed next decision.
    A weak answer calls an imprecise positive difference proven savings.

## 15. Key terms

- [Sector playbook](#4-definitions-and-vocabulary) — versioned guide to an operating decision and its evidence.
- [Decision owner](#step-3--map-stakeholders-and-authority) — accountable authority for the action.
- [Decision cadence](#step-1--specify-the-decision) — frequency and deadline of a choice.
- [Baseline](#step-6--establish-an-executable-baseline) — executable alternative to the candidate.
- [Policy](#step-7--select-a-method-for-the-decision) — rule mapping available information to action.
- [Forecast](#four-different-statements) — estimate of an uncertain future quantity.
- [Plan](#four-different-statements) — proposed feasible actions.
- [Target](#four-different-statements) — desired outcome, not a prediction.
- [Commitment](#four-different-statements) — authorized obligation or action.
- [Constraint](#step-7--select-a-method-for-the-decision) — condition the decision must satisfy.
- [Guardrail](#step-4--translate-objectives-into-measures) — predefined boundary on acceptable risk.
- [Point-in-time data](#step-5--inventory-the-data-and-its-timing) — information available at the actual cutoff.
- [Leakage](#10-failure-modes-and-red-flags) — unavailable future information in a historical decision.
- [Censoring](#10-failure-modes-and-red-flags) — incomplete observation such as stockout-limited sales.
- [Fill rate](#6-worked-example) — fulfilled units divided by eligible requested units.
- [Contribution margin](#step-10--convert-outcomes-into-economics) — incremental revenue less specified variable costs.
- [Waste](#6-worked-example) — unusable units or their specified cost.
- [Counterfactual](#step-9--design-the-local-comparison) — outcome under the alternative action.
- [Randomization](#step-9--design-the-local-comparison) — chance assignment under a defined protocol.
- [Spillover](#10-failure-modes-and-red-flags) — an intervention's effect on a comparison unit.
- [Transfer test](#step-8--evaluate-sources-and-transfer) — five-question assessment of case fit.
- [Evidence grade](#9-industry-practice-and-case-evidence) — source-type classification, not causal proof.
- [Data contract](#step-5--inventory-the-data-and-its-timing) — agreed data definitions, delivery, and ownership.
- [Traceability](#8-data-standards-and-regulatory-context) — linkage of relevant product and movement records.
- [Lot](#8-data-standards-and-regulatory-context) — defined quantity tied to a production or handling history.
- [Override](#step-11--operationalize-the-decision) — authorized departure from a recommendation.
- [Fallback](#step-11--operationalize-the-decision) — executable alternative when the candidate cannot operate.
- [Applicability](#record-the-applicability-decision) — determination of whether a rule governs a specific context.
- [Version](#step-12--review-and-retire) — identifiable snapshot of the decision process and assumptions.

## 16. Sources

Sources below are checkable starting points; consult current official texts for live compliance decisions.
The evidence grade describes source type in this module, not the strength of every claim that might be derived from it.

1. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. “The M5 accuracy competition: Results, findings and conclusions.” *International Journal of Forecasting*, 38(4), 2022, 1346–1364. [https://doi.org/10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013). Peer-reviewed benchmark report, grade A.
2. UPS. [ORION: On-Road Integrated Optimization and Navigation](https://about.ups.com/us/en/our-company/innovation/orion.html). Company description, grade C; verify current page and attribution when using a reported result.
3. U.S. FDA. [FSMA Final Rule on Requirements for Additional Traceability Records for Certain Foods](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods). Official explanation, grade B; check current dates and scope.
4. U.S. Electronic Code of Federal Regulations. [21 CFR Part 1, Subpart S](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-1/subpart-S). Current regulatory text, grade B.
5. U.S. FDA. [Drug Supply Chain Security Act](https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa). Official overview and linked guidance, grade B; check current implementation policy.
6. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Standards body source; identification conventions and technical definitions.
7. GS1. [EPCIS and CBV](https://www.gs1.org/standards/epcis). Standards body source; visibility-event and vocabulary specifications.
8. U.S. Federal Motor Carrier Safety Administration. [Hours of Service](https://www.fmcsa.dot.gov/regulations/hours-service). Official regulatory overview, grade B; applicability and exceptions require review.
9. ISA. [ISA-95 Standard: Enterprise-Control System Integration](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard). Standards body overview; not itself a universal legal requirement.
10. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Official legal text, grade B; assess territorial and substantive scope.
11. National Institute of Standards and Technology. [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework). Voluntary risk-management resource; not a blanket certification or legal mandate.

## 17. Links

- [Module 28 index](./_index.md) — topic sequence and scope.
- [Case anatomy](./01_case-anatomy.md) — structure a case before extracting a playbook.
- [Evidence grades](./02_evidence-grades.md) — assess source types and limitations.
- [The transfer test](./03_transfer-test.md) — determine whether a case is a plan or an analogy.
- [Module 28 survey and its labs](../../../course/28_domain_case_library.md) — apply the case and transfer exercises.
