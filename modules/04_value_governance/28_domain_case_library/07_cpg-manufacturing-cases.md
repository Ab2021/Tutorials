# Consumer-Goods, Manufacturing, and Industrial Cases

> Module 28 — Domain Case Library and Sector Playbooks  
> Topic 07 of 10  
> Estimated study time: 4 hours  
> Prerequisites: case anatomy, evidence grades, and the transfer test  
> Version 2026.5

## 1. Why this topic exists

A consumer packaged goods (CPG) manufacturer can ship more cases and still destroy value.
A production line can run at high utilization and still miss the customer because one component is absent.
A well-known company can describe its practices without proving that those practices caused its results.
These are decision problems, not just data problems.

A replenishment manager needs to decide whether a supplier may propose store orders.
A category manager needs to distinguish incremental demand from a promotion that merely shifts purchases in time.
A plant manager needs a safe response to an abnormal process.
A supply-risk team needs to locate a critical sub-tier plant before an earthquake or a chip shortage interrupts it.
Aerospace leaders must deliver certified configurations while constrained suppliers and labor compete for capacity.
The relevant constraints and evidence differ across these decisions.

Case studies offer useful mechanisms and questions, but a famous name is not a control group.
This topic reads six families of evidence without claiming that any one company provides a universal blueprint.
The continuous-replenishment and trade-spend examples concern information and commercial incentives.
Toyota's production-system description concerns management design.
The 2011 Tohoku earthquake concerns disruption and recovery.
The Boeing-related Government Accountability Office (GAO) study concerns contemporary aviation supply constraints.
The semiconductor request for information concerns a market-wide shortage, not a controlled intervention.

A decision memo should say what the source actually establishes, what remains a company claim, and what is inferred.
It should also say which local decision would change, which data would support it, and how to stop a damaging pilot.
The example and calculations below are synthetic and are not estimates for any named company.

## 2. Learning outcomes

After this topic you can:

1. Identify the decision, unit, evidence source, and missing counterfactual in each case family; see exercises 1 and 2.
2. Trace how shared demand signals and replenishment authority can change an order; see exercise 3.
3. Calculate incremental promotion contribution after variable costs and trade spending; see exercise 4.
4. Distinguish Toyota's stated production principles from demonstrated causal impact; see exercise 1.
5. Reconstruct the time sequence and the stated limits of the earthquake evidence; see exercise 2.
6. Compare aircraft and chip bottlenecks without equating reported association with causation; see exercise 5.
7. Design an accountable transfer pilot with guardrails and an evaluation; see exercises 6–8.
8. Explain how data provenance, product identity, and applicable rules constrain the decision; see exercise 8.

## 3. The core mental model

```text
Evidence source -> claim and scope -> mechanism hypothesis -> local conditions
                                                    |             |
                                                    v             v
                                           candidate decision -> pilot and control
                                                                  |
                                              service / margin / safety / resilience
                                                                  |
                                                   review, stop, or scale
```

The source establishes a bounded observation.
The mechanism explains how a proposed action might change a decision.
The destination is the operation in which you hope to apply it.
The counterfactual is what would have happened there without the action.
The pilot supplies local evidence about the difference, subject to its own limitations.

Information sharing alone does not put a product on a shelf.
A usable signal must reach an authorized decision-maker before the next order cutoff.
A promotion forecast does not create incremental margin by itself.
It must be converted into an approved offer, funded, executed, and evaluated against a defensible baseline.
A supply map alone does not prevent disruption.
It must connect a vulnerable site to a qualified alternative, buffer, or recovery action.

Use four labels throughout a case note:

- **Documented observation:** a source reports an event or measurement within a defined scope.
- **Company claim:** a company reports its own action, performance, or explanation.
- **Inference:** an interpretation that goes beyond the source and requires testing.
- **Synthetic example:** invented values used to teach a calculation; not an empirical claim.

A source can contain several labels at once.
Toyota's account of a production recovery is a company-reported observation.
Attributing all recovery to one management principle would be an unsupported inference.
Similarly, a government survey finding is an observation about respondents, not automatically a population estimate.

The transfer test asks five practical questions.
Is the destination decision the same?
Is the mechanism feasible under local incentives and constraints?
Are the necessary data available at the necessary grain and latency?
Are outcomes comparable, including cost and safety?
Can the team measure the incremental effect and stop safely?
A failed answer does not make the original case false.
It means the case is not yet an implementation plan.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| CPG | Consumer packaged goods, typically repeat-purchase products | Assuming all products have the same shelf life or demand |
| Sell-through | Consumer purchases at retail, in units per period | Calling manufacturer shipments consumer demand |
| Sell-in | Supplier shipments to the channel, in units per period | Treating a channel fill as consumption |
| Continuous replenishment | Frequent replenishment driven by shared consumption and agreed rules | Assuming a particular algorithm or universal ownership |
| Vendor-managed inventory | Supplier participates in inventory decisions under an agreement | Assuming title, liability, and order authority are identical |
| Point of sale (POS) | Store transaction record, normally units, price and time | Treating incomplete data as unconstrained demand |
| Lead time | Elapsed time from decision to usable receipt, in days | Replacing an uncertain distribution with an unqualified average |
| Order-up-to level | Desired inventory position, in units | Confusing it with physical on-hand stock |
| Inventory position | On hand plus eligible on-order minus backorders, in units | Counting unusable or duplicate receipts |
| Fill rate | Fulfilled quantity divided by requested quantity | Confusing it with order or line service level |
| Days of supply | Inventory divided by a stated daily consumption rate, in days | Comparing denominators from different periods |
| Trade spend | Supplier spending under retailer commercial agreements, in currency | Treating every payment as a price discount |
| Slotting allowance | Payment for initial placement or initial warehouse access | Treating it as proof of promotion effectiveness |
| Baseline | Estimated result without the intervention | Assuming last week's sales equal the counterfactual |
| Incremental volume | Observed or forecast volume minus counterfactual volume | Calling all promoted volume incremental |
| Pull-forward | Purchases shifted from a later period into a promotion | Counting the same demand twice |
| Cannibalization | Lost contribution from displaced products | Ignoring the rest of the portfolio |
| Contribution | Revenue less specified variable costs | Calling it net profit without fixed costs |
| Jidoka | Toyota's principle of stopping to address abnormalities | Reducing it to automation alone |
| Just-in-time (JIT) | Toyota's principle of making what is needed, when and in the amount needed | Equating it with zero inventory everywhere |
| Tier | Position in a supply network relative to the focal manufacturer | Assuming the direct supplier is the source of every component |
| Recovery time | Time to restore a specified capability after an interruption | Assuming plant restart equals delivery recovery |
| Buffer | Inventory, capacity, time, or other reserve against variation | Assuming a buffer fixes a permanent capacity deficit |
| Qualification | Approval of a source or process against relevant requirements | Treating a second supplier as immediately interchangeable |
| Bottleneck | Constraint limiting throughput under current conditions | Assuming the same constraint holds for all products |
| Counterfactual | Outcome that would have occurred without the action | Treating an unadjusted comparison as causal proof |
| Transfer | Application of a mechanism to a distinct setting | Copying an outcome or percentage from a source |
| Forecast | Conditional estimate of a future outcome | Calling it a commitment |
| Plan | Proposed set of actions and resources | Treating it as an approved obligation |
| Target | Desired outcome | Treating it as an unbiased prediction |
| Commitment | Authorized obligation to act or deliver | Confusing it with a scenario |

## 5. The mechanics, step by step

### Step 1 — State the decision and accountability

Input: a real business question and the next decision date.
Output: a decision statement naming the owner and alternatives.
For replenishment, specify which party can create, approve, and override an order.
For trade spend, specify who approves the offer and who pays each component.
For manufacturing, specify the authority to stop the process and release product.
For resilience, specify who can reserve capacity, approve substitutions, and accept risk.
Do not start with a tool selection.
If nobody can act on the analysis, a more accurate forecast will not necessarily change an outcome.

### Step 2 — Delimit the case and the source

Input: an original document, publication date, and relevant passage.
Output: a source-to-claim ledger with unit, time period, geography, and limitations.
Record whether the statement is direct evidence, company-reported performance, or inference.
A named case may have multiple sources of different quality.
Read the actual passage rather than quoting an attractive number from a later presentation.
If the evidence is missing, label it missing instead of filling the gap with a plausible mechanism.

### Step 3 — Map the physical and commercial flow

Input: bill of materials or product hierarchy, order path, and ownership terms.
Output: a flow map from demand to supply, including a money flow.
Locate stock at stores, distribution centers, in transit, and at suppliers.
For a chip, connect design, fabrication, packaging, test, and the consuming assembly where known.
For an aircraft, connect approved configuration, supplier work, quality acceptance, and final assembly.
For a promotion, connect the retail price, supplier allowance, claim, invoice, and settlement.
A financial credit may lag the physical shipment by months.
A supplier relationship is not necessarily the location of the physical bottleneck.

### Step 4 — Specify a candidate mechanism

Input: source observation and a plausible pathway to the local decision.
Output: a mechanism hypothesis with necessary conditions and possible side effects.
In replenishment, a fresher consumption signal could reduce delayed orders if the retailer shares timely, complete data.
In trade spend, a randomized promotion could reveal incremental demand if execution and control are comparable.
In production, abnormality detection could reduce escaped defects if workers can stop and resolve the process.
In disruption response, a mapped sub-tier site could shorten identification time if the map is current and actionable.
In aircraft manufacturing, supplier monitoring could reveal risk earlier if reporting captures actual capacity and quality constraints.
In chips, inventory buffers could bridge a temporary interruption if the buffer matches the required part and duration.
These are hypotheses, not estimates of achieved savings.

### Step 5 — Build the data contract

Input: decisions and the mechanism's required information.
Output: a documented grain, owner, refresh time, validation, and permitted use for each dataset.
Identify the item, location, calendar, quantity, and unit of measure.
Define event time separately from load time.
Capture stockouts, discontinued items, substitutions, returns, and price changes.
Preserve historical versions of the product hierarchy and bill of materials.
Include a reliable route for correction when a partner disputes a record.
If agreement prohibits use of a record for a purpose, do not proceed on an assumption of consent.

### Step 6 — Construct the counterfactual

Input: a baseline design fixed before intervention where possible.
Output: expected outcomes without the action, uncertainty, and known threats.
Randomize eligible stores or comparable store clusters where operationally feasible.
Pre-specify assignment, treatment dates, exclusion rules, and outcome windows.
If randomization is impossible, use a justified comparison group and assess pre-period trends.
Monitor concurrent price, distribution, weather, capacity, and competitor changes.
A before-and-after contrast alone cannot separate the action from all concurrent changes.
Include post-event weeks to detect pull-forward and delayed replenishment.
Do not use a control location that is materially contaminated by the intervention.

### Step 7 — Calculate total decision value

Input: incremental units, price and costs, trade payments, operational cost, and risk effects.
Output: a bridge from baseline to intervention, in a specified currency and time window.
Use contribution per incremental unit, not revenue per gross unit.
Deduct promotion fees, execution costs, and any required markdown.
Include cannibalization and pull-forward if they occur inside the agreed assessment boundary.
For inventory, separate carrying cost from stockout and expediting costs.
For resilience, include inventory and qualification cost as well as disruption exposure.
Avoid adding the same cost twice when it is already in a unit contribution.
Do not monetize safety impacts with a made-up financial value to justify breaching a safety requirement.

### Step 8 — Design guardrails and escalation

Input: a candidate intervention and its foreseeable failure modes.
Output: a pilot charter with decision rights, limits, and stop criteria.
A replenishment pilot can pause on stale POS, negative inventory, or a service decline.
A promotion can pause when the approved spend cap or stock availability threshold is breached.
A production line must follow approved safety and quality escalation, irrespective of an efficiency target.
A supply-risk pilot can pause when an alternative source is not qualified or a part substitution is unauthorized.
Define who is notified, what happens to in-flight decisions, and how normal operations resume.

### Step 9 — Evaluate and decide

Input: observed outcomes, baseline, deviations, and uncertainty.
Output: a decision to stop, redesign, extend, or scale, with an accountable owner.
Report treatment adoption separately from model output.
Report the service and quality distribution, not just an average.
Explain uncertainty and the effect of missing data.
Review both financial and nonfinancial guardrails before expanding the scope.
The pilot result applies first to the tested locations, items, and dates.
Extrapolation requires a second transfer argument.

### Step 10 — Maintain the decision system

Input: approved rules, contracts, data feeds, and change requests.
Output: an operating cadence with owners, review dates, and audit records.
Revalidate when products, suppliers, price terms, or lead-time distributions change.
Investigate recommendations that are repeatedly overridden.
Record the reason for each override rather than assuming all overrides are errors.
Refresh the supply map when a sub-tier plant, part, or route changes.
A source citation is not a substitute for a current local operating record.

### Operational decision path A — shared replenishment

1. Select the product-location combinations for the test.
   Exclude products without a stable identifier or a known pack conversion.
   Keep a log of exclusions so results are not mistaken for the whole assortment.
2. Record the existing ordering cadence before changing it.
   Include the order cutoff, receiving schedule, minimum quantity, and normal exceptions.
   Obtain the actual observed lead-time distribution rather than an undocumented promise.
3. Agree which party supplies each field.
   The retailer may provide transactions and store stock.
   The supplier may provide open orders, shipment status, and available supply.
   Each side should identify a person who can resolve a disagreement.
4. Decide how to treat missing observations.
   A missing POS file is not a zero-sales day.
   A missing stock file is not an empty shelf.
   Hold a recommendation when a critical input fails its freshness threshold.
5. Validate opening stock against a physical sample.
   Record whether the count includes damaged, expired, or reserved inventory.
   Restrict available stock to quantities that can satisfy the relevant demand.
6. Construct a demand signal at the item-location-day grain.
   Mark closures, stockouts, returns, and unusual events separately.
   Use a documented treatment for incomplete days rather than silently dropping them.
7. Translate the demand signal into a replenishment position.
   Include the review period as well as the physical lead time.
   The coverage interval must match the cadence at which a new order can be placed.
8. Set a buffer using the destination's uncertainty and service requirement.
   Check demand variation, lead-time variation, and constraints on stock.
   A single percentage copied from a case has no demonstrated local validity.
9. Apply minimum order and pack rules.
   Review the impact of rounding on expiry, display capacity, and transport.
   If a constraint makes the proposed order infeasible, send an exception instead.
10. Define approval and fallback.
    The person with contractual authority accepts or rejects the suggestion.
    When a feed fails, use a documented manual process and retain the reason.
11. Evaluate on the same calendar and product set.
    Report units short, fulfilled units, inventory, and costs separately.
    Compare against an appropriately assigned or adjusted control group.
12. Review whether the intervention changed decisions.
    A precise suggestion that was never accepted cannot explain a measured operational effect.
    Track adoption, overrides, and the time from exception to resolution.

### Operational decision path B — trade-spend evaluation

1. Write the commercial objective.
   It might be profitable incremental demand, distribution expansion, or an agreed launch.
   These are different objectives and need different success measures.
2. Identify the exact instrument.
   A slotting payment, temporary price reduction, display fee, and volume rebate are distinct.
   A single campaign can include several instruments at once.
3. Obtain the executed agreement and amendments.
   Record who is eligible, when, at which locations, and on what quantity basis.
   An assumed allowance rate is not a substitute for the settlement clause.
4. Separate the manufacturer and retailer economics.
   A manufacturer's incremental contribution need not equal the retailer's incremental margin.
   A joint business case should show the flows between the parties rather than adding transfers twice.
5. Fix the unit of analysis.
   Store-week can support a store-level promotion comparison.
   Item-day may be necessary when execution starts on different dates.
6. Identify comparison locations before launch.
   Match or randomize using pre-period demand, distribution, and relevant characteristics.
   Record exclusions and keep the assignment stable through the analysis.
7. Verify execution.
   An authorized display may not be installed or may be installed late.
   Separate the effect of assignment from the effect among locations that actually executed.
8. Include product availability.
   An out-of-stock promotion has different implications from a fully supplied one.
   Do not interpret zero sales in an empty store as a lack of customer interest.
9. Extend the outcome window beyond the event.
   A pantry-loading product may see a later decline after the discounted period.
   Choose follow-up based on the purchase cycle, not on the accounting close alone.
10. Measure portfolio displacement.
    Identify related products and plausible switching before the event.
    Include their contribution change where the decision boundary covers the portfolio.
11. Reconcile the financial outcome.
    Match contract, execution evidence, claim, invoice, and settlement.
    Include late claims and adjustments in the stated reporting cut.
12. Decide whether to repeat.
    A positive point estimate with wide uncertainty may justify another test, not national deployment.
    A negative estimate can still be informative if the design and execution were credible.

### Operational decision path C — manufacturing quality and flow

1. Select a process with a clearly defined output and acceptance condition.
   Name the customer of that process, which may be the next internal station.
   Distinguish an observed defect from an upstream process condition.
2. Record normal operating limits.
   Identify who set them and which instrument supplies each measurement.
   Calibration status and measurement error affect the usefulness of a signal.
3. Describe what constitutes an abnormality.
   Specify the threshold, duration, and relevant product or equipment.
   Avoid alarm rules that trigger so frequently that workers routinely ignore them.
4. Define the immediate safe response.
   Operators need authority and a clear route to stop or contain the affected work.
   The response must follow the site's safety procedures.
5. Define containment separately from root-cause analysis.
   Containment prevents further potentially affected product from moving onward.
   Root-cause work tests why the abnormal condition occurred.
6. Record the affected time and material interval.
   Link the event to serial numbers, lots, equipment, and personnel where appropriate.
   Traceability enables targeted investigation instead of an unsupported assumption.
7. Assign a resolution owner.
   Maintenance, engineering, production, and quality may have different responsibilities.
   Specify the handoff and required evidence for each.
8. Define restart authorization.
   A process is not ready merely because the alarm has cleared.
   Approved checks must establish that the problem is contained and restart is permitted.
9. Monitor both leading and lagging measures.
   Time to respond and repeat alarms are leading operational signals.
   Escaped defects and rework are downstream outcomes.
10. Review the incentive system.
    A throughput target that penalizes every stop can discourage reporting.
    Compare actual stop records with the stated authority to stop.
11. Investigate recurring exceptions.
    A repeated override may indicate a poorly designed threshold or an unresolved failure.
    It may also reflect a capacity pressure that management has not addressed.
12. Transfer only the mechanism that fits.
    A regulated or hazardous process may require different approval and containment procedures.
    The original company's description is not an authorization for the destination site.

### Operational decision path D — geographic disruption

1. Define a disruption scenario and its boundary.
   Specify the affected area, start time, infrastructure condition, and uncertainty.
   Consider the safety of affected people before production targets.
2. Identify the potentially exposed physical facilities.
   Supplier headquarters and contract addresses do not necessarily identify manufacturing sites.
   Include critical sub-tier sites when the information is available.
3. Connect sites to parts and finished products.
   Record versioned part numbers, manufacturing steps, and model applications.
   Mark unknown links rather than assuming no exposure.
4. Check the timing of the mapping data.
   A plant that previously made a component may have transferred the work.
   Obtain a current confirmation from the relevant supplier contact where possible.
5. Determine usable inventory.
   Identify stock in transit, in inspection, at the supplier, and at the receiving plant.
   Exclude inventory that is damaged, obsolete, reserved, or incompatible.
6. Estimate consumption under the response plan.
   Normal run rates may not hold if demand, product mix, or other components change.
   Model the likely range rather than treating one daily average as certain.
7. Estimate time to recover usable supply.
   Facility repair is only one step.
   Utilities, worker safety, transport, material, quality release, and customer approval may all matter.
8. Compare recovery time with coverage.
   If coverage expires first, identify when output would become constrained.
   Consider multiple scenarios because both coverage and recovery estimates are uncertain.
9. Evaluate feasible actions.
   Options include allocating inventory, adjusting schedules, qualifying another source, and supporting recovery.
   Each option has costs, lead times, and decision rights.
10. Escalate before the reserve is exhausted.
    Assign triggers to a responsible person and establish a communication cadence.
    Record what is known, unknown, and due for confirmation at each review.
11. Measure recovery against a defined baseline.
    A restarted machine, accepted components, and customer deliveries are different milestones.
    Record each separately so a restart announcement is not mistaken for full recovery.
12. Preserve the learning.
    Compare actual response times with the initial scenario.
    Update the mapping and playbook when the network or product design changes.

### Operational decision path E — aerospace and semiconductor constraints

1. Define the constrained end product and time bucket.
   An aircraft delivery, a qualified engine, and a finished wafer are not equivalent outputs.
   Different bottlenecks can dominate in different periods.
2. Map requirements down to the critical component.
   Include part revision, production location, process, and required certification or qualification.
   A family name alone is not enough to establish interchangeability.
3. Separate open orders from executable capacity.
   A purchase order expresses a request or commitment under its terms.
   It does not prove that labor, tooling, material, and quality capacity are available.
4. Identify the constraint at each operation.
   Consider skilled labor, test capacity, foundry allocation, packaging, and transportation.
   Record which evidence supports each alleged constraint.
5. Evaluate upstream dependencies.
   Two direct suppliers can rely on the same foundry, special process, or raw material.
   A network map should flag these common nodes.
6. Inspect inventory by compatibility and status.
   Separate accepted parts from those awaiting inspection or engineering disposition.
   A substitute with the wrong specification cannot cover a shortage of the required part.
7. Estimate coverage for each scenario.
   Divide usable units by the scenario's daily consumption only when the demand rate is appropriate.
   Recalculate for changes in product mix and incomplete days.
8. Assess the lead time of alternatives.
   Include design evaluation, testing, approval, contracting, production, and delivery.
   A source that can make a sample is not necessarily ready for production volume.
9. Respect independent quality gates.
   A schedule or financial target does not authorize a change to an approved configuration.
   Route exceptions to engineering and quality authorities.
10. Compare mitigation portfolios.
    A buffer can bridge a finite interruption.
    Capacity reservation, redesign, or qualification may address different durations and constraints.
11. Include interactions between interventions.
    More finished inventory may compete for scarce working capital.
    A design change may consume engineering capacity while reducing future dependence.
12. Report the evidence boundary.
    A shortage survey documents reported conditions for its respondents.
    A local mitigation's effectiveness requires local measurement or a defensible comparative design.

### Worked micro-scenario — chip coverage

The values in this micro-scenario are synthetic and do not describe any named company.
An assembly plant uses 180 compatible chips per day at the planned production mix.
It holds 3,600 accepted chips and has no reliable incoming supply for the next 30 days.
The 3,600 accepted chips cover 3,600 / 180 = 20 days at the assumed rate.
The plant therefore faces a 10-day gap if demand and recovery follow the scenario.
It would need 1,800 additional compatible chips to cover that gap at the same rate.
This is a conditional arithmetic result, not a forecast of a real shortage.
The plan could change if assembly mix, usable stock, or delivery dates change.
The reserve target is a management choice; the 30-day outage is a scenario, not a commitment.
A proposed alternate must be qualified before its inventory is counted as usable.
If disruption lasts 60 days, the 20-day reserve covers only one-third of that duration.
That sensitivity illustrates why a finite reserve cannot by itself solve a persistent supply deficit.

### Worked micro-scenario — recovery milestones

The following dates are synthetic and are intended only to distinguish measures.
Assume a supplier stops production on day 0.
Power returns on day 5, one line restarts on day 8, and the first lot passes inspection on day 11.
The first accepted delivery reaches the customer on day 14.
Plant restart time is eight days, while accepted-delivery recovery time is 14 days.
Neither number alone describes a return to the pre-disruption daily output rate.
If the customer has 10 days of usable coverage, restart on day 8 is not enough to prevent a gap.
The four-day gap is an illustrative consequence under constant consumption and no other receipts.
The appropriate response depends on safety, substitution approvals, and feasible inventory allocation.


## 6. Worked example

### Scenario and boundaries

The following is a **synthetic** CPG case, not an observation about any named company.
A manufacturer considers a four-week retailer promotion and a replenishment pilot for one shelf-stable product.
Currency is U.S. dollars (USD); volumes are consumer units.
The evaluation window includes four promotion weeks and two post-promotion weeks.
The retailer and supplier have agreed a fixed store group and a comparable control group.
Figures below are assumed incremental estimates, not randomized trial results.

The promotion forecast is 12,400 units sold in the treated stores during the window.
The estimated no-promotion baseline is 9,800 units for the same stores and period.
The promotion plan is a temporary retail discount supported by a manufacturer allowance.
The target is a positive incremental contribution without reducing availability below the agreed threshold.
The only commitment so far is to review the proposal; the expenditure is not yet approved.

Assume the product's manufacturer net contribution before the promotional allowance is USD 3.20 per unit.
This contribution already deducts ordinary variable production and distribution costs.
The allowance is USD 0.40 for each of the 12,400 treated units, including units that would have sold anyway.
The retailer charges a fixed USD 1,200 execution fee for the window.
Estimated displaced contribution from another product is USD 900.
Estimated lost contribution from post-window pull-forward is USD 640.
These amounts are not included in the baseline or unit contribution and must be deducted separately.

### Arithmetic

Incremental units = 12,400 − 9,800 = 2,600 units.
Gross incremental contribution = 2,600 × USD 3.20 = USD 8,320.
Allowance = 12,400 × USD 0.40 = USD 4,960.
Total incremental contribution = USD 8,320 − USD 4,960 − USD 1,200 − USD 900 − USD 640.
The result is USD 620 over the six-week assessment window.
This is a modeled contribution, not a measured causal result or company profit.
Fixed overhead and taxes are outside the stated boundary.

Break-even incremental units, holding all other inputs fixed, are:

(USD 4,960 + USD 1,200 + USD 900 + USD 640) / USD 3.20 = 2,406.25 units.
At whole-unit precision, at least 2,407 incremental units are needed.
The allowance is held fixed at the assumed promoted volume for this sensitivity calculation.
If promoted volume changes, the per-unit allowance changes too and the full equation must be recalculated.

Consider uncertainty in the baseline.
If the baseline is 10,400 units rather than 9,800, incremental volume is 2,000 units.
At the same 12,400 promoted units, the result becomes USD 6,400 − USD 7,700 = −USD 1,300.
If the baseline is 9,200 units, incremental volume is 3,200 units.
The result becomes USD 10,240 − USD 7,700 = USD 2,540.
The decision is sensitive to baseline error, not just to the headline sales lift.
A reasonable next step is an appropriately designed experiment, not claiming the USD 620 is certain.

### Replenishment decision

Suppose expected consumption over a seven-day lead time plus review period is 1,150 units.
The safety buffer is 240 units, set for the example rather than derived from a service model.
The order-up-to level is 1,390 units.
Suppose usable on-hand stock is 710 units, eligible on-order stock is 340 units, and backorders are 60 units.
Inventory position is 710 + 340 − 60 = 990 units.
The unconstrained suggested order is 1,390 − 990 = 400 units.
If the supplier has an agreed case pack of 24 units, rounding up gives 17 cases, or 408 units.
Confirm shelf life, capacity, existing receipts, and contractual authority before release.
A stale or duplicated on-order feed can invalidate the recommendation.
A POS stockout can make observed sales lower than unconstrained demand.
The pilot should compare availability and inventory against control locations rather than claim that the arithmetic alone creates value.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the synthetic promotion and replenishment calculations above.
It treats contribution and costs as whole cents to avoid binary floating-point rounding.
It assumes the allowance applies to all treated units and that displacement and pull-forward are separate costs.
The program is a transparent calculation, not a forecasting model or an automated purchasing authorization.

```python
from decimal import Decimal, ROUND_CEILING

# Synthetic six-week promotion inputs; monetary values are USD.
promoted_units = 12_400
baseline_units = 9_800
unit_contribution = Decimal('3.20')
allowance_per_unit = Decimal('0.40')
execution_fee = Decimal('1200')
cannibalization = Decimal('900')
pull_forward = Decimal('640')

incremental_units = promoted_units - baseline_units
gross = Decimal(incremental_units) * unit_contribution
allowance = Decimal(promoted_units) * allowance_per_unit
other_costs = execution_fee + cannibalization + pull_forward
net = gross - allowance - other_costs
break_even = ((allowance + other_costs) / unit_contribution).to_integral_value(
    rounding=ROUND_CEILING
)

# Synthetic replenishment inputs, all in sellable units.
expected_demand = 1_150
safety_buffer = 240
on_hand = 710
on_order = 340
backorders = 60
case_pack = 24
order_up_to = expected_demand + safety_buffer
position = on_hand + on_order - backorders
raw_order = max(0, order_up_to - position)
cases = (raw_order + case_pack - 1) // case_pack
order_units = cases * case_pack

print(f'Incremental units: {incremental_units:,}')
print(f'Gross incremental contribution: USD {gross:,.2f}')
print(f'Allowance: USD {allowance:,.2f}')
print(f'Net incremental contribution: USD {net:,.2f}')
print(f'Break-even incremental units: {break_even:,}')
print(f'Inventory position: {position:,} units')
print(f'Raw order: {raw_order:,} units')
print(f'Rounded order: {cases} cases / {order_units:,} units')
```

Expected output:

```text
Incremental units: 2,600
Gross incremental contribution: USD 8,320.00
Allowance: USD 4,960.00
Net incremental contribution: USD 620.00
Break-even incremental units: 2,407
Inventory position: 990 units
Raw order: 400 units
Rounded order: 17 cases / 408 units
```

To adapt this calculation, first agree the commercial settlement basis.
For example, an allowance based on shipments instead of retail sales needs shipment quantities and a different exposure calculation.
If the baseline includes post-promotion demand already, do not deduct the same pull-forward a second time.
If the on-order feed contains late or canceled orders, exclude those quantities according to a documented rule.
A case-pack rounding rule must also respect shelf life, minimum order, maximum inventory, and authorized exceptions.

## 8. Data, standards, and regulatory context

Data are not interchangeable just because they describe the same product.
A transaction record, a retailer invoice, a supplier shipment, and a physical count represent different events.
The joins between them are an operational control, not a clerical detail.

### Product and location identity

A Global Trade Item Number (GTIN) identifies a trade item under GS1 conventions.
A consumer unit and a case can have different identifiers and conversion factors.
Store a time-valid hierarchy so that an old transaction is not silently mapped to a newly configured pack.
A location identifier should distinguish store, distribution center, supplier, and physical production site.
GS1 identifiers facilitate communication; adopting them does not by itself certify inventory accuracy or legal compliance.

### Commercial and POS records

Capture retail sales date, quantity, shelf price, promotion indicator, and returns.
Capture contract terms, eligible products, stores, settlement period, and evidence of execution.
Link allowances and claims to invoices and credit notes.
Trade terms may be confidential and subject to contractual access limits.
Restrict access to the minimum necessary group and retain the agreed audit trail.
The FTC's 2003 slotting study is descriptive; it is not a ruling that every allowance is lawful or unlawful.
Competition and trade-practice questions depend on jurisdiction and facts; seek qualified legal advice.

### Manufacturing and traceability

A bill of materials connects an end item to components and approved revisions.
A routing connects a part to process steps and equipment.
A lot or serial record links material to specific production and quality events.
Store effective dates and the identity of the approving authority.
For aviation, applicable certification, design, production, and maintenance requirements must be verified for the actual part and jurisdiction.
A suggested alternate part is not automatically approved for flight use.
A supply-network dashboard cannot waive an engineering or regulatory release.

### Resilience and chips

Supplier master data alone may identify a contracting entity but not a wafer fabrication plant.
Map part number, plant, process, sub-tier supplier, geography, and the period for which each link is valid.
Record confidence and the date of verification; supplier disclosure may be incomplete.
Record inventory separately by compatible component, ownership, location, and available quantity.
A substitute must be assessed for electrical, packaging, software, qualification, and customer requirements.
The Commerce request for information is a historical snapshot of voluntary responses, not a real-time inventory database.

### Privacy, security, and retention

Aggregate or de-identify person-level records when they are not needed.
Apply relevant privacy laws and contract terms to personal data where applicable.
Limit supplier-sensitive information to authorized recipients.
Record retention periods, access, corrections, and incident contacts.
These practices are controls; their exact legal requirements depend on jurisdiction and facts.

## 9. Industry practice and case evidence

### A. Continuous replenishment — P&G and Walmart

**Source:** the teaching case “Procter & Gamble: Improving Consumer Value through Process Redesign” describes a supplier-retailer relationship and process redesign.
Its authors prepared the case for class discussion, not as a controlled causal study.
It is a source for the historical partnership and its described practices, not a general proof of quantified savings.
The company-affiliated account by Grean and Shaw describes information sharing between the companies.
The accounts concern a specific partnership and period, not all modern retailers.

**Mechanism hypothesis:** shared consumption and inventory information can align replenishment decisions with demand.
Necessary conditions include dependable identifiers, timely data, explicit decision rights, and product availability.
The retailer may retain approval or exception authority; inventory ownership must be specified separately.
Sales are censored during stockouts, and retailer promotions can generate legitimate demand spikes.
A transfer should compare the current ordering process with a controlled or credible comparison group.
Measure shelf availability, inventory, spoilage, order frequency, and total cost.
A supplier-operated replenishment process can shift work or stock between firms rather than eliminate cost.
No public source cited here establishes the synthetic example's results.

### B. Trade spend — FTC grocery slotting study

**Documented observation:** the FTC's 2003 staff study examined selected anonymous cases in five grocery categories and certain geographic areas.
Slotting allowances are one-time payments for initial shelf placement or warehouse access in that study.
The FTC reported variation across categories and warned that the small sample might not represent U.S. retailers generally.
Most surveyed retailers said the payments helped defray new-product introduction costs; this is a reported explanation, not a proven causal effect.
The report also distinguishes slotting from the broader set of promotional allowances and discounts.

**Transfer:** a category team can use the study to ask which payment buys what service.
The study does not prove any individual promotion is profitable or that all payments have the same competitive effect.
Build a settlement-level ledger and an incremental-demand design before choosing a budget.
Include retailer, supplier, and consumer perspectives without assuming their incentives coincide.
Calculate contribution net of all relevant costs and check whether the product displaced other sales.

### C. Toyota production system

**Company description:** Toyota's 2025 Form 20-F describes the Toyota Production System through jidoka and just-in-time.
Jidoka includes identifying abnormalities and stopping rather than passing problems downstream.
Just-in-time concerns producing the needed items at the needed time and quantity.
The description is evidence of the company's stated management principles, not an experiment isolating their effects.
It does not establish that every firm should minimize every inventory buffer.

**Transfer:** start with a specific process, defect mechanism, safe stop authority, and resolution path.
Observe whether the team can detect a meaningful abnormality and respond before the next process consumes the output.
Track escaped defects, time to respond, safety, throughput, and rework together.
A signal that operators cannot act on is not equivalent to jidoka.
A fragile external supply chain may require a deliberate buffer even when internal process waste is reduced.

### D. Tohoku earthquake and Toyota recovery

**Company-reported observation:** Toyota's 2011 annual-report feature says the March 11 earthquake affected suppliers in Tohoku and northern Kanto and temporarily halted domestic production.
Toyota reports approximately 800,000 units of lost production through June relative to its production plan.
It also reports June output at around 70% of normal overall across domestic and overseas plants, and volume recovery to roughly normal by July.
The same passage gives a June 10 forecast of recovery and remaining shortfall; those projections are not realized outcomes merely because they appear in the report.
The cited passage combines domestic and overseas production and does not identify a controlled counterfactual.
It does not quantify the causal contribution of inventory, mapping, supplier support, or any particular practice.

**Transfer:** locate indirect dependencies and rehearse response while avoiding hindsight certainty.
A supplier being outside the immediate disaster area does not establish independence from affected sub-tier sites.
Map the time to identify a disruption, the time to recover usable supply, and the authorized response.
Review human safety and infrastructure before pursuing production recovery.
Measure recovery against an explicit baseline and distinguish capacity restart from customer delivery.

### E. Boeing and the aviation manufacturing network

**Official observation:** GAO's 2024 report examines aviation manufacturing, including Boeing and Airbus, after the demand rebound.
GAO reports that 15 of 17 manufacturers interviewed said they or their suppliers had difficulty hiring enough skilled workers.
It also reports 15 manufacturers described difficulties procuring materials, including engines, semiconductors, and aluminum.
Nine of 15 component suppliers interviewed reported difficulty filling orders.
GAO describes its stakeholder interviews as a non-generalizable sample.
It notes that safety actions after a January 2024 in-flight fuselage failure affected Boeing production levels early in 2024.
The report describes multiple concurrent constraints; it does not isolate the effect of any one supplier-monitoring action.

**Transfer:** measure capacity at the constrained operation and qualified source, not just purchase-order totals.
A second source needs engineering approval, process capability, and capacity before it can protect output.
Separate labor availability, material delivery, quality acceptance, and certification constraints.
A production-rate target cannot override a safety or airworthiness requirement.
The GAO source concerns the 737/A320 and the wider network; it is not evidence about the causes of a particular 787 delay.

### F. Semiconductor shortage

**Official observation:** the U.S. Department of Commerce published findings from a 2021 semiconductor supply-chain request for information in January 2022.
Commerce reported that median inventory of key chips among responding consumers fell from 40 days in 2019 to less than five days in 2021.
The agency also reported higher demand among chips highlighted by responding buyers.
These are findings from respondents and selected products, not a random census of all semiconductor inventories.
A median is not the inventory of every company or part.
The finding documents tight conditions; it does not prove which intervention would have prevented a specific shortage.

**Transfer:** identify the exact part and qualification path before estimating coverage.
Measure demand, on-hand usable stock, open supply, capacity allocation, and lead-time uncertainty by part.
Compare a buffer, a qualified alternate, redesign, and reserved capacity under plausible disruption durations.
Do not assume a generic chip or a different fabrication node is immediately interchangeable.
A long-lasting structural shortfall needs capacity or demand changes, not a finite buffer alone.

### Cross-case comparison: the decision is the transferable unit

The six cases do not all describe the same kind of intervention.
Their business contexts also differ in demand frequency, qualification, shelf life, and the cost of disruption.
The table identifies the decision an analyst can investigate locally, not a proven benefit to copy.

| Case | Primary decision | Operational signal | Decision authority | Important local outcome |
|---|---|---|---|---|
| P&G–Walmart | When and how much to replenish | Store demand and inventory | Agreed buyer or supplier | Availability and total inventory cost |
| FTC grocery study | Whether to pay for placement and under what terms | Contract and category economics | Supplier and retailer negotiators | Incremental contribution and access |
| Toyota production principles | Whether to stop and address an abnormality | Process and quality observations | Trained operator and escalation chain | Quality, safety, flow, and response time |
| Tohoku recovery | How to restore supply after disruption | Site, part, and recovery status | Incident and production leaders | Accepted supply and customer recovery |
| Aviation manufacturing | How to resolve constrained aircraft and part output | Supplier capacity and quality status | Manufacturer, supplier, and quality authority | Safe, approved deliveries |
| Semiconductor shortage | How to allocate and protect compatible chip supply | Part-specific inventory and supply | Procurement, engineering, and production | Coverage and feasible production |

The replenishment signal can be observed every day but may arrive after an order cutoff.
An aviation production constraint may require months of qualification rather than a daily reorder.
A chip constraint can appear as a shortage of one part even when other components are abundant.
These differences affect the window in which an analysis is useful.

Use the table to ask where an action can actually intervene.
A promotion manager can adjust a future offer but cannot change an already settled contract retroactively.
A plant operator may stop a process but may not authorize a design change.
A buyer may expedite a shipment but cannot create capacity at a fully loaded fabrication facility.
The decision rights must match the proposed action.

### Comparison 1 — visibility and control

Visibility is information about the state of a system.
Control is the ability to choose and execute an action that changes that state.
The two are related but not equivalent.

In the P&G–Walmart partnership, shared information is relevant because it can support a replenishment decision.
The historical source does not prove that a new shared feed will improve every destination.
The receiving operation must still be able to process an order and handle exceptions.
If the supply network cannot deliver within the relevant horizon, a better signal alone may not prevent a stockout.

In the Tohoku case, the annual report identifies affected suppliers and a production interruption.
The report does not isolate the value of a particular visibility system.
For a local exercise, a map can help identify exposed parts, but its useful action depends on verified links and an available response.
If the only approved site is damaged, simply knowing that fact does not create a second qualified site.

In aviation, supplier monitoring can reveal a labor or material problem.
It cannot independently certify an alternate part or waive a quality requirement.
The GAO study reports multiple manufacturer responses, including increased oversight and new suppliers for some supplies.
It does not establish a causal estimate for any one response.

In semiconductors, a low days-of-supply number can be an important warning.
The warning must be connected to the exact compatible item and the time to replenish it.
The Commerce finding concerns a selected set of respondents and chips.
It cannot determine the correct reserve for a particular automobile, medical device, or aircraft.

### Comparison 2 — average performance and tail exposure

A normal-period average can conceal the effects of a rare interruption.
It can also conceal ordinary variation between locations or product families.
Therefore report a distribution or a set of scenarios alongside the average.

For store replenishment, report the fraction of item-location days with an availability problem.
Also inspect how severe the worst shortages were and whether they were concentrated in a few stores.
An average inventory reduction is not a complete benefit if service deteriorates for particular customers.
Conversely, occasional high inventory may reflect an intentional response to a known lead-time risk.

For promotions, an average lift across stores can combine highly effective and ineffective executions.
Separate pre-specified segments when sample sizes allow and report uncertainty.
Do not search dozens of segments after the fact and present the most favorable one as a confirmed finding.
Show the number of stores and the total incremental contribution alongside any average percentage.

For a critical component, average delivery time may be less relevant than the duration of a severe interruption.
Test whether usable inventory covers several explicitly stated disruption durations.
Include the possibility that the recovery distribution changes when a whole region is affected.
Do not infer an earthquake probability from the Toyota event alone.

For aviation, a delivery schedule can be constrained by one of many safety, quality, workforce, or material conditions.
The most visible constraint may change during a recovery.
The GAO interviews identify categories of reported constraints; they do not rank a universal bottleneck for every factory.
Local evidence is needed to decide which constraint is active.

### Comparison 3 — incentives and cost allocation

Each case includes more than one organization or function.
An intervention can improve one party's metric while moving costs or risk to another.
The analysis should identify the economic boundary before claiming system-wide savings.

Under shared replenishment, the retailer may carry less stock while the supplier carries more.
That is a transfer of inventory unless total stock or total cost falls.
Add the carrying, handling, transport, shortage, and administration costs that are within the agreed boundary.
Do not add a transfer payment as if it were a net system cost to both parties.
The distribution of benefits can still matter for negotiation even when a payment nets out in a joint view.

Under a promotion, an allowance is a cost to the supplier and may be income or price support to the retailer.
The effect on consumers depends on the retail price and availability actually observed.
The FTC study documents selected slotting arrangements and retailer explanations.
It does not establish how a particular local payment changes final consumer welfare.

Under a production stop rule, the local line may record less output in the short term.
The next process may avoid receiving defective material.
Measuring only the stopped line can punish the behavior the quality system asks workers to perform.
Include downstream rework, escaped defects, and safe restart alongside the stop duration.

Under resilience planning, the team paying for a reserve may not be the team that avoids a shutdown.
Make the budget owner and the beneficiary explicit.
Agree who owns excess or obsolete material if demand changes.
Without an allocation rule, an apparently attractive reserve can fail to receive authorization.

### Comparison 4 — what can and cannot be learned from the sources

The P&G–Walmart material is useful for understanding a documented partnership and a proposed mechanism.
It cannot provide an unbiased estimate for a different supplier, assortment, and retailer.
Before transfer, examine whether the destination shares reliable item and inventory data.
Then test whether the resulting recommendation changes orders and service.

The FTC report is useful because it examines actual selected grocery cases and describes variation.
The agency explicitly cautions against treating its small sample as representative.
Before transfer, inspect the agreement and the relevant market rather than assuming a standard fee.
Then evaluate the specific business decision and obtain legal review where appropriate.

Toyota's production-system disclosure explains stated principles.
It is not a controlled evaluation comparing identical factories with and without those principles.
Before transfer, identify a process where workers can observe an abnormality and safely stop it.
Then measure whether containment and resolution occur in practice.

Toyota's earthquake account provides a dated company-reported chronology and production comparison with a plan.
It does not quantify the independent effects of all recovery measures.
Before transfer, compare the destination's exposure, suppliers, and recovery options.
Then rehearse a scenario without assuming the historical recovery rate will recur.

GAO's report combines data analysis and stakeholder interviews on aviation supply challenges.
The interview sample is expressly non-generalizable.
Before transfer, identify whether a proposed aerospace lesson concerns labor, materials, quality, or approved configuration.
Then verify that the corresponding constraint exists in the destination.

Commerce's request for information is useful as a historical indicator of respondent-reported chip conditions.
Its reported median cannot be applied as an individual company's stock position.
Before transfer, calculate coverage for the exact compatible component in the destination.
Then compare that coverage with a range of replenishment and recovery scenarios.

### Transfer worksheet: a recommendation with explicit gates

Use the following worksheet when a sponsor requests that a famous case be replicated.
Complete it for one decision, one destination, and one evaluation window.
If a gate fails, revise the proposal or record why it should not proceed.

**Gate 1: outcome and baseline**

- State the proposed outcome in units, currency, or an operational rate.
- State the time period and population to which the outcome applies.
- Identify the current process that the proposed action would replace.
- Explain how the no-action outcome will be estimated.
- Record whether the source provides an actual outcome or only a management description.

**Gate 2: mechanism**

- State the sequence from signal to action to physical or commercial result.
- Identify the person or system authorized to make the decision.
- Identify the point at which an operator can override or stop the action.
- Name the operational dependency that could prevent the action from working.
- Label the proposed causal chain as a hypothesis until locally evaluated.

**Gate 3: comparable operating conditions**

- Compare product substitutability and the consequence of an incorrect decision.
- Compare demand frequency, lead time, and the need for qualification.
- Compare the role of the retailer, supplier, manufacturer, and customer.
- Compare the degree of shared exposure across supplier sites.
- Document differences that make the source an analogy rather than a close match.

**Gate 4: data and execution**

- Name each required dataset and its owner.
- State item, location, time, and unit definitions.
- Set a freshness rule and the response to missing data.
- Verify that the planned use is permitted under applicable agreements.
- Explain how an observed recommendation becomes an approved operational action.

**Gate 5: evaluation and safeguards**

- Specify the comparison or randomization method.
- Define service, cost, safety, quality, and risk guardrails where applicable.
- Include the period after the immediate intervention when effects can be delayed.
- Specify how deviations and missing observations will be handled.
- Name the person who can stop, extend, or scale the pilot.

### A synthetic transfer decision: one intervention, two destinations

Assume a supplier proposes using daily retailer data to recommend orders for two product families.
All quantities and costs in this example are synthetic.
The first family consists of a stable, shelf-stable product at 40 stores.
The second family consists of short-life chilled products with variable store deliveries.
The sponsor cites the P&G–Walmart partnership as the reason to deploy the same rules everywhere.

For the shelf-stable family, assume valid sales and inventory data arrive before the daily cutoff.
The stores use a consistent item identifier and can accept a proposed replenishment order.
An eight-week controlled pilot can compare service, average usable inventory, and incremental transport cost.
This is a plausible test of the information-to-order mechanism.
It does not guarantee that the pilot will generate savings.

For the chilled family, assume stock counts arrive two days late and expiry is not included in the feed.
The same order rule could recommend stock that cannot be sold before its expiration.
The appropriate immediate decision is to fix the data and define a usable-life rule before testing.
The destination has not failed because the historical partnership was invalid.
It has failed a prerequisite for this particular transfer.

Suppose the shelf-stable pilot estimates a USD 9,000 reduction in inventory carrying cost over eight weeks.
Suppose incremental deliveries cost USD 3,400 and extra manual review costs USD 1,100 over the same period.
The net modeled cost reduction is USD 9,000 − USD 3,400 − USD 1,100 = USD 4,500.
This arithmetic assumes the three estimates refer to the same eligible stores and do not overlap.
It excludes any monetized service benefit and any fixed implementation cost.
If service is below a pre-agreed guardrail, a positive modeled cost reduction does not automatically authorize scaling.
If the USD 9,000 already includes transport, deducting transport again would double-count the cost.

Now assume the pilot's USD 9,000 estimate has substantial uncertainty.
The team should report the comparison design and interval or range alongside the point estimate.
It should also disclose whether other interventions changed during the pilot.
A short pilot may not capture seasonal peaks, supplier failures, or renewal of commercial terms.
The scale decision should therefore distinguish the observed period from the intended future deployment.

### A synthetic transfer decision: when the reserve is insufficient

Assume a manufacturer needs one qualified chip per unit of a finished product.
The plant expects to make 250 finished units per day under its current plan.
It has 5,000 accepted chips allocated to the product and no confirmed deliveries.
Coverage is 5,000 / 250 = 20 days at that planned rate.
The production plan is not a forecast of actual output if other constraints intervene.

Consider a 12-day interruption and a 35-day interruption.
In the first scenario, the reserve covers demand if the assumptions hold and chips remain usable.
In the second, the reserve is exhausted 15 days before the assumed supply recovery.
At 250 chips per day, bridging the uncovered interval requires 3,750 additional compatible chips.
This quantity is a scenario calculation, not proof that those chips can be purchased.

If another supplier can provide compatible chips in 10 days and is already approved, it may cover the gap.
If qualification takes 90 days, that same supplier cannot bridge the 35-day event under this schedule.
If both suppliers rely on the same disrupted fabrication plant, their apparent independence is misleading.
Check the site and qualification facts before treating the alternative as available capacity.
An allocation decision should also account for other products and contractual commitments competing for the same stock.

For longer disruptions, reassess the intervention rather than extrapolating the reserve indefinitely.
Consider qualification, design changes, capacity agreements, and demand prioritization where feasible.
Engineering and quality authorities must approve any substitution.
The Commerce survey motivates asking about exposure; it does not establish this synthetic plant's answer.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Shipment mistaken for consumption | Orders rise while store sales do not | Channel inventory builds | False promotion success | Reconcile sell-in and sell-through |
| Stockout-censored POS | Zero sales with empty shelf | Missing demand appears as no demand | Under-ordering | Capture availability and estimate lost demand |
| Stale inventory | Late or negative positions | Decisions use an incorrect state | Excess or shortage | Freshness gate and physical reconciliation |
| Duplicate orders | Same receipt appears twice | Inventory position overstated | Under-ordering | Idempotent order identifiers |
| Pack conversion error | Case and unit totals disagree | Wrong factor in item hierarchy | Mis-sized order | Time-valid unit conversions |
| Allowance applied to lift only | Settlement exceeds analysis | Fee applies to all eligible units | Overstated contribution | Model contract settlement basis |
| Pull-forward omitted | Post-event volume falls | Purchases shifted earlier | Spurious lift | Include a post-event window |
| Cannibalization ignored | Related product declines | Customer switches within portfolio | Inflated net value | Evaluate portfolio contribution |
| Confounded control | Different promotions run in control | Counterfactual is not comparable | Misleading causal result | Randomize or document adjustment |
| Stop signal without authority | Operators bypass alarms | Incentive or policy discourages stopping | Defects propagate | Give clear stop and escalation rights |
| JIT interpreted as zero stock | One supplier delay stops assembly | Buffer removed without recovery plan | Lost throughput | Size buffer to risk and recovery |
| Outdated sub-tier map | Supplier cannot verify site | Network changes are unrecorded | Slow incident response | Refresh links and confidence |
| Nominal alternate unqualified | Alternate lacks approval | Source is not usable | False resilience | Complete qualification and capacity checks |
| Common-mode sourcing | Two suppliers share one plant | Apparent diversity hides concentration | Simultaneous disruption | Map shared sub-tier facilities |
| Aircraft rate overrides quality | Open nonconformances increase | Schedule pressure dominates gates | Safety and delivery risk | Independent quality release |
| Generic chip substitution | Similar part fails integration | Electrical or software incompatibility | Redesign and qualification delay | Validate full specification |
| Buffer covers wrong horizon | Reserve expires before restart | Disruption exceeds coverage | Production stops anyway | Scenario-test recovery durations |
| Double-counted benefits | Finance cannot reconcile bridge | Same savings appear in several lines | Inflated business case | Assign cost ownership once |
| Vendor success story extrapolated | No baseline or denominator | Marketing substitutes for comparison | Bad local commitment | Demand scope, controls, and raw definitions |
| Missing safety owner | No escalation contact | Decision ignores operational constraints | Unsafe or noncompliant action | Name accountable authority before pilot |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Local test |
|---|---|---|---|
| Eligible units are identifiable | Apply commercial terms accurately | Item or store hierarchy changes | Reconcile a sample to invoices |
| Baseline is credible | Estimate incremental demand | Unobserved concurrent events | Inspect pre-trends and randomized balance |
| Post-window is adequate | Detect shifted demand | Purchase cycles exceed six weeks | Extend follow-up by category cycle |
| Unit contribution is stable | Translate volume into value | Overtime or freight changes with volume | Use volume-dependent costs |
| Allowance applies to all promoted units | Compute the worked example | Contract applies to shipments or tiers | Read executed agreement |
| Inventory position is accurate | Derive an order | Duplicates, damage, or late receipts | Reconcile physical and system records |
| Case packs are feasible | Round to an order | Shelf-life or capacity constraint | Check storage and expiry |
| Alternative supplier is usable | Reduce concentration | Unapproved process or shared sub-tier | Verify qualification and genealogy |
| Historical recovery is comparable | Use an event as a scenario | Different geography, infrastructure, or disruption | Compare network and hazard drivers |
| Survey respondents represent the target | Extrapolate a survey statistic | Voluntary, selected sample | Inspect sampling and composition |
| Stop authority is real | Translate detection into action | Production incentives penalize stoppage | Audit actual escalations |
| Reported results are causal | Estimate an intervention effect | No counterfactual or multiple simultaneous changes | Demand a credible design |

None of these assumptions is guaranteed by a prestigious company name.
A missing assumption should become a pilot prerequisite or an explicit limitation.
The 2011 Toyota production loss is relative to its plan, not a measurement of demand lost forever.
The 2022 Commerce inventory finding describes the request's respondent data and selected chips.
The GAO interview counts are not population prevalence estimates.
The FTC's 2003 category sample is not a current universal price schedule.

## 12. Exercises

### Exercise 1 — Claim classification

Time: 15 minutes.
Deliverable: a four-row ledger for Toyota production principles and the FTC's slotting observations.
Classify each row as company description, official observation, reported respondent explanation, or inference.
Acceptance: every row has a specific source, scope, and one thing that the source does not establish.

### Exercise 2 — Earthquake chronology

Time: 15 minutes.
Deliverable: a timeline containing March, June, and July 2011 and the June 10 forecast.
Acceptance: distinguish company-reported production observations from the projection and name the units.
Explain why the timeline cannot establish the effect of a single resilience measure.

### Exercise 3 — Inventory position

Time: 15 minutes.
Deliverable: calculate a suggested order for expected lead-time demand of 870 units, safety stock of 180 units, on-hand of 490 units, on-order of 260 units, and backorders of 30 units.
Use a 20-unit case pack and round upward.
Acceptance: show inventory position, order-up-to level, raw order, and final cases.
List two checks before release.

### Exercise 4 — Promotion economics

Time: 20 minutes.
Deliverable: calculate incremental contribution for 8,000 treated units, a 6,500-unit baseline, USD 4.00 contribution per incremental unit, a USD 0.50 allowance on all treated units, and USD 1,200 fixed execution cost.
Assume no other effects and a four-week period.
Acceptance: show incremental units, allowance, and net contribution; explain whether this is actual causal evidence.

### Exercise 5 — Compare constrained systems

Time: 20 minutes.
Deliverable: a side-by-side comparison of the GAO aviation findings and the Commerce chip findings.
Acceptance: state a source, decision, sampling limitation, and one non-interchangeable constraint for each.
Do not claim either report proves a specific intervention works.

### Exercise 6 — Design a replenishment pilot

Time: 30 minutes.
Deliverable: a one-page pilot charter for 30 stores, including treatment and comparison assignment.
Acceptance: name the ordering owner, data freshness gate, service and inventory metrics, stop rule, and cost boundary.
Explain how promotions and stockouts affect the analysis.

### Exercise 7 — Design a resilience response

Time: 30 minutes.
Deliverable: a response plan for a single-source chip shared by two vehicle models.
Acceptance: map direct and sub-tier dependencies, inventory by compatible part, recovery scenarios, and qualification requirements.
Give an accountable owner, trigger, and stop condition for each action.
Distinguish short-duration buffering from long-duration capacity remedies.

### Exercise 8 — Audit the commercial data contract

Time: 25 minutes.
Deliverable: a data dictionary and claim reconciliation procedure for one retailer promotion.
Acceptance: define item, store, period, price, units, allowance, settlement identifier, and ownership.
Specify how to handle late claims, returns, data permissions, and conflicting item versions.
State which local legal and contractual questions require expert review.

## 13. Self-check answers

### Exercise 1

Toyota's Form 20-F is a company description of its management system.
It does not isolate the system's incremental effect on quality or cost.
The FTC observed variation in slotting arrangements within selected grocery cases.
That observation does not establish prevalence across all U.S. retailers.
The surveyed retailers' cost-defraying explanation is a reported respondent view.
Whether fees actually covered those costs in each case requires further evidence.
The statement that a specific local slotting fee will increase profit is an inference, not a conclusion of the FTC study.

### Exercise 2

March 11: the earthquake struck and Toyota reports production and supplier disruption.
Through June: Toyota reports about 800,000 units of production loss against its plan.
June: Toyota reports overall production at around 70% of normal.
July: Toyota reports volume returned to approximately normal.
The June 10 projections of subsequent recovery and annual shortfall are forecasts at that date.
Multiple actions and outside factors changed together, and no untreated comparison identifies the contribution of one action.

### Exercise 3

Order-up-to level = 870 + 180 = 1,050 units.
Inventory position = 490 + 260 − 30 = 720 units.
Raw order = 1,050 − 720 = 330 units.
Rounding upward to a 20-unit case pack gives 17 cases, or 340 units.
Check whether on-order quantities are still expected and whether capacity and expiry allow the rounded quantity.

### Exercise 4

Incremental units = 8,000 − 6,500 = 1,500 units.
Gross incremental contribution = 1,500 × USD 4.00 = USD 6,000.
Allowance = 8,000 × USD 0.50 = USD 4,000.
Net incremental contribution = USD 6,000 − USD 4,000 − USD 1,200 = USD 800.
The arithmetic is correct only under the stated inputs and boundaries.
The baseline is assumed; no causal result follows until its validity is evaluated.

### Exercise 5

A strong answer cites GAO-24-106493 and identifies the non-generalizable stakeholder interviews.
It separates skilled labor, material access, quality, and certified configuration in aviation.
It cites Commerce's request and recognizes respondent and product selection in the chip finding.
It notes that part-specific compatibility and fabrication capacity limit immediate substitution.
It does not equate an interview count with a population percentage or a survey median with every firm's stock.

### Exercise 6 rubric

A strong charter randomizes comparable eligible stores when feasible.
It states the unit and time horizon for availability, inventory, and cost.
It rejects stale records before an order is released and names a human escalation owner.
It specifies a control, post-promotion follow-up, and a pre-agreed review of adverse service outcomes.
A weak charter chooses software and forecasts savings without a baseline or decision rights.

### Exercise 7 rubric

A strong response connects the part to physical manufacturing and assembly sites, not merely supplier names.
It tests common-mode exposure and records mapping confidence.
It specifies usable inventory, duration scenarios, and approved substitution and qualification owners.
It gives escalation for a reserve below its trigger and does not substitute an unapproved part.
A weak response assumes two contractual suppliers necessarily imply two independent plants.

### Exercise 8 rubric

A strong dictionary records keys, units, timestamps, version history, ownership, and allowable use.
It reconciles settlement to contractual eligibility and explains correction and late-arriving records.
It separates sales, shipments, returns, invoices, and claim transactions.
It assigns legal review instead of making a universal competition-law conclusion.

## 14. Interview questions

1. Why is a rise in supplier shipments not sufficient evidence of promotion success?
   A strong answer separates channel fill from consumer sell-through and checks later stock.
   A weak answer assumes every shipped unit represents new consumption.
2. What must be true for a shared POS feed to improve replenishment?
   A strong answer names data latency, item identity, stockout censoring, authority, and response time.
   A weak answer treats a dashboard as an order decision.
3. How do you compute the ROI of an allowance?
   A strong answer specifies incremental contribution, the allowance base, costs, and counterfactual.
   A weak answer divides gross sales by spend and calls it profit.
4. What does the FTC study say about slotting?
   A strong answer states its five-category selected-case scope and its nonrepresentative limitation.
   A weak answer says all slotting fees have the same outcome or legality.
5. What is the difference between jidoka and simply automating inspection?
   A strong answer includes abnormality detection, stopping authority, and resolution.
   A weak answer equates the principle with replacing people with machines.
6. What can be learned from Toyota's 2011 recovery?
   A strong answer reports the bounded chronology and distinguishes reported output from causal attribution.
   A weak answer attributes the entire recovery to one unmeasured intervention.
7. Why is a second chip supplier sometimes not a second source of risk?
   A strong answer traces shared fabrication, packaging, equipment, or materials and qualification.
   A weak answer counts supplier names without mapping sites.
8. How should the GAO aviation interview counts be interpreted?
   A strong answer states the interview denominators and the non-generalizable selection.
   A weak answer generalizes the counts to every manufacturer.
9. When can a safety stock buffer fail?
   A strong answer compares compatible usable stock with recovery duration and demand, including structural deficits.
   A weak answer assumes any positive inventory guarantees continuity.
10. What would justify scaling a case-inspired pilot?
    A strong answer requires local incremental evidence, safety and service guardrails, operational ownership, and a transfer assessment.
    A weak answer cites the original company name or a forecast as proof.

## 15. Key terms

- [Availability](#5-the-mechanics-step-by-step): whether an item is available for the decision or customer.
- [Baseline](#4-definitions-and-vocabulary): estimate of outcome without intervention.
- [Bill of materials](#8-data-standards-and-regulatory-context): versioned component structure.
- [Bottleneck](#4-definitions-and-vocabulary): constraint on throughput.
- [Case pack](#6-worked-example): units in an ordering package.
- [Cannibalization](#4-definitions-and-vocabulary): displaced contribution from another product.
- [Commitment](#4-definitions-and-vocabulary): authorized obligation.
- [Continuous replenishment](#9-industry-practice-and-case-evidence): frequent replenishment using shared information and rules.
- [Contribution](#6-worked-example): revenue less specified variable costs.
- [Counterfactual](#3-the-core-mental-model): outcome without the intervention.
- [Demand censoring](#10-failure-modes-and-red-flags): sales fail to reveal unmet demand.
- [Disruption](#9-industry-practice-and-case-evidence): event interrupting normal supply or production.
- [Forecast](#6-worked-example): conditional prediction.
- [GTIN](#8-data-standards-and-regulatory-context): GS1 trade-item identifier.
- [Incremental units](#6-worked-example): treated units minus baseline units.
- [Inventory position](#4-definitions-and-vocabulary): on hand plus eligible on order minus backorders.
- [Jidoka](#9-industry-practice-and-case-evidence): identifying abnormalities and stopping to address them.
- [Just-in-time](#9-industry-practice-and-case-evidence): producing needed items when and in quantities needed.
- [Lead time](#4-definitions-and-vocabulary): elapsed time to usable receipt.
- [Order-up-to level](#6-worked-example): desired inventory position.
- [Plan](#6-worked-example): proposed actions and resources.
- [POS](#8-data-standards-and-regulatory-context): retail transaction data.
- [Pull-forward](#4-definitions-and-vocabulary): demand shifted to an earlier period.
- [Qualification](#11-assumptions-and-limits): approval of a source or process.
- [Sell-in](#4-definitions-and-vocabulary): supplier shipments into the channel.
- [Sell-through](#4-definitions-and-vocabulary): consumer purchases from the channel.
- [Slotting allowance](#9-industry-practice-and-case-evidence): payment for initial placement or warehouse access.
- [Sub-tier](#8-data-standards-and-regulatory-context): supplier beyond the direct contracting relationship.
- [Target](#6-worked-example): desired outcome.
- [Trade spend](#4-definitions-and-vocabulary): supplier spending under retailer agreements.
- [Transfer](#3-the-core-mental-model): application of a mechanism to another context.

## 16. Sources

Sources support only the specific claims attributed to them above.
Official reports and company publications are primary evidence for their own observations or statements; neither automatically establishes causality.

1. McKenney, James L., and Theodore H. Clark. [*Procter & Gamble: Improving Consumer Value through Process Redesign*](https://www.hbs.edu/faculty/Pages/item.aspx?num=4673), Harvard Business School case 9-195-126, 1995. Teaching case; historical description, not a randomized impact evaluation.
2. Grean, Michael, and Michael J. Shaw. [*Supply-Chain Integration through Information Sharing: Channel Partnership between Wal-Mart and Procter & Gamble*](https://www.sobtell.com/images/questions/1496388650-week_9.pdf). Company-affiliated account of the partnership; treat reported benefits as claims, not independently identified effects.
3. U.S. Federal Trade Commission. [*Slotting Allowances in the Retail Grocery Industry: Selected Case Studies in Five Product Categories*](https://www.ftc.gov/sites/default/files/documents/reports/use-slotting-allowances-retail-grocery-industry/slottingallowancerpt031114.pdf), 2003. See also the [FTC release and sample limitations](https://www.ftc.gov/news-events/news/press-releases/2003/11/ftc-releases-grocery-industry-slotting-allowance-report).
4. Toyota Motor Corporation. [*Form 20-F*, fiscal year ended March 31, 2025](https://global.toyota/pages/global_toyota/ir/library/sec/20-F_202503_final.pdf), Toyota Production System section. Company description of principles.
5. Toyota Motor Corporation. [*Annual Report 2011*, “Restore and Renew Our Production Structure for Further Growth”](https://www.toyota-global.com/pages/contents/investors/ir_library/annual/pdf/2011/p21.pdf). Company-reported earthquake impacts, recovery, and dated forecasts.
6. U.S. Government Accountability Office. [*Commercial Aviation Manufacturing: Supply Chain Challenges and Actions to Address Them*](https://www.gao.gov/products/gao-24-106493), GAO-24-106493, March 6, 2024. Official report with non-generalizable interviews.
7. U.S. Department of Commerce. [*Results from Semiconductor Supply Chain Request for Information*](https://commerce.gov/news/blog/2022/01/results-semiconductor-supply-chain-request-information), January 25, 2022. Agency summary of respondent data; not a census.
8. GS1. [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin). Identification standard; no guarantee of local master-data quality.
9. Hernán, Miguel A., and James M. Robins. [*Causal Inference: What If*](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/), Chapman & Hall/CRC, 2020. Methodological reference on counterfactual inference, not a supply-chain case study.

## 17. Links

- [Module 28 topic index](_index.md).
- [Case anatomy](01_case-anatomy.md): structure a decision and source record.
- [Evidence grades](02_evidence-grades.md): assess the evidence supporting individual claims.
- [Transfer test](03_transfer-test.md): decide which local conditions must hold.
- [Selection bias](04_selection-bias.md): examine cases missing from the visible sample.
- [Sector playbooks](05_sector-playbooks.md): organize decisions and operating constraints.
- [Module 28 survey and Lab 28.2](../../../course/28_domain_case_library.md): apply the transfer test in a case portfolio.
