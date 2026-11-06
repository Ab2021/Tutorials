# Retail and E-commerce Cases: Decisions, Evidence, and Transfer

> Module 28 — Domain Case Library and Sector Playbooks  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 07, 10, and 26; topics 01–03  
> Version 2026.5

## 1. Why this topic exists

Retail leaders often inherit a persuasive story before they inherit a defined decision.
A competitor is said to forecast better, deliver faster, automate a warehouse, or reduce waste.
The underlying decisions, economics, and evidence are rarely interchangeable.
A demand forecast does not place inventory by itself.
A regional fulfillment network may reduce transport while increasing inventory duplication.
A low return rate is not automatically more profitable if it also reduces conversion.
A fast replenishment cycle is useful only when suppliers can respond responsibly.
A robotic center can increase throughput without earning a return on invested capital.
A grocery marketplace may improve picking efficiency without controlling shelf availability.

This unit examines six distinct cases through that lens.
The M5 forecasting competition is a public research benchmark, not an implementation case.
Amazon regionalization is a company-reported U.S. network redesign.
Zalando provides company descriptions of returns and resale operations.
Ocado describes an automated grocery platform and related technology.
SHEIN describes a demand-led apparel production model.
Instacart describes marketplace and retailer technology, including shopper support.
Each source has a different vantage point and a different incentive to present results favorably.

The manager's decision is therefore not whether to copy a famous organization.
It is whether a specific local intervention beats an available alternative under local constraints.
An analyst must connect an observed quantity to its population, period, denominator, and source.
A company statement is evidence of what the company reports, not automatically independent proof of causation.
Vendor descriptions demonstrate a product proposition, not guaranteed performance in another operation.
A simulation is an explicitly synthetic calculation, not evidence about a named company.

The examples below use public material as of the cited source's date or the text available when consulted.
Company web pages can change; preserve a dated copy when using them in a decision record.
This is educational material, not legal, labor, investment, or environmental assurance advice.

## 2. Learning outcomes

After this unit you can:

1. **Classify** each case as a benchmark, operational description, or interested-party claim; see exercise 1.
2. **Specify** a decision, owner, cadence, alternatives, and constraint for each case; see exercise 2.
3. **Compute** fulfillment and return economics without counting transfer payments as revenue; see exercises 3 and 4.
4. **Explain** why benchmark accuracy, fulfillment speed, and realized profit differ; see exercise 5.
5. **Design** a controlled pilot with guardrails, a comparison, and a stopping rule; see exercises 6 and 7.
6. **Evaluate** transfer across demand, data, network, labor, and customer conditions; see exercise 8.
7. **Write** a decision memo that separates evidence, claim, inference, and unknowns; see exercise 9.

A correct recommendation can be to defer deployment until missing evidence is collected.
A good answer names the conditions under which the recommendation would change.

## 3. The core mental model

```text
PUBLIC SOURCE --> scope + source incentives --> claim ledger
                                                   |
LOCAL DECISION --> owner + options + constraints   |
       |                                           |
       v                                           v
 DATA AT DECISION TIME --> model/workflow --> local experiment
       |                          |                |
       v                          v                v
 demand and availability     action actually    service, cost,
 are not identical           taken              risk, uncertainty
                                                   |
                                                   v
                          economics + transfer test + decision
```

Follow the arrow from a forecast to the operational action.
In M5, the recorded output is forecast accuracy, not replenishment savings.
In Amazon's example, the decision includes where goods sit before orders arrive.
In Zalando's example, a sale may generate a return, inspection, refurbishment, and resale.
In Ocado's example, a capital-intensive center changes capacity, labor, and delivery routing.
In SHEIN's example, an initial batch can reveal demand before additional production is committed.
In Instacart's example, a shopper must find or substitute items in a retailer's store.

Ask five transfer questions before recommending any of these patterns:

1. **Decision:** Is the receiving organization choosing the same action at the same cadence?
2. **Economics:** Are contribution, fixed cost, capacity, and risk measured on the same boundary?
3. **Data:** Are the necessary signals available at decision time, at the required grain?
4. **Constraints:** Are lead time, labor, privacy, safety, and contractual limits comparable?
5. **Evidence:** Does the source support the magnitude and attribution being proposed locally?

A partial match is not automatically a rejection.
It is an instruction to test the mismatched link rather than assume it away.
The final business case needs incremental outcomes against a feasible counterfactual.
A description of an industry leader is not that counterfactual.

### Evidence labels used here

- **Reported:** A source states a measurement or describes an action within a named scope.
- **Company claim:** An interested company asserts an outcome or causal explanation.
- **Marketing claim:** A seller describes a capability or benefit in promotional material.
- **Inference:** A reasoned interpretation that is not directly established by the source.
- **Synthetic:** Invented data or assumptions used to teach a calculation.
- **Unknown:** A material item that the cited source does not establish.

A research paper can describe a competition accurately while still not proving production value.
An official company statement can document its reported practice while leaving causality unverified.
The label attaches to a proposition, not an organization as a whole.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Stock-keeping unit (SKU) | Product variant tracked for stock; units per SKU. | Treating all colors and sizes as interchangeable. |
| Demand | Desired purchases during a period, including unmet demand. | Equating observed sales with unconstrained demand. |
| Sales | Completed or recorded units sold during a period. | Assuming zero sales means zero customer interest. |
| Forecast | Estimate of a future outcome at a stated origin and horizon. | Treating it as an approved plan or customer promise. |
| Plan | Chosen operational response with resource assumptions. | Assuming the forecast determines it uniquely. |
| Commitment | A promise or contractual obligation to a customer or supplier. | Treating a projected service level as a guarantee. |
| Horizon | Time from forecast origin to predicted period; days here. | Comparing forecasts with different horizons as if equivalent. |
| Forecast origin | Time at which a forecast is issued. | Using information observed after that time. |
| Backtest | Evaluation using historical origins and information available then. | Randomly splitting time-dependent transactions. |
| Leakage | Use of future or otherwise unavailable information. | Assuming a historically observed promotion was known in advance. |
| Hierarchy | Related series, such as item, store, region, and total. | Summing unrelated metrics across aggregation levels. |
| WRMSSE | Weighted root mean squared scaled error over hierarchical series. | Calling it a percentage or an inventory-cost metric. |
| Aggregation | Summing or grouping lower-grain observations to a higher level. | Ignoring that errors can cancel in totals. |
| Sell-through | Sales relative to available or received inventory over a stated period. | Omitting replenishments or the measurement window. |
| Stockout | Product unavailable when demand arrives. | Inferring lost demand precisely from zero sales alone. |
| Censoring | Observation constrained by availability or another mechanism. | Fitting sales as if unconstrained customer demand. |
| Regionalization | Serving geographic areas through a network with locally positioned capacity or inventory. | Assuming complete geographic isolation. |
| Pooling | Combining variability across demand streams or inventory locations. | Assuming aggregation removes all correlated shocks. |
| Fill rate | Fraction of requested units supplied, with a defined time and denominator. | Confusing it with fraction of orders delivered on time. |
| Order split | An order fulfilled through multiple shipments or locations. | Counting units rather than shipments in transport expense. |
| Lead time | Elapsed time between a triggering action and availability; days. | Using average lead time when tail risk drives service. |
| Return rate | Returned units or orders divided by an explicitly matched sales cohort. | Mixing item-level and order-level rates. |
| Cohort maturity | Enough time has elapsed to observe outcomes for a sales cohort. | Comparing a new cohort with a fully matured cohort. |
| Recovery | Value from resale or other disposition after a return. | Counting potential resale as already realized cash. |
| Refurbishment | Inspecting, cleaning, repairing, or preparing a returned unit for sale. | Assuming every returned unit can be sold at full price. |
| Contribution | Revenue less specified variable costs; currency per order or period. | Calling contribution full net profit. |
| Avoided cost | Expense that would otherwise have been incurred. | Counting a fixed salary as saved without capacity removal. |
| Customer fulfillment center (CFC) | Facility that picks and packs customer orders. | Assuming every grocery order uses a fully automated center. |
| Capacity utilization | Used capacity divided by available capacity in the same period. | Inferring high return on investment from high utilization alone. |
| Fixed cost | Cost not changing with small volume changes over the stated horizon. | Treating it as fixed forever or independent of capacity steps. |
| Variable cost | Cost changing with an activity driver over a stated range. | Ignoring labor minimums and volume discounts. |
| Unit economics | Revenue and relevant costs per consistent unit of activity. | Mixing cost per order with revenue per item. |
| Initial batch | First quantity made before a demand-response decision. | Calling a company-reported range a universal production rule. |
| Reorder | Additional production or purchasing after observing a signal. | Ignoring lead time, minimum quantities, and supplier capacity. |
| Substitution | Replacing an unavailable requested item with an alternative. | Assuming the customer accepts all substitutes. |
| Found rate | A defined share of requested items located; definition must be explicit. | Assuming all platforms use identical denominators. |
| Intent-to-treat | Analysis by assigned intervention regardless of actual use. | Excluding inconvenient non-adopters after assignment. |
| Counterfactual | Outcome that would have occurred under the alternative action. | Using a post-intervention outcome alone to infer causation. |
| Guardrail | Measured constraint that prevents improvement in one metric harming another. | Treating a guardrail as an optional afterthought. |
| Transfer test | Structured evaluation of whether a case's mechanism applies locally. | Copying a headline metric as a local target. |

## 5. The mechanics, step by step

### Step 1 — Convert the story into an owned decision

Record the decision maker and an actual choice.
Specify the period, geography, customer population, and units.
Name the alternative that is operationally possible today.
Record the decision deadline and its update cadence.
Separate decisions made before the customer orders from decisions made after the order.
Document which constraints cannot be traded away, such as temperature or product safety.

For M5, the analyst may choose a forecasting method for daily SKU–store demand.
The operational owner still has to decide order quantities and availability constraints.
For regionalization, a network planner chooses inventory locations and replenishment rules.
For returns, a commercial team chooses a sizing intervention or a disposition path.
For automated grocery, an investment committee chooses capacity and operating design.
For apparel, a merchant and supplier choose initial quantities and reorder thresholds.
For marketplace grocery, an operations team chooses how shoppers see shelf or substitute information.

**Input:** A stated business problem and the feasible actions.
**Output:** One sentence naming owner, action, cadence, population, and boundary.
**Branch:** If nobody can act on the forecast, stop before selecting a modeling technique.

### Step 2 — Capture the original case and source

Record the named source, publication date if available, URL, and retrieval date.
Copy the exact claim into a ledger before paraphrasing it.
Preserve denominators: units versus orders, processed returns versus all sales, and so on.
State whether the source is a paper, a company disclosure, or a supplier's promotional page.
Identify its commercial interest without assuming that every statement is false.
Label the difference between what the source reports and what you infer from it.
If a source page is updated, do not silently attribute its current figures to an earlier year.

**Input:** Original source and its claimed scope.
**Output:** Claim–source–scope–qualification ledger.
**Branch:** If the source provides no comparator, report a descriptive observation only.

### Step 3 — Map the data generating process

Identify the grain of each table: order line, SKU–day, store–day, or route–shift.
Record the meaning of each timestamp: event, ingestion, and availability to the model.
Join product and location identifiers with effective dates and a documented mapping.
Record substitutions, canceled orders, returned items, and stockout intervals separately.
Do not infer purchases a customer would have made from sales alone.
Collect historical price and promotion plans as they were known at each forecast origin.
Exclude information introduced after the actual decision time.
For returns, follow the order cohort through its complete return eligibility window.
For grocery, record product availability at the time the shopper looked for the item.
For production, capture supplier capacity, lead-time distributions, and rejected lots.

**Input:** Source tables, event definitions, and access rules.
**Output:** Data lineage with grain, availability, coverage, and missingness.
**Branch:** If joins miss a material share of activity, quantify the missing share before modeling.

### Step 4 — Select a meaningful baseline

A seasonal-naive forecast is a possible starting point for a retail series.
A current network and its existing order assignment rule are the regionalization baseline.
An existing size chart is a baseline for a randomized size-advice experiment.
An existing manual picking operation is one baseline for automation investment.
A current buying rule is a baseline for small initial batches.
An existing shopper interface is a baseline for a shelf-location intervention.

The baseline must be feasible, contemporaneous, and measured over the same outcome window.
Document whether its operators can respond to the same information as the intervention.
A baseline that is deliberately outdated inflates the estimated benefit.
A historical comparison can be confounded by seasonality, product mix, or prices.
A controlled comparison reduces some confounding but can still face spillovers.

**Input:** Current policy and at least one candidate policy.
**Output:** A reproducible baseline and evaluation period.
**Branch:** When randomization is infeasible, document the design's assumptions and weaknesses.

### Step 5 — Select a metric that follows the decision

Forecast error should be measured at the operational horizon and decision-relevant grain.
M5's weighted metric emphasizes a specified hierarchy and revenue-weighting scheme.
The same ranking need not hold for a different assortment, horizon, or cost function.
Network evaluation should include delivered orders, service, inventory, and transport.
Returns evaluation should include kept items, contribution, handling, and customer behavior.
Automation evaluation should include total cost, utilization, service, and capital requirements.
Apparel evaluation should include demand capture, markdowns, production cost, and labor conditions.
Marketplace evaluation should include accepted fulfillment, shopper time, substitutions, and pay.

Avoid combining incompatible denominators in a single improvement percentage.
A reduction from 20% to 15% is 5 percentage points, or 25% relative to the starting rate.
Neither number alone shows the value of orders kept or customers retained.
Report absolute counts and time periods beside rates.

**Input:** Outcome definitions, denominators, cost rates, and guardrails.
**Output:** Metric dictionary and an economic bridge.
**Branch:** If a measured improvement has no plausible route to a decision, revise the metric.

### Step 6 — Translate predictions into actions

Forecasts can inform order quantities, placement, labor schedules, and customer promises.
Each action requires a rule, an owner, a cadence, and an exception path.
A forecast of 100 units is not necessarily a purchase order for 100 units.
Starting stock, inbound inventory, lead time, and desired service affect that order.
An inventory positioning rule may trade fewer delivery miles for additional stock.
A returns model may suggest an intervention but cannot override consumer rights.
A demand signal for apparel may not justify reordering if ethical capacity is unavailable.
A shelf prediction may be wrong even if the store's inventory file claims availability.

Capture actual interventions and overrides rather than merely the model's recommendations.
Measure adoption separately from model quality.
If the recommendation was not followed, estimate the effect of assignment and usage separately.
Avoid attributing the impact of a policy bundle to one component without component-level evidence.

**Input:** Prediction, operational state, feasible actions, and constraints.
**Output:** Versioned decision rule and action log.
**Branch:** If an action cannot be safely executed, run in shadow mode rather than claim a live effect.

### Step 7 — Design a local comparison

Randomize eligible stores, customers, SKU groups, or time blocks when practical.
Choose the unit at which interference is least likely and operations can implement assignment.
For regional inventory, shared stock and cross-region spillovers can violate independence.
For sizing advice, the same customer can see both variants unless assignment is stable.
For grocery automation, center-level treatment may leave too few independent facilities.
For production, styles differ substantially, so predefine matching and stratification rules.
For shopper software, the same shopper can serve treated and untreated orders.

Specify the primary outcome and analysis window before the pilot starts.
Measure all eligible units, including cancellations and non-adopters.
Track sample sizes, seasonality, and the precision of the estimated difference.
Predefine minimum economic significance in addition to statistical evidence.
Monitor safety, workload, customer service, and compliance throughout the test.
Choose a stopping authority and a rollback process.

**Input:** Eligible population, assignment mechanism, data, and guardrails.
**Output:** Pilot protocol, analysis plan, and implementation checklist.
**Branch:** If a safe randomized pilot is impossible, state the causal limitations of the alternative.

### Step 8 — Calculate the full incremental business case

Use the same population and time window for both alternatives.
Calculate incremental contribution from units kept, recovered, or delivered.
Subtract incremental handling, transport, production, and technology costs once.
Include fixed cost and capital costs on the appropriate investment horizon.
Do not count inventory's book value as an annual profit saving when it is merely released cash.
Quantify tax, financing, depreciation, and opportunity cost only if the decision boundary requires them.
State whether costs are actual cash outflows, accounting charges, or modeled assumptions.
Avoid crediting revenue from units that would have sold through another channel anyway.
Treat a transfer between marketplace and retailer as revenue to one party and cost to another.
Calculate both partner-level and combined economics when incentives differ.

**Input:** Incremental activity counts and agreed cost and revenue definitions.
**Output:** Reconciled economic bridge and sensitivity range.
**Branch:** If the result depends on one uncertain value, design a measurement for that value.

### Step 9 — Assess transfer and approve a bounded decision

Compare the source and receiving contexts for decision, demand, network, and data.
Compare lead time and supplier responsiveness before borrowing a replenishment policy.
Compare catchment density and route distances before borrowing a fulfillment network.
Compare assortment and return windows before using a fashion return rate.
Compare capacity scale, labor rules, and capital access before borrowing an automated design.
Compare platform control and retailer data rights before borrowing a marketplace workflow.

Document what is known, unknown, and testable without a full rollout.
A reasonable output may be a small controlled pilot, not a deployment commitment.
Set a review date and assign the person who can stop or expand the intervention.
Preserve the original source and the local calculation in the decision record.

**Input:** Case ledger, pilot results or assumptions, and local constraints.
**Output:** Transfer memo with decision, owner, conditions, and next evidence.
**Branch:** Reject an intervention whose downside violates a non-negotiable guardrail even if average value is positive.

## 6. Worked example

### Synthetic regional fulfillment and returns decision

This example is invented for instruction and is not a result reported by any named company.
A retailer is considering an eight-week regional inventory pilot for a fashion assortment.
The decision maker is the network planning director.
The alternative is the current centralized inventory and returns process.
Forecasts below are planning assumptions, not orders or service commitments.
The pilot plan is to position extra stock regionally and process returns under the same policy.
The service commitment remains the existing customer promise; no new promise is assumed.
All currency is U.S. dollars (USD), and quantities cover the same eight-week cohort.

| Input | Central baseline | Regional pilot | Interpretation |
|---|---:|---:|---|
| Customer orders | 12,000 orders | 12,000 orders | Equal cohort size assumed. |
| Units per order | 1.25 units | 1.25 units | 15,000 units sold in each scenario. |
| Returned units | 4,500 units | 4,350 units | Return window is fully mature. |
| Contribution per kept unit | $18.00/unit | $18.00/unit | Net of purchase and selling variable costs, excluding costs listed below. |
| Shipping expense | $5.80/order | $5.10/order | Includes outbound transport only. |
| Return handling expense | $3.40/returned unit | $3.40/returned unit | Includes reverse transport and processing. |
| Incremental regional operating expense | $0 | $5,200 | Eight-week expense, not capital expenditure. |
| Incremental inventory carrying expense | $0 | $1,600 | Eight-week financing and holding estimate. |
| Incremental setup expense | $0 | $3,000 | One-time pilot expense in this horizon. |

The illustrative reduction in returns is an assumption, not a consequence demonstrated by regionalization.
No new size-advice intervention is included; the model should be rerun with zero return-rate change.
The model assumes the same orders, product mix, prices, stock availability, and unit contribution.
Returned inventory recovery is excluded from the contribution definition for both alternatives.
If resale value differs between scenarios, measure and add the incremental recovery separately.
The carrying expense is an expense estimate; the inventory principal is not counted as profit.

### Step A — Compute units kept

Baseline units ordered: 12,000 orders × 1.25 units/order = 15,000 units.
Baseline units kept: 15,000 − 4,500 = 10,500 units.
Pilot units kept: 15,000 − 4,350 = 10,650 units.
Incremental units kept: 10,650 − 10,500 = 150 units.
Baseline item return rate: 4,500 / 15,000 = 30%.
Pilot item return rate: 4,350 / 15,000 = 29%.
The difference is 1 percentage point, not 1% relative improvement.
The relative reduction is 1 / 30 = 3.33% of the baseline rate.

### Step B — Compute incremental contribution

Incremental kept-unit contribution: 150 units × $18.00/unit = $2,700.
The calculation assumes customers keep the same number of units they do not return.
If customers cancel or reduce future purchases, this narrow bridge overstates total value.
If the 150 units would have been resold anyway, the full $18 per unit is not incremental.
A real evaluation must reconcile resale, avoided markdowns, and inventory availability.

### Step C — Compute logistics differences

Baseline outbound shipping: 12,000 orders × $5.80/order = $69,600.
Pilot outbound shipping: 12,000 orders × $5.10/order = $61,200.
Outbound shipping saving: $69,600 − $61,200 = $8,400.
Baseline return handling: 4,500 returned units × $3.40/unit = $15,300.
Pilot return handling: 4,350 returned units × $3.40/unit = $14,790.
Return handling saving: $15,300 − $14,790 = $510.
The analysis uses expense per returned unit, not expense per returned order.

### Step D — Reconcile incremental eight-week value

Gross modeled contribution and logistics improvement: $2,700 + $8,400 + $510 = $11,610.
Incremental pilot expense: $5,200 + $1,600 + $3,000 = $9,800.
Net modeled incremental contribution after listed expenses: $11,610 − $9,800 = $1,810.
This is not a complete accounting profit or a causal estimate.
It excludes tax, capital depreciation, unmodeled service changes, resale, and longer-term customer effects.
The value is a projection under assumptions, not a result from an actual pilot.

### Step E — Change the uncertain return assumption

With no change in the number of returns, kept-unit contribution and handling savings are zero.
The projected change is then $8,400 − $9,800 = **−$1,400** over eight weeks.
With a 150-unit return reduction, the projection is **+$1,810**.
Each avoided returned unit contributes $18.00 + $3.40 = $21.40 under the stated assumptions.
Break-even avoided returns are $1,400 / $21.40 = 65.42 units.
Because returned units are whole, at least 66 fewer returns are needed to break even.
This calculation does not justify assuming the return reduction occurs.
A network-only experiment should measure the change and explain a plausible mechanism.
If return reduction is unrelated to placement, the no-return-change result is the relevant base case.

### Step F — Decision and information needed

Do not approve a permanent rollout on the +$1,810 projection alone.
First test outbound shipping savings and service at comparable order mix.
Measure regional stockouts, split shipments, late delivery, and incremental inventory.
Measure return behavior by mature order cohort, but do not claim an effect without a valid comparison.
Predefine whether the one-time setup expense recurs for each additional region.
Require a stop if late delivery or availability breaches the agreed guardrail.
The analysis supports a bounded experiment, not a guaranteed financial outcome.

## 7. Code, query, or calculation

The following standard-library Python reproduces the synthetic eight-week bridge.
It assumes equal order counts and product mix, a fully matured returns cohort, and the contribution definition above.
It does not simulate uncertainty, establish causation, or represent data from any company.
Save as `retail_bridge.py` and run `python retail_bridge.py`.

```python
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Scenario:
    orders: int
    units_per_order: float
    returned_units: int
    contribution_per_kept_unit: float
    outbound_per_order: float
    handling_per_return: float
    regional_expense: float = 0.0
    carrying_expense: float = 0.0
    setup_expense: float = 0.0


def value(s: Scenario) -> float:
    ordered_units = s.orders * s.units_per_order
    kept_units = ordered_units - s.returned_units
    return (
        kept_units * s.contribution_per_kept_unit
        - s.orders * s.outbound_per_order
        - s.returned_units * s.handling_per_return
        - s.regional_expense
        - s.carrying_expense
        - s.setup_expense
    )


baseline = Scenario(
    orders=12_000,
    units_per_order=1.25,
    returned_units=4_500,
    contribution_per_kept_unit=18.00,
    outbound_per_order=5.80,
    handling_per_return=3.40,
)
pilot = Scenario(
    orders=12_000,
    units_per_order=1.25,
    returned_units=4_350,
    contribution_per_kept_unit=18.00,
    outbound_per_order=5.10,
    handling_per_return=3.40,
    regional_expense=5_200,
    carrying_expense=1_600,
    setup_expense=3_000,
)
no_return_change = Scenario(
    orders=pilot.orders,
    units_per_order=pilot.units_per_order,
    returned_units=baseline.returned_units,
    contribution_per_kept_unit=pilot.contribution_per_kept_unit,
    outbound_per_order=pilot.outbound_per_order,
    handling_per_return=pilot.handling_per_return,
    regional_expense=pilot.regional_expense,
    carrying_expense=pilot.carrying_expense,
    setup_expense=pilot.setup_expense,
)

net = value(pilot) - value(baseline)
net_without_return_change = value(no_return_change) - value(baseline)
benefit_per_avoided_return = (
    pilot.contribution_per_kept_unit + pilot.handling_per_return
)
break_even_returns = ceil(
    max(0.0, -net_without_return_change) / benefit_per_avoided_return
)
print(f"Incremental value: ${net:,.2f}")
print(f"No return change: ${net_without_return_change:,.2f}")
print(f"Break-even avoided returns: {break_even_returns} units")
```

Expected output:

```text
Incremental value: $1,810.00
No return change: $-1,400.00
Break-even avoided returns: 66 units
```

The function calculates a comparable operating contribution, not total company profit.
Both scenarios include kept-unit contribution and outbound and reverse logistics.
The difference removes costs that are common to the two scenarios.
Use integer cents or decimal arithmetic in a financial production system.
For a real pilot, replace assumptions with cohort-level observations and uncertainty intervals.

## 8. Data, standards, and regulatory context

### Retail identifiers and event histories

A product identifier, a SKU, and a product description are not necessarily the same thing.
A retailer can sell multiple sizes under one product family, each with its own return behavior.
Use stable product, location, order, shipment, and return identifiers where available.
[GS1's Global Trade Item Number guidance](https://www.gs1.org/standards/id-keys/gtin) describes identification of trade items.
GTIN is an identification standard, not proof of item availability or product quality.
Internal SKU mappings still need validity dates when assortments and packaging change.

Events must distinguish order placement, fulfillment, delivery, return request, receipt, inspection, and resale.
A returned item can be physically received before its refund, or vice versa.
A return's order-line link prevents counting it against the wrong period or product.
Store event time separately from ingestion time and correction time.
A late correction should not leak into an earlier historical forecast snapshot.
If item-level returns are unavailable, do not use order-level rates as a substitute without qualification.

### Forecasting and point-in-time availability

M5 data are a benchmark built around historical retail series and specified rules.
A production forecast can require stock availability, lead time, commitments, and cost information absent from a public benchmark.
Price data and calendar features must be available to the forecaster at the specified origin.
Future observed values are not valid substitutes for genuinely known future values.
A benchmark score measures forecast error under its own metric, not customer service or cash value.
Match sales, inventory, and forecast timestamps before interpreting model drift.

### Consumer returns and privacy

For relevant EU distance sales, the [Consumer Rights Directive 2011/83/EU](https://eur-lex.europa.eu/eli/dir/2011/83/oj) establishes withdrawal rules subject to conditions and exceptions.
National implementation and product-specific exceptions matter; commercial return policies may provide more than the legal minimum.
Do not treat a company's voluntary return period as the minimum legal entitlement everywhere.
Legal counsel should determine current obligations for the product, customer, jurisdiction, and transaction.

Where applicable, the EU [General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj) governs personal-data processing.
A size recommendation using body measurements or purchase history requires a lawful basis and appropriate safeguards.
Minimize data, restrict access, state retention periods, and assess purpose and necessity with privacy counsel.
A forecast accuracy target does not itself authorize collecting more sensitive information.
This paragraph is educational and does not determine a particular controller's obligations.

### Worker, supplier, and environmental evidence

Route or picking optimization affects people as well as order economics.
Monitor workload, worker safety, compensation, and meaningful recourse alongside speed.
Supplier monitoring needs production-site identification, working-time evidence, and a credible remediation process.
An announced code of conduct is not independent proof of compliance throughout a supply chain.
A fast replenishment cycle cannot justify unrecorded overtime or unsafe subcontracting.

Environmental comparisons require a stated functional unit, system boundary, and baseline.
At minimum consider outbound and reverse transport, packaging, inventory, and product disposition.
Fewer delivery miles do not automatically prove lower total lifecycle emissions.
Resale eligibility is not equivalent to a verified reduction in manufactured goods.
A company's emissions or waste claim must be read with its boundary and methodology.

### Data access and confidentiality

A marketplace may rely on retailer-controlled inventory or shelf information.
Contracts determine who may access data, share it, and retain derived information.
Data freshness, stock adjustments, and store-level exclusions can differ across partners.
Aggregate results should not inadvertently disclose a competitor's commercially sensitive data.
A model transferred to another retailer requires independent validation of coverage and definitions.

## 9. Industry practice and case evidence

### 9.1 M5: forecasting benchmark, not a retailer ROI study

**Source:** Makridakis, Spiliotis and Assimakopoulos describe the [M5 accuracy competition and results](https://doi.org/10.1016/j.ijforecast.2021.11.013).
The competition used Walmart retail data and evaluated hierarchical forecasting.
Its design included 42,840 aggregated and disaggregated series with a 28-day horizon.
The competition used weighted root mean squared scaled error (WRMSSE) as its accuracy metric.
See also the organizers' [background and implementation paper](https://www.sciencedirect.com/science/article/pii/S0169207021001187).
The bottom level contains 30,490 item–store series; the full hierarchy has 42,840 series.
These are series counts, not counts of independent stores or independent retailers.

**Decision:** Choose a forecast for item and aggregation levels in the stated evaluation setup.
**Evidence:** Published competition design and rankings show relative accuracy under its metric.
**Inference:** A method that handles hierarchy and calendar signals may be worth testing elsewhere.
**Unknown:** The score does not establish an inventory, revenue, service, or profit impact for a new retailer.
A winning approach's complexity and training cost may not be justified at another scale.
The available history, product mix, stockouts, lead times, and promotion mechanisms may differ.
A retailer should backtest at its own decision horizons and compare a simple feasible baseline.
The operational test must connect the forecast to order decisions and observed service.

### 9.2 Amazon: interconnected regional fulfillment

**Source:** Amazon executive Doug Herrington's [July 31, 2023 operations article](https://www.aboutamazon.com/news/operations/doug-herrington-amazon-prime-delivery-speed).
Amazon reported dividing its U.S. operations network into eight interconnected regions.
It reported that more than 76% of orders it fulfilled came from the customer's region at the time of the article.
It also reported a 15% reduction in distance between its sites and customers since the beginning of that year.
The company attributed faster and less expensive fulfillment partly to regionalization and product placement.
These are company-reported descriptive figures and causal explanations, not an independent controlled evaluation.
The eight regions were interconnected; the source does not support a claim that cross-region shipping disappeared.
The 76% figure is a share of orders, not a share of units, customers, or delivery miles.
The article also describes other initiatives, including same-day facilities and improved product placement.

**Decision:** Position selection and capacity so orders can be fulfilled near customers.
**Mechanism, inference:** Shorter paths and fewer handoffs can reduce delivery time and transport expense.
**Trade-off:** Additional inventory positions can weaken pooling and increase capital or stockouts.
**Unknown:** The article does not isolate the causal contribution of the regionalization change from the other initiatives.
**Transfer:** Test local demand density, inventory breadth, transfer rules, and order-splitting costs.
Small retailers should not assume that the same number of regions is appropriate.

### 9.3 Zalando: returns as reverse-flow decisions

**Source:** Zalando's [returns explanation](https://corporate.zalando.com/en/about-us/what-we-do/returns-zalando), a company-operated page that may be updated.
Zalando says that, across its markets, an average of 50% of ordered items are returned.
That is an item-level company-reported average, not a statement that half of all orders are returned.
The company says it can offer around 98% of returned fashion items processed in its returns centers again through its shop or shopping club.
Its stated scope includes wholesale items and partner items processed through its logistics services.
That is an offer-again claim, not proof that 98% of all returned merchandise is eventually sold at full price.
The company describes inspection, refurbishment, consolidated movement, and predictive placement.
It describes size advice and richer product information as ways to address avoidable returns.
These descriptions do not establish the incremental causal impact of any single intervention.

**Decision:** Prevent avoidable returns while maintaining conversion and handle unavoidable returns economically.
**Mechanism, inference:** Better fit information may change what a customer buys and keeps.
**Trade-off:** Restrictive return policies can lower returns and also deter valuable purchases.
**Unknown:** A public average does not reveal a local category's return causes, margin, or customer lifetime effect.
**Transfer:** Use mature item cohorts and randomized advice tests, with kept-unit contribution and satisfaction guardrails.

### 9.4 Ocado: automation as a capacity and network investment

**Source:** [Ocado Group's technology page](https://www.ocadogroup.com/about-us/our-technology).
Ocado describes its platform as combining grocery e-commerce, fulfillment, supply chain, and last-mile systems.
It describes fleets of bots moving containers in CFC storage grids, with robotic arms or people packing goods.
The page describes forecasting, fleet coordination, and route planning applications.
It also makes performance and efficiency claims as a supplier marketing its technology.
Those claims are not independently verified comparisons for a particular prospective customer.
The platform description does not establish an individual retailer's payback period or realized margin.

**Decision:** Choose a fulfillment design and capacity investment for an online grocery network.
**Mechanism, inference:** Centralized automation can reduce some handling effort at sufficient utilization.
**Trade-off:** Fixed capital, ramp-up, downtime, chilled capacity, and delivery distances can outweigh variable savings.
**Unknown:** Local throughput, maintenance, financing, route density, and partner contract economics.
**Transfer:** Model demand scenarios and compare automation with store picking or a smaller facility over the same horizon.
Do not assume that every operation is fully automated or that a robot's technical speed equals end-to-end order speed.

### 9.5 SHEIN: small initial batches and responsive replenishment

**Source:** SHEIN's [on-demand business model description](https://sheingroup.com/our-business-model), an interested company explanation.
SHEIN describes producing in small initial batches and using demand signals to inform production decisions.
The company presents its model as matching supply to preferences and reducing waste.
These are company descriptions and environmental marketing claims, not an independently established lifecycle comparison.
The source does not establish a universal batch size or a local supplier's feasible lead time.
Nor does a small initial batch by itself establish ethical labor conditions or total environmental improvement.

**Decision:** Choose the initial buy and whether and when to reorder a style.
**Mechanism, inference:** Early sales can reduce uncertainty before committing additional production.
**Trade-off:** Waiting for demand information can lose sales if replenishment cannot arrive in time.
**Unknown:** Full product lifecycle impacts, supplier-level working conditions, and comparable counterfactual waste.
**Transfer:** Test responsive production on a controlled set of styles with audited capacity and lead-time distributions.
Include freight, additional setups, markdowns, unmet demand, and supplier compensation in the economics.

### 9.6 Instacart: marketplace information and shopper execution

**Source:** Instacart's [fourth-quarter 2024 shareholder letter](https://s25.q4cdn.com/251036341/files/doc_financials/2024/q4/Fourth-Quarter-2024-Shareholder-Letter.pdf).
The company reported that nearly 10% of Instacart orders came from stores with its Carrot Tags software.
It described the software as using electronic shelf labels to help shoppers find items and improve efficiency.
The 10% refers to orders from stores with the software, not the percentage of orders proven to benefit.
The company statement is an interested-party description, not a published causal estimate of efficiency.
The letter also described testing an out-of-stock solution; a planned test is not an established outcome.

**Decision:** Decide whether to deploy shelf-location assistance and how to manage unavailable items.
**Mechanism, inference:** Accurate shelf information can reduce search time and support finding the requested item.
**Trade-off:** Stale shelf data, incorrect inventory, shopper workload, and rejected substitutes can erode benefits.
**Unknown:** Treatment effect, installation cost, shopper compensation impact, and performance by retailer and store.
**Transfer:** Randomize eligible stores or stable shopper groups and measure found rate, elapsed time, and customer outcomes.
A marketplace and retailer should agree in advance who pays, who benefits, and who controls the relevant data.

### 9.7 M5: a decision-ready evaluation design

The benchmark asks participants for predictions at a fixed forecast origin and horizon.
The operational team instead needs to make recurring decisions over many origins.
Begin with a series of historical origins separated by at least one relevant decision cycle.
At each origin, freeze all inputs as they would have appeared at that time.
Produce the simple seasonal-naive forecast and the proposed model's forecast.
Score both at the granular level needed for actual replenishment.
Also score the aggregates needed for labor, transport, or procurement planning.
Aggregation can make errors look smaller when positive and negative errors cancel.
Disaggregated errors can matter even when the regional total looks accurate.

Calculate bias as well as an absolute-error metric.
A forecast that systematically underpredicts can cause different costs from one that overpredicts.
Segment results by volume, intermittency, season, and availability.
Identify where a history contains prolonged stockouts or delistings.
Do not assume that a zero observation represents a zero-demand day.
Record whether an item was actually available at the relevant store and time.
If the true lost demand is unavailable, present a range rather than inventing a label.

Then pass the competing forecasts through the same ordering policy.
The policy must include initial inventory, inbound orders, review cadence, and lead time.
It must also apply the same minimum order quantities and service constraints.
Compute simulated inventory, lost sales under a stated demand assumption, and order costs.
Keep forecast selection and policy optimization separate to avoid giving one model extra tuning.
Use historical periods not used to tune either method for the final comparison.
Report both mean results and the distribution across stores and product groups.

The simulation is useful for filtering candidates, not proving actual business impact.
Real operators can override orders, suppliers can deliver late, and customers can substitute.
Run a subsequent controlled pilot if the simulated benefit is material.
Log recommendations, approvals, overrides, and observed outcomes.
Compare assigned groups even when some operators do not follow the model.
Check that the pilot does not reduce availability in hard-to-serve stores.
The decision is a policy choice with uncertainty, not a prize for the lowest score.

### 9.8 Regionalization: placement, pooling, and routing

A regional network begins with a geographic demand map and a feasible set of nodes.
Demand needs to be represented at a product and destination grain useful for placement.
The candidate network should identify which node can serve each destination.
Include shipping cutoffs, capacity limits, carrier service, and inventory availability.
An optimizer that ignores any of these can recommend a physically impossible assignment.
For each SKU, decide whether it will be stocked centrally, regionally, or both.
A high-volume item may support local inventory at multiple nodes.
A slow-moving item may be better pooled centrally despite a longer delivery path.
Those are hypotheses to test, not universal SKU classification rules.

Estimate replenishment lead times to each node, including their variability.
Regional stock must be replenished before a customer places the order.
A surge in one region can require an emergency transfer or a longer shipment.
Simultaneous demand shocks across regions reduce the benefit of geographic pooling.
Local nodes may need duplicated safety stock when inventory is not freely movable.
Measure both average stock and the long tail of shortages and excess.
Price the extra inventory using a clear carrying-cost definition.
Keep the value of the stock principal distinct from recurring expense.

Simulate order assignment with realistic inventory snapshots.
An order containing multiple SKUs may be split across sites.
Each split can add packaging, handling, and a separate parcel.
Do not multiply a single per-unit freight saving by units without counting parcels.
Use delivered order as well as shipped unit as a reporting denominator.
Check delivery promise accuracy and cancellation rates by customer region.
Track an out-of-region fulfillment path rather than assuming a regional wall.
Track planned versus actual interfacility transfers and their total cost.

A pilot can allocate comparable areas or SKU cohorts to different policies.
If locations share stock, randomizing individual orders may contaminate the comparison.
Cluster assignment at the relevant inventory pool can reduce that interference.
Document any exceptional transfer between treatment and comparison pools.
Report service, inventory, travel distance, and cost jointly.
Include emissions only with a declared boundary and a defensible measurement method.
Do not call observed time improvements causal if network changes occurred simultaneously.
A decision to expand should depend on the combined outcome and its uncertainty.

### 9.9 Returns: separate prevention, processing, and resale

Returns prevention starts before purchase with product information and customer expectations.
Candidate actions include better measurements, images, product descriptions, and fit guidance.
The outcome is not just whether a customer initiates a return.
Record orders placed, units bought, units kept, revenue, and customer experience.
A restrictive policy can reduce both returns and valuable purchases.
An attractive intervention should improve contribution without unacceptable customer harm.

Define a sales cohort by purchase date, country, and product category.
Wait until its return window has elapsed before comparing final return rates.
Include return requests that have not yet reached the warehouse in a separate in-transit state.
Deduplicate return events caused by corrections or repeated scans.
Match each returned item to its original order line and selling price.
Do not divide current-month returns by current-month sales and call that a cohort rate.
That ratio mixes items sold at different times and can change with growth.
Show both unit counts and percentages so shifts in mix remain visible.

Once a return arrives, the processor must inspect and choose a disposition.
Possible paths include direct resale, repair, outlet, donation, recycling, or compliant disposal.
Each path has a delay, cost, recovery value, and applicable product constraint.
The best immediate resale price may not be the best total contribution after handling and time.
Trace the unit until its actual final disposition where the data permit.
An item listed again is not necessarily sold, and its eventual sales price may be discounted.
Avoid reporting an estimated sale as realized cash.

For a controlled fit-advice experiment, stabilize assignment at customer level if feasible.
Otherwise one customer may receive both variants and contaminate the comparison.
Predefine whether the primary outcome is contribution per eligible customer or per order.
The choice matters if the intervention affects conversion and order frequency.
Measure customers who never use the advice rather than excluding them after assignment.
Analyze heterogeneous outcomes by size, category, and market without claiming certainty from tiny samples.
Monitor whether body-measurement data are actually necessary and appropriately protected.
The final report should show both intended benefit and service or privacy guardrails.

### 9.10 Ocado: comparing fulfillment architectures

The investment decision is not simply robots versus people.
Alternatives may include store picking, a manually operated center, a smaller automated site, or a larger CFC.
Each alternative has a different capital profile, service radius, and route structure.
Use the same demand scenarios and customer service requirements for every alternative.
State the beginning and end of the investment horizon.
Record whether land, buildings, software, maintenance, and delivery fleets are included.
Make all one-time implementation costs and recurring operating costs visible.

Capacity must be modeled by shift and by temperature regime, not just annual volume.
Fresh, chilled, frozen, and ambient goods can impose different constraints.
An annual average can hide peak-hour bottlenecks or unused off-peak capacity.
Include maintenance windows, outages, training, and a realistic operating ramp.
Order batching may improve picking while worsening delivery-window availability.
Measure time from order acceptance through picking, packing, staging, and delivery.
Include substitutions and food waste when data support a defensible estimate.

Estimate variable labor, packaging, energy, maintenance, transport, and customer service costs.
Do not assume all warehouse labor disappears after automation.
Different roles can emerge in replenishment, maintenance, exception handling, and supervision.
Model workforce transition and safety requirements rather than treating them as externalities.
Allow for supplier contract minimums and penalties where they apply.
Where costs are uncertain, use sensitivity ranges instead of a single promotional number.

Compare the present value of cash flows using a stated discount rate and time horizon.
Include residual value only if it can be supported by a realistic exit scenario.
Evaluate low, base, and high order volumes without changing service assumptions silently.
Calculate the utilization or volume at which the automated option breaks even.
If that threshold is above a credible demand range, the investment case is weak.
If the model excludes a facility outage, make the resulting exposure explicit.
Approval should also require a contingency plan for fulfillment during downtime.

### 9.11 SHEIN: responsive production without unsupported sustainability claims

An initial production decision balances information value against missed sales.
A large first order can capture demand immediately but risks leftover stock.
A small first order limits initial exposure but may sell out before replenishment.
The reorder decision depends on how quickly the demand signal becomes reliable.
It also depends on remaining selling time, supplier capacity, and replenishment lead time.
Early sales are affected by advertising, placement, stock availability, and price.
They are not a pure or unbiased measure of unconstrained consumer demand.

At the style level, record first-batch quantity, launch date, and available sizes.
Record traffic and conversion alongside units sold and stockout dates.
Record promotion exposures and any changes to the product page.
Record supplier capacity, material availability, production time, and freight time.
Measure the distribution of lead times rather than assuming a single promised value.
For a reorder, distinguish the forecast, approved order, supplier commitment, and actual receipt.
If the selling season ends before the reorder arrives, the signal cannot create full value.

Compare the responsive policy against the merchant's actual initial-buy rule.
Track gross contribution, lost sales, markdowns, handling, freight, and setup costs.
Account for returns and actual end-of-season disposition.
Supplier costs and payment terms must not be silently excluded from the comparison.
Rapid changes can shift risk to factories or workers even if the buyer's inventory falls.
Verify authorized sites, working time, pay, and access to remediation through appropriate processes.
A supplier code or an audit announcement alone does not establish end-to-end compliance.

For environmental analysis, specify a unit such as one delivered and kept garment.
Include production, transport, returns, use where relevant, and disposition in the chosen boundary.
Smaller batches may reduce excess inventory, but additional freight or more total sales can offset benefits.
Do not infer lifecycle emissions from batch size alone.
Use a comparable baseline, consistent accounting methods, and independently review material claims.
The company case motivates a test of information timing; it does not settle that test's result.

### 9.12 Instacart: marketplace incentives and shelf accuracy

A grocery marketplace connects customers, retailers, shoppers, and the platform.
The same change can benefit one participant while imposing cost on another.
A retailer may pay for a shelf-label installation or maintain location data.
A shopper may save time if the indicated shelf location is correct.
A customer may receive more requested items and fewer unwanted substitutes.
The platform may benefit from retention, fees, or lower support costs.
These are potential mechanisms, not verified outcomes of the cited letter.

Define a requested-item event before calculating a found rate.
State whether weighted produce, unavailable items, canceled orders, and substitutions are included.
Measure the time from the shopper starting an item search to finding it or selecting an alternative.
Record whether the proposed substitute was accepted, rejected, or never presented.
Capture store, aisle, timestamp, and the freshness of the guidance.
Inventory records and shelf records can disagree even when both are generated by the retailer.
Some items may be in a back room, misplaced, damaged, or reserved for another channel.
An incorrect confident recommendation can make search slower than no recommendation.

Choose a comparison that accounts for store layout and shopper experience.
Store-level randomization may be appropriate when shelf labels affect all shoppers in a store.
Stratify by store volume and retailer if enough stores are available.
If few stores participate, quantify the resulting uncertainty rather than treating every item as independent.
Track the same shoppers over time only with an analysis that accounts for repeated observations.
Report intent-to-treat results and adoption separately.
Monitor customer refunds, substitution acceptance, elapsed time, and shopper earnings.
Check whether workload or safety deteriorates while the average search time improves.

Calculate platform, retailer, shopper, and combined economics separately.
An installation fee is income to one party and an expense to another.
It should not be counted as a net benefit to the combined system.
Value shopper time only with an explicit pay and opportunity-cost assumption.
Include installation, device support, data integration, and upkeep.
If data-sharing agreements prevent measurement of a key outcome, record the gap.
The go/no-go decision requires a contract and operating plan as well as an attractive estimate.

### 9.13 Cross-case comparison: where the uncertainty enters

These six cases do not merely offer six different implementations of the same idea.
They place uncertainty at different points in the operating process.
The following comparisons are analytical inferences from the cited source descriptions, not reported cross-company results.

**M5 versus Amazon: prediction and positioning.**
In M5, a participant produces a forecast that is scored against subsequent observations.
The benchmark does not specify the retailer's actual stock placement or replenishment action.
In Amazon's description, products are already positioned across an operating network.
The relevant local decision is how much inventory to place at each location before demand occurs.
More accurate predictions could inform that decision, but their value depends on the placement rule.
For example, a 10-unit forecast correction has little value if a supplier's minimum order is 500 units.
Conversely, a modest change may matter if it prevents an expensive expedited shipment.
These quantities are illustrative assumptions, not facts from either company.
Compare forecasts at the replenishment lead time and then evaluate the resulting feasible action.

**Amazon versus Zalando: forward and reverse flows.**
Regional fulfillment positions products before purchase to reduce the distance to customers.
Return handling begins after a customer sends an item back and may require inspection.
The same regional location need not be optimal for both processes.
The nearest return center may lack the equipment needed to repair an item.
The best resale location may depend on demand, remaining season, and inventory already held.
A design that minimizes outbound distance may therefore increase reverse-flow handling.
Measure the full round trip for returned items rather than assuming symmetry.
Amazon's company report does not establish Zalando's optimal returns network, or vice versa.

**Zalando versus SHEIN: information at different commitment points.**
Fit advice attempts to improve a customer's choice before or during a purchase.
Responsive production uses early demand signals to inform a later supply commitment.
Both may change the number of units requiring a later disposition.
Neither guarantees a reduction in total waste without a complete comparison.
Fit advice can also alter conversion, order composition, and customer satisfaction.
Small batches can alter freight, production setups, and availability.
For each intervention, record the original decision, subsequent actions, and all material flows.
Do not transfer one company's return percentage or production practices as a target for the other.

**Ocado versus Instacart: different operating boundaries.**
Ocado describes technology for fulfillment centers and other parts of the grocery process.
Instacart's cited letter describes assistance for shoppers working in stores with shelf-label software.
The two descriptions do not imply that all orders on either platform follow one uniform workflow.
The automated-center investment decision includes long-lived capacity and route design.
The shelf-assistance decision includes data freshness, store deployment, and shopper adoption.
Both require item availability and a workable substitution process.
Compare customer service at the same basket, time-window, and product-availability definitions.
Also compare labor, capital, and delivery costs across the complete order boundary.
A per-pick speed claim is not enough to choose between the operating models.

**M5 versus SHEIN: observations and response timing.**
A forecast competition evaluates future observations over a specified horizon.
An apparel reorder needs a signal early enough to arrive before the selling opportunity ends.
A model can improve aggregate accuracy but still deliver a recommendation after the supplier deadline.
Record forecast issuance, decision cutoff, purchase commitment, and expected arrival separately.
Test whether the information is available in time to change an order.
If no feasible action remains after the forecast arrives, the information has no decision value for that cycle.
This does not mean the forecast has no value for other decisions or periods.

**Instacart versus Zalando: denominator discipline.**
The Instacart letter's nearly 10% figure concerns orders from stores with a specified software capability.
Zalando's cited return rate concerns the share of ordered items returned.
The two percentages describe different populations and neither is an experimental treatment effect.
Neither identifies the fraction of customers who changed behavior because of the intervention.
A local evaluation should define exposure, eligible population, outcome, and time window independently.
Retain counts of eligible orders and items as well as any reported percentages.
If one order contains several items, order-based and item-based rates can move in opposite directions.
Never average these company percentages to create a sector benchmark.

Across every comparison, the practical output is a local decision hypothesis.
Write down the mechanism, the action it changes, and the observation that could disprove it.
State the cost of delaying the decision as well as the risk of moving too early.
An inexpensive information-gathering step can be preferable when the investment is hard to reverse.
If the operational constraint is already known to bind, test that constraint before buying a complex model.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Benchmark-to-ROI leap | Forecast score appears as a savings percentage. | Metric and action economics differ. | Unsupported investment. | Backtest decisions and then pilot. |
| Sales treated as demand | Zero sales coincide with empty shelves. | Stockouts censor observations. | Chronic under-ordering. | Join availability and estimate lost-demand uncertainty. |
| Future information leakage | Backtest uses final promotion files. | Changes were unknown at forecast origin. | Inflated accuracy. | Reconstruct point-in-time snapshots. |
| Wrong geographic denominator | Regional order share described as share of all units. | Order and unit counts differ. | Misleading transfer target. | Keep original denominator and population. |
| Ignoring inventory pooling | Regional stock rises while average transit falls. | More stocking points require protection stock. | Working capital and obsolescence increase. | Model joint inventory and service. |
| Split shipment omission | Transport cost per unit falls but parcels per order rise. | Orders traverse multiple nodes. | Higher total delivery cost. | Measure cost per delivered order and parcel. |
| Immature return cohort | Recent orders show implausibly low returns. | Return window has not elapsed. | False apparent improvement. | Compare fully matured cohorts. |
| Return rate without conversion | Returns fall while kept units fall faster. | Restrictions discourage purchases. | Lower contribution and satisfaction. | Measure kept units and conversion together. |
| Resale eligibility equals resale | Returned units are marked offerable but not tracked to sale. | Listing is confused with realized recovery. | Overstated value. | Reconcile disposition and actual resale. |
| Automation utilization bias | Business case assumes peak throughput year-round. | Demand and maintenance create idle capacity. | Missed payback. | Model ramps, outages, and lower-volume scenarios. |
| Automation boundary error | Pick time excludes delivery and staging. | Local optimization hides end-to-end cost. | Worse total service or economics. | Measure order-to-door and full cost. |
| Small-batch labor risk | Supplier lead times shrink while hours are unavailable. | Work is shifted to overtime or subcontractors. | Worker harm and disruption. | Verify capacity, hours, and remediation. |
| Fast-reorder stockout | Winning styles sell out before replenishment. | Lead time exceeds selling window. | Lost contribution. | Test time-to-replenish and capacity reservation. |
| Platform data staleness | System says in stock but shopper cannot find item. | Inventory and shelf state diverge. | Substitutions and customer dissatisfaction. | Track freshness and actual found events. |
| Shopper cost shifting | Platform time falls while unpaid effort rises. | Costs move across partner boundary. | Unfairness and poor adoption. | Report shopper pay, time, and retailer economics. |
| Marketing claim laundering | A supplier's assertion becomes a local target. | Claim loses attribution and scope. | Procurement overconfidence. | Preserve source label and demand local evidence. |
| Pilot spillover | Control stores receive treatment-generated inventory. | Treatment affects comparison units. | Biased effect estimate. | Cluster design and track cross-unit flows. |
| Environmental boundary mismatch | Only outbound miles are included. | Returns, packaging, and production are omitted. | Unsupported sustainability claim. | Specify full boundary and functional unit. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Orders are comparable | The bridge attributes cost differences to the intervention. | Mix, geography, or basket size changes. | Stratify by customer, basket, SKU, and destination. |
| Return cohorts are mature | Returned-unit counts are final enough for comparison. | Different order dates or return windows. | Apply a fixed maturity cutoff. |
| Contribution per kept unit is stable | Converts kept units to incremental value. | Different markdown, margin, or mix. | Use line-level realized contribution. |
| Avoided returns mean extra kept units | Converts return changes to value. | Cancellations or substitution offset them. | Measure full order and customer outcomes. |
| Shipping cost per order is stable | Simplifies the regional comparison. | Parcel splits, fuel, or carrier mix change. | Use actual shipment invoices and weights. |
| Holding expense captures incremental inventory cost | Charges stock positioning. | Capital, shrinkage, or obsolescence omitted. | Reconcile finance assumptions and inventory aging. |
| Setup expense occurs once in the pilot horizon | Gives an eight-week result. | Repeated setup or ongoing licenses. | Separate one-time and recurring contract costs. |
| No resale difference | Avoids an unsupported recovery benefit. | Faster returns change resale or markdown. | Track each return through actual disposition. |
| Comparison is not contaminated | Supports a local causal contrast. | Shared inventory or shopper crossovers. | Cluster randomization and monitor movement. |
| Supplier capacity is available | Makes a reorder feasible. | Labor or material bottlenecks. | Validate capacity and lead-time tails. |
| Shelf data are current enough | Makes shopper guidance actionable. | Delayed shelf or inventory updates. | Compare timestamped guidance to observations. |
| Source scope matches the claim | Prevents misquoting a public example. | A company changes definitions or updates its page. | Retain dated original and denominator. |

A plausible mechanism is not sufficient evidence that it operated at a useful magnitude.
A source can be authentic and still lack a valid counterfactual.
Local experiments may be too small for precise inference; report intervals and decision risk.
The synthetic example intentionally isolates a few costs and should not be used as a budget.
It assumes no service effect, which is particularly consequential in regional fulfillment.
A narrower operational saving can coexist with worse whole-system outcomes.

## 12. Exercises

### Exercise 1 — Classify the evidence (15 minutes)

For each of the six cases, label the source type and identify one directly reported statement.
Separately write one plausible inference and one important unknown.
**Deliverable:** A six-row claim ledger.
**Acceptance:** No inference is presented as a measured causal outcome; M5 is not called an ROI study.

### Exercise 2 — Specify decisions (20 minutes)

Choose three cases and state the owner, action, cadence, alternative, and non-negotiable constraint.
At least one decision must be made before a customer order and one after it.
**Deliverable:** Three decision statements.
**Acceptance:** Each statement names an actionable choice and a feasible comparison.

### Exercise 3 — Calculate the bridge (15 minutes)

Reproduce the eight-week pilot's return rate, logistics savings, and incremental contribution.
Show both the assumed 150-unit return improvement and the no-change case.
**Deliverable:** A calculation table in USD and units.
**Acceptance:** Results match section 6 and identify the return assumption as synthetic.

### Exercise 4 — Change the economics (15 minutes)

Suppose outbound saving is $0.50 per order rather than $0.70, with all other assumptions unchanged.
Calculate incremental value with the 150-unit return reduction and with no return change.
**Deliverable:** Two values and a short recommendation.
**Acceptance:** Apply the changed shipping cost to all 12,000 orders and retain the same expenses.

### Exercise 5 — Critique transfer (20 minutes)

A manager proposes using the M5 winning model because it will cut inventory by the same percentage as its forecast-score advantage.
Explain at least three unsupported links in that argument and propose evidence for each.
**Deliverable:** A one-page critique.
**Acceptance:** Distinguish benchmark score, forecast behavior, ordering rule, and measured stock outcomes.

### Exercise 6 — Design a returns experiment (25 minutes)

Design a test of size advice for a fashion retailer.
Define randomization unit, primary metric, cohort maturity, two guardrails, and an analysis of non-adopters.
**Deliverable:** A short experiment protocol.
**Acceptance:** Measure kept-unit economics and do not rely on a raw return-rate reduction alone.

### Exercise 7 — Compare grocery options (25 minutes)

A grocery retailer is considering an automated CFC versus continued store picking.
List required capital, variable cost, utilization, service, labor, and route data.
Describe low-, base-, and high-volume scenarios and a credible operating baseline.
**Deliverable:** A scenario table and investment decision rule.
**Acceptance:** Include fixed and variable cost, ramp, downtime, and customer service guardrails.

### Exercise 8 — Apply the transfer test (20 minutes)

Choose either SHEIN's responsive production or Instacart's shopper assistance.
Compare source and local contexts on decision, economics, data, constraints, and evidence.
**Deliverable:** A five-row transfer table and conditional pilot recommendation.
**Acceptance:** Name a material mismatch, a measurable mitigation, and a stop condition.

### Exercise 9 — Write a case memo (20 minutes)

Write a decision memo on Amazon's regionalization or Zalando's return process.
Use one cited reported figure, one inference, one unknown, and a proposed local test.
**Deliverable:** At most 250 words.
**Acceptance:** Preserve the original denominator and avoid asserting causality from a company description.

## 13. Self-check answers

### Answer 1 — Evidence classifications

M5 is a published research competition using a specified dataset and scoring metric.
Amazon's eight interconnected regions are described in an executive company article.
Zalando's item return rate and offer-again rate are company-reported with different denominators.
Ocado's technology description is supplier material that includes promotional claims.
SHEIN's on-demand model description is company material and includes a waste-reduction claim.
Instacart's shareholder letter is an interested-company communication about stores and software.
A reader may infer a mechanism in each case but needs local evidence to quantify its effects.

### Answer 2 — Example decisions

A network director chooses weekly SKU–region inventory allocations versus central stocking, subject to availability.
A returns manager chooses daily disposition routes versus the current process, subject to product safety.
A merchant chooses initial production and weekly reorder quantities versus the existing buy rule, subject to supplier capacity.
Other answers are acceptable if owner, action, cadence, comparator, and constraints are explicit.

### Answer 3 — Baseline bridge

The baseline return rate is 4,500 / 15,000 = 30%.
The assumed pilot return rate is 4,350 / 15,000 = 29%.
Shipping saves $8,400 and returns handling saves $510 over eight weeks.
Additional kept-unit contribution is $2,700 under the given assumptions.
After $9,800 in added expense, the projected difference is $1,810.
If returns do not change, the projected difference is −$1,400.

### Answer 4 — Changed shipping saving

A $0.50 saving per order across 12,000 orders is $6,000.
With the assumed return change, $6,000 + $510 + $2,700 − $9,800 = **−$590**.
Without the return change, $6,000 − $9,800 = **−$3,800**.
The modeled case is negative in both scenarios; do not treat the prior positive estimate as robust.
This arithmetic alone does not include the value of any verified service change.

### Answer 5 — Forecast transfer

A competition metric is not a percentage inventory improvement.
A different retailer has different series, data availability, and future conditions.
An ordering policy translates a forecast into stock and can amplify or dampen forecast differences.
Inventory changes may come at the expense of availability and service.
A good answer proposes rolling-origin backtests, decision simulation, and a controlled operational pilot.

### Answer 6 — Returns experiment rubric

A strong protocol randomizes eligible customers consistently and defines exposure and assignment.
It compares mature cohorts on kept-unit contribution, not merely returned-item percentage.
It monitors conversion, satisfaction, product mix, and privacy safeguards.
It analyzes all assigned customers and reports actual uptake separately.
It predefines sample size, effect size of interest, uncertainty, and stopping authority.
A weak protocol compares this month's treated users with last month's untreated users without controlling mix.

### Answer 7 — Grocery investment rubric

A strong answer compares like-for-like capacity and service over a common multi-year horizon.
It includes capital, financing or discount rate, maintenance, labor, technology, and routing.
It models ramp-up, utilization, downtime, and scenario uncertainty.
It specifies a safety and service threshold as well as financial criteria.
It avoids treating an equipment vendor's claimed pick rate as a local business outcome.

### Answer 8 — Transfer rubric

A strong answer explicitly compares all five transfer dimensions.
For SHEIN, it tests supplier capacity and replenishment before depending on a short selling window.
For Instacart, it verifies shelf-data freshness, incentives, and shopper-level outcomes.
It names a pilot population, comparator, measurement plan, and enforceable stop condition.
It does not assume the company's public description proves the proposed local effect.

### Answer 9 — Memo rubric

A strong memo attributes the reported figure and retains its time, scope, and denominator.
It distinguishes an inference from the company's statement and identifies at least one important unknown.
It proposes a comparison and explains the decision the test will inform.
A weak memo turns the company claim into a guaranteed local percentage saving.

## 14. Interview questions

1. **What did M5 establish?** Strong: a benchmark ranking under a specified dataset and metric. Weak: proof of retailer profit.
2. **How would you use M5 locally?** Strong: point-in-time backtests, operational baselines, then decision evaluation. Weak: deploy the winner unchanged.
3. **What is the trade-off in regionalization?** Strong: transport and speed versus inventory pooling, splits, and service. Weak: fewer miles always means lower total cost.
4. **Why retain Amazon's order denominator?** Strong: order share is not unit share or mileage share. Weak: the percentages are interchangeable.
5. **How do you evaluate a returns intervention?** Strong: mature cohorts, kept units, contribution, and conversion. Weak: minimize return rate in isolation.
6. **Does an item offered again equal an item resold?** Strong: no; disposition and actual proceeds must be tracked. Weak: every offer is realized revenue.
7. **When might a grocery CFC be unattractive?** Strong: low or variable utilization, long routes, and large fixed costs. Weak: automation is always cheaper.
8. **What is the value of a small first batch?** Strong: information before a commitment, bounded by lead time and supplier capacity. Weak: guaranteed lower waste.
9. **How would you evaluate shelf-label assistance?** Strong: define found rate and compare assigned groups, time, substitutions, and worker outcomes. Weak: count stores with the software as proven benefit.
10. **How do you use company evidence responsibly?** Strong: cite exact scope, preserve attribution, and test causal claims locally. Weak: dismiss all corporate evidence or accept it uncritically.

## 15. Key terms

- [Aggregation](#4-definitions-and-vocabulary): combining lower-level series.
- [Backtest](#5-the-mechanics-step-by-step): a historical point-in-time forecast evaluation.
- [Capacity utilization](#4-definitions-and-vocabulary): used capacity relative to available capacity.
- [Censoring](#5-the-mechanics-step-by-step): constrained observation of underlying demand.
- [Cohort maturity](#8-data-standards-and-regulatory-context): sufficient time to observe returns.
- [Commitment](#6-worked-example): a customer or supplier promise, not a forecast.
- [Contribution](#6-worked-example): revenue less specified variable costs.
- [Counterfactual](#3-the-core-mental-model): the alternative outcome under another action.
- [Customer fulfillment center](#94-ocado-automation-as-a-capacity-and-network-investment): facility for customer order fulfillment.
- [Demand](#4-definitions-and-vocabulary): desired purchases, including unmet demand.
- [Fill rate](#4-definitions-and-vocabulary): requested units supplied under a defined rule.
- [Forecast](#4-definitions-and-vocabulary): an estimate for a stated future horizon.
- [Found rate](#96-instacart-marketplace-information-and-shopper-execution): a defined fraction of requested items located.
- [Guardrail](#10-failure-modes-and-red-flags): a constraint on unacceptable side effects.
- [Hierarchy](#91-m5-forecasting-benchmark-not-a-retailer-roi-study): related levels of time series.
- [Initial batch](#95-shein-small-initial-batches-and-responsive-replenishment): production committed before a response decision.
- [Leakage](#8-data-standards-and-regulatory-context): use of information unavailable at decision time.
- [Lead time](#4-definitions-and-vocabulary): elapsed time from trigger to availability.
- [Order split](#10-failure-modes-and-red-flags): fulfillment of an order through multiple shipments.
- [Plan](#6-worked-example): an operational action chosen under assumptions.
- [Pooling](#92-amazon-interconnected-regional-fulfillment): combining uncertainty across demand or inventory.
- [Recovery](#93-zalando-returns-as-reverse-flow-decisions): realized value after a return.
- [Regionalization](#92-amazon-interconnected-regional-fulfillment): geographic positioning of capacity and inventory.
- [Return rate](#93-zalando-returns-as-reverse-flow-decisions): returns divided by a matched cohort.
- [Stockout](#4-definitions-and-vocabulary): unavailability when demand arises.
- [Substitution](#96-instacart-marketplace-information-and-shopper-execution): replacing a requested item with another.
- [Transfer test](#3-the-core-mental-model): assessing whether a mechanism applies in a new setting.
- [WRMSSE](#91-m5-forecasting-benchmark-not-a-retailer-roi-study): the competition's weighted scaled error metric.

## 16. Sources

1. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). “M5 accuracy competition: Results, findings, and conclusions.” *International Journal of Forecasting*. [DOI: 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013). Research paper on the benchmark; not an operational ROI evaluation.
2. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. “The M5 competition: Background, organization, and implementation.” *International Journal of Forecasting*. [Publisher record](https://www.sciencedirect.com/science/article/pii/S0169207021001187). Competition design and hierarchy.
3. Herrington, D. (2023-07-31). [Amazon is delivering its largest selection of products to U.S. Prime members at the fastest speeds ever](https://www.aboutamazon.com/news/operations/doug-herrington-amazon-prime-delivery-speed). Amazon company article; executive-reported figures and explanations.
4. Zalando. [Returns at Zalando](https://corporate.zalando.com/en/about-us/what-we-do/returns-zalando). Company description; page content may change and its figures have specific denominators.
5. Ocado Group. [Our technology](https://www.ocadogroup.com/about-us/our-technology). Company technology and marketing description; not independent customer ROI evidence.
6. SHEIN Group. [Our on-demand business model](https://sheingroup.com/our-business-model). Company model description and marketing claims; not a lifecycle assessment.
7. Instacart (Maplebear Inc.). [Fourth-quarter 2024 shareholder letter](https://s25.q4cdn.com/251036341/files/doc_financials/2024/q4/Fourth-Quarter-2024-Shareholder-Letter.pdf). Company-reported marketplace and product information.
8. GS1. [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin). Primary identifier-standard guidance.
9. European Parliament and Council. [Directive 2011/83/EU on consumer rights](https://eur-lex.europa.eu/eli/dir/2011/83/oj). Official legal text, subject to amendments and national implementation.
10. European Parliament and Council. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Official legal text; application requires context-specific advice.

The company sources are primary evidence of what each company said, not independent proof of every claim.
The worked example and code use synthetic inputs and are not sourced company outcomes.

## 17. Links

- [Module 28 index](_index.md).
- [Case anatomy](01_case-anatomy.md).
- [Evidence grades](02_evidence-grades.md).
- [Transfer test](03_transfer-test.md).
- [Module 28 survey](../../../course/28_domain_case_library.md).
- [Simulation and optimization labs survey](../../../course/30_simulation_optimization_production_labs.md).
- [Available-to-promise in omnichannel fulfillment](../../03_commerce/23_omnichannel_fulfillment_returns/02_available-to-promise.md).
