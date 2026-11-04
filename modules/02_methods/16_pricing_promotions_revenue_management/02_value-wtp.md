# Price, Value, and Willingness to Pay

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 02 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 04, 05, and 07; Topic 01 is helpful  
> Version 2026.5

## 1. Why this topic exists

A distributor is considering an express replenishment service for retailers.
The service reserves a daily shipping slot and promises a shorter replenishment lead time.
Operations needs to know whether to allocate scarce warehouse capacity to it.
Sales needs a defensible offer, not just a percentage markup on cost.
Finance needs to know whether expected incremental contribution covers implementation.
Customers need to decide whether the service is worth changing their ordering process.

Cost-plus pricing can establish whether an offer is financially sustainable.
It does not establish whether the customer receives enough benefit to buy.
A competitor's list price can supply an external reference.
It does not establish which service features or contract terms are comparable.
A customer's stated maximum price can be informative.
It is not necessarily the amount that customer would pay when a real alternative is available.

Value-based pricing starts with a named customer decision and the next-best alternative.
It estimates the economic consequences of choosing the offer instead.
It then investigates how much of those consequences the customer recognizes and will exchange for the offer.
The analysis separates a value estimate from a willingness-to-pay estimate.
A value estimate is a model of consequences under stated assumptions.
Willingness to pay is a threshold for a specific buyer, offer, alternative, and context.
Neither is automatically the correct price.

The distinction matters in supply chains because benefits are distributed across functions.
A buyer may pay the invoice while a store operations team receives the labor savings.
A stockout reduction may increase availability without increasing demand if another product substitutes.
A shorter lead time may lower safety stock only if variability and replenishment policy also change.
The value of a reliable delivery slot depends on the customer's actual receiving constraints.
An analyst who adds every plausible benefit risks counting the same cash flow twice.

This topic gives the commercial team a repeatable evidence-to-decision process.
You will identify the relevant alternative, quantify differentiated value, collect demand evidence, and make an offer decision under uncertainty.
You will also identify when a small real-world price experiment is more informative than another survey.
The method is useful for business-to-business contracts, freight services, inventory programs, and retail offers.
It is not a substitute for legal advice, financial approval, or a causal estimate of price elasticity.

The worked case is fictional and uses U.S. dollars per store-month unless otherwise stated.
Its calculations illustrate a decision; they are not claims about a real company's performance.
The proposed price is a decision to pilot, not a forecast, target, or customer commitment.

## 2. Learning outcomes

After this topic you can:

1. **Define** the focal offer, buyer, use case, time horizon, and next-best alternative.
   Exercise 1 checks whether these elements form a comparable decision.
2. **Calculate** a customer's incremental economic value and avoid double counting.
   Exercises 2 and 3 require a value bridge and a sensitivity calculation.
3. **Distinguish** value, stated willingness to pay, observed purchase behavior, and contribution.
   Exercise 4 asks you to interpret conflicting evidence.
4. **Design** surveys and choice studies that make alternatives and price units explicit.
   Exercise 5 asks for a short instrument and a sampling plan.
5. **Design** a randomized price experiment with guardrails and an analysis unit.
   Exercise 6 asks for an experiment plan and stopping rules.
6. **Segment** customers using actionable, defensible differences in value and behavior.
   Exercise 7 tests segment economics and operational feasibility.
7. **Recommend** a price, pilot, or no-go decision with uncertainties and an owner.
   Exercise 8 asks for a decision memo and an evidence-triggered revision rule.

A successful answer exposes the comparison and the unit before performing arithmetic.
It does not present a single point estimate as a guarantee.

## 3. The core mental model

Value-based pricing is a sequence of linked decisions, not one equation.
The arrows below represent evidence flow, not guaranteed causality.

```text
Buyer + job + decision period
             |
             v
Comparable next-best alternative
             |
             v
Differential outcomes and customer costs
             |
             v
Economic value bridge -----> uncertainty and double-count checks
             |
             v
Recognition and adoption ----> survey / choice / observed behavior
             |                         |
             +------------+------------+
                          v
                Price-response evidence
                          |
                          v
          Contribution + capacity + legal guardrails
                          |
                          v
               Pilot, revise, or do not launch
                          |
                          v
             Measure outcomes and update belief
```

Start with the customer's choice, not with a product feature.
A feature such as a daily shipment slot is not value by itself.
It matters only if it changes a customer outcome relative to a feasible alternative.
The alternative could be the existing supplier, an internal workaround, or doing nothing.
It must include the price, terms, and operational performance available to the customer.

Build a bridge from the alternative to the focal offer.
For each change, specify the operational driver, quantity, unit value, frequency, and payer.
Deduct new customer costs, including integration and switching.
Do not deduct the supplier's own delivery cost in the customer value bridge.
Supplier cost belongs in the supplier contribution calculation.
Keeping the perspectives separate makes both calculations auditable.

The economic value bridge is not a demand curve.
It estimates consequences if the modeled effects occur and the customer can realize them.
A customer may not believe the benefit, may not control the affected budget, or may prefer a familiar supplier.
Some effects are uncertain or difficult to monetize without making heroic assumptions.
Thus a value bridge suggests hypotheses and negotiation boundaries; it does not prove purchases.

Willingness to pay (WTP) is the maximum amount a buyer would exchange for a specified offer instead of the specified alternative.
For an individual buyer, that threshold is usually unobserved.
A purchase at a given price indicates that WTP was at least that price only under restrictive conditions.
Those conditions include an actual choice, available budget, understanding, and no overriding constraint.
A nonpurchase at a price is also ambiguous when the customer never saw the offer or could not order.

At the market level, the share willing to buy at each price is a demand relationship.
A stated-intent survey can estimate expressed interest under its design.
A randomized price test can estimate causal differences in measured purchase rates for the tested population and context.
Neither automatically predicts behavior for a different contract, segment, season, or competitor response.

Finally, price must be assessed against the supplier's costs and capacity.
A customer can receive positive value at a price that is unprofitable for the supplier.
The supplier can earn contribution per order while losing contribution overall if volume, cannibalization, or fixed costs are ignored.
The decision is the best feasible action among the alternatives, including waiting.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or interpretation |
|---|---|---|---|
| Focal offer | Specific product or service whose price is being considered | Treating an unspecified bundle as a single stable offer | Defined features and contract |
| Customer job | Outcome the buyer seeks in a use context | Equating a job with a feature | Narrative plus operational measure |
| Decision maker | Person or group authorized to choose | Assuming the user signs the contract | Role and buying unit |
| Economic buyer | Role controlling relevant budget or approval | Assuming invoice payer captures all benefits | Role and budget |
| Next-best alternative | Feasible option the buyer would choose without the focal offer | Using a nonavailable theoretical option | Offer, terms, price, and performance |
| Reference price | Price used as a comparison by the analyst or buyer | Assuming every buyer has the same reference | Currency per same unit |
| Differential benefit | Change in customer benefit caused by focal offer versus alternative | Adding total benefits from both options | Currency per period |
| Differential customer cost | Additional cost incurred by customer to use focal offer | Mixing with supplier fulfillment cost | Currency per period |
| Economic value | Monetized incremental benefit less incremental customer cost | Calling it observed WTP | Currency per customer-period |
| Economic value to customer | Reference alternative price plus net differential value when prices and units are comparable | Treating this benchmark as a guaranteed sale price | Currency per comparable purchase unit |
| Willingness to pay | Maximum exchange amount preferred over alternative for specified choice | Assuming a survey response is exact truth | Currency per offer unit |
| Reservation price | Individual threshold at which a buyer is indifferent | Assuming the threshold is observed exactly | Currency per offer unit |
| Consumer surplus | Buyer's value or reservation threshold minus actual price for a purchase | Treating all modeled value as realized cash | Currency per transaction |
| Value capture | Portion of incremental customer value retained through a price premium | Maximizing capture without considering demand | Fraction or currency per period |
| Contribution | Revenue less specified variable and incremental costs | Calling revenue profit | Currency per period |
| Fixed launch cost | Cost incurred for the pilot or launch regardless of observed units within relevant range | Hiding it in unit margin | Currency per decision period |
| Avoidable cost | Cost that changes if the decision changes | Assigning all overhead to a marginal decision | Currency per relevant action |
| Cannibalization | Sales or contribution shifted from another own offer | Counting all focal sales as incremental | Units or currency per period |
| Switching cost | Customer cost of changing from alternative to offer | Treating all inconvenience as free | Currency, time, or qualitative burden |
| One-time cost | Cost occurring once at adoption | Subtracting the entire amount every month | Currency per adoption |
| Time horizon | Period over which outcomes and costs are compared | Mixing annual savings with monthly prices | Months or years |
| Discount rate | Rate used to bring future cash flows to a common time | Ignoring timing in long-lived decisions | Percent per stated period |
| Stockout | Failure to meet demand from available inventory | Assuming every unavailable unit is permanently lost | Events, units, or duration |
| Lost contribution | Contribution forgone when a sale is truly lost | Using sales revenue as profit | Currency per lost unit |
| Lead time | Time from order to receipt under an agreed definition | Using average lead time as variability | Days or hours |
| Safety stock | Buffer inventory against uncertainty | Assuming lead-time reduction automatically realizes cash savings | Units |
| Working-capital cost | Cost of financing inventory over time | Claiming the entire inventory reduction is recurring annual savings | Currency per year or month |
| Segment | Group with meaningful differences relevant to a decision | Assuming demographic labels prove need | Named group and membership rule |
| Segment size | Number of eligible buyers or opportunities | Using all survey respondents as the addressable market | Buyers or store-periods |
| Penetration | Share of eligible opportunities that buy | Treating survey intent as realized penetration | Fraction per period |
| Stated preference | Response to a hypothetical choice or question | Treating stated intent as a binding commitment | Response under survey design |
| Revealed preference | Choice observed in an actual decision environment | Assuming observed behavior identifies price causally | Observed action and context |
| Hypothetical bias | Difference between hypothetical response and actual behavior | Assuming its direction and magnitude are universal | Context-dependent difference |
| Discrete choice experiment | Survey presenting alternatives with systematically varied attributes | Treating arbitrary unrealistic bundles as market evidence | Choices per respondent |
| Conjoint analysis | Family of methods inferring attribute preferences from evaluated bundles | Treating part-worths as automatically causal market shares | Model-dependent utilities |
| Monadic price test | Each respondent sees one price or offer condition | Asking one respondent many prices and calling it monadic | Randomized respondent condition |
| Price ladder | Set of candidate prices or price questions | Presenting ascending prices without anchoring concern | Currency per offer unit |
| Incentive-compatible mechanism | Mechanism where truthful reporting is optimal under stated rules | Calling an intention survey incentive compatible | Mechanism and assumptions |
| Randomization unit | Entity assigned independently to a test arm | Randomizing sessions when contract price is account-level | Account, store, or other unit |
| Exposure | Opportunity to see a treatment or offer | Calling every assigned unit exposed | Measured event |
| Intention to treat | Comparison by original randomized assignment | Excluding nonbuyers or nonexposed units after assignment | Outcome per assigned unit |
| Conversion | Purchases divided by a defined eligible or exposed denominator | Comparing rates with different denominators | Fraction per period |
| Elasticity | Percentage change in quantity per percentage change in price | Inferring causality from raw price-sales correlation | Dimensionless local or arc measure |
| Confidence interval | Procedure-based interval for an estimand with a stated coverage property | Probability that a fixed parameter lies in an already computed interval | Interval and confidence level |
| External validity | Relevance of an estimate outside its study population and conditions | Assuming a pilot generalizes everywhere | Target population and setting |
| Guardrail | Metric or rule limiting unacceptable experimental outcomes | Treating it as an after-the-fact justification | Threshold and action |
| Price discrimination | Different prices across buyers or transactions for comparable offers | Assuming every segmentation practice is lawful or unlawful | Jurisdiction-specific practice |
| Forecast | Probabilistic or point prediction of future outcomes | Calling an aspiration a forecast | Quantity and horizon |
| Plan | Chosen operational actions and resources | Treating the plan as certain demand | Actions and dates |
| Target | Desired performance level | Treating a target as a model output | Metric and horizon |
| Commitment | Binding or authorized promise by a party | Treating survey intent as a contract | Party, terms, and period |

## 5. The mechanics, step by step

### Step 1 — Frame the decision

**Input:** a proposed commercial change and an accountable decision maker.
**Action:** write the buyer, offer, alternative, period, decision deadline, and feasible actions.
**Output:** a one-paragraph decision statement.

Identify the payer, user, approver, and person who experiences any disruption.
In a retailer, procurement may negotiate while store teams capture labor benefits.
A finance owner may need to approve technology integration.
A price justified only to the store manager may not pass procurement.

Specify whether the action is a list price, surcharge, subscription, rebate, or bundle.
State whether the charge is per shipment, per store-month, or per unit sold.
Include minimum volume, service levels, exclusions, and cancellation terms.
A lower nominal charge with a longer commitment may be less attractive.

List the available actions before seeing results.
Typical actions are pilot at one of several prices, change the bundle, collect more evidence, or stop.
Do not let a model choose a price outside legal, contractual, or operational boundaries.
Record the decision date; information arriving after it may not be useful for that decision.

**Branch:** If the customer cannot actually choose the alternative, replace it with a feasible one.
If no credible alternative is known, interview buyers before monetizing a premium.
If the offer is not operationally deliverable, pause pricing and resolve feasibility.

### Step 2 — Define the comparison and unit of analysis

**Input:** focal offer, alternative, and customer use case.
**Action:** normalize price and outcomes to the same unit and period.
**Output:** comparison table with gaps and evidence sources.

Compare the complete alternative, not just its headline price.
Include freight charges, service levels, expected lead time, and minimum order size.
Account for differences in implementation and termination costs.
If the customer's alternative is internal replenishment, estimate its avoidable costs rather than allocated overhead alone.

Choose the decision unit that matches the contract and the data.
A chain may contract centrally but realize value store by store.
A monthly subscription should be compared with monthly benefits and monthly variable costs.
Convert annual rates using an explicit approximation or a properly timed cash-flow model.

**Branch:** When alternatives differ by customer, build separate bridges.
Do not average a competitor's premium service and an internal workaround into a fictional baseline.
If the baseline is uncertain, calculate scenarios rather than inserting a hidden assumption.

### Step 3 — Map operational effects before monetizing them

**Input:** process map, service-level history, customer interviews, and baseline measures.
**Action:** list causal pathways from service change to customer outcome.
**Output:** a driver tree with measurable quantities and ownership.

For express replenishment, one pathway is faster response to a demand shock.
Faster response may reduce out-of-stock events if order processing and store receiving are also ready.
Fewer lost sales can increase retailer contribution, but only for sales not recovered later or shifted to another own product.
Another pathway is a more predictable receiving slot.
A predictable slot may reduce rescheduling labor if a store can actually change staffing.
A third pathway is lower inventory requirements.
That effect requires a change in ordering policy and may be limited by case packs or minimum stock.

Write the causal chain in measurable terms:
service change → changed operational behavior → changed quantity → unit economics → cash or service outcome.
For each link, record a baseline, an intervention estimate, and the evidence supporting the estimate.
Separate the value of existing service from incremental value of the new offer.

**Branch:** If an outcome cannot be attributed plausibly to the offer, exclude it from the base case.
Keep it as an unmonetized benefit or a sensitivity scenario.
If the same outcome appears through two pathways, choose one measurement and reconcile the overlap.

### Step 4 — Quantify incremental customer value

**Input:** driver tree, baseline, effect sizes, and customer economics.
**Action:** calculate each benefit and customer cost in a common currency and period.
**Output:** a value bridge with low, base, and high cases.

For each benefit, use a transparent identity:

`incremental value = change in operational quantity × value per unit × frequency`.

Examples include avoided truly lost units times contribution per unit,
avoided labor hours times the avoidable labor cost per hour,
and inventory reduction times the appropriate carrying rate for the same period.
Use customer contribution rather than selling price for incremental sales.
Labor savings are cashable only if hours, overtime, staffing, or another avoidable cost actually change.
Released capacity that cannot be redeployed may have operational value but not immediate cash value.

Subtract incremental customer operating costs.
Examples are new receiving tasks, software fees, training, and incremental handling.
Separate one-time implementation costs from recurring costs.
For a short pilot, show the one-time cost separately and calculate total pilot economics.
For a longer contract, discount cash flows at the customer's appropriate rate if timing matters.
Simple monthly amortization is an illustration, not a substitute for net present value when timing is material.

Check units in every row.
If an annual carrying rate is used with monthly inventory value, divide by 12 only as a stated simple approximation.
Do not count released inventory principal as a recurring monthly benefit.
Do not add both avoided stockout contribution and a revenue-based stockout benefit for the same units.

**Branch:** If a driver is uncertain, use a range tied to a measurable assumption.
If a cost is shared across stores, allocate it using a defensible rule and show the total.
If the buyer cannot act on the benefit, separate theoretical value from realizable value.

### Step 5 — Translate the value bridge into a price hypothesis

**Input:** net incremental value, alternative price, and commercial constraints.
**Action:** compare candidate premiums with realizable customer value and supplier contribution.
**Output:** candidate price range and explicit hypotheses.

When the same service unit and terms are comparable:

`economic value benchmark = alternative price + net differential customer value`.

This is a model benchmark, not an observed reservation price.
The candidate premium is the focal price less the alternative price.
At a proposed price, modeled customer surplus is net differential value less that premium.
If switching costs are one-time, do not hide them in a recurring surplus without stating the horizon.

Consider how much value the customer must retain to justify uncertainty and internal effort.
A positive modeled surplus is not proof that a buyer will purchase.
Customers differ in risk tolerance, liquidity, authority, and trust in the service claim.
A price below the modeled benchmark may still fail if the benefit is not credible.

Calculate supplier contribution separately:

`incremental supplier contribution = incremental revenue − incremental variable supplier cost − displaced contribution − incremental fixed cost`.

Use the relevant counterfactual: what the supplier would earn without the new offer.
Do not subtract the whole baseline service cost if it is already incurred under both options.
If a new offer displaces an existing premium service, include the lost contribution.

**Branch:** If no candidate price offers both acceptable customer value and supplier economics, change the offer or stop.
A larger survey cannot make an infeasible service profitable.
If capacity is scarce, compare contribution per constrained resource with alternative uses.

### Step 6 — Design segments around decisions

**Input:** customer heterogeneity, operational data, and feasible commercial policies.
**Action:** define segments using observable, relevant, and governable criteria.
**Output:** segment definitions, estimated value ranges, and offer eligibility rules.

A useful segment has a meaningful difference in need or value driver.
It is identifiable before the offer decision and large enough to serve efficiently.
The supplier can reach it with an offer or service level it can deliver.
The segmentation must also pass legal, contractual, privacy, and fairness review.

For replenishment, relevant variables might include frequency of emergency orders,
store receiving constraints, baseline service performance, and contribution at risk.
Firm size by itself is not a value driver unless linked to an operational outcome.
A segment built on a protected characteristic or a proxy can create unacceptable risk.
Even apparently neutral variables can produce disparate effects and require review.

Do not assume each segment requires a different individual price.
Differentiated bundles, service levels, or published eligibility rules can address different needs.
Track cost-to-serve and capacity by segment as well as modeled WTP.
A high-value segment may be small, costly to serve, or difficult to identify reliably.

**Branch:** If the segment cannot be identified without restricted data, redesign the offer or obtain appropriate review.
If sample sizes are too small, pool information cautiously and report uncertainty.
If behavior is inconsistent within a segment, revise the segment hypothesis rather than inventing precision.

### Step 7 — Select willingness-to-pay evidence

**Input:** candidate prices, target population, decision risk, and research constraints.
**Action:** choose a mix of interviews, surveys, choice studies, and behavioral tests.
**Output:** evidence plan with estimand, sample, assumptions, and decision threshold.

Interviews reveal decision criteria, vocabulary, and hidden switching costs.
They are useful for designing a survey or identifying alternative offers.
A handful of interviews does not estimate population conversion reliably.
Avoid asking leading questions such as whether a useful service is worth a low price.

A direct WTP question is quick and transparent.
Respondents may anchor on a suggested number or report a strategic answer.
Specify the full offer, alternative, unit, period, and purchase authority.
Include a genuine no-purchase option and ask about the respondent's role.
Open-ended values can be difficult for respondents to construct without context.

A monadic survey randomly assigns respondents to one candidate price.
Each respondent evaluates the same offer at one price, reducing within-person price anchoring.
Compare stated purchase intent across randomized groups using a consistent denominator.
Randomization does not remove hypothetical bias or selection bias in the recruited sample.
For each price, report sample size, intent rate, uncertainty, and nonresponse.

A discrete choice experiment (DCE) presents several alternatives with varying attributes and prices.
Include realistic levels, a no-choice option, and feasible combinations.
Avoid attribute combinations that imply impossible lead times or service terms.
Randomize or balance the design according to a pre-specified experimental design.
Interpret model-derived WTP cautiously, especially when the estimated price coefficient is weak or heterogeneous.
A simulated market share is conditional on the modeled alternatives, sample, and assumptions.

A real offer or transaction is closer to the decision of interest.
It can still be biased if price varies with customer quality, seller discretion, season, or inventory.
Randomly assigning eligible accounts or clusters to prices is often more informative than comparing historical deals.
An incentive-compatible elicitation mechanism can improve incentives only when its rules and assumptions fit the actual context.
Do not call a survey incentive compatible merely because respondents receive a participation payment.

**Branch:** Use qualitative research first when the decision or alternatives are poorly defined.
Use a survey when testing concepts or attributes across many respondents is valuable.
Use a feasible real transaction experiment when purchase behavior is the critical unknown.
If legal or operational constraints prohibit random prices, document the limitation and assess alternative designs.

### Step 8 — Draft and field the survey

**Input:** evidence plan, realistic offer descriptions, and population frame.
**Action:** pretest neutral questions and recruit eligible decision makers.
**Output:** documented questionnaire and analysis-ready responses.

Start with screening for role, authority, relevant use, and eligibility.
Explain the offer and the alternative in the same level of detail.
State the price unit, contract duration, service conditions, and what is included.
Do not reveal the research sponsor's preferred answer.

Ask each respondent to choose among realistic options, including no purchase where appropriate.
Ask what operational constraint could prevent adoption and who would approve the purchase.
Collect relevant segment variables without unnecessary sensitive information.
Separate a customer's current contract price from the survey's randomized price.

Pilot the wording with a small number of people for comprehension, not precision.
Ask them to explain what they think the unit and commitment mean.
Revise ambiguous wording before collecting the main sample.
Document exclusions, quotas, recruitment source, incentives, and response rates.
Do not claim representativeness merely because the sample is large.

Before seeing results, specify the primary outcome and analysis population.
Define how missing responses, duplicate accounts, and partial completions will be handled.
Record subgroup hypotheses and distinguish planned analysis from exploration.
Report question wording and offer presentation alongside summary results.

**Branch:** If a respondent cannot realistically authorize or influence the purchase, analyze separately or exclude by a predeclared rule.
If important segments are underrepresented, report the gap and recruit more where possible.
Weighting may address measured sampling imbalance but cannot repair unknown selection bias.

### Step 9 — Plan a real-world price experiment

**Input:** candidate prices, eligible population, operational capacity, and approvals.
**Action:** randomize a feasible offer and predefine measurement and decision rules.
**Output:** experiment protocol with estimand, randomization, guardrails, and analysis.

Define the estimand before assignment.
For example: the difference in 30-day incremental supplier contribution per eligible account assigned to price A versus price B.
Specify whether the outcome includes implementation, cannibalization, returns, and service penalties.
A conversion-only experiment may select a price with lower total contribution.

Randomize at the level where price is actually set and information can be shared.
For a chain contract, randomize the chain or account, not individual store visits.
If stores share a purchasing team, individual store assignment can contaminate the comparison.
A cluster design requires analysis and sample-size calculations that account for clustering.

Use an eligible-account denominator fixed before treatment where practical.
Compare all assigned accounts by their assigned arm for intention-to-treat analysis.
Measure exposures and compliance separately for diagnosis.
Do not remove accounts because they did not view the offer after assignment.
That exclusion can break the protection of randomization.

Keep nonprice terms, fulfillment capacity, sales training, and offer timing comparable.
If price affects inventory availability, predefine whether availability is part of the treatment effect.
Prevent sales staff from overriding assigned prices unless the protocol explicitly permits and records it.
Record account-level assignment, timestamps, price shown, actual contract, and outcome window.

Calculate sample size before launch using a meaningful effect, baseline outcome, desired power, and significance level.
For contribution outcomes, account for variance and potentially skewed account sizes.
If the eligible population is too small, acknowledge that a test may be inconclusive.
A short, underpowered test is not evidence of no effect.
Plan for seasonality, renewal cycles, and delayed purchases.

Specify guardrails such as service performance, customer complaints, legal compliance, and contribution loss.
Give each guardrail an owner, threshold, measurement frequency, and action.
Predefine whether stopping is for safety, feasibility, or statistical evidence.
Repeated significance checks can inflate false-positive rates unless a valid sequential method is planned.
Do not stop solely because a favorable result appears early under a fixed-horizon design.

Analyze price effects with uncertainty intervals, not only p-values.
Report assigned counts, exposures, purchases, contribution, and any deviations by arm.
Predefine subgroup analysis and avoid claiming a segment effect from a small exploratory slice.
Check whether the test population and season resemble the planned rollout.

**Branch:** If customers must receive one uniform contractual price, randomize timing or another permissible design only if it answers the decision question.
If interference is likely, randomize a larger cluster or choose a different setting.
If an arm breaches an agreed guardrail, follow the protocol and document the action.

### Step 10 — Turn evidence into a decision

**Input:** value bridge, survey or experiment evidence, supplier economics, and constraints.
**Action:** compare candidate actions under base and downside scenarios.
**Output:** decision memo with owner, next measurement, and revision trigger.

Show customer value, expected conversion, supplier contribution, and capacity use separately.
Do not multiply a survey intent rate by the entire market and label the result a forecast without calibration and uncertainty.
If evidence comes from a randomized transaction test, carry its uncertainty into the rollout decision.
Include the do-nothing baseline and the cost of delaying the decision.

State which quantities are measured, estimated, assumed, or negotiated.
Name the decision owner and the operational owner.
Describe the chosen price and target population, as well as who is excluded and why.
Record customer-facing claims that can be substantiated and contract terms that remain negotiable.

Separate forecast, plan, target, and commitment.
A forecast is a prediction with uncertainty; a plan is the chosen deployment action.
A target is a desired outcome; a commitment is an authorized promise.
Do not write an expected pilot conversion as though a customer has committed to buy.

Set a review date and triggers before rollout.
A trigger might be verified service cost exceeding the assumption,
conversion falling below a pre-specified range,
or insufficient customer benefit after a measured operational pilot.
At review, update the bridge and evaluate whether to expand, change, or stop.

## 6. Worked example

### Situation and comparison

The fictional distributor Northbank supplies 240 eligible independent grocery stores.
It currently sells a standard replenishment service for $420 per store-month.
The standard service is the customer's next-best feasible alternative for this case.
The proposed express service reserves a receiving slot and improves replenishment response.
The proposed subscription is $465 per store-month with the same basic product prices.
The comparison holds contract length, product assortment, and core service terms constant.
The express premium is therefore $45 per store-month.

The store owner can approve the subscription after the store manager validates the receiving process.
The distributor has enough pilot capacity for 60 stores, but not necessarily for all 240 stores.
This capacity constraint affects the launch plan, not the per-store customer value calculation.
The horizon for the initial comparison is six months.
All numerical inputs below are invented assumptions for instruction, not observed outcomes.

### Baseline and operational changes

The base-case store currently loses an estimated 34 units per month to genuine stockouts.
The estimate excludes demand that shifts to another product or is recovered later.
Express service is assumed to avoid 11 of those lost units per month.
The retailer earns $5.40 contribution per recovered unit after product and variable selling costs.
The incremental recovered-sales benefit is:

`11 units/month × $5.40/unit = $59.40/month`.

The store also expects to avoid 1.5 hours of overtime rescheduling each month.
Only the overtime portion is treated as avoidable expense.
The assumed fully avoidable overtime rate is $24 per hour.
The resulting benefit is:

`1.5 hours/month × $24/hour = $36.00/month`.

The new service permits an assumed $1,800 reduction in average inventory investment.
The customer uses a 24% annual carrying rate as a simple financing and carrying-cost assumption.
The illustrative monthly carrying-cost benefit is:

`$1,800 × 0.24/year ÷ 12 months/year = $36.00/month`.

The $1,800 inventory reduction is a one-time release of working capital if realized.
It is not an additional recurring $1,800 monthly benefit.
The $36 monthly figure is a simplified carrying-cost benefit, not the released principal itself.
A more detailed cash-flow model would place the inventory release at its actual date and avoid double counting financing effects.

Express receiving creates an extra monthly scheduling task worth $14 per store-month.
This is a customer operating cost, not the distributor's fulfillment cost.
The recurring bridge is:

| Customer bridge item | Calculation | USD per store-month |
|---|---|---:|
| Recovered lost-sales contribution | 11 units × $5.40/unit | +59.40 |
| Avoided overtime | 1.5 hours × $24/hour | +36.00 |
| Inventory carrying-cost reduction | $1,800 × 24% ÷ 12 | +36.00 |
| Additional customer scheduling | Given | −14.00 |
| Net recurring differential value | Sum | **117.40** |

The modeled economic value benchmark is $420 + $117.40 = $537.40 per store-month.
This benchmark assumes the effects are incremental, realizable, and not overlapping.
It does not demonstrate that a store will pay $537.40.
At the proposed $465 price, the premium is $45 and modeled recurring customer surplus is:

`$117.40 − $45.00 = $72.40 per store-month`.

The implied capture of modeled recurring differential value is $45 / $117.40 = 38.3%, rounded.
This ratio is descriptive; it is not a recommended universal capture rate.

### Switching and the six-month customer decision

The customer also incurs a one-time onboarding cost of $120 per store.
Ignoring discounting over this short illustrative horizon, six-month modeled net benefit is:

`6 months × $72.40/month − $120 = $314.40 per store`.

The implied undiscounted payback period is $120 / $72.40 = 1.66 months.
This calculation assumes each month's modeled benefit is realized immediately and evenly.
If onboarding requires two months before any benefit, the cash-flow timing changes.
The one-time inventory principal release is not included in the $314.40 figure.
Whether to include it in a full cash-flow analysis depends on timing and financing treatment.

### Sensitivity of the customer value bridge

Consider a downside case with only 4 recovered units per month,
0.5 avoidable overtime hour per month, and $900 lower average inventory.
Keep the $5.40 unit contribution, $24 hourly rate, 24% annual carrying rate,
and $14 monthly scheduling cost unchanged.

| Downside item | Calculation | USD per store-month |
|---|---|---:|
| Recovered contribution | 4 × $5.40 | 21.60 |
| Avoided overtime | 0.5 × $24 | 12.00 |
| Carrying-cost reduction | $900 × 24% ÷ 12 | 18.00 |
| Extra scheduling | Given | −14.00 |
| Net differential value | Sum | **37.60** |
| Surplus at $45 premium | $37.60 − $45.00 | **−7.40** |

In this downside case the customer is worse off by $7.40 per month before onboarding.
The model therefore does not support saying that every eligible store benefits at the proposed price.
A pilot should measure lost units, labor, inventory, and scheduling costs separately.
If operational effects are correlated, a combined downside scenario is more informative than changing one driver in isolation.

### Segment hypotheses

Northbank distinguishes two proposed segments using pre-offer operational records.
The high-urgency segment has at least three documented emergency orders per store-month.
The routine segment has fewer than three, subject to data-quality review.
The segment definition is a hypothesis, not proof of different WTP.
The threshold is used to stratify the pilot and is not a customer-specific price rule.

| Segment | Eligible stores | Base recurring value assumption | Main uncertainty |
|---|---:|---:|---|
| High urgency | 100 stores | $117.40/store-month | Whether emergency orders represent truly lost sales |
| Routine | 140 stores | $37.60/store-month | Whether labor or inventory policy can change |

At the same $45 premium, the base bridge suggests positive recurring surplus for the high-urgency case.
It suggests negative recurring surplus for the routine downside profile.
Northbank should not infer conversion or customer identity from this value bridge alone.
The team could test a lower-cost bundle for routine stores rather than assume a uniform response.

### Evidence plan and illustrative evidence

Suppose Northbank interviews 12 decision makers to check wording and baseline alternatives.
The interviews identify uncertainty about the reserved slot and the $120 onboarding cost.
The team then revises the description to include both.
The interviews are directional; 12 participants do not establish market-wide conversion.

Suppose a monadic survey randomly assigns eligible decision makers to the same express offer at $450 or $465 per store-month.
The questionnaire states the $420 standard alternative, six-month horizon, and $120 onboarding cost.
At $450, 24 of 40 respondents say they would purchase: 60.0% stated intent.
At $465, 20 of 40 respondents say they would purchase: 50.0% stated intent.
These values are illustrative, not empirical evidence about Northbank.
The ten-percentage-point difference is not automatically a statistically reliable or causal estimate of actual purchases.
The 40-person arms are small, stated intent may differ from actual buying, and recruitment may not represent all stores.
The study does not justify treating 60% or 50% as a rollout forecast.

### Supplier economics and pilot decision

Assume the incremental variable cost of express fulfillment is $18 per participating store-month.
Assume no cannibalization beyond replacement of standard service in this example.
The incremental contribution over the standard alternative is premium less incremental variable cost:

- At $450, the premium is $30 and incremental contribution is $12 per store-month.
- At $465, the premium is $45 and incremental contribution is $27 per store-month.

For illustration only, multiplying the stated-intent fractions by contribution gives $7.20 per surveyed eligible store-month at $450 and $13.50 at $465.
These are stated-intent-weighted comparisons, not forecasts of actual contribution.
They exclude fixed pilot cost and assume that variable cost occurs only when a store subscribes.
The comparison does not account for the uncertainty of either intent rate or costs.

The pilot plan assigns 60 eligible stores, stratified by urgency, equally to the two price arms.
Each store is independently assigned because the fictional stores do not share contract decision makers.
A real chain would need account-level or cluster-level assignment instead.
Assume fixed pilot setup costs are $900 for the six-month decision window.
If 15 stores actually subscribe at $465 for all six months, their incremental contribution before setup is:

`15 stores × 6 months × $27/store-month = $2,430`.

After the $900 fixed setup cost, the illustrative pilot contribution is $1,530.
This is a scenario, not a forecast, target, or commitment.
It assumes all 15 subscriptions last six months, no displacement, and no extra service penalties.
At $465, the simple break-even number of full six-month subscriptions is:

`$900 / (6 × $27) = 5.56 stores`, rounded up to 6 stores.

Break-even counts alone do not establish that the pilot is the best use of scarce capacity.
Compare the capacity consumed with the contribution from its next-best use.

The decision is to run a controlled pilot, not to roll out $465 to all stores.
The plan reserves capacity for 60 offers and measures six-month contribution and operational outcomes.
A target could be established by the sponsor separately; none is assumed here.
No buyer commitment exists until an actual agreement is signed.
Stop or revise according to preapproved service and loss guardrails, and review customer value after enough observed operating time.
An inconclusive price comparison calls for a decision under uncertainty, not a declaration that both prices are equivalent.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the worked customer bridge and supplier scenarios.
It also calculates the observed survey proportions without presenting them as actual conversion.
Copy the complete block into a file and run `python value_wtp.py`.
The code is deterministic; it performs no statistical test and loads no external data.
Inputs use U.S. dollars, stores, units, hours, and months as shown.

```python
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class CustomerDrivers:
    recovered_units_per_month: float
    contribution_per_unit: float
    avoidable_hours_per_month: float
    avoidable_cost_per_hour: float
    inventory_reduction: float
    annual_carry_rate: float
    extra_customer_cost_per_month: float


def customer_bridge(d: CustomerDrivers) -> dict[str, float]:
    """Return customer differential value in USD per store-month.

    The inventory amount is not added as a recurring benefit.
    Annual carrying cost is converted using a simple 12-month year.
    """
    recovered = d.recovered_units_per_month * d.contribution_per_unit
    labor = d.avoidable_hours_per_month * d.avoidable_cost_per_hour
    carrying = d.inventory_reduction * d.annual_carry_rate / 12
    net = recovered + labor + carrying - d.extra_customer_cost_per_month
    return {
        "recovered": recovered,
        "labor": labor,
        "carrying": carrying,
        "customer_cost": d.extra_customer_cost_per_month,
        "net": net,
    }


def supplier_unit_contribution(
    focal_price: float,
    alternative_price: float,
    incremental_variable_cost: float,
) -> float:
    """USD per subscribing store-month; no displaced contribution assumed."""
    return focal_price - alternative_price - incremental_variable_cost


def print_bridge(label: str, bridge: dict[str, float], premium: float) -> None:
    print(f"{label} customer bridge (USD/store-month)")
    print(f"  Recovered contribution: {bridge['recovered']:.2f}")
    print(f"  Avoided overtime:       {bridge['labor']:.2f}")
    print(f"  Carrying cost:          {bridge['carrying']:.2f}")
    print(f"  Extra customer cost:   -{bridge['customer_cost']:.2f}")
    print(f"  Net value:              {bridge['net']:.2f}")
    print(f"  Surplus after premium:  {bridge['net'] - premium:.2f}")


base = CustomerDrivers(11, 5.40, 1.5, 24, 1800, 0.24, 14)
downside = CustomerDrivers(4, 5.40, 0.5, 24, 900, 0.24, 14)
alternative_price = 420.0
focal_price = 465.0
premium = focal_price - alternative_price
base_bridge = customer_bridge(base)
downside_bridge = customer_bridge(downside)

print_bridge("Base", base_bridge, premium)
print_bridge("Downside", downside_bridge, premium)
print(f"Economic value benchmark: ${alternative_price + base_bridge['net']:.2f}/month")
print(f"Six-month customer net: ${6 * (base_bridge['net'] - premium) - 120:.2f}")
print(f"Base value capture: {100 * premium / base_bridge['net']:.1f}%")

unit_450 = supplier_unit_contribution(450, alternative_price, 18)
unit_465 = supplier_unit_contribution(465, alternative_price, 18)
print(f"Incremental supplier contribution at $450: ${unit_450:.2f}/store-month")
print(f"Incremental supplier contribution at $465: ${unit_465:.2f}/store-month")

for price, yes, assigned, unit in ((450, 24, 40, unit_450), (465, 20, 40, unit_465)):
    intent = yes / assigned
    print(
        f"${price} survey intent: {intent:.1%}; "
        f"intent-weighted contribution: ${intent * unit:.2f}/eligible store-month"
    )

fixed_pilot_cost = 900.0
months = 6
subscriptions = 15
pilot_net = subscriptions * months * unit_465 - fixed_pilot_cost
break_even = ceil(fixed_pilot_cost / (months * unit_465))
print(f"Illustrative pilot contribution: ${pilot_net:.2f}")
print(f"Full-duration subscriptions to break even: {break_even}")
```

Expected output:

```text
Base customer bridge (USD/store-month)
  Recovered contribution: 59.40
  Avoided overtime:       36.00
  Carrying cost:          36.00
  Extra customer cost:   -14.00
  Net value:              117.40
  Surplus after premium:  72.40
Downside customer bridge (USD/store-month)
  Recovered contribution: 21.60
  Avoided overtime:       12.00
  Carrying cost:          18.00
  Extra customer cost:   -14.00
  Net value:              37.60
  Surplus after premium:  -7.40
Economic value benchmark: $537.40/month
Six-month customer net: $314.40
Base value capture: 38.3%
Incremental supplier contribution at $450: $12.00/store-month
Incremental supplier contribution at $465: $27.00/store-month
$450 survey intent: 60.0%; intent-weighted contribution: $7.20/eligible store-month
$465 survey intent: 50.0%; intent-weighted contribution: $13.50/eligible store-month
Illustrative pilot contribution: $1530.00
Full-duration subscriptions to break even: 6
```

Interpretation checklist:

- The `CustomerDrivers` values are assumptions, not measurements from a real company.
- `net` is customer differential value before the express premium and onboarding.
- The benchmark adds customer net value to the standard-service price.
- Customer surplus subtracts the incremental premium, not the entire express price.
- The six-month customer result subtracts onboarding once.
- Supplier contribution subtracts only the express incremental cost because standard service is the counterfactual.
- The survey lines show hypothetical intent-weighted arithmetic, not predicted transactions.
- The pilot example assumes 15 full-duration subscriptions; it does not derive 15 from the survey.
- The break-even calculation applies only when unit contribution and months are positive.
- A production model should explicitly handle partial months, churn, displaced contribution, uncertainty, and time value where relevant.

To adapt the calculation, replace the input assumptions with documented measurements.
Keep the underlying data, timestamp, owner, and evidence grade with each input.
Do not add uncertainty intervals unless their statistical basis is defensible.
For a choice model or a randomized experiment, use a separately specified analysis appropriate to its design.

## 8. Data, standards, and regulatory context

### Data needed to define the decision

The minimum commercial dataset identifies eligible buyers and the alternatives available to them.
A contract table should record the price, currency, price unit, start date, end date, service terms, and customer account.
Transaction data should distinguish list price from invoiced price, discounts, credits, and rebates.
For a monthly service, the analytical record should represent a store-month or account-month consistently.
If contracts are account-wide, retain the account identifier even when benefits are measured at store level.

A product or service definition should identify features and exclusions for each offer version.
A price without the corresponding version and validity period is not sufficient evidence.
A $465 subscription with a reserved slot is not automatically comparable to a $465 subscription without one.
Record when the customer was eligible, when the offer was shown, and when a decision was made.
Store timezone and timestamp conventions so that outcome windows can be reconstructed.

The operational dataset should include actual orders, promised dates, receipt dates, inventory, and stockout measures.
Define whether lead time ends at warehouse dispatch, store arrival, or available-for-sale receipt.
Distinguish missing inventory observations from zero inventory.
Record backorders, substitutions, recovered sales, and canceled orders where available.
Observed lost sales generally require estimation because demand during unavailability may not be observed.
A model of lost sales must state its counterfactual and uncertainty.

The financial dataset should define contribution consistently.
For a recovered sale, record selling price, acquisition cost, and relevant variable expenses.
For labor, distinguish scheduled hours, paid hours, overtime, and redeployable time.
For inventory, document valuation basis and the carrying-rate components.
Avoid mixing nominal and real currency or monthly and annual rates.

### Survey and experiment records

Keep the exact questionnaire, display order, device context, recruitment source, and field dates.
Retain randomized assignment and the actual version shown separately.
Use a stable account identifier where appropriate to detect multiple respondents from one buying unit.
Do not equate responses with independent purchasing accounts if several people represent the same account.
Record missing and partial responses without silently recoding them as nonpurchase.

An experiment assignment table needs randomization unit, arm, assignment time, eligibility, and assignment mechanism.
Outcome tables should include price presented, purchase, cancellation, realized revenue, and incremental costs.
Document deviations, outages, sales overrides, and cross-arm exposure.
Analysis should use the predeclared unit and account for cluster assignment if applicable.

### Standards and interoperability

GS1 publishes standards for identifying products, locations, and parties.
A Global Trade Item Number (GTIN) can identify a trade item; a Global Location Number (GLN) can identify a location or party when implemented appropriately.
These identifiers help join product and location records across organizations.
They do not define a customer's WTP, prove causality, or replace contract-specific identifiers.
Use the applicable GS1 specifications and trading-partner agreements rather than assuming every organization uses the same key.

Electronic data interchange (EDI) and related transaction standards can transmit orders, invoices, and related commercial documents.
The relevant message format and implementation guide depend on the parties and jurisdiction.
EDI conveys business records; it does not by itself make a price comparable across different service bundles.
Map contract periods, currency, units of measure, and adjustments before estimating realized prices.

ISO 20252:2019 specifies requirements for market, opinion, and social research, including aspects of data collection and reporting.
It is a research-process standard, not a regulation or a guarantee that stated WTP equals actual WTP.
Applicability depends on contractual and organizational requirements.
Its existence does not remove the need to describe sampling, question wording, and limitations.

### Privacy, fairness, and competition

As of November 2026, applicable privacy and consumer-protection requirements depend on the jurisdictions, data subjects, and uses involved.
The European Union General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, applies to personal-data processing within its territorial and material scope.
It sets obligations and rights whose application depends on the facts, including lawful basis, transparency, and data minimization.
Do not assume a business-to-business survey is outside the GDPR merely because the buyer is a company; identifiable individual respondents may still be involved.
Have qualified counsel assess the processing and any automated-decision issues before collecting or using personal data.

In the United States, the Federal Trade Commission (FTC) enforces the FTC Act's prohibition on unfair or deceptive acts or practices.
Its guidance on advertising and disclosures is relevant when communicating savings claims or comparison prices.
A modeled $117.40 monthly value is not a substantiated guarantee for every customer.
Claims should reflect the supporting evidence, qualifications, and actual terms; legal review depends on context.

The FTC and U.S. Department of Justice jointly publish antitrust guidance concerning competitors and information exchange.
Do not obtain or share competitively sensitive future pricing information with competitors in a way that risks coordination.
Public competitor prices and independently collected customer research are not a blanket exemption from legal review.
The Robinson-Patman Act can be relevant to some discriminatory pricing of commodities in U.S. commerce, subject to its detailed scope and defenses.
It should not be summarized as a universal ban on differentiated prices or as permission for any particular scheme.
Counsel must assess facts, product type, customers, jurisdiction, and any applicable exemptions.

A segmentation model may use lawful operational data yet still produce unfair or unlawful outcomes.
Document the purpose, features, eligibility rules, human review, and potential disparate effects.
Limit access to identifiable survey responses and retain them only under the organization's policy and applicable law.
Aggregating data reduces some exposure but does not automatically anonymize small groups.

### Data quality and evidence grading

For each bridge driver, record source, period, unit, owner, uncertainty, and whether it is observed or estimated.
Check that the baseline and intervention populations are comparable.
Reconcile total monthly invoice revenue with the transaction extract before calculating realized price.
Reconcile store counts with eligibility and contract tables.
Investigate missing costs and unusually large value claims before using them in a price recommendation.

A peer-reviewed method paper supports the method under its assumptions; it does not validate a particular customer's inputs.
A regulator's guidance explains a legal framework; it is not approval of a proposed price test.
A company case may document a program but often lacks a credible counterfactual.
Treat a vendor's product claim as a claim unless independent evidence supports the measured effect.

## 9. Industry practice and case evidence

### Case 1 — UPS and differentiated delivery services

United Parcel Service (UPS) publishes service descriptions and rate materials for differentiated shipping services.
The documented practice is that customers can choose among services with different delivery commitments and terms.
Published rates and conditions should be checked for the relevant origin, destination, package, and date.
The existence of differentiated services does not reveal UPS's internal WTP estimates or the causal effect of any price.

**Inference:** A shipper can value a faster delivery service using avoided delay costs and the customer's next-best service.
The incremental value will differ for urgent replacement parts and nonurgent replenishment.
A higher rate is not necessarily better for the carrier once cost and capacity are included.
A buyer comparing services must also consider actual delivery reliability and the value of delay risk.
The case illustrates an observable differentiated offer, not proof of a particular pricing method.

### Case 2 — Amazon Business and business purchasing

Amazon Business publicly describes business purchasing features, including business pricing and quantity discounts.
Those features are observable offer mechanics, not evidence of a particular buyer's WTP.
Terms, eligibility, sellers, and displayed prices can vary; consult the current offer before comparison.
A quantity discount changes the effective price schedule and can interact with inventory carrying costs.
A buyer should compare total acquisition cost at its actual order quantity, not the lowest advertised tier alone.

**Inference:** A value bridge for a quantity offer should include any additional inventory investment, storage, and obsolescence risk.
A large nominal discount can be unattractive if the buyer must overstock.
The seller should assess whether the discount induces incremental orders or merely reduces price on purchases that would occur anyway.
This is an analytical implication, not a claim about Amazon's internal experiment results.

### Case 3 — Michelin tire service contracts

Michelin publicly describes fleet tire-management services under its MICHELIN Connected Fleet and related business offerings.
A service arrangement can include monitoring, support, and operational commitments in addition to tires.
Check the relevant agreement and market for actual scope; offers vary by geography and customer.
Public service descriptions do not establish a universal per-kilometer saving or WTP threshold.

**Inference:** For a fleet, potential value drivers include tire life, downtime, fuel consumption, and administration.
Each driver needs a baseline and an attributable effect, and some may overlap.
If a claimed fuel saving is already included in a measured total operating-cost reduction, it must not be added again.
The appropriate comparison is the fleet's actual alternative and its actual operating conditions.
A performance guarantee may change risk allocation and WTP even if expected physical performance is unchanged.

### Research evidence on stated and revealed preferences

Louviere, Hensher, and Swait's *Stated Choice Methods* describes the design and analysis of stated-choice studies.
The methodology uses systematically varied alternatives to study preferences under designed choice conditions.
Its practical value is separating price and feature effects when the design permits identification.
The output remains conditional on the presented alternatives and model assumptions.

List and Gallet's meta-analysis examines hypothetical bias across experimental valuation studies.
It supports caution about equating hypothetical stated values with real economic behavior.
The size and direction of bias in a particular business survey cannot be borrowed mechanically from a meta-analysis.
An analyst should test calibration or use actual purchase evidence when feasible.

Camerer and Hogarth review the effects of financial incentives in experiments.
Incentives can matter, but their effect depends on task and setting.
Paying respondents for completion does not automatically make an answer about a hypothetical contract truthful.
A proper incentive-compatible mechanism requires the consequences and assumptions to be specified.

### What a practitioner can and cannot infer

Public menus show that firms differentiate services and prices.
They rarely reveal the counterfactual, buyer-level reservation prices, or contribution at each price.
Company marketing can describe a feature but cannot alone establish an unbiased causal value estimate.
A rigorous local decision combines public context with the buyer's operational data and tested assumptions.
Use a documented case as an example of a mechanism, not as a transferable conversion rate.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong alternative | Teams compare only with list price | Buyer can choose a different realistic option | Inflated premium | Document actual feasible alternatives |
| Feature mistaken for value | Bridge lists service attributes only | No operational consequence is measured | Unsupported claim | Trace feature to measurable outcome |
| Revenue counted as profit | Lost units multiplied by selling price | Product and variable costs omitted | Overstated customer value | Use contribution per genuinely incremental unit |
| Double-counted stockouts | Both revenue and margin benefits appear | Same recovered sale enters twice | Inflated bridge | Reconcile every benefit to a unique cash flow |
| Inventory principal repeated | Full inventory release shown every month | Stock and flow are confused | Large recurring overstatement | Separate one-time release from carrying cost |
| Noncash labor treated as savings | Hours saved but payroll unchanged | Capacity cannot be monetized immediately | Forecast cash savings fail | Identify actual avoidable overtime or redeployment |
| Supplier cost mixed into buyer bridge | Same service cost deducted from buyer and supplier | Perspectives are blended | Misleading surplus and margin | Maintain separate customer and supplier ledgers |
| Unit mismatch | Annual benefits compared to monthly fee | Time bases differ | Invalid price benchmark | Normalize currency, unit, and period |
| Price-only comparison | Contract length or onboarding omitted | Offer is not comparable | Biased preference or value | Show full bundle and switching costs |
| Leading survey | Question says service is obviously valuable | Demand characteristics and anchoring | Inflated intent | Pretest neutral wording and include no choice |
| Unrepresentative respondents | Mostly nonbuyers or convenient contacts | Sample does not represent decision units | Poor rollout prediction | Screen role, document frame, assess coverage |
| Hypothetical intent treated as sales | Survey percentage labeled forecast | No real money or competing constraints | Overestimated adoption | Calibrate with real choices and report uncertainty |
| Historical correlation called causal | Higher prices coincide with higher sales | Season or demand drives both variables | Wrong price response | Randomize when feasible; otherwise document identification |
| Wrong randomization unit | Different stores in one account see different prices | Shared buyer contaminates arms | Biased and operationally infeasible comparison | Randomize at account or appropriate cluster |
| Post-assignment filtering | Only exposed accounts analyzed as randomized | Exposure depends on assignment or behavior | Selection bias | Analyze all assigned; report exposure separately |
| Repeated unplanned peeking | Test stops on first favorable p-value | Multiple looks inflate false positives | Unreliable winner | Fix horizon or preplan valid sequential monitoring |
| Capacity ignored | Test service deteriorates as volume grows | Treatment changes service availability | Rollout effect differs from pilot | Model constraints and monitor service guardrails |
| Segment overfitting | Tiny subgroup has dramatic apparent response | Noise and multiple comparisons | Unstable targeting | Prespecify groups and report subgroup uncertainty |
| Sensitive targeting | Eligibility relies on protected attributes or proxies | Legal and fairness risk | Customer harm and enforcement exposure | Review features, policy, and disparate outcomes |
| Value claim overpromised | Modeled average advertised as guaranteed saving | Assumptions and variability are hidden | Deception and customer distrust | Substantiate claims and communicate conditions |
| Competitor coordination | Sales team seeks rivals' future plans | Sensitive information can facilitate coordination | Antitrust exposure | Use independent sources and legal guidance |
| Fixed costs omitted | High per-store contribution but low volume | Launch cost not covered | Loss despite positive unit economics | Include incremental fixed costs and break-even |
| Cannibalization omitted | Most buyers were existing premium subscribers | New revenue replaces old contribution | Incremental profit overstated | Estimate displaced contribution against counterfactual |
| Pilot extrapolated blindly | Test stores differ from rollout population | External validity fails | Wrong scale decision | Compare populations and test new contexts |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Standard service is the true alternative | Defines incremental comparison | Store has a cheaper credible competitor | Interview buyers and inspect current contracts |
| Express service changes replenishment outcomes | Connects feature to value | Store process cannot use the reserved slot | Measure pre/post operations with a credible control |
| Recovered units are genuinely incremental | Values lost sales correctly | Buyers substitute or purchase later | Track substitution and delayed orders where possible |
| Unit contribution is stable | Converts recovered units to money | Mix, discounts, or costs change | Recalculate by product and period |
| Overtime is avoidable | Makes labor value cashable | Paid hours cannot change | Check schedules, payroll, and redeployment |
| Inventory policy can change | Connects lead time to inventory | Minimum order or shelf constraint dominates | Test policy change and measure average stock |
| Carrying rate is appropriate | Converts inventory investment to period cost | Financing and storage components differ | Agree rate with customer finance and show sensitivity |
| Benefits do not overlap | Allows bridge components to be summed | Same cost reduction appears in multiple rows | Map each line to a unique operational and cash-flow driver |
| Switching cost is $120 once | Determines six-month customer net | Integration is larger or repeated | Obtain implementation estimate and track actual spend |
| Six months is the relevant horizon | Supports simple total and payback | Customer renews, exits, or discounts cash flows differently | Model contract-specific duration and discounting |
| Incremental service cost is $18 per month | Determines supplier unit contribution | Cost rises with distance, volume, or congestion | Measure cost-to-serve by account and capacity utilization |
| No displaced supplier contribution | Makes premium-minus-cost sufficient | Existing premium service is cannibalized | Identify source offer and compare counterfactual margin |
| Survey arms are comparable | Supports intent comparison | Assignment fails or recruitment differs by arm | Audit assignment and baseline characteristics |
| Survey respondents represent buying units | Supports population interpretation | One account supplies many responses | Sample or aggregate at account level and report coverage |
| Stated intent relates to behavior | Makes survey direction useful | Hypothetical bias or budget constraints dominate | Compare against randomized real offers |
| No cross-arm interference | Supports simple randomized comparison | Accounts share decision makers or prices | Randomize clusters and document information sharing |
| Pilot capacity matches rollout | Supports external validity | Service degrades at scale | Test load and monitor operational constraints |
| Customer recognizes the benefit | Connects modeled value to purchase | Benefit accrues to a different budget owner | Interview stakeholders and validate claims |
| Price and service terms remain stable | Makes treatment interpretable | Sales staff negotiate other terms | Log deviations and constrain or model permitted negotiation |
| Legal and privacy review permits the design | Makes execution feasible | Jurisdiction or contract restricts use or differential prices | Obtain qualified review before fielding |

These assumptions are not equally uncertain or equally important.
Prioritize testing drivers with both high sensitivity and high uncertainty.
A precise estimate of a minor cost does not compensate for an unsupported lost-sales assumption.
A value bridge is a decision model, not an accounting guarantee.

## 12. Exercises

### Exercise 1 — Specify the decision

**Time:** 15 minutes.
A supplier proposes a $90 monthly premium for earlier delivery to a pharmacy chain.
The current service costs $600 per location-month and delivers twice weekly.
The new service delivers three times weekly but requires a new receiving process.
The chain can also use a local courier on demand.

**Deliverable:** a decision statement and a comparison table.
Identify the buyer, users, focal offer, alternatives, unit, and horizon.
**Acceptance criteria:** the table includes both feasible alternatives,
receiving-process costs, and at least two missing facts that must be obtained.
Do not assume the local courier has a known price or service level.

### Exercise 2 — Calculate the value bridge

**Time:** 20 minutes.
A customer recovers 9 truly lost units per month at $7 contribution per unit.
It avoids 2 hours of overtime per month at $22 per hour.
Average inventory falls by $1,200 and the annual carrying rate is 18%.
Additional receiving work costs $16 per month.
The alternative costs $300 per month and the proposed service costs $360 per month.

**Deliverable:** a bridge, benchmark, premium, and customer surplus in USD per month.
**Acceptance criteria:** show every formula, use the monthly carrying-rate approximation,
and do not treat the released inventory principal as monthly income.

### Exercise 3 — Include onboarding and sensitivity

**Time:** 15 minutes.
Use Exercise 2 and assume onboarding costs $150 once.
The contract lasts six months and discounting is ignored for this exercise.
Now reduce recovered units from 9 to 3 per month while holding other drivers fixed.

**Deliverable:** base and downside six-month net customer benefit.
**Acceptance criteria:** subtract the $60 monthly premium and one-time cost exactly once,
show units, and state whether the downside is positive or negative.

### Exercise 4 — Critique evidence

**Time:** 15 minutes.
A sales director reports that 70% of 20 friendly customers said they would buy at $500 per month.
The director concludes that 70% of all 5,000 eligible accounts will buy immediately.
No alternative, contract duration, or onboarding fee was shown.

**Deliverable:** a short critique and two next evidence steps.
**Acceptance criteria:** identify at least four distinct problems,
distinguish stated intent from observed behavior, and explain why 3,500 is not a justified forecast.

### Exercise 5 — Write a survey design

**Time:** 25 minutes.
You need evidence about a six-month express service for independent stores.
You can recruit 200 eligible account decision makers.
The standard service costs $420 per store-month and onboarding for express costs $120 once.
Candidate express prices are $450 and $465 per store-month.

**Deliverable:** a neutral offer description, one choice question, and a sampling and analysis plan.
**Acceptance criteria:** state all prices and units, show a no-purchase or standard-service choice,
randomize the candidate express price between respondents,
and specify respondent role, denominator, missing responses, and limitations.

### Exercise 6 — Design a transaction experiment

**Time:** 30 minutes.
The 200 accounts can actually be offered either candidate price for six months.
Several accounts operate multiple stores under a single purchasing contract.
The business can serve at most 80 new subscribers during the test.

**Deliverable:** a one-page experimental protocol.
**Acceptance criteria:** define an account-level estimand and randomization,
assignment and analysis population, contribution outcome, capacity rule,
minimum sample-size inputs, at least two guardrails, and a stopping policy.
Explain how to handle cluster size and incomplete exposure.

### Exercise 7 — Compare segment actions

**Time:** 20 minutes.
Segment A has 100 eligible stores, expected actual conversion of 30%,
and $27 monthly incremental supplier contribution per subscribing store.
Segment B has 140 eligible stores, expected actual conversion of 15%,
and $12 monthly incremental supplier contribution per subscribing store.
These rates are assumptions for the exercise, not survey intent.
Assume subscriptions last six months and fixed costs are excluded.

**Deliverable:** expected six-month contribution for each segment and a recommendation with caveats.
**Acceptance criteria:** calculate expected subscribers and contribution correctly,
name a capacity issue, and explain why expected values do not establish legal or operational eligibility.

### Exercise 8 — Write the decision memo

**Time:** 25 minutes.
Use the Northbank case, its customer value sensitivity, and its illustrative survey.
A sponsor wants an immediate rollout at $465 for all 240 stores.
You may recommend rollout, a pilot, a changed bundle, or no launch.

**Deliverable:** a decision memo of no more than 300 words.
**Acceptance criteria:** name the decision owner, action, baseline, and major uncertainty;
separate forecast, plan, target, and commitment; include supplier economics,
a downside customer outcome, and a measurable review trigger.
Do not present the illustrative survey percentages as observed purchases.

## 13. Self-check answers

### Exercise 1

The buyer is the pharmacy chain's authorized purchasing decision maker.
Store receiving staff are users and may control the operational feasibility of the new process.
The focal offer is three deliveries per week for $690 per location-month,
assuming the $90 premium is added to the $600 standard price.
The standard alternative costs $600 per location-month for two deliveries per week.
A local courier is another potential alternative, but its cost, availability, and service terms are unknown.
The appropriate horizon must be established from contract and implementation terms.

A strong comparison includes the incremental receiving process,
frequency and timing of urgent orders, and any different product or freight charges.
At least two missing facts are the local courier's actual all-in price and its service reliability.
Other important missing facts are the chain's contract duration and the operational value of a third delivery.
Without these, a $90 premium cannot be judged solely from frequency.

### Exercise 2

Recovered contribution is 9 × $7 = $63 per month.
Avoided overtime is 2 × $22 = $44 per month.
Monthly carrying-cost benefit is $1,200 × 18% / 12 = $18 per month.
Subtract $16 of extra receiving cost.
Net differential customer value is $63 + $44 + $18 − $16 = $109 per month.
The economic value benchmark is $300 + $109 = $409 per month.
The proposed premium is $360 − $300 = $60 per month.
Modeled monthly customer surplus is $109 − $60 = $49.
The $1,200 inventory release is a stock change, not $1,200 of recurring monthly benefit.

### Exercise 3

The base six-month benefit after premium and onboarding is 6 × $49 − $150 = $144.
In the downside case, recovered contribution is 3 × $7 = $21 per month.
Downside net differential value is $21 + $44 + $18 − $16 = $67 per month.
Downside monthly surplus after the $60 premium is $7.
Six-month downside benefit is 6 × $7 − $150 = −$108.
The recurring surplus is positive, but it does not recover onboarding within six months.
This difference illustrates why the horizon and one-time costs matter.

### Exercise 4

The sample is small and consists of friendly customers rather than a documented representative frame.
Respondents may not be the decision makers and may answer strategically or politely.
The offer lacks a specified alternative, duration, and onboarding cost.
A hypothetical statement is not a paid purchase or a customer commitment.
The 70% result has sampling uncertainty even for the sampled population.
Multiplying it by 5,000 produces an arithmetic figure, not a justified demand forecast.

A reasonable next step is to interview eligible decision makers to define the actual alternative and offer terms.
A second step is a neutral randomized survey or, when feasible, a randomized real offer.
Both should document the sampling frame, denominator, and uncertainty.

### Exercise 5 rubric

A complete offer description says standard service is $420 per store-month,
express service costs either $450 or $465 per store-month according to random assignment,
the contract lasts six months, and express onboarding costs $120 once.
The choice question offers express, standard, and an appropriate no-purchase option.
The wording describes operational features without promising unverified savings.

A strong sampling plan recruits independent eligible account decision makers,
assigns roughly half to each price, and records the role and eligible account.
The primary survey outcome is the express-choice proportion per assigned respondent or account,
with the denominator and treatment of missing responses stated before analysis.
Report counts and intervals or other uncertainty, not just percentages.
Explain selection and hypothetical bias; do not label the result actual conversion.

### Exercise 6 rubric

The randomized unit is the purchasing account when stores share a contract.
A suitable estimand is the six-month difference in incremental supplier contribution per assigned eligible account.
The outcome includes actual premium revenue, incremental service costs, displaced contribution, and relevant penalties.
Analyze by original assignment; report exposure and contract adherence separately.
Account for unequal numbers of stores per account and the cluster-level design in sample-size and analysis choices.

A capacity rule must specify how the 80-subscription limit is allocated without confounding price arms.
If capacity may bind, the protocol must say whether constrained fulfillment is part of the treatment and how it affects interpretation.
Sample-size planning needs baseline conversion, meaningful contribution difference,
outcome variability, desired power, significance level, and relevant cluster structure.
Guardrails can include service-level breaches and cumulative financial loss, each with owner and action.
A fixed-horizon analysis should not stop for an unplanned favorable significance result.

### Exercise 7

Segment A has 100 × 30% = 30 expected subscribers.
Its six-month expected contribution is 30 × $27 × 6 = $4,860.
Segment B has 140 × 15% = 21 expected subscribers.
Its six-month expected contribution is 21 × $12 × 6 = $1,512.
The combined expected count is 51 subscribers, but actual demand can differ.
If capacity is less than the realized number of subscriptions,
the firm needs a predeclared allocation and a comparison with other uses of the constrained resource.
Expected contribution excludes fixed costs and depends on the stated conversion assumptions.
A segment's apparent profitability does not establish lawful or feasible targeting.

### Exercise 8 rubric

A strong memo recommends a feasible action with an accountable sponsor.
A controlled pilot is defensible because the value bridge has a negative downside at the $45 premium
and the illustrative survey does not measure actual conversion.
It cites $27 incremental supplier contribution per subscribing store-month at $465 before fixed cost,
and the $900 fixed pilot cost if using the case assumptions.
It states that the plan is to offer two prices to randomized eligible stores,
not that a forecast or target has been achieved or customers have committed.
A useful trigger specifies a measured service-cost, benefit, or guardrail threshold and an owner.
Alternative recommendations can score well if they use the same evidence honestly and address uncertainty.

## 14. Interview questions

1. **How is economic value different from WTP?**
   A strong answer defines a modeled differential outcome versus an individual threshold in a specified choice.
   A weak answer treats a calculated savings figure as an observed purchase price.

2. **What is the correct alternative in a value bridge?**
   A strong answer selects the feasible option the buyer would actually use and matches terms and period.
   A weak answer always chooses the lowest public list price without checking availability.

3. **How would you value reduced stockouts?**
   A strong answer estimates truly incremental units and multiplies by contribution, accounting for substitution.
   A weak answer multiplies every unavailable unit by sales revenue.

4. **How do you value inventory reduction?**
   A strong answer separates the one-time working-capital release from recurring carrying cost and states the rate.
   A weak answer books the full inventory reduction as income each month.

5. **Why might a customer reject an offer with positive modeled value?**
   A strong answer discusses credibility, risk, budget ownership, switching, and alternatives.
   A weak answer assumes that all customers know and can realize the model's benefits.

6. **When is a monadic survey useful, and what can it not prove?**
   A strong answer describes randomized price conditions, a consistent denominator, and hypothetical bias.
   A weak answer equates stated purchase intent with observed conversion.

7. **What makes a discrete choice experiment credible?**
   A strong answer names realistic alternatives, attribute variation, no-choice, sampling, and model limitations.
   A weak answer assumes any collection of price questions identifies market demand.

8. **At what level should a price experiment randomize?**
   A strong answer matches assignment to the contract decision and considers interference and clustering.
   A weak answer randomizes page views for accounts that negotiate one shared price.

9. **Why analyze by original assignment?**
   A strong answer explains intention to treat and the selection risk of filtering on post-assignment exposure.
   A weak answer excludes nonbuyers because they did not receive the treatment successfully.

10. **How would you decide between two candidate prices?**
    A strong answer compares incremental contribution, volume uncertainty, fixed cost, capacity, and customer outcomes.
    A weak answer selects the price with the highest conversion or highest unit margin alone.

11. **What makes a segment actionable?**
    A strong answer links need to observable eligibility, reachable offers, cost, legal review, and sufficient evidence.
    A weak answer assumes every demographic difference justifies personalized prices.

12. **How should a modeled benefit be communicated?**
    A strong answer states assumptions, uncertainty, evidence, and substantiation limits.
    A weak answer markets an illustrative average as a guaranteed saving for everyone.

## 15. Key terms

- [Alternative](#4-definitions-and-vocabulary) — feasible option without the focal offer.
- [Avoidable cost](#4-definitions-and-vocabulary) — cost that changes with the decision.
- [Cannibalization](#4-definitions-and-vocabulary) — displacement of contribution from an existing offer.
- [Choice experiment](#4-definitions-and-vocabulary) — designed selection among varying alternatives.
- [Commitment](#4-definitions-and-vocabulary) — authorized promise under specified terms.
- [Confidence interval](#4-definitions-and-vocabulary) — uncertainty interval from a stated procedure.
- [Consumer surplus](#4-definitions-and-vocabulary) — buyer value less paid price.
- [Contribution](#4-definitions-and-vocabulary) — revenue less specified incremental costs.
- [Customer job](#4-definitions-and-vocabulary) — outcome the buyer seeks.
- [Decision maker](#4-definitions-and-vocabulary) — person or group authorized to choose.
- [Differential benefit](#4-definitions-and-vocabulary) — benefit relative to the alternative.
- [Economic value](#4-definitions-and-vocabulary) — modeled net differential customer benefit.
- [Economic value benchmark](#5-the-mechanics-step-by-step) — alternative price plus net differential value.
- [Elasticity](#4-definitions-and-vocabulary) — relative quantity response to relative price change.
- [Exposure](#4-definitions-and-vocabulary) — opportunity to see the offer.
- [External validity](#4-definitions-and-vocabulary) — applicability outside the study conditions.
- [Fixed launch cost](#4-definitions-and-vocabulary) — decision-period cost not varying with trial units.
- [Forecast](#4-definitions-and-vocabulary) — prediction of an uncertain future outcome.
- [Guardrail](#4-definitions-and-vocabulary) — predeclared constraint and response.
- [Hypothetical bias](#4-definitions-and-vocabulary) — gap between hypothetical and actual behavior.
- [Incremental cost](#5-the-mechanics-step-by-step) — cost changed by the selected action.
- [Intention to treat](#4-definitions-and-vocabulary) — comparison by original assignment.
- [Lead time](#4-definitions-and-vocabulary) — defined elapsed replenishment time.
- [Lost contribution](#4-definitions-and-vocabulary) — margin forgone on genuinely lost sales.
- [Monadic test](#4-definitions-and-vocabulary) — one assigned price condition per respondent.
- [Plan](#4-definitions-and-vocabulary) — selected operational actions and resources.
- [Randomization unit](#4-definitions-and-vocabulary) — independently assigned entity.
- [Reference price](#4-definitions-and-vocabulary) — price used for a specified comparison.
- [Reservation price](#4-definitions-and-vocabulary) — individual indifference threshold.
- [Revealed preference](#4-definitions-and-vocabulary) — observed choice under actual conditions.
- [Segment](#4-definitions-and-vocabulary) — decision-relevant group with explicit membership.
- [Stated preference](#4-definitions-and-vocabulary) — response to a hypothetical choice.
- [Switching cost](#4-definitions-and-vocabulary) — customer cost of changing alternatives.
- [Target](#4-definitions-and-vocabulary) — desired performance, not a prediction.
- [Value capture](#4-definitions-and-vocabulary) — premium as a portion of modeled value.
- [Willingness to pay](#4-definitions-and-vocabulary) — maximum exchange amount for a specified choice.
- [Working-capital cost](#4-definitions-and-vocabulary) — cost of financing inventory over time.

## 16. Sources

The sources support concepts, public offer descriptions, or regulatory context as indicated.
They do not validate the invented Northbank inputs or promise a particular business result.
Access conditions and service terms can change; check the relevant current documents before use.

1. Anderson, James C., Nirmalya Kumar, and James A. Narus.
   *Value Merchants: Demonstrating and Documenting Superior Value in Business Markets.*
   Harvard Business School Press, 2007.
   A practitioner treatment of customer value assessment and business-market value communication.
   [Publisher listing](https://store.hbr.org/product/value-merchants-demonstrating-and-documenting-superior-value-in-business-markets/10036).
2. Louviere, Jordan J., David A. Hensher, and Joffre D. Swait.
   *Stated Choice Methods: Analysis and Applications.* Cambridge University Press, 2000.
   Design and interpretation of stated-choice studies.
   [Publisher page](https://www.cambridge.org/core/books/stated-choice-methods/21EDB15CE627494FE1CA258331D34154).
3. List, John A., and Craig A. Gallet.
   “What Experimental Protocol Influence Disparities Between Actual and Hypothetical Stated Values?”
   *Environmental and Resource Economics* 20, 241–254, 2001.
   Meta-analysis of hypothetical versus actual valuation.
   [DOI: 10.1023/A:1012791822804](https://doi.org/10.1023/A:1012791822804).
4. Camerer, Colin F., and Robin M. Hogarth.
   “The Effects of Financial Incentives in Experiments: A Review and Capital-Labor-Production Framework.”
   *Journal of Risk and Uncertainty* 19, 7–42, 1999.
   Review of incentives and experimental performance.
   [DOI: 10.1023/A:1007850605129](https://doi.org/10.1023/A:1007850605129).
5. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Identification and data-carrier specifications; consult the applicable edition and implementation.
6. ISO. [ISO 20252:2019, Market, opinion and social research, including insights and data analytics](https://www.iso.org/standard/73671.html).
   Research-process requirements, not proof of purchase behavior.
7. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   Primary legal text; applicability depends on the processing and facts.
8. U.S. Federal Trade Commission. [Advertising and Marketing](https://www.ftc.gov/business-guidance/advertising-marketing).
   Official business guidance on advertising and substantiation.
9. U.S. Federal Trade Commission. [Price Discrimination: Robinson-Patman Violations](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/price-discrimination-robinson-patman-violations).
   Official overview; the detailed statutory scope and defenses require legal assessment.
10. U.S. Federal Trade Commission and Department of Justice.
    [Antitrust Guidelines for Collaborations Among Competitors](https://www.ftc.gov/sites/default/files/documents/public_events/joint-venture-hearings-antitrust-guidelines-collaboration-among-competitors/ftcdojguidelines-2.pdf), 2000.
    Historical official guidance relevant to competitor collaboration and information exchange; check current agency policy.
11. UPS. [Shipping Services](https://www.ups.com/us/en/shipping/services.page).
    Public service descriptions; actual terms and rates depend on the shipment and date.
12. Amazon Business. [Business Pricing and Quantity Discounts](https://business.amazon.com/en/work-with-us/sell-to-business/business-pricing-and-quantity-discounts).
    Public description of offer mechanics, not a causal pricing study.
13. Michelin. [MICHELIN Connected Fleet](https://connectedfleet.michelin.com/).
    Public service description; local offers and contract terms vary.

## 17. Links

- [Module 16 topic index](./_index.md).
- [Module 16 survey and lab sections](../../../course/16_pricing_promotions_revenue_management.md).
- [Pricing decision stack](./01_pricing-decision-stack.md) — list price, terms, and decision ownership.
- [Own-price and cross-price elasticity](./03_elasticity-identification.md) — identifying price response.
- [Price testing](./10_price-testing.md) — experiments and reference-price effects.
- [Course outline](../../../course/COURSE_OUTLINE.md) — program sequence and assessment context.

Sibling topics listed in the index may be planned rather than available at the time of reading.
Use the module survey's lab sections until a linked expanded topic is published.
