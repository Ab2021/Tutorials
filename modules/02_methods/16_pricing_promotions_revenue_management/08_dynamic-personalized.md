# Dynamic and Personalized Pricing

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 08 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 04, 05, and 07; contribution-margin arithmetic  
> Version 2026.1

## 1. Why this topic exists

A distributor has a limited number of delivery slots for a perishable product.
Demand varies by day, location, and lead time.
The commercial team wants a price that reflects the cost of serving each order without wasting capacity.
The operations team wants enough notice to staff the delivery route.
The customer wants a quote that is understandable and will not change at checkout.

A dynamic price changes as conditions change.
A personalized price or offer varies with information about the customer or a group to which the customer belongs.
These are different choices.
A published peak delivery charge that applies equally to everyone at the same time can be dynamic without being personalized.
A customer-specific coupon that remains available for a month can be personalized without changing dynamically.
A system can do both, and therefore inherits both kinds of risk.

The attraction is straightforward: different orders create different revenue, costs, and capacity demands.
A high-demand period may justify a different publicly available price from a quiet period.
A voluntary bulk commitment may legitimately receive a different business quote from a small, urgent order, subject to the applicable law.
But the existence of different observed purchase rates is not itself permission or evidence to charge different people different prices.
The observed rate reflects previous targeting, availability, customer needs, and price exposure.

A poor system can increase apparent margin while losing long-run demand, shifting sales to substitutes, or increasing costs elsewhere.
It can train on discounts given to customers already likely to buy.
It can charge different prices because a proxy encodes a protected characteristic or a vulnerable population.
It can also recommend a legally or operationally unavailable price.
A quote that cannot be honored is not a viable recommendation.

This unit treats a price as a controlled business intervention.
It begins with the decision and the eligible population, not a model architecture.
It asks for evidence on incremental response, constraints on the feasible offer, and a design for monitoring outcomes that matter to customers as well as finance.
The fictional worked case uses a public time-based charge, not an individual willingness-to-pay score.
A separate design discussion shows the additional controls needed before considering personalization.
Nothing in this unit is legal advice or a substitute for jurisdiction-specific review.

## 2. Learning outcomes

After working through this topic you can:

1. Distinguish dynamic pricing, personalized pricing, segmentation, and a public eligibility rule.
   Exercise 1 asks you to classify each intervention.
2. Define an intervention, outcome, unit of assignment, and estimand for a price decision.
   Exercises 2 and 5 test whether your comparison is causal.
3. Calculate incremental contribution per eligible customer and capacity impact.
   Exercise 3 requires arithmetic and an operational recommendation.
4. Construct a feasible-price set from customer, commercial, and operational constraints.
   Exercise 4 requires a stop decision when a constraint is breached.
5. Design a controlled test with allocation, measurement, stopping rules, and interference controls.
   Exercise 5 asks for a field-ready protocol.
6. Assess distributional effects, privacy, consumer trust, and legal exposure without treating a fairness metric as legal proof.
   Exercise 6 requires a governance decision.
7. Distinguish a forecast, plan, target, and customer commitment.
   Exercise 7 applies these terms to the case.
8. Prepare an operating and rollback plan with named accountable roles and evidence for review.
   Exercise 8 asks for an approval packet.

## 3. The core mental model

Price is not a prediction returned by a model and immediately shown to a customer.
It is an intervention selected from an approved set and presented within an accountable process.

```text
Business objective + customer promise + legal boundaries
                       |
                       v
          Define eligible population and horizon
                       |
                       v
    Measure costs, capacity, historical exposure, outcomes
                       |
                       v
       Design causal comparison of feasible interventions
                       |
                       v
   Estimate incremental value and distributional consequences
                       |
                       v
     Apply guardrails, approval, and customer disclosures
                       |
                       v
       Publish quote -> honor quote -> fulfill -> measure
                       |                         |
                       +------ audit and learn <-+
```

The arrows are deliberately directional.
A favorable estimate does not override a legal prohibition, a contract, or an inability to honor the quote.
If the feasible set is empty, the correct output is a safe existing price or a human review, according to a preapproved policy.
The fallback is not an excuse to hide an error; an operator must see and investigate it.

### Four separable choices

The first choice is **whether prices vary over time**.
Time variation can reflect a public schedule, a forecast, inventory, capacity, or a specific event.
The business should know what changes the price, who can change it, and how long a customer can rely on a quote.

The second choice is **whether offers vary across customers**.
A segment might be defined by a contract, an order size, membership eligibility, or observed behavior.
Being able to label a segment does not establish that segment-specific pricing is fair, lawful, or profitable.
A segment can be too small for reliable evidence or can indirectly encode sensitive characteristics.

The third choice is **how the organization learns response**.
A model trained on historical sales predicts outcomes under the old targeting policy unless additional identification assumptions hold.
A randomized test compares preassigned offers, including people who did not purchase.
The test identifies a specific contrast for a specific eligible population and period, not a universal demand curve.

The fourth choice is **how the offer is governed**.
A customer sees an all-in amount and a promise, not the internal model's objective function.
The company must prevent unstable quotes, unexpected charges, and discriminatory implementation.
It must be able to reconstruct which rule produced a quote and what was actually displayed.

### A counterfactual, not a before-and-after story

For one eligible session, define potential outcome Y(A) as contribution if assigned offer A and Y(B) as contribution if assigned offer B.
Only one of these outcomes is observed for that session.
Random assignment makes groups comparable in expectation, so the group-average difference estimates the average effect of assignment under the test conditions.
It does not reveal both outcomes for any one customer.

If the business changes the price during a holiday and compares with last month, the holiday itself is a competing explanation.
If it targets a coupon to likely buyers, the recipients may have purchased anyway.
If it raises prices when inventory is scarce, scarcity both affects the price and reduces sales.
These are different mechanisms, but each can make a sales-versus-price correlation misleading.

### Operating at the right level

A price for a perishable seat or delivery slot is coupled to the remaining capacity.
A price for a durable consumer item is coupled to stock, replenishment, returns, and possible substitutes.
A contract quote may apply for months and cannot be evaluated as a single checkout conversion.
Choose the decision unit and outcome window to fit the actual promise and supply constraints.

## 4. Definitions and vocabulary

The table defines the terms used throughout this unit.
Amounts are in the stated currency and per the stated unit of assignment unless specified otherwise.

| Term | Meaning | Common misuse or unit |
|---|---|---|
| Dynamic pricing | A price changes with time or changing conditions according to a rule. | Not automatically personalized; price per unit or order. |
| Personalized pricing | A price offered varies with information about an identifiable customer or their characteristics. | Not the same as a uniformly published time schedule. |
| Personalized offer | A targeted benefit or condition, such as an eligible coupon. | Face price alone may conceal an effective price difference. |
| Segmentation | Grouping eligible decision units using specified characteristics or behavior. | A segment label does not establish a causal effect. |
| Public fence | A transparent condition, such as booking time, that determines an offer. | A fence still needs fairness and legal review. |
| Effective price | Amount the customer actually owes for comparable goods and conditions. | Include mandatory fees and discounts, not just advertised price. |
| Quote | A price and conditions communicated for a stated validity period. | An internal recommendation is not a customer commitment. |
| Price floor | Minimum allowed price for a defined item and context. | A margin floor alone may be inadequate. |
| Price ceiling | Maximum permitted price under policy, contract, or law. | A voluntary ceiling is not necessarily a legal cap. |
| Change limit | Maximum permitted price movement over a stated interval. | Define whether measured in currency units or percent. |
| Capacity | Maximum fulfillable orders, units, or service time in a period. | A sales forecast is not capacity. |
| Opportunity cost | Value of the next-best use of constrained capacity. | Not necessarily the accounting cost of a unit. |
| Contribution | Revenue less specified variable costs and incremental offer costs. | Not profit after fixed overhead unless explicitly included. |
| Cannibalization | Sales displaced from the firm's other products or offers. | Do not count all shifted sales as net new. |
| Pull-forward | Purchases moved earlier from a later period. | A short-window uplift need not be incremental over time. |
| Eligible population | All units entitled to be assigned an intervention before assignment. | Buyers alone omit nonconversion. |
| Assignment unit | Unit randomized to a policy, such as customer, store, or market. | Repeated sessions from one person may not be independent. |
| Exposure | Offer actually seen by the customer. | Assignment is not necessarily exposure. |
| Intention to treat (ITT) | Effect of assignment, analyzed by assigned group. | Do not restrict to people who purchase or comply. |
| Estimand | Precisely defined target effect, population, contrast, outcome, and horizon. | A metric name alone is not an estimand. |
| Counterfactual | Outcome that would have occurred under an alternative intervention. | Cannot observe both alternatives for one unit at one time. |
| Confounding | A common cause of treatment selection and outcome. | Correlation after targeting is not automatically causal. |
| Positivity | Each relevant unit has a nonzero chance of each compared assignment. | No observational adjustment can recover an unsupported comparison without assumptions. |
| Interference | One unit's assignment affects another unit's outcome. | Shared capacity can violate individual-level independence. |
| Censoring | The observed outcome is limited or cut off, for example by stockout. | Zero sales during a stockout are not necessarily zero demand. |
| Randomization | Assignment by a known chance mechanism. | Rotating offers by day may confound with day effects. |
| Holdout | A group assigned to the existing policy or other comparison policy. | A convenience group is not necessarily a valid holdout. |
| Propensity | Probability of assignment under a known or estimated policy. | A model score is not a probability unless calibrated for that purpose. |
| Overlap | Availability of comparable treatment alternatives for the relevant population. | A sparse edge of the data may not support personalization. |
| Heterogeneous effect | Treatment effect that varies across a specified subgroup. | A subgroup's raw conversion difference is not a treatment effect. |
| Protected characteristic | A characteristic protected by applicable law in a given context. | The list and legal test differ by jurisdiction. |
| Proxy | Variable correlated with another characteristic. | Removing the sensitive variable does not remove every proxy. |
| Fairness metric | A defined numerical comparison across groups. | No single metric proves fairness or compliance. |
| Calibration | Agreement between predicted and observed quantities for a defined population. | Calibration does not itself identify a causal response. |
| Regret | Loss relative to a stated benchmark policy. | A model cannot know the true best policy from one observed outcome. |
| Drift | Change over time in inputs, outcomes, or relationships. | Not all seasonal variation is model deterioration. |
| Logging | Recording decision inputs, rule version, assignment, and offer outcome. | Collection must be minimized and access-controlled. |
| Rollback | Return to a specified safe policy after a trigger or incident. | A rollback must preserve already issued commitments. |

## 5. The mechanics, step by step

### Step 1 — Specify the decision and the accountable owner

**Input:** the business problem, product, channel, market, and decision deadline.
**Output:** a decision statement with one accountable owner and a defined action set.

Write the decision in operational language.
For example: choose the public delivery-slot charge for the next four weeks in a particular city.
State whether the decision changes the product price, the delivery fee, a discount, or the conditions of service.
Different components can have different tax, contract, and display requirements.
Do not report a lower product price as a customer benefit if a mandatory fee more than offsets it.

Name the decision maker who can authorize the policy.
Identify finance, operations, legal, privacy, customer service, and engineering reviewers.
The pricing team may own the commercial recommendation, but it cannot independently authorize a contract exception outside its authority.
If a pricing service fails, designate who can suspend it and what rule takes over.

Specify the decision horizon.
A daily replenished delivery slot has a different learning cycle from an annual business contract.
The outcome horizon must be long enough to include normal cancellation, returns, and substitution.
When it cannot be, label the missing effects and treat the estimate as provisional.

**Branch:** if no owner can honor the price or authorize rollback, do not launch.
If the action is only a forecast for planning, do not present it as an approved price.

### Step 2 — Separate time variation from customer variation

**Input:** proposed decision variables and how they enter the rule.
**Output:** a classification of dynamic, personalized, both, or neither.

List every variable affecting the customer-visible amount.
Include eligibility, base price, shipping, rebates, coupon targeting, taxes where applicable, and rounding.
Determine whether two otherwise comparable customers at the same time can receive different offers because of customer information.
If they can, document the information and why it is needed.

A published early-booking discount available to anyone who meets the same timing condition is a time or purchase-condition fence.
A unique offer generated from account history is personalized even if the underlying list price stays fixed.
A membership discount may have a public eligibility rule, but it still creates a group-level difference that needs review.
The classification is an analytical aid, not a determination of legal status.

Ask whether the same business purpose could be served by a less intrusive rule.
If the objective is to shift demand from peak to off-peak, compare a public time-based schedule with individual targeting.
If it achieves the operational aim, the simpler schedule may reduce data use and explainability burdens.
It can still affect groups differently if access to off-peak service is unequal.

### Step 3 — Define value and the decision baseline

**Input:** current policy, incremental cost, outcomes, and planning horizon.
**Output:** a contribution definition and the comparison policy.

Define contribution per eligible unit as collected revenue less specified variable product, service, payment, and incentive costs.
Add return and cancellation costs when the observation window permits.
In a capacity-constrained setting, separately report units of scarce capacity and their opportunity cost.
Do not silently substitute revenue for contribution.

Select the baseline that would actually be used without the proposal.
A comparison with no offer is misleading if the current business always offers a public discount.
If the existing policy also changes over time, record its rules and estimate the effect of assignment to the new policy versus assignment to the current policy.
Include the cost of operating the new system when moving from test economics to a deployment business case.

Count outcomes for every eligible assigned unit, including those with no purchase.
If purchase value is measured only among purchasers, conversion changes can produce a misleading improvement in average basket value.
Report both contribution per eligible unit and total eligible units.
For a store-randomized test, construct the estimator and uncertainty at the store or assignment level, accounting for the design.

### Step 4 — Map constraints before optimization

**Input:** contract terms, applicable law, approved commercial rules, system capabilities, and capacity.
**Output:** the feasible set and an exception path.

Write constraints as operational rules with units and owners.
A simple per-order feasible set can include a minimum and maximum amount, a limit on changes per day, and an available capacity requirement.
A quote already accepted by a customer must remain honored according to its terms.
The policy may need a required display of mandatory charges, eligibility, or automated-personalization information depending on jurisdiction.
A published business rule can be more restrictive than the legal minimum.

Identify the order of operations.
The service first determines eligibility and cost basis, then selects only among approved prices, then checks the all-in displayed amount and quote validity.
Clamping an unconstrained model output to a floor and ceiling can create a feasible number but does not resolve an impermissible input or discriminatory objective.
A prohibited attribute should not be fed to a model merely because the output later falls within a price band.

State the exception behavior.
For unavailable capacity, the safe response might be to offer a later slot rather than claim an unavailable one can be purchased.
For missing data, the system might use a preapproved public tariff.
For a legal hold or contract conflict, it may require human review.
Log the reason code and prevent the fallback from rewriting an outstanding quote.

### Step 5 — Define segments and review their meaning

**Input:** candidate segments, business purpose, lawful basis for processing where required, and sufficient sample.
**Output:** an approved segmentation specification or a decision not to segment.

Describe each segment using variables available at the moment of assignment.
Avoid variables created after purchase or after exposure; those would leak outcome information into the decision.
Document segment size, stability, missingness, update frequency, and whether a customer can contest an incorrect classification.
A high-value account classification can reflect a sales team's earlier discount decisions rather than independent willingness to pay.

Distinguish features for decisioning from features used only for auditing.
Sensitive information may have stricter collection and use rules, and collecting it for fairness measurement also requires an appropriate legal and privacy basis.
If direct sensitive attributes cannot lawfully be used, a review may still examine available aggregate evidence and known proxy pathways.
Do not infer individual protected status from a proxy and treat that inference as a verified fact.

For each proposed segment, ask what customer-facing explanation is accurate.
An explanation such as 'this delivery window has a published peak charge' is materially different from 'your device indicates you will pay more.'
A commercially useful segment that cannot be justified or explained may be an unacceptable policy choice.
This is a governance decision, not a statistical threshold alone.

### Step 6 — Draw the causal pathways

**Input:** historical decision records and knowledge of how prices were set.
**Output:** a causal diagram and a list of threats to identification.

Map demand forecasts, seasonality, inventory, marketing, competitor actions, and account selection to both price and outcomes where relevant.
Ask which variables were known before assignment and which are consequences of it.
A post-price purchase, basket size, or cancellation is generally an outcome or mediator, not a baseline confounder to adjust away casually.

A demand shock can raise both the price and the number of orders.
This can make a higher price appear to increase demand even if the price itself reduces conversion.
Conversely, a retailer may discount slow-moving inventory, making the discount look associated with low demand.
Both examples demonstrate why the sign of historical bias is not predetermined.

Document stockouts and capacity rationing.
Observed fulfilled orders equal demand only when the product or slot was available and customers could complete the purchase.
If a test changes capacity allocation, the outcome includes that mechanism; do not interpret it as a pure price sensitivity estimate.

### Step 7 — Choose a learning design

**Input:** eligible population, feasible actions, expected interference, and operational risk.
**Output:** a test protocol or an explicit reason why causal evidence is unavailable.

When feasible, randomize eligible units between the current policy and a proposed policy.
Assign before the customer sees the offer and record the probability of each assignment.
Use persistent customer-level assignment when repeated visits could otherwise expose one customer to conflicting quotes.
Honor prior quotes even when a later assignment or tariff changes.

When customers share a scarce delivery slot, individual assignment can create interference.
A treatment customer occupying a slot reduces capacity for a control customer.
Consider randomizing comparable markets, routes, or time blocks if these units can be separated operationally.
Choose the cluster unit before examining results, and account for the smaller effective number of independent clusters.
Cluster randomization can reduce, but does not automatically eliminate, spillovers between nearby markets or time blocks.

Record randomization, stratification, and planned analysis before launch.
Stratify on preassignment factors that materially affect demand or capacity when the design supports it.
Predefine the effect of assignment as the primary estimand, even if some customers do not see the assigned offer.
An analysis of the effect of actual exposure requires additional assumptions or a suitable instrumental-variable design; it is not simply the exposed-versus-unexposed difference.

If randomization is infeasible, document the alternative identification strategy.
A difference-in-differences design requires a defensible parallel-trends assumption and no material coincident intervention that differs across groups.
An instrumental variable requires a relevant assignment mechanism and an exclusion restriction, among other assumptions.
Neither is validated merely by adding many control variables.
When overlap or identifying assumptions are implausible, report the evidence as descriptive and narrow the decision.

### Step 8 — Specify measurement and guardrails before assignment

**Input:** decision economics, customer commitments, test design, and risk tolerance.
**Output:** a signed metric dictionary and stopping plan.

For each eligible unit, log assignment, assignment probability, timestamp, context, displayed price and fees, actual exposure, order outcome, and rule version.
Link later cancellations, returns, refunds, and fulfillment costs using a privacy-preserving identifier.
Record missingness and failed exposures rather than deleting them from the primary analysis.
Keep the allocation log separate from any optional demographic audit data where appropriate.

Define the primary metric, unit, horizon, and denominator.
A useful primary metric is incremental contribution per eligible customer over 14 days, including zero contribution for nonbuyers.
Secondary metrics can include conversion, average effective price, orders per scarce slot, cancellation, complaints, and retention.
A customer complaint rate must have a consistent denominator and should not be treated as a complete measure of harm.

Set operational guardrails before observing effects.
Examples include a hard cap on slot utilization, a ceiling on quote errors, and a limit on complaint-rate deterioration.
A hard safety or compliance trigger can stop a test regardless of statistical significance.
Specify who can pause it, who informs customers, and whether preexisting commitments continue.

Avoid repeatedly testing the same result and stopping when a conventional fixed-horizon p-value first crosses a threshold.
Use a prespecified fixed observation period or a valid sequential design with its own error control.
Do not claim a subgroup win merely because one of many exploratory comparisons is significant.

### Step 9 — Estimate the causal effect and its uncertainty

**Input:** all eligible assignments and linked outcomes through the stated horizon.
**Output:** an intention-to-treat estimate, uncertainty interval, and data-quality assessment.

For equal randomization without clustering, subtract mean contribution per eligible control unit from mean contribution per eligible treatment unit.
This estimates the average effect of assigning the new policy in the test population.
When assignment probabilities vary, use an estimator consistent with the design and its known probabilities.
When randomization is clustered, quantify uncertainty at the cluster level or use an appropriate design-based procedure.
Do not use the number of transactions as the independent sample size when whole stores were assigned together.

Show the number assigned, exposed, purchasing, canceled, and missing for each arm.
A difference in exposure can be a consequence of the assignment or a system defect; it is not a reason to silently exclude units.
Investigate assignment imbalance, duplicate identifiers, logging failures, price mismatches, and capacity changes.
Balance checks are diagnostics, not a replacement for a valid randomization mechanism.

Estimate outcomes over the full horizon and report uncertainty, not only a point estimate.
Compare the interval with the minimum improvement needed to cover operating cost and risk.
A non-significant result does not prove no effect; it can be compatible with a range of gains and losses.
An estimate from one city and season should not automatically be transferred to all cities and seasons.

### Step 10 — Evaluate distribution, privacy, and trust

**Input:** customer journeys, eligibility rules, permissible audit data, and disaggregated outcomes.
**Output:** an impact assessment, unresolved risks, and a decision by accountable reviewers.

Compare who receives each offer, who can use it, and who bears the effective price.
A public time fence can disadvantage people with fixed working hours even though the same rule applies to all.
A member-only discount can exclude people without a bank account or reliable digital access.
A favorable average contribution result does not establish that these impacts are acceptable.

Choose group comparisons based on a documented harm model and legally permissible data.
Possible measures include offer access, average all-in price for comparable baskets, conversion, complaint rate, and availability of an alternative channel.
Define each denominator, comparator, and minimum sample size.
Compare like service conditions where possible, but do not condition away the very mechanism that makes an offer inaccessible.
Report uncertainty and small-cell suppression; absence of evidence in small groups is not evidence of equal treatment.

Examine how a customer would understand and challenge the price.
Show the price, mandatory fees, eligibility, and expiration before commitment.
Provide a channel to correct account data or dispute an incorrectly applied offer.
A disclosure does not cure every unfair, deceptive, discriminatory, or otherwise unlawful practice.

### Step 11 — Select, approve, and implement a policy

**Input:** causal estimates, cost, capacity, risk assessment, and approved action set.
**Output:** a signed decision with an operational configuration and effective dates.

Compare each feasible policy with the current policy on contribution and nonfinancial guardrails.
If the expected gain is smaller than the cost to operate or the uncertainty is too wide for the risk, continue the current policy or run a better test.
If the system cannot fulfill the increased volume, include the cost of additional capacity or reject the policy.
Do not call a modeled optimum a commitment until the responsible owner approves it.

Version the policy, its input definitions, price bands, expiration, and rollback rule.
Test the exact price displayed through the entire customer journey and on relevant channels before release.
Require customer-service training and a route for correcting an incorrect quote.
Maintain an immutable or otherwise controlled record of actual offers and decisions for the applicable retention period.

Launch in a bounded scope when appropriate.
A gradual rollout should use a preapproved allocation and monitored control if the firm wants a causal comparison.
A gradual rollout alone is not a randomized experiment.
Retain the ability to stop the policy without canceling customer commitments.

### Step 12 — Monitor, investigate, and retire

**Input:** production outcomes, customer feedback, rule changes, and business context.
**Output:** periodic review, change record, and a decision to continue, revise, or retire.

Monitor data freshness, eligibility errors, the distribution of recommended and displayed prices, capacity, and financial outcomes.
Track both aggregate and permitted subgroup measures against prespecified thresholds.
Check whether prices are concentrated at a floor or ceiling, which can signal a changed environment or an unsuitable model.
Review whether a previously justified segment still corresponds to a legitimate business purpose.

Drift in conversion does not by itself prove the policy caused the change.
Seasonality, competitors, stock availability, and the customer mix can all change.
Use a maintained holdout where ethically and operationally appropriate, or design a new test when a causal estimate is needed.
A held-out group must receive a valid offer, not a degraded or undisclosed customer experience.

Review complaints and correction requests qualitatively as well as quantitatively.
One serious case may reveal a systematic defect even when the aggregate rate is low.
Record incident ownership, customer remediation, and the evidence required to restore service.
Retire data and access when their approved purpose and retention period end.

## 6. Worked example

### The operating decision

A fictional grocery distributor, Harbor Basket, offers scheduled delivery in one city.
It has experienced peak-hour route congestion and idle capacity during midday.
Management is considering a public peak delivery charge in place of a flat delivery charge.
All customers who choose the same slot face the same charge under the proposed schedule.
The proposal uses time of service, not a customer-specific willingness-to-pay estimate.

The objective is to increase contribution while keeping peak orders within route capacity.
Finance counts product and delivery revenue less product cost, delivery variable cost, payment cost, and refunds.
The delivery charge is included in revenue; it is not added a second time to contribution.
All amounts below are United States dollars (USD).
The measurement unit is one eligible customer over a 14-day outcome window.
The example is illustrative; its figures are not evidence of a real company's results.

### The two policies

Policy A is the existing public flat delivery fee of USD 4.00 per completed order.
Policy B is a public USD 5.50 peak fee and a USD 3.00 off-peak fee per completed order.
Both policies show the complete delivery fee before checkout.
Both honor a displayed quote for 30 minutes while the slot remains reserved under the stated terms.
Product prices, service quality, and the slot menu are unchanged by design.

Policy B is dynamic in the sense that the fee differs by scheduled time.
It is not individualized by customer history in this example.
A customer choosing the same slot receives the same fee as another eligible customer under the same assigned policy.
A randomized experiment can temporarily show different policies to different assigned customers; that experimental assignment itself needs review, disclosure where required, and a consistent quote policy.

### Assignment and limitations

For teaching arithmetic, suppose 2,000 eligible customers are randomly assigned equally to each policy.
Assignment persists for each customer throughout the experiment.
The business logs every eligible customer before the offer is displayed.
Outcomes are collected for 14 days after assignment, including customers who do not purchase.

Assume the experiment reserves enough separate test capacity for both arms that one arm does not block the other.
In a real shared route this assumption would be questionable.
If separate capacity is infeasible, randomize independent routes or time blocks and analyze at the assignment-unit level.
The numerical illustration below does not provide enough information to calculate a defensible confidence interval or establish statistical significance.

### Observed results

| Measure over 14 days | Policy A: current | Policy B: proposed |
|---|---:|---:|
| Eligible customers assigned | 2,000 customers | 2,000 customers |
| Customers placing an order | 600 customers | 620 customers |
| Completed peak orders | 360 orders | 280 orders |
| Completed off-peak orders | 240 orders | 340 orders |
| Total contribution after specified variable costs | USD 7,800 | USD 8,370 |
| Peak slots reserved for this arm | 400 orders | 400 orders |

Suppose each ordering customer places exactly one completed order in the observation window.
The table's orders then reconcile with the number of ordering customers.
If customers could place multiple orders, the number of orders and the number of customers must be separately recorded.
Suppose the stated contribution totals already include the delivery charges, product margin, fulfillment, payment, and observed refund costs.
They exclude incremental annual platform costs, which are considered separately below.

### Calculate conversion and contribution

Policy A conversion is 600 / 2,000 = 0.30, or 30.0% of assigned eligible customers.
Policy B conversion is 620 / 2,000 = 0.31, or 31.0%.
The difference is 1.0 percentage point, not a 1.0% relative increase.
The relative change in conversion is 0.01 / 0.30 = 3.33%, rounded.

Policy A contribution per eligible customer is USD 7,800 / 2,000 = USD 3.90.
Policy B contribution per eligible customer is USD 8,370 / 2,000 = USD 4.185.
The estimated difference is USD 4.185 − USD 3.90 = USD 0.285 per eligible customer over 14 days.
Among 2,000 assigned customers, the observed difference in total contribution is USD 570.
That total difference is useful for reconciliation but the per-eligible effect is the transport unit used in the forecast.

The average contribution per order is USD 7,800 / 600 = USD 13.00 under A.
Under B it is USD 8,370 / 620 = USD 13.50.
Those descriptive purchaser averages are not the primary causal estimand.
They condition on ordering, which the policy can affect.
The ITT comparison includes the 1,400 A and 1,380 B customers who did not order.

### Calculate the capacity effect

Peak orders fall from 360 to 280 in the equally sized groups.
The observed reduction is 80 peak orders per 2,000 assigned customers.
Peak use falls from 360 / 400 = 90% to 280 / 400 = 70% of the separately reserved capacity.
Off-peak orders rise by 100, from 240 to 340.
The 20-order overall increase is consistent with the difference between the peak decrease and off-peak increase.
These figures do not prove that 80 specific customers moved from peak to off-peak; individual counterfactual moves are unobserved.

The 400-order capacities are separate experimental allocations, not a single pool available twice in production.
Before scaling, the planner must verify off-peak capacity can handle the additional 100 orders per 2,000 eligible customers.
The change in slot mix may also change route length and labor cost.
Those costs must be in the contribution measure or separately added to the scale-up case, but not counted twice.

### Build a conditional forecast

Suppose the city expects 20,000 eligible customers in a comparable 14-day period.
A simple forecast using the point estimate is 20,000 × USD 0.285 = USD 5,700 incremental contribution per 14 days.
The same linear scaling forecasts 20,000 / 2,000 × 80 = 800 fewer peak orders per 14 days.
It also forecasts 1,000 more off-peak orders per 14 days.
These are forecasts, conditional on comparable customer mix, available capacity, no interference, stable costs, and transportable response.
They are not guaranteed results.

Suppose the incremental operating expense for the policy is USD 1,200 per 14-day period and is not already included in contribution.
The point-estimate net value after that expense is USD 5,700 − USD 1,200 = USD 4,500 per 14 days.
The break-even incremental contribution is USD 1,200 / 20,000 = USD 0.06 per eligible customer.
Without an interval around the USD 0.285 estimate, the decision maker cannot judge how confidently the effect exceeds USD 0.06.
A larger test or properly analyzed repeated experiment may be needed.

The 20,000-customer forecast is not a plan.
A plan would state the approved rollout scope, staffing, budget, and effective dates.
A target might be a 10% reduction in peak orders, chosen by management rather than inferred from the experiment.
A commitment is the 30-minute quote promised to a customer under the stated reservation terms.
No forecast changes that customer commitment.

### Make the decision

The observed arithmetic suggests that B could improve contribution and reduce peak congestion.
It does not establish statistical confidence, long-term retention, or acceptable distributional impact.
The commercial owner should request uncertainty estimates, capacity confirmation, and a customer-impact review before production approval.
The reviewer should also verify that the actual price and fee displays satisfy applicable local requirements.

If B is approved, publish the slot schedule, monitor effective prices and complaints, and retain the current schedule as a preapproved rollback.
Honor every valid quote issued before rollback.
If off-peak capacity is insufficient or the legal review is unresolved, do not scale B.
If an audit shows that customers with inflexible schedules disproportionately bear the peak fee, evaluate mitigation such as additional affordable slots rather than assuming a public rule is automatically fair.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It reproduces the descriptive test arithmetic and the conditional scale-up calculation.
It does not estimate uncertainty, simulate randomization, or determine legal compliance.
Each arm has equal assigned customers, so the simple difference of arm means is appropriate for the displayed point calculation.
The aggregate data cannot support a customer-level variance estimate.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    name: str
    assigned: int
    ordering_customers: int
    peak_orders: int
    off_peak_orders: int
    contribution_usd: float
    peak_capacity: int

    def conversion(self) -> float:
        return self.ordering_customers / self.assigned

    def contribution_per_eligible(self) -> float:
        return self.contribution_usd / self.assigned

    def peak_utilization(self) -> float:
        return self.peak_orders / self.peak_capacity


current = Arm(
    name="A",
    assigned=2000,
    ordering_customers=600,
    peak_orders=360,
    off_peak_orders=240,
    contribution_usd=7800.0,
    peak_capacity=400,
)
proposed = Arm(
    name="B",
    assigned=2000,
    ordering_customers=620,
    peak_orders=280,
    off_peak_orders=340,
    contribution_usd=8370.0,
    peak_capacity=400,
)

# This sample records exactly one completed order per ordering customer.
for arm in (current, proposed):
    assert arm.peak_orders + arm.off_peak_orders == arm.ordering_customers
    assert arm.peak_orders <= arm.peak_capacity

increment_per_eligible = (
    proposed.contribution_per_eligible()
    - current.contribution_per_eligible()
)
conversion_pp = 100 * (proposed.conversion() - current.conversion())
peak_change_per_eligible = (
    proposed.peak_orders / proposed.assigned
    - current.peak_orders / current.assigned
)

forecast_eligible = 20000
incremental_operating_cost_usd = 1200.0
forecast_contribution_usd = forecast_eligible * increment_per_eligible
forecast_net_usd = forecast_contribution_usd - incremental_operating_cost_usd
break_even_usd = incremental_operating_cost_usd / forecast_eligible
forecast_peak_change = forecast_eligible * peak_change_per_eligible

print(f"A conversion: {current.conversion():.1%}")
print(f"B conversion: {proposed.conversion():.1%}")
print(f"Conversion difference: {conversion_pp:+.1f} percentage points")
print(f"A contribution per eligible: USD {current.contribution_per_eligible():.3f}")
print(f"B contribution per eligible: USD {proposed.contribution_per_eligible():.3f}")
print(f"Increment per eligible: USD {increment_per_eligible:.3f}")
print(f"A peak utilization: {current.peak_utilization():.1%}")
print(f"B peak utilization: {proposed.peak_utilization():.1%}")
print(f"Forecast contribution: USD {forecast_contribution_usd:,.0f}")
print(f"Forecast net after operating cost: USD {forecast_net_usd:,.0f}")
print(f"Break-even increment per eligible: USD {break_even_usd:.2f}")
print(f"Forecast peak-order change: {forecast_peak_change:+,.0f} orders")
```

Expected output, allowing ordinary floating-point rounding:

```text
A conversion: 30.0%
B conversion: 31.0%
Conversion difference: +1.0 percentage points
A contribution per eligible: USD 3.900
B contribution per eligible: USD 4.185
Increment per eligible: USD 0.285
A peak utilization: 90.0%
B peak utilization: 70.0%
Forecast contribution: USD 5,700
Forecast net after operating cost: USD 4,500
Break-even increment per eligible: USD 0.06
Forecast peak-order change: -800 orders
```

The assertions check only the example's stated order accounting and peak capacity.
They do not validate the experiment design, completeness of costs, or comparability of the two groups.
A production implementation should use an appropriate money representation and retain original currency precision.
The code uses floating-point arithmetic only for this illustrative reporting calculation.
The forecast treats the point estimate as stable and proportional to eligible volume.
Actual deployment must incorporate uncertainty and operational constraints.

## 8. Data, standards, and regulatory context

### Minimum data contract

Start with a documented event for every eligible unit, including people who do not buy.
A transaction table alone omits nonbuyers and cannot supply the correct assignment denominator.
The event should specify its timestamp, market, channel, eligible policy set, assignment unit, assigned policy, and assignment probability.
Record the rule version and data snapshot used at assignment.

A separate exposure event records what the customer actually saw.
Store the displayed product price, mandatory fees, discount, currency, eligibility terms, and quote expiry.
Record delivery-slot availability and reservations at the relevant time.
A later order event should connect to payment, cancellation, return, refund, and fulfillment costs.
Each record needs an event definition and a clear effective time.

Customer identifiers should be stable enough to prevent inconsistent repeated assignment and link outcomes.
Use pseudonymous identifiers and access restrictions where appropriate.
Pseudonymization reduces some risks but does not necessarily make data anonymous under applicable law.
Set retention based on purpose, legal obligations, and policy rather than retaining all behavioral history indefinitely.
Keep a correction process for wrong account or eligibility information.

A price log should distinguish assigned price, displayed price, accepted quote, charged amount, and refunded amount.
They can differ because of defects, tax changes, coupon stacking, or valid customer actions.
Without those distinctions, a model evaluation may attribute a checkout error to customer price sensitivity.
Reconcile amounts to the order and finance systems using consistent currency and time buckets.

### Data quality checks

Check duplicate assignments, missing assignment probabilities, and treatment changes within a quote-validity period.
Compare counts of assigned, exposed, and purchased units by arm and channel.
Investigate a sudden drop in exposure before interpreting a drop in conversion.
Check that capacity and stock were available when the offer was shown.
Record when a slot filled, rather than treating all nonpurchases as price rejection.

Check costs against the period in which the order was fulfilled.
A delivered order's route cost may be posted after the initial transaction.
Choose a close date for the analysis and quantify costs still missing at that date.
Where estimated costs are used, label them and assess sensitivity to the estimate.
Do not mix revenue including tax with costs and revenue excluding tax without an explicit accounting convention.

For fairness audits, first decide whether the data can lawfully and responsibly be collected or processed for that purpose.
Describe the basis, access, purpose, retention, and aggregation safeguards.
If small group counts make estimates unstable or risk reidentification, suppress or combine cells only with a documented rationale.
Suppression protects privacy but also limits what can be concluded about those groups.

### European Union consumer information

As of November 2026, Directive 2011/83/EU, as amended by Directive (EU) 2019/2161, includes a precontractual information requirement concerning a price personalized on the basis of automated decision-making in the relevant consumer-contract context.
Its exact application depends on the transaction and national implementation.
The directive does not mean that every time-varying public price is necessarily a price personalized on that basis.
A firm should review the actual eligibility, interface, and implementation with qualified local counsel.
See the consolidated Consumer Rights Directive and the amending directive in the sources.

EU price-indication rules are also relevant where a seller announces a price reduction.
Directive 98/6/EC, as amended, contains rules on indicating a prior price, subject to its scope and national implementation.
Do not assume that every personalized offer, ordinary fee, or price change is the same as a public reduction announcement.
The exact presentation and applicable exceptions require review.

### European Union personal data

The General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, applies when its material and territorial conditions are met.
It contains principles including purpose limitation, data minimization, accuracy, storage limitation, and security.
A controller also needs an applicable legal basis for processing personal data.
Information duties, individual rights, and any impact-assessment requirement depend on the facts.

Article 22 addresses certain decisions based solely on automated processing, including profiling, that produce legal or similarly significant effects, subject to its conditions and exceptions.
It is not correct to say that every algorithmic price automatically falls within Article 22.
Nor is it safe to assume that pricing can never produce a similarly significant effect.
Document the use, the human role, likely effects, and legal advice for the particular context.
Special-category data and fairness-audit data raise additional questions under applicable law.

### United States and other jurisdictions

In the United States, the Federal Trade Commission (FTC) Act prohibits unfair or deceptive acts or practices within the FTC's authority.
The FTC's 2025 staff report on surveillance pricing describes the collection and use of personal data in pricing and raises consumer-protection concerns.
A staff study is evidence of agency attention, not itself a statute or a finding that every personalized price is unlawful.
Federal and state rules, sector rules, contract terms, and antidiscrimination law can also matter depending on the transaction.
Do not infer a universal nationwide ban or universal permission from one source.

For business-to-business sales, competition and price-discrimination rules can depend on the parties, products, transaction, defenses, and jurisdiction.
The U.S. Robinson-Patman Act is a specific federal statute, not a blanket prohibition on every difference between business quotes.
Contract and legal specialists should examine the relevant facts before offers are implemented.
Competition law also constrains coordination with competitors; independent price research is not permission to exchange future pricing intentions.

Other jurisdictions may have different consumer, privacy, competition, and anti-discrimination requirements.
A global pricing system needs a market-specific rules inventory, not a single worldwide assumption.
Record the date and owner of each legal interpretation and review changes to the law and product design.
This educational discussion is not legal advice.

### Standards and conventions

The National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 is a voluntary framework for mapping, measuring, managing, and governing AI risks.
It does not itself authorize a price or replace binding law.
Its governance concepts are useful even where a policy is a simple rule rather than machine learning.

GS1 identifiers can help distinguish products and trading items across systems where adopted.
They are identification conventions, not evidence that two offers have identical service conditions or that a price is fair.
A common identifier should be joined to versioned product, pack size, currency, delivery terms, and effective date.
This prevents a pack-size change from being mistaken for a price change.

## 9. Industry practice and case evidence

### Public transport and capacity-based variation

Rail, airlines, and hotels commonly sell time-sensitive capacity at different prices or conditions.
A seat or room not sold before departure or occupancy cannot generally be stored for the next day.
This economic feature supports the business case for controlling availability and prices over time.
It does not establish that every individualized offer is justified.

The U.S. Department of Transportation's airline consumer information describes the requirement to disclose total airfare including mandatory government taxes and fees in applicable advertised prices.
That example concerns how a price is presented, not a general endorsement of any fare-setting algorithm.
The operational lesson is to evaluate the total customer-visible amount and applicable rules, not just an attractive base fare.
Taxes, optional services, and regulatory requirements vary by product and jurisdiction.

Airline revenue management also illustrates an interference problem.
Selling one seat to an early buyer removes it from the capacity available to a later buyer.
A customer-level price test that ignores this interaction can estimate the effect of a mixed allocation rather than a clean standalone pricing policy.
Littlewood's rule provides a stylized capacity-allocation framework under restrictive assumptions; see the module's revenue-management topic when available.

### Uber surge pricing

Uber publicly explains that prices can rise during periods of high demand and that the app shows an upfront price before a trip is requested in markets where upfront pricing is offered.
Its public description distinguishes a time-and-place market condition from an individual customer's personal characteristics, but it should not be read as a complete audit of every input to every product.
The documented business rationale is balancing rider demand and driver availability.
The precise effect on supply, demand, and customer welfare varies with conditions and the design of the market.

[INFERENCE] A logistics company considering a peak delivery fee can borrow the operational idea of aligning demand with scarce capacity.
It cannot infer Uber's causal results, regulatory obligations, or acceptable customer experience from a description of another market.
It should test its own policy and show a reliable all-in delivery charge before the customer commits.
A temporary surge mechanism also needs a clear rule for emergencies, exceptional events, and public trust.

### FTC surveillance-pricing inquiry

The FTC's January 2025 staff report describes a study of firms that supply pricing products using personal data and identifies a range of data sources and possible uses.
It is a real regulator's account of an inquiry, not a randomized estimate of the effect of a particular retailer's pricing policy.
The report highlights why an organization should map vendors, data sources, and customer-facing outcomes instead of assuming a vendor's model is a neutral input.
An internal owner remains responsible for understanding the deployed rule and its consequences.

[INFERENCE] A retailer buying a third-party optimization service should request an inventory of input categories, decision outputs, update frequency, and audit rights.
The exact obligations depend on the contract and applicable law.
A black-box vendor claim that a model improves margin does not establish incremental contribution, fairness, or compliance.
The retailer should be able to test a policy against a valid comparison and disable it without breaking existing quotes.

### Online experimentation evidence

Kohavi and coauthors describe controlled experiments on the web and practical challenges such as trustworthy assignment, measurement, and interpretation.
Their work supports a disciplined experiment process; it does not guarantee that an arbitrary pricing test is ethically or legally acceptable.
The sample, assignment mechanism, interference, and outcome horizon still determine what a particular experiment can identify.
The business should assess customer impact before testing, not only after a positive result.

Randomization may be technically easy when a web service can assign a price.
It is operationally harder when a customer sees a price on multiple devices, contacts a service agent, or purchases after a quote expires.
An experiment must specify identity resolution, persistent assignment, customer communications, and the handling of an existing quote.
A mismatch between tested and deployed experiences undermines transportability.

### What the evidence does not show

A public case demonstrates that a practice exists or that a source makes a documented claim.
It does not prove the same practice is profitable in another company's market.
It does not settle the legality of a different jurisdiction or the fairness of a particular segment.
Distinguish source fact, interpretation, and proposed action in the decision packet.

For Harbor Basket, the USD 0.285 per eligible customer estimate is fictional illustrative arithmetic.
It is not evidence about Uber, an airline, or the FTC inquiry.
A real launch would require actual assignment records, customer outcomes, costs, and a review of the relevant legal environment.

### Business case comparison: three feasible policies

Consider three hypothetical ways to address a congested delivery window.
The figures in this comparison are illustrations, not results from a named company.
They are assessed against the same existing flat-fee baseline, customer population, and 14-day horizon.
Assume each estimate comes from an appropriate experiment with uncertainty reported separately.

| Policy | Customer-facing rule | Main operating question | Main trust question |
|---|---|---|---|
| Public peak fee | Same published charge for the same slot | Can off-peak routes absorb shifted demand? | Can customers realistically choose the lower fee? |
| Voluntary advance booking discount | Same published discount for qualifying lead time | Does advance demand improve route planning? | Is the booking window accessible and clearly explained? |
| Account-specific coupon | Discount depends on account information | Is incremental response large enough to cover targeting costs? | Why did one customer receive a different offer? |

The public peak fee directly changes the relative cost of using scarce peak capacity.
Its success should be judged on total contribution, peak utilization, and customer impact rather than the peak fee revenue alone.
The advance-booking discount rewards information that the operator values.
Its cost depends on how many customers would have booked early without an incentive.
The account-specific coupon can target an estimated incremental response, but its incremental value must exceed both the discount and the added operating and governance costs.

Suppose a proposed advance-booking discount costs USD 1.20 per redeemed order.
Among 1,000 eligible customers assigned to the discount, 300 redeem it, for USD 360 of discount expense.
Suppose contribution including the discount is USD 4,100 in the treatment group and USD 3,900 in an equally sized control group.
The estimated incremental contribution is USD 200 per 1,000 eligible customers, or USD 0.20 per eligible customer.
Do not subtract the USD 360 again: the stated treatment contribution already includes it.
The 300 redemptions do not imply 300 incremental orders.

Suppose the targeted coupon has an estimated incremental contribution of USD 0.24 per eligible customer before its separate targeting-platform cost.
The public booking discount has an estimated USD 0.20 per eligible customer and requires no incremental platform expense.
If the targeting platform costs USD 0.08 per eligible customer, the targeted coupon's point-estimate net increment is USD 0.16 per eligible customer.
On these point estimates alone, the public booking discount has the higher net contribution.
The decision still requires uncertainty intervals, capacity evidence, lawful data use, and distributional review.

If the USD 0.24 targeted estimate already included the platform cost, subtracting it again would be incorrect.
The accounting boundary should be printed beside every alternative in the decision table.
An optimization score that omits an operational cost cannot be directly compared with one that includes it.

### Business case comparison: a business-to-business quote

A fictional packaging supplier quotes a customer for 10,000 cartons delivered over a month.
Its current quote is USD 1.20 per carton, or USD 12,000 before applicable taxes and freight terms.
Variable production and delivery costs are estimated at USD 0.90 per carton.
If the entire order is incremental and those costs are complete, contribution is 10,000 × (USD 1.20 − USD 0.90) = USD 3,000.

A proposed discount to USD 1.15 per carton yields USD 2,500 contribution on the same 10,000-carton order.
The difference is USD 500 if order volume and all other conditions remain fixed.
The supplier should not assume that a USD 500 lower margin is justified merely because the customer asked for a discount.
It should ask whether the reduced price changes the probability, quantity, duration, or conditions of the order.
It should also consider whether reserved production capacity displaces a more valuable order.

Suppose the supplier estimates a 60% chance of winning at USD 1.20 and a 75% chance at USD 1.15.
Under the simplifying assumption of no other outcome costs, expected contribution is 0.60 × USD 3,000 = USD 1,800 at the higher quote.
It is 0.75 × USD 2,500 = USD 1,875 at the lower quote.
The point-estimate advantage of the discount is USD 75 for this opportunity.
That is a small difference relative to the uncertainty in the two win probabilities.

The probabilities must describe the same opportunity under alternative quotes, not two different customer populations.
Historical win rates can be confounded if sales teams discount only the hardest deals.
Contracts, competition law, and any applicable price-discrimination requirements also constrain the quote.
No numerical calculation here determines legal compliance.

If a customer commits to a minimum quantity or flexible delivery window, those conditions can change the economics.
Document how the condition changes the cost or capacity forecast rather than inventing a post hoc justification.
An approved term should be understandable to the sales team and the customer.
It must also be implemented consistently in the contract and order systems.

### Choosing between simple rules and an adaptive policy

A fixed public schedule is easy to explain but can respond slowly to an unexpected capacity shock.
An adaptive rule can react to changes in demand or supply, but it needs reliable data and a documented update cadence.
An account-specific model may add flexibility while also increasing privacy, audit, and customer-trust burdens.
The relevant comparison is not simple versus sophisticated in the abstract.
It is the incremental value and risk of each feasible policy for the decision at hand.

Before using an adaptive policy, specify what information can change the next offer.
State whether the policy updates hourly, daily, or only after a human review.
Separate the learning period from the period during which a customer holds a valid quote.
Log the policy version and assignment probability when random exploration is used.

An adaptive allocation can preferentially assign an apparently successful arm over time.
This changes the distribution of customers and conditions observed under each arm.
A naive pooled comparison may therefore be misleading even if the initial allocation was random.
The analysis should account for the actual allocation mechanism and time variation.
An experienced statistician should select a method appropriate to the adaptive design before launch.

A business can also choose not to personalize.
If a public capacity signal meets the objective with less data and a simpler customer explanation, that is a substantive design outcome.
It is not evidence that all personalized offers are inherently unlawful or ineffective.
The organization should document why the selected rule is proportionate to the problem and what evidence would cause it to be reconsidered.

### Customer service as evidence

Customer service records can reveal unexpected behavior that a conversion dashboard misses.
For example, a customer may believe that a previously quoted amount changed because a delivery slot expired.
The underlying issue may be a display that failed to explain the expiry or a system that did not preserve the quote.
These require different remedies.

Create categories for quote mismatch, unclear fees, eligibility disputes, unavailable lower-priced slots, and refund requests.
Record the policy version and channel where the issue arose.
Do not infer the frequency of all customer harm solely from complaint counts, because many affected customers do not complain.
Investigate representative cases, and verify the actual displayed and charged amounts.

Provide agents with the current public schedule and a way to honor or correct a valid quote.
An agent should not need to guess an individual model's reasoning.
Where personal data affect an offer, provide an appropriate process to correct incorrect information and route legal requests.
Record the resolution without retaining unnecessary sensitive narrative data.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Historical targeting mistaken for causality | High coupon redemption among previously targeted buyers | Buyers were selected for purchase propensity | Overstated incremental margin | Randomize assignment and count all eligible units. |
| Demand shock confounds price | Prices and sales rise together during holidays | Demand drives both price and volume | Wrong response estimate | Randomize within comparable periods or use a defensible design. |
| Shared capacity interference | One arm fills slots before the other | Assigned customers compete for supply | Biased or policy-mix-specific estimate | Reserve capacity or randomize separable clusters. |
| Stockout interpreted as low demand | Zero orders coincide with unavailable inventory | Fulfillment censors demand | Excessively low demand forecast | Log availability and analyze stockouts explicitly. |
| Buyer-only analysis | Basket value rises while conversion falls | Conditioning on purchase changes the population | Misleading business case | Measure contribution per eligible assigned unit. |
| Missing mandatory fee | Checkout total exceeds displayed comparison | Offer components were evaluated separately | Trust loss and possible legal exposure | Compare and disclose the applicable all-in amount. |
| Quote instability | Customers see a different amount during checkout | Repricing occurs inside a validity window | Abandoned orders and complaints | Persist and honor quotes under defined terms. |
| Proxy discrimination | Price differences track geography or device | Inputs correlate with sensitive characteristics | Disparate harm or legal risk | Review features, permissible group outcomes, and alternatives. |
| Unexplained targeted offer | Service agents cannot explain eligibility | Model or vendor rule is opaque | Customer distrust and correction failures | Document an accurate explanation and challenge process. |
| Too-small subgroup | Large effect in a handful of customers | Sampling variation and multiple comparisons | False personalization decision | Prespecify comparisons and report uncertainty. |
| Price band treated as full governance | All outputs are inside limits but inputs are sensitive | Output clamping ignores input and policy harms | Impermissible or unfair targeting persists | Review purposes, features, and outcomes before deployment. |
| Cost omission | Revenue grows while route expense rises | Delivery, returns, or opportunity cost excluded | Apparent profit disappears | Reconcile contribution and capacity costs. |
| Duplicate operating cost | Net forecast subtracts the same cost twice | Accounting boundary is unclear | A viable policy appears unattractive | Document each cost and whether already included. |
| Simpson's paradox | Aggregate effect differs from segment effects | Mix changes across relevant groups | Wrong diagnosis of customer impact | Inspect prespecified group rates and denominators. |
| Adaptive policy without logging | No record of assignment probabilities | The allocation rule changes as it learns | Evaluation and audit become difficult | Version decisions and log probabilities and context. |
| Repeated peeking | Launch occurs after the first favorable daily result | Repeated tests inflate false-positive risk | Unreliable rollout decision | Prespecify horizon or valid sequential procedure. |
| Vendor opacity | Contract cannot explain input categories or rollback | Control is delegated without accountability | Audit and remediation gaps | Require documentation, monitoring, and disablement rights. |
| Distribution shift | New channel or season has different customers | Tested response no longer transports | Forecast misses and inequitable outcomes | Revalidate with a bounded test and monitor drift. |
| Unusable fallback | Pricing service outage returns invalid amounts | Safe policy was not specified or deployed | Checkout failure or broken commitments | Preapprove a public fallback and preserve quotes. |
| Retention harm | Short-term margin rises while repeat orders fall | Customer resentment or pull-forward | Lower lifetime contribution | Extend outcome horizon and monitor retention. |

For each warning, investigate the underlying mechanism rather than simply suppressing the metric.
A red flag can coexist with a positive contribution estimate.
Escalation thresholds are business decisions set before launch, not proof that unflagged outcomes are safe.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Assignment is random as specified | Gives comparable groups in expectation | Code defect or manual override | Audit allocation logs and investigate imbalances. |
| Eligible population is logged before assignment | Defines a valid ITT denominator | Missing nonbuyers or post-treatment selection | Reconcile eligibility events with independent traffic counts. |
| Outcome windows are complete | Captures comparable contribution | Late refunds or fulfillment costs | Mature the cohort and report missing costs. |
| Separate capacity prevents interference | Supports interpretation of individual assignment | Shared route, inventory, or customer referrals | Inspect allocation; use cluster assignment if needed. |
| Policies are implemented as designed | Connects assignment to the tested offer | Fee or display mismatch | Compare assignment, exposure, quote, and charge logs. |
| Contributions use the same accounting boundary | Makes arm differences comparable | One arm excludes delivery or refund expense | Reconcile component definitions with finance. |
| One order per buyer in the illustration | Reconciles customers and orders in the table | Repeat orders during the window | Log order count and analyze customer-level totals. |
| Test capacity is not exceeded | Prevents unmodeled rationing | Unexpected peak demand | Monitor utilization and record availability. |
| The outcome horizon captures important effects | Supports the business case | Pull-forward, retention, and late returns | Extend follow-up or bound omitted effects. |
| Scale-up population resembles the test | Supports transport of the point estimate | New geography, channel, or season | Compare preassignment distributions and retest. |
| Incremental cost is not already in contribution | Prevents double counting | Platform expense included in arm totals | Trace the cost ledger before scaling. |
| Group audit data are lawful and sufficiently reliable | Makes subgroup analysis meaningful | Missing consent, proxy error, or small cells | Review legal basis and report limitations. |
| The public schedule is understood and accessible | Supports meaningful customer choice | Hidden fees or inaccessible off-peak slots | Test the customer journey and offer alternatives. |
| The current policy remains feasible as fallback | Enables rollback | Contract or capacity change | Reapprove fallback and test operational readiness. |

The example assumes no statistical uncertainty calculation because only aggregate totals are provided.
It also does not observe long-run behavior or competitor response.
These limitations prevent a claim that B is ready for unrestricted rollout.
A favorable point estimate is a prompt for a decision under uncertainty, not proof of one.

## 12. Exercises

### Exercise 1 — Classify the offer

A retailer proposes four rules.
A: everyone pays a published USD 2.00 peak-slot fee.
B: an account receives a unique coupon based on its browsing history.
C: every qualifying bulk order receives the same published discount.
D: a customer-specific price is recalculated each hour from account activity.

**Deliverable:** classify each as dynamic, personalized, both, or neither, and state one caveat.
**Acceptance criteria:** distinguish time variation from customer-specific information; do not equate public eligibility with legal approval.
**Time:** 10 minutes.

### Exercise 2 — Define the causal question

A manager compares conversion for coupon recipients with nonrecipients.
Recipients were selected because their previous purchase probability was high.

**Deliverable:** write an estimand, a causal threat, and an improved assignment design.
**Acceptance criteria:** include population, intervention, comparison, outcome, and time horizon; name selection as a threat and count nonbuyers.
**Time:** 15 minutes.

### Exercise 3 — Compute policy economics

A test assigns 1,000 customers to each policy.
Control produces USD 3,200 contribution and 250 peak orders.
Treatment produces USD 3,500 contribution and 190 peak orders.
A forecast covers 12,000 eligible customers, and incremental operating cost is USD 1,000 per forecast period.

**Deliverable:** calculate incremental contribution per eligible customer, forecast gross and net incremental contribution, and forecast peak-order change.
**Acceptance criteria:** use all assigned customers as denominator; state the scale and transport assumptions; do not claim statistical significance.
**Time:** 15 minutes.

### Exercise 4 — Apply feasible-set rules

A proposed model returns USD 12.00 for a delivery fee.
The approved range is USD 2.00–USD 8.00, and the maximum increase from the prior fee of USD 5.00 is USD 1.00 per day.
The customer's valid existing quote is USD 5.00.
The model used an unapproved sensitive proxy.

**Deliverable:** state the maximum new fee allowed by the numerical limits, what happens to the existing quote, and whether the model output can be deployed.
**Acceptance criteria:** calculate the intersection of constraints, preserve the quote, and identify the feature violation separately from the price limits.
**Time:** 15 minutes.

### Exercise 5 — Design a capacity-aware experiment

Two pricing policies share a fixed set of delivery routes.
A customer can visit multiple times and may place an order after a previous visit.

**Deliverable:** write a one-page test protocol covering assignment unit, policy contrast, primary estimand, capacity interference, quote persistence, analysis, and stop rules.
**Acceptance criteria:** use assignment before exposure, preserve customer commitments, explain how clusters or separate capacity address spillovers, and state how uncertainty follows the assignment unit.
**Time:** 30 minutes.

### Exercise 6 — Review a personalization proposal

A vendor proposes a higher delivery price for customers predicted to have low price sensitivity.
Its features include device type, postcode, and prior purchase history.
The vendor cannot explain how individual features affect decisions.

**Deliverable:** write an approval, conditional approval, or rejection memo with a data map, impact measures, legal/privacy questions, and customer remedies.
**Acceptance criteria:** distinguish proxy risk from proof of discrimination; examine group access and effective prices; require an accountable owner and an auditable fallback.
**Time:** 25 minutes.

### Exercise 7 — Label business statements

Classify each statement as a forecast, plan, target, or commitment.
A: the team expects 800 fewer peak orders per 14 days under stated assumptions.
B: management approves a four-week rollout with two additional midday drivers.
C: the team aims to reduce peak orders by 10%.
D: a customer is promised a quoted fee for 30 minutes under stated terms.

**Deliverable:** one label and one reason per statement.
**Acceptance criteria:** do not describe an aspiration as an estimate or a forecast as a customer promise.
**Time:** 10 minutes.

### Exercise 8 — Build a launch and rollback packet

Assume a larger test shows a positive estimate with an uncertainty interval above the preapproved break-even amount.
Off-peak capacity remains tight, and some customers report that only peak slots are practical.

**Deliverable:** propose a bounded decision, named roles, monitoring metrics, customer communications, and rollback triggers.
**Acceptance criteria:** address capacity and distributional impact, preserve existing quotes, define evidence and ownership for review, and avoid declaring fairness from the average result.
**Time:** 25 minutes.

## 13. Self-check answers

### Exercise 1

A is time-based dynamic pricing if the fee varies by slot; it is not individualized merely because a customer selects a slot.
B is a personalized offer because account browsing determines a unique coupon; it need not be dynamic if the offer stays fixed.
C is a publicly fenced bulk discount, not necessarily dynamic or individualized; eligibility and applicable business-to-business law still need review.
D is both personalized and dynamic because account activity determines a price that changes hourly.
Actual legal classification depends on the facts and jurisdiction.

### Exercise 2

One estimand is the 14-day average difference in contribution per eligible customer from assignment to a coupon versus assignment to the current offer among eligible customers in a named market.
The existing comparison is confounded because prior purchase probability affects both coupon receipt and later purchasing.
Randomize eligible customers before offer display, persist assignment across visits, and record all assignments and nonpurchases.
Analyze the assignment effect and report uncertainty at the assignment unit.
Check for shared capacity or spillovers that require a cluster design.

### Exercise 3

Control contribution per eligible customer is USD 3,200 / 1,000 = USD 3.20.
Treatment contribution per eligible customer is USD 3,500 / 1,000 = USD 3.50.
The point-estimate increment is USD 0.30 per eligible customer.
For 12,000 comparable eligible customers, the gross forecast is 12,000 × USD 0.30 = USD 3,600 per forecast period.
After USD 1,000 incremental operating cost, the point-estimate net forecast is USD 2,600 per period.
The peak-order difference is (190 − 250) / 1,000 = −0.06 order per eligible customer.
Scaled to 12,000 eligible customers, the forecast change is −720 peak orders per period.
These calculations assume a stable response and sufficient other capacity; the data provided do not establish uncertainty.

### Exercise 4

The approved range is USD 2.00 to USD 8.00.
The change limit permits at most USD 5.00 + USD 1.00 = USD 6.00 today.
The maximum new fee satisfying both numerical constraints is therefore USD 6.00.
The existing valid USD 5.00 quote is honored under its terms.
The model must not be deployed merely by clipping USD 12.00 to USD 6.00 because it used an unapproved proxy.
The owner must resolve the feature and governance issue and use the approved fallback or review path.

### Exercise 5 rubric

A strong protocol defines the eligible population and the existing and proposed policies before assignment.
It chooses route or other cluster assignment when that meaningfully separates shared capacity, or reserves separate capacity and explains remaining interference.
It persists the relevant assignment and quote for repeat visits and specifies how cross-cluster customers are handled.
Its primary outcome is contribution per eligible assignment unit over a defined window, with nonbuyers included.
It plans uncertainty at the level randomized and prespecifies stop rules, quote preservation, and accountable operators.
A weak protocol randomizes page views independently and treats every order as independent without addressing the shared routes.

### Exercise 6 rubric

A strong memo does not treat device or postcode as automatically prohibited or automatically harmless.
It asks why each feature is necessary, what it proxies, whether processing is lawful, and whether a less intrusive policy achieves the goal.
It specifies permitted comparisons of access and all-in price, with denominators and uncertainty.
It requires an explanation, correction channel, owner, vendor audit provisions, and a safe fallback.
It holds approval until the business can understand and govern the decision.
A weak memo accepts a vendor's aggregate uplift as proof that the price is fair and lawful.

### Exercise 7

A is a forecast because it predicts an outcome conditional on the stated assumptions.
B is a plan because resources and a rollout are approved.
C is a target because management sets an intended result.
D is a commitment because it is a promise to a customer with a stated validity condition.
A single sentence can contain more than one category, so separate the promise from any forecast surrounding it.

### Exercise 8 rubric

A strong packet authorizes only the scope that capacity can serve and names a commercial owner and an operations owner.
It records the estimate, interval, accounting boundary, and reason the result is transportable to the proposed scope.
It defines price-display, capacity, complaint, and permissible distributional measures with accountable reviewers.
It sets a hard stop for broken quotes or capacity breaches and describes customer remediation.
It also gives customers understandable pricing information and preserves valid quotes after rollback.
A weak packet calls the policy fair because average contribution increased and leaves the response to a capacity breach unspecified.

## 14. Interview questions

1. **What is the difference between dynamic and personalized pricing?**
   A strong answer separates time or condition changes from customer-specific information and gives an example of each.
   A weak answer treats every changing price as individual discrimination.
2. **Why is purchase conversion among coupon recipients not necessarily the coupon's incremental effect?**
   A strong answer identifies selection and defines a randomized comparison including nonbuyers.
   A weak answer assumes redemption equals incremental purchase.
3. **What should the denominator be for a contribution-based price experiment?**
   A strong answer uses eligible assigned units for an ITT effect and states the outcome horizon.
   A weak answer averages only successful transactions without acknowledging selection.
4. **How can shared capacity invalidate a customer-level test?**
   A strong answer explains interference and proposes separated capacity or an appropriate cluster design.
   A weak answer assumes randomization eliminates all spillovers.
5. **How would you handle a valid quote after the pricing policy is rolled back?**
   A strong answer preserves the existing commitment according to its terms and logs the policy transition.
   A weak answer simply replaces every outstanding quote with the latest tariff.
6. **What is the difference between a price floor and a fairness control?**
   A strong answer explains that a numerical band cannot cure prohibited features or unequal access.
   A weak answer treats a bounded price as proof of fairness.
7. **How would you evaluate a subgroup effect?**
   A strong answer defines the subgroup before analysis, considers sample size and multiplicity, and reports uncertainty.
   A weak answer selects the most favorable subgroup after looking at many cuts.
8. **What makes a price experiment transportable to a new city?**
   A strong answer compares population, capacity, costs, channel, timing, and policy implementation and tests material differences.
   A weak answer assumes that randomization in one city guarantees the same effect everywhere.
9. **What should a vendor provide before a pricing system is approved?**
   A strong answer asks for input categories, decision logic or meaningful explanation, logging, evaluation, controls, and disablement rights.
   A weak answer accepts a single uplift number without a comparison or audit trail.
10. **How do you interpret an uncertainty interval that includes both loss and gain?**
    A strong answer compares plausible effects with the decision threshold and considers further learning.
    A weak answer concludes the effect is exactly zero or definitely positive.
11. **What does an EU automated-personalization disclosure requirement imply?**
    A strong answer identifies the relevant consumer-contract context and national implementation and seeks legal review.
    A weak answer claims it universally bans dynamic prices or automatically authorizes all personalized prices.
12. **Why can a public time-based price still raise fairness concerns?**
    A strong answer considers whether some customers can realistically access the lower-priced option.
    A weak answer assumes identical formal rules guarantee identical opportunity.

## 15. Key terms

- [Dynamic pricing](#4-definitions-and-vocabulary): a price varies over time or conditions.
- [Personalized pricing](#4-definitions-and-vocabulary): an offer varies with customer information.
- [Segmentation](#5-the-mechanics-step-by-step): grouping eligible units for an explicitly stated purpose.
- [Public fence](#5-the-mechanics-step-by-step): a published eligibility condition available to those who meet it.
- [Effective price](#4-definitions-and-vocabulary): the comparable customer amount including relevant fees and discounts.
- [Quote](#6-worked-example): a communicated amount and conditions with a validity period.
- [Price floor](#5-the-mechanics-step-by-step): a lower bound on an approved price.
- [Price ceiling](#5-the-mechanics-step-by-step): an upper bound on an approved price.
- [Capacity](#6-worked-example): the amount of service that can be fulfilled in a period.
- [Contribution](#6-worked-example): revenue less defined variable and incremental costs.
- [Cannibalization](#4-definitions-and-vocabulary): displacement of another offer's sales.
- [Pull-forward](#11-assumptions-and-limits): demand moved from a later period.
- [Eligible population](#5-the-mechanics-step-by-step): the units entitled to assignment before exposure.
- [Assignment unit](#5-the-mechanics-step-by-step): the customer or cluster randomized to a policy.
- [Exposure](#8-data-standards-and-regulatory-context): the offer actually shown.
- [Intention to treat](#5-the-mechanics-step-by-step): the effect of policy assignment for all assigned units.
- [Estimand](#3-the-core-mental-model): a precisely stated causal quantity.
- [Counterfactual](#3-the-core-mental-model): an outcome under an alternative intervention.
- [Confounding](#5-the-mechanics-step-by-step): shared causes of assignment and outcome.
- [Interference](#6-worked-example): an assignment affecting another unit's outcome.
- [Censoring](#10-failure-modes-and-red-flags): incomplete observation of demand or outcomes.
- [Holdout](#5-the-mechanics-step-by-step): a comparison group retained under a specified policy.
- [Overlap](#4-definitions-and-vocabulary): support for comparing alternatives in the relevant population.
- [Heterogeneous effect](#4-definitions-and-vocabulary): an effect that differs across a defined group.
- [Proxy](#10-failure-modes-and-red-flags): a variable associated with another characteristic.
- [Fairness metric](#5-the-mechanics-step-by-step): a specified comparison of outcomes or access across groups.
- [Drift](#5-the-mechanics-step-by-step): change in data, outcomes, or relationships over time.
- [Rollback](#5-the-mechanics-step-by-step): return to an approved safe policy after a trigger.
- [Forecast](#6-worked-example): an assumption-dependent estimate of an outcome.
- [Commitment](#6-worked-example): a promise that must be honored under its terms.

## 16. Sources

1. European Parliament and Council, [Directive 2011/83/EU on consumer rights, consolidated text](https://eur-lex.europa.eu/eli/dir/2011/83/oj).
   See Article 6 as amended for precontractual information; consult the current consolidated text and national implementation.
2. European Parliament and Council, [Directive (EU) 2019/2161](https://eur-lex.europa.eu/eli/dir/2019/2161/oj).
   Amendments include provisions concerning automated personalized prices and price-reduction announcements.
3. European Parliament and Council, [Directive 98/6/EC on consumer protection in the indication of prices](https://eur-lex.europa.eu/eli/dir/1998/6/oj).
   See the amended text and relevant national rules for price-reduction announcements.
4. European Parliament and Council, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   See Articles 5, 6, 9, 13–15, 22, and 35 as relevant to a particular use.
5. U.S. Federal Trade Commission, [Surveillance Pricing Study: Findings from Initial Staff Report](https://www.ftc.gov/reports/surveillance-pricing-study-findings-initial-staff-report), January 2025.
   This is a staff report about an inquiry, not a universal legal ruling on personalized prices.
6. U.S. Federal Trade Commission, [The Antitrust Laws](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/antitrust-laws).
   Overview of U.S. federal antitrust statutes and enforcement context.
7. National Institute of Standards and Technology, [AI Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1), 2023.
   A voluntary risk-management resource, not a pricing law.
8. U.S. Department of Transportation, [Airline Ticketing and Airfare](https://www.transportation.gov/individuals/aviation-consumer-protection/airline-ticketing-and-airfare).
   Official consumer information on airfare and related rules.
9. Uber, [Surge pricing](https://www.uber.com/us/en/drive/driver-app/how-surge-works/).
   Company explanation of how surge relates to demand and driver availability; not an independent impact evaluation.
10. Kohavi, Ron, Roger Longbotham, Dan Sommerfield, and Randal M. Henne, [Controlled experiments on the web: survey and practical guide](https://doi.org/10.1007/s10618-008-0114-1), *Data Mining and Knowledge Discovery*, 2009.
    Practical evidence on web experiments and measurement pitfalls.
11. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin).
    Identification standard for trade items; not a pricing or fairness standard.

## 17. Links

- [Module 16 topic map](./_index.md).
- [Module 16 survey and embedded lab sections](../../../course/16_pricing_promotions_revenue_management.md).
- [Pricing decision stack](./01_pricing-decision-stack.md).
- [Price, value, and willingness to pay](./02_value-wtp.md).
- [Own-price and cross-price elasticity](./03_elasticity-identification.md).
- [Lerner rule](./04_lerner-rule.md).
- [Markdown optimization](./05_markdown-optimization.md).
- [Production labs map](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
