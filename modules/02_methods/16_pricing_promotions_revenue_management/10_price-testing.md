# Price Testing, Reference Prices, and Psychological Pricing

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 04 and 05; Topics 02, 03, and 06; basic probability and contribution accounting  
> Version 2026.1

## 1. Why this topic exists

A price committee must choose the online price for a replenishable household product.
The buyer's spreadsheet predicts that a higher price will improve unit margin.
The demand planner expects some shoppers to leave, buy a substitute, or wait for a promotion.
The finance partner needs the effect on contribution, not just the effect on conversion.
The customer-experience team is concerned that shoppers may see different prices for the same offer.
The legal team must determine whether the offer, reference price, and disclosures are lawful.

Historical transactions alone rarely settle this decision.
The company may have changed price during a holiday, while advertising, stock, and competitors also changed.
A conversion difference at two points in time can reflect all of those changes.
Even a randomized test can answer the wrong question if it randomizes price and an advertised comparison price together.
A price that attracts more clicks may earn less after fulfillment costs and substitution.
A high-confidence estimate of clicks is not a high-confidence estimate of profit.

A price experiment creates a defensible comparison among feasible commercial actions.
Its design defines who is assigned, what each person or market is offered, and the time horizon.
Its analysis keeps the assignment denominator intact and attaches costs to outcomes.
Its decision rule makes uncertainty and customer harm visible before implementation.
When the rule does not favor either action, the answer can be to retain the current price and learn more.

The worked case is fictional and uses U.S. dollars (USD).
It concerns a one-unit offer, a stable variable cost, and a short online exposure window.
It is not a claim that any named company obtained these results.
Real organizations need a separate accounting, legal, and operational review.

## 2. Learning outcomes

After completing this topic you can:

1. Define a causal price estimand, target population, assignment unit, and outcome window.
   Exercise 1 asks for the assignment and estimand.
2. Choose among customer, store, geographic, and switchback assignments using interference and execution constraints.
   Exercise 2 asks you to justify a design.
3. Separate an absolute price change from a reference-price or price-ending treatment.
   Exercise 3 asks you to diagnose a confounded offer.
4. Compute conversion, contribution per eligible visitor, and the contribution lift from arm totals.
   Exercise 4 requires the arithmetic.
5. Calculate and interpret an approximate confidence interval for a contribution difference.
   Exercise 5 tests the interval and the decision.
6. Size a test around a minimum economically relevant effect and explain the role of power.
   Exercise 6 asks for an independent-observation approximation.
7. Review data quality, fairness, disclosures, and legal risks before deployment.
   Exercise 7 asks for a prelaunch audit.
8. Write a decision memo distinguishing a forecast, a plan, a target, and a commitment.
   Exercise 8 asks for the four labels.

## 3. The core mental model

A randomized offer changes what a defined eligible unit is invited to buy.
The unit responds under that offer and within the study window.
You observe the response for the assigned offer, not the response the same unit would have produced under every alternative.
Random assignment makes the groups comparable in expectation before exposure.
It does not make an incorrectly recorded or selectively analyzed outcome causal.

```text
Decision and population
        |
        v
Eligible unit -> randomized assignment -> delivered offer -> customer response
       |                 |                    |                   |
       |                 v                    v                   v
       +----------> assignment log      exposure audit      orders, returns,
                         |                                  substitutes, cost
                         +------------------+---------------------+
                                            v
                              intention-to-treat arm outcomes
                                            |
                                            v
                           uncertainty + guardrails + decision rule
                                            |
                                            v
                              retain / test further / deploy
```

An *offer* is more than a number.
It can include the item, quantity, currency, tax treatment, shipping, eligibility, display position, reference price, and expiration.
A price-only estimand holds the other components constant by design.
A bundle or presentation test can be useful, but its result applies to the entire bundle.
The customer's remembered price can influence choice without appearing in the data.
A displayed comparison price is controlled by the seller and must not be confused with that internal reference.

The causal contrast is between potential outcomes for the same eligible population under two feasible offers.
Only one outcome is observed for each unit, so the other is counterfactual.
Random assignment supplies an unbiased comparison in expectation if assignment is implemented and outcomes are observed as planned.
This is an effect of an offer policy, not a measurement of every person's maximum willingness to pay.
It is local to the prices, market, season, product, and observation horizon studied.

Profit per eligible unit is often the right primary commercial outcome.
For a one-unit order it is the probability of purchase multiplied by contribution per order.
A larger conversion rate can coexist with a smaller contribution because the lower price applies to all purchases in that arm.
For multiple products, costs, and repeat purchases, use observed net contribution across the defined window instead of a single-product shortcut.
A contribution estimate remains uncertain even when the observed point estimate is positive.
A confidence interval communicates a range of values compatible with the data and the stated procedure; it is not a probability distribution for the fixed effect.

## 4. Definitions and vocabulary

| Term | Meaning | Common misuse | Unit or coding |
|---|---|---|---|
| Eligible unit | Entity that meets prespecified entry criteria before assignment | Counting purchasers only | Customer, store, region, or period |
| Assignment | Random allocation to a tested policy | Replacing the assignment with the observed price | Arm identifier and timestamp |
| Exposure | Opportunity to encounter the assigned offer | Treating a page rendering as proof of attentive reading | Event and timestamp |
| Treatment | Complete intervention a unit is assigned | Calling a price plus a badge a pure price treatment | Price, presentation, and conditions |
| Control | Comparator policy in the decision | Assuming it is universally the best policy | Arm identifier |
| Potential outcome | Outcome a unit would have under one policy | Treating both potential outcomes as observed | Outcome units per assigned unit |
| Estimand | Target causal contrast for a population and window | Reporting a metric without its denominator | USD/eligible unit or percentage points |
| Intention to treat (ITT) | Contrast by original assignment, including delivery failures | Dropping people who did not see or accept an offer | Arm-level mean difference |
| Conversion | Share of eligible units with a defined purchase | Counting orders over page views as customers converted | Purchasers/eligible units |
| Contribution | Revenue less specified incremental variable costs | Calling revenue or gross merchandise value profit | USD per period or unit |
| Reference price | Comparison price in memory or explicitly displayed | Treating an advertised former price as verified | USD per comparable unit |
| Internal reference | Customer's remembered or expected benchmark | Assuming it equals the last transaction | Usually latent, USD/unit |
| External reference | Benchmark shown by the seller or another source | Assuming a listed price was actually charged | USD/unit and source |
| Price ending | Final digits of a price, such as .99 | Attributing an effect to digits when the price also changed | Currency digits |
| Anchor | Initial comparison value that may influence judgment | Assuming all anchors shift every customer's valuation | Displayed amount and context |
| Minimum detectable effect (MDE) | Effect a specified design can detect at a stated power | Smallest true effect that could exist | USD/unit or percentage points |
| Minimum economically relevant effect | Smallest effect that changes the business decision after costs | Setting it from the p-value | USD/unit or total USD |
| Statistical power | Probability of meeting the planned rejection rule under a specified alternative | Probability that the alternative is true | Probability at a specified effect |
| Type I error | Probability of rejecting a true null under the specified procedure | Probability that a finding is false | Probability |
| Confidence interval | Procedure-based range for a parameter | Probability that a fixed parameter is in the realized interval | Outcome units at stated confidence |
| Randomization unit | Smallest entity independently assigned | Counting dependent visits as independent | Customer, store, region, or period |
| Interference | One unit's treatment affects another unit's outcome | Assuming randomization alone prevents it | Design-dependent |
| Intracluster correlation | Similarity of outcomes within the same cluster | Ignoring clustering when estimating uncertainty | Correlation |
| Spillover | Effect of assignment outside its intended unit | Counting substitution in another product as independent evidence | Outcome and boundary |
| Stockout censoring | Sales cannot reveal unmet demand when stock is unavailable | Calling capped sales unconstrained demand | Units and availability time |
| Sample-ratio mismatch | Observed assignment totals differ unusually from planned allocation | Automatically excluding an inconvenient arm | Arm counts and expected proportion |
| Sequential monitoring | Inspecting accumulating outcomes during a test | Applying a fixed-horizon significance rule at every look | Analysis schedule and error control |
| Multiplicity | More than one chance to make a statistical claim | Reporting only the smallest p-value | Number and family of claims |
| Practical significance | Decision value of an effect relative to cost and risk | Treating any statistically significant change as worthwhile | USD and operational constraints |
| Guardrail | Prespecified constraint or safety indicator | Using a non-significant guardrail test as proof of no harm | Metric, threshold, and action |
| Interquartile range | Difference between upper and lower sample quartiles | Treating it as a confidence interval | Same unit as outcome |

## 5. The mechanics, step by step

### 5.1 Write the decision before collecting data

The input is a feasible decision, not a preferred statistical test.
Name the current policy, alternative policy, decision owner, and deadline.
State whether the choice is between two prices, two presentations of one price, or two complete promotional offers.
Record what would remain unchanged after either choice.
Record constraints on supply, customer treatment, contracts, and technology.

Define the decision horizon.
A two-week conversion experiment can inform a two-week offer but may miss returns, renewals, and reference-price adaptation.
If repeat buying matters, define a follow-up period long enough to capture it.
If the full horizon is unavailable before the decision, label the missing value and plan for follow-up.
The output is a decision brief that a pricing owner can approve without interpreting a regression.

Example decision: choose between USD 50 and USD 55 for the same one-unit item in the same online channel.
A feasible alternative must actually be deployable for the eligible market.
If contracts require a uniform price, a customer-level test may need a different design or should not proceed.
Do not test a price that cannot be honored or explained to affected customers.

### 5.2 Define the population and eligibility point

Choose the unit whose outcome counts once.
For a customer-randomized test, identify eligible customers before assignment.
Define eligibility using information available before assignment, such as geography and whether the product is deliverable.
Do not define eligibility by reaching checkout if the offer can itself affect checkout.
That selection could differ by arm and destroy comparability.

A returning customer should normally retain the same assigned offer during the experiment.
Create a stable assignment key with appropriately controlled identity resolution.
If a customer uses multiple devices and cannot be reliably linked, document contamination and consider a different design.
Record the first eligible timestamp, assignment version, and all subsequent offer events.
The denominator is all assigned eligible units, including those who never purchase.

Choose a target population separately from the sampled population.
A test on signed-in repeat buyers cannot automatically represent first-time anonymous visitors.
A test on one metropolitan area cannot automatically represent every market.
Transport requires evidence that response drivers and operating conditions are sufficiently similar.
If not, repeat or extend the test in the intended population.

### 5.3 Define the treatment precisely

Specify the item identifier, size, seller, currency, tax basis, shipping terms, stock status, and offer duration.
Specify the exact numeric price and when it becomes effective.
Specify whether coupons, loyalty discounts, and delivery fees stack with it.
If a former price is shown, specify its source, eligibility, dates, and verification record.
Preserve screenshots or rendered offer logs for each arm.

For a clean price comparison, keep the reference-price display, visual hierarchy, message, and placement constant.
If the crossed-out comparison price appears only in the lower-price arm, the result combines price and framing.
The combined effect can still answer a decision about the full offer.
It cannot establish what a price-only change would have done.

If the intervention is a price ending, distinguish digit effects from monetary effects.
USD 9.99 and USD 10.00 differ both in left digit and by USD 0.01.
A single comparison therefore does not isolate a pure left-digit mechanism.
Use multiple thresholds and a prespecified design if you need evidence about the mechanism rather than a practical offer choice.
Avoid claiming a universal psychological effect from a single product test.

### 5.4 Specify the causal estimand

For each eligible unit, let Y(55) be its net contribution over the specified window if assigned the USD 55 offer.
Let Y(50) be the corresponding outcome under USD 50.
The population average effect is E[Y(55) - Y(50)] in USD per eligible unit.
The higher-price arm minus the current-price arm fixes the sign convention.
A positive number favors the higher offer on the selected contribution measure.

Y includes zero for a nonpurchaser and the appropriate net contribution for a purchaser.
It can be negative for returns, refunds, or customer service costs if the accounting definition permits that.
Use realized or consistently accrued costs and a fixed outcome window in both arms.
Prespecify treatment of missing outcomes and late returns.
Specify whether substitutes and repeat transactions are included.
If they are excluded, identify the omitted economics rather than calling the outcome total profit.

Separate the primary estimand from diagnostic metrics.
Conversion helps explain customer response but is not the primary objective when contribution is the decision criterion.
Measure availability, delivered price, returns, complaints, and support burden as operational or safety metrics.
A checkout conversion metric conditional on reaching checkout is diagnostic, not a replacement for the all-eligible ITT effect.

### 5.5 Choose the randomization unit

Customer randomization can be efficient when customers do not share offers, inventory, or accounts.
It also allows contemporaneous comparison under common demand conditions.
It can create fairness, communication, or legal concerns if comparable customers discover different prices.
A consistent offer, clear policy, and legal review are prerequisites, not afterthoughts.

Store randomization makes shelf execution and staff training more coherent.
It also reduces the number of independent units because customers within a store share conditions.
Stratify or pair stores on prespecified characteristics when doing so improves balance and is operationally practical.
Analyze store-level dependence rather than pretending each receipt was independently randomized.
If there are few stores, use methods appropriate to the actual randomization and small cluster count.

Geographic randomization helps contain store and advertising spillovers.
It is useful when media, inventory, or local competition affect whole markets.
Markets can differ substantially in customer mix, so balanced assignment and a credible preperiod improve precision.
Geographic boundaries do not eliminate cross-border shopping or online exposure.
Measure those pathways and specify how they affect interpretation.

A switchback randomizes a policy across time blocks for the same operating units.
It can suit shared capacity when individual randomization would cause interference.
Balance time-of-day and day-of-week exposure across policies.
Allow washout when customers delay purchases or reference prices persist.
If carryover is longer than practical block length, use a different design or explicitly estimate the resulting policy effect.
A block is not an independent observation merely because it has a separate timestamp.

### 5.6 Control interference and operational spillovers

Consider customers who share an account, household, or purchasing agent.
Consider a promotion shown on social media to customers assigned to another arm.
Consider competitors changing price after observing an offer.
Consider shared inventory: an increase in one arm's purchases can cause a stockout for the other arm.
These pathways violate the simplest independent-unit interpretation.

Choose clusters that contain the most important spillovers when feasible.
Record actual boundary crossings and inventory availability.
If shared inventory remains, interpret the estimated effect as a policy effect in that shared environment.
Do not extrapolate to separate or much larger inventory systems without additional evidence.
If the experiment triggers a market-wide competitor response, the response may be part of the total policy effect at that scale.
An individual test may miss it because competitors cannot observe a market-wide price change.

### 5.7 Randomize and log assignment

Use a reproducible randomization service or audited random number generator.
Keep the assignment mechanism independent of predicted demand and customer outcomes.
Record the allocation probability and any stratification or blocking variables.
Freeze the allocation and analysis rules before observing outcomes.
Restrict access to the assignment key and retain a reproducible version of the assignment algorithm.

Check that each eligible unit receives one consistent arm.
Log assignment even if a page fails to render or the customer does not purchase.
Record the actual rendered offer separately from assignment.
Do not replace assignment with receipt of treatment in the primary ITT analysis.
Delivery failures are evidence about whether the offer policy can be implemented, not permission to discard inconvenient units.

Prespecify an expected allocation, such as one half in each arm.
A large sample-ratio mismatch can indicate a broken eligibility filter, missing event logs, assignment bug, or differential attrition.
Investigate at the assignment stage, not only among converters.
A balanced final purchase count does not prove assignment was correctly implemented.
A significant mismatch does not by itself identify the cause or show which arm is biased.

### 5.8 Plan sample size around economics

State the smallest incremental contribution that would justify implementation costs, risk, and operational disruption.
Convert a total deployment hurdle into USD per eligible unit over the decision horizon.
For example, USD 40,000 of incremental cost across 200,000 eligible customers implies USD 0.20 per eligible customer before any risk allowance.
A statistical MDE should be aligned with that economic hurdle, not chosen merely because it makes the test short.

Estimate outcome variance using a relevant historical sample or a separate pilot.
Because contribution has zeros, variable quantities, refunds, and potentially long tails, a binary conversion variance may understate the real variance.
For equal independent arms and a two-sided normal approximation, per-arm sample size is approximately 2(z_alpha/2 + z_power)^2 sigma^2 / delta^2.
Here sigma is the assumed standard deviation in USD per eligible unit, and delta is the effect in the same units.
This expression assumes equal arm variances, independent units, no clustering, and a fixed-horizon analysis.
It is a planning approximation rather than a guarantee of power for an actual distribution.

For clusters of average size m and intracluster correlation rho, an approximate design effect is 1 + (m - 1)rho.
Multiply a simple independent-unit requirement by that factor as a rough first pass.
The number of clusters, unequal cluster sizes, and assignment strategy also matter.
Use cluster-level simulation or a method matched to the actual design before committing budget.
A single store with many receipts does not replace many independently assigned stores.

Plan for observed eligible units rather than expected raw traffic.
Account for exclusions defined before assignment, missing data, seasonality, and operational feasibility.
Precommit the maximum test duration and the eligible stopping rule.
If the necessary sample is infeasible, reduce the scope of the claim, choose a different design, or accept greater uncertainty explicitly.
Do not report an underpowered no-difference result as proof of equivalence.

### 5.9 Pre-register inference and decision rules

Record the primary outcome, sign, denominator, outcome window, and statistical procedure before launch.
Specify the null hypothesis and whether the interval is two-sided.
Specify the randomization unit and the variance estimator.
List secondary outcomes and separate exploratory subgroup analyses from confirmatory claims.
If multiple primary claims are required, use a prespecified familywise or false-discovery approach suited to the decision.

A fixed-horizon interval assumes the planned analysis time and procedure are honored.
Repeatedly checking a conventional p-value and stopping at the first favorable result inflates the false-positive rate.
If early stopping is necessary, choose a valid group-sequential or always-valid procedure in advance and use its boundaries.
Operational safety monitoring can require an early pause; document it and do not pretend the final inferential procedure was unchanged.
Do not selectively choose the endpoint, timeframe, segment, or cost definition after seeing the results.

A business rule need not be identical to a hypothesis test.
One conservative rule is to deploy only if the lower confidence bound for incremental contribution exceeds the implementation hurdle and no guardrail is breached.
Another may use expected utility and explicit downside tolerance.
Either rule must be agreed before viewing the experiment and must state what happens when evidence is inconclusive.
The selected rule should reflect the cost of a false deployment and the cost of delaying a beneficial change.

### 5.10 Execute and monitor implementation

Run the two policies concurrently when possible.
Check initial offer rendering and price consistency before material traffic reaches the test.
Monitor assignment integrity, stock availability, errors, complaints, and severe customer harm.
Document pauses, outages, inventory interventions, and changes in price feeds.
Predefine which operational events trigger a pause and who has authority to act.

A guardrail is not a license to keep exposing customers to known harm until a p-value crosses a threshold.
If an offer is inaccurate or cannot be honored, stop or correct it using the incident procedure.
Record affected units and the remedy.
A correction that changes the treatment may require a new experiment or a revised interpretation.
Keep the original randomization record even if a run is paused.

### 5.11 Analyze the assigned population

Reconstruct one row per randomized unit using the prespecified key and eligibility timestamp.
Join order, returns, costs, and offer logs without multiplying rows through many-to-many joins.
A unit with no order has zero order contribution, not a missing outcome.
A unit with an unavailable cost record is missing contribution and needs the prespecified missing-data treatment.
Compare missingness and data latency by arm.

For an independent individual-randomized test, calculate the mean contribution per eligible unit in each arm.
The ITT estimate is treatment mean minus control mean.
A large-sample standard error is the square root of sample variance in treatment divided by treatment count plus sample variance in control divided by control count.
The confidence interval is the estimated difference plus or minus the appropriate critical value times this standard error.
For small samples, skewed outcomes, or clusters, use an inference method compatible with the assignment and outcome distribution.

For cluster assignment, preserve clusters in resampling or use cluster-aware model-based or randomization-based inference.
A randomization test should reproduce the actual assignment mechanism, including blocks and allocation constraints.
When clusters are few, asymptotic cluster-robust intervals may be unreliable.
Report the cluster count and the method, not merely the number of transactions.
If the estimand is customer-weighted rather than store-weighted, choose an estimator that targets customer weighting and explain the weighting.

### 5.12 Translate the result into action

Start with data integrity, not the sign of the estimated lift.
If assignment or outcome capture is compromised, determine whether the result can still support the original estimand.
Report arm counts, outcome rates, contribution means, the difference, and the interval.
Show the economic hurdle and the predefined rule on the same scale.
Explain exclusions and missing data before displaying a winner.

A positive point estimate with an interval that spans harmful and beneficial values is inconclusive under a conservative deployment rule.
It does not prove the policies are equivalent.
A statistically detectable effect can still be too small to cover implementation costs.
A practically useful effect can be uncertain if the experiment is small.
When evidence is inconclusive, retain the current policy or run a better-powered test if that is the approved default.

Translate the per-unit effect into a forecast using a stated future eligible population.
Keep the interval and transport assumptions visible when scaling.
A forecast is an estimate of what could happen under stated assumptions.
A plan is the action and resources approved by the organization.
A target is the desired result, not a causal estimate.
A commitment is an authorized promise or obligation, not an inference from a point estimate.

### 5.13 Distinguish price level from price presentation

A presentation experiment can keep the payable amount unchanged while changing how the amount is communicated.
For example, compare a USD 30 price with USD 30 accompanied by an accurate unit-price explanation.
Both arms must use the same item, required charges, and eligibility conditions.
The resulting contrast concerns the communication policy, not the demand response to a USD price change.
If one display makes a required fee less prominent, the intervention is no longer a benign formatting change.
The customer sees a different representation of the total economic obligation.

In a factorial design, assign price level and display independently where all combinations are feasible.
The four policies might be low price without a comparison, low price with a comparison, high price without a comparison, and high price with a comparison.
Each comparison price must be truthful and legally appropriate in the cell in which it appears.
The display effect at the low price is the difference between the two low-price cells.
The display effect at the high price is the difference between the two high-price cells.
Their difference is an interaction on the chosen outcome scale.
An interaction can matter because the same reference value implies different savings at the two selling prices.
More cells divide traffic and may require a materially larger experiment.
Predefine which contrast supports the decision and which contrasts are exploratory.

### 5.14 Interpret the displayed discount

A displayed saving has a numerator, denominator, product basis, and time basis.
For USD 40 compared with a substantiated USD 50 reference, the absolute difference is USD 10.
The arithmetic percentage reduction from the reference is USD 10 / USD 50 = 20%.
It is not a 25% reduction, even though USD 10 is 25% of the new price.
If the reference concerns a different package size, normalize to a comparable unit before making an analytic comparison.
Normalization alone does not establish that a public advertising claim is lawful.

The display can affect perceived value even if actual customer savings versus the next-best alternative are different.
A customer might compare a product with a competitor's offer rather than the seller's previous price.
Another customer might remember a past promotion rather than the current regular price.
Those internal comparisons are heterogeneous and often unobserved.
Avoid interpreting a single aggregate effect as the same mental process for every customer.
If you survey remembered prices, record the timing and wording of the question.
Asking for a remembered price before purchase can itself influence the subsequent decision.
Consider a separate randomized measurement arm if the survey may change behavior.
Report missing survey responses and avoid treating respondents as representative without analysis.

### 5.15 Plan a price-ending investigation

Suppose a merchant can choose USD 19.99 or USD 20.00 for a product.
The direct business question is which feasible offer gives better contribution under the actual display.
Both the one-cent price difference and the change in leftmost digits are part of that policy contrast.
The experiment can choose between these offers without proving a psychological mechanism.
For a mechanism question, include multiple price thresholds and multiple products where practicable.
Include comparisons of similar one-cent changes that do not cross a round-number boundary.
For example, USD 19.98 versus USD 19.99 provides a non-boundary comparison at a nearby level.
Those comparisons still require sufficient sample, comparable products, and prespecified analysis.
Do not assume a common digit effect without checking variation by category and price level.

Rounding rules, taxes, and fees can change the amount the customer ultimately pays.
State whether the ending is displayed before tax, after tax, or in a subscription renewal.
A price ending in .99 can also change the perceived category positioning of a product.
That perception may affect customer satisfaction or later purchase beyond immediate conversion.
Measure those outcomes if they are material to the decision and can be observed responsibly.
Avoid using confusing presentation to exploit a known customer misunderstanding.
Experimental evidence about behavior does not authorize deceptive conduct.

### 5.16 Select the appropriate outcome horizon

An outcome window begins at a precisely defined anchor such as first eligible assignment.
An assignment-relative window avoids giving early entrants more opportunity to purchase than later entrants.
Fix a data extraction date after every included unit has completed the intended observation period.
If returns can occur later, either wait for a mature return window or explicitly report the limitation.
Estimate the fraction of returns historically observed by the planned cutoff using separate data.
That fraction is a diagnostic, not a guaranteed correction for a treatment-induced change in return timing.

For subscriptions, initial sign-up contribution may exclude renewals, cancellations, and servicing costs.
The price could increase immediate revenue while encouraging early cancellation.
Define both short-run and longer-run outcomes before the experiment when those dynamics matter.
For replenishable goods, observe whether a discount merely shifts a planned purchase earlier.
An immediate sales increase followed by a later decrease may produce little net demand over the full horizon.
For durable goods, a long purchase cycle may make a short test especially uninformative.
When a longer horizon is infeasible, state which effects are not yet known.
Do not silently add an assumed lifetime value to a measured short-run contribution.
If using a model of later value, label it as a model-based forecast and evaluate sensitivity to its assumptions.

### 5.17 Handle noncompliance without losing randomization

Noncompliance occurs when the delivered offer differs from the assigned offer.
An integration error might show the control price to some customers assigned to treatment.
A customer might also use an allowed coupon that changes the price ultimately paid.
First decide whether the coupon was part of the assigned policy or an implementation deviation.
Prespecify the answer based on the offer rules rather than the realized outcome.

The ITT contrast retains the original assigned groups.
It estimates the effect of assigning the policies as implemented, including the observed delivery failures.
That makes it relevant to a rollout with similar implementation quality.
It may not equal the effect of perfect delivery of the intended offers.
Comparing those who actually paid USD 55 with those who paid USD 50 is generally not randomized.
Actual payment can depend on purchase choice, coupon use, and other customer characteristics.
An instrumental-variable or complier analysis needs additional assumptions and a clearly different estimand.
In particular, assignment must influence receipt and affect outcomes through receipt in the required way.
Those assumptions are not guaranteed by randomization and need substantive defense.
If implementation failure is substantial, fix the system and consider a fresh randomized test.

### 5.18 Investigate heterogeneity responsibly

An aggregate effect can conceal variation across prespecified business-relevant groups.
Potential groups might include new versus returning customers or regions with different fulfillment costs.
The reason to analyze a group should be linked to a feasible decision and stated in advance.
Define membership using variables measured before assignment.
Do not create a subgroup from post-treatment behavior such as purchase or coupon redemption.

Compare treatment effects between groups directly with an interaction contrast.
One subgroup being statistically significant while another is not does not establish a significant difference between them.
Report group sizes, point estimates, intervals, and the number of comparisons considered.
Small groups can have very wide uncertainty even when their observed lift appears large.
Avoid publishing granular results that could reveal individuals.
If a subgroup result motivates a new pricing policy, validate it in a fresh test where practical.

A predictive difference is not automatically permission to charge different prices.
Assess fairness, the business rationale, customer expectations, and applicable law.
Excluding protected characteristics does not remove correlated proxies.
Consider impacts on vulnerable customers and access to essential goods.
Provide a review and escalation route for proposed differentiated offers.
Document the policy owner and how customer complaints will be resolved.

### 5.19 Quantify decision uncertainty

The point estimate is the observed best estimate under the chosen estimator.
Sampling uncertainty describes variability from the random assignment or sampling process under the method used.
Measurement uncertainty arises from incomplete costs, delayed refunds, or imperfect identity resolution.
Transport uncertainty arises when the future market differs from the test market.
These are distinct sources of uncertainty and need not be covered by one confidence interval.

Translate a per-customer effect to a planning population only after specifying the eligible count.
For example, USD 0.10 per eligible customer across 100,000 customers implies USD 10,000 of estimated contribution.
That multiplication assumes the same average effect in the future population.
It also assumes the price policy does not change the count of eligible customers in an unmodeled way.
If an interval is scaled by a fixed eligible count, the result is conditional on that count.
It does not automatically include future demand volatility or competitor reaction.
Run separate sensitivity scenarios for cost, eligible population, and plausible transport differences.
Label the scenarios as assumptions rather than confidence bounds.
Do not multiply a forecasted customer count by a sample lift and call the result guaranteed profit.

### 5.20 Use a decision matrix

Begin with the predefined economic hurdle and non-negotiable guardrails.
If the lower confidence bound exceeds the hurdle and guardrails pass, the conservative example rule supports deployment.
If the upper confidence bound is below the hurdle, the evidence does not support exceeding that hurdle under the same rule.
If the interval crosses the hurdle, the decision remains uncertain on that criterion.
If the data integrity review fails, resolve the integrity issue before interpreting any apparent winner.
If an offer is misleading or not authorized, do not deploy it even if its measured contribution is positive.

A decision maker may still choose differently when delay is costly or the downside is limited.
Such a decision needs an explicit utility, risk tolerance, and accountable approval.
It should not be presented as though it satisfied the original conservative rule.
The option to learn has value when additional data can change the decision.
Additional learning also has costs: experiment exposure, delay, implementation work, and potential customer harm.
Compare those costs with the plausible value of reducing uncertainty.
Predefine the action for inconclusive evidence rather than negotiating after viewing the result.
Document the chosen policy and whether the evidence supports a full rollout or only a limited pilot.

### 5.21 Account for substitution and baskets

A price change can move purchases between pack sizes, brands, and channels.
For a category decision, measure contribution across the predefined category rather than only the focal item.
Link products using a stable hierarchy that is not changed after seeing results.
Include nonpurchasers and customers who buy only a substitute.
Subtract the same types of variable costs in both arms.
If an item moves to a channel with different fulfillment cost, include that cost where observable.

Basket associations are not automatically caused by the price treatment.
The randomized arm contrast in the full basket outcome can identify the effect of the assigned policy under the design.
A correlation between buying the focal item and buying another item does not identify incremental attachment.
Do not count a complementary product's full revenue as incremental without a control comparison.
Beware double counting when an order line is joined to several product classifications.
Reconcile the total category contribution with the finance ledger.
If only one channel is observable, report the missing channel as a limit.
When a customer can shop elsewhere under a different identifier, assess the likely direction and magnitude of missing substitution.

### 5.22 Document execution and approvals

The experiment record should show who approved the decision, design, offer, accounting basis, and legal review.
It should include the assignment version, start and stop times, expected allocation, and primary metric.
Archive the rendered offer for each arm and any material changes during execution.
Preserve the analysis plan as it existed before outcomes were inspected.
Track deviations with a reason, timestamp, approver, and expected analytical consequence.

At close, reconcile assigned counts with the source system.
Record exclusions by reason and by arm.
Record how missing costs, returns, and identity issues were handled.
Attach the primary estimate and uncertainty interval in the same units as the hurdle.
List guardrail results and any customer incidents.
Name the decision owner and state the action, scope, and review date.
If the result is used for a forecast, retain the target population and transport assumptions.
The archive should let a future reviewer distinguish the original plan from hindsight.

### 5.23 Distinguish statistical and commercial thresholds

A null hypothesis of zero incremental contribution answers whether the average contrast differs from zero under the specified test.
It does not directly answer whether the improvement covers a positive implementation hurdle.
For a hurdle of USD 0.20 per eligible customer, the relevant boundary is USD 0.20 rather than USD 0.
Under the conservative interval rule, the lower endpoint must exceed that boundary.
The same interval can exclude zero while still crossing the hurdle.
In that situation a positive effect may be statistically detectable but insufficiently certain to meet the commercial rule.

Conversely, failure to reject a zero-effect null is not an equivalence conclusion.
An equivalence claim requires a prespecified range of economically negligible effects.
An appropriate equivalence analysis must establish that the interval lies within those bounds under its chosen error convention.
The bounds need a business rationale and must be selected before looking at the results.
An interval that includes zero but also includes serious loss is not evidence that loss is negligible.
An interval that includes zero and a very large gain does not rule out a valuable opportunity.
Report the relevant interval, hurdle, and action together.

### 5.24 Prepare a test handoff

The pricing owner supplies the feasible prices and any contractual limits.
Finance supplies the variable-cost definition and confirms the implementation hurdle.
Engineering supplies the assignment logic, offer rendering, and data lineage.
Operations supplies inventory plans and procedures for outages or stockouts.
Customer support supplies a complaint route and a documented remedy.
Privacy and legal owners review the actual data use and offer in each jurisdiction.
Analytics supplies the estimand, analysis plan, sample-size assumptions, and reporting template.

Before launch, each owner signs off on the artifact within that person's authority.
During the test, the incident owner can pause an unsafe or inaccurate offer.
At close, the analysis owner reports deviations before the commercial owner makes a deployment decision.
Afterward, the deployment owner monitors whether the real policy matches the tested offer.
A change to fees, eligibility, or comparison-price language can make the deployed treatment materially different.
If the policy changes, the original estimate may no longer describe the new action.

## 6. Worked example

### 6.1 Business setting and assumptions

A fictional home-goods retailer sells a replacement filter online in the United States.
The current offer is USD 50 per filter, excluding sales tax and shipping collected separately.
The proposed offer is USD 55 per filter on the same basis.
The variable product and fulfillment cost attributable to each purchased filter is USD 20.
The test uses a one-filter order cap and a fixed fourteen-day outcome window.

The retailer randomly assigns 1,000 eligible customers to each arm.
Each customer is assigned once and remains in that arm.
The scenario assumes no shared-inventory interference, refunds, repeat transactions, or effects on other products.
All eligible customers are observed through the end of the window.
Those assumptions simplify arithmetic and are not claims about a real retailer.

The primary outcome is net contribution from the focal filter per eligible customer over fourteen days.
Each nonpurchaser contributes USD 0 in this simplified case.
Each purchaser contributes the offered price minus USD 20.
The point estimate compares the proposed arm to the current arm by original assignment.
The planned deployment hurdle is USD 0.20 per eligible customer.
The conservative decision rule requires the lower endpoint of a two-sided 95% confidence interval to exceed USD 0.20 and all guardrails to pass.

### 6.2 Arm counts and conversion

| Quantity | Current offer | Proposed offer |
|---|---:|---:|
| Price per filter | USD 50 | USD 55 |
| Eligible assigned customers | 1,000 | 1,000 |
| Customers purchasing one filter | 120 | 105 |
| Customers not purchasing | 880 | 895 |
| Conversion | 12.0% | 10.5% |
| Contribution per purchased filter | USD 30 | USD 35 |

The current conversion is 120 / 1,000 = 0.120, or 12.0%.
The proposed conversion is 105 / 1,000 = 0.105, or 10.5%.
The difference is -0.015, or -1.5 percentage points.
The relative change is -0.015 / 0.120 = -12.5%.
Percentage points and percent are different units and must not be interchanged.
These are realized arm differences, not exact values of the unknown population effect.

### 6.3 Revenue and contribution

Current-arm revenue is 120 × USD 50 = USD 6,000.
Current-arm variable cost is 120 × USD 20 = USD 2,400.
Current-arm contribution is USD 3,600, or USD 3.600 per eligible customer.
Proposed-arm revenue is 105 × USD 55 = USD 5,775.
Proposed-arm variable cost is 105 × USD 20 = USD 2,100.
Proposed-arm contribution is USD 3,675, or USD 3.675 per eligible customer.

The proposed price yields USD 225 less revenue and USD 300 less variable cost in these equal-size arms.
Its observed contribution is USD 75 higher across 1,000 eligible customers.
The ITT point estimate is USD 3.675 - USD 3.600 = USD 0.075 per eligible customer.
This point estimate is smaller than the USD 0.20 hurdle even before uncertainty is considered.
The higher price can produce lower revenue but higher contribution because fewer units also require fewer variable costs.
This is contribution, not a complete statement of net income, fixed cost, or long-run customer value.

### 6.4 Break-even conversion and threshold

To match current contribution of USD 3.60 per eligible customer, the USD 55 arm needs conversion q such that USD 35 × q = USD 3.60.
Thus q = 3.60 / 35 = 0.102857, or approximately 10.29%.
The observed 10.5% exceeds that simple break-even rate by about 0.21 percentage points.
To exceed the USD 0.20 hurdle, the required conversion is (3.60 + 0.20) / 35 = 0.108571, or approximately 10.86%.
The observed 10.5% is below this hurdle, ignoring uncertainty.
The calculation holds current-arm mean and cost constant; both would require review for a real deployment.

### 6.5 Approximate inference

In this simplified binary-purchase setting, a customer outcome is either USD 0 or the arm's fixed contribution per purchase.
The unbiased sample variance is c² n p-hat(1 - p-hat)/(n - 1), where c is the arm's contribution per purchase.
Use n = 1,000 in each arm.
For the current arm the sample variance is approximately USD² 95.14 per customer squared.
For the proposed arm it is approximately USD² 115.23 per customer squared.
The standard error is sqrt(95.14 / 1,000 + 115.23 / 1,000), approximately USD 0.459 per eligible customer.

The normal-approximation 95% interval is USD 0.075 ± 1.96 × USD 0.459.
That is approximately [−USD 0.824, +USD 0.974] per eligible customer.
The interval includes meaningful loss, zero, and a gain above the hurdle.
It is much wider than the point estimate and the USD 0.20 hurdle.
Under the precommitted conservative rule, retain the current price; the experiment does not justify deployment.
This conclusion is not proof that the prices have equal long-run value.

### 6.6 Forecast, plan, target, and commitment

A naive forecast for 200,000 future eligible customers is 200,000 × USD 0.075 = USD 15,000 incremental contribution.
It is a forecast conditional on transport, constant mix, costs, and the observed point estimate.
The corresponding simple scaled interval is approximately [−USD 164,800, +USD 194,800], before future variability and transport uncertainty.
The approved plan in this example is to retain USD 50 and consider a better-powered test.
A management target might be an incremental USD 40,000, but a target does not change the estimate.
No customer or supplier commitment is implied by the forecast or target.

### 6.7 What a broader ledger could change

The calculation excludes substitute products, returns after fourteen days, acquisition costs, and subsequent customer purchases.
If the higher price shifts customers to a more profitable substitute, focal-product contribution understates category contribution.
If it shifts customers to a lower-margin substitute, the focal calculation may overstate the benefit.
If delivery capacity is constrained, saved capacity may have value not included in USD 20 per unit.
The owner must name those effects and collect the corresponding data before claiming total business profit.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the aggregate one-unit illustration, including unbiased sample variances and a normal-approximation interval.
It assumes independent customer assignments and complete observation.
It is not a cluster-aware or sequential-testing implementation.
Run the entire block as a file or in a Python session.

```python
from math import sqrt

# Each arm: eligible customers, purchasers, offered price, variable unit cost.
control = {"n": 1000, "purchases": 120, "price": 50.0, "cost": 20.0}
treatment = {"n": 1000, "purchases": 105, "price": 55.0, "cost": 20.0}


def summarize(arm):
    n = arm["n"]
    purchases = arm["purchases"]
    if n <= 1 or not 0 <= purchases <= n:
        raise ValueError("Need n > 1 and purchases between zero and n")
    conversion = purchases / n
    contribution_per_purchase = arm["price"] - arm["cost"]
    total_contribution = purchases * contribution_per_purchase
    mean = total_contribution / n
    # Sample variance for n observations equal to either zero or c.
    variance = (contribution_per_purchase ** 2 * n * conversion
                * (1 - conversion) / (n - 1))
    return conversion, total_contribution, mean, variance


pc, total_c, mean_c, var_c = summarize(control)
pt, total_t, mean_t, var_t = summarize(treatment)
effect = mean_t - mean_c
se = sqrt(var_t / treatment["n"] + var_c / control["n"])
lower, upper = effect - 1.96 * se, effect + 1.96 * se
hurdle = 0.20
break_even = mean_c / (treatment["price"] - treatment["cost"])
required = (mean_c + hurdle) / (treatment["price"] - treatment["cost"])

print(f"conversion: {pc:.1%} vs {pt:.1%}")
print(f"contribution: ${total_c:.2f} vs ${total_t:.2f}")
print(f"ITT effect/customer: ${effect:.3f}")
print(f"standard error/customer: ${se:.3f}")
print(f"95% interval/customer: [${lower:.3f}, ${upper:.3f}]")
print(f"break-even conversion: {break_even:.2%}")
print(f"conversion for $0.20 hurdle: {required:.2%}")
print("decision:", "deploy" if lower > hurdle else "retain current price")
```

Expected output, allowing for rounding:

```text
conversion: 12.0% vs 10.5%
contribution: $3600.00 vs $3675.00
ITT effect/customer: $0.075
standard error/customer: $0.459
95% interval/customer: [$-0.824, $0.974]
break-even conversion: 10.29%
conversion for $0.20 hurdle: 10.86%
decision: retain current price
```

The code's final decision assumes all non-statistical guardrails pass.
It does not ingest transactional records or validate operational data.
If customers buy multiple units, replace the Bernoulli shortcut with one observed net-contribution value per assigned customer.
If assignment is by store or geography, replace the independent-customer standard error with an appropriate cluster-aware analysis.
If interim stopping is planned, use the prespecified sequential procedure rather than this fixed-horizon interval.
If costs vary across customers, compute each customer's contribution before aggregation.
If a purchase is refunded outside the observation window, the program will not correct for it.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

A defensible experiment starts with a written data contract between pricing, engineering, analytics, finance, and customer operations.
The assignment table needs a stable pseudonymous unit identifier, eligibility timestamp, assignment arm, randomization version, and allocation probability.
The offer table needs item, quantity, currency, displayed price, reference-price text, delivery terms, and render timestamp.
The transaction table needs an order identifier, purchaser linkage, quantity, paid amount, tax, shipping, discount, and cancellation status.
The cost ledger needs a defined effective date, variable cost components, and a finance-approved accounting basis.
The returns table needs the original order link, refund amount, return timestamp, and relevant handling cost.

Store records of offer changes and experiment pauses separately from event data.
Identify how anonymous sessions are reconciled with later customer identifiers without exposing raw personal data to every analyst.
Specify time zones, event latency, deduplication, and the rule for customers whose observation window crosses the data extract date.
Retain a data dictionary and versioned query or calculation code with the decision record.
The logging design is a business control as well as an analytic convenience.

### 8.2 Product and price comparability

Use a stable product identifier and a clear unit of measure so that USD per filter is not compared with USD per package of two filters.
GS1 Global Trade Item Numbers can identify trade items when used and maintained correctly.
GS1 identification standards do not certify a former price, measure demand, or establish legal compliance.
Map bundle and pack changes explicitly rather than treating identifiers as interchangeable.
A reference price for a different package, seller, or fulfillment condition may be misleading and is not an equivalent control.

Tax-inclusive and tax-exclusive prices are not interchangeable.
Specify whether shipping or required fees are included in the displayed and analyzed amount.
A nominal discount can disappear after unavoidable charges.
Keep the information the customer saw distinct from the final amount paid.
Both can be relevant to an offer's legality and to behavioral interpretation.

### 8.3 U.S. reference-price context

The U.S. Federal Trade Commission's Guides Against Deceptive Pricing, 16 CFR Part 233, describe principles for former-price, retail-price, and comparable-value representations.
The guides discuss genuine former prices and misleading comparisons; their application depends on the representation and facts.
They are guides to the FTC's interpretation of deceptive pricing principles, not a blanket authorization for a particular display.
The FTC Act prohibition on unfair or deceptive acts or practices and other federal, state, and local requirements may apply.
Document the underlying offer history, dates, geography, product identity, and substantiation before displaying a comparison.
A randomized experiment does not exempt an offer from advertising law.

### 8.4 European Union context

Article 6a of the EU Price Indication Directive, as amended by Directive (EU) 2019/2161, sets a prior-price rule for announced price reductions within its scope.
The prior price is generally the lowest price applied by the trader during a period not shorter than 30 days before the reduction.
Member-state implementation, scope, exceptions, and particular facts require review.
The rule does not mean that every possible reference-price claim or personalized offer is automatically compliant.
Assess the current national law and official guidance for the relevant market before launch.

The General Data Protection Regulation (GDPR) applies when its territorial and material scope is met.
It imposes obligations on processing personal data, including applicable principles, lawful basis, transparency, and security.
Pseudonymization can reduce exposure but does not automatically make personal data anonymous.
The relevance of automated-decision provisions depends on the facts and legal interpretation; do not assume every price test triggers the same rule.
Consult qualified privacy and consumer-law counsel for the actual deployment.

### 8.5 Responsible test operations

Run accessibility checks on each treatment, including screen-reader labels and price readability.
A visual framing change should not hide essential terms from people using assistive technology.
Use a customer complaint route and a remedy if an offer is misrepresented or not honored.
Document whether customers may receive different prices, the policy rationale, and the approved explanation.
Review whether proposed segmentation is lawful and whether proxies could create unfair disparate outcomes.
A fairness review considers access, burden, transparency, and context; it is not satisfied by excluding a protected attribute alone.

### 8.6 Security and retention

Limit assignment and transaction data to authorized personnel.
Use retention periods and deletion procedures consistent with organizational policy and applicable law.
Publish aggregate results with suitable suppression where small cells might reveal individuals.
Keep raw identifiers out of notebooks and screenshots whenever possible.
A statistically useful dataset can still create unacceptable privacy risk if access is uncontrolled.

## 9. Industry practice and case evidence

### 9.1 A/B testing in digital commerce

Kohavi, Tang, and Xu describe controlled online experiments and operational pitfalls in their book *Trustworthy Online Controlled Experiments*.
The work discusses real experimentation practice, including sample-ratio mismatch and the importance of trustworthy metrics.
It is evidence that experimental operations matter, not evidence that every pricing experiment produces profit.
An organization should adapt the design to its own assignment, legal, and operational constraints.

A common digital workflow is to assign an eligible user to a stable experience and compare outcomes across groups.
That workflow is compatible with a price test only when the offer can be delivered consistently and responsibly.
The observed result is a policy comparison in the tested channel.
It does not establish the causal effect of each visual element when multiple elements change together.

### 9.2 Price endings and field evidence

Anderson and Simester's paper “Effects of $9 Price Endings on Retail Sales: Evidence from Field Experiments” reports field experiments concerning retail price endings.
The documented study is evidence that endings can matter in particular catalog settings.
It does not establish that a .99 ending always beats a round price or that its effect is independent of the absolute price difference.
The appropriate practical conclusion is to test the complete feasible offer and report its context.
Do not import a published percentage lift as a guaranteed outcome for a different category or channel.

### 9.3 Reference prices and consumer behavior

Kalyanaram and Winer review empirical generalizations on internal reference price in *Marketing Science*.
The literature describes how consumers' remembered or expected prices can influence purchase response.
Internal reference prices are not directly observed in most transaction feeds.
A displayed comparison price is a different variable and may also carry a legal claim.
An experiment that alters both selling price and displayed comparison cannot separately attribute the response to either mechanism.

### 9.4 Commercial interpretation

A catalog field experiment, an online visitor test, and a store-cluster test face different exposure and interference conditions.
A company should document where its evidence came from and whether the delivery environment matches the proposed rollout.
Inference: a reference-price cue that affects a one-time purchase may have a different long-run effect if customers learn that the cue is routinely displayed.
That hypothesis requires a suitable repeated-exposure design; it is not a result established by this fictional case.
Inference: shared stock can transmit demand changes across arms even when customers are randomized separately.
A practical test must measure inventory and consider whether a clustered design is necessary.

### 9.5 Evidence hierarchy

A well-executed randomized study supports a causal claim about its assigned policies and studied population.
A peer-reviewed field study offers useful external evidence but does not remove the need for local validation.
A regulator's document states the agency's guidance or legal text, not a company's treatment effect.
A company case study may describe its own experience but can omit negative tests or details of selection.
Vendor marketing claims should not substitute for a reproducible design, denominator, and uncertainty interval.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Price and badge change together | Arm screenshots differ in several elements | Multiple treatments move jointly | Cannot isolate price effect | Hold presentation fixed or label bundle effect |
| Eligibility occurs after exposure | Denominator uses checkout arrivals | Treatment changes selection into sample | Biased contrast | Define eligibility before assignment |
| Purchasers-only analysis | No nonpurchasers in extract | Conditioning on post-treatment outcome | Conversion and profit effects distorted | Retain all assigned eligible units |
| Duplicate assignment | Customer sees two arm values | Device or identity mismatch | Contamination and trust harm | Stable key, audit, and household-aware design |
| Missing assignment events | Allocation ratio differs at entry | Logging or routing failure | Causal comparison may be compromised | Investigate source and repair before inference |
| Stockout differs by arm | Availability falls with orders | Sales are capped by shared supply | Demand response obscured | Monitor supply and design around interference |
| Wrong cost basis | Finance cannot reconcile ledger | Revenue used as profit proxy | Bad commercial decision | Approve incremental cost definition beforehand |
| Returns ignored | Refund rates or lags differ | Benefits counted before reversal | Contribution overstated | Follow orders through defined refund window |
| Frequent peeking | Team reports daily p-values as final | Optional stopping inflates false positives | Spurious winner | Fixed horizon or valid sequential method |
| Many unplanned segments | Only one subgroup is highlighted | Multiple comparisons and selection | Unreliable personalized rollout | Prespecify claims and adjust inference |
| Cluster ignored | Store assignment analyzed as customer assignment | Within-store outcomes correlate | Standard error too small | Cluster-aware inference and adequate clusters |
| Competitor reaction omitted | Rival price changes during market rollout | Small test differs from full deployment | Transport fails | Monitor competition and test at relevant scale |
| Misleading reference price | Former-price history unavailable | Unsubstantiated comparison claim | Customer and legal harm | Verify claim and obtain legal review |
| Price fairness ignored | Complaints cluster by customer group | Similar customers receive unexplained differences | Trust and access harm | Review segmentation, policy, and remedy |
| Bad joining | Orders exceed source totals after join | Many-to-many duplication | Inflated contribution | Validate one-to-one keys and reconciliation |
| Missing outcomes treated as zero | Cost events arrive late in one arm | Missing is confused with no purchase | Biased profit estimate | Track missingness and apply prespecified treatment |
| Novelty mistaken for persistence | Early lift fades in later cohorts | Customer learning or attention changes | Forecast overstates durable impact | Prespecify longer follow-up and replication |
| Underpowered equivalence claim | Wide interval includes zero | Absence of significance mistaken for equality | Harmful or valuable effects dismissed | Report interval and design an equivalence test if needed |
| Price ending overclaim | Only USD 9.99 and USD 10.00 compared | Digit and amount change together | Mechanism is not identified | Use multiple thresholds and factorial structure |
| Late change to primary metric | Metric switches after outcomes appear | Researcher selects favorable result | False confidence | Version and timestamp analysis plan |
| Unclear window | One arm has more time to return | Unequal outcome maturity | Biased net contribution | Fix assignment-relative windows and cutoff |
| Inconsistent total price | Shipping differs across treatments | Effective customer price differs from listed price | Intervention mislabeled | Log required fees and standardize terms |
| Differential delivery | Render failures cluster in one arm | Assigned and received policies diverge | Effect of policy differs from intended offer | Audit receipt and retain ITT |
| Unaffordable rollout | Per-customer gain is tiny | Fixed implementation cost omitted | Positive test loses money at scale | Convert cost to per-unit hurdle |
| Protected proxy targeting | Higher prices concentrate in vulnerable group | Model uses correlated behavioral variables | Unequal burden and possible legal exposure | Review features, outcomes, and policy |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to assess |
|---|---|---|---|
| Assignment is random as specified | Makes potential outcomes comparable in expectation | Routing based on customer value | Audit code, seed, allocation, and balance |
| Eligibility precedes assignment | Prevents post-treatment selection | Defining entry after seeing price | Inspect event sequence and query |
| Outcomes are complete | Makes arm averages interpretable | Delayed refunds or missing costs | Reconcile sources and compare missingness |
| Unit identity is stable | Keeps assignment consistent | Multiple devices or shared accounts | Estimate cross-device and cross-account exposure |
| Interference is negligible or designed for | Supports unit-level policy contrast | Shared inventory or word of mouth | Examine stock and cross-boundary exposure |
| Treatment is well defined | Connects estimate to an actionable policy | Unlogged discounts or inconsistent fees | Audit rendered offers and payment records |
| Costs reflect the decision | Makes contribution economically meaningful | Allocated fixed costs or missing handling cost | Finance review and sensitivity analysis |
| Outcome window is adequate | Captures consequences relevant to decision | Late returns or delayed repeat purchase | Analyze cohort maturity and extended follow-up |
| Variance method matches assignment | Provides meaningful uncertainty | Treating receipts as independent stores | Review assignment and variance units |
| Fixed-horizon analysis is honored | Preserves stated error properties | Stopping on a favorable interim result | Compare timestamps with pre-registration |
| Future population is sufficiently similar | Supports transport of the effect | New season, channel, or customer mix | Compare covariates, operations, and replicate |
| Capacity remains comparable | Prevents an unmodeled stock constraint | Rollout exhausts inventory | Model and monitor supply under rollout |
| Offer is lawful and truthful | Makes the action feasible and responsible | Unsupported comparison or prohibited segmentation | Counsel reviews actual market and records |
| Point estimates are not commitments | Prevents forecast from becoming a promise | Management treats sample lift as certain | State interval, uncertainty, and approval status |
| Normal approximation is adequate | Supports the example interval | Very small sample or extreme heavy tails | Inspect distribution and use design-suitable inference |

No single diagnostic establishes every assumption.
Balance in measured pre-treatment variables does not prove absence of interference or complete outcome capture.
A randomized assignment gives a strong starting point, not immunity from post-assignment selection.
If the decision changes, revisit the estimand rather than reusing the same number without qualification.

## 12. Exercises

### Exercise 1 — Specify the causal question

A retailer considers USD 40 versus USD 44 for a reusable bottle in an online store.
A shopper is eligible upon first viewing the product page during the study period.
The primary business concern is fourteen-day net contribution from all bottle orders.
Write the eligible population, assignment unit, treatment, control, outcome, and signed estimand.
Deliverable: six labeled lines and one sentence explaining the denominator.
Acceptance: the denominator includes assigned nonpurchasers and the contrast is expressed in USD per eligible shopper.
Time: 10 minutes.

### Exercise 2 — Choose a design

A grocery chain has 40 stores and a price label is visible to every shopper in a store.
Stores share regional advertising, and several nearby stores compete for the same customers.
Recommend a randomization unit and identify one spillover and one inference consequence.
Deliverable: a design note of 150–200 words.
Acceptance: the note recognizes that 40 stores are not thousands of independent assignments and states how to contain or measure cross-store exposure.
Time: 20 minutes.

### Exercise 3 — Separate price and framing

Arm A shows USD 20 with no comparison price.
Arm B shows USD 18 and a crossed-out USD 25 former price.
The observed conversion difference favors B.
Explain what the comparison identifies and propose a design to distinguish price and reference-price effects.
Deliverable: a two-by-two design table and two interpretation sentences.
Acceptance: the former price is used only if substantiated and each cell changes one factor relative to an appropriate comparator.
Time: 15 minutes.

### Exercise 4 — Calculate contribution

Two independent arms each have 2,000 eligible customers.
At USD 40, 300 customers buy one unit; at USD 44, 270 customers buy one unit.
The incremental variable cost is USD 16 per purchased unit in both arms.
Calculate conversion, contribution per purchase, contribution per eligible customer, and the signed ITT difference.
Deliverable: a four-row comparison table and a one-sentence business interpretation.
Acceptance: all figures carry units and the calculation includes nonpurchasers.
Time: 15 minutes.

### Exercise 5 — Interpret an interval

The estimated contribution effect is USD 0.12 per eligible customer with a 95% confidence interval of [−USD 0.30, +USD 0.54].
The approved hurdle is USD 0.10 per eligible customer, and the rule requires the lower bound to exceed it.
Give the decision and explain whether the test proves no effect.
Deliverable: three sentences.
Acceptance: the answer applies the rule and distinguishes inconclusive evidence from equivalence.
Time: 10 minutes.

### Exercise 6 — Approximate sample size

Assume independent equal-sized arms, outcome standard deviation USD 10 per eligible customer, and an economically relevant effect USD 1 per eligible customer.
Use z values 1.96 for a two-sided 5% test and 0.84 for 80% power.
Apply n ≈ 2(z_alpha/2 + z_power)² sigma² / delta².
Deliverable: per-arm sample size rounded up and two assumptions that could change it.
Acceptance: arithmetic and units are correct, and clustering is identified as a possible reason for more observations.
Time: 15 minutes.

### Exercise 7 — Audit a proposed launch

A seller proposes different prices by browsing history and displays a former price that marketing cannot substantiate.
The system stores raw customer identifiers indefinitely and has no complaint route.
List four stop-or-remediate items and name an accountable owner for each.
Deliverable: a four-row issue, action, and owner table.
Acceptance: include truthful reference prices, privacy, fairness, and customer remedy without claiming a universal legal conclusion.
Time: 20 minutes.

### Exercise 8 — Label the decision packet

A manager says: “We predict USD 10,000 additional contribution; we will retain the existing offer; our goal is USD 40,000; we have promised no discount to a distributor.”
Label each clause as forecast, plan, target, or commitment.
Deliverable: four labels and one sentence on why uncertainty belongs beside the forecast.
Acceptance: each label appears once and the promise is not described as statistical evidence.
Time: 10 minutes.

## 13. Self-check answers

### Answer 1

The population is eligible shoppers at first qualifying product-page view in the stated channel and period.
The unit is the shopper, assigned once before the offer is rendered.
Treatment is the USD 44 offer and control is the USD 40 offer, with all other offer conditions held fixed.
The outcome is fourteen-day net bottle contribution in USD per assigned shopper, including zero for a shopper without an order.
The estimand is E[Y(44) − Y(40)] for that eligible population.
Restricting the denominator to purchasers would condition on a response affected by the offer.

### Answer 2

A strong response considers geography or store clusters, depending on the advertising footprint and shopping travel patterns.
It prespecifies blocking or pairing based on pre-treatment characteristics when appropriate.
It treats the number of independently randomized clusters as the relevant basis for inference.
It measures cross-store visits and shared advertising, and it explains the impact of remaining interference.
Forty stores may be feasible but are not automatically sufficient; sample size depends on variability and the decision threshold.
A weak response randomizes individual receipts while every customer sees the same shelf label.

### Answer 3

The observed comparison is the combined effect of the USD 18 offer plus the reference-price display versus the USD 20 offer without it.
It does not isolate a price effect or a reference-price effect.
A factorial design compares USD 20 and USD 18, each with and without the same substantiated comparison display, subject to legal and operational approval.
The display must be accurate and appropriate in every cell; otherwise omit that design or choose a lawful alternative.
Estimate the effect of the display at each price and allow for interaction rather than assuming the two effects are additive.

### Answer 4

At USD 40, conversion is 300 / 2,000 = 15.0%.
Contribution per purchase is USD 40 − USD 16 = USD 24.
Total contribution is 300 × USD 24 = USD 7,200, or USD 3.60 per eligible customer.
At USD 44, conversion is 270 / 2,000 = 13.5%.
Contribution per purchase is USD 44 − USD 16 = USD 28.
Total contribution is 270 × USD 28 = USD 7,560, or USD 3.78 per eligible customer.
The signed ITT difference is +USD 0.18 per eligible customer.
The higher price has a positive observed contribution difference despite lower conversion; uncertainty is not calculated from these totals alone in this exercise.

### Answer 5

Retain the current offer under the stated rule because −USD 0.30 does not exceed the USD 0.10 hurdle.
The interval contains both a meaningful loss and a gain above the hurdle.
It does not prove that the true effect is zero or that the two offers are equivalent.

### Answer 6

The required independent sample per arm is approximately 2 × (1.96 + 0.84)² × 10² / 1² = 1,568 eligible customers.
The standard deviation and target effect are both measured per eligible customer in USD.
The calculation assumes equal variance, independent observations, a fixed-horizon two-sided test, and the stated normal approximation.
Clustered assignment, unequal allocation, missing outcomes, or a smaller effect of interest can increase the requirement.
This is a planning approximation, not a guarantee that a realized test will meet its target power.

### Answer 7

A strong audit stops the unsubstantiated former-price claim and assigns evidence review to legal and pricing.
It assigns data minimization, retention, and lawful processing review to privacy and security owners.
It asks a commercial governance owner to evaluate fairness, proxies, accessibility, and segmentation rationale.
It requires customer operations to provide a complaint and remedy process before launch.
The rubric rewards concrete accountable owners and an actual stop condition, not a generic promise to be ethical.

### Answer 8

“Predict USD 10,000” is a forecast.
“Will retain the existing offer” is a plan.
“Goal is USD 40,000” is a target.
“Promised no discount” is a commitment.
The forecast needs uncertainty and assumptions because the future population and realized response can differ from the experiment.

## 14. Interview questions

### 1. What is the primary estimand in a price test?

A strong answer names the eligible population, assigned offers, outcome window, unit, and signed contrast.
A weak answer gives a conversion rate without a comparator or denominator.

### 2. Why keep nonpurchasers in the analysis?

A strong answer explains intention to treat and post-treatment selection.
A weak answer treats no purchase as a missing record.

### 3. When would you randomize stores instead of customers?

A strong answer discusses visible shelf prices, shared exposure, interference, and cluster-aware inference.
A weak answer assumes thousands of receipts eliminate the need for independent stores.

### 4. How can conversion fall while contribution rises?

A strong answer multiplies conversion by per-purchase contribution and accounts for all eligible customers.
A weak answer treats revenue, units, and contribution as interchangeable.

### 5. What is wrong with selecting a winner from daily ordinary p-values?

A strong answer describes optional stopping and proposes fixed-horizon or preplanned sequential inference.
A weak answer says more looks have no effect on false-positive risk.

### 6. How would you test a crossed-out reference price?

A strong answer verifies the underlying claim, defines a lawful treatment, and separates display from selling price.
A weak answer assumes an invented former price is acceptable because both arms were randomized.

### 7. What does a wide confidence interval containing zero mean?

A strong answer compares its endpoints with the business hurdle and declines to claim equivalence.
A weak answer says the policies have exactly equal profit.

### 8. How do price endings differ from an absolute price effect?

A strong answer notes that USD 9.99 versus USD 10.00 changes both digits and amount and suggests multiple thresholds.
A weak answer attributes the entire difference to a psychological mechanism.

### 9. How would you diagnose a sample-ratio mismatch?

A strong answer checks assignment-stage counts, logging, eligibility, routing, and differential loss.
A weak answer removes observations until the ratio looks balanced.

### 10. What changes when scaling a test to every market?

A strong answer discusses customer mix, seasonality, inventory, competitor response, cost, and uncertainty.
A weak answer multiplies the point estimate by all traffic and calls it a commitment.

### 11. What makes price testing responsible?

A strong answer covers truthful offers, fairness, privacy, accessibility, customer support, and market-specific legal review.
A weak answer assumes statistical significance establishes ethical or legal acceptability.

### 12. How would you handle shared stock across experiment arms?

A strong answer identifies interference, measures availability, and considers cluster assignment or a policy-level interpretation.
A weak answer excludes stockout customers after seeing the outcome without assessing selection.

## 15. Key terms

- [Assignment](#4-definitions-and-vocabulary): randomized policy allocated to an eligible unit.
- [Eligibility](#52-define-the-population-and-eligibility-point): rule applied before assignment.
- [Exposure](#4-definitions-and-vocabulary): opportunity to see an assigned offer.
- [Treatment](#53-define-the-treatment-precisely): the complete policy delivered in an arm.
- [Control](#4-definitions-and-vocabulary): comparator policy.
- [Potential outcome](#54-specify-the-causal-estimand): outcome under one possible policy.
- [Estimand](#54-specify-the-causal-estimand): population contrast to be estimated.
- [Intention to treat](#511-analyze-the-assigned-population): comparison by original assignment.
- [Conversion](#62-arm-counts-and-conversion): purchasers divided by eligible assigned units.
- [Contribution](#63-revenue-and-contribution): revenue less specified variable costs.
- [Reference price](#4-definitions-and-vocabulary): internal or displayed comparison amount.
- [Internal reference](#4-definitions-and-vocabulary): remembered or expected customer benchmark.
- [External reference](#4-definitions-and-vocabulary): comparison amount displayed or otherwise observed.
- [Price ending](#53-define-the-treatment-precisely): final digits in a displayed amount.
- [Anchor](#4-definitions-and-vocabulary): comparison value that may influence judgment.
- [Randomization unit](#55-choose-the-randomization-unit): entity independently assigned to a policy.
- [Interference](#56-control-interference-and-operational-spillovers): one unit's assignment affects another's outcome.
- [Intracluster correlation](#58-plan-sample-size-around-economics): similarity of outcomes within clusters.
- [Switchback](#55-choose-the-randomization-unit): policy assignment across time blocks.
- [Minimum detectable effect](#58-plan-sample-size-around-economics): effect detectable at stated power.
- [Economic hurdle](#58-plan-sample-size-around-economics): minimum gain needed for the decision.
- [Power](#58-plan-sample-size-around-economics): rejection probability under a specified alternative.
- [Confidence interval](#65-approximate-inference): procedure-based range for the causal contrast.
- [Sequential monitoring](#59-pre-register-inference-and-decision-rules): analysis during accumulation.
- [Multiplicity](#59-pre-register-inference-and-decision-rules): multiple chances to make a claim.
- [Sample-ratio mismatch](#57-randomize-and-log-assignment): unexpected assignment allocation imbalance.
- [Guardrail](#510-execute-and-monitor-implementation): safety or operational limit.
- [Transport](#512-translate-the-result-into-action): applying evidence to a different population or context.
- [Forecast](#66-forecast-plan-target-and-commitment): conditional estimate of a future outcome.
- [Plan](#66-forecast-plan-target-and-commitment): approved action and resources.
- [Target](#66-forecast-plan-target-and-commitment): desired outcome.
- [Commitment](#66-forecast-plan-target-and-commitment): authorized promise or obligation.

## 16. Sources

1. Federal Trade Commission. [Guides Against Deceptive Pricing, 16 CFR Part 233](https://www.ecfr.gov/current/title-16/chapter-I/subchapter-B/part-233). U.S. regulatory guidance; consult current text and applicable law.
2. European Parliament and Council. [Directive (EU) 2019/2161](https://eur-lex.europa.eu/eli/dir/2019/2161/oj), including amendments to Directive 98/6/EC on price reductions. Official legal text; national implementation matters.
3. European Parliament and Council. [Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj), General Data Protection Regulation. Official legal text.
4. GS1. [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin). Identifier standard and official documentation.
5. Anderson, Eric T., and Duncan I. Simester. “Effects of $9 Price Endings on Retail Sales: Evidence from Field Experiments.” *Quantitative Marketing and Economics* 1 (2003): 93–110. [DOI: 10.1023/A:1023581927405](https://doi.org/10.1023/A:1023581927405).
6. Kalyanaram, Gurumurthy, and Russell S. Winer. “Empirical Generalizations from Reference Price Research.” *Marketing Science* 14, no. 3, supplement (1995): G161–G169. [DOI: 10.1287/mksc.14.3.G161](https://doi.org/10.1287/mksc.14.3.G161).
7. Kohavi, Ron, Diane Tang, and Ya Xu. *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press, 2020. [Publisher page](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/trustworthy-online-controlled-experiments/5FB3E0EFAF2980661AED5BE66511792B).
8. Gerber, Alan S., and Donald P. Green. *Field Experiments: Design, Analysis, and Interpretation*. W. W. Norton, 2012. Randomization, interference, and experimental inference.
9. NIST/SEMATECH. [e-Handbook of Statistical Methods](https://www.itl.nist.gov/div898/handbook/). Official statistical reference for estimation and experimental design.

Regulatory sources are educational references, not legal advice.
Rules and official interpretations can change; obtain current advice in the relevant jurisdiction.
The worked example is fictional and is not evidence about a named company's commercial results.

## 17. Links

- [Module 16 topic map](_index.md).
- [Module 16 survey and lab context](../../../course/16_pricing_promotions_revenue_management.md).
- [Price, value, and willingness to pay](02_value-wtp.md).
- [Own-price and cross-price elasticity](03_elasticity-identification.md).
- [Promotion optimization and trade spend](06_promotion-trade-spend.md).
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).
