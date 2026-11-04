# Segmentation and Customer Lifetime Value

> Module 17 — Assortment, Category, and Retail Analytics  
> Topic 08 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Topics 01, 03 and 04; basic contribution margin and experimental design  
> Version 2026.5

## 1. Why this topic exists

A category manager must decide whether to reserve scarce shelf space for a low-volume product.
The item has modest unit sales, but some customers buy it frequently and buy other goods on the same trips.
A campaign manager proposes a discount to the same customers.
A supply planner wants a demand forecast before changing the range.
None of those decisions follows automatically from a customer label or from historical revenue.

Segmentation groups customers when the differences between groups help choose an action.
Customer lifetime value (CLV) estimates the present value of future customer-level economic contribution over a stated horizon.
Both are decision aids, not properties that a retailer can observe perfectly.
A profitable-looking segment may simply have had more time to purchase.
A high-CLV customer may have bought anyway without an offer.
A category can have strategic value without each item in it earning a high margin.

The expensive error is to confuse a prediction with the incremental effect of a commercial intervention.
If a customer has an expected future contribution of $100, giving that customer a $20 incentive is not justified unless the incentive creates enough additional contribution, after costs, to cover its own cost.
The $100 is a baseline forecast, not a treatment effect.
Likewise, identifying customers who buy organic food does not prove that removing an organic item causes those customers to leave the retailer.

This unit gives the category, customer, finance, analytics and privacy teams a shared way to frame the decision.
It connects observable customer behavior to range decisions, economics and experimental measurement.
It also shows where identity coverage, missing observations and model assumptions limit the answer.
The goal is not to maximize a model score.
The goal is to choose a feasible, lawful action that improves incremental economic outcomes without unjustified customer harm.

### Decisions in scope

- Whether a segment has sufficiently distinct needs to justify an assortment or service change.
- Whether to retain, replace or test an item in a particular store group.
- Whether an eligible customer should receive an offer, and what budget ceiling is economically defensible.
- How to prioritize a retention experiment when customer-level future value is uncertain.
- How to measure outcomes that include repeat visits, substitutions and whole-basket contribution.
- When evidence is too weak to deploy a differentiated action.

A segment can inform a store-level plan without sending individual offers.
A customer-level decision needs additional identity, eligibility and privacy controls.
Where a category has nutritional, accessibility or community importance, management should specify those constraints separately from predicted commercial value.
The model does not decide what the organization is permitted or willing to trade away.

### What this unit does not promise

A segmentation solution is not a discovery of natural customer types.
It is a representation built for a purpose and a measurement window.
CLV is not a contractual minimum future profit, a valuation of a person, or an exact estimate for each individual.
Past loyalty-card data do not reveal every shopper or every reason for a visit.
A well-calibrated forecast still needs a credible counterfactual before it can justify a targeted intervention.

## 2. Learning outcomes

After working through this unit, you can:

1. **Frame** an assortment or retention decision with an owner, action, unit, horizon and constraint.
   Exercise 1 assesses this outcome.
2. **Construct** reproducible behavioral segments using an as-of date and leakage-safe observation windows.
   Exercises 2 and 3 assess this outcome.
3. **Compute** contribution-based, discounted finite-horizon CLV and distinguish it from revenue and acquisition economics.
   Exercise 4 assesses this outcome.
4. **Compare** an offer's incremental expected value with the cost of reaching and serving customers.
   Exercises 5 and 6 assess this outcome.
5. **Design** a randomized measurement plan that captures substitution and whole-basket effects.
   Exercise 7 assesses this outcome.
6. **Critique** identity coverage, uncertainty, drift and legal constraints before operational use.
   Exercises 3 and 8 assess this outcome.
7. **Recommend** a feasible action and stopping rule rather than present a segment chart as a conclusion.
   Exercise 8 assesses this outcome.

Your final deliverable should enable another person to reproduce the calculations and challenge the assumptions.
It should also state what would change the decision.
A forecast is an estimate of what may happen.
A plan specifies an intended action.
A target is an aspiration or threshold.
A commitment is an authorized obligation.
The four are not interchangeable.

## 3. The core mental model

```text
Commercial question + constraints + decision owner
                      |
                      v
As-of data --> coverage and quality checks --> customer features
                      |                            |
                      |                            v
                      |                    segment rules or model
                      |                            |
                      v                            v
           future contribution forecast <-- segment-level validation
                      |                            |
                      +------------+---------------+
                                   v
                  candidate action and eligibility
                                   |
                                   v
                  randomized comparison or holdout
                                   |
                                   v
       incremental whole-basket contribution - incremental costs
                                   |
                                   v
       decision, guardrails, monitoring and scheduled review
```

The flow is deliberately not a straight line from model to deployment.
You may return from validation to data design if the segments are unstable or missingness is systematic.
You may stop before targeting if the action is not legal, feasible or financially defensible.
You may keep a simple rule if a complicated model does not improve the decision.

### Description, prediction and intervention

Description asks what people in the observed data did.
For example, 28% of identified customers bought the category during the last 90 days.
Prediction asks what is likely to happen under a specified future environment.
For example, an identified customer's expected contribution over the next three months is $42.
Intervention asks what changes because the retailer acts.
For example, what extra contribution will an offer cause compared with no offer?
The third question requires an explicit counterfactual: what would have happened to the same eligible population without that action.
A randomized control group estimates that counterfactual under the experiment's conditions.

A customer can have high predicted CLV and low incremental response.
A customer can also have modest baseline CLV and a high incremental response to a cost-effective service improvement.
Ranking by baseline CLV alone directs spending toward value that may already be secured.
It may also systematically exclude less visible customers from benefits.

### The economic boundary matters

A category margin view includes contribution from that category only.
A retailer view may include other categories, fulfillment, returns and service costs.
A campaign view includes the costs and revenue changes caused by the intervention.
A customer acquisition view may also subtract acquisition costs, depending on whether those costs are still avoidable at the decision date.
The relevant boundary must be declared before comparing options.
Otherwise a category can appear unprofitable while supporting valuable trips, or appear profitable while expensive fulfillment destroys contribution.

### The customer is not the store

Customer segments describe people or households, depending on the identifier.
Store clusters describe locations.
A store can serve several customer segments, and a customer can shop at multiple stores or online.
Map segments to stores using observed trips or demand only after documenting this many-to-many relationship.
Do not treat a store's surrounding demographics as the identity or preferences of every customer.
Anonymous purchases and shoppers without loyalty cards remain part of the commercial population.

### Redraw the model

Start with the decision and its constraints.
Draw separate paths for data quality, segmentation and value forecasting.
Join them at the candidate action, not at an assumption of causality.
Finish with incremental measurement and feedback into the next decision.
If your drawing contains no control group, no costs or no review loop, it omits a necessary decision step.

## 4. Definitions and vocabulary

All currency examples in this unit use U.S. dollars (USD).
Unless stated otherwise, a month is a model period, not a claim that every calendar month contains the same number of days.

| Term | Definition and unit | Common misuse |
|---|---|---|
| Decision unit | Entity assigned an action, such as customer, household or store | Switching units after selecting the treatment |
| Customer identifier | Key linking observed transactions to a person, account or household | Treating an account key as a verified individual |
| Identity coverage | Fraction of a defined transaction or sales denominator linked to a usable identity | Treating linked customers as representative of all shoppers |
| Observation window | Period whose records are available for features | Including transactions after the decision date |
| As-of date | Latest time at which input information could have been known | Using an updated customer profile that includes future events |
| Outcome window | Future period in which performance is measured | Measuring outcomes before treatment assignment |
| Cohort | Population sharing an entry time or other defined event | Comparing cohorts with different observation opportunities without adjustment |
| Left truncation | Customers entered the data after their true relationship began | Assuming the first observed purchase was acquisition |
| Right censoring | The full future outcome is not observed by the data cutoff | Labeling a recently inactive customer definitively lost |
| Recency | Time since the most recent observed transaction, usually days | Assuming absence of a linked transaction means absence of all shopping |
| Frequency | Count of eligible purchases within a fixed window | Comparing customers with unequal observation periods as if exposure were equal |
| Monetary value | Revenue or contribution during a stated period, in USD | Calling revenue profit |
| RFM | Recency, frequency and monetary features used for segmentation | Treating the conventional acronym as a validated causal model |
| Segment | Group defined by a reproducible rule or model for a decision | Presenting arbitrary clusters as immutable customer identities |
| Behavioral segment | Group based on observed actions such as visits and category purchases | Inferring unobserved motives from observed behavior |
| Needs-based segment | Group defined by a measured need or job to be done | Assuming a purchase pattern reveals the underlying need |
| Actionability | Ability to apply a feasible differentiated decision to a group | Counting a group as actionable when the action is unavailable |
| Stability | Degree to which membership or characteristics persist across comparable periods | Expecting all legitimate segments to be permanent |
| Calibration | Agreement between predicted and observed average outcomes in a specified population and horizon | Evaluating only ranking accuracy |
| Discrimination | Ability to order observations by outcome or risk | Assuming a model with good ranking has accurate dollar values |
| Customer lifetime value | Expected present value of future customer contribution over a defined horizon | Calling historic revenue or gross sales CLV |
| Contribution | Revenue less specified variable costs, in USD | Treating an undefined gross margin as net profit |
| Gross margin | Sales less cost of goods sold, under the stated accounting policy | Assuming it includes delivery, returns or campaign costs |
| Retention probability | Probability of remaining in a modeled active state for a period, conditional on the model | Treating a repeat-purchase rate as a causal retention effect |
| Survival probability | Probability of remaining in a specified state through a future time | Multiplying survival twice when projected contribution already includes attrition |
| Discount rate | Rate used to convert future cash flows to present value for the same time unit | Applying an annual rate as a monthly rate |
| Discount factor | Present-value multiplier for cash flow at a stated time | Applying end-of-period discounting to an undisclosed beginning-of-period payment |
| Finite horizon | The explicitly modeled future period, such as 12 months | Adding an unsupported perpetual terminal value |
| Terminal value | Assumed value beyond the explicit forecast horizon | Presenting a speculative tail as measured profit |
| Acquisition cost | Cost of acquiring a customer under a defined attribution rule | Subtracting a sunk acquisition cost again when evaluating a current retention offer |
| Incremental value | Difference in expected outcome caused by an action versus its counterfactual | Equating the treated group's value with the treatment effect |
| Cannibalization | Sales or contribution transferred from another product or channel | Counting all promoted-item sales as incremental |
| Halo | Change in other products' outcomes associated with an intervention | Assuming every correlated basket item is causally affected |
| Uplift | Difference in expected outcome under treatment and control for a population or subgroup | Using an untreated response probability as uplift |
| Holdout | Eligible group withheld from an intervention for comparison | Treating an ineligible group as a valid holdout |
| Intention to treat | Comparison by randomized assignment regardless of actual treatment receipt | Comparing only those who redeemed an offer |
| Contamination | Exposure of control units to treatment or treatment-related effects | Ignoring shared households or store-wide signage |
| Guardrail | Predefined constraint on harms or operational degradation | Treating a guardrail as a secondary metric that can always be ignored |
| Suppression | Preventing use of a record or contact when required by a rule | Deleting an opt-out signal and then accidentally re-contacting the customer |
| Protected characteristic | Attribute protected under applicable law, varying by jurisdiction and use | Assuming removing the explicit field eliminates proxy discrimination |
| Proxy | Variable correlated with another attribute or condition | Assuming every correlation proves prohibited discrimination |

Definitions are operational contracts.
The same label can mean different things in different organizations.
Write the numerator, denominator, unit, window and data source next to every reported metric.

## 5. The mechanics, step by step

### Step 1 — Define the decision before the segmentation

**Input:** a commercial problem, decision owner, decision deadline and available actions.

**Procedure:**

1. Name the decision: for example, whether to test a local range change in eligible stores.
2. Specify the unit: account, household, store, item-store-week or another entity.
3. Specify the action and the realistic alternative.
4. Set a forecast horizon and an evaluation horizon separately.
5. Name the economic boundary, such as retailer contribution after fulfillment and treatment costs.
6. Identify operational, legal and customer guardrails before looking for attractive segments.
7. Write the minimum evidence and decision threshold that would change the plan.

**Output:** a one-page decision contract with owner, alternatives and acceptance rules.

**Branch:** if the action cannot differ by customer or store, customer segmentation may be useful for diagnosis but not for directly assigning treatment.
If no one can act on the result, do not build a segmentation merely to populate a dashboard.

A useful contract might state: “At the next range review, choose whether to test a 12-week specialty assortment in 40 comparable stores, subject to shelf capacity, service levels, privacy requirements and a nonnegative estimated incremental retailer contribution.”
The 12-week test is a plan, not a forecast of its outcome.

### Step 2 — Define the population and identity boundary

**Input:** transaction records, identity crosswalk, store and channel definitions, customer eligibility policy and as-of date.

**Procedure:**

1. Decide whether the target population is all trips, loyalty-linked accounts, active customers or another group.
2. Define a customer key and its resolution level.
3. Record when the identity link was valid; avoid retroactive identity joins that were not available at the decision date.
4. Identify merged accounts, shared cards, duplicate identities, employee transactions and returns.
5. Document customers who cannot be linked, contacted or included in the proposed analysis.
6. Calculate linked share by transaction count and separately by sales value.
7. Compare coverage by store, channel, week and other lawful, appropriate dimensions.
8. Escalate large coverage differences before interpreting segments as population demand.

**Output:** an eligibility table and a coverage report with denominators.

**Branch:** if linked sales are 72% of total sales, the unlinked 28% are not necessarily distributed like the linked 72%.
Report both numbers; do not silently scale all segments by 1/0.72.
Weighting requires defensible adjustment variables and overlap, not just a convenient multiplier.

A shared household account can make a category look like one person's preference even when several people contribute.
If assignment occurs at household level, analysis should respect that unit.
If treatment is applied at store level, customer records are observations within a randomized store, not independent randomized units.

### Step 3 — Freeze time and define clean windows

**Input:** transaction timestamps, ingestion timestamps, returns and corrections, feature specifications and forecast cutoff.

**Procedure:**

1. Set an as-of timestamp with time zone and store it in the run metadata.
2. Define the feature window, for example the preceding 180 calendar days.
3. Define the outcome window, for example the next 90 calendar days.
4. Use only records available by the cutoff, including availability delays in the production feed.
5. Specify how returns posted after the transaction are handled without using future knowledge as a feature.
6. Separate new accounts from mature accounts or normalize for observed tenure.
7. Use multiple historical cutoffs to emulate repeated real decisions.
8. Preserve a final period that is not used for model or threshold selection.

**Output:** a leakage-safe feature table, an outcome table and reproducible window definitions.

**Branch:** if a customer was observed for only 20 of the 180 days, a low 180-day count is not comparable to a fully observed customer's count without an exposure adjustment or a separate new-customer treatment.
If historical ingestion timestamps are unavailable, flag the backtest as optimistic rather than pretending the data were available in real time.

Time-based validation matters because random row splits can put the same customer's later behavior into model training and earlier behavior into evaluation.
It can also leak seasonal conditions across training and validation.
A later holdout better represents a future deployment, though no holdout guarantees resilience to a new regime.

### Step 4 — Build interpretable candidate features

**Input:** cleaned, time-frozen transaction and product data.

**Procedure:**

1. Calculate recency in days since the last eligible transaction.
2. Calculate purchase frequency in the observation window, identifying returns and canceled orders.
3. Calculate net sales and, where available, contribution rather than revenue alone.
4. Calculate category participation as customers or trips with a category purchase divided by the appropriate eligible denominator.
5. Calculate category spend share only when the denominator is positive.
6. Describe channel and store use as shares, not a single assumed home store.
7. Include availability or stockout exposure where reliable data exist.
8. Record missingness and eligibility explicitly rather than coding missing as zero.
9. Exclude features unavailable or prohibited at the point of action.
10. Limit features to those needed for the decision; document intended use and retention period.

**Output:** a feature dictionary with formulas, source, timing, units, missingness and owner.

**Branch:** if product cost or fulfillment cost is unavailable, use an explicitly labeled proxy and test whether plausible cost ranges alter the decision.
Do not call sales minus product cost full customer profit when delivery and service costs are material.

A simple recency-frequency-monetary (RFM) table can be enough for an initial hypothesis.
It is not automatically the best representation for a category decision.
A store assortment problem may need unmet need, substitution, physical space and availability information that RFM omits.
Surveyed needs can complement purchase data, but survey participation and sampling must also be assessed.

### Step 5 — Select a segmentation method appropriate to the action

**Input:** decision contract, feature dictionary, coverage report and feasible interventions.

**Procedure:**

1. Begin with a simple rule that a business owner can audit.
2. Make segment boundaries mutually exclusive and collectively exhaustive for the eligible population, including an unknown group when needed.
3. If using quantiles, fit cut points on the training population and freeze them for validation.
4. If using clustering, standardize features only using training data and document the distance metric.
5. Test whether clusters are reproducible across time, initialization and reasonable feature choices.
6. Profile each group with counts, coverage, contribution, uncertainty and potential action.
7. Compare the value of differentiated action with a single policy applied to everyone.
8. Consolidate groups that do not lead to a distinct feasible decision.

**Output:** versioned segment rules and a mapping from each segment to an action or a deliberate no-action decision.

**Branch:** if a group is too small to measure or operationalize, combine it with a defensible adjacent group or collect more evidence.
If its needs are distinctive but an individualized action would be invasive, consider a store-level or universal service change instead.

Statistical separation alone is insufficient.
A segment is decision-useful when it is identifiable with permitted data, sufficiently stable for the decision cadence, reachable through a legitimate channel, large enough for measurement and different in a way that changes a feasible action.
Stability is relative to purpose: a seasonal segment may legitimately change each season.

### Step 6 — Validate segments against real alternatives

**Input:** frozen segment rules, later-period records, operational alternatives and stakeholder review.

**Procedure:**

1. Apply the original rules to a later time period without refitting thresholds on the evaluation outcomes.
2. Compare group sizes and membership transitions.
3. Check whether segment-level outcomes differ by amounts large enough to matter economically.
4. Inspect small cells and suppress reports where re-identification risk is unacceptable.
5. Compare performance with a simpler benchmark such as a single policy or recency rule.
6. Evaluate whether the segment definition produces materially different action choices.
7. Review coverage and outcomes for groups potentially affected by the policy.
8. Document who approved the rule and the next review date.

**Output:** validation report and a decision to adopt, revise or abandon the segmentation.

**Branch:** if segment membership changes mainly because a customer crossed an arbitrary threshold, use an uncertainty band or less frequent reassignment where operationally justified.
If a segment is predictive but cannot change a decision, do not confuse predictive separation with business value.

### Step 7 — Define the CLV cash-flow model

**Input:** horizon, customer population, period-level margin, cost definitions, retention assumptions and discount rate.

A finite-horizon model with contribution conditional on being active is:

```text
CLV_H = sum over t=1..H of [S_t × m_t / (1 + d)^t] - C_0
```

Here `H` is the number of future periods, `S_t` is the probability of being active through period `t`, `m_t` is expected contribution in period `t` conditional on being active, `d` is the discount rate per period, and `C_0` is an explicitly relevant cost at time zero.
Cash flows in this formula occur at the end of each period.
If `m_t` is already unconditional expected contribution per assigned customer, set `S_t` to 1 rather than applying attrition a second time.
If a model allows customers to stop and later return, use a model of future purchase opportunities rather than an absorbing “lost forever” state unless that assumption is justified.

For a constant conditional retention probability `r`, `S_t = r^t` when the customer must survive the first interval before the first end-of-period contribution.
Different timing conventions produce different exponents; state the convention rather than silently mixing them.
Model the entire future period rather than extrapolating a short observed history forever.

**Output:** a versioned CLV definition and inputs with their source and date.

**Branch:** if customer-level contribution is unavailable, report a contribution proxy and a sensitivity range.
If the decision horizon is only three months, a 36-month valuation may add uncertainty without improving the action.
A long horizon can matter for a strategic decision, but the tail must be supported and separately displayed.

### Step 8 — Specify costs and avoid double counting

**Input:** finance definitions, product costs, returns, delivery costs, service costs and campaign budgets.

**Procedure:**

1. Start with net sales after returns and refunds according to the accounting definition.
2. Subtract product cost and variable fulfillment and service costs where relevant.
3. Decide whether promotional discounts are already reflected in net sales.
4. Include incremental campaign costs once, not in both baseline contribution and a separate campaign deduction.
5. Separate avoidable future acquisition or onboarding costs from historical sunk acquisition spending.
6. Decide how shared fixed costs are handled and avoid treating arbitrary allocations as avoidable costs.
7. Include stockout, substitution and capacity effects if material and measurable.
8. Reconcile a sample of customer contribution to finance totals at the same boundary.

**Output:** a cost bridge from observed revenue to decision-relevant contribution.

**Branch:** for evaluating acquisition, include the relevant acquisition cost in the acquisition decision.
For an existing customer's retention offer, historical acquisition cost is generally sunk and does not change when the offer is sent.
A cost that varies because of the current action remains relevant even if accounting classifies it outside gross margin.

### Step 9 — Estimate baseline future value and uncertainty

**Input:** historical cohorts, leakage-safe features, contribution outcomes and horizon.

**Procedure:**

1. Choose a benchmark such as average future contribution by tenure and recency group.
2. Estimate repeat behavior, expected order value and variable cost consistently with the chosen cash-flow formula.
3. Account for new customers whose full horizon is not yet observable; do not label censored outcomes as zero.
4. Validate aggregate predicted versus actual contribution in later cohorts.
5. Examine calibration by segment, tenure, channel and store where sample sizes permit.
6. Examine ranking only after verifying the dollar scale relevant to the decision.
7. Quantify uncertainty using suitable cohort-level or cluster-aware resampling, scenario ranges or a probabilistic model.
8. Examine drift in feature distribution, coverage and actual versus forecast contribution.
9. Specify retraining and retirement conditions.

**Output:** forecast distributions or ranges, calibration tables and a documented limitation statement.

**Branch:** if a model ranks customers but overpredicts total contribution by 30%, recalibrate or use it only for a purpose that does not require accurate dollar estimates.
If an intervention changes the shopping environment, old baseline relationships may no longer apply.
Forecast error and uncertainty about the effect of treatment are different sources of risk.

### Step 10 — Convert baseline value into an incremental decision

**Input:** candidate action, eligible population, baseline outcome, plausible treatment effect, costs and constraints.

For an offer assigned to each eligible customer:

```text
incremental net value per assigned customer
  = expected contribution with offer before offer-specific cost
  - expected contribution without offer
  - incremental offer and contact costs
```

The difference must include effects on all relevant categories, channels, returns and service costs.
If a discount is already reflected in the contribution outcome, do not subtract it again.
If offer cost is incurred only on redemption, use expected redemptions per assigned customer rather than charging every assignee as though they redeemed.
Also account for nonredeemer effects where relevant.

**Output:** expected incremental net value and a sensitivity table, not merely an average CLV ranking.

**Branch:** if the estimated lift is uncertain around zero, run a suitably powered experiment or choose a lower-risk action.
If the proposed action violates a guardrail, its positive expected value alone does not authorize deployment.

### Step 11 — Design the test around the actual assignment unit

**Input:** treatment specification, eligibility, decision metric, sample size assumptions and operational constraints.

**Procedure:**

1. Define the eligible population before assignment.
2. Choose the unit that can receive the action without unacceptable spillover.
3. Randomize at the customer or household level for individual offers when interference is limited.
4. Randomize at store or matched-store level for a physical assortment change.
5. Stratify on a small number of pre-treatment variables if needed for balance.
6. Record assignment before exposure and retain the full assigned population in analysis.
7. Predefine the primary outcome, horizon, denominator, minimum worthwhile effect and stopping rule.
8. Include the costs of implementing and measuring the action.
9. Plan cluster-aware uncertainty for store-level randomization.
10. Track stockouts, contamination, opt-outs, fulfillment performance and adverse customer outcomes.

**Output:** a registered measurement plan and an auditable assignment table.

**Branch:** if treatment changes the entire store shelf, customer-level randomization inside the same store cannot isolate the assortment exposure.
If customers regularly shop both test and control stores, measure cross-shopping and consider geographic clusters or other designs.
A power calculation must use the number of randomized clusters, expected variance and feasible minimum detectable effect, not merely the number of transaction rows.

### Step 12 — Analyze, decide and monitor

**Input:** randomized assignments, complete outcomes, operational logs and the prespecified analysis plan.

**Procedure:**

1. Compare outcomes by original assignment using intention to treat.
2. Express the estimated effect in USD per assigned unit and in total for the eligible population.
3. Calculate a confidence interval appropriate to the assignment design.
4. Inspect guardrails and implementation fidelity without excluding inconvenient observations after the fact.
5. Report both the point estimate and uncertainty relative to the decision threshold.
6. Examine prespecified segment effects cautiously; small subgroups can be noisy.
7. Decide to scale, revise, stop or collect additional evidence using the agreed rule.
8. Monitor performance, customer complaints, stock availability and data coverage after rollout.
9. Review whether the decision still meets legal and commercial requirements when data or market conditions change.

**Output:** a signed decision record, evidence, implementation owner and review schedule.

**Branch:** an interval that includes economically important benefit and harm is inconclusive, not proof of no effect.
A statistically detectable effect can still be too small to cover deployment costs.
A positive average may mask a guardrail breach or an operationally unsustainable workload.

## 6. Worked example

### Decision and setting

A regional grocery retailer is considering a 90-day retention offer for loyalty-linked shoppers who buy a specialty breakfast category.
The customer team can send an eligible customer a digital coupon.
The category team also wants to understand whether that category contributes to broader retailer value.
The offer decision is individual; a later shelf-space decision would need a store-level design.

The decision date is 1 April.
The observation period is the preceding 180 calendar days.
The forecast horizon is the next three months.
All examples are fictional and in USD.
They illustrate calculations, not evidence about a real retailer.

The retailer has 10,000 loyalty-linked active customer accounts eligible for analysis.
Eligibility for the digital coupon is narrower: only customers with the necessary contact permission and a usable channel can be randomized.
For the illustrated segment, assume 1,000 accounts meet all eligibility and permission rules.
A shared account is treated as the assignment unit.
The 1,000-account figure is a modeling assumption, not a forecast of reach.

The team defines a practical behavioral segment:

- At least two purchases in the specialty breakfast category during the previous 180 days.
- At least one retailer transaction during the previous 45 days.
- At least 90 days of observable account history.
- Not currently opted out of the proposed contact channel.
- Exclude known employee test accounts and accounts under a valid suppression rule.

These conditions define eligibility for this particular example, not an assertion that shoppers who fail them lack value.
The segment is descriptive and behavioral.
It does not infer dietary needs, health status or a protected characteristic.
The team will review whether targeting the category nevertheless creates sensitive inferences or other unacceptable effects under applicable policy.

### Establish the economic boundary

Finance supplies estimated monthly contribution for an active account:

| Component | Amount per active account per month | Treatment |
|---|---:|---|
| Net sales after expected returns | $76.00 | Retailer-wide, all included categories |
| Product cost | −$51.50 | Variable cost of goods |
| Fulfillment and service cost | −$8.50 | Expected variable cost |
| Contribution before campaign | $16.00 | $76.00 − $51.50 − $8.50 |

The figures are forecast averages conditional on an account being active in that month.
They are not a commitment that each active account spends $76.00.
They exclude the proposed offer cost, which is evaluated separately.
Fixed corporate overhead is outside this illustrative incremental boundary.
If the change requires extra fixed labor or equipment, the decision analysis must include those avoidable costs.

### Forecast future contribution

Assume the probability of remaining active across each monthly interval is 0.80, conditional on being active at the beginning of the interval.
Assume that the same conditional contribution applies in each active month.
Assume an effective monthly discount rate of 1%.
Cash flows occur at the end of each month.
The active probabilities are therefore 0.80, 0.64 and 0.512 for months one, two and three.
These are model assumptions to validate, not measurements established by the segment rule.

| Month | Active probability | Conditional contribution | Expected contribution before discount | Discount factor | Present value |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.800 | $16.00 | $12.800 | 1/1.01 | $12.673 |
| 2 | 0.640 | $16.00 | $10.240 | 1/1.01² | $10.038 |
| 3 | 0.512 | $16.00 | $8.192 | 1/1.01³ | $7.951 |
| **Total** | — | — | **$31.232** | — | **$30.662** |

The modeled three-month baseline CLV is approximately $30.66 per eligible account before any offer-specific cost.
This is a forecast of discounted contribution over a finite horizon.
It is neither total lifetime revenue nor an estimate of what the coupon causes.
No terminal value is added.
No historical acquisition cost is deducted because the retention decision does not change that sunk cost.

For the 1,000 eligible accounts, multiplying by $30.662 gives an illustrative baseline forecast of approximately $30,662.
The forecast is not a campaign budget or a management target.
Actual outcomes will differ and accounts may be correlated through stores, seasonality or service conditions.

### Compare offer economics

Suppose the proposed coupon is expected to create $2.40 in additional discounted retailer-wide contribution per assigned eligible account before offer-specific costs.
This incremental contribution includes any changes in other categories and channels, but excludes the coupon discount and delivery cost.
That $2.40 is a planning assumption until supported by a causal test.
Suppose 30% of assigned accounts redeem a $4.00 coupon, and the digital contact costs $0.10 per assigned account.

Expected coupon cost per assigned account is `0.30 × $4.00 = $1.20`.
Expected contact cost is `$0.10` per assigned account.
Expected incremental net value is `$2.40 − $1.20 − $0.10 = $1.10` per assigned account.
For 1,000 assigned accounts, the illustrative expected net value is `$1,100` over the specified horizon.
The break-even pre-offer-cost incremental contribution is `$1.30` per assigned account.

The break-even condition does not require the baseline CLV to exceed $1.30.
It requires the effect of the coupon on contribution, before coupon and contact costs, to exceed $1.30.
A high baseline customer who would shop anyway can generate a negative incremental outcome.
A customer who redeems is not necessarily incremental: redemption can subsidize a purchase that would have occurred at full price.

### Sensitivity before deciding

| Incremental contribution before offer costs | Expected offer and contact cost | Incremental net per assigned account | Interpretation |
|---:|---:|---:|---|
| $0.80 | $1.30 | −$0.50 | Do not scale under this scenario |
| $1.30 | $1.30 | $0.00 | Break-even before any omitted implementation costs |
| $2.40 | $1.30 | $1.10 | Positive planning case, subject to validation |
| $3.50 | $1.30 | $2.20 | Favorable scenario, not evidence of effect |

If redemption rises to 50% while the other assumptions stay fixed, cost becomes `0.50 × $4.00 + $0.10 = $2.10` per assigned account.
At $2.40 pre-cost incremental contribution, net value falls to $0.30 per assigned account.
In practice, redemption and contribution response may move together; changing one assumption at a time is a stress test, not a full joint forecast.

### Plan a randomized comparison

Randomly assign the 1,000 eligible accounts to 500 offer and 500 no-offer accounts.
Assignment is at the account level, after eligibility and suppression checks.
The primary outcome is discounted retailer-wide contribution per assigned account over 90 days, net of coupon and contact costs.
Both groups remain in their assigned groups whether or not the message is opened or the coupon is redeemed.
A separate coupon redemption measure is diagnostic, not the primary causal outcome.

Suppose the observed mean discounted contribution is $32.20 per assigned offer account after offer and contact costs and $31.10 per assigned control account.
The intention-to-treat estimate is `$32.20 − $31.10 = $1.10` per assigned account.
For 1,000 comparable eligible accounts, the point estimate translates to $1,100 if the effect transports unchanged.
That extrapolation is an assumption, not a guaranteed outcome.

Suppose a correctly computed 95% confidence interval for the difference is −$0.80 to $3.00 per assigned account.
The interval is illustrative; the necessary individual-level variance data are not supplied here.
It is compatible with both a loss and a material gain.
A positive point estimate alone is insufficient under a prespecified rule requiring the lower confidence bound to exceed $0.00.
The correct decision would be not to scale yet; the team could revise the offer or plan more evidence if the expected value of learning justifies it.
The example does not assert that 500 accounts per group provide adequate power.

### What about the shelf decision?

The coupon experiment estimates an offer effect, not the causal effect of retaining a shelf item.
An item can attract trips, but the category's observed whole-basket contribution cannot be credited entirely to that item.
Some customers would substitute another product, shop another store, or make the same trip anyway.
A shelf decision should compare a feasible item assortment against a control assortment in randomized stores or store clusters.
Measure total retailer contribution, availability, substitutions and service constraints across the relevant stores and channels.
Use the customer segmentation to interpret and design that test, not to manufacture a causal result.

### Decision record

| Field | Example entry |
|---|---|
| Owner | Customer marketing director with category and finance approval |
| Decision | Test the offer; do not commit to rollout on the planning forecast |
| Forecast | $30.66 baseline discounted contribution per eligible account over three months |
| Plan | Randomized offer versus no-offer comparison |
| Target | Positive incremental net contribution with the prespecified evidence threshold |
| Commitment | Only approved test budget and channel capacity |
| Key uncertainty | Treatment effect, redemption, identity coverage and cost completeness |
| Stop or revise | Guardrail breach, inadequate operational fidelity or evidence below the agreed threshold |

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It reproduces the finite-horizon forecast and the offer sensitivity calculation above.
Save it as a local `.py` file and run it with `python`.
There are no external files, random numbers or network calls.

The model assumes constant conditional retention, constant conditional contribution, end-of-month cash flows and a constant effective monthly discount rate.
The offer analysis assumes the stated pre-cost increment is already discounted and measured per assigned account over the full horizon.
It does not estimate causal lift, uncertainty or sample size.
It treats the coupon face value as the incremental coupon cost per redemption; actual accounting should use the appropriate incremental cost definition.

```python
from typing import Iterable


def finite_horizon_clv(
    monthly_retention: float,
    monthly_contribution: float,
    monthly_discount: float,
    months: int,
) -> tuple[list[tuple[int, float, float, float]], float]:
    """Contribution conditional on active status; cash flows at month end."""
    if not 0.0 <= monthly_retention <= 1.0:
        raise ValueError("retention must lie between 0 and 1")
    if monthly_discount <= -1.0:
        raise ValueError("monthly discount rate must be greater than -1")
    if months < 0:
        raise ValueError("months must be nonnegative")

    rows = []
    survival = 1.0
    for month in range(1, months + 1):
        survival *= monthly_retention
        expected_contribution = survival * monthly_contribution
        present_value = expected_contribution / ((1.0 + monthly_discount) ** month)
        rows.append((month, survival, expected_contribution, present_value))
    return rows, sum(row[3] for row in rows)


def offer_net_value(
    incremental_contribution: float,
    redemption_rate: float,
    coupon_cost_per_redemption: float,
    contact_cost: float,
) -> float:
    """All inputs are expected USD per assigned customer, except rate/coupon."""
    if not 0.0 <= redemption_rate <= 1.0:
        raise ValueError("redemption rate must lie between 0 and 1")
    return (
        incremental_contribution
        - redemption_rate * coupon_cost_per_redemption
        - contact_cost
    )


def print_scenarios(increments: Iterable[float]) -> None:
    for increment in increments:
        net = offer_net_value(increment, 0.30, 4.00, 0.10)
        print(f"Increment ${increment:.2f}: net ${net:.2f} per account")


def main() -> None:
    rows, total = finite_horizon_clv(0.80, 16.00, 0.01, 3)
    for month, survival, expected, present in rows:
        print(
            f"Month {month}: active={survival:.3f}, "
            f"expected=${expected:.3f}, PV=${present:.3f}"
        )
    print(f"Three-month baseline CLV: ${total:.2f}")
    print_scenarios([0.80, 1.30, 2.40, 3.50])
    net = offer_net_value(2.40, 0.30, 4.00, 0.10)
    print(f"Planning net for 1000 accounts: ${1000 * net:.2f}")


if __name__ == "__main__":
    main()
```

Expected output, with displayed values rounded to the precision shown:

```text
Month 1: active=0.800, expected=$12.800, PV=$12.673
Month 2: active=0.640, expected=$10.240, PV=$10.038
Month 3: active=0.512, expected=$8.192, PV=$7.951
Three-month baseline CLV: $30.66
Increment $0.80: net $-0.50 per account
Increment $1.30: net $0.00 per account
Increment $2.40: net $1.10 per account
Increment $3.50: net $2.20 per account
Planning net for 1000 accounts: $1100.00
```

The display uses conventional floating-point rounding.
Do not round intermediate amounts before aggregating in a production decision.
For audited financial settlement, use an appropriate decimal representation and the organization's rounding policy.

### Adapting the calculation

Replace the constant `monthly_contribution` with period-specific estimates when seasonality or price changes matter.
Replace constant retention with estimated conditional retention by period when the survival pattern is not geometric.
If a customer's forecast already incorporates lapses and returns, use unconditional expected contributions directly.
If a customer can make purchases after inactivity, define active status and reactivation consistently with the model.
To compare offers, use the same outcome boundary and horizon for every alternative.
To include fixed deployment cost, subtract the total avoidable fixed cost after multiplying per-account net value by the number of assigned accounts.

The code is a transparent calculation, not a validated production model.
Production use requires data access controls, test coverage, versioned assumptions, appropriate uncertainty and review of eligibility and legal requirements.

## 8. Data, standards, and regulatory context

### Minimum data contract

| Dataset | Required fields | Practical check | Limitation |
|---|---|---|---|
| Transactions | Transaction ID, timestamp, store, channel, account key, net sales | Deduplicate transaction ID and reconcile to sales ledger | Anonymous transactions cannot be attributed to an account |
| Transaction lines | Product ID, quantity, net line sales, discounts, returns | Reconcile lines to transaction total | Late corrections can be unavailable at cutoff |
| Product master | Product ID, category, dates of validity | Join using effective dates | Reclassifications change historical comparability |
| Customer identity | Pseudonymous account key, link validity, creation date | Check duplicate and shared identifiers | Account is not necessarily one person |
| Cost data | Product cost, variable fulfillment and service cost | Reconcile sample and totals with finance | Average costs may hide customer-level variation |
| Eligibility | Contact permissions, suppression, channel validity | Resolve status before assignment and contact | Applicable rules vary by channel and jurisdiction |
| Inventory | Store-product availability and stockout periods | Compare sales gaps with availability | Sales alone do not reveal unmet demand |
| Experiment | Assignment, assignment time, treatment version, exposure | Verify assignment before exposure | Exposure is not equivalent to assignment |
| Outcomes | Sales, contribution, returns and costs over fixed horizon | Reconcile denominators to randomized population | Delayed returns may require a maturation window |

A product identifier should be stable enough for the stated measurement period.
GS1's Global Trade Item Number (GTIN) is a standard identifier for trade items, but a GTIN does not by itself determine the business category, customer identity or profitability.
Use the product master and effective dates to interpret item histories.
A retailer may have internal identifiers and mapping rules in addition to GTINs.

### Identity and representativeness

Loyalty enrollment is a selection process.
Some customers prefer not to identify themselves; others forget their cards, share accounts or use different channels.
Coverage can differ by store, age of account, payment type and transaction size.
Measure both linked transaction share and linked sales share, because they answer different questions.
If linked sales are 72% of sales but only 55% of trips, linked trips have a different average value from unlinked trips.
Neither percentage proves that the linked population is representative.

A customer acquired before the data system began is left-truncated.
The first observed transaction is not necessarily the true first purchase.
A recently enrolled customer has not had time to produce a full outcome horizon and is right-censored.
Use mature cohorts for complete fixed-horizon outcomes or an appropriate method that accounts for censoring.
Document observation opportunity when comparing cohorts.

### Data minimization and access

Use pseudonymous keys for routine modeling where possible.
Keep direct contact information in a separately controlled activation system.
Limit access by job role, purpose and retention period.
Pseudonymization reduces exposure but does not automatically make data anonymous.
Record the provenance of permission, suppression and eligibility signals.
A suppression record may need to be retained in a restricted form to avoid contacting someone who opted out; handle retention under the applicable policy and law.

Small segment reports can reveal information about individuals even when names are omitted.
Apply appropriate minimum-cell and access rules; no universal cell-size threshold guarantees anonymity.
When using external enrichment, document source, permission, accuracy and the relevance of each field.
Do not infer medical status, financial hardship or other sensitive circumstances merely because such inference improves a model.
Escalate potentially sensitive features and proxies for legal and ethics review.

### EU and UK context

The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, applies where its territorial and material scope are met.
Articles 5 and 6 address principles including purpose limitation and data minimization, and the need for a lawful basis.
Article 13 addresses information to be provided when personal data are collected from the data subject.
Article 21 includes a right to object to processing for direct marketing, including related profiling.
Article 22 imposes conditions on certain solely automated decisions producing legal or similarly significant effects; not every retail offer is automatically such a decision.
Whether any particular activity falls within these provisions depends on the facts and applicable law.

The UK has a separate legal framework, including the UK GDPR and the Data Protection Act 2018, with direct-marketing rules also affected by the Privacy and Electronic Communications Regulations.
Electronic marketing rules can differ by channel and recipient.
Do not assume a GDPR lawful basis alone answers every electronic-marketing consent question.
Consult the current regulator guidance and qualified counsel for the relevant jurisdiction and communication.

### United States context

U.S. privacy and marketing obligations vary by jurisdiction, entity and activity.
The California Consumer Privacy Act as amended by the California Privacy Rights Act, and implementing regulations, provide rights and obligations for covered businesses and qualifying personal information.
Applicability, exemptions, notices and consumer rights depend on the facts.
The federal CAN-SPAM Act establishes requirements for covered commercial email, including an opt-out mechanism; it is not a general permission to disregard state privacy law or other channel rules.
The Federal Trade Commission's guidance describes commercial email requirements and enforcement context.
For texts or calls, different rules may apply; obtain advice rather than extrapolating from email.

### Standards versus legal obligations

A GS1 identifier is a voluntary interoperability convention unless incorporated into a contract or applicable requirement.
An internal customer segmentation taxonomy is an organizational convention, not a legal classification.
Privacy laws are binding only when their scope and conditions apply.
Regulator guidance explains regulators' positions but does not replace the underlying law or fact-specific legal advice.
This educational unit is not legal advice.
The legal references in the sources were checked as publicly available sources for this edition; requirements and guidance can change after publication.

### Data quality controls at the decision boundary

- Reconcile net sales to the ledger for each store and period.
- Measure return and correction latency before finalizing an outcome window.
- Verify all features precede the decision timestamp.
- Keep an unknown category for unmapped products and quantify it.
- Check that opt-out and suppression status is current at activation time.
- Monitor the share of accounts with missing or implausible costs.
- Check that offer and control groups are drawn from the same eligibility frame.
- Distinguish missing customer linkage from a genuine zero-purchase outcome.
- Record model version, rule version and input snapshots for reproducibility.
- Limit dashboards to aggregated information appropriate to the audience.

A failed control has a defined owner and resolution path.
For example, if the suppression feed is stale, pause activation rather than assume the last available status is sufficient.
If only sales are available, describe the output as sales-based and do not relabel it contribution CLV.

## 9. Industry practice and case evidence

### Tesco Clubcard: data used in retail decisions

Tesco's published history describes the launch of Clubcard in 1995 and its use as a loyalty program.
The case illustrates a large retailer's sustained use of an identified loyalty relationship.
It does not establish that a particular segment, coupon or assortment decision produced a specific profit uplift.
Loyalty participation can help connect trips, but it also creates selection and privacy questions.

**Documented fact:** Tesco identifies Clubcard as its loyalty program and dates its launch to 1995 in its corporate history.
**Inference:** a retailer with linked purchases can construct behavioral segments that are unavailable from anonymous transactions alone.
**Not established:** the causal return of any specific targeting rule, or representativeness of all shoppers in the loyalty file.

A category manager should therefore ask for coverage by store and category before using loyalty-linked demand to allocate space.
A marketing manager should ask whether the data are permitted for the intended action.
The existence of a successful loyalty scheme is not proof that every personalization model adds value.

### Fader and Hardie: customer-base valuation

Fader and Hardie's work on probability models for customer-base analysis provides a formal framework for predicting future purchases from transaction histories.
Their models illustrate that purchase timing and repeat behavior can be modeled explicitly rather than equating lifetime value with historic sales.
The assumptions of any particular model, including its purchase and dropout process, must fit the business setting.
A noncontractual retail setting does not directly reveal a customer's permanent departure.

**Documented contribution:** the cited paper develops a probabilistic model for forecasting transactions in a noncontractual setting.
**Inference:** an analyst may use such a model as one component of a future-contribution forecast after validating its assumptions.
**Not established:** that a purchase-count model by itself estimates contribution margin, offer uplift or the causal effect of an assortment change.

### Retail experimentation

Kohavi, Tang and Xu describe controlled online experiments as a method for making product decisions.
The principle applies to retail marketing, while the unit and interference structure differ for physical stores.
A digital offer may be randomized to customers; a physical shelf change often must be randomized to stores.
Control contamination, supply constraints and delayed returns require a design grounded in the actual operation.

**Documented contribution:** controlled experimentation can compare alternative experiences under randomized assignment.
**Inference:** a retailer can use the same causal logic to estimate an eligible coupon's incremental contribution.
**Not established:** that any example in this unit reflects an actual retailer's experiment or realized outcome.

### Common operating pattern

A mature process often separates three artifacts.
First, an analytics team publishes a descriptive segmentation and its coverage report.
Second, finance and analytics agree on a contribution forecast with a stated horizon and uncertainty.
Third, a commercial team tests a proposed action against a control, with privacy and operational review.
These artifacts may share data but answer different questions.
They should not be collapsed into a single “high-value customer” score.

A retailer may prefer a universal improvement if the predicted incremental benefit of targeting is small relative to operational complexity.
For example, improving availability of a widely used staple could benefit identified and anonymous customers alike.
Alternatively, a narrowly targeted coupon might be appropriate if eligibility, consent, incremental economics and fairness checks all support it.
The best option is an empirical and governance question, not a default preference for individualization.

### Evidence hierarchy in use

A randomized test in the intended population and setting can provide stronger causal evidence for an action than an observational segment profile.
A well-designed observational study can still inform hypotheses and may be necessary when randomization is infeasible.
A vendor's claimed improvement should be read with its population, control design, cost boundary and uncertainty.
A public case describing a program's existence should not be treated as proof of profitability.
A finance reconciliation supports the cost definition, not the causal interpretation of a campaign.

The relevance of outside evidence decreases when customer base, category, channel, pricing, fulfillment or legal context differ.
Replicate the decision logic locally rather than copying another retailer's segment thresholds.
Record what was learned and what remains uncertain in the decision log.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Revenue called CLV | No variable cost bridge | Sales are mistaken for contribution | Overinvestment in costly customers or channels | Reconcile a contribution definition with finance |
| Baseline value called uplift | Business case cites only high predicted CLV | Existing purchases are attributed to the offer | Coupon spend subsidizes behavior that would occur anyway | Estimate incremental value against a valid control |
| Future leakage | Features contain later returns or updated profiles | Training uses information unavailable at decision time | Backtest overstates real-world performance | Freeze data by as-of and availability timestamp |
| Incomplete identity coverage | Linked share differs sharply by store | Enrollment or card use selects the sample | Range decisions ignore anonymous shoppers | Report coverage and triangulate with all-trip data |
| Treating missing as zero | Unlinked records enter as no purchase | Missing identity is interpreted as no demand | Understates customer or category activity | Preserve unknown status and quantify missingness |
| Unequal tenure | New accounts dominate low-frequency segment | Exposure time differs | Recently acquired shoppers are mislabeled low value | Adjust exposure or analyze new cohorts separately |
| Double-counting attrition | Forecast already includes lapse and is multiplied by survival again | Same risk applied twice | CLV is understated | Declare conditional versus unconditional contribution |
| Double-counting discounts | Discount in net sales and deducted again | Same campaign cost appears twice | Rejects potentially worthwhile action | Trace each cost through a documented bridge |
| Ignoring coupon cost | High redemption celebrated without margin impact | Redemption is confused with incremental sales | Profitable-looking campaign loses money | Use net contribution per assigned customer |
| Post-treatment selection | Analysis includes only redeemers | Redemption depends on customer behavior | Treated and control groups are no longer comparable | Analyze by randomized assignment |
| Wrong randomization unit | Customers randomized within one changed shelf | Control customers see treatment | Diluted or biased effect | Randomize stores or appropriate clusters |
| Cross-shopping contamination | Control customers frequently visit test stores | Treatment spills across assignment units | Effect can be hard to interpret | Measure cross-shopping and redesign clusters |
| Poor calibration | Predicted dollars exceed observed totals | Model ranks well but has wrong scale | Budget and value estimates are overstated | Validate and recalibrate on later cohorts |
| Segment instability | Many customers flip labels at small boundary changes | Arbitrary thresholds or noisy features | Inconsistent customer treatment and planning | Test stability and use sensible review cadence |
| Unsupported extrapolation | Most CLV comes from an assumed terminal value | Small retention errors compound far into future | Forecast dominated by speculation | Use finite horizon and show tail sensitivity |
| Proxy discrimination | Offers correlate unexpectedly with sensitive groups | Behavioral or location features proxy protected attributes | Unequal access or legal and reputational harm | Review purpose, features, outcomes and lawful mitigations |
| Stale suppression | Opt-outs arrive after audience export | Activation pipeline uses outdated permissions | Unwanted contact and potential violation | Recheck suppression at send time |
| Category-only metric | Promoted item grows while total basket falls | Substitution or discount transfer is omitted | Category win destroys retailer value | Measure whole-basket incremental contribution |
| Stockout confounding | Test stores have different availability | Sales change reflects supply, not preference | Wrong assortment conclusion | Measure availability and use appropriate randomized design |
| Underpowered store test | Thousands of transactions but few randomized stores | Observations within stores are correlated | False precision and inconclusive decisions | Power and analyze at the assignment-cluster level |
| Multiple subgroup searching | Only one favorable slice is reported | Many comparisons produce chance extremes | Fragile targeted rollout | Prespecify subgroups and report uncertainty |
| Sunk-cost distortion | Old acquisition spend is charged to retention option | Cost unaffected by current action is treated as avoidable | Rejects a beneficial retention action | Separate historical and incremental costs |
| Inappropriate group label | Segment name asserts motivation or character | Behavior is interpreted as inherent trait | Stigma and poor commercial judgment | Use neutral, operational descriptions |
| Forecast presented as commitment | A forecast is entered as a required sales number | Model output is converted into an obligation | Supply and budget decisions lack uncertainty | Label forecast, plan, target and commitment separately |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test or respond |
|---|---|---|---|
| Identity links are sufficiently accurate | Customer histories must refer to the intended unit | Shared cards, duplicates, account merges | Audit links and report account versus household unit |
| Linked behavior informs the decision population | Segments use observed records | Anonymous shoppers differ systematically | Compare coverage and observable basket distributions |
| Feature data existed at cutoff | Backtest must resemble actual deployment | Late corrections or future-enriched profiles | Use availability timestamps and frozen snapshots |
| Contribution definition captures material variable costs | Economic ranking depends on costs | Delivery, returns or service costs omitted | Reconcile with finance and run cost sensitivity |
| Future contribution conditional on activity is estimable | CLV needs period cash flows | Price changes, mix shifts, seasonality | Validate by later cohort and scenario |
| Retention model matches customer process | Survival weights future cash flows | Reactivation or nonconstant hazard | Compare predicted and observed repeat patterns |
| Monthly rate matches monthly periods | Discounting must use consistent units | Annual rate applied monthly without conversion | Document rate conversion and timing |
| Horizon captures relevant effects | Decision should include material future impacts | Delayed repeat purchases or returns beyond horizon | Extend follow-up or bound the omitted tail |
| Treatment and control are comparable | Randomization supports a causal contrast | Eligibility changes after assignment or flawed randomizer | Audit assignment and baseline balance |
| Limited interference | One unit's assignment should not alter another's outcome | Shared households, store signage, cross-shopping | Randomize clusters or measure spillovers |
| Costs are counted once | Net effect must be arithmetically coherent | Discount embedded in net sales and deducted again | Trace the cost bridge line by line |
| Eligible population is reachable lawfully | Action requires a valid activation pathway | Opt-out, channel restrictions or stale permissions | Check eligibility at assignment and at contact |
| Forecast transport is reasonable | Test results may inform rollout | New stores, different seasons or supply conditions | Test in representative settings and monitor rollout |
| Sample supports intended precision | Decision depends on uncertainty | Too few customers or randomized clusters | Calculate power using plausible variance and cluster structure |
| Segments remain decision-relevant | Action rules have a useful lifetime | Market, behavior or policy changes | Monitor drift and review action thresholds |
| Guardrails are measurable | Positive value must not hide unacceptable harm | Missing complaint, accessibility or service data | Define measures and escalation before launch |

No single test proves all assumptions.
Document the plausible direction and economic materiality of violations.
Where the evidence is weak, reduce reliance on the assumption, bound the decision or collect better data.
A confidence interval around a randomized effect does not automatically capture data quality failures, external validity or an omitted cost.

## 12. Exercises

### Exercise 1 — Frame the decision

**Type:** recall and application.  
**Time:** 12 minutes.

A retailer wants to “find valuable customers” before its next range review.
The category team can change a store assortment, but cannot make individual price offers.

**Deliverable:** a decision contract of at most 200 words.

**Acceptance criteria:**

- State an owner, two feasible alternatives and the assignment unit.
- Specify a horizon and a retailer-wide economic outcome.
- Distinguish the forecast from the proposed plan.
- Name one operational guardrail and one privacy constraint.

### Exercise 2 — Define a behavioral segment

**Type:** application.  
**Time:** 15 minutes.

Use a 180-day observation window ending at the decision timestamp.
Account A has 3 category purchases, a most recent retailer visit 20 days ago and 200 days of observable tenure.
Account B has 2 category purchases, a most recent retailer visit 50 days ago and 200 days of observable tenure.
Account C has 2 category purchases, a most recent retailer visit 10 days ago and 30 days of observable tenure.
All three meet contact and suppression requirements.
Apply the segment rule from the worked example.

**Deliverable:** classify each account and explain the reason.

**Acceptance criteria:**

- Apply all three numerical conditions correctly.
- Do not classify short observed tenure as proof of low underlying value.
- Identify the unit as an account, not necessarily a person.

### Exercise 3 — Diagnose coverage and leakage

**Type:** critique.  
**Time:** 18 minutes.

A dataset contains 8,000 linked trips from 10,000 total trips.
The linked trips account for $720,000 of $1,000,000 total net sales.
The analyst uses a customer profile updated two weeks after the decision date as a feature.

**Deliverable:** calculate two coverage measures and identify two decision risks.

**Acceptance criteria:**

- Give the linked-trip share and linked-sales share with denominators.
- Explain why neither share proves representative coverage.
- Identify the time leakage and a practical correction.

### Exercise 4 — Calculate finite-horizon CLV

**Type:** calculation.  
**Time:** 20 minutes.

An account has conditional monthly retention probability 0.75, conditional contribution $20.00 per active month and a 0% monthly discount rate.
The first contribution occurs after the first retention interval.
Use a two-month horizon with no terminal value and no time-zero cost.

**Deliverable:** show active probability and expected contribution for each month and the total CLV.

**Acceptance criteria:**

- Apply survival as 0.75 and 0.75².
- Do not subtract or add costs not given in the problem.
- State the result in USD per account over two months.

### Exercise 5 — Calculate offer economics

**Type:** calculation.  
**Time:** 15 minutes.

An offer creates an estimated $1.80 incremental contribution per assigned account before offer-specific costs.
A $5.00 coupon is redeemed by 20% of assigned accounts.
Contact costs $0.15 per assigned account.
Assume all figures use the same horizon and present-value basis.

**Deliverable:** calculate expected coupon cost, net incremental value and break-even pre-cost increment.

**Acceptance criteria:**

- Use the assigned-account denominator for each component.
- State whether the point estimate is positive or negative.
- Explain why redemption alone does not demonstrate incrementality.

### Exercise 6 — Interpret an experiment

**Type:** application and critique.  
**Time:** 15 minutes.

A randomized offer group averages $24.40 contribution per assigned account after all offer costs.
The control group averages $23.80 on the same basis.
A 95% confidence interval for the difference is −$0.50 to $1.70.
The prior decision rule requires a lower bound above $0.00 to scale.

**Deliverable:** calculate the intention-to-treat effect and recommend an action.

**Acceptance criteria:**

- State the point estimate and the interval in USD per assigned account.
- Apply the stated decision rule rather than inventing a new one.
- Explain that the interval includes both loss and gain.

### Exercise 7 — Design an assortment experiment

**Type:** design.  
**Time:** 25 minutes.

A retailer proposes replacing two shelf items in a specialty category in 40 stores.
Customers can shop at more than one store.
The objective is incremental retailer-wide contribution over 12 weeks.

**Deliverable:** a brief experimental protocol.

**Acceptance criteria:**

- Define eligibility, treatment, control and randomization unit.
- Specify the primary outcome and denominator.
- Address cross-shopping, stock availability and cluster-aware uncertainty.
- Include at least two guardrails and a decision rule.
- State why a customer-level coupon test cannot answer this shelf question.

### Exercise 8 — Responsible deployment review

**Type:** critique and design.  
**Time:** 25 minutes.

A model uses purchase history and postal code to target offers for a category that may reveal sensitive needs.
Loyalty-linked sales coverage is low in several stores.
The business wants to roll out next week without a holdout.

**Deliverable:** write a 250-word recommendation and a short evidence checklist.

**Acceptance criteria:**

- Separate baseline CLV, predicted response and causal incremental value.
- Identify proxy and sensitive-inference risks without asserting a legal violation from the limited facts.
- Address unlinked shoppers, permissions and appropriate access controls.
- Recommend a feasible experiment or alternative action with a review owner.
- State what evidence or guardrail result would stop deployment.

## 13. Self-check answers

### Exercise 1 — Example answer and rubric

A suitable owner is the category director, with finance and store operations approval.
The decision is whether to retain the current assortment or test a specified replacement in comparable stores.
The unit is the store or store cluster because the shelf change is visible to shoppers in that location.
The primary outcome is retailer-wide contribution per randomized store over 12 weeks, including material implementation costs.
A sales forecast is a prediction; the proposed store test is the plan.
A shelf-capacity or availability requirement is an operational guardrail.
Use only permitted, appropriately aggregated customer data and avoid exposing individual purchase histories to store staff.

Full credit requires an actual decision and alternative, not merely a promise to build a dashboard.
Other defensible horizons and guardrails are acceptable if tied to the decision.

### Exercise 2 — Segment classification

Account A qualifies: three category purchases meet the minimum of two, the most recent visit is within 45 days, and observed tenure is at least 90 days.
Account B does not qualify because its most recent visit was 50 days ago, outside the 45-day requirement.
Account C does not qualify because only 30 days of history are observable, below the 90-day requirement.
Neither exclusion proves that the account has low CLV or that the customer is unimportant.
The rule operates on accounts and may combine several household members.

### Exercise 3 — Coverage and leakage

Linked-trip coverage is `8,000 / 10,000 = 80%` of trips.
Linked-sales coverage is `$720,000 / $1,000,000 = 72%` of net sales.
They have different denominators and should not be presented as the same quantity.
The difference indicates that linked and unlinked trips differ in average net sales in this dataset, but does not reveal every source of selection.
A profile updated after the decision date may contain future information and creates leakage.
Use a version available at the decision timestamp, or mark the backtest as limited if historical snapshots do not exist.
Compare coverage by store and period before generalizing to all shoppers.

### Exercise 4 — CLV calculation

Month-one active probability is 0.75, so expected contribution is `0.75 × $20.00 = $15.00`.
Month-two active probability is `0.75² = 0.5625`, so expected contribution is `0.5625 × $20.00 = $11.25`.
With no discounting, total finite-horizon CLV is `$15.00 + $11.25 = $26.25` per account over two months.
The first contribution occurs only after surviving the first interval, which determines the exponent.
An alternative timing convention would require a different formula and must be stated explicitly.

### Exercise 5 — Offer economics

Expected coupon cost is `0.20 × $5.00 = $1.00` per assigned account.
Contact cost is $0.15 per assigned account.
Net incremental value is `$1.80 − $1.00 − $0.15 = $0.65` per assigned account.
The break-even pre-cost increment is `$1.15` per assigned account.
The planning point estimate is positive, subject to uncertainty and any omitted costs.
Redemption does not establish incrementality because some redeemers would have purchased without the coupon.
A randomized comparison estimates the difference attributable to the offer under the test conditions.

### Exercise 6 — Experiment interpretation

The intention-to-treat point estimate is `$24.40 − $23.80 = $0.60` per assigned account.
The stated 95% confidence interval is −$0.50 to $1.70 per assigned account.
Because its lower bound is below $0.00, the prespecified scale rule is not met.
Do not scale on this evidence under that rule.
The interval includes both economically negative and positive effects; it is not proof that the true effect is zero.
Further testing or a revised offer may be considered if justified by the value of information and operational constraints.

### Exercise 7 — Design rubric

A strong protocol randomizes eligible stores or suitable store clusters to the replacement assortment or the current assortment.
It fixes a 12-week observation window and defines retailer-wide contribution per assigned store, net of incremental implementation costs, as the primary outcome.
It retains all assigned stores in the intention-to-treat analysis.
It measures cross-shopping and considers geographic clustering if spillover is substantial.
It records stockouts and fulfillment problems in both arms.
It uses uncertainty appropriate to the number and structure of randomized clusters.
It identifies guardrails such as shelf capacity, availability, customer complaints or service levels.
It specifies a minimum economically worthwhile effect and a rule for scaling, revising or stopping.
A coupon changes an individual offer, not the physical assortment visible to all shoppers in a store.
A protocol missing the assignment unit or control condition is incomplete.

### Exercise 8 — Review rubric

A strong answer does not claim that a high CLV score measures incremental offer effect.
It recognizes that purchase history and postal code may reveal or proxy sensitive attributes, and asks for purpose-specific legal and ethics review.
It reports low loyalty coverage and avoids assuming the unseen shoppers behave like linked shoppers.
It requires current permissions, suppression checks, restricted data access and a documented retention policy.
It proposes a randomized holdout among eligible customers or a less individualized action if targeting cannot be justified.
It names a commercial owner and a privacy reviewer.
It pauses activation if permission status cannot be established or a required guardrail fails.
It avoids asserting that a legal provision applies without the necessary jurisdictional and factual analysis.

## 14. Interview questions

### 1. What makes a customer segment useful?

A strong answer connects a reproducible, measurable group to a distinct feasible decision and compares it with a simpler policy.
It discusses coverage, stability, actionability and constraints.
A weak answer says that any statistically separated cluster is automatically useful.

### 2. How does CLV differ from customer revenue?

A strong answer defines the horizon, future contribution, discounting and relevant costs.
It distinguishes predicted future value from past revenue.
A weak answer sums historical sales and calls the result lifetime profit.

### 3. Why is a high-CLV customer not automatically the best coupon target?

A strong answer distinguishes baseline value from incremental treatment effect and subtracts offer and contact costs.
It mentions that a loyal shopper may buy anyway.
A weak answer assumes the entire future value is caused by the coupon.

### 4. How would you prevent leakage in a customer model?

A strong answer uses as-of snapshots, availability timestamps, time-based validation and fixed outcome windows.
It discusses late-arriving returns and updated profiles.
A weak answer randomly splits rows after building features from the full data history.

### 5. When is survival multiplied into a CLV calculation?

A strong answer explains that it is used when contribution is conditional on active status and matches the timing convention.
It avoids multiplying survival again if contribution is already unconditional.
A weak answer multiplies by retention repeatedly without defining the cash-flow model.

### 6. How would you evaluate a shelf-space change?

A strong answer identifies store or cluster assignment, an appropriate control, whole-basket contribution and spillovers.
It considers availability and cluster-aware uncertainty.
A weak answer compares the changed item's sales before and after without a counterfactual.

### 7. What does a 70% loyalty-linked sales share tell you?

A strong answer states the denominator and notes that the remaining 30% may differ systematically.
It asks for coverage by store, channel and transaction type.
A weak answer assumes the linked customers are automatically a random sample of everyone.

### 8. How do you handle acquisition costs in a retention decision?

A strong answer separates sunk historical acquisition costs from avoidable future costs caused by the current decision.
It includes relevant acquisition costs when evaluating acquisition itself.
A weak answer applies every historical cost to every future intervention without regard to the alternative.

### 9. What should you do when the effect interval includes both material loss and gain?

A strong answer compares the interval with a prespecified economic threshold and considers more evidence or a lower-risk alternative.
It does not equate lack of precision with proof of zero effect.
A weak answer scales based only on the positive point estimate.

### 10. How would you assess responsible personalization?

A strong answer examines lawful basis and channel permissions where applicable, sensitive inferences, proxy effects, coverage, access controls and customer outcomes.
It escalates jurisdiction-specific legal questions.
A weak answer says that deleting names makes all use anonymous and permissible.

## 15. Key terms

Each term links to the definition table or the procedure where it is used.

- [Actionability](#4-definitions-and-vocabulary) — capacity to change a feasible decision.
- [Acquisition cost](#4-definitions-and-vocabulary) — cost of acquiring a customer under a stated attribution rule.
- [As-of date](#4-definitions-and-vocabulary) — information cutoff for a decision.
- [Behavioral segment](#4-definitions-and-vocabulary) — group defined by observable actions.
- [Calibration](#4-definitions-and-vocabulary) — agreement of predicted and observed averages.
- [Cannibalization](#4-definitions-and-vocabulary) — transferred sales or contribution.
- [Censoring](#4-definitions-and-vocabulary) — incomplete observation of a future outcome.
- [Cohort](#4-definitions-and-vocabulary) — population sharing a defined entry event or time.
- [Contribution](#4-definitions-and-vocabulary) — revenue less specified variable costs.
- [Customer lifetime value](#4-definitions-and-vocabulary) — discounted expected future contribution within a defined horizon.
- [Decision unit](#4-definitions-and-vocabulary) — entity assigned an action.
- [Discount factor](#4-definitions-and-vocabulary) — conversion from future value to present value.
- [Discrimination](#4-definitions-and-vocabulary) — ability to rank outcomes.
- [Eligibility](#step-2--define-the-population-and-identity-boundary) — inclusion under operational and policy rules.
- [Finite horizon](#4-definitions-and-vocabulary) — explicit future evaluation period.
- [Frequency](#4-definitions-and-vocabulary) — purchases in a stated window.
- [Guardrail](#4-definitions-and-vocabulary) — pre-established operational or customer constraint.
- [Holdout](#4-definitions-and-vocabulary) — eligible group withheld from treatment for comparison.
- [Identity coverage](#4-definitions-and-vocabulary) — linked share of a specified denominator.
- [Incremental value](#4-definitions-and-vocabulary) — causal difference relative to the alternative.
- [Intention to treat](#4-definitions-and-vocabulary) — comparison by original randomized assignment.
- [Observation window](#4-definitions-and-vocabulary) — period used to calculate features.
- [Outcome window](#4-definitions-and-vocabulary) — period used to measure future outcomes.
- [Proxy](#4-definitions-and-vocabulary) — variable correlated with another attribute.
- [Recency](#4-definitions-and-vocabulary) — time since the latest observed transaction.
- [Retention probability](#4-definitions-and-vocabulary) — probability of remaining in a modeled state.
- [RFM](#4-definitions-and-vocabulary) — recency, frequency and monetary features.
- [Segment](#4-definitions-and-vocabulary) — reproducibly defined group for a decision.
- [Suppression](#4-definitions-and-vocabulary) — preventing prohibited or unwanted use or contact.
- [Survival probability](#4-definitions-and-vocabulary) — probability of remaining in a state through a time.
- [Terminal value](#4-definitions-and-vocabulary) — assumed value beyond the explicit horizon.
- [Uplift](#4-definitions-and-vocabulary) — incremental expected response to treatment.

## 16. Sources

Sources distinguish general methods and documented cases from the fictional worked example.
URLs identify the original publishers or official texts.
Regulatory references describe the law and guidance available for this edition, not a legal opinion.

1. Fader, Peter S., and Bruce G. S. Hardie. “A Note on Deriving the Pareto/NBD Model and Related Expressions.” *Working paper*, 2005. [Author-hosted paper](https://www.brucehardie.com/notes/009/). Provides derivation and assumptions for noncontractual customer-base modeling; it does not estimate a retailer's coupon effect.
2. Fader, Peter S., Bruce G. S. Hardie, and Ka Lok Lee. “Counting Your Customers the Easy Way: An Alternative to the Pareto/NBD Model.” *Marketing Science* 24, no. 2 (2005): 275–284. [DOI: 10.1287/mksc.1040.0098](https://doi.org/10.1287/mksc.1040.0098). Probabilistic repeat-purchase and dropout modeling.
3. Kohavi, Ron, Diane Tang, and Ya Xu. *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press, 2020. [Publisher page](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/trustworthy-online-controlled-experiments/C9D51C4DEB6DF5785DF74F3D76CE8173). General experimentation practice; physical-store assignment requires additional design considerations.
4. European Parliament and Council. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). See Articles 5, 6, 13, 21 and 22 for the provisions discussed.
5. UK Information Commissioner's Office. [Direct marketing guidance](https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/). Guidance on UK marketing and electronic-communications obligations; check current applicability.
6. California Office of the Attorney General. [California Consumer Privacy Act](https://oag.ca.gov/privacy/ccpa). Official overview and links to California privacy requirements.
7. U.S. Federal Trade Commission. [CAN-SPAM Act: A Compliance Guide for Business](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business). Official guidance for covered commercial email.
8. GS1. [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Identifier standard and context for trade-item data.
9. Tesco PLC. [Our history](https://www.tescoplc.com/about/our-history). Company history, including Clubcard; company source for program history, not independent evidence of campaign incrementality.
10. National Institute of Standards and Technology. [NIST Privacy Framework](https://www.nist.gov/privacy-framework). Voluntary privacy risk management resource, not a substitute for applicable law.

## 17. Links

- [Module 17 index](_index.md) — sequence and topic map.
- [Module 17 survey and lab sections](../../../course/17_assortment_category_retail_analytics.md) — broader assortment context and hands-on activities.
- [Assortment decision stack](01_assortment-decision-stack.md) — range, depth and constrained portfolio decisions.
- [Category roles and decision trees](03_category-roles.md) — why a category's role changes the decision.
- [Market basket analysis](04_market-basket.md) — association rules and their noncausal interpretation.
- [Basket affinity and cross-selling](05_basket-affinity.md) — attach rates, counterfactuals and whole-basket outcomes.
- [Space elasticity and planograms](06_space-elasticity.md) — store-level space allocation and measurement.
