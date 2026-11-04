# Markdown Optimization with Salvage

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 05 of 10  
> Estimated study time: 4 hours  
> Prerequisites: 01 pricing decision stack; 03 elasticity identification; basic probability and contribution margin  
> Version 2026.1

## 1. Why this topic exists

A retailer buys inventory before knowing exactly how many customers will want it.
Once a season is underway, the merchant must decide when to reduce the selling price and how deep to cut.
The buyer cannot recover missed demand after the selling window closes, and often cannot sell leftover units at the original price.
A markdown trades contribution per unit against the probability and timing of selling that unit.
The decision is not simply whether a lower price increases unit sales.
It is whether the additional attainable contribution exceeds the contribution surrendered on units that would have sold anyway, after considering the remaining inventory and its best alternative use.

The planner owns a finite stock of winter coats, fashion sizes, dated food, or discontinued components.
The stock may be available in several stores, channels, batches, and condition grades.
Each unit has a remaining useful selling life.
The merchant sees inventory and recent sales, but not every customer's willingness to pay or future demand.
A forecast is therefore a distribution of possible outcomes, not a commitment that the midpoint will happen.

Waiting can preserve margin if enough customers purchase at the current price.
Waiting can also destroy selling opportunities if demand arrives too late or promotional execution requires lead time.
An early deep cut can clear stock but unnecessarily discount customers who would have paid more.
A late deep cut may be too small or may not run long enough to empty shelves.
Both decisions can look successful when judged only by sell-through.

The owner needs an actionable calendar: eligible products, allowable prices, decision dates, inventory snapshots, escalation rules, and a terminal disposition.
Finance needs expected incremental contribution, cash, and risk rather than a gross revenue story.
Operations needs enough lead time to update labels, sites, warehouse availability, and returns handling.
Legal and customer-service teams need a truthful, consistent customer-facing price.

This topic models a bounded decision for stock already owned.
It is not a procurement recommendation, a personalized pricing system, or a claim that a mathematical optimum is automatically deployable.
The same discipline can inform perishable goods, but food safety and product quality can remove units from the sellable pool before the commercial season ends.
The business decision is the best feasible sequence of prices given what is known at each decision time.

## 2. Learning outcomes

After studying this topic you can:

1. **Define** a markdown decision's horizon, state, action, objective, and terminal value.
   Exercise 1 tests this framing.
2. **Calculate** constrained sales, lost demand, terminal stock, and contribution under a price path.
   Exercise 2 tests the inventory arithmetic.
3. **Compare** an early cut, a later cut, and a state-dependent policy under uncertainty.
   Exercises 3 and 4 test expected-value calculations.
4. **Explain** why a stockout censors demand and why observed sales do not identify price response on their own.
   Exercise 5 tests the evidence critique.
5. **Choose** a feasible markdown ladder with salvage, operational, and customer-facing constraints.
   Exercise 6 tests design and execution.
6. **Evaluate** uncertainty, downside, sensitivity to salvage, and the cost of waiting without using future information.
   Exercise 7 tests decision robustness.
7. **Communicate** a recommendation as a forecast, plan, target, and commitment, each clearly distinguished.
   Exercise 8 tests the handoff.

## 3. The core mental model

```text
starting sellable stock + time remaining + observed history
                         |
                         v
          choose a feasible current price and channel
                         |
                         v
       uncertain customer demand and operational execution
                         |
            +------------+------------+
            |                         |
       sales capped by stock     unmet demand if stockout
            |                         |
            v                         v
      contribution now      lost opportunity, not observed sales
            |
            v
     remaining sellable stock and updated information
            |
     more selling periods? ---- yes ----> choose again
            |
            no
            v
  net salvage, donation value, disposal cost, or carryover value
```

The system contains both a physical flow and an information flow.
Physical inventory declines through sales, damage, expiry, transfers, or disposition.
Information improves as sales and availability become visible, although a low sales count may reflect low demand, unavailable stock, or an incorrect price.
An optimizer must not interpret a unit that cannot be sold as evidence that customers did not want it.

A **price path** is a sequence fixed before future demand is known.
A **policy** is a rule that can choose a later price based on information then available.
A policy can be more valuable because it responds to actual remaining stock.
The policy is not allowed to use tomorrow's actual demand when deciding today's price.
A comparison between an adaptive policy and a fixed path must give both the same starting stock, price options, distribution, and objective.

The opportunity cost of selling now is the value of the unit in the best future use.
That use could be a sale at a later price, transfer, or net salvage.
The opportunity cost of waiting includes the chance that a future customer never arrives, the selling window closes, or remaining stock loses value.
This is why a lower current price is neither always good nor always bad.

**Lost demand** is demand that arrives when no saleable stock is available.
It is not necessarily a lost company sale: customers may substitute another item, store, or channel.
**Lost revenue** at a given price is a counterfactual estimate of the foregone transaction value; it is not observed revenue.
For a profit decision, use lost incremental contribution and adjust for substitution rather than treating every missing dollar of revenue as lost profit.

Terminal value must be net of incremental costs.
If a liquidator pays $8 per unit and handling costs $3 per unit, net salvage is $5 per unit.
If disposal costs $2 per unit and there is no recovery, terminal value is negative $2 per unit.
The best feasible terminal route may vary by condition, quantity, channel, and time.
Do not include a hypothetical salvage route unless capacity and eligibility make it available.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Markdown | A reduction from a defined reference selling price; dollars per unit or percent. | Treating the percent reduction as incremental profit. |
| Depth | Magnitude of a reduction from the stated base price. | Comparing percentages with inconsistent reference prices. |
| Timing | Period in which a price takes effect. | Using approval date instead of customer-visible effective time. |
| Horizon | Last period in which the modeled selling action is possible. | Assuming every SKU has the same end date. |
| Period | Decision and demand bucket, such as a week. | Using weekly totals when inventory runs out on Monday. |
| State | Information available at a decision, including stock and time remaining. | Including realized future demand. |
| Action | A price or other permitted decision chosen in a state. | Allowing prices that stores cannot execute. |
| Policy | Mapping from observed state to feasible action. | Calling a fixed calendar an adaptive policy. |
| Price ladder | Ordered allowed prices or markdown levels. | Assuming each price has sufficient demand evidence. |
| Regular price | Price before the markdown under a specified reference convention. | Assuming it is a legally valid prior price. |
| Demand | Requested quantity at a price during a period, including potentially unfilled requests. | Equating it with sales during stockout. |
| Sales | Completed units, bounded by available sellable inventory. | Counting canceled orders as final transactions. |
| Lost demand | Demand not filled because stock is unavailable, in units. | Treating it as observed without evidence. |
| Lost revenue | Counterfactual sales value foregone, in currency. | Equating it with contribution lost after substitution. |
| Censoring | Observation of a lower bound on demand when inventory limits sales. | Assuming demand exactly equals stock. |
| Inventory | Units on hand or available to sell, as specifically defined. | Mixing on-hand, in-transit, and allocated units. |
| Sellable stock | Stock eligible and available for the modeled customer and period. | Counting expired, damaged, or reserved units. |
| Depletion | Reduction in inventory through sales or other flows. | Ignoring shrink, returns, or transfers. |
| Stockout | No inventory available for a relevant customer or channel. | Applying a store-wide flag to every size. |
| Sell-through | Units sold divided by an explicitly defined initial or available quantity. | Maximizing it without valuing margin. |
| Salvage | Best feasible terminal recovery, net of incremental disposition costs, per unit. | Using gross liquidation proceeds as net value. |
| Disposal | Terminal removal of stock, which may have a net cost. | Setting its value to zero when destruction costs money. |
| Carryover | Future use or sale after the modeled horizon. | Counting full next-season price without holding and obsolescence costs. |
| Contribution | Revenue less relevant variable costs; define included costs and currency. | Subtracting sunk purchase cost selectively between options. |
| Sunk cost | A prior expenditure unaffected by the current choice. | Treating it as a marginal cost of keeping versus cutting. |
| Incremental cost | A cost changed by the current action, per unit or per event. | Excluding fulfillment or label costs from an option comparison. |
| Forecast | A probabilistic statement about possible outcomes conditional on assumptions. | Calling a forecast an operational commitment. |
| Plan | A chosen action sequence or state-contingent rule. | Treating an unapproved candidate as an executed plan. |
| Target | A desired result used for management. | Substituting a target for expected demand. |
| Commitment | An authorized promise or operational obligation. | Promising uncertain customer sales. |
| Expected value | Probability-weighted average of modeled outcomes, in currency. | Treating the mean as a guaranteed outcome. |
| Scenario | A coherent combination of demand and operational conditions. | Adding independently selected extremes without considering dependence. |
| Value function | Best expected future objective from a state, in currency. | Treating it as an observed accounting balance. |
| Opportunity cost | Value forgone by choosing one feasible use rather than another. | Calling historical purchase price the opportunity cost. |
| Incrementality | Difference in an outcome under an action versus a credible counterfactual. | Counting all markdown-period sales as incremental. |
| Cannibalization | Sales displaced from other products or channels. | Treating the item-level lift as company-level lift. |
| Pull-forward | Demand moved earlier from a later period. | Adding both the uplift and the displaced future sale. |
| Perishability | Decline in eligibility, value, or remaining life over time. | Assuming all remaining units have the same expiry. |
| Execution lag | Time between decision and customer-visible, operationally effective change. | Assuming a midnight system update guarantees shelf compliance. |
| Reference-price claim | Customer-facing comparison with a stated earlier or other price. | Assuming every internally stored list price is a lawful comparison. |
| Nonanticipativity | Requirement that decisions use only information available when made. | Selecting a historical price using future realized demand. |

## 5. The mechanics, step by step

### 5.1 Define the decision and commercial boundary

1. Name the decision owner and the date of the first action.
   The owner may be a category merchant, not the forecasting team.
   The data scientist supplies analysis; the merchant accepts commercial accountability.
2. Define the stock pool.
   Specify product, size, color, location, channel, condition, and ownership.
   A consolidated pool is valid only if stock can reach the modeled demand before it expires.
3. Define the horizon.
   It might end at the end of a season, a product launch, a food expiry, or a liquidation pickup.
   The modeled end date must reflect customer-facing availability, not only an accounting period.
4. Define the objective.
   A typical objective is expected incremental contribution from now through disposition.
   State whether this is before or after tax, fixed overhead, and funding.
5. Identify required constraints.
   Examples include minimum ticket price, price-ending rules, markdown frequency, legal review, store workload, and vendor funding terms.
6. Record excluded decisions.
   If replenishment, transfers, advertising, or substitution are outside scope, state that the result is conditional on their absence or fixed treatment.

**Input:** decision charter and inventory ownership.
**Output:** a precise decision unit and feasible operating horizon.
**Branch:** if inventory is not controllable by the owner, obtain the necessary operating agreement before optimization.

### 5.2 Reconcile the stock that can actually be sold

1. Start with the timestamped inventory snapshot.
2. Remove units reserved for existing orders and units not yet legally or physically sellable.
3. Separate stock in transit from stock available in each period.
4. Separate damaged, expired, and restricted-condition inventory.
5. Record expected returns only if their timing and saleability are modeled.
6. Reconcile starting stock plus receipts and returns minus sales, transfers, shrink, and disposition to ending stock.
7. For location-specific pricing, reconcile at the location and SKU level.
8. Flag stale inventory records and units with unknown availability.

A one-unit inventory error can matter near the end of a small season.
For a large chain, a small percentage error can still involve substantial value.
Store counts should not be silently pooled if shipment time or fulfillment capacity prevents the pool from serving demand.
For perishable items, track batch and expiry rather than assuming first-expiry-first-out is perfectly executed.

**Input:** inventory ledger, availability, and expiry data.
**Output:** sellable units by state and period.
**Branch:** if availability cannot be reconstructed, widen uncertainty and avoid claiming precise lost demand.

### 5.3 Build a credible price-response distribution

1. Define demand as customer requests, not just completed orders.
2. Gather realized transaction prices, timestamps, availability, promotion exposure, holidays, weather where material, and relevant competitor conditions.
3. Use actual price seen by the customer, including eligible discounts, not merely the intended ticket.
4. Identify stockout intervals and partial availability.
5. Estimate or bound latent demand in censored intervals rather than setting demand equal to sales.
6. Look for variation in price that is not simply a reaction to weak demand or excess stock.
7. Where feasible, use a randomized or otherwise credible quasi-experimental design.
8. Evaluate response on comparable stores, products, and seasons with time-respecting backtests.
9. Translate uncertainty in estimated response into scenarios or predictive distributions.
10. Preserve dependence across time and prices when common shocks, weather, or customer composition matter.

A historic low-price period often follows poor early sales.
A naive regression can confuse the merchant's reaction with customer response.
An estimated elasticity from an unrelated product or regular-price range may not apply to end-of-season clearance.
The relevant response changes with stock depth, assortment, customer awareness, and days remaining.

A model may use counts, a demand curve, an empirical distribution, or simulations.
The model does not need an elaborate functional form to improve a decision, but it does need defensible support for the actions being compared.
Do not extrapolate an exact response to an untested discount merely because the formula returns a number.
If a price has weak evidence, show a range and consider a controlled pilot.

**Input:** price, sales, exposure, availability, and contextual history.
**Output:** conditional distributions of demand under feasible actions.
**Branch:** if causal evidence is weak, report scenario dependence instead of a spurious point optimum.

### 5.4 Value sales and the terminal alternative

For a period with price `p`, unit sales `q`, and incremental per-sale cost `c`, modeled contribution is `(p - c) * q`.
A one-time markdown execution cost is deducted when that action is taken.
Use net price after discounts and expected returns when that is the decision-relevant measure.
If returns are material and happen after the horizon, explicitly model their expected financial and physical effects.

For terminal stock `r` and net salvage value `s`, terminal contribution is `s * r`.
If salvage has a fixed collection fee, capacity limit, tiered price, or minimum lot size, model those terms explicitly.
The highest quoted recovery is not necessarily the highest feasible net recovery.
A donation's financial value depends on actual incremental costs and legally reviewed tax treatment; do not assume the retail price is recoverable.

Past acquisition cost is sunk when stock has already been purchased and all candidate actions deal with the same units.
It can be shown in a separate full-period accounting profit bridge.
It should not be subtracted from only one action, and it should not change the ranking when it is identical across candidates.
Future replenishment or return-to-vendor rights would change that assumption.

**Input:** net transaction economics and feasible disposition contracts.
**Output:** incremental contribution by action and terminal state.
**Branch:** if salvage can be negative, include disposal rather than forcing a zero floor.

### 5.5 Generate executable candidate actions

1. List allowable prices at each decision time.
2. Apply approved price endings, minimum margins if applicable, and vendor restrictions.
3. Decide whether price increases after a markdown are operationally and commercially allowed.
4. Set minimum durations and lead times for labels and digital updates.
5. Remove dates when the product is unavailable or the team cannot execute a change.
6. Record price effective time, channel, and store coverage.
7. Include a no-change action as a legitimate candidate.
8. Include a terminal disposition action when normal selling is no longer possible.
9. If markdowns require approval, capture approval deadlines and fallback actions.

A regular-to-clearance ladder often forbids returning to the higher regular price.
That restriction is a business rule, not a theorem of optimization.
Test a broader action space only as a diagnostic; do not present an infeasible policy as the deployable answer.

**Input:** price governance, operational calendar, and systems constraints.
**Output:** feasible action set for every relevant state.
**Branch:** if a candidate cannot be implemented across intended channels, either model partial implementation or exclude it.

### 5.6 Simulate the inventory and revenue path

For each candidate and demand scenario:

1. Start with available units for the period.
2. Apply any modeled receipts, returns, expiry, and transfers in their actual sequence.
3. Draw or specify demand conditional on the current action and history.
4. Set sales equal to the smaller of available stock and demand when all eligible demand can be served.
5. Record any demand exceeding availability as modeled lost demand.
6. Subtract sales and other outflows from stock.
7. Record price times sales less incremental costs.
8. Carry the state forward to the next decision.
9. Apply terminal salvage or disposal to units remaining at the end.
10. Repeat across scenarios and probability-weight the resulting outcomes.

When demand arrives throughout a period and stockouts occur mid-period, use finer time buckets or an explicit arrival model.
An end-of-week `min(stock, demand)` gives total sold in the simple one-item case, but may miss time-dependent service, cross-channel allocation, and substitution.
For multi-item decisions, the inventory and demand transitions may be coupled.
Do not sum separately optimized SKU values if they compete for the same clearance capacity or customers.

**Input:** starting state, candidate action, response distribution, and cost inputs.
**Output:** scenario-level sales, ending stock, lost demand, and contribution.
**Branch:** if common shocks drive demand across periods, use joint rather than independently sampled scenarios.

### 5.7 Compare timing, depth, and the cost of waiting

A cost of waiting must be defined relative to a specific alternative.
It is not simply the revenue of an unsold unit at today's price.
For example, compare the expected value of choosing a lower price now with the expected value of keeping the current price now and following an optimized feasible continuation.
The difference can be positive or negative.

A deep early cut gains sales from some price-sensitive customers, but discounts customers who would have purchased at the regular price.
A shallow early cut may preserve margin but leave too little time to clear excess units.
A delayed cut preserves the option to sell at a high price if early demand is strong.
That option has value only if the merchant can still observe, approve, and execute the later decision in time.

The following distinctions prevent common errors:

- Price times projected demand is not attainable revenue when demand exceeds stock.
- A missed regular-price sale is not necessarily lost forever if the customer buys later.
- A full sell-through rate does not show whether the price was unnecessarily low.
- Leftover stock is not automatically a total loss if salvage is feasible.
- A low salvage value does not automatically justify selling below the incremental cost of serving the order.
- The cheapest price is not the only way to clear inventory; transfers or a later channel may be feasible alternatives.
- A later markdown is not free if systems and store teams need lead time.
- Comparing forecasts with different assumptions about demand or salvage is not an isolated timing test.

**Input:** scenario outcomes for comparable alternatives.
**Output:** expected and downside contribution, sell-through, lost demand, and terminal stock.
**Branch:** when alternatives are close relative to estimation error, prefer a robust and executable plan rather than an unstable exact optimum.

### 5.8 Optimize a state-dependent policy without foresight

Let `V_t(i)` be the maximum expected future contribution at the start of period `t` with `i` units available.
A fuller state may include location, expiry, previous price, elapsed exposure, and information learned from earlier sales.
At the end of the selling horizon, `V_T(i)` equals the feasible terminal value of `i` units.

For an allowed price `p` and random demand `D_t(p)`, define `q = min(i, D_t(p))`.
With constant per-sale incremental cost `c`, no other flows, and no fixed action cost:

```text
V_t(i) = max over feasible p of
         E[(p - c) * q + V_(t+1)(i - q)]
```

The expectation uses the information and conditional distribution available at period `t`.
If the feasible price depends on a previous markdown, include the previous price in the state.
If early demand signals later demand, update the later distribution conditional on observed history.
If execution is delayed, include the pending action and effective date in the state.

Backward induction evaluates terminal states first, then works back one period at a time.
It evaluates a current action before seeing the current demand realization.
Future actions can react to observations available at their later decision points.
A model that chooses each period's price after seeing that period's demand reports a clairvoyant upper bound, not an implementable plan.

**Input:** action sets, state transitions, conditional demand, and terminal value.
**Output:** a feasible state-to-action policy and its expected value.
**Branch:** if states are too numerous, use a simpler ladder, approximate optimization, or scenario simulation and disclose approximation error.

### 5.9 Stress-test and select

Change one assumption at a time to understand direction, then test coherent joint scenarios.
Useful sensitivities include weaker markdown response, higher salvage recovery, lower sellable stock, slower execution, higher returns, and greater demand correlation.
Examine expected contribution and a downside measure such as a lower outcome percentile.
A percentile is meaningful only relative to the stated probability model.

Compare the candidate policy with the actual current rule, not just a deliberately weak benchmark.
Use the same starting stock, demand process, costs, and decision dates for both.
Separate the modeled value of a policy from the cost and difficulty of deployment.
If the expected gain is smaller than price-label cost or uncertainty in the forecast, a simpler plan may be preferable.
Document the decision rule before inspecting test outcomes.

**Input:** candidate policies and plausible assumption ranges.
**Output:** recommendation, downside exposure, and conditions that would change the choice.
**Branch:** if a policy wins only at an unsupported demand response or salvage assumption, do not call it robust.

### 5.10 Execute, monitor, and learn

1. Assign one owner for each price, channel, product, and effective interval.
2. Confirm the inventory and terminal disposition before publication.
3. Approve the customer-facing price and any comparison claim.
4. Submit the change with enough time for store and digital propagation.
5. Validate a sample of shelf labels, point-of-sale prices, and online checkout prices.
6. Capture actual execution timestamps and exceptions.
7. Monitor sales, available stock, and price realization against the distribution, not just the mean.
8. Trigger a review when inventory, demand, or execution falls outside an agreed range.
9. Preserve the approved fallback when the planned update fails.
10. Reconcile the final inventory with sales, returns, transfers, and disposition.
11. Compare outcome with a credible counterfactual before attributing all improvement to the policy.
12. Feed the evidence back into future model updates and governance.

The plan is the selected action rule.
The forecast is the conditional range of possible results under it.
The target is a management aspiration and must not be fed back as if it were an unbiased forecast.
The commitment is what operations has accepted to execute, such as an approved price update by a stated time.
No team can commit customers to purchase a forecast quantity.

### 5.11 Quantify the value of one more unit

The value function also gives a practical way to explain inventory scarcity.
For a state with `i` units, subtract the value with `i - 1` units from the value with `i` units.
The difference is the marginal value of the additional unit at that time and state.
It reflects the best expected future selling and terminal opportunities, not the original invoice cost.
It can vary with inventory, time remaining, prior price, and the demand outlook.

Use the marginal value carefully:

1. Fix the decision time and all state variables except the one-unit stock difference.
2. Recalculate the best feasible continuation at `i` units.
3. Recalculate it at `i - 1` units using the same information.
4. Subtract the second value from the first.
5. Compare the additional unit's value with the avoidable cost of obtaining or retaining it.
6. Do not apply the result to a large transfer without recomputing all affected units.
7. The first moved unit can have a different value from the twentieth.
8. Check destination capacity and the source location's lost sales before proposing the move.

For a transfer, the relevant benefit is not simply the destination's sales price.
It is the increase in expected destination contribution less the decrease in expected source contribution.
Subtract freight, handling, delay, damage, and any other incremental costs.
A one-way transfer may also make an item unavailable in the source store while it is in transit.
That time gap matters when the remaining season is short.

**Input:** value estimates for adjacent inventory states and feasible transfer economics.
**Output:** an indicative per-unit opportunity value and a transfer comparison.
**Branch:** if the additional unit changes a fixed fee or contract tier, evaluate the complete lot rather than multiplying a single marginal value.

### 5.12 Represent a constrained disposition contract

A real liquidator may impose a minimum lot, accept only selected conditions, or pay different rates by volume.
A single per-unit terminal value hides those discontinuities.
Write the terminal function before comparing price paths.
For example, suppose a buyer accepts at most 50 eligible units for $5 per unit net of variable handling.
The buyer also charges $40 for collection whenever any units are sent.
Excess units cost $2 each to dispose of.
If all remaining units are eligible, the modeled value of sending all accepted units is:

```text
accepted = min(remaining, 50)
excess = max(remaining - 50, 0)
value = 5 * accepted - 2 * excess - 40 * indicator(accepted > 0)
```

At 20 units, that route yields `20 × $5 − $40 = $60`.
At 60 units, it yields `50 × $5 − 10 × $2 − $40 = $190`.
At zero units, the route yields zero because there is no collection.
This calculation assumes that collection is available at every positive eligible quantity.
If disposal without collection is also feasible, evaluate that alternative for the same quantity.
For 5 units, collection yields `5 × $5 − $40 = −$15`.
Disposal at $2 per unit yields `−$10`, so the model should choose disposal instead.
If another collection agreement has a minimum lot, omit it when that minimum is not met.

The terminal function is the best *feasible* alternative, not necessarily the route with the highest quoted unit price.
Its input should include eligibility and capacity, not only total stock.
If a partner accepts different grades at different prices, track grade in the inventory state.
If collection capacity is shared across products, optimize the joint allocation or reserve capacity explicitly.
Otherwise, every product may independently claim the same 50-unit limit.

**Input:** disposition offers, costs, eligibility, capacity, and pickup timing.
**Output:** a terminal-value function by quantity and relevant product attributes.
**Branch:** when contracts are not yet agreed, use scenario ranges and do not represent an indicative quote as a commitment.

### 5.13 Translate a policy into operating thresholds

An optimizer can output an action for every stock count.
Stores and price teams often need a compact rule.
The rule must reflect the actual decision time, allowed actions, and information quality.
Start from the full action table, then identify contiguous ranges with the same recommended action.
Combine ranges only if the resulting loss of expected contribution is understood and accepted.

For a hypothetical later-week review, a table might say:

| Sellable units at review | Approved action | Operational meaning |
|---:|---|---|
| 0 | No price change | There is no stock to clear. |
| 1–40 | Retain $40 | Preserve the higher price under this example rule. |
| 41–80 | Reduce to $30 | Increase the chance of selling larger residual stock. |
| More than 80 | Escalate | Confirm inventory and use an approved exception process. |

These ranges are an illustration of a deployable table, not a result established by the two observed jacket states.
Section 6 only computes the choices at 40 and 80 remaining units.
To approve intermediate thresholds, evaluate every relevant inventory state and its corresponding demand distribution.
If counts arrive late, define which snapshot applies and whether an updated count can change an already submitted action.
If the data are stale beyond the approved tolerance, follow the fallback rather than pretending the state is current.

Check the rule against execution realities:

1. Specify the local date and time at which the inventory snapshot is frozen.
2. Specify the time zone and the time when customers first see the price.
3. Match each rule to eligible products, locations, and channels.
4. Define treatment for negative, missing, or inconsistent inventory.
5. Define who can override a price and what reason must be recorded.
6. Check that the price can be displayed and charged consistently.
7. Record the policy version and the particular inputs used for each decision.
8. Set a fallback if the automated feed, approval, or publication fails.
9. Communicate the rule to customer service before customers encounter it.
10. Close the loop with an exception report and an owner for each unresolved issue.

**Input:** action table, decision rights, system lead times, and tolerances.
**Output:** a short, auditable runbook with approved fallback actions.
**Branch:** if threshold simplification changes the choice in material states, keep the richer rule or explicitly accept the quantified loss.

### 5.14 Evaluate a pilot without confusing selection and impact

A pilot asks whether a proposed decision process improves a business outcome relative to the current process.
It does not establish that every forecast for every product is correct.
Agree on the comparison before the pilot begins.
Choose a unit of assignment that limits interference between treatment and control.
For example, nearby stores sharing customers may not be independent if shoppers travel to the cheaper location.

Document the design in the following order:

1. Define the eligible stores, products, remaining life, and decision dates.
2. Exclude unsafe or otherwise ineligible stock under the same rules in both groups.
3. Define treatment as an operationally executable policy, not just an optimizer score.
4. Define control as the actual business-as-usual rule, including ordinary overrides.
5. Randomize comparable units when feasible and retain the assignment record.
6. Fix the primary outcome as incremental contribution over a stated selling and disposition window.
7. Predefine secondary measures such as terminal stock, waste, and service complaints.
8. Include implementation costs, returns, and net salvage consistently in both groups.
9. Specify how transfers and customer substitution between groups will be handled.
10. Keep assigned units in the main comparison even when a store fails to execute its assigned policy.
11. Report execution compliance separately rather than silently dropping difficult stores.
12. Quantify uncertainty and explain the minimum business gain needed to justify rollout.
13. Inspect whether effects vary by season, stock depth, location, and product category.
14. Decide in advance who can stop the pilot for safety or customer-facing price issues.

The comparison of all assigned units estimates the effect of offering the process under the pilot's actual execution conditions.
It is not automatically the effect of perfect compliance.
An analysis limited to stores that executed correctly can be biased if those stores are systematically easier to operate.
When randomization is impossible, disclose the reasons and use a defensible comparison with pre-period checks.
An observational comparison still requires caution about unmeasured differences.

**Input:** eligible population, baseline practice, outcome ledger, and assignment design.
**Output:** an impact estimate with uncertainty, execution rates, and rollout conditions.
**Branch:** if the feasible pilot is too small to distinguish a plausible gain from noise, present it as a feasibility study rather than proof of financial impact.

## 6. Worked example

### 6.1 The decision

A regional retailer has 100 sellable insulated jackets on Monday morning.
The jackets are not replenishable during the remaining two-week season.
All units are in one accessible fulfillment pool, with no size or location differences in this teaching case.
The retailer can charge $40 or $30 per jacket in each week.
The customer-facing price is the net realized price; tax is excluded.
The initial purchase cost has already been incurred and is the same under every candidate.
There are no additional per-sale or price-change costs in this example.
At the end of week 2, every leftover jacket can be liquidated for $5 net of handling.

The forecast is a probability distribution of weekly demand, not a promised sales volume.
At $40, weekly demand is 20 or 60 jackets, each with probability 0.50.
At $30, weekly demand is 60 or 100 jackets, each with probability 0.50.
Demand realizations are independent across weeks conditional on the chosen prices.
The two price distributions are assumptions for instruction, not empirical elasticity estimates.
Customers who cannot find stock do not substitute to another retailer-owned product in this simplified case.
Sales in a week are capped by stock available at the start of that week.
A decision made at the start of week 2 may use week-1 sales and remaining inventory, but not week-2 demand.

The merchant's initial plan is to hold $40 in week 1 and then decide after seeing remaining stock.
The aspirational target is to sell all 100 jackets during the two-week season.
That target is not the demand forecast and is not a customer commitment.
Operations can commit to updating the week-2 price before opening, assuming a preapproved ladder.

### 6.2 Fixed $40 then $40

The four demand pairs each have probability 0.25.
All figures below are US dollars except units and probabilities.

| Week-1 demand | Week-2 demand | Sales in weeks 1 and 2 | Leftover | Selling revenue | Net salvage | Total contribution |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 20 + 20 | 60 | $1,600 | $300 | $1,900 |
| 20 | 60 | 20 + 60 | 20 | $3,200 | $100 | $3,300 |
| 60 | 20 | 60 + 20 | 20 | $3,200 | $100 | $3,300 |
| 60 | 60 | 60 + 40 | 0 | $4,000 | $0 | $4,000 |

In the last row, 20 jackets of requested demand cannot be filled.
The potential transaction revenue at the current price is 20 × $40 = $800.
That $800 is a modeled lost-revenue opportunity, not observed revenue and not an extra cost to subtract from the already inventory-constrained contribution.
Expected total contribution is `(1,900 + 3,300 + 3,300 + 4,000) / 4 = $3,125`.
Expected terminal stock is `(60 + 20 + 20 + 0) / 4 = 25 jackets`.
Expected sell-through is 75% of the initial stock.

### 6.3 Fixed $40 then $30

| Week-1 demand at $40 | Week-2 demand at $30 | Week-1 sales | Week-2 sales | Leftover | Total contribution |
|---:|---:|---:|---:|---:|---:|
| 20 | 60 | 20 | 60 | 20 | $800 + $1,800 + $100 = $2,700 |
| 20 | 100 | 20 | 80 | 0 | $800 + $2,400 = $3,200 |
| 60 | 60 | 60 | 40 | 0 | $2,400 + $1,200 = $3,600 |
| 60 | 100 | 60 | 40 | 0 | $2,400 + $1,200 = $3,600 |

Expected contribution is `(2,700 + 3,200 + 3,600 + 3,600) / 4 = $3,275`.
Expected terminal stock is 5 jackets; expected sell-through is 95%.
In this model the fixed later cut earns $150 more than holding $40 throughout.
The two policies share the same week-1 action, so the $150 difference isolates the consequence of their fixed week-2 action under these assumptions.

### 6.4 Cut immediately

For a fixed $30 then $30 path, at least 60 units of demand arrive in each week.
All 100 units sell at $30 under each scenario, yielding $3,000 contribution and no salvage.
The expected sell-through is 100%, which is higher than under the fixed late-cut policy, but expected contribution is $275 lower.
This is an explicit example of why maximizing sell-through alone can be misleading.

For completeness, suppose a diagnostic path charges $30 in week 1 then $40 in week 2.
If week-1 demand is 60, week-2 $40 demand of 20 or 60 produces $2,700 or $3,400 total contribution.
If week-1 demand is 100, all stock is gone and total contribution is $3,000 regardless of week-2 demand.
The expected total is `(2,700 + 3,400 + 3,000 + 3,000) / 4 = $3,025`.
A real merchant may prohibit the $30-to-$40 increase; it is shown here only as a diagnostic, not a recommended clearance ladder.
The best feasible fixed non-increasing path in this example is $40 then $30, at $3,275.

### 6.5 Adapt the second-week decision

After selling 20 jackets in week 1, 80 remain.
At $40 in week 2, the two total outcomes are $1,900 and $3,300, averaging $2,600.
At $30 in week 2, the two total outcomes are $2,700 and $3,200, averaging $2,950.
The merchant should therefore cut to $30 in week 2 in this state.

After selling 60 jackets in week 1, 40 remain.
At $40 in week 2, the outcomes are $3,300 and $4,000, averaging $3,650.
At $30 in week 2, both outcomes are $3,600, averaging $3,600.
The merchant should therefore keep $40 in week 2 in this state.

The resulting policy holds $40 in week 1, cuts to $30 if 80 units remain, and holds $40 if 40 units remain.
Its expected contribution is `0.5 × $2,950 + 0.5 × $3,650 = $3,300`.
It improves expected contribution by $25 relative to the best fixed path, assuming zero extra execution cost and the stated demand distribution.
The $25 is the modeled value of this particular adaptive decision, not a guaranteed realized benefit.
Its four outcomes are $2,700, $3,200, $3,300, and $4,000.
Expected terminal stock is `(20 + 0 + 20 + 0) / 4 = 10 jackets`.
Expected sell-through is 90%, below the fixed later cut's 95%, while expected contribution is higher.

The example's best adaptive first-week price is also $40: starting at $30 yields $3,025 under the unconstrained diagnostic continuation and $3,000 if price increases are prohibited.
Thus $3,300 is best among the available adaptive choices in this two-period model.
With no useful new information, a $25 execution charge for the adaptive rule would eliminate its expected advantage over the fixed late-cut path.
Actual deployment costs and uncertainty may favor the simpler policy.

### 6.6 Interpret the trade-off

The wait-or-cut comparison depends on remaining inventory.
With 80 units left, the expected $350 gain from cutting in week 2 reflects the risk of carrying too much stock to $5 salvage.
With 40 units left, retaining $40 gains $50 in expectation relative to cutting because available inventory is scarce relative to demand.
These are conditional differences after the corresponding week-1 observation, not amounts that can be added without their 0.5 probabilities.

The numerical recommendation changes if price response, salvage, demand dependence, or the execution cost changes.
For example, a liquidator contract with limited capacity invalidates the assumption that every residual unit earns $5.
If weak first-week demand also predicts weak second-week demand, independence understates or overstates the value of waiting depending on the conditional model.
The case demonstrates a decision procedure, not a transferable markdown percentage.

### 6.7 Trace the conditional choice as a manager would

The first week ends with either 80 or 40 jackets remaining in this deliberately simple case.
Both states are observed before the second-week price is chosen.
The merchant does not observe whether the second week's demand will be the low or high outcome.
The following bridge separates already-earned value from the decision still available:

| Decision-time state | Week-1 contribution | Week-2 choice | Expected week-2 sales value | Expected salvage | Expected total |
|---|---:|---:|---:|---:|---:|
| 80 remain | $800 | $40 | $1,600 | $200 | $2,600 |
| 80 remain | $800 | $30 | $2,100 | $50 | $2,950 |
| 40 remain | $2,400 | $40 | $1,200 | $50 | $3,650 |
| 40 remain | $2,400 | $30 | $1,200 | $0 | $3,600 |

For 80 units at $40, the two possible sales quantities are 20 and 60.
Their mean is 40 jackets, giving `40 × $40 = $1,600` in expected sales value.
The corresponding terminal quantities are 60 and 20, whose mean is 40 jackets.
Expected net salvage is `40 × $5 = $200`.
For 80 units at $30, the sales quantities are 60 and 80, with mean 70.
Their expected sales value is `70 × $30 = $2,100`.
Their mean terminal stock is 10 jackets, giving $50 expected salvage.
The cut adds $500 of expected sales value but sacrifices $150 of expected salvage.
The net gain in this state is therefore $350.

For 40 units at $40, the sales quantities are 20 and 40, with mean 30.
Expected sales value is `30 × $40 = $1,200`, and expected salvage is $50.
For 40 units at $30, all 40 jackets sell in both demand outcomes.
Expected sales value is also $1,200, but there is no salvage.
The higher price therefore wins by $50 in this state.
That comparison does not rely on knowing which demand outcome will occur.
If the merchant could see the demand outcome first, the resulting value would describe an unattainable information advantage.

The two state-specific improvements have opposite signs when compared with always cutting.
In the 80-unit state, always cutting already takes the preferred action, so adaptation adds $0.
In the 40-unit state, adaptation improves value by $50.
Each state has probability 0.50, so the overall gain is `0.50 × $0 + 0.50 × $50 = $25`.
This bridge explains the $25 gain without adding the $350 conditional gain a second time.

### 6.8 Sensitivity to the terminal offer

Keep the original demand assumptions, prices, and inventory unchanged.
Let `s` denote the net terminal recovery in dollars per jacket.
For each fixed price path, expected contribution is expected sales revenue plus `s` times expected leftover stock.
The three feasible fixed non-increasing paths have these expressions:

| Fixed path | Expected sales revenue | Expected leftover | Expected contribution as a function of `s` |
|---|---:|---:|---|
| $40 then $40 | $3,000 | 25 jackets | `$3,000 + 25s` |
| $40 then $30 | $3,250 | 5 jackets | `$3,250 + 5s` |
| $30 then $30 | $3,000 | 0 jackets | `$3,000` |

At the original `s = $5`, the first two paths produce $3,125 and $3,275.
At `s = $10`, they produce $3,250 and $3,300.
The hold path gains more from a higher salvage value because it leaves more stock.
Set the first two expressions equal to find the switch point:

```text
3,000 + 25s = 3,250 + 5s
20s = 250
s = $12.50 per jacket
```

At exactly $12.50 per jacket, both paths yield $3,312.50.
Below that value, the fixed later cut beats holding $40 under these assumptions.
Above that value, holding $40 beats the fixed later cut.
The early-cut path remains at $3,000 in this comparison.
This is a switch point for fixed paths, not necessarily the switch point for an adaptive policy.
It also assumes the terminal offer is linear and feasible for every leftover jacket.
If a higher offer covers only a limited quantity, the expected-value expressions need a scenario-level capacity calculation.

### 6.9 Distinguish a target from a decision gate

The manager may prefer zero leftover stock as an operational target.
The expected-value model instead recommends the adaptive policy before execution costs.
That policy leaves an average of 10 jackets, so the target and recommended forecast differ.
A target is not an authorization to substitute 100 jackets of assumed demand into every scenario.
If zero residual stock is a mandatory constraint, state whether it must hold in every scenario or only with a specified probability.
Then solve the constrained decision rather than describing the target as a forecast.
The fixed $30-then-$30 path clears stock in every modeled scenario, but earns $3,000 before execution costs.
Its expected contribution is $300 below the adaptive policy's $3,300.
That difference is the modeled contribution cost of this particular all-scenarios clearance choice.
It is not a universal cost of meeting a clearance target in other products or demand distributions.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Copy it into a file and run `python markdown_example.py`.
It enumerates discrete demand outcomes exactly, not by random simulation.
It uses the case's independent weekly distributions, $5 net terminal salvage, zero incremental selling cost, and 100 starting jackets.
The adaptive rule allows only non-increasing prices.
It compares fixed feasible paths with the best adaptive policy and prints conditional week-2 decisions.
Future demand is averaged before the action is selected; no future demand is observed by the policy.

```python
from functools import lru_cache
from itertools import product

STOCK = 100
HORIZON = 2
SALVAGE = 5
PRICES = (40, 30)
DEMAND = {
    40: ((20, 0.5), (60, 0.5)),
    30: ((60, 0.5), (100, 0.5)),
}


def fixed_value(path):
    """Expected contribution for a fixed sequence of prices."""
    def visit(period, inventory):
        if period == len(path):
            return SALVAGE * inventory
        price = path[period]
        total = 0.0
        for demand, probability in DEMAND[price]:
            sold = min(inventory, demand)
            total += probability * (
                price * sold + visit(period + 1, inventory - sold)
            )
        return total
    return visit(0, STOCK)


@lru_cache(None)
def optimal(period, inventory, previous_price):
    """Return (expected future contribution, action) for an observed state."""
    if period == HORIZON:
        return SALVAGE * inventory, None
    actions = [price for price in PRICES if price <= previous_price]
    candidates = []
    for price in actions:
        value = 0.0
        for demand, probability in DEMAND[price]:
            sold = min(inventory, demand)
            continuation, _ = optimal(period + 1, inventory - sold, price)
            value += probability * (price * sold + continuation)
        candidates.append((value, price))
    # On a tie this chooses the higher price; either tied action has equal value.
    return max(candidates)


for path in product(PRICES, repeat=HORIZON):
    if all(path[t + 1] <= path[t] for t in range(HORIZON - 1)):
        print(f"fixed {path}: ${fixed_value(path):,.2f}")

value, first_price = optimal(0, STOCK, max(PRICES))
print(f"adaptive: ${value:,.2f}; first-week price ${first_price}")
for remaining in (80, 40):
    future_value, price = optimal(1, remaining, 40)
    print(f"week 2, {remaining} left: price ${price}, "
          f"future value ${future_value:,.2f}")
```

Expected output:

```text
fixed (40, 40): $3,125.00
fixed (40, 30): $3,275.00
fixed (30, 30): $3,000.00
adaptive: $3,300.00; first-week price $40
week 2, 80 left: price $30, future value $2,150.00
week 2, 40 left: price $40, future value $1,250.00
```

The conditional future values omit week-1 revenue: $800 for the 80-unit state and $2,400 for the 40-unit state.
Adding those amounts produces $2,950 and $3,650 respectively, matching the worked example.
The recursion's state includes the previous price because the allowed next action depends on it.
A production implementation would additionally model action costs, execution delays, time-varying demand, and any constraints that matter to the decision.
Use integer currency minor units or an appropriate decimal type when financial rounding matters.
Do not run the example unchanged on a different product and present its assumed demand as an estimate.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

Each transaction record needs a stable product identifier, store or channel, transaction time, quantity, realized unit price, discount, currency, and return or cancellation status.
Each availability record needs product, location, timestamp, available units, reserved units, and stockout or visibility status.
Each price record needs intended and actual effective times, eligible customer scope, and the applicable price or promotion rule.
A price change table without execution timestamps is a plan, not proof that a customer could buy at that price.

Inventory facts need a defined as-of time and a reconciliation to the stock ledger.
If batches expire at different times, include lot, best-before or use-by data where applicable, and disposition eligibility.
For a financial decision, attach unit-level fulfillment, handling, liquidation, and disposal economics where material.
Keep gross receipts, tax, returns, and vendor funding distinguishable so that the objective can be rebuilt.

An availability flag is not automatically proof of customer exposure.
Online an item can be technically in stock but hidden in search results or undeliverable to the shopper's location.
In store, a unit can be in a back room while the shelf is empty.
Record the practical availability definition and assess its measurement error.

### 8.2 Identifiers and traceability

GS1's Global Trade Item Number (GTIN) identifies trade items in the GS1 system; it is a widely used identifier standard, not proof that two records refer to interchangeable inventory in every business context.
Store, size, lot, unit of measure, and channel may still be needed to define a decision state.
GS1 standards support consistent identification and data sharing; they do not supply a causal demand model or determine an optimal markdown.
Check the applicable current GS1 specification and trading-partner requirements before building an integration.

### 8.3 Price claims and consumer protection

For EU price-reduction announcements, Directive 98/6/EC Article 6a, added by Directive (EU) 2019/2161, establishes a prior-price rule generally based on the lowest price applied during a period of at least 30 days before the reduction.
The directive allows Member States options, including for goods liable to deteriorate or expire rapidly and for certain progressively increased reductions.
National implementation, scope, exceptions, and current guidance matter.
A price optimizer cannot decide from the discount percentage alone whether a public “was/now” claim is lawful.
The European Commission's Article 6a guidance is a useful interpretive starting point, not a substitute for local legal advice.

In the United States, the Federal Trade Commission's Guides Against Deceptive Pricing address potentially misleading former-price comparisons and other price claims.
The guides describe enforcement principles, while applicable federal and state law and current facts determine obligations.
A model's internal regular price is not automatically a truthful public reference price.
Legal review should cover the actual claim, relevant sales history, geography, and channel before publication.
This lesson is educational, not legal advice.

### 8.4 Food, quality, and disposal

Where food or other regulated perishables are involved, a commercial markdown does not override applicable safety, labeling, recall, or disposal requirements.
Applicable requirements differ by product and jurisdiction.
The US Food and Drug Administration's Food Code is a model for retail food regulation, not itself uniformly binding law; local adoption and applicable rules control.
Exclude recalled or unsafe units from sellable inventory and assess an authorized disposition route.
Do not use a positive salvage assumption for stock that cannot lawfully be transferred or sold.

### 8.5 Governance and retention

Retain a versioned decision record: input snapshots, forecast version, feasible ladder, approvals, selected policy, actual execution, and realized outcomes.
Define data access and retention rules in accordance with applicable privacy and commercial requirements.
A product-level markdown analysis does not require identifying individual customers by default.
If personal data are introduced, obtain a purpose-specific privacy review rather than assuming that a pricing use is automatically permitted.
Protect commercially sensitive prices and contracts under organizational access controls.

### 8.6 Reconciliation and data-quality controls

Define a common grain before joining price, sales, and inventory.
For an item-store-day analysis, a single row should represent one product, one location, and one local trading date.
If prices change during the day, retain a finer effective interval or a transaction-level price.
A daily average price can hide which customers saw which price.
Use a consistent calendar for week boundaries, daylight-saving transitions, and late-arriving transactions.

Reconcile physical movement using a signed ledger:

```text
opening sellable stock
+ eligible receipts and returns
- completed sales
- transfers out, damage, expiry, and other removals
= closing sellable stock
```

Transfers in belong among eligible receipts only when available to the modeled pool.
Reserved orders should not be deducted twice if the opening snapshot already excludes them.
Returns should enter sellable stock only after inspection and release.
Compare the calculated close with the recorded close and investigate material differences.
Keep adjustment reasons rather than forcing the ledger to balance with an unexplained residual.

Before a decision, review:

1. The fraction of products with a current inventory snapshot.
2. The fraction of sellable units whose location and condition are known.
3. Price records with overlapping effective intervals.
4. Transactions whose realized price differs from the authorized price.
5. Inventory records with negative balances or impossible movement sequences.
6. Items with sales recorded after a supposed stockout.
7. Missing disposition contracts, costs, or capacity records.
8. Returns that arrive outside the modeled contribution window.
9. Records where a product identifier maps to multiple materially different sizes or grades.
10. Sales and stock data received after the decision timestamp.

A reconciliation failure is not always proof that the model must stop.
The owner can sometimes bound the impact and use a conservative fallback.
The decision record should show the failed control, exposure, chosen response, and accountable owner.
Do not silently repair records using information that was unavailable at the historical decision time.

## 9. Industry practice and case evidence

### 9.1 What public research establishes

Cachon and Swinney study dynamic pricing and inventory management when consumers may strategically delay purchases.
Their published model shows that customer anticipation can change the value of a markdown policy.
That is a theoretical result under specified assumptions, not evidence that every customer waits for every sale.
In a practical setting, frequent predictable end-of-season discounts may train some customers to wait; the size of that effect is an empirical question.
A static demand curve that ignores intertemporal customer behavior can therefore misstate the gain from a deep late cut.

Talluri and van Ryzin's revenue-management text treats finite capacity, stochastic demand, and the value of preserving capacity for later customers.
The same opportunity-cost logic helps explain a finite retail inventory decision.
An airline seat and a physical jacket differ: the jacket may have positive liquidation value and transfer costs, whereas a departed seat generally has no later sale value.
Do not transfer an airline protection rule directly to a jacket without adapting the state, terminal value, and demand model.

### 9.2 Named public retail example

Zara's owner Inditex describes its integrated store and online model, logistics, and inventory-related practices in its annual reports.
Those reports provide public company evidence about the operating setting, not a public disclosure of the exact markdown optimization algorithm used for every garment.
**Inference:** an integrated view of store and online inventory can make transfers or fulfillment alternatives relevant before a terminal markdown.
Whether a specific transfer is profitable depends on shipping cost, time, availability, and demand; it is not established by a general annual-report statement.
The worked example's single inventory pool is an idealization, not a claim about Zara's actual system.

### 9.3 Perishable retail example

The UK Waste and Resources Action Programme (WRAP) publishes guidance and evidence on food surplus and waste in retail supply chains.
Its material supports considering redistribution and waste prevention as alternatives in the disposition hierarchy.
It does not establish that discounting always reduces waste or that the same markdown threshold is optimal for every store.
**Inference:** for a short-dated product, a timed price reduction may compete with donation, redistribution, and disposal for the remaining usable life.
The relevant comparison uses feasible net value and operational timing, including safety restrictions.
A positive headline selling price can still be inferior to a feasible alternative if fulfillment or handling costs are higher.

### 9.4 What to transfer into a business decision

Use published theory to identify mechanisms and failure modes.
Use company reports to understand disclosed operating context, not to fabricate numerical outcomes.
Use the organization's own transaction, inventory, and execution records to calibrate demand and contribution.
Test whether the same policy behaves differently by product life cycle, store, channel, and remaining stock.
When evidence is weak, a pilot with a predefined comparison and decision rule is more informative than a highly precise unsupported optimization.

### 9.5 Commercial handoff and post-season review

In a practical review, merchants typically need a small number of comparable alternatives rather than an unexplained mathematical score.
Show the current rule, the proposed rule, and at least one operationally simple alternative.
For each, report expected contribution, a plausible outcome range, expected terminal units, and the assumptions that most affect the ranking.
Separate cash timing from contribution because a retailer may care about both.
Cash proceeds can arrive in a different period from a financial recognition entry or a liquidation pickup.

Ask each stakeholder a different question:

| Stakeholder | Decision question | Evidence to bring |
|---|---|---|
| Merchant | Is the price and brand treatment acceptable? | Price ladder, customer scope, and forecast. |
| Finance | Is the incremental gain material? | Comparable contribution bridge and cost assumptions. |
| Store operations | Can the change be executed on time? | Lead time, staffing, label process, and fallback. |
| Digital commerce | Will online display and checkout agree? | Publication timing and channel exception report. |
| Supply chain | Is terminal capacity available? | Inventory by condition, collection schedule, and contract. |
| Legal or compliance | Is the actual claim appropriate for the jurisdiction? | Customer-facing wording and relevant price history. |

After the season, distinguish four questions.
First, did the team execute the approved actions at the intended times?
Second, were the available stock and disposition assumptions accurate?
Third, did the demand distribution cover the observed sales and stockout patterns?
Fourth, what was the policy's incremental effect against a credible comparison?
The first three questions can reveal an implementation or forecasting problem.
They do not by themselves answer the fourth, because the unchosen action's outcome is not observed.

Record the findings in the next season's decision inputs.
For example, a late pickup may imply that a future terminal value needs extra storage cost.
A failed label update may imply that the modeled execution lag was too short.
A stockout may indicate stronger demand, less inventory than recorded, or both.
Investigate those explanations before changing the price-response model.
Do not generalize a single favorable season to all products without checking comparability.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Demand treated as sales | Many periods sell exactly all available stock. | Stockouts censor demand. | Underestimated demand and poor prices. | Model availability and bound latent demand. |
| Endogenous price | Deep cuts follow weak sales. | Price responds to unobserved demand. | Biased price response. | Use credible experiments or identification and sensitivity. |
| Maximizing sell-through | Every recommended price is the lowest permitted. | Margin and salvage are ignored. | Unnecessary discounting. | Optimize incremental contribution. |
| Gross salvage used | Liquidation invoice excludes handling. | Costs and capacity are omitted. | Waiting looks too attractive. | Use feasible net terminal recovery. |
| Sunk cost double counted | Historic purchase cost differs across candidate rows for the same units. | Unchanged cost enters selectively. | Wrong decision ranking. | Use a consistent incremental bridge. |
| Future demand leakage | Backtest selects the realized best price each week. | The model sees future outcomes. | Inflated estimated gain. | Enforce decision-time information and nonanticipativity. |
| Stale inventory | System stock exceeds shelf count. | Damaged or reserved stock is counted. | Unfillable orders and lost trust. | Reconcile sellable stock and timestamp. |
| Pooling unavailable stock | A region appears fully stocked while local sizes are absent. | Transfers cannot meet local demand. | False service and revenue estimates. | Model location, lead time, and transfer capacity. |
| Missing price execution | Plan shows a cut but checkout remains unchanged. | Publication or label lag. | Forecast-to-actual gap and customer complaints. | Capture actual prices and exception resolution. |
| Weak clearance evidence | An untested 70% cut gets a precise lift estimate. | Extrapolation beyond observed prices. | Unreliable optimum. | Bound response and pilot feasible alternatives. |
| Ignored substitution | Item sales rise while category sales fall. | Customers switch from full-price products. | Item gain overstates company gain. | Evaluate category contribution and substitution. |
| Ignored pull-forward | Sales drop after the promotion. | Earlier purchases replace later ones. | Double-counted incremental demand. | Extend the measurement window. |
| Common shocks ignored | All stores miss together after weather changes. | Independent demand draws understate joint risk. | Underestimated leftover stock. | Model correlated scenarios and backtest. |
| Flat expiry assumption | Some lots are already near expiry. | One terminal date hides earlier removal. | Unsafe or impossible sales forecast. | Track lot-specific eligibility and outflows. |
| Infeasible salvage | Buyer accepts only a fraction of stock. | Capacity or quality limits omitted. | Overvalued residual stock. | Model contract limits and alternative disposal. |
| Misleading prior price | Reference price was briefly applied or is not documented. | Internal list price is used as a public claim. | Customer harm and regulatory exposure. | Preserve history and obtain local legal review. |
| Returns omitted | Clearance returns arrive after the horizon. | Revenue counted without reversal or handling. | Overstated contribution. | Model expected return cost and eligibility. |
| Policy churn | Recommended prices change daily on noisy data. | The optimizer reacts to random fluctuations. | Label costs and customer confusion. | Set review cadence and action thresholds. |
| Optimization without ownership | Nobody approves or deploys the proposed ladder. | Analytical and operational processes are disconnected. | No realizable benefit. | Assign decision rights and a fallback. |
| Double-counted lost revenue | Lost-demand revenue is added as a penalty after stock-constrained sales. | The same opportunity is valued twice. | Distorted price choice. | Define the objective and counterfactual once. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or relax |
|---|---|---|---|
| Starting stock is known. | Sales cannot exceed available units. | Shrink, stale counts, or hidden reservations. | Reconcile snapshots to physical counts and orders. |
| Stock is interchangeable. | One inventory state represents all demand. | Sizes, locations, or expiry lots differ. | Model relevant pools and transfer constraints. |
| Demand responds to the chosen price as specified. | Candidate prices need outcome distributions. | Confounding or an untested price range. | Use experiments, backtests, and response bounds. |
| Weekly demands are independent in the case. | Four pairs have equal probability. | Weather, persistent demand, or learning. | Estimate conditional or joint scenarios. |
| Customers cannot be served after stockout. | `min(stock, demand)` defines sales. | Backorders, substitution, or replenishment. | Model these flows explicitly. |
| No substitution or pull-forward in the case. | Single-item contribution is sufficient. | Buyers switch products or purchase earlier. | Measure a wider category and time horizon. |
| $5 salvage is feasible for every leftover unit. | Terminal value is linear. | Contract caps, quality restrictions, or fees. | Use tiered net disposition functions. |
| No incremental sales or action costs in the case. | Revenue plus salvage equals contribution. | Shipping, labor, returns, or label changes vary. | Add action-specific costs and execution lag. |
| Prior purchase cost is fixed. | It cancels in a comparison of actions. | Return-to-vendor or replenishment rights differ. | Include avoidable acquisition or return cash flows. |
| Decision maker observes the state before acting. | Week-2 policy can use remaining stock. | Delayed or inaccurate sales feeds. | Model observation lag and error. |
| Price changes are non-increasing in code. | The policy respects the stated ladder. | A business permits or requires other changes. | Change action sets after governance review. |
| Risk-neutral expected contribution is the objective. | It gives a simple ranking. | Cash, waste, service, or risk constraints dominate. | Add explicit constraints and report downside. |
| Prices do not affect demand through competing actions. | Single-item prices can be evaluated alone. | Competitor reaction or internal cannibalization. | Evaluate joint scenarios or category response. |
| All demand arrives during modeled periods. | End-of-horizon salvage follows week 2. | Sales or returns occur after the cutoff. | Extend the horizon or estimate carryover value. |

The arithmetic is exact for the stated discrete model; the model itself is a simplified forecast.
There is no universal rule that the deepest cut should occur in the final week.
There is also no universal rule that a higher salvage value always changes the selected policy: the answer depends on the entire distribution and action set.

## 12. Exercises

### Exercise 1 — Frame the decision

A merchant has 240 units of seasonal footwear across three stores and two sizes.
The season ends in four weeks, and an approved liquidator will collect eligible leftovers afterward.
Define the decision owner, state, horizon, action set, objective, and terminal disposition.
**Deliverable:** a one-page decision charter.
**Acceptance:** all six elements are explicit; store and size pooling are justified or rejected; forecast and commitment are distinguished.
**Time:** 15 minutes.

### Exercise 2 — Inventory and contribution

Start with 70 units; demand at $25 is 48 units in week 1 and 40 units in week 2.
There are no receipts, returns, or variable selling costs; net salvage is $4 per remaining unit.
**Deliverable:** week-by-week sales, lost demand, terminal stock, and total contribution.
**Acceptance:** sales never exceed available stock; the calculation separately identifies modeled lost revenue without double-counting it.
**Time:** 10 minutes.

### Exercise 3 — Compare fixed paths

Use the jacket case in section 6.
Compare $40 then $40 with $40 then $30 using all four equally likely outcomes.
**Deliverable:** expected contribution and terminal stock for each path, plus a decision.
**Acceptance:** salvage is included once, scenarios are probability weighted, and the common first-week action is acknowledged.
**Time:** 20 minutes.

### Exercise 4 — Conditional policy

Use the same jacket assumptions and suppose 60 jackets sold in week 1 at $40.
Choose the week-2 price without seeing week-2 demand.
**Deliverable:** conditional expected total contribution at each price and the selected action.
**Acceptance:** cap week-2 sales at 40; include the already earned week-1 revenue consistently; explain why realized demand is unavailable at decision time.
**Time:** 15 minutes.

### Exercise 5 — Critique evidence

A colleague reports that low-price weeks sell 30% more than full-price weeks and concludes that a cut caused a 30% lift.
Some full-price weeks had stockouts and many cuts followed slow sales.
**Deliverable:** a short critique and two practical evidence improvements.
**Acceptance:** identify both censoring and endogenous markdown timing; distinguish sales from demand and describe a credible comparison.
**Time:** 20 minutes.

### Exercise 6 — Design a deployable ladder

A grocery chain has dated stock, a 24-hour shelf-label lead time, a donation partner with limited daily capacity, and a local legal review requirement for price claims.
**Deliverable:** a state-based price ladder and execution checklist.
**Acceptance:** address expiry, action lag, donation capacity, actual price verification, approvals, and a fallback for failed execution; do not assume a universal legal exception.
**Time:** 25 minutes.

### Exercise 7 — Test a salvage sensitivity

For the fixed $40-then-$40 jacket path, keep demand assumptions unchanged but raise net salvage from $5 to $10 per leftover jacket.
Compare it with the original $40-then-$30 result recalculated at the same $10 salvage.
**Deliverable:** both expected contributions and the new difference.
**Acceptance:** change salvage for both policies, retain all other assumptions, and show expected terminal units.
**Time:** 15 minutes.

### Exercise 8 — Write the decision memo

Use the adaptive jacket policy and its $3,300 expected contribution.
The fixed $40-then-$30 alternative has $3,275 expected contribution and needs no extra execution.
Assume adapting costs $40 in additional incremental labor.
**Deliverable:** a five-sentence recommendation to the merchant.
**Acceptance:** include the net comparison, downside or uncertainty, a forecast, a target, and an operational commitment; identify a condition that would change the decision.
**Time:** 15 minutes.

## 13. Self-check answers

### Exercise 1

A strong charter identifies a merchant with authority to approve prices and a named operations owner for execution.
The state should include sellable units by store and size, weeks remaining, prior price, and any transfer or liquidation constraints.
The horizon ends after four weeks of selling and includes the subsequent feasible liquidation action.
The action set specifies allowed prices, dates, and locations, including a no-change option.
The objective is expected incremental contribution through selling and net disposition, with relevant constraints explicit.
Pooling is justified only if stock can be moved in time and demand can be served without material size or location differences.
A forecast is a distribution of sales; a commitment might be publishing an approved price by a specified time.

### Exercise 2

Week 1 sells 48 units, leaving 22 units.
Week 2 demand is 40 units but sales are capped at 22 units, leaving zero units.
Lost demand is 18 units, with $450 of modeled lost transaction revenue at $25 per unit before substitution or other adjustments.
Contribution is `(48 + 22) × $25 + 0 × $4 = $1,750`.
Do not subtract the $450 again from this already capped sales calculation.

### Exercise 3

The $40-then-$40 path has expected contribution $3,125 and expected terminal stock 25 jackets.
The $40-then-$30 path has expected contribution $3,275 and expected terminal stock 5 jackets.
The fixed later cut is better by $150 under the assumptions.
Both paths use $40 in week 1, so their difference is attributable within this model to the fixed week-2 choice.

### Exercise 4

Week-1 revenue is `60 × $40 = $2,400`.
At $40 in week 2, demand of 20 produces $800 sales plus $100 salvage; demand of 60 produces 40 sales for $1,600 and no salvage.
Total contributions are therefore $3,300 and $4,000; their conditional mean is $3,650.
At $30 in week 2, both demand outcomes exceed the 40 units left, so the total is $2,400 + $1,200 = $3,600.
Keep $40: its conditional expected advantage is $50.
The choice uses the observed 40-unit inventory state, not the unobserved next demand draw.

### Exercise 5

The sales comparison is not a causal estimate because price cuts may be responses to weak underlying demand.
Stockouts also cap observed sales and can make a high-price period look weaker than its latent demand.
A suitable randomized store or time assignment, where feasible and legally approved, can improve identification.
Availability-aware demand estimation and a matched comparison with pre-period checks can complement it.
Measure the category and subsequent weeks to assess substitution and pull-forward.
No single observational adjustment automatically proves causation.

### Exercise 6

A strong design conditions decisions on sellable quantity by batch and remaining safe life.
It schedules decisions at least 24 hours before the intended shelf change and includes a safe fallback if labels do not change.
It treats donation capacity and collection windows as constraints on net terminal value.
It requires legal approval of the actual customer-facing claim under applicable local rules.
It samples shelf and checkout prices and removes unsafe or recalled goods from the sellable pool.
There is no single numeric ladder without demand, costs, eligibility, and jurisdiction-specific information.

### Exercise 7

The $40-then-$40 path has 25 expected leftover jackets, so its value rises by `25 × $5 = $125` to $3,250.
The $40-then-$30 path has 5 expected leftover jackets, so its value rises by `5 × $5 = $25` to $3,300.
The late-cut path remains better by $50, down from $150.
The comparison changes both terminal values, not just the path being tested.

### Exercise 8

The adaptive rule nets $3,300 − $40 = $3,260 in expected contribution, below the fixed policy's $3,275 by $15.
Recommend the fixed later cut under these inputs, rather than paying more for adaptation.
A good memo calls $3,275 an expectation under the stated distribution, not a commitment.
It may set 100 units sold as a target while acknowledging expected sell-through is 95%.
Operations can commit to the approved week-2 update, not to realized customer demand.
A sufficiently lower adaptation cost, or a different validated demand model, could change the recommendation.

## 14. Interview questions

1. **Why can full sell-through be a poor objective?**  
   Strong: compares incremental contribution, discount given to baseline buyers, and salvage.  
   Weak: assumes selling every unit at any price maximizes profit.
2. **How do you identify a stockout-censored observation?**  
   Strong: uses time-stamped availability and recognizes observed sales as a demand lower bound.  
   Weak: treats a zero sale as proof of zero demand.
3. **What is the opportunity cost of selling a unit today?**  
   Strong: names its best feasible future sale or net salvage value under uncertainty.  
   Weak: uses only the already-paid acquisition cost.
4. **What information can an adaptive policy use?**  
   Strong: observed state and history available at the decision time.  
   Weak: selects the action after seeing the future demand draw.
5. **How does a markdown's timing affect value?**  
   Strong: balances early conversion, margin erosion, later uncertainty, and execution lead time.  
   Weak: asserts that later or earlier is always better.
6. **How should liquidation be valued?**  
   Strong: uses feasible net proceeds after costs, capacity, quality, and timing.  
   Weak: uses a gross quote for every unit without checking eligibility.
7. **Why is historical price-sales correlation not sufficient?**  
   Strong: explains reactive markdown timing, availability, and possible experimental designs.  
   Weak: equates a correlation coefficient with causal elasticity.
8. **How would you model dated food?**  
   Strong: distinguishes lots, remaining life, safety eligibility, and alternative disposition.  
   Weak: discounts unsafe goods as if price resolved compliance.
9. **How do you compare a policy with an existing calendar?**  
   Strong: holds inputs and feasible actions consistent, measures contribution and uncertainty, and includes execution costs.  
   Weak: compares different demand assumptions or reports only total sales.
10. **What makes a proposed markdown deployable?**  
    Strong: names approval, price propagation, inventory accuracy, verification, monitoring, and fallback.  
    Weak: assumes an optimizer output changes every shelf automatically.

## 15. Key terms

- [Markdown](#4-definitions-and-vocabulary) — reduction from a defined reference price.
- [Depth](#4-definitions-and-vocabulary) — size of the price reduction.
- [Timing](#5-the-mechanics-step-by-step) — when the reduced price becomes effective.
- [Horizon](#5-the-mechanics-step-by-step) — modeled selling and disposition window.
- [Sellable stock](#5-the-mechanics-step-by-step) — units eligible and available to serve demand.
- [Demand](#4-definitions-and-vocabulary) — desired quantity, including unfilled requests.
- [Censoring](#10-failure-modes-and-red-flags) — loss of demand information when sales hit a stock limit.
- [Lost demand](#3-the-core-mental-model) — unmet customer demand in units.
- [Lost revenue](#3-the-core-mental-model) — counterfactual transaction value not realized.
- [Sell-through](#6-worked-example) — sold units as a fraction of a defined available base.
- [Salvage](#5-the-mechanics-step-by-step) — feasible terminal recovery net of relevant costs.
- [Disposal](#4-definitions-and-vocabulary) — terminal removal that can have a negative value.
- [Carryover](#4-definitions-and-vocabulary) — stock value after the modeled horizon.
- [Contribution](#5-the-mechanics-step-by-step) — revenue less specified incremental costs.
- [Sunk cost](#5-the-mechanics-step-by-step) — historical cost unchanged by the decision.
- [Opportunity cost](#3-the-core-mental-model) — value of the best forgone feasible use.
- [Policy](#3-the-core-mental-model) — mapping from observed state to action.
- [State](#5-the-mechanics-step-by-step) — decision-time information needed for a choice.
- [Action](#5-the-mechanics-step-by-step) — permitted decision in a state.
- [Price ladder](#5-the-mechanics-step-by-step) — permitted sequence of prices.
- [Expected value](#6-worked-example) — probability-weighted model outcome.
- [Scenario](#5-the-mechanics-step-by-step) — coherent set of uncertain inputs and outcomes.
- [Nonanticipativity](#5-the-mechanics-step-by-step) — decisions cannot use future observations.
- [Value function](#5-the-mechanics-step-by-step) — best expected future contribution from a state.
- [Incrementality](#9-industry-practice-and-case-evidence) — difference from a credible counterfactual.
- [Cannibalization](#10-failure-modes-and-red-flags) — displacement of another product's sales.
- [Pull-forward](#10-failure-modes-and-red-flags) — earlier demand replacing later demand.
- [Execution lag](#5-the-mechanics-step-by-step) — delay between price decision and effective change.
- [Forecast](#2-learning-outcomes) — conditional distribution of possible outcomes.
- [Commitment](#2-learning-outcomes) — authorized operational obligation.

## 16. Sources

1. Cachon, Gérard P., and Robert Swinney. “Purchasing, Pricing, and Quick Response in the Presence of Strategic Consumers.” *Management Science* 55, no. 3 (2009): 497–511. [https://doi.org/10.1287/mnsc.1080.0940](https://doi.org/10.1287/mnsc.1080.0940). Peer-reviewed model of pricing, inventory, and strategic customer timing.
2. Talluri, Kalyan T., and Garrett J. van Ryzin. *The Theory and Practice of Revenue Management*. Springer, 2004. [https://doi.org/10.1007/b139000](https://doi.org/10.1007/b139000). Reference on stochastic capacity allocation and dynamic pricing.
3. European Parliament and Council. [Directive (EU) 2019/2161](https://eur-lex.europa.eu/eli/dir/2019/2161/oj), amending consumer-protection directives including price-indication rules. Consult national implementation and current law.
4. European Commission. [Guidance on the interpretation and application of Article 6a of Directive 98/6/EC](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52021XC1229(06)), 2021. Guidance on price-reduction announcements, including exceptions and national options.
5. US Federal Trade Commission. [16 CFR Part 233, Guides Against Deceptive Pricing](https://www.ecfr.gov/current/title-16/chapter-I/subchapter-B/part-233). Federal guidance on pricing representations; state law and current facts also matter.
6. US Food and Drug Administration. [FDA Food Code](https://www.fda.gov/food/fda-food-code). Model retail food code and links to current editions; adoption varies by jurisdiction.
7. GS1. [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/). Standard for assigning global trade item identifiers; not a markdown methodology.
8. Inditex. [Annual reports](https://www.inditex.com/itxcomweb/en/investors/investor-relations/annual-reports). Company-reported operating context; no algorithmic outcome is inferred as fact.
9. WRAP. [Food and drink](https://www.wrap.ngo/taking-action/food-drink). Public resources on food waste prevention and redistribution; applicability depends on the actual product and local arrangements.

## 17. Links

- [Module 16 topic map](./_index.md).
- [Module 16 survey and Lab 16.2: markdown calendar optimisation](../../../course/16_pricing_promotions_revenue_management.md#lab-162--markdown-calendar-optimisation).
- [Pricing decision stack](./01_pricing-decision-stack.md).
- [Price, value, and willingness to pay](./02_value-wtp.md).
- [Own-price and cross-price elasticity](./03_elasticity-identification.md).
