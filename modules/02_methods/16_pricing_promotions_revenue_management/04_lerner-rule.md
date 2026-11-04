# From Elasticity to Price: The Lerner Rule

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 04 of 10  
> Estimated study time: 4 hours  
> Prerequisites: 01 pricing decision stack, 02 value and willingness to pay, 03 elasticity identification  
> Version 2026.5

## 1. Why this topic exists

A pricing manager must recommend next quarter's price before the supply plan is locked.
The sales team wants a lower price to protect volume.
Operations wants higher volume to fill a production line.
Finance wants to protect contribution.
The analyst may have an estimated own-price elasticity but no decision rule connecting it to margin.

The Lerner rule connects marginal cost and demand response at an unconstrained, single-product profit optimum.
It does not say that every observed margin is optimal.
It does not say that a historical elasticity will remain constant after a price change.
It does not replace a demand model, a cost model, or commercial judgment.

The practical question is narrower and more useful:
if you change the price by a small amount, is the contribution lost on units no longer sold offset by the extra contribution on units still sold?
At a smooth interior optimum, these marginal effects balance.
If they do not balance, the current price is not the optimum under the stated model.

The financial stakes can be material even when a price move is small.
A lower price applies to all units sold, not only incremental units.
A higher price earns more on retained units but can cause lost contribution on units no longer purchased.
A decision based only on incremental revenue can therefore recommend the wrong direction.

The supply-chain stakes are equally concrete.
The recommended quantity determines material orders, line loading, distribution capacity, and inventory exposure.
A recommendation outside the demand model's evidence range can create an unrealistic production plan.
A price optimized before a capacity constraint is considered can promise volume that cannot be supplied.

This topic develops the rule from profit rather than presenting it as a memorized markup table.
You will then use a demand curve to check the proposed price, evaluate an operational constraint, and decide when a portfolio calculation or an experiment is needed instead.

> **Note:** Examples are educational and fictional unless a source is explicitly identified.
> A calculated optimum is a model output, not an authorization to change a customer price.

### Decision owner and handoff

The commercial owner defines the objective, customer scope, approval route, and permitted choices.
The analyst estimates demand and quantifies uncertainty.
Finance confirms the relevant marginal economics and accounting reconciliation.
Operations confirms capacity, service risk, and replenishment timing.
Legal or compliance reviews competition, contractual, and consumer-protection issues where applicable.
The decision record must identify who approved the model, who approved the price, and when it will be reviewed.

A completed analysis produces a recommended option and at least one credible alternative.
It also states the conditions under which the recommendation should be withheld.
A business can deliberately choose a price below the modeled contribution maximum for a documented reason.
That choice is not a failure of the rule; it is a different objective or an additional constraint.

### The question the rule cannot answer alone

The rule is local: it evaluates the slope of demand at the candidate price.
It cannot identify that slope from a list of invoices by itself.
Price changes are often caused by expectations about demand, inventory, promotions, or a competitor.
Those same forces affect quantity, so a naive price-versus-sales correlation need not be causal.
Topic 03 describes identification; this topic assumes the demand response is usable and asks what follows.

## 2. Learning outcomes

After this topic, you can perform the following actions.
Each outcome points to an exercise in section 12.

1. **Define** own-price elasticity, contribution, marginal cost, and the Lerner index, including signs and units.
   Exercise 1 checks the definitions.
2. **Derive** the single-product first-order condition and explain the role of the elasticity sign.
   Exercise 2 checks the algebra.
3. **Calculate** a candidate price, test whether its elasticity is evaluated at that price, and compare feasible choices.
   Exercises 3 and 4 require these calculations.
4. **Distinguish** a local optimum, a global optimum, a boundary solution, and a commercial decision.
   Exercise 4 tests this distinction.
5. **Evaluate** the impact of cross-price effects and contribution from other products.
   Exercise 5 checks the direction and magnitude of a portfolio adjustment.
6. **Critique** demand evidence, marginal cost, and operational assumptions before authorizing a change.
   Exercise 6 requires an evidence and risk review.
7. **Design** a decision guardrail and post-launch measurement plan that does not coordinate with competitors.
   Exercises 7 and 8 require a decision memo and a test design.

A successful learner can explain why a 25% gross margin and a 25% markup on cost are different.
A successful learner can also explain why a point elasticity estimated at today's price does not automatically justify substituting that same elasticity into a formula for a far-away price.

## 3. The core mental model

The rule is a balance of two marginal flows.
One flow earns a higher margin on customers who continue to buy.
The other loses the margin that would have been earned on customers who leave.
The business should compare both flows in the same currency, quantity, and period.

```text
Demand evidence + candidate price + relevant marginal cost
                    |
                    v
       Q(p), slope dQ/dp, elasticity e(p)
                    |
                    v
   extra margin on retained units vs lost unit contribution
                    |
                    v
     candidate stationary price and predicted contribution
                    |
          +---------+----------+
          |                    |
    constraints and      other products,
    supply feasibility   uncertainty, policy
          |                    |
          +---------+----------+
                    |
                    v
     approve / test / revise / do not change
                    |
                    v
       monitor causal evidence and outcomes
```

Start with an economically meaningful outcome, not a formula.
For a single product and period, that outcome is contribution after price-dependent revenue and relevant costs.
Then describe how quantity changes when the product's own realized price changes, holding the other modeled conditions fixed.
The demand curve supplies both the predicted number of units and its local slope.
The cost model supplies the marginal cost of serving an additional unit.

An elasticity of -2 means that near a particular price, a 1% price increase is associated with an approximately 2% decrease in quantity under the causal interpretation and held-fixed conditions of the model.
The approximation is local.
The same numerical elasticity should not be interpreted as an exact percentage change for a large price move.

At a smooth interior optimum the extra revenue from one more unit, marginal revenue, equals the marginal cost of that unit.
Equivalently, the share of price left after marginal cost equals the reciprocal of the absolute own-price elasticity at the optimum.
The equality is a necessary condition under the stated assumptions, not a universal guarantee of the best business decision.

The rule is sometimes called the inverse elasticity rule.
Its index is a price-cost margin expressed as a fraction of price.
The index uses marginal cost, not necessarily average cost, fully allocated cost, or the cost of goods sold reported on a financial statement.

### Read the diagram backwards

Begin with the authorization decision.
Ask which options are feasible and whether the organization can implement them fairly and lawfully.
Ask whether adjacent products, downstream contractual terms, or supply constraints alter the economics.
Only then ask whether the model gives a stable optimum.
Finally, inspect the demand evidence and cost inputs that produce that model.

This reverse reading protects against an attractive point estimate becoming an unexamined target.
It also makes the work reproducible: a reviewer can trace a decision back through the model to the data and assumptions.

### A comparison that prevents common mistakes

Revenue equals price times units.
Contribution equals revenue less relevant variable costs.
Accounting profit may additionally subtract fixed and allocated costs.
Cash flow also reflects timing, inventory, working capital, and potentially taxes.
A price maximizing one of these measures need not maximize another.

A volume target, a sales forecast, a production plan, and a contractual commitment are not interchangeable.
The forecast is a prediction with uncertainty.
The plan is an operational choice informed by the prediction.
The target is an aspiration or performance benchmark.
A commitment creates an obligation and must be validated independently.

## 4. Definitions and vocabulary

The terms in this table are used with the stated scope throughout the topic.
A price and a cost must refer to the same transaction basis and currency.

| Term | Meaning and unit | Frequent misuse |
|---|---|---|
| Price, `p` | Net realized revenue per unit, USD/unit in the example | Substituting list price when discounts or returns change receipts |
| Quantity, `Q(p)` | Units demanded per defined period at price `p` | Treating observed shipments during stockouts as unconstrained demand |
| Demand curve | Relationship between a hypothetical price and quantity, conditional on stated conditions | Connecting observed points without accounting for price endogeneity |
| Own-price elasticity, `e` | `(dQ/dp)(p/Q)`, dimensionless, typically negative | Omitting the sign and reversing the interpretation |
| Absolute elasticity, `E` | `-e` when `e < 0`, dimensionless | Calling an elasticity of -2 a 2% margin |
| Cross-price elasticity | Percentage response of product `j` quantity to a 1% change in product `i` price | Assuming all substitute effects are captured by own demand |
| Marginal cost, `MC` | Incremental cost of one more unit over the decision horizon, USD/unit | Replacing it blindly with allocated full cost |
| Incremental cost | Difference in total relevant cost between two options, USD/period | Including sunk costs that do not change |
| Contribution | Revenue less stated relevant variable costs, USD/period | Claiming it is net income without fixed costs and tax |
| Gross margin | Price less a defined unit cost, divided by price | Confusing a stated accounting cost with economic marginal cost |
| Markup on cost | `(p-c)/c`, dimensionless for positive `c` | Using it as if its denominator were price |
| Lerner index | `(p-MC)/p`, dimensionless | Calling the observed index proof of optimality or unlawful conduct |
| Marginal revenue | Change in revenue from selling one additional unit, USD/unit | Treating average revenue `p` as marginal revenue under downward-sloping demand |
| Stationary point | Point at which a differentiable objective has zero first derivative | Assuming every stationary point is a maximum |
| Interior solution | Choice strictly inside the feasible price interval | Applying an equality at an active price cap or floor |
| Boundary solution | Best feasible price at a limit of the choice set | Calling it impossible because the derivative is not zero |
| Global optimum | Feasible option with highest objective over the full feasible set | Inferring global optimality from a single first-order condition |
| Local optimum | Best choice in a neighborhood of a point | Ignoring a better feasible peak elsewhere |
| Constant elasticity | Elasticity has the same value over the modeled price range | Assuming linear demand has constant elasticity |
| Linear demand | `Q(p)=a-bp` over a defined range, with `b>0` | Using it where predicted demand is negative |
| Cannibalization | Sales lost from one owned product because another owned product gains or changes price | Assuming category sales equal product sales |
| Complement | Product whose demand tends to decrease when another product's price increases, all else fixed | Treating an observed correlation as a causal cross effect |
| Substitute | Product whose demand tends to increase when another product's price increases, all else fixed | Assuming a competitor's response is fixed indefinitely |
| Capacity constraint | Limit on units or resources that can be supplied in a period | Using unconstrained demand as a committed shipment plan |
| Shadow cost | Marginal value of relaxing a binding constraint, USD per constrained unit | Adding an arbitrary capacity surcharge without optimizing the constraint |
| Price floor or cap | Lower or upper permitted price, USD/unit | Assuming the unconstrained optimum is implementable |
| Net price waterfall | Reconciliation from list price through discounts, rebates, returns, and other adjustments | Estimating elasticity on list price while optimizing net receipts |
| Endogeneity | Price is related to unobserved demand determinants or measurement error | Treating an observational correlation as an intervention effect |
| Confidence interval | Range produced by a method with stated repeated-sampling coverage properties | Calling it a guarantee that the true value lies within the interval |
| Holdout | Comparable observations not exposed to the proposed change | Calling any untreated group comparable without checking assignment |
| Contribution risk | Sensitivity of contribution to uncertain demand, cost, or response | Presenting only the best-case forecast as a commitment |

## 5. The mechanics, step by step

### Step 1 — Define the economic decision

**Input:** a proposed price decision, product, customer scope, geography, and time horizon.

Define the controllable price and the unit of sale.
A case price and an each price require consistent conversion.
A manufacturer net price and a retailer shelf price are different decisions.
If the manufacturer does not control shelf price, a shelf-price demand curve alone is not an implementable manufacturer rule.

Specify the objective before estimating the optimum.
For this derivation, maximize single-period contribution from one product.
Ignore fixed costs only when they do not vary across the feasible alternatives.
State whether returns, fulfillment, payment fees, and promotional funding are included.
If the choice affects fixed spending, include the difference directly in option-level contribution.

**Output:** one sentence such as: “Choose the realized USD per unit price for SKU A in the domestic direct channel for the next four weeks to maximize expected contribution, subject to supply and contractual limits.”

**Branch:** if the business objective is revenue, service, market entry, or portfolio value rather than contribution, do not silently substitute this objective.
Translate that objective into an explicit function or constraints, or treat the Lerner result as a diagnostic only.

### Step 2 — Build the net price and cost waterfall

**Input:** invoices, discounts, returns, payment terms, and cost records aligned to the period and customer scope.

Start with the price actually earned by the decision maker.
Subtract transaction discounts and expected rebates to the extent they vary with the choice.
Identify sales taxes or value-added taxes separately where they are collected for a tax authority rather than retained as revenue.
Use the organization's accounting and tax treatment; the analysis is not a substitute for tax advice.

For a constant-cost illustration, define `c` as relevant variable cost per unit.
For a realistic setting, calculate total relevant cost `C(Q)` and its marginal cost `C'(Q)`.
This may include materials, labor that varies within the horizon, transaction fees, freight, and expected returns.
A fee that is a percentage of price requires explicit modeling because it changes when price changes.
A cost step at a supplier tier or overtime threshold makes the objective nondifferentiable at that threshold.

Separate an accounting allocation from an incremental cash or economic cost.
An allocation can help a longer-horizon profitability review without being the cost of one additional unit today.
Conversely, a scarce resource can have an opportunity cost not visible in a standard unit-cost report.
Do not exclude a cost merely because finance labels it fixed; ask whether it differs between the actual choices.

**Output:** a traceable table of price components and relevant cost assumptions.
Record the currency, per-unit basis, effective dates, and owner of each input.

### Step 3 — Describe a causal demand response

**Input:** a demand function, identification strategy, estimation range, and uncertainty.

Let `Q(p)` represent expected demand per period at price `p`, with relevant conditions held fixed.
For a differentiated product, conditions can include competitor prices, distribution, season, advertising, and product availability.
The interpretation is the expected outcome of setting `p`, not merely the mean of records that happened to have price `p`.

Compute the derivative `Q'(p)=dQ/dp` in units per period per USD/unit.
Compute signed own-price elasticity:

`e(p) = Q'(p) * p / Q(p)`.

When `Q(p)>0` and demand slopes downward, `e(p)<0`.
Write `E(p)=-e(p)>0` for the absolute value.
A finite difference or midpoint elasticity can approximate local response when a differentiable curve is not available, but the approximation becomes less reliable for a large move or a kink.

Use the estimation evidence from topic 03 to judge whether a causal interpretation is defensible.
A price discount targeted to weak-demand locations can make an effective discount look ineffective.
A price increase taken during an unexpected demand surge can make a harmful increase appear beneficial.
Stockouts censor demand and can create a false impression of inelasticity.

**Output:** a demand function or a clearly defined local slope and an evidence range.
State what was held fixed and whether uncertainty reflects parameter estimation, experimental sampling, or scenario judgment.

**Branch:** if price assignment is not plausibly identified, treat the curve as a scenario rather than as established causal evidence.
Use a bounded test or additional evidence rather than reporting a false precision optimum.

### Step 4 — Write contribution as a function of price

**Input:** demand and relevant cost on the same time and unit basis.

With a constant marginal cost `c` and fixed costs `F` that do not change with price:

`pi(p) = p Q(p) - c Q(p) - F`.

Equivalently:

`pi(p) = (p-c) Q(p) - F`.

For nonlinear costs, write `pi(p)=p Q(p)-C(Q(p))-F` instead.
If `C` already includes the fixed costs in `F`, do not subtract them twice.
At this stage, `Q` is expected demand; if actual fulfilled units are constrained or stochastic, use a fulfillment model rather than assuming every demanded unit ships.

**Output:** one objective with all price-dependent economics visible.

### Step 5 — Differentiate and interpret the terms

Differentiate the constant-cost objective using the product rule:

`d pi/dp = Q(p) + (p-c) Q'(p)`.

The first term is the direct revenue gain on units retained when price rises by a small amount.
The second term is the contribution change from the induced quantity change.
Under downward-sloping demand, `Q'(p)` is negative, so the second term is normally a loss.
The units of the derivative are contribution per period per USD/unit of price.

At a differentiable interior optimum `p*`:

`0 = Q(p*) + (p*-c) Q'(p*)`.

Divide by positive `Q(p*)` and use `e(p*)=Q'(p*) p*/Q(p*)`:

`0 = 1 + ((p*-c)/p*) e(p*)`.

Therefore:

`(p*-c)/p* = -1/e(p*) = 1/E(p*)`.

The left-hand side is the Lerner index.
The right-hand side is the inverse absolute own-price elasticity evaluated at the optimum.
The sign matters: `-1/e` is positive when demand slopes downward.

For differentiable `C(Q)`, the same algebra replaces `c` by `C'(Q(p*))`.
That marginal cost is evaluated at the output induced by the candidate price.
If marginal cost depends on quantity, it too may change as price moves.

**Output:** a necessary first-order condition with all assumptions named.

### Step 6 — Connect the rule to marginal revenue

When the inverse demand function `p(Q)` exists locally, revenue is `R(Q)=p(Q)Q`.
Differentiating with respect to quantity gives:

`MR = p + Q dp/dQ`.

Because `e=(dQ/dp)(p/Q)`, the reciprocal relationship gives `Q dp/dQ = p/e` where the inverse exists.
Hence `MR = p(1+1/e)`.
At an interior contribution maximum, `MR=MC`.
Rearranging yields `(p-MC)/p=-1/e`.

This derivation says the same thing as the price derivative.
A downward-sloping curve means the incremental unit may require a lower price on other units, so marginal revenue is below price.
The relevant comparison is marginal revenue versus marginal cost, not price versus cost alone.

### Step 7 — Rearrange cautiously

If `E` is a known constant greater than 1 and `c>0`, the algebra gives:

`p* = c / (1 - 1/E) = c E/(E-1)`.

This expression also holds as a fixed-point relationship when `E=E(p*)` is evaluated at the solution.
It is not generally valid to insert `E(p_current)` and assume it remains unchanged at the resulting new price.

For `E=2`, the implied price-cost margin is 50% of price, and the implied markup is 100% of cost.
For `E=4`, the margin is 25% of price and the markup is one-third of cost.
With positive marginal cost, the ordinary interior solution must lie on the elastic part of the demand curve, where `E>1`.
This conclusion follows from `0 < (p-c)/p < 1` when `p>c>0`.

If `E=1`, the rearranged denominator is zero.
If `E<1`, the formula yields a nonpositive or economically inappropriate price with positive cost.
Do not repair that result with a sign flip.
It indicates that the assumed positive-cost interior solution is not supported at that local elasticity; inspect the actual curve, boundaries, and constraints.

### Branch — A payment fee that varies with price

A percentage fee cannot be folded into a constant USD/unit cost at every candidate price.
Suppose the decision price `p` is the amount paid by the customer and the payment provider retains fraction `f` of that amount.
With other relevant unit costs `c`, contribution is `((1-f)p-c)Q(p)`.
This treatment assumes the fee is incurred on every fulfilled unit, is not refunded, and does not change with volume.
Change the model if rebates, returns, taxes, or fee tiers alter that basis.

Differentiating gives `(1-f)Q(p)+((1-f)p-c)Q'(p)`.
At a smooth interior stationary point, the corresponding condition is:

`(((1-f)p-c)/((1-f)p)) = -1/e(p)`.

The denominator is the amount retained after the proportional fee, not the customer price.
Equivalently, use the retained price as the optimization variable and transform the demand curve consistently.
Do not subtract the fee from price and also add it to unit cost.

For the Northstar curve in section 6, suppose the USD 10.40/unit cost excludes an additional 2% payment fee.
The fee-adjusted objective is `((0.98p)-10.40)(24,000-800p)`.
Its stationary price is `(0.98*24,000+800*10.40)/(2*0.98*800) = USD 20.306122/unit`.
Forecast demand is approximately 7,755.10 units per four weeks and the retained unit contribution is USD 9.50/unit.
Forecast contribution is approximately USD 73,673.47 per four weeks before unchanged fixed costs.
At the original USD 20.20/unit candidate, fee-adjusted contribution is USD 73,664.64 per four weeks.
The difference is approximately USD 8.83 per four weeks before rounding or implementation costs.
This illustration changes the cost assumption from the base case; do not add the fee to that case without first confirming it was excluded.
The exact candidate is within the example's USD 17–22/unit evidence range, but a commercial price must also respect permitted price increments and other constraints.

### Step 8 — Solve for a candidate with a real curve

For linear demand `Q(p)=a-bp`, with `a>0` and `b>0`, elasticity varies with price:

`e(p) = -bp/(a-bp)`.

The contribution derivative is `a-bp-b(p-c)`.
Setting it to zero gives `p*=(a+bc)/(2b)`, subject to feasibility.
The second derivative is `-2b`, so contribution is strictly concave for this model.
If the candidate lies within the valid demand and price range, it is the unique unconstrained maximum within that range.

At a different price the linear demand elasticity differs.
Thus a formula fed with a baseline elasticity is a diagnostic at that baseline, not a valid shortcut to the linear model's optimum.
Evaluate the elasticity again at the candidate.

For an estimated nonlinear curve, solve the derivative equation numerically if it is smooth.
Search all relevant stationary points, any kinks, and the endpoints of the feasible interval.
Compare contribution directly, because the first-order condition alone does not establish the global maximum.
If the demand model is valid only over an observed range, constrain the optimization to that range unless a separately justified extrapolation is approved.

### Step 9 — Incorporate operational constraints

A price floor, cap, or contractual band changes the feasible set.
If the unconstrained maximum is outside the band, compare feasible endpoints and any interior candidates.
A boundary optimum need not satisfy the Lerner equality.

If production capacity is scarce, determine whether demand can be fulfilled and whether the firm must use price to ration demand.
A binding capacity constraint can make the opportunity cost of capacity relevant.
If the firm has several products sharing capacity, optimizing each one separately can allocate scarce capacity badly.
For stochastic demand or backorders, model service, lost sales, and capacity explicitly; a deterministic cap on demand is not the only possible formulation.

A price that would imply a quantity above available capacity is not a credible shipment commitment.
The firm may maintain the price and ration units, increase capacity, reallocate capacity, or change price, depending on contracts and the objective.
State which of these actions the model permits.

### Step 10 — Include other products when necessary

Let `p_i`, `q_i`, and `c_i` denote product `i` price, demand, and marginal cost.
With a portfolio of `n` owned products and constant unit costs:

`Pi(p) = sum_j (p_j-c_j) q_j(p_1,...,p_n)`.

Holding other prices fixed, the derivative for product `i` is:

`dPi/dp_i = q_i + sum_j (p_j-c_j) (d q_j/dp_i)`.

The `j=i` term is the familiar own-demand effect.
The `j!=i` terms are contribution changes on the other products.
A substitute can recover some units lost by the focal product; a complement can lose units along with it.
A single-product margin rule omits both effects.

Define cross elasticity `e_ji=(d q_j/dp_i)(p_i/q_j)` where the quantities and prices are positive.
If product `j` is an owned substitute, `e_ji` is often positive under a causal interpretation.
If it is a complement, `e_ji` is often negative.
The signs and magnitudes are empirical questions, not labels that can be assigned from the product catalog alone.

At a joint interior optimum all price derivatives must be zero simultaneously.
A one-product derivative holding other prices fixed is not a complete joint optimization.
A portfolio model also needs feasible price bounds, stable estimated cross effects, and any relevant supply constraints.
Cross-price effects involving competitor products can change focal demand; their profit is not the firm's profit unless a legitimate economic relationship makes it relevant.
Do not coordinate or exchange competitively sensitive intentions with competitors.

### Step 11 — Compare uncertainty rather than only point optima

Construct demand and cost scenarios that preserve their relationships.
For example, an estimated intercept and slope may be correlated; independently choosing the most favorable value of each can create an implausible curve.
Where the estimator supplies a joint parameter distribution, draw parameters jointly or use a suitable bootstrap.
If only a range is credible, describe it as a scenario range, not a confidence interval.

For each candidate price, report expected units, contribution, capacity use, and downside under each scenario.
Compare regret: the difference between the scenario's best feasible contribution and the contribution from the chosen action.
A price near a flat peak may be preferable to a volatile point optimum if it is easier to implement and produces little expected loss.
This is a decision criterion to agree in advance, not a mathematical consequence of the Lerner equality.

**Output:** an option comparison, not a single unexplained number.

### Step 12 — Approve, implement, and learn

Document the candidate, the feasible alternatives, the baseline, and the expected effect in both units and currency.
State the scope of the estimate and the prices at which it is supported.
Identify a monitoring owner and a rollback authority before deployment.

Implementation checks include price configuration, customer communication, channel-specific terms, and effective dates.
Use an eligible holdout or randomized test when possible and lawful.
Predefine the primary outcome, duration, minimum data needs, and rules for early stopping.
Monitor demand, contribution, service, returns, customer complaints, and unintended distribution changes.
A rollback trigger should use measured deterioration relative to a comparable control or a prespecified safety threshold, not a single noisy day's sales.

The final output is a decision with evidence and guardrails.
It is not a claim that an algebraic optimum is a mandate.

## 6. Worked example

### Situation and scope

Northstar Components is a fictional supplier of a consumable industrial filter.
The commercial team controls the realized direct-channel net price per filter.
The decision period is four weeks.
The company wants to maximize the expected contribution of this one filter, subject to a feasible price range and supply availability.
The example assumes competitors' prices, marketing, and distribution remain fixed.

The current realized price is USD 18.00/unit.
Relevant variable cost is assumed constant at USD 10.40/unit over the considered range.
The cost includes variable acquisition, handling, and fulfillment costs.
No fixed cost changes among the alternatives in the base case.
Demand is assumed to be:

`Q(p) = 24,000 - 800 p` filters per four weeks.

Here `p` is USD/unit.
The slope is -800 filters per four weeks per USD/unit.
The model is assumed valid for prices from USD 17.00/unit through USD 22.00/unit.
The model's causal validity is an assumption for teaching; no real estimation dataset is claimed.

At USD 18.00/unit, forecast demand is:

`24,000 - 800 * 18 = 9,600` units per four weeks.

Unit contribution is USD 7.60/unit.
Forecast period contribution is:

`(18.00 - 10.40) * 9,600 = USD 72,960` per four weeks.

The 9,600 units are a model forecast, not a production plan or a customer commitment.
The operations team has not yet accepted a supply plan.
A commercial target may differ from the forecast and needs its own justification.

### Diagnose the current price

At USD 18.00/unit, signed elasticity is:

`e(18) = -800 * 18 / 9,600 = -1.5`.

Absolute elasticity is 1.5.
The current price-cost margin is:

`(18.00 - 10.40) / 18.00 = 0.4222`, or 42.22% of price.

The inverse absolute elasticity at the current price is `1/1.5=0.6667`, or 66.67%.
These quantities are unequal, so this price does not satisfy the interior first-order condition.
The derivative gives the direction under the assumed curve:

`d pi/dp = 9,600 + 7.60 * (-800) = 3,520`.

Its unit is USD per four weeks per USD/unit change in price.
It is positive, so a sufficiently small increase improves modeled contribution at USD 18.00/unit.
It does not by itself quantify a large price change accurately.

If one incorrectly treats the baseline elasticity as constant and inserts `E=1.5`, the shortcut produces USD 31.20/unit.
That value is outside the evidence range and is not the linear-demand optimum.
It is a useful warning: the elasticity of this curve changes when price changes.

### Find the linear-demand optimum

Substituting `a=24,000`, `b=800`, and `c=10.40` gives:

`p* = (24,000 + 800 * 10.40) / (2 * 800)`.

The numerator is 32,320.
The denominator is 1,600.
The candidate is USD 20.20/unit.
It lies inside the assumed USD 17.00–22.00/unit valid range.

Forecast demand at the candidate is:

`Q(20.20) = 24,000 - 800 * 20.20 = 7,840` units per four weeks.

Unit contribution is USD 9.80/unit.
Expected four-week contribution is:

`7,840 * 9.80 = USD 76,832`.

The contribution increase relative to the baseline is USD 3,872 per four weeks.
The forecast volume decrease is 1,760 units per four weeks.
The modeled price increase is USD 2.20/unit, or approximately 12.22% of the original price.
This is a substantial change and depends on the full demand curve, not a local percentage approximation.

The own-price elasticity at USD 20.20/unit is:

`e(20.20) = -800 * 20.20 / 7,840 = -2.061224...`.

The margin as a share of price is:

`9.80 / 20.20 = 0.4851485...`.

The inverse absolute elasticity is:

`1 / 2.061224... = 0.4851485...`.

The equality holds apart from rounding.
The second derivative is -1,600 in the corresponding contribution-per-price-squared units.
It is negative everywhere for this linear model, establishing a unique maximum within the valid unconstrained interval.

### Compare practical prices

| Net price, USD/unit | Forecast units/four weeks | Unit contribution, USD/unit | Forecast contribution, USD/four weeks |
|---:|---:|---:|---:|
| 18.00 | 9,600 | 7.60 | 72,960 |
| 19.00 | 8,800 | 8.60 | 75,680 |
| 20.00 | 8,000 | 9.60 | 76,800 |
| 20.20 | 7,840 | 9.80 | 76,832 |
| 20.50 | 7,600 | 10.10 | 76,760 |
| 21.00 | 7,200 | 10.60 | 76,320 |
| 22.00 | 6,400 | 11.60 | 74,240 |

The peak is flat near USD 20.20/unit.
USD 20.00/unit sacrifices only USD 32 per four weeks in this deterministic model.
A firm may reasonably prefer USD 20.00/unit if its implementation or customer communication is materially easier, provided constraints and objectives are met.
The table does not establish that USD 20.00/unit is always the best choice under uncertainty.

### Add a supply restriction

Suppose only 7,600 units can be supplied in the next four weeks.
For this illustration, management requires a deterministic price that brings forecast demand no higher than capacity; it does not permit rationing, overtime, or backlog.
The feasibility condition is:

`24,000 - 800p <= 7,600`.

Therefore `p >= USD 20.50/unit`.
Because the unconstrained contribution curve is decreasing above USD 20.20/unit, the best feasible price within USD 17.00–22.00/unit is USD 20.50/unit.
Forecast demand equals the 7,600-unit capacity.
Forecast contribution is USD 76,760 per four weeks.

At this boundary the ordinary Lerner equality need not hold.
The supply requirement, not the unconstrained marginal balance, determines the chosen price.
In reality, demand is uncertain, so matching mean forecast demand to capacity would not guarantee the service level.
A service-constrained or stochastic model would be needed to assess shortage risk.

The supply plan is a separate decision: reserve no more than the approved quantity, assess variability, and confirm materials and labor.
Do not describe 7,600 units as orders already received.
If the business must honor an existing contract at USD 18.00/unit, this price-only exercise may not apply to that volume.

### Add a portfolio diagnostic

Suppose the filter has an owned premium substitute, product B.
At the focal price USD 20.20/unit, B's contribution is USD 5.00/unit.
Assume raising the focal price by USD 1.00/unit increases B demand by 120 units per four weeks, holding B's price and other factors fixed.
The resulting first-order portfolio contribution effect is:

`5.00 * 120 = USD 600` per four weeks per USD/unit of focal price.

The focal product's own derivative is zero at its standalone optimum.
The combined derivative at that same price is therefore positive USD 600 per four weeks per USD/unit, if no other cross effects occur.
A small focal price increase would improve modeled portfolio contribution under these assumptions.
The exact new optimum requires the full portfolio demand functions, costs, constraints, and possible changes to B's price.

If B were instead an owned complement and its demand fell by 120 units per four weeks per USD/unit, the cross contribution term would be negative USD 600 per four weeks per USD/unit.
The standalone optimum would then be too high locally for the modeled portfolio objective, other things equal.
These are illustrative derivatives, not evidence about any actual company or market.

### Recommendation and handoff

For the unconstrained, single-product base case, present USD 20.20/unit as the model's candidate and USD 20.00/unit as a nearby practical alternative.
Do not claim either is approved or experimentally validated.
For the specified 7,600-unit no-rationing capacity constraint, present USD 20.50/unit as the deterministic constrained candidate.
Before implementation, validate demand identification, capacity uncertainty, contracts, net price realization, and portfolio effects.
If these checks fail, retain the current price or run an appropriately controlled test rather than treating the mathematical answer as a commitment.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It reproduces the linear-demand case and compares a grid of feasible prices.
It also demonstrates how the deterministic capacity requirement changes the candidate.
The grid is a transparent comparison; the analytical optimum remains the exact result for the stated model.

```python
from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True)
class LinearDemand:
    intercept: float       # units per four weeks
    slope: float           # units per four weeks per USD/unit
    unit_cost: float       # USD/unit

    def quantity(self, price: float) -> float:
        return self.intercept - self.slope * price

    def contribution(self, price: float) -> float:
        return (price - self.unit_cost) * self.quantity(price)

    def elasticity(self, price: float) -> float:
        units = self.quantity(price)
        if price <= 0 or units <= 0:
            raise ValueError("Elasticity requires positive price and demand")
        return -self.slope * price / units

    def price_derivative(self, price: float) -> float:
        return self.quantity(price) - self.slope * (price - self.unit_cost)

    def stationary_price(self) -> float:
        return (self.intercept + self.slope * self.unit_cost) / (2 * self.slope)


def lerner_margin(price: float, cost: float) -> float:
    return (price - cost) / price


def best_on_grid(model, prices, capacity=None):
    feasible = [
        price for price in prices
        if model.quantity(price) >= 0
        and (capacity is None or model.quantity(price) <= capacity)
    ]
    if not feasible:
        raise ValueError("No feasible price in the grid")
    return max(feasible, key=model.contribution)


def main():
    model = LinearDemand(intercept=24_000, slope=800, unit_cost=10.40)
    low, high = 17.00, 22.00
    optimum = model.stationary_price()
    if not low <= optimum <= high:
        raise ValueError("Stationary price lies outside modeled range")

    elasticity = model.elasticity(optimum)
    margin = lerner_margin(optimum, model.unit_cost)
    assert isclose(margin, -1 / elasticity, rel_tol=1e-12)
    assert isclose(model.price_derivative(optimum), 0.0, abs_tol=1e-9)

    print("price  units  contribution")
    for price in (18.00, 19.00, 20.00, 20.20, 20.50, 21.00, 22.00):
        print(f"{price:5.2f}  {model.quantity(price):5.0f}  "
              f"{model.contribution(price):12.2f}")

    print(f"stationary_price={optimum:.2f}")
    print(f"elasticity={elasticity:.6f}")
    print(f"lerner_margin={margin:.6f}")

    # A one-cent grid includes the endpoints and the analytical solution.
    prices = [cents / 100 for cents in range(1700, 2201)]
    unconstrained = best_on_grid(model, prices)
    capacity_price = best_on_grid(model, prices, capacity=7_600)
    print(f"grid_best={unconstrained:.2f}")
    print(f"capacity_best={capacity_price:.2f}")
    print(f"capacity_units={model.quantity(capacity_price):.0f}")
    print(f"capacity_contribution={model.contribution(capacity_price):.2f}")


if __name__ == "__main__":
    main()
```

Expected output, with columns measured in USD/unit, units/four weeks, and USD/four weeks:

```text
price  units  contribution
18.00   9600      72960.00
19.00   8800      75680.00
20.00   8000      76800.00
20.20   7840      76832.00
20.50   7600      76760.00
21.00   7200      76320.00
22.00   6400      74240.00
stationary_price=20.20
elasticity=-2.061224
lerner_margin=0.485149
grid_best=20.20
capacity_best=20.50
capacity_units=7600
capacity_contribution=76760.00
```

The assertions verify algebraic consistency for these inputs; they do not validate the demand curve.
The grid has a one-cent increment and is not a general continuous optimizer.
The capacity calculation assumes deterministic demand must not exceed capacity, with no rationing or backorders.
A production model would need evidence for the demand coefficients and would include feasible contractual and operational choices.
The script does not estimate elasticity or provide uncertainty intervals.
Its outputs are forecasts conditional on assumptions, not plans or commitments.

### Manual audit trail

1. Confirm the coefficient units match the four-week horizon.
2. Check positive demand across the modeled interval.
3. Calculate the stationary price from the specified linear curve.
4. Verify that the candidate is within the interval.
5. Compare its contribution with the endpoints and commercially practical alternatives.
6. Recalculate elasticity at the candidate, not just at the baseline.
7. Verify that the capacity branch is permitted by the business decision.
8. Review contracts and data quality before using the result commercially.

## 8. Data, standards, and regulatory context

### Minimum data contract

A price recommendation is only as good as the transaction and demand records behind it.
Define a row grain before combining datasets.
A useful grain might be product, customer segment, channel, geography, and week.
At that grain, join price, quantity, discounts, availability, cost, and relevant external conditions using effective dates.
Do not aggregate away price variation before checking whether prices and quantities represent the same transactions.

| Field | Required definition | Key control |
|---|---|---|
| Product identifier | Item and pack at the price decision grain | Maintain mappings when pack sizes change |
| Period | Start, end, timezone, and sales or shipment basis | Align price effective dates with orders and returns |
| Channel and geography | Scope that owns or observes the price | Avoid mixing wholesale and retail prices |
| Net realized price | Revenue retained per unit after defined adjustments | Reconcile to finance and trace the waterfall |
| Units | Ordered, shipped, returned, or consumed units | Separate stockout-censored sales from demand |
| Promotions | Mechanics, duration, funding, eligibility | Separate regular-price and promotion effects |
| Availability | On-hand, stockout, distribution and fulfillment status | Flag periods when demand could not be observed |
| Cost | Marginal cost assumptions and effective dates | Distinguish incremental from allocated cost |
| Other prices | Related product and competitor price scope | Document provenance and lawful collection |
| Exposure | Customers or locations eligible to see a price | Exclude observations not actually exposed |
| Treatment assignment | Experimental assignment or observational selection | Preserve assignment and deviations separately |
| Returns and claims | Time-lagged units and financial adjustments | Avoid attributing returns to the wrong price period |

A sales transaction file alone may lack unfilled demand and exposure.
A shipment file may reflect replenishment rather than end-customer consumption.
A list-price file may not capture rebates negotiated after the invoice.
These differences change the meaning of elasticity and therefore the recommendation.

### Product and transaction standards

GS1 Global Trade Item Numbers (GTINs) identify trade items in a standardized way when properly assigned.
They can support matching a product and its pack configuration across organizations.
A GTIN does not establish that two rows have the same realized price, customer exposure, or economic cost.
The GS1 General Specifications describe identification and barcode standards; they do not prescribe an optimal price or elasticity estimator.
Use the edition and implementation rules applicable to the trading partners and products in scope.

An electronic data interchange (EDI) invoice or purchase order can carry transaction information using a trading-partner implementation guide.
The existence of a standard message does not ensure that rebates, returns, or delayed adjustments are complete.
Reconcile the transmitted transaction to the decision maker's net receipts.
A partner's specific message requirements may be contractual rather than a universal statutory requirement.

### Competition and consumer protection

U.S. competition law and European Union competition law constrain conduct, not the Lerner equation itself.
The U.S. Federal Trade Commission (FTC) guidance on price fixing explains that agreements among competitors to fix prices are generally illegal.
Article 101 of the Treaty on the Functioning of the European Union prohibits certain agreements that restrict competition, subject to its terms and applicable exemptions.
Consult current law and qualified counsel for the jurisdictions, business relationships, and facts involved.
This educational explanation is not a legal opinion.

Use internal data and lawfully obtained public information to form independent decisions.
Do not seek a competitor's future intended prices, negotiate a common price, or use a third party to coordinate price changes.
A high observed price-cost margin is not by itself proof of market power, illegality, or optimality.
An estimated elasticity may reflect product differentiation, data limitations, costs, or strategic conditions.
The legal assessment is separate from the business model.

Published prices, advertised discounts, and customer-specific offers can also be subject to consumer-protection, contract, privacy, and sector-specific rules.
For example, the FTC's Guides Against Deceptive Pricing address certain comparisons to former and competing prices in U.S. advertising.
They do not create a general numerical Lerner margin requirement.
Where offers use personal data, determine the applicable privacy obligations and access controls before deployment.
Legal requirements depend on jurisdiction, the transaction, and current rules.

### Evidence retention and governance

Record the source extracts, versions, estimation code, analysis date, and approval decision.
Restrict access to identifiable customer and transaction data to authorized users.
Retain enough information to reconstruct the recommendation, subject to the organization's retention schedule and applicable law.
Separate a reproducible model artifact from confidential deal-level exports.

Document model validation independently from business approval.
A validator can challenge data leakage, confounding, and out-of-range extrapolation.
A commercial approver can decide whether the expected contribution and risks meet the business objective.
Neither role can infer the other's approval from a completed calculation.

## 9. Industry practice and case evidence

### Evidence boundaries

The fictional Northstar example is a teaching case, not an industry result.
The sources below establish that demand estimation and pricing optimization have been studied and applied, but they do not validate Northstar's numerical assumptions.
Do not transfer a published elasticity or optimal price to another product without checking the customer, channel, period, and identification method.

### Supermarket demand and price response

Dominick's Finer Foods scanner data have been used in published research on retail pricing and demand.
The University of Chicago Booth School of Business provides documentation for the Dominick's dataset.
The dataset contains retail scanner and pricing information from the chain and has supported empirical work on consumer response.
The availability of detailed store-level observations does not, by itself, make price changes randomly assigned.
Promotions, local competition, and inventory conditions can still influence both prices and observed sales.

**Inference:** A practitioner using such data for an own-price rule should first reconstruct realized prices and availability and then defend identification.
That inference is a methodological recommendation, not a claim that the dataset establishes a particular optimal price.
The relevant price level is retail, so a manufacturer would also need to model how its own decisions translate to retail prices.

### Empirical differentiated-product demand

Berry, Levinsohn, and Pakes study differentiated-product demand in the automobile market in their 1995 paper.
Their framework addresses demand estimation when product characteristics and unobserved product attributes affect demand and prices.
The application demonstrates why substitution across products and price endogeneity matter in a real market.
It does not show that a single-product Lerner formula alone determines an implementable price for every automobile.

**Inference:** A portfolio pricing team should consider substitution and identification together rather than treating cross-price effects as an optional cosmetic adjustment.
The specific model, instruments, and market definitions need to be evaluated for the decision at hand.

### Retail promotion evidence

Hendel and Nevo study consumer response to storable goods and sales in published empirical work.
Their analysis highlights that purchase timing and inventory behavior can affect observed responses to price promotions.
A temporary sale may shift purchases across time rather than create the same amount of incremental consumption.

**Inference:** A permanent regular-price change should not automatically inherit a short promotional elasticity.
The optimization horizon must include the relevant post-promotion period when pull-forward or stockpiling matters.
A single-period contribution model is a useful benchmark, not a complete promotion model.

### Revenue management practice

Talluri and van Ryzin's revenue-management text describes pricing and capacity allocation as linked decisions in industries with constrained inventory and time-sensitive demand.
That literature provides a broader setting than a one-period static elasticity rule.
The constrained price in the Northstar example is an elementary illustration, not a reproduction of a named company's proprietary policy.

**Inference:** When a production bottleneck or expiring inventory dominates the economics, the team should compare an explicit capacity or dynamic model with the static benchmark.
The Lerner index can still help explain local price-cost economics, but it cannot stand in for the missing state variables.

### What a commercial review should demand

Ask whether the source estimates a price response for the same decision maker and customer exposure.
Ask whether the response is short-run, long-run, promotional, or regular-price.
Ask whether the cited evidence measures category expansion, substitution, or purchase timing.
Ask whether the paper's empirical conditions resemble the decision's constraints.
Ask which assumptions are model assumptions and which are observed facts.

A named public dataset or paper strengthens the explanation of a method.
It does not supply an evidence-free estimate for a new product.
Document transfer limitations alongside the candidate price.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong sign | Positive recommended margin computed from `1/e` with negative `e` | Signed and absolute elasticity confused | Invalid price or reversed interpretation | Write `-1/e=1/E` and state signs |
| Margin versus markup | Spreadsheet divides by cost in one place and price in another | Different denominators | Materially wrong target price | Label both measures and reconcile algebra |
| Baseline elasticity reused | A distant candidate is computed from one local estimate | Elasticity changes with price | Candidate may be outside the evidence range | Solve the demand curve and recalculate at candidate |
| Unidentified response | High prices coincide with strong demand periods | Demand shocks influence price | Correlation mistaken for intervention effect | Use a credible design, instruments, or controlled test |
| Stockout censoring | Sales plateau when inventory hits zero | Sales understate latent demand | Elasticity and forecast are biased | Model availability and lost sales |
| Wrong price basis | Model uses shelf price but decision controls wholesale net price | Decision and response refer to different levels | Nonimplementable recommendation | Model pass-through and align price waterfall |
| Average cost substitution | Fully allocated fixed overhead dominates unit cost | Nonincremental expenses treated as marginal | Wrong contribution comparison | Reconcile incremental costs by option and horizon |
| Fee omitted | Percentage payment fee increases with price | Cost changes with the decision | Overstated gain from increase | Include fee as a price-dependent term |
| Local maximum treated as global | Multiple peaks or kinks ignored | First-order condition is only local | Better feasible price is missed | Compare stationary points, boundaries, and kinks |
| Invalid extrapolation | Candidate outside observed price range | Demand shape is assumed without support | Forecast and margin are unreliable | Constrain range or gather new evidence |
| Product silo | Adjacent owned product sales shift | Cross contribution omitted | Portfolio contribution falls | Estimate cross effects and optimize the portfolio |
| Capacity ignored | Forecast units exceed feasible supply | Price choice and fulfillment decoupled | Shortages, penalties, or lost sales | Model capacity and operational alternatives |
| Uncertain demand treated as certain | Capacity equals mean forecast with no buffer | Forecast error causes overload | Service level deteriorates | Model variability and service constraints |
| Competitor reaction ignored | Response persists beyond the analysis horizon | Other firms change their own actions | Forecast fails after launch | Scenario-test independent competitor responses |
| Time effects ignored | Promotion spike reverses after the offer | Pull-forward or stockpiling | Incrementality overstated | Extend the horizon and measure post-period |
| Contract conflict | Candidate violates negotiated price or notice period | Mathematical feasible set is incomplete | Dispute or failed implementation | Obtain contract review before approval |
| Misleading price claim | Advertised discount lacks a valid comparison basis | Marketing communication is separated from transaction evidence | Consumer-protection risk | Review claims against applicable rules and records |
| Single noisy result | Daily sales trigger immediate reversal | Seasonality and noise dominate | Oscillation and biased learning | Predefine control, monitoring window, and stop rules |
| Sensitivity suppressed | Only a favorable point estimate is shown | Parameter uncertainty hidden | False precision in approval | Present scenario contributions and regret |
| Privacy ignored | Customer-level model is shared broadly | Access and purpose are not controlled | Confidentiality or compliance incident | Minimize data and follow applicable governance |

### Triage order

First stop a recommendation that violates a binding constraint or lacks authorization.
Next challenge the price, quantity, and cost definitions.
Then challenge identification, range, and uncertainty.
Finally compare cross-product and operational effects.
This ordering prevents precision work on a decision that cannot legally or operationally be executed.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Practical test or response |
|---|---|---|---|
| A well-defined decision price | The model needs a controllable intervention | Different customers receive different effective prices | Reconcile list-to-net waterfall and contractual scope |
| Positive price and demand | Elasticity divides by price and quantity | Zero sales or free distribution | Exclude undefined points or use an appropriate discrete model |
| Differentiable demand | The derivative and first-order condition must exist locally | Price thresholds, discrete packages, or kinks | Compare discrete options and one-sided changes |
| Interior feasible optimum | Equality assumes no active price or operational boundary | Contract floor, cap, or capacity restriction | Compare feasible endpoints and constrained candidates |
| Causal demand response | The predicted change must correspond to setting price | Confounding, selection, stockouts, reverse causality | Audit design and use credible experiments or instruments |
| Demand stable over horizon | The fitted curve must apply during implementation | New entrants, season shifts, supply shocks | Backtest by period and scenario-test plausible changes |
| Other conditions fixed | The partial derivative holds other inputs constant | Advertising, assortment, or competitor prices move | Model these effects or use explicitly conditional scenarios |
| Correct marginal economics | The lost unit has the modeled incremental cost | Cost tiers, capacity shadow cost, returns, price-based fees | Rebuild option-level costs and evaluate breakpoints |
| Single-product objective | The basic rule omits other product contributions | Substitutes, complements, shared costs | Estimate cross effects and use a portfolio objective |
| Local condition identifies maximum | A stationary point is not necessarily best | Nonconcavity, multiple peaks, discontinuities | Inspect curvature and compare all feasible candidates |
| Known demand curve over range | Optimization relies on prices beyond one observed point | Candidate leaves the estimation support | Constrain the range or gather additional evidence |
| No dynamic consequences | One period contains all relevant economics | Stockpiling, retention, reference price, future capacity | Expand the horizon and model state transitions |
| Competitor prices fixed | Own derivative is conditional on other prices | Competitors independently respond after the move | Evaluate independent reaction scenarios without coordination |
| Deterministic fulfillment | Demand and supply are known in the simple capacity example | Forecast error, lead-time variability, allocation | Use stochastic service and inventory models |
| Risk-neutral expected contribution | Maximizing the expectation matches the stated goal | Downside limits, customer risk, cash constraints | Agree on risk measures and scenario constraints |
| Fixed costs unchanged | Dropping `F` does not change ranking | A price choice requires a new campaign or capacity | Include incremental fixed expenditures by option |

These are conditions for interpretation, not a checklist that can be signed without evidence.
Some assumptions can be relaxed by changing the model.
Others require a different decision process or additional data.
A mathematically correct result remains conditional on the model and its inputs.

### When not to use the closed form

Do not use the constant-elasticity closed form with an elasticity at or below one and positive marginal cost as though it gave a normal positive interior optimum.
Do not use it when the price point is outside the demand model's range.
Do not use it as a substitute for a discrete menu comparison where only a few prices are permitted.
Do not use it to declare a legal violation or a firm's market power.
Do not use it alone for dynamic markdowns, auctions, two-sided markets, negotiated nonlinear tariffs, or shared-capacity portfolios.
Use a model that represents the actual decision and explain what the simple benchmark omits.

## 12. Exercises

Use USD and the stated period throughout.
Unless stated otherwise, assume a single product, deterministic demand, constant marginal cost, and a feasible interior price.
Show the arithmetic and the interpretation, not merely the final number.

### Exercise 1 — Vocabulary and units

**Time:** 12 minutes.

A product has price USD 40/unit, marginal cost USD 24/unit, and own-price elasticity -2.5 at that price.
Calculate the price-cost margin and markup on cost.
Calculate the inverse absolute elasticity.
State whether the Lerner equality holds at the given price.

**Deliverable:** three numerical measures and two sentences explaining their denominators and interpretation.

**Acceptance criteria:** distinguish 40% of price from 66.67% of cost; retain the negative elasticity sign until converting to absolute elasticity; avoid claiming that equality alone proves a global optimum.

### Exercise 2 — Derivation

**Time:** 15 minutes.

Starting with `pi(p)=(p-c)Q(p)-F`, derive the Lerner equality for an interior stationary point.
Identify the direct price effect and the induced volume effect.
Explain why fixed cost `F` disappears from the derivative under the assumption that it is unchanged by price.

**Deliverable:** a short derivation with the signed elasticity explicitly defined.

**Acceptance criteria:** include the product rule, the zero derivative, and the substitution leading to `-1/e`.
State at least two conditions that could make the equality inappropriate.

### Exercise 3 — Local versus constant elasticity

**Time:** 18 minutes.

Suppose a constant-elasticity demand curve has `E=4` over a valid range and marginal cost is USD 18/unit.
Compute the interior candidate price and price-cost margin.
Then explain why the same calculation is not justified merely because elasticity at the current price is -4.

**Deliverable:** candidate price, margin, and a two-sentence caveat.

**Acceptance criteria:** report USD 24/unit and 25%; distinguish a constant elasticity model from a local point estimate.

### Exercise 4 — Linear demand and a price ceiling

**Time:** 25 minutes.

Demand per week is `Q(p)=1,000-20p` units and marginal cost is USD 10/unit.
The allowed price range is USD 10–28/unit.
Find the unconstrained stationary price, the best feasible price, and its weekly contribution.
Compare the feasible answer with the contribution at USD 25/unit.

**Deliverable:** arithmetic, constraint interpretation, and a short decision statement.

**Acceptance criteria:** show USD 30/unit as the unconstrained stationary point and USD 28/unit as the constrained choice; show 440 units and USD 7,920/week at the constrained choice; explain why the interior equality need not hold there.

### Exercise 5 — Cross-price contribution

**Time:** 20 minutes.

At the standalone optimum for product A, the derivative of A contribution with respect to A's price is zero.
Product B is an owned substitute with unit contribution USD 6/unit.
A USD 1/unit increase in A's price increases B demand by 40 units/week, holding B's price fixed.
Compute the cross contribution derivative and state the local direction for portfolio contribution.
Repeat the sign interpretation if B is a complement whose demand decreases by 40 units/week instead.

**Deliverable:** both derivative values in USD/week per USD/unit and a statement of what further information is needed for an exact optimum.

**Acceptance criteria:** obtain +USD 240 and -USD 240 respectively; avoid claiming the numerical optimum can be found from the given local derivative alone.

### Exercise 6 — Audit the price recommendation

**Time:** 25 minutes.

An analyst estimates elasticity by correlating weekly sales with invoice prices across stores.
Promotions were targeted to low-demand stores, some stores stocked out, and a rebate was paid after the invoice.
The proposed price is 30% above the largest observed price.

**Deliverable:** a prioritized audit memo with at least four distinct risks, the likely mechanism of each, and an evidence or design response.

**Acceptance criteria:** address targeted promotions, stockout censoring, net-price measurement, and extrapolation; do not assign a confident bias direction without sufficient facts.

### Exercise 7 — Approve or withhold

**Time:** 25 minutes.

The Northstar base case suggests USD 20.20/unit.
A contract caps the relevant customer's net price at USD 19.50/unit, and the operations forecast indicates a possible shortage.
The commercial director wants immediate deployment across all customers.

**Deliverable:** a one-page decision memo with a recommended action, alternatives, named owners, and approval or stop conditions.

**Acceptance criteria:** do not recommend violating the cap; distinguish customer scopes and forecast from commitment; explicitly address supply and demand uncertainty.
Include a feasible comparison rather than assuming a single price can be rolled out everywhere.

### Exercise 8 — Design a learning plan

**Time:** 30 minutes.

The historical elasticity is uncertain and the USD 20.00/unit alternative is close to the modeled peak.
Design a lawful, operationally feasible test comparing the current USD 18.00/unit price with an eligible alternative.

**Deliverable:** a test outline specifying unit of assignment, treatment, primary contribution outcome, demand and service monitoring, duration or stopping logic, and a decision rule.

**Acceptance criteria:** prevent contamination where possible, distinguish randomized assignment from voluntary selection, address delayed rebates and returns, and include an independent legal and contract review.
Do not coordinate test prices with competitors.

## 13. Self-check answers

### Exercise 1

The price-cost margin is `(40-24)/40 = 0.40`, or 40% of price.
The markup on cost is `(40-24)/24 = 0.6667`, or approximately 66.67% of cost.
The inverse absolute elasticity is `1/2.5=0.40`, or 40%.
The equality holds at that point under the stated values.
It is a necessary stationary condition under the model assumptions, not proof of a global optimum or causal validity.

### Exercise 2

The derivative is `Q(p)+(p-c)Q'(p)`.
The first term is the gain from a small price increase applied to the current quantity.
The second is the change in contribution due to the induced quantity change.
Set the sum to zero at a differentiable interior stationary point.
Divide by positive quantity and replace `Q'(p)/Q(p)` with `e(p)/p`.
This gives `1+((p-c)/p)e=0`, hence `(p-c)/p=-1/e`.
Fixed `F` has derivative zero because it is unchanged across prices.
An active price ceiling, a kinked demand curve, or unmodeled portfolio effects can invalidate this simple equality as a decision rule.

### Exercise 3

With `E=4`, the margin is `1/4=0.25`.
The price is `18/(1-0.25)=USD 24/unit`.
The unit contribution is USD 6/unit, which is 25% of price and one-third of cost.
A local elasticity of -4 is only the slope at the current price.
It may change at the candidate, so a demand curve or a justified constant-elasticity assumption is needed.

### Exercise 4

The unconstrained stationary price is `(1,000+20*10)/(2*20)=USD 30/unit`.
The allowed ceiling is USD 28/unit.
The linear contribution function is concave, and the objective is increasing throughout the feasible interval up to the ceiling.
At USD 28/unit, demand is `1,000-20*28=440` units/week.
Contribution is `(28-10)*440=USD 7,920/week`.
At USD 25/unit, demand is 500 units/week and contribution is `15*500=USD 7,500/week`.
The ceiling candidate improves modeled contribution by USD 420/week relative to USD 25/unit.
The derivative at the ceiling is positive, so the unconstrained Lerner equality does not hold there.

### Exercise 5

For the substitute, the cross term is `6*40=+USD 240/week per USD/unit`.
Because the focal derivative is zero, the portfolio derivative is positive under the stated assumptions.
For the complement, the cross term is `6*(-40)=-USD 240/week per USD/unit`, giving a negative portfolio derivative.
An exact optimum needs demand functions or other information about derivatives away from the point, other product costs and prices, feasible ranges, and relevant constraints.
A local directional statement does not establish a global portfolio optimum.

### Exercise 6 rubric

A strong memo identifies promotion targeting as a confounding or selection risk.
It identifies lost demand during stockouts as censoring rather than zero demand.
It reconstructs net receipts after rebates and aligns them to units and exposure.
It flags the proposed 30% extrapolation beyond the observed range.
It proposes a credible design, such as eligible randomized price assignment, or a defensible alternative identification strategy.
It treats direction of bias as uncertain unless the underlying selection and measurement mechanisms establish it.

### Exercise 7 rubric

A strong memo withholds the broad rollout until contract and supply checks are complete.
It compares eligible prices at or below the contractual cap for the affected customer scope.
It identifies commercial, legal, finance, and operations owners.
It distinguishes modeled demand from available supply and committed orders.
It states the trigger for a revised recommendation and shows a feasible alternative for each relevant segment.

### Exercise 8 rubric

A strong outline defines eligible assignment units that reduce spillovers and preserves the assignment record.
It specifies randomized treatment where feasible, a comparable control, and a primary incremental contribution measure after relevant costs.
It allows enough time to observe returns and rebates or defines a defensible accrual.
It prespecifies monitoring and stopping decisions without using competitor coordination.
It includes contract, privacy, and customer communication review where relevant.
A specific sample size requires variability, minimum detectable effect, power, and a design; none is supplied in the question.

## 14. Interview questions

### 1. Derive the Lerner rule.

**Strong answer:** differentiates contribution, defines signed elasticity, and obtains `(p-MC)/p=-1/e` at an interior stationary point.

**Weak answer reveals:** memorization of a formula without the economic balance or sign convention.

### 2. Why is the margin denominator price rather than cost?

**Strong answer:** distinguishes the Lerner price-cost margin from markup on cost and computes each for a numerical example.

**Weak answer reveals:** interchangeable use of two percentages with different denominators.

### 3. Can an elasticity at today's price be used to calculate tomorrow's optimal price?

**Strong answer:** only with a justified demand model or assumption that remains valid at the candidate; linear demand has changing elasticity.

**Weak answer reveals:** extrapolation from a point estimate without checking the response curve.

### 4. What if the absolute elasticity is less than one?

**Strong answer:** with positive marginal cost it cannot satisfy the usual positive-price interior Lerner equality; inspect the curve and constraints rather than force a result.

**Weak answer reveals:** sign flipping or a negative price presented as a commercial recommendation.

### 5. How do you determine relevant cost?

**Strong answer:** identifies incremental costs and opportunity costs over the decision horizon and reconciles them to finance data.

**Weak answer reveals:** automatic substitution of accounting average cost or an assumption that all overhead is marginal.

### 6. How can the rule be valid at a price ceiling?

**Strong answer:** the unconstrained equality generally need not hold at an active boundary; compare feasible options and the derivative's direction.

**Weak answer reveals:** rejection of a constrained optimum because the derivative is nonzero.

### 7. Why can a product-level optimum hurt a portfolio?

**Strong answer:** a price move changes contribution from substitutes and complements, and shared capacity can add opportunity costs.

**Weak answer reveals:** treating product demand as independent without evidence.

### 8. How would you handle a stockout in elasticity data?

**Strong answer:** recognizes censored sales, checks availability, and develops an identification or lost-demand approach.

**Weak answer reveals:** assuming recorded sales equal unconstrained demand.

### 9. What should happen when the modeled optimum is outside the estimation range?

**Strong answer:** constrain the recommendation to credible support or acquire new evidence and disclose extrapolation.

**Weak answer reveals:** treating a curve-fitting equation as reliable at any price.

### 10. How should a team use competitor prices?

**Strong answer:** uses lawful independent information, models uncertainty and independent reactions, and avoids exchanging future pricing intentions or coordination.

**Weak answer reveals:** proposing competitor agreement or confusing market monitoring with a legal safe harbor.

## 15. Key terms

- **Absolute elasticity:** magnitude of a negative own-price elasticity; see [definitions](#4-definitions-and-vocabulary).
- **Boundary solution:** optimum at a feasible-set limit; see [constraints](#step-9--incorporate-operational-constraints).
- **Cannibalization:** transfer of demand between owned products; see [portfolio effects](#step-10--include-other-products-when-necessary).
- **Capacity constraint:** limit on supply or resource use; see [worked example](#add-a-supply-restriction).
- **Constant elasticity:** unchanged elasticity across a modeled range; see [rearrangement](#step-7--rearrange-cautiously).
- **Contribution:** revenue less specified relevant variable costs; see [objective](#step-4--write-contribution-as-a-function-of-price).
- **Cross-price elasticity:** response of another product's demand to the focal price; see [portfolio effects](#step-10--include-other-products-when-necessary).
- **Demand curve:** expected quantity conditional on price and other stated conditions; see [demand response](#step-3--describe-a-causal-demand-response).
- **Endogeneity:** relationship between price and omitted demand determinants or measurement error; see [failure modes](#10-failure-modes-and-red-flags).
- **Feasible set:** all price choices satisfying stated constraints; see [constraints](#step-9--incorporate-operational-constraints).
- **First-order condition:** zero derivative at a smooth interior stationary point; see [derivation](#step-5--differentiate-and-interpret-the-terms).
- **Global optimum:** best objective value over all feasible choices; see [candidate solving](#step-8--solve-for-a-candidate-with-a-real-curve).
- **Interior solution:** choice strictly inside the feasible set; see [assumptions](#11-assumptions-and-limits).
- **Lerner index:** price less marginal cost divided by price; see [definitions](#4-definitions-and-vocabulary).
- **Linear demand:** quantity described by `a-bp` in a valid range; see [worked example](#6-worked-example).
- **Local optimum:** best choice in a neighborhood; see [candidate solving](#step-8--solve-for-a-candidate-with-a-real-curve).
- **Marginal cost:** cost of an additional unit at the relevant output; see [cost waterfall](#step-2--build-the-net-price-and-cost-waterfall).
- **Marginal revenue:** revenue change associated with an additional unit; see [marginal revenue](#step-6--connect-the-rule-to-marginal-revenue).
- **Markup on cost:** price-cost difference divided by cost; see [exercise 1](#exercise-1--vocabulary-and-units).
- **Net realized price:** receipts per unit after specified transaction adjustments; see [data contract](#minimum-data-contract).
- **Own-price elasticity:** percentage demand response to a 1% change in the product's own price; see [demand response](#step-3--describe-a-causal-demand-response).
- **Portfolio objective:** total contribution across products; see [portfolio effects](#step-10--include-other-products-when-necessary).
- **Regret:** difference between a scenario's best feasible outcome and the selected outcome; see [uncertainty](#step-11--compare-uncertainty-rather-than-only-point-optima).
- **Shadow cost:** marginal value of a constrained resource; see [assumptions](#11-assumptions-and-limits).
- **Stationary point:** point with zero derivative; see [derivation](#step-5--differentiate-and-interpret-the-terms).
- **Substitute:** product whose demand rises when another price rises, all else fixed; see [portfolio diagnostic](#add-a-portfolio-diagnostic).
- **Complement:** product whose demand falls when another price rises, all else fixed; see [portfolio diagnostic](#add-a-portfolio-diagnostic).
- **Stockout censoring:** observed sales limited by availability; see [failure modes](#10-failure-modes-and-red-flags).

## 16. Sources

1. Lerner, Abba P. “The Concept of Monopoly and the Measurement of Monopoly Power.” *The Review of Economic Studies* 1, no. 3 (1934), 157–175. [https://doi.org/10.2307/2967480](https://doi.org/10.2307/2967480). Original price-cost index context; the index alone is not a legal finding.
2. Berry, Steven, James Levinsohn, and Ariel Pakes. “Automobile Prices in Market Equilibrium.” *Econometrica* 63, no. 4 (1995), 841–890. [https://doi.org/10.2307/2171802](https://doi.org/10.2307/2171802). Differentiated-product demand and equilibrium estimation.
3. Hendel, Igal, and Aviv Nevo. “Measuring the Implications of Sales and Consumer Inventory Behavior.” *Econometrica* 74, no. 6 (2006), 1637–1673. [https://doi.org/10.1111/j.1468-0262.2006.00723.x](https://doi.org/10.1111/j.1468-0262.2006.00723.x). Intertemporal effects of sales and inventory behavior.
4. Talluri, Kalyan T., and Garrett J. van Ryzin. *The Theory and Practice of Revenue Management*. Springer, 2004. [https://doi.org/10.1007/b139000](https://doi.org/10.1007/b139000). Pricing and capacity context.
5. University of Chicago Booth School of Business, Kilts Center for Marketing. [Dominick's Data Manual and documentation](https://www.chicagobooth.edu/research/kilts/datasets/dominicks). Public research dataset and scope.
6. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Identification and barcode specifications; use the applicable edition.
7. U.S. Federal Trade Commission. [Price Fixing](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/dealings-competitors/price-fixing). Official explanatory guidance on agreements among competitors; consult current law and counsel.
8. U.S. Federal Trade Commission. [Guides Against Deceptive Pricing, 16 CFR Part 233](https://www.ecfr.gov/current/title-16/chapter-I/subchapter-B/part-233). Current regulatory text for specified advertising practices.
9. European Union. [Treaty on the Functioning of the European Union, Article 101](https://eur-lex.europa.eu/eli/treaty/tfeu_2012/art_101/oj). Primary treaty text on specified restrictive agreements.

Sources are used for the claims identified in the text, not as evidence for the fictional numerical case.
Legal references were selected for educational context; applicability and current requirements must be checked for the decision date and jurisdiction.

## 17. Links

- [Module 16 topic index](_index.md).
- [Pricing decision stack](01_pricing-decision-stack.md).
- [Price, value, and willingness to pay](02_value-wtp.md).
- [Own-price and cross-price elasticity identification](03_elasticity-identification.md).
- [Module 16 survey](../../../course/16_pricing_promotions_revenue_management.md).
- [Module 16 labs: elasticity estimation and endogeneity audit](../../../course/16_pricing_promotions_revenue_management.md#7-labs).
