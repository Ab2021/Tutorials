# Competitive Response and Price Wars

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 09 of 10  
> Estimated study time: 4 hours  
> Prerequisites: 01, 03, and causal inference in Module 04  
> Version 2026.5

## 1. Why this topic exists

A price decision changes the environment in which the decision earns its return.
Customers compare alternatives.
Retailers may change displays.
Competitors may change price, assortment, advertising, capacity, or service.
A margin forecast that assumes those responses away can be precise and still be wrong.

The category manager must decide whether to hold, cut, promote, or change the offer.
The supply planner needs a volume range before reserving inventory and transport.
Finance needs incremental contribution, not just sales revenue.
Legal and compliance need to assess how evidence was collected and how decisions are communicated.
The customer needs an accurate, accessible final price.

A competitor's lower shelf price is an observation, not an instruction.
It may be a short promotion, a new list price, a clearance, a regional test, or a data error.
Matching before confirming the effective price can destroy contribution without protecting demand.
Ignoring a sustained competitor move can lose customers and leave stock stranded.
Neither automatic matching nor automatic inaction is a general rule.

The practical problem is to choose an independently determined action under uncertainty.
The analyst estimates customer demand conditional on the firm's action and plausible rival actions.
The analyst also considers how those rival actions might depend on the firm's choice.
The decision maker compares outcomes after costs, capacity, inventory, and implementation constraints.
A price recommendation is a forecast-based plan, not a promise of volume or a commitment by anyone else.

Price wars are repeated sequences of aggressive moves, not proof of illegal conduct.
A sustained price decline can reflect competition, lower costs, or excess capacity.
It can also be made worse by short review cycles, noisy data, and goals that reward share without margin.
The firm can assess its own economics without negotiating a shared pricing outcome.

Signaling requires particular care.
A public, truthful customer offer has a legitimate commercial function.
A communication intended to invite, coordinate, or reassure competitors about future pricing may create serious risk.
Consult qualified counsel before using public or private communications in a way that concerns rivals' future behavior.
Nothing in this lesson is legal advice or a substitute for jurisdiction-specific review.

The worked case uses a fictional consumer-goods company and invented numbers.
Its purpose is to make the decision arithmetic inspectable, not to describe an actual company's strategy.
The example also shows why a model can recommend a cut even when a static comparison favors holding.
That result depends on assumed rival-response probabilities; the assumptions are not established facts.

## 2. Learning outcomes

By the end of this topic you can:

1. **Describe** the difference between customer demand response and a competitor reaction function.
   Exercise 1 checks the distinction and the direction of each relationship.
2. **Compute** contribution for each own-price and rival-price scenario with consistent units.
   Exercises 2 and 3 check margin, quantity, and expected contribution.
3. **Choose** a feasible action using explicit response probabilities and a sensitivity threshold.
   Exercise 4 checks whether a recommendation changes when beliefs change.
4. **Critique** historical response estimates for simultaneous moves, common shocks, and selection bias.
   Exercise 5 asks for a defensible evidence plan rather than an unsupported causal claim.
5. **Design** operational guardrails for price, inventory, latency, and model monitoring.
   Exercise 6 checks decision rights and an escalation process.
6. **Identify** information-sharing and signaling red flags and specify a stop-and-escalate response.
   Exercise 7 checks what to do with an inappropriate invitation from a competitor.
7. **Explain** why public cases do not prove that every price-matching algorithm is unlawful.
   Exercise 8 checks the difference between evidence, inference, and legal conclusion.
8. **Communicate** a decision memo separating forecast, plan, target, and commitment.
   Exercise 9 checks commercial clarity and uncertainty disclosure.

These outcomes are deliberately commercial as well as quantitative.
A correct spreadsheet that uses impermissible information or excludes supply constraints is not decision-ready.
Likewise, a legal escalation does not replace an economic calculation of alternatives.

## 3. The core mental model

```text
Independent business objective and lawful information
                         |
                         v
          Feasible own actions and decision horizon
                         |
                         v
   +--------- own action p -----------+-------------------+
   |                                  |                   |
   v                                  v                   v
Customer demand                Rival may react      Operations react
D(p, rival price, X)           R(rival action | p)  stock, service, cost
   |                                  |                   |
   +------------------+---------------+-------------------+
                      v
       Scenario contribution and risk distribution
                      |
                      v
     Independent choice, documented approval, execution
                      |
                      v
        Observe outcomes, diagnose, update assumptions
```

The arrows represent information and possible causal influence, not a communication channel between competitors.
In particular, the rival-response box is a forecast made independently from lawful evidence.
It is not a request to a competitor for a future price or agreement.

### Two distinct functions

The **demand function** asks how many units customers buy at specified own and rival prices.
For a substitute, a higher rival price may increase demand for the firm's product.
Own and cross-price effects are properties of customer choices, subject to context and identification.

The **reaction function** describes a competitor's action conditional on an observed state or action.
A reduced-form forecast might estimate the probability of a rival promotion after an own-price cut.
A strategic best-response model instead computes the rival action that maximizes an assumed rival objective.
The latter needs assumptions about the rival's costs, demand, constraints, and information.
These are usually uncertain and may not be observable.

A demand function and a reaction function are not interchangeable.
A positive cross-price effect does not mean a competitor will match a cut.
A historical match rate does not measure customers' sensitivity to the rival's price.

### Timing matters

At time zero, the firm chooses an action using information then available.
Customers may react immediately, while a competitor may take days or weeks to respond.
After a rival response, customer demand and the firm's inventory position may change again.
If a model assigns a rival move to the same instant as the firm's price change, it may miss interim contribution.
If it assumes the rival never observes a public offer, it may overstate a durable advantage.

A static one-period scenario table is useful when the decision is short and response is summarized in that period.
A multi-period model is appropriate when response delays, switching, or inventory depletion matter materially.
Do not sum daily and weekly forecasts without aligning their time buckets.

### Feedback does not imply a stable equilibrium

A firm cuts, a competitor cuts, and the firm observes lost share again.
A second cut may follow if the decision rule simply matches the most recent price.
The loop can continue until a floor, capacity constraint, or human intervention stops it.
A modeled equilibrium is a prediction under specific assumptions, not a jointly agreed price.
Do not use an equilibrium concept as a goal to coordinate pricing with rivals.

### What the diagram omits

It compresses several operational layers: promotions, bundles, stock availability, and customer eligibility.
It also compresses geography and channel.
A rival's online price is not necessarily the relevant delivered price for a store customer.
Reconstruct the decision context before fitting a function or setting a response rule.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scope |
|---|---|---|---|
| Own price | Price paid for the focal product under the modeled offer | Using a list price when the customer pays less | Currency per unit, channel, time |
| Rival price | Comparable customer price for an alternative | Comparing different sizes or terms | Currency per comparable unit |
| Net price | Realized revenue per unit after modeled discounts and rebates | Mixing gross shelf price with net cost | Currency per unit |
| Price gap | Own price minus rival price, or a stated ratio | Reversing the sign across reports | Currency per unit or percent |
| Demand function | Units expected conditional on prices and context | Treating association as causal demand | Units per period |
| Own-price effect | Demand change from own-price change, other conditions specified | Applying promotion lift to a permanent price | Units per currency unit |
| Cross-price effect | Demand change from a rival-price change | Calling a rival's reaction a cross-price effect | Units per currency unit |
| Elasticity | Percentage demand change per percentage price change, locally | Treating an elasticity as a constant unit change | Dimensionless |
| Reaction function | Mapping from observed state or action to a rival action or distribution | Assuming it is an agreement | Rival action conditional on state |
| Best response | Action maximizing one actor's assumed objective given another's action | Treating a modeled optimum as observed behavior | Price or other action |
| Sequential response | An action following observation of an earlier move | Assuming response is instantaneous | Time and action |
| Simultaneous move | Decisions made without observing the other's current move | Confusing common timing with coordination | Defined decision interval |
| Scenario | Coherent combination of own action, rival action, and other conditions | Adding inconsistent assumptions | One modeled state |
| Conditional probability | Probability of a rival outcome given an own action and information | Using the same probability without checking action dependence | Probability from 0 to 1 |
| Contribution | Revenue less specified variable costs | Calling contribution net profit | Currency per period |
| Incrementality | Difference from a specified counterfactual | Comparing against an arbitrary sales target | Currency or units per period |
| Price war | Sustained competitive price reductions with material economic consequences | Treating any discount as evidence of collusion or predation | Market and time horizon |
| Promotion | Offer with defined eligibility and duration | Treating a promotion as permanent list price | Offer, channel, dates |
| Price match | An independent decision to charge a comparable price | Assuming all matching is illegal or always profitable | Product and period |
| Signaling | A communication or conduct that conveys information about intended behavior | Assuming all customer-facing advertisements are unlawful | Message, audience, context |
| Public price | A price genuinely available through lawful public observation | Assuming publication makes an invitation to coordinate safe | Offer and timestamp |
| Non-public sensitive data | Confidential or commercially sensitive information such as future individualized prices | Treating vendor access as automatic permission to share | Source, owner, date |
| Hub-and-spoke | Potential coordination involving a common intermediary and market participants | Assuming every common vendor arrangement is unlawful | Facts and applicable law |
| Common shock | Factor affecting multiple firms simultaneously | Attributing all correlated price changes to reaction | Market and date |
| Endogeneity | Correlation between an explanatory variable and unobserved demand or decision factors | Treating regression coefficients as causal automatically | Estimation setting |
| Censoring | Observed sales limited by stock or capacity rather than demand | Interpreting stockout sales as full demand | Units and availability |
| Price floor | Internal feasibility boundary or separately defined legal concept | Assuming every below-cost price is automatically illegal | Currency per unit and cost measure |
| Capacity | Maximum feasible service or sales volume in the horizon | Forecasting sales above available stock | Units per period |
| Holdout | Units or markets not exposed to an intervention for comparison | Calling an unrepresentative market a valid control | Store, geography, or period |
| Spillover | Effect on units outside the intended treatment | Assuming nearby stores or rivals cannot respond | Geography and time |
| Decision horizon | Period over which consequences are compared | Comparing a weekly gain to a monthly loss | Days, weeks, or months |
| Robust decision | An action acceptable across a stated set of plausible assumptions | Claiming certainty from one scenario | Defined uncertainty set |
| Commitment | Authorized operational obligation such as a purchase order | Calling a forecast an order | Units, date, owner |

The terms are reused in the mechanics and cases below.
A full glossary with section links appears in section 15.

## 5. The mechanics, step by step

### Step 1: Frame the decision

**Input:** the customer offer, market, current price, available actions, review date, and objective.
**Output:** a written decision question with a specific horizon.

Name the product, pack size, channel, geography, and customer eligibility.
State whether the action is a list-price change, a temporary discount, a bundle, or a service change.
Specify the start and end dates and whether the offer can be reversed.

Write the objective as an economic quantity, such as incremental contribution in USD per week.
A share objective may be relevant, but give it an explicit constraint or value rather than treating share as profit.
Include acquisition, fulfillment, and return costs if they change with the action.
State which overhead is excluded and do not label the result net income.

For example, the decision might be whether to hold USD 24 per unit or offer USD 23 per unit for one week.
The relevant customers are online buyers in a defined region.
The available stock must cover the scenario volumes, or sales must be capped.
The owner is the pricing director, with separate finance and legal review where required.

**Branch:** If the objective or eligible offer is unclear, stop before estimating a response.
A model cannot resolve a disagreement about which business outcome matters.

### Step 2: Construct comparable observed prices

**Input:** permitted public offer data and the firm's transaction and cost records.
**Output:** an auditable set of normalized price observations.

Record source, observation time, effective time, geography, channel, pack size, and currency.
Capture availability, delivery fees, minimum order, membership requirement, and coupon eligibility where observable.
Convert packs to a common unit only when the products are meaningfully comparable.
Distinguish an advertised price from the price a customer can actually obtain.

A USD 20 pack containing 10 units is USD 2 per unit before fees.
A USD 18 pack containing 8 units is USD 2.25 per unit before fees.
A raw sticker-price comparison would rank them incorrectly on a per-unit basis.
Quality, usage rate, and product features may still make the unit-price comparison incomplete.

Identify stale quotes and missing availability.
A sold-out listing may not constrain a customer who needs immediate delivery.
A scraper that uses a member-only offer for every customer can create a false undercut.
Preserve source evidence under the organization's data retention and access rules.

**Branch:** If the comparison cannot be validated, do not automatically change price.
Use a manual review or a sensitivity range while the observation is investigated.
Do not seek non-public competitor plans to fill the gap.

### Step 3: Map the customer's alternatives

**Input:** category definitions, substitution evidence, customer segment, and offer attributes.
**Output:** a short list of relevant alternatives and their limitations.

A competitor is relevant when customers plausibly switch to its offer for the same need.
A national rival may be irrelevant in a region it cannot serve.
A private-label alternative may matter even if its pack or service differs.
A wholesale customer may compare a contract rebate and delivery terms, not shelf price alone.

Separate direct substitutes from complementary products.
Raising the price of a substitute can increase own demand; raising a complement's price can reduce it.
The empirical sign may vary across segments and time periods.
Avoid assigning a single competitor weight to every customer without evidence.

Use observed substitution, experiments, customer research, or choice data when available.
Explain when the evidence is descriptive rather than causal.
Record excluded alternatives and the conditions under which they would become relevant.
A competitor's price should not enter a model solely because it is easy to collect.

### Step 4: Separate customer response from competitor response

**Input:** aligned prices, sales, availability, marketing, and decision timing.
**Output:** distinct customer-demand and rival-action hypotheses.

Write a demand equation such as `q = a + b*p + c*r`, with `b < 0` and `c > 0` for a substitute.
Here `p` is own price, `r` is rival price, and `q` is expected units in a defined period.
The signs are assumptions or estimates, not universal laws.
A linear function is local and should not be extrapolated indefinitely.

Write a rival response distribution separately, for example `Pr(r | p, history)`.
It may assign probabilities to no change, a promotion, and a larger price reduction.
Use a distribution when rival intent and cost are uncertain.
A single point forecast hides the downside of a wrong rival assumption.

A best-response curve is a special strategic model, not a measured fact.
Given a specified rival cost and demand function, it gives the rival's optimal action for each own price.
Unknown costs, contracts, and objectives make the curve particularly uncertain.
Never request those inputs from a competitor or infer a commitment from a scenario.

### Step 5: Examine identification before estimating

**Input:** candidate history and a causal question tied to a price change.
**Output:** an evidence grade and an estimation plan.

Prices can respond to expected demand rather than cause it alone.
The firm may discount when it anticipates weak demand.
A competitor may simultaneously discount because both observe the same season or cost shock.
A naive regression can therefore misstate own elasticity, cross elasticity, and rival response.

Draw a timeline with data available before each decision.
Do not use future observations as predictors of an earlier decision.
Record holidays, advertising, distribution, weather where relevant, and stockouts.
Distinguish omitted variables that can be measured from those that remain unobserved.

A lawful independent price experiment may help identify customer response.
A credible design needs treatment assignment, a comparison group, sufficient scale, and a predefined outcome.
Spillovers, competitor reactions, and supply constraints must be measured rather than assumed absent.
A test can estimate the combined effect of own action and induced rival action in the tested environment.
That combined effect is not automatically the direct effect holding the rival fixed.

Natural experiments or instruments may be useful when their assumptions are defensible.
An instrument must affect the relevant price and influence demand only through the specified price pathway, conditional on controls.
A shared cost shock may affect both firms and customers, violating that exclusion restriction.
Pre-trends and placebo checks are informative, but none alone proves identification.
Do not claim causal identification merely because a model has many controls.

### Step 6: Estimate response timing and uncertainty

**Input:** lawful public price histories, event definitions, and a baseline comparison.
**Output:** an action-conditional range of possible rival outcomes and response delays.

Define a rival response in advance: for example, a price drop of at least USD 1 per unit within seven days.
Specify whether a temporary coupon counts and how missing observations are treated.
Choose an event window consistent with how fast the rival can revise an offer.
A seven-day response rate is not a same-day response rate.

Compare own moves with appropriate non-move periods or control markets when feasible.
Adjust for common calendar shocks and known changes in supplier costs.
A raw fraction of matching events is descriptive unless the design supports a causal interpretation.
The rival may have planned its action before observing the firm's move.

Report sample counts, missingness, confidence or credible intervals, and changes over time.
Stratification can reveal different behavior by channel or promotion type, but small cells are unstable.
A beta-binomial or other shrinkage approach may reduce extreme estimates when data are sparse.
Its prior and pooling assumptions must be disclosed.

When data are insufficient, elicit a range from internal experts using independently available information.
Record the rationale and disagreements before seeing which action the assumptions favor.
Expert probabilities are assumptions, not hidden observations of a competitor's intentions.

### Step 7: Create coherent scenario paths

**Input:** feasible own actions, rival outcomes, probabilities, and timing.
**Output:** a scenario tree whose probabilities sum to one for each own action.

For each own action, list mutually exclusive rival outcomes over the same horizon.
Include no response, a moderate response, and a more adverse response if they are material.
Assign conditional probabilities separately for each own action when justified.
If probabilities are not credible, compare a range or use a robust criterion instead of inventing precision.

A scenario must align the rival price with the period in which it applies.
If the rival responds halfway through a week, calculate two subperiods or use a defensible weighted approximation.
If own price changes midway too, track both prices and quantities by interval.
Ensure the horizon contains the same total number of days in every action.

Check that each branch is economically possible and that conditional probabilities sum to 1.
A scenario tree is not a forecast of an agreement among firms.
It is a record of uncertain independent actions.

### Step 8: Translate each path into demand and feasible sales

**Input:** demand model, scenario prices, availability, and capacity.
**Output:** sales units by scenario, along with unmet demand where relevant.

Compute latent or unconstrained demand at each price pair.
Cap sales at the available stock and fulfillment capacity for the horizon.
Use `sales = min(demand, available units)` only if demand is nonnegative and lost sales cannot be backordered.
A model that allows backorders or substitution requires a different fulfillment rule.

If a product sells out, recorded sales underestimate demand.
Stockout timing and availability can help distinguish low demand from constrained sales.
Include the effect of lost service and replenishment lead time when material.
A price cut that consumes scarce stock may reduce later high-margin sales.

For a multi-period model, update remaining inventory after each interval.
Do not reuse the same starting stock independently in every interval.
Treat returns and cancellation rates consistently with the price and demand measures.

### Step 9: Calculate contribution and opportunity costs

**Input:** net realized price, feasible units, variable cost, and action-specific cost.
**Output:** contribution by path in a common currency and period.

For the simple case, contribution equals `(price - variable cost) * sales - action cost`.
The cost must include costs that vary with each unit sold and are attributable to the decision.
A one-time promotion fee should be subtracted once, not per unit.
A supplier reimbursement should be included only if the contract actually provides it for that scenario.

Include cross-product cannibalization if the offer diverts sales from another profitable product.
Include later pull-forward if a temporary deal moves future orders into the current week.
Subtract incremental logistics or service costs if the volume surge makes them necessary.
For finite inventory, compare the contribution from selling now with the contribution or salvage value of waiting.

Expected contribution is the probability-weighted sum of scenario contributions.
Incremental expected contribution is the difference between actions against the same decision horizon.
Do not multiply average price by average volume when their joint distribution matters.
Do not subtract a fixed cost twice or compare revenue for one action to contribution for another.

### Step 10: Apply risk and operational constraints

**Input:** scenario outcomes, inventory, cash, service, and governance limits.
**Output:** the feasible action set and a view of downside exposure.

Ask whether the firm can fulfill the high-demand branch without unacceptable stockouts.
Ask whether the lowest contribution branch meets the organization's cash or margin requirement.
Use minimum contribution, a downside quantile, or a scenario loss limit only when the criterion is explicit.
Different risk criteria can produce different choices from the same scenarios.

A minimum price may reflect contract terms, customer commitments, cost recovery, or authorization limits.
Document its commercial purpose and owner.
It is not a substitute for legal analysis of pricing conduct.
A model may reject an action for a business reason without stating that the action is unlawful.

State whether the decision maker can change price during the horizon.
If reversals are possible, include customer trust, implementation delay, and any applicable pricing rules.
Do not assume a free, instantaneous reversal of a public offer.

### Step 11: Test sensitivity and choose independently

**Input:** scenario values and credible ranges for the uncertain inputs.
**Output:** a recommendation, threshold, and conditions that would change it.

Vary reaction probability, response delay, cross-price effect, unit cost, and available stock.
Change one assumption at a time to show directional effects.
Then change related assumptions together in a coherent adverse case.
Do not assign independent extreme values if they cannot plausibly occur together.

For two actions, solve for the reaction probability at which their expected contributions are equal.
The threshold tells the manager how much belief in the rival response matters.
It does not identify the true probability.
A decision near the threshold warrants a wider range, better evidence, or a reversible plan.

Record the chosen action and reasons in an internal decision memo.
The memo should demonstrate an independent decision based on lawful information.
Do not publish a message designed to obtain reciprocal pricing behavior.

### Step 12: Implement, monitor, and learn

**Input:** approved action, controls, and a monitoring plan.
**Output:** a time-stamped change and a post-decision review.

Record the authorized price, eligible customers, effective dates, channel, and system owner.
Validate that the offer displayed to customers matches the approved offer.
Track data freshness, sales, stock, contribution, competitor observations, and service levels.
Establish clear intervention triggers before execution.

Separate a forecast from the operational plan and the inventory commitment.
A forecast is an uncertain expectation; a plan is the chosen course of action.
A target is a desired performance level; a commitment is an authorized obligation.
A high-demand forecast does not itself authorize an unreviewed purchase order.

After the period, compare observed outcomes with each forecast branch.
Investigate data errors, demand misspecification, rival timing, and stock constraints.
Update assumptions only with an honest account of selection and common shocks.
A successful contribution outcome does not prove the rival-response probability was correct.

## 6. Worked example

### Case: one-week online offer

A fictional household-products seller, Harbor, faces a competing offer from Ridge.
Both sell comparable packs through the same online region.
Harbor must set its independently chosen customer price for the next seven days.
The example uses USD throughout and assumes no sales tax or delivery-price difference.

Harbor's current net price is USD 24 per pack.
Its relevant variable cost is USD 14 per pack.
The price alternatives are to hold at USD 24 or offer USD 23 per pack.
The USD 23 offer has no incremental fixed execution cost in this simplified case.
Ridge's observed current customer price is USD 24 per pack.
Its possible next-week prices in this simplified model are USD 24 and USD 22 per pack.
Harbor does not know Ridge's future action and does not communicate with Ridge about it.

The planning horizon is one week.
Harbor has 12,000 packs available and sufficient fulfillment capacity for 12,000 packs.
There is no backordering, pull-forward, cannibalization, or salvage effect in the calculation.
Those exclusions are simplifications, not claims that such effects are usually zero.

The demand model is an illustrative local forecast, not a fitted causal result:

```text
Expected Harbor demand, packs/week = 20,000 - 800*p + 400*r
p = Harbor net price, USD/pack
r = Ridge customer price, USD/pack
```

The own-price coefficient is minus 800 packs per week per USD/pack.
The rival-price coefficient is plus 400 packs per week per USD/pack.
Within this range a lower own price increases forecast demand.
Within this range a lower rival price decreases forecast Harbor demand.

### Step A: calculate the four demand outcomes

At own USD 24 and rival USD 24:

```text
q = 20,000 - 800*24 + 400*24
  = 20,000 - 19,200 + 9,600
  = 10,400 packs/week
```

At own USD 24 and rival USD 22:

```text
q = 20,000 - 800*24 + 400*22
  = 20,000 - 19,200 + 8,800
  = 9,600 packs/week
```

At own USD 23 and rival USD 24:

```text
q = 20,000 - 800*23 + 400*24
  = 20,000 - 18,400 + 9,600
  = 11,200 packs/week
```

At own USD 23 and rival USD 22:

```text
q = 20,000 - 800*23 + 400*22
  = 20,000 - 18,400 + 8,800
  = 10,400 packs/week
```

Every scenario is within the 12,000-pack capacity, so sales equal modeled demand.
The model does not say how many unique customers switch, how long loyalty persists, or what happens in later weeks.

### Step B: calculate scenario contribution

The unit contribution at USD 24 is USD 24 minus USD 14, or USD 10 per pack.
The unit contribution at USD 23 is USD 23 minus USD 14, or USD 9 per pack.
Multiply each unit contribution by the relevant scenario's sales.

| Harbor price | Ridge price | Forecast sales | Contribution per pack | Weekly contribution |
|---:|---:|---:|---:|---:|
| USD 24 | USD 24 | 10,400 packs | USD 10 | USD 104,000 |
| USD 24 | USD 22 | 9,600 packs | USD 10 | USD 96,000 |
| USD 23 | USD 24 | 11,200 packs | USD 9 | USD 100,800 |
| USD 23 | USD 22 | 10,400 packs | USD 9 | USD 93,600 |

If Ridge holds at USD 24, holding generates USD 3,200 more contribution than cutting.
If Ridge cuts to USD 22, holding generates USD 2,400 more contribution than cutting.
Those comparisons hold Ridge's action fixed and therefore do not yet account for action-dependent response.

### Step C: specify a hypothetical reaction distribution

Suppose Harbor's internal scenario workshop uses these illustrative beliefs:

| Harbor action | Probability Ridge holds at USD 24 | Probability Ridge cuts to USD 22 |
|---|---:|---:|
| Hold at USD 24 | 0.20 | 0.80 |
| Cut to USD 23 | 0.90 | 0.10 |

These probabilities are intentionally strong assumptions.
The table does not claim that an observed cut would deter Ridge or that deterrence is generally expected.
The numbers are chosen to show how action-dependent reactions can change a decision.
They would require a defensible evidence base and sensitivity analysis before an actual business uses them.
They do not represent information from Ridge or a mutual understanding.

### Step D: compute expected contribution

For holding:

```text
Expected contribution = 0.20 * USD 104,000 + 0.80 * USD 96,000
                      = USD 20,800 + USD 76,800
                      = USD 97,600 per week
```

For cutting:

```text
Expected contribution = 0.90 * USD 100,800 + 0.10 * USD 93,600
                      = USD 90,720 + USD 9,360
                      = USD 100,080 per week
```

On these assumptions, the cut has USD 2,480 more expected weekly contribution.
Expected units are 9,760 packs for holding and 11,120 packs for cutting.
Both expected values combine different potential states; neither is a guaranteed outcome.
If Ridge cuts, the cut action earns only USD 93,600, below the hold action's USD 96,000 in that same state.

### Step E: calculate the decision threshold

Let `h` be the probability Ridge cuts if Harbor holds.
Let `c` be the probability Ridge cuts if Harbor cuts.
Both probabilities are conditional on the indicated Harbor action.

```text
Hold expected contribution = 104,000 - 8,000*h
Cut expected contribution  = 100,800 - 7,200*c
Cut is better when         8,000*h - 7,200*c > 3,200
```

If `c` is fixed at 0.10, the cut is better when `h > 0.49`.
At `h = 0.49`, both actions have expected contribution of USD 100,080 per week.
At `h = 0.30` and `c = 0.10`, holding yields USD 101,600, so holding wins by USD 1,520.
The chosen USD 23 offer is therefore sensitive to a probability that is difficult to know.

If instead both actions had the same probability of Ridge cutting, call it `x`:

```text
Hold minus cut = (104,000 - 8,000*x) - (100,800 - 7,200*x)
               = 3,200 - 800*x
```

For any `x` from 0 to 1, holding has at least USD 2,400 more expected contribution.
A model that accidentally reuses the action-dependent probabilities as if they were identical changes the conclusion.
Conversely, action dependence must not be invented solely to justify a favored cut.

### Step F: check a constrained variation

Suppose only 10,800 packs can be fulfilled in the week.
Only the USD 23 own-price, USD 24 rival-price branch exceeds capacity.
Its sales fall from 11,200 to 10,800 packs and its contribution falls from USD 100,800 to USD 97,200.
The cut's expected contribution becomes `0.90*97,200 + 0.10*93,600 = USD 96,840`.
The hold remains at USD 97,600 because both of its branches remain under capacity.
Under the original response probabilities, holding now wins by USD 760.

The operational constraint reverses the recommendation.
That result assumes excess demand is lost and that the rival-response probabilities do not change with availability.
If customers backorder or competitors respond to a visible stockout, this simplified variation is incomplete.

### Step G: write the decision memo

**Forecast:** under the stated probability assumptions, 11,120 packs and USD 100,080 contribution for the cut.
**Plan:** offer USD 23 per pack for seven days only if 12,000 packs of capacity are confirmed.
**Target:** management may set an internal contribution target, but it is not the forecast or a guarantee.
**Commitment:** a purchase or transport order requires separate authorization and is not implied by the forecast.
**Sensitivity:** hold becomes preferable if `h` is below 0.49 when `c` remains 0.10, or if capacity is 10,800 packs.
**Review:** verify the price observation and review demand and fulfillment daily using authorized internal information.
**Legal boundary:** Harbor decides independently and does not seek Ridge's future prices or reciprocal conduct.

A prudent manager could still hold if the probability assumptions are too weak to support the cut.
The numerical expected-value recommendation is conditional, not an unconditional instruction.

## 7. Code, query, or calculation

The following Python 3 program reproduces the four scenario values, expected values, threshold, and capacity sensitivity.
It uses only the standard library and requires no external data or packages.
Save it as `competitive_case.py` and run `python competitive_case.py`.
The code represents the fictional one-week case, not a recommendation for a real market.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Action:
    name: str
    own_price: float  # USD per pack
    rival_cut_probability: float  # conditional on this action


VARIABLE_COST = 14.0  # USD per pack
RIVAL_HOLD_PRICE = 24.0  # USD per pack
RIVAL_CUT_PRICE = 22.0  # USD per pack
DEFAULT_CAPACITY = 12_000.0  # packs per week


def demand(own_price: float, rival_price: float) -> float:
    """Forecast unconstrained packs/week within the example's price range."""
    return max(0.0, 20_000.0 - 800.0 * own_price + 400.0 * rival_price)


def scenario(
    own_price: float,
    rival_price: float,
    capacity: float = DEFAULT_CAPACITY,
) -> tuple[float, float]:
    """Return fulfilled packs and USD contribution for one week."""
    sales = min(demand(own_price, rival_price), capacity)
    contribution = (own_price - VARIABLE_COST) * sales
    return sales, contribution


def expected(
    action: Action,
    capacity: float = DEFAULT_CAPACITY,
) -> tuple[float, float]:
    """Return probability-weighted packs and USD contribution per week."""
    probability_cut = action.rival_cut_probability
    hold_sales, hold_profit = scenario(action.own_price, RIVAL_HOLD_PRICE, capacity)
    cut_sales, cut_profit = scenario(action.own_price, RIVAL_CUT_PRICE, capacity)
    sales = (1.0 - probability_cut) * hold_sales + probability_cut * cut_sales
    contribution = (1.0 - probability_cut) * hold_profit + probability_cut * cut_profit
    return sales, contribution


def hold_cut_threshold(cut_response_probability: float) -> Optional[float]:
    """Hold-response probability where the two unconstrained actions tie."""
    hold_high = scenario(24.0, RIVAL_HOLD_PRICE)[1]
    hold_low = scenario(24.0, RIVAL_CUT_PRICE)[1]
    cut_high = scenario(23.0, RIVAL_HOLD_PRICE)[1]
    cut_low = scenario(23.0, RIVAL_CUT_PRICE)[1]
    cut_expected = (
        (1.0 - cut_response_probability) * cut_high
        + cut_response_probability * cut_low
    )
    gap = hold_high - hold_low
    if gap == 0:
        return None
    return (hold_high - cut_expected) / gap


def main() -> None:
    actions = [Action("hold", 24.0, 0.80), Action("cut", 23.0, 0.10)]
    for action in actions:
        for rival_price in (RIVAL_HOLD_PRICE, RIVAL_CUT_PRICE):
            sales, contribution = scenario(action.own_price, rival_price)
            print(
                f"{action.name} rival={rival_price:.0f}: "
                f"sales={sales:,.0f} contribution=${contribution:,.0f}"
            )
        sales, contribution = expected(action)
        print(f"{action.name} expected: sales={sales:,.0f} contribution=${contribution:,.0f}")

    threshold = hold_cut_threshold(0.10)
    if threshold is not None:
        print(f"tie threshold for hold response={threshold:.2f}")
    for action in actions:
        _, contribution = expected(action, capacity=10_800.0)
        print(f"capacity 10,800 {action.name}: contribution=${contribution:,.0f}")


if __name__ == "__main__":
    main()
```

Expected output:

```text
hold rival=24: sales=10,400 contribution=$104,000
hold rival=22: sales=9,600 contribution=$96,000
hold expected: sales=9,760 contribution=$97,600
cut rival=24: sales=11,200 contribution=$100,800
cut rival=22: sales=10,400 contribution=$93,600
cut expected: sales=11,120 contribution=$100,080
tie threshold for hold response=0.49
capacity 10,800 hold: contribution=$97,600
capacity 10,800 cut: contribution=$96,840
```

The script floors modeled demand at zero and caps fulfilled sales at capacity.
It does not validate inputs, model response timing, estimate the probabilities, or consider future-period effects.
The threshold function is tied to the example's 12,000-pack capacity and its two specific actions.
A production implementation needs approved data access, input checks, versioning, monitoring, and owner sign-off.
The absence of those features here is deliberate: this is a reproducible calculation, not a deployed pricing system.

## 8. Data, standards, and regulatory context

### Data needed for an independent decision

A useful observation has a product identifier, a timestamp, a source, a customer-facing price, and availability.
It also records the geography, channel, currency, pack size, and relevant offer eligibility.
The firm's own transaction data should distinguish orders, cancellations, returns, discounts, and net revenue.
Costs should align with the same product, horizon, and decision.

A product identifier supports matching, but it does not make offers equivalent.
A Global Trade Item Number (GTIN) identifies a trade item under GS1 conventions.
It does not prove that two offers have the same delivery terms, membership rules, or customer value.
Pack-size conversions and product substitutions still require business judgment.
GS1 identifiers are a data standard, not an antitrust safe harbor or a mandatory rule for every dataset.

Keep the observation time separate from the offer's effective time.
A price seen on Tuesday may have started Monday and ended before Wednesday.
A rival response may be incorrectly attributed to an own move if effective times are missing.
Preserve historical price versions rather than overwriting them with the latest scrape.

Define a data-quality contract for completeness, latency, duplicates, and missing availability.
Use access controls for confidential own-company data.
Do not ingest a rival's non-public future prices simply because a vendor makes them available.
Ask counsel and data governance to review provenance, permission, aggregation, and use.

### United States competition law

Section 1 of the U.S. Sherman Act prohibits certain agreements in restraint of trade.
Section 2 addresses monopolization and related conduct under its own standards.
The Federal Trade Commission Act also authorizes the Federal Trade Commission (FTC) to address unfair methods of competition.
The precise legal standard depends on the conduct, evidence, and jurisdiction.

The U.S. Department of Justice (DOJ) and FTC explain that agreements among competitors to fix prices can be unlawful.
An agreement may be proved through evidence of conduct and context, not only a signed contract.
Independent parallel pricing alone is not automatically an agreement.
The distinction is fact-specific and should be assessed by counsel.

An intermediary or algorithm does not automatically eliminate risk.
A common vendor can be useful and lawful, but arrangements involving sensitive competitor data and coordination require review.
The agencies' 2024 RealPage complaint alleged an unlawful scheme involving competitively sensitive information and pricing recommendations.
That allegation is not a final judicial finding that every algorithmic pricing service is unlawful.
The November 2025 proposed consent decree is a settlement proposal in that litigation, not a universal statutory rule.
Consult the current court record and counsel for any subsequent procedural outcome.

Information exchange can be problematic even without a conventional price-fixing agreement.
Its competitive effects depend on the nature, age, granularity, recipients, purpose, market, and other facts.
There is no universal rule that a specified age or aggregation threshold automatically makes a dataset safe.
The DOJ and FTC withdrew certain historical healthcare antitrust policy statements in 2023; do not treat old numerical information-sharing thresholds as a general safe harbor.

### European Union and other jurisdictions

Article 101 of the Treaty on the Functioning of the European Union prohibits certain agreements and concerted practices that restrict competition.
Article 102 addresses abuse of a dominant position.
The European Commission's 2023 Horizontal Guidelines discuss information exchange and the assessment of horizontal cooperation.
The guidelines explain enforcement analysis; the treaty and applicable law provide the legal framework.
National rules and enforcement practices can differ, and other jurisdictions have their own laws.

Unilateral public pricing and lawful market observation are not automatically prohibited.
The content, audience, timing, intent, and surrounding conduct of a communication matter.
A message seeking a rival's assurance about future price can create risk even if phrased as a public announcement.
Do not use a press release, trade association, supplier, platform, or algorithm to solicit a coordinated response.
Escalate ambiguous communications to counsel rather than trying to decide the legal outcome from a checklist.

### Consumer and operational obligations

Customer-facing price claims, reference prices, eligibility, taxes, and advertised availability may be regulated separately from competition law.
The relevant requirements depend on jurisdiction and channel.
A competitor's misleading offer does not authorize the firm to make a misleading offer of its own.
Review displayed prices and promotions with the appropriate compliance owner.

Retain decision records according to company policy and applicable legal holds.
Records should support auditability and independent decision making.
They should not contain speculative accusations or unnecessary competitor-sensitive information.
A compliance escalation is not a license to collect or circulate the material further.

## 9. Industry practice and case evidence

### Retail monitoring and customer-facing offers

Retailers commonly observe public prices, compare equivalent products, and independently choose promotions.
A price-comparison tool can support a legitimate business decision when its inputs and authority are controlled.
The presence of a competitor price in a model does not itself establish an unlawful agreement.
What matters legally includes how information was obtained, who receives it, and the surrounding conduct.

The commercial issue is more subtle than matching the lowest displayed number.
A membership discount may apply to only a subset of customers.
A competitor may be out of stock or charge different delivery fees.
A channel-specific promotion can change the effective competitive gap without changing list prices.
A responsible process distinguishes those conditions before changing the firm's offer.

**Inference:** a retailer using stale or ineligible prices can trigger unnecessary reactive cuts.
That inference follows from the decision mechanics; it is not a finding about a named retailer.
Measure the frequency of stale inputs and rejected matches before concluding that automation is harming margin.

### U.S. airline tariff case: public announcements can raise concerns

In United States v. Airline Tariff Publishing Company and several airlines, the DOJ challenged alleged coordination involving airline fare information disseminated through a shared system.
The government's 1992 complaint alleged that the system was used to communicate and coordinate proposed fare changes.
The case was resolved through consent decrees, including a 1994 final judgment.
A consent decree resolves claims under its terms and is not the same as a litigated finding on every allegation.
The final judgment imposed specific restrictions on covered conduct; it is not a general ban on publishing airline fares.

The case illustrates why the purpose and structure of communications matter.
A genuine offer to customers differs from a message structured to convey a proposed future price to competitors and invite a response.
Whether a particular modern announcement has legal consequences is fact-specific.
The business lesson is to review intended audience and commercial purpose, not to avoid ordinary accurate advertising.

### U.S. RealPage litigation: common intermediaries need scrutiny

The DOJ and state plaintiffs filed a civil antitrust complaint against RealPage in 2024 concerning rental pricing software.
The complaint alleged that the company collected non-public, competitively sensitive landlord information and used it to generate pricing recommendations.
The complaint alleged that landlords' use of those recommendations harmed competition.
Those allegations are the plaintiffs' claims, not proof of liability.

In November 2025 the DOJ announced a proposed settlement with RealPage.
The announced proposal included restrictions addressing use of competitors' sensitive non-public data and coordination-related services.
The scope and status of any final order must be checked in the current case record.
The named case concerns rental housing; its specific facts cannot simply be transferred to a retailer's public-price monitor.

**Inference:** a pricing team considering a shared vendor should map data flows and permitted uses before deployment.
Review whether inputs are confidential, whether outputs expose individual competitors, and who can see recommendations.
A vendor assurance of legality is not a substitute for the buyer's own review.
The practical control is an approved data and decision architecture, not a blanket ban on analytics.

### U.S. online poster case: algorithms can implement agreements

In 2015 the DOJ announced a guilty plea by an online poster seller, David Topkins, for conspiring to fix prices of posters sold online.
The DOJ described an agreement with competitors and the use of pricing algorithms to implement the agreed prices.
The public enforcement account concerns the specific plea and facts in that case.
It does not establish that independently written price-adjustment software is itself illegal.

The analytical distinction is between unilateral automation and an agreement implemented through automation.
A pricing rule's source code should be reviewed together with contracts, communications, data flows, and operating practice.
An algorithm can act faster than a manual team, making monitoring and a human stop control important.
Technical speed does not change the requirement that each firm's decisions be independent.

### A second fictional business case: avoid a false price war

A beverage distributor sees a competitor's public USD 15 per case offer against its own USD 18 per case.
The competitor's price appears to be 16.7% lower on a case basis.
A proposed automatic match would reduce the distributor's contribution by USD 3 per case before any volume effect.
At 4,000 cases per week, the immediate contribution exposure on unchanged volume is USD 12,000 per week.

Review finds the rival's case contains 18 bottles while the distributor's contains 24 bottles.
The rival price is USD 0.833 per bottle; the distributor price is USD 0.750 per bottle.
The apparent sticker-price gap reverses on a bottle basis before adjusting for quality and delivery.
Assume the distributor's per-case variable cost is USD 12.
At USD 18 it earns USD 6 per case, or USD 24,000 on 4,000 cases.
At USD 15 it earns USD 3 per case and would need 8,000 cases to retain USD 24,000 contribution, ignoring capacity and other effects.

The decision is to hold while monitoring actual customer substitution, not to assume unit price is the whole customer value.
If evidence later shows customers prefer the smaller pack or face a cash-budget constraint, the distributor should revisit the offer.
This case is fictional; it illustrates a workflow and is not evidence of a named company's outcome.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Automatic match on stale price | Source timestamp exceeds refresh limit | Old observation drives current action | Unnecessary margin loss | Expire observations and require review |
| Non-comparable pack | Unit conversion missing | Sticker prices measure different quantities | False gap | Normalize and review customer value |
| Coupon eligibility ignored | Membership price treated as universal | Price is unavailable to most customers | Overreaction | Model eligibility and delivered price |
| Rival action confused with demand | Cross elasticity labeled match rate | Two distinct processes are conflated | Wrong scenario probabilities | Estimate and document separately |
| Common shock omitted | Multiple firms move after a cost change | Shared cause mimics response | Spurious reaction function | Align timing and control for shocks |
| Own price endogeneity | Discounts cluster in weak-demand weeks | Price responds to expected sales | Biased elasticity | Use credible experimental or quasi-experimental design |
| Stockout censoring | Sales plateau as availability falls | Observed sales are capped demand | Understated demand | Include availability and model lost sales |
| Immediate-response assumption | Rival changes price days later | Entire week assigned to post-response state | Misstated contribution | Split the horizon by response timing |
| Invalid probabilities | Scenario weights exceed one | Inputs are not a distribution | Impossible expected value | Validate by action and version assumptions |
| Cost omitted | Contribution calculated from revenue only | Fulfillment and discounts excluded | False profitability | Reconcile net price and incremental cost |
| Capacity ignored | Forecast exceeds available inventory | Demand is mistaken for fulfilled sales | Infeasible plan | Cap sales and model lost demand |
| Future effects ignored | Post-promotion demand falls | Customers pull purchases forward | Overstated incrementality | Extend horizon or quantify limitation |
| Share-only target | Unit share rises while margin falls | Incentive rewards volume regardless of economics | Value destruction | Use contribution and explicit constraints |
| Below-cost legal shortcut | Team calls any loss-making price illegal | Business and legal standards conflated | Bad escalation and missed context | Consult counsel on facts and law |
| Vendor data provenance unclear | Non-public competitor fields appear in feed | Sensitive information enters a common system | Legal and confidentiality risk | Stop use and seek legal review |
| Invitation to coordinate | Competitor asks for future-price assurance | Communication can support agreement | Serious competition-law exposure | Do not agree; stop and escalate |
| Pricing loop | Frequent alternating automated cuts | Rules react to each other's current output | Rapid margin erosion | Rate limits, bounds, and human review |
| Public-message misuse | Draft discusses desired rival response | Communication may be a coordination invitation | Enforcement and reputation risk | Counsel review and customer-focused purpose |
| Unvalidated treatment | Intervention overlaps a promotion | Price and marketing change together | Unidentified effect | Separate treatment or interpret combined effect |
| Overconfident point estimate | No intervals or scenario range | Sparse history hidden by precision | Fragile decision | Show ranges and decision thresholds |
| Forecast used as commitment | Procurement order issued from mean demand | Planning and authorization confused | Excess stock or shortage | Separate plan, target, and approval |
| Wrong time bucket | Daily observations paired with weekly quantities | Units and timing inconsistent | Invalid coefficient | Aggregate with documented rules |
| Unsupported vendor claim | Model promises to prevent all price wars | Strategic response is inherently uncertain | False assurance | Test assumptions and monitor outcomes |
| No override owner | Alert fires but no one can pause rule | Control exists only on paper | Continued losses | Assign authority, coverage, and escalation |

### Diagnosing an apparent escalation

First verify that each observed offer is comparable and in stock.
Then establish whether the price changes actually alternate or simply follow a common shock.
Calculate the incremental contribution of each own action against a consistent counterfactual.
Review whether the system's own decision rule amplifies noise.
Escalate suspicious communications separately from the commercial analysis.
Do not infer an agreement or predation solely from a time series of falling prices.

### What not to do

Do not ask the competitor why it cut or whether it will restore its price.
Do not tell a competitor that the firm will raise price if the competitor does likewise.
Do not route a proposed price understanding through a supplier or platform.
Do not obtain a competitor's private future price through a third party.
Do not discard evidence subject to a retention rule or legal hold.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or manage it |
|---|---|---|---|
| Prices are comparable | Demand depends on relevant alternatives | Different pack, quality, delivery, or eligibility | Audit offers and normalize units |
| Demand model is locally stable | Scenario arithmetic uses fixed coefficients | New product, regime shift, or unusual promotion | Backtest by period and bound extrapolation |
| Own-price effect is identified | Decision needs response to intervention | Price chosen in response to hidden demand | Use credible design and document uncertainty |
| Cross-price effect is identified | Rival price enters demand forecast | Shared shocks or correlated marketing | Examine timing and credible controls |
| Response probabilities are calibrated | Expected value weights branches | Small sample, structural change, or expert bias | Compare forecasts with outcomes and use ranges |
| Action-dependent response is plausible | Case assigns different conditional probabilities | No evidence that own action changes rival behavior | Report equal-probability sensitivity |
| Competitor action occurs within horizon | One-week branch represents a week | Delayed response or partial-week offer | Model intervals and response delay |
| Stock is available | Unconstrained demand equals sales | Inventory or fulfillment shortage | Cap sales and track unmet demand |
| Unit variable cost is constant | Simple contribution uses one cost | Overtime, freight, rebates, or tiered costs | Use scenario-specific cost curves |
| No pull-forward | Current gain represents incrementality | Customers buy early instead of later | Extend the measurement window |
| No cannibalization | Focal product contribution captures the impact | Own products lose sales | Model category-level contribution |
| No fixed execution fee | Simple calculation excludes offer cost | Campaign, platform, or setup fee | Subtract documented incremental fee |
| Rival alternatives are exhaustive | Two branch probabilities sum to one | Another price or non-price move occurs | Add branches or bound omitted states |
| Sales are nonnegative | Linear model is applied meaningfully | Extrapolation beyond valid price range | Restrict range and inspect predictions |
| Customers can observe the offer | Price affects choice | Display latency or eligibility restrictions | Measure exposure and actual checkout price |
| Rival cost and objective are unknown | Scenario model avoids false certainty | Analyst assumes a precise best response | Use ranges and lawful evidence only |
| Probabilities are independent judgments | Decision is unilateral | Sensitive information exchange or coordination | Restrict sources and obtain counsel review |
| One-week contribution is sufficient | Horizon captures relevant effects | Loyalty, inventory, or contracts persist | Extend horizon or label excluded effects |

An assumption ledger is not a waiver of accountability.
If a critical assumption is unsupported, widen the range or withhold a point recommendation.
The correct response to missing rival intent is not to request it from the rival.

## 12. Exercises

Work independently and state all units.
For design exercises, explain the intended deliverable and trade-offs rather than inventing a uniquely correct answer.

### Exercise 1 — distinguish the functions

A seller observes that demand rises when a rival's price rises.
It also observes that the rival often changes price within three days of the seller's own price change.
Explain which observation relates to demand and which relates to a reaction function.
**Deliverable:** two definitions and one warning about causal interpretation.
**Acceptance:** names the direction of each mapping and identifies at least one common shock.
**Time:** 8 minutes.
**Outcome:** 1.

### Exercise 2 — calculate scenario contribution

Use the Harbor demand model and a USD 14 per pack variable cost.
Calculate demand and weekly contribution at own USD 23 and rival USD 22.
Assume capacity is 12,000 packs and no fixed execution cost.
**Deliverable:** equation, units, and contribution.
**Acceptance:** 10,400 packs and USD 93,600 per week with correct arithmetic.
**Time:** 8 minutes.
**Outcome:** 2.

### Exercise 3 — calculate expected value

A hold action earns USD 104,000 if the rival holds and USD 96,000 if it cuts.
The conditional probability of the rival cut is 0.30.
A cut action earns USD 100,800 if the rival holds and USD 93,600 if it cuts.
The conditional probability of the rival cut after the cut action is 0.10.
**Deliverable:** expected contribution for each action and the higher-valued action.
**Acceptance:** USD 101,600 for hold, USD 100,080 for cut, and a USD 1,520 difference.
**Time:** 10 minutes.
**Outcome:** 2.

### Exercise 4 — find a threshold

Retain the Harbor scenario contributions and hold the rival-cut probability after Harbor cuts at 0.10.
Let the rival-cut probability after Harbor holds vary.
Find the probability at which the two actions tie.
Explain the decision on either side, assuming the other inputs stay fixed.
**Deliverable:** equation, threshold, and two conditional recommendations.
**Acceptance:** threshold 0.49; hold wins below it and cut wins above it.
**Time:** 12 minutes.
**Outcome:** 3.

### Exercise 5 — critique historical response

An analyst finds that rival prices declined within a week of 70% of 20 own-price cuts.
Most cuts occurred around a major supplier cost decrease and a seasonal event.
The analyst calls 0.70 the causal rival-response probability.
**Deliverable:** a critique and an independent evidence plan.
**Acceptance:** explains at least two identification problems, defines the event window, and proposes a comparison and uncertainty reporting.
**Time:** 18 minutes.
**Outcome:** 4.

### Exercise 6 — design a control

A price engine can update an online offer every 15 minutes.
It consumes public competitor data and can change the price without human approval.
**Deliverable:** a one-page control design with inputs, constraints, owners, alerts, and rollback.
**Acceptance:** includes comparable-offer validation, stale-data handling, cost and inventory limits, a human stop, monitoring, and a decision log.
**Time:** 20 minutes.
**Outcome:** 5.

### Exercise 7 — handle an inappropriate contact

At a trade event, a competitor says: “We will raise next month's price if you will too.”
The competitor asks for a yes or no.
**Deliverable:** immediate response and escalation steps.
**Acceptance:** does not agree or exchange future prices, ends the discussion, and promptly follows company counsel and reporting procedures.
**Time:** 8 minutes.
**Outcome:** 6.

### Exercise 8 — interpret case evidence

A manager says the RealPage complaint proves every algorithm using a competitor's public price is unlawful.
**Deliverable:** a three-part correction distinguishing allegation, data facts, and legal conclusion.
**Acceptance:** states that allegations and settlement proposals are not universal findings and calls for fact-specific review.
**Time:** 10 minutes.
**Outcome:** 7.

### Exercise 9 — write the decision memo

Use the base Harbor case and the 10,800-pack capacity variation.
Write a memo to pricing and supply owners that distinguishes forecast, plan, target, and commitment.
**Deliverable:** at most 250 words including the threshold and a conditional recommendation.
**Acceptance:** cites USD 2,480 expected advantage under base assumptions, the capacity reversal, and an evidence limitation.
**Time:** 15 minutes.
**Outcome:** 8.

## 13. Self-check answers

### Exercise 1

The demand function maps own price, rival price, and context to expected customer demand.
The cross-price observation concerns that function.
The reaction function maps the seller's observed action and other information to a rival action or distribution.
The three-day observation concerns that second relationship.
A supplier-cost change or seasonal event may cause both firms' prices to move.
Neither observation alone establishes a causal mechanism.

### Exercise 2

At own USD 23 and rival USD 22:

```text
Demand = 20,000 - 800*23 + 400*22 = 10,400 packs/week
Unit contribution = USD 23 - USD 14 = USD 9/pack
Contribution = 10,400 * USD 9 = USD 93,600/week
```

Capacity is 12,000 packs, so it does not bind.
The answer is contribution before omitted fixed and other costs, not accounting profit.

### Exercise 3

For holding, `0.70*104,000 + 0.30*96,000 = USD 101,600`.
For cutting, `0.90*100,800 + 0.10*93,600 = USD 100,080`.
Holding has USD 1,520 more expected contribution per week.
The different conditional probabilities must remain attached to the correct actions.

### Exercise 4

The expected hold contribution is `104,000 - 8,000*h`.
The expected cut contribution is `100,800 - 7,200*0.10 = 100,080`.
Equating them gives `8,000*h = 3,920`, so `h = 0.49`.
Hold wins below 0.49 and cut wins above 0.49, given all other assumptions.
The threshold is not evidence that the actual response probability is above or below it.

### Exercise 5 rubric

A strong critique recognizes the common supplier-cost shock and the seasonal event.
It notes that the firm's own decision to cut may be selected based on expected demand.
It states that 14 of 20 events is a small, descriptive sample, not a causal estimate.
It defines response magnitude, timing, and effective price before analysis.
A stronger design compares appropriate non-cut periods or markets and checks pre-event trends.
It discusses spillovers, missing prices, and uncertainty intervals.
It identifies which assumptions would be needed to interpret an estimated difference causally.
It does not seek confidential future-price plans from the rival.

### Exercise 6 rubric

A strong design checks source, freshness, product, pack, channel, eligibility, and availability.
It sets approved price ranges, update frequency, inventory and cost constraints, and ownership.
It pauses or escalates on missing data, unusual moves, or rapid repeated cuts.
It logs the input snapshot, decision rule version, approved action, and override.
It identifies a person who can stop and reverse a change subject to customer-facing obligations.
It separates legal review from the operational decision and tests the process before deployment.

### Exercise 7

Do not agree or discuss future pricing.
End the discussion and promptly report it through company competition-law procedures to counsel.
Preserve relevant records according to counsel's direction and company policy.
Do not independently investigate by seeking additional competitor information.
The precise escalation procedure and legal implications require jurisdiction-specific advice.

### Exercise 8

The complaint contains allegations by the plaintiffs, not a universal finding about all pricing tools.
The case concerns alleged use of non-public competitively sensitive information and other alleged coordination-related conduct.
A system that independently observes public customer offers has different facts and is not automatically lawful or unlawful on that fact alone.
The proposed settlement has a case-specific scope and procedural status.
Counsel should review actual data flows, conduct, applicable law, and current case documents.

### Exercise 9 rubric

The memo identifies expected base contribution of USD 100,080 for the cut versus USD 97,600 for the hold.
It states that the USD 2,480 advantage depends on action-specific probability assumptions.
It reports that at 10,800 packs of capacity, cut contribution is USD 96,840 versus USD 97,600 for hold.
It gives the 0.49 hold-response threshold when the cut-response probability is 0.10.
It labels forecast, plan, target, and commitment separately.
It assigns a decision owner and a supply-capacity confirmation.
It describes a review trigger and the independent-decision legal boundary.
A memo that states a forecast as a guaranteed outcome does not meet the rubric.

## 14. Interview questions

1. **How is a competitor reaction function different from cross-price elasticity?**
   A strong answer distinguishes rival action from customer demand and defines the conditioning information.
   A weak answer treats both as one demand coefficient.
2. **Why might a static price comparison select a different action from a reaction-aware model?**
   A strong answer explains action-dependent outcome probabilities and demonstrates expected contribution.
   A weak answer assumes a competitor's future price is fixed or known.
3. **What makes a historical match rate noncausal?**
   A strong answer identifies shared shocks, anticipation, selection, timing, and uncertainty.
   A weak answer says correlation becomes causation after adding any control variable.
4. **How do you compare two offers with different pack sizes?**
   A strong answer normalizes units while checking quality, availability, eligibility, and delivery.
   A weak answer compares only displayed case prices.
5. **When can a stockout make an elasticity estimate misleading?**
   A strong answer explains censored sales and the distinction between demand and fulfillment.
   A weak answer assumes every unsold unit indicates zero willingness to buy.
6. **What is the value of a decision threshold?**
   A strong answer derives the point where actions tie and states what evidence would move belief across it.
   A weak answer mistakes the threshold for an estimate of the true probability.
7. **How do you include a delayed rival response?**
   A strong answer partitions the horizon, updates inventory, and weights paths by timing.
   A weak answer assigns the final rival price to all periods without justification.
8. **What distinguishes a forecast from a commitment?**
   A strong answer identifies an uncertain expectation versus an authorized operational obligation.
   A weak answer treats a demand model output as an automatic purchase order.
9. **What controls would you add to an automated pricing engine?**
   A strong answer covers data validity, bounds, inventory, rate limits, logs, ownership, and human intervention.
   A weak answer relies solely on a model score or a blanket promise of legal compliance.
10. **What should an employee do after a rival proposes a coordinated increase?**
    A strong answer declines discussion, avoids agreement or exchange, and escalates promptly under policy.
    A weak answer continues negotiating or assumes a public setting makes the proposal safe.
11. **What does the RealPage litigation establish about all pricing algorithms?**
    A strong answer distinguishes allegations, case-specific procedural status, and different public-data scenarios.
    A weak answer asserts either that all algorithms are illegal or that all are safe.
12. **How would you respond to an apparent price war?**
    A strong answer validates offers, measures contribution and operational constraints, and independently chooses a response.
    A weak answer automatically matches or seeks a shared pricing understanding.

## 15. Key terms

- [Action-dependent probability](#step-7-create-coherent-scenario-paths): probability conditional on the firm's chosen action.
- [Availability](#step-8-translate-each-path-into-demand-and-feasible-sales): whether the customer can actually purchase the offer.
- [Best response](#step-4-separate-customer-response-from-competitor-response): modeled optimal action given another actor's action.
- [Capacity](#step-8-translate-each-path-into-demand-and-feasible-sales): feasible units in the specified period.
- [Censoring](#step-8-translate-each-path-into-demand-and-feasible-sales): limitation of observed sales by stock or capacity.
- [Commitment](#step-12-implement-monitor-and-learn): authorized obligation, distinct from a forecast.
- [Common shock](#step-5-examine-identification-before-estimating): event influencing multiple firms at once.
- [Contribution](#step-9-calculate-contribution-and-opportunity-costs): net price less specified variable and action costs.
- [Cross-price effect](#step-4-separate-customer-response-from-competitor-response): demand change associated with a rival price change.
- [Decision horizon](#step-1-frame-the-decision): period over which consequences are compared.
- [Demand function](#two-distinct-functions): mapping from prices and context to expected demand.
- [Endogeneity](#step-5-examine-identification-before-estimating): correlation between an input and unobserved determinants of the outcome.
- [Expected contribution](#step-d-compute-expected-contribution): probability-weighted contribution across scenarios.
- [Forecast](#step-g-write-the-decision-memo): uncertain prediction, not a promise.
- [GTIN](#data-needed-for-an-independent-decision): GS1 trade-item identifier, not proof of offer equivalence.
- [Hub-and-spoke](#united-states-competition-law): possible coordination involving an intermediary and participants.
- [Incrementality](#step-9-calculate-contribution-and-opportunity-costs): difference from a defined counterfactual.
- [Net price](#step-2-construct-comparable-observed-prices): realized price after modeled discounts and rebates.
- [Own-price effect](#step-4-separate-customer-response-from-competitor-response): demand change from an own-price change.
- [Plan](#step-g-write-the-decision-memo): selected course of action, conditional on approval and constraints.
- [Price gap](#step-2-construct-comparable-observed-prices): stated difference or ratio between comparable prices.
- [Price war](#1-why-this-topic-exists): repeated aggressive reductions with economic consequences.
- [Reaction function](#two-distinct-functions): mapping from observed state or action to rival action or probability.
- [Scenario](#step-7-create-coherent-scenario-paths): coherent combination of actions and conditions.
- [Signaling](#european-union-and-other-jurisdictions): communication or conduct conveying information about intended behavior.
- [Spillover](#step-5-examine-identification-before-estimating): an intervention's effect outside its intended treatment units.
- [Target](#step-g-write-the-decision-memo): desired outcome, not automatically a forecast or commitment.
- [Threshold](#step-e-calculate-the-decision-threshold): assumption value at which competing actions tie.
- [Variable cost](#step-9-calculate-contribution-and-opportunity-costs): cost changing with units or the decision.

## 16. Sources

The legal sources below are educational starting points, not legal advice.
Case allegations, settlements, and judicial findings have different evidentiary status.
Check current case records and jurisdiction-specific law before making a legal determination.

1. U.S. Congress, [Sherman Act, 15 U.S.C. §§ 1–2](https://uscode.house.gov/view.xhtml?path=/prelim@title15/chapter1&edition=prelim). Statutory text on agreements and monopolization.
2. U.S. Congress, [Federal Trade Commission Act, 15 U.S.C. § 45](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section45&num=0&edition=prelim). Statutory text on unfair methods of competition.
3. FTC, [Price Fixing](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/dealings-competitors/price-fixing). Agency explanation of competitor agreements and related conduct.
4. DOJ, [Justice Department Sues RealPage for Algorithmic Pricing Scheme That Harms Millions of American Renters](https://www.justice.gov/opa/pr/justice-department-sues-realpage-algorithmic-pricing-scheme-harms-millions-american-renters), August 23, 2024. Source for the government's allegations, not a final merits finding.
5. DOJ, [RealPage Proposed Final Judgment](https://www.justice.gov/opa/media/1419406/dl), November 2025. Source for the proposed settlement; consult the court docket for current status.
6. DOJ, [Former E-Commerce Executive Charged with Price Fixing in the Antitrust Division's First Online Marketplace Prosecution](https://www.justice.gov/opa/pr/former-e-commerce-executive-charged-price-fixing-antitrust-divisions-first-online-marketplace), April 6, 2015. Government account of the Topkins plea and alleged agreement.
7. DOJ, [United States v. Airline Tariff Publishing Company et al.](https://www.justice.gov/atr/case/us-v-airline-tariff-publishing-company-et-al). Case documents and consent decrees concerning airline fare dissemination.
8. DOJ, [Justice Department Withdraws Outdated Enforcement Policy Statements](https://www.justice.gov/opa/pr/justice-department-withdraws-outdated-enforcement-policy-statements), February 3, 2023. Context on withdrawn healthcare policy statements.
9. European Union, [Treaty on the Functioning of the European Union, Articles 101 and 102](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:12012E/TXT). Primary treaty text.
10. European Commission, [Guidelines on the Applicability of Article 101 of the Treaty on the Functioning of the European Union to Horizontal Co-operation Agreements](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52023XC0721(01)), 2023. Guidance including information exchange.
11. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Identifier definition and scope.
12. J. D. C. Little, “Models and Managers: The Concept of a Decision Calculus,” *Management Science* 16(8), 1970. [DOI: 10.1287/mnsc.16.8.B466](https://doi.org/10.1287/mnsc.16.8.B466). Decision-model design and managerial use.
13. V. R. Nijs, M. G. Dekimpe, J.-B. E. M. Steenkamp, and D. M. Hanssens, “The Category-Demand Effects of Price Promotions,” *Marketing Science* 20(1), 2001. [DOI: 10.1287/mksc.20.1.1.10197](https://doi.org/10.1287/mksc.20.1.1.10197). Evidence on category promotion effects and their timing.
14. G. E. P. Box, “Science and Statistics,” *Journal of the American Statistical Association* 71(356), 1976. [DOI: 10.1080/01621459.1976.10480949](https://doi.org/10.1080/01621459.1976.10480949). Context for model assumptions and checking.

## 17. Links

- [Module 16 topic index](_index.md).
- [Module 16 survey and lab context](../../../course/16_pricing_promotions_revenue_management.md).
- [Pricing decision stack](01_pricing-decision-stack.md).
- [Own-price and cross-price elasticity](03_elasticity-identification.md).
- [Promotion optimization and trade spend](06_promotion-trade-spend.md).
- [Price testing](10_price-testing.md).
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).

Sibling topic links follow the module index; some may remain planned until their assigned files are published.
