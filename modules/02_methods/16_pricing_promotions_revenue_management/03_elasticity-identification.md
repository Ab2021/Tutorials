# Own-Price and Cross-Price Elasticity

> Module 16 — Pricing, Promotions, and Revenue Management  
> Topic 03 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 04, 05, and 07; contribution-margin arithmetic  
> Version 2026.1

## 1. Why this topic exists

A category manager must decide whether to raise the shelf price of a popular product.
The finance team can calculate the extra margin on each unit sold.
It cannot calculate the lost units without a demand response.
The supply planner needs that response before reserving inventory.
The retailer also needs to know whether buyers will migrate to a substitute or leave the category.

Historical prices and sales are available, but price was not assigned by a neutral mechanism.
The commercial team discounted before holidays, when demand was already expected to increase.
It also funded extra displays in some discounted weeks.
A simple comparison of those weeks with ordinary weeks combines the price effect with seasonality and merchandising.
It might make a discount look more effective than the price change alone actually is.
In another setting a firm may raise prices during a demand surge, making price and sales move together even when a price increase reduces demand.
Neither direction of bias is guaranteed without a model of the pricing process.

Elasticity expresses a percentage demand response to a percentage price change.
Own-price elasticity concerns the product whose price changes.
Cross-price elasticity concerns another product's demand when the focal price changes.
Both are useful only when the outcome, price, population, time horizon and intervention are specified.
An elasticity for a coupon plus a prominent display is not automatically the elasticity of an unadvertised shelf-price change.

The commercial choice is not merely to produce a coefficient.
You need to decide whether evidence supports a particular action, whether a limited test is warranted, or whether a decision should wait.
A good estimate is accompanied by uncertainty, economic exposure, implementation constraints and a plan for learning.
A precise historical correlation is not a substitute for a credible causal comparison.

This topic follows one fictional retailer through a small randomized price test.
The example is illustrative rather than evidence about an actual retailer.
It shows why unit response, substitution, net contribution and supply capacity belong in the same decision packet.
It also shows why the randomized effect is local to the tested price change and implementation conditions.

## 2. Learning outcomes

After this topic you can:

1. Compute own-price and cross-price elasticities from an explicitly defined intervention and interpret their signs.
   Exercise 1 checks both calculations.
2. Draw a causal diagram showing demand shocks, price decisions and merchandising as sources of confounding.
   Exercise 2 asks you to distinguish a causal effect from an observed association.
3. Specify a randomized price experiment, its estimand, unit of assignment and operational guardrails.
   Exercise 3 asks you to design a cluster test.
4. Calculate an intention-to-treat effect and an experiment-based finite-change elasticity.
   Exercise 4 uses store-level outcomes.
5. Diagnose when fixed effects, difference-in-differences or an instrumental variable might and might not identify a price effect.
   Exercise 5 asks you to critique identification strategies.
6. Translate own and cross responses into incremental category contribution with uncertainty and constraints.
   Exercise 6 asks you to calculate the decision economics.
7. Explain censoring, interference, dynamic response and lack of support as limits on transport to a new decision.
   Exercise 7 asks you to audit a transfer request.
8. Distinguish an evidence-based forecast, an approved plan, a target and a commitment.
   Exercise 8 asks you to label each statement.

## 3. The core mental model

Think of a price intervention as one input to a system, not as a label attached to a sales row.
The intervention changes what eligible buyers face; buyers then choose quantities and alternatives.
The outcome recorded in a transaction system is constrained by availability and measurement.
The relevant comparison is the same population under another feasible intervention, not another population selected for convenience.

```text
             expected demand, season, local events
                 |                     |
                 v                     v
       manager chooses price -------> sales of A
                 |                     ^
                 v                     |
        display / coupon -----------> choices
                                       |   \
                                       v    v
                                  sales of B  category exit
                                       ^
                              availability and exposure

 Random assignment -> offered price -> customer choices -> units and contribution
       |                   |
       +-> assignment log  +-> implementation / actual receipt
```

The top half illustrates the observational problem.
A manager's expectation can influence both the chosen price and sales.
That common cause creates an association even if the price itself has no effect.
Display and coupons may accompany the price change, making the effect of price alone difficult to separate.
Stockouts can prevent high underlying demand from appearing in recorded sales.

The bottom half illustrates a randomized offer.
Assignment is created before outcomes and, if implemented correctly, is independent of potential outcomes by design.
The assignment changes the price offered; it does not guarantee every transaction sees or pays that price.
Analyze by original assignment first so that selective execution does not undo the randomized comparison.
A comparison by actual paid price can be confounded by customer choice, eligibility and implementation.

For product A, define own elasticity as the proportional change in demand for A divided by the proportional change in A's price.
For product B, define cross elasticity as the proportional change in demand for B divided by the proportional change in A's price.
A negative own elasticity is typical but not a mathematical necessity in every observed dataset.
A positive cross elasticity is consistent with B substituting for A when A becomes more expensive.
A negative cross elasticity is consistent with complementarity, such as a product used with A.
The sign is conditional on the intervention and need not remain the same across horizons or market contexts.

Elasticity is a ratio, not a direct profit calculation.
A rise in A's price can lower A units and still increase A contribution.
Customers switching to B can add or subtract category contribution depending on B's economics.
A lower price can increase units yet destroy contribution because the lower margin applies to units that would have sold anyway.
A price test should therefore record outcomes for the entire decision scope, not just the promoted item.

The experiment identifies a comparison between tested price policies under its design and execution.
It does not reveal every point on a demand curve.
It does not establish how a competitor would respond to a permanent national rollout.
It does not license inference about a different customer group without evidence of transportability.
The decision owner must connect the estimated effect to the next feasible decision and its constraints.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Demand | Quantity customers would purchase under specified conditions, in units per period. | Equating latent demand with sales when inventory is unavailable. |
| Sales | Observed completed units per period after stock, returns and recording conventions. | Calling all lost sales zero demand. |
| Price | The specified amount per unit faced or paid, in currency per unit. | Mixing list price, shelf price and net receipt. |
| Own-price elasticity | Proportional response of demand for A divided by proportional change in A price; dimensionless. | Reporting a raw unit change as an elasticity. |
| Cross-price elasticity | Proportional response of B demand divided by proportional change in A price; dimensionless. | Reversing the outcome and price indices. |
| Point elasticity | Derivative of log demand with respect to log price at a point. | Treating a large observed change as infinitesimal. |
| Arc elasticity | Finite-change response using midpoint denominators. | Comparing it directly with a baseline-denominator ratio without saying so. |
| Baseline ratio | Finite percentage quantity change from baseline divided by finite percentage price change from baseline. | Calling it an exact derivative. |
| Counterfactual | Outcome for the same unit under an alternative treatment. | Treating a selected historical period as automatically comparable. |
| Potential outcome | Outcome a unit would have under a specified treatment assignment. | Assuming both potential outcomes are observed for the same unit and time. |
| Estimand | Precisely defined population-level quantity to estimate. | Confusing the estimand with an estimator or business target. |
| Estimator | Rule applied to observed data to estimate the estimand. | Assuming any regression coefficient is causal by definition. |
| Confounder | Common cause of treatment and outcome, or a variable needed for exchangeability under a causal model. | Calling every correlated variable a confounder. |
| Endogeneity | Dependence between the price regressor and unobserved outcome determinants in a specified model. | Assuming fixed effects always eliminate it. |
| Randomization | Chance-based allocation according to a known rule. | Calling a manager's arbitrary choice randomized. |
| Cluster | Group assigned together, such as a store or market. | Treating every transaction as an independently randomized unit. |
| Intention to treat (ITT) | Comparison by assigned policy regardless of receipt. | Dropping noncompliant units after randomization. |
| Compliance | Degree to which implementation matches assigned treatment. | Assuming a compliance rate proves the delivered-price effect. |
| Interference | One unit's outcome depends on another unit's assignment. | Assuming nearby stores cannot share customers. |
| Stockout censoring | Observed sales limited by unavailable stock. | Interpreting constrained sales as unconstrained demand. |
| Fixed effect | Model parameter absorbing a specified group or time intercept. | Assuming it absorbs time-varying local shocks. |
| Difference-in-differences (DiD) | Difference between treatment and control outcome changes over time. | Claiming causality without a defensible counterfactual trend. |
| Instrumental variable (IV) | Variable used to isolate treatment variation under relevance, exclusion and independence assumptions. | Calling a cost series valid without examining its other pathways. |
| First stage | Relationship between instrument and treatment in an IV design. | Treating a strong first stage as proof of exclusion. |
| Contribution | Net revenue less specified avoidable variable costs, in currency per period. | Substituting gross revenue for incremental profit. |
| Cannibalization | Sales displaced from another product by an intervention. | Counting every contemporaneous decrease as caused by the intervention. |
| Pull-forward | Earlier purchase displacing a later purchase. | Measuring a one-week lift as permanent incremental consumption. |
| Confidence interval | A procedure-based interval with a stated long-run coverage property. | Reading a 95% interval as a 95% probability for a fixed parameter without a Bayesian model. |
| Forecast | Conditional estimate of a future outcome, including uncertainty. | Treating it as an approved plan or a guarantee. |
| Plan | An approved set of intended operational actions and quantities. | Calling an analytical scenario an authorized commitment. |
| Target | Desired performance level. | Presenting ambition as an unbiased forecast. |
| Commitment | A binding or explicitly accepted obligation to deliver, buy or execute. | Assuming a demand forecast commits the supply team. |

## 5. The mechanics, step by step

### Step 1 — Define the decision and the intervention

Start with the decision that a named owner can actually make.
Write the current and proposed prices in the same currency, tax basis and unit of measure.
Specify the effective start and end times, eligible customers, channel, product and geography.
Say whether a shelf label, coupon, delivery fee or net invoice changes.
An online personalized coupon and a storewide shelf-price change are different treatments.

Decide whether other marketing conditions are held fixed or intentionally included.
If a display is part of the business package, the experiment can estimate the package effect.
Label that effect as a package effect rather than attributing all incremental sales to price.
If price alone is the decision lever, keep display and other treatments constant or independently vary them.
A two-factor experiment can separate effects and interactions if it has adequate power and execution discipline.

Specify the horizon of interest before inspecting results.
A weekly effect may include pantry loading that disappears over the following month.
A permanent list-price decision needs more than an event-week response.
Specify whether the decision is local to tested stores or meant to generalize to a wider population.
The output of this step is a short intervention card with owner, alternatives, eligibility and horizon.

### Step 2 — Define quantities and elasticities

Let A be the product whose price changes and B a potential substitute.
Write $Q_A(p_A,p_B,\ldots)$ for A units per store-week under a defined price and context.
Write $Q_B(p_A,p_B,\ldots)$ for B units on the same geographic and temporal basis.
For a differentiable demand function, own elasticity is

$$\varepsilon_{AA}=\frac{\partial Q_A}{\partial p_A}\frac{p_A}{Q_A}=\frac{\partial\log Q_A}{\partial\log p_A}.$$

Cross elasticity of B with respect to A's price is

$$\varepsilon_{BA}=\frac{\partial Q_B}{\partial p_A}\frac{p_A}{Q_B}=\frac{\partial\log Q_B}{\partial\log p_A}.$$

The first subscript names the outcome; the second names the price.
Hold the other elements of the intervention fixed in the causal definition.
Elasticities have no physical units, but the input quantities and time buckets must be stated.
The response is not necessarily symmetric: $\varepsilon_{BA}$ need not equal $\varepsilon_{AB}$.
The two products can have different baseline units, buyer populations and price points.

For a discrete randomized comparison, define a baseline-denominator ratio:

$$E_{BA}^{base}=\frac{(Q_B^1-Q_B^0)/Q_B^0}{(p_A^1-p_A^0)/p_A^0}.$$

The superscripts indicate policies 1 and 0, not a before and after time comparison.
The same construction with A quantities gives the own-price finite-change ratio.
This ratio describes the tested interval and is not necessarily the point elasticity at either endpoint.
If either baseline quantity or baseline price is zero, this percentage ratio is undefined.
Use a different outcome scale or a model designed for zeros rather than adding arbitrary constants silently.

For midpoint or arc elasticity use

$$E_{BA}^{arc}=\frac{(Q_B^1-Q_B^0)/[(Q_B^1+Q_B^0)/2]}{(p_A^1-p_A^0)/[(p_A^1+p_A^0)/2]}.$$

The midpoint formula gives the same result when the endpoints are swapped.
It answers a different scaling question from the baseline ratio.
Report the convention with every value and avoid spurious precision.

### Step 3 — Draw the causal process before fitting a model

List who sets the price and what information they use.
Ask whether an expected demand increase, a competitor promotion or a stock imbalance influences that choice.
Map each such driver to both price assignment and the sales outcome when appropriate.
Write down which drivers are observed and which are not.
The causal diagram is a hypothesis about the operational process, not proof that the assumptions hold.

Separate variables measured before assignment from variables affected by assignment.
A pre-period local holiday calendar can be useful for design or adjustment.
The number of customers entering after seeing a price can be part of the price effect.
Controlling for that post-treatment traffic can remove a real pathway and change the estimand.
Likewise, adjusting for a stockout caused by the treatment can introduce selection bias.
Prevent or measure availability problems by design and report their consequences rather than conditioning casually.

Sales and price can be jointly determined.
When a heat wave increases demand and a retailer raises price in anticipation, a positive correlation may reflect the weather shock.
When a price cut is deployed with extra advertising, the combined outcome can overstate the price-only response.
When inventory is tight, management may raise price and observed sales may be capped; the resulting bias need not have the same direction.
A sign of bias requires reasoning about the specific mechanism, not a universal rule.

### Step 4 — Select a credible identification design

Prefer a feasible randomized intervention when price authority, ethics, customer rules and operations permit it.
For physical retail, assign stores or non-overlapping catchments to price policies.
For a digital offer, assignment may be at customer or account level if consistent treatment can be delivered and spillovers are acceptable.
For a shared platform with time-varying pricing, randomized time blocks may be appropriate if carryover and time patterns are addressed.
The randomization unit should reflect how the price is actually implemented and how interference can occur.

Randomize within pre-specified strata such as region and baseline sales when useful.
Record the random seed, algorithm, eligible population, assignment timestamp and assignment probabilities.
Keep all eligible randomized units in the analysis even when a store fails to execute its assigned price.
Collect pre-period measurements for precision and balance checks, not as a license to search for a favorable specification.
Cluster standard errors and power calculations at the randomization unit or a defensible higher level.
A million transactions do not make 12 independently randomized stores into a million independent experiments.

If randomization is unavailable, evaluate observational designs against the actual decision process.
Store fixed effects remove stable store-level differences in the model, such as a persistent baseline demand level.
Week fixed effects remove common shocks shared by all stores in that week.
Neither removes a store-specific, time-varying demand shock that also drives its price.
A local event known to a manager but missing from the data remains a plausible source of bias.

A DiD design compares the change in treated units with the change in comparable control units.
Its central identifying assumption is that, absent treatment, the treated group's average outcome would have followed the control group's trend after the chosen adjustments.
Pre-period trend comparisons can reveal some incompatibilities but cannot prove the counterfactual post-period trend.
Anticipation, different interventions at the same time, changing composition and spillovers can invalidate the comparison.
Staggered timing and heterogeneous effects require an estimator appropriate to that design rather than an automatic two-way fixed-effect regression.

An IV design uses an external source of price variation rather than the price variation itself.
The instrument must predict the price change in the relevant population.
It must also be as good as independent of unobserved demand determinants after any valid adjustment.
Its effect on the outcome must operate through the specified price channel, not through another direct or indirect path excluded by the model.
A supplier cost change may move price, but it may also affect availability, quality, advertising or competitors' costs.
Those pathways make exclusion a substantive claim requiring institutional evidence, not a regression diagnostic.
Weak instruments can produce unstable and misleading estimates; assess instrument strength using methods appropriate to the model.
With heterogeneous responses, a valid IV may identify a local response for units whose price changes because of the instrument, not the whole market.
State the estimand and additional assumptions rather than calling any IV estimate a universal elasticity.

### Step 5 — Design outcomes and data collection

Choose the primary outcome before launch.
For a policy test it may be A units per eligible store-week or incremental category contribution per eligible store-week.
Specify the exact numerator, denominator, currency and treatment of tax, returns and rebates.
For a cross-price question, include B units even in stores with zero transactions.
The denominator must include all eligible stores and periods, not only those with sales.

Record both assigned price and actual customer-facing price over time.
A transaction-weighted average paid price can change because buyers, coupons or basket composition change.
It is not necessarily the assigned shelf price or a valid randomized treatment variable.
Capture price-label checks and price-file timestamps separately from transaction receipts.
If policy compliance is poor, report the policy ITT and investigate implementation.
Do not silently exclude failed stores or interpret a ratio using actual paid price as a randomized causal effect.

Record availability, on-hand inventory, delivery interruptions and store closure hours.
Include pre-specified related products and category units to measure substitution and category expansion.
Include post-test periods when pull-forward or delayed competitor response is plausible.
A return recorded in a later period requires a consistent attribution and maturity rule.
Freeze the outcome definition and data-extraction version before the primary readout.

### Step 6 — Plan scale, power and safeguards

Estimate baseline variation at the cluster and time level from prior data.
Use a minimum economically meaningful contribution effect, not a convenient percentage alone, to frame power.
Translate that effect into plausible unit responses and estimate variance under realistic intracluster correlation.
Simulate the proposed assignment and analysis, including exclusions fixed before treatment.
Power is the probability of meeting a specified detection criterion under a specified alternative; it is not a guarantee of a useful estimate.
If the number of independent stores is small, a long transaction table may still yield a wide interval.

Pre-register primary outcome, contrast, assignment, duration and decision rule.
Specify secondary outcomes and how multiple comparisons will be interpreted.
If a test looks at many products and segments, some apparent effects can occur by chance.
Plan either multiplicity control for confirmatory claims or label those analyses exploratory.
Stopping repeatedly whenever the result looks favorable can distort ordinary fixed-horizon inference.
Use an approved sequential design if early analysis is required, including valid stopping boundaries.

Agree on a maximum customer harm, stockout rate, price-file error rate and contribution loss that trigger review.
A safety stop is not the same as claiming a conclusive effect from an unplanned early read.
Assign owners for implementation, customer communication, supply monitoring and incident escalation.
Legal and privacy specialists should review the design before launch when customer-level pricing or sensitive data is involved.
The output is a signed test brief, not an instruction to change a price without authorization.

### Step 7 — Estimate policy effects

Compute mean outcomes in the assigned treatment and control groups on the same time and unit basis.
The ITT contrast is treatment mean minus control mean.
For a completely randomized balanced design, the unadjusted difference in means is transparent and unbiased for the corresponding finite-population assignment effect under the design assumptions.
For stratified or unequal-probability assignment, use an estimator and weights consistent with the assignment scheme and target population.
If pre-period adjustment is used, specify it in advance and use design-appropriate inference.
Report absolute unit effects as well as percentage effects; percentages can become unstable near zero.

For a known policy price change, divide the relative ITT quantity response by the relative assigned price change to obtain a policy finite-change elasticity.
This is not necessarily the elasticity of a change in actual transaction price when compliance is imperfect.
If different stores face different assigned price changes, define the target contrast and weighting explicitly instead of dividing incompatible averages.
Uncertainty in the outcome means and their correlation propagates into the elasticity ratio.
A cluster bootstrap or randomization inference can be suitable if it matches the assignment and has sufficient independent units.
If the denominator is uncertain or close to zero, ratio inference requires particular care.

Report a confidence interval for the primary decision outcome, not only a point estimate and a significance label.
If the interval includes both economically favorable and unfavorable outcomes, state that the decision remains uncertain.
Avoid claiming no effect merely because a test is not statistically significant.
Report execution, missing data, stock availability and any deviations alongside the estimate.
An anomalous null or positive own-price response should trigger an execution and context review, not automatic deletion.

### Step 8 — Translate the response into a commercial decision

Construct incremental contribution over the decision scope.
For two items with constant variable cost and no other changes:

$$\Delta C=(p_A^1-c_A)Q_A^1-(p_A^0-c_A)Q_A^0+(p_B-c_B)(Q_B^1-Q_B^0)-K.$$

Here $K$ is incremental fixed or event-specific cost over the same period.
Net receipts, funding, handling, returns and variable fulfillment costs belong in the relevant unit margins.
Do not count the same rebate as both a price reduction and a separate cost.
If inventory, delivery capacity or service levels change, include those effects in scenarios or constraints.
The equation assumes B's price and unit cost are unchanged; revise it if they are not.

Calculate the break-even A volume after accounting for expected B migration and incremental cost.
Stress the joint distribution of A and B outcomes rather than combining two unrelated worst cases and labeling it a confidence interval.
Price and unit cost may also vary, and their uncertainty matters to the decision.
Report category economics, not just the apparent success of A.
If the proposed price affects other products or future weeks, expand the accounting boundary before scaling.

Distinguish the forecast from the plan.
The forecast is a conditional range of units and contribution if the proposed policy is implemented.
The plan is the approved price, supply and operational action.
The target is a desired performance benchmark and need not equal the forecast.
A commitment is a separately accepted obligation, such as a signed volume contract.
Do not let an optimistic target silently replace the model's forecast in capacity planning.

### Step 9 — Monitor and decide whether to transfer

Compare test and rollout on geography, customer mix, competitor activity, price magnitude, channel and time horizon.
The more the new decision differs, the more speculative a direct transfer becomes.
A short local test cannot automatically predict a permanent nationwide change with competitive responses.
A price increase and an equal-sized price decrease may have different effects because reference prices and switching costs differ.
The baseline ratio is local to the tested endpoints even if a fitted constant-elasticity model extends further.

Check actual execution and outcomes against the assumptions behind the forecast.
Define review triggers before launch, including out-of-range prices, changing supply and a category contribution shortfall.
Use new evidence to update the decision, preserving the original experiment and its analysis.
If the decision lacks support, propose another appropriately sized test rather than extrapolating until the economics look attractive.
Archive the assignment, code, data definitions, approval and limitations so another analyst can reproduce the decision.

### Step 10 — Choose the experimental contrast deliberately

An experiment can answer only questions represented by its assigned policies.
Suppose a business considers a $9.50, $10.00 or $10.45 price per pack.
A two-arm test of $9.50 versus $10.45 directly estimates only that contrast.
It does not separately reveal the response from $9.50 to $10.00.
Adding a third arm allows additional contrasts but leaves fewer clusters per arm at a fixed sample size.
That trade-off should be resolved during design, using decision value and power.

If a business wants to estimate price and display separately, consider four policy cells.
The cells are low price without display, low price with display, high price without display, and high price with display.
Each cell represents an actual operational package.
Randomize eligible clusters to those cells using a documented allocation rule.
The contrast between high and low price without display estimates the price effect without display.
The analogous contrast with display estimates the price effect when display is present.
The difference between those two contrasts is an interaction on the selected outcome scale.
A nonzero interaction means the combined effect is not the sum of two invariant main effects.

The design requires enough independent clusters in each cell.
If display cannot be controlled independently of price, the price-only effect may not be separately identified.
It is better to estimate the feasible package honestly than label the bundle as a price-only intervention.
Make sure the customer-facing signage does not inadvertently announce one arm differently.
If a label or display necessarily changes with price, define the intervention to include it.
That intervention can still be useful if it matches the actual commercial choice.

For online experiments, define how the policy persists when a buyer returns on another device.
If the same buyer alternates between prices, the intervention becomes inconsistent and customer trust may suffer.
Account-level assignment can improve consistency but may require identity resolution and privacy review.
Shared devices and households can also create interference across assigned accounts.
Specify eligibility and the point of assignment before a buyer sees the offer.
Excluding buyers after they do not click or purchase conditions on a post-assignment event.

For switchbacks, randomize time blocks rather than selecting high-demand hours for a preferred price.
Balance blocks across day of week and time of day where the design permits.
Set a block length long enough for the operational system to settle.
If effects carry into later blocks, specify a washout or model the carryover explicitly.
Longer blocks reduce rapid switching but reduce the number of independent policy changes.
Use inference that reflects the time-series dependence and the randomization schedule.
A switchback does not automatically solve customer memory or competitor reaction.

### Step 11 — Separate statistical precision from decision value

The point estimate is one possible summary of experimental outcomes.
Its uncertainty is essential because the business decision concerns future consequences.
For each cluster, calculate the chosen contribution outcome for the full pre-specified period.
Compare assigned policy means using the allocation and weighting dictated by the design.
Retain the relationship between A and B outcomes within a cluster when estimating category contribution.
Analyzing their sum directly preserves that covariance.

Imagine an estimated incremental contribution of +$111 per store-week.
An interval from −$40 to +$260 per store-week would leave meaningful downside and upside.
An interval from +$80 to +$140 per store-week would support a different risk assessment.
These intervals are hypothetical examples, not results from the coffee data.
The decision threshold depends on implementation cost, reversibility and the owner's risk tolerance.
Statistical significance at a conventional level is not itself a commercial objective.

A sufficiently narrow interval can rule out effects worth pursuing even if its point estimate is positive.
A wide interval can make both a large gain and a large loss plausible.
Neither result should be described simply as “the price works” or “the price does not work.”
Define a minimum economically relevant effect before looking at the result.
Compare the interval with that threshold and any downside guardrail.
If additional evidence is cheaper than the expected cost of a wrong irreversible decision, a larger test may be justified.
This is a business judgment; do not infer exact decision value without a specified loss function and uncertainty model.

When sample sizes are small, standard asymptotic cluster intervals may be unreliable.
Randomization inference can use the known allocation mechanism to construct a reference distribution under a specified sharp null.
It is not interchangeable with an interval for any arbitrary population effect without additional procedure and assumptions.
Consult statistical expertise on the relevant design before choosing the method.
Report the number of independent randomized clusters and the analysis population alongside the interval.
Describe any missing observations and sensitivity to pre-specified treatment of missing data.

### Step 12 — Inspect heterogeneity without creating false certainty

Price response may differ by store, customer, region or baseline category penetration.
Heterogeneity is commercially interesting because an average effect can hide meaningful variation.
It is also easy to overfit because many possible segment cuts exist.
Begin with hypotheses supported by the business mechanism and available sample size.
For example, distance to an independently observed alternative retailer may matter to switching.
Define the segment before assignment where possible.

Do not classify stores using treatment-period sales and then claim causal differences between the resulting groups.
The intervention can itself change the classification.
Similarly, a segment formed from a customer's observed coupon redemption is selected after the offer.
Use baseline characteristics and preserve all randomized units within each defined segment.
Report segment sample sizes and intervals as well as point estimates.
When the experiment is powered for the overall effect, subgroup estimates may be much less precise.

Testing many candidate segments creates a selection problem.
The largest observed effect can reflect random noise rather than durable customer behavior.
Pre-register a small number of confirmatory interactions and appropriate multiplicity handling.
Treat additional patterns as hypotheses for a separate validation test.
Pooling or hierarchical models can stabilize estimates, but their assumptions and shrinkage should be explained.
Never present a model-generated store-specific coefficient as directly randomized evidence for that individual store.

Segmentation can also create legal, privacy and fairness risks.
An attribute may be a proxy for protected or sensitive characteristics in a particular context.
Consult the organization's policies and applicable law before using individual-level characteristics to set prices.
Reporting an average treatment effect for a group does not itself authorize differentiated pricing.
Separate analytical diagnosis from the business approval to act on a segment.
Record who can access the data, what is retained and when it will be deleted.

### Step 13 — Handle dynamic and competitive effects

A price can affect the timing of purchases as well as total consumption.
For durable or storable products, a temporary discount may shift purchases from later weeks into the event week.
Let the primary event window and the post-event window be specified before the experiment.
Compare cumulative units and contribution over both windows.
If treatment adds 20 packs in the event week but removes 15 packs later, the observed cumulative increase is 5 packs.
That calculation still requires a valid control for the later window.

Repeated promotions can change the reference price that buyers expect.
An experiment on a first promotion need not predict the tenth promotion.
Customers may wait for the next discount, and suppliers may respond by changing funding.
The relevant treatment can therefore be a schedule of prices rather than a single price.
If the decision concerns a schedule, randomize feasible schedules and measure across the full cycle.
Changing the schedule midway complicates the original interpretation and should be recorded.

Competitors may react to a visible rollout in ways not observed in a small test.
Their response can affect the focal price effect and the category outcome.
A local trial may be too small to trigger a reaction, whereas a national rollout may trigger one.
This limits transport; it does not make the original randomized comparison invalid for its own setting.
Create scenarios for plausible reactions without claiming they are identified by the original test.
Use only lawfully obtained information and do not coordinate future pricing with competitors.

Inventory can interact with these dynamics.
A discount may deplete shelves early, making later weeks appear weak because the item is unavailable.
A price increase may preserve stock, increasing later sales relative to a depleted control.
Report on-hand inventory and replenishment by assignment over time.
If supply is an intended component of the policy, define its effect as part of the policy package.
If the decision seeks unconstrained demand, a stock-limited sales contrast is insufficient by itself.

### Step 14 — Prepare the handoff to finance and supply

The experiment owner should provide a decision table with one row per feasible policy.
Each row should show the offered price, eligibility, expected units and contribution in a common period.
State the analytical population and whether figures are per store-week or total across stores.
Include the actual experiment duration and the follow-up period.
Put the uncertainty interval next to the point estimate, not in a separate appendix alone.
Mark any assumption that was not directly observed.

Finance should confirm whether price is gross or net of tax and which rebates fund the policy.
Confirm whether supplier funding applies to all units or only incremental volume.
Distinguish invoice cost, avoidable logistics cost, returns and fixed implementation expense.
A one-time shelf-label cost should not be multiplied by units unless the costing model requires it.
If funding ends after the pilot, calculate the future decision with the future terms.
The test's measured sales response and the finance team's cost forecast are separate inputs.

Supply should translate the conditional unit forecast into an inventory and replenishment proposal.
Include lead time, minimum order quantities, shelf capacity and service-level constraints.
A price cut that increases demand beyond capacity may produce stockouts and customer harm.
A price increase that reduces demand can create excess inventory if the initial order is fixed.
Use scenarios rather than treating one elasticity point estimate as a guaranteed volume.
Escalate when the approved plan differs materially from the forecast used in the financial case.

The final decision record should identify the owner and the approval date.
It should state whether the decision is to roll out, extend the test, revise the policy or reject it.
It should link the analysis version, data extract and experiment registration.
It should list outstanding uncertainty and the monitoring trigger that would prompt review.
Record separately the forecast, plan, target and any external commitment.
This separation allows a later reviewer to distinguish analytical error from a deliberate risk-taking decision.

## 6. Worked example

### Business situation and assumptions

A fictional grocery chain is deciding whether to increase the regular shelf price of a 500 g pack of coffee A.
A comparable 500 g coffee B is sold in the same stores.
The proposed A price is $10.45 per pack, versus $9.50 per pack under the current policy.
The change is $0.95 per pack, or 10.0% of the current price.
The B shelf price stays at $8.80 per pack in both policies.
The decision horizon is one store-week, with an additional follow-up period planned for persistent effects.
All amounts are U.S. dollars, excluding sales tax, per pack unless otherwise stated.

Assume the retailer controls the price and receives the shelf price as net revenue.
Assume A's avoidable variable cost is $6.00 per pack.
Assume B's avoidable variable cost is $5.30 per pack.
Assume no extra display, coupon, fee, trade funding or event-specific cost changes with the price.
Assume the two items are the complete contribution boundary for this illustration.
In practice, that last assumption requires checking category exit and other products.

The proposed policy is an unadvertised shelf-price increase, not a promotion package.
The chain assigns 20 similar stores to each policy for four weeks by a documented random mechanism.
Each store receives its assigned policy for the full four weeks.
Treat stores as the randomization units, not the individual receipts.
The numbers below are illustrative group averages, not a claim about a real experiment.
The analysis does not invent standard errors or statistical significance from aggregate means.

### Observed policy averages

| Outcome | Current policy | Higher-price policy | Difference |
|---|---:|---:|---:|
| Assigned A price | $9.50/pack | $10.45/pack | +$0.95/pack |
| A sales | 200 packs/store-week | 176 packs/store-week | −24 packs/store-week |
| B sales | 100 packs/store-week | 108 packs/store-week | +8 packs/store-week |
| A plus B sales | 300 packs/store-week | 284 packs/store-week | −16 packs/store-week |

Assume complete price execution and no stockouts or closures for the arithmetic.
In a real test verify those conditions using shelf checks, price logs, inventory and opening hours.
Assignment protects the comparison in expectation; it does not ensure exact baseline balance in one realization.
Store-level variation is needed to assess precision.

### Own-price response

The A quantity change relative to the current-policy mean is

$$\frac{176-200}{200}=-0.12=-12.0\%.$$

The A price change relative to the current policy is

$$\frac{10.45-9.50}{9.50}=0.10=10.0\%.$$

The baseline-denominator own-price finite-change ratio is

$$E_{AA}^{base}=\frac{-12.0\%}{10.0\%}=-1.20.$$

For this policy contrast, a 10.0% higher A price corresponds to 12.0% fewer A packs per store-week.
Do not infer that every 1.0% increase anywhere on the price curve necessarily causes exactly 1.20% fewer packs.
The estimate is local to the compared policies and depends on the identified experiment effect.

The corresponding midpoint calculation uses an A quantity midpoint of 188 packs/store-week.
The midpoint price is $9.975 per pack.
The midpoint quantity change is −24/188, or approximately −12.77%.
The midpoint price change is $0.95/$9.975, or approximately 9.52%.
The arc own elasticity is approximately −1.34.
The two answers differ because the denominators differ; neither convention should be hidden.

### Cross-price response

The B quantity change relative to the current-policy mean is

$$\frac{108-100}{100}=0.08=8.0\%.$$

Dividing by the 10.0% A price increase gives

$$E_{BA}^{base}=\frac{8.0\%}{10.0\%}=+0.80.$$

The positive sign is consistent with B gaining units as A becomes more expensive.
It does not prove every additional B buyer previously purchased A.
Some switching may involve other products, new customers, or correlated basket choices.
Measure a wider category and customer movement if the business needs that attribution.

The B quantity midpoint is 104 packs/store-week.
The midpoint B response is 8/104, or approximately 7.69%.
Dividing by the midpoint A price change of approximately 9.52% gives an arc cross elasticity of approximately +0.81.
The cross elasticity of A with respect to B's price is a different quantity and cannot be computed from this experiment.

### Contribution comparison

The current A unit contribution is $9.50 − $6.00 = $3.50 per pack.
Current A contribution is 200 packs × $3.50 = $700 per store-week.
The higher-price A unit contribution is $10.45 − $6.00 = $4.45 per pack.
Higher-price A contribution is 176 packs × $4.45 = $783.20 per store-week.
The A contribution change is therefore +$83.20 per store-week.

B's unit contribution is $8.80 − $5.30 = $3.50 per pack under both policies.
Current B contribution is 100 packs × $3.50 = $350 per store-week.
Higher-price B contribution is 108 packs × $3.50 = $378 per store-week.
The B contribution change is +$28 per store-week.
Combined A and B contribution changes from $1,050 to $1,161.20 per store-week.
The point estimate of incremental contribution is +$111.20 per store-week.
It is not a guarantee, and it excludes unmeasured effects outside the stated two-product boundary.

The A price increase raises contribution despite fewer A packs because the higher unit margin more than offsets the lost volume at these observed means.
The gain on B adds to the category result under the assumed constant B price and cost.
Combined unit volume falls by 16 packs per store-week, so a revenue or volume target may lead to a different decision than a contribution objective.
If the 16 missing packs instead move to another product, its contribution also belongs in a complete category calculation.
If customers leave the category, their wider basket value may matter but is not estimated here.

### Break-even and decision packet

With no B effect, the higher-price A policy breaks even on A when

$$Q_A^1=\frac{700}{4.45}\approx157.30\text{ packs/store-week}.$$

That is approximately 21.35% fewer A packs than the 200-pack baseline.
With an independently supported +$28 per store-week B contribution effect, the A quantity threshold becomes

$$Q_A^1=\frac{700-28}{4.45}\approx151.01\text{ packs/store-week}.$$

That threshold assumes the B effect remains +8 packs even as A volume varies.
It is a conditional arithmetic scenario, not a joint statistical confidence bound.
If the B effect is uncertain or correlated with A loss, model their joint outcomes using store-level data.

The observed point estimates justify further review of the higher-price policy.
They do not by themselves justify a national rollout because no interval, full category accounting or competitor response is provided.
A suitable next action is to calculate a cluster-aware interval and assess the follow-up window and execution.
The forecast is a conditional +$111.20 per store-week point estimate with unquantified uncertainty in this illustration.
The plan would be a separately approved choice of stores, dates, shelf labels and inventory allocations.
A target might be +$100 per store-week, but it does not establish that the effect is certain.
A commitment would require a separate accepted obligation; the experiment's mean is not one.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
Save it as a local `.py` file and run `python filename.py`.
It reproduces the example's aggregate ratios and contribution arithmetic.
Inputs are illustrative means per store-week, not individual observations.
It does not estimate standard errors, confidence intervals or experimental power.
A real analysis must use the actual cluster-level assignment and outcome data.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    a_price: float          # USD per pack
    a_units: float          # packs per store-week
    b_units: float          # packs per store-week


def baseline_elasticity(
    q_control: float,
    q_treatment: float,
    p_control: float,
    p_treatment: float,
) -> float:
    if q_control <= 0 or p_control <= 0:
        raise ValueError("Baseline quantity and price must be positive")
    price_change = (p_treatment - p_control) / p_control
    if price_change == 0:
        raise ValueError("The policy price must change")
    quantity_change = (q_treatment - q_control) / q_control
    return quantity_change / price_change


def arc_elasticity(
    q_control: float,
    q_treatment: float,
    p_control: float,
    p_treatment: float,
) -> float:
    q_mid = (q_control + q_treatment) / 2
    p_mid = (p_control + p_treatment) / 2
    if q_mid <= 0 or p_mid <= 0:
        raise ValueError("Quantity and price midpoints must be positive")
    price_change = (p_treatment - p_control) / p_mid
    if price_change == 0:
        raise ValueError("The policy price must change")
    return ((q_treatment - q_control) / q_mid) / price_change


def contribution(
    policy: Policy,
    a_cost: float,
    b_price: float,
    b_cost: float,
) -> float:
    a = (policy.a_price - a_cost) * policy.a_units
    b = (b_price - b_cost) * policy.b_units
    return a + b


control = Policy(a_price=9.50, a_units=200, b_units=100)
treatment = Policy(a_price=10.45, a_units=176, b_units=108)
a_cost = 6.00
b_price = 8.80
b_cost = 5.30

own = baseline_elasticity(
    control.a_units, treatment.a_units,
    control.a_price, treatment.a_price,
)
cross = baseline_elasticity(
    control.b_units, treatment.b_units,
    control.a_price, treatment.a_price,
)
own_arc = arc_elasticity(
    control.a_units, treatment.a_units,
    control.a_price, treatment.a_price,
)
cross_arc = arc_elasticity(
    control.b_units, treatment.b_units,
    control.a_price, treatment.a_price,
)
control_c = contribution(control, a_cost, b_price, b_cost)
treatment_c = contribution(treatment, a_cost, b_price, b_cost)
b_delta = (treatment.b_units - control.b_units) * (b_price - b_cost)
a_baseline_c = (control.a_price - a_cost) * control.a_units
break_even_a = (a_baseline_c - b_delta) / (treatment.a_price - a_cost)

print(f"Own, baseline: {own:.2f}")
print(f"Cross, baseline: {cross:+.2f}")
print(f"Own, arc: {own_arc:.2f}")
print(f"Cross, arc: {cross_arc:+.2f}")
print(f"Control contribution: ${control_c:.2f}/store-week")
print(f"Treatment contribution: ${treatment_c:.2f}/store-week")
print(f"Contribution difference: ${treatment_c - control_c:.2f}/store-week")
print(f"A break-even with B effect: {break_even_a:.2f} packs/store-week")
```

Expected output, allowing for ordinary rounding:

```text
Own, baseline: -1.20
Cross, baseline: +0.80
Own, arc: -1.34
Cross, arc: +0.81
Control contribution: $1050.00/store-week
Treatment contribution: $1161.20/store-week
Contribution difference: $111.20/store-week
A break-even with B effect: 151.01 packs/store-week
```

The code assumes that prices and quantities have the same interpretation across policies.
It treats the displayed policy price as the relevant net receipt.
The contribution formula assumes constant per-pack costs and excludes incremental fixed cost.
The output is descriptive arithmetic on supplied group means.
A causal interpretation requires the design, implementation and outcome assumptions discussed above.

For a real dataset, retain one record per randomized store with assignment and pre-specified outcome.
Use a store-week panel only with an analysis that respects repeated observations within store.
Document missing-store outcomes and compare missingness by assignment.
Apply weights matching the randomization and the desired population when the design is not balanced.
Compute uncertainty for the combined contribution contrast directly from each cluster's combined outcome.
If treatment is assigned by market, stores within a market are not independent randomized units.

## 8. Data, standards, and regulatory context

### Data contract

A usable analysis starts with an eligibility table of all stores or customers assigned to each policy.
It includes assignment ID, assignment time, allocation probability, stratum and randomization unit.
The product table records stable item identifiers, pack size and effective dates.
The price table records the price customers were offered, currency, tax basis and validity interval.
The transaction table records item, quantity, net price, discounts, returns and timestamp.
The inventory table records opening availability, replenishment, stockouts and closures.
The marketing table records displays, feature advertisements, coupons and eligibility.
The cost table records avoidable cost and commercial funding on the same time basis as contribution.

Join by item, location and effective time, not by a current product description alone.
Preserve historical price and product versions rather than overwriting them.
If a pack size changes, convert to a comparable unit or treat it as a separate intervention.
An average price per transaction is influenced by the customer mix and is not automatically the price exposure.
For an ITT analysis, assigned policy is the treatment variable even when actual receipts differ.
Maintain a separate execution measure so the ITT is not confused with a perfect-compliance effect.

Zero sales for an open, stocked store are an outcome.
Missing transactions for a store whose feed failed are missing data.
A closed store and an in-stock store with no buyers are different operational states.
Predefine the treatment of each state and report how often it occurs in each assignment arm.
Do not discard low-selling stores after seeing treatment outcomes.

### Identifiers and data conventions

GS1's Global Trade Item Number (GTIN) identifies trade items in a shared identification system.
Use it where available to reconcile products across internal and trading-partner data.
A GTIN does not itself prove equivalent pack size, current price, availability or causal comparability.
Preserve the source-specific product mapping and effective dates, especially for multipacks and relaunches.
A location identifier can help join store records but does not define a randomization unit by itself.
See the GS1 source in section 16 for the identification convention.

A store-week should have an explicit time zone and week boundary.
A price that starts halfway through a week requires either a finer time grain or a pre-specified exposure rule.
Normalize currency only with a documented exchange-rate convention when pooling markets.
A cross-country model must not treat a currency conversion as proof that customers face comparable economic conditions.
Separate consumer tax from retailer net receipt consistently throughout the calculation.
The relevant cost for a decision is avoidable cost under that decision, not automatically the fully allocated accounting cost.

### Privacy and competition boundaries

Customer-level price experiments may process personal data when identifiers can be linked to people.
The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, sets obligations for applicable processing, including principles such as purpose limitation and data minimization.
Its applicability and lawful basis depend on the specific facts and jurisdiction.
Pseudonymization reduces some risk but does not automatically remove data from the regulation's scope.
Consult the regulation and the organization's privacy counsel before selecting personal data or launching a personalized test.
This educational discussion is not legal advice.

Pricing teams should use independently authorized information and avoid sharing competitively sensitive non-public prices or future pricing plans with competitors.
U.S. antitrust law and other jurisdictions' competition rules depend on conduct and context; an analytics design is not a safe harbor.
The U.S. Department of Justice and Federal Trade Commission provide official antitrust guidance linked below.
Involve competition counsel when an experiment involves competitors, shared platforms, third-party data or coordinated market practices.
Do not use this topic to infer that a particular data exchange or algorithm is lawful.

Consumer-protection, advertised-price, accessibility, sector-specific and local rules may also apply.
A randomized test does not override price-display rules or customer commitments.
Record which legal and policy reviews were completed, by whom and for which market.
Document fairness and customer trust concerns, including consistent treatment within eligible groups and accessible disclosures where required.
These controls constrain implementation; they do not by themselves identify a causal effect.

## 9. Industry practice and case evidence

### What public evidence can and cannot establish

Retailers routinely use promotions, price variation and customer response data to understand demand.
Public research documents the methodological challenge that prices and marketing activities are chosen rather than naturally randomized.
The evidence below supports the need for careful design; it is not evidence that the fictional coffee test occurred.
A published estimate for one product, period and market should not be copied into another company's price file without validation.

### Dominick's Finer Foods research data

The University of Chicago Booth School makes the Dominick's Finer Foods database available for academic research, subject to its access conditions.
The database contains store-level scanner data for a supermarket chain and has supported extensive empirical research on pricing and promotions.
It is useful for exploring the relationship among observed prices, promotions and category sales.
It is not a randomized price assignment merely because the observations are detailed.
Managers can choose prices and promotions using information the analyst does not observe.
The analyst must therefore defend a causal design rather than treating database size as identification.

A practical exercise is to choose a category, reconstruct effective unit prices and identify weeks with promotion indicators.
First plot the support of prices by store and week.
Then compare a raw regression with a specification containing store and week effects.
The difference between coefficients reveals sensitivity to model choice, not necessarily which coefficient is causal.
Check whether display, availability and competitor actions are observed before claiming a price-only effect.
The data access terms and available fields must be checked before promising a particular analysis.

### A field experiment on retail pricing

Hoch, Kim, Montgomery and Rossi studied determinants of store-level price elasticity using supermarket data and field experimentation.
Their published research connects store and market characteristics with price sensitivity rather than assuming one chainwide response.
The study is evidence that retail price response can vary across locations and that field evidence can inform pricing analysis.
It does not establish the appropriate elasticity for the fictional coffee product or for a current retailer.
Consult the paper for its sampling, design and statistical limits before transferring any substantive result.

**Inference:** A chain considering rollout should examine whether the test stores represent the intended rollout population.
A small set of high-volume stores may produce a precise estimate for those stores but miss customer and competitor differences elsewhere.
A model can partially pool store responses, but the amount of pooling and its assumptions should be disclosed.
Designing a representative test often provides stronger transfer evidence than retrospectively adjusting a convenience sample.

### Randomized pricing and experimentation practice

Anderson and Simester report field experiments on price endings in retail settings.
Their work illustrates that a change in the price customers see can be tested rather than inferred exclusively from historical correlations.
Its results concern specific settings and pricing manipulations, not a universal rule for every product.
A test of a price ending is not automatically a test of a comparable percentage change with the same display and reference price.
The intervention definition must include the visible price, not only a rounded percentage discount.

**Inference:** When a firm tests two prices that differ in both level and ending, its estimated effect combines those differences.
Separating the effect of a price ending from the effect of the price level requires a design with appropriate independent variation.
This distinction matters if a manager wants to use a price-ending result for an unrelated price increase.
A single two-arm comparison cannot separately identify all mechanisms that changed between arms.

### From evidence to governance

The first decision meeting should review the intervention card and test plan before launch.
The readout should show assignment, execution, unit response, category economics and uncertainty together.
Commercial owners should state the business objective and the next feasible action.
Supply owners should assess stock, capacity and replenishment implications.
Finance should reconcile price, funding, costs and contribution with the ledger.
Legal and privacy owners should assess the proposed use where relevant.

A price model can be useful for forecasting even when causal identification is weak, provided its purpose and validation are explicit.
It should not be presented as the expected effect of intervention merely because it predicts historical sales.
A well-calibrated forecast under historical pricing policies can fail when the policy changes.
Conversely, a credible experimental contrast can inform an intervention but still require an operational forecast for a different time horizon.
Keep these two uses and their evidence standards distinct in the decision record.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Demand-driven pricing | High-demand weeks have high prices. | Managers anticipate demand shocks. | Correlation can understate the negative price response or reverse its sign. | Randomize or defend an identification strategy with the pricing process. |
| Promotion bundle | Discounts always coincide with display. | Price and merchandising vary together. | Package effect is mislabeled price-only elasticity. | Hold display fixed or independently vary factors. |
| Stockout censoring | High-demand stores have empty shelves. | Sales cannot exceed available inventory. | Understates response in constrained stores. | Protect stock, measure availability and report constrained outcomes. |
| Selective compliance | Stores dropping price differ from assigned stores. | Actual implementation depends on local conditions. | Comparison by receipt loses randomization. | Report ITT and execution separately. |
| Cross-store interference | Nearby control stores lose traffic. | Customers shop across catchments. | Treatment-control contrast is contaminated. | Randomize catchments or quantify spillovers. |
| Category omission | A units fall while B units rise. | Switching is outside the outcome. | Decision misses displaced or recaptured margin. | Measure predefined alternatives and category contribution. |
| Pull-forward | Event lift is followed by a dip. | Customers advance future purchases. | Short-window gain overstates lasting benefit. | Include pre-specified follow-up periods. |
| Invalid control | Treated and control markets face different events. | Counterfactual trends differ. | DiD estimate is biased. | Defend comparison and inspect pre-period patterns and concurrent events. |
| Weak instrument | Price barely changes with instrument. | Little exogenous price variation exists. | IV ratio is unstable. | Assess first stage and use robust methods or another design. |
| Exclusion violation | Instrument also affects availability. | Instrument changes outcome outside price. | IV is not identified by the claimed channel. | Investigate institutional pathways and reject unsupported instrument. |
| Wrong price denominator | Net paid price changes with buyer mix. | Coupon redemption and selection change averages. | Elasticity denominator is endogenous. | Define assigned price and measure paid price separately. |
| Wrong unit of inference | Millions of receipts but few stores. | Transactions share cluster shocks and assignment. | Intervals are too narrow. | Analyze at assignment cluster or use appropriate cluster inference. |
| Over-extrapolation | Proposed price is outside test range. | Demand curve and competitors may change. | Large rollout misses actual response. | Test the new range and show scenario limits. |
| Multiple testing | Only the best segment is reported. | Chance extremes are selected. | Spurious segment-specific pricing. | Pre-specify primary claims and control multiplicity or label exploration. |
| Margin leakage | Revenue rises but finance cannot reconcile profit. | Funding, returns or costs are omitted. | Incorrect commercial recommendation. | Reconcile a net contribution waterfall. |
| Post-treatment adjustment | Model controls for treatment-induced traffic. | A pathway of the effect is conditioned away. | Estimand changes and selection bias can arise. | Pre-specify causal adjustment and avoid casual mediator control. |
| Zero baseline | A newly launched item has no baseline units. | Percentage response is undefined. | Infinite or arbitrary elasticity. | Report unit effects or use an appropriate model for zeros. |
| Price implementation error | Shelf and checkout prices differ. | Price file or label failure. | Customer harm and treatment ambiguity. | Monitor execution, escalate and preserve assigned analysis. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to assess or respond |
|---|---|---|---|
| Assignment follows the stated random mechanism. | Supports exchangeability by design. | Manual reassignment after outcomes are anticipated. | Audit assignment logs, timing and deviations. |
| Outcomes are measured consistently across arms. | Ensures comparable numerator and denominator. | A treatment store's transaction feed fails. | Reconcile eligible stores and quantify differential missingness. |
| Interference is absent or incorporated in the estimand. | Gives each unit a well-defined policy contrast. | Shoppers move between adjacent assigned stores. | Map catchments and randomize larger clusters if needed. |
| The assigned policy is clearly defined. | Makes potential outcomes interpretable. | Price, display and eligibility differ unpredictably. | Freeze the intervention card and audit execution. |
| Costs and receipts reflect the decision. | Makes contribution relevant. | Off-invoice rebates or returns are omitted. | Reconcile with finance and perform sensitivity analysis. |
| Inventory is sufficient for the interpreted outcome. | Allows sales to approximate demand response. | Treatment induces a stockout. | Measure availability and report sales versus latent demand limits. |
| Tested population is relevant to rollout. | Supports transport to the decision. | Test stores have unusual customers or competitors. | Compare distributions and run a representative follow-up test. |
| Tested horizon captures the decision horizon. | Avoids mistaking timing changes for lasting effects. | Pantry loading or delayed competitor reaction. | Observe a longer pre-specified follow-up. |
| The price contrast is nonzero and baseline positive. | Makes the reported finite ratio defined. | Zero baseline sales or no price separation. | Report units or use an alternative estimand. |
| Other decision effects are included or bounded. | Supports category-level economics. | Shoppers move to omitted products or channels. | Expand the outcome boundary and show residual uncertainty. |
| Observational adjustments block relevant confounding. | Needed for a causal regression interpretation. | Managers use unobserved local information. | Compare designs and conduct substantive sensitivity analysis. |
| IV exclusion and independence hold if IV is used. | Connects the instrument to a price-only effect. | A cost shock changes availability as well as price. | Investigate pathways; do not infer validity from first-stage strength. |

None of these assumptions can be made true merely by using a complex model.
A balance table is useful for diagnosing mistakes but does not prove the absence of all confounding.
A statistically significant first stage does not prove the IV exclusion restriction.
A stable short-run average does not prove that a long-run rollout will have the same effect.
When assumptions are doubtful, state the direction and possible scale of the business exposure where evidence permits.
Otherwise label the effect as uncertain rather than inventing a bias correction.

## 12. Exercises

### Exercise 1 — Elasticity arithmetic

**Time:** 15 minutes.
A's assigned price rises from $8.00 to $8.80 per pack.
A sales change from 150 to 132 packs per store-week.
B sales change from 80 to 84 packs per store-week.
**Deliverable:** Calculate baseline-denominator own and cross elasticities.
**Acceptance criteria:** Show both percentage changes, identify the outcome and price in each ratio, and interpret both signs.

### Exercise 2 — Draw the confounding story

**Time:** 15 minutes.
Managers raise price before a local festival because they expect high demand.
The festival also directly increases store visits.
A separate display is sometimes installed at the same time.
**Deliverable:** Draw a causal diagram and explain why a raw price-sales regression may mislead.
**Acceptance criteria:** Include the festival, expected demand, price, display and sales; explain at least one backdoor path and one bundled-treatment problem.

### Exercise 3 — Design a cluster test

**Time:** 25 minutes.
A chain can change shelf price only at store level.
Nearby stores share shoppers, and marketing can vary displays.
The chain wants a four-week price-only effect on category contribution.
**Deliverable:** Write a one-page test brief.
**Acceptance criteria:** Specify assignment unit and interference mitigation, price policies, outcome, ITT analysis, execution checks, power inputs, guardrails and follow-up.

### Exercise 4 — ITT and finite response

**Time:** 15 minutes.
In a randomized trial, the control mean is 240 packs per store-week at $12.00 per pack.
The treatment mean is 216 packs per store-week at an assigned price of $13.20 per pack.
Some stores fail to update their shelf labels, but all remain in their assigned arms.
**Deliverable:** Calculate the ITT unit contrast and baseline-denominator policy ratio.
**Acceptance criteria:** Report units, percentage changes and the meaning of incomplete compliance; do not claim the ratio is a perfect-compliance effect.

### Exercise 5 — Critique identification

**Time:** 20 minutes.
An analyst regresses store sales on price with store and week fixed effects.
Managers vary local prices in response to unrecorded local events.
A second analyst proposes supplier cost as an instrument, but supplier disruption also causes empty shelves.
**Deliverable:** Write a short identification critique.
**Acceptance criteria:** Explain the remaining time-varying confounding, the IV relevance and exclusion requirements, and a reason the proposed instrument may fail.

### Exercise 6 — Compare contribution

**Time:** 20 minutes.
A sells 100 packs per store-week at $10.00 per pack with $6.00 variable cost.
A proposed price of $11.00 per pack produces 85 packs per store-week.
B's unit contribution is $2.00 and its sales rise from 40 to 45 packs per store-week.
Assume no other effects or costs.
**Deliverable:** Calculate the change in A contribution, B contribution and combined contribution.
**Acceptance criteria:** Show each policy's A contribution and the $/store-week difference; state the two-product boundary.

### Exercise 7 — Audit a transfer request

**Time:** 20 minutes.
A two-week online coupon test estimates a −1.5 own-price ratio for a small metropolitan sample.
A manager wants to apply it to a permanent nationwide shelf-price increase twice as large.
The test had frequent stockouts and no post-test measurement.
**Deliverable:** Prepare a transfer-risk memo with a recommended next action.
**Acceptance criteria:** Identify at least four distinct threats, distinguish effect from observed sales, and propose evidence that addresses the gaps.

### Exercise 8 — Label the decision packet

**Time:** 10 minutes.
The analyst projects 900–1,100 packs per week under the proposed policy.
The category director wants 1,200 packs per week.
Operations has approved an order of 1,050 packs per week.
The supplier has signed an obligation to deliver 1,000 packs per week.
**Deliverable:** Label each statement as forecast, target, plan or commitment.
**Acceptance criteria:** Explain why the labels are not interchangeable and identify the units and period.

## 13. Self-check answers

### Exercise 1

A price increases by ($8.80 − $8.00)/$8.00 = 10%.
A units fall by (132 − 150)/150 = −12%.
The baseline own ratio is −12%/10% = −1.20.
B units rise by (84 − 80)/80 = +5%.
The baseline cross ratio is +5%/10% = +0.50.
The positive cross sign is consistent with substitution toward B, conditional on a credible causal design.
The arithmetic alone does not establish that the price change caused the unit changes.

### Exercise 2

One valid diagram has festival → expected demand → price and festival → store visits → sales.
There is a noncausal path from price back through expected demand and the festival to sales.
A positive correlation can therefore coexist with a negative causal price effect.
Display → sales and price-associated display assignment also complicate a price-only contrast.
A good answer explains that adjusting for some observed event variables may help but does not automatically solve unobserved confounding.

### Exercise 3

A strong brief assigns catchments or sufficiently separated store clusters when shared shoppers are material.
It states the price in each arm, keeps display constant and defines category contribution per eligible cluster-week.
It analyzes by assignment, verifies prices and availability, and retains noncompliant stores in ITT.
It bases power on the number and variance of independent clusters, the effect worth detecting and the planned horizon.
It identifies privacy and competition review, an operational stop process and a post-test window.
A weak brief randomizes receipts although the same shelf price applies to the whole store.

### Exercise 4

The ITT contrast is 216 − 240 = −24 packs per store-week.
The assigned price increases by 10% and the assigned-policy outcome falls by 10%.
The baseline-denominator policy ratio is −1.00.
The ratio includes implementation failures because the comparison remains by assignment.
It does not directly identify the effect of everyone receiving the higher price.
A separate compliance-adjusted estimand would require additional assumptions and an appropriate design.

### Exercise 5

Store fixed effects absorb time-invariant store differences; week effects absorb shocks common to all stores.
An unrecorded local event varying by store and week can still influence both price and sales.
Supplier cost is relevant only if it moves the price sufficiently in the relevant data.
A disruption that affects shelves directly offers a path from the instrument to sales other than the focal price.
That path threatens exclusion; a strong first stage cannot repair it.
A stronger answer discusses independence and the population or local effect identified under additional IV assumptions.

### Exercise 6

Current A contribution is 100 × ($10 − $6) = $400 per store-week.
Proposed A contribution is 85 × ($11 − $6) = $425 per store-week.
A increases contribution by $25 per store-week.
B adds (45 − 40) × $2 = $10 per store-week.
Combined contribution rises by $35 per store-week under the stated assumptions.
The calculation omits other products, fixed implementation costs and effects beyond the stated period.

### Exercise 7

A strong memo identifies coupon versus shelf-price treatment, online versus store channel, metro versus nationwide population and short versus permanent horizon.
It also identifies larger price movement, possible asymmetry of a price increase, stockout censoring and missing post-test pull-forward.
It says the test measured observed sales under its policy, not unconstrained demand under the new policy.
It recommends a representative shelf-price test with sufficient inventory, category outcomes and an appropriate follow-up window.
A weak memo simply multiplies −1.5 by the larger percentage and labels the result a guaranteed forecast.

### Exercise 8

The 900–1,100 packs per week range is a conditional forecast.
The 1,200 packs per week desired outcome is a target.
The approved order of 1,050 packs per week is a plan.
The signed obligation to deliver 1,000 packs per week is a commitment.
The plan and commitment may create a supply gap, and the target does not override the forecast.

## 14. Interview questions

1. **What does an own-price elasticity of −1.2 mean?**
   A strong answer specifies a local percentage response, population and horizon and distinguishes a finite ratio from a derivative.
   A weak answer says every price increase always causes exactly the same response.
2. **How do you interpret a positive cross-price elasticity?**
   A strong answer names the demand outcome and the other product's price and says the sign is consistent with substitution.
   A weak answer reverses the indices or assumes the sign proves individual switching.
3. **Why might price and sales be positively correlated when raising price reduces demand?**
   A strong answer explains demand shocks and managers' anticipatory pricing.
   A weak answer treats correlation as a causal demand curve.
4. **When does a promotion estimate fail to answer a shelf-price question?**
   A strong answer identifies display, advertising, coupons, timing and different customer populations as intervention differences.
   A weak answer assumes any observed discount effect is pure price response.
5. **Why analyze a price experiment by assignment?**
   A strong answer explains the ITT estimand and how conditioning on compliance can destroy randomized comparability.
   A weak answer deletes every store that failed implementation without discussing selection.
6. **What is the relevant unit for uncertainty?**
   A strong answer starts with the randomization unit, clustering, repeated measurements and spillovers.
   A weak answer treats every receipt as independently randomized when stores received one price.
7. **What can store and week fixed effects remove?**
   A strong answer distinguishes stable store effects and common time shocks from local time-varying confounding.
   A weak answer claims all endogeneity is removed.
8. **What makes a price instrument credible?**
   A strong answer states relevance, independence and exclusion and investigates institutional pathways.
   A weak answer offers only a high first-stage correlation.
9. **Why can a volume increase reduce profit?**
   A strong answer calculates the lower unit margin on baseline units, incremental units and cross-product effects.
   A weak answer equates revenue or units with contribution.
10. **What would block nationwide use of a local price test?**
    A strong answer discusses support, population, horizon, execution, competition and uncertainty.
    A weak answer assumes a statistically significant local coefficient transfers unchanged.

## 15. Key terms

- [Assignment](#step-4--select-a-credible-identification-design) — allocation of an eligible unit to a policy.
- [Arc elasticity](#step-2--define-quantities-and-elasticities) — finite-change elasticity using midpoint denominators.
- [Baseline-denominator ratio](#step-2--define-quantities-and-elasticities) — finite quantity response divided by finite price response, each relative to baseline.
- [Cannibalization](#step-8--translate-the-response-into-a-commercial-decision) — displacement of sales from another product.
- [Cluster](#step-4--select-a-credible-identification-design) — group assigned or analyzed together.
- [Compliance](#step-5--design-outcomes-and-data-collection) — implementation of assigned policy.
- [Confounding](#step-3--draw-the-causal-process-before-fitting-a-model) — mixing of treatment effects with common causes.
- [Contribution](#step-8--translate-the-response-into-a-commercial-decision) — net revenue less specified avoidable costs.
- [Counterfactual](#3-the-core-mental-model) — outcome under an alternative policy for the same unit.
- [Cross-price elasticity](#step-2--define-quantities-and-elasticities) — response of one product's demand to another product's price.
- [Demand](#4-definitions-and-vocabulary) — desired purchases under specified conditions.
- [Difference-in-differences](#step-4--select-a-credible-identification-design) — comparison of outcome changes across groups.
- [Endogeneity](#step-3--draw-the-causal-process-before-fitting-a-model) — dependence of modeled price on unobserved outcome determinants.
- [Estimand](#step-1--define-the-decision-and-the-intervention) — quantity defined as the target of estimation.
- [Exclusion restriction](#step-4--select-a-credible-identification-design) — instrument affects the outcome only through the specified treatment pathway.
- [Fixed effect](#step-4--select-a-credible-identification-design) — modeled intercept for a defined group or time.
- [Forecast](#step-8--translate-the-response-into-a-commercial-decision) — conditional prediction, distinct from a target.
- [Instrumental variable](#step-4--select-a-credible-identification-design) — source of treatment variation used under explicit causal assumptions.
- [Intention to treat](#step-7--estimate-policy-effects) — effect of assigned policy regardless of implementation.
- [Interference](#step-4--select-a-credible-identification-design) — one unit's outcome depends on another unit's assignment.
- [Own-price elasticity](#step-2--define-quantities-and-elasticities) — demand response to the same product's price.
- [Plan](#step-8--translate-the-response-into-a-commercial-decision) — approved intended action.
- [Point elasticity](#step-2--define-quantities-and-elasticities) — derivative-based proportional response at a point.
- [Pull-forward](#step-5--design-outcomes-and-data-collection) — shifting purchases into an earlier period.
- [Randomization](#step-4--select-a-credible-identification-design) — chance allocation under a known design.
- [Stockout censoring](#step-5--design-outcomes-and-data-collection) — recorded sales limited by availability.
- [Target](#step-8--translate-the-response-into-a-commercial-decision) — desired outcome, not a forecast.
- [Transportability](#step-9--monitor-and-decide-whether-to-transfer) — relevance of an estimate to another decision setting.

## 16. Sources

- Joshua D. Angrist and Jörn-Steffen Pischke, *Mostly Harmless Econometrics: An Empiricist's Companion*, Princeton University Press, 2009. [Publisher page](https://press.princeton.edu/books/hardcover/9780691120348/mostly-harmless-econometrics). Identification, IV and regression interpretation.
- Paul R. Rosenbaum and Donald B. Rubin, “The Central Role of the Propensity Score in Observational Studies for Causal Effects,” *Biometrika* 70(1), 1983, 41–55. [DOI: 10.1093/biomet/70.1.41](https://doi.org/10.1093/biomet/70.1.41). Observational causal comparisons and assumptions.
- Alberto Abadie, Susan Athey, Guido W. Imbens and Jeffrey Wooldridge, “When Should You Adjust Standard Errors for Clustering?”, *Quarterly Journal of Economics* 138(1), 2023, 1–35. [DOI: 10.1093/qje/qjac038](https://doi.org/10.1093/qje/qjac038). Design and sampling considerations for clustered inference.
- Steven J. Hoch, Byung-Do Kim, Alan L. Montgomery and Peter E. Rossi, “Determinants of Store-Level Price Elasticity,” *Journal of Marketing Research* 32(1), 1995, 17–29. [DOI: 10.1177/002224379503200103](https://doi.org/10.1177/002224379503200103). Retail store-level heterogeneity.
- Eric T. Anderson and Duncan I. Simester, “Effects of $9 Price Endings on Retail Sales: Evidence from Field Experiments,” *Quantitative Marketing and Economics* 1, 2003, 93–110. [DOI: 10.1023/A:1023581927405](https://doi.org/10.1023/A:1023581927405). Field evidence on price endings and intervention context.
- Chicago Booth, Kilts Center for Marketing, [Dominick's Finer Foods data](https://www.chicagobooth.edu/research/kilts/datasets/dominicks). Public description and access conditions for supermarket scanner data.
- GS1, [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin). Trade-item identification; not a causal or price standard.
- European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Primary legal text; applicability depends on facts.
- U.S. Federal Trade Commission, [Guide to Antitrust Laws](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws). General official competition guidance; not legal advice.
- U.S. Department of Justice, Antitrust Division, [Antitrust Laws and You](https://www.justice.gov/atr/antitrust-laws-and-you). General official explanation of U.S. antitrust law.

## 17. Links

- [Module 16 topic map](./_index.md).
- [The pricing decision stack](./01_pricing-decision-stack.md).
- [Module 16 survey and its lab exercises](../../../course/16_pricing_promotions_revenue_management.md).
- [Module 04 survey: statistics and causal inference](../../../course/04_statistics_causal_inference.md).
- [Module 05 survey: experimentation and pilots](../../../course/05_experimentation_pilots_evidence.md).
- [Price, value, and willingness to pay](./02_value-wtp.md).
- The topic map lists planned sibling topics on the Lerner rule and price testing.
