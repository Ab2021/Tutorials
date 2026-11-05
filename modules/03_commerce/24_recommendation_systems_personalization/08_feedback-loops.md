# Feedback Loops and Bias in Recommendations

> Module 24 — Recommendation Systems and Personalization  
> Topic 08 of 11  
> Estimated study time: 4 hours  
> Prerequisites: recommendation tasks, baselines, and offline/online evaluation  
> Version 2026.5

## 1. Why this topic exists

A retailer sees sales concentrate in a few recommended products.
The ranking model learns from those sales and gives the same products still more exposure.
The pattern can reflect genuine customer preference, unequal visibility, or both.
A dashboard of clicks alone cannot distinguish those explanations.

The commercial decision is whether to replace an existing recommendation policy, not whether a model can predict its own historical labels.
A policy includes candidate eligibility, ranking, placement, and any randomization.
It determines which customer sees which product in which position.
Its effects reach inventory allocation, supplier access, margin, and the customer's ability to discover a useful alternative.

Consider an online grocer with a new regional supplier.
The supplier's cereal rarely enters the candidate set because it has few historical purchases.
A model that uses purchases as relevance labels learns almost nothing about that cereal.
Displaying it in a safe, randomized exploration slot can reveal demand, but costs scarce screen space and may displace a proven product.
The right experiment depends on the value of information, expected contribution, supply constraints, and customer experience.

A category manager owns assortment and commercial guardrails.
A product owner owns the recommendation surface and treatment definition.
A data scientist owns the estimand and analysis, subject to independent review.
Engineering owns correct logging and policy execution.
Inventory and fulfillment owners define availability and service constraints.
Privacy and legal teams advise on lawful data use and applicable disclosures.
The experiment owner has explicit authority to stop or roll back a test.

> **Note:** This topic distinguishes observed association from a causal policy effect.
> A mathematically valid estimator does not rescue a policy comparison when the underlying logging, support, or experiment design is wrong.

The operational questions are practical.
Which items were eligible but not displayed?
Which were actually viewable, rather than merely returned by an API?
How often would the historical policy have shown an alternative?
What did customers do after the display, including returns or substitutions?
Is the proposed exploration safe for both the shopper and the supply chain?

## 2. Learning outcomes

After working through this unit, you can:

1. Draw the causal loop from eligibility and ranking through exposure, outcomes, training data, and the next policy.
2. Distinguish exposure bias, position bias, selection bias, and popularity reinforcement in an observed log.
3. Define a policy-value estimand with a population, action, outcome, attribution window, and unit.
4. Compute inverse propensity scoring (IPS), self-normalized IPS (SNIPS), and effective sample size for a small logged dataset.
5. Check support, propensity provenance, variance, and the assumptions behind counterfactual evaluation.
6. Design a bounded randomized exploration policy and its event contract without bypassing eligibility or safety rules.
7. Interpret an online experiment with customer, financial, inventory, and supplier guardrails.
8. Diagnose a feedback failure and assign a corrective decision to an accountable owner.

Exercises 1–8 map to these outcomes in order.
The answers provide arithmetic where the result is identifiable and rubrics where it depends on design choices.

## 3. The core mental model

```text
context + inventory + customer permissions
                 |
                 v
        eligible candidate set
                 |
                 v
     logging policy + randomization -----> logged action probability
                 |
                 v
        ordered product slate
                 |
                 v
       rendered / viewable exposure
                 |
                 v
       attention -> click -> order -> return
                 |                    |
                 +---- event log -----+
                          |
                          v
         labels + sampling + model training
                          |
                          v
                next serving policy
                          |
                          +---- changes future exposure ----+
```

The policy affects which labels can exist.
The training pipeline treats some of those labels as evidence about relevance.
The resulting policy then changes the labels collected next time.
This is a closed loop, not an independent sample from a fixed catalogue.

Separate four stages: eligible, selected, rendered, and examined.
An eligible product can be omitted by retrieval.
A selected product can fail to render because the page is abandoned or an asset fails.
A rendered product can be below the fold and never enter the viewport.
A viewable product can receive no attention.
Each stage removes information, and none is identical to a negative preference.

Position changes the opportunity to be noticed.
A highly placed item can receive more clicks even when its intrinsic appeal is lower.
Page layout, device, scrolling, price labels, imagery, and neighboring products can all change examination.
Therefore, a raw click-through rate by rank is not a clean estimate of a rank effect unless placement is randomized or stronger assumptions are justified.

The loop can be beneficial as well as harmful.
Exploitation uses reliable signals to show useful products.
Feedback becomes problematic when the system overinterprets policy-induced data as an unbiased view of demand, or when concentration undermines commercial or customer objectives.
A rising share for a popular product is not by itself proof of bias.
A randomized comparison can help distinguish the effect of visibility from differences in underlying demand.

Counterfactual evaluation asks what an alternative policy would have achieved on the same population.
The missing outcomes under unchosen actions cannot be observed for each request.
Randomized logging and known action probabilities make a population-level estimate possible when the alternative has adequate support.
No estimator observes the unshown counterfactual for an individual shopper.

## 4. Definitions and vocabulary

| Term | Working definition | Common misuse | Unit or record |
|---|---|---|---|
| Request | A decision opportunity on a defined surface and time | Treating every page event as independent | Request ID |
| Context | Information available before the policy acts | Including post-display clicks as features | Request-level fields |
| Candidate set | Items eligible for the action at decision time | Assuming it equals the full catalogue | Item IDs per request |
| Logging policy | Policy that generated historical actions | Using the current model version in place of the historical one | Probability distribution over actions |
| Target policy | Policy being evaluated or proposed | Calling observed logged performance its value | Probability distribution over actions |
| Action | Choice to be evaluated, such as one item or an ordered slate | Mixing per-item and whole-slate probabilities | Item or ordered list |
| Exposure | Presentation opportunity defined by an event contract | Equating a server response with a view | Impression or viewable impression |
| Position | Location within a surface under a particular layout | Assuming rank 1 means the same across layouts | Rank and surface ID |
| Examination | A shopper's attention to an item | Assuming viewability proves attention | Latent event |
| Click | Recorded activation of an item or link | Treating no click as certain dislike | Binary or count per attribution window |
| Outcome | Prespecified response to a policy action | Changing the outcome after looking at results | USD/request, orders/request, etc. |
| Exposure bias | Observations are conditioned on what was shown | Equating missing outcomes with zero outcomes | Selection mechanism |
| Position bias | Outcome opportunity depends on placement | Claiming all rank differences are causal position effects | Rank-dependent mechanism |
| Feedback loop | Policy influences data that influence later policy | Calling every retraining cycle harmful | Time-indexed process |
| Popularity bias | Disproportionate visibility or reinforcement of popular items relative to a stated reference | Treating every popular item as unfairly exposed | Share by item or cohort |
| Propensity | Logging policy probability of the observed action given context | Using the item score as a probability | Number in (0, 1] |
| Target probability | Probability assigned to an action by the target policy | Assuming it is always 1 for a stochastic target | Number in [0, 1] |
| Support | Target actions have positive logging probability in relevant contexts | Assuming a displayed item has support in every context | Overlap condition |
| Positivity | Another name for the relevant nonzero-probability condition | Confusing tiny probability with adequate data | Probability and sample size |
| IPS | Average of observed rewards weighted by target/logging probability | Averaging only matched rows without correct normalization | Outcome per request |
| SNIPS | Weighted reward sum divided by weight sum | Claiming exact unbiasedness in finite samples | Outcome per effective weighted request |
| Importance weight | Target probability divided by logging probability | Omitting target probability for a stochastic target | Dimensionless |
| Effective sample size | Squared sum of weights divided by sum of squared weights | Treating it as an independent-row count in clustered data | Equivalent weighted observations |
| Clipping | Replacing large weights by a threshold | Describing clipped IPS as unbiased for original policy value | Dimensionless threshold |
| Reward model | Prediction of outcome for a context-action pair | Treating prediction as observed counterfactual truth | Expected outcome |
| Doubly robust estimate | Combines reward predictions with propensity-weighted residuals | Assuming robustness without overlap or correct policy logging | Outcome per request |
| Exploration | Deliberate random variation to collect information | Randomizing ineligible or unsafe items | Probability and traffic share |
| Exploitation | Choosing actions preferred by current estimates | Assuming a deterministic action provides universal support | Policy rule |
| Guardrail | A decision threshold limiting harm during a test | Treating a non-significant result as proof of safety | Prespecified metric and threshold |
| Interference | One unit's treatment affects another unit's outcome | Assuming customer randomization isolates shared inventory | Cross-unit effect |
| Attribution window | Time allowed to connect an outcome to a request or assigned unit | Extending one arm's window after the fact | Hours or days |
| Intent-to-treat | Analysis by randomized assignment regardless of receipt | Analyzing only customers who clicked | Assigned unit |
| Availability | An item can be promised for a specific customer and time | Using catalogue status as guaranteed stock | Location, quantity, timestamp |
| Contribution | Revenue less specified variable costs | Using gross revenue as profit | Currency per unit or request |

## 5. The mechanics, step by step

### Step 1: State the decision and estimand

Input: a proposed business change and a defined population.
Output: a single primary estimand and a decision rule.

Specify the eligible customer population, surface, geography, dates, and unit of randomization.
State the exact historical and proposed policies, including candidate generation and ranking.
Define the outcome and its time horizon before looking at results.
For example: average 7-day net contribution in USD per eligible recommendation request under target policy A, in a fixed region and week.
If a customer can generate several requests, also define customer-level aggregation and inference.

The target policy value is an expectation over the target population's contexts and the actions sampled from that policy.
For one action and immediate outcome, write $V(\pi)=E_X[\sum_a \pi(a\mid X) E(R(a)\mid X)]$.
Here $R(a)$ is the potential reward if action $a$ were taken under the specified intervention.
This expression is a definition, not evidence that the value is identifiable.

A policy-value estimate is not the same as incremental business value.
To claim lift, estimate both target and comparator values for the same population, period, reward definition, and eligible opportunity.
An incremental value per request does not automatically scale linearly to all future traffic.
Stock, customer behavior, and competitors can change at rollout.

Separate four planning concepts.
A forecast estimates likely future outcome under stated assumptions.
A plan allocates resources to actions.
A target is the desired performance threshold.
A commitment is an approved promise to deliver or fund a specific outcome.
An offline estimate is evidence for a decision; it is not a revenue commitment.

### Step 2: Map the exposure mechanism

Input: user journey, UI implementation, and event architecture.
Output: a causal and measurement map.

Trace retrieval, filters, ranker, re-ranker, page response, rendering, scrolling, interaction, and downstream transaction.
Record the policy-controlled choices at each stage.
Mark stock availability and merchandising rules that affect eligibility before ranking.
Mark campaign, price, season, device, customer context, and product features that influence both selection and response.

Ask whether the chosen unit is an item, a slot, a slate, or a session.
If the page shows several items, a click on one can depend on its neighbors.
A single-item analysis that ignores those neighbors evaluates a narrower intervention and requires extra assumptions.
If items are chosen sequentially without replacement, the joint probability of a slate is the product of the actual conditional probabilities in order.
That product must include every random selection that defines the action being evaluated.
It is not the product of marginal probabilities unless the appropriate independence condition holds.

Use a diagram to identify conditioning on post-treatment events.
For example, filtering the analysis to customers who clicked selects on a variable affected by the recommendation.
A click-only sample cannot identify purchase lift for all assigned customers.
Compare assigned or eligible units according to the defined estimand rather than conditioning away nonresponders.

### Step 3: Define and log the decision opportunity

Input: an eligible request before the policy samples its action.
Output: a joinable, versioned request and action record.

Log an immutable request ID, pseudonymous unit ID where permitted, timestamp, surface, and experiment assignment.
Capture the eligible set or an auditable reconstruction snapshot, policy version, and relevant pre-action context.
Record action identity, ordering, slot, and exact conditional or joint propensity appropriate to the analysis.
Keep the policy probability that was used at the time, not a probability recalculated from a later model.

A system may retrieve a fixed set before randomizing the ranking.
If so, the logged propensity only supports interventions inside that retrieved set unless retrieval itself has support and is logged.
A product excluded with probability one cannot be evaluated by reweighting those logs for that request.
Support must cover the whole target intervention, not merely the last visible ranking step.

Log filters and reasons: unavailable, restricted, incompatible, duplicate, and merchandising exclusion.
Do not encode a restricted product as a negative preference label.
Keep the inventory snapshot and store or fulfillment location used by the decision.
Record whether exposure was selected, rendered, viewable, clicked, purchased, canceled, and returned as distinct event types.

Define event deduplication and stable joins before analysis.
Use event time for the decision and transaction time for attribution.
Specify how late events and data corrections update the mature outcome dataset.
A missing event can mean instrumentation failure, not absence of an action.
Monitor event-loss and join rates by treatment and device.

### Step 4: Define the logging and target probabilities

Input: deployed sampling algorithm, eligible set, and target policy.
Output: $p_i=\mu(a_i\mid x_i)$ and $q_i=\pi(a_i\mid x_i)$ for every logged decision.

For a deterministic target, $q_i$ is 1 when the logged action matches the target action and 0 otherwise.
For a stochastic target, compute its probability for the logged action on that same historical context and eligible set.
The logging probability must include the probability of entering the exploration branch and choosing the action within that branch.
If both exploitation and exploration can choose the same item, add both contributions.

For example, with two eligible items, an exploitation choice A, and 20% uniform exploration, $\mu(A)=0.8+0.2/2=0.9$ and $\mu(B)=0.2/2=0.1$.
The fact that B was selected from a uniform exploration branch does not make its unconditional propensity 0.5.
For stateful algorithms, condition on the state available before the draw and log it or an auditable policy record.
A random seed alone does not supply the probability unless the sampling distribution and context are recoverable.

Apply eligibility rules before computing the final selection distribution.
If the set has three products on one request and two on another, a uniform exploratory probability changes accordingly.
Ensure probabilities sum to one over possible actions for the actual decision point.
A score, click prediction, or softmax input is not itself a propensity unless the deployed policy samples from that exact normalized distribution.

### Step 5: Check overlap before estimating

Input: target action probabilities and the actual historical logging distribution.
Output: support diagnostics and an explicit scope of inference.

The required condition is $\pi(a\mid x)>0$ only where $\mu(a\mid x)>0$ for contexts in the target population.
This is an action- and context-specific condition.
A product shown somewhere in the dataset does not establish support for every store, customer, or page position.
A deterministic logger usually supplies no nonparametric evidence for alternatives it never selected in a context.

Report unsupported target probability mass when it is computable from reconstructed distributions.
Report the fraction of requests with any unsupported target action.
Break the diagnostic down by surface, customer segment where lawful, inventory state, and candidate-set size.
A small average unsupported fraction can hide an entirely unsupported subgroup.
If only selected-action propensities were logged and the full distribution cannot be reconstructed, the missing-support diagnostic itself may not be identifiable.

Tiny positive propensities satisfy formal positivity but can still be practically useless.
One rare matched action can dominate the estimate.
Compare weight distributions, the largest weights, and effective sample size.
If support is inadequate, restrict the target population or policy and name that restriction, collect randomized data, or run an online experiment.
Do not fill unsupported cells with zero and claim unbiased policy value.
A model prediction in an unsupported cell is extrapolation, not recovered experimental evidence.

### Step 6: Calculate the one-step IPS estimate

Input: $n$ logged requests with action $a_i$, context $x_i$, reward $r_i$, and probabilities $p_i$ and $q_i$.
Output: an estimated target-policy mean on the logged context distribution.

Define $w_i=q_i/p_i$.
The IPS estimate is $\widehat V_{IPS}=n^{-1}\sum_i w_i r_i$.
Every request remains in the denominator, including zero-weight mismatches.
Under correct logging probabilities, consistency, overlap, a suitable sampling design, and no relevant unmeasured selection beyond the logged policy, IPS recovers the target mean in expectation for the defined one-step setting.
The reward for the observed action must be measured under a comparable attribution rule.

For a deterministic target, unmatched actions have zero weight.
A matching action logged with probability 0.1 receives weight 10.
That one reward represents ten comparable opportunities in expectation, not ten actual purchases.
If the action definition is a full slate, use the joint slate probability, which may be very small.
Slot-level weights cannot generally substitute for a slate probability when outcomes depend on the full slate.

To compare policies, compute both values on the same eligible records when possible.
Use the paired difference and an uncertainty method appropriate to the logging design.
Report denominators, missing outcomes, distribution of weights, and segmentation rules with the estimate.
A large number of logged events is not enough when very few have material target weight.

### Step 7: Use SNIPS, clipping, and doubly robust methods carefully

Input: the same matched data plus optional out-of-sample reward predictions.
Output: sensitivity estimates, not a license to ignore missing support.

SNIPS is $\sum_i w_i r_i/\sum_i w_i$, provided the denominator is positive.
It often stabilizes scale because the weights are normalized by their observed sum.
It is generally biased in finite samples even when its assumptions hold, although it can be consistent under standard conditions.
If all weights are zero, SNIPS is undefined, not zero.

Weight clipping replaces $w_i$ by $\min(w_i,c)$ for a prespecified threshold $c$.
Clipping trades variance for bias relative to the original target value.
Show the threshold, share of rows and weight mass affected, and a sensitivity curve across reasonable thresholds.
A clipped result must not be labeled an unbiased IPS estimate of the original policy.
Clipping cannot manufacture outcomes for unsupported actions.

A contextual doubly robust estimate combines a reward model $\hat m(x,a)$ with an importance-weighted residual:

$\widehat V_{DR}=n^{-1}\sum_i[\sum_a\pi(a\mid x_i)\hat m(x_i,a)+w_i\{r_i-\hat m(x_i,a_i)\}]$.

Under appropriate regularity and overlap conditions, the estimator can remain consistent if either the reward model or propensity model is correctly specified, subject to the estimator's assumptions and fitting procedure.
With known randomized probabilities, use those known probabilities rather than replacing them with an unvalidated classifier.
Fit the reward model on separate data or use cross-fitting to reduce overfitting-related bias.
A reward model can extrapolate poorly for rare items, new contexts, and policy changes that alter neighboring slate items.
Even a doubly robust method cannot identify arbitrary unsupported actions without additional modeling assumptions.

### Step 8: Quantify uncertainty at the right unit

Input: weighted request-level outcomes and experimental design.
Output: uncertainty interval and a statement of its limitations.

Effective sample size is $(\sum_i w_i)^2/\sum_i w_i^2$ when the denominator is nonzero.
It describes concentration of weights, not the number of independent customers.
Repeated requests by one customer and shared inventory create dependence.
Use customer-clustered uncertainty if customers are the independent sampling units and that assumption is credible.
For store-level randomization or shared inventory, use the actual randomization unit or a design-appropriate method.

Bootstrap clusters, not individual requests, when the sampling design supports that procedure.
Recompute the full estimator within each resample.
Very few clusters, severe weights, or adaptive logging can invalidate a simple bootstrap approximation.
Consult a statistician on suitable inference rather than reporting a narrow interval produced by an inappropriate resampling unit.

A confidence interval summarizes uncertainty under its sampling and modeling assumptions.
It does not include all bias from logging errors, nonoverlap, unmeasured interference, or future drift.
Prespecify the minimum business-relevant effect and a decision rule.
An interval spanning both meaningful harm and meaningful gain supports more information, not a claim of equivalence.

### Step 9: Plan safe exploration

Input: candidate action space, costs, and guardrails.
Output: an auditable randomized policy with a stop rule.

Begin with hard constraints: legal restrictions, compatibility, location-specific availability, service promise, and customer permissions.
Randomize only among products that pass these checks.
A 5% exploration allocation means 5% of the defined decision opportunities enter a randomization branch; it does not mean 5% of sales are at risk.
State whether exploration is per request, per customer, or per slot.

A simple epsilon-greedy policy exploits the preferred action with probability $1-\epsilon$ and samples from a specified eligible distribution with probability $\epsilon$.
The action probabilities combine both paths when the preferred action can also be sampled during exploration.
Uniform exploration is transparent but can waste impressions on unlikely products.
A nonuniform exploration distribution may be more efficient if it retains adequate support for the policy comparisons of interest.
Always log the actual distribution or enough information to reconstruct it.

Stratify exploration by surface and context if this is part of the intended policy.
Set a budget in traffic, expected downside, and operational capacity.
Estimate possible opportunity cost using plausible reward ranges, not a guaranteed loss figure.
Give the inventory owner authority to remove products whose available-to-promise status changes.
Define a safe fallback if probabilities, inventory, or eligibility information are missing.
A fallback changes the action distribution and must itself be logged.

Adaptive bandits change allocation based on prior outcomes.
They can be useful for balancing learning and current reward, but introduce time dependence and potentially very uneven exposure.
Log pre-draw conditional propensities and the policy state or version at each decision.
Use inference that accounts for adaptive assignment and dependence.
Do not assume a conventional fixed-allocation test remains valid merely because the bandit is randomized.

### Step 10: Run the online policy comparison

Input: an offline assessment and an approved experiment design.
Output: a causal comparison for the randomized population, subject to experiment validity.

Choose a randomization unit that limits contamination and matches the decision.
Customer-level assignment can keep repeated visits in the same arm.
It does not by itself prevent treatment from consuming inventory needed by control customers.
When stock sharing is material, consider a cluster or switchback design and analyze it at the assigned level.
State the assumptions and trade-offs: clusters reduce interference but typically reduce effective sample size; switchbacks can have carryover and time trends.

Pre-register the primary outcome, eligibility, duration, mature outcome window, randomization, analysis, and stopping rule.
Specify treatment and control policy versions and how the candidate set and inventory checks differ.
Run an assignment and logging quality check, including sample-ratio mismatch and differential event loss.
Examine availability, cancellations, returns, and service metrics as well as clicks.
Analyze the assigned population by intent to treat; a separate per-exposure analysis is descriptive unless its selection issues are addressed.

Avoid peeking with an unadjusted fixed-horizon test and stopping at the first favorable result.
If interim monitoring is required, use an approved sequential design or a separately specified safety-monitoring procedure.
Make rollout conditional on both decision-quality evidence and operational readiness.
A neutral experiment with an interval excluding commercially important harm can support a different decision from one with an interval too wide to rule out harm.

### Step 11: Close the loop responsibly

Input: experiment results, action logs, and downstream operational data.
Output: a release, revision, or stop decision with an owner and review date.

Archive policy versions, eligibility rules, model features, experiment assignment, and outcome definitions.
Track exposure distribution, concentration, new-item opportunity, customer value, margin, inventory, and service by relevant segments.
Use randomized holdouts or periodic exploration to detect drift and distinguish demand changes from policy reinforcement.
Review product availability before interpreting a drop in exposure as a ranking failure.

The model owner evaluates data quality and retraining risks.
The commercial owner decides whether observed upside meets the agreed threshold.
The operations owner decides whether fulfillment constraints remain acceptable.
The experiment owner documents whether to roll out, continue learning under an approved design, or revert.
Do not transfer the accountability for a business decision to the estimator.

### Step 12: Separate item appeal from exposure opportunity

Input: a position comparison that appears to favor an established item.
Output: a testable decomposition, not a presumed causal explanation.

Write the observed click rate at a given position as a product of several mechanisms.
A shopper must have an opportunity to see the unit.
They must notice it, consider it, and choose to click.
The first two mechanisms are affected by presentation; the latter two also depend on the offer.
This conceptual decomposition is useful even when examination is not directly observed.

An examination model sometimes assumes that an item can be clicked only after examination.
It may further assume that the examination probability depends on position and the click probability after examination depends on item relevance.
These are modeling assumptions, not identities guaranteed by the interface.
Product images, neighboring offers, price badges, and visual emphasis can violate the separation.
A shopper may recognize a product without triggering the chosen viewability event.
Conversely, an item can be inside the viewport without receiving attention.

Suppose a position model estimates examination probabilities of 0.8 at rank 1 and 0.4 at rank 3.
If the conditional probability of clicking after examination were 0.10 at both ranks, expected click probabilities would be 0.08 and 0.04.
The twofold raw difference would arise entirely from the stipulated examination probabilities.
This is an arithmetic illustration, not an empirical claim that these are real position rates.
Estimating the examination parameters from observational click logs requires additional identification assumptions.

An order-randomization test can hold the eligible products and page design fixed while varying their order.
If A and B appear in both orders, compare outcomes for A in rank 1 versus rank 2 and for B in rank 1 versus rank 2.
The estimand must specify whether the neighboring item also changes position.
That is an order intervention: it does not necessarily isolate a pure rank effect independent of neighboring items.
If the rest of the slate changes too, the interpretation becomes a broader page-policy effect.

Use the full request population as the denominator for a policy-level outcome.
If the outcome is an item click, record whether the user could interact with other items.
If multiple clicks are possible, define whether the outcome is any click, count, or a particular item's click.
If only one click is possible due to interface design, item outcomes are competing events.
The analysis must match this mechanism rather than pretending that each item has an independent click trial.

Document whether rank is absolute on the page or relative within the recommendation component.
On a mobile device, rank 2 in a horizontally scrolling carousel may be partly visible.
On a desktop device, rank 2 may be fully visible.
The same numerical rank is therefore not necessarily the same treatment.
Use surface, layout, and device definitions to narrow the comparison or explicitly model the differences.

Position correction is not a license to multiply all observed clicks by a global factor indefinitely.
The correction depends on the population, layout, and exposure mechanism used to estimate it.
Large inverse examination factors can produce the same variance problems as rare logging actions.
Compare corrected estimates with randomized holdouts and report uncertainty.
Refresh the evidence when the interface changes substantially.

### Step 13: Specify a complete reward and attribution ledger

Input: the business objective and the transaction lifecycle.
Output: a reproducible reward table with a clear maturity rule.

Start with the unit on which the policy makes a decision.
Choose whether reward is measured per request, session, assigned customer, or another unit.
Customer-level assignment can coexist with request-level logging, but the aggregation and weighting must be explicit.
An average per request weights customers with many requests more heavily.
An average per customer weights each assigned customer once.
These are different estimands, even when calculated from the same events.

Define gross sales as price paid after discounts if that is the business convention.
List costs to subtract, such as cost of goods, pick and pack, payment fees, delivery, and expected or realized return expense.
If a cost is unavailable, label the metric partial contribution rather than net profit.
Include negative rewards when refunds or variable costs exceed revenue.
Do not clip negative rewards to zero merely to simplify an estimator.

Choose the treatment-to-outcome window before outcomes are inspected.
A 7-day window captures only transactions and reversals occurring within its defined rules.
A product with a 30-day return window may need a longer maturity period for realized return cost.
Alternatively, a modeled return reserve can be included, but then the reward incorporates a model assumption.
Compare estimates based on observed returns and modeled reserves as separate quantities.

Consider repeated exposures.
If one order follows three recommendation requests, a last-touch rule assigns the order to the last eligible request.
A first-touch rule assigns it to the first.
A fractional rule allocates a fixed total across eligible requests.
Each rule describes measurement, not necessarily the causal contribution of each exposure.
For a randomized customer-level experiment, all-customer contribution can avoid some arbitrary event attribution while still requiring a defined follow-up window.

Maintain a ledger for each reward component.
Record source system, timestamp, currency, conversion rate if applicable, return status, and revision history.
Ensure a correction is applied consistently to all experiment arms.
Report the share of records that has not matured and the share that cannot be joined.
If event loss differs by treatment, a complete-case estimate can be biased even when the original assignment was random.

Before using a reward for model training, ask whether the signal is policy-dependent.
Purchases reflect both customer demand and previous exposure.
Attribution rules can further amplify the apparent value of frequently displayed items.
Preserving the raw events and policy record allows later analyses to revise attribution without recreating the past.
The reward definition should be versioned along with the policy and model.

### Step 14: Evaluate concentration without confusing it with causation

Input: eligible-item and exposure histories over successive model versions.
Output: a concentration report and a causal question for follow-up.

Calculate the share of eligible impressions received by the top 10 products over a stated interval.
Use the same surface and denominator across periods.
Also calculate shares for new items, suppliers, and product groups where these distinctions are legitimate and relevant.
Report the number of eligible items and their in-stock opportunities.
An item that was unavailable for the whole week should not be counted as having the same opportunity as an in-stock item.

Define an exposure opportunity for a supplier as the number of requests on which at least one of its products passed the eligibility rules.
This is a business diagnostic, not an inherent fairness entitlement.
The numerator can be the number of displayed supplier products, but a multi-slot page may show more than one.
Therefore, say whether the share counts requests, slots, or distinct item impressions.
Do not divide displayed item impressions by eligible requests without noting that the result may exceed one.

Inspect the training-data distribution alongside the exposure distribution.
An item can gain training weight because it was displayed more, because it converted better, or because a sampling process duplicated its events.
A count of purchases alone does not separate those possibilities.
Check exposure-normalized click and purchase summaries, recognizing that these are still confounded by position and context.
Use randomized traffic to estimate the effect of allocating a defined exposure opportunity.

Examine the dynamic horizon.
A single-week randomization can identify a near-term policy contrast under its assumptions.
It might not capture the value of learning about a new product over several retraining cycles.
To study a learning policy, define the entire adaptive procedure as the treatment, including update frequency and exploration rules.
Compare it with a baseline procedure over an appropriately long horizon.
The analysis must account for adaptation and any interference caused by shared capacity.

Avoid optimizing a concentration metric in isolation.
Forcing all products to equal exposure can reduce relevance, increase stockouts, or promote products that are unsuitable for some shoppers.
Conversely, maximizing short-term clicks alone can reduce discovery and create dependence on a small group of suppliers.
Make the trade-off explicit in a policy constraint or decision rule.
The relevant owner should approve the constraint and its business rationale.

### Step 15: Diagnose distribution shift and feasibility

Input: a target deployment plan and historical evaluation population.
Output: a transportability assessment and any population restrictions.

Historical contexts represent the customers, locations, season, inventory, and interface observed during collection.
The deployment population may differ along any of these dimensions.
A target policy that performs well during a holiday promotion need not have the same value in a normal week.
The model may also change shopper behavior, altering the distribution of future sessions and requests.
One-step offline evaluation generally treats the context distribution as fixed.
It does not automatically capture those longer-term changes.

Compare the historical and proposed deployment distributions on pre-action variables.
Useful checks include store, surface, candidate-set size, device, stock status, price range, and customer lifecycle where lawful.
Large differences do not prove failure, but they make direct extrapolation less credible.
An apparent match on recorded variables does not prove that all relevant differences were observed.
State which covariates were compared and which important factors were unavailable.

If deployment will use a different eligible set, do not silently apply a historical action probability from the old set.
Reconstruct the historical policy and the target policy on a common, decision-time set where possible.
If the intervention includes changing eligibility or retrieval, evaluate that change as part of the policy.
A ranking experiment inside a fixed candidate set cannot establish the value of a broader retrieval change.
If historical support for new retrieval is absent, collect new randomized data or rely on explicitly labeled extrapolation.

Inventory creates both feasibility and outcome risks.
A product may be eligible when the page is generated and unavailable by checkout.
Log the timing of the inventory snapshot and the later order outcome.
Define whether the policy values an attempted display, a promised order, or a fulfilled order.
The answer affects the reward and the time needed for maturity.
Operations should review the difference between displayed availability and actual fulfillment.

### Step 16: Turn the assessment into an operating decision

Input: policy-value evidence, experiment diagnostics, and guardrails.
Output: a written decision with conditions and accountable owners.

Use a decision table with four possible results.
If upside is credible and all guardrails pass, the authorized owner may approve a staged rollout.
If upside is plausible but uncertainty spans material harm, keep the comparison in an approved learning phase.
If a hard safety or service guardrail fails, stop or revert according to the predefined protocol.
If measurement validity fails, repair instrumentation before interpreting the outcome as a policy effect.

For each result, specify what evidence changes the decision.
Examples include additional mature transactions, a repaired propensity log, or a stock-synchronization fix.
Set the review date and the owner who has authority to approve the next stage.
Do not continue exploration indefinitely without a learning objective or cost limit.
Do not silently redefine success after observing the result.

Rollout itself is another intervention.
Increasing traffic can change supply pressure, supplier behavior, and latency.
Use staged allocation and monitor guardrails at each stage.
Preserve a randomized control or holdout where the business and experimental design allow it.
A holdout is useful only if its assignment, policy version, and measurement remain stable enough to interpret.

When the policy is retrained, record the source-data window and exposure distribution.
Maintain the ability to reproduce which policy generated each training example.
Review whether a recent exploration campaign dominates the labels for a small category.
Ensure that emergency overrides and stock exclusions are represented in the logs.
A model trained on the new policy's data should be evaluated as a new policy version, not assumed to inherit the previous experiment's result.

## 6. Worked example

### Setting and policy definition

A regional grocer considers recommending one of two eligible breakfast products in a single slot.
The following eight records are illustrative, not a company result.
The action is the one selected product for a recommendation request.
The outcome is net contribution in USD attributed within 7 days to that request under a fixed deduplication rule.
For this exercise, assume each request belongs to a different customer and no shared-stock interference occurs.
Actual systems often violate both simplifying assumptions.

The historical policy chooses item A with probability 0.75 and item B with probability 0.25 on each of these requests.
Both products are eligible on all eight requests.
The target policy chooses B with probability 1 on every request.
The logged action probability is the probability of the observed action, not the probability of the target action.
Outcomes of zero mean no attributed contribution during the window, not necessarily permanent disinterest.

| Request | Logged action | Logging probability | Reward, USD/request | Target probability of logged action | Weight | Weighted reward, USD |
|---:|---|---:|---:|---:|---:|---:|
| 1 | A | 0.75 | 0.00 | 0 | 0 | 0.00 |
| 2 | B | 0.25 | 1.20 | 1 | 4 | 4.80 |
| 3 | A | 0.75 | 0.80 | 0 | 0 | 0.00 |
| 4 | B | 0.25 | 0.00 | 1 | 4 | 0.00 |
| 5 | A | 0.75 | 0.00 | 0 | 0 | 0.00 |
| 6 | A | 0.75 | 1.00 | 0 | 0 | 0.00 |
| 7 | B | 0.25 | 0.60 | 1 | 4 | 2.40 |
| 8 | A | 0.75 | 0.40 | 0 | 0 | 0.00 |

The reward and attribution rule are deliberately simplified.
A real contribution metric must specify discounts, cost of goods, payment costs, shipping, returns, and whether negative outcomes are allowed.
Each transaction must be attributed at most as specified; otherwise one order can be counted against several recommendation requests.
The example is a one-slot policy value, not a full-page recommendation value.

### IPS arithmetic

The target probability of the observed action is zero for the five A rows.
For each B row, the target probability is 1 and the logging probability is 0.25.
Thus each B row has weight $1/0.25=4$.
The weighted reward sum is $4.80+0.00+2.40=7.20$ USD.
There are eight original requests, not three, in the IPS denominator.
The IPS estimate is $7.20/8=0.90$ USD per request.

The arithmetic does not mean the business earned 7.20 USD on the three B requests.
Observed B contribution is $1.20+0+0.60=1.80$ USD.
The larger weighted total is a statistical expansion used to estimate a target-policy mean.
The estimate is noisy because only three requests match the deterministic target.

### SNIPS and weight concentration

The total weight is $4+4+4=12$.
SNIPS is $7.20/12=0.60$ USD per request-equivalent.
The sample has three B actions when the logging policy expected two on average over eight requests.
SNIPS normalizes for this realized weight total, while IPS retains the fixed request denominator.
The two methods need not agree on a small sample.
Neither estimate establishes that the target will improve business performance.

The squared-weight sum is $4^2+4^2+4^2=48$.
The effective sample size is $12^2/48=3$ weighted observations.
This is much smaller than eight logged requests.
It is not a guarantee of three independent customers in a real system; here independence was assumed for illustration.

### Historical comparison and decision

The observed total contribution for all eight logged actions is $4.00$ USD.
The observed historical-policy sample mean is $4.00/8=0.50$ USD per request.
The IPS point estimate difference between target B and the observed historical mean is $0.90-0.50=0.40$ USD per request.
The SNIPS point estimate difference is $0.60-0.50=0.10$ USD per request.
These are two different estimators of a comparison, not two independent business outcomes.
With only eight observations and three weighted matches, neither difference is a reliable rollout decision.

Suppose the category owner needs evidence of at least 0.05 USD additional contribution per request without a material cancellation increase.
That threshold is a decision target, not a forecast or commitment.
The plan is to collect a larger randomized sample, measure cancellations and returns after maturity, and examine uncertainty.
The forecast would require additional modeling and assumptions about traffic, season, and supply.
No revenue commitment follows from these eight rows.

### What changes with a position effect?

Suppose a future page offers two slots and B is usually placed second.
A click rate difference between A in slot 1 and B in slot 2 combines item appeal, placement, and potentially neighbor effects.
Randomizing the order among otherwise eligible products can identify an order-policy contrast for the randomized population.
If the outcome is measured for the entire two-item page, the action is the ordered slate and its joint probability is required for full-slate IPS.
If the analysis estimates a marginal item-position effect instead, it needs a specifically defined intervention and defensible assumptions about neighboring items and interference.

### What changes when B is out of stock?

If B cannot be promised to a customer, it is outside that request's eligible action set.
A target policy that always chooses B is then infeasible for that request.
Do not assign B a hypothetical reward of zero and call the policy evaluated.
Either define an availability-aware fallback as part of the target policy or restrict the estimand to requests where B was eligible.
The two choices answer different commercial questions.
A current stock status is not proof of the historical stock status at the time of a logged request.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Copy it into a file and run it with `python filename.py`.
It reproduces the single-slot example, checks probability inputs, calculates IPS, SNIPS, effective sample size, and the observed logging-policy mean.
It also demonstrates a configurable clipping sensitivity calculation.
The program cannot determine support for unobserved actions from selected-action probabilities alone.
It does not calculate a confidence interval and does not simulate independence or infer causal validity.

```python
from math import isfinite

# Each row is (observed action, logging propensity, reward in USD).
rows = [
    ("A", 0.75, 0.00),
    ("B", 0.25, 1.20),
    ("A", 0.75, 0.80),
    ("B", 0.25, 0.00),
    ("A", 0.75, 0.00),
    ("A", 0.75, 1.00),
    ("B", 0.25, 0.60),
    ("A", 0.75, 0.40),
]


def policy_summary(records, target_action, clip=None):
    """One-step deterministic-target estimates; caller establishes support."""
    if not records:
        raise ValueError("at least one request is required")
    if clip is not None and (not isfinite(clip) or clip <= 0):
        raise ValueError("clip must be a positive finite number")

    weights = []
    rewards = []
    clipped_count = 0
    for action, propensity, reward in records:
        if not isfinite(propensity) or not 0 < propensity <= 1:
            raise ValueError("observed-action propensity must be in (0, 1]")
        if not isfinite(reward):
            raise ValueError("reward must be finite")
        weight = (1.0 if action == target_action else 0.0) / propensity
        if clip is not None and weight > clip:
            weight = clip
            clipped_count += 1
        weights.append(weight)
        rewards.append(reward)

    weighted_reward = sum(w * r for w, r in zip(weights, rewards))
    weight_sum = sum(weights)
    weight_squared_sum = sum(w * w for w in weights)
    ips = weighted_reward / len(records)
    snips = weighted_reward / weight_sum if weight_sum else None
    ess = weight_sum * weight_sum / weight_squared_sum if weight_squared_sum else 0.0
    observed = sum(rewards) / len(records)
    return ips, snips, ess, observed, clipped_count


ips, snips, ess, observed, _ = policy_summary(rows, "B")
print(f"IPS: {ips:.2f} USD/request")
print(f"SNIPS: {snips:.2f} USD/request" if snips is not None else "SNIPS: undefined")
print(f"Effective sample size: {ess:.2f}")
print(f"Observed logging mean: {observed:.2f} USD/request")
clipped_ips, _, _, _, count = policy_summary(rows, "B", clip=2.0)
print(f"Clipped IPS: {clipped_ips:.2f} USD/request; clipped rows: {count}")
```

Expected output:

```text
IPS: 0.90 USD/request
SNIPS: 0.60 USD/request
Effective sample size: 3.00
Observed logging mean: 0.50 USD/request
Clipped IPS: 0.45 USD/request; clipped rows: 3
```

The clipped value is lower because each positive target weight falls from 4 to 2.
This is a sensitivity illustration, not proof that 0.45 USD is a less biased estimate.
When a stochastic target is evaluated, replace the deterministic indicator with the actual target action probability.
When action probabilities are context-specific, supply the correct probability for each logged decision.
When outcomes have not matured, wait for the full attribution window or use a separately justified delayed-outcome method.

## 8. Data, standards, and regulatory context

### Event contract

A practical minimum event contract includes request ID, event time, policy and model versions, candidate eligibility, selected action, probability, surface, slot, and experiment assignment.
The probability must be tied to the same sampling stage and action definition as the analysis.
Record a render or viewability event separately from the server-side selection.
Include device and layout version where permitted because rank is not comparable across all surfaces.

Join clicks and transactions through stable identifiers and a documented attribution rule.
Specify whether an item can be recommended repeatedly within the attribution window.
Define how returns and cancellations affect net reward and when the reward is final.
Late-arriving events should have a documented maturity cutoff and correction process.
Monitor missing propensity, invalid probability, duplicate request, impossible slot, and unmatched purchase rates by policy arm.

Data minimization matters for both privacy and operational quality.
Store only the identifiers and context needed for the stated measurement purpose, with controlled access and retention.
A pseudonymous customer identifier can still be personal data when it can be linked to a person.
Separate identity resolution from analytic access where practical.
Document legal basis, consent or objection handling where applicable, and deletion or retention obligations with qualified counsel.

### Standards and identifiers

GS1 Global Trade Item Numbers (GTINs) identify trade items when properly assigned and maintained.
They can help reconcile product records across retailer and supplier systems.
A GTIN does not prove current stock availability, establish a recommendation propensity, or define the user's preference.
A retailer's own item and offer identifiers may also be needed to distinguish location, price, or pack configuration.
The identifier mapping must be versioned when products or offers change.

The W3C Web Content Accessibility Guidelines (WCAG) provide accessibility guidance for web interfaces.
They are not a method for causal inference or a universal legal determination.
Layout and accessibility changes can change examination behavior, so record the interface version when interpreting position effects.
Applicable accessibility obligations depend on jurisdiction and context; consult the relevant legal and accessibility owners.

### Privacy and platform rules

The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, governs processing of personal data within its scope.
Its principles include purpose limitation, data minimization, and storage limitation; relevant rights and lawful-basis requirements depend on the processing.
GDPR Article 22 addresses certain solely automated decisions producing legal or similarly significant effects; ordinary product ranking is not automatically within that provision.
Counsel must assess the actual processing, jurisdiction, and applicable exceptions.
This is educational material, not legal advice.

For services subject to the EU Digital Services Act, Regulation (EU) 2022/2065 includes recommender-system transparency provisions, including Article 27, with obligations depending on service and scope.
Very large online platforms and very large online search engines have additional provisions, including Article 38 on a non-profiling option for recommender systems.
Do not assume these rules apply to every retailer or surface; ask counsel to assess the service and current law.
A disclosure is not a substitute for valid causal measurement, and an experiment's statistical design is not a legal basis for data processing.

The California Consumer Privacy Act, as amended by the California Privacy Rights Act, establishes rights and obligations for covered businesses and personal information, subject to statutory scope and exceptions.
Applicable implementing rules and definitions must be checked at deployment time.
A retention table and access-control design support compliance but do not establish compliance on their own.
Review cross-border transfers, sensitive data, children's data, and contractual supplier terms where relevant.

### Measurement boundary

A valid event schema does not by itself validate a randomization distribution.
A valid probability does not prove an impression was seen.
A viewability event does not prove human attention.
A recorded click does not prove satisfaction or net contribution.
A GTIN does not encode inventory by fulfillment location.
Treat each claim as requiring its own evidence.

## 9. Industry practice and case evidence

### Yahoo news recommendation exploration

Li and colleagues describe a contextual-bandit approach for personalized news recommendation and offline evaluation using randomized traffic in a Yahoo setting.
The paper explains how logged randomized choices can support replay-style evaluation of a new policy under its assumptions.
The setting concerns news recommendations, not grocery inventory.
It supports the general lesson that deliberate randomization creates evaluation data; it does not establish a grocery conversion lift.
Source: Li et al., *A Contextual-Bandit Approach to Personalized News Article Recommendation*, WWW 2010, linked below.

### Search position bias

Joachims and colleagues study how click data are influenced by result presentation and discuss using propensity weighting for unbiased learning to rank under specified assumptions.
Search result ranking is not identical to a retail recommendation carousel.
The transferable mechanism is that visibility affects observable clicks.
It does not follow that a single fixed position correction applies across devices, surfaces, and designs.
Source: Joachims, Swaminathan, and Schnabel, *Unbiased Learning-to-Rank with Biased Feedback*, WSDM 2017.

### Feedback in recommender systems

Chaney, Stewart, and Engelhardt model how algorithmic confounding can arise when recommender systems affect the data used to train future recommenders.
Their analysis illustrates how a feedback loop can distort subsequent learning and user experience under modeled conditions.
It is evidence about a mechanism, not a quantified estimate for every retailer.
A system-specific randomized holdout is still required to assess the magnitude of an actual business effect.

### Counterfactual risk minimization

Swaminathan and Joachims develop counterfactual risk minimization using logged bandit feedback and emphasize the role of variance in learning from importance-weighted data.
The work supports reporting uncertainty and weight concentration instead of optimizing only a point estimate.
Its assumptions and objective must be translated carefully to a particular product, reward, and action space.
The paper does not eliminate the need for logging support or operational guardrails.

### Business interpretation

A mature recommendation program commonly has separate retrieval, ranking, and constraints.
This structural decomposition is a practical design pattern, not evidence that any particular company runs the same experiment.
A rank-only randomized test does not measure the value of items excluded by retrieval.
An inventory-aware policy may improve clicks among available products while reducing canceled orders, or it may lose margin; those outcomes require measurement.
Supplier exposure is a commercial and governance concern, but exposure parity is not automatically the same as customer utility or a legal entitlement.
Select metrics and thresholds through an explicit business and legal review.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unshown treated as disliked | Negatives grow with catalogue size | Missing exposure becomes zero label | Established products dominate | Separate missing from observed feedback |
| Position ignored | Click rate falls sharply by rank | Visibility differs by slot | Misranked items | Randomize order and model placement |
| No target support | Target picks never-logged items | Logging probability is zero | Value not identified | Restrict policy or collect randomized data |
| Tiny support | Few rows carry most weight | Rare target matches | Unstable point estimate | Increase exploration and report ESS |
| Wrong propensity | Logged score differs from sampling chance | Branch probability omitted | Biased estimate | Log actual unconditional action probability |
| Wrong slate probability | Item marginals used for full-page action | Selection without replacement | Incorrect weights | Log joint conditional probabilities |
| Eligibility leakage | Later stock status used as context | Post-decision data enters reconstruction | Invalid policy comparison | Snapshot pre-action inventory |
| Render loss | Server impressions exceed client renders in one arm | UI failure differs by treatment | False exposure or click interpretation | Log render and analyze assigned units |
| Click-only analysis | Nonclickers disappear | Selection on treatment outcome | Inflated downstream conversion | Use intent-to-treat denominator |
| Reward immaturity | Recent cohort has fewer returns | Unequal observation windows | Artificial contribution lift | Use mature cohorts and same window |
| Double attribution | Orders exceed unique transaction count | Several impressions claim one order | Inflated reward | Specify deterministic attribution and deduplicate |
| Shared stock interference | Control availability falls as treatment sells | Treatment depletes common inventory | Contaminated contrast | Cluster or model spillovers where feasible |
| Unbounded exploration | Revenue or service declines in learning traffic | Unsafe candidates sampled | Customer and commercial harm | Hard eligibility, caps, stop rule |
| Clipping concealed | Estimate changes with threshold | Weight truncation introduces bias | Overconfident value claim | Publish thresholds and sensitivity |
| Adaptive inference ignored | Allocation varies sharply over time | Bandit state affects assignment | Invalid fixed-test uncertainty | Use design-aware inference |
| Simpson's paradox | Overall lift disagrees with key segments | Arm mix or traffic shifts | Wrong rollout decision | Prespecify and inspect relevant strata |
| Feedback concentration | New-item exposure collapses over retrains | Prior selections become labels | Discovery and supplier risk | Maintain measured exploration and holdout |
| Model leakage | Offline result exceeds online performance | Post-action features enter model | Unrealistic offline performance | Enforce feature time boundaries |
| Missing fallback logs | Policy error rate correlates with treatment | Fallback actions have another distribution | Wrong probabilities and outcomes | Version and log fallback decisions |
| Privacy overcollection | Unused identifiable fields retained | Logging exceeds purpose | Legal and security risk | Minimize, restrict, and expire data |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or assess |
|---|---|---|---|
| Correct action definition | Matches intervention to probability | Item-level weights used for slate outcome | Trace policy and action schema |
| Accurate logging propensity | IPS corrects selection probabilities | Missing branch or fallback probability | Audit sampler and compare empirical frequencies |
| Positive support | Target actions need historical chance | Deterministic logger excludes target | Inspect target mass against full logging distribution |
| Practical overlap | Sufficient weighted information | Nearly zero propensities | Report weight tail and ESS |
| Consistency | Observed reward corresponds to chosen action | Different UI version or hidden intervention | Version policies, layouts, and outcomes |
| Stable eligibility | Target action is feasible in context | Stale inventory or legal restrictions | Reconstruct decision-time candidate set |
| Comparable reward | Outcome measures the same construct | Different attribution or immature returns | Check outcome definitions and maturity by arm |
| Appropriate sampling population | Logged contexts represent target population | Season, geography, or traffic shifts | Compare context distributions and scope claims |
| No unmodeled interference | One unit's action does not alter another's outcome | Shared stock or supplier capacity | Inspect stock flows; choose cluster design |
| Valid independent unit | Uncertainty uses correct dependence structure | Multiple requests per shopper | Cluster by randomization or sampling unit |
| No unmeasured decision path | Probability describes all actual choices | Manual overrides not logged | Audit overrides and experiment assignment |
| Reward model adequacy for DR | Model side of DR reflects expected outcome if relied upon | New items or changing context | Out-of-sample calibration and subgroup error |
| Stable post-action measurement | Outcome collection does not differ by policy | UI prevents click logging in one arm | Compare event loss and audit client events |
| Appropriate horizon | Defined reward captures relevant impact | Returns or repeat purchase occur later | Compare mature windows and longer holdouts |

These assumptions are not a checklist that proves causality by declaration.
Some can be audited directly; others remain judgment calls supported by design and sensitivity analysis.
If an assumption fails, change the estimand, collect better data, or report the limitation rather than hiding it in a confidence interval.

### Interpreting the ledger in practice

An assumption can fail for only part of the traffic.
For example, the logging policy may have valid propensities for the main recommendation surface but not for an emergency fallback.
Exclude neither group silently.
Measure how much traffic used the fallback, explain its action mechanism, and define whether the target estimand covers it.
If no reliable fallback probabilities exist, the unsupported portion needs new data or an explicitly narrower claim.

The assumption of correct propensity is separate from the assumption of correct outcome.
A perfectly randomized action can still be paired with a broken purchase join.
A complete purchase table can still be paired with the wrong probability.
Assign independent owners and checks to both components.
Neither defect is repaired by a large sample.

Interference can also operate across time.
An early treatment purchase may deplete a product before a later control request.
The later control experience then reflects earlier treatment allocation.
A switchback that changes policy by hour may carry inventory effects across the boundary.
Its analysis needs an assessment of carryover and restocking, not just a time-of-day adjustment.

The same customer can experience both arms if assignment is made per request.
That may be acceptable for an immediate request-level estimand with negligible carryover.
It is inappropriate to assume no carryover when a recommendation changes a customer's basket or future interest.
For longer outcomes, stable customer-level assignment may better match the decision.
The price is potentially fewer independent units and slower learning.

If the policy changes a displayed price, the action includes more than product identity.
A probability for product selection alone cannot support a counterfactual about a different price.
The same applies to badge, promotion, placement, and bundle composition when those are changed by the target policy.
Represent the combined treatment faithfully, even if doing so makes support harder to obtain.
Otherwise the estimate answers a different question from the planned launch.

### A governance checklist for the decision record

Before data collection, record the business question in a form a reviewer can challenge.
Name the decision owner and the amount of downside that requires intervention.
List the policy surfaces and the geography in scope.
List excluded populations, restricted products, and stock constraints.
Identify the control policy and any existing holdout.

Record the randomization unit and how assignment is persisted.
Describe the candidate and action distributions, including fallback and overrides.
Identify the event that establishes an opportunity and the event that establishes a rendered item.
Describe how propensity values are generated, stored, and audited.
Identify who can change the policy during the collection period.

Record primary and secondary outcomes with units and time horizons.
List return, cancellation, and late-event rules.
State the minimum decision-relevant effect and uncertainty criterion.
Describe the inferential unit, clustering, and any planned interim analysis.
Prespecify how to respond to missing events or sample-ratio mismatch.

Record retention, access controls, and deletion procedures.
Identify data subject rights and platform-specific requirements for legal assessment.
Document whether supplier reporting exposes confidential commercial information.
Identify the operational owner for inventory and fulfillment metrics.
Record the experiment owner who can pause or revert the policy.

After analysis, archive the actual dates and population.
Record deviations from the planned protocol without rewriting history.
Show effect estimates and uncertainty alongside support and event-quality diagnostics.
Note any uncertainty from immature outcomes or model-based assumptions.
Document the rollout, continuation, revision, or stop decision.

### Example of a bounded exploration contract

The following is a design example, not a universal recommended allocation.
The eligible decision population consists of signed-in or anonymous sessions that are legally permitted to receive the defined recommendation, on one regional grocery surface.
Each request has a precomputed, location-specific eligible set.
The set excludes unavailable items, incompatible products, and restricted products for which the necessary conditions are not met.
The policy chooses one item for a single slot.

On 95% of requests, the policy chooses the highest-scoring eligible item.
On 5% of requests, it samples uniformly from the eligible set.
If there are five eligible items, the exploration probability of each is 0.05 divided by 5, or 0.01.
The exploitation winner also receives probability 0.95, for a total probability of 0.96.
Each nonwinner has probability 0.01.
These five probabilities sum to 1.00.

When only one item is eligible, it is chosen with probability 1.
When no items are eligible, a defined no-recommendation action is taken with probability 1.
The no-recommendation action is logged as such rather than represented by a missing request.
If the policy service fails, the fallback is separately versioned and its action distribution is recorded.
If the distribution cannot be reconstructed, those requests cannot be treated as if they followed the normal 95%/5% mixture.

Suppose the exploratory branch costs at most 0.20 USD of expected contribution per exploratory request under a planning assumption.
For 10,000 total requests, the expected number of exploratory requests is 500.
The resulting planning estimate of opportunity cost is 100 USD.
This is a budget scenario, not a measured loss or an upper confidence bound.
The actual cost depends on which items are sampled and what would have happened under exploitation.

Define a hard stop for invalid eligibility or restricted-item display.
Define a separate statistical monitoring plan for contribution and cancellations.
Inventory operations can remove items that cease to be available.
The experiment owner controls the overall pause and rollback.
The analyst reports realized exposure, propensity quality, and outcome maturity rather than assuming the plan was executed perfectly.

For off-policy evaluation of a target that always selects a particular nonwinner, the relevant propensity is 0.01 in this five-item context.
That produces an importance weight of 100 when it is selected and zero otherwise.
Positive support alone is therefore not evidence of useful precision.
A higher exploration rate, a targeted exploration distribution, or a direct online comparison might be needed.
The choice should be made before looking for a favorable estimate.

### Distinguishing three claims

Claim one is descriptive: an item received 10% of slots during the measured week.
It requires reliable slot counts and a defined denominator.
It does not establish what would have happened under a different policy.

Claim two is predictive: requests where an item was shown had a 4% observed purchase rate.
It requires reliable joins and a defined observation window.
It can be useful for forecasting within similar logged conditions.
It does not establish the purchase probability if the item were shown to all eligible shoppers.

Claim three is causal: assigning a defined policy rather than its comparator changed average contribution by a stated amount for the randomized population.
It requires a valid intervention, assignment, outcome measurement, and suitable treatment of interference.
It is limited to the population, policy versions, and period supported by the design.
The three claims should not be substituted for one another in an executive summary.

## 12. Exercises

### Exercise 1 — Draw the loop

Time: 15 minutes.
Deliverable: a diagram with at least eight nodes and two distinct event types.
Include candidate eligibility, policy selection, rendered exposure, action, reward, training labels, and the next policy.
Acceptance criteria: arrows identify the feedback path, and an unshown product is not treated as a negative outcome.

### Exercise 2 — Diagnose position bias

Time: 15 minutes.
A new product has 3 clicks in 100 impressions at position 3; an established product has 20 clicks in 100 impressions at position 1.
Deliverable: identify at least three competing explanations and a randomized comparison.
Acceptance criteria: do not infer a causal position effect or intrinsic preference from these two rates alone; specify what remains fixed and what is randomized.

### Exercise 3 — Define an estimand

Time: 20 minutes.
A retailer wants to replace a recommendation carousel for shoppers in one region.
Deliverable: one sentence defining population, action, comparator, reward, and time horizon, plus an interference risk.
Acceptance criteria: outcome has units; eligibility and attribution are defined; the intervention matches the action probability to be logged.

### Exercise 4 — Calculate policy value

Time: 20 minutes.
Four requests have rewards of 2, 0, 1, and 0 USD.
The target policy chooses B deterministically.
The logged actions are B, A, B, A, with logging probabilities 0.5, 0.5, 0.5, and 0.5 respectively.
Deliverable: compute the weights, IPS, SNIPS, and ESS.
Acceptance criteria: use all four requests in the IPS denominator and explain what the ESS counts.

### Exercise 5 — Check support and probability

Time: 20 minutes.
A policy chooses A for exploitation with probability 0.8 and otherwise samples uniformly from eligible A and B.
A proposed target always chooses B, but a second store's eligibility filter always removes B.
Deliverable: calculate the first store's action probabilities and explain the second store's identification and feasibility problem.
Acceptance criteria: account for both routes to A and distinguish zero probability from a very small positive probability.

### Exercise 6 — Design exploration

Time: 25 minutes.
A grocer wants information about 20 new products without compromising availability or regulated-product controls.
Deliverable: a short design naming randomization unit, eligible pool, sampling distribution, propensity log, budget, fallback, and stop authority.
Acceptance criteria: selection occurs only after hard constraints; the exact action probability includes every branch; inventory and privacy owners are involved.

### Exercise 7 — Assess an experiment

Time: 25 minutes.
An experiment reports a click lift, but treatment has more missing client render events and more orders later canceled for lack of stock.
Deliverable: a decision memo stating the primary uncertainty, two diagnostics, and one operational action.
Acceptance criteria: separate logging bias from business harm, preserve intent-to-treat analysis, and avoid claiming a verified net benefit.

### Exercise 8 — Diagnose reinforcement

Time: 25 minutes.
After three retraining cycles, top-ten items receive a higher exposure share while new-item sales fall.
Deliverable: a causal diagram, at least two non-feedback explanations, and an intervention with ownership.
Acceptance criteria: include availability and seasonal demand as alternatives; propose a safe holdout or randomization; identify the owner of the decision and the owner of instrumentation.

## 13. Self-check answers

### 1. Loop diagram rubric

A strong diagram runs from pre-action context to eligible candidates, selection, rendering, customer response, logged labels, training, and future selection.
It distinguishes selected actions from viewable impressions.
It labels inventory and business rules as determinants of eligibility.
A missing impression is not a recorded rejection.
A useful extension shows a randomized holdout as an independent source of comparison.

### 2. Position critique

The observed click rates are 3% and 20%, respectively.
They differ by 17 percentage points, but that difference combines product identity and position.
Different price, imagery, stock, audience, page layout, and season are other possible causes.
Randomize order among the same eligible items and comparable requests while logging the assigned order and render status.
The resulting contrast estimates the defined ordering intervention under the experiment's assumptions, not a universal position correction.

### 3. Estimand rubric

One valid answer is: difference in average 7-day net contribution in USD per eligible customer assigned to target versus current ordered-carousel policy in the named region and study period.
The answer must define transaction attribution and the eligible customer population.
Because this is a full-carousel intervention, log the assignment or joint slate probability appropriate to the design.
Shared fulfillment stock may make one customer's treatment affect another customer's control outcome.

### 4. Calculation

The weights are 2, 0, 2, and 0.
The weighted reward sum is $2\times2+0+2\times1+0=6$ USD.
IPS is $6/4=1.50$ USD per request.
SNIPS is $6/(2+2)=1.50$ USD per request-equivalent.
The squared-weight sum is $2^2+2^2=8$ and the weight sum is 4.
ESS is $4^2/8=2$ weighted observations, not necessarily two independent customers.
The equality of IPS and SNIPS in this sample is not a general identity.

### 5. Support

The first store has $P(A)=0.8+0.2(0.5)=0.9$ and $P(B)=0.2(0.5)=0.1$.
A target that always selects B has positive support in that store, although the small probability can yield noisy estimates.
In the second store, B is not eligible, so the always-B policy is infeasible there and has no support.
Use a defined eligible fallback or restrict the population; each changes the question.

### 6. Exploration rubric

A strong response filters products for availability, customer suitability, and restrictions before randomization.
It defines the probability of every chosen action including exploitation and exploration paths.
It specifies a capped traffic and downside budget, fallback behavior, and a named stop owner.
It logs the candidate set and policy version and includes operations and privacy review.
It does not claim a fixed exploration rate guarantees a particular sample size for every item.

### 7. Experiment rubric

A strong memo does not equate click lift with incremental profit.
It checks render-event completeness by arm and uses assignment-level outcomes rather than restricting to renderers.
It compares cancellation and mature net contribution by assignment.
It asks the fulfillment owner to investigate stock synchronization and applies the prespecified safety rule.
If measurement is unreliable or guardrails are breached, pause or revert according to the approved decision protocol.

### 8. Reinforcement rubric

A strong diagram connects prior exposure to observed sales, training labels, and later exposure.
It also includes availability, price, promotions, seasonality, and product quality as alternative explanations.
A randomized eligible-item exploration or holdout can isolate part of the policy effect.
The commercial owner decides the exposure and economic trade-off; engineering owns event fidelity; the inventory owner validates availability.
A short observational trend alone does not prove the feedback loop caused the sales decline.

## 14. Interview questions

1. **Why is an unshown item not a negative label?**  
   Strong: distinguishes missing outcome from observed rejection and describes selection.  
   Weak: assumes all catalogue items had equal exposure.
2. **What is position bias?**  
   Strong: explains differential examination and confounding with item quality.  
   Weak: equates a rank-by-click association with a causal rank effect.
3. **How do you log a propensity for epsilon-greedy selection?**  
   Strong: sums the exploration and exploitation routes for the chosen action.  
   Weak: records the model score or exploration-branch probability alone.
4. **What happens without overlap?**  
   Strong: identifies nonidentifiability and proposes feasible restriction or randomized data.  
   Weak: clips an infinite weight and claims support is restored.
5. **How do IPS and SNIPS differ?**  
   Strong: states denominators and finite-sample bias/variance trade-offs.  
   Weak: says SNIPS is always unbiased or always better.
6. **What does ESS tell you?**  
   Strong: quantifies weight concentration and separates it from independent cluster count.  
   Weak: treats total log rows as effective sample size.
7. **Can you use per-item propensity for a slate?**  
   Strong: defines the intervention and explains neighbor effects and joint probabilities.  
   Weak: multiplies arbitrary marginals without checking dependence.
8. **Why can an online test still be misleading?**  
   Strong: addresses event loss, interference, assignment, outcome maturity, and stopping.  
   Weak: assumes randomization fixes every measurement problem.
9. **How do you safely explore?**  
   Strong: enforces hard eligibility, logs probabilities, budgets downside, and assigns stop authority.  
   Weak: uniformly randomizes the entire catalogue regardless of stock.
10. **When does doubly robust estimation fail?**  
    Strong: names unsupported actions, incorrect action definition, and failure of both nuisance models.  
    Weak: treats the name as a guarantee against all bias.
11. **What is a feedback loop's commercial risk?**  
    Strong: links historical exposure to training labels and potential loss of discovery while recognizing genuine demand.  
    Weak: claims all concentration is evidence of discrimination.
12. **Who decides rollout?**  
    Strong: separates statistical evidence, business accountability, operational guardrails, and privacy review.  
    Weak: gives unilateral authority to the model or metric.

## 15. Key terms

- **Action:** The precisely defined choice, such as an item or ordered slate; see [definitions](#4-definitions-and-vocabulary).
- **Attribution window:** The interval in which outcomes are assigned to the intervention; see [event contract](#event-contract).
- **Bandit:** An adaptive decision procedure that balances exploration and exploitation; see [exploration](#step-9-plan-safe-exploration).
- **Candidate set:** Products eligible before the decision; see [exposure mapping](#step-2-map-the-exposure-mechanism).
- **Clipping:** A cap on weights that trades variance for bias; see [estimator choices](#step-7-use-snips-clipping-and-doubly-robust-methods-carefully).
- **Consistency:** The observed outcome corresponds to the specified intervention; see [assumptions](#11-assumptions-and-limits).
- **Contribution:** Defined revenue minus specified variable costs; see [worked example](#6-worked-example).
- **Counterfactual:** An outcome under an alternative action not observed for the same decision; see [mental model](#3-the-core-mental-model).
- **Doubly robust estimation:** Reward-model predictions plus propensity-weighted residuals; see [estimator choices](#step-7-use-snips-clipping-and-doubly-robust-methods-carefully).
- **Effective sample size:** A weight-concentration diagnostic; see [uncertainty](#step-8-quantify-uncertainty-at-the-right-unit).
- **Eligibility:** Pre-action business and safety conditions defining allowed products; see [logging](#step-3-define-and-log-the-decision-opportunity).
- **Estimand:** The precise population-level quantity to be estimated; see [decision definition](#step-1-state-the-decision-and-estimand).
- **Examination:** Latent customer attention, not guaranteed by rendering; see [mental model](#3-the-core-mental-model).
- **Exploitation:** Selection based on the present preferred action; see [exploration](#step-9-plan-safe-exploration).
- **Exploration:** Randomized choice intended partly to collect information; see [exploration](#step-9-plan-safe-exploration).
- **Exposure bias:** Selection of observable outcomes by the display policy; see [definitions](#4-definitions-and-vocabulary).
- **Feedback loop:** Policy-induced data shape subsequent policies; see [mental model](#3-the-core-mental-model).
- **Guardrail:** A prespecified constraint on harm or operational performance; see [online comparison](#step-10-run-the-online-policy-comparison).
- **Importance weight:** Target action probability divided by logging probability; see [IPS](#step-6-calculate-the-one-step-ips-estimate).
- **Intent-to-treat:** Comparing units by randomized assignment; see [online comparison](#step-10-run-the-online-policy-comparison).
- **Interference:** One unit's treatment changes another unit's outcome; see [assumptions](#11-assumptions-and-limits).
- **IPS:** Inverse propensity scoring, an importance-weighted policy-value estimate; see [IPS](#step-6-calculate-the-one-step-ips-estimate).
- **Logging policy:** The policy that generated the historical action; see [probabilities](#step-4-define-the-logging-and-target-probabilities).
- **Position bias:** Differential response opportunity by location; see [mental model](#3-the-core-mental-model).
- **Propensity:** Conditional probability of the selected action under the logger; see [probabilities](#step-4-define-the-logging-and-target-probabilities).
- **Reward model:** A predicted outcome for a context and action; see [estimator choices](#step-7-use-snips-clipping-and-doubly-robust-methods-carefully).
- **Slate:** The ordered set of items displayed together; see [exposure mapping](#step-2-map-the-exposure-mechanism).
- **SNIPS:** Self-normalized inverse propensity scoring; see [estimator choices](#step-7-use-snips-clipping-and-doubly-robust-methods-carefully).
- **Support:** Positive logging probability for target actions in relevant contexts; see [overlap](#step-5-check-overlap-before-estimating).
- **Target policy:** The alternative distribution over actions being evaluated; see [decision definition](#step-1-state-the-decision-and-estimand).

## 16. Sources

1. Lihong Li, Wei Chu, John Langford, and Robert E. Schapire, [“A Contextual-Bandit Approach to Personalized News Article Recommendation”](https://doi.org/10.1145/1772690.1772758), *Proceedings of the 19th International Conference on World Wide Web*, 2010. Randomized news recommendation and offline evaluation; context differs from retail.
2. Thorsten Joachims, Adith Swaminathan, and Tobias Schnabel, [“Unbiased Learning-to-Rank with Biased Feedback”](https://doi.org/10.1145/3018661.3018699), *Proceedings of WSDM*, 2017. Propensity-weighted learning from biased search clicks.
3. Allison J. B. Chaney, Brandon M. Stewart, and Barbara E. Engelhardt, [“How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility”](https://doi.org/10.1145/3240323.3240370), *Proceedings of RecSys*, 2018. Modeled feedback effects; not a universal effect estimate.
4. Adith Swaminathan and Thorsten Joachims, [“Counterfactual Risk Minimization: Learning from Logged Bandit Feedback”](https://proceedings.mlr.press/v37/swaminathan15.html), *Proceedings of ICML*, 2015. Importance weighting and variance in logged feedback.
5. Miroslav Dudík, John Langford, and Lihong Li, [“Doubly Robust Policy Evaluation and Learning”](https://arxiv.org/abs/1103.4601), *Proceedings of ICML*, 2011. Doubly robust policy evaluation assumptions.
6. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), official text, particularly Articles 5, 6, and 22. Applicability depends on facts and jurisdiction.
7. European Union, [Regulation (EU) 2022/2065, Digital Services Act](https://eur-lex.europa.eu/eli/reg/2022/2065/oj), official text, especially Articles 27 and 38. Obligations depend on service category and scope.
8. California Legislature, [California Consumer Privacy Act, Civil Code sections 1798.100 et seq.](https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?division=3.&part=4.&lawCode=CIV&title=1.81.5), official statutory text. Check current law and implementing regulations.
9. GS1, [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin), official identifier documentation. Identification is distinct from availability or propensity.
10. W3C, [Web Content Accessibility Guidelines (WCAG) 2.2](https://www.w3.org/TR/WCAG22/), W3C Recommendation. Accessibility guidance is not itself a causal measurement method.

## 17. Links

- [Module 24 index](_index.md)
- [Recommendation tasks](01_recommendation-tasks.md)
- [Popularity and content-based baselines](02_popularity-content-baselines.md)
- [Collaborative filtering and ALS](03_collaborative-filtering.md)
- [Two-tower retrieval](04_two-tower-retrieval.md)
- [Sequential and session models](05_sequential-session.md)
- [Module 24 survey and lab sections](../../../course/24_recommendation_systems_personalization.md)
- [Simulation and production labs survey](../../../course/30_simulation_optimization_production_labs.md)
