# Offline and Online Evaluation of Recommendations

> Module 24 — Recommendation Systems and Personalization  
> Topic 07 of 11  
> Estimated study time: 4 hours  
> Prerequisites: topics 01–05; familiarity with experimental design  
> Version 2026.5

## 1. Why this topic exists

A recommendation team must decide whether a new list should replace the current list.
The decision is not simply which model produces the largest score in a notebook.
It is whether the change improves an agreed outcome for eligible customers without unacceptable operational or distributional harm.

The merchandising owner may want more fulfilled contribution dollars per visitor.
The customer experience owner may care about relevant choices and low cancellation rates.
The fulfillment owner may care about stockouts, late deliveries, and substitutions.
The data scientist may measure historical relevance because it can be computed before deployment.
The experiment owner must connect these perspectives without pretending they are interchangeable.

A retrieval model can retrieve more historical purchases and still recommend items that are unavailable today.
A ranker can improve clicks by putting cheap, eye-catching products first while reducing contribution after returns.
A system can increase average revenue while making a small region's service unacceptably worse.
An apparent win can also be a logging change, a data leak, a seasonal shift, or random noise.

Offline evaluation is a controlled comparison on recorded data.
It is useful for eliminating candidates, diagnosing ranking quality, and checking constraints cheaply.
It cannot by itself establish the causal effect of showing a different list.
Recorded behavior reflects what the previous policy chose to expose.

An online randomized experiment changes the policy shown to comparable groups.
It can estimate an effect on the selected population and period when assignment, measurement, and interference assumptions hold.
It is more expensive and exposes customers to risk, so candidates need offline checks and release safeguards first.

This topic gives a single decision chain from historical ranking metrics to a randomized launch decision.
The worked case is illustrative rather than a reported company result.
Numbers are examples in US dollars and orders unless stated otherwise.

## 2. Learning outcomes

After this unit you can:

1. Define the evaluation unit, relevance, eligible catalogue, cutoff, and no-positive convention for a recommendation task.
   Exercise 1 checks this contract.
2. Calculate Recall@k, normalized discounted cumulative gain (NDCG@k), and mean average precision (MAP@k).
   Exercises 2 and 3 check calculation and interpretation.
3. Construct a leakage-resistant temporal evaluation and explain exposure, position, and selection biases.
   Exercise 4 tests the design.
4. Describe when inverse propensity weighting is identifiable and why it is not an automatic correction.
   Exercise 5 checks the assumptions.
5. Design a randomized online experiment with an estimand, unit of assignment, primary outcome, guardrails, and analysis plan.
   Exercises 6 and 7 assess this design.
6. Translate changes in customer and operational outcomes into incremental contribution, with attribution and costs stated.
   Exercise 8 tests the calculation.
7. Make a launch, iterate, or stop recommendation that distinguishes evidence, uncertainty, ownership, and unresolved risks.
   Exercise 9 assesses the decision.

Each verb represents a reproducible decision or artifact, not familiarity with a metric name.
A score without its evaluation contract does not satisfy these outcomes.

## 3. The core mental model

```text
Customer need + business objective + hard constraints
                         |
                         v
             Define decision and eligible population
                         |
                         v
    Time-correct data --> Offline candidate comparisons
                         |    relevance, ranking, coverage, cost
                         |    historical policy and labels limit inference
                         v
             Safety, feasibility, and readiness gate
                         |
                         v
       Randomized assignment --> Policy A or Policy B
                         |       same eligibility and measurement
                         v
     Primary business outcome + guardrails + uncertainty
                         |
                         v
         Owner decision: launch / iterate / stop / learn
                         |
                         v
     Monitor delayed outcomes, feedback, and operational drift
```

The left-to-right story is a sequence of increasingly consequential decisions.
The causal story is different: assignment changes the policy, which may change exposure, behavior, orders, fulfillment, and realized contribution.
The intermediate steps are mechanisms, not proof of the final effect.

The offline gate asks whether a model can recover or rank observed future interactions in a defined replay.
It compares alternatives under a shared dataset and protocol.
Its answer is conditional on the historical population, exposure policy, catalogue, labels, and time window.

The online gate asks what would happen if eligible traffic received one policy rather than another.
Randomization makes treatment assignment independent of potential outcomes in expectation.
It does not ensure that everyone sees a recommendation, that logging works, or that effects generalize to a later period.

A funnel can be useful for diagnosis:

1. A request is eligible to receive a recommendation.
2. A policy selects and orders eligible items.
3. An impression is rendered and measured.
4. A customer may view, click, add to basket, or purchase.
5. The order may be fulfilled, cancelled, returned, or substituted.
6. Contribution is realized after product, fulfillment, and other specified costs.

The funnel also contains selection traps.
A click-through rate computed only among rendered impressions excludes requests whose rendering failed.
A margin per order can rise while the number of orders falls.
An experiment analyzed only among people who clicked compares groups selected after assignment.
Use all randomized eligible units for the primary intention-to-treat analysis.

A useful model score must fit its place in this chain.
Recall@k is primarily a candidate-coverage question when the relevant items are defined.
NDCG@k values higher placement and can support graded relevance.
MAP@k rewards placing multiple binary-relevant items early.
None of the three measures incremental revenue, customer welfare, or fulfillment directly.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Evaluation query | One recommendation decision context, such as a user-session at a request timestamp. | Treating every request as an independent person. |
| Evaluation unit | Object over which a metric is computed before aggregation. | Changing from sessions to users without noting weighting. |
| Assignment unit | Entity randomized to an online policy, for example customer account. | Treating repeated requests as independent assignments. |
| Target population | Eligible entities to which the decision is intended to apply. | Claiming results for every customer from a narrow eligible segment. |
| Candidate set | Items the system may consider for a particular query. | Including future or unavailable items in historical comparisons. |
| Eligible item | Item satisfying the decision-time availability and policy rules. | Assuming a product present in the catalogue was orderable in every region. |
| Impression | A recommendation actually rendered according to the instrumentation definition. | Counting a server response as a viewed item. |
| Exposure | Opportunity for a customer to encounter an item, including position and context. | Treating all unclicked catalogue items as equally observed negatives. |
| Label | Recorded or derived relevance value in a defined outcome window. | Calling a click a purchase or treating a missing event as disinterest. |
| Positive | An item meeting the prespecified binary relevance rule. | Changing the positive definition between models. |
| Graded relevance | Relevance with ordered levels, such as click, basket, and fulfilled order. | Assuming numerical grades reflect true utility without validation. |
| Holdout | Data withheld from fitting or model selection for evaluation. | Repeatedly tuning on the final test set. |
| Temporal split | Training and evaluation divided by event time. | Using a user's future history to generate past features. |
| Leakage | Information unavailable at decision time entering features, labels, or candidate generation. | Restricting only the model fit timestamp while ignoring feature snapshots. |
| Cutoff k | Maximum rank positions included in a metric, measured in items. | Comparing scores at different cutoffs as if directly comparable. |
| Recall@k | Relevant items in the first k positions divided by all relevant items in the specified evaluation universe. | Calling it the probability that a customer sees at least one relevant item. |
| Hit rate@k | Fraction of queries with at least one relevant item in the top k. | Calling it recall when queries have multiple relevant items. |
| Precision@k | Relevant items in the first k positions divided by k, subject to the declared short-list convention. | Using it without stating whether fewer than k results are possible. |
| Rank discount | Weight that declines as an item's position increases. | Assuming a mathematical discount is a measured attention curve. |
| DCG@k | Discounted cumulative gain of the top k ranked relevance grades. | Comparing unnormalized DCG across queries with different relevance potential. |
| IDCG@k | DCG of the ideal ordering of the query's eligible labeled items at k. | Building the ideal list from only a model's retrieved candidates. |
| NDCG@k | DCG divided by IDCG when IDCG is positive. | Interpreting a 0.02 absolute increase as a 2% revenue lift. |
| AP@k | Average precision for a query with an explicitly declared truncated denominator. | Assuming every library uses the same AP@k denominator. |
| MAP@k | Mean AP@k across evaluation queries using a specified aggregation. | Averaging only successful queries after looking at the outcome. |
| Macro average | Equal weight per evaluation unit. | Assuming it weights each customer equally when the unit is a request. |
| Micro average | Aggregate numerator divided by aggregate denominator under a stated definition. | Treating it as interchangeable with the macro average. |
| Propensity | Probability that the logging policy takes an observed action in context. | Substituting a fitted score for a verified probability of the full action. |
| Positivity | Every action required by the target policy has positive logging probability in the relevant contexts. | Assuming more logged data repairs zero exposure probability. |
| Counterfactual evaluation | Estimation of outcomes under a policy other than the logging policy. | Assuming inverse weights remove unmeasured confounding. |
| Randomization | Allocation by a known random mechanism independent of potential outcomes at assignment. | Alternating policies by time and calling it random assignment. |
| Estimand | Precisely defined population, treatment contrast, outcome, and time horizon. | Naming a metric without specifying who and when. |
| Intention to treat | Analysis by assigned group, including units not successfully exposed. | Filtering to clicks or actual exposure after assignment. |
| Guardrail | Predeclared metric and decision rule limiting acceptable harm. | Declaring every unowned dashboard metric a binding guardrail after seeing results. |
| Sample-ratio mismatch | Material disagreement between observed and expected allocation counts. | Treating it as proof of a treatment effect rather than an instrumentation warning. |
| Interference | One unit's treatment affects another unit's outcome. | Assuming scarce inventory makes individual outcomes independent. |
| Contribution | Revenue minus specified variable costs and adjustments, in currency per stated unit. | Calling gross merchandise value or revenue profit. |
| Maturity window | Time allowed for delayed outcomes such as returns to be observed. | Comparing mature control orders with immature treatment orders. |
| Novelty | A measure of how unfamiliar or unpopular recommended items are under a declared definition. | Assuming novelty is always desirable for every task. |
| Coverage | Breadth of users, requests, or catalogue items reached, depending on the definition. | Reporting a percentage without saying what is in the denominator. |

## 5. The mechanics, step by step

### Step 1: State the decision and owner

**Input:** a proposed recommendation change and the current policy.
**Output:** a written decision card signed by a business owner and an experiment owner.

Specify the surface: for example, a product-detail-page complement carousel in one country.
State the customer action the surface is meant to assist.
A substitute for an unavailable item has a different success condition from a complementary purchase.

Name the current policy, the proposed policy, and what stays fixed.
A comparison of two rankers is not isolated if their retrieval systems also differ.
A comparison of complete policies can change retrieval and ranking together, but then the estimate is for the bundle.

Name the decision date and who can stop exposure.
Record who owns merchandising, inventory feasibility, experiment operations, and financial definition.
A model developer should not be the sole owner of both the metric and the go/no-go decision.

Write the business outcome as a direction and unit, not as an unsupported forecast.
For example: increase realized contribution in US dollars per randomized eligible customer over 28 days.
A target may be set for planning; it is not a commitment and is not an estimate of the actual effect.

### Step 2: Define the offline query and label

**Input:** historical decision logs, catalogue snapshots, and outcomes.
**Output:** one row per query and the corresponding relevant-item set.

Choose a query identifier that matches the use case.
For a carousel, a request with timestamp and context may be appropriate.
For a weekly personalized email, the user-week may be more meaningful.
Multiple queries from one customer must remain grouped when estimating uncertainty.

Define a forward label window before looking at model comparisons.
For a purchase label, specify order placement versus fulfillment, cancellation handling, and item-level attribution.
A seven-day purchase window may capture deliberation but overlap other recommendation opportunities.

Decide whether repeated events for an item count once per query.
Set relevance from data available only after the query while ensuring features use only data available before it.
For graded relevance, specify each grade and tie handling.
Graded relevance levels represent a design choice, not a cardinal measurement of customer value.

Define whether purchases unrelated to a surface count as relevant.
A broader label may reflect customer intent but also other marketing or navigation channels.
A surface-attributed label is narrower but is itself affected by the old exposure policy.
Report the choice and use sensitivity analyses where plausible.

### Step 3: Reconstruct the eligible universe

**Input:** dated catalogue, inventory, geography, and policy records.
**Output:** candidate universe and eligible relevant items for every query.

Evaluate only items that could be recommended under the specified decision-time rules.
A grocery item must be orderable in the relevant store and delivery window, not merely listed nationally.
A restricted item must pass age, region, and other applicable policy checks.

Preserve the state known at request time.
Using end-of-day stock for a morning request introduces future information.
If historical snapshots are missing, describe the approximation and quantify its coverage.
Do not claim perfect replay.

Decide how to treat a later purchase of an item that was ineligible at query time.
It cannot be recovered by a compliant model at that moment.
Exclude it from the eligible-relevance denominator and report the excluded share.
If the goal is to measure total unmet demand instead, retain a separate demand metric.

For ranking-only evaluation, hold the candidate set fixed across rankers.
For retrieval evaluation, compare each retriever against the same eligible relevance universe.
For end-to-end policy evaluation, measure both retrieval and final ordering.
A ranker cannot recover a relevant item that retrieval omitted.

### Step 4: Split time and isolate decisions

**Input:** dated interactions, features, model artifacts, and outcomes.
**Output:** training, validation, and untouched test periods with mature labels.

An illustrative split uses January through March for fitting, April for selection, and May for final testing.
The exact dates must fit event volume and the intended deployment season.
Leave enough time after each query for its label window to close.
A seven-day label for a May 31 query cannot be complete on June 1.

Fit popularity, normalization, embeddings, and candidate indices only from information available at the relevant cutoff.
Historical aggregates computed over the whole dataset are leakage even if the supervised model is trained only on earlier rows.
Avoid features that contain later inventory, later user behavior, or revised product information unavailable at the decision time.

A single fixed train-test split may hide seasonal sensitivity.
Use rolling temporal backtests to expose changes in events, stock, assortment, and customer mix.
Select the evaluation windows before the final comparison, and reserve a final period from tuning.

Do not randomly split individual events when future interactions can reveal earlier decisions.
A random split may be useful for a narrowly defined diagnostic, but its deployment interpretation is limited.
If a new-user scenario is important, separately test users with no eligible prior history.

### Step 5: Compute Recall@k

For a query q let Rq be its set of eligible relevant items and Lq,k be the distinct top-k recommended items.
For |Rq| > 0:

```text
Recall@k(q) = |Rq intersect Lq,k| / |Rq|
```

The denominator is the number of eligible relevant items in the evaluation window.
It is not k, the number of catalogue items, or the number retrieved by a particular model.
When the list has fewer than k eligible items, use the actual list and record the shortfall separately.
Duplicate recommendations must not count twice.

For multiple queries, macro recall is the arithmetic mean of the per-query recalls over the declared cohort.
Micro recall pools all relevant-item hits and divides by all relevant-item counts.
The two differ when queries contain different numbers of relevant items.
A query with 20 relevant items has 20 times the denominator weight of one with one relevant item in the micro measure.

Queries with zero observed positives have undefined recall.
One defensible convention is to exclude them from the conditional recall average while reporting their fraction and separate request coverage.
Another is to assign zero for a particular product objective, but this changes the estimand.
Prespecify the rule and do not compare metrics computed with different conventions.

Recall ignores order within the top k.
A relevant item at rank 1 and the same item at rank k receive the same credit.
For retrieval this can be appropriate because the ranker will reorder the candidates.
For a visible carousel it can obscure a large user-experience difference.

### Step 6: Compute NDCG@k

Let rel_i be the relevance grade of the item placed at one-indexed rank i.
Choose a gain function before evaluation.
This unit uses linear gain, gain(rel) = rel, with log base 2 rank discount:

```text
DCG@k  = sum from i=1 to k of gain(rel_i) / log2(i + 1)
IDCG@k = DCG@k of the ideal ordering of all eligible labeled items
NDCG@k = DCG@k / IDCG@k, when IDCG@k > 0
```

Missing ranks contribute zero.
For binary relevance, a hit at rank 1 contributes 1 and a hit at rank 3 contributes 0.5.
The ideal ordering is built from the full eligible label set, not just what the model happened to retrieve.
Under nonnegative grades and a common candidate universe, NDCG lies between 0 and 1.

Some implementations use gain(rel) = 2^rel - 1.
That function places much larger relative value on high grades.
Others define logarithmic discounts or tie handling differently.
State the exact formula, grades, cutoff, candidate universe, and aggregation whenever a score is reported.
Do not compare values from different implementations without reconciling those choices.

Queries with IDCG equal to zero have no defined normalized score.
Handle and report them consistently with the no-positive convention.
A log discount is a modeling convention; it is not proof that a customer sees every rank with that probability.
Measure actual viewability separately when possible.

### Step 7: Compute AP@k and MAP@k

For binary relevance define P@i as the number of relevant items in the first i positions divided by i.
For a query with m = |Rq| > 0, this unit defines:

```text
AP@k(q) = sum over i=1..k of [P@i * indicator(item at i is relevant)] / m
MAP@k   = mean of AP@k(q) across the declared query cohort
```

This denominator is the full number of eligible relevant items, including those beyond k or not retrieved.
It therefore penalizes a model that misses positives even when k is smaller than m.
A perfect top-k ordering has AP@k = min(k, m) / m under this convention.

Another common convention divides by min(k, m), making the maximum 1 even when more than k items are relevant.
Both can be legitimate when explicitly defined; they answer different questions.
The worked example and code consistently use the full-m denominator.
A library called `MAP@k` is not a sufficient specification of the formula.

AP rewards early relevant placements and recognizes multiple relevant items.
It does not use graded relevance in this binary formulation.
As with recall, a positive-free query has an undefined per-query value unless another convention is declared.

### Step 8: Compare models fairly

**Input:** identical evaluation queries, labels, candidate rules, and cutoff.
**Output:** a paired comparison with uncertainty and diagnostic slices.

Run the production baseline and a simple popularity or content baseline on the same requests.
Compare the candidate against the existing policy, not just against a deliberately weak reference.
Freeze model selection before the final holdout evaluation.

Calculate each query's difference between candidate and control metrics.
The paired design recognizes that some queries are intrinsically harder than others.
If multiple requests belong to a user, resample users, not requests, in a cluster bootstrap.
For geographic or temporal dependence, consider the appropriate higher-level unit and the limits of the available clusters.
A bootstrap interval measures sampling variability under its resampling assumptions; it does not fix biased labels.

Report the point estimate, interval, number of queries, number of independent clusters, and no-positive fraction.
Also report the test period, model versions, relevant-item definition, k, and availability rule.
Specify whether an interval is for an absolute difference or a relative change.

Inspect predefined segments such as new users, low-stock markets, and long-tail items.
Include sample sizes and uncertainty because small slices produce unstable estimates.
Do not declare a subgroup success merely because its point estimate is positive.
If many slices are inspected, distinguish exploratory findings from confirmatory hypotheses.

### Step 9: Diagnose offline biases

**Exposure bias:** the old policy determines which items are seen and can receive clicks.
Unexposed items can be relevant even though they have no observed click.
A model imitating the old list can look strong because the test labels favor old exposure.

**Position bias:** items near the top have more opportunity to be examined.
A high click count at rank 1 combines item appeal with placement.
A click model or randomized position experiment may help distinguish these mechanisms under its assumptions.

**Selection bias:** the logged population may contain only users who visited, consented, or encountered a surface.
The offline cohort may differ from all eligible production users.
Training on buyers alone excludes nonbuyers and can misstate value for a broad recommendation task.

**Popularity and feedback bias:** frequently exposed items accumulate more interactions and become still more likely to be recommended.
A high offline score can reinforce past concentration.
Track item and supplier exposure distributions along with relevance.

**Survivorship bias:** deleted products, unavailable inventory, returned orders, and cancelled transactions may disappear from a later extract.
A point-in-time dataset must preserve what existed and what was known then.
A metric computed only on ultimately fulfilled orders answers a selected-outcome question.

**Temporal bias:** an item or behavior that was predictive last season may not remain so.
Rolling periods reveal variation but do not guarantee the next period will match any backtest.
For promotion-heavy retail, state which promotions occurred in each window.

**Sampling bias:** reducing millions of negatives to a small sampled set makes a ranking problem easier.
A score from 100 sampled negatives is not automatically comparable with one from the full eligible catalogue.
State the sampling distribution, seed, and whether negatives are shared across models.

**Measurement bias:** events may be blocked by device, consent state, or client failure.
Evaluate event completeness by platform and group.
An apparent improvement can arise from logging more impressions or clicks rather than changing behavior.

### Step 10: Consider counterfactual offline evaluation

A randomized logging policy can provide known action probabilities for eligible contexts.
For contextual bandits with one selected action a and observed reward r, the basic inverse propensity score (IPS) estimator for target policy pi is:

```text
IPS value = mean over logged rows of [pi(a | x) / p_log(a | x) * r]
```

Here x is context, p_log is the logging probability, and pi is the target-policy probability.
For a deterministic target policy, pi is 1 for the selected action and 0 otherwise.
The estimator is for the stated reward and action definition, not automatically for long-term business value.

Identification needs consistency of action and reward definitions, correct logged propensities, appropriate randomization or exchangeability, and positivity.
Every action that the target policy may select in the relevant context must have positive logging probability.
A deterministic historical policy with zero probability for alternatives cannot identify their outcomes from those logs alone.

Large weights from very small logging probabilities create high variance.
Clipping weights or self-normalizing can improve stability but changes finite-sample behavior and can introduce bias.
Report effective sample size, propensity distribution, clipping rule, and sensitivity.
Compare with known randomized results when possible.

A recommendation slate is more complicated than a single action.
The action may be the entire ordered slate, with a joint propensity too small to estimate reliably.
Item-level propensities need additional assumptions about position, interactions, and reward attribution.
Do not multiply or substitute marginal probabilities without a valid policy and reward model.
A small randomized exploration policy designed with product and safety owners is often more informative than retrofitting propensities to deterministic logs.

### Step 11: Choose an online estimand

**Input:** offline evidence, operational review, and business decision.
**Output:** an experiment protocol signed before assignment begins.

Write the causal contrast in plain language.
Example: among all eligible signed-in customers randomized during the test, compare 28-day realized contribution per assigned customer under policy B versus policy A.
The treatment is the complete serving policy, including fallbacks and operational failures.

Choose an assignment unit aligned with the experience and outcome.
A customer-level assignment is appropriate when a customer may make multiple requests and should see a stable policy.
A request-level assignment can answer a request-level question but may contaminate customers across variants.
A household, store, or geographic unit may be necessary where customers share accounts, inventory, or local operations.

Use persistent assignment, a documented hash or randomization service, and a versioned eligibility rule.
Define behavior for logged-out users and cross-device identity.
Assignment before exposure supports intention-to-treat analysis, including failures to render.

Specify whether the estimate covers all assigned users, only eligible requests, or another population.
Avoid filtering on post-assignment actions such as viewing the carousel, clicking, or purchasing.
Those actions can themselves be affected by treatment.
A per-protocol analysis may be informative descriptively but requires additional assumptions for causal interpretation.

### Step 12: Select outcomes and guardrails

Choose one primary decision metric when practical.
Its numerator, denominator, attribution horizon, source system, and maturity rule must be written down.
For contribution per assigned customer, all assigned customers remain in the denominator, including those with no orders.

Document revenue and every included variable cost.
Examples include product cost, fulfillment, shipping subsidy, payment fees, returns, and incremental serving cost.
Avoid subtracting costs already embedded in the contribution numerator.
Customer lifetime value may be useful but requires assumptions and enough follow-up; it is not observed immediately.

Use secondary metrics to explain the path: impression rate, clicks per eligible customer, add-to-basket rate, fulfilled orders per customer, and average order value.
A secondary mechanism metric is not a replacement for the primary outcome.
Measure the whole customer's eligible activity if the policy can move purchases between surfaces.
Otherwise surface-attributed revenue can rise merely by taking credit from another channel.

Choose a small number of operational and customer guardrails with named owners.
Examples include cancellation percentage, delivery lateness, customer complaints, availability violations, latency, and error rate.
For each, specify the direction of harm, unit, threshold, minimum detectable harm or precision requirement, and action.
Some safety violations may trigger immediate suspension regardless of statistical significance.

Define the denominator of each guardrail carefully.
Cancellation per placed order and cancellations per assigned customer answer different questions.
Treatment can change the number and composition of orders, so interpret per-order ratios with this selection in mind.
Report both ratios and assigned-customer rates where useful.

A guardrail test that fails to reject zero harm does not establish that harm is acceptably small.
A noninferiority claim needs a prespecified acceptable harm margin and an appropriately bounded confidence interval.
If the interval includes material harm, the result is inconclusive for that guardrail.

### Step 13: Plan sample size and duration

Record the baseline outcome distribution, minimum business-relevant effect, allocation ratio, significance level, and desired power.
Use a variance estimate compatible with the randomization unit and the chosen metric.
Revenue and contribution are often skewed, with many zero values and a few large orders.
A binary conversion formula is not automatically valid for contribution per user.

For a simple two-sided comparison of independent proportions with equal group sizes, an approximate per-group planning calculation is:

```text
n ≈ 2 * (z_(1-alpha/2) + z_(1-beta))^2 * p_bar * (1-p_bar) / delta^2
```

Here p_bar is a planning baseline proportion, delta is the absolute proportion-point effect expressed as a fraction, alpha is the false-positive rate, and 1-beta is power.
The formula is an approximation and assumes independent observations and a fixed-horizon test.
Use a suitable design or simulation for clusters, rare events, revenue, sequential monitoring, and multiple metrics.

Define the minimum duration needed to cover weekly patterns and delayed outcomes.
Avoid starting or stopping based only on whether the current p-value is favorable.
If interim efficacy looks are required, use a prespecified sequential design or alpha-spending approach.
Operational safety monitoring and efficacy testing have different purposes and decision rules.

Record planned traffic, allocation, ramp stages, exposure cap, and stop authority.
A low-powered test with a wide interval can be useful for safety learning but cannot prove absence of a small business effect.
If the needed sample is infeasible, revise the decision or accept an explicitly uncertain answer rather than relabeling the test conclusive.

### Step 14: Run and monitor the experiment

Before launch, validate assignment, policy versions, event schemas, timestamps, treatment exposure, and outcome joins using synthetic or test traffic.
Run an assignment-only or identical-policy check when practical to reveal instrumentation defects.
A clean check increases confidence but does not prove future data will be correct.

During the ramp, monitor allocation counts against the expected ratio at the assignment unit.
Investigate sample-ratio mismatch before interpreting treatment effects.
Possible causes include eligibility differences, dropped logs, inconsistent hashing, or treatment-dependent redirects.
Do not simply reweight away an unexplained mismatch.

Monitor server errors, recommendation latency, empty lists, stock eligibility, and customer safety.
Compare logging coverage by platform and arm.
If critical measurement or guardrails fail, the named owner pauses or rolls back according to the protocol.
Record the time and reason; do not silently exclude the incident period.

Monitor inventory interference.
If treatment increases demand for scarce stock, control customers may encounter less stock because of treatment customers.
The resulting contrast may underestimate, overestimate, or otherwise differ from the all-treated rollout effect.
Cluster randomization, geographic switchbacks, or explicit inventory modeling may help, but each needs its own assumptions.

### Step 15: Analyze, decide, and follow through

Freeze the assignment population and follow the prespecified outcome-maturity rule.
Join outcomes to assignment, not only to impression events.
Report missingness and unresolved data-quality issues by arm.

Compute absolute treatment minus control differences and confidence intervals at the assignment unit or an appropriate cluster level.
Report relative changes only alongside their baseline and absolute values.
For ratio metrics, use an appropriate variance estimator or cluster bootstrap; do not treat individual events as independent users.

Check the primary metric, then guardrails under the decision protocol.
Address multiple testing if several co-primary metrics or many confirmatory comparisons determine launch.
Exploratory slices can suggest hypotheses; they do not create a prespecified success after the fact.

A decision memo should include the experiment dates, population, interventions, outcome windows, sample sizes, estimates, uncertainty, and costs.
Separate observed facts, causal interpretation, assumptions, and remaining uncertainty.
A business owner decides launch, iterate, or stop; safety and operational owners retain their agreed vetoes.

After launch, monitor mature returns, cancellations, inventory, exposure concentration, and model drift.
A randomized holdout can help measure continuing incremental value if traffic and governance permit.
Changing seasonality, assortment, or competitor behavior can change the effect even when the original experiment was valid.

### Additional decision practice: make the offline contract executable

The procedure above is easier to audit when the team turns each conceptual choice into a field or assertion.
The following contract is illustrative for an online retail carousel.
It is not a universal schema.

| Contract element | Example setting | Responsible owner | Failure action |
|---|---|---|---|
| Query | One eligible carousel request | Product | Reject rows without request ID. |
| Query timestamp | Server request time in UTC | Data engineering | Quarantine malformed timestamps. |
| Customer link | Stable pseudonymous ID at assignment | Privacy and engineering | Report unlinked assignment percentage. |
| Region | Delivery region resolved at request | Fulfillment | Reject requests without a valid region. |
| Surface | Product-detail complement carousel | Product | Keep other surfaces out of this analysis. |
| Catalogue | Snapshot valid at request time | Catalogue operations | Flag missing snapshots. |
| Stock | Store and delivery-window availability | Inventory | Suppress infeasible items. |
| Relevance | Eligible item bought within seven days | Analytics | Exclude labels outside the window. |
| Outcome maturity | Seven days plus the specified ingestion lag | Data engineering | Do not score immature queries. |
| Recall cutoff | Ten retrieved items | Machine learning | Store k with every result. |
| Ranking cutoff | Six visible positions | Product and analytics | Use the same display rule across policies. |
| Positive-free queries | Exclude from conditional ranking metrics | Analytics | Report count and fraction separately. |
| Repeated purchases | One distinct item per query | Analytics | Deduplicate by query and item. |
| Duplicate list items | Invalid output | Serving engineering | Fail the data-quality check. |
| Comparison | Existing policy and candidate on common queries | Experiment owner | Reject unmatched evaluation cohorts. |
| Primary launch metric | Realized contribution per assigned customer | Finance and product | Do not substitute a click metric. |
| Guardrail | Unavailable recommendation rate | Inventory | Trigger the documented pause rule. |

Each choice can change the answer.
For instance, a relevant item purchased on day 8 is excluded by the seven-day label.
That is not evidence that the customer did not value the item.
It is a consequence of the defined observation window.

Similarly, a product newly eligible after the query may be purchased during the label window.
Its later purchase is evidence of demand but is not a recoverable hit under the original eligibility rule.
Record that distinction rather than changing the eligibility rule after observing the results.

The contract must be versioned with the experiment and data pipeline.
If the team changes from placed orders to fulfilled orders, comparisons across versions need a bridge.
One bridge is to recompute both definitions on a shared mature period.
The resulting difference is descriptive and does not identify the true relevance of every item.

### Additional decision practice: understand denominator sensitivity

Consider a query with five eligible relevant items and k = 3.
The system ranks three relevant items first.
Its Recall@3 is 3 / 5 = 0.6.
Under the full-relevant-count convention used in this lesson, AP@3 is (1 + 1 + 1) / 5 = 0.6.
Under the alternative min(k, m) denominator, AP@3 is 3 / 3 = 1.0.
Both values follow from their respective definitions.
The former rewards the three early hits while penalizing the two unrecovered positives.
The latter asks whether the available top-three positions were filled perfectly.

NDCG@3 for that same query is 1.0 under binary relevance because the displayed three positions match the best possible top-three ordering.
That score does not say all five relevant items were retrieved.
The difference is useful: recall emphasizes coverage of all observed positives, while NDCG@3 judges the quality of the achievable top-three ordering.

Now suppose two systems both retrieve two of five relevant items.
System A places them at ranks 1 and 2.
System B places them at ranks 2 and 3.
Both have Recall@3 = 2 / 5 = 0.4.
System A has DCG@3 approximately 1.6309.
System B has DCG@3 approximately 1.1309.
Because IDCG@3 is approximately 2.1309, their NDCG values are approximately 0.7654 and 0.5307 respectively.
The ranking-sensitive metric detects a difference that recall intentionally ignores.

Under the full-relevant-count convention, system A has AP@3 = (1 + 1) / 5 = 0.4.
System B has AP@3 = (1 / 2 + 2 / 3) / 5 = 0.2333.
AP rewards early hits but still reflects the three unrecovered items.
These arithmetic examples are conditional on the observed label set, not on the true set of all items the customer might like.

### Additional decision practice: quality checks before scoring

The analytics owner can use a checklist before producing model comparisons.
Each check should include a count and a denominator, not merely a pass or fail label.

1. Count all requests in the intended time window.
2. Count requests meeting the predeclared surface and country eligibility rules.
3. Count requests removed for missing or invalid timestamps.
4. Count requests missing an interpretable customer or session identifier.
5. Count requests with no valid catalogue snapshot.
6. Count requests with stale or missing inventory.
7. Count requests whose label windows have not matured.
8. Count requests with no eligible observed positives.
9. Count ranked lists with duplicate product identifiers.
10. Count ranked items absent from the relevant catalogue snapshot.
11. Count recommendations blocked by hard safety or availability policy.
12. Count lists shorter than the intended cutoff.
13. Count label events with missing item IDs or ambiguous product mappings.
14. Count duplicate outcome records before deduplication.
15. Count requests retained for the headline conditional metrics.

The counts should reconcile from the original population to the scored population.
If multiple exclusion reasons overlap, specify a hierarchy or show the intersections.
Otherwise the sum of reason counts can exceed the number of excluded requests without explanation.

Calculate these diagnostics separately for every policy when comparing logged responses.
A missingness difference can create an apparent model improvement.
For a replay, the same query population should feed both policies even if one policy returns an empty list.
An empty list is a product outcome, not a reason to drop a difficult query.

Audit a small number of records directly with the relevant operational owners.
For each, reconstruct what was known before the request and when the observed outcome occurred.
Manual inspection does not replace automated validation, but it can reveal a wrong join or misunderstood status code.
Keep sensitive identifiers out of broadly shared review material.

### Additional decision practice: build a metric hierarchy

Metrics should have distinct decision roles.
Using the same metric as an optimization target, a diagnostic, and a safety threshold without explanation creates confusion.

| Role | Example | What it answers | What it cannot establish alone |
|---|---|---|---|
| Offline screening | Recall@100 | Are observed relevant items retrieved? | Whether customers benefit from the new policy. |
| Offline ranking diagnosis | NDCG@6 | Are labeled items placed early? | Incremental sales or margin. |
| Delivery health | Empty-list rate per eligible request | Does the system return a list? | Whether the list is useful. |
| Engagement | Clicks per assigned customer | Does assigned traffic generate more clicks? | Whether clicks create durable value. |
| Commerce | Fulfilled orders per assigned customer | Does the policy affect completed transactions? | Whether orders are profitable. |
| Primary value | Contribution per assigned customer | What is the observed incremental economic result? | Every long-term or distributional effect. |
| Hard guardrail | Policy-violating impressions | Does a prohibited condition occur? | Overall customer experience. |
| Operational guardrail | Late deliveries per assigned customer | Does the policy burden fulfillment? | All aspects of customer satisfaction. |
| Distributional diagnostic | Supplier exposure by segment | How is attention allocated? | Whether any difference is legally impermissible. |
| Long-term monitoring | Repeat purchase in a mature cohort | Is there a sustained behavioral change? | A near-term causal claim without a suitable design. |

The hierarchy makes trade-offs explicit.
A policy can pass an offline screen and fail a launch guardrail.
It can improve engagement and reduce contribution.
It can increase contribution while a confidence interval leaves operational harm unresolved.

Each metric should have a numerical definition and a source of truth.
If finance calculates contribution monthly while analytics calculates it daily, reconcile the two before using either for a consequential decision.
The reconciliation needs an explicit treatment of cancellations, returns, taxes, discounts, and cost timing.

### Additional decision practice: read confidence intervals correctly

Suppose a fixed-horizon experiment estimates an increase of $0.05 per customer.
Assume its two-sided 95% confidence interval is from -$0.03 to +$0.13 per customer.
This interval is illustrative; it cannot be derived from the aggregate totals in the worked case.
It is compatible with a small loss and with a meaningful gain under the analysis assumptions.
It does not support a precise claim that the true gain is $0.05.

Suppose the launch owner requires an improvement of at least $0.02 per customer to justify operating cost.
Because the interval extends below that decision threshold, the observed estimate does not establish that requirement.
The owner can decide whether more evidence has value.
The result is not proof that the policy has no effect.

For a cancellation guardrail, suppose the maximum acceptable increase is 0.20 percentage points.
If the appropriate one-sided upper confidence bound on treatment minus control is 0.12 percentage points, it is below the stated margin.
Under the prespecified design and assumptions, that is consistent with passing the noninferiority rule.
If the upper bound is 0.34 percentage points, material harm remains plausible.
The guardrail is inconclusive even if the point estimate is near zero.

Confidence levels and margins must match the prespecified test.
If the team simultaneously requires several guardrails to pass, plan how each test contributes to the overall decision.
For rare but severe failures, a statistical bound may be less useful than an operational incident trigger.
Use the documented safety process instead of waiting for a conventional significance threshold.

### Additional decision practice: choose a randomization design for stock constraints

An individual-customer experiment is straightforward when inventory is abundant relative to incremental demand.
Customers can be persistently assigned, and per-customer outcomes can be compared.
The key limitation is that one customer's treatment may change the stock available to another.

A store-level cluster design assigns a store or fulfillment region to a policy.
It can reduce within-store contamination if inventory is genuinely separate.
It also reduces the effective number of independent units and may require more stores or a longer test.
Store differences in demand and assortment make blocking or stratification useful when planned in advance.

A time-based switchback assigns policies to time blocks within a store or region.
It can be useful when stock is shared and the same location can receive both policies at different times.
Time effects, carryover, replenishment cycles, and products sold in an earlier block can contaminate later blocks.
Randomize the block sequence, define a washout or carryover analysis where appropriate, and use the block as the relevant assignment unit.

Neither cluster nor switchback design is automatically superior.
The design must fit the stock network and the effect the business wants to estimate.
An item shared across multiple stores may still generate interference in a store-level test.
A replenishment process spanning several blocks may violate a simple switchback assumption.

Before selecting a design, draw the stock-flow network.
Mark which customers share fulfillment nodes, which nodes share central inventory, and how replenishment is scheduled.
Then identify where a treated customer's demand can change another customer's available options.
Document remaining spillovers in the estimand and launch memo.

### Additional decision practice: build an economic bridge

The business metric should not be assembled by multiplying unrelated percentage changes.
For example, a 3% relative click increase does not imply a 3% relative order increase.
Each funnel transition can change under treatment.
Increased clicks can be low intent, while fewer clicks can be more decisive.

Start with assigned customers and measure outcomes at the appropriate customer horizon.
Sum eligible realized order revenue under a consistent accounting definition.
Subtract product cost, fulfillment costs, shipping subsidies, payment fees, returns, and other agreed variable costs.
Add treatment-specific infrastructure or licensing costs only if they are not already included.
Divide each arm's total by its number of assigned customers.
The difference between arm means is the observed incremental contribution per assigned customer.

If the assignment arms differ in size, do not subtract raw totals.
Compare per-assigned-unit means or use a prespecified analysis that accounts for the allocation ratio.
If the outcome is delayed, compare equally mature cohorts.
If a return arrives after the maturity cutoff, report the residual exposure to late adjustments.

An annualized projection is a scenario, not a measured experiment result.
For instance, multiplying $0.05 per customer by 1 million future customers yields a $50,000 scenario only if the effect, traffic, costs, and customer mix remain comparable.
It ignores uncertainty, seasonality, saturation, and operational changes unless those are explicitly modeled.
Label it as a projection and present a sensitivity range based on defensible inputs.

The product owner may choose not to launch even with a positive estimated effect.
Reasons include an unresolved safety issue, a wide interval, implementation complexity, or a more valuable competing project.
The purpose of evaluation is to make that decision auditable, not to force one outcome from a single score.

### Additional decision practice: a launch-review worksheet

Use the worksheet below to move from analysis to a defensible decision.
An unanswered item is a reason to investigate, not an invitation to invent a reassuring answer.

**Population and intervention**

1. Which customer and request eligibility rules were fixed before randomization?
2. How many entities were assigned, by arm, and over what dates?
3. Was assignment persistent across sessions and devices?
4. Did either arm change eligibility or attribution after assignment?
5. What policy versions, filters, and fallbacks actually ran?
6. Were there outages or partial rollbacks during the period?
7. What proportion of assigned customers received a rendered recommendation?
8. Is that delivery rate descriptive rather than a condition for primary analysis?
9. Did the customer population differ materially from the intended rollout population?
10. Were any assignments excluded, and were exclusions specified before outcomes were known?

The experiment owner signs off on assignment integrity.
The product owner confirms the intended experience and eligibility.
Engineering documents any serving deviations and the precise periods affected.

**Outcome and measurement**

1. What is the primary numerator in currency or other units?
2. What is its denominator, and does it include customers with zero activity?
3. What is the observation window following assignment?
4. Are outcomes mature under the same rule in both arms?
5. Which order, return, and cost systems supply the measure?
6. How are currency conversion, refunds, and late adjustments handled?
7. What percentage of assigned entities could not be linked to outcomes?
8. Is missingness different by arm, device, or consent state?
9. Is a surface-specific outcome being mistaken for whole-customer incrementality?
10. Can finance reproduce the reported totals from the documented ledger?

The analytics owner supplies the metric specification.
Finance confirms the cost and reconciliation rules.
The privacy owner confirms that linkage and retention are permitted for the stated purpose.

**Inference and decision**

1. What is the estimate in absolute units per assigned entity?
2. What is the corresponding interval and its confidence level?
3. What randomization or dependence unit was used for uncertainty?
4. Was the analysis fixed-horizon or sequential, as planned?
5. Were interim efficacy looks or deviations documented?
6. What is the smallest business-relevant effect?
7. Does the interval rule out an effect smaller than that threshold?
8. Which guardrails passed, failed, or remain inconclusive?
9. Did any safety event trigger an operational stop independently of the statistical analysis?
10. Are segment findings confirmatory or exploratory?

The business owner documents the decision.
The operational owner confirms whether constraints remain acceptable.
An unexplained sample-ratio mismatch or material logging failure prevents a trustworthy causal interpretation until investigated.

**After the decision**

1. Record the exact policy and configuration approved for rollout.
2. Set the initial rollout scope and escalation contacts.
3. Preserve the experiment protocol and analysis version.
4. Schedule a review after the return window matures.
5. Track inventory and cancellation outcomes under expanded traffic.
6. Monitor exposure concentration across products and suppliers.
7. Compare ongoing traffic mix with the tested population.
8. Document any major assortment or pricing changes.
9. Define when the team will reconsider the decision.
10. State who can reverse the rollout if a guardrail fails.

The worksheet is not a substitute for the experiment protocol.
It is a compact handoff between model builders, business decision makers, and operating owners.
Its purpose is to make the limits of evidence visible at the point of commitment.

### Additional decision practice: reconcile common stakeholder claims

| Stakeholder statement | Clarifying question | Appropriate evidence |
|---|---|---|
| “The model is 5% better.” | Is that an absolute or relative change, and in which metric? | Formula, baseline, interval, and evaluation population. |
| “Customers clicked more.” | Was the rate per impression or per assigned customer? | Both numerator and denominator, plus delivery rate. |
| “The model sells more.” | Did total customer purchases rise or only attributed carousel purchases? | Assigned-customer outcome across surfaces. |
| “We improved profit.” | Which costs and returns were included? | Finance-approved contribution ledger. |
| “The result is safe.” | What harm margin was prespecified and what does its interval exclude? | Guardrail estimate, margin, interval, and owner. |
| “There was no stock problem.” | Was the test powered to detect rare violations and spillovers? | Violation count, exposure denominator, and stock-flow review. |
| “The baseline is weak.” | Is it the real production alternative? | Current-policy specification and matched evaluation. |
| “The test is valid because it was randomized.” | Did assignment, eligibility, and logging work as designed? | Allocation, integrity, and missingness diagnostics. |
| “The offline result proves the launch.” | Were labels biased by prior exposure and are business outcomes measured? | Replay contract and randomized outcome evidence. |
| “This will earn the same next year.” | Are traffic, costs, stock, and seasonality stable? | Scenario assumptions and ongoing holdout or monitoring. |

The purpose of these questions is not to make every decision slower.
It is to prevent one well-defined number from being used to answer a different question.
When stakeholders agree on the metric contract in advance, many of the questions become routine.

## 6. Worked example

### Business and offline contract

A retailer evaluates a complement carousel on product-detail pages.
The existing policy ranks eligible items by recent category popularity.
The candidate retrieves personalized products and ranks them with a relevance model.
Both policies use the same decision-time regional availability and safety filters.

The business owner wants realized contribution per assigned customer to improve without materially increasing cancellations or violating availability policy.
The offline label is a distinct eligible item purchased within seven days of a query.
Only items eligible at the original query time enter the relevance denominator.
This example has three queries with at least one positive; real datasets also need to report the zero-positive fraction.

The list cutoff is k = 3 items.
Grades are binary; all purchases have relevance 1.
NDCG uses linear gains and log2(i + 1) discounts.
AP@3 uses the full number of relevant items as its denominator.

| Query | Top-three list | Eligible relevant set | Hit positions | Relevant count |
|---|---|---|---|---:|
| Q1 | A, B, C | B, C, D | 2, 3 | 3 |
| Q2 | D, E, F | D, F | 1, 3 | 2 |
| Q3 | A, D, G | G, H | 3 | 2 |

Each row is one evaluation query, not necessarily one independent customer.
The letters are product identifiers.
An item in a relevant set may be absent from the displayed list.

### Recall arithmetic

For Q1, two of three relevant items appear: Recall@3 = 2 / 3 = 0.6667.
For Q2, both relevant items appear: Recall@3 = 2 / 2 = 1.0000.
For Q3, one of two relevant items appears: Recall@3 = 1 / 2 = 0.5000.

Macro Recall@3 = (0.6667 + 1.0000 + 0.5000) / 3 = 0.7222, allowing for rounding.
Micro Recall@3 = (2 + 2 + 1) / (3 + 2 + 2) = 5 / 7 = 0.7143.
The difference is due to the different query weights.
Neither value reveals whether hits are early or late within the list.

### NDCG arithmetic

The binary rank discounts for positions 1, 2, and 3 are approximately 1.0000, 0.6309, and 0.5000.

For Q1, DCG@3 = 0.6309 + 0.5000 = 1.1309.
Q1 has three relevant items, so IDCG@3 = 1.0000 + 0.6309 + 0.5000 = 2.1309.
NDCG@3 for Q1 is approximately 1.1309 / 2.1309 = 0.5307.

For Q2, DCG@3 = 1.0000 + 0.5000 = 1.5000.
Its ideal two-positive score is 1.0000 + 0.6309 = 1.6309.
NDCG@3 for Q2 is approximately 1.5000 / 1.6309 = 0.9197.

For Q3, DCG@3 = 0.5000 and IDCG@3 = 1.6309.
NDCG@3 for Q3 is approximately 0.3066.
Macro NDCG@3 is approximately (0.5307 + 0.9197 + 0.3066) / 3 = 0.5857.
Small differences in the last decimal place can arise from intermediate rounding.

### AP and MAP arithmetic

For Q1, the hit at rank 2 has precision 1 / 2 and the hit at rank 3 has precision 2 / 3.
AP@3 = (1 / 2 + 2 / 3) / 3 = 0.3889.

For Q2, hit precisions are 1 / 1 and 2 / 3.
AP@3 = (1 + 2 / 3) / 2 = 0.8333.

For Q3, the only hit is at rank 3 with precision 1 / 3.
AP@3 = (1 / 3) / 2 = 0.1667.

MAP@3 = (0.3889 + 0.8333 + 0.1667) / 3 = 0.4630 approximately.
This MAP is not directly comparable to a library using min(k, relevant count) as the AP denominator.
In this particular three-query example, all relevant counts are at most k, so the two denominators happen to coincide.

### From offline score to a proposed test

Suppose a larger, temporally held-out evaluation shows a candidate NDCG@3 increase of 0.025 absolute versus the existing policy.
That value is a hypothetical offline finding, not a conversion or contribution estimate.
It justifies further investigation only after leakage, inventory feasibility, and slice diagnostics pass.

The team randomizes 10,000 eligible customers to each policy.
Assignment is persistent for the test and analysis follows all assigned customers.
The outcome window is 28 days after assignment, with an additional fixed maturity period for recorded returns.
For simplicity, assume all outcome records are complete and no customer changes assignment.

| Observed item over the stated windows | Policy A | Policy B |
|---|---:|---:|
| Assigned eligible customers | 10,000 | 10,000 |
| Customers placing at least one order | 800 | 840 |
| Conversion | 8.0% | 8.4% |
| Total realized contribution before incremental serving cost | $16,000 | $16,800 |
| Incremental serving cost attributable to policy B | $0 | $300 |
| Contribution after stated serving cost | $16,000 | $16,500 |

Assume contribution before serving cost already includes the observed product, fulfillment, shipping, and return adjustments under the same accounting rule in both arms.
No additional return deduction should be made from those totals.
Actual contribution is generally not a fixed $20 per order; the totals here are illustrative.

The absolute conversion difference is 8.4% minus 8.0% = 0.4 percentage points.
The relative conversion change is 0.4 / 8.0 = 5.0% relative to control.
The observed net contribution difference is $16,500 minus $16,000 = $500 across 10,000 assigned customers per arm.
The difference in mean contribution is therefore $0.05 per assigned customer over the stated window.

This calculation is not automatically a statistically conclusive win.
A confidence interval for contribution requires the distribution of customer-level outcomes, not just the two totals.
The conversion counts alone also do not establish that the contribution guardrails passed.
The owner must examine uncertainty, missing data, and the predeclared guardrail rules before deciding.

If the policy diverts orders from another surface, a carousel-only revenue calculation could exaggerate incremental business value.
The customer-level outcome captures eligible cross-surface orders within the stated window, subject to identity and attribution limitations.
If treatment consumes inventory and changes control outcomes, even that experiment may not represent a full rollout.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It evaluates the exact three-query example above and prints per-query and aggregate scores.
It deliberately rejects duplicate ranked items because a production list should not receive duplicate credit.
It excludes no-positive queries from these conditional metrics and reports their count.
The code assumes every relevant item is eligible at query time and every list has already passed policy filtering.

```python
from math import log2


def score_query(ranked, relevant, k):
    """Binary metrics; AP denominator is the full relevant-set size."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranked list contains duplicate item IDs")
    relevant = set(relevant)
    if not relevant:
        return None

    top = ranked[:k]
    hits = 0
    dcg = 0.0
    precision_sum = 0.0
    for rank, item in enumerate(top, start=1):
        if item in relevant:
            hits += 1
            dcg += 1.0 / log2(rank + 1)
            precision_sum += hits / rank

    ideal_count = min(k, len(relevant))
    idcg = sum(1.0 / log2(rank + 1)
               for rank in range(1, ideal_count + 1))
    return {
        "hits": hits,
        "relevant": len(relevant),
        "recall": hits / len(relevant),
        "ndcg": dcg / idcg,
        "ap": precision_sum / len(relevant),
    }


def evaluate(rows, k):
    scored = []
    no_positive = 0
    for query_id, ranked, relevant in rows:
        result = score_query(ranked, relevant, k)
        if result is None:
            no_positive += 1
            continue
        scored.append((query_id, result))
    if not scored:
        raise ValueError("no queries have eligible relevant items")

    n = len(scored)
    for query_id, result in scored:
        print(
            f"{query_id}: recall={result['recall']:.4f}, "
            f"ndcg={result['ndcg']:.4f}, ap={result['ap']:.4f}"
        )
    for name in ("recall", "ndcg", "ap"):
        average = sum(result[name] for _, result in scored) / n
        print(f"macro_{name}={average:.4f}")
    hits = sum(result["hits"] for _, result in scored)
    positives = sum(result["relevant"] for _, result in scored)
    print(f"micro_recall={hits / positives:.4f}")
    print(f"scored_queries={n}, no_positive_queries={no_positive}")


rows = [
    ("Q1", ["A", "B", "C"], {"B", "C", "D"}),
    ("Q2", ["D", "E", "F"], {"D", "F"}),
    ("Q3", ["A", "D", "G"], {"G", "H"}),
]
evaluate(rows, k=3)
```

Expected output, rounded to four decimal places:

```text
Q1: recall=0.6667, ndcg=0.5307, ap=0.3889
Q2: recall=1.0000, ndcg=0.9197, ap=0.8333
Q3: recall=0.5000, ndcg=0.3066, ap=0.1667
macro_recall=0.7222
macro_ndcg=0.5857
macro_ap=0.4630
micro_recall=0.7143
scored_queries=3, no_positive_queries=0
```

To use this in a real pipeline, construct rows from a point-in-time query table and a mature outcome table.
Join by stable query and item identifiers, and preserve the query timestamp and candidate eligibility snapshot.
Do not interpret this small script as an online experiment analyzer or as a bias correction.
Use paired customer-cluster resampling or another suitable method to quantify offline uncertainty.

A production metric implementation should also define invalid IDs, duplicate labels, empty lists, late-arriving events, and tied scores.
Record the implementation version and compare its denominator convention against any external library before replacing it.
If a query has zero eligible positives, the printed count makes the conditional cohort explicit.
A separate eligible-request and coverage report is needed to understand those excluded queries.

## 8. Data, standards, and regulatory context

### Data contract

The minimum decision log includes assignment ID where applicable, query ID, pseudonymous subject ID, event timestamp and time zone, surface, region, and policy version.
A recommendation response should record the ordered item IDs, candidate and filter versions, and fallback reason.
A client impression should identify which positions actually rendered under a documented viewability rule.
Do not confuse a response timestamp with a view timestamp.

The outcome table needs an order or event ID, item ID, quantity, currency, event time, and status.
Financial calculations require a documented cost basis and a way to connect refunds and returns to the original order.
Use a consistent exchange-rate policy when comparing multiple currencies.
A global revenue total without currency conversion and timing rules is not interpretable.

The inventory snapshot needs item, location, time, availability state, and source-system freshness.
A catalogue entry alone does not prove that a customer could order the product.
Keep the matching rule for fulfillment location and delivery window alongside the metric version.
Late and corrected records should be handled with a fixed reconciliation and maturity policy.

Log the historical selection probabilities when using randomized exploration for counterfactual evaluation.
The probability must correspond to the actual action and eligible set at that time.
If the action is an ordered slate, a single item probability is not automatically its joint probability.
Missing or fabricated propensities cannot be repaired by labeling a deterministic score a probability.

### Identifiers and product standards

GS1 Global Trade Item Numbers (GTINs) identify trade items under GS1 rules where applicable.
GTINs are useful for linking product information across organizations but are not a universal user ID, inventory state, or relevance label.
Internal stock-keeping units can differ by retailer, packaging, or market.
Document the mapping and its effective dates; one-to-many mappings can duplicate outcomes if joined carelessly.
The [GS1 GTIN standard](https://www.gs1.org/standards/id-keys/gtin) describes the identifier and its use.

A ranking metric has no single mandatory global convention for gain, truncation, or zero-positive handling.
The formulas here are analytical choices, not a regulatory standard.
Publish the metric definition so that model versions and teams can reproduce comparisons.

### Privacy and applicable law

Recommendation evaluation commonly uses personal data or pseudonymous data that can be linked to individuals.
Pseudonymization does not necessarily make information anonymous.
Apply the organization's retention, access-control, minimization, and security policies to logs and joins.
The legal basis and permissible processing depend on jurisdiction, purpose, and facts; this teaching unit is not legal advice.

In the European Union, the General Data Protection Regulation (GDPR) includes principles of purpose limitation, data minimization, and storage limitation in Article 5.
Article 22 addresses certain solely automated decisions producing legal or similarly significant effects, with specified conditions and exceptions.
A routine product recommendation does not automatically satisfy that threshold; a qualified assessment is needed for the actual use case.
See the [official GDPR text](https://eur-lex.europa.eu/eli/reg/2016/679/oj).

The EU Digital Services Act includes transparency obligations for recommender systems in Article 27 for covered online platforms, and additional obligations in Article 38 for very large online platforms and very large online search engines as applicable.
Applicability depends on the service and designation; these articles do not prescribe Recall@k or an experiment's statistical test.
See the [official regulation](https://eur-lex.europa.eu/eli/reg/2022/2065/oj).

In California, the California Consumer Privacy Act as amended provides rights and obligations for covered businesses and qualifying data, subject to statutory and regulatory details.
See the [California Department of Justice overview](https://oag.ca.gov/privacy/ccpa).
Do not assume all jurisdictions require the same consent mechanism or permit the same retention period.
Consult privacy and legal owners before collecting exploration data or connecting recommendation logs to sensitive outcomes.

### Governance and ownership

The product owner defines the customer task and launch decision.
The data owner documents events, access, retention, and quality.
The analytics owner defines metric computation, uncertainty, and reproducible reporting.
The inventory owner certifies feasibility rules and operational thresholds.
Finance owns contribution accounting and reconciliation.
Legal and privacy owners review applicable use and disclosures.
The incident owner can suspend the experiment under the predeclared safety policy.

Keep an auditable record of the protocol, approval, model and data versions, deviations, and decision.
A documented policy is not a substitute for consent, a lawful basis, or an operational control where those are required.

## 9. Industry practice and case evidence

### Netflix and the limits of offline prediction

The Netflix Prize evaluated prediction accuracy on held-out movie ratings using root mean squared error.
The public competition is evidence of an offline predictive benchmark, not evidence that every leaderboard improvement improved subscriptions or viewing.
The [Netflix Prize rules and data description](https://www.netflixprize.com/rules.html) describe the benchmark.

Gómez-Uribe and Hunt discuss the Netflix recommender as a collection of algorithms and describe experimentation in the product context.
Their account provides a useful contrast between prediction tasks and the full member experience.
See [The Netflix Recommender System: Algorithms, Business Value, and Innovation](https://doi.org/10.1145/2843948).

**Inference:** a retail team should not equate a holdout ranking gain with incremental contribution merely because both are measured precisely.
Retail has additional availability, price, fulfillment, and returns constraints that need direct measurement.
The Netflix case does not quantify any result for the hypothetical retailer in this lesson.

### Microsoft and controlled experiments

Kohavi and coauthors describe practical challenges in large-scale online controlled experiments, including trustworthiness and operational issues.
See [Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing](https://doi.org/10.1017/9781108653985).
Their evidence supports treating experiment infrastructure and data quality as part of the causal measurement system.

A sample-ratio mismatch is a reason to investigate assignment and logging before reading effect estimates.
It does not identify the root cause by itself.
The appropriate remedy depends on whether the defect is in assignment, eligibility, instrumentation, or analysis.

**Inference:** a retailer should give an independent experiment owner authority to pause a test whose assignment or measurement is not trustworthy.
That governance recommendation is not a claim about the exact internal practices of every Microsoft product.

### Counterfactual evaluation in recommendation

Schnabel and coauthors study recommendations as treatments and discuss the bias introduced by selectively observed user feedback.
See [Recommendations as Treatments: Debiasing Learning and Evaluation](https://proceedings.mlr.press/v48/schnabel16.html).
The paper motivates propensity-based methods and the importance of the observation mechanism.

The method does not make every historical log sufficient for every target policy.
Without support for an action, a logged reward cannot reveal its outcome under that action.
With very small propensities, an estimator can have high variance even when positivity holds mathematically.

**Inference:** logging controlled exploration can create valuable evaluation evidence, but it also changes customer exposure and requires business and safety review.
The appropriate exploration rate depends on risk, data volume, and the action space.

### Offline and online disagreement

The RecSys literature has examined the relationship between offline evaluation and online performance.
For an explicit study of offline-to-online disagreement, see Rossetti, Stella, and Zanker, [Contrasting Offline and Online Results when Evaluating Recommendation Algorithms](https://doi.org/10.1145/2959100.2959176).
The existence of disagreement is a warning against treating one offline metric as a universal causal surrogate.
It does not imply that offline testing is useless or that every pair of rankings will reverse online.

**Inference:** teams should retain past paired offline and online results to calibrate which diagnostics are informative for their own surface.
That historical calibration is descriptive and can drift as the product, audience, and catalogue change.
A strong historical correlation is not a guarantee for an untested policy.

### Supply-chain interpretation

A recommendation can increase orders for a constrained item while reducing availability to other customers.
The business effect includes substitutions, fulfillment cost, customer cancellations, and possible cross-customer interference.
These mechanisms are plausible and should be tested in the actual operation; they are not claimed outcomes of the named cases above.

A high click rate can reflect an attractive price rather than a profitable transaction.
A high margin per order can reflect a change in customer mix rather than a gain per assigned customer.
A surface-specific sale can substitute for a sale that would have occurred elsewhere.
The experiment's primary outcome and denominator should match the decision the business actually needs to make.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future leakage | Implausibly large holdout gain | Features contain post-query events | Offline winner fails in production | Rebuild point-in-time joins and audit feature timestamps. |
| Wrong eligibility | Many recommended items unavailable in replay | Catalogue is not regional inventory | Inflated attainable recall | Reconstruct availability and report snapshot gaps. |
| Outcome immaturity | Recent cohorts show low purchase or return rates | Windows have not closed | Arm or period comparisons are biased | Freeze mature cohorts and reconcile late events. |
| Exposure bias | Candidate closely mirrors prior impressions | Only shown items can generate clicks | Historical policy is rewarded for its own choices | Use randomized exploration or qualify the estimate. |
| Position bias | Same item clicks change sharply by rank | Attention differs by position | Click labels confuse relevance and exposure | Log positions and conduct a suitable randomized study. |
| Negative sampling mismatch | Scores rise when sampled negatives shrink | Evaluation task becomes easier | Misleading model comparison | Share samples and disclose sampling distribution. |
| Inconsistent AP convention | Library score differs from hand calculation | Denominator differs | Incorrect model or vendor comparison | Publish formula and reconcile a small example. |
| Positive-free query exclusion | Large share of requests has no labels | Conditional metric hides the population | Overstated performance for typical traffic | Report exclusion rate and coverage. |
| Duplicate items | Repeated SKU in a list | Ranking or join defect | Inflated hits or poor experience | Deduplicate and reject invalid lists. |
| Post-treatment filtering | Analysis includes only exposed or clicked users | Treatment affects inclusion | Biased causal contrast | Analyze all randomized eligible units. |
| Sample-ratio mismatch | Assignment counts miss expected allocation | Hashing, eligibility, or logging defect | Untrustworthy experiment | Investigate and repair before interpreting outcomes. |
| Repeated unplanned peeking | Test stops at first favorable p-value | Optional stopping inflates false positives | False launch decision | Prespecify sequential monitoring or fixed horizon. |
| Underpowered guardrail | Wide interval includes serious harm | Insufficient sample or rare event | Harm cannot be ruled out | Increase precision, restrict rollout, or remain inconclusive. |
| Inventory interference | Treatment-heavy area runs out earlier | Users compete for shared stock | Individual randomization misses rollout effect | Consider cluster design and monitor stock flows. |
| Revenue substitution | Carousel sales rise but total sales do not | Attribution moves between surfaces | Claimed uplift is not incremental | Measure whole-customer outcomes. |
| Cost omission | Revenue rises while margin falls | Shipping, returns, or serving excluded | Value is overstated | Finance approves contribution ledger. |
| Missing logs by arm | Impression rate falls on one device | Client errors or consent handling differs | Apparent engagement effect is measurement | Compare telemetry completeness and use server-side assignment. |
| Multiple comparisons | One of many slices appears significant | Chance finding selected after inspection | False segment claim | Prespecify confirmatory slices and label exploration. |
| Serving fallback ignored | Many assigned users receive baseline | Treatment implementation differs from intended model | Wrong interpretation of policy impact | Analyze assignment and report actual delivery separately. |
| Feedback loop | Exposure concentrates on fewer items | Labels reflect previous policy | Long-tail relevance and supplier opportunities decline | Track distribution and plan controlled exploration. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or bound it |
|---|---|---|---|
| Query time is accurate | Prevents future features entering replay | Clock skew or delayed ingestion | Compare event and processing times; audit samples. |
| Eligible universe is reconstructible | Denominator reflects feasible items | Missing store-level stock snapshots | Measure snapshot coverage and run sensitivity cases. |
| Labels represent the intended task | Metrics need a meaningful outcome | Clickbait or unrelated purchases | Compare click, purchase, and fulfilled-order labels. |
| Labels are sufficiently mature | Windows are comparable | Returns or delayed orders arrive later | Plot arrival lag and choose a fixed maturity rule. |
| Candidate comparisons use a common universe | Differences should reflect models | One model sees more eligible items | Log and reconcile candidate rules by query. |
| Offline queries represent deployment | Model selection should transfer | New geography, promotion, or assortment | Compare population distributions and temporal slices. |
| IPS propensities are correct | Inverse weighting relies on actual action probability | Deterministic or mislogged policy | Audit policy implementation and randomization logs. |
| Positivity holds for target actions | Counterfactual outcomes must be observable | Target policy chooses never-shown items | Inspect support and restrict the estimand if needed. |
| Randomization works | Groups should be comparable at assignment | Buggy hashing or treatment-dependent eligibility | Check allocation and pre-assignment covariates. |
| Assignment is stable | Treatment contrast must be well-defined | Cross-device identity or reassignment | Audit assignment changes and identity resolution. |
| Interference is negligible or modeled | Unit-level effects should map to target decision | Shared scarce inventory and social influence | Monitor stock and consider cluster or switchback design. |
| Outcomes are measured equally | Arm differences should not be logging differences | Client-specific event loss | Compare independent sources and missingness by arm. |
| Contribution costs are complete | Business value needs a consistent ledger | Omitted returns, shipping, or serving expenses | Reconcile with finance and perform cost sensitivity. |
| Inference matches dependence | Standard errors require valid units | Repeated requests treated as independent | Cluster at randomization or higher dependence level. |
| Trial period is relevant | Result should inform launch conditions | Seasonal demand or promotional changes | Repeat or segment over predefined periods. |
| Guardrail thresholds are meaningful | A pass must correspond to acceptable harm | Threshold set after results | Approve margins and owners before allocation. |

None of these tests proves its assumption perfectly.
The decision memo should say which checks were possible, which failed, and how uncertainty affects the recommendation.

## 12. Exercises

### Exercise 1 — Evaluation contract

**Time:** 15 minutes.
A retailer recommends substitutes when a requested product is unavailable.
Define one offline query, eligible item universe, relevant label, label window, and no-positive rule.
Name the business owner and one operational owner.

**Deliverable:** a one-page metric contract.
**Acceptance criteria:** it distinguishes request-time availability from later purchase, identifies the denominator, and states how zero-positive requests are reported.

### Exercise 2 — Recall and aggregation

**Time:** 15 minutes.
At k = 3, query A has three relevant items and retrieves two; query B has one relevant item and retrieves one.
Calculate macro and micro recall.
Explain why they differ.

**Deliverable:** two values and one sentence of interpretation.
**Acceptance criteria:** macro weights queries equally; micro weights relevant items equally.

### Exercise 3 — NDCG and AP

**Time:** 20 minutes.
At k = 3, the binary relevance sequence is [0, 1, 1] and there are three eligible relevant items in total.
Use linear gain, log2 discount, and the full-relevant-count AP denominator.
Calculate NDCG@3 and AP@3.

**Deliverable:** the intermediate DCG, IDCG, hit precisions, and final values.
**Acceptance criteria:** the missing third relevant item is included in both the ideal list and the AP denominator.

### Exercise 4 — Temporal and exposure critique

**Time:** 20 minutes.
A team trains on all of last year's interactions, randomly holds out 20% of clicks, and evaluates only products shown by the old policy.
It uses today's stock table for every historical query.
Identify at least four distinct concerns and propose a repair or limitation for each.

**Deliverable:** a four-row issue-and-remedy table.
**Acceptance criteria:** includes leakage, exposure, availability, and label maturity or selection.

### Exercise 5 — Counterfactual feasibility

**Time:** 15 minutes.
A deterministic logging policy always showed item A for context X; a proposed target policy always shows item B there.
The logs contain a million observations of A and no observations of B in X.
Can ordinary IPS identify the target policy's outcome in X?
Explain what data or design could change the answer.

**Deliverable:** a yes/no answer and a short justification.
**Acceptance criteria:** identifies zero support and does not claim that a larger sample of A solves it.

### Exercise 6 — Experiment protocol

**Time:** 30 minutes.
Design a comparison of the current carousel with a new ranking policy for signed-in shoppers.
Specify the estimand, assignment unit, primary outcome, treatment definition, and outcome maturity window.
Include one risk caused by shared inventory.

**Deliverable:** a protocol card of at most one page.
**Acceptance criteria:** the denominator includes all assigned eligible units; outcome and time horizon are explicit; treatment includes fallback behavior.

### Exercise 7 — Guardrails and inference

**Time:** 25 minutes.
The treatment has a positive primary point estimate, but the cancellation guardrail confidence interval contains both zero and a prespecified unacceptable increase.
The team wants to declare the guardrail safe because the p-value for a difference exceeds 0.05.
Critique this conclusion and specify an appropriate decision.

**Deliverable:** a decision paragraph and a revised guardrail rule.
**Acceptance criteria:** distinguishes lack of significance from evidence of noninferiority and names the threshold owner.

### Exercise 8 — Business calculation

**Time:** 15 minutes.
Each arm contains 20,000 assigned customers.
Control has $48,000 of realized contribution; treatment has $50,800 before an additional $600 treatment-only serving cost.
All other listed costs and returns are already included.
Calculate the observed net difference per assigned customer.
Explain one reason it is not sufficient by itself for launch.

**Deliverable:** arithmetic in dollars per customer and a qualification.
**Acceptance criteria:** subtracts serving cost once and does not infer a confidence interval from totals alone.

### Exercise 9 — Decision memo

**Time:** 30 minutes.
A candidate has better offline NDCG, a positive online contribution estimate with a wide interval, and an unexplained sample-ratio mismatch.
The product owner requests immediate rollout.
Write the decision, next investigation, and conditions for reconsideration.

**Deliverable:** a short memo addressed to product, analytics, and operations owners.
**Acceptance criteria:** does not call the observed difference causal without resolving assignment quality; states who can pause and what evidence is missing.

## 13. Self-check answers

### Exercise 1

A suitable query is one substitute request at a store and time.
Eligible items are products orderable for the same customer, location, and delivery window at that time and allowed by substitution policy.
A relevant label might be an accepted, fulfilled substitute within a declared window.
Its denominator is the number of distinct eligible accepted substitutes per query.

Recall is undefined for a query with no observed eligible positive under this unit's convention.
Exclude such queries from conditional recall and report their count and percentage separately.
The product owner owns customer experience and the inventory or fulfillment owner owns stock feasibility.
Other defensible definitions require an explicit rationale and consistent application.

### Exercise 2

Macro recall is ((2 / 3) + 1) / 2 = 5 / 6 = 0.8333.
Micro recall is (2 + 1) / (3 + 1) = 3 / 4 = 0.7500.
The first gives both queries equal weight; the second gives the three-positive query three times the relevance-denominator weight.

### Exercise 3

DCG@3 = 1 / log2(3) + 1 / log2(4) = approximately 1.1309.
IDCG@3 = 1 + 1 / log2(3) + 1 / log2(4) = approximately 2.1309.
NDCG@3 is approximately 0.5307.
The hit precisions are 1 / 2 and 2 / 3.
AP@3 = (1 / 2 + 2 / 3) / 3 = 0.3889.

### Exercise 4

Randomly splitting events can use future behavior to predict past decisions; split by time and audit feature cutoffs.
Click labels are restricted by the previous exposure policy; collect randomized exposure or qualify the conditional estimate.
Today's inventory cannot reconstruct historical eligibility; use dated store-level snapshots and disclose missing coverage.
Click-only data omit people who did not click and may omit unexposed relevance; define the target population and report selection.
If recent labels have not had time to arrive, use mature query windows.

### Exercise 5

No: the logging probability of B in context X is zero, violating positivity for the target action.
No finite inverse weight for an observed B outcome exists because there are no such outcomes.
A safe randomized exploration policy with known nonzero B probability could collect supporting data.
Other data may help only if their selection and transport assumptions are justified.

### Exercise 6

A strong protocol defines an eligible signed-in customer population at assignment and persistently randomizes each customer to A or B.
The primary outcome could be 28-day realized contribution per assigned customer after a common return-maturity period.
It specifies the two complete policies, fallback behavior, assignment dates, currency, cost definitions, and all assigned customers in the denominator.
It also identifies shared inventory as possible interference and selects a mitigation or explicit limitation.

### Exercise 7

The result is inconclusive for the guardrail: failure to reject zero harm does not rule out an unacceptable increase.
A prespecified noninferiority margin and one-sided interval or equivalent test should show the plausible harm is below the approved threshold before claiming safety.
The operational or customer-safety owner approves the margin and the action when the interval overlaps it.
Additional data or a restricted rollout may be appropriate if authorized; neither is automatic proof of safety.

### Exercise 8

Treatment net contribution is $50,800 minus $600 = $50,200.
The observed arm difference is $50,200 minus $48,000 = $2,200.
Dividing by 20,000 assigned customers per arm gives $0.11 per assigned customer.
Customer-level variance, guardrail outcomes, and experiment validity are still required for a launch decision.

### Exercise 9 rubric

A strong memo pauses or withholds a causal launch claim until the mismatch is investigated.
It identifies the assignment and logging owners, preserves the randomized population, and checks eligibility and event loss.
It describes the wide contribution interval and guardrail status rather than treating an offline gain as a substitute for online evidence.
It names the authorized stop owner and gives a concrete condition for a trustworthy reanalysis or a new test.
A weak memo chooses the most favorable metric, excludes inconvenient traffic, or changes the success criterion after results.

## 14. Interview questions

1. **When is Recall@k more useful than NDCG@k?**
   A strong answer distinguishes retrieval coverage from rank-sensitive display quality and states the relevant-item universe.
   A weak answer says the two metrics are interchangeable because both range from zero to one.
2. **How do you define MAP@k?**
   A strong answer writes the hit-precision sum and explicitly chooses the truncated denominator and aggregation.
   A weak answer cites a library name without checking its convention.
3. **What happens when a query has no observed positive?**
   A strong answer recognizes undefined denominators and reports the chosen cohort and exclusion rate.
   A weak answer silently drops such queries and describes the score as applying to all requests.
4. **Why can a temporal split still leak?**
   A strong answer checks point-in-time features, inventory snapshots, embeddings, and label maturity.
   A weak answer assumes a timestamp filter on the model-training table is sufficient.
5. **Why are unclicked items not necessarily negative?**
   A strong answer describes exposure and position mechanisms and distinguishes missing feedback from dislike.
   A weak answer assumes every catalogue item had the same chance of being seen.
6. **When does inverse propensity scoring fail?**
   A strong answer includes positivity, accurate action probabilities, variance, and the slate-action complication.
   A weak answer says weights can infer outcomes for actions never logged.
7. **Why analyze all assigned customers?**
   A strong answer explains intention to treat and treatment-induced selection from filtering on clicks or impressions.
   A weak answer says only exposed users matter without discussing the resulting estimand.
8. **How do you choose a guardrail?**
   A strong answer names harm, denominator, owner, threshold, uncertainty, and response.
   A weak answer treats a nonsignificant test as proof of no harm.
9. **Why can an individual A/B test misstate a rollout effect?**
   A strong answer describes interference through shared inventory and possible cluster-level designs.
   A weak answer assumes randomization removes all interactions between customers.
10. **How do you connect clicks to contribution?**
    A strong answer measures customer-level incremental outcomes, mature returns, relevant costs, and uncertainty.
    A weak answer multiplies an offline NDCG gain by revenue without a causal bridge.

## 15. Key terms

- [Assignment unit](#step-11-choose-an-online-estimand): entity allocated to an online policy.
- [Availability](#step-3-reconstruct-the-eligible-universe): item feasibility at a query's time and location.
- [Candidate set](#step-3-reconstruct-the-eligible-universe): items considered for a decision.
- [Cluster bootstrap](#step-8-compare-models-fairly): resampling of independent groups rather than individual correlated requests.
- [Contribution](#step-12-select-outcomes-and-guardrails): revenue less the explicitly included costs.
- [Counterfactual](#step-10-consider-counterfactual-offline-evaluation): outcome under an alternative policy.
- [DCG](#step-6-compute-ndcgk): discounted cumulative gain of a ranked list.
- [Eligibility](#step-3-reconstruct-the-eligible-universe): rules determining which items or customers can participate.
- [Estimand](#step-11-choose-an-online-estimand): target causal contrast and population.
- [Exposure bias](#step-9-diagnose-offline-biases): labels depend on prior presentation.
- [Gain](#step-6-compute-ndcgk): numerical value assigned to a relevance grade.
- [Guardrail](#step-12-select-outcomes-and-guardrails): approved limit on an adverse outcome.
- [Hit rate](#4-definitions-and-vocabulary): fraction of queries with at least one relevant hit.
- [IDCG](#step-6-compute-ndcgk): ideal discounted cumulative gain.
- [Intention to treat](#step-11-choose-an-online-estimand): analysis according to assigned policy.
- [Interference](#step-14-run-and-monitor-the-experiment): one unit's treatment affects another's outcome.
- [IPS](#step-10-consider-counterfactual-offline-evaluation): inverse propensity score estimator.
- [Label maturity](#step-4-split-time-and-isolate-decisions): time required to observe the defined outcome.
- [Leakage](#step-4-split-time-and-isolate-decisions): use of information unavailable at decision time.
- [Macro average](#step-5-compute-recallk): equal weight for each declared evaluation unit.
- [MAP](#step-7-compute-apk-and-mapk): mean of query-level average precision.
- [Micro average](#step-5-compute-recallk): pooled numerator divided by pooled denominator.
- [NDCG](#step-6-compute-ndcgk): DCG normalized by ideal DCG.
- [Noninferiority](#step-12-select-outcomes-and-guardrails): evidence that harm is smaller than a specified margin.
- [Positivity](#step-10-consider-counterfactual-offline-evaluation): target actions have logging support.
- [Position bias](#step-9-diagnose-offline-biases): position affects opportunity to interact.
- [Propensity](#step-10-consider-counterfactual-offline-evaluation): logged probability of an action given context.
- [Recall@k](#step-5-compute-recallk): fraction of relevant items recovered in the first k positions.
- [Sample-ratio mismatch](#step-14-run-and-monitor-the-experiment): unexpected imbalance in assigned counts.
- [Temporal split](#step-4-split-time-and-isolate-decisions): division of training and evaluation by time.

## 16. Sources

1. Järvelin, K. and Kekäläinen, J. (2002). “Cumulated Gain-Based Evaluation of IR Techniques.” *ACM Transactions on Information Systems*. [DOI: 10.1145/582415.582418](https://doi.org/10.1145/582415.582418). Foundational DCG and normalized gain treatment.
2. Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval*, chapter 8. [Online text](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html). Precision, recall, and ranked retrieval evaluation.
3. Schnabel, T. et al. (2016). “Recommendations as Treatments: Debiasing Learning and Evaluation.” *Proceedings of Machine Learning Research* 48. [Paper](https://proceedings.mlr.press/v48/schnabel16.html). Selection and propensity-based evaluation.
4. Joachims, T., Swaminathan, A., and Schnabel, T. (2017). “Unbiased Learning-to-Rank with Biased Feedback.” *WSDM*. [DOI: 10.1145/3018661.3018699](https://doi.org/10.1145/3018661.3018699). Position bias and propensity methods.
5. Rossetti, M., Stella, F., and Zanker, M. (2016). “Contrasting Offline and Online Results when Evaluating Recommendation Algorithms.” *RecSys*. [DOI: 10.1145/2959100.2959176](https://doi.org/10.1145/2959100.2959176). Offline and online comparison.
6. Kohavi, R., Tang, D., and Xu, Y. (2020). *Trustworthy Online Controlled Experiments*. Cambridge University Press. [DOI: 10.1017/9781108653985](https://doi.org/10.1017/9781108653985). Experimental design and operational practice.
7. Gómez-Uribe, C. A. and Hunt, N. (2015). “The Netflix Recommender System: Algorithms, Business Value, and Innovation.” *ACM Transactions on Management Information Systems*. [DOI: 10.1145/2843948](https://doi.org/10.1145/2843948). Public industry case.
8. Netflix Prize. [Rules](https://www.netflixprize.com/rules.html). Historical offline rating-prediction benchmark.
9. GS1. [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin). Trade-item identification, not inventory or relevance measurement.
10. European Union. [Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj). GDPR primary text; consult current applicable law and counsel.
11. European Union. [Regulation (EU) 2022/2065](https://eur-lex.europa.eu/eli/reg/2022/2065/oj). Digital Services Act primary text.
12. California Department of Justice. [California Consumer Privacy Act overview](https://oag.ca.gov/privacy/ccpa). Official explanatory resource, not a substitute for legal assessment.

## 17. Links

- [Module 24 topic map](_index.md).
- [Recommendation tasks](01_recommendation-tasks.md): match evaluation to the customer decision.
- [Popularity and content baselines](02_popularity-content-baselines.md): establish a credible comparison.
- [Two-tower retrieval](04_two-tower-retrieval.md): evaluate candidate coverage before ranking.
- [Sequential and session models](05_sequential-session.md): evaluate context and timing.
- [Module 24 survey and lab sections](../../../course/24_recommendation_systems_personalization.md): module narrative and applied exercises.
