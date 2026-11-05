# Popularity and Content-Based Baselines

> Module 24 — Recommendation Systems and Personalization  
> Topic 02 of 11  
> Estimated study time: 4 hours  
> Prerequisites: recommendation tasks; basic probability and tabular data  
> Version 2026.5

## 1. Why this topic exists

A recommendation team is often asked to improve a product carousel before anyone can say what improvement means.
The merchandising owner wants products that sell.
The customer wants relevant, orderable items.
The supply planner wants the displayed offer to be fulfillable.
Finance wants incremental contribution, not a larger attributed revenue number.
A sophisticated model is not a substitute for agreement among these owners.

A baseline is a reproducible decision rule that can be deployed or simulated under the same constraints as a proposed model.
A useful baseline is not deliberately weak.
It makes sensible use of the information available at decision time.
It has an explicit fallback for missing data and thin segments.
It is refreshed at a documented cadence.
It is evaluated on the same population, inventory state, and business objective as the challenger.

Popularity asks what customers have recently chosen in a comparable context.
Content similarity asks which eligible items resemble an item or a customer's demonstrated interests.
Both can be explained to a business owner and implemented without learning a user-item latent representation.
Neither establishes causality merely because an item is frequently purchased or shares attributes with another item.

Consider a grocery substitution surface.
A global best-seller list can suggest a popular cereal that contains an ingredient the customer avoids.
A content match can identify a similar cereal that is unavailable in the customer's fulfillment location.
A high-margin item may be irrelevant or unsafe for the intended use.
The baseline must first enforce the task's eligibility rules and then rank what remains.

A fair comparison also matters for investment decisions.
If the challenger uses current store availability but the baseline uses yesterday's global catalog, the measured gap combines ranking and data access.
If the challenger sees clicks after the prediction time, its apparent advantage is leakage.
If one policy serves only easy, well-stocked requests, its conversion rate is not directly comparable to a policy serving all requests.
The decision is whether the extra complexity produces incremental value under equivalent conditions.

This topic equips a product manager, analyst, merchandising owner, and supply-chain owner to specify that comparison.
It covers candidate construction, segment definitions, popularity scoring, content similarity, inventory controls, and business evaluation.
It does not claim that a baseline provides an unbiased estimate of its own online impact.
Use randomized exposure to estimate incremental effects when the decision warrants it.

### The decisions and their owners

| Decision | Accountable owner | Required input |
|---|---|---|
| Define the customer task and surface | Product owner | Placement, user intent, success and guardrails |
| Approve eligible products | Merchandising and policy owners | Catalog, safety, regional and contractual rules |
| Define fulfillment eligibility | Operations owner | Location, sellable inventory, delivery promise and freshness |
| Define event quality and identity | Data owner | Event contracts, consent, deduplication and timestamps |
| Choose ranking and experiment design | Recommender lead and experiment owner | Baselines, uncertainty, exposure and power |
| Approve economic value definition | Finance | Contribution components, returns and attribution rules |
| Make launch or rollback decision | Product owner with operations and finance | Pre-agreed thresholds and experiment results |

An accountable owner can delegate implementation but must not delegate the decision implicitly to a model score.
The product owner resolves trade-offs only after the safety and eligibility owners have approved the feasible set.
A ranking score is not authority to relax a mandatory restriction.

## 2. Learning outcomes

After this topic you can perform the following observable tasks.
Each outcome is assessed in the exercises and self-checks.

1. **Specify** a baseline decision contract with a task, request time, target population, and fallback.
   Exercise 1 asks you to write this contract.
2. **Compute** a recency-weighted popularity score and explain why exposure affects its interpretation.
   Exercise 2 provides a hand-calculation check.
3. **Choose** useful segment boundaries and a support-based fallback without treating missing values as preferences.
   Exercise 3 asks you to route sparse and well-supported requests.
4. **Calculate** content similarity for normalized item vectors and identify invalid substitutes.
   Exercise 4 tests the calculation and the eligibility gate.
5. **Apply** a location- and time-specific inventory rule and identify stale-data behavior.
   Exercise 5 asks for an eligible list and a stockout response.
6. **Compare** baseline and challenger rankings using matched replay and an online randomized business metric.
   Exercise 6 asks you to distinguish predictive metrics from causal outcomes.
7. **Design** a launch scorecard and ownership plan that includes contribution, service, distribution, and rollback.
   Exercises 7 and 8 use decision rubrics rather than one unique solution.

The outcomes are intentionally operational.
You should be able to reproduce a number and explain what action the number supports.
A formula alone is insufficient if nobody can describe its inputs, exclusions, and owner.

## 3. The core mental model

A recommendation is a constrained decision made at a particular request time.
The same ranker can produce a different list when the fulfillment location or catalog state changes.
Separate *who may be considered* from *how eligible items are ordered*.

```text
request: task, user/session context, location, time
                 |
                 v
       point-in-time data contract
     catalog + policy + stock + history
                 |
                 v
       eligibility / candidate set
       |                    |
       v                    v
  popularity score    content similarity
       |                    |
       +---------+----------+
                 v
       stable rank and fallback
                 |
                 v
       display, exposure log, outcome
                 |
        +--------+---------+
        v                  v
   offline replay    randomized online test
        |                  |
        +--------+---------+
                 v
     product + operations + finance decision
```

The arrows describe information flow, not proof of causation.
Historic orders are caused partly by customer demand and partly by past exposure, price, promotion, stock, and placement.
A popularity score inherits those influences.
A similarity score inherits whatever distinctions and errors are present in the product attributes.

At request time, choose a task such as related items, a replacement, or a complement.
The task determines the initial universe and mandatory exclusions.
A replacement normally needs compatibility with the original item's use.
A complement may deliberately differ in category.
Do not apply a same-category substitution rule to a complement task without justification.

The eligibility gate receives a catalog snapshot and a fulfillment snapshot available by the request time.
It removes items that cannot be offered to this customer on this surface.
It does not know whether an eligible item is relevant.
The ranker then orders eligible candidates using a documented score and tie-breaker.
If no candidate remains, a fallback may show a permitted message or no recommendation.
An empty safe result is preferable to an impermissible product.

Offline replay measures agreement with observed historical behavior under a defined logging policy.
The observed outcome is not a label for every unshown item.
Randomized online comparison can estimate the causal effect of a policy assignment, if the randomization and analysis are valid.
Both forms of evidence should use the same unit of analysis and clearly report coverage.

A forecast is a prediction, such as an estimated future conversion rate.
A plan is a proposed operational action, such as refreshing lists daily.
A target is a desired metric, such as an increase in contribution per eligible session.
A commitment is an approved promise, such as a delivery window offered at checkout.
A predicted stock level is not a delivery commitment.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Recommendation task | A decision about which items to show in a defined context | Treating every surface as the same relevance problem |
| Request | One opportunity to generate a list at a timestamp | Counting only requests that received a click |
| Candidate | An item considered for ranking on a request | Assuming every catalog item is orderable |
| Eligibility | Mandatory conditions an item must meet before display | Treating a safety rule as a soft score |
| Ranking | Ordering candidates by a decision rule | Confusing retrieval coverage with ranking quality |
| Top-k | First k items returned; k is a count of slots | Omitting requests that return fewer than k items |
| Global popularity | Aggregate behavior over a defined population and window | Calling lifetime sales universal customer preference |
| Segment | Group defined by request-time attributes | Using post-purchase attributes to assign the group |
| Segment support | Number of relevant historical observations or independent entities | Counting repeated events as independent customers |
| Backoff | Moving from a sparse segment to a broader rule | Quietly substituting a different task |
| Event | Timestamped recorded action, such as view, cart, or purchase | Assuming one event equals one satisfied customer |
| Exposure | A logged opportunity for a customer to see an item | Inferring it from a candidate list that was never rendered |
| Conversion | A specified desired action divided by a defined denominator | Switching between users, sessions, and impressions |
| Recency weight | Weight assigned according to event age, dimensionless | Using future events in a historical replay |
| Decay half-life | Time after which an event weight is halved | Choosing a half-life without stating time units |
| Smoothing | Combining a noisy segment estimate with a prior | Claiming a prior removes systematic exposure bias |
| Content feature | Item attribute represented in a vector or matching rule | Assuming all missing fields mean an attribute is absent |
| Token | A discrete term derived from an attribute or text | Treating unrelated units or identifiers as equivalent |
| TF-IDF | Term frequency–inverse document frequency representation | Fitting vocabulary or document frequencies on future data |
| Cosine similarity | Dot product divided by vector norms, dimensionless | Treating similarity as purchase probability |
| Zero vector | Feature vector with zero norm | Dividing by zero or pretending it is a close match |
| Seed item | Item used as the reference for related-item similarity | Recommending the seed itself as a substitute |
| Profile | Aggregated history-based representation of customer interests | Using activity collected without a permitted purpose |
| Cold start | Insufficient user or item interaction history | Concluding that content data are always complete |
| Sellable inventory | Stock that can actually be offered for a location and channel | Equating physical on-hand count with available-to-promise |
| Available-to-promise | Quantity the business can promise after applicable reservations and rules | Treating a point-in-time estimate as guaranteed delivery |
| Freshness | Age of the latest valid data at decision time, usually seconds or minutes | Reporting event time while ignoring ingestion delay |
| Stock buffer | Quantity withheld from recommendations to reduce overselling risk | Treating a buffer as a reservation |
| Constraint | Hard eligibility rule or a declared soft ranking preference | Hiding mandatory rules inside an opaque score |
| Impression | A defined rendered or viewable display event | Counting all returned candidates as seen |
| CTR | Click-through rate, clicks divided by specified impressions | Comparing CTRs with different impression definitions |
| Recall@k | Relevant held-out items in top k divided by eligible held-out relevant items | Treating missing, unexposed items as known negatives |
| NDCG@k | Normalized discounted cumulative gain at cutoff k | Comparing values computed from different label universes |
| Coverage | Fraction of a defined request or item universe receiving an outcome | Reporting catalog coverage without a denominator |
| Incrementality | Difference in outcomes caused by policy assignment | Calling last-touch attributed revenue incremental revenue |
| Contribution | Revenue less a defined set of variable costs, in currency | Calling gross revenue profit |
| Guardrail | Metric or rule used to limit unacceptable side effects | Looking at guardrails only after selecting a winner |
| Stable tie-break | Deterministic ordering when scores are equal | Allowing database iteration order to determine exposure |
| Leakage | Use of information unavailable at decision time | Treating a chronological split as sufficient without point-in-time joins |
| Positivity | Nonzero probability of each action being evaluated in the relevant context | Estimating an unshown policy from logs with no support |

## 5. The mechanics, step by step

### 5.1 Define the task and decision boundary

**Input:** product brief, placement, customer journey, and business rules.
**Output:** a written request contract and success criterion.

1. Name the action: for example, show up to three substitutes after a grocery item becomes unavailable.
2. Identify the decision time: for example, the instant the customer opens the substitution dialog.
3. Identify the decision unit: request, session, order, or customer.
4. State the unit used for business analysis separately from the serving request.
5. Define the population, channel, geography, device, and observation period.
6. Specify the exact destination of each item, including checkout and fulfillment flow.
7. Identify whether the recommendation is a related item, substitute, complement, or next offer.
8. Set a primary decision metric and guardrails before seeing the results.
9. Record who can approve a fallback, stop a test, or override a policy.

A request contract should also specify what constitutes an empty result.
For a substitute, the approved fallback may be a customer choice to remove the item.
For a discovery carousel, it may be to omit the carousel.
Do not silently fill vacant slots with items that fail hard eligibility rules.

### 5.2 Define the point-in-time data contract

**Input:** catalog, inventory, customer context, events, and policy tables.
**Output:** versioned fields with explicit timestamps and quality checks.

For every event, retain an event timestamp and an ingestion timestamp.
For every feature, retain when it became available to the serving system.
In a replay at time `t`, include only data that were available by `t`.
An order event that occurred before `t` but arrived after `t` is unavailable to the online policy at `t`.
The same principle applies to corrected product attributes and late inventory adjustments.

Record item identifier, catalog version, segment fields, location, channel, policy status, and ranking version.
Log a request identifier and the final ordered list actually rendered.
Log whether an item was merely returned, rendered, or viewable, according to a consistent measurement standard.
Store price, stock freshness, candidate exclusions, and score components where permitted and proportionate.
A candidate log can be sampled if storage or privacy constraints require it; document the sampling rule.
Do not infer unrecorded eligible candidates from the final list.

Define deduplication keys for retries and order updates.
Distinguish an initial order, cancellation, return, and replacement.
A net purchase signal may require an observation window because a return occurs later than an order.
Never attach a future return to a score computed before the return was known.
A training label can mature later, while the features must remain point-in-time correct.

### 5.3 Create the eligible universe

**Input:** task, location, policy and catalog snapshots, and inventory state.
**Output:** eligible item set `E(t, request)` and exclusion reasons.

Start with active products for the sales channel and territory.
Exclude the seed item when suggesting alternatives, unless the task explicitly permits it.
Apply age, safety, regulatory, and contractual restrictions as approved by the relevant owner.
Apply task-specific compatibility constraints such as product type, dietary requirements, or technical fit when these are known and required.
Use approved handling for unknown values; unknown is not automatically compatible.

Check the correct fulfillment node rather than a national total.
The node may be a store, warehouse, or delivery zone selected by the order-routing service.
Evaluate the intended fulfillment time when the business maintains valid future availability, not merely the current shelf count.
If a product has variants, check the purchasable variant rather than a parent product with aggregate stock.

A simple gate is `available_to_promise >= requested_quantity + buffer`.
All quantities must use compatible units and reflect reservations according to the inventory system's contract.
A threshold cannot guarantee fulfillment when updates are delayed or demand changes.
If the inventory record is stale beyond the permitted age, follow the approved conservative rule, such as withholding the product or showing a noncommittal availability status.
The proper branch depends on customer harm and the reliability of the fallback.

Capture the count excluded for each reason and the final eligible-set size.
A high relevance score cannot resurrect an excluded item.
When fewer than `k` products are eligible, return fewer than `k` or invoke a separately approved fallback.
Never use inventory from a later timestamp to make a historical decision appear feasible.

### 5.4 Build the global popularity baseline

**Input:** eligible candidates, events known by the scoring cutoff, and a scoring window.
**Output:** a score and deterministic rank for each candidate.

Choose the behavioral signal that corresponds to the task.
Completed orders may be more meaningful than clicks for a purchase surface, but they arrive less often and can be biased by prior placement.
For a substitute acceptance surface, use accepted substitutions if sufficient and appropriately logged.
Avoid combining views, clicks, and orders without documenting the relative weights and their rationale.

A simple recency-weighted purchase score for item `i` at time `t` is:

```text
pop(i, t) = sum over eligible events e for i:
            w(type_e) * 2 ** (-(t - event_time_e) / half_life)
```

The score has weighted-event units, not dollars or probability.
The half-life and elapsed time must use the same time unit.
Filter events to a finite historical lookback and known data at `t`.
The event-type weight should be nonnegative unless a separately justified negative-signal model is specified.
A half-life of seven days gives an event from seven days ago half the weight of an otherwise identical event now.

Count unique orders or units deliberately.
An item bought in large quantities by one business customer can dominate unit sales while being irrelevant to individual shoppers.
Choose the unit aligned to the request population and measure sensitivity to outliers.
Deduplicate retries before aggregation.
Use explicit rules for canceled and returned items, with availability times respected.

Sort by descending score, then by an approved stable item identifier.
Items with zero observed events can receive zero and remain eligible.
A predetermined merchandising or content fallback can prevent arbitrary identifiers from dominating long zero-score tails.
A popularity score is a popularity measure under historical exposure, not an unbiased latent-demand estimate.

### 5.5 Add segments and a support ladder

**Input:** request-time segment fields and historical events with matching definitions.
**Output:** segment-specific ranking or a documented broader fallback.

Segment only on fields available at serving time and allowed for the intended use.
Examples include fulfillment region, channel, product category, season, and a coarse explicit preference.
A store-specific list may reflect local demand but have too few observations for a stable rank.
A broad list has more support but may ignore local preferences and assortment.

Define a hierarchy before inspecting the test outcomes.
One example is store-category, then region-category, then national-category.
For each level specify the minimum number of distinct eligible orders and the lookback period.
The threshold is a policy choice; it is not a guarantee of statistical precision.
Monitor segment size and how often each fallback is used.
If location is unknown, do not invent a location; route to an approved broader eligible universe or withhold location-dependent recommendations.

A simple smoothed estimate for a segment share is:

```text
share(i | s) = (n_si + alpha * p_parent_i) / (N_s + alpha)
```

Here `n_si` is the segment's count for item `i`, `N_s` is the total count over the defined item universe, and `p_parent_i` is the parent share on the same universe.
`alpha` is a nonnegative prior strength measured in pseudo-events.
The shares sum to one only when the counts and parent probabilities cover the same exhaustive universe.
Choose `alpha` using historical validation, then freeze it for the comparison.
Smoothing reduces variance under its assumptions; it does not correct selection, exposure, or eligibility bias.
If the score is recency weighted, call its units weighted events and ensure the parent score uses the same definition.

Do not build hundreds of tiny demographic groups merely because the fields exist.
More partitions increase variance, operational burden, and the risk of sensitive or unfair treatment.
Review permitted use with privacy and legal owners.
Report performance and fallback rates by meaningful approved groups without revealing individual behavior.

### 5.6 Build an item-content baseline

**Input:** seed item, point-in-time catalog attributes, and eligible candidate set.
**Output:** a similarity score and stable ranking.

Choose features that represent relevant properties of the task.
For a grocery substitute, category, dietary attributes, flavor, pack size, and brand may matter.
For a technical replacement, physical dimensions, compatibility, and certification may be mandatory gates rather than similarity features.
For complements, use a different relation or curated compatibility mapping; nearest neighbors are often the wrong objective.

Represent categorical values with controlled tokens such as `category:tea` and `diet:caffeine_free`.
Prefix tokens with their field name so that unrelated attributes cannot accidentally match.
Normalize units before creating numeric features.
For example, express package size in grams rather than mixing grams and ounces as incomparable text.
Validate catalog source, locale, and attribute freshness.

For binary vectors, cosine similarity between a seed vector `x` and candidate vector `y` is:

```text
cos(x, y) = (x dot y) / (||x||_2 * ||y||_2)
```

When both vectors contain only zeros and ones, the numerator is the number of shared active tokens.
The denominator normalizes by the square roots of the token counts.
Similarity is undefined if either norm is zero.
Route an undefined score to an approved fallback rather than divide by zero.
For nonnegative vectors, the score ranges from zero to one; a score of one does not establish interchangeability.

You may weight attributes by multiplying feature values before computing similarity.
Document the weights, and recognize that a feature multiplier is squared in the dot product when both vectors contain the feature.
Alternatively define an explicit weighted overlap rule when that is easier to audit.
Fit a text vocabulary and inverse document frequency only from documents available by the training cutoff.
Use a frozen vocabulary during replay; new terms require an approved unknown-token or refresh strategy.

Apply mandatory restrictions before similarity ranking.
A shared brand and flavor cannot compensate for an incompatible allergen or connection standard.
Use popularity only as a stated tie-break or fallback if that is the baseline being tested.
A hybrid that mixes similarity and popularity is a distinct policy and needs its own version and comparison.

### 5.7 Create a customer-content profile when appropriate

**Input:** permitted prior interactions, item vectors, and request context.
**Output:** a normalized profile or an explicit cold-start fallback.

Aggregate vectors of previously interacted-with items using weights and decay.
Use only events known by the request time and consistent with the relevant purpose and consent choices.
A purchase may indicate gift buying or a one-time need, so do not equate interaction with enduring preference.
Optionally exclude previously purchased products if the task seeks discovery, but do not impose that rule on a replenishment task.

Normalize the resulting profile before cosine scoring.
An empty history or zero profile requires a fallback, often a context-specific popularity list.
The fallback must still obey the same eligibility and inventory rules.
If the profile is built at session level, expire it according to the approved retention and session definitions.
Do not treat absence of an interaction as evidence of dislike.

### 5.8 Compare policies on matched opportunities

**Input:** frozen policy versions, a time-ordered test period, point-in-time snapshots, and common labels.
**Output:** comparable offline results with denominators and caveats.

Choose a training window, validation window, and strictly later test window.
Tune half-life, segment thresholds, content weights, and fallback behavior on validation data only.
Freeze the selected policy before evaluating on the final test period.
Keep customers or sessions grouped as appropriate to avoid dependent observations being treated as independent.
A time split does not by itself prevent leakage from revised features or future catalog state.

Run every policy on the same set of requests with the same request-time information and hard eligibility rules.
For a retrieval comparison, report the candidate universe and candidate recall before comparing ranking metrics.
If a policy cannot serve a request, include that request in coverage and business denominators rather than silently dropping it.
Record list length, missing scores, fallback frequency, and exclusion counts for each policy.
Evaluate both overall and by approved, prespecified segments.

For one eligible held-out item per request, hit@k is one if the item appears in the first `k` positions and zero otherwise.
Mean hit@k averages that indicator over the specified requests.
For multiple relevant items, recall@k divides the number retrieved in the top `k` by the number of eligible relevant items.
State how requests with no observed relevant item are treated; they do not have a defined recall denominator.
For NDCG, document relevance grades, logarithmic discount, cutoff, and treatment of zero ideal gain.
Do not compare metrics across evaluations with different label universes or exclusions.

Historical labels are conditional on the old policy's exposure.
A held-out purchase of item A supports an observed positive label for A, but an unshown item B is not necessarily irrelevant.
Offline replay is therefore a diagnostic and screening tool, not proof that online revenue will rise.
Counterfactual estimators require logged propensities, overlap, and further assumptions; they cannot recover outcomes for actions with zero probability.
When those conditions fail, collect exploration data or use a randomized test.

### 5.9 Estimate business impact online

**Input:** eligible traffic, randomized assignment, policy logs, outcome data, and a prespecified analysis plan.
**Output:** an incremental effect estimate, uncertainty, guardrails, and a decision.

Randomize at a unit that prevents meaningful interference between variants.
Customer-level assignment may be preferable when a customer sees multiple requests, while a store-level design may be needed when inventory interference is substantial.
Keep assignment stable for the planned exposure period and record the assignment independently of rendering.
Use an intention-to-treat analysis over all assigned eligible units unless a different estimand is justified in advance.
This preserves the benefit of randomization when a policy fails to serve or a customer never sees the surface.

A suitable primary metric might be contribution per assigned eligible session over a defined attribution window.
Compute contribution from an agreed ledger, for example net sales less product cost, fulfillment cost, promotion cost, and expected return cost.
Specify whether taxes, fixed overhead, cancellations, refunds, and cross-category effects are included.
Use the same definition in both variants and wait for outcome maturity or disclose censoring.
Do not add item-attributed revenue to total order revenue; they overlap.

Report the absolute difference and relative difference with an uncertainty interval.
Use methods compatible with the randomization unit and repeated observations.
A statistically uncertain result is not evidence of zero impact.
A precisely estimated tiny gain may still be below the minimum economically worthwhile effect.
Plan sample size, duration, stopping rules, and multiple comparisons before launch; involve an experiment statistician.

Track guardrails such as stockout after click, canceled lines, return rate, delivery promise misses, customer complaints, latency, and eligible-list coverage.
Monitor exposure concentration and supplier or category distribution where relevant.
The supply owner can require a stop when a service guardrail breaches its pre-agreed rule even if click-through increases.
Document missing data and assignment imbalance before interpreting the effect.

### 5.10 Operate, refresh, and retire

**Input:** approved policy, service-level objectives, monitoring, and ownership.
**Output:** a reproducible production decision with rollback path.

Schedule the popularity refresh to match demand volatility and data latency.
Define the maximum acceptable age of a list separately from the maximum acceptable age of inventory.
A daily popularity list can be combined with a current serving-time inventory gate.
Record training cutoff, data availability cutoff, code version, catalog version, parameters, and activation time.

Test behavior for missing location, empty history, stale stock, missing attributes, ties, and empty eligible sets.
Use predeployment checks and production monitoring under the team's change process.
Provide a simple prior policy to roll back to, but never bypass hard safety and stock rules during rollback.
Track whether observed outcomes and segment support drift after launch.
Reassess the baseline when the surface, assortment, pricing regime, or fulfillment network changes.

### 5.11 Diagnose disagreement between baselines

**Input:** matched lists, exclusion reasons, item attributes, and observed outcomes.
**Output:** a decision-ready explanation of where each policy succeeds or fails.

Do not stop after reporting one average metric.
Partition matched requests into cases where both policies serve the same first item, different first items, or no item.
For each partition, report the number of requests and its fraction of all eligible requests.
List agreement is a diagnostic, not an outcome metric.
A high agreement rate can conceal disagreement precisely on the difficult requests.

For disagreements, inspect whether the content policy favors a matching attribute while popularity favors a frequently purchased item.
Check if that attribute is trustworthy, current, and appropriate to the task.
Check whether the popularity signal reflects genuine demand, placement, discounting, or a temporary promotion.
A manually reviewed sample can expose catalog defects, but reviewers should not see which policy produced an item when feasible.
Agree on a review rubric before inspecting outcomes.
Track inter-reviewer disagreement and escalate restrictions to the appropriate subject-matter owner.

Compare coverage and list length at the same cutoff.
Calculate `served requests / all eligible requests`, not `served requests / requests with a click`.
Calculate mean list length including zero-length lists.
If the product surface requires three slots but only one safe item is available, record the shortfall.
Do not increase coverage by bypassing the eligibility gate.
Investigate whether the shortfall comes from true scarcity, stale stock, sparse features, or incorrect policy data.
Each cause has a different owner and remedy.

Inspect diversity using a clearly specified measure.
For example, report the number of distinct brands in each returned list when brand diversity is relevant.
The number of distinct brands is not a measure of customer satisfaction or supplier fairness by itself.
If a brand cap is a hard requirement, apply it identically to policies being compared.
If it is merely an optimization objective, include its trade-off in a separate policy and evaluate it as such.

### 5.12 Plan refresh and computational boundaries

Popularity can be precomputed for each approved segment and refreshed on a schedule.
At request time, retrieve the appropriate list and intersect it with the current eligible set.
If the cached list contains too few eligible items, retrieve a deeper ranked pool or use the approved broader fallback.
Choose the pool depth based on observed exclusions rather than assuming the first `k` cached items will always survive.
Log the retrieved depth and the fraction of requests that exhaust it.

Precomputing popularity does not justify precomputing a final list without regard to current stock.
The final availability check must use the approved request-time inventory contract.
If the stock service is unavailable, follow the specified outage behavior.
The outage branch should be monitored separately from ordinary fallback.
Avoid interpreting an outage-driven drop in list length as a ranking regression.

Content vectors can be computed when a catalog item is created or updated.
Version the tokenization, feature weights, and source catalog snapshot.
If a feature changes after a product reformulation, the vector should not continue to describe the old formulation indefinitely.
During a historical replay, select the vector version available at that historical time.
For large catalogs, approximate nearest-neighbor retrieval may change which candidates are considered.
Measure retrieval recall against exact neighbors on a representative sample before attributing differences to the ranking score.

The business owner should understand the cost of each policy.
Record refresh compute, storage, serving latency, and the operational cost of maintaining attributes.
Compare these costs against incremental contribution, not against a model's attributed revenue.
A more complex policy with a small uncertain benefit may not justify its ongoing maintenance.
Conversely, a simple policy with unsafe data dependencies is not acceptable merely because it is inexpensive.

### 5.13 Interpret counterfactual evidence carefully

Suppose the historical policy displayed item A on every request and never displayed item B.
The resulting logs cannot directly identify how those same customers would have reacted to B.
More observations of A do not create support for B.
An inverse-propensity estimator cannot divide by a zero exposure probability to manufacture that missing evidence.
Its validity also depends on correctly logged assignment probabilities and the assumptions of the estimator.

Randomized exploration can create support, subject to safety, policy, and commercial constraints.
Specify the exploration population and eligible actions before collecting data.
Log the probability with which each action was selected under the actual serving policy.
If ranking multiple slots, the probability of a slate may differ from each item's marginal probability.
The appropriate estimator depends on what action and outcome are being studied.
Work with an experiment statistician rather than assuming an item-level propensity is always sufficient.

An online experiment itself needs an interference assessment.
If a treatment causes one group to consume scarce stock, a control group sharing that stock may experience a different outcome.
This violates a simple no-interference interpretation of customer-level assignment.
Potential responses include larger allocation units, inventory partitions, or a design explicitly accounting for spillovers.
Each response changes the estimand, operational burden, or statistical precision.
Document the chosen design and its limits before interpreting a treatment effect.

### 5.14 Translate the scorecard into a decision

Write the launch rule before reading the experiment result.
State the primary metric, minimum worthwhile effect, interval or decision criterion, and guardrail thresholds.
Define the outcome window and the procedure for late refunds and returns.
Name the person authorized to stop the experiment for an urgent service issue.
Name the person authorized to approve a full rollout.
Set a review date for unresolved or inconclusive evidence.

If the estimate is positive but the uncertainty interval includes economically meaningful losses, describe the result as inconclusive under the chosen criterion.
If the estimate is precise but smaller than the agreed economic threshold, the policy may not be worth operating.
If a hard guardrail fails, do not average the failure away using an unrelated engagement improvement.
If the estimate meets the criterion and guardrails hold, plan a staged rollout with monitored eligibility and fallback rates.
Retain the previous policy version and the same mandatory controls for rollback.
The decision record should state what was observed, what remains uncertain, and who accepted the residual risk.

## 6. Worked example

### 6.1 Decision and data

A regional grocer needs up to two substitutes for an unavailable 500 gram bag of unsweetened oat cereal.
The decision occurs at 10:00 local time on a Tuesday.
The request is for delivery from store S1, with one unit requested.
The example uses illustrative data, not a company result.
The operations owner requires at least one sellable unit after a one-unit buffer.
The catalog owner requires cereal category and unsweetened status; the seed itself is excluded.
The product owner approves showing fewer than two items if necessary.

The table uses store-specific available-to-promise (ATP), after reservations, at 09:59.
All values are illustrative snapshots available before the request.
A fresh snapshot is defined here as no more than five minutes old.

| Item | Category | Unsweetened | Shared feature tokens with seed | Total tokens | ATP, units | Result |
|---|---|---|---:|---:|---:|---|
| A | Cereal | Yes | 3 | 4 | 5 | Eligible |
| B | Cereal | Yes | 2 | 3 | 2 | Eligible |
| C | Cereal | Yes | 3 | 4 | 1 | Excluded: buffer |
| D | Cereal | No | 3 | 4 | 9 | Excluded: compatibility |
| E | Cereal | Yes | 1 | 2 | 4 | Eligible |

The threshold is `1 requested + 1 buffer = 2 units`.
A and B pass; C does not, even though one unit is physically available to promise.
D fails a task rule even though its stock is ample.
E passes the hard rules but may be a weaker match.
The eligible set is `{A, B, E}`.
The seed has four binary content tokens.
The shared-token counts include the required category and unsweetened tokens only where present in the illustrative vector.
For E, the hard unsweetened rule is verified in a separate authoritative policy field; its content vector is incomplete.
That distinction is deliberate: policy eligibility and model features need not have identical completeness.

### 6.2 Popularity score

Use a seven-day half-life and count one weighted event per completed order in the last 14 days.
All events were ingested before 10:00.
For simplicity the table aggregates events by age, and no subsequent cancellation was known by the cutoff.
The weights for zero, seven, and 14 days are 1, 0.5, and 0.25 respectively.
The inclusive 14-day boundary is part of this example's contract.

| Item | Orders age 0 days | Orders age 7 days | Orders age 14 days | Weighted score |
|---|---:|---:|---:|---:|
| A | 4 | 2 | 0 | `4 + 2 × 0.5 = 5.00` |
| B | 3 | 0 | 4 | `3 + 4 × 0.25 = 4.00` |
| C | 8 | 0 | 0 | `8.00`, excluded before ranking |
| D | 7 | 0 | 0 | `7.00`, excluded before ranking |
| E | 1 | 2 | 0 | `1 + 2 × 0.5 = 2.00` |

Popularity returns A then B.
C's high score does not override the stock buffer.
D's high score does not override compatibility.
Neither score is a probability of a customer accepting the substitute.

### 6.3 Content score

Compute binary cosine using the four-token seed.
The score is shared tokens divided by the square root of the product of token counts.
The candidate's content tokens need not include every authoritative policy field.

| Item | Calculation | Similarity | Eligible? |
|---|---|---:|---|
| A | `3 / sqrt(4 × 4)` | 0.750 | Yes |
| B | `2 / sqrt(4 × 3)` | 0.577 | Yes |
| C | `3 / sqrt(4 × 4)` | 0.750 | No: buffer |
| D | `3 / sqrt(4 × 4)` | 0.750 | No: compatibility |
| E | `1 / sqrt(4 × 2)` | 0.354 | Yes |

Content also returns A then B for this request.
Agreement on this request does not imply the policies are equivalent across all requests.
A missing token reduces E's similarity; it does not prove the real product lacks that attribute.
The team should improve completeness rather than interpreting a feature omission as a customer preference.

### 6.4 Segment smoothing

Suppose a separate store-category sample contains 10 eligible historical orders: A has 6, B has 3, and E has 1.
The parent region-category shares over the same three-item universe are A 0.40, B 0.35, and E 0.25.
Set prior strength `alpha = 10` pseudo-orders.
This example uses unweighted counts, not the weighted-event score above.

| Item | Smoothed calculation | Smoothed share |
|---|---|---:|
| A | `(6 + 10 × 0.40) / (10 + 10)` | 0.5000 |
| B | `(3 + 10 × 0.35) / (10 + 10)` | 0.3250 |
| E | `(1 + 10 × 0.25) / (10 + 10)` | 0.1750 |

The shares total 1.0000 because the parent and segment use the same exhaustive item universe.
The ten pseudo-orders pull the local estimate toward the parent distribution.
They do not mean ten new customers actually bought anything.
If the minimum local support is 20 distinct orders, this segment instead backs off under the specified ladder.
A team can use smoothing and a minimum support rule together, but must specify when each applies.

### 6.5 Illustrative business comparison

Suppose a randomized pilot assigns 1,000 eligible sessions to each policy.
The unit is the eligible session and every assigned session remains in the denominator.
After the same outcome window, the popularity arm has USD 2,400 contribution and the content arm has USD 2,520 contribution.
The absolute difference is `(2520 / 1000) - (2400 / 1000) = USD 0.12 per assigned session`.
The relative difference is `0.12 / 2.40 = 5%`.
These are illustrative sample estimates, not evidence that content truly improves contribution.
No variance or confidence interval is supplied, so a launch decision cannot be justified from the point estimates alone.

Suppose the content arm has 18 stockout-after-click events out of 200 clicked sessions and popularity has 12 out of 200.
The rates are 9% and 6% of clicked sessions, respectively.
This is a diagnostic among post-assignment clickers, not automatically a causal comparison for a fixed population: the policies may change who clicks.
Also report stockout events per assigned eligible session and apply the prespecified service guardrail.
The supply owner must review the breach rule before product and finance approve launch.
Do not extrapolate the USD 0.12 estimate as a commitment or forecast without uncertainty, scaling assumptions, and a separate plan.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the popularity and cosine results above, applies eligibility before ranking, and prints both top-two lists.
It is a compact teaching implementation, not a production inventory reservation service.
The event ages are already calculated relative to the request and all events are assumed known at the cutoff.
Inventory freshness, reservations, policy-field provenance, and ingestion-time checks would normally be enforced upstream.
The catalog lists the seed's feature tokens and each candidate's tokens explicitly.
E's missing content token is not the authoritative unsweetened policy field.

```python
from math import sqrt

SEED_TOKENS = {"category:cereal", "diet:unsweetened", "base:oat", "format:bag"}
REQUEST_QTY = 1
BUFFER = 1
HALF_LIFE_DAYS = 7.0
LOOKBACK_DAYS = 14.0

items = {
    "A": {
        "category": "cereal", "unsweetened": True, "atp": 5,
        "tokens": {"category:cereal", "diet:unsweetened", "base:oat", "format:box"},
    },
    "B": {
        "category": "cereal", "unsweetened": True, "atp": 2,
        "tokens": {"category:cereal", "diet:unsweetened", "base:rice"},
    },
    "C": {
        "category": "cereal", "unsweetened": True, "atp": 1,
        "tokens": {"category:cereal", "diet:unsweetened", "base:oat", "format:box"},
    },
    "D": {
        "category": "cereal", "unsweetened": False, "atp": 9,
        "tokens": {"category:cereal", "base:oat", "format:bag", "diet:sweetened"},
    },
    "E": {
        "category": "cereal", "unsweetened": True, "atp": 4,
        "tokens": {"category:cereal", "base:rice"},
    },
}

# Each pair is (item ID, event age in days); one event is one completed order.
events = (
    [("A", 0)] * 4 + [("A", 7)] * 2
    + [("B", 0)] * 3 + [("B", 14)] * 4
    + [("C", 0)] * 8 + [("D", 0)] * 7
    + [("E", 0)] + [("E", 7)] * 2
)


def eligible(item):
    return (
        item["category"] == "cereal"
        and item["unsweetened"] is True
        and item["atp"] >= REQUEST_QTY + BUFFER
    )


def popularity(item_id):
    return sum(
        2 ** (-age / HALF_LIFE_DAYS)
        for event_item, age in events
        if event_item == item_id and 0 <= age <= LOOKBACK_DAYS
    )


def cosine_binary(seed, candidate):
    if not seed or not candidate:
        return None
    return len(seed & candidate) / sqrt(len(seed) * len(candidate))


eligible_ids = [item_id for item_id, item in items.items() if eligible(item)]
pop_scores = {item_id: popularity(item_id) for item_id in eligible_ids}
content_scores = {
    item_id: cosine_binary(SEED_TOKENS, items[item_id]["tokens"])
    for item_id in eligible_ids
}
# Undefined similarity falls behind defined scores; ID is the stable tie-break.
pop_order = sorted(eligible_ids, key=lambda i: (-pop_scores[i], i))
content_order = sorted(
    eligible_ids,
    key=lambda i: (
        content_scores[i] is None,
        -(content_scores[i] if content_scores[i] is not None else 0.0),
        i,
    ),
)

print("eligible:", ", ".join(eligible_ids))
print("popularity:", ", ".join(f"{i}={pop_scores[i]:.3f}" for i in pop_order))
print("content:", ", ".join(
    f"{i}={content_scores[i]:.3f}" if content_scores[i] is not None else f"{i}=undefined"
    for i in content_order
))
print("top-2 popularity:", ", ".join(pop_order[:2]))
print("top-2 content:", ", ".join(content_order[:2]))
```

Expected output:

```text
eligible: A, B, E
popularity: A=5.000, B=4.000, E=2.000
content: A=0.750, B=0.577, E=0.354
top-2 popularity: A, B
top-2 content: A, B
```

The code gives an undefined similarity a deterministic last position.
That is acceptable for this small example only if the product owner accepts identifier ordering for the remaining undefined-score slots.
In production, a declared popularity fallback or shorter list is usually more interpretable.
The example does not learn weights from outcomes, reserve inventory, or estimate an online treatment effect.

## 8. Data, standards, and regulatory context

### 8.1 Data contracts

A product identifier should identify the exact purchasable variant.
A parent product may aggregate several sizes or flavors with different inventory and eligibility.
A location identifier must map to the node responsible for the intended promise.
Document the mapping's effective dates and what happens when routing changes.

The Global Trade Item Number (GTIN), defined by GS1, can identify trade items across organizations where it is used.
A GTIN is not itself proof of stock, compatibility, a safe substitution, or a valid customer preference.
Internal stock-keeping unit identifiers may still be necessary for channel-specific variants.
Do not assume a one-to-one mapping between every internal item and every external identifier without validation.

Catalog attributes need a source, effective time, permitted values, units, and missing-value convention.
An ingredient list may be authoritative for a restriction while a search index is merely a derived feature.
A mismatch needs an owner and correction workflow.
If a mandatory field is missing, the policy should specify whether to exclude or seek an approved alternative source.

Inventory fields need explicit semantics.
On-hand, reserved, damaged, inbound, and available-to-promise quantities are not interchangeable.
A single number cannot convey update lag, expected demand, and allocation policy without supporting metadata.
Synchronize the displayed fulfillment channel with the node whose inventory was checked.
A reservation is a separate transaction; ranking an item does not reserve it.

### 8.2 Measurement standards

Define impressions using a consistent rendering or viewability rule and retain that definition with reports.
An item returned by a service but never rendered is not necessarily an impression.
A rendered item below the visible viewport may not have been seen.
The right definition depends on the surface and the business question; record it rather than mixing definitions.

Event ordering matters when a user updates an order or an item is replaced.
Deduplicate retries using stable event identifiers, not only identical timestamps.
For financial reporting, reconcile recommendation-attributed events to order lines and refund records.
Keep the analysis denominator independently of successful logging of clicks and purchases.
Investigate sample-ratio mismatch and missing exposure events before treating a test as valid.

### 8.3 Privacy and legal scope

A recommendation profile can contain personal data even when it contains only pseudonymous identifiers.
The European Union General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, sets requirements for processing personal data, including principles such as purpose limitation and data minimization in Article 5.
Applicable lawful basis and transparency obligations depend on the processing and facts; consent is not the only possible lawful basis under the GDPR.
Review retention, access, deletion, and cross-border handling with qualified counsel and the privacy owner.
This course material is educational and is not legal advice.

Article 22 concerns certain solely automated decisions producing legal or similarly significant effects; it does not automatically apply to every product carousel.
Assess applicability with counsel rather than inferring it from the presence of a model.
The EU Digital Services Act, Regulation (EU) 2022/2065, includes recommender-system transparency obligations for covered online platforms in Article 27, subject to its scope and conditions.
Its applicability depends on the service, jurisdiction, exemptions, and facts.
Do not treat these examples as a complete list of applicable rules.

Avoid using sensitive characteristics or proxies without a justified and reviewed basis.
An explicit food restriction may be relevant to customer safety, but collection and retention can create privacy risks.
Use the minimum fields necessary for the approved task, protect access, and document the retention policy.
Customers' preferences, opt-outs, and deletion requests must flow into both profile construction and downstream caches as applicable.
A pseudonymous key is not a substitute for a privacy assessment.

### 8.4 Policy and supply obligations

Product safety, consumer protection, advertising, and sector-specific rules vary by jurisdiction and product.
The business must determine which requirements apply to its service.
For example, a paid placement should be handled under the organization's applicable disclosure policy rather than disguised as an organic relevance signal.
A supplier agreement or internal assortment rule may be binding contractually even when it is not a statute.
Label internal conventions and external legal requirements separately.

A delivery estimate is not a firm customer promise unless the relevant process and wording make it one.
A recommendation screen should not assert availability more strongly than the inventory service can support.
Record which operational service owns the promise and how stale or uncertain stock is communicated.
When an offer is excluded, retain a reason code suitable for audit without retaining unnecessary personal information.

## 9. Industry practice and case evidence

### 9.1 Amazon item-to-item collaborative filtering

Linden, Smith, and York describe Amazon.com's item-to-item collaborative filtering in an IEEE Internet Computing article published in 2003.
The paper describes recommending items similar to items in a customer's history using item relationships derived from customer behavior.
It is not a claim that a purely catalog-content baseline powered the complete Amazon recommendation system.
The useful lesson for this topic is that item-to-item retrieval can be operationally attractive, but the origin of the similarity signal matters.
Behavioral co-occurrence and attribute cosine answer different questions and have different cold-start and bias properties.

**Inference:** A team evaluating a modern item-to-item model should compare against both an attribute-based neighbor list and an appropriate popularity list under the same candidate rules.
The cited paper does not establish a current performance lift for the illustrative grocer.

### 9.2 Netflix evaluation and experimentation

Gomez-Uribe and Hunt describe the Netflix recommender system and its business value in a 2015 ACM Transactions on Management Information Systems article.
They discuss personalization, the product context, and the role of experimentation.
Their setting involves media choices and engagement, not perishable grocery stock.
Do not transfer a reported media result directly to physical inventory or assume the same cost structure.

**Inference:** Multiple relevant baselines and online experiments are useful because offline agreement with historical choices cannot by itself establish business value.
The paper does not validate this topic's numerical example.

### 9.3 Exposure bias in logged interactions

Schnabel and colleagues' 2016 work on recommendations as treatments explains the selection problem created by which items users are exposed to.
A historical click or purchase is observed after a prior policy influenced exposure.
The absence of a click on an item never shown to a customer is not equivalent to an observed rejection.
Propensity-based correction needs appropriate logged probabilities and overlap; it is not a universal repair for missing support.

**Inference:** A popularity baseline should log exposure and share its candidate and eligibility rules with challengers.
Even then, a randomized policy comparison is often needed for an incremental launch decision.

### 9.4 Inventory and local context

The public sources above do not establish a universal stock buffer or a required location hierarchy.
A buffer is a local risk policy that depends on inventory accuracy, request volume, reservation behavior, and customer harm.
Test it with operations and monitor missed sales as well as oversell.
A conservative buffer may protect service while excluding otherwise deliverable stock.
A small buffer may increase apparent coverage while increasing disappointment.

The illustrative grocer's threshold is an assumption, not an industry standard.
The same applies to its seven-day decay and twenty-order support threshold.
Choose those values using the business's history and validate them before freezing a test.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Global list ignores local assortment | Frequent unavailable clicks in one store | National purchases mask node-specific supply | Lost sales and customer frustration | Apply local eligibility and report by node |
| Inventory snapshot is stale | Large update lag or sudden cancellations | Ranking uses obsolete availability | False availability promise | Set freshness threshold and approved stale-data branch |
| Buffer uses wrong units | Threshold differs across pack sizes | Cases and eaches are mixed | Excess exclusions or oversell | Normalize units and validate master data |
| Popularity reinforces exposure | Same items dominate despite assortment growth | Historical placement creates more recorded events | Entrenched concentration | Log exposure, monitor distribution, test exploration |
| Popularity counts retries | Event volume jumps after a client release | Duplicate events inflate scores | Artificial ranking changes | Deduplicate by stable event identity |
| Segment is too small | Rank flips after a handful of orders | High sampling variability | Unstable customer experience | Prespecify support ladder and smoothing |
| Segment leaks future information | Offline scores collapse online | Segment uses later order or corrected field | Inflated offline metrics | Join features as known at request time |
| Content fields are sparse | Many zero vectors or identical low scores | Attributes are missing or unstandardized | Arbitrary ranking and poor cold start | Monitor completeness and use explicit fallback |
| Similarity bypasses hard rules | Close matches include incompatible items | Soft score substitutes for eligibility | Safety, returns, or compliance harm | Enforce verified gates before ranking |
| Score scales are mixed | Hybrid ordering changes after feature refresh | Popularity and similarity have unrelated units | Uncontrolled weighting | Define and validate transformations as a distinct policy |
| Old and new models see different candidates | Challenger coverage is much larger | Different inventory or catalog inputs | Unfair claimed model lift | Hold input and policy rules constant |
| Offline test treats unshown as negative | Very high precision on logged items only | Historical exposure determines labels | Misleading model selection | Disclose selection; use randomized exposure |
| List-only denominator | High CTR but more empty requests | Unserved requests disappear from analysis | Coverage loss hidden | Report assigned-request metrics and coverage |
| Post-click comparison is called causal | Clickers differ sharply across variants | Treatment changes who enters the denominator | Selection bias | Analyze assigned units for causal metrics |
| Finance counts gross sales as profit | High revenue accompanies costly fulfillment | Variable costs and returns omitted | Negative incremental contribution | Agree and reconcile contribution ledger |
| Ties have no stable rule | Lists change between identical requests | Nondeterministic iteration order | Inconsistent exposure and hard debugging | Version deterministic tie-breaks |
| Stock gate is mistaken for reservation | Items sell out between display and checkout | Concurrent demand depletes stock | Customer sees an unavailable offer | Use promise-aware service and monitor lag |
| Fallback bypasses eligibility | More invalid items during outages | Global fallback ignores channel rules | Safety and service regression | Apply same hard gate to every fallback |
| Experiment is stopped opportunistically | Decision changes with daily fluctuations | Repeated unplanned looks inflate false positives | Spurious launch | Prespecify monitoring and stopping design |
| Profile survives an opt-out | User continues receiving history-based lists | Cache and downstream deletion are not coordinated | Privacy and trust harm | Propagate changes and audit retention paths |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or monitor |
|---|---|---|---|
| Catalog IDs identify purchasable variants | Joins connect events, features, and stock | Parent/variant mapping changes | Reconcile catalog and order-line identifiers |
| Policy fields are authoritative | Mandatory compatibility gates need trustworthy data | Attribute is missing, stale, or incorrectly sourced | Sample against source documents and track unknowns |
| Inventory snapshots represent the correct node | Stock should match the promised channel | Routing changes after ranking | Compare selected node with fulfilled node |
| Inventory is fresh enough | Availability gate needs a current estimate | Integration outage or late updates | Monitor age distribution and stale-branch counts |
| Buffer reflects risk tolerance | Threshold balances availability and oversell | Demand or accuracy shifts | Compare post-click stockouts and excluded stock over time |
| Event timestamps and ingestion times are reliable | Replay must use only known information | Clock skew or missing arrival timestamps | Audit event ordering and late-arrival distribution |
| Deduplication is correct | Counts should represent real actions | Retry identifiers change or collide | Reconcile event counts with order ledger |
| Recent history contains relevant signal | Popularity uses past choices to rank future choices | Promotion, season, or assortment changes | Compare time windows and monitor rank drift |
| Parent and child shares cover the same universe | Smoothing is interpretable as a distribution | Different filters or item sets | Check denominator and probability sum |
| Content features encode task relevance | Similarity must track meaningful product properties | Missing attributes or task requires complements | Audit nearest neighbors with task experts |
| Offline labels are useful diagnostics | Observed actions contain some relevance signal | Prior policy has narrow or biased exposure | Inspect exposure support and compare online results |
| Randomization is implemented as planned | Treatment contrast identifies an incremental effect | Assignment bugs or cross-arm interference | Audit assignment, balance, and contamination |
| Contribution costs are complete enough | Economic outcome should reflect variable economics | Returns or fulfillment costs arrive late | Reconcile mature cohorts to finance ledger |
| Results generalize to the launch population | A pilot is used for a broader decision | Different regions, season, or supply conditions | Report segment effects and stage rollout |
| Stable fallback is acceptable | Missing scores still need a defined action | Fallback violates task or policy constraints | Measure fallback usage and audit returned lists |

These assumptions are not interchangeable.
A correct cosine calculation cannot compensate for an incorrect inventory join.
A randomized test cannot make a prohibited product eligible.
A good average effect does not establish that every store or customer group benefits.
A post-test investigation of many subgroups is exploratory unless appropriately planned and analyzed.

## 12. Exercises

### Exercise 1 — Write the decision contract

**Time:** 15 minutes.
**Scenario:** A retailer wants a two-slot carousel of alternatives when a customer views an out-of-stock coffee item.
The customer has selected a store for pickup.

**Deliverable:** A concise contract naming task, decision time, analysis unit, required inputs, hard exclusions, fallback, owners, primary metric, and two guardrails.
**Acceptance criteria:** The store and inventory freshness appear; the seed is handled; empty lists are permitted; the primary metric includes all eligible assigned units.
**Outcome assessed:** 1.

### Exercise 2 — Calculate popularity

**Time:** 10 minutes.
**Scenario:** Item X has 3 orders aged zero days and 4 orders aged seven days.
Item Y has 2 orders aged zero days and 8 orders aged 14 days.
All events are known, eligible, and within the window.
Use a seven-day half-life and a unit weight per order.

**Deliverable:** Both weighted-event scores and the ordering with a stable identifier tie-break.
**Acceptance criteria:** Show the weights for seven and 14 days; distinguish weighted events from purchase probability.
**Outcome assessed:** 2.

### Exercise 3 — Route a sparse segment

**Time:** 15 minutes.
**Scenario:** The ladder is store-category, region-category, then national-category.
A level requires at least 20 distinct orders over 28 days.
Store S has 12 orders; its region has 85; the national category has 900.
All counts use the same event definition and are available at request time.

**Deliverable:** Select a level and explain the fallback for an unknown store.
**Acceptance criteria:** Do not treat 12 orders as 20; do not infer an unknown store; retain location-specific eligibility when available.
**Outcome assessed:** 3.

### Exercise 4 — Calculate similarity and apply a rule

**Time:** 15 minutes.
**Scenario:** A seed has four binary tokens.
Candidate P has four tokens and shares three; candidate Q has three tokens and shares two.
Both are in stock, but P violates a mandatory dietary restriction.

**Deliverable:** Calculate both cosine scores and choose the top eligible result.
**Acceptance criteria:** Use vector norms; exclude P despite its higher score; state that a missing mandatory field needs an approved policy.
**Outcome assessed:** 4.

### Exercise 5 — Inventory and freshness

**Time:** 15 minutes.
**Scenario:** One unit is requested, with a buffer of two units.
Freshness must be no older than five minutes.
A has ATP 3 units and age 2 minutes; B has ATP 4 units and age 8 minutes; C has ATP 2 units and age 1 minute.
The policy says stale stock must be withheld.

**Deliverable:** List eligible items and reasons for exclusions.
**Acceptance criteria:** Apply both the three-unit threshold and freshness rule; do not call the buffer a reservation.
**Outcome assessed:** 5.

### Exercise 6 — Interpret offline and online comparisons

**Time:** 20 minutes.
**Scenario:** A challenger has higher offline hit@10 than popularity on historical purchases.
It also returns a list on 80% of eligible requests versus 99% for popularity.
In a randomized test, its contribution is USD 1.20 per assigned eligible session versus USD 1.10 for popularity; no uncertainty interval is reported.

**Deliverable:** Write three justified conclusions and two missing pieces needed for a launch decision.
**Acceptance criteria:** Address logged exposure bias, coverage denominator, causal randomization conditions, and uncertainty; do not claim a proven lift from point estimates alone.
**Outcome assessed:** 6.

### Exercise 7 — Design an experiment scorecard

**Time:** 25 minutes.
**Scenario:** A proposed local popularity policy may improve selection but concentrate demand on a few stores' fast-moving items.

**Deliverable:** Define an estimand, randomization unit, contribution metric, three guardrails, monitoring cadence, stop authority, and a plan for delayed returns.
**Acceptance criteria:** Define denominators and windows; address repeated customers and shared inventory; distinguish an assignment-based primary metric from post-click diagnostics.
**Outcome assessed:** 7.

### Exercise 8 — Audit a content baseline

**Time:** 25 minutes.
**Scenario:** A product team uses cosine similarity on title text and fills empty results with global best sellers.
The list ignores delivery location and has no restriction fields.

**Deliverable:** Produce a prioritized five-item remediation plan and name an owner for each item.
**Acceptance criteria:** Include task compatibility, inventory, feature completeness, fallback eligibility, and matched evaluation; do not propose a model upgrade as a replacement for those controls.
**Outcome assessed:** 4, 5, 6, 7.

## 13. Self-check answers

### Exercise 1

A strong contract defines substitution at the time the customer opens the out-of-stock item view.
The store selected for pickup determines the fulfillment node, subject to routing rules.
Eligible alternatives must be active, compatible, permitted, sufficiently stocked, and fresh under the approved inventory contract.
The unavailable seed is excluded unless the task explicitly supports a different variant.
If no eligible item remains, show no product and offer an approved non-product action.

A suitable primary outcome is contribution per assigned eligible session over a defined outcome window.
Stockout after selection and pickup promise misses can be guardrails.
The product owner owns the decision; operations owns fulfillment rules; merchandising and policy owners approve compatibility.
The data owner documents input quality and finance approves the contribution definition.
Other sensible contracts receive credit if they define equivalent boundaries and ownership.

### Exercise 2

The seven-day weight is `2 ** -1 = 0.5`.
The 14-day weight is `2 ** -2 = 0.25`.
X scores `3 × 1 + 4 × 0.5 = 5` weighted events.
Y scores `2 × 1 + 8 × 0.25 = 4` weighted events.
X ranks before Y; the tie-break is not needed for these values.
Neither score is a probability or an estimate of incremental contribution.

### Exercise 3

The store level has only 12 distinct orders and fails the 20-order threshold.
The region has 85 and is the first eligible level in the ladder.
Use the regional ranking after applying the request's store-level eligibility and stock gate.
If the store is unknown, do not invent one or claim store-level stock is known.
Route to an approved non-location-specific experience or withhold location-dependent offers until a location is available.

### Exercise 4

P has `3 / sqrt(4 × 4) = 0.75` similarity.
Q has `2 / sqrt(4 × 3) ≈ 0.577` similarity.
P is excluded because it violates the mandatory restriction.
Q is the top eligible result if it meets every other rule.
Unknown is not automatically compliant; follow the approved unknown-field branch.

### Exercise 5

The quantity threshold is `1 + 2 = 3` units.
A passes both quantity and freshness.
B has sufficient quantity but is too stale and must be withheld under the stated rule.
C is fresh but fails the quantity threshold.
Only A is eligible, and no stock has been reserved merely by recommending it.

### Exercise 6

The challenger has a higher observed offline hit rate on the specified logged test set.
That fact does not prove it would cause higher contribution, because observed purchases depend on past exposure.
The challenger serves fewer eligible requests, so the missing 20% must remain visible in coverage and assignment-based business analysis.
The randomized point-estimate difference is USD 0.10 per assigned session, provided assignment and measurement are valid.
Without uncertainty, design details, mature outcomes, and guardrail results, a launch claim is not supported.

### Exercise 7 rubric

A strong answer specifies the eligible population, assignment unit, outcome window, and contribution components.
It accounts for repeated sessions and considers whether shared stock makes customer-level interference material.
It names service, coverage, and concentration guardrails with denominators and thresholds.
It gives the supply owner a documented stop path and explains mature versus immature return data.
A weak answer uses clicks alone, conditions the primary metric on clicking, or assigns responsibility to nobody.

### Exercise 8 rubric

First establish mandatory task and safety rules with the merchandising or policy owner.
Second add location-specific, freshness-controlled inventory gating with operations.
Third improve and audit content attributes with the catalog owner.
Fourth ensure all fallback lists use the same eligibility gate with the product and engineering owners.
Fifth compare policies on matched requests and run an appropriately designed experiment with analytics and finance.
An equally defensible ordering earns credit if it prevents unsafe or unavailable offers before optimization.

## 14. Interview questions

1. **Why is a global best-seller list not necessarily a weak baseline?**
   A strong answer specifies a relevant window, task, eligibility rules, and fair comparison.
   A weak answer assumes complexity guarantees improvement.
2. **What does a recency-weighted popularity score measure?**
   A strong answer explains weighted historic events and the effect of past exposure.
   A weak answer calls it a calibrated probability.
3. **When should a segment fall back to a broader group?**
   A strong answer uses a prespecified support rule and preserves local eligibility.
   A weak answer treats a handful of events as a stable local preference.
4. **What does smoothing fix, and what does it not fix?**
   A strong answer distinguishes variance reduction from exposure bias and mismatched universes.
   A weak answer claims pseudo-counts create unbiased observations.
5. **How do you handle an item with no usable content features?**
   A strong answer names an undefined cosine, a permitted fallback, and completeness monitoring.
   A weak answer divides by zero or invents a similarity.
6. **Why can a high-similarity item still be a bad substitute?**
   A strong answer separates mandatory compatibility, stock, and safety from similarity.
   A weak answer assumes nearest neighbors are automatically interchangeable.
7. **What is the difference between stock gating and reservation?**
   A strong answer describes a time-sensitive estimate versus an allocation transaction.
   A weak answer promises fulfillment solely because an item passed a threshold.
8. **What makes an offline comparison fair?**
   A strong answer includes shared requests, point-in-time inputs, eligible candidates, coverage, and frozen tuning.
   A weak answer compares different populations or excludes failures.
9. **Why is historical hit@k not an incremental revenue estimate?**
   A strong answer explains logged exposure and the need for causal identification.
   A weak answer treats unshown products as rejected.
10. **How would you choose an online primary metric?**
    A strong answer defines contribution, denominator, window, assignment, and uncertainty.
    A weak answer reports attributed sales without cost or a control group.
11. **What happens when the inventory feed is stale?**
    A strong answer states the agreed threshold, safe branch, owner, and monitoring.
    A weak answer silently uses stale stock or removes all constraints.
12. **How can privacy change the fallback design?**
    A strong answer discusses permitted data use, opt-outs, retention, and non-profiled alternatives.
    A weak answer assumes pseudonymization makes all processing unrestricted.

## 15. Key terms

Each term links to the definition table in [section 4](#4-definitions-and-vocabulary).

- [Available-to-promise](#4-definitions-and-vocabulary): stock quantity available under the service's allocation rules.
- [Backoff](#4-definitions-and-vocabulary): move to a broader rule when support is insufficient.
- [Candidate](#4-definitions-and-vocabulary): item considered for a recommendation.
- [Cold start](#4-definitions-and-vocabulary): inadequate interaction history for an item or user.
- [Constraint](#4-definitions-and-vocabulary): mandatory rule or explicitly soft preference.
- [Content feature](#4-definitions-and-vocabulary): encoded item attribute.
- [Contribution](#4-definitions-and-vocabulary): revenue less specified variable costs.
- [Conversion](#4-definitions-and-vocabulary): specified action over a defined denominator.
- [Cosine similarity](#4-definitions-and-vocabulary): normalized vector dot product.
- [Coverage](#4-definitions-and-vocabulary): share of a specified universe served or represented.
- [Decay half-life](#4-definitions-and-vocabulary): time required to halve an event's weight.
- [Eligibility](#4-definitions-and-vocabulary): required conditions for consideration.
- [Event](#4-definitions-and-vocabulary): timestamped recorded action.
- [Exposure](#4-definitions-and-vocabulary): logged opportunity to encounter an item.
- [Freshness](#4-definitions-and-vocabulary): age of a valid data snapshot.
- [Global popularity](#4-definitions-and-vocabulary): aggregated behavior across a stated population.
- [Guardrail](#4-definitions-and-vocabulary): condition limiting unacceptable effects.
- [Impression](#4-definitions-and-vocabulary): defined display event.
- [Incrementality](#4-definitions-and-vocabulary): outcome difference caused by a policy.
- [Leakage](#4-definitions-and-vocabulary): unavailable future information used in a decision.
- [NDCG@k](#4-definitions-and-vocabulary): normalized rank-discounted gain at a cutoff.
- [Positivity](#4-definitions-and-vocabulary): sufficient action probability for the relevant comparison.
- [Profile](#4-definitions-and-vocabulary): representation of a customer's permitted history.
- [Ranking](#4-definitions-and-vocabulary): ordering candidates by a rule.
- [Recall@k](#4-definitions-and-vocabulary): share of eligible relevant items retrieved in the top k.
- [Request](#4-definitions-and-vocabulary): one opportunity to serve a recommendation.
- [Seed item](#4-definitions-and-vocabulary): reference product for item similarity.
- [Segment](#4-definitions-and-vocabulary): group defined from available context.
- [Smoothing](#4-definitions-and-vocabulary): combining an estimate with prior information.
- [Stock buffer](#4-definitions-and-vocabulary): withheld quantity used to manage risk.
- [TF-IDF](#4-definitions-and-vocabulary): text representation using term and inverse document frequency.
- [Zero vector](#4-definitions-and-vocabulary): representation with no active features.

## 16. Sources

The following sources support the concepts or describe the stated cases; they do not validate the illustrative dataset or its numeric results.
Access and legal applicability should be checked for the relevant deployment date and jurisdiction.

1. Linden, G., Smith, B., and York, J. (2003). “Amazon.com Recommendations: Item-to-Item Collaborative Filtering.” *IEEE Internet Computing*, 7(1), 76–80. [DOI: 10.1109/MIC.2003.1167344](https://doi.org/10.1109/MIC.2003.1167344). Peer-reviewed description of an item-to-item behavioral approach.
2. Gomez-Uribe, C. A., and Hunt, N. (2015). “The Netflix Recommender System: Algorithms, Business Value, and Innovation.” *ACM Transactions on Management Information Systems*, 6(4), Article 13. [DOI: 10.1145/2843948](https://doi.org/10.1145/2843948). Company-authored account of a media recommendation system and experimentation.
3. Schnabel, T., Swaminathan, A., Singh, A., Chandak, N., and Joachims, T. (2016). “Recommendations as Treatments: Debiasing Learning and Evaluation.” *Proceedings of the 33rd International Conference on Machine Learning*. [PMLR 48:1670–1679](https://proceedings.mlr.press/v48/schnabel16.html). Research on selection bias and propensity-based methods.
4. Järvelin, K., and Kekäläinen, J. (2002). “Cumulated Gain-Based Evaluation of IR Techniques.” *ACM Transactions on Information Systems*, 20(4), 422–446. [DOI: 10.1145/582415.582418](https://doi.org/10.1145/582415.582418). Source for discounted gain measures.
5. GS1. [GTIN Management Standard](https://www.gs1.org/standards/gtin-management). Official guidance on identifying trade items; does not prescribe an inventory or recommender policy.
6. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5, 6, and 22. Binding text subject to scope, interpretation, and later applicable law.
7. European Union. [Regulation (EU) 2022/2065, Digital Services Act](https://eur-lex.europa.eu/eli/reg/2022/2065/oj), especially Article 27. Binding text subject to scope and applicable conditions.
8. scikit-learn. [Cosine similarity documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html). Official implementation reference for normalized dot-product similarity.

## 17. Links

- [Module 24 topic index](_index.md).
- [Module 24 survey and lab sections](../../../course/24_recommendation_systems_personalization.md).
- [Recommendation tasks](01_recommendation-tasks.md) — sibling topic, listed as planned in the module index.
- [Offline and online evaluation](07_offline-online-evaluation.md) — sibling topic, listed as planned in the module index.
- [Business constraints and re-ranking](09_constraints-re-ranking.md) — sibling topic, listed as planned in the module index.
- [Inventory-aware substitution](10_inventory-aware-substitution.md) — sibling topic, listed as planned in the module index.

Sibling topic links may not resolve until those planned files are published; the module index is the source of their status.
