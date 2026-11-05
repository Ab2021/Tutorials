# Cold Start: New Users, New Items, and the Fallback Ladder

> Module 24 — Recommendation Systems and Personalization  
> Topic 06 of 11  
> Estimated study time: 4 hours  
> Prerequisites: Topics 01 and 02; Topic 05 is helpful  
> Version 2026.5

## 1. Why this topic exists

A customer opens a commerce app for the first time.
The service must decide what to show before it has an interaction history.
A supplier simultaneously lists a new product.
The service must decide whether that item deserves exposure before it has clicks or sales.
These are related but distinct cold-start problems.
A new user has little reliable preference evidence; a new item has little reliable response evidence.
A new market or surface can have both problems and no trustworthy local popularity history.

The immediate decision is not which sophisticated model to buy.
It is which eligible items to show, to whom, under what privacy permission, and when to replace a provisional policy with learned evidence.
A merchandising manager needs a defensible way to give new suppliers an opportunity without displacing useful products arbitrarily.
A supply planner needs to prevent a launch from advertising stock that cannot be delivered.
A privacy owner needs to know whether a signal can be collected or reused.
A product owner needs an accountable default when the model, identity service, inventory feed, or consent service fails.

The cost of a bad decision is asymmetric.
An irrelevant tile consumes attention and may reduce trust.
An unavailable substitute may cause an abandoned basket, a cancelled order, or an unnecessary service contact.
An incompatible spare part may stop equipment, not merely lose a click.
Uncontrolled exploration can expose a restricted product, transfer demand to an unable supplier, or make an experiment impossible to interpret.
A no-personalization choice must not silently become a different kind of tracking.

Cold start is not solved by waiting for enough clicks.
Exposure determines which clicks can exist, and the initial allocation of exposure is itself a policy choice.
A product with zero clicks after zero impressions has not failed a customer test.
A product with zero clicks after 1,000 relevant impressions has evidence, although position and audience still matter.
The evidence threshold is therefore conditional on opportunity, data quality, and the cost of being wrong.

This lesson builds a fallback ladder with eligibility gates, context-dependent candidates, bounded exploration, and explicit handoffs.
It treats forecasts, plans, targets, and commitments as different objects:
- A **forecast** predicts future outcomes under stated assumptions.
- A **plan** specifies the intended exposure allocation and operational response.
- A **target** is a desired metric, not evidence that it will be met.
- A **commitment** is an approved obligation, such as inventory reserved for a launch.

The worked example is synthetic; it is not evidence of a real company's results.
You should leave able to make an interim decision and specify what would change it.

## 2. Learning outcomes

After this lesson you can:

1. **Classify** a request as user-cold, item-cold, both-cold, or sufficiently observed, and distinguish missing data from prohibited data. Exercise 1 checks this classification.
2. **Construct** a fallback ladder whose rungs use only permitted information and whose hard gates apply at every rung. Exercise 2 checks ordering and failure behavior.
3. **Calculate** an exploration allocation and compare incremental contribution with operational and risk costs. Exercises 3 and 4 check the arithmetic.
4. **Explain** why observed clicks on exposed items are selected evidence, and specify the logging needed for valid comparison. Exercise 5 checks this explanation.
5. **Design** an item launch and evidence-based graduation or rollback rule with a named decision owner. Exercise 6 checks operational completeness.
6. **Choose** a privacy-respecting response for anonymous, consented, revoked, and unknown-permission requests without treating consent as universally the only lawful basis. Exercise 7 checks the choices.
7. **Critique** a cold-start rollout across customer, supplier, inventory, and fairness dimensions. Exercise 8 checks the trade-offs.

The exercises require calculations and decision records, not a particular machine-learning library.
A correct answer can reject personalization when data, permission, or operational readiness is insufficient.

## 3. The core mental model

Treat each recommendation request as a sequence of decisions, not as a single score.

```text
Request + surface + delivery context
              |
              v
   Permission and purpose check -------> restricted / unknown:
              |                         approved nonpersonal fallback
              v
   Catalogue + delivery hard gates ----> empty / uncertain:
              |                         safe empty state or verified fallback
              v
   Select strongest permitted evidence:
   session -> explicit preference -> cohort/context -> local popularity
              |                            -> curated catalogue
              v
   Blend bounded eligible new-item exploration
              |
              v
   Rank, validate again, render and log policy + probability
              |
              v
   Observe outcomes and delayed operational costs
              |
              v
   Review thresholds, graduate, continue, pause, or retire
```

The arrow from permission applies to every downstream use, including logging and model training.
A valid session signal is not automatically a valid long-term profile.
A safe nonpersonal fallback is an independently approved policy, not an invitation to infer an identity from a device.

The two directions of uncertainty differ.
For a user, ask: what evidence indicates the person's intent, and may it be used for this purpose?
For an item, ask: what is known about its attributes, availability, and response in the relevant audience?
For a new-user/new-item pair, both uncertainties are high.
It may still be reasonable to show the item if attributes support relevance, inventory is sound, and an approved exploration policy provides a measurable opportunity.
It is not reasonable to pretend that an unknown affinity score is a measured zero.

A fallback ladder expresses degradation, not a strict hierarchy of intrinsic quality.
A broad popularity list can outperform a noisy one-click session model.
The rungs are candidate policies with explicit readiness tests, and experiments determine which should win in a particular surface.
A contextual bestseller is not necessarily nonpersonal: a sufficiently granular location, persistent identifier, or sensitive context may be personal data.
The privacy owner must define allowed fields and retention for each surface.

Availability and policy constraints are gates rather than optional weights.
A high predicted click probability does not make a recalled item sellable.
A permitted item that has no reliable delivery promise is not equivalent to a verified in-stock item.
Depending on the business and surface, uncertainty in inventory can mean suppressing the item or showing an explicit availability disclaimer.
The policy must state which choice applies and who approves it.

Explore only among candidates that passed those gates.
Exploration is a deliberate allocation of limited display opportunities to learn; it is not an excuse to show any new item to anyone.
The policy's randomization and logging must support the measurement question.
As evidence accumulates, the system can promote an item, retain it in a measured exploration pool, or stop showing it.
A human may override these decisions for safety or contractual reasons, but the override must be logged separately from the model.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| User cold start | Insufficient permitted user or session evidence for a particular decision | Assuming every new account has no intent signal |
| Item cold start | Insufficient relevant, attributed exposure and outcome evidence for an item | Calling an item bad because nobody was shown it |
| System cold start | A new market, surface, or catalogue lacks reliable local history | Reusing another market's data without checking transfer |
| Sparse history | Some observations exist, but not enough for a stable decision | Treating a fixed event count as universal proof |
| Eligibility | Boolean result of legal, safety, stock, delivery, and commercial rules for a request | Treating eligibility as merely another ranking feature |
| Candidate | An item eligible for consideration before final ordering | Equating retrieval with an impression |
| Impression | A defined opportunity to see a rendered item, counted in impressions | Counting an API response the user never saw |
| Click-through rate (CTR) | Clicks divided by defined impressions, usually a percentage | Treating CTR as contribution or causation |
| Conversion | A defined downstream action, such as a paid order, per eligible opportunity | Mixing order, unit, and customer denominators |
| Contribution | Revenue less specified variable costs, in currency per order or impression | Calling gross revenue profit |
| Exploration | Deliberate exposure allocation to reduce decision uncertainty | Unlogged random shuffling |
| Exploitation | Selecting using the current estimated objective | Assuming the current estimate is unbiased |
| Propensity | Probability that the logging policy chose an action in a given context | Logging only the eventual score as though it were a probability |
| Positivity | Each action evaluated has nonzero probability under the logging policy in relevant contexts | Claiming an offline estimate for never-exposed items |
| Position bias | Outcome differences caused by placement and attention, not only item relevance | Comparing top and bottom slots as if exchangeable |
| Exposure bias | Observed responses are conditional on what the serving policy chose to show | Treating the observed interaction table as all preferences |
| Fallback ladder | Ordered set of authorized policies with readiness and failure rules | A single popularity list without defined gates |
| Context | Request-time information such as surface, delivery area, or time | Assuming all contextual data is anonymous |
| Cohort | A group defined by approved shared attributes | Assuming a small cohort cannot identify someone |
| Explicit preference | A user-provided choice such as selected product category | Treating a one-time answer as indefinite permission |
| Session signal | An event within the current bounded interaction | Automatically retaining it in a persistent profile |
| Content feature | Item attribute, text, or image used to assess similarity | Trusting unvalidated supplier attributes |
| Prior | Initial assumption or information used before local response data accumulates | Treating the prior as observed local demand |
| Bayesian smoothing | Combining observations with a specified prior to stabilize a rate | Hiding a discriminatory or stale prior behind a formula |
| Graduating | Moving an item from provisional exploration to ordinary ranking under a stated rule | Promoting solely after a fixed calendar age |
| Guardrail | A monitored constraint or stop condition, with denominator and owner | A dashboard metric without a decision threshold |
| Consent | A specific permission mechanism where applicable under law and policy | Assuming consent is always required or always sufficient |
| Purpose limitation | Restricting processing to specified purposes and compatible uses where applicable | Reusing data for unrelated targeting because it is available |
| Retention | The period and mechanism for keeping data, including deletion or de-identification | Leaving raw events indefinitely in debug logs |
| Backfill | A lower-rung policy supplying items when a higher rung cannot fill the slate | Filling with ineligible items to reach a fixed count |
| Slate | Ordered set of items shown in one request | Treating each slot as independent when selections interact |
| Incrementality | Difference between outcomes under a treatment and a counterfactual | Calling total observed sales incremental |
| Minimum detectable effect | Smallest effect a particular experiment is designed to detect at stated error rates and sample size | Treating a nonsignificant result as proof of no effect |

## 5. The mechanics, step by step

### 5.1 Define the decision and its owner

1. Name the surface, such as a home-page product carousel, a replenishment page, or an out-of-stock substitution prompt.
2. State the action: select and order up to `k` products for one eligible request.
3. Define the business objective and the unit of analysis before looking at results.
4. Assign a product decision owner and separately name owners for inventory, privacy, safety, analytics, and supplier operations.
5. State the decision horizon: a session, a 28-day trial, or a seasonal launch have different costs and learning windows.
6. Define the counterfactual: the current approved fallback policy is often a better comparator than an empty page.
7. Specify what the team will do if evidence is inconclusive.

Input: surface specification, user journey, catalogue, operating constraints, and the current baseline.
Output: a versioned decision brief and an owner authorized to stop or change the policy.
Branch: if showing a wrong item creates safety or compatibility risk, require stronger deterministic validation and narrower exploration.
For a regulated or high-stakes surface, legal and domain owners must approve the design before launch.

An example objective is incremental contribution per eligible session over 28 days, with cancellation and complaint guardrails.
A click can be a diagnostic but should not silently replace the objective.
A supplier exposure target is a target, not a claim that demand exists or a contract to buy inventory.
If the business commits display volume, distinguish that obligation from the learning policy and disclose it in measurement.

### 5.2 Separate absence, uncertainty, and prohibition

1. Determine whether an identifier exists, whether a reliable join exists, and whether the relevant feature is present.
2. Independently determine whether using that feature for this decision and purpose is permitted.
3. Record missing, stale, invalid, and prohibited as different states.
4. Do not impute a prohibited feature from a proxy merely to make the model work.
5. On an unknown consent or authorization state, use the organization's approved fail-closed policy for optional processing.
6. Do not write sensitive raw identifiers to the fallback log to make debugging easier.

Input: consent and policy decision, source metadata, and feature availability.
Output: a list of permitted features with freshness and purpose restrictions.
Branch: if a visitor declines optional personalization, use only the separately approved nonpersonal experience and permitted operational data.
If a visitor revokes permission, stop future processing as required and route historical records through the applicable deletion or restriction workflow.
The applicable obligations depend on jurisdiction, purpose, lawful basis, and exceptions; privacy counsel makes that determination.

A profile may be absent because a person is genuinely new, has chosen not to identify, or has asked for data to be deleted.
Those cases can look identical to a scoring service, but their permissible follow-up differs.
Do not ask the recommendation service to reverse-engineer the reason through device fingerprinting.

### 5.3 Establish hard eligibility before relevance

1. Verify the item is active and approved for the selling surface and destination.
2. Check applicable age, safety, recall, and category restrictions.
3. Check product compatibility when the use case requires it.
4. Check current price, order minimum, available-to-promise quantity, and delivery serviceability.
5. Exclude products whose supplier or fulfilment capability is suspended.
6. Apply contractual or commercial restrictions approved for this use case.
7. Capture the version and freshness of the feeds that support each decision.
8. Repeat volatile checks immediately before render or purchase according to the service-level design.

Input: catalogue, policy rules, inventory and fulfilment state, destination, and time.
Output: the eligible candidate pool plus reason codes for exclusion.
Branch: stale or unavailable eligibility inputs invoke a documented fail-safe, rather than being interpreted as unlimited stock.
If no item passes, return a safe empty state, alternative navigation, or a separately verified list.
Do not relax a hard restriction to meet a slate-length target.

An available-to-promise figure is a snapshot, not a reservation.
Concurrent purchases can still consume stock after a tile is rendered.
If a launch requires reserved stock, obtain an explicit commitment and expiry from the inventory owner.
Keep displayed price and delivery promise consistent with the checkout contract.

### 5.4 Choose the strongest permitted user evidence

A possible ladder follows; its order is a hypothesis to validate, not a universal law.
All rungs receive the same eligible pool and policy checks.

| Rung | Trigger and candidate basis | Back off when | Required distinction |
|---|---|---|---|
| 1. Current intent | A sufficient, permitted current-session action indicates category or compatibility | Signal is ambiguous, stale, disallowed, or retrieval is empty | Session use does not imply permission to retain a profile |
| 2. Explicit choice | A visitor voluntarily selects a need, category, or delivery requirement | Choice is missing, contradictory, or withdrawn | A preference is not automatically marketing consent |
| 3. Approved context or cohort | Surface, broad location, or approved group has enough reliable coverage | Group is too small, unrepresentative, or not authorized | A cohort may still be personal data |
| 4. Local popularity | Recent demand within an approved market and surface has sufficient exposure and coverage | Market is new, data is stale, or popularity is dominated by stockouts | Sales reflect prior exposure and availability |
| 5. Curated safe default | Merchandising supplies a diversified, verified list for the surface | List is stale, unavailable, or empty after filtering | Curation requires accountable refresh and audit |
| 6. Safe empty state | No recommendation can be supported safely | A verified option becomes available | Empty is preferable to unsafe backfill |

A known user with prohibited profiling data follows the permitted ladder, not a special identity-recovery path.
A new account with a reliable current search can be less cold than a long-standing account with stale history.
A session click should be interpreted according to its context: a click on a return policy is not necessarily a product preference.
Use freshness windows and contradictory-action rules documented by the product team.

Each rung needs an entry criterion, candidate source, quality threshold, timeout, and owner.
For example, a session-based rung could require an explicit product-category selection within the last 20 minutes of the current session.
That interval is a design assumption, not a legal rule or an empirically validated universal threshold.
When a higher rung returns only two eligible candidates for a four-slot slate, fill the remaining slots from lower rungs after deduplication and the same eligibility checks.
Log which rung supplied each displayed item.
Avoid delaying the page indefinitely for a personalized response: a timeout should resolve to the approved lower rung.

### 5.5 Create an item launch record

1. Receive a new item with a stable identifier, supplier, launch date, category, and destination scope.
2. Validate required content attributes and use compatibility rules where appropriate.
3. Confirm item status, price, delivery window, stock, returns policy, and any restrictions.
4. Identify comparable products by defensible attributes and customer need, not merely the supplier's marketing label.
5. Choose an initial prior or content-based score and document its source, age, and limitations.
6. Decide which surfaces and audiences are relevant and approved.
7. Set an exploration budget, maximum per-item exposure, and stop conditions.
8. Confirm the supplier and fulfilment owner can handle the plausible range of demand.
9. Define how evidence will be reviewed and when the item's provisional status ends.

Input: item master, operational approvals, category taxonomy, and comparable item data.
Output: a versioned launch record, eligible audience, and bounded exploration plan.
Branch: missing critical compatibility or safety attributes block exposure until corrected.
A missing optional description may instead reduce ranking confidence or route the item to manual review.

A cold item can borrow information from its category or similar products, but the prior is not the item's measured conversion rate.
A recently launched private-label item may inherit a prior from comparable private-label products, but a category-level prior could conceal unequal supplier exposure.
A specialist should review whether comparison items cover the relevant price band, use case, and market.

### 5.6 Set a bounded exploration policy

1. Select an exploration unit: one slot in a slate, one request, or an item-audience pair.
2. Reserve a maximum share of eligible traffic or slots for exploration.
3. Restrict the exploration pool to items passing the same hard gates as exploitation candidates.
4. Define a relevance floor using permitted attributes and the surface's task.
5. Randomize among eligible items or assign known probabilities using a documented policy.
6. Cap item-level and supplier-level exposure where needed to avoid concentration and stock exhaustion.
7. Account for frequency caps so one person is not repeatedly shown an unsuitable item.
8. Log the actual probability conditional on the eligible set, position, preceding selections, and policy state.
9. Pause or reduce exploration when a safety, inventory, or quality guardrail is breached.

Input: candidate pool, allowed budget, estimated downside, stock constraints, and approved objectives.
Output: per-request action probabilities, selected item, and a complete decision log.
Branch: if no exploration item is eligible, use the ordinary eligible fallback; do not force exploration.
If the logging service cannot record required assignment information, stop the experiment or switch to its approved nonexperimental policy.

A random choice among two candidates has probability one-half only if both are eligible and equally weighted at the time of selection.
If one candidate becomes ineligible, the remaining candidate's conditional probability is one, not one-half.
For a multi-slot slate sampled without replacement, record the probability for each selection conditional on earlier choices, or the joint slate probability when required by the estimator.
A marginal probability per item is not necessarily enough for slate-level effects.

Randomization can compare a new-item policy with a baseline, but it cannot override customer restrictions or an unsafe product.
Stratification by market, surface, or user-cold status can improve balance when specified before assignment.
Keep the assignment unit consistent with the interference risk: if a user sees many sessions, user-level assignment may be preferable to request-level assignment.
Supply is shared, so even user-level randomization can have interference if the treatment exhausts inventory available to control.

### 5.7 Design the evidence and logging contract

1. Define a rendered impression, a viewable impression if used, and the attribution window.
2. Log request time, surface, allowed context, eligible-set version, and policy version.
3. Log treatment assignment, selected item, slot, source rung, and probability under the actual policy.
4. Log exclusions and feed freshness in aggregated or otherwise approved form where useful.
5. Join permitted outcome events such as click, order, cancellation, return, complaint, and contribution using approved identifiers.
6. Record the maturity of each outcome window; returns may arrive well after clicks.
7. Document late-arriving events, deduplication, missing identifiers, and consent-related deletion effects.
8. Publish denominator definitions and an audit trail accessible only to authorized roles.

Input: instrumented serving and event pipelines, data dictionary, and approved retention rules.
Output: a decision-level dataset fit for the planned experiment and diagnostics.
Branch: if assignment or propensity is missing, do not pretend the affected records support the planned causal estimate.
Investigate logging loss by treatment and device before interpreting a difference.

The set of candidates that could have been chosen matters for interpreting a probability.
Recording a final model score does not establish the probability with which the item was selected.
An item with no eligible exposure in a context cannot be evaluated there by inverse propensity weighting from those logs alone.
Even with nonzero probability, very small probabilities can make weights unstable.
Predefine diagnostics, weight handling, and sensitivity analyses rather than selecting them after seeing results.

### 5.8 Evaluate the choice and graduate cautiously

1. Compare the exploration policy with the approved baseline in the randomized population.
2. Report effect estimates with uncertainty, sample sizes, and the prespecified unit of analysis.
3. Review outcomes at their correct maturity: clicks quickly, orders after their window, returns later.
4. Check customer, supplier, market, cold-user, and stock-availability slices where sample size and privacy permit.
5. Check exposure opportunity as well as outcomes, because low exposure can conceal promising items.
6. Review guardrails and any material operational incident before promoting a policy.
7. Make an explicit continue, graduate, pause, revise, or retire decision.
8. Record the owner, decision time, evidence, uncertainty, and next review date.

Input: experiment readout, operational incidents, supplier feedback, and governance review.
Output: a signed decision record and the next policy version.
Branch: an inconclusive estimate is not evidence of equivalence; continue if the budget and decision value justify it, or stop for a stated operational reason.
If a safety stop triggers, pause immediately under the approved incident procedure rather than waiting for statistical significance.

Graduation need not mean an item receives unrestricted traffic.
It means that the ordinary ranking and eligibility policy may consider the item using the evidence available.
The item can become cold again after a material price change, stock interruption, formulation change, or market transfer.
A reviewed threshold should include relevant impressions, outcome maturity, and stability, not age alone.

### 5.9 Allocate responsibilities and prepare rollback

| Decision | Accountable owner | Supporting evidence | Escalation |
|---|---|---|---|
| Business objective and launch | Product owner | Baseline, objective, opportunity cost | Commerce leadership |
| Item eligibility and recall | Category or product safety owner | Catalogue validation and restriction feed | Safety incident process |
| Stock and promise | Inventory or fulfilment owner | Stock freshness, serviceability, capacity | Operations on-call |
| Processing and retention | Privacy/legal owner | Purpose, data map, jurisdiction, rights workflow | Privacy incident process |
| Randomization and analysis | Experimentation/analytics owner | Assignment, power assumptions, logging | Experiment review |
| Supplier commitments | Commercial owner | Contract and exposure plan | Procurement/legal |
| Deployment and rollback | Service owner | Feature flag, health checks, approved default | Engineering on-call |

A rollback must have a named policy, not just a model version.
It must keep current eligibility and consent checks, even if the ranking model is disabled.
The default list itself needs a refresh owner and an expiry time.
Write the incident timestamp and affected traffic window so the experiment can exclude or separately analyze the disruption under a prespecified rule.

### 5.10 Work through the new-user decision

The following sequence illustrates how evidence changes within one request journey.
It is a policy design example, not a prescription for a particular jurisdiction.
The privacy owner must authorize each input before engineering enables it.

**Arrival with no permitted individual history.**
The service knows which page was requested and which catalogue is configured for that page.
It does not invent a preference vector filled with zeros and then call the result personalized.
It requests candidates from a verified curated or appropriately aggregated popularity policy.
The catalogue and fulfilment gates remove items that cannot be offered on this surface.
The service may show fewer than the requested number of items.
The product owner should measure the frequency of this reduced-slate outcome.

**A user supplies an explicit need.**
The interface asks an optional, understandable question such as which equipment model needs a replacement part.
The user can continue without answering.
The answer may be used for the immediate compatibility decision when approved for that purpose.
It is not automatically permission for cross-session profiling, advertising, or supplier disclosure.
The service filters by verified compatibility before ranking by other features.
If the answer is ambiguous, it asks for clarification or provides neutral navigation rather than guessing.

**The user performs a current-session action.**
A category selection, search term, or product view may provide an intent signal.
The policy specifies which actions qualify and how long they remain usable.
For example, a category selection may be stronger evidence than a single incidental product view.
A change of category can invalidate the earlier signal rather than being averaged blindly.
If a visitor views several unrelated categories, a diverse fallback may be safer than assuming a narrow preference.
An authorized session feature can guide the current result without being copied into a durable identity graph.

**The user returns later.**
Whether a previous interaction can be used depends on the approved processing purpose, identity, retention, and applicable rights.
If persistent use is not authorized, the service begins again with the approved nonpersonal policy.
If persistent use is authorized but the data is stale or sparse, the service should compare it with fresher permitted intent.
The existence of an account does not guarantee that a historical preference remains correct.
A user who once ordered office supplies may now be looking for industrial maintenance parts.

This sequence gives product and privacy teams separate choices.
They can enable helpful in-session controls without silently making a long-term profile.
They can also allow an existing profile while applying a freshness rule and an explicit override.
The correct choice depends on the approved use and measured performance, not on a blanket claim that more history is always better.

### 5.11 Select comparable items without hiding uncertainty

Choosing a comparison set is an operational decision that affects the first opportunities given to new products.
Start with the need the item serves, rather than a broad department name alone.
Then compare attributes that plausibly affect demand and fulfilment:

1. Product function and compatibility.
2. Price band and unit size.
3. Destination and delivery promise.
4. Brand or supplier characteristics when justified and permitted.
5. Channel, surface, and customer task.
6. Season and time since launch.
7. Availability during the historical measurement window.
8. Treatment of promotions, sponsored placements, and returns.

A category average can be misleading if its bestsellers were continuously in stock while the new product has intermittent supply.
A historical product shown only in a sponsored slot should not be treated as an ordinary organic comparable without accounting for placement.
A product with a different pack size may have a different unit economics profile despite sharing a barcode family.
A geographically distant item can supply a hypothesis, but local pricing and fulfilment still need validation.

Document both the included and excluded comparable groups.
State the aggregation window and the number of eligible impressions behind the prior.
Review whether the comparison set gives new suppliers a systematically different starting point from established suppliers.
Small groups may be unstable or raise privacy concerns, so suppress or combine them under approved reporting rules.
If the comparison set is weak, widen uncertainty rather than pretending the estimate is precise.
Where no useful prior exists, a conservative manually reviewed value can be used as an explicit assumption.
The owner should schedule a review when local exposure accumulates.

### 5.12 Decide how much exploration to buy

Exploration consumes scarce attention and can also consume stock.
The opportunity cost has at least four components:

1. The predicted contribution of the eligible item that would otherwise occupy the slot.
2. The predicted contribution of the exploratory item on that same opportunity.
3. Additional variable costs, including delivery, returns, customer support, and supplier handling where applicable.
4. The expected value of learning, which may improve later allocation but is uncertain.

The first three components are not always observed at decision time.
Write their assumptions and range rather than hiding them in one point estimate.
The fourth is not automatically positive enough to justify any experiment.
The team should compare potential learning with the number of future decisions that could actually change.
If an item will be retired tomorrow and has no similar future products, a costly experiment may have little reusable value.
If the product will be sold for a year across many markets, careful early learning may have more value.
Neither statement is a quantitative valuation without further assumptions.

Translate the budget into operational limits.
The traffic share controls aggregate opportunity cost but does not by itself cap an individual item's exposure.
An item-level cap prevents a single product from consuming the whole pool.
A supplier-level cap can reduce concentration, subject to fair and contractual treatment.
A per-user frequency cap avoids repeatedly testing the same irrelevant product on one visitor.
A stock-derived cap keeps projected orders below a conservative share of available supply.
Each cap needs an owner, a measurement interval, and a rule for what happens when it is reached.

For example, suppose an item has 40 verified available units.
The team estimates an order probability of 4% per eligible impression, with substantial uncertainty.
A naive calculation gives `40 / 0.04 = 1,000` impressions before expected demand reaches 40 units.
That is not a safe guarantee: the rate can change, other channels consume stock, and orders arrive in clusters.
The inventory owner could instead allocate a smaller impression cap, reserve stock, or require frequent stock checks.
The numerical cap must reflect the business's risk tolerance and operational data.

### 5.13 Distinguish item evidence from policy evidence

A randomized item choice can inform a comparison among eligible items within the tested contexts.
A randomized policy assignment can inform the effect of offering the exploration policy instead of the baseline.
Those are different estimands and need different denominators.

For the policy question, define treatment as the exploration-enabled experience and control as the existing approved experience.
Include all assigned eligible units in the primary analysis when the design calls for an intention-to-treat comparison.
Some treatment-assigned requests may receive fallback because no exploratory item passes the gates.
Dropping those requests after assignment can change the population and bias the comparison.
Report that fallback rate separately as an implementation diagnostic.

For the item question, define which item, audience, surface, and time window are being compared.
If the item was unavailable in part of the window, identify the relevant eligible opportunities.
If the selection probability changed with availability, use the logged conditional probability appropriate to each decision.
If the item only appeared in one slot, a claim about a different slot requires an additional assumption or experiment.
If the same visitor sees several items, account for correlated outcomes at the appropriate analysis unit.

Do not compare a treatment item's gross sales with the baseline item's sales and call the difference incremental profit.
Some orders might have occurred anyway, and a recommendation can shift sales among items rather than create new demand.
Supplier fees, cannibalization, returns, and fulfilment costs can change the commercial conclusion.
Finance and analytics should agree the contribution definition before inspecting the experiment.

### 5.14 Plan exposure logging for a multi-slot slate

Suppose a slate has three positions and items are sampled sequentially without replacement.
At the first position the eligible pool contains A, B, and C.
If the policy chooses uniformly, each has conditional probability `1/3` at that position.
After selecting A, the remaining pool contains B and C.
Under the same uniform rule, B has conditional probability `1/2` for the second position.
The remaining item C then has probability `1` for the third position.
The joint probability of the ordered slate A, B, C is `(1/3) × (1/2) × 1 = 1/6`.

That number describes the ordered slate under this particular policy and eligibility state.
It changes if the policy uses relevance weights, position constraints, caps, or a different candidate pool.
It also changes if a safety check removes B before the second choice.
The decision log must reflect the actual state when each choice is made.
For a later estimator, record enough information to reconstruct the policy and the candidate eligibility at each step.
Avoid logging unnecessary sensitive raw features when a policy version or privacy-approved summary is sufficient.

If the baseline and exploration policies can both select the same item, the probability of the final item can be the sum of the probabilities of all paths that produce it.
Recording only the probability of the path that happened can be insufficient for an item-action estimator.
Conversely, an estimator of an assigned policy may require the assignment probability rather than an item marginal.
Analytics and engineering must agree on the estimand before deciding what to log.

### 5.15 Specify failure and recovery behavior

The fallback ladder is a production control only if its behavior is defined under partial failure.
Consider each dependency independently.

**Permission service unavailable.**
Optional personalized processing follows the approved fail-closed policy.
The service uses only a separately authorized fallback and records the outage without exposing a user identifier unnecessarily.
The privacy owner determines whether any operational processing remains permitted.

**Catalogue service unavailable.**
Use a verified, time-bounded cache only if the safety and catalogue owners have approved its maximum age.
Do not rely on a cached listing after a recall update could have arrived.
If the cache cannot meet the safety policy, return a safe empty state.

**Inventory service delayed.**
Apply the documented freshness threshold for the surface.
A browsing surface may have a different approved response from an immediate substitution or checkout surface.
Do not represent stale inventory as a firm delivery commitment.
Escalate sustained failures to the fulfilment owner.

**Recommendation model times out.**
Use the next permitted rung, subject to the same eligibility rules.
Record the timeout and the actual source rung.
Prevent automatic retries from exhausting the latency budget or causing duplicate logging.
When the model recovers, resume only under the deployment and incident policy.

**Experiment logging is incomplete.**
If the required assignment or action probability cannot be recovered, halt randomized exploration or use the approved nonexperimental fallback.
Do not backfill a missing probability with an assumed value after the fact.
Mark the affected interval and notify analytics.
The analysis plan should specify how interruptions and incomplete outcomes are handled.

**No eligible candidates remain.**
Display an honest empty state, category navigation, or a separately verified alternative.
Do not remove a safety or consent rule to fill a visual slot.
Measure how often this occurs and why, without treating the empty state itself as a defect in a safety-critical case.

### 5.16 Close the learning loop

The periodic review should bring together product, analytics, inventory, privacy, commercial, and category representatives.
Start with the decision the trial was designed to support.
Present the number of eligible opportunities, randomized units, rendered impressions, and mature outcomes.
Show the baseline and treatment estimates with uncertainty and the original analysis population.
Report safety incidents and stockouts even when they occur in a small subgroup.
Separate operational failures from evidence about item relevance.

Next, review supplier and customer distribution.
A supplier's low observed sales may reflect low eligible inventory rather than an unfavorable ranking rule.
A high impression share may come from a paid commitment rather than organic model preference.
A customer group's low click rate may reflect poor fit, placement, or measurement failure.
Avoid drawing conclusions about small groups from unstable or privacy-inappropriate breakdowns.

Finally, choose one of five documented outcomes.

| Outcome | Appropriate condition | Required follow-through |
|---|---|---|
| Continue | Uncertainty remains and remaining learning value justifies risk and cost | Set a new review date, budget, and stop conditions |
| Graduate | Relevant evidence and operational readiness satisfy the approved rule | Move to ordinary ranking while retaining eligibility controls |
| Revise | The policy or input quality is unsuitable but correctable | Version the change and decide whether a new experiment is needed |
| Pause | Safety, legal, supply, or instrumentation incident requires intervention | Route traffic to the approved fallback and open an incident |
| Retire | The item or policy no longer merits exposure under the stated objective | Record the reason and ensure contractual obligations are addressed |

An item may be retired for commercial reasons despite uncertain response evidence.
Record that as a business decision rather than claiming a statistically established preference.
An item may be paused for safety even when its observed contribution is high.
Record that as a constraint, not as a model error to be optimized away.
When the product changes materially, reconsider whether previous observations still apply.
The final record should identify what was learned, what remains uncertain, and who owns the next decision.

## 6. Worked example

### 6.1 The decision

A fictional grocery marketplace launches a home-page carousel in one delivery region.
The surface has one eligible carousel opportunity per session for this example.
The team compares a curated local-popularity baseline with a policy that reserves a fraction of opportunities for eligible new items.
The decision horizon is 28 days.
Currency is U.S. dollars (USD).
The numbers are assumptions for planning, not measured company results.

The product owner forecasts 20,000 eligible sessions over 28 days.
This is a forecast, not a traffic commitment.
The plan reserves 10% of opportunities for new-item exploration.
The resulting planned exploration volume is `20,000 × 0.10 = 2,000` impressions.
The team sets a target of at least 1,800 successfully logged exploration impressions, subject to eligibility and traffic.
The inventory owner commits 400 sellable units across the candidate pool, with daily refresh and no guarantee that each item will receive equal demand.
The commitment is a stock allocation, not a guarantee of customer orders.

The baseline contribution rate is assumed to be USD 0.12 per impression after specified variable fulfilment costs.
For the new-item exploration pool, the planning estimate is USD 0.09 contribution per impression.
The expected immediate opportunity cost is therefore `2,000 × (0.12 − 0.09) = USD 60`.
This calculation assumes the displayed alternatives would otherwise have earned the stated rate and does not estimate long-term learning value.
The team budgets up to USD 80 of immediate opportunity cost plus a separate cap on operational incidents.
The USD 60 estimate is below that budget, but uncertainty and inventory capacity still matter.

### 6.2 Eligibility and fallback

Three prospective items arrive:

| Item | Relevant category | Stock | Status | Decision |
|---|---|---:|---|---|
| A | Breakfast cereal | 180 units | Approved, deliverable | Eligible for breakfast context |
| B | Breakfast cereal | 120 units | Approved, deliverable | Eligible for breakfast context |
| C | Chilled meal | 100 units | Cold-chain status unresolved | Excluded pending verification |

A new visitor selects breakfast during the current session.
Assume the selection is permitted for the current request and no persistent personalization is authorized.
A and B pass the relevance and stock checks.
C does not pass the operational gate and is not in the randomization pool.
The policy cannot use C merely because the item needs exposure.
If neither A nor B remains available, the carousel uses the verified local-popularity fallback.
If that list also has no eligible item, the surface shows a safe empty state or category navigation.

On an exploration-assigned request with both A and B eligible, choose one uniformly.
The conditional selection probability of each is `1 / 2 = 0.50`.
If B becomes unavailable before selection, A is selected with probability `1.00` conditional on that eligible set.
The decision log records the exploration assignment, eligible-set version, choice, conditional probability, position, and policy version.
It does not record a persistent user profile merely because the visitor selected breakfast.

### 6.3 Illustrative observed outcomes

Suppose an analysis at day 28 finds the following exploration results after deduplication.
Assume all reported order windows have matured; returns are reported separately if their window is not yet mature.

| Item | Rendered impressions | Clicks | Attributed orders | Contribution after variable costs |
|---|---:|---:|---:|---:|
| A | 980 | 98 | 49 | USD 98.00 |
| B | 920 | 64 | 23 | USD 64.40 |
| Total | 1,900 | 162 | 72 | USD 162.40 |

The achieved exploration volume is 1,900 impressions, above the 1,800-impression operational target.
That target is not proof of commercial success.
A's CTR is `98 / 980 = 0.10`, or 10.00%.
B's CTR is `64 / 920 ≈ 0.0696`, or 6.96%.
A's order rate is `49 / 980 = 0.05`, or 5.00%.
B's order rate is `23 / 920 = 0.025`, or 2.50%.
A's contribution per impression is `USD 98.00 / 980 = USD 0.10`.
B's contribution per impression is `USD 64.40 / 920 = USD 0.07`.
The pooled contribution per exploration impression is `USD 162.40 / 1,900 ≈ USD 0.0855`.

At the assumed baseline of USD 0.12 per impression, the simple opportunity-cost estimate for 1,900 exploration impressions is:

`1,900 × USD 0.12 − USD 162.40 = USD 65.60`.

This is an arithmetic comparison to a planning assumption, not a causal estimate.
A randomized control group is necessary to estimate the incremental policy effect under the experiment's assumptions.
The result is within the USD 80 planning budget, but the final decision also depends on uncertainty, returns, stock effects, and guardrails.
Neither item should automatically graduate solely because A's observed rate is higher.
The two item audiences, eligibility sets, inventory interruptions, and positions may differ.
The analytics owner should report uncertainty and verify the randomized comparison.

### 6.4 Smoothing and its limits

For a simple teaching calculation, suppose the prior order rate is 3% and the team chooses a prior strength of 100 impression-equivalents.
This corresponds to a Beta prior with parameters `alpha = 3` and `beta = 97` for a binary order-per-impression outcome.
Under a simplified independent Bernoulli model, the posterior mean is `(orders + alpha) / (impressions + alpha + beta)`.

For A: `(49 + 3) / (980 + 100) = 52 / 1,080 ≈ 4.81%`.
For B: `(23 + 3) / (920 + 100) = 26 / 1,020 ≈ 2.55%`.
For an item with no exposure, the posterior mean remains `3 / 100 = 3%`; it is the prior, not observed item performance.

The calculation assumes comparable independent impressions and a stable order definition.
Repeated users, exposure selection, attribution errors, and shared stock can violate those assumptions.
The prior strength is a modeling choice and needs sensitivity analysis, especially when suppliers or categories differ.
It does not solve position bias, establish causation, or justify collecting prohibited data.

### 6.5 Decision record

The product owner decides to continue the trial pending the randomized control readout and mature return data.
The inventory owner confirms that remaining stock remains sellable and updates availability daily.
The privacy owner confirms that session-only preference processing and the experimental log comply with the approved purpose and retention schedule.
The analytics owner checks propensity and assignment completeness and reports uncertainty by the agreed analysis unit.
The category owner holds C until cold-chain status is verified.
This is a decision under uncertainty, not a promise that exploration will produce a positive return.

## 7. Code, query, or calculation

The following Python 3 script uses only the standard library.
It demonstrates deterministic eligibility, an approved fallback, a bounded exploration draw, the conditional item probability, and the example arithmetic.
Save it as `cold_start_demo.py` and run `python cold_start_demo.py`.
It is a teaching simulation, not a production privacy, inventory, or experimentation service.
The caller supplies already-authorized session intent; `None` means no permitted intent is available.
The static item status and stock stand in for verified, fresh operational feeds.
The random generator is seeded for reproducibility, not for security.
The displayed probability is conditional on an exploration assignment; the joint probability also includes the 10% exploration assignment.
The sample code logs no customer identifier and does not persist session data.

```python
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Item:
    item_id: str
    category: str
    approved: bool
    deliverable: bool
    available_units: int
    is_new: bool


def eligible(item: Item) -> bool:
    return item.approved and item.deliverable and item.available_units > 0


def choose_item(items, fallback_ids, permitted_category, rng, explore_rate=0.10):
    if not 0.0 <= explore_rate <= 1.0:
        raise ValueError("explore_rate must be between 0 and 1")

    by_id = {item.item_id: item for item in items}
    pool = [
        item for item in items
        if eligible(item)
        and item.is_new
        and permitted_category is not None
        and item.category == permitted_category
    ]
    assigned_explore = rng.random() < explore_rate

    if assigned_explore and pool:
        selected = rng.choice(pool)
        conditional_probability = 1.0 / len(pool)
        return {
            "item": selected.item_id,
            "rung": "exploration",
            "conditional_probability": conditional_probability,
            "joint_probability": explore_rate * conditional_probability,
            "eligible_exploration": tuple(item.item_id for item in pool),
        }

    for item_id in fallback_ids:
        item = by_id.get(item_id)
        if item is not None and eligible(item):
            # If no exploration candidate exists, both assignment paths
            # produce this fallback. Otherwise only the nonexplore path does.
            probability = 1.0 if not pool else 1.0 - explore_rate
            return {
                "item": item.item_id,
                "rung": "curated_fallback",
                "conditional_probability": 1.0,
                "joint_probability": probability,
                "eligible_exploration": tuple(item.item_id for item in pool),
            }
    return {
        "item": None,
        "rung": "safe_empty",
        "conditional_probability": 1.0,
        "joint_probability": 1.0 if not pool else 1.0 - explore_rate,
        "eligible_exploration": tuple(item.item_id for item in pool),
    }


def posterior_order_rate(orders, impressions, alpha=3, beta=97):
    if not 0 <= orders <= impressions:
        raise ValueError("orders must be between zero and impressions")
    return (orders + alpha) / (impressions + alpha + beta)


items = [
    Item("A", "breakfast", True, True, 180, True),
    Item("B", "breakfast", True, True, 120, True),
    Item("C", "meal", True, False, 100, True),
    Item("D", "breakfast", True, True, 200, False),
]

# Seed 31 makes the first assignment draw less than 0.10.
result = choose_item(items, ["D"], "breakfast", Random(31))
print("selected:", result["item"])
print("rung:", result["rung"])
print("eligible exploration:", result["eligible_exploration"])
print("conditional probability:", f'{result["conditional_probability"]:.2f}')
print("joint probability:", f'{result["joint_probability"]:.2f}')

# No permitted category: no item-level session targeting or exploration.
private_result = choose_item(items, ["D"], None, Random(31))
print("no permitted intent:", private_result["rung"], private_result["item"])

print("A smoothed order rate:", f'{100 * posterior_order_rate(49, 980):.2f}%')
print("B smoothed order rate:", f'{100 * posterior_order_rate(23, 920):.2f}%')
actual_contribution = 98.00 + 64.40
opportunity_cost = 1900 * 0.12 - actual_contribution
print("estimated opportunity cost:", f'USD {opportunity_cost:.2f}')
```

Expected output for Python's standard `random.Random` implementation:

```text
selected: A
rung: exploration
eligible exploration: ('A', 'B')
conditional probability: 0.50
joint probability: 0.05
no permitted intent: curated_fallback D
A smoothed order rate: 4.81%
B smoothed order rate: 2.55%
estimated opportunity cost: USD 65.60
```

The example returns a joint probability for the sampled decision path.
A production system evaluating final item actions must account for all paths that can produce the same item, including any overlap between exploration and fallback pools.
For a slate, it must also account for position and previous selections.
The function does not enforce age restrictions, destination rules, stale-feed timeouts, consent decisions, frequency caps, or atomic stock reservations.
Those omissions are explicit because these require organization-specific services and policy, not because they are optional in deployment.

## 8. Data, standards, and regulatory context

### 8.1 Data contract

The item master needs stable item identifiers, valid category and compatibility attributes, effective dates, and source provenance.
The serving layer also needs destination-specific policy and inventory data with a documented freshness service level.
The recommendation event needs a clear distinction among candidate, served response, rendered impression, click, and attributable order.
A returned JSON response is not automatically an impression.
A cancelled or returned order should not be counted as identical to a retained order in a contribution analysis.

Record purpose, authorization decision, and permissible retention for each data element.
Minimize what is collected; keep raw event access narrower than aggregate reporting access.
Define deletion propagation across event stores, feature stores, training datasets, backups, and derived artifacts according to the applicable legal and technical policy.
A deletion request can change available analysis data; predefine how reporting handles this without retaining prohibited identifiers for convenience.
A salted or hashed identifier may still be personal data where re-identification or linkage is possible.

### 8.2 Regulatory considerations

The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, provides principles including purpose limitation, data minimization, storage limitation, and a lawful basis for processing in its applicable scope.
Articles 5 and 6 are starting points for those questions; Article 7 sets conditions when consent is the basis, and Articles 12–22 address transparency and data-subject rights subject to their terms and exceptions.
The law does not make consent the only possible legal basis for every recommendation use.
Whether a particular use, device storage, profiling practice, or right applies depends on the facts and relevant law.
The EU ePrivacy framework and national implementations can impose separate requirements for accessing or storing information on a device.
Obtain jurisdiction-specific advice rather than assuming a server-side profile makes device rules irrelevant.

In California, the California Consumer Privacy Act as amended by the California Privacy Rights Act provides rights and obligations subject to statutory definitions, applicability thresholds, exceptions, and regulations.
Whether a recommendation or data sharing practice constitutes a sale, sharing, targeted advertising, or another regulated activity is fact-specific.
Honor applicable opt-out and other rights through the organization's approved process.
Do not assert that a generic consent checkbox resolves every obligation.

Rules and interpretations change; the references below should be checked against the current law and regulator guidance before deployment.
This course is educational and is not legal advice.
The privacy owner should document legal basis, disclosures, retention, applicable rights, and cross-border considerations.
The product owner remains accountable for implementing the resulting decision.

### 8.3 Identifiers and standards

GS1 Global Trade Item Numbers (GTINs) are a common standard for identifying trade items.
A GTIN can help match catalogue records, but it does not itself prove that an item is safe, in stock, compatible, or permitted for a destination.
Organizations may have internal identifiers in addition to GTINs; specify mapping, effective dates, and versioning.
For variants, verify that the identifier and attributes refer to the correct size, pack, and formulation.
A substituted identifier can make a new item appear to have the history of a different product.

### 8.4 Sensitive data and fairness

Do not infer protected or sensitive attributes simply to improve an initial ranking.
An apparently neutral context can be a proxy for a sensitive attribute or can make a small cohort identifiable.
Fairness measurement may require carefully governed data and cannot be replaced by simply suppressing all diagnostics.
Discuss the authorized measurement design with privacy, legal, and relevant affected stakeholders.
Supplier fairness and customer fairness are distinct: equal supplier impressions do not guarantee appropriate products or equitable customer outcomes.
Document the chosen exposure and outcome measures, their denominators, and trade-offs.

## 9. Industry practice and case evidence

### 9.1 Netflix and contextual cold start

Netflix's public paper by Gomez-Uribe and Hunt describes a recommender system combining multiple algorithms and the role of context, member interactions, and presentation.
It discusses the cold-start problem and the need to choose recommendations for members and titles with limited data.
The paper describes a specific video service and period; it does not establish that the same ladder or economics work for a grocery marketplace.
[Inference] A commerce team can borrow the principle of combining multiple evidence sources while adding explicit inventory and fulfilment gates.
Do not cite the paper as proof of the synthetic outcome in Section 6.

### 9.2 News exploration and selection bias

Li and coauthors describe a contextual-bandit approach for personalized news recommendation and an offline evaluation method using randomized traffic.
Their work illustrates why deliberately randomized exposure can produce useful learning and evaluation data.
It also depends on the assumptions of the evaluation procedure and the logging policy.
[Inference] For a new retail item, an approved exploration slot can similarly create evidence that pure popularity ranking would not collect.
A retail deployment must additionally account for inventory depletion, margin, delivery, safety, and returns.
The paper is not a guarantee that every retail bandit improves contribution.

### 9.3 Music recommendations and implicit feedback

Hu, Koren, and Volinsky analyze collaborative filtering for implicit feedback, where observed interactions carry confidence but unobserved items are not equivalent to explicit dislikes.
The distinction matters for new products that have not yet had a chance to be observed.
[Inference] A zero-sale item with no eligible impressions should be treated differently from one repeatedly shown to relevant customers without response.
Their paper does not by itself specify an exploration policy, legal basis, or inventory rule.

### 9.4 What transfers and what does not

Public cases are useful for understanding general design choices, not for importing point estimates.
A market with few suppliers and constrained stock can have very different exposure economics from a large media catalogue.
Before applying a published technique, check the surface, audience, feedback delay, action space, counterfactual, and operational constraints.
A documented increase in clicks in another domain is not a forecast of incremental profit in yours.
A vendor demo without randomized, independently interpretable evidence is weaker than a peer-reviewed study for the question it actually evaluates.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No exposure for new products | New-item impression share stays near zero | Popularity model favors incumbents | Learning never starts | Bounded eligible exploration with opportunity reporting |
| Unsafe backfill | Restricted items appear after model timeout | Fallback bypasses hard gates | Safety or compliance incident | Apply shared gates to every rung; fail safely |
| Stale inventory | Clicks rise while cancellations rise | Cache shows stock that has sold out | Customer harm and wasted exposure | Freshness thresholds and recheck at transaction |
| Missing consent state | Personalization continues during permission outage | Unknown treated as granted | Unauthorized processing | Approved fail-closed default and audit |
| Session becomes profile | Session event appears in long-term features | Retention and purpose boundaries are unclear | Privacy and trust harm | Separate stores, retention enforcement, access controls |
| Position confounding | New items mostly appear in a lower slot | Attention differs by slot | False item-performance conclusion | Randomize or adjust with a justified position-aware design |
| Missing propensities | Many records have scores but no probabilities | Logging contract omits selection mechanics | Offline causal estimate is unsupported | Log conditional action probabilities and eligible set |
| Zero support | Some items never eligible under logging policy | Deterministic allocation | Counterfactual cannot be identified from logs | Run a bounded randomized experiment where safe |
| Tiny propensities | A few records dominate weighted estimates | Extreme inverse weights | High variance and unstable decisions | Diagnose overlap; redesign allocation; prespecify sensitivity analysis |
| Bad prior transfer | New segment systematically underperforms prior | Comparables do not match audience or product | Misallocated exposure | Validate comparables and test prior sensitivity |
| Misleading CTR | Clicks improve but returns and contribution worsen | Engagement is optimized as a surrogate | Commercial loss | Use mature contribution and customer guardrails |
| Supplier concentration | Few suppliers receive most new-item slots | Score or cap design favors incumbents | Reduced supplier opportunity | Monitor eligible exposure by supplier and review caps |
| Forced slate fill | Ineligible items appear only when pool is small | Display count overrides constraints | Unsafe or unavailable recommendations | Permit fewer items and a safe empty state |
| Premature graduation | Promotion occurs after a few favorable clicks | Noise mistaken for stable evidence | Unreliable traffic allocation | Require relevant exposure, maturity, and uncertainty review |
| No graduation | Item remains experimental indefinitely | Owner or review date missing | Wasteful exploration and supplier frustration | Assign review owner, schedule, and decision options |
| Shared-stock interference | Control availability falls as treatment sells stock | Randomized groups compete for inventory | Biased experiment comparison | Monitor inventory, consider cluster/time designs and sensitivity |
| Misidentified item variant | Return rate spikes for one pack size | Incorrect item mapping or attributes | Wrong product and corrupted history | Validate GTIN and variant mappings before launch |
| Expired curated fallback | Default list contains discontinued products | No owner or refresh schedule | Degraded experience during outages | Version, expire, and periodically validate defaults |
| Incomplete outcomes | Early orders look good but returns later increase | Readout precedes outcome maturity | Overstated contribution | Report mature windows and later follow-up |
| Hidden contractual promotion | Sponsored item is analyzed as organic exploration | Commercial override is not logged | Misleading attribution and trust issues | Label and separately log contractual placement |

A single warning may have several causes.
Investigate feed freshness, population mix, and instrumentation before attributing the change to the recommender.
Safety and legal incidents follow their own escalation rules; they do not wait for a causal analysis.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or monitor |
|---|---|---|---|
| Eligible items are actually sellable | The policy is only safe if gates work | Stale recall or stock feed | Compare decisions to authoritative feed and incident logs |
| Permission state is current | Data use depends on authorization | Revocation propagation delay | Audit latency from rights event to serving and stores |
| Session intent is meaningful | A context rung should improve relevance | Accidental or ambiguous clicks | Compare against baseline by intent type and age |
| Curated list is representative enough | It must provide a useful safe fallback | Seasonal change or regional mismatch | Review list age, coverage, and outcomes by market |
| Comparable items inform the prior | Borrowed evidence stabilizes early estimates | New product differs materially | Compare calibration and outcomes across comparable groups |
| Exploration allocation is logged | Evaluation needs actual policy probabilities | Sampling changes after log calculation | Reconstruct choices from decision logs and versions |
| Randomized groups are comparable | Treatment differences support causal interpretation | Assignment leakage or differential missingness | Check balance, assignment integrity, and event loss |
| Stock interference is limited | Control should represent counterfactual outcomes | Treatment exhausts shared inventory | Track availability by arm and conduct sensitivity analysis |
| Outcomes are mature | Contribution requires delayed costs | Returns and cancellations arrive later | Report maturity curves and a later final readout |
| Costs are correctly allocated | Profit comparison uses consistent units | Missing fulfilment or support costs | Reconcile with finance definitions and invoices |
| Exposure is measured consistently | CTR denominator must be meaningful | Render failure or hidden placement | Compare client render events and server responses |
| Traffic forecast is plausible | Budget converts percentages into impressions | Seasonality, outages, or promotion | Monitor actual eligible sessions against forecast |
| Statistical uncertainty is acceptable | A decision needs sufficient evidence | Small or correlated samples | Compute intervals at the assignment unit and plan sample size |
| No prohibited proxy is used | Privacy restrictions must hold in practice | Device or location reconstructs identity | Review feature lineage and access controls |
| The item remains comparable over time | Evidence should describe the current offer | Price, formulation, or fulfilment changes | Version item attributes and re-evaluate after changes |

The simple Beta calculation is descriptive under strong assumptions, not a substitute for an experiment.
A highly personalized policy may create many low-support contexts where reliable off-policy evaluation is impossible.
An experiment can still be operationally inappropriate if the expected harm is unacceptable.

### 11.1 Sensitivity and decision boundaries

Use a range of plausible assumptions rather than treating the forecast as known.
In the worked example, 1,900 exploration impressions and USD 162.40 of observed contribution are held fixed for illustration.
If the appropriate baseline were USD 0.10 per impression, the simple difference would be `USD 190.00 − USD 162.40 = USD 27.60`.
If the baseline were USD 0.14 per impression, it would be `USD 266.00 − USD 162.40 = USD 103.60`.
The latter exceeds the illustrative USD 80 opportunity-cost budget.
These are sensitivity calculations, not confidence intervals or causal effects.
They show that the decision depends on the counterfactual, which must be estimated rather than chosen to fit a preferred outcome.

The inventory assumption is also consequential.
An average order rate does not determine the maximum number of units needed on any single day.
Concurrent demand from search, replenishment, and supplier campaigns can consume the same stock.
An experiment should record stock availability and important simultaneous commercial activity.
An inventory reserve may protect the trial, but it can itself change availability for the control group or other customers.
An operations owner must assess that trade-off before treating a reserve as neutral.

### 11.2 Distinguish monitoring from proof

Monitoring alerts can protect customers and operations without establishing why an outcome changed.
For example, a sudden cancellation increase may trigger a pause even before the team knows whether the recommendation policy caused it.
An incident review can then inspect order mix, inventory feed health, and fulfilment capacity.
Conversely, a quiet alert dashboard does not establish that a new policy is commercially beneficial.
Estimate the prespecified effect and uncertainty before making that claim.

Repeatedly checking results and stopping only when an ordinary significance threshold is crossed can distort error rates.
If interim efficacy decisions are needed, the analytics owner should choose an appropriate sequential design before launch.
Safety monitoring can require immediate action regardless of the statistical readout.
Document which decisions are safety stops and which are planned statistical analyses.
If too little traffic is available to answer the question at a useful precision, say so.
The team may select a conservative operational policy without claiming the alternatives are equivalent.

## 12. Exercises

### Exercise 1 — Classify cold starts

A visitor has no account but selects a product category in an approved session control.
A newly listed item has complete attributes but no impressions.
A second item has 400 impressions from another country, but no local delivery approval.

**Deliverable:** Classify each relevant user/item state and identify one usable signal and one missing or disallowed signal.
**Acceptance criteria:** Distinguish user-cold from item-cold; do not count foreign impressions as local delivery approval; distinguish permitted session use from persistent profiling.
**Time:** 10 minutes.

### Exercise 2 — Construct the fallback ladder

A four-slot home carousel needs recommendations during a feature-store outage.
The visitor has declined optional profile personalization, and the local inventory feed is stale.

**Deliverable:** Write a rung-by-rung decision flow, including the output when no inventory data is reliable.
**Acceptance criteria:** No prohibited profile or inferred identity; each candidate passes applicable eligibility; stale stock invokes a documented fail-safe; fewer than four items or a safe empty state is allowed.
**Time:** 20 minutes.

### Exercise 3 — Calculate the exploration budget

A forecast has 30,000 eligible impressions in 30 days.
The plan allocates 8% to exploration.
The baseline is forecast at USD 0.15 contribution per impression, and exploration at USD 0.11.

**Deliverable:** Calculate planned exploration impressions and the immediate opportunity-cost estimate; name two assumptions.
**Acceptance criteria:** Correct units and arithmetic; identify the result as a forecast comparison, not a causal effect.
**Time:** 10 minutes.

### Exercise 4 — Compare observed item rates

Item X receives 500 impressions, 45 clicks, 20 attributed orders, and USD 40 contribution.
Item Y receives 800 impressions, 56 clicks, 24 orders, and USD 48 contribution.

**Deliverable:** Calculate CTR, order rate, and contribution per impression for both items; explain whether X is proven superior.
**Acceptance criteria:** Correct denominators and units; mention uncertainty and differences in exposure or position.
**Time:** 15 minutes.

### Exercise 5 — Audit propensity logging

A service logs the winning item's model score but not its eligible competitors, position, or randomization probability.
Analysts propose inverse propensity weighting to estimate a different policy.

**Deliverable:** Identify at least four missing pieces and state whether the estimate is supported.
**Acceptance criteria:** Include actual selection probabilities and support; distinguish a score from a probability; discuss slate dependencies and missing data.
**Time:** 15 minutes.

### Exercise 6 — Design a launch record

A new spare part arrives with an uncertain compatibility attribute and 25 units in one warehouse.
A wrong recommendation could interrupt a customer's equipment operation.

**Deliverable:** Produce a launch checklist, exposure plan, owners, graduation rule, and rollback trigger.
**Acceptance criteria:** Compatibility is a hard gate; no exposure until verified; stock, customer harm, evidence maturity, and owner accountability are addressed.
**Time:** 30 minutes.

### Exercise 7 — Apply purpose boundaries

Consider four requests: no identifier, an explicit session choice, a previously consented profile with revocation, and an unavailable authorization service.

**Deliverable:** For each, specify permitted decision inputs only after an organizational privacy decision, a fallback, and a logging/retention implication.
**Acceptance criteria:** No assumption that an identifier or consent is always required; revocation is propagated as applicable; unknown optional authorization fails closed; no fingerprinting workaround.
**Time:** 20 minutes.

### Exercise 8 — Critique a commercial proposal

A supplier offers a payment for a guaranteed top slot for its new product.
The product has uncertain stock and the team wants to include the placement in an organic exploration experiment.

**Deliverable:** Write a one-page recommendation to product, commercial, analytics, and operations owners.
**Acceptance criteria:** Address contractual and disclosure review, stock verification, clear labeling where required, separate treatment/logging, customer relevance, counterfactual integrity, and stop authority.
**Time:** 25 minutes.

## 13. Self-check answers

### Exercise 1

The visitor is user-cold for persistent preference history but has an explicit current-session category choice.
That choice can be used only under the approved purpose and permission rules; it is not automatic permission for a durable profile.
The newly listed item is item-cold on behavioral response despite complete attributes.
Its validated attributes can support content-based matching, but zero exposure is not evidence of zero customer interest.
The foreign item has some behavioral history, but transferability is uncertain and it lacks a local delivery approval.
It must not be shown as deliverable locally until the relevant gate passes.

### Exercise 2

A strong ladder first applies the permission decision, then uses a verified approved nonpersonal catalogue source.
If the local inventory feed is stale, use an independently verified approved fallback only if its freshness and delivery policy meet the surface's requirements.
Otherwise show a safe empty state or category navigation without making an unsupported stock promise.
Do not use the unavailable feature store as a reason to recover a declined profile from another service.
A four-slot visual design is subordinate to eligibility.

### Exercise 3

`30,000 × 0.08 = 2,400` planned exploration impressions over 30 days.
`2,400 × (USD 0.15 − USD 0.11) = USD 96` estimated immediate opportunity cost.
The calculation assumes the forecast traffic occurs and that both per-impression contribution estimates are applicable to the same eligible opportunities.
It also assumes no additional incremental operational cost or inventory interference.
The USD 96 is not a measured treatment effect or a commitment.

### Exercise 4

X: CTR `45 / 500 = 9%`, order rate `20 / 500 = 4%`, contribution `USD 40 / 500 = USD 0.08` per impression.
Y: CTR `56 / 800 = 7%`, order rate `24 / 800 = 3%`, contribution `USD 48 / 800 = USD 0.06` per impression.
X has higher observed rates in this sample.
That does not prove superiority: the rates have sampling uncertainty, and exposure, audience, position, timing, and attribution may differ.
A suitable randomized comparison and mature cost data are needed for a causal policy decision.

### Exercise 5

The winning score does not reveal the probability of selection under a policy.
At minimum, recover the eligible action set, actual conditional action probabilities, policy version, position and prior slate selections, assignment context, and outcome linkage.
Check whether each action of the proposed policy had nonzero logging probability in relevant contexts.
Without those facts the proposed inverse-propensity estimate is not supported.
Even after recovery, inspect extreme weights, missingness, and estimator assumptions.

### Exercise 6 rubric

A strong design blocks the item until compatibility and operational approval are verified.
It specifies the responsible compatibility authority, inventory owner, target audience, exposure cap, and safe fallback.
It defines an exposure and outcome review with an owner, uncertainty, and appropriate delay.
A safety complaint or compatibility discrepancy triggers immediate pause and investigation.
A weak design merely recommends the item to random customers until it has a fixed number of clicks.

### Exercise 7 rubric

A strong answer separates identity availability from permission and states that the authorized nonpersonal fallback is independently reviewed.
The explicit session choice is used only for an approved purpose and retained no longer than permitted.
Revocation triggers future restrictions and applicable historical-data workflows; an authorization outage invokes the approved fail-closed optional-processing rule.
Operational logging must itself have a documented purpose, access control, and retention.
A weak answer treats absence of an account as permission to fingerprint or assumes consent is the only possible legal basis in every jurisdiction.

### Exercise 8 rubric

A strong answer separates paid placement from the randomized organic policy and records the commercial override.
It requires stock and delivery verification, relevance and safety gates, applicable disclosure review, and a named pause owner.
It quantifies opportunity cost against a defensible counterfactual rather than treating the supplier fee as pure profit.
It considers the impact of shared inventory on both experiment arms and documents contract obligations before promising impressions.
A weak answer either silently mixes the placement into the experiment or assumes all paid placement is automatically prohibited.

## 14. Interview questions

1. **How do new-user and new-item cold starts differ?**
   A strong answer distinguishes missing preference evidence from missing response evidence and recognizes the both-cold case.
   A weak answer treats all missing interactions as dislikes.
2. **How would you serve a visitor who declines profiling?**
   A strong answer uses an independently approved nonpersonal policy and honors purpose and retention restrictions.
   A weak answer reconstructs identity through device fingerprints.
3. **What belongs before ranking?**
   A strong answer names safety, eligibility, serviceability, availability, and permission gates with freshness rules.
   A weak answer gives unsafe items a lower score rather than excluding them.
4. **Why explore a new item?**
   A strong answer explains the exposure-feedback loop and a bounded, relevant experiment.
   A weak answer assumes an unshown item has failed.
5. **What is a propensity?**
   A strong answer gives the actual conditional selection probability and explains eligible-set and slate dependence.
   A weak answer points to a model score.
6. **Can inverse propensity weighting evaluate every item?**
   A strong answer discusses positivity, variance, consistency, and logging quality.
   A weak answer claims it recovers outcomes for actions never exposed.
7. **How do you choose an initial item prior?**
   A strong answer defines comparable data, strength, calibration, and sensitivity to transfer.
   A weak answer calls the prior the item's observed conversion.
8. **When does an item graduate?**
   A strong answer combines relevant exposure, mature outcomes, uncertainty, operational guardrails, and ownership.
   A weak answer uses age alone.
9. **How do you compare a cold-start policy commercially?**
   A strong answer specifies an incremental contribution counterfactual, costs, delayed outcomes, and uncertainty.
   A weak answer reports clicks or gross sales only.
10. **What can invalidate a randomized trial with inventory constraints?**
    A strong answer identifies shared-stock interference, spillovers, and availability differences and proposes monitoring or design changes.
    A weak answer assumes random assignment makes every operational effect disappear.

## 15. Key terms

- [User cold start](#4-definitions-and-vocabulary): insufficient permitted user evidence for a decision.
- [Item cold start](#4-definitions-and-vocabulary): insufficient item response evidence in the relevant setting.
- [System cold start](#4-definitions-and-vocabulary): insufficient local history for a surface or market.
- [Eligibility](#53-establish-hard-eligibility-before-relevance): hard decision on whether an item may be shown.
- [Candidate](#4-definitions-and-vocabulary): an item considered before final selection.
- [Impression](#57-design-the-evidence-and-logging-contract): defined rendered exposure opportunity.
- [CTR](#63-illustrative-observed-outcomes): clicks divided by impressions, interpreted with exposure context.
- [Conversion](#4-definitions-and-vocabulary): a specified downstream action with a stated denominator.
- [Contribution](#63-illustrative-observed-outcomes): revenue less defined variable costs.
- [Exploration](#56-set-a-bounded-exploration-policy): deliberate allocation to learn.
- [Exploitation](#4-definitions-and-vocabulary): selection based on current estimated value.
- [Propensity](#57-design-the-evidence-and-logging-contract): probability of selection under the actual policy.
- [Positivity](#57-design-the-evidence-and-logging-contract): nonzero logging support for actions being evaluated.
- [Position bias](#10-failure-modes-and-red-flags): response changes caused by placement.
- [Exposure bias](#3-the-core-mental-model): selection affects which outcomes become observable.
- [Fallback ladder](#54-choose-the-strongest-permitted-user-evidence): approved ordered degradation across policies.
- [Context](#4-definitions-and-vocabulary): request-time information, subject to privacy review.
- [Cohort](#4-definitions-and-vocabulary): a group defined by permitted shared attributes.
- [Explicit preference](#54-choose-the-strongest-permitted-user-evidence): a user-provided choice, not blanket consent.
- [Session signal](#52-separate-absence-uncertainty-and-prohibition): a bounded current-interaction event.
- [Content feature](#55-create-an-item-launch-record): validated item attribute used for matching.
- [Prior](#64-smoothing-and-its-limits): assumption or borrowed evidence before local observations.
- [Smoothing](#64-smoothing-and-its-limits): combining a prior with observed outcomes.
- [Graduation](#58-evaluate-the-choice-and-graduate-cautiously): moving from provisional to ordinary ranking consideration.
- [Guardrail](#10-failure-modes-and-red-flags): monitored boundary with a defined response.
- [Consent](#82-regulatory-considerations): a permission mechanism where required or chosen as the legal basis.
- [Purpose limitation](#82-regulatory-considerations): restricting data use to specified and compatible purposes where applicable.
- [Retention](#81-data-contract): approved duration and handling of stored data.
- [Backfill](#54-choose-the-strongest-permitted-user-evidence): lower-rung items used when a slate is short.
- [Slate](#56-set-a-bounded-exploration-policy): ordered collection of items displayed together.
- [Incrementality](#63-illustrative-observed-outcomes): difference from the relevant counterfactual.
- [Available to promise](#53-establish-hard-eligibility-before-relevance): stock availability estimate for a specified fulfilment context.

## 16. Sources

1. European Parliament and Council, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5–7 and 12–22. Official legal text; check current consolidated law and national implementation.
2. European Parliament and Council, [Directive 2002/58/EC on privacy and electronic communications](https://eur-lex.europa.eu/eli/dir/2002/58/oj). Official framework; national implementation and subsequent amendments matter.
3. California Department of Justice, [California Consumer Privacy Act](https://oag.ca.gov/privacy/ccpa). Official overview; consult current statutes and regulations for applicability.
4. GS1, [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/). Identifier allocation guidance; not a safety or stock certification.
5. Carlos A. Gomez-Uribe and Neil Hunt, [The Netflix Recommender System: Algorithms, Business Value, and Innovation](https://doi.org/10.1145/2843948), *ACM Transactions on Management Information Systems*, 2015. Company-authored description of a media recommender.
6. Lihong Li, Wei Chu, John Langford, and Robert E. Schapire, [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://doi.org/10.1145/1772690.1772758), *Proceedings of the 19th International Conference on World Wide Web*, 2010. Research case on randomized recommendation and evaluation.
7. Yifan Hu, Yehuda Koren, and Chris Volinsky, [Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22), *IEEE International Conference on Data Mining*, 2008. Research on implicit feedback and confidence.
8. Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims, [Recommendations as Treatments: Debiasing Learning and Evaluation](https://proceedings.mlr.press/v48/schnabel16.html), *International Conference on Machine Learning*, 2016. Research on selection bias and propensity-based methods.
9. NIST, [Privacy Framework](https://www.nist.gov/privacy-framework). Voluntary risk-management resource, not a substitute for applicable law.

Sources were selected for concepts and official context; none validates the synthetic example's numerical estimates.
Legal references are educational and should be checked for amendments and local applicability before use.

## 17. Links

- [Module 24 index](_index.md)
- [Module 24 survey](../../../course/24_recommendation_systems_personalization.md)
- [Recommendation tasks](01_recommendation-tasks.md)
- [Popularity and content-based baselines](02_popularity-content-baselines.md)
- [Collaborative filtering](03_collaborative-filtering.md)
- [Two-tower retrieval](04_two-tower-retrieval.md)
- [Sequential and session models](05_sequential-session.md)
- The survey's lab sections provide the module's practical evaluation context.
