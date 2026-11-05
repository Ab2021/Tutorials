# Business Constraints and Re-Ranking

> Module 24 — Recommendation Systems and Personalization  
> Topic 9 of 11  
> Estimated study time: 4 hours  
> Prerequisites: topics 01, 02, 07, and 08  
> Version 2026.5

## 1. Why this topic exists

A recommender can predict clicks accurately and still fail a customer.
It may suggest an item that cannot be delivered, present five nearly identical products, or promote a high-margin product that is a poor substitute.
It may also expose a restricted product to a customer who is not eligible to buy it.
These are not merely cosmetic ranking defects: they can create cancelled orders, unsafe purchases, supplier disputes, and regulatory exposure.

Re-ranking is the decision layer between a scored candidate set and the ordered list that the customer sees.
It combines relevance with current operational facts and explicit commercial rules.
It does not replace retrieval, product safety controls, checkout validation, or customer choice.
A product not retrieved cannot be rescued by a re-ranker, and an availability forecast is not a reservation.

The merchandising owner wants a commercially sustainable assortment.
The customer wants useful choices, accurate information, and a realistic delivery promise.
The supply-chain owner wants recommendations that do not create avoidable fulfillment failures.
The policy owner wants restrictions enforced consistently and evidenced.
These goals can conflict, so the team needs a documented priority order and a way to resolve infeasibility.

Consider a grocery app with three recommendation slots.
A click model puts three products from one brand first.
One is unavailable for the customer's delivery window, and another is age restricted.
A margin-only adjustment may lift a third product even if it is a poor fit for the meal.
A defensible system first establishes eligibility, then selects a useful slate while respecting the remaining assortment rules.
It logs the decision and rechecks stock at the point where an order is committed.

The practical decision is not whether to optimize margin or relevance in isolation.
It is which combinations of products may be shown, what value they create for customers and the business, and who can change the rules.
This unit gives a small, auditable procedure and identifies when a more sophisticated optimizer is needed.

## 2. Learning outcomes

After this unit you can:

1. **Distinguish** retrieval, scoring, eligibility filtering, slate selection, and order confirmation; exercise 1 checks the boundary.
2. **Classify** an availability, safety, diversity, or margin requirement as hard or soft and name its owner; exercise 2 checks the classification.
3. **Calculate** a comparable relevance-and-margin score and a feasible slate from candidate data; exercises 3 and 4 check the arithmetic.
4. **Diagnose** an infeasible slate and apply a preapproved relaxation or fallback without bypassing safety rules; exercise 5 checks the decision.
5. **Critique** a commercially biased proposal using customer-value and distributional guardrails; exercise 6 checks the reasoning.
6. **Design** logging, experiments, ownership, and rollback for a policy change; exercises 7 and 8 check the operating design.
7. **Explain** the limitations of a greedy implementation and when an exact or constrained optimizer is warranted; exercise 9 checks this judgment.

## 3. The core mental model

```text
request + context + policy version
       |                     
       v                     
retrieval --> candidate scores --> hard eligibility --> feasible candidates
                   |                         |                 |
             relevance estimate       reasons for exclusion    v
                                                     slate selection
                                         relevance + margin + diversity
                                                               |
                                        no feasible full slate? |
                                        /                       \
                           approved fallback                 ordered slate
                                        \                       /
                                         --> impression + audit log
                                                      |
                                  fulfillment recheck and outcomes
```

A *candidate* is a product that retrieval nominated for this request.
A *slate* is an ordered collection of products selected together, not a set of independent decisions.
For example, showing two similar yogurts may be less useful than showing one yogurt and a suitable alternative even if each yogurt scores well alone.

Separate the decisions into three kinds.
First, hard constraints define what may be shown: legal eligibility, contractual exclusions, and the minimum data needed to make a safe availability claim.
Second, soft preferences trade off eligible options: margin, relevance, variety, and supplier exposure can enter a score or a bounded objective.
Third, slate constraints concern the list as a whole: a maximum per brand, a minimum category mix, or a limit on repeated variants.
A requirement should be hard only if the business can identify its owner, exact semantics, enforcement point, and behavior when no solution exists.

Hard does not mean that a source is infallible.
If an age-verification service is unavailable, the restriction can still be mandatory: the system must choose its approved unknown-status behavior rather than guess eligibility.
If the inventory feed is late, the system must use an explicit freshness rule; it cannot make stale data current by assigning a high score.

The sequence matters.
A finite relevance or margin bonus must never make an ineligible item eligible.
A diversity penalty must not push the system toward an irrelevant or unsafe product simply to increase variety.
Finally, the displayed slate is a forecast-informed suggestion, not a commitment that inventory has been allocated.
The checkout or reservation service remains authoritative for committing stock.

**Decision boundary:** re-ranking chooses among the candidates and facts it receives.
It cannot repair systematically missing retrieval candidates, incorrect product attributes, invalid consent, or a biased upstream label on its own.
Those are separate controls with separate owners.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Request | One recommendation opportunity for a context and time | Treating all surfaces as one decision |
| Candidate | Item retrieved for possible display | Assuming a candidate is eligible |
| Retrieval | Finding a manageable candidate set from a catalogue | Calling final ordering retrieval |
| Relevance score | Estimated customer fit on a defined scale | Treating an uncalibrated score as a probability |
| Calibrated probability | Predicted event frequency conditioned on an event and horizon, in 0–1 | Assuming calibration implies causal lift |
| Slate | Ordered items displayed in one request | Evaluating every item independently |
| Slot | A display position in a slate | Assuming all positions receive equal attention |
| Hard constraint | A rule that every returned slate must satisfy | Representing a legal ban as a small penalty |
| Soft preference | A tradable objective or penalty | Calling a target mandatory without a fallback |
| Eligibility | Item-level permission to consider a candidate for a request | Equating positive stock with eligibility |
| Feasibility | Existence of a slate satisfying all hard rules | Assuming a solver can always produce K items |
| Infeasible | No slate meets the specified hard rules | Treating a timeout as proof of infeasibility |
| Margin | Defined contribution per sale, here USD per unit | Mixing gross revenue and profit |
| Expected margin | Purchase probability times contribution, under stated assumptions | Multiplying by an arbitrary rank score |
| Diversity | A defined difference across displayed items, such as brand or category | Assuming more variety always improves utility |
| Redundancy penalty | Cost for similarity to items already selected | Treating any brand match as an objectively bad outcome |
| Availability | Ability to fulfill for a location, channel, quantity, and time | Using global on-hand as delivery availability |
| Freshness | Age of a data observation at decision time | Assuming a recent feed guarantees a sale |
| Safety stock | Inventory withheld from general promise, in units | Counting it as freely sellable stock |
| Policy version | Immutable identifier of the rules applied | Logging a mutable rule name alone |
| Fallback | Approved behavior when the preferred decision cannot be made | Silently dropping legal requirements |
| Relaxation | Ordered weakening of a designated constraint | Relaxing an unrelaxable restriction |
| Exposure | Opportunity for an item or supplier to be seen, often position weighted | Counting retrieval as customer exposure |
| Customer value | Benefit to the customer, measured by useful outcomes and harms | Equating clicks with welfare |
| Guardrail | Metric or rule limiting unacceptable harm during optimization | Assuming a dashboard alone enforces a rule |
| Experiment | Assignment of traffic to variants to estimate effects | Comparing pre/post periods as if randomized |
| Propensity | Probability of an action under the logging policy | Logging only a score and calling it a propensity |
| Reservation | Allocation of stock under an authoritative inventory contract | Calling an availability prediction a reservation |

## 5. The mechanics, step by step

### 5.1 Define the decision and the service boundary

**Input:** surface, customer context, candidate set, and the business problem.
**Output:** a request contract and the decision the service is authorized to make.

1. Specify the surface: for example, three product cards on a grocery item page.
2. Define the customer task: substitute for a missing product, find a complement, or discover a new product.
3. Name the request-time facts: delivery zone, promised window, channel, basket, age status when lawfully available, and currency.
4. Define K, the maximum number of slots, and whether a shorter slate is permissible.
5. Define latency and timeout behavior; a recommendation must not block checkout indefinitely.
6. State what happens after display: checkout independently validates price, restrictions, and inventory.

A substitution surface needs functional compatibility and customer preference, while a discovery surface can accept a wider assortment.
A candidate with high margin and low relevance is not automatically a substitute.
A business target is a desired aggregate outcome, not permission to alter an individual customer's eligibility.

### 5.2 Establish rights, duties, and decision owners

**Input:** applicable laws, contractual terms, internal product rules, and decision context.
**Output:** a versioned rule register with an accountable owner for every rule.

For each rule, record the scope, input fields, effective interval, severity, and whether unknown input means exclude, show a neutral alternative, or invoke another approved process.
Legal and compliance owners interpret applicable restrictions; catalog operations maintains product classifications; the inventory owner defines sellability and freshness; merchandising proposes commercial priorities; product and data science own the customer-value measurement design.
An engineering owner implements enforcement and provides auditable evidence.
None should be able to bypass another owner's mandatory control through an undocumented score weight.

Use a policy register rather than burying a rule in a model coefficient.
For each commercial override, require a named approver, reason, start and end time, affected surfaces and suppliers, expected impact, and rollback trigger.
Separate a paid placement from a recommendation when required by local rules and product design; do not imply organic personalization if a material commercial relationship changes the result.
An ethical review is especially important where customers cannot easily assess the trade-off.

### 5.3 Validate candidate and context data

**Input:** retrieved item identifiers and context attributes.
**Output:** typed, timestamped facts and explicit unknowns.

Check item identifiers against the active catalog and deduplicate exact item IDs.
Resolve variant relationships deliberately: a size or flavor may be a useful option in one task and redundant in another.
Validate currency, unit of measure, timestamps, and price basis before scoring.
A missing margin is not zero margin unless the commercial policy expressly defines that fallback.
A missing stock record is not positive stock.

For inventory, join the item to the correct fulfillment node, quantity, channel, and delivery window.
Record observation time and the maximum accepted age for that surface.
Sellable-to-promise may subtract reserved, damaged, safety, or otherwise unavailable units; the exact formula belongs to the inventory owner.
A store count cannot establish availability from a different warehouse.
A source update can still be wrong; freshness limits reduce a risk but do not guarantee truth.

Avoid leaking unnecessary personal data into the audit log.
Use a request identifier and the minimum context needed for diagnosis; keep retention and access aligned with the organization's privacy controls.
Log the policy decision and inputs at an appropriate level, not raw sensitive attributes by default.

### 5.4 Apply non-negotiable item-level eligibility

**Input:** candidates and validated request-time facts.
**Output:** eligible candidates and reason codes for exclusions.

A useful ordering is:

1. Remove inactive or prohibited catalog entries.
2. Enforce product, geography, customer, and channel restrictions with the approved unknown-status handling.
3. Exclude items incompatible with the task when compatibility is mandatory, such as an unsafe substitute for a declared constraint.
4. Apply the availability and freshness policy for the promised location, time, and quantity.
5. Remove duplicates and any contractual exclusions that apply to this surface.

Each rule should emit a stable reason code and rule version.
One item can fail multiple checks; choose whether logs record all failures or just the first and document the consequence for diagnostics.
When a restriction input is missing, do not infer approval from absence of evidence.
A compliance owner determines the safe behavior for the actual jurisdiction and product.

Hard filtering should be evaluated both for correctness and candidate coverage.
An unusually large increase in `inventory_unknown` could mean a broken feed rather than a genuinely empty catalog.
An exclusion rate by region or device may reveal a data-quality issue that disproportionately harms a group.
Only examine group outcomes with a lawful basis and appropriate privacy controls.

### 5.5 Express slate rules precisely

**Input:** eligible items and approved assortment policy.
**Output:** a mathematical or executable definition of valid slates.

Let binary variable x(i,p) equal 1 if candidate i is placed at position p, and 0 otherwise.
Let K be the maximum number of positions and E be the eligible set.
The basic rules are:

- At most one item in each position: sum over i of x(i,p) <= 1.
- At most one position for each item: sum over p of x(i,p) <= 1.
- No item outside E can be selected.
- A full-slate requirement, if appropriate, sets the number selected to K.
- A brand cap limits the sum of x(i,p) for a brand across all positions.
- A category minimum sets a lower bound for the relevant category across the slate.

A maximum of one item per brand is not the same as a penalty for repeating a brand.
The first prohibits a second item, even when the alternatives are poor; the second allows a second item if its incremental value outweighs the penalty.
A minimum number of distinct categories requires a definition of category hierarchy and multi-category assignment; otherwise a solver and a report can disagree.
Do not add a hard minimum just because a dashboard has a target.

If position matters, specify whether a rule applies to the entire slate or only the first positions.
A supplier exposure target accumulated across a week is a portfolio constraint, not necessarily a per-request quota.
Per-request quotas can cause customer harm if a supplier has no relevant item for a particular request.

### 5.6 Choose a defensible objective

**Input:** eligible candidates, utility estimates, commercial contribution, and slate rules.
**Output:** a comparable objective with units, calibration, and approved trade-offs.

One illustrative objective is

```text
maximize sum over slots p and eligible items i:
    x(i,p) * position_weight[p] * (relevance[i] + lambda * normalized_margin[i])
    - diversity_penalty(slate)
```

Here relevance and normalized margin must be dimensionless before addition, lambda is a documented trade-off, and position weights are nonnegative if used.
The formula is an example, not a claim that scores are comparable by nature.
Normalization must specify the reference distribution, clipping, date, and what happens outside its range.
Otherwise a change in catalog mix can change the effective margin weight without any policy approval.

A more economic formulation can use expected contribution: probability of the defined purchase event times contribution in USD per unit.
The contribution definition might subtract product cost, fulfillment, returns, and incremental promotion expense, but only if each component is measured on a comparable basis.
Do not include revenue twice or treat a click probability as a purchase probability.
Do not subtract fixed costs per item unless that accounting allocation is appropriate for the decision.
Expected contribution still omits customer outcomes that are difficult to monetize; preserve explicit relevance and harm guardrails.

A simple sequential redundancy penalty may charge a fixed amount when a candidate's brand already appears in the selected slate.
This is interpretable but depends on selection order and does not represent all forms of similarity.
A richer metric might account for category, price range, attributes, or embeddings, subject to validation.
Some similarity is useful: a customer comparing sizes may welcome several variants.
Measure whether diversity actually improves task completion rather than maximizing it for its own sake.

### 5.7 Check feasibility before deciding what to relax

**Input:** eligible set, K, and all slate constraints.
**Output:** feasible slate, explicit infeasibility result, or an approved fallback.

A necessary condition for a full K-item slate is at least K distinct eligible items.
With a one-per-brand cap, at least K eligible brands are also necessary.
These checks are not sufficient when category, position, and other constraints interact.
A capable solver can determine feasibility for the encoded problem, provided it finishes and the inputs are valid.
A timeout or an algorithm's failure to find a slate is not a mathematical proof that no slate exists.

Before launch, business and policy owners should approve a fallback ladder, for example:

1. Return the optimal or approximate slate under all hard rules when available.
2. If a designated assortment target is soft, relax only that target in the preapproved order and record the change.
3. If fewer than K valid items remain, return a shorter slate when the surface supports it.
4. If no item qualifies, omit the module or present a separately approved non-product message.
5. If a required policy or inventory dependency is unknown, follow its documented safe behavior; do not bypass it to fill slots.

A contractual brand cap that is genuinely mandatory is not relaxable merely because the slate is short.
A team may elect to make a merchandising brand target soft from the outset, but that is a policy decision, not an emergency algorithmic exception.
Distinguish policy infeasibility from missing inputs, a solver timeout, a stale catalog, and an upstream retrieval shortage.
Each requires a different owner and remedy.

### 5.8 Select, order, and validate the slate

**Input:** valid candidates, rules, objective, and compute budget.
**Output:** ordered slate with a decision trace.

For small candidate sets, enumerate feasible ordered slates or use an integer optimization solver.
For larger sets, teams often use greedy selection, beam search, or other approximations.
A greedy algorithm scores the remaining candidates at each position conditional on the already selected items, then picks the highest incremental score.
It is easy to explain but can choose an early item that prevents a better or even feasible later combination.
A local optimum is not proof of the best slate.

For instance, if the only eligible category-B item shares a capped brand with the greedy first pick, the first pick may make a hard category-B minimum impossible despite a feasible alternative first pick.
A greedy implementation must either account for future feasibility, use backtracking, or avoid claiming to enforce that minimum.
An exact method is appropriate when the constraints and decision value justify its cost; otherwise compare approximation quality and constraint violations with an exact solution on representative small samples.

After selection, independently validate every hard item and slate rule.
Use deterministic tie-breaking, such as item ID, so that the same inputs and policy produce the same result unless randomized exploration is intentional.
For exploration, record the actual assignment probabilities and seed or decision ID needed to reconstruct the policy.
An arbitrary relevance score is not a probability of selection.

### 5.9 Log, monitor, and learn

**Input:** selection trace, impressions, downstream events, and policy versions.
**Output:** audit trail and measured customer and commercial outcomes.

Log the request and experiment identifiers, surface, timestamp, candidate IDs, scores or sufficient reconstruction inputs, eligibility reasons, displayed order, fallback reason, and versions of catalog, features, models, and policy.
Where randomized selection is used, record actual propensities and the eligible action set to the extent required by the evaluation design.
Log an impression only when the item had a defined opportunity to be seen; distinguish rendered cards from candidates that were never displayed.
Keep order, cancellation, substitution, return, and complaint events joinable with privacy-appropriate identifiers.

Monitor hard-rule violations as incidents, not merely as a decline in average metrics.
Monitor eligibility yield, short-slate rate, missing and stale stock, latency, solver timeouts, coverage, category and supplier exposure, and customer outcome changes.
Set denominators and windows before interpreting a rate.
For example, a short-slate rate is short slates divided by eligible requests, not divided by impressions.
Segment by surface, fulfillment region, new versus returning customers, and other appropriate operational dimensions.

A margin lift measured only on purchased items can hide a fall in conversion or an increase in cancellations.
Report contribution per eligible request as well as per order, with a consistent attribution window and a treatment of returns.
Customer guardrails can include successful fulfillment, task completion, complaint rate, relevant-item coverage, and price or delivery-promise accuracy.
No single proxy is a complete measure of customer value.

### 5.10 Release and govern changes

**Input:** candidate policy and offline evidence.
**Output:** approved rollout, monitoring plan, and rollback authority.

Replay historical requests using point-in-time inputs and compare the proposed slate with the current policy.
Check constraint compliance and representative slices; offline ranking metrics alone cannot establish incremental business impact because historic exposure is selective.
Use an online randomized experiment when lawful and operationally appropriate, with allocation at a unit that limits interference.
Predefine primary outcome, guardrails, analysis window, minimum detectable effect or precision target, and stopping rules.

Ramp gradually with a kill switch and a known safe policy.
The policy owner approves restrictions, the inventory owner signs off on the availability contract, merchandising approves commercial parameters, and product and analytics own experiment interpretation.
The on-call engineering owner can roll back immediately when a hard control fails.
Escalate conflicts rather than allowing commercial targets to override safety by default.
Periodic reviews should check whether suppliers or customer segments bear concentrated harm and whether overrides have expired.

### 5.11 Turn commercial objectives into an accountable trade-off

A coefficient is a policy choice, even if an optimizer later tunes it.
The owner should be able to explain what change the coefficient is intended to produce.
Start by writing the business decision in plain language.
For example: “Among useful, available complements, favor higher-contribution products modestly without increasing cancellation or degrading task completion.”
That statement is more useful than “maximize margin,” because it specifies the permitted decision space.

Next, define the commercial outcome at the request level.
If 1,000 eligible requests lead to 100 orders and USD 250 of attributable contribution, the observed contribution is USD 0.25 per eligible request.
If another policy leads to 80 orders and USD 240 of contribution, its contribution per eligible request is USD 0.24.
Although contribution per order rises from USD 2.50 to USD 3.00, the second policy has lower contribution per eligible request.
These observations are arithmetic, not causal conclusions unless the comparison design supports causality.
Return costs and delayed cancellations should be included when their attribution window is complete.

Write down which costs are incremental to the recommendation.
A product cost is generally relevant to unit contribution, but a warehouse's fixed rent may not change when a different product is recommended.
Incremental shipping expense may matter when products have different handling requirements.
A supplier-funded promotion may change the net contribution, but its accounting treatment and disclosure consequences must be reviewed.
If the data cannot support a defensible per-item estimate, use an explicitly limited proxy and report its limitations.

Do not assume the coefficient learned on one surface transfers to another.
On a replacement surface, the customer has a specific need and a poor match may cause abandonment.
On a discovery surface, a broader assortment may be welcome.
On a checkout surface, an irrelevant addition may delay completion.
Estimate each setting separately or demonstrate that a shared policy remains acceptable across them.

Before changing a weight, specify:

1. The current and proposed policy versions.
2. The measured quantity and its unit.
3. The target population and surfaces.
4. The permitted range of the coefficient.
5. The customer guardrails and their denominators.
6. The approvers and review date.
7. The experiment and rollback plan.

### 5.12 Measure diversity as a task property

Diversity has no single natural unit.
Count distinct brands if the task is intended to expose different manufacturers.
Count distinct categories if the task is to assemble a meal.
Measure attribute distance if the customer needs a range of product sizes or flavors.
Measure price dispersion if affordability and comparison matter, but do not infer affordability from dispersion alone.
Each measure can move independently of the others.

Consider the two slates below:

| Slate | Products | Distinct brands | Distinct categories |
|---|---|---:|---:|
| S1 | Three pasta brands | 3 | 1 |
| S2 | Pasta, sauce, vegetables from one brand | 1 | 3 |

S1 may help a customer compare pasta alternatives.
S2 may be more appropriate for a meal-building task.
Neither is universally more diverse without specifying the dimension.
The category labels must have a versioned hierarchy so that a reporting change does not appear to be an algorithmic effect.

An intra-list similarity measure averages similarity over distinct item pairs in the slate.
For K items, there are K × (K − 1) / 2 unordered pairs.
If a three-item list has pair similarities 0.8, 0.4, and 0.3 on a 0–1 scale, its average similarity is 0.50.
A diversity score of 1 − average similarity would be 0.50 under that chosen definition.
This score is undefined for a single-item slate unless a reporting convention is specified.
Report the proportion of requests excluded from such an aggregate.

The similarity function itself is a modeling choice.
An embedding may encode popularity or supplier identity rather than functional substitutability.
Two flavors from the same brand might be useful alternatives for one customer and redundant for another.
Validate the similarity measure with task-specific judgments and actual customer outcomes.
Do not use sensitive characteristics in a similarity function without an appropriate governance review.

If the optimization adds a diversity bonus, test for boundary behavior.
A very high bonus can make a low-relevance item outrank a useful alternative purely because it is unusual.
A very low bonus may have no effect on the actual slate.
Plot both customer outcomes and slate composition over the proposed weight range.
The approved setting should survive realistic catalog changes rather than work only for a single replay sample.

### 5.13 Separate inventory prediction, promise, and allocation

The inventory service may expose an on-hand count, available-to-promise count, or a probabilistic availability estimate.
These are different facts.
On-hand is a physical or recorded quantity at a location.
Available-to-promise is a business-defined quantity after appropriate commitments and protections.
A predicted probability represents uncertainty about future fulfillment, not an allocation.
The inventory owner must define which field the recommendation service can use.

For example, suppose a node reports 12 units on hand, 5 reserved, 2 damaged, and 1 protected as safety stock.
If those buckets are mutually exclusive and the approved formula subtracts all three, available-to-promise is 12 − 5 − 2 − 1 = 4 units.
If damaged stock is already excluded from the on-hand field, subtracting it again would double count the deduction.
The numbers alone are insufficient without the data contract.
The re-ranker must not independently invent the accounting formula.

Time matters as much as quantity.
A product available for pickup now may not be available for delivery tomorrow.
A warehouse replenishment forecast is not a confirmed receipt.
An item may be geographically eligible but have an unacceptable delivery window.
The request should carry the time window and fulfillment method used to query the inventory decision.

Concurrent demand can exhaust stock immediately after the recommendation is displayed.
The risk increases for low-stock items, popular items, and long periods between impression and checkout.
An operational threshold might withhold an item below a certain buffer, but the inventory owner must approve and validate it.
The right threshold depends on the cost of lost exposure, cancellation, and holding stock.
Do not claim that the threshold guarantees availability.

When a feed is stale, three distinct situations need different handling:

- A delayed update may conceal a stockout.
- A delayed update may conceal replenishment.
- A service outage may prevent determining whether either occurred.

All three create uncertainty, but they do not imply the same operational root cause.
Use a consistent reason such as `stale_stock` for the decision and more detailed telemetry for the feed owner.
If a separate verified availability source exists, use it only under an approved precedence and freshness contract.
Never silently replace an unavailable authoritative source with an unrelated global stock count.

### 5.14 Design the fallback as a decision tree

A fallback is part of the product experience, not an error message added after deployment.
It needs the same policy review and observability as the main path.
Start with the distinction between no eligible products and an unavailable dependency.
In the first case, the system has a valid empty result.
In the second, it may not know whether products are eligible.
Treating both as an ordinary empty catalog conceals an outage.

| Condition | Example response | Responsible owner |
|---|---|---|
| Enough eligible products and slate is feasible | Return the selected slate | Recommendation service |
| Fewer than K eligible products | Return a shorter slate if supported | Product owner |
| Eligible products exist but hard slate rules conflict | Return an approved shorter slate or omit the module | Policy and product owners |
| Optional assortment target conflicts | Apply the preapproved relaxation, with a reason | Merchandising owner |
| Mandatory restriction input unavailable | Use the approved unknown-status path | Compliance owner |
| Inventory source stale | Exclude or otherwise handle under its availability contract | Inventory owner |
| Solver exceeds time limit | Use a validated safe fallback, not an infeasibility claim | Engineering owner |
| No valid product can be displayed | Hide the module or use an approved neutral message | Product owner |

A shorter slate is not always possible.
Some interfaces have fixed card layouts or accessibility behaviors that assume a particular count.
The client and service owners must agree on empty, one-item, and partial-slate representations.
An empty card should not look like a product, an error, or a loading indicator indefinitely.
The fallback should also be tested with screen readers and on supported devices.

If the preapproved fallback is a cached slate, check every item against the current mandatory restrictions and availability policy before returning it.
A cached eligible result can become ineligible when a rule, product status, or stock fact changes.
If those checks cannot be performed, use the dependency-failure behavior rather than assuming the cached result is safe.
Record whether the returned slate came from the primary algorithm, an alternate policy, or a cache.

### 5.15 Audit the entire funnel

The re-ranker can change only what reaches it.
For each request cohort, count candidates at these stages:

1. Active items in the relevant catalog.
2. Items accessible to the retrieval index.
3. Items retrieved for the request.
4. Items with usable scoring features.
5. Items passing hard eligibility.
6. Items selected for the slate.
7. Items actually rendered.
8. Items clicked, purchased, fulfilled, returned, or cancelled.

These counts have different denominators.
For example, eligible-candidate yield is eligible candidates divided by retrieved candidates with a defined treatment of missing data.
Impression rate may be rendered items divided by selected items.
Purchase conversion per request is requests with a qualifying purchase divided by eligible requests.
Do not compare these rates as though they measure the same event.

Suppose a supplier has 100 active products, 50 indexed products, 20 retrieved products, 10 eligible products, and 2 displayed products in a particular aggregate.
The two displays are not enough evidence that the re-ranker alone caused low exposure.
The loss of 50 products before retrieval may be more important.
The counts may also represent different request opportunities, so the analysis must specify its unit and window.
Inspect product quality, geographic scope, stock, and customer relevance before interpreting supplier shares.

For every funnel step, assign an owner and an actionable alert.
The catalog owner investigates missing attributes.
The indexing owner investigates stale or absent catalog entries.
The retrieval owner investigates candidate recall.
The inventory owner investigates sellability and feed health.
The policy owner investigates unexpected exclusions.
The recommendation owner investigates selection and rendering differences.
This division prevents the last component in the pipeline from receiving every incident regardless of cause.

### 5.16 Make commercial governance inspectable

A governing record should let a reviewer answer “why was this product shown here?” without relying on the memory of an employee.
The record need not expose every internal model detail to every viewer.
It should establish what decisions were made, under which authority, with which inputs, and where more detail is held.
Access should be restricted according to the sensitivity of the underlying data.

For each policy change, retain:

- A concise purpose and expected customer effect.
- A list of surfaces, regions, and fulfillment contexts in scope.
- The current and proposed policy identifiers.
- The rule or coefficient that changes.
- The eligible products or suppliers affected, if scoped.
- The beginning and expiry of the change.
- The source of any commercial consideration.
- The required legal and disclosure review.
- The owner of the customer and supplier outcome analysis.
- A rollout schedule, kill switch, and rollback owner.
- The final decision and evidence used to make it.

An expiry is important because a temporary boost can become a permanent hidden policy through neglect.
The default at expiry should be an explicitly approved state, not an undefined continuation.
Changes to the weight, eligible suppliers, or period should create a new version and approval record.
Monitoring should distinguish organic exposure from paid placement or other interventions where the system supports that distinction.

Review conflicts of interest in the approval process.
The person with a supplier revenue target should not be the only person judging customer harm from the placement.
An independent product, compliance, or governance reviewer can challenge the proposal according to local process.
That reviewer needs enough information to assess both the benefit and the downside.
Commercial confidentiality can restrict distribution, but should not prevent the authorized reviewer from seeing the decision basis.

### 5.17 Plan an experiment around constraints

Randomization compares eligible policies; it does not license a treatment that violates a mandatory rule.
Both control and treatment must enforce the same non-negotiable restrictions unless a qualified authority has changed the policy itself.
If the proposed change is a margin coefficient, hold eligibility definitions stable or explicitly account for any simultaneous policy change.
Otherwise the effect cannot be attributed cleanly to the coefficient.

Choose the assignment unit before collecting outcomes.
User-level assignment can reduce a customer's exposure to inconsistent ranking policies across visits.
Request-level assignment can provide more observations but may create cross-request contamination.
Inventory is shared: treatment customers may buy scarce items and change what control customers can receive.
This interference can make a simple independent-user analysis misleading.
Cluster randomization or inventory-aware analysis may be appropriate, depending on the setting.

Define the primary metric with numerator, denominator, attribution window, and accounting treatment.
An example is net contribution in USD divided by eligible requests, measured over a 14-day outcome window and adjusted for observed returns.
The exact window must reflect the business's return and cancellation processes.
If outcomes mature after the experiment ends, report interim estimates as incomplete.
For sparse outcomes, aggregate enough traffic or use a decision criterion based on uncertainty rather than declaring a small observed lift definitive.

Define guardrails before seeing results.
Possible guardrails include the rate of unavailable items shown, order cancellation, customer complaints, and task completion.
Specify a denominator and minimum sample for every rate.
An increase in hard-rule violations should trigger an incident even if total contribution improves.
Choose who is empowered to stop the experiment and how quickly the kill switch propagates.

Supplier and segment analysis should be planned, not improvised solely after an unfavorable result.
Use appropriate sample sizes and uncertainty intervals.
A small subgroup can have a large estimated percentage change with very few observations.
Repeatedly searching many slices can generate apparent differences by chance.
Protect sensitive attributes and consult the organization's privacy and fairness governance before collecting or using them.

### 5.18 Prepare a production decision trace

For one request, a compact trace might contain the following fields:

| Field | Example | Purpose |
|---|---|---|
| Request ID | `r-104` | Join decision and impression events |
| Surface | `meal-complements` | Identify the task and interface |
| Policy version | `policy-17` | Reconstruct rule semantics |
| Model version | `ranker-8` | Identify score producer |
| Inventory snapshot | `stock-431` | Locate the observed stock facts |
| Candidate ID | `B` | Identify an item considered |
| Eligibility result | `eligible` | Establish the hard-rule outcome |
| Utility | `0.940` | Explain the example's score |
| Selected position | `1` | Connect decision to the rendered slate |
| Fallback reason | `none` | Distinguish primary and alternate paths |
| Experiment assignment | `control` | Support analysis of policy variants |

These are synthetic identifiers and are not a complete logging schema.
A real trace also needs the event time, the exact score definition, and enough retained data or references to reproduce the decision.
For excluded candidates, record reason codes at the granularity needed to distinguish the relevant failure modes.
If the system uses randomized slate selection, capture the probability of the action under the actual logging policy, not an estimated click probability.

Logs can be sampled for volume, but sampling must be designed for the intended audit and analysis.
Sampling away rare safety failures or the control population can make important rates unmeasurable.
Record the logging sample rate and keep mandatory incident evidence according to policy.
Protect trace data against unauthorized access, alteration, and excessive retention.
If a replay cannot recover a result, report the missing version or input rather than fabricating a reconstruction.

### 5.19 Review edge cases before implementation

**Missing brand.** A null brand does not necessarily mean that all null-brand products share one manufacturer.
Choose an approved representation, assess its impact on the cap, and send the catalog defect to its owner.
Do not convert every null to a common label without understanding the resulting exclusions.

**Negative margin.** A promotion or fulfillment cost can make contribution negative.
Negative contribution is not automatically prohibited; an approved customer acquisition or substitution strategy may still justify display.
The objective and guardrails must explicitly allow that trade-off, and the cost input must be validated.

**Duplicate product identifiers.** If two catalog records represent the same sellable item, item-ID deduplication alone may not prevent repeated display.
Use a canonical item or variant group where the task requires it.
Ensure the relationship is versioned and does not incorrectly collapse distinct pack sizes.

**Tied scores.** A deterministic tie-break makes replay predictable.
If the tie-break systematically favors a supplier, that can create persistent exposure differences.
An approved randomized tie-break can reduce deterministic bias, but requires actual propensity logging and experiment-aware analysis.

**Expired commercial boost.** A cached policy may continue to apply an expired boost.
Enforce effective times in the serving path and monitor policy-version propagation.
The fallback after expiration should be explicit.

**New supplier with missing cost data.** Zero-filling the margin field can disadvantage that supplier regardless of product relevance.
Quarantining every new item can also harm discovery.
Choose and document a missing-data treatment with finance and merchandising, and measure its effect.

**A high-value item with no stock.** Relevance and margin do not override an availability requirement.
If the use case permits showing unavailable products for discovery, that is a different surface policy with clear status and expectations.
It must not be represented as immediately deliverable.

**Conflicting mandatory rules.** If the encoded requirements cannot all be satisfied, return no invalid slate.
Escalate the conflict to the accountable policy owners and preserve the evidence.
An engineer should not independently decide which legal or contractual obligation to ignore.

**Unreliable category labels.** A minimum category count computed from stale taxonomy can appear satisfied while showing near-duplicates.
Review the underlying classifications and test representative customer tasks.
The metric is only as meaningful as its classification system.

**Multiple currencies.** A USD margin coefficient cannot be applied directly to an amount denominated in another currency.
Specify conversion rate source, timing, rounding, and the relevant reporting currency.
If local price and costs are uncertain, do not hide uncertainty inside the coefficient.

**High return rate.** A product may appear attractive before returns but produce negative realized contribution afterward.
Measure returns on a mature cohort and investigate product fit and description accuracy.
A short attribution window should be labeled provisional.

**Repeated requests.** A slate can be diverse within one impression while showing the same items on every visit.
If longitudinal variety matters, define the history window and consented data use explicitly.
Do not assume an intra-list metric captures cross-session fatigue.

**Data deletion.** A privacy request may require deletion or other handling of identifiable event data under applicable rules.
Design the decision trace and retention scheme with the responsible privacy owner.
Auditability does not justify keeping every raw personal attribute indefinitely.

**Time-zone boundaries.** A policy effective at midnight in one region may start at a different instant elsewhere.
Store timestamps with offsets or in a standardized time representation and document local-time rules.
Evaluate daylight-saving transitions and expiration at the serving boundary.

**Last-unit demand.** Two customers may both be shown the same final unit.
The recommendation service is not a locking service.
Use the authoritative allocation process at the point a promise is made and measure disappointed demand.

## 6. Worked example

### 6.1 Decision and assumptions

An online grocer wants up to three recommendations for one meal-planning request.
The customer can receive delivery from node N at 18:00 local time on the same day.
The decision uses one unit of each product; currency is USD.
The relevance score estimates task fit on a 0–1 scale and is **not** assumed to be a purchase probability.
The margin figure is estimated contribution per purchased unit, after the costs defined by finance.
It is not a guaranteed profit and it is not a delivery commitment.

For illustration, a product must be active, eligible, and have at least one sellable unit in a sufficiently fresh node-N feed.
The product restriction policy excludes items whose eligibility is known to be false or unknown for this request.
For this example only, the maximum accepted feed age is 10 minutes.
The merchandising team wants at most one item per brand, but has explicitly made that a relaxable assortment preference rather than a legal or contractual requirement.
The approved first fallback is a shorter slate; a change to the brand rule requires the documented alternate policy.

We use the following base utility, with a fixed reference of USD 4.00 per unit:

```text
base = relevance + 0.20 * (contribution_USD / 4.00)
```

This normalization makes the margin term dimensionless.
The coefficient 0.20 expresses a chosen trade-off, not a causal estimate of customer value.
The reference value is fixed for the example; production must version and govern it.
The cap is hard within this policy, so no separate repeated-brand penalty is needed.
All positions have equal weight in this arithmetic example; in a real interface position effects require measurement.

### 6.2 Candidate data

| ID | Product | Brand | Relevance | Contribution, USD/unit | Sellable units | Feed age, minutes | Restriction status |
|---|---|---|---:|---:|---:|---:|---|
| A | Tomato pasta | Alder | 0.82 | 1.60 | 4 | 3 | eligible |
| B | Lentil pasta | Alder | 0.79 | 3.00 | 8 | 2 | eligible |
| C | Basil sauce | Birch | 0.73 | 2.40 | 2 | 4 | eligible |
| D | Cheese topping | Cedar | 0.69 | 3.20 | 0 | 1 | eligible |
| E | Sparkling drink | Dune | 0.66 | 3.60 | 5 | 2 | ineligible |
| F | Roasted vegetables | Elm | 0.70 | 1.20 | 6 | 6 | eligible |
| G | Herb dressing | Fir | 0.76 | 2.80 | 9 | 14 | eligible |

The table is synthetic and illustrates the calculation, not a claim about a real retailer.
D fails the availability rule because zero sellable units remain.
E fails the product restriction rule regardless of its margin.
G has stock but fails the freshness rule: its 14-minute observation is older than the 10-minute limit.
A, B, C, and F are eligible at the decision time.
A fresh feed alone would not reserve any unit.

### 6.3 Calculate the scores

| ID | Margin calculation | Base utility |
|---|---|---:|
| A | 0.82 + 0.20 × (1.60 / 4.00) | 0.900 |
| B | 0.79 + 0.20 × (3.00 / 4.00) | 0.940 |
| C | 0.73 + 0.20 × (2.40 / 4.00) | 0.850 |
| F | 0.70 + 0.20 × (1.20 / 4.00) | 0.760 |

Without the brand cap, the three highest scores are B, A, and C, with total utility 2.690.
That list contains two Alder products.
With the one-per-brand cap, choose B over A, then C and F.
The resulting order is B, C, F and its total utility is 0.940 + 0.850 + 0.760 = **2.550**.
For this small example, that slate is optimal under the equal-position-weight additive objective: B is the best Alder option, and C and F are the only remaining eligible brands.

The objective value falls by 0.140 relative to the unconstrained list.
This is a model-score difference, not USD and not proof of a welfare loss or benefit.
The customer receives a more varied list, but whether it improves task completion needs experimental evidence.
The expected contribution of the slate cannot be calculated from the relevance values because those values are not purchase probabilities.
The sum of per-unit contribution figures is not expected profit either: not every displayed item is purchased.

### 6.4 Handle a shortage and state ownership

If F becomes unavailable and no other eligible brand remains, the brand-capped policy returns B and C rather than inserting A silently.
That is a two-item slate, assuming the surface supports short results.
The service logs `short_slate` and the count of available brands.
If the interface cannot render fewer than three cards, it omits the module or follows another approved fallback; it does not include E or stale G just to fill the space.

The inventory owner investigates a feed freshness issue if G is frequently excluded.
The retrieval owner investigates whether too few relevant brands reach the re-ranker.
The merchandising owner can propose a new policy allowing a second Alder item, but must obtain the required approvals and evaluate customer impact.
The compliance owner alone decides the permitted treatment of restriction uncertainty.
The checkout system rechecks stock and eligibility when the customer attempts to order.

### 6.5 Compare a hypothetical alternate policy

Suppose an approved alternate policy replaces the hard brand cap with a 0.16 utility penalty for an additional same-brand item.
After choosing B, A's incremental utility becomes 0.900 − 0.160 = 0.740; C remains 0.850 and F remains 0.760.
A greedy selector therefore chooses C and then F, producing B, C, F.
If the penalty were 0.10 instead, A's incremental utility would be 0.800 and the greedy slate would be B, C, A with total adjusted utility 2.590.
These are different policies with different trade-offs, not emergency overrides to the original policy.
A penalty must be applied consistently to each repeated occurrence and its interpretation documented.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the base policy and computes an exact optimum by enumerating permutations of this small candidate set.
It returns a short slate if a full slate is impossible while preserving the item eligibility and brand cap.
This is appropriate for teaching and small sets, not for an unbounded production catalog: enumeration grows rapidly with candidate count and K.
The code uses integer thousandths of utility to avoid floating-point tie ambiguity in this example.
The underlying input scores and contributions are already fixed-point integers.

```python
from itertools import permutations

# Scores are thousandths; contribution is cents per unit.
# Stock age is minutes. Restriction is True only when eligibility is known.
candidates = [
    {"id": "A", "brand": "Alder", "rel": 820, "margin_cents": 160,
     "stock": 4, "age": 3, "restriction": True},
    {"id": "B", "brand": "Alder", "rel": 790, "margin_cents": 300,
     "stock": 8, "age": 2, "restriction": True},
    {"id": "C", "brand": "Birch", "rel": 730, "margin_cents": 240,
     "stock": 2, "age": 4, "restriction": True},
    {"id": "D", "brand": "Cedar", "rel": 690, "margin_cents": 320,
     "stock": 0, "age": 1, "restriction": True},
    {"id": "E", "brand": "Dune", "rel": 660, "margin_cents": 360,
     "stock": 5, "age": 2, "restriction": False},
    {"id": "F", "brand": "Elm", "rel": 700, "margin_cents": 120,
     "stock": 6, "age": 6, "restriction": True},
    {"id": "G", "brand": "Fir", "rel": 760, "margin_cents": 280,
     "stock": 9, "age": 14, "restriction": True},
]


def exclusion_reason(item):
    if item["restriction"] is not True:
        return "restriction"
    if item["stock"] < 1:
        return "unavailable"
    if item["age"] > 10:
        return "stale_stock"
    return None


def utility_milli(item):
    # 1000 * 0.20 * (margin_cents / 400) = margin_cents / 2.
    # These example margins are all even; production needs explicit rounding.
    assert item["margin_cents"] % 2 == 0
    return item["rel"] + item["margin_cents"] // 2


def select(items, k):
    eligible = [item for item in items if exclusion_reason(item) is None]
    # Prefer a longer valid slate; only then maximize utility.
    for size in range(min(k, len(eligible)), 0, -1):
        feasible = []
        for slate in permutations(eligible, size):
            if len({item["brand"] for item in slate}) != size:
                continue
            score = sum(utility_milli(item) for item in slate)
            # Equal position weights: order the same set by individual utility.
            ordered = tuple(sorted(slate, key=lambda x: (-utility_milli(x), x["id"])))
            ids = tuple(item["id"] for item in ordered)
            feasible.append((score, ids))
        if feasible:
            # Highest score, then lexicographically smallest IDs on a tie.
            score, ids = min(feasible, key=lambda result: (-result[0], result[1]))
            return ids, score
    return (), 0


for item in candidates:
    reason = exclusion_reason(item)
    if reason is not None:
        print(f"excluded {item['id']}: {reason}")
ids, score = select(candidates, 3)
print("slate:", ", ".join(ids))
print(f"utility: {score / 1000:.3f}")
```

Expected output:

```text
excluded D: unavailable
excluded E: restriction
excluded G: stale_stock
slate: B, C, F
utility: 2.550
```

The `select` function prioritizes length before score, which is itself a policy choice.
It uses a maximum one-per-brand constraint and does not implement category minima, position weights, policy-service outages, stock reservations, or randomized exploration.
It enumerates equivalent permutations more than once to keep the decision explicit; a production implementation should select an algorithm appropriate to the scale and constraints.
Its restriction boolean represents a prior policy decision; it is not an age-verification implementation.
The example assumes nonnegative ages and complete, valid fields; production inputs need validation and explicitly governed unknown handling.
The comparison of a score across slates is valid only under the fixed objective and policy version used for the calculation.

## 8. Data, standards, and regulatory context

### 8.1 Product and inventory data contracts

A catalog record should identify the sellable item, its variants, brand and category, lifecycle status, restrictions, and applicable market.
A stock record should identify the fulfillment location, channel, quantity, observation timestamp, and the semantics of each inventory bucket.
A price record needs currency, tax treatment, validity period, and the customer context for any personalized offer.
Margin data needs an owner, unit, cost definitions, refresh schedule, and explicit treatment of missing values.

GS1 Global Trade Item Numbers (GTINs) provide a standardized way to identify trade items where used.
A GTIN does not prove current stock, item safety, eligibility, or equivalence of two pack sizes.
A retailer may use an internal stock-keeping unit (SKU) for operational inventory; the mapping between internal SKUs and GTINs must be maintained.
When a pack or formulation changes, a stale mapping can produce an incorrect eligibility or availability decision.
See the GS1 GTIN standard in section 16 for authoritative identifier rules.

Inventory can change after the observation but before the customer acts.
An availability score or timestamp cannot guarantee delivery.
Use an authoritative reservation or checkout validation process when the business promises a unit.
Monitor the interval from stock observation to display and from display to purchase, plus resulting cancellation and substitution rates.

### 8.2 Privacy and eligibility

The EU General Data Protection Regulation (GDPR) applies only where its territorial and material scope conditions are met.
It includes principles such as purpose limitation and data minimization and gives individuals specified rights; determine the lawful basis and specific obligations with counsel.
Article 22 addresses certain solely automated decisions with legal or similarly significant effects, subject to its conditions and exceptions; not every recommendation automatically falls within it.
Sensitive data and inferences need particular care, including applicable special-category restrictions.
The regulation does not prescribe a single ranking objective or permit unrestricted collection merely because a model benefits from it.

If an eligibility check depends on an attribute such as age, use an approved source and process, and avoid logging the raw attribute when a policy outcome suffices.
Customer consent, where relied upon, must be assessed for the actual purpose and jurisdiction; a generic product preference is not necessarily consent for every use.
Retention, deletion, access control, and incident response belong in the data governance plan, not just model documentation.

### 8.3 Ranking transparency and commercial influence

The EU Platform-to-Business Regulation (Regulation (EU) 2019/1150), where applicable, requires covered online intermediation services and online search engines to describe main ranking parameters and their relative importance under its terms.
The EU Digital Services Act (Regulation (EU) 2022/2065), where applicable, includes recommender-system transparency provisions and additional obligations for designated very large online platforms and search engines.
The precise covered service, user, and obligation require legal review.
These regulations do not mean every internal score weight must be published or that every business is subject to every provision.

In the United States, the Federal Trade Commission's endorsement guidance discusses disclosure of material connections in endorsements; applicability depends on the facts.
Sponsored placement and a recommendation affected by payment must be reviewed under the relevant advertising and consumer-protection rules.
A label is not a substitute for an accurate ranking explanation or for compliance with other applicable obligations.
Rules and guidance may change; the cited primary sources should be checked against current law and the actual deployment date.

### 8.4 Data quality controls

- Check that the stock timestamp is in a known time zone and not in the future beyond permitted clock skew.
- Check that the node and delivery window correspond to the customer's request.
- Check that product restrictions have effective dates and jurisdictional scope.
- Check that margin uses a common unit and currency and reflects the current promotion.
- Check that category and brand labels are neither null nor silently defaulted into a shared brand.
- Check that a candidate's identity is stable across retrieval, ranking, and impression logging.
- Alert on sudden shifts in missingness, negative stock, duplicate IDs, or feed latency.
- Preserve point-in-time snapshots for reproducible offline analysis.

These checks detect defects; they do not by themselves establish that every source record is true.
Feed ownership, correction workflow, and the approved behavior for unavailable dependencies must be explicit.

## 9. Industry practice and case evidence

### 9.1 Documented examples

**YouTube candidate generation and ranking.** Covington, Adams, and Sargin describe a two-stage recommender architecture in their 2016 RecSys paper: candidate generation narrows a large corpus and a ranking model orders candidates.
The paper concerns video recommendations, not grocery stock or the exact algorithm in this unit.
Its useful lesson is architectural: a later ranking stage only sees candidates supplied by an earlier stage.
[Inference] A commerce re-ranker that excludes unavailable items should also monitor retrieval coverage, because filtering can expose gaps in the candidate pool.

**Netflix diversity.** Netflix describes using personalized row selection and ranking in its public recommender-system account by Gomez-Uribe and Hunt.
The article discusses the presentation of diverse rows and personalization of the member experience.
It does not establish that a fixed one-per-brand rule improves grocery purchases.
[Inference] List composition and presentation matter beyond an individual item's predicted score, but their appropriate constraints depend on the customer task.

**Maximal marginal relevance.** Carbonell and Goldstein's 1998 work proposes maximal marginal relevance for balancing relevance and novelty in information retrieval.
It motivates penalizing redundancy with already selected results.
It is not a rule that every recommender should maximize diversity, and its original retrieval context differs from product selection.
[Inference] A retailer can adapt a redundancy term, but must define similarity, validate the outcome, and preserve mandatory product restrictions separately.

### 9.2 What evidence does and does not establish

A published architecture demonstrates a plausible pattern, not its effectiveness on your catalog, fulfillment network, or customer population.
Vendor reports may describe successful deployments but can lack independent controls or full cost accounting.
A policy that improves clicks in a media product may worsen fulfillment or customer trust in a grocery substitution task.
The correct comparison is against the current valid policy under the same candidate coverage, data freshness, and experiment population.

When reporting a case internally, identify the source and date, the intervention, the baseline, the observed outcome, and the limitations.
Separate measured effects from a design rationale or inference.
For example, a higher brand count is directly measurable; claiming it improved customer choice requires an outcome study.
An increase in contribution per purchased item is directly measurable, but it does not by itself establish higher contribution per visitor.

### 9.3 An operating pattern

A mature team maintains a shared rule register and a separate experiment log.
The register states what must always be true and who owns each rule.
The experiment log states which tunable choices were compared and what was learned.
Policy releases should link the two: a commercial experiment cannot redefine an eligibility requirement without the policy owner's approval.
A change in inventory definitions can invalidate previous offline evaluations even if the ranker code did not change.

Supplier governance needs particular attention.
Supplier exposure can be concentrated by retrieval popularity, catalog quality, availability, or a margin objective.
Measure each stage separately before attributing the observed concentration to re-ranking.
If commercial agreements affect exposure, document the decision and applicable disclosure; do not assume a fairness metric can replace contractual or legal review.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unavailable item displayed | Cancellation rate rises for recommended items | Global stock substituted for node-level sellable stock | Broken delivery promise | Join on node and window; revalidate at checkout |
| Stale feed accepted | Feed age distribution shifts upward | Timestamp ignored or defaulted | Incorrect availability | Enforce freshness contract and safe unknown behavior |
| Restriction bypass | Eligible flag is missing unusually often | Unknown treated as allowed | Unsafe or unlawful exposure | Approved unknown policy and independent hard-rule validation |
| Margin overwhelms relevance | Conversion or satisfaction drops while margin per order rises | Incomparable score scales | Poor customer fit | Calibrate, bound weights, and monitor per-request outcomes |
| Repeated variants | Low distinct-brand or category count | Independent item ranking ignores slate | Narrow choice | Define and test task-appropriate diversity |
| Forced irrelevant diversity | Relevant coverage falls for constrained requests | Hard quota applied without fit threshold | Less useful results | Make targets soft where appropriate and measure task completion |
| Infeasible slate | Short-slate rate spikes | Conflicting caps and minimums | Empty or misleading module | Feasibility checks and approved fallback ladder |
| Greedy dead end | Greedy returns short results while exact replay finds full slate | Early pick consumes scarce constraint capacity | Lost coverage | Backtracking, lookahead, or exact solver |
| Solver timeout misreported | Timeout count rises with candidate volume | Runtime limit interpreted as infeasibility | Wrong operational diagnosis | Separate timeout reason and apply safe fallback |
| Duplicate identities | Same item appears twice under different IDs | Broken SKU/variant mapping | Wasted slots and misleading diversity | Canonical identity and deduplication rules |
| Hidden paid influence | Unexplained supplier exposure shift | Manual or sponsored boost not recorded | Trust and compliance risk | Register overrides and review disclosure |
| Missing exposure log | Downstream clicks lack impression IDs | Logging only candidates or purchases | Biased evaluation and weak audit | Instrument rendered impressions and joins |
| Incorrect propensity | Propensities sum incorrectly or are constant | Score logged as selection probability | Invalid off-policy estimates | Log actual randomization probabilities |
| Feedback loop | Popular suppliers gain exposure each retrain | Past exposure drives training labels | Reduced discovery and concentration | Controlled exploration and exposure-aware evaluation |
| Metric denominator drift | Margin rate rises but total contribution falls | Reporting per order rather than per request | False business conclusion | Fix denominators and report a metric suite |
| Silent commercial override | Policy version does not explain output | Untracked manual rule | Inconsistent and unauditable decisions | Version rules, approval, expiry, and rollback |
| Disparate data coverage | Certain regions have high unknown-stock rates | Unequal feed quality or catalog enrichment | Unequal service quality | Monitor operational slices and repair the source |
| Reservation confusion | Customers see stock then receive cancellation | Display treated as allocation | Customer frustration | Separate forecast from commitment and recheck |
| Unknown margin defaulted to zero | New suppliers systematically rank lower | Missing cost interpreted as poor value | Supplier and customer bias | Explicit missing-data policy and coverage monitoring |
| Outdated policy in cache | Rule changes but old behavior persists | Cache invalidation or version mismatch | Continued prohibited exposure | Versioned deployment and cache-expiry controls |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or govern |
|---|---|---|---|
| Candidate set contains useful alternatives | Selection only chooses from retrieved items | Retrieval misses relevant categories or suppliers | Audit recall and excluded-candidate coverage |
| Relevance scores are comparable within a request | The objective orders candidates | Different models or surfaces use different scales | Calibrate and compare by surface and cohort |
| Margin values share a definition | A single coefficient can compare items | Costs or promotion treatment differ | Finance reconciliation and data contracts |
| Contribution is a useful commercial proxy | Margin represents a business outcome | Returns, basket effects, or fulfillment costs omitted | Compare attributed contribution with realized outcomes |
| Stock facts match the requested fulfillment context | Availability filter is meaningful | Wrong node, time window, or channel | Join audits and cancellation analysis |
| Freshness threshold is acceptable | A bounded delay is used as a risk control | Fast-moving stock or feed outages | Measure stale-to-failure relationship by item and node |
| Eligibility classification is accurate | Mandatory rules depend on it | Incorrect catalog attributes or policy service errors | Ownership, audits, and exception investigation |
| Brand labels are well defined | The slate cap has consistent meaning | Shared private labels or null brand values | Catalog quality checks and policy review |
| A short slate is renderable | Fallback preserves all hard rules | Client requires exactly K cards | Test interface contract before rollout |
| Utility trade-off is approved | Margin can influence ordering | Coefficient changes without governance | Version and approve parameters; use controlled experiments |
| Equal position weights in the example | Simple sum identifies the best set | Attention differs by slot | Estimate position effects in an appropriate experiment |
| Online experiment is interpretable | Randomization supports causal comparison | Interference through shared stock or repeated users | Choose assignment unit and assess spillovers |
| Customer outcomes are observable | Guardrails can detect harm | Delayed returns or unrecorded offline outcomes | Use suitable windows and acknowledge missing outcomes |
| Historical snapshots are point-in-time correct | Replay reflects what was knowable | Future stock or corrected attributes leak backward | Audit timestamps and replay lineage |
| Enumeration finishes | Exact sample code can find an optimum | Candidate count or K grows | Set bounded inputs or use a suitable optimizer |

The worked example assumes all items are otherwise task-compatible; it does not prove suitability for a particular customer.
A complete product system needs additional controls for allergies, substitutions, and other context when those are relevant.
Similarly, slate-wide objectives can have complex interactions that a local greedy score cannot capture.
Do not generalize the example's single coefficient or ten-minute freshness window to another deployment without evidence and approval.

## 12. Exercises

### Exercise 1 — Identify the boundaries (10 minutes)

A retrieval service returns 100 items, a ranker scores them, a policy service excludes restricted items, and a checkout service confirms stock.
**Deliverable:** one sentence for each service and one sentence explaining why re-ranking is not a stock reservation.
**Acceptance criteria:** all four boundaries are distinct and the answer identifies the checkout recheck.

### Exercise 2 — Classify constraints (15 minutes)

A team proposes: (a) do not show prohibited products, (b) prefer higher margin, (c) show at least two brands whenever possible, and (d) avoid out-of-stock items.
**Deliverable:** classify each as a hard rule or soft preference under an explicitly stated business policy, name an owner, and describe unknown data handling.
**Acceptance criteria:** no finite margin boost bypasses the prohibition; the answer explains that a diversity target can be soft and specifies availability semantics.

### Exercise 3 — Calculate the score (10 minutes)

Use the formula in section 6 for an eligible item with relevance 0.74 and contribution USD 2.20 per unit.
**Deliverable:** normalized margin term and total base utility.
**Acceptance criteria:** both steps are shown and the result is identified as dimensionless rather than USD.

### Exercise 4 — Select a slate (15 minutes)

Three eligible candidates have `(ID, brand, utility)` values `(J, Moss, 0.91)`, `(K, Moss, 0.87)`, and `(L, Pine, 0.78)`.
You need up to two items with a one-per-brand cap and equal position weights.
**Deliverable:** ordered slate and total utility; compare with an unconstrained top-two slate.
**Acceptance criteria:** applies the cap before interpreting the score and reports both totals.

### Exercise 5 — Resolve infeasibility (15 minutes)

A hard geographic prohibition excludes three candidates; the two remaining eligible candidates have the same brand.
The approved policy limits each brand to one item and allows a shorter slate.
**Deliverable:** describe a valid result, a reason code, and who can approve a future policy change.
**Acceptance criteria:** no prohibited item is displayed and the cap is not silently relaxed.

### Exercise 6 — Critique a margin proposal (20 minutes)

A commercial manager reports that the new ranking increased margin per purchased unit by 12% and asks to launch immediately.
Conversion fell, and cancellation data has not yet matured.
**Deliverable:** a decision memo with two missing metrics, one customer risk, and the release decision.
**Acceptance criteria:** distinguishes per-unit from per-request results and does not claim a causal effect without a suitable comparison.

### Exercise 7 — Design logging (20 minutes)

An engineer proposes recording only the displayed product IDs and eventual purchases.
**Deliverable:** an expanded event schema and two privacy safeguards.
**Acceptance criteria:** includes exclusions or reconstruction inputs, policy version, impression position, experiment assignment, and actual propensities when randomization is used.

### Exercise 8 — Govern a change (25 minutes)

A supplier pays for increased exposure on a recommendation surface.
**Deliverable:** an approval and monitoring plan with at least four distinct roles, an expiry rule, and a rollback trigger.
**Acceptance criteria:** separates commercial influence from mandatory eligibility, addresses disclosure review, and specifies customer and supplier exposure monitoring.

### Exercise 9 — Diagnose greedy selection (20 minutes)

K is two; exactly one category B item must be included and each brand can appear at most once.
Candidates are P `(brand X, category A, score 0.95)`, Q `(brand X, category B, score 0.90)`, and R `(brand Y, category A, score 0.80)`.
A greedy algorithm chooses P first based only on score.
**Deliverable:** show why it gets stuck, identify a feasible slate, and propose a better method.
**Acceptance criteria:** includes both the brand cap and category minimum and does not call the problem globally infeasible.

## 13. Self-check answers

### 1. Boundaries

Retrieval reduces the catalog to candidates; ranking estimates an ordering or customer fit.
The policy service removes candidates that fail the applicable eligibility rules; re-ranking then chooses and orders an eligible slate under slate rules.
Checkout validates the live transaction, including inventory and any applicable restrictions.
A recommendation uses information observed earlier and does not allocate a unit unless a separate reservation contract says it does.

### 2. Constraint classification

A product prohibition is hard and owned by the relevant compliance or product-safety authority; unknown eligibility follows its approved safe policy.
Margin is normally a soft commercial preference, with finance owning its definition and merchandising owning the proposed trade-off.
The brand target can be soft, owned by merchandising and product, with a documented way to handle missing labels.
An availability rule can be hard for display, owned by inventory operations: define sellability, node, window, freshness, and unknown behavior.
The actual legal and business requirements determine the classification; the label alone does not.

### 3. Score

The normalized margin is 2.20 / 4.00 = 0.55.
The margin contribution to the objective is 0.20 × 0.55 = 0.11.
Total base utility is 0.74 + 0.11 = **0.85**, a dimensionless score.
It is not USD 0.85 and it is not a purchase probability.

### 4. Slate

J and L satisfy the cap and total 0.91 + 0.78 = **1.69** utility units.
J precedes L when ordered by individual utility.
The unconstrained top two are J and K, totaling 0.91 + 0.87 = **1.78** utility units, but both are Moss.
The 0.09 difference is a score trade-off, not a measured change in customer welfare.

### 5. Infeasibility

Select the higher-scoring of the two remaining eligible candidates and return a one-item slate.
Record `short_slate` with an eligible-brand count of one and separately record the geographic exclusions.
The merchandising policy owner can propose a different brand rule, subject to the organization's change approvals.
The compliance owner governs the prohibition; the commercial owner cannot override it to fill a slot.

### 6. Decision memo rubric

A strong memo holds the launch pending a suitable comparison and maturation of relevant outcomes.
It requests contribution per eligible request, conversion, cancellation and return rates, and an appropriate customer-value measure.
It distinguishes a selected subset of purchased items from all eligible requests.
It states that the reported 12% is not automatically incremental profit or causal lift and asks for experiment design and uncertainty.
A weak memo treats the per-unit margin increase as conclusive or ignores cancellations.

### 7. Logging rubric

A strong schema includes request and decision IDs, time, surface, candidate IDs, eligibility reasons or sufficient reconstruction inputs, model and policy versions, displayed order, and impression event IDs.
It links experiment assignment and downstream events and records actual action probabilities when randomized choices are used.
It controls access, retention, and sensitive fields; pseudonymization alone is not a guarantee of anonymity.
A weak schema logs only purchases and therefore cannot distinguish no exposure from no interest.

### 8. Governance rubric

A strong plan assigns merchandising to the commercial proposal, legal/compliance to applicability and disclosures, product to customer experience, analytics to experiment design, engineering to implementation and rollback, and inventory to availability.
It records supplier, scope, parameter, approver, start and expiration, and contractual basis.
It monitors customer outcomes, fulfillment, supplier exposure, and policy violations.
An immediate hard-rule failure triggers rollback; changes to soft metrics trigger the preapproved review procedure.

### 9. Greedy diagnosis

P consumes brand X, so Q cannot be selected afterward; R cannot satisfy the category B minimum.
The valid slate is Q and R, with utility 0.90 + 0.80 = **1.70**.
A selector that checks remaining feasibility before choosing P, backtracks, or enumerates the small set can find it.
A timeout or this greedy dead end does not prove the original problem infeasible.

## 14. Interview questions

1. **What is the difference between retrieval and re-ranking?** Strong answers describe candidate coverage and slate-level selection; weak answers claim the re-ranker can choose any catalog item.
2. **Why not put a legal restriction into a penalty?** Strong answers distinguish hard eligibility from tradable scores; weak answers assume a sufficiently large but finite penalty is always safe.
3. **How would you define availability?** Strong answers include item, node, channel, quantity, time, freshness, and later revalidation; weak answers cite global on-hand stock alone.
4. **What makes margin scores comparable?** Strong answers define contribution, currency, unit, cost horizon, and normalization; weak answers add unrelated raw scores without calibration.
5. **When can diversity harm the customer?** Strong answers discuss task compatibility, relevance thresholds, and forced quotas; weak answers assume a higher brand count is always better.
6. **What do you do when there is no full feasible slate?** Strong answers use an approved fallback and preserve non-negotiable restrictions; weak answers silently relax every constraint.
7. **Why can greedy selection fail?** Strong answers identify interactions and show a counterexample; weak answers equate a locally best item with a globally optimal slate.
8. **How do you know a margin increase is beneficial?** Strong answers compare per-request contribution and customer outcomes in a valid experiment; weak answers report only margin per order.
9. **What would you log for randomized exploration?** Strong answers name eligible actions, displayed slate, policy version, and actual selection propensities; weak answers call a relevance score a propensity.
10. **Who owns a sponsored boost?** Strong answers describe commercial approval, legal review, customer-impact monitoring, expiry, and rollback; weak answers make it an untracked engineering parameter.
11. **How would you detect stock-related bias?** Strong answers segment data freshness, exclusion, exposure, and fulfillment by operational context and repair the source; weak answers only inspect overall click-through rate.
12. **What does a solver timeout mean?** Strong answers distinguish incomplete computation from proven infeasibility and specify fallback; weak answers report the problem as impossible.

## 15. Key terms

- **Availability:** ability to fulfill for a specified context; see [mechanics](#53-validate-candidate-and-context-data).
- **Brand cap:** upper bound on a brand's slate count; see [slate rules](#55-express-slate-rules-precisely).
- **Candidate:** retrieved item considered for a request; see [mental model](#3-the-core-mental-model).
- **Commercial override:** approved deviation or parameter affecting exposure; see [governance](#52-establish-rights-duties-and-decision-owners).
- **Contribution:** defined commercial value per unit; see [objective](#56-choose-a-defensible-objective).
- **Customer value:** usefulness and harm to the customer; see [monitoring](#59-log-monitor-and-learn).
- **Diversity:** defined variation within a slate; see [objective](#56-choose-a-defensible-objective).
- **Eligibility:** permission to consider an item; see [filtering](#54-apply-non-negotiable-item-level-eligibility).
- **Expected margin:** purchase probability times contribution under stated assumptions; see [objective](#56-choose-a-defensible-objective).
- **Exposure:** opportunity for an item to be seen; see [monitoring](#59-log-monitor-and-learn).
- **Fallback:** approved alternative when the preferred slate cannot be produced; see [feasibility](#57-check-feasibility-before-deciding-what-to-relax).
- **Feasibility:** existence of a slate satisfying hard rules; see [feasibility](#57-check-feasibility-before-deciding-what-to-relax).
- **Freshness:** age of a fact at decision time; see [data contracts](#81-product-and-inventory-data-contracts).
- **Greedy selection:** sequential local choice; see [selection](#58-select-order-and-validate-the-slate).
- **Guardrail:** rule or metric limiting unacceptable harm; see [release](#510-release-and-govern-changes).
- **Hard constraint:** mandatory rule on a valid result; see [mental model](#3-the-core-mental-model).
- **Impression:** defined opportunity for a displayed item to be seen; see [logging](#59-log-monitor-and-learn).
- **Margin:** specified contribution per sale; see [worked example](#61-decision-and-assumptions).
- **Policy version:** identifier of the applicable rule set; see [logging](#59-log-monitor-and-learn).
- **Position weight:** modeled relative value of a display slot; see [objective](#56-choose-a-defensible-objective).
- **Propensity:** probability of a randomized action; see [logging](#59-log-monitor-and-learn).
- **Redundancy penalty:** reduction for similarity to already selected items; see [objective](#56-choose-a-defensible-objective).
- **Relevance:** estimated task fit; see [worked example](#61-decision-and-assumptions).
- **Reservation:** authoritative stock allocation, distinct from prediction; see [data contracts](#81-product-and-inventory-data-contracts).
- **Retrieval:** candidate generation from a larger catalog; see [mental model](#3-the-core-mental-model).
- **Slate:** ordered list selected together; see [slate rules](#55-express-slate-rules-precisely).
- **Soft preference:** tradable objective term or target; see [mental model](#3-the-core-mental-model).
- **Utility:** defined objective score, not necessarily money; see [calculation](#63-calculate-the-scores).

## 16. Sources

Sources below support the specific statements identified above; they do not certify the synthetic example or provide legal advice.
Access and applicability should be rechecked for the deployment jurisdiction and date.

1. Covington, P., Adams, J., and Sargin, E. (2016). “Deep Neural Networks for YouTube Recommendations.” *Proceedings of the 10th ACM Conference on Recommender Systems*. [DOI: 10.1145/2959100.2959190](https://doi.org/10.1145/2959100.2959190). Candidate generation and ranking architecture.
2. Gomez-Uribe, C. A., and Hunt, N. (2015). “The Netflix Recommender System: Algorithms, Business Value, and Innovation.” *ACM Transactions on Management Information Systems*, 6(4). [DOI: 10.1145/2843948](https://doi.org/10.1145/2843948). Personalized presentation and industry practice.
3. Carbonell, J., and Goldstein, J. (1998). “The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.” *Proceedings of SIGIR '98*. [DOI: 10.1145/290941.291025](https://doi.org/10.1145/290941.291025). Relevance and novelty in retrieval.
4. GS1. [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/). Identifier assignment rules; not an availability standard.
5. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). See particularly Articles 5, 6, 9, and 22, subject to applicability and conditions.
6. European Union. [Regulation (EU) 2019/1150 on fairness and transparency for business users of online intermediation services](https://eur-lex.europa.eu/eli/reg/2019/1150/oj). See Article 5 for ranking transparency in covered services.
7. European Union. [Regulation (EU) 2022/2065, Digital Services Act](https://eur-lex.europa.eu/eli/reg/2022/2065/oj). See Articles 27 and 38 for recommender-system provisions, subject to scope.
8. U.S. Federal Trade Commission. [FTC's Endorsement Guides: What People Are Asking](https://www.ftc.gov/business-guidance/resources/ftcs-endorsement-guides-what-people-are-asking). Guidance on endorsements and material connections; fact-specific applicability.

## 17. Links

- [Module 24 index](_index.md)
- [Module 24 survey and lab sections](../../../course/24_recommendation_systems_personalization.md)
- [Recommendation tasks](01_recommendation-tasks.md)
- [Offline and online evaluation](07_offline-online-evaluation.md)
- [Feedback loops and bias](08_feedback-loops.md)
- [Production labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
