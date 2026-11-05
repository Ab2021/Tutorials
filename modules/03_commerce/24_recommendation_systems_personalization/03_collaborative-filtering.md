# Collaborative Filtering and Alternating Least Squares

> Module 24 — Recommendation Systems and Personalization  
> Topic 03 of 11  
> Estimated study time: 4 hours  
> Prerequisites: recommendation tasks and popularity/content baselines  
> Version 2026.5

## 1. Why this topic exists

A distributor's digital channel may carry tens of thousands of items.
A buyer may interact with only a few dozen of them in a year.
A merchandiser cannot hand-author related-item rules for every customer and product combination.
A single popularity list is easy to operate but treats different customer needs alike.
Collaborative filtering uses patterns shared across customers to nominate plausible items for each customer.

The practical question is not whether a factorization can be trained.
It is whether historical interactions contain enough comparable, lawfully usable evidence to improve a specific decision.
For example, the owner of a business-to-business replenishment page wants to suggest compatible products that are in stock at the customer's fulfilment location.
The merchant owns the commercial objective and exceptions.
The data scientist owns the offline comparison and uncertainty.
The platform owner owns the serving and rollback contract.
Inventory and compliance owners define eligibility, not latent factors.

Showing a relevant but unavailable item wastes a slot and can create an unfulfillable order.
Showing a substitute that is dimensionally incompatible can interrupt a customer's operation.
A return or complaint may reflect a delivery defect, not dislike of the product.
An interaction recorded after the evaluation cutoff cannot legitimately teach the model what was known before that cutoff.
These are business and measurement problems as much as algorithmic ones.

A collaborative model can generate candidates from transaction histories without needing a complete product taxonomy.
It can generalize across similar customers and items that never directly co-occurred.
It cannot infer the preferences of an entirely new customer without another information source.
Nor does it know whether a product is safe, contractually allowed, profitable, or available now.
Use it as one component of a decision system with explicit constraints, fallbacks, and an experiment.

The material below distinguishes an observed event, an inferred preference, a recommendation score, and the incremental outcome caused by showing an item.
Those are four different objects.
Confusing them is a common reason a strong offline result does not translate into a better operation.

## 2. Learning outcomes

After this topic you can:

1. Construct a time-valid user–item interaction matrix and explain its sparsity and missingness.
   Exercise 1 checks the structure and the effect of duplicate events.
2. Convert implicit events into preference and confidence values without treating missing events as observed dislikes.
   Exercise 2 checks the arithmetic and interpretation.
3. Derive and execute one alternating least squares (ALS) user update for a fixed item-factor matrix.
   Exercise 3 checks the normal equation and numerical solution.
4. Design temporal holdouts and compute ranking metrics against a defensible baseline.
   Exercises 4 and 5 test the ranking and the candidate universe.
5. Diagnose exposure bias, cold start, catalogue restrictions, and transaction-data limitations.
   Exercise 6 asks for an evidence-based critique.
6. Specify ownership, a launch experiment, and a fallback for a constrained business use case.
   Exercise 7 asks for a decision memo.
7. Assess what data-protection and product-identification sources do and do not require.
   Exercise 8 asks for a control map.

Completion means explaining the decision and its limitations, not merely producing a factor matrix.

## 3. The core mental model

```text
Historical events, as known at cutoff
    | identity resolution, event policy, time window
    v
User x item sparse interactions r_ui
    | derive preference p_ui and confidence c_ui
    v
Implicit ALS: alternate fixed-item user solves and fixed-user item solves
    | vectors x_u and y_i; affinity score x_u^T y_i
    v
Candidate generation for a known user
    | remove seen items if appropriate; enforce real-time eligibility
    v
Ranked slate or downstream ranker + fallback
    | randomized measurement, guardrails, operational ownership
    v
Outcomes and exposure logs -> later training window
```

Rows are users and columns are items.
Each cell summarizes what was observed about one pair within an explicitly defined time window.
Most cells are empty because the customer did not interact with most of the catalogue.
Empty is not equivalent to an explicit negative rating.
The customer might never have been exposed, might have been out of stock, or might have bought through an unobserved channel.

A factorization assigns each user and item a short vector.
Their dot product is an affinity score on an arbitrary scale.
The dimensions are learned jointly and need not correspond to named product attributes.
A high score means the fitted model ranks the pair relatively highly under its training objective.
It is not automatically a calibrated purchase probability, expected profit, or causal effect.

Implicit ALS makes a modelling compromise.
A pair with a positive interaction gets preference one and increased confidence.
A pair with no recorded interaction gets preference zero at a lower baseline confidence.
The latter is an optimization target representing uncertain absence, not a claim that the user explicitly dislikes the item.
Because there are many more unobserved pairs than observed pairs, their aggregate contribution still matters.

With item vectors held fixed, each user's vector is found by solving a regularized weighted least-squares problem.
Then user vectors are held fixed while the corresponding item problems are solved.
The two steps repeat until a chosen stopping condition.
Each block update has a solvable quadratic objective, but the full joint factorization is non-convex.
Initialization and convergence choices can affect the resulting local solution.

The model works by sharing statistical strength.
If customers with overlapping histories also interact with another item, a user with a similar pattern may receive a high score for it.
This is association, not proof that the item is a substitute or that the recommendation will change behaviour.
For scarce stock, a good demand-prediction score may still produce an undesirable allocation.
The business policy layer must decide who is allowed to see or order it.

A recommendation pipeline has at least three distinct tests.
First, can retrieval include useful eligible items in a manageable candidate set?
Second, does ranking order those items well for the task?
Third, does the deployed policy improve a business outcome relative to an appropriate control?
A pass at one level does not prove a pass at the next.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse or unit |
|---|---|---|
| User | Entity receiving a personalized recommendation. | Assuming account, employee, device, and person are interchangeable. |
| Item | Catalogue entity being recommended. | Mixing product families and stock-keeping units without a mapping. |
| Stock-keeping unit (SKU) | Operational product variant tracked in inventory. | Treating a family-level product as a warehouse-specific SKU. |
| Interaction | Observed user–item event such as a purchase or click. | Calling every event a preference. |
| Event time | Time at which an action occurred. | Substituting ingestion time when reconstructing history. |
| Ingestion time | Time at which the system learned of an event. | Ignoring late arrival when simulating what was knowable. |
| As-of snapshot | Data available at a stated decision time. | Joining present-day catalogue attributes into past decisions. |
| Implicit feedback | Behaviour used as an indirect signal of preference. | Assuming a missing event means dislike. |
| Explicit feedback | Stated rating or preference supplied by a user. | Treating a click as a stated rating. |
| Exposure | Opportunity to see an item in a specific context. | Equating a fetched candidate with an actually rendered impression. |
| Impression | Logged rendering under a defined visibility rule. | Assuming it proves attention or choice. |
| Sparse matrix | Matrix with relatively few recorded nonzero cells. | Forgetting that the logical unobserved cells still influence implicit ALS. |
| Density | Observed distinct user–item pairs divided by eligible matrix cells. | Counting repeated events as separate occupied cells. |
| Interaction strength | Nonnegative summary of chosen events for a pair, $r_{ui}$. | Mixing clicks and purchases as though they share natural units. |
| Preference | Binary modelling target $p_{ui}=1[r_{ui}>0]$. | Calling it observed satisfaction. |
| Confidence | Positive loss weight $c_{ui}$ assigned to a pair. | Treating it as a statistical probability. |
| Factor | Learned vector coordinate associated with a user or item. | Assigning a business label without evidence. |
| Embedding | Low-dimensional vector representing an entity in a model. | Assuming different training runs have aligned coordinates. |
| Rank | Number of latent dimensions, denoted $f$. | Confusing factor rank with a recommendation's position. |
| Score | Dot product $x_u^\top y_i$. | Reading an uncalibrated score as purchase probability. |
| Regularization | Penalty controlling factor magnitudes, here $\lambda\|x\|^2+\lambda\|y\|^2$. | Assuming one package's regularization convention transfers unchanged. |
| ALS | Alternating least squares, solving one factor block while fixing the other. | Claiming a globally optimal joint solution. |
| Candidate | Item proposed for later eligibility checks or ranking. | Claiming a retrieved item was necessarily shown. |
| Eligible item | Item permitted for this user, channel, location, and moment. | Filtering with future availability in an offline replay. |
| Holdout | Events withheld from model fitting for evaluation. | Randomly splitting events and leaking future behaviour. |
| Recall at $K$ | Share of held-out relevant items found in top $K$. | Calling it precision or ignoring users with no relevant items. |
| Precision at $K$ | Relevant retrieved items divided by $K$, under a fixed-length convention. | Hiding missing slots or duplicate items. |
| NDCG | Normalized discounted cumulative gain; rewards relevant items near the top. | Comparing scores calculated with different relevance or candidate rules. |
| Coverage | Fraction of a declared user or item population served or recommended. | Reporting it without the denominator and time period. |
| Cold start | Inability to estimate a factor from adequate interaction history. | Assuming a zero vector is a meaningful personalized model. |
| Exposure bias | Distortion because observed actions depend on prior opportunity. | Treating historical choice as an unbiased sample of the catalogue. |
| Leakage | Use of information unavailable at decision time. | Restricting only event timestamps while leaking later identity or availability. |
| Propensity | Probability an action was selected under a known logging policy. | Inventing it retrospectively for a deterministic policy. |
| Fallback | Defined response when the personalized model cannot produce valid candidates. | Silently returning an empty slate or unrestricted global popularity. |

## 5. The mechanics, step by step

### 5.1 Frame the decision before building the matrix

1. Specify the surface and the decision timestamp.
   A reorder page at account login is different from product-detail-page complements.
   The former can use past purchases to suggest replenishment.
   The latter may need session intent, product compatibility, and a different item universe.
2. Define the user entity.
   A business account may have multiple buyers with different responsibilities.
   A person-level model may not be appropriate if purchases are centrally negotiated.
   An account-level model may leak one buyer's activity to another if access is not authorized.
3. Define the item entity.
   Pick a SKU, product family, or offer and retain an effective-dated mapping.
   A variant-level decision needs variant-level availability and compatibility.
   A family-level recommendation needs a documented way to select the purchasable variant.
4. Define the action the system can take.
   A retrieved candidate is not necessarily shown.
   A shown item is not automatically ordered or fulfilled.
   State the maximum number of slots and what happens when fewer valid items exist.
5. Set the decision objective and constraints.
   Examples include incremental fulfilled contribution margin per eligible session and no increase in wrong-item complaints.
   Margin, stock, contracts, and safety requirements are not learned automatically by ALS.
6. Assign approval rights.
   Commercial leadership accepts the business objective.
   Data governance and security approve permitted data uses and access.
   The product owner approves the user experience and experiment.
   The operations owner defines stock freshness and exception procedures.

**Input:** a documented placement, users, items, timing, objectives, and constraints.
**Output:** an accountable decision contract.
**Branch:** if the action requires compatibility proof, require a verified compatibility rule or another validated mechanism before offering a candidate.

### 5.2 Establish the event contract

1. Inventory event sources, including transactions, page views, cart additions, returns, cancellations, and exposure logs.
2. For each source record the event identifier, event timestamp, arrival timestamp, entity keys, channel, and semantic meaning.
3. Deduplicate using a source-specific immutable event identifier where possible.
   A retry is not another purchase.
   Multiple genuinely separate purchases remain separate events.
4. Decide how order changes are represented.
   A submitted order that is cancelled before fulfilment may be poor evidence of a successful purchase.
   A return may be caused by damage or shipping error and should not automatically become a negative preference.
5. Resolve identities using mappings effective at the decision time.
   Audit mergers, account migrations, deleted identities, and consent changes.
6. Define a training lookback and a data-latency buffer.
   A 180-day lookback is a business hypothesis, not a universal optimum.
   A model for annual spare parts may need a longer period than one for daily groceries.
7. Record what was not captured.
   Offline sales, direct sales representatives, and unlogged impressions may produce systematic gaps.

**Input:** raw source events and historical identity mappings.
**Output:** a versioned, deduplicated, as-of event table.
**Branch:** if source semantics or lawful use cannot be established, exclude that signal until its owner resolves the issue.

### 5.3 Form interaction strengths

For each user $u$ and item $i$, aggregate eligible events into a nonnegative strength $r_{ui}$.
A simple starting rule might assign 1 unit to a valid product view, 3 units to an add-to-cart event, and 6 units to a fulfilled purchase.
These are illustrative modelling weights, not empirical conversion probabilities.
A pair's strength could be the sum of the event weights in the lookback window.

Check what repeated events mean.
One hundred automated page refreshes should not dominate a single completed order.
Options include deduplication by session, capping events per day, or applying a monotone transform such as $\log(1+r)$.
Each choice changes what the model is asked to fit.
A recency factor can be applied before aggregation, but its decay should reflect the buying cycle.
For example, $w(\Delta t)=2^{-\Delta t/h}$ halves event weight every $h$ days.
Validate the half-life against temporally later behaviour rather than selecting it because it looks smooth.

Do not introduce signed values into the simple implicit preference mapping without an explicit design.
A cancellation, refund, or dislike signal has a different mechanism from a missing positive action.
Treat it in a separate policy or adopt a model whose loss explicitly supports negatives.
The choice should be reviewed with the signal owner.

**Input:** clean events and a documented weighting policy.
**Output:** distinct $(u,i,r_{ui})$ rows for $r_{ui}>0$.
**Branch:** if action frequencies vary sharply by customer size, compare caps or normalization in validation, and check whether the change removes meaningful scale.

### 5.4 Quantify the matrix and its limits

Let $U$ be the number of users and $I$ the number of items in the chosen historical universe.
Let $N$ be the number of distinct positive user–item pairs.
The interaction density is $N/(UI)$ when the rectangular universe is the declared denominator.
Sparsity is $1-N/(UI)$ under that same convention.
A regional or contractual eligibility mask may make $UI$ an overstatement of the genuinely possible pairs.
Report both the rectangular calculation and the policy-aware denominator if it can be reconstructed reliably.

Also report users with one, two to five, and more than five distinct items.
Report items with one, two to five, and more than five distinct users.
Check the age and volume of the last interaction, the distribution by channel, and the concentration in head items.
A low global density does not by itself prove that factorization will fail.
It does indicate a need to examine the connectivity of the interaction graph and cohort performance.

If a catalogue has disconnected groups of customers and items, their embeddings can be poorly comparable across components.
If only a small group of heavy buyers connects them, that group can dominate the learned relationships.
Separate regional or contractual catalogues may need separate candidate restrictions or even separate models, depending on data volume and operating needs.
Document the trade-off rather than treating one global matrix as automatically correct.

**Input:** distinct pair table and entity universe.
**Output:** density, degree distributions, graph and cohort diagnostics.
**Branch:** if users or items have no interactions, route them to a cold-start strategy rather than pretending ALS learned their preferences.

### 5.5 Define the implicit ALS objective

A standard implicit-feedback formulation sets

$$p_{ui}=\begin{cases}1 & r_{ui}>0\\0 & r_{ui}=0\end{cases},\qquad c_{ui}=1+\alpha r_{ui},\quad \alpha\ge 0.$$

Given user vector $x_u\in\mathbb{R}^f$ and item vector $y_i\in\mathbb{R}^f$, minimize

$$L(X,Y)=\sum_{u,i}c_{ui}(p_{ui}-x_u^\top y_i)^2
+\lambda\left(\sum_u\|x_u\|_2^2+\sum_i\|y_i\|_2^2\right).$$

Here $f$ is the factor dimension, $\lambda>0$ is the regularization coefficient in this stated convention, and $\alpha$ controls the additional confidence attached to strength.
The sum includes all pairs in the modelled rectangular universe, including unobserved pairs with $p=0$ and $c=1$.
That is a modelling assumption: many uncertain zeros collectively exert pressure on scores.
The objective is not a likelihood of purchases unless further assumptions establish that interpretation.

Some implementations use different confidence transformations, scaling, regularization conventions, or handling of supplied sparse matrix values.
Read the library documentation and confirm whether an input matrix contains $r$, $\alpha r$, or $c$.
Do not multiply by $\alpha$ twice.
Use this lesson's equations as the contract for its example code, not as a guarantee of a library's defaults.

**Input:** nonnegative interaction strengths and selected $f$, $\alpha$, and $\lambda$.
**Output:** an explicit mathematical objective and reproducible hyperparameter specification.
**Branch:** if absence is strongly determined by structural ineligibility, evaluate whether the rectangular-zero assumption is appropriate; eligibility filtering alone does not correct a biased training objective.

### 5.6 Alternate user and item updates

Let $Y$ be the $I\times f$ matrix whose rows are item vectors.
For a user $u$, let $C_u$ be the diagonal matrix of pair confidences and $p_u$ the preference vector.
Holding $Y$ fixed gives the user update

$$x_u=(Y^\top C_uY+\lambda I_f)^{-1}Y^\top C_up_u.$$

The inverse in the expression denotes the solution of a linear system.
In implementation, solve the system rather than explicitly forming a matrix inverse.
For a fixed user, the positive regularizer makes the normal matrix positive definite when $\lambda>0$.
Analogously, with users fixed,

$$y_i=(X^\top C_iX+\lambda I_f)^{-1}X^\top C_ip_i.$$

The confidence matrix for an item is diagonal over users.
Initialize one factor block using a documented random seed and scale.
Solve every user factor with item factors fixed.
Then solve every item factor with the new user factors fixed.
Repeat the pair of passes for a fixed iteration budget or a stated stopping rule.
Training loss need not translate monotonically to held-out business performance.
Select iterations and hyperparameters using validation, not the final test.

A sparse implementation exploits

$$Y^\top C_uY=Y^\top Y+\sum_{i:r_{ui}>0}(c_{ui}-1)y_i y_i^\top.$$

Likewise, because $p_{ui}=0$ for unobserved pairs,

$$Y^\top C_up_u=\sum_{i:r_{ui}>0}c_{ui}y_i.$$

The shared Gram matrix $Y^\top Y$ avoids materializing every zero-confidence entry.
The equivalent item update uses $X^\top X$.
Memory and computation still grow with the number of entities, factor dimension, nonzero pairs, and iterations.
Production training must budget for sparse storage, factor tables, solver workspace, and data movement.

**Input:** sparse interactions, factors, confidence rule, and solver settings.
**Output:** versioned user and item factors and training diagnostics.
**Branch:** if training is unstable or slow, check scales, duplicates, factor dimension, numerical precision, regularization, and solver configuration before interpreting the output.

### 5.7 Score and retrieve safely

For a known user, compute $s_{ui}=x_u^\top y_i$ for candidate items.
Sort by decreasing score and use a deterministic secondary key for ties.
The dot product can be negative or exceed one.
Neither outcome is a probability error because the score is not a probability.

Select the appropriate candidate universe first or enforce the same constraints before display.
Typical constraints include effective customer contract, region, product status, channel, stock, minimum remaining shelf life, and safety restrictions.
If the user is reordering consumables, previously purchased items may be desirable.
If the task is new-item discovery, suppressing prior purchases may be appropriate.
The rule depends on the decision, not on a universal recommender convention.

A fast approximate nearest-neighbour index can reduce retrieval latency at scale, but the index must implement maximum inner-product search or a mathematically equivalent transformation.
Cosine similarity is not generally identical to a raw dot product when vector norms differ.
If approximate retrieval is used, measure its recall against exact eligible top-$K$ on a representative sample.
Refresh the index when item factors change and version it alongside user factors.

If too few valid results remain after filtering, apply a documented fallback in the same eligibility universe.
For example, use regional popularity among contract-eligible, available items.
If no valid item exists, return fewer slots rather than an invalid product.
Log the model, index, policy versions, candidate counts, filter reasons, and fallback reason.

**Input:** factors, current eligibility, suppression policy, and inventory snapshot.
**Output:** a valid candidate list with lineage.
**Branch:** if a user or item is unknown to the current factor version, use an approved cold-start route.

### 5.8 Compare models without leaking the future

Choose an evaluation date and simulate information available at that date.
Train only on events and mappings available before the cutoff, accounting for ingestion delay.
Tune on a later validation interval.
Reserve a still later test interval for the final comparison.
For repeating decisions, consider multiple rolling-origin windows to reveal seasonality and drift.

For each evaluation user, construct the candidate set the system could have recommended at the decision time.
Use historical eligibility, not today's catalogue and inventory, when those historical data are available.
Declare whether previously interacted items are eligible and whether the task is replenishment or discovery.
Remove held-out events that were not eligible at the evaluation moment from the relevant set, or report their exclusion and the associated limitation explicitly.
Do not let a future purchase determine whether an item was historically in stock.

A time window can contain multiple purchases for one user.
Define whether the relevance set is distinct purchased items, first purchase after the cutoff, or another target.
State how users with no held-out relevant items are handled.
For macro recall, a common choice is to evaluate users with at least one eligible held-out relevant item and separately report their fraction of all eligible users.
This conditional metric is not the expected outcome for every session.

Compute top-$K$ recall, precision, and optionally NDCG under the same candidate and tie rules for every model.
Compare to a time-valid popularity baseline and, where useful, a content-based baseline.
Report user coverage, item coverage, stock-filter losses, latency, and cohort performance alongside ranking quality.
A model that wins on the subset of warm users but cannot serve many users may lose at the system level.

**Input:** historical snapshots, frozen model, baseline, and evaluation protocol.
**Output:** comparable metrics, denominators, coverage, uncertainty, and limitations.
**Branch:** if eligibility or exposure cannot be reconstructed, label the result an imperfect proxy and do not claim it simulates the deployed policy exactly.

### 5.9 Make the business decision and monitor

Use offline results to decide whether an online test is justified, not to claim incremental revenue.
A randomized experiment should compare the complete ALS-based policy, including filters and fallbacks, with the actual current policy.
Randomize at an entity level that avoids contamination, such as account when multiple buyers share an account-level model.
Check that assignment, exposure, and outcome logging use the same identity definition.

Predefine a primary outcome, time horizon, guardrails, analysis population, and minimum effect worth detecting.
For a distributor, fulfilled contribution margin per eligible account over a stated period may be more meaningful than clicks.
Measure wrong-item returns, complaints, stockouts, and latency as guardrails where data permit.
Investigate heterogeneous effects for cold-start users, regions, and small suppliers without treating exploratory subgroup results as confirmed causal findings.

The product owner decides launch or rollback using the agreed decision rule.
Operations owns inventory freshness and stop conditions.
Data science owns analysis and reports uncertainty, including no detectable effect.
Security and privacy owners review access, retention, and incident response.
The model owner monitors stale factors, degraded coverage, distribution drift, latency, and fallback rates.
If the catalogue or business policy changes, re-evaluate the whole serving policy, not only the model file.

### 5.10 Decide whether the data can support collaboration

Begin with the business cycle, not a universal threshold for matrix density.
An account that buys specialist valves once per year may have little recent activity but a meaningful long-term relationship.
An account ordering office supplies daily may create many events without much variety in intent.
Count active users and distinct items at several lookback lengths.
Plot the share of users that would receive an estimated factor under each length.
Also plot the share of those users whose latest interaction predates the intended buying cycle.
This identifies the trade-off between more data and more stale data.

Measure user degree as the number of distinct items with a positive interaction.
Measure item degree as the number of distinct users with a positive interaction.
Compute these distributions separately for each decision surface and major market.
A single global average can conceal a market with insufficient history.
It can also conceal a few customers responsible for most events.
For each cohort, compare the median and upper tail with the mean.
Review source records for extreme values before treating them as real behaviour.

Inspect connectivity in the bipartite graph of users and items.
Two nodes are connected by an edge when their pair has a positive interaction.
Identify the largest connected component and the proportions of users, items, and events it contains.
List the largest disconnected components and the business reasons for separation.
Different jurisdictions, incompatible product categories, or restricted contracts can create genuine separation.
A mapping defect can create accidental separation.
Do not merge identifiers solely to make the graph look better.

Check whether the interactions bridge the groups the system is expected to serve.
If all links between two catalogue regions come from one reseller, its unusual purchasing strategy can influence both regions.
Temporarily excluding that reseller in an offline sensitivity analysis can reveal dependence on one bridge.
The result is a diagnostic comparison, not a causal estimate of that customer's effect.
If removing a small number of users changes the ranking drastically, report the instability.
Consider separate models or a fallback where cross-group transfer lacks evidence.

Examine item age and account tenure at the historical decision date.
Items introduced after that date must not appear in the training catalogue.
Accounts created after that date must not receive a retrospectively fitted user vector.
Split evaluation results into genuinely new, lightly observed, and established cohorts.
State the actual boundaries, such as zero, one to four, and at least five prior distinct items.
The boundaries are operational choices, not universal statistical laws.

For every cohort, report both model-only and full-policy results.
Model-only results reveal factor quality on entities that have factors.
Full-policy results include fallback and reflect what users actually receive.
If model-only recall rises but the fallback covers most traffic, the business gain may be small.
The serving share and the quality on the served share are both required for a decision.

### 5.11 Translate action strength into a defensible policy

Start with one event family whose meaning is understood, such as fulfilled orders.
Build a documented baseline before adding views, searches, or cart activity.
A click can indicate curiosity, accidental navigation, or an attempt to resolve a problem.
A fulfilled order is stronger evidence of a completed transaction but still not proof of satisfaction.
An invoice line can represent a contractual obligation rather than active selection.
The same event type can have different meaning for a direct buyer and an automated replenishment system.

For each signal, name its unit and observation mechanism.
A view count is in logged views, not dollars or probability points.
An order-line count is in order lines, not necessarily physical units.
A quantity measure may be dominated by pack-size conventions and account scale.
If quantities are used, verify whether cases and individual units have been normalized consistently.
Do not sum quantities across incompatible units without a defined transformation.

Choose weights on a training and validation process, not the held-out test.
Compare a small number of interpretable candidates against the order-only baseline.
For instance, assess whether adding session-deduplicated views improves later eligible purchase ranking.
Separately inspect whether view-heavy customers dominate the loss.
If a signal produces a lift only for a few heavy users, decide whether that cohort matches the business objective.
Avoid selecting weights from a single lucky time window.

Investigate the role of repeated orders.
Repeated purchases can mean an item is essential and worth recommending again.
They can also reflect an automated schedule that recommendations cannot change.
The strength function should match the intended task.
For discovery, large repeat counts may overemphasize known staples.
For replenishment, frequency and elapsed time may be informative but a static factor alone cannot model reorder timing.
Evaluate those alternatives against task-specific baselines.

Set treatment for promotions and price changes.
A promotion can generate real interactions while revealing little about normal-price preference.
Do not remove every promotional order automatically; the future surface may also run promotions.
Instead, stratify evaluation by promotion state and check robustness across windows.
If the model needs to respond to current price, provide that information to a suitable downstream policy.
Implicit ALS on historical counts does not directly condition on the current offer price.

Write a decision record for exclusions.
It should identify source owner, reason, effective date, and possible bias.
For example, excluding returns may remove useful evidence of incorrect fit but avoid treating shipping damage as product dislike.
Monitor the excluded population and revisit the rule when reason codes improve.
Use explicit reason-specific signals only when their semantics and permitted use are established.

### 5.12 Select hyperparameters and compare compute cost

The factor dimension controls representation capacity.
Too few factors can blur distinct customer needs.
Too many can fit noise and increase storage and computation.
Compare a small candidate set of dimensions on the same validation windows.
For each candidate, report ranking, cohort coverage, training time, artifact size, and serving latency.
Choose the smallest model that meets the decision criteria when differences are practically immaterial.

The confidence coefficient controls the incremental weight of observed strengths relative to the baseline weight of one.
Its meaning depends on the interaction-strength scale.
Multiplying all strengths by ten while keeping alpha fixed changes the objective.
Therefore report the transform and alpha together.
A value selected for raw order counts cannot be copied safely to logarithmically transformed counts.
When migrating libraries, verify whether confidence is formed internally.
Repeat a hand-checkable calculation before interpreting performance changes.

Regularization limits factor magnitudes under the stated loss.
Its effective strength also depends on the number of observations and implementation conventions.
Some implementations scale regularization by interaction count; others do not.
Record the exact convention alongside the numeric setting.
An apparently identical lambda in two libraries may optimize different objectives.
Tune it using validation rather than assuming an external default is portable.

Iteration count is a computational budget and a modelling choice.
Track the training objective if the implementation exposes a comparable definition.
Also track validation ranking after selected iterations.
Improving training loss does not guarantee improved later ranking.
Stop according to a declared budget or validation rule; reserve the test interval for final assessment.
If several configurations are tried, report that selection occurred on validation.

Estimate factor storage before deployment.
With $U$ users, $I$ items, $f$ dimensions, and 32-bit floats, raw factor storage is approximately $4f(U+I)$ bytes.
This excludes identifiers, indexes, replicas, metadata, and solver workspace.
For one million users, one hundred thousand items, and 64 dimensions, the raw vectors need about 281.6 million bytes.
That is approximately 268.6 mebibytes, before operational overhead.
The estimate is a capacity input, not a promise about total system memory.

Estimate the exact-scoring workload for the actual candidate universe.
Scoring every eligible item for every request can be expensive at catalogue scale.
Approximate retrieval can reduce work but introduces index build, refresh, and approximation error.
Measure candidate recall against exact scoring on a sample with the same eligibility rules.
An index trained on stale item factors can silently produce irrelevant candidates even when application health checks pass.
Keep index version, factor version, and catalogue mapping together.

### 5.13 Define an experiment that measures incrementality

The experiment compares policies, not isolated vectors.
The treatment policy includes ALS retrieval, filtering, ranking, display, and fallback.
The control policy is the current complete experience, subject to the same mandatory safety rules.
If the control has a different number of filled slots, record that difference.
Slot count may be part of the treatment effect rather than a defect, depending on the decision question.

Choose the assignment unit before exposure.
An account-level model shared by several employees generally calls for account-level randomization to limit contamination.
If one warehouse serves several accounts, examine possible spillovers through shared stock.
For example, treatment may exhaust inventory that control customers would otherwise buy.
Such interference complicates an ordinary account-level causal interpretation.
The experiment owner should decide whether a different unit or inventory guardrail is needed.

Specify eligibility for experiment entry without using future outcomes.
Record all assigned eligible accounts, including those that never click a recommendation.
An intention-to-treat analysis compares outcomes by assignment for that population.
An exposure-conditioned analysis can be useful operationally but is vulnerable to selection after assignment.
Report the two populations separately and do not relabel the latter as the randomized effect.

Define the primary outcome in operational units.
For example, fulfilled contribution margin in currency per eligible account over 28 days has a clear numerator and denominator.
Define contribution margin with finance, including the treatment of discounts, fulfilment costs, cancellations, and returns.
Allow sufficient maturation time for returns and late-arriving transactions.
A short click window and a long profit window answer different questions.
Do not add attributed recommendation revenue to total order revenue if that double-counts the same transaction.

Agree on a minimum effect worth the operational cost and risk.
Determine the required sample and duration using the planned unit and observed outcome variability.
Include seasonality and account-level clustering in the design discussion.
Do not promise that any fixed sample will detect every plausible effect.
If the experiment is underpowered, an inconclusive interval is not proof of no effect.

Set guardrails with explicit action owners.
Examples include an increase in incompatible-item returns, a decrease in order fill rate, excessive latency, or elevated fallback errors.
Specify whether a threshold triggers investigation, pause, or automatic rollback.
Review safety-critical exceptions immediately rather than waiting for a scheduled analysis.
Document any interim monitoring and its effect on statistical interpretation before the test begins.

At analysis, report effect estimates and uncertainty in the primary unit.
Check assignment counts and logging completeness without excluding inconvenient outcomes.
Use prespecified subgroup analyses carefully; many comparisons can yield chance findings.
If the trial fails a guardrail, a positive click result does not by itself authorize launch.
The product owner records the decision and the responsible owners for any remediation.

### 5.14 Operate the model after launch

Create an artifact manifest for every release.
Include training cutoff, source snapshot identifiers, event policy, factor dimension, confidence convention, regularization, iteration count, and library version.
Include checksums or immutable identifiers for user factors, item factors, and retrieval index.
Record the eligibility policy and fallback version independently because they can change without retraining.
Maintain a known-good prior release for rollback under the organization's change-control process.

Monitor ingestion and transformation before monitoring recommendations.
A missing purchase feed can cause a delayed decline in model quality without an immediate serving error.
Alert on source freshness, duplicate rate, entity mapping failures, and unusual shifts in interaction strength.
Compare catalogue and inventory snapshots against their service-level agreements.
Treat the absence of a signal differently from a measured zero.

Monitor serving at each stage.
Count requested recommendations, known users, retrieved candidates, valid candidates, rendered slots, and fallback uses.
Break down filter reasons by inventory, contract, safety, and catalogue status where policy permits.
Track latency percentiles rather than only average latency.
An apparently healthy model can still create empty pages if a downstream filter rejects most candidates.
Investigate changes by region and customer cohort.

Monitor outcomes over appropriately matured periods.
Report engagement alongside orders, fulfilment, returns, and customer complaints.
Separate a model refresh from a simultaneous price or catalogue change where possible.
An observational decline after a refresh is a warning to investigate, not automatically proof of causation.
Maintain the ability to disable personalization without disabling mandatory eligibility rules.

Schedule periodic review of business assumptions.
An account merger, new product line, supply disruption, or change in buying process can invalidate old interaction patterns.
Reassess whether the user entity, lookback window, and objective still match the decision.
Review retention and data access at the same time.
The model owner recommends action; the relevant policy and business owners approve changes within their authority.

## 6. Worked example

### 6.1 Business setting and input

A fictional industrial-parts distributor wants a three-slot discovery panel for existing business accounts.
A recommendation is a suggestion, not an order or an inventory commitment.
The merchant wants to find previously unpurchased items that customers might investigate.
All amounts below are illustrative; no company outcome is claimed.

The training window ends at 23:59 UTC on 31 March.
Only events that arrived by that cutoff are included.
The held-out interval is 1–30 April.
The example assumes four items were active, legally eligible, and available to every illustrated account at the decision timestamp.
In production those conditions would be reconstructed per account and location.

| Item | Description | Operational note |
|---|---|---|
| A | Nitrile gloves, box | Consumable; repeat purchase is plausible. |
| B | Cleaning wipes, pack | Consumable; repeat purchase is plausible. |
| C | Safety goggles, pair | Protective item; suitability must be verified. |
| D | Ear protection, pack | Protective item; suitability must be verified. |

For this discovery panel only, items purchased before the cutoff are excluded from the slate.
The policy does not assert that a previous purchase makes a product unsuitable for replenishment.
The business could run a separate reorder panel with a different rule.

Assume a fulfilled purchase contributes one interaction-strength unit.
Duplicate order messages have already been removed.
No views are included in this example.
The historical pair strengths are:

| Account | A | B | C | D |
|---|---:|---:|---:|---:|
| U1 | 2 | 1 | 0 | 0 |
| U2 | 1 | 0 | 1 | 0 |
| U3 | 0 | 1 | 1 | 1 |

There are 3 accounts and 4 items, so the rectangular universe has 12 pairs.
There are 7 distinct positive pairs.
Density is $7/12=58.33\%$ and sparsity is $5/12=41.67\%$.
This tiny example is deliberately far denser than many real catalogues; it is chosen for hand calculation.
The count of 8 purchase events is not the count of 7 distinct positive pairs.

### 6.2 Preferences and confidence

Set $\alpha=2$ per interaction-strength unit.
A pair with strength 2 has $p=1$ and $c=1+2(2)=5$.
A pair with strength 1 has $p=1$ and $c=3$.
A pair with strength 0 has $p=0$ and $c=1$.
For U1 the preference vector is $(1,1,0,0)$.
Its confidence diagonal is $(5,3,1,1)$.
The two zero-preference entries are low-confidence modelling targets, not recorded statements of dislike.

### 6.3 One exact user update

To make the computation transparent, suppose the item vectors at a particular ALS iteration are fixed as follows:

| Item | $y_{i1}$ | $y_{i2}$ |
|---|---:|---:|
| A | 1 | 0 |
| B | 0 | 1 |
| C | 1 | 1 |
| D | 1 | -1 |

These are assumed current factors, not factors claimed to have been learned from the table.
Use rank $f=2$ and $\lambda=1$ in the stated objective.
For U1, the four contributions to $Y^\top C_1Y$ are:

$$5\begin{bmatrix}1&0\\0&0\end{bmatrix},\quad
3\begin{bmatrix}0&0\\0&1\end{bmatrix},\quad
\begin{bmatrix}1&1\\1&1\end{bmatrix},\quad
\begin{bmatrix}1&-1\\-1&1\end{bmatrix}.$$

Their sum is $\begin{bmatrix}7&0\\0&5\end{bmatrix}$.
After adding $\lambda I_2$, the normal matrix is $A=\begin{bmatrix}8&0\\0&6\end{bmatrix}$.
The right-hand side is $5(1,0)+3(0,1)=(5,3)$.
Solving $Ax=b$ gives $x_{U1}=(5/8,1/2)=(0.625,0.5)$.

The scores under these fixed item vectors are:

| Item | Calculation | Score | Discovery eligibility |
|---|---|---:|---|
| A | $0.625(1)+0.5(0)$ | 0.625 | Excluded: previously purchased. |
| B | $0.625(0)+0.5(1)$ | 0.500 | Excluded: previously purchased. |
| C | $0.625(1)+0.5(1)$ | 1.125 | Eligible, subject to suitability. |
| D | $0.625(1)+0.5(-1)$ | 0.125 | Eligible, subject to suitability. |

The discovery order for U1 is C then D.
Although the panel has three slots, only two eligible unseen items exist in this four-item example.
Returning two is correct; it is not permission to show an excluded or unsafe item.
The score 1.125 is not a 112.5% probability.

The next ALS step would update item vectors using all user vectors, not only U1.
Further iterations would change the scores.
The hand calculation illustrates one conditional solve, not convergence or evidence that this model wins.

### 6.4 Evaluate a frozen ranking

Suppose a separate, already-frozen full model produces these eligible discovery lists at $K=2$.
Assume each listed held-out item was eligible at the decision time.
Do not infer these U2 and U3 scores from the single U1 update above.

| Account | ALS top 2 | Baseline top 2 | Distinct eligible April purchases |
|---|---|---|---|
| U1 | C, D | D, C | C |
| U2 | B, D | B, D | D |
| U3 | A | A | A |

For U1, both policies retrieve the one relevant item.
For U2, both retrieve the one relevant item.
For U3, both retrieve the one relevant item.
Macro recall@2 is therefore $(1+1+1)/3=1.0$ for each policy.
Each account has one relevant hit; with the unfilled U3 slot counted in the fixed-$K$ denominator, macro precision@2 is $0.5$ for each policy.

Binary NDCG@2 differentiates position here.
With exactly one relevant item, a rank-one hit receives 1 and a rank-two hit receives $1/\log_2(3)\approx0.6309$.
ALS obtains $(1+0.6309+1)/3\approx0.8770$.
The baseline obtains $(0.6309+0.6309+1)/3\approx0.7539$.
The difference of about 0.1231 is descriptive for three accounts, not evidence of a population effect.
U3 has only one eligible unseen item because B, C, and D were previously purchased.
Under the fixed-$K$ precision convention, the unfilled slot counts in the denominator.
The candidate eligibility audit precedes metric interpretation.
In a real catalogue, valid fallback items could fill the slot when available.

### 6.5 Decision and ownership

The merchant should not launch on these three accounts.
The analysis is too small, and the hand-factor calculation is not a trained model evaluation.
The data scientist should run rolling time-based evaluation with realistic catalogue snapshots and compare the full fallback policy with the existing policy.
The safety owner should specify which protective items require compatibility or certification checks.
The platform owner should demonstrate that invalid candidates cannot reach the display.
If those gates pass, the product owner can sponsor a randomized account-level trial with a predefined guardrail for incorrect protective-equipment orders.

## 7. Code, query, or calculation

The following complete Python program uses only NumPy and the standard library.
Install NumPy in the environment if it is not already available.
It computes the exact U1 update above, trains a small implicit ALS model, and reports scores for unseen items.
It materializes a tiny dense matrix for teaching; do not use this implementation for a production-scale catalogue.
The program uses the objective and regularization convention stated in section 5.
It is deterministic for the stated seed, although small floating-point differences across numerical libraries are possible.

```python
import numpy as np

# Rows are U1, U2, U3; columns are A, B, C, D.
# Each value counts fulfilled purchases during the training window.
r = np.array([
    [2.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 1.0],
])
alpha = 2.0
regularization = 1.0
preference = (r > 0).astype(float)
confidence = 1.0 + alpha * r

# One conditional user solve using the hand-specified item factors.
y_fixed = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0, -1.0],
])
cu = confidence[0]
normal = y_fixed.T @ (cu[:, None] * y_fixed)
normal += regularization * np.eye(y_fixed.shape[1])
rhs = y_fixed.T @ (cu * preference[0])
x_first = np.linalg.solve(normal, rhs)
print('normal matrix:', normal.tolist())
print('right-hand side:', rhs.tolist())
print('U1 factor:', x_first.tolist())
print('fixed-item scores:', (y_fixed @ x_first).tolist())


def update_rows(other, p, c, reg):
    """Solve one regularized weighted least-squares problem per row."""
    factors = np.empty((p.shape[0], other.shape[1]))
    identity = np.eye(other.shape[1])
    for row in range(p.shape[0]):
        a = other.T @ (c[row, :, None] * other) + reg * identity
        b = other.T @ (c[row] * p[row])
        factors[row] = np.linalg.solve(a, b)
    return factors


def fit_implicit_als(p, c, rank=2, reg=1.0, iterations=12, seed=7):
    """Dense educational implementation; rows of p are users."""
    rng = np.random.default_rng(seed)
    item_factors = rng.normal(0.0, 0.1, size=(p.shape[1], rank))
    for _ in range(iterations):
        user_factors = update_rows(item_factors, p, c, reg)
        item_factors = update_rows(user_factors, p.T, c.T, reg)
    return user_factors, item_factors


users, items = fit_implicit_als(preference, confidence)
scores = users @ items.T
item_names = np.array(['A', 'B', 'C', 'D'])
for user in range(r.shape[0]):
    unseen = np.flatnonzero(r[user] == 0)
    # Sort ties by the item name for a reproducible policy.
    ordered = sorted(unseen, key=lambda i: (-scores[user, i], item_names[i]))
    print(f'U{user + 1} unseen:', [item_names[i] for i in ordered])
```

The first four output lines are exactly, apart from floating-point display formatting:

```text
normal matrix: [[8.0, 0.0], [0.0, 6.0]]
right-hand side: [5.0, 3.0]
U1 factor: [0.625, 0.5]
fixed-item scores: [0.625, 0.5, 1.125, 0.125]
```

The final three lines show ranked unseen item names from the iterative fit.
They depend on the numerical fit and are not the manually specified rankings in section 6.4.
The training code does not implement time splitting, production eligibility, inventory, fallback, or a business experiment.
Those are deliberate boundaries of this calculation, not optional production controls.

For a production implementation, use a maintained sparse library, verify its confidence convention, and compare its output against a small hand-checkable case.
Record the library and numerical backend versions, random seed, dataset snapshot, feature policy, hyperparameters, and factor/index versions.
A seed alone does not guarantee bitwise reproducibility across hardware or parallel execution strategies.

## 8. Data, standards, and regulatory context

### 8.1 Data lineage and identifiers

The event table should identify the source system, original event identifier, event time, ingestion time, user key, item key, event type, and event status.
An effective-dated mapping connects these keys to account and product entities.
A catalogue merge or account migration must not retroactively reveal information that was unavailable at the evaluation cutoff.
Preserve transformation versions so an analyst can reproduce a historical training snapshot.

GS1's Global Trade Item Number (GTIN) standard identifies trade items across trading partners.
A GTIN is not automatically a seller's internal SKU, a warehouse location, or an account-specific offer.
Product variants can have different identifiers, and identifier assignment must follow the relevant GS1 rules.
A correct identifier helps reconcile events and catalogue data; it does not certify compatibility, availability, or relevance.
The GS1 source in section 16 is a standards reference, not evidence that every dataset has valid GTINs.

A well-defined inventory record includes item, fulfilment location, timestamp, available quantity, reservation rules, and freshness.
A positive warehouse balance may not mean a particular account can purchase the item.
Contract eligibility, allocation, regulatory restrictions, and shipping constraints can independently disqualify it.
An as-of inventory snapshot is especially important when the evaluation asks whether an item could actually have been shown.

### 8.2 Personal data and lawful use

Depending on context, user identifiers, browsing histories, and inferred preferences may be personal data.
Business-to-business accounts can still contain identifiable individual buyers.
Pseudonymous identifiers reduce direct exposure but do not automatically make data anonymous.
The controller must determine the applicable jurisdiction, lawful basis, notice, retention, and data-subject-rights processes with qualified counsel.

The European Union General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, includes principles such as purpose limitation, data minimization, accuracy, storage limitation, and security in Article 5.
Article 6 addresses lawful bases for processing.
Articles 13 and 14 address information to be provided in specified circumstances.
Article 22 addresses certain solely automated decisions with legal or similarly significant effects; ordinary product ranking should not automatically be assumed to meet or avoid that threshold.
These provisions are binding where applicable, but their application depends on facts and legal interpretation.
This lesson is educational, not legal advice.

For California, the California Consumer Privacy Act as amended by the California Privacy Rights Act provides rights and obligations subject to statutory definitions, exemptions, and applicability conditions.
The California Privacy Protection Agency's published materials are a starting point, not a substitute for current legal review.
Do not infer a universal consent requirement or exemption from this short overview.
A deployment should map the actual data flow and obtain jurisdiction-specific advice.

### 8.3 Controls for interaction data

Minimize event attributes to those needed for the stated purpose.
Separate access to raw histories, training data, factors, and serving logs.
A learned embedding can encode information about behaviour and should not be treated as inherently anonymous.
Encrypt data in transit and at rest under the organization's security policy.
Restrict exports and record access to sensitive datasets.
Set retention windows for raw events, intermediate snapshots, factors, logs, and backups.

A deletion or correction process needs to address derived artifacts as well as source records.
The operational response may include excluding the record from subsequent training and rebuilding affected models, depending on policy and legal advice.
Document propagation time, owners, and verification of the process.
Do not promise that deleting a source row immediately removes all influence from a previously trained model.

### 8.4 Exposure, consent, and auditability

An exposure log should distinguish candidate retrieval, policy filtering, actual rendering, and subsequent interaction.
Record the policy version and timestamp for each stage.
If randomized exposure is used for measurement, retain the assignment and known selection probability where lawful and appropriate.
A deterministic historical policy usually does not supply nonzero propensity for alternatives it never showed.
Inverse propensity methods require known or defensibly estimated propensities and support; they cannot recover outcomes for actions that were impossible under the logging policy without extra assumptions.

Privacy and product-safety review are separate controls.
A lawful data use does not prove a product is safe to recommend.
A safe product does not justify an otherwise impermissible data use.
The merchant, legal, security, and operations owners must each approve their part of the deployment.

## 9. Industry practice and case evidence

### 9.1 Implicit feedback in practice

Hu, Koren, and Volinsky's 2008 paper describes collaborative filtering for implicit feedback and develops a confidence-weighted factorization objective.
The source distinguishes positive observations from confidence in preference and discusses how to exploit sparse interactions.
This lesson follows the general mathematical approach while making its own explicit assumptions about event weights and the evaluation contract.
The paper does not establish that any particular distributor's purchase history is representative or that ALS will outperform a baseline in that distributor's catalogue.

Koren, Bell, and Volinsky's survey on matrix factorization describes the broad role of latent factors in recommendation.
Much of the public literature discusses rating prediction; purchase and exposure data require different missingness assumptions.
A historical rating benchmark is not a direct measurement of online incremental sales.
The appropriate operational baseline and evaluation population must be specified for the actual placement.

### 9.2 Public company case: Amazon

Linden, Smith, and York describe Amazon.com's item-to-item collaborative filtering approach in a 2003 IEEE Internet Computing article.
Their account describes relating products through customer behaviour and generating recommendations from similar items.
The published case supports the feasibility and historical use of item-based collaborative techniques at a large retailer.
It does not establish that Amazon used the particular implicit ALS implementation in this lesson.
Nor does it supply a transferable causal uplift for a different company's model.

**Inference:** an item-to-item approach may be easier to explain operationally than a latent-factor score when the product owner wants to inspect specific related items.
That is a design hypothesis, not a measured outcome from the cited case.
Both approaches remain vulnerable to exposure and availability distortions in their input data.

### 9.3 Public company case: Netflix

Gomez-Uribe and Hunt describe the Netflix recommender system and its business context in a 2015 ACM Transactions on Management Information Systems article.
The article describes multiple recommendation problems and a broader personalization system rather than a single universally sufficient model.
It is useful evidence that recommendation is a product and evaluation problem as well as a modelling problem.
The service's media catalogue, consumption patterns, and commercial objectives differ from industrial inventory and replenishment.
The article should not be read as proof that its reported business context or methods transfer unchanged to a supply-chain setting.

### 9.4 Operational implications

A commerce team can use collaborative candidates as input to a later ranker, but must test the complete policy.
Item coverage can improve or worsen depending on data, hyperparameters, and filters.
More personalization can concentrate demand on a small number of items, especially when training data already reflect popularity.
Conversely, a long-tail candidate that is unavailable or restricted is not a useful recommendation.
Neither result should be asserted without measurement in the target setting.

Organizations often operate a portfolio of retrieval methods rather than insisting on one model for every case.
Popularity serves as an interpretable benchmark and fallback.
Content-based rules help with new products when reliable attributes exist.
Collaborative filtering can exploit repeated interactions for warm entities.
Session signals can reflect immediate intent that long-run account factors miss.
The correct mixture is a business and empirical choice, not an inherent property of ALS.

A public case is evidence about the source's stated setting and period.
It is not a guarantee of present-day architecture, an audited causal effect, or an implementation recipe.
Record the source date and distinguish company statements from independent evaluation.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Exposure bias | Items with no impressions appear as zeros. | The old policy controls what can be clicked or purchased. | The model repeats historical visibility. | Log exposure; test exploration under appropriate controls. |
| Stockout confounding | Popular items vanish from training after stockouts. | Lack of supply suppresses observed demand. | The model treats unavailability like weak interest. | Retain as-of availability; analyze stockout periods separately. |
| Future leakage | Offline lift disappears in prospective replay. | Future events, identities, or attributes enter training. | Inflated offline metrics. | Use event and arrival cutoffs with effective-dated joins. |
| Identity collapse | Different buyers share unexpectedly similar histories. | Account and individual identifiers are merged improperly. | Irrelevant or unauthorized personalization. | Define the user entity and access rules; audit mappings. |
| Duplicate messages | A few pairs have implausibly large counts. | Retries are counted as independent orders. | Excessive confidence and distorted factors. | Deduplicate by event identity and inspect outliers. |
| Cold user | Large fallback fraction for new accounts. | No user factor can be estimated from interactions. | Low personalized coverage. | Use a permitted popularity or content fallback. |
| Cold item | New products never appear in recommendations. | No interactions exist to estimate their factors. | Slow discovery and supplier disadvantage. | Add a content or controlled exploration route. |
| Wrong confidence scale | Training changes dramatically after a library migration. | The input is scaled by alpha twice or interpreted as confidence. | Unstable comparisons and poor fit. | Document library semantics and check a small example. |
| Repeat suppression error | Replenishment customers stop seeing staples. | A discovery rule is applied to a reorder surface. | Missed repeat orders and poor experience. | Define suppression per task and compare policies. |
| Unavailable items | Large post-retrieval filter rate. | Item factors are not inventory-aware. | Empty slots or unfulfillable recommendations. | Use current eligibility and a valid fallback; monitor freshness. |
| Unsafe substitution | High predicted affinity for incompatible parts. | Co-purchase is mistaken for functional equivalence. | Damage, returns, or safety risk. | Apply verified compatibility and safety rules before display. |
| Catalogue drift | New factor scores reference retired SKUs. | Model, index, and catalogue snapshots are out of sync. | Invalid items or silent candidate loss. | Version artifacts and enforce effective status at serving. |
| Head-item domination | A few items occupy most slots. | Popularity and high-degree users dominate the loss. | Reduced discovery and supplier concentration. | Measure concentration and evaluate policy constraints. |
| Misleading offline gain | Recall rises but fulfilled margin falls online. | Predictive relevance is not incremental value. | Commercial harm despite a metric win. | Run randomized policy experiments with guardrails. |
| Noncomparable evaluation | One model sees more eligible candidates than another. | Candidate and filtering rules differ. | Apparent lift is not attributable to the model. | Freeze the candidate universe and report coverage. |
| Deletion gap | A removed user persists in factor snapshots. | Derived artifacts outlive source records. | Privacy and governance breach. | Define propagation, retraining, access, and retention controls. |
| Position bias | Top-slot items receive most clicks. | Visibility rather than relevance drives engagement. | Learned ranking reinforces old positions. | Log positions and use randomized measurement where permitted. |
| Factor incompatibility | Serving scores shift after a partial deployment. | User and item vectors come from different model versions. | Invalid score geometry. | Atomically version and deploy compatible factors and index. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Interactions reflect some useful signal. | ALS learns from observed behaviour. | Purchases are mandated or automated without choice. | Compare signal types and evaluate by procurement process. |
| Recorded events are sufficiently complete. | Missing channels distort user histories. | Sales occur through a disconnected field channel. | Reconcile source totals and report channel coverage. |
| Entity keys are stable and authorized. | Rows and columns must refer to meaningful entities. | Account merges, shared logins, or consent changes. | Audit effective-dated identity and permissions. |
| Strength is nonnegative under this formulation. | Preference and confidence are defined from positive $r$. | Returns or dislikes are supplied as negative counts. | Separate the signal or use an explicitly designed objective. |
| Unobserved pairs have baseline confidence one. | This is part of the stated implicit objective. | Most unobserved pairs were structurally impossible. | Examine eligibility and compare alternative formulations. |
| Similar historical patterns have predictive value. | Factorization transfers information across entities. | Requirements are unique or change abruptly. | Evaluate temporal cohorts and compare with baselines. |
| Historical availability can be approximated. | Offline candidate sets must resemble feasible choices. | Inventory snapshots are missing or delayed. | Report proxy bias and prioritize logging. |
| Item identity corresponds to the served choice. | Factors must attach to the actual selectable item. | Family-level data are mapped to an incompatible variant. | Audit mappings and enforce variant eligibility. |
| Future resembles the validation windows enough. | Hyperparameter selection relies on temporal transfer. | A promotion, shock, or assortment reset changes demand. | Use rolling windows and monitor post-launch drift. |
| Relevant outcomes are observed. | Offline recall needs a held-out target. | Buyers purchase through another channel. | Link permissible sources and quantify missingness. |
| Online assignment is randomized as planned. | Causal comparison requires comparable groups. | Cross-account contamination or selective exposure. | Randomize at the shared identity level and audit assignment. |
| Safety policy is external and current. | Affinity alone does not establish product suitability. | Compatibility tables are stale or incomplete. | Assign a safety owner and fail closed on mandatory rules. |

No factor dimension is inherently interpretable.
The factorization can be rotated without changing dot products, so coordinate labels should not be used as stable explanations across runs without additional evidence.
A recommendation score may be useful for ordering and still unsuitable for comparing absolute preferences across users or model versions.
The numerical objective does not encode causality, product compatibility, fair allocation, or contractual entitlement.

## 12. Exercises

### Exercise 1 — Matrix structure and sparsity

**Time:** 15 minutes.
A training window contains the following fulfilled-order events: U1–A twice, U1–B once, U2–A once, and U3–C once.
The declared universe contains three users and four items A–D.
One duplicate delivery of the U1–B event is removed before counting.

**Deliverable:** a 3-by-4 strength matrix, the number of distinct positive pairs, density, and sparsity.
**Acceptance criteria:** use event counts for strength but distinct pairs for density; state the denominator.

### Exercise 2 — Preference and confidence

**Time:** 10 minutes.
Use $r=(2,1,0,0)$ and $\alpha=2$.

**Deliverable:** the four preference values and the four confidence values.
**Acceptance criteria:** compute $p=1[r>0]$ and $c=1+2r$; explain why the zero entries are not observed dislikes.

### Exercise 3 — One ALS update

**Time:** 25 minutes.
Use the item factors, U1 confidence, and regularization from section 6.3.

**Deliverable:** the normal matrix, right-hand side, U1 factor, and scores for C and D.
**Acceptance criteria:** include the contributions from unobserved C and D at confidence one; solve the linear system rather than asserting a trained result.

### Exercise 4 — Ranking metrics

**Time:** 20 minutes.
For one user, the top three items are A, B, C and the relevant eligible set is B and D.
Use binary relevance, $K=3$, and $\mathrm{DCG}@3=\sum_{j=1}^3\mathrm{rel}_j/\log_2(j+1)$.

**Deliverable:** recall@3, precision@3, DCG@3, and NDCG@3.
**Acceptance criteria:** use both relevant items in the recall denominator; compute ideal DCG using two relevant items at ranks one and two.

### Exercise 5 — Time and eligibility

**Time:** 20 minutes.
A training cutoff is 31 March at 23:59 UTC.
An order occurred on 30 March but arrived on 2 April.
An item became eligible for an account on 5 April.
An account bought it on 10 April.
The simulated recommendation decision was made on 1 April.

**Deliverable:** a short inclusion decision for the late order and the held-out item, plus one missing data requirement.
**Acceptance criteria:** distinguish event time from availability of information and evaluate eligibility at the decision time.

### Exercise 6 — Critique a deployment claim

**Time:** 20 minutes.
A team says: “Our offline recall improved 8 percentage points, so showing the ALS list will increase fulfilled profit by 8%.”
The team evaluated only users with a held-out purchase and used today's inventory to filter historical recommendations.

**Deliverable:** identify at least four independent defects or missing facts and propose one action per defect.
**Acceptance criteria:** address causal interpretation, metric units, selected population, and time-valid inventory.

### Exercise 7 — Design a decision and experiment

**Time:** 30 minutes.
A distributor wants to personalize a safety-equipment discovery panel for multi-buyer accounts.
Some items require compatibility checks; new accounts lack history.

**Deliverable:** a one-page decision memo defining user, target, eligibility, fallback, randomization unit, primary outcome, two guardrails, and owners.
**Acceptance criteria:** make the control a complete existing policy; do not treat an affinity score as a safety approval; include how a no-history account is served.

### Exercise 8 — Data controls

**Time:** 20 minutes.
The input includes identifiable buyer events, supplier identifiers, catalogue records, and learned factors.
The system serves EU and California customers.

**Deliverable:** a four-row control map covering purpose and lawful use, retention, access, and deletion propagation.
**Acceptance criteria:** distinguish source records from derived factors; cite appropriate official sources and route jurisdiction-specific decisions to counsel.

## 13. Self-check answers

### Answer 1

The matrix rows U1, U2, U3 and columns A, B, C, D are

$$\begin{bmatrix}2&1&0&0\\1&0&0&0\\0&0&1&0\end{bmatrix}.$$

There are four distinct positive pairs in a universe of $3\times4=12$ pairs.
Density is $4/12=33.33\%$ and sparsity is $8/12=66.67\%$.
The five valid order events do not imply five occupied cells.
The duplicate message does not add a sixth event.

### Answer 2

The preference vector is $(1,1,0,0)$.
The confidence vector is $(5,3,1,1)$.
The zero entries are low-confidence targets in the objective; they do not prove the customer saw and rejected the items.

### Answer 3

The normal matrix is $\operatorname{diag}(8,6)$ and the right-hand side is $(5,3)$.
The user factor is $(0.625,0.5)$.
The score for C is $1.125$ and the score for D is $0.125$.
The unobserved items contribute to the normal matrix even though they contribute zero to the right-hand side.
This is one conditional update, not a complete trained model.

### Answer 4

Only B appears in both the top-three list and the relevant set.
Recall@3 is $1/2=0.5$ and precision@3 is $1/3\approx0.3333$.
DCG@3 is $1/\log_2(3)\approx0.6309$.
Ideal DCG@3 is $1+1/\log_2(3)\approx1.6309$.
NDCG@3 is approximately $0.6309/1.6309\approx0.3869$.
The metric uses binary relevance; graded relevance would require a different stated gain rule.

### Answer 5

Exclude the late-arriving order from a faithful 1 April as-of training snapshot because it was not yet available, even though its event time precedes the cutoff.
The item first became eligible after the simulated decision, so it was not a valid candidate on 1 April.
Do not silently count the later purchase as a missed eligible recommendation.
An effective-dated eligibility record and an arrival-time field are necessary to make these decisions.
Report excluded outcomes and reassess the protocol if eligibility changes make the chosen horizon unrepresentative.

### Answer 6

A strong critique includes the following points.
Offline recall is predictive, not a causal estimate of incremental fulfilled profit; run a randomized policy comparison.
An eight-percentage-point change in recall is not an 8% change in profit; report metric units and a business outcome separately.
Conditioning on users with a held-out purchase excludes other eligible users; report coverage and evaluate the full policy online.
Today's inventory leaks future availability and may misstate feasible historical recommendations; reconstruct as-of eligibility or disclose the limitation.
Also check whether baselines, candidate sets, tie rules, uncertainty, and time windows were identical.

### Answer 7 rubric

A strong memo defines whether the account or buyer receives the decision and explains the privacy implications.
It specifies an eligible product universe including compatibility, stock, and account permissions.
It uses a valid cold-start fallback and a fail-closed response for unknown mandatory safety data.
It randomizes at account level if buyers share account histories and defines the analysis population before launch.
It compares complete policies on fulfilled contribution margin over a stated period and names safety and operational guardrails.
It assigns product, data-science, operations, safety, and governance ownership and states a rollback rule.

### Answer 8 rubric

A strong map records a documented purpose and a counsel-reviewed lawful basis where applicable.
It sets retention for raw events, snapshots, factors, and logs rather than only source tables.
It limits access and recognizes that factors can encode personal information.
It specifies how requests and corrections affect derived artifacts, including rebuilding or expiry when required by policy and legal advice.
It links the GDPR and California regulator sources while avoiding an unsupported universal consent claim.

## 14. Interview questions

1. **Why is an unobserved interaction not a negative label?**
   A strong answer mentions exposure, availability, and channel gaps, then distinguishes those facts from the model's low-confidence zero target.
   A weak answer says every unpurchased product was rejected.
2. **What do preference and confidence mean in implicit ALS?**
   A strong answer gives $p=1[r>0]$ and $c=1+\alpha r$, and calls confidence a loss weight rather than a probability.
   A weak answer treats confidence as calibrated purchase likelihood.
3. **What does the ALS user update solve?**
   A strong answer states the regularized normal equation for fixed item factors and recommends solving the system.
   A weak answer claims all user and item factors are jointly solved in one convex step.
4. **How can sparse computation account for all missing pairs?**
   A strong answer describes the shared Gram matrix and corrections for observed pairs.
   A weak answer says unobserved pairs have no effect on this objective.
5. **Why might a score exceed one?**
   A strong answer identifies an unconstrained dot product and distinguishes ranking from probability calibration.
   A weak answer interprets the score as an impossible probability and clips it without analysing the model.
6. **How do you evaluate a replenishment panel?**
   A strong answer permits repeat purchases when appropriate, uses time-valid eligibility, and compares a relevant baseline.
   A weak answer blindly excludes all prior purchases.
7. **What is the effect of an out-of-stock period?**
   A strong answer explains censored opportunity and distinguishes unavailable demand from lack of interest.
   A weak answer interprets no orders as evidence of zero demand.
8. **How should new users and items be handled?**
   A strong answer describes explicit eligible fallbacks and coverage measurement.
   A weak answer assumes ALS learned a useful factor with no interactions.
9. **Why can a better offline ranking fail online?**
   A strong answer covers selection bias, exposure, operational constraints, and incrementality.
   A weak answer assumes recall improvement equals profit uplift.
10. **Who may approve an ALS candidate as a replacement part?**
    A strong answer assigns compatibility and safety to an accountable domain authority and treats ALS as candidate generation.
    A weak answer delegates safety to the model score.
11. **How can factor versions break serving?**
    A strong answer explains that independently trained user and item factors need a compatible shared coordinate system and atomic versioning.
    A weak answer assumes equal vector length guarantees compatibility.
12. **What data would you log for a later experiment?**
    A strong answer includes assignment, candidate and filter versions, rendered impressions, timestamps, outcomes, and relevant eligibility under privacy controls.
    A weak answer logs only clicks and claims non-exposed alternatives can be evaluated causally.

## 15. Key terms

- [Account](#4-definitions-and-vocabulary): possible business-level user entity.
- [Affinity score](#57-score-and-retrieve-safely): uncalibrated dot product used for ranking.
- [Alternating least squares](#56-alternate-user-and-item-updates): alternating fixed-block linear solves.
- [As-of snapshot](#58-compare-models-without-leaking-the-future): data available at a stated decision time.
- [Candidate](#57-score-and-retrieve-safely): item proposed for later policy checks.
- [Cold start](#57-score-and-retrieve-safely): missing interaction history for an entity.
- [Confidence](#55-define-the-implicit-als-objective): weight on a pair's model error.
- [Coverage](#58-compare-models-without-leaking-the-future): share of a declared population reached.
- [Density](#54-quantify-the-matrix-and-its-limits): observed distinct pairs divided by a defined universe.
- [Dot product](#57-score-and-retrieve-safely): user–item factor score.
- [Eligibility](#57-score-and-retrieve-safely): permission and feasibility for a particular decision.
- [Embedding](#3-the-core-mental-model): learned vector representation.
- [Event time](#52-establish-the-event-contract): time the underlying action occurred.
- [Exposure](#84-exposure-consent-and-auditability): opportunity to observe a recommendation.
- [Factor](#3-the-core-mental-model): coordinate in a learned representation.
- [Fallback](#57-score-and-retrieve-safely): defined response when personalized candidates are unavailable.
- [Gram matrix](#56-alternate-user-and-item-updates): shared factor cross-product used in sparse updates.
- [Holdout](#58-compare-models-without-leaking-the-future): data reserved for later evaluation.
- [Implicit feedback](#3-the-core-mental-model): behaviour used as indirect preference evidence.
- [Ingestion time](#52-establish-the-event-contract): time an event became available to the system.
- [Interaction strength](#53-form-interaction-strengths): nonnegative aggregate $r_{ui}$.
- [Leakage](#58-compare-models-without-leaking-the-future): information from outside the legitimate decision-time view.
- [NDCG](#64-evaluate-a-frozen-ranking): position-sensitive normalized ranking metric.
- [Preference](#55-define-the-implicit-als-objective): binary target derived from positive interaction strength.
- [Propensity](#84-exposure-consent-and-auditability): probability of an action under a logging policy.
- [Recall at K](#64-evaluate-a-frozen-ranking): fraction of eligible relevant items retrieved in $K$ positions.
- [Regularization](#55-define-the-implicit-als-objective): penalty controlling factor magnitude.
- [Sparsity](#54-quantify-the-matrix-and-its-limits): complement of density under a declared rectangular universe.
- [Stock-keeping unit](#4-definitions-and-vocabulary): operational item variant.
- [Temporal holdout](#58-compare-models-without-leaking-the-future): later period reserved for evaluation.

## 16. Sources

1. Yifan Hu, Yehuda Koren, and Chris Volinsky, “Collaborative Filtering for Implicit Feedback Datasets,” *2008 Eighth IEEE International Conference on Data Mining*, 2008, pp. 263–272. [DOI: 10.1109/ICDM.2008.22](https://doi.org/10.1109/ICDM.2008.22). Primary reference for confidence-weighted implicit factorization.
2. Yehuda Koren, Robert Bell, and Chris Volinsky, “Matrix Factorization Techniques for Recommender Systems,” *Computer*, 42(8), 2009, pp. 30–37. [DOI: 10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263). Overview of latent-factor approaches.
3. Greg Linden, Brent Smith, and Jeremy York, “Amazon.com Recommendations: Item-to-Item Collaborative Filtering,” *IEEE Internet Computing*, 7(1), 2003, pp. 76–80. [DOI: 10.1109/MIC.2003.1167344](https://doi.org/10.1109/MIC.2003.1167344). Historical company case, not evidence of this lesson's ALS implementation.
4. Carlos A. Gomez-Uribe and Neil Hunt, “The Netflix Recommender System: Algorithms, Business Value, and Innovation,” *ACM Transactions on Management Information Systems*, 6(4), 2015. [DOI: 10.1145/2843948](https://doi.org/10.1145/2843948). Company account of recommendation in product context.
5. Kalervo Järvelin and Jaana Kekäläinen, “Cumulated Gain-Based Evaluation of IR Techniques,” *ACM Transactions on Information Systems*, 20(4), 2002, pp. 422–446. [DOI: 10.1145/582415.582418](https://doi.org/10.1145/582415.582418). Basis for discounted gain measures.
6. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Official identifier overview; consult current applicable specifications for implementation.
7. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Official legal text; applicability requires legal analysis.
8. California Privacy Protection Agency, [California Consumer Privacy Act resources](https://cppa.ca.gov/regulations/). Official regulatory materials; consult current law and regulations.
9. NumPy project, [`numpy.linalg.solve` documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html). Numerical solver used in the teaching code.

Sources are dated or versioned as published; legal and library requirements can change.
Company cases describe the authors' reported context, not guaranteed outcomes elsewhere.

## 17. Links

- [Module 24 topic map](_index.md).
- [Recommendation tasks](01_recommendation-tasks.md).
- [Popularity and content-based baselines](02_popularity-content-baselines.md).
- [Module 24 survey and lab sections](../../../course/24_recommendation_systems_personalization.md).
