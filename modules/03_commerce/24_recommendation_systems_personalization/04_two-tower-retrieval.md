# Two-Tower Retrieval

> Module 24 — Recommendation Systems and Personalization  
> Topic 04 of 11  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 07 and 17; recommendation tasks and baseline methods  
> Version 2026.5

## 1. Why this topic exists

A retailer with millions of sellable offers cannot afford to run its most expensive ranking model on every offer for every page request.
A recommendation request may have a budget of tens of milliseconds for candidate generation.
Inventory and customer context may change between two requests.
The business still expects the page to contain useful, purchasable choices.

Candidate generation narrows the catalogue to a few hundred or thousand plausible items.
A later ranking stage can spend more computation on each candidate.
A final policy stage applies business and safety constraints.
The retrieval model therefore controls what the later stages have a chance to consider.

A two-tower model makes this narrowing efficient.
One tower encodes a customer and request context into a query vector.
The other encodes each item into a vector that can be computed ahead of time.
A similarity search finds item vectors near the query vector.
The item tower is not normally run on the entire catalogue during each request.

The attractive speed comes with commitments.
The two towers must share a compatible vector space.
Item identifiers, feature transforms, embedding model version, and index version must agree.
A fast search over stale or incompatible vectors can return confidently wrong candidates.

A merchandising manager wants relevant choices without suppressing new or niche inventory.
A supply planner wants recommendations to respect what can actually be fulfilled.
A platform owner wants predictable latency and a safe fallback.
A privacy owner wants a clear purpose and retention policy for user features.
A data scientist wants evidence that retrieval adds value beyond simple baselines.
None of these decisions is solved by training loss alone.

Consider a catalogue of 2 million offers, with a ranking model costing 0.2 milliseconds per offer on one serial compute stream.
Scoring every offer would require 400 seconds of serial model time per request, before overhead.
Parallel hardware changes wall time but does not eliminate the total compute requirement.
Retrieving 500 candidates reduces the nominal ranking work to 100 milliseconds of serial model time.
Batching and parallel inference may reduce wall time further.
These illustrative numbers are capacity-planning inputs, not service guarantees.

The expensive error is not merely a poor score.
If the right item is missing from the candidate set, the ranker cannot rescue it.
If an unavailable item occupies a scarce candidate slot, a different sellable item may be excluded.
If a model version mismatch changes vector geometry, similarity scores may have no valid interpretation.

This topic builds the candidate-generation decision from data contract to operational rollback.
It also distinguishes a business target from a forecast, a deployment plan, and a contractual commitment.
A target for recall is an aspiration; measured recall is an estimate.
A plan to refresh an index is not proof that a fresh index was published.
A service commitment exists only if the responsible team has actually agreed to it.

## 2. Learning outcomes

After this topic you can:

1. Draw a two-tower retrieval pipeline and identify which work is offline and online.
2. Specify point-in-time interaction, user-feature, item-feature, eligibility, and embedding contracts.
3. Construct positive and negative training examples without treating unobserved items as proven dislikes.
4. Calculate dot-product scores, exact candidate recall, approximate-search fidelity, and operational cost from stated inputs.
5. Choose an eligibility strategy and explain its recall, latency, and freshness tradeoffs.
6. Design an offline evaluation and an online experiment that separate retrieval quality from ranking and business impact.
7. Diagnose version skew, staleness, biased exposure, cold start, and missing candidates.
8. Assign decision owners, launch gates, monitoring, and a reversible fallback.

Exercises 1 and 2 assess the pipeline and data contracts.
Exercises 3 and 4 assess the arithmetic and eligibility choices.
Exercises 5 and 6 assess evaluation and negative sampling.
Exercises 7 and 8 assess deployment and ownership.
The self-check supplies calculations and rubrics.

## 3. The core mental model

### Two independently computable representations

```text
Historical events + point-in-time features       Item catalogue + point-in-time features
                  |                                              |
            Training pairs --------------------------> shared training objective
                  |                                              |
       query-tower model version V                    item-tower model version V
                  |                                              |
request + allowed user context                           encode eligible item snapshot
                  |                                              |
           query vector q                               item vectors v_i + IDs
                  |                                              |
                  +---------- ANN index version I <--------------+
                                  |
                     retrieve candidate IDs and scores
                                  |
                    current eligibility and policy checks
                                  |
                    expensive ranker and re-ranking
                                  |
                     impressions, outcomes, monitoring
```

ANN means approximate nearest neighbor.
The index is built from item vectors and an explicit item-ID mapping.
A query vector is computed when the request arrives.
The score is typically a dot product or another compatible similarity function.
The index returns an approximation to the highest-scoring item vectors.

The towers are separate computations, not independent models in the business sense.
They are trained jointly so their outputs are comparable.
Their output dimension, scoring convention, and version are part of a shared contract.
The query and item inputs need not have the same fields or architecture.

A dot product is separable because the query representation does not depend on the candidate item and the item representation does not depend on the request.
That separability enables precomputation and indexed search.
It also limits interactions: an item-specific promotion combined with a request-specific coupon cannot be modeled arbitrarily in a fixed item vector.
A cross-feature ranker can consider such interactions after retrieval.

### Retrieval is a gate, not the complete recommender

The retrieval objective asks whether useful eligible candidates enter the pool.
The ranker orders that pool for the surface and objective.
The policy layer enforces requirements such as availability, legal restrictions, and customer-specific blocks.
A candidate set is not an impression, and an impression is not a purchase.

The ordering of eligibility and search is a design choice.
Searching an eligible partition can be efficient when a stable partition exists.
Filtering after search is simpler when eligibility changes frequently, but wastes candidate slots.
Oversampling can compensate for filtering, subject to latency and capacity budgets.
A final authoritative check is still needed when availability or policy may change after index construction.

### Three different comparisons

Model quality compares exact model retrieval with held-out relevant outcomes.
Index fidelity compares approximate results with exact results from the same model and catalogue snapshot.
Business impact compares user and operational outcomes under randomized serving policies when an experiment is feasible.
Conflating these comparisons makes diagnosis difficult.

For example, ANN fidelity can be excellent while the trained model retrieves irrelevant items.
Offline recall can rise while a ranker or checkout constraint prevents incremental orders.
An online lift can be misleading if treatment and control differ in stock availability or instrumentation.
Measure the stages separately before attributing a result.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Retrieval | Selection of a manageable candidate set from a larger catalogue; candidates per request. | Calling the final sorted page retrieval. |
| Candidate | Item or offer ID passed to later processing; IDs. | Assuming every candidate was shown. |
| Query tower | Function mapping allowed user and request inputs to a vector; dimension d. | Assuming it is a text-search query only. |
| Item tower | Function mapping item inputs to a vector; dimension d. | Assuming item metadata never changes. |
| Embedding | Numeric representation in a learned vector space; d floating-point values. | Treating coordinates as stable business meanings. |
| Dot product | Sum of pairwise coordinate products; similarity score. | Treating the raw value as a probability. |
| Cosine similarity | Dot product of unit-normalized vectors; dimensionless. | Assuming it equals an unnormalized dot product. |
| Vector norm | Magnitude of a vector; nonnegative. | Ignoring that norms affect dot-product order. |
| Maximum inner product search | Search for highest dot products, often abbreviated MIPS. | Assuming every nearest-neighbor index supports it directly. |
| ANN | Approximate nearest-neighbor search; returns a potentially incomplete top-k. | Describing it as an unbiased sample of the catalogue. |
| Exact search | Exhaustive scoring over the defined eligible snapshot. | Calling an ANN result ground truth. |
| Index | Search structure plus item-ID mapping and metadata. | Treating vectors alone as the entire deployable artifact. |
| Index fidelity | Overlap between ANN and exact top-k under the same model and universe. | Confusing it with relevance recall. |
| Positive | Observed action chosen as a training or evaluation target. | Equating any click with satisfaction. |
| Negative | Item used as a contrast against a positive during training. | Calling every unobserved item a verified dislike. |
| In-batch negative | Another example's positive reused as a contrast in the same batch. | Assuming no duplicate or shared-positive collisions. |
| Hard negative | A confusing, usually high-scoring contrast. | Assuming high score proves lack of relevance. |
| False negative | A contrast that is actually relevant or would be positive under exposure. | Treating it as harmless label noise at any rate. |
| Exposure | Opportunity to see an item, usually recorded as an impression. | Treating all eligible items as exposed. |
| Exposure bias | Logged outcomes reflect a previous serving policy's choices. | Assuming historical clicks represent unbiased preferences. |
| Sampling probability | Probability under a specified negative or traffic sampling design. | Treating an undocumented heuristic as a known propensity. |
| Feature availability time | Earliest time a value was usable by the serving system; timestamp. | Using event time alone to establish availability. |
| Point-in-time join | Feature join that uses only values available by the example time. | Joining the current catalogue onto past interactions. |
| Leakage | Use of information unavailable at the prediction decision. | Treating delayed data as safe merely because its event timestamp is old. |
| Eligibility | Whether a particular item may be recommended for a request now. | Assuming a global in-stock flag covers every customer. |
| Oversampling | Retrieving more than the desired final candidate count before filtering. | Assuming a fixed multiplier guarantees enough survivors. |
| Candidate recall | Fraction of defined relevant targets present among retrieved candidates. | Confusing it with ANN overlap. |
| Hit rate | Fraction of requests with at least one target retrieved. | Calling it recall when requests have multiple targets. |
| Coverage | Fraction of a stated item or user universe reached. | Reporting a percentage without a denominator. |
| Freshness | Age or lag of a feature, embedding, index, or policy snapshot; time. | Assuming all artifacts refresh together. |
| Cold start | Insufficient user or item history for a collaborative signal. | Assuming metadata always resolves new-item discovery. |
| Feature contract | Schema, semantics, timing, ownership, and allowed use of a feature. | Treating a column name as a complete contract. |
| Embedding contract | Compatibility rules for tower outputs, index, metric, and mapping. | Matching only vector dimensions. |
| Canary | Limited production exposure used to detect deployment problems. | Treating a canary alone as a causal experiment. |
| Rollback | Restore a compatible prior serving bundle and routing policy. | Reverting only the query tower. |
| Fallback | Safe alternative when primary retrieval fails or is unsuitable. | Returning an empty or unrestricted list by default. |

## 5. The mechanics, step by step

### Step 1 — Define the business decision

**Input:** a recommendation surface, customer journey, and business objective.
**Output:** a written retrieval charter and decision owner.

Specify whether the request is for substitutes, complements, replenishment, or discovery.
Each task defines relevance differently.
A complementary item may be useful even when it is not similar to the current product.
A substitution must satisfy compatibility, availability, and other constraints.
A browse-page candidate need not meet the same rules as an order-replacement candidate.

Write the request grain: for example, one authenticated customer, one storefront, one timestamp, and one page surface.
Specify the item grain: product, stock-keeping unit (SKU), or offer.
A product-level embedding cannot by itself prove that a particular seller's offer is deliverable.
Resolve the ID mapping before calculating metrics or constructing an index.

Name the business target separately from forecasts and commitments.
A target might be a 2 percentage-point increase in eligible target recall at 500 candidates.
A forecast might estimate incremental margin from a pilot under explicit assumptions.
A plan might describe the proposed rollout by region.
None is a binding service commitment unless the responsible owners agree and document it.

The product owner approves the surface and outcome definition.
The operations owner approves inventory and fulfillment rules.
The engineering owner approves the latency and recovery budgets.
Privacy and legal owners review the intended data use.
The model owner documents tradeoffs and measurement.

### Step 2 — Define the eligible universe

**Input:** catalogue, fulfillment rules, policy rules, and request context.
**Output:** a reproducible universe definition for training, evaluation, and serving.

Eligibility can depend on region, warehouse, seller, delivery promise, customer restrictions, or current inventory.
Distinguish indexable from currently sellable.
An item may be indexed because it is generally active but fail a request-specific check.
A held-out target must be judged against the eligibility information available for that request.

Write each rule with owner, source, event time, availability time, refresh interval, and failure behavior.
For example, the fulfillment system may be authoritative for available-to-promise quantity.
The index may contain a stale copy for broad pruning, but must not silently replace an authoritative live check.
A missing eligibility signal should follow an agreed fail-closed or safe-fallback policy.
Do not assume the same policy is appropriate for every rule.

Decide whether customer-specific restrictions are applied before search, within the search engine, after search, or at more than one stage.
The decision depends on filter selectivity, supported index operations, and update frequency.
Measure candidate loss under real filter combinations.
A 90% global survivor rate does not imply a 90% survivor rate for a restricted region.

### Step 3 — Build event and feature contracts

**Input:** request logs, impressions, actions, catalogue history, and feature snapshots.
**Output:** point-in-time examples with traceable lineage.

A minimum interaction record contains a stable request ID, pseudonymous user or session key, item ID, surface, event type, and timestamp.
Where available, include impression position, serving policy version, eligibility snapshot, and experiment assignment.
Use event ingestion time and availability time as well as event occurrence time.
A purchase can arrive late, be canceled, or be returned.
Define the observation window and maturity rule before assigning labels.

The user context might include prior categories, recency, region, and session signals.
Only use history available before the decision time.
The item context might include category, brand, price band, and text or image representations when permitted.
Snapshot changing attributes as they existed and were available at the request.
Using an item's later category correction on an earlier example can leak future knowledge.

Define null handling, vocabulary version, units, timezone, privacy classification, retention, and owner for each field.
Document whether an identifier identifies a product, a variant, or an offer.
Document whether price is in minor currency units and whether tax and shipping are included.
Specify how deletions, consent changes, and prohibited features propagate to derived artifacts.
A schema-compatible feature can still be semantically incompatible.

### Step 4 — Choose examples and labels

**Input:** a point-in-time eligible request and mature interaction outcomes.
**Output:** positive query-item pairs and an explicitly defined contrast set.

A purchase, add-to-cart, long dwell, or click may be a positive depending on the task.
A click label is cheap and frequent but can reward misleading thumbnails.
A purchase label is closer to commercial value but is sparse and affected by stock, price, and checkout friction.
Multiple outcomes can be weighted, but the weights are business choices, not natural probabilities.

Choose whether a request with several relevant items contributes several pairs or one set-valued target.
Prevent repeated events from allowing a heavy user or duplicate logging to dominate without deliberate weighting.
Account for the maturity window: a same-day evaluation of a seven-day conversion target is incomplete.
Separate training and evaluation by time rather than randomly splitting near-duplicate events.

Impressions without clicks can be useful contrasts when the impression was actually visible and eligible.
They are still ambiguous: position, price, and attention affect the outcome.
Unshown items are missing outcomes, not proven dislikes.
A mixture of exposed non-clicks and sampled catalogue items can improve training, but document each source.

### Step 5 — Sample negatives deliberately

**Input:** positive pairs, candidate sampling distribution, and exposure logs.
**Output:** labeled or contrastive examples with known sampling behavior.

Uniform catalogue negatives cover the long tail but are often too easy.
Popularity-weighted negatives are more realistic but can overemphasize already popular items.
In-batch negatives are cheap but become false negatives when two users like the same item or the same item appears twice.
Deduplicate or mask known shared positives where appropriate.
Hard negatives from a previous model teach fine distinctions but can include good items the user never saw.

For a batch of query vectors and item vectors, a common objective applies a softmax to similarity scores.
The observed positive is encouraged to outrank sampled alternatives.
A temperature parameter changes score sharpness; tune it on a held-out period rather than declaring it universal.
If negative sampling is nonuniform, corrections may be appropriate for the intended objective; their derivation depends on the sampling design.
Do not invent propensities for deterministic legacy logs.

Audit negative composition by category, popularity, customer segment, availability, and item age.
Keep a stable evaluation set when comparing sampling strategies.
Record the random seed and sampler version for reproducibility.
Monitor whether hard-negative mining disproportionately penalizes relevant alternatives.

### Sampling design clinic

The choice of negatives changes which ranking errors receive training pressure.
Consider a request for industrial gloves with one observed purchase of a medium-size pair.
A uniformly sampled heavy appliance is an easy contrast.
The model can separate it using category alone.
That example contributes little information about competing glove offers.

An exposed but unclicked glove offer is a harder contrast.
Its non-click is still ambiguous because the offer may have appeared below the visible area.
The customer may have preferred the item but encountered an unsuitable delivery date.
A training pipeline should preserve the source and position of the contrast.
The model owner can then compare outcomes with and without that negative source.

A previously retrieved glove offer that the customer never saw is harder still.
The previous model's high score does not establish that the offer was unwanted.
It may even have been a better substitute than the observed purchase.
Use hard-negative mining with a documented source, refresh cadence, and audit.
Avoid repeating the same false negative across every training epoch without inspection.

In-batch negatives require particular care in concentrated catalogues.
Suppose two customers independently buy offer A within one batch.
Treating the second occurrence of A as a negative for the first customer is contradictory.
Mask duplicate item IDs from the denominator for the relevant example.
If another observed positive for that user is known, consider masking that too.
The available history may still be incomplete, so masking cannot eliminate all false negatives.

The sampler also defines an implicit training population.
Sampling uniformly by item gives a rare offer more probability than it receives in typical traffic.
Sampling by logged impressions reflects the old serving policy and may amplify its concentration.
Sampling by category can improve local discrimination but neglect cross-category discovery.
There is no universally correct mixture independent of the task.
Document each source's proportion and compare alternatives on the same future evaluation set.

When applying sampling corrections, specify the exact probability being corrected.
An item-draw probability is not the same as the probability that a customer saw an item.
An impression propensity is not necessarily the probability of purchase given exposure.
A mathematical correction based on the wrong probability can introduce rather than remove bias.
If historical propensities are unavailable, state that limitation and consider a prospective logging design.

### Example objective and score interpretation

For one query q, suppose the positive item is p and the sampled contrasts are n1 and n2.
Let the temperature be a positive scalar T.
The uncorrected sampled softmax probability assigned to p is:

```text
exp(score(q, p) / T)
---------------------------------------------------------
exp(score(q, p) / T) + exp(score(q, n1) / T)
                         + exp(score(q, n2) / T)
```

The per-example loss is the negative natural logarithm of that quantity.
For scores 0.8, 0.3, and 0.1 with T = 1, the exponential terms are approximately 2.226, 1.350, and 1.105.
The positive probability within this sampled set is approximately 2.226/4.681 = 0.476.
The associated loss is approximately 0.743 natural-log units.
This is not the probability of purchase among all catalogue offers.
Changing either the negative set or its sampling design changes the denominator.

Lower temperature makes score differences more pronounced in the training softmax.
It can increase the contribution of hard examples and sensitivity to false negatives.
The final serving metric and whether vectors are normalized remain separate configuration choices.
Record the temperature with training artifacts and compare models on the same evaluation universe.
Do not compare raw losses across different samplers as if the tasks were identical.

### Update and publication design

A catalogue may change faster than its model weights.
Separate the item-feature refresh interval, embedding refresh interval, index publication interval, and live eligibility refresh interval.
For example, item text might update daily while stock changes every minute.
A daily embedding refresh does not provide minute-level stock correctness.
A live availability check can compensate for some changes but cannot retrieve a new item absent from the index.

Incremental index updates reduce rebuild work when supported by the selected index.
They also introduce ordering and consistency questions.
An update must use the item encoder and transformations compatible with the current index.
A deletion must remove or invalidate the intended ID without affecting a different offer.
Out-of-order updates must not restore an older item state over a newer one.
Record an item version or effective timestamp and specify conflict handling.

A full rebuild can simplify snapshot consistency but may require more compute and memory.
During a blue-green publication, both old and new indexes may coexist.
Budget for this temporary overlap as well as steady-state replicas.
Only route query traffic to the new index after its matching query model is available and checks pass.
Keep the prior bundle available for an agreed recovery window.

Index checks should include item count, duplicate-ID count, nonfinite vector count, and snapshot age.
Also check representative queries for unexpected empty results and measure eligibility rejection.
These checks detect certain implementation errors; they do not prove business relevance.
The release owner records the artifact identifiers and the check outcomes.
The incident owner needs the same identifiers in request logs.

### Capacity and latency decomposition

A retrieval budget is an end-to-end request budget, not only ANN compute time.
It includes feature fetching, query encoding, network transfer, index search, filtering, merging, and serialization.
Queueing and retries can add time that is invisible in an isolated model benchmark.
Measure each stage and the end-to-end distribution under realistic concurrency.

Suppose a design budget allocates 5 milliseconds to feature access, 3 milliseconds to query encoding, and 12 milliseconds to search.
It allocates another 5 milliseconds to eligibility and 3 milliseconds to merging and transport.
The arithmetic sum is 28 milliseconds for those nominal stage allocations.
This is a planning sum, not the p95 of the end-to-end request.
Percentiles cannot generally be added to obtain the percentile of a sum.
Measure the actual end-to-end percentile from request traces.

If a service has a 40-millisecond p95 retrieval budget, the nominal 28-millisecond sum leaves 12 milliseconds of arithmetic headroom.
That does not prove the budget will be met under load.
Network tails, cache misses, fan-out, and correlated slowdowns may consume the headroom.
Define timeouts and the fallback path for each external dependency.
Include the cost of checking current eligibility in the benchmark, not merely index lookup.

Increasing k normally gives the ranker more possible choices and may improve target recall.
It can also increase search effort, network payload, filter calls, and ranking cost.
The incremental value of candidates 501 through 1,000 may differ from candidates 1 through 500.
Plot recall, eligible count, and cost across candidate budgets.
Choose a point based on the downstream ranker and business constraints rather than maximum recall alone.

### A practical evaluation ledger

For each evaluated request, retain a reproducible request ID and decision timestamp.
Associate the eligible universe definition, relevant target IDs, and model/index versions.
Record the global exact top-k when feasible and the ANN top-k before filtering.
Record post-filter candidates and reasons for each rejection.
Record the final displayed items separately from candidate IDs.

Aggregate metrics with a declared unit of analysis.
Macro recall averages the per-request recall of requests with at least one target.
Micro recall divides the total retrieved relevant targets by the total relevant targets.
These measures can differ when requests contain different numbers of targets.
Neither specifies what to do with requests with no observed positive target.
Report the count and share of such requests alongside the evaluable population.

For example, one request has one target and retrieves it.
Another request has four targets and retrieves one.
Macro recall is (1/1 + 1/4)/2 = 0.625 across two requests.
Micro recall is (1 + 1)/(1 + 4) = 0.40 across five targets.
The first weights the two requests equally.
The second weights a request in proportion to its number of targets.
Choose the measure that reflects the decision and disclose the alternative when informative.

Use confidence intervals appropriate to the sampling and assignment unit.
Repeated requests from the same user are not necessarily independent observations.
For online experiments, compute uncertainty at the randomized unit or use a defensible clustered method.
Report sample size, time window, missingness, and exclusions with each estimate.
Inspect both absolute differences and uncertainty before declaring a practical improvement.

### Decision gates for catalogue-scale serving

The first gate asks whether the baseline is adequate for the surface.
If a current popularity or content system meets the business need at lower cost, complexity may not be justified.
The second gate asks whether training and evaluation can reconstruct sufficiently reliable historical states.
When they cannot, qualify offline estimates and improve logging before relying on them.

The third gate asks whether the exact model retrieves useful eligible targets.
If exact model recall is weak, tuning ANN parameters cannot fix the underlying representation.
The fourth asks whether the ANN index preserves enough of the exact model's quality within the latency budget.
If not, adjust the index, search effort, or candidate budget and repeat the matched comparison.

The fifth gate asks whether the filtered pool is large and varied enough for ranking.
Investigate shortfall by region and inventory state rather than using only a global average.
The sixth asks whether the complete serving system meets operational and policy requirements.
This includes fallback behavior and a compatible rollback artifact.

The final gate asks whether randomized business evidence supports deployment at the proposed scale.
The primary metric should match the product objective; guardrails protect fulfillment, latency, and customer experience.
When shared inventory creates interference, document the experiment's limitations and consider a design that reduces spillover.
The accountable owners decide whether the observed benefit justifies operating cost and residual risk.

### Step 6 — Design tower inputs and model geometry

**Input:** approved feature contracts and compute constraints.
**Output:** query and item encoders with a declared scoring function.

The query tower can combine user history, session context, region, and surface.
The item tower can combine item ID, category, textual features, and other approved attributes.
Categorical lookup tables, pooling, and multilayer networks are possible implementation choices.
The exact architecture should be justified against simpler baselines and measured cost.

Let q be the d-dimensional query vector and v_i the d-dimensional item vector.
For dot-product retrieval, score item i as s_i = q · v_i.
With unit-normalized vectors, the dot product equals cosine similarity.
Without normalization, vector length affects the score and may encode popularity or confidence.
Changing normalization changes the retrieval problem and requires an aligned index.

Embedding dimension affects memory, network transfer, and search cost.
For N item vectors of d float32 values, raw vector storage is N × d × 4 bytes.
Two million vectors at 128 dimensions require 1,024,000,000 bytes before index and metadata overhead.
That is approximately 0.954 GiB, using 1 GiB = 2^30 bytes.
Replicas, training artifacts, mapping tables, and ANN structures add to the requirement.
Quantization may reduce memory while changing ranking fidelity.

A new item without an ID history may still receive a vector from metadata.
That does not ensure it is retrieved: sparse features, training distribution, and exploration policy still matter.
A new user may use session features or an approved popularity/content fallback.
Measure cold-start performance as a separate slice.

### Step 7 — Train and validate without future information

**Input:** training snapshot, feature transforms, sampling configuration, and objective.
**Output:** a versioned pair of tower weights and reproducible training metadata.

Split by decision time with a training window, validation window, and later test window.
Use a gap if label maturity or feature delivery makes neighboring windows unsafe.
Define whether users and items may appear in multiple periods; this is realistic for established entities but not a cold-start test.
Add explicit new-user and new-item slices rather than relying on an accidental split.

Train only on features available at each example time.
Fit normalization statistics and vocabularies on the training period, not the full dataset.
Evaluate exact retrieval against the same eligible universe used for each historical request, where it can be reconstructed.
If historical eligibility cannot be reconstructed, disclose the proxy and its likely bias.

Track data snapshot IDs, code revision, seeds, sampling parameters, and model artifacts.
Check whether the score distribution, norms, missing rates, and item-ID coverage are stable by slice.
A falling training loss does not establish a useful candidate set.
Compare to popularity and content baselines under the same constraints and time windows.

### Step 8 — Construct and publish an index

**Input:** item tower, item snapshot, preprocessing, item IDs, and search configuration.
**Output:** an immutable, validated index bundle.

Encode every item in the chosen snapshot with the item tower and its matching transforms.
Reject duplicate IDs, missing mappings, nonfinite coordinates, wrong dimensions, and inconsistent metric settings.
Count expected versus encoded items and explain exclusions.
Record feature snapshot time and embedding generation time separately.

Build an exact-search reference on a representative sample or the full catalogue when affordable.
Choose an ANN method supported by the serving stack and the desired metric.
Graph, inverted-file, and quantized methods have different build, memory, update, and fidelity properties.
Do not assume a cosine index can directly serve an unnormalized dot-product model.

Benchmark approximate overlap with exact top-k, plus end-to-end relevant-target recall and tail latency.
Test the actual request distribution and filter selectivity, not just random vectors.
Tune search effort, partitions, and candidate count on validation data.
Freeze the configuration before evaluating the final test period.

An index manifest should bind the query tower version, item tower version, transforms, metric, normalization, dimension, item snapshot, ID mapping, and index configuration.
Include build timestamp, checksum, validation outcome, and rollback pointer.
Publish the index and compatible query model as an atomic bundle or use explicit compatibility routing.
A matching dimension alone is insufficient evidence that two vectors share a space.

### Step 9 — Serve, filter, and rank

**Input:** request context, compatible query encoder, published index, and live eligibility service.
**Output:** eligible candidate IDs and diagnostic metadata.

Validate request and feature availability against the contract.
Encode the query once, then query the compatible index.
Retrieve enough candidates to allow for downstream filtering and deduplication.
Apply request-specific constraints with an explicit timeout and fallback.
Pass candidate IDs, retrieval scores, versions, and provenance to the ranker.

If search is partitioned, route to the appropriate partition and define what happens when it is absent.
If results from multiple retrieval sources are merged, deduplicate IDs and reserve or cap source contributions deliberately.
Scores from different models are not automatically calibrated or comparable.
A ranker may use retrieval-source indicators instead of comparing raw scores across models.

Recheck dynamic constraints close to display or checkout as needed.
Even a live eligibility check cannot guarantee future stock under concurrent demand.
The fulfillment system remains authoritative for a sale or promise.
Define whether a final short list is acceptable or whether a safe backup pool should fill it.

### Step 10 — Measure the stages independently

**Input:** time-held-out outcomes, serving logs, and randomized experiment data when available.
**Output:** model, index, operational, and business evidence with uncertainty.

For a request with R relevant eligible targets and H retrieved targets, recall@k is H/R when R > 0.
Aggregate per request or across all targets only after stating the weighting rule.
Requests with no defined target need separate reporting; do not silently assign them recall zero.
Hit rate@k is the fraction of evaluable requests with H ≥ 1.
When every request has exactly one target, hit rate and per-request recall coincide.

ANN fidelity@k can be defined as |ANN top-k ∩ exact top-k| / k for the same request, universe, model, and deterministic tie rule.
It measures the index approximation, not whether the items are useful.
Compare relevant-target recall before and after eligibility filtering to locate candidate loss.
Also measure coverage, item age, category distribution, and availability by meaningful segments.

Log total and stage latency at median and high percentiles, including timeouts and error rates.
Averages alone hide queueing and slow partitions.
Report index age, user-feature age, embedding coverage, version mismatch count, and fallback rate.
Separate platform errors from legitimate low-inventory situations.

For online evaluation, randomize an appropriate unit such as user or session, with consistent assignment.
Keep ranking and policy constant when testing retrieval alone where feasible.
Use intention-to-treat analysis and prespecified primary, guardrail, and diagnostic metrics.
Consider interference through shared inventory, marketplace supply, and repeated users.
An experiment can estimate the effect of the whole serving policy, not a universal preference truth.

### Step 11 — Decide launch, monitor, and recover

**Input:** offline and online evidence, service readiness, and owner approvals.
**Output:** a launch decision and an operational runbook.

The product owner decides whether business benefits justify the change.
Engineering owns latency, capacity, alerting, and rollback execution.
The data science owner owns evaluation, drift interpretation, and model lineage.
Operations owns stock and fulfillment correctness.
Privacy and legal owners review applicable data use and customer rights.

Set numeric thresholds appropriate to the surface before launch.
For example, a team may require an agreed minimum relevant-target recall and maximum p95 latency.
Those numbers are local acceptance criteria, not a claim about any actual service.
Require a safe fallback for missing query features, stale indexes, index timeouts, and invalid version pairs.
The fallback must also enforce mandatory eligibility rules.

Start with a canary and compare health against a concurrent control.
Canary health is not sufficient evidence of business lift unless allocation and analysis support causal inference.
Retain the prior compatible bundle until rollback is tested by the owning team.
Alert on index age, candidate shortfall, filter rejection, model/index version skew, and business guardrails.
Define who can stop a rollout and who must be notified.

## 6. Worked example

### Decision and assumptions

A fictional regional distributor recommends replacement maintenance parts on a product page.
The objective is to supply the ranker with 3 eligible candidate offers per request in this small example.
A real catalogue would use a much larger candidate pool and higher-dimensional vectors.
There is no claim that the example reflects an actual company.

The request is for one customer in one delivery region at 10:00 UTC.
The product owner wants a 3-offer candidate pool.
The fulfillment owner defines eligible offers as active and deliverable to this region at the decision time.
The ranking owner will apply compatibility and business ordering after retrieval.
The operations team retains authority over the actual fulfillment promise.

Assume the compatible query and item towers emit two-dimensional, unnormalized vectors.
The scoring metric is the dot product.
The item snapshot has five offers, A through E.
All five are indexed, but offer B has become unavailable for this region.
The authoritative request-time eligibility check identifies B as unavailable.

| Offer | Item vector | Eligible at 10:00 UTC? | Held-out relevant target? |
|---|---|---|---|
| A | (0.8, 0.4) | Yes | Yes |
| B | (1.0, 0.6) | No | No |
| C | (0.2, 0.9) | Yes | Yes |
| D | (0.5, 0.2) | Yes | No |
| E | (0.1, 0.3) | Yes | No |

The query vector is q = (0.7, 0.5).
The held-out target set {A, C} is a hypothetical evaluation label, not a claim that both offers were purchased simultaneously.
The targets were defined independently of the score and are eligible in this example.

### Calculate exact scores

The score for A is 0.7 × 0.8 + 0.5 × 0.4 = 0.76.
The score for B is 0.7 × 1.0 + 0.5 × 0.6 = 1.00.
The score for C is 0.7 × 0.2 + 0.5 × 0.9 = 0.59.
The score for D is 0.7 × 0.5 + 0.5 × 0.2 = 0.45.
The score for E is 0.7 × 0.1 + 0.5 × 0.3 = 0.22.
These scores are dimensionless similarities, not purchase probabilities.

| Exact global position | Offer | Score | Eligible? |
|---:|---|---:|---|
| 1 | B | 1.00 | No |
| 2 | A | 0.76 | Yes |
| 3 | C | 0.59 | Yes |
| 4 | D | 0.45 | Yes |
| 5 | E | 0.22 | Yes |

The exact eligible top 3 are A, C, and D.
Both relevant targets are present, so eligible recall@3 is 2/2 = 1.00 for this one request.
The hit indicator is 1 because at least one target is present.
The value 1.00 is not a population estimate from a single request.

### Compare filter strategies

Suppose the system retrieves the global top 3 and filters B afterward.
The surviving list is A and C, only 2 offers.
It contains both relevant targets, so recall relative to {A, C} remains 1.00.
However, it misses the 3-candidate pool target and gives the ranker less choice.
This distinction shows why recall alone does not measure candidate sufficiency.

If the system retrieves the global top 4 and filters B, the survivors are A, C, and D.
The result now meets the 3-candidate target.
Here the oversampling factor is 4/3, approximately 1.33.
One request does not establish that factor as a production guarantee.
Other regions may reject more than one of the top four offers.

If a search engine can correctly enforce region eligibility during retrieval, it can search the eligible universe directly.
That approach needs a sufficiently fresh filter and adequate index support.
A final authoritative check remains necessary if the state can change between retrieval and display.
Compare the end-to-end latency and candidate sufficiency of both approaches on real traffic.

### Isolate approximation error

Assume an ANN index returns B, A, and D for the global top 3.
The exact global top 3 are B, A, and C.
Their intersection contains B and A, so ANN fidelity@3 is 2/3, approximately 0.667.
The approximate list misses target C even before the eligibility check.
After B is filtered, the survivors A and D contain 1 of 2 targets.
Thus relevant-target recall is 1/2 = 0.50 for this request.

The exact top 3 followed by filtering had target recall 1.00 but only 2 eligible candidates.
The approximate top 3 followed by filtering had target recall 0.50 and only 2 eligible candidates.
The gap in this constructed example is attributable to the approximate candidate set, holding the model and global universe fixed.
In a real system, also check for timing and eligibility snapshot differences before attributing the gap.

### Translate results into an operating decision

Assume one benchmark measures 100,000 requests over a representative time window.
Suppose 4,000 requests have fewer than 3 eligible candidates after the current filter.
The shortfall rate is 4,000/100,000 = 4% of requests.
Suppose an alternative retrieves one additional item and reduces shortfalls to 1,500 requests.
The absolute reduction is 2.5 percentage points; the relative reduction is 62.5%.
This is an illustrative calculation, not a measured outcome.

The decision also needs p95 latency, memory, freshness, and eligible recall under the alternative.
The product owner may accept higher compute if the ranker gains enough useful choices.
The engineering owner may reject it if the agreed service budget is exceeded.
The operations owner may require an authoritative stock check regardless of retrieval strategy.
The result is a documented tradeoff, not an automatic instruction to increase k.

### Forecast, target, plan, and commitment

A target could be fewer than 2% of requests with a candidate shortfall.
The 1.5% result above is a hypothetical benchmark observation under stated traffic and inventory assumptions.
A forecast could project the shortfall rate next month using anticipated demand and catalogue changes.
A plan could deploy the new search configuration to one region after a controlled trial.
A commitment requires an explicit agreement by the service owner and its stakeholders.
None of these labels implies that an experiment has been run.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the exact dot products and candidate/ANN calculations above.
It does not train a model or implement ANN; the approximate results are declared inputs for the exercise.
Run it as a standalone script with `python retrieval_example.py`.
Tie handling is deterministic by offer ID.
The eligibility dictionary is assumed to represent the request-time authoritative snapshot.

```python
from math import isclose

# Illustrative two-dimensional embeddings, not trained model outputs.
query = (0.7, 0.5)
items = {
    "A": (0.8, 0.4),
    "B": (1.0, 0.6),
    "C": (0.2, 0.9),
    "D": (0.5, 0.2),
    "E": (0.1, 0.3),
}
eligible = {"A": True, "B": False, "C": True, "D": True, "E": True}
targets = {"A", "C"}


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return sum(x * y for x, y in zip(left, right))


def exact_top_k(query_vector, catalogue, k, allowed=None):
    if k < 0:
        raise ValueError("k must be nonnegative")
    scores = []
    for item_id, vector in catalogue.items():
        if allowed is None or item_id in allowed:
            scores.append((item_id, dot(query_vector, vector)))
    return sorted(scores, key=lambda row: (-row[1], row[0]))[:k]


def recall_at_k(retrieved, relevant):
    if not relevant:
        return None  # A request without targets is not recall-evaluable.
    return len(set(retrieved) & relevant) / len(relevant)


def ann_fidelity(approx_ids, exact_ids):
    if len(approx_ids) != len(exact_ids) or not exact_ids:
        raise ValueError("compare equally sized nonempty top-k lists")
    if len(set(approx_ids)) != len(approx_ids):
        raise ValueError("duplicate approximate IDs")
    return len(set(approx_ids) & set(exact_ids)) / len(exact_ids)


allowed_ids = {item for item, allowed in eligible.items() if allowed}
global_top3 = exact_top_k(query, items, 3)
global_top4 = exact_top_k(query, items, 4)
eligible_top3 = exact_top_k(query, items, 3, allowed_ids)
post_filter3 = [item for item, _ in global_top3 if eligible[item]]
post_filter4 = [item for item, _ in global_top4 if eligible[item]]
ann_ids = ["B", "A", "D"]  # Hypothetical approximate index output.
ann_survivors = [item for item in ann_ids if eligible[item]]
exact_ids = [item for item, _ in global_top3]

print("scores:", ", ".join(f"{item}={score:.2f}" for item, score in exact_top_k(query, items, 5)))
print("eligible exact top 3:", [item for item, _ in eligible_top3])
print("global top 3 after filter:", post_filter3)
print("global top 4 after filter:", post_filter4)
print(f"ANN fidelity@3: {ann_fidelity(ann_ids, exact_ids):.3f}")
print(f"ANN post-filter target recall: {recall_at_k(ann_survivors, targets):.2f}")

assert isclose(recall_at_k(post_filter4, targets), 1.0)
assert isclose(ann_fidelity(ann_ids, exact_ids), 2 / 3)
assert isclose(recall_at_k(ann_survivors, targets), 0.5)
```

Expected output:

```text
scores: B=1.00, A=0.76, C=0.59, D=0.45, E=0.22
eligible exact top 3: ['A', 'C', 'D']
global top 3 after filter: ['A', 'C']
global top 4 after filter: ['A', 'C', 'D']
ANN fidelity@3: 0.667
ANN post-filter target recall: 0.50
```

The assertions check this small example only.
They are not a production test suite or a statistical evaluation.
The helper treats a request with no relevant targets as not recall-evaluable.
A production evaluator also needs temporal joins, filtering provenance, deduplication, population aggregation, and uncertainty estimates.
It must use actual search output and a reference search over the same snapshot.

## 8. Data, standards, and regulatory context

### Contracts across systems

| Artifact | Required fields or metadata | Owner and control |
|---|---|---|
| Request | Request ID, timestamp, surface, locale, pseudonymous context key. | Serving team; validate and limit collection. |
| Interaction | Request ID, item ID, event type, event and ingestion times. | Analytics team; deduplicate and reconcile. |
| Impression | Visible opportunity, position, policy version, experiment arm. | Product instrumentation owner; define visibility. |
| User features | Value, availability time, transform version, permissible purpose. | Feature owner and privacy owner; point-in-time access. |
| Item features | Item grain, attributes, effective and availability times, owner. | Catalogue team; maintain history. |
| Eligibility | Rule, region, state, freshness, authority, failure behavior. | Fulfillment and policy owners; current checks. |
| Embedding | ID, vector, dimension, metric, normalization, model and transform versions. | Model owner; validate compatibility. |
| Index | Snapshot, mapping, build settings, checksum, validation and rollback reference. | Search platform owner; atomic publication. |
| Outcome | Action, attribution window, cancellation or return state, maturity time. | Analytics and finance owners; consistent definitions. |

An event timestamp says when the event occurred.
An availability timestamp says when the prediction system could have used it.
For a historical decision, both must be at or before the decision time if the feature is intended to represent known information.
A late-arriving event with an old event timestamp can still leak information.
Backfills should preserve the original availability history or mark an evaluation as unable to reconstruct it.

A catalogue identifier is meaningful only at a stated grain.
A global trade item number (GTIN), where used, identifies a trade item under the applicable GS1 identification system.
An internal offer ID can additionally identify seller, price, and fulfillment arrangement.
GTIN presence does not determine offer eligibility or guarantee live inventory accuracy.
Consult the relevant GS1 General Specifications and the organization's implementation rules rather than assuming one ID fits every workflow.

### Privacy and automated decision context

The EU General Data Protection Regulation (GDPR) applies where its territorial and material scope conditions are met.
Its Article 5 sets principles including purpose limitation, data minimization, accuracy, storage limitation, and integrity and confidentiality.
Article 6 addresses lawful bases for processing.
These provisions do not prescribe a two-tower architecture or make a particular feature automatically lawful.
A responsible privacy team must assess the actual purpose, data, legal basis, jurisdiction, and deployment.

GDPR Article 22 concerns certain solely automated decisions with legal or similarly significant effects, subject to its conditions and exceptions.
Ordinary product recommendations should not automatically be assumed to fall within that category; context matters.
Transparency and other obligations can apply independently of Article 22.
The European Data Protection Board guidance on automated decision-making and profiling provides interpretive context.
As of November 2026, consult current law and competent counsel rather than treating this teaching example as advice.

A pseudonymous embedding may still be personal data if it relates to an identifiable person under applicable law.
A vector is not a privacy guarantee merely because coordinates are hard to interpret.
Restrict access, document retention, and design deletion propagation for derived data and indexes.
The technically appropriate deletion procedure depends on system design and legal requirements.
Track the relevant user-feature and model retraining implications rather than promising instant removal from every aggregate model without evidence.

### Operational controls

Separate business-purpose approval from technical access approval.
Limit logged features and avoid storing raw sensitive information merely because a model can consume it.
Encrypt data in transit and at rest according to organizational policy.
Restrict who may export embeddings and join them back to identity systems.
Record model, index, and feature lineage for investigation and reproducibility.

Monitor data quality by source and region.
A schema-valid null can still invalidate a segment if its missing rate rises suddenly.
Distinguish event delay from true inactivity and catalogue removal from stale ingestion.
Use thresholds and escalation owners rather than silently replacing missing features with misleading defaults.

Standards and law constrain some choices, not every product decision.
A privacy review does not validate retrieval relevance.
A GS1 identifier does not validate customer consent.
A security control does not prove that a recommendation is in stock.
Each decision needs its own evidence and accountable owner.

## 9. Industry practice and case evidence

### YouTube retrieval and ranking

Covington, Adams, and Sargin describe a two-stage recommendation architecture for YouTube in their 2016 RecSys paper.
Their candidate-generation model retrieves a smaller set from a large video corpus, followed by a separate ranking model.
The paper discusses learned embeddings, implicit feedback, and sampled training for a large output space.
It is a published description of that system at the time, not evidence of the current implementation.

The transferable lesson is the division of labor between broad retrieval and more expensive ranking.
The paper's video watch signals and ecosystem are not identical to a distributor's inventory-constrained offer recommendation.
A commerce team must independently define eligibility, availability, and value.
Do not transfer a reported benchmark or business outcome to a different surface without measurement.

### Web-scale retrieval research

Google researchers describe large-scale deep retrieval and efficient index-based serving in the 2020 paper on mixed negative sampling.
Their work examines sampling effects in two-tower retrieval settings.
The paper supports the claim that the training sampler is a substantive modeling choice.
It does not imply that a particular sampler is optimal for every catalogue.

The 2020 ScaNN paper describes efficient vector similarity search using anisotropic vector quantization.
It provides evidence for a specific research method and evaluation setting.
An organization considering any ANN library must benchmark its own metric, filter workload, hardware, update pattern, and latency distribution.
A research paper's benchmark is not a service-level commitment.

### TensorFlow Recommenders example

TensorFlow Recommenders publishes a retrieval tutorial using a query model, candidate model, and retrieval task.
The example demonstrates a practical software pattern for training and evaluating representations.
It is documentation and a teaching example, not an independent claim that a deployed system will meet a business target.
The production contract still needs eligibility, privacy, reproducibility, and serving controls.

### Inventory-constrained inference

A catalogue search index is a useful engineering mechanism for candidate generation.
[INFERENCE] In a distributor with highly volatile regional stock, a frequently updated eligibility layer is likely to matter more than a global daily stock flag.
This is a hypothesis about that environment, not an observed result from the published cases above.
Test it by comparing candidate sufficiency and fulfillment outcomes across regions and stock-volatility bands.

[INFERENCE] A model optimized only for historical clicks may under-retrieve newly listed or less-exposed items.
The mechanism is limited exposure and sparse positive feedback, not a guaranteed outcome for every model.
Measure item-age and exposure slices and compare against a controlled exploration or content-based source.
Avoid claiming that more exposure automatically produces incremental demand.

### Evidence hierarchy in a launch review

A peer-reviewed architecture paper can justify considering a technique.
Vendor documentation can establish supported API behavior for a specified version.
An internal replay can estimate offline behavior under its historical data assumptions.
A randomized online experiment can estimate the effect of the tested policy in its study population.
None alone guarantees future outcomes under different inventory, users, or policy.
Document which claim each piece of evidence supports.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Query/item model skew | Sudden score or overlap shift after deployment. | Vectors come from incompatible spaces. | Arbitrary candidates despite valid dimensions. | Bind versions in one bundle and reject mismatch. |
| Stale item index | Growing index age and missing recent items. | Refresh or publication failed. | New or changed offers are absent. | Alert on age; use freshness gate and fallback. |
| Stale eligibility | High post-retrieval rejection or stock complaints. | Index state lags the authoritative source. | Candidate waste and unfulfillable suggestions. | Live checks and measured refresh policy. |
| Wrong item grain | Duplicates or unmappable IDs. | Product IDs are confused with offer IDs. | Incorrect price, seller, or stock. | Explicit grain and validated mapping. |
| Future leakage | Improbably strong offline metrics. | Features joined using later availability. | Overstated model quality. | Point-in-time joins and availability audit. |
| False negatives | Popular shared positives repeatedly sampled as negatives. | In-batch collisions or unobserved interest. | Suppressed relevant items. | Mask known positives and audit samplers. |
| Exposure bias | Model mostly returns historically promoted items. | Labels reflect prior display choices. | Narrow discovery and biased evaluation. | Log exposure; explore responsibly; slice results. |
| Metric confusion | ANN overlap improves but business recall falls. | Approximation fidelity substitutes for relevance. | Wrong launch decision. | Report exact relevance and ANN overlap separately. |
| Filter starvation | Many requests return fewer than required candidates. | Selective filters remove top results. | Weak ranker choice or empty surface. | Tune oversampling or filtered search by segment. |
| Cold item suppression | New-item coverage remains low after indexing. | Metadata signal and exposure are insufficient. | New catalogue never receives traffic. | Evaluate cold slices and use governed source mixing. |
| Norm drift | Vector norm distribution changes sharply. | Transform or training change affects dot products. | Ranking dominated by magnitude. | Track norms and enforce metric contract. |
| Unsupported metric | Cosine index used with raw dot-product model. | Index geometry differs from training objective. | Exact and ANN rankings disagree. | Select compatible search or retrain consistently. |
| Duplicate events | A few users dominate positives. | Retries or event replay inflate counts. | Biased model and metrics. | Idempotent event keys and deduplication. |
| Late labels | Recent cohort appears unusually negative. | Purchases or returns have not matured. | Incorrect model selection. | Maturity windows and cohort reporting. |
| Fallback bypasses policy | Errors coincide with unsafe or unavailable offers. | Backup path skips current checks. | Customer harm and operational breach. | Apply mandatory rules to all paths. |
| Shared inventory interference | Treatment changes control stockout rate. | Experimental arms compete for the same supply. | Biased treatment effect. | Design clusters or model interference; monitor stock. |
| Missing latency tail | Mean latency looks healthy while timeouts rise. | Queueing or slow shards affect a minority. | Poor customer experience. | Monitor percentiles, timeouts, and saturation. |
| Untraceable artifact | Cannot reproduce a prior result. | Missing snapshot, transform, or ID-map version. | Slow incident response and unreliable comparisons. | Immutable manifests and retained bundles. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Query and item vectors share geometry. | Similarity must be meaningful. | Independent deployment or changed transforms. | Enforce bundle versions and compare known reference outputs. |
| Item features are sufficiently fresh. | Precomputed vectors should describe current offers. | Price, category, or status changes after encoding. | Measure feature age and recompute affected vectors. |
| Eligibility can be evaluated at request time. | Candidate pool must be usable. | Service outage, missing region, stale state. | Exercise fallback and monitor rejected or unknown states. |
| Historical labels capture some relevance. | Offline recall needs a target. | Unexposed items have no observed outcomes. | Report exposure-conditioned limitations and run online tests. |
| Evaluation time split reflects deployment. | Future performance is estimated from past behavior. | Seasonal shift or major policy change. | Repeat across windows and monitor post-launch drift. |
| Sampling procedure is documented. | Training objective and bias depend on contrasts. | Hidden sampler changes or unknown propensities. | Version sampler and compare controlled alternatives. |
| Exact reference uses the same universe. | ANN fidelity needs a fair comparator. | Different filter snapshot or ID mapping. | Pin snapshots, metric, and tie behavior. |
| Item-level target is appropriate. | Recall denominator must match the decision. | Product labels evaluated against offer IDs. | Map targets at a documented grain and inspect collisions. |
| Latency measurements represent load. | Service decisions depend on tail performance. | Benchmark omits concurrent traffic or filters. | Load-test representative mixtures under owner supervision. |
| Experiment assignment remains stable. | Treatment comparison needs consistent exposure. | User switches arms across sessions or devices. | Audit assignment and choose an appropriate unit. |
| Inventory interference is manageable. | Standard experiments assume limited cross-arm spillover. | Treatment depletes shared stock. | Measure spillover and consider cluster designs. |
| Retained features are permitted. | Training and serving must honor the approved purpose. | Consent or legal requirements change. | Review lineage, retention, and deletion processes. |

Two-tower retrieval is not always the best choice.
A small catalogue may be served adequately by exact scoring or a transparent baseline.
A highly dynamic catalogue may make embedding and index maintenance expensive.
A task requiring rich query-item interactions may need multiple retrieval sources or a stronger subsequent ranker.
The decision should be based on measured incremental value and operational cost.

The dot-product score is not automatically calibrated.
Even a well-ranked candidate is not guaranteed to be relevant, in stock, profitable, or safe.
Offline held-out recall estimates performance against observed targets, not against all items a customer might have wanted.
A business outcome estimate requires a suitable causal design and a stated population.

## 12. Exercises

### Exercise 1 — Draw the flow

**Time:** 15 minutes.
Draw the offline training and indexing paths and the online request path.
Label the query tower, item tower, index, eligibility check, ranker, and outcome log.
**Deliverable:** a one-page diagram with two sentences describing the embedding contract.
**Acceptance:** the diagram shows precomputation, request-time work, and version compatibility.

### Exercise 2 — Write a point-in-time contract

**Time:** 20 minutes.
A purchase occurs at 09:55, arrives in the feature store at 10:07, and the recommendation request is at 10:00.
An item price effective at 09:00 arrives at 09:10.
State which values can be used at 10:00 and why.
**Deliverable:** a table with event time, availability time, decision, and owner.
**Acceptance:** availability time is distinguished from event time; missing purchase is not treated as a negative.

### Exercise 3 — Score and retrieve

**Time:** 20 minutes.
Let q = (0.6, 0.8), X = (0.5, 0.2), Y = (0.1, 0.9), and Z = (0.7, 0.3).
Use unnormalized dot products.
Y is ineligible, and {X, Z} are the relevant eligible targets.
**Deliverable:** scores, global order, eligible top 2, and eligible recall@2.
**Acceptance:** arithmetic, filtering, and denominator are explicit.

### Exercise 4 — Quantify approximation

**Time:** 15 minutes.
The exact top 4 are A, B, C, D; an ANN index returns A, C, E, F.
Relevant targets are B and C, both eligible.
**Deliverable:** ANN fidelity@4 and relevant-target recall@4.
**Acceptance:** both denominators and the distinction between metrics are correct.

### Exercise 5 — Design negative sampling

**Time:** 25 minutes.
You have impressions with positions, clicks, purchases, and a catalogue snapshot.
Some items are never shown and some users share the same purchased items in a batch.
**Deliverable:** a positive definition, two negative sources, collision handling, and one bias audit.
**Acceptance:** unobserved items are not labeled as proven dislikes; point-in-time eligibility is respected.

### Exercise 6 — Design evaluation

**Time:** 30 minutes.
Your proposed model improves validation exact recall but reduces ANN fidelity and raises p95 latency.
The catalogue contains volatile stock and new products.
**Deliverable:** an evaluation table with model, index, candidate sufficiency, and business measures.
**Acceptance:** comparisons use matched snapshots, include cold-item and regional slices, and distinguish offline from online evidence.

### Exercise 7 — Plan serving and rollback

**Time:** 25 minutes.
A deployment changes the item tower and a region's eligibility service may time out.
**Deliverable:** a bundle manifest, routing decision, fallback policy, and rollback trigger.
**Acceptance:** incompatible vectors cannot be mixed and the fallback does not bypass mandatory eligibility.

### Exercise 8 — Make the business decision

**Time:** 25 minutes.
A fictional benchmark moves candidate shortfall from 4% to 1.5%, while p95 latency rises by 8 milliseconds.
The agreed latency budget and experiment results are not yet provided.
**Deliverable:** a decision memo listing owners, required evidence, target, forecast, plan, and commitment separately.
**Acceptance:** the memo does not declare a launch based on shortfall alone and identifies an experiment and guardrails.

## 13. Self-check answers

### Exercise 1

A strong diagram shows events and historical features feeding training.
The trained item tower encodes a catalogue snapshot into an index before serving.
The compatible query tower encodes the request online.
The index returns candidates that pass through eligibility and ranking before display.
The outcome log records exposure and later actions.
The contract binds both tower versions, transforms, dimension, similarity metric, normalization, and item-ID mapping.

### Exercise 2

The purchase occurred before the request but was not available until 10:07.
It cannot be used by a 10:00 serving decision or a faithful replay of that decision.
The price was effective and available before 10:00, so it may be used if its contract permits the purpose and the snapshot is correct.
The absence of the purchase in the 10:00 features is not evidence of dislike.
The event owner and feature-store owner should document the delay and availability semantics.

### Exercise 3

X scores 0.6 × 0.5 + 0.8 × 0.2 = 0.46.
Y scores 0.6 × 0.1 + 0.8 × 0.9 = 0.78.
Z scores 0.6 × 0.7 + 0.8 × 0.3 = 0.66.
The global order is Y, Z, X.
Removing ineligible Y leaves Z, X as the eligible top 2.
Both relevant targets are retrieved, so recall@2 = 2/2 = 1.00.
If only the global top 2 were fetched before filtering, only Z would survive and target recall would be 1/2.

### Exercise 4

The intersection of exact and ANN lists is {A, C}, containing 2 items.
ANN fidelity@4 = 2/4 = 0.50.
The ANN list contains target C but not B.
Relevant-target recall@4 = 1/2 = 0.50.
The equal numbers in this exercise are incidental; the measures answer different questions.

### Exercise 5 rubric

A strong response selects a mature positive outcome and states the decision grain.
It uses visible non-click impressions with caveats and sampled eligible items as contrasts rather than verified dislikes.
It masks duplicate or known shared positives and records sampling probabilities where they are genuinely known.
It audits composition and performance by exposure, position, popularity, and item age.
A weak response calls the entire unobserved catalogue negative and assumes every impression was seen.

### Exercise 6 rubric

A strong response evaluates exact model recall, ANN overlap, post-filter recall, candidate shortfall, and latency separately.
It pins model, metric, item universe, and time snapshot for each comparison.
It includes regional stock volatility, item age, and user slices.
It proposes a randomized experiment with business and safety guardrails before making a causal claim.
It does not claim that a validation lift offsets a service-budget breach without owner agreement.

### Exercise 7 rubric

A strong manifest names both tower versions, transforms, metric, dimension, normalization, index snapshot, mapping, and checksum.
Routing admits only an approved query/index pair.
An eligibility timeout invokes an explicitly approved safe path, not an unfiltered candidate list.
A rollback restores the prior compatible bundle and records the trigger and responsible team.

### Exercise 8 rubric

The shortfall reduction is 4% − 1.5% = 2.5 percentage points.
The relative reduction is 2.5/4 = 62.5%.
The latency increase is 8 milliseconds at p95; whether it is acceptable depends on the agreed budget and full service distribution.
A strong memo requests relevance, availability, cost, uncertainty, and online business evidence.
It identifies product, engineering, operations, data science, and privacy owners.
It separates a proposed target, a conditional forecast, an implementation plan, and any formally agreed commitment.

## 14. Interview questions

1. **Why use two towers instead of scoring every item with a cross-feature model?**
   Strong: explains precomputed item vectors, indexed retrieval, and lost interaction expressiveness.
   Weak: says two towers are always more accurate.
2. **What must match between a query model and an index?**
   Strong: names model versions, transforms, metric, normalization, dimension, and ID mapping.
   Weak: checks only dimension.
3. **What does a dot product represent?**
   Strong: computes similarity and explains vector norm effects; it is not automatically a probability.
   Weak: interprets every score as purchase probability.
4. **How do you construct negatives?**
   Strong: distinguishes exposure, unobserved outcomes, sampling distribution, and false negatives.
   Weak: declares all non-clicked catalogue items dislikes.
5. **How do you evaluate an ANN index?**
   Strong: compares with exact top-k on the same snapshot and also measures relevance and latency.
   Weak: reports only index throughput.
6. **Why can filtering after retrieval fail?**
   Strong: explains selective rejection, candidate shortfall, and segment-dependent oversampling.
   Weak: assumes any fixed oversampling factor guarantees enough candidates.
7. **How do you avoid temporal leakage?**
   Strong: uses feature availability time, label maturity, and time-based splits.
   Weak: relies only on event timestamps.
8. **How do you handle a newly listed item?**
   Strong: discusses metadata encoding, indexing freshness, exposure, and cold-item evaluation.
   Weak: assumes a vector guarantees discovery.
9. **How would you prove business impact?**
   Strong: proposes a randomized comparison, defined outcome, guardrails, and interference analysis.
   Weak: treats offline recall as incremental revenue.
10. **What do you do when the eligibility service fails?**
    Strong: follows an approved safe fallback and preserves mandatory restrictions.
    Weak: bypasses filters to keep the page populated.
11. **How do you roll back an index deployment?**
    Strong: restores a prior compatible bundle and routing, with clear triggers and ownership.
    Weak: reverts only one tower.
12. **When should you not use a two-tower model?**
    Strong: weighs catalogue size, baseline value, feature dynamics, interaction needs, and operating cost.
    Weak: assumes sophistication is always beneficial.

## 15. Key terms

- [ANN](#4-definitions-and-vocabulary): approximate vector search with an explicit fidelity tradeoff.
- [Candidate](#4-definitions-and-vocabulary): an item or offer passed to a later decision stage.
- [Candidate recall](#10--measure-the-stages-independently): relevant targets retrieved divided by defined targets.
- [Cold start](#6--design-tower-inputs-and-model-geometry): sparse history for a new user or item.
- [Cosine similarity](#4-definitions-and-vocabulary): dot product after unit normalization.
- [Dot product](#6--design-tower-inputs-and-model-geometry): sum of coordinate-wise products.
- [Eligibility](#2--define-the-eligible-universe): request-specific permission and ability to recommend.
- [Embedding](#4-definitions-and-vocabulary): vector representation in a learned space.
- [Embedding contract](#8--construct-and-publish-an-index): compatibility of towers, transforms, metric, and index.
- [Event time](#contracts-across-systems): when an event occurred.
- [Exposure bias](#5--sample-negatives-deliberately): bias from the previous policy's display choices.
- [Fallback](#11--decide-launch-monitor-and-recover): approved alternative when primary retrieval fails.
- [False negative](#5--sample-negatives-deliberately): a sampled contrast that is actually relevant.
- [Feature availability time](#contracts-across-systems): when a value became usable.
- [Fidelity](#10--measure-the-stages-independently): overlap of approximate and exact model top-k.
- [Hard negative](#5--sample-negatives-deliberately): a high-scoring or otherwise confusing contrast.
- [Hit rate](#10--measure-the-stages-independently): share of evaluable requests with at least one retrieved target.
- [Index](#8--construct-and-publish-an-index): searchable vectors plus mappings and deployment metadata.
- [In-batch negative](#5--sample-negatives-deliberately): another example's positive used as a contrast.
- [Item tower](#two-independently-computable-representations): encoder of item features.
- [MIPS](#4-definitions-and-vocabulary): maximum inner product search.
- [Negative sampling](#5--sample-negatives-deliberately): selection of contrast items for training.
- [Normalization](#6--design-tower-inputs-and-model-geometry): scaling vectors under a declared scoring convention.
- [Oversampling](#compare-filter-strategies): retrieving extra candidates before filtering.
- [Point-in-time join](#3--build-event-and-feature-contracts): join using information available at the decision.
- [Query tower](#two-independently-computable-representations): encoder of user and request features.
- [Recall@k](#10--measure-the-stages-independently): target recovery within a candidate budget.
- [Retrieval](#retrieval-is-a-gate-not-the-complete-recommender): first-stage selection from a catalogue.
- [Rollback](#11--decide-launch-monitor-and-recover): restoration of a compatible prior serving bundle.
- [Sampling probability](#5--sample-negatives-deliberately): probability specified by the actual selection design.
- [Shortfall](#translate-results-into-an-operating-decision): fewer eligible candidates than the desired pool size.
- [Tower](#two-independently-computable-representations): a separately computable representation function.

## 16. Sources

1. Covington, P., Adams, J., and Sargin, E. (2016), “Deep Neural Networks for YouTube Recommendations,” *Proceedings of the 10th ACM Conference on Recommender Systems*. [DOI: 10.1145/2959100.2959190](https://doi.org/10.1145/2959100.2959190). Architecture and sampling context.
2. Yi, X. et al. (2019), “Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations,” *Proceedings of the 13th ACM Conference on Recommender Systems*. [DOI: 10.1145/3298689.3346996](https://doi.org/10.1145/3298689.3346996). Sampling bias in large-corpus retrieval.
3. Yang, J. et al. (2020), “Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations,” *Companion Proceedings of The Web Conference 2020*. [DOI: 10.1145/3366424.3386195](https://doi.org/10.1145/3366424.3386195). Negative sampling choices.
4. Guo, R. et al. (2020), “Accelerating Large-Scale Inference with Anisotropic Vector Quantization,” *Proceedings of the 37th International Conference on Machine Learning*. [PMLR paper](https://proceedings.mlr.press/v119/guo20h.html). Approximate vector search.
5. TensorFlow Recommenders, [Basic retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval). Practical query and candidate tower example; documentation behavior may change with library versions.
6. GS1, [General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Identification standards and their scope.
7. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5, 6, and 22. Primary legal text; applicability depends on facts.
8. European Data Protection Board, [Guidelines on automated individual decision-making and profiling](https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/automated-individual-decision-making-and-profiling_en). Interpretive guidance; consult current versions and local advice.

Sources describe their own settings and dates; none establishes a result for the fictional distributor.
The legal discussion is educational and not legal advice.

## 17. Links

- [Module 24 topic map](_index.md).
- [Module 24 survey and its lab sections](../../../course/24_recommendation_systems_personalization.md).
- [Recommendation tasks](01_recommendation-tasks.md).
- [Popularity and content-based baselines](02_popularity-content-baselines.md).
- [Collaborative filtering and ALS](03_collaborative-filtering.md).
- [Sequential and session models](05_sequential-session.md).
- [Offline and online evaluation](07_offline-online-evaluation.md).
- [Business constraints and re-ranking](09_constraints-re-ranking.md).
- [Inventory-aware recommendation and substitution](10_inventory-aware-substitution.md).
