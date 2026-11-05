# Sequential and Session Models

> Module 24 — Recommendation Systems and Personalization  
> Topic 05 of 11  
> Estimated study time: 4 hours  
> Prerequisites: [Recommendation tasks](01_recommendation-tasks.md) and [Popularity and content baselines](02_popularity-content-baselines.md)  
> Version 2026.5

## 1. Why this topic exists

A customer who just viewed a cordless drill and then a masonry bit may be assembling a project.
A recommendation based on the customer's average purchases over several years may not represent that immediate need.
The distinction matters when the next screen has only a few positions, when an item is unavailable, and when an irrelevant suggestion interrupts checkout.

A session model uses an ordered sequence of interactions available before the recommendation decision.
It can adapt to a new intent without requiring a durable customer profile.
It can also amplify a mistaken click, infer sensitive circumstances, and make a visitor's experience feel intrusive.
Those risks are not solved by calling the model anonymous.

The commercial decision is not simply whether the next click can be predicted.
A merchandising owner must decide whether a surface should recommend at all, what it should optimize, and which products may be shown.
A supply-chain owner must decide how availability and delivery promises enter the decision.
A privacy owner must decide whether the proposed collection, purpose, retention, and access are appropriate and lawful.
An experimentation owner must decide whether an observed gain justifies a release.

Consider a visitor browsing rain jackets who suddenly searches for a child's school backpack.
A durable profile may still favor jackets.
A session-aware system can respond to the new activity, but it must not assume that the visitor is a child or that the items represent a family relationship.
The sequence is evidence about a task, not a verified account of identity or intent.

An organization may instead choose a simple recent-item similarity rule.
That can be the right choice when traffic is low, event order is unreliable, latency is tight, or the expected benefit of a complex model does not cover its cost.
A more expressive model earns its place only by improving a specified decision against a credible baseline under the same eligibility rules.

The business consequences include:

- A relevant complement can reduce the effort required to complete a purchase.
- An irrelevant suggestion can consume attention and obscure a customer's chosen path.
- An unavailable item can create a broken promise even if its predicted click probability is high.
- A low-margin click can be worse than a modestly less likely but useful, deliverable product.
- Excessive repetition can make the interface feel manipulative.
- A sensitive inference can create harm even when a prediction is statistically accurate.
- A model trained on its own previous exposure can mistake familiarity for preference.
- A slow recommendation call can delay the page whose experience it is meant to improve.

This topic builds from an explicit decision to a point-in-time data contract, candidate generation, sequential scoring, evaluation, and a governed release.
The numerical case is illustrative, not a claim about a company's results.
The Python example is deliberately small enough to inspect without a machine-learning framework.

## 2. Learning outcomes

After this topic you should be able to:

1. **Frame** a session recommendation decision with an actor, surface, trigger, eligible set, outcome, and owner.
   Exercise 1 asks for a decision contract.
2. **Construct** an ordered prefix and next-event label without using future events.
   Exercise 2 tests the timestamp and boundary rules.
3. **Compute** transition probabilities, top-$k$ hit rate, and reciprocal rank for a small example.
   Exercises 3 and 4 require the calculations.
4. **Compare** transition rules, recurrent models, and attention models against an appropriate baseline.
   Exercise 5 asks for a model choice with constraints.
5. **Design** a global temporal evaluation and describe the difference between offline prediction and causal online impact.
   Exercise 6 requires an evaluation plan.
6. **Apply** privacy, inventory, and customer-experience controls to a proposed session feature.
   Exercise 7 asks for a feature review and release decision.
7. **Translate** experiment estimates into contribution economics and an explicit deployment decision.
   Exercise 8 tests an arithmetic and governance recommendation.

Observable competence means explaining not just which model wins a leaderboard but why the comparison is valid, what the model can know, and which owner can stop release.
An answer that reports an accuracy metric but ignores eligibility and exposure is incomplete.

## 3. The core mental model

A session recommender is a decision pipeline operating on a prefix, not a magic reader of intent.

```text
Customer interaction stream
        |
        v
Collect permitted events + timestamps + context
        |
        v
Order events and form an as-of-decision prefix
        |
        +--> session representation / recent-item rule
        |                 |
        |                 v
        +----------> retrieve candidate products
                          |
Current catalogue,        v
inventory, policy ---> eligibility and ranking
                          |
                          v
                 show a limited slate or fallback
                          |
             exposure, response, fulfilment
                          |
                          v
       temporal evaluation + randomized experiment
                          |
           owner decision, monitoring, rollback
```

The arrow from events to a prefix is constrained by time.
At a decision time $t$, the system may use only information available by $t$ under its event and feature availability rules.
An event that occurred before $t$ but arrived after the serving decision was not available to the online system.
A later correction to a catalogue attribute is not necessarily the attribute value known at $t$.

The arrow from prefix to representation is a compression.
A last-item rule retains only the most recent eligible item.
A transition model estimates which items tend to follow another item.
A recurrent neural network (RNN) updates a hidden state as events arrive.
A Transformer uses attention over a bounded prefix and explicit or implicit order information.
All of these discard some information, make modelling assumptions, and inherit biases from logged behaviour.

The arrow from scoring to display is a separate commercial and policy decision.
A high score is not permission to recommend an item that is prohibited, unavailable, duplicative, or inconsistent with the customer's selected delivery context.
A score is also not a commitment that the customer will purchase or that stock will be available at checkout.

The observed response is generated by both the customer and the existing interface.
An item never shown has less opportunity to be clicked.
A click after a recommendation does not prove the recommendation caused the click.
An offline next-item metric tests predictive agreement with the logged world, not incremental value in a different world.

### Session intent is a working hypothesis

The prefix `view drill → view masonry bit` is compatible with a drilling project.
It is also compatible with comparison shopping, a mistaken click, or a shared device.
A system should use the evidence to choose helpful candidates rather than asserting a personal attribute.
The difference is important for product copy, privacy review, and customer trust.

### Sessions and durable profiles answer different questions

A session may summarize immediate context.
A longer-term profile may capture stable preferences, subject to permission and purpose.
Neither is automatically superior.
A design can combine them with a recency-dependent weight, but it must evaluate whether older behaviour improves the current decision.
It must also define what happens when no profile is available or permitted.

### Three clocks matter

- **Event time:** when an interaction occurred according to the event producer.
- **Availability time:** when the event became usable by the decision service.
- **Decision time:** when the recommendation was requested.

Offline reconstruction that filters only by event time can accidentally use late-arriving events.
The information boundary must reproduce what the serving system actually knew.
When availability timestamps are missing, document the approximation and test its sensitivity.

### A decision is not a forecast, plan, target, or commitment

A predicted next-click probability is a model estimate, not a demand forecast for replenishment.
A merchandising plan describes intended placement and business rules.
An experiment target is a threshold chosen for a decision, not a guaranteed result.
A delivery promise is a customer-facing commitment and must come from the approved fulfilment system.
A recommendation score cannot substitute for that promise.

## 4. Definitions and vocabulary

The table defines terms used throughout the lesson.
Times should include a timezone and a precision; prices use the currency of the relevant market.

| Term | Definition and unit | Common misuse |
|---|---|---|
| Decision | Choice of slate or no slate at a request | Treating a training row as a customer decision |
| Surface | Interface location, such as product page | Pooling surfaces with different intent and constraints |
| Trigger | Event or request that initiates ranking | Assuming every event triggers a visible recommendation |
| Event | Recorded interaction with an event type and time | Treating every log row as a meaningful action |
| Event time | Time interaction occurred, in UTC or an explicit timezone | Assuming it equals ingestion time |
| Availability time | Time data became available to the serving system | Ignoring late arrival in backtests |
| Decision time | Time at which the recommendation must be chosen | Using data produced after the request |
| Session | Operational grouping of events within a defined boundary | Assuming it is a single human or purchase intent |
| Inactivity timeout | Maximum permitted gap before a new session, measured in minutes | Treating a conventional value as a natural law |
| Prefix | Events usable before a particular decision | Including the target action in its own input |
| Target | Later event selected for prediction | Treating observed target as the only relevant item |
| Label window | Allowed interval after a decision for a target | Counting actions indefinitely into the future |
| Candidate | Item considered by the selection process | Assuming an ungenerated item was scored |
| Eligible item | Candidate satisfying current hard constraints | Using today's stock status for historical decisions |
| Retrieval | Producing a manageable candidate set | Confusing retrieval recall with final-slate quality |
| Ranking | Ordering eligible candidates for a surface | Assuming raw scores are calibrated probabilities |
| Slate | Ordered set of displayed items | Treating each position as independent |
| Transition count | Number of eligible observed adjacent item pairs | Interpreting counts as causal effects |
| Markov assumption | Next-step distribution summarized by a current state | Assuming all user intent depends on only one item |
| Smoothing | Combining sparse estimates with a prior or baseline | Hiding that a candidate has no observations |
| Backoff | Switching to a simpler rule when evidence is insufficient | Returning an empty list unnecessarily |
| Recency | Time or order distance from the decision | Assuming most recent always means most relevant |
| Dwell time | Approximate time between an event and a subsequent signal, in seconds | Interpreting an idle tab as attention |
| Impression | Record that an item was actually presented | Treating a generated but unseen slate as exposure |
| Position bias | Response variation caused partly by placement | Attributing position-driven clicks entirely to relevance |
| Exposure bias | Logged responses reflect what prior systems displayed | Treating unobserved items as dislikes |
| Point-in-time join | Join using values known at the historical decision | Joining with a later corrected value |
| Leakage | Information unavailable at decision time enters training or evaluation | Assuming a chronological row order alone prevents leakage |
| Global temporal split | All training data precedes a common time cutoff | Training on future behaviour from other users |
| Hit rate at $k$ | Fraction of decisions whose target appears in the first $k$ results | Calling it conversion rate |
| Reciprocal rank | $1/r$ for target rank $r$, otherwise zero | Treating the average rank as its reciprocal |
| NDCG | Normalized discounted cumulative gain, an ordering metric | Comparing values without defining labels and candidate universe |
| Randomization unit | Entity assigned to an experiment arm | Counting repeated requests as independent assignments |
| Incrementality | Difference caused by assigning treatment instead of control | Calling an association a causal effect |
| Contribution | Revenue less specified variable costs, in currency per period | Calling gross revenue profit |
| Retention period | Authorized duration of storage, measured in days or another unit | Equating a session timeout with data deletion |
| Pseudonymous identifier | Identifier replacing direct identity but potentially linkable to a person | Treating it as automatically anonymous |
| Fallback | Safe alternative when ranking cannot be served | Treating fallback traffic as model success |

## 5. The mechanics, step by step

### Step 1 — Specify the decision contract

**Input:** a business problem, surface, customer task, and operational constraints.

**Output:** a written decision contract approved by the relevant owners.

1. Name the surface and the precise request that triggers a decision.
2. Identify whether the system recommends a related product, complement, replacement, or next action.
3. Define whether the system can show no recommendation.
4. Define how many positions are available and where they appear.
5. Record the customer segment and geography for which the surface is enabled.
6. Define hard exclusions before discussing model scores.
7. Record the measurement window, commercial objective, and experience guardrails.
8. Assign an accountable product owner and owners for data, policy, supply, and experimentation.

For a product-page complement carousel, the trigger could be the page request after a product detail view.
The decision is the next four eligible complements, not a prediction of the visitor's entire life history.
The offer must be purchasable in the relevant market and compatible with the selected delivery option when those facts are known.

**Branch:** if the surface has no visible placement or no reliable exposure log, first fix measurement or choose a surface that can be measured.
A model score without a defined display decision is not a deployable recommendation system.

### Step 2 — Define the session boundary

**Input:** event identifiers, event time, availability time, permitted session key, and surface context.

**Output:** reproducible session identifiers and ordered event sequences.

A session can be defined as a sequence of events associated with a permitted session key, split when the gap between successive events exceeds a selected inactivity threshold.
For example, a 30-minute threshold is an operational hypothesis, not a universal definition.
A gap of exactly 30 minutes needs an explicit rule: this lesson starts a new session only when the gap is greater than 30 minutes.

Decide how to handle:

- Duplicate event IDs from retries.
- Multiple events with identical timestamps.
- Device clock errors and timezone conversion.
- Late-arriving or corrected events.
- Concurrent browser tabs.
- Authentication during an existing session.
- Consent changes or an opt-out during a session.
- A session that crosses midnight or an evaluation cutoff.
- Bot traffic, synthetic monitoring, and internal staff activity.
- Long pauses followed by a restored browser tab.

A deterministic tie-breaker such as a producer sequence number can resolve ties if it existed at serving time.
If no trustworthy order exists, do not invent one using a future ingestion order and claim perfect sequence fidelity.
You may instead exclude ambiguous transitions, group simultaneous events, or evaluate the uncertainty.

A session identifier is an operational key, not proof that events came from one person.
A shared browser can contain several people; one person can use several devices.
Do not merge identities solely because a longer sequence would be convenient for modelling.

**Branch:** if the permitted identifier or event timestamps are too unreliable, use a contextual or non-personalized baseline instead of fabricating a session.

### Step 3 — Define the prediction target

**Input:** decision contract, event taxonomy, and customer outcome of interest.

**Output:** a target event and a bounded label window.

Possible targets include next product-detail view, next add-to-cart, or a purchase within a specified interval.
They have different frequencies, commercial meaning, and susceptibility to interface effects.
A product view is plentiful but weak evidence of value.
A purchase is more meaningful but sparse, delayed, and affected by price, delivery, and payment.

Specify whether a target must be:

- A different item from the item currently viewed.
- A newly displayed recommendation or any item reached by another route.
- Available at the historical decision time.
- In the same market, fulfilment mode, and customer context.
- Observed within a fixed number of minutes or hours.
- A first action, any action, or a weighted combination of actions.

For a next-view task, the label is the next eligible product view after the decision, not the last item in the entire future session.
For a complement purchase task, a same-item repeat may be a poor label even if it is common in logs.
Define handling for returns, cancellations, and multi-item baskets before using purchase as a value label.

**Branch:** if the next event is not in the historical eligible catalogue, report its rate and exclude it only under a predeclared rule.
Otherwise the evaluation can hide impossible tasks or silently change the denominator.

### Step 4 — Build point-in-time prefixes

**Input:** ordered permitted events, decision times, label windows, and historical item state.

**Output:** one example per eligible decision, with a prefix and a target if observable.

For each decision time $t$:

1. Select events with event time strictly before $t$, unless a documented trigger event is known to be available at $t$.
2. Require availability time no later than $t$.
3. Apply the session boundary and privacy rules in force at $t$.
4. Limit the prefix to the selected length or lookback duration.
5. Construct features using historical values available at $t$.
6. Select the target only from the allowed future label window.
7. Store the eligible candidate universe or a reproducible historical snapshot.
8. Save the decision identifier, split assignment, and exclusion reason.

The recommendation request may include the just-viewed product in its payload.
If so, it is legitimate to include that product even if the corresponding asynchronous click event has not arrived, provided production has the same payload.
The data contract should state this explicitly.

A prefix with one item is still useful.
A prefix with zero usable items needs a fallback.
A prefix extending across a cutoff can be valid at evaluation time if the prior events were actually observable; it must not place evaluation labels into model training.

Do not compute a feature from the final session length for an earlier decision.
At the third event, the system cannot know that the visitor will produce eight events in total.
The same warning applies to eventual purchase, eventual return, and future inventory correction.

### Step 5 — Audit feature families

**Input:** point-in-time prefix and the approved data inventory.

**Output:** feature definitions with source, availability, transformation, and retention.

Useful families include:

1. **Item identity:** current product and recent product identifiers.
2. **Event type:** view, search, cart addition, removal, or purchase where permitted and meaningful.
3. **Ordering:** event index, order relative to the decision, and bounded sequence length.
4. **Time:** elapsed seconds since the last interaction and time gaps between events.
5. **Content:** category, brand, size, compatibility, or product text as known at decision time.
6. **Context:** surface, market, language, and fulfilment mode when relevant and allowed.
7. **Interaction summaries:** number of distinct categories so far, recent search terms, or repeated views.
8. **Operational state:** available-to-promise, price, delivery eligibility, and policy flags from authorized systems.

Item IDs are high-cardinality and can expose sparse or newly introduced products.
Content can provide a backoff when IDs have not been seen before.
A category path can be more stable than a product title, but taxonomy changes need versioning.

Time gaps may distinguish a rapid comparison from a resumed visit, but an open tab can produce misleading dwell estimates.
Use capped or bucketed gaps if that is consistent with the problem; record the bucket boundaries.
Never infer attention from the absence of a subsequent event without considering telemetry gaps.

Repeated views can indicate interest, confusion, or accidental navigation.
Represent repetition as a measurable signal rather than interpreting it as certainty.
Similarly, a cart removal is not necessarily a dislike: the customer may have chosen another size or encountered delivery costs.

Search text can carry sensitive information.
Decide whether the task requires raw text, a limited category, an on-device representation, or no search feature at all.
A more detailed feature is not automatically justified by a small offline gain.

For every feature, document the producer, event-time definition, availability delay, null behaviour, retention, access policy, and owner.
A feature with no serving-time source is not deployable merely because it exists in a warehouse.

### Step 6 — Establish non-sequential and simple sequential baselines

**Input:** training data before the cutoff, candidate eligibility rules, and evaluation protocol.

**Output:** baseline slates and reproducible metrics.

At minimum compare:

- Eligible popularity by market and surface, computed only from earlier data.
- Content similarity to the current item, with an explicit tie rule.
- A last-item transition model trained on earlier sessions.
- Where feasible, the current production policy under its actual constraints.

Popularity must use a stated window and event definition.
A trend computed using the evaluation period leaks future demand.
If popularity is global but inventory is local, the resulting slate may be infeasible for the customer's location.

Content similarity should use a stable catalogue snapshot and an interpretable similarity rule.
It can work for a new product with descriptive attributes even when no transitions exist.
However, similarity alone can return near-duplicates where the task calls for complements.

For adjacent training items $i$ and $j$, let $C(i,j)$ be the number of observed transitions.
The unsmoothed estimate is:

$$P(j\mid i)=\frac{C(i,j)}{\sum_{k} C(i,k)}.$$

It is undefined for an item with no observed outgoing transition.
Back off to an eligible content or popularity list in that case.
If the model uses smoothing, state the prior distribution and weight; do not silently treat a missing transition as strong evidence of dislike.

A last-item model is fast, inspectable, and often competitive.
It cannot distinguish `drill → bit` from `paint → bit` if both end with the same item, unless context is added to the state.
That limitation creates a testable reason to consider a richer sequence model.

### Step 7 — Select a sequential representation

**Input:** baseline residual errors, volume, latency budget, feature contract, and governance requirements.

**Output:** a model family and an explicit justification for its additional complexity.

A first-order transition table uses one previous state.
A higher-order table uses a fixed history, such as the last two items, but the number of possible states grows rapidly.
A variable-order rule can back off from a sparse two-item context to a one-item transition and then to popularity.
Its strength is transparency; its weakness is sparsity and limited generalization to new combinations.

An RNN processes a sequence in order and updates a hidden state.
A gated recurrent unit is a common gated RNN variant that can retain or forget information across steps.
Training typically presents prefixes and optimizes a next-item objective.
At serving time, the state must reflect only available events and must be reset at the selected session boundary.
Batching, cached state, and sequence truncation affect latency and consistency.

An attention-based model computes context-dependent weights across a bounded prefix.
A causal mask prevents an earlier training position from attending to later positions when predicting a future item.
Positional information is needed because a bag of item embeddings alone does not distinguish order.
A bidirectional masked-item training objective can be useful, but its training mask and serving use must be aligned; do not expose future events to a next-item evaluation.

A graph-based session model represents items and observed within-session transitions as nodes and edges.
It can express repeated transitions and local structure but adds graph construction and serving complexity.
Its benefit must be demonstrated for the particular task rather than inferred from architecture alone.

A sequential encoder can feed retrieval, ranking, or both.
If it produces an embedding for approximate nearest-neighbour retrieval, the index version and item embedding freshness become part of the serving contract.
A ranker that only sees a narrow retrieved candidate set cannot recover a useful item that retrieval omitted.

**Branch:** prefer the simpler approach when the richer model's incremental value is uncertain, its data requirements are unmet, or its operating costs exceed its benefit.
Architecture is a means, not a business objective.

### Step 8 — Construct the training objective

**Input:** permissible training prefixes, targets, candidate universe, and model family.

**Output:** training examples and a reproducible objective.

Next-item classification can optimize the likelihood of the observed next item among a specified universe.
With a large catalogue, training may use sampled negatives or an approximate objective.
The sampling distribution changes what the model learns and how the loss should be interpreted.
A sampled item is not automatically a true negative: the customer may never have seen it.

An impression-based ranking objective can use exposed items and responses.
This gives a better-defined opportunity set but inherits the previous policy's position and selection biases.
Record the displayed slate, order, visibility, and any randomization propensities needed by the chosen analysis.

Purchase-weighted losses can emphasize business outcomes, but a large weight on a sparse event can make estimates unstable.
A cancellation or return may arrive after training extraction.
Specify a maturity window and whether financial outcomes are net of returns.

Separate training-example weighting from the business decision rule.
For example, weighting cart additions more heavily does not guarantee an increase in margin or customer satisfaction.
Only a properly designed online comparison can estimate the incremental effect of changing the policy.

### Step 9 — Make candidate generation and constraints explicit

**Input:** current prefix, catalogue, inventory, policy rules, and retrieval sources.

**Output:** an eligible, ranked slate or a documented fallback.

Candidate sources might include transition neighbours, content neighbours, a sequential embedding index, and a popular-item backoff.
Deduplicate by the commercial entity appropriate to the surface, such as product family rather than variant, if the interface should avoid near-identical items.
Preserve source and model-version metadata for diagnosis.

Hard exclusions should run against authoritative, sufficiently fresh sources.
Examples include restricted goods, unavailable fulfilment, incompatible parts, and products the customer cannot purchase in the selected market.
Whether to exclude an item already in the cart depends on the task; a consumable replenishment scenario may differ from an accessory scenario.

A model can rank eligible items using relevance and business signals.
Any margin or promotion adjustment should be reviewed for customer utility and transparency.
Do not invent a benefit for the customer merely because an item has a higher margin.

For an empty or stale candidate set, use a safe fallback or suppress the carousel.
For a timeout, record the failure and show the preapproved fallback within the page latency budget.
A stale inventory service may require a conservative fallback rather than an unsupported claim of availability.

### Step 10 — Evaluate in time

**Input:** decision rows, training cutoff, historical feature availability, eligible universe, and frozen evaluation rules.

**Output:** offline metrics with denominators, uncertainty, and segment diagnostics.

Choose a common global cutoff $T$.
Train using events and mature labels available before $T$.
Tune on a later validation interval and evaluate once on a subsequent untouched test interval.
If training labels can arrive after their decision, use a gap or label-maturity rule so the model does not learn from outcomes unavailable at the simulated training time.

For a rolling evaluation, repeat the train, validate, and test process at several historical cutoffs.
Each fold must reconstruct the model and historical candidate state using only data available at its own cutoff.
Report seasonal or promotional conditions because a single period may be atypical.

For each test decision, rank the same eligible universe or use the same candidate-generation budget across methods.
If a method retrieves a smaller candidate set, measure retrieval recall separately from ranking performance.
A target outside the candidate set contributes zero to end-to-end hit rate unless the evaluation explicitly reports a conditional ranking metric.

For $N$ decisions and one target per decision:

$$\mathrm{Hit@}k=\frac{1}{N}\sum_{n=1}^{N}\mathbf{1}[r_n\leq k],$$

where $r_n$ is the target's rank and a missing target has infinite rank.
Mean reciprocal rank (MRR) averages $1/r_n$, with zero for a missing target.
State whether ranks are measured across the full eligible universe or a sampled set.
Sampled-negative results are not directly comparable to full-catalogue results.

Report coverage, no-candidate rate, latency, catalogue concentration, and outcomes by relevant slices alongside relevance metrics.
Useful slices can include first-event sessions, long sessions, new items, market, device, and availability context.
Small slices need uncertainty intervals or counts; avoid declaring a fairness conclusion from an unstable estimate.

### Step 11 — Design an online experiment

**Input:** offline evidence, proposed treatment, control policy, business threshold, and approved guardrails.

**Output:** randomized experiment protocol and decision rule.

Randomize at an appropriate stable unit, such as an account or permitted browser identifier, and keep assignment consistent for its defined lifetime.
The choice depends on privacy, cross-device behaviour, and the expected interference between requests.
Repeated page views from one unit are not independent experimental assignments.

Define the primary metric before launch, for example net contribution per randomized eligible session over a fixed observation window.
Also define customer-experience guardrails, such as page latency, unsuccessful clicks, stock-related disappointment, and complaint rates where measurable.
A click-through increase alone is not sufficient if it comes from misleading placement or erodes contribution.

Log assignment even when no recommendation is delivered.
An intention-to-treat analysis compares assigned groups, including service failures and fallback, preserving the benefit of randomization.
A separate delivered-treatment diagnostic can help debugging but is not automatically causal because delivery may depend on post-assignment conditions.

Choose duration and sample size based on baseline variance, minimum worthwhile effect, statistical power, and business constraints.
Account for repeated looks, seasonality, and multiple comparisons in the analysis plan.
Do not stop merely because an early metric looks favorable.
Document what happens if the stock system, price policy, or campaign changes during the test.

### Step 12 — Operate, review, and retire

**Input:** experiment result, operational health, privacy assessment, and owner approvals.

**Output:** release, hold, revise, or rollback decision with monitoring thresholds.

Monitor event freshness, session construction failures, candidate availability, model and index versions, timeout rate, fallback rate, and customer-experience guardrails.
Monitor outcome and traffic mix by relevant segments, taking care not to collect new sensitive attributes solely for monitoring without a lawful and justified basis.

The product owner owns the user-facing decision.
The supply or fulfilment owner owns stock and delivery eligibility feeds.
The data owner owns event quality and lineage.
The privacy and legal owners review purposes, retention, access, and applicable obligations.
The experiment owner owns randomization, metric definitions, and analysis.
An on-call owner must have authority and a tested route to revert to the approved baseline.

Set a review date for the model, features, and retention policy.
A feature that once helped may become unnecessary after a surface or catalogue change.
Retire it rather than keeping it indefinitely because it is already available.

## 6. Worked example

### Situation and decision

A fictional home-improvement retailer has a product-page carousel for accessory recommendations.
The carousel has two positions.
A request occurs after the visitor opens a product detail page.
The retailer wants to recommend relevant, in-stock accessories without delaying the page or showing incompatible products.

The product owner proposes a last-item transition baseline before investing in a neural model.
The fulfilment owner supplies the historical market-specific eligibility snapshot.
The privacy owner approves only the described session event fields for this fictional exercise; that approval is an assumption, not a legal conclusion.
The experiment owner defines offline metrics and an eventual randomized test.

The following product codes are illustrative:

| Code | Product | Role in example |
|---|---|---|
| A | Cordless drill | Current product in training sessions |
| B | Masonry bit set | Current product in test decisions |
| C | Dust extraction attachment | Candidate accessory |
| D | Safety glasses | Candidate accessory |

Compatibility is assumed verified independently of the model.
The example is intentionally small so the arithmetic can be checked by hand.
The counts are not evidence that these products are commonly bought together.

### Training data

Three historical sessions were completed before the training cutoff.
Each arrow denotes the next recorded product view within that session.

| Session | Ordered product views | Adjacent transitions |
|---|---|---|
| S1 | A → B → C | A → B; B → C |
| S2 | A → B → D | A → B; B → D |
| S3 | A → C → D | A → C; C → D |

The transition counts are:

| Origin | Destination | Count, transitions |
|---|---|---:|
| A | B | 2 |
| A | C | 1 |
| B | C | 1 |
| B | D | 1 |
| C | D | 1 |

The total outgoing count from A is $2+1=3$ transitions.
Therefore $P(B\mid A)=2/3$ and $P(C\mid A)=1/3$.
The total outgoing count from B is $1+1=2$ transitions.
Therefore $P(C\mid B)=1/2$ and $P(D\mid B)=1/2$.
The total outgoing count from C is one transition, so $P(D\mid C)=1$ in this tiny sample.
That last estimate is highly uncertain; it does not establish that every customer viewing C wants D.

### Test decisions

Two later decisions occur after the cutoff.
Both have B as their most recent item.
For this example, C and D are eligible at each decision and both labels are observed in the specified next-view window.
The rule breaks equal scores alphabetically by product code, so the slate is `[C, D]` for both decisions.

| Decision | Prefix ending | Eligible slate | Observed next view | Target rank |
|---|---|---|---|---:|
| T1 | B | C, D | C | 1 |
| T2 | B | C, D | D | 2 |

At one position, T1 is a hit and T2 is not.
Hit@1 is $(1+0)/2=0.50$, or 50% of the two decisions.
At two positions, both are hits.
Hit@2 is $(1+1)/2=1.00$, or 100% of the two decisions.

The reciprocal ranks are $1/1=1$ for T1 and $1/2=0.5$ for T2.
MRR is $(1+0.5)/2=0.75$.
These statistics describe agreement with two observed next views, not the effect of displaying the carousel.
With only two decisions they are far too unstable to support a launch.

If D had been ineligible at T2, the model should not display D merely to improve the metric.
The evaluation must apply the prespecified target-eligibility rule and report how many decisions were affected.
If the rule retains the decision in an end-to-end measure, D is a miss; if a different explicitly defined eligible-target metric excludes it, the exclusion count and changed denominator must be visible.

### Larger illustrative business calculation

Suppose a properly randomized pilot later has 20,000 eligible sessions per arm over a fixed period.
Assume the measured incremental contribution is USD 0.018 per eligible session, before the incremental system cost.
The point estimate of incremental contribution in the treatment arm is:

$$20{,}000\ \text{sessions}\times\text{USD }0.018/\text{session}=\text{USD }360.$$

If incremental variable serving and operating cost allocated to those treatment sessions is USD 110, the estimated net contribution is USD 250 for that treatment volume.
This is an illustrative point estimate, not a forecast or a commitment.
It does not include any fixed implementation cost unless stated.
A confidence interval that includes a material loss, or a guardrail breach, can still justify holding the launch.

For a decision, the team should compare the incremental net benefit with the cost of development and operation at the intended scale, examine uncertainty, and check that the randomized comparison remained valid.
The product owner does not get to override a binding policy exclusion simply because the point estimate is positive.

### What the example does and does not establish

- It establishes how to count adjacent transitions from the three specified sessions.
- It establishes how the fixed tie-breaker creates the two example slates.
- It establishes how to calculate Hit@1, Hit@2, and MRR on the two labels.
- It does not establish a causal impact on clicks, revenue, or contribution.
- It does not establish that a larger sequential model is necessary.
- It does not validate inventory accuracy or accessory compatibility.
- It does not establish compliance with any privacy law.
- It does not turn an estimated recommendation score into a stock or delivery promise.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the training transition counts and test metrics in Section 6.
Copy the entire block into a file and run it with `python`.
The data are synthetic; no external files, network access, or packages are required.

Assumptions:

- Each training sequence is already ordered and belongs to a separate historical session.
- All training sessions precede the test decisions.
- Both test candidates are eligible at decision time.
- Each test decision has exactly one observed next-view target.
- Ties are broken by product code in ascending order.
- A target absent from the slate receives reciprocal rank zero.
- The program demonstrates a transition rule, not a production event pipeline.

```python
from collections import Counter, defaultdict

# Historical sessions, ordered before the evaluation cutoff.
train_sessions = [
    ["A", "B", "C"],
    ["A", "B", "D"],
    ["A", "C", "D"],
]

# The prefix is represented here by its last observed item.
# A real system must reconstruct the complete as-of-decision context.
test_decisions = [
    {"last_item": "B", "eligible": {"C", "D"}, "target": "C"},
    {"last_item": "B", "eligible": {"C", "D"}, "target": "D"},
]


def fit_transitions(sessions):
    counts = defaultdict(Counter)
    for session in sessions:
        for origin, destination in zip(session, session[1:]):
            counts[origin][destination] += 1
    return counts


def recommend(counts, last_item, eligible):
    # Sorting makes ties deterministic. This toy example does not backfill
    # candidates with zero transition counts; production needs a fallback.
    neighbours = counts.get(last_item, {})
    candidates = [item for item in neighbours if item in eligible]
    return sorted(candidates, key=lambda item: (-neighbours[item], item))


def rank_of(target, slate):
    try:
        return slate.index(target) + 1
    except ValueError:
        return None


def evaluate(counts, decisions):
    hit1 = 0
    hit2 = 0
    reciprocal_rank_sum = 0.0
    for decision in decisions:
        slate = recommend(counts, decision["last_item"], decision["eligible"])
        rank = rank_of(decision["target"], slate)
        print(f"target={decision['target']} slate={slate} rank={rank}")
        if rank is not None:
            hit1 += rank <= 1
            hit2 += rank <= 2
            reciprocal_rank_sum += 1.0 / rank
    n = len(decisions)
    if n == 0:
        raise ValueError("At least one test decision is required")
    return hit1 / n, hit2 / n, reciprocal_rank_sum / n


counts = fit_transitions(train_sessions)
for origin in sorted(counts):
    total = sum(counts[origin].values())
    for destination in sorted(counts[origin]):
        count = counts[origin][destination]
        print(f"{origin}->{destination}: count={count}, p={count / total:.3f}")

hit1, hit2, mrr = evaluate(counts, test_decisions)
print(f"Hit@1={hit1:.2f} Hit@2={hit2:.2f} MRR={mrr:.2f}")
```

Expected output:

```text
A->B: count=2, p=0.667
A->C: count=1, p=0.333
B->C: count=1, p=0.500
B->D: count=1, p=0.500
C->D: count=1, p=1.000
target=C slate=['C', 'D'] rank=1
target=D slate=['C', 'D'] rank=2
Hit@1=0.50 Hit@2=1.00 MRR=0.75
```

The code filters observed transition neighbours by eligibility.
It does not add an eligible item with zero observed transitions, so an unseen last item yields an empty slate.
A production implementation needs the approved fallback described earlier.
This omission is explicit because the artifact is for reproducing the calculation, not for deployment.

The program also does not infer the historical eligibility snapshot, check event arrival times, or estimate uncertainty.
Those require the actual event and catalogue contracts.
Do not interpret a correct toy calculation as evidence that a real temporal backtest is leakage-free.

## 8. Data, standards, and regulatory context

### Event and catalogue contract

A minimally useful event record contains a stable event ID, permitted session key, event type, product ID when applicable, event time, and availability time.
A decision record contains a decision ID, trigger, surface, timestamp, eligible context, model version, and treatment assignment where applicable.
An impression record should distinguish a candidate generated by the system from an item actually displayed to the customer.

A catalogue record needs an item identifier, validity interval, market, product attributes, and the relevant eligibility attributes.
Inventory should be joined at the location and time granularity used by the customer-facing promise.
A warehouse's current product table is not a historical availability ledger.

A reliable event pipeline needs explicit handling for:

- Retry deduplication and idempotent event IDs.
- Invalid or missing item IDs.
- Clock skew and timezone normalization.
- Late arrival and backfill.
- Deletion and correction requests.
- Schema changes and event-type versioning.
- Test traffic and automated activity.
- Attribution between a decision and actual display.
- Catalogue or inventory snapshots missing at decision time.
- Retention and access restrictions for raw and derived data.

The feature contract should identify an owner and a freshness service-level objective in units of seconds or minutes.
A freshness objective is an operating target, not a legal safe harbour and not proof that every individual event arrived on time.

### Identifiers and product standards

GS1 Global Trade Item Numbers (GTINs) provide a standardized way to identify trade items when correctly assigned and maintained.
A retailer may also use internal stock-keeping unit identifiers.
A mapping between them is a data-governance decision; neither identifier proves that two variants are interchangeable or that a customer can receive one at a particular location.
Refer to GS1's published GTIN guidance for the applicable identification rules.

If the model works at product-family level but the cart and inventory systems work at variant level, record the mapping and validity period.
A recommendation for a jacket family does not establish that the selected size is in stock.
A compatible substitution requires product and fulfilment rules beyond identifier matching.

### Privacy and lawful processing

The EU General Data Protection Regulation (GDPR) applies under its territorial and material scope; applicability and obligations require assessment of the facts.
Article 5 sets out principles including purpose limitation, data minimization, storage limitation, integrity and confidentiality, and accountability.
Article 6 addresses lawful bases for processing.
Article 25 addresses data protection by design and by default.
These are legal requirements where applicable, not a universal rule that every session recommender must obtain the same form of consent.

Online identifiers can be personal data when they relate to an identifiable person in context.
Pseudonymization can reduce risk but does not automatically make data anonymous.
The controller should assess identifiability, linkage, access, and purpose rather than relying on a label such as `anonymous_session_id`.

The EU ePrivacy framework and national implementations can impose separate rules for storing or accessing information on a user's device.
The applicable rules and exceptions depend on the technology, jurisdiction, and purpose.
A short-lived identifier is not automatically exempt solely because it expires at the end of a session.
Seek jurisdiction-specific legal review before deployment.

The California Consumer Privacy Act, as amended by the California Privacy Rights Act, may apply to covered businesses and personal information under its statutory conditions.
It provides specified consumer rights and business obligations, subject to exceptions and implementing regulations.
Do not assume that an identifier is outside scope merely because it does not contain a name.
The California Privacy Protection Agency provides primary guidance and regulations.

As of November 2026, legal duties and regulator interpretations can change.
This material is educational, not legal advice.
The responsible legal and privacy teams must determine applicable law, lawful basis, notice, rights handling, retention, and any required assessment for the actual deployment.

### Data minimization in a session system

Before collecting a feature, document:

1. The specific decision it supports.
2. Why a less detailed or less persistent alternative is insufficient.
3. The lawful basis and notice or consent requirements determined by counsel.
4. Whether the feature can be processed locally or transiently.
5. The permitted retention duration and deletion method.
6. Who can access the raw field and derived representation.
7. Whether it can reveal sensitive circumstances or affect vulnerable customers.
8. How opt-outs and deletion requests propagate to logs, features, and downstream models as required.

A 30-minute inactivity timeout controls session grouping.
It does not by itself delete raw logs, embeddings, backups, or model parameters after 30 minutes.
Separate policies are needed for those stores and derivatives.

Sensitive search terms and inferred circumstances deserve particular care.
Even when a system does not explicitly store a health category, a sequence can reveal a sensitive pattern.
Avoid displaying explanations that expose another person's browsing on a shared screen.
A relevance gain does not settle whether that processing is proportionate or lawful.

### Security and governance

Limit access to event-level data to authorized roles.
Use encryption and access logging appropriate to the organization's risk assessment.
Document retention in backups and the procedure for responding to security incidents.
A recommendation embedding is derived data and may retain information about input behaviour; do not assume it is harmless because it is numerical.

When a deletion or correction request is received, the required downstream treatment depends on applicable law and the technical and organizational context.
Document the decision and the treatment of raw events, feature stores, training sets, and model artifacts.
Do not promise instantaneous removal from every model without an implemented, validated process.

### Standards versus business policy

A product identifier standard can improve interoperability.
A legal requirement can constrain collection and use.
A retailer's rule such as “show no more than one variant from a family” is an internal experience policy.
These are distinct kinds of authority and should not be described interchangeably.
The model team cannot create a legal exemption by choosing a particular architecture.

## 9. Industry practice and case evidence

### Public research: session-based recommendations

Hidasi and colleagues' paper *Session-based Recommendations with Recurrent Neural Networks* describes a recurrent approach to recommendation from session sequences.
Its documented contribution is a modelling and evaluation approach for session data, not proof that every retailer should deploy an RNN.
The paper illustrates why ordering can be useful when durable profiles are absent or unhelpful.

The practical lesson is to define an ordered prefix and compare the RNN against strong simple methods under a matching evaluation protocol.
The evidence does not remove the need to measure current eligibility, latency, or contribution in a new business.

### Public research: self-attention

Kang and McAuley's *Self-Attentive Sequential Recommendation* describes a self-attention architecture for sequential recommendation.
The model uses sequence order and attention to represent historical interactions for next-item prediction.
Its research results concern the reported datasets and evaluation settings.
They are not a guarantee of incremental profit in a particular commerce surface.

A deployment team should test whether the model improves over last-item and popularity baselines on its own point-in-time data.
It should separately measure retrieval recall and system latency.
A good offline score cannot demonstrate that a customer saw an item or benefited from it.

### Public research: graph structure

Wu and colleagues' *Session-based Recommendation with Graph Neural Networks* describes representing a session as a graph and learning a session representation from it.
This can represent transitions and repeated item relationships in a different way from a plain linear sequence.
The paper provides a model design and benchmark results, not a universal rule that graph construction is operationally worthwhile.

A graph model may be reasonable when a team can explain which observed behaviour the graph captures that a simpler model misses.
It still needs a safe fallback for unseen products, a bounded serving cost, and the same candidate eligibility controls.

### Public research: exposure and evaluation

Schnabel and colleagues' *Recommendations as Treatments* discusses selection bias in recommendation data and approaches to evaluation under biased observations.
The core relevance here is that observed ratings or responses depend on selection and exposure.
A log of clicks is not a complete list of customer preferences.

An organization should log actual impressions and evaluate policy changes with randomization when feasible.
Inverse-propensity methods depend on correct propensities, support, and other assumptions; they are not a substitute for missing exposure data by declaration.

### What can be inferred for a retailer

**Inference:** a retailer with short, anonymous browsing sessions may find that recent-item context complements a longer-term profile.
The inference is plausible because recent behaviour can reflect a temporary task, but the cited research does not establish the size of that benefit for this retailer.

**Inference:** a retailer with highly volatile local inventory may gain more from accurate eligibility and a simple relevance rule than from a larger sequence model with stale stock data.
This is a decision hypothesis to test, not a published result attributed to any named company.

**Inference:** a customer may find a session-aware recommendation intrusive if the interface reveals a sensitive inference or repeatedly follows an accidental click.
A customer-research study and complaint monitoring are appropriate ways to investigate that risk; an offline ranking metric alone cannot settle it.

### Translating evidence into practice

Use published methods to generate candidate designs, not to import claimed performance.
Reproduce the decision contract, data availability, candidate universe, and metric definitions before comparing results.
The relevant operational evidence includes how often a fallback is used and whether the system is eligible to show the predicted item.

A mature review separates three questions:

1. Does a model predict logged behaviour better under a valid temporal protocol?
2. Does the whole system improve the customer experience and business outcome under random assignment?
3. Is the processing and display appropriate, lawful, reliable, and sustainable?

A positive answer to one question does not automatically answer the others.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future-event leakage | Offline gain disappears in a realistic replay | Prefix contains later or late-arriving data | Inflated model selection | Join on event and availability time |
| Global split leakage | Newer training events overlap test calendar | Per-user splits expose future trends | Unrealistic evaluation | Use common chronological cutoffs |
| Target leakage | Session length or final purchase predicts early decisions | Future summary included as a feature | Unservable model | Compute features as of each decision |
| Broken ordering | Many tied or reversed timestamps | Clock skew or concurrent tabs | False transitions | Preserve producer ordering and flag ambiguity |
| Session fragmentation | High fraction of one-event sessions | Timeout too short or identifiers reset | Lost context | Audit gaps and compare timeout sensitivity |
| Session merging | Extremely long sessions or mixed categories | Shared devices or timeout too long | Misread intent | Cap and test boundaries; avoid identity claims |
| Exposure bias | Popular displayed items dominate labels | Prior policy controls opportunity | Self-reinforcing ranking | Log impressions; randomize where appropriate |
| Sampled-negative distortion | Large metric changes with sampling seed | Easy or mismatched negative pool | Misleading comparisons | Report sampling and full-universe checks |
| Candidate bottleneck | Ranker looks strong conditionally but end-to-end recall is low | Relevant item never retrieved | Customer never sees useful candidate | Measure retrieval and end-to-end metrics |
| Stale inventory | High click-to-unavailable rate | Eligibility snapshot lags stock changes | Broken customer promise | Fresh authoritative checks and conservative fallback |
| Incompatible accessory | High return or complaint rate for a pair | Co-view interpreted as compatibility | Harm and unnecessary returns | Apply verified compatibility rules |
| New-item starvation | New products receive almost no exposure | No transition history | Narrow catalogue and poor cold start | Content fallback and governed exploration |
| Repetition | Same items appear across consecutive requests | No slate memory or diversity rule | Fatigue and reduced trust | Bound repetition and measure outcomes |
| Timeout cascade | Fallback and page latency rise together | Model service misses deadline | Slower customer journey | Deadline, circuit breaker, safe fallback |
| Privacy overcollection | Raw search text retained without a clear need | Convenience replaces purpose review | Exposure of sensitive information | Minimize, restrict, and review retention |
| Hidden identity linkage | Session IDs are joined across devices without review | Convenience-driven profile enrichment | Unexpected tracking | Approve linkage purpose and controls |
| Metric gaming | Clicks rise while contribution or satisfaction falls | Model optimizes a proxy | Commercial and experience loss | Predeclare primary and guardrail metrics |
| Experiment contamination | One unit receives both treatments | Unstable assignment or shared identity | Biased treatment estimate | Stable assignment and diagnostics |
| Fallback blind spot | Model dashboard excludes failed requests | Only served predictions are measured | Inflated impact | Analyze all assigned eligible units |
| Nonstationarity | Recent sessions diverge from training distribution | Campaign, season, or catalogue changes | Relevance degradation | Monitor slices and retrain under governance |
| Unmatured outcomes | Recent purchase cohorts show unusually few returns | Evaluation closes before returns arrive | Inflated net value | Define a maturity window and report lag |
| Sensitive explanation | Customer sees a revealing reason on a shared device | Copy exposes inferred context | Privacy harm and loss of trust | Review explanations and allow suppression |

A warning sign is not proof of a particular mechanism.
For example, a long session can be a legitimate research journey rather than a bot or merged identities.
Investigate with aggregate diagnostics and authorized data access before making a corrective change.

The most dangerous failures often cross ownership boundaries.
A data team may report excellent offline relevance while fulfilment sees stale stock and customer support sees complaints.
A release review should combine these signals before a model is promoted.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Event ordering reflects the actual usable sequence | Transitions require order | Clock skew, retries, concurrent tabs | Compare producer sequence and arrival distributions |
| Availability timestamps are reliable | Point-in-time replay depends on them | Backfill overwrites original arrival time | Audit immutable ingestion metadata |
| Session key is appropriate for the purpose | Events need a grouping rule | Shared browser, opt-out, resets | Review policy and measure boundary anomalies |
| Timeout captures useful context | Model needs a bounded prefix | Long research gaps or rapid intent switching | Sensitivity analysis across justified thresholds |
| Historical eligibility is reconstructible | Fair ranking requires feasible options | Missing inventory or catalogue snapshots | Measure snapshot coverage and label exclusions |
| Next view is a useful proxy | Provides a frequent training signal | Accidental clicks or misleading presentation | Compare with later outcomes and randomized impact |
| Training and serving features match | Offline scores must be deployable | Different transformations or freshness | Compare feature definitions and online distributions |
| Sufficient transition support exists | Counts need enough observations | Sparse items and rapid catalogue turnover | Report count distribution and use backoff |
| Past behaviour remains informative | Model transfers from train to test | Seasonality, promotions, changed interface | Rolling temporal tests and drift monitoring |
| Experiment assignment is stable | Randomization supports causal comparison | Cross-device switching or reassignment | Audit assignment consistency and contamination |
| No material unmeasured interference | One arm should not change another's outcomes | Shared inventory or social influence | Assess stock constraints and consider cluster designs |
| Contribution costs are correctly defined | Business calculation requires consistent units | Omitted returns, shipping, or service costs | Reconcile with finance and mature outcomes |
| Privacy controls apply to derivatives | Embeddings and logs may remain linkable | Downstream copies or undocumented access | Maintain lineage, access review, and retention controls |
| Observed segments are sufficient for diagnostics | Aggregate averages can hide harm | Missing or inappropriate demographic data | Use justified proxies carefully and report blind spots |

No model can infer true intent with certainty from clicks alone.
The same sequence may correspond to different people, tasks, or interface accidents.
The available data support a bounded decision, not a definitive personal characterization.

Offline metrics depend on the observed policy and label definition.
They do not establish counterfactual customer choices under a new policy.
Online experiments can estimate effects under their own population and conditions, but generalization to other markets, seasons, or supply conditions still requires judgment.

A positive point estimate is not a guarantee.
Sample uncertainty, operational failures, and fixed costs can reverse a business decision.
A privacy or safety restriction may prohibit an otherwise profitable feature.

## 12. Exercises

### Exercise 1 — Decision contract

**Time:** 15 minutes. **Type:** application.

A retailer wants to recommend accessories after a visitor opens a product page.
The page has three positions, a 150-millisecond recommendation deadline, and market-specific availability.

**Deliverable:** a one-page decision contract.

**Acceptance criteria:** name the trigger, decision, eligible set, customer objective, commercial metric, two guardrails, fallback, and at least four accountable owners.
Distinguish the recommendation score from the delivery promise.

### Exercise 2 — Point-in-time prefix

**Time:** 15 minutes. **Type:** application and critique.

A request occurs at 10:00:00 UTC.
Event A occurred at 09:58:00 and was available at 09:58:02.
Event B occurred at 09:59:50 but was available at 10:00:05.
Event C occurred at 10:00:03 and was available at 10:00:04.
Assume the request payload contains no independent copy of B.

**Deliverable:** list the usable events and explain why the others are excluded.

**Acceptance criteria:** apply both event time and availability time; do not use a later arrival to reconstruct a feature that was unavailable online.

### Exercise 3 — Transition probabilities

**Time:** 15 minutes. **Type:** calculation.

Training sessions are `A → B → C`, `A → B → D`, and `A → C → D`.
Use adjacent transitions and no smoothing.

**Deliverable:** counts and probabilities for all outgoing transitions from A and B.

**Acceptance criteria:** show denominators in transition units and explain the undefined case for an unseen origin.

### Exercise 4 — Ranking metrics

**Time:** 15 minutes. **Type:** calculation.

For four decisions, observed target ranks are 1, 2, absent, and 4.
Treat an absent target as a miss and reciprocal rank zero.

**Deliverable:** Hit@1, Hit@2, Hit@4, and MRR.

**Acceptance criteria:** use all four decisions in every denominator and show the reciprocal-rank arithmetic.

### Exercise 5 — Model choice

**Time:** 20 minutes. **Type:** critique and design.

A team proposes a Transformer trained on six months of product views.
Thirty percent of test items are new, the event pipeline has substantial late arrival, and the current baseline is global popularity.

**Deliverable:** a prioritized model and data plan, including at least two alternative baselines.

**Acceptance criteria:** address temporal reconstruction, cold-start coverage, eligibility, operational cost, and a condition under which a more complex model is justified.

### Exercise 6 — Temporal evaluation

**Time:** 25 minutes. **Type:** design.

You have one year of logs, delayed purchase labels, and weekly catalogue updates.
The proposed system predicts a purchase within seven days of a product-page decision.

**Deliverable:** a train/validation/test protocol with an example calendar and a diagram or numbered timeline.

**Acceptance criteria:** use global chronological splits, a mature-label rule, point-in-time catalogue and feature joins, a common candidate policy, and at least three reported denominators or slices.
Explain why an offline win is not causal evidence.

### Exercise 7 — Privacy and experience review

**Time:** 20 minutes. **Type:** critique.

A proposal stores raw search text and a cross-device session identifier for 18 months to improve a product carousel.
The interface displays “recommended because of your recent searches.”

**Deliverable:** a risk review and a recommended approve, revise, or reject decision.

**Acceptance criteria:** discuss purpose and necessity, potentially sensitive inference, shared-device display, jurisdiction-specific legal review, retention, deletion and access, and a less intrusive alternative.
Do not assert that a particular consent choice is universally sufficient.

### Exercise 8 — Commercial release decision

**Time:** 20 minutes. **Type:** calculation and decision.

A randomized test has 50,000 eligible sessions assigned to treatment.
The estimated incremental contribution before incremental system cost is USD 0.012 per eligible session.
Incremental system cost is USD 180 for those sessions.
The predeclared latency guardrail is breached, and the uncertainty interval for net value includes a material loss.

**Deliverable:** calculate estimated net contribution and recommend release, hold, or revise.

**Acceptance criteria:** show units and arithmetic; distinguish the point estimate from uncertainty; make the guardrail and accountable owner explicit.

## 13. Self-check answers

### Exercise 1

A strong contract defines the trigger as a product-page request after the current item is known.
The decision is up to three compatible and eligible accessories or no carousel.
Eligible means purchasable in the customer's market and consistent with current fulfilment and policy constraints.

An example primary metric is net contribution per randomized eligible session over a fixed, mature observation window.
Guardrails can include page latency and the rate of clicks leading to unavailable items.
The approved fallback might be a cached eligible content list or suppression if eligibility cannot be established.

The product owner owns the surface; the supply owner owns availability; the data owner owns event quality; the privacy owner reviews the processing; and the experiment owner owns measurement.
The delivery promise comes from the fulfilment system, not the ranker's score.
Other defensible contracts receive credit if their rules and ownership are explicit.

### Exercise 2

Only A is usable from the event stream at 10:00:00 UTC.
A both occurred and became available before the request.
B occurred before the request but arrived at 10:00:05, so the online system could not use it at 10:00:00.
C occurred after the request and is also unavailable at that instant.

If B were independently included in the request payload, the answer could change, but that is explicitly excluded here.
A warehouse replay that joins B solely by event time would leak information into this decision.

### Exercise 3

From A, the counts are A → B: 2 and A → C: 1.
There are 3 outgoing A transitions.
Thus $P(B\mid A)=2/3$ and $P(C\mid A)=1/3$.

From B, the counts are B → C: 1 and B → D: 1.
There are 2 outgoing B transitions.
Thus $P(C\mid B)=1/2$ and $P(D\mid B)=1/2$.
An unseen origin has denominator zero, so the unsmoothed formula is undefined and a fallback is required.

### Exercise 4

Hit@1 is $1/4=0.25$.
Hit@2 is $2/4=0.50$.
Hit@4 is $3/4=0.75$.

MRR is $(1+1/2+0+1/4)/4=1.75/4=0.4375$.
The absent target remains in the denominator because it is one of the four evaluated decisions.

### Exercise 5

A strong answer first repairs or explicitly models event availability and defines the historical eligible catalogue.
It adds market-aware popularity, current-item content similarity, and a last-item transition rule as comparable baselines.
Content features and a safe fallback help cover new items, but their historical availability must also be reconstructed.

The Transformer is justified only if it adds meaningful, robust value over those baselines under the same temporal and eligibility protocol and fits the latency, cost, and privacy constraints.
A weak answer chooses the architecture solely because it is more expressive or compares it only against global popularity.

### Exercise 6

One valid calendar trains on decisions through September, subject to labels having matured by the training extraction date.
A later October validation period tunes choices, and a November test period remains untouched until the policy is frozen.
For a seven-day purchase window, a decision's label must be fully observable before it enters a training or evaluation dataset; allow additional delay if purchase events arrive late.

At each decision, join the catalogue and features as they were available then, not as corrected today.
Use the same historical candidate policy and report eligible decisions, missing snapshots, target-outside-catalogue rate, and performance by new versus established items.
A rolling series of such cutoffs gives stronger evidence about time variation.

The logged policy affected what customers saw and clicked, so an offline result predicts agreement with historical observations rather than the causal effect of a new policy.
A randomized online comparison is needed for an incremental-effect estimate when feasible.

### Exercise 7

A strong answer recommends revision or rejection until the team demonstrates necessity and completes the applicable legal review.
Raw text and cross-device linkage can expose sensitive information; 18 months is a retention proposal, not a justified requirement merely because storage is cheap.
An explanation based on recent search can reveal behaviour to someone viewing a shared device.

Consider using a transient, coarser category or recent product context instead of raw search text, and avoid cross-device linkage unless it is necessary and authorized.
Specify access, deletion, retention for derived data, notice and applicable rights, and a less revealing explanation or no explanation.
The correct legal basis and any consent requirement depend on the actual jurisdiction and processing.

### Exercise 8

Estimated incremental contribution before system cost is:

$$50{,}000\ \text{sessions}\times\text{USD }0.012/\text{session}=\text{USD }600.$$

Subtracting USD 180 gives an estimated net contribution of USD 420 for the 50,000 treatment sessions.
This is a point estimate, not a guaranteed gain; the uncertainty interval includes a material loss.
The predeclared latency guardrail is breached, so hold or revise rather than release under the stated rules.
The product owner and experiment owner should document the decision, while the engineering owner addresses latency and the relevant governance owners approve any new test.

## 14. Interview questions

### 1. What is a session model trying to capture?

A strong answer describes immediate context from an ordered prefix and distinguishes it from a durable profile.
A weak answer claims the model knows a customer's true intent or identity.

### 2. How would you define a session?

A strong answer proposes an operational boundary, a gap rule, tie handling, and sensitivity checks.
A weak answer treats a 30-minute timeout as universal or equates it with deletion.

### 3. What is the difference between event time and availability time?

A strong answer explains late arrival and reconstructs what serving knew at the decision.
A weak answer assumes an earlier event timestamp proves the feature was available online.

### 4. Why compare a Transformer with a transition rule?

A strong answer discusses incremental value, sparse data, latency, interpretability, and a common candidate universe.
A weak answer assumes architecture complexity guarantees business improvement.

### 5. How can a model accidentally see the future?

A strong answer names final session length, future catalogue corrections, overlapping global time splits, and late-arriving events.
A weak answer considers only whether the row's label column was removed.

### 6. Why is next-click accuracy not the same as causal impact?

A strong answer describes prior exposure, position, selection, and randomized comparison.
A weak answer treats every observed click as a response caused by the candidate policy.

### 7. What should happen when a predicted item is unavailable?

A strong answer applies authoritative eligibility rules, uses a safe alternative or suppresses the slate, and records the outcome.
A weak answer shows the item anyway because it had the highest score.

### 8. How do you evaluate retrieval and ranking together?

A strong answer reports retrieval recall and end-to-end metrics, counting targets not retrieved as misses in the latter.
A weak answer evaluates only the ranker's ordering of candidates that were already retrieved.

### 9. How would you handle a new item or a one-event session?

A strong answer proposes content or popularity backoff, defines eligibility, and reports cold-start coverage.
A weak answer silently omits the request from evaluation.

### 10. Is a pseudonymous session ID anonymous?

A strong answer explains linkability and contextual identifiability and refers to applicable legal review.
A weak answer asserts that removing a name automatically removes privacy obligations.

### 11. How would you decide whether to launch?

A strong answer combines incremental contribution, uncertainty, customer guardrails, policy approval, operational cost, and rollback readiness.
A weak answer launches on a higher offline hit rate alone.

### 12. Who owns a session recommendation?

A strong answer assigns distinct accountability for product, data, supply, privacy, experimentation, and operations.
A weak answer makes the model developer responsible for every decision without authority over dependent systems.

## 15. Key terms

- **Availability time:** when a feature or event became usable; see [point-in-time prefixes](#step-4--build-point-in-time-prefixes).
- **Backoff:** simpler model used when the preferred estimate is sparse; see [baselines](#step-6--establish-non-sequential-and-simple-sequential-baselines).
- **Candidate:** item considered for display; see [candidate generation](#step-9--make-candidate-generation-and-constraints-explicit).
- **Causal mask:** prevents future positions from entering an earlier prediction; see [sequential representation](#step-7--select-a-sequential-representation).
- **Contribution:** revenue less specified variable costs; see [business calculation](#larger-illustrative-business-calculation).
- **Decision time:** instant a slate must be chosen; see [three clocks](#three-clocks-matter).
- **Dwell time:** approximate elapsed interaction time, not necessarily attention; see [feature families](#step-5--audit-feature-families).
- **Eligibility:** hard conditions an item must satisfy; see [constraints](#step-9--make-candidate-generation-and-constraints-explicit).
- **Event time:** when the interaction occurred; see [three clocks](#three-clocks-matter).
- **Exposure bias:** feedback depends on what was shown; see [industry evidence](#public-research-exposure-and-evaluation).
- **Fallback:** approved alternative when a model cannot serve; see [operating procedure](#step-12--operate-review-and-retire).
- **Global temporal split:** common calendar cutoff for all entities; see [temporal evaluation](#step-10--evaluate-in-time).
- **Hit@k:** fraction of targets within the first $k$ positions; see [test decisions](#test-decisions).
- **Impression:** record of an item actually displayed; see [event contract](#event-and-catalogue-contract).
- **Incrementality:** causal difference between assigned policies; see [online experiment](#step-11--design-an-online-experiment).
- **Inactivity timeout:** gap rule for session boundaries; see [session boundary](#step-2--define-the-session-boundary).
- **Label window:** interval in which an outcome counts; see [prediction target](#step-3--define-the-prediction-target).
- **Leakage:** use of information unavailable at a decision; see [failure modes](#10-failure-modes-and-red-flags).
- **MRR:** mean reciprocal rank across decisions; see [ranking metrics](#test-decisions).
- **Point-in-time join:** historical join using data then available; see [prefix construction](#step-4--build-point-in-time-prefixes).
- **Prefix:** sequence available before a decision; see [core model](#3-the-core-mental-model).
- **Pseudonymous identifier:** potentially linkable replacement for direct identity; see [privacy](#privacy-and-lawful-processing).
- **Randomization unit:** entity assigned to treatment or control; see [online experiment](#step-11--design-an-online-experiment).
- **Retrieval:** generation of a candidate set; see [candidate generation](#step-9--make-candidate-generation-and-constraints-explicit).
- **Session:** operational grouping of events; see [session boundary](#step-2--define-the-session-boundary).
- **Slate:** ordered set of displayed items; see [decision contract](#step-1--specify-the-decision-contract).
- **Transition count:** observed adjacent pair frequency; see [worked example](#training-data).
- **Transformer:** attention-based sequence architecture; see [sequential representation](#step-7--select-a-sequential-representation).

## 16. Sources

1. Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk, “Session-based Recommendations with Recurrent Neural Networks,” ICLR 2016. [Paper](https://arxiv.org/abs/1511.06939).
   Research source for recurrent session modelling; benchmark results are context-specific.
2. Wang-Cheng Kang and Julian McAuley, “Self-Attentive Sequential Recommendation,” IEEE International Conference on Data Mining, 2018. [Paper](https://arxiv.org/abs/1808.09781).
   Research source for causal self-attention in sequential recommendation.
3. Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan, “Session-based Recommendation with Graph Neural Networks,” AAAI Conference on Artificial Intelligence, 2019. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/3804).
   Research source for graph-based session representations.
4. Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims, “Recommendations as Treatments: Debiasing Learning and Evaluation,” International Conference on Machine Learning, 2016. [Paper](https://proceedings.mlr.press/v48/schnabel16.html).
   Research source for selection bias and counterfactual recommendation evaluation.
5. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5, 6, and 25.
   Primary legal text; applicability and current interpretation require legal review.
6. European Union, [Directive 2002/58/EC on privacy and electronic communications](https://eur-lex.europa.eu/eli/dir/2002/58/oj), as amended and implemented nationally.
   Primary framework for relevant electronic communications and device-access rules; national implementation matters.
7. California Privacy Protection Agency, [California privacy law and regulations](https://cppa.ca.gov/regulations/).
   Official regulatory resource; check current text and applicability for the actual deployment.
8. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin).
   Official identification guidance; product identity does not establish availability or compatibility.

The research sources describe methods and reported experiments, not audited outcomes for the fictional retailer.
The regulatory references are educational and are not a substitute for jurisdiction-specific advice.

## 17. Links

- [Module 24 topic index](_index.md).
- [Recommendation tasks](01_recommendation-tasks.md): choose the decision and recommendation task.
- [Popularity and content baselines](02_popularity-content-baselines.md): establish simpler alternatives.
- [Module 24 survey and lab sections](../../../course/24_recommendation_systems_personalization.md): broader module context and hands-on exercises.
- [Module expansion specification](../../../course/MODULE_EXPANSION_SPEC.md): teaching-unit requirements.
