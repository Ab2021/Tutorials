---
title: "Recommendation architectures"
topic: "Recommender architectures: matrix factorisation, two-tower retrieval, sequence models, and cascades"
collection: "14_recommendation_search_ads_and_ranking"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Recommendation architectures

Every number here is constructed for teaching. Arithmetic is recomputed and re-checked
against an independent route, but no system was profiled and no organisation's result is
claimed. Inputs that stand in for real catalogue or traffic values are marked `ILLUSTRATIVE`
where they first enter an equation.

## 1. File Header

**Purpose.** Explain, at the level of mechanism and cost, why recommender systems are built
as a cascade of architecturally different models, and how to choose the model for each stage
without confusing the stage's job with the other stages' jobs.

**Decision this sheet supports.** Given a catalogue size, an interaction matrix, a latency
budget, and a cold-start population, choose the retrieval architecture, the ranking
architecture, and the training objective, and state which failure each choice is exposed to.

**Audience.** Engineers moving from a prototype recommender to a production one, and analysts
who must explain why offline metrics from a factorisation model and an online system diverge.

**Prerequisites.** Dot products and norms, the idea of a loss function and a gradient, matrix
shapes, and basic probability. Familiarity with logistic regression and with a softmax is
assumed; familiarity with transformers is helpful but only needed for section 4.3.

**Outcomes.** Derive the factorisation model from a likelihood and compute its gradients by
hand; compute the storage and compute of a two-tower retriever for a stated catalogue; explain
why sampled softmax changes what the model learns; identify the six architecture-level failure
modes and the check that separates them; and size a training job in MACs rather than in epochs.

**Notation convention.** Fenced `text` blocks carry all display mathematics in plain ASCII.
Symbols are defined at first use as meaning, unit, and shape. A vector is bold-free: `u` is a
vector of length d, `u_i` is its i-th component, `u·v` is a dot product. Matrices are capital
letters. Interaction counts are integers; rates are dimensionless in [0,1]; money is not used.

**Evidence boundary.** Sections 4 and 5 are standard results, restated with derivations and
recomputed arithmetic. Section 9 counts multiply-accumulate operations and bytes; those counts
are exact given the stated shapes, while the wall-clock throughput figures are illustrative
hardware assumptions. No accuracy improvement is claimed for any architecture.

**Exclusions.** This sheet does not cover feature engineering for ranking in detail, does not
cover exploration and bandit policies, and does not cover ads auction mechanics. It also does
not attempt a survey of published models; the point is the mechanism and the trade-off, not
a bibliography.

**How to use it.** Section 2 is the map of stages. Section 4 is the mathematics you must be
able to reproduce on a whiteboard. Section 7 is the selection table to consult during design
review. Section 8 is the checklist when a model behaves worse in production than offline.

## 2. Concept Map / Mental Model

A recommender answers a question the user did not ask, from a matrix that is 99.95% empty.
Everything architectural follows from those two facts: the emptiness forces the model to
generalise across users and items, and the unasked question forces a proxy objective that is
only loosely connected to what the user wanted.

```text
interaction matrix R (users x items), density ~5e-4
   |
   +--> retrieval: cheap model, whole catalogue, high recall
   |        matrix factorisation / two-tower / co-visitation / ANN
   v
candidate set (hundreds to low thousands)
   |
   +--> ranking: expensive model, candidate set only, high precision
   |        GBDT or deep model on rich features, calibrated
   v
ordered list (tens)
   |
   +--> policy layer: diversity, business rules, freshness, dedup
   v
presented slate
```

The cascade exists because cost per item differs by orders of magnitude between stages. A
dot product between two 64-dimensional vectors costs 64 multiply-accumulates and can be done
for every item in the catalogue. A gradient-boosted ranker with 300 features costs hundreds
of tree traversals per item and can only be run on a shortlist. The architecture is therefore
a budget allocation: spend the cheap model on breadth and the expensive model on depth.

```text
stage       items scored   cost/item            total cost        (ILLUSTRATIVE)
retrieval   100,000       64 MAC               6.4e6 MAC
ranking         500       1,800 comparisons    9.0e5 comparisons
deep ranker     500       1.3e5 MAC           6.6e7 MAC
policy           50       1,000 ops            5.0e4 ops
```

Three invariants define a healthy cascade, and each one fails in a way that is invisible from
the stage's own metrics.

**Invariant 1 — retrieval must not be judged by precision.** A retriever optimised for the
precision of its top 10 is a ranker, and it will be bad at both jobs. The retriever's job is to
put the eventual winners somewhere in its output, which is a recall question at a fixed budget.

**Invariant 2 — ranking cannot recover what retrieval never sent.** The ranker's score function
is defined only on items that arrive. This is why a two-tower model that improves ranking
metrics can leave the product unchanged: the same items were already arriving.

**Invariant 3 — the training distribution must resemble the serving distribution.** A model
trained on all logged interactions learns from items that were shown because an older model
ranked them highly. That selection is a feedback loop, and its statistical signature is a
model that is excellent on popular items and blind on the tail.

A second mental model concerns what the latent factors mean. In a factorisation, the dot
product `u·v` is a learned similarity in a space with no natural units. Factor 3 is not
"price sensitivity"; it is whatever direction reduced the loss. Interpretations are
post-hoc and must be validated externally before anyone writes them into a design document.

A third model is the difference between implicit and explicit feedback. An explicit rating is
a measurement with a scale; an implicit click is a behaviour with a position bias and a
censoring problem (a non-click is not a dislike). The two require different losses: regression
for the first, ranking losses for the second.

Finally, picture the sequence view. A recommender that ignores order treats a user as a bag of
items; a sequence model treats a user as a prefix of a session and predicts the next item. The
sequence view buys short-term relevance at the cost of stability, and it is the reason two
architectures with identical offline metrics can behave completely differently in a session.

## 3. Foundational Definitions

**Interaction matrix (R).** A matrix of shape users x items holding an implicit count or an
explicit rating. Stored sparse: in coordinate form as (user, item, value) triples. Density is
non-zeros divided by the number of cells, and it is the quantity that makes the problem hard.

**Implicit feedback.** A signal derived from behaviour without an explicit judgement: click,
add-to-cart, dwell above a threshold, purchase. Non-observation is ambiguous, so implicit data
supports "this happened" but not "this was disliked".

**Explicit feedback.** A rating on a stated scale. It supports regression and it carries its
own biases: raters self-select, and the scale's middle is used inconsistently across users.

**Latent factor.** One coordinate of a learned embedding. Factors are unidentifiable up to
rotation and sign flips: if Q is any orthogonal matrix, replacing every user vector u by Qu
and every item vector v by Qv leaves all dot products unchanged. Consequence: comparing factor
values across two training runs is meaningless without alignment.

**Embedding dimension (d).** The number of latent factors, a hyperparameter. It bounds the
model's capacity: a rank-d factorisation of a U x I matrix has at most d(U + I) free parameters
and cannot represent structure of higher rank.

**Matrix factorisation (MF).** A model in which the score for a (user, item) pair is the dot
product of a user vector and an item vector. Trained by regression on ratings, by
classification on implicit counts, or by a pairwise ranking loss.

**Bias terms.** A global mean plus a user bias plus an item bias, added to the dot product. In
practice these capture most of the signal in explicit data, and a model without them spends
capacity re-learning popularity.

**BPR (Bayesian personalised ranking).** A pairwise loss that asks the model to score an
observed item above an unobserved one, using the logistic function of the margin. It optimises
an ordering, not a rating.

**Sampled softmax.** A training shortcut in which the normalising constant is estimated over
one positive plus a small number of sampled negatives instead of the full catalogue. It makes
large-catalogue training tractable and it changes the objective's asymptote.

**Two-tower architecture.** Separate encoders for user features and item features producing
vectors in a shared space, so that scoring becomes a dot product and retrieval becomes an
approximate nearest-neighbour search. The defining constraint is that the two towers never see
each other's inputs.

**Sequence model.** An encoder over the user's recent item sequence producing a state that is
used to score the next item. Attention-based variants weight the history by relevance to the
current context.

**Candidate generation budget (B).** The number of items the retriever emits per request. It is
the primary recall/latency dial and must be written next to every recall number.

**Negative sampling.** The process of choosing unobserved items to act as negatives. Uniform
sampling over the catalogue, popularity-weighted sampling, and in-batch sampling produce
different learned geometries from the same data.

**Cold start.** The regime where a user or item has too few interactions to estimate its
vector. Item cold start is typically served by content features; user cold start by
demographics, context, or popularity.

**Position bias.** The tendency of users to examine items near the top of a list regardless of
relevance. It contaminates training data whenever the model learns from logged clicks.

**Feature store.** The serving system that returns per-entity features at request time. Its
consistency with the training pipeline is the most common source of silent accuracy loss.

**Calibration.** Agreement between predicted probabilities and observed frequencies. Distinct
from discrimination (AUC): a model can rank perfectly and be off by a factor of 1.67 in
predicted probability, which matters the moment the score is multiplied by a value.

**Slate.** The ordered set actually presented, as distinct from the scored candidate list. Slate
constraints (diversity, dedup, business rules) are applied after ranking and can change the
effective model.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

Start from the probabilistic statement that makes factorisation a model rather than a heuristic.
Assume an explicit rating r_ui is a noisy observation of a latent preference, with Gaussian
noise around the dot product and biases.

```text
FORMULA (base case, biased matrix factorisation, squared loss)
  r_hat_ui = mu + b_u + b_i + u·v
  L = sum over observed (u,i) of  (r_ui - r_hat_ui)^2  +  lambda * (||u||^2 + ||v||^2 + b_u^2 + b_i^2)
WHERE
  mu is the global mean rating (same units as r)
  b_u, b_i are scalar biases (same units as r)
  u, v are vectors of length d (unitless)
  lambda >= 0 is the regularisation strength (unitless)
CONSTRAINT
  the sum runs over observed cells only; unobserved cells are not zeros
  lambda > 0 is required: with lambda = 0 the model can fit any observed entry exactly
NUMERICAL EXAMPLE (d = 2, one pair, mu = 3.5, b_u = 0.2, b_i = -0.1)
  u = (0.8, 0.1), v = (0.7, 0.2) -> u·v = 0.56 + 0.02 = 0.58
  r_hat = 3.5 + 0.2 - 0.1 + 0.58 = 4.18
  if the observed rating is 4, the squared error is (4 - 4.18)^2 = 0.0324
INTERPRETATION
  biases supply the level and the dot product supplies the deviation from it
```

The Gaussian assumption is what turns squared error into a likelihood, and stating it makes
the failure obvious: explicit ratings are bounded, often 1 to 5, so a Gaussian is wrong at the
edges. A bounded likelihood (ordered logit, or a clipped Gaussian) is the defensible choice,
and the practical symptom of getting this wrong is predicted values outside the scale.

For implicit feedback, the observation model changes completely. Treat the interaction count
as a draw from a Poisson, whose rate is an exponential of the score.

```text
FORMULA (implicit MF, Poisson likelihood)
  rate_ui = exp(mu + b_u + b_i + u·v)
  L = sum over observed (u,i) of  [ rate_ui - c_ui * (mu + b_u + b_i + u·v) ]  +  regularisation
  where c_ui is the observed count
WHERE
  the first term is the Poisson negative log-likelihood up to a constant
  rate has units of counts per unit time
CONSTRAINT
  exp() overflows for large scores; clamp the linear predictor before exponentiating
  c_ui = 0 cells still contribute rate_ui, so the loss is dense even when R is sparse
NUMERICAL EXAMPLE (score 0.58 with mu + biases = 0)
  rate = exp(0.58) = 1.7862 expected interactions
  with c_ui = 2 the per-cell loss is 1.7862 - 2 * 0.58 = 0.6262
INTERPRETATION
  the model must explain zeros too, which is why implicit factorisation is more stable
```

The pairwise objective abandons the value of the interaction entirely and keeps only the
ordering. It is the right loss when the product question is "which item should be first",
which is the usual recommendation question.

```text
FORMULA (BPR loss and gradient)
  s = u·v_pos - u·v_neg
  L = -log( sigmoid(s) ) = log( 1 + exp(-s) )
  dL/du = -(1 - sigmoid(s)) * (v_pos - v_neg)
  dL/dv_pos = -(1 - sigmoid(s)) * u
  dL/dv_neg = +(1 - sigmoid(s)) * u
WHERE
  v_pos is the vector of an observed item, v_neg of a sampled unobserved item
  sigmoid(s) = 1 / (1 + exp(-s))
CONSTRAINT
  the loss saturates for large margins, so a well-separated pair contributes no gradient
  sampling of negatives controls the effective decision boundary
NUMERICAL EXAMPLE (u = (0.8, 0.1), v_pos = (0.7, 0.2), v_neg = (0.5, -0.1))
  s = 0.58 - (0.40 - 0.01) = 0.58 - 0.39 = 0.19
  sigmoid(0.19) = 0.5474; loss = -log(0.5474) = 0.6027
  dL/du = -(1 - 0.5474) * (0.2, 0.3) = (-0.0905, -0.1358)
INTERPRETATION
  the gradient moves u away from the negative item's direction, scaled by how unsure
  the model is; a confidently ranked pair contributes nothing
```

### 4.2 General case

The general case has three additions: the model must be trained by alternating closed-form
updates or by stochastic gradients, negatives must be sampled from a distribution, and the
score must be normalised so it can be interpreted as a probability of engagement.

Alternating least squares is the closed form that makes the structure visible. Fix every item
vector, then the optimal user vector is the solution of a d x d linear system.

```text
FORMULA (ALS user update)
  u = (V_obs^T V_obs + lambda I)^-1 V_obs^T r_obs
WHERE
  V_obs is the (n_obs x d) matrix of item vectors the user interacted with
  r_obs is the n_obs vector of observed values
  I is the d x d identity; the inverse exists for lambda > 0
CONSTRAINT
  V_obs^T V_obs is d x d regardless of how many items the user touched
  with lambda = 0 the system is singular when n_obs < d or when items are collinear
NUMERICAL EXAMPLE (d = 2, lambda = 0.5, items (0.7,0.2), (0.1,0.8), (-0.2,0.3), r = (1,0,1))
  V^T V without ridge = [[0.54, 0.16],[0.16, 0.77]]; adding 0.5 to the diagonal gives
  [[1.04, 0.16],[0.16, 1.27]]
  V^T r = (0.7 - 0.2, 0.2 + 0.3) = (0.50, 0.50)
  determinant = 1.04*1.27 - 0.16*0.16 = 1.3208 - 0.0256 = 1.2952
  u_1 = (0.50*1.27 - 0.16*0.50) / 1.2952 = (0.635 - 0.08) / 1.2952 = 0.4285
  u_2 = (1.04*0.50 - 0.50*0.16) / 1.2952 = (0.52 - 0.08) / 1.2952 = 0.3397
INTERPRETATION
  the ridge term keeps the inverse finite and shrinks the solution toward zero, which is
  exactly the behaviour wanted for users with three interactions
```

Check the fit for that user: scores are 0.4285*0.7 + 0.3397*0.2 = 0.29995 + 0.06794 = 0.3679;
0.4285*0.1 + 0.3397*0.8 = 0.04285 + 0.27176 = 0.3146; 0.4285*(-0.2) + 0.3397*0.3 = -0.0857 +
0.10191 = 0.0162. The observed items score 0.368 and 0.016 against targets 1 and 1, and the
unobserved one scores 0.315. The fit is poor in absolute terms because lambda = 0.5 is large
relative to a three-item fit — which is the correct lesson: regularisation is not free, and its
scale must be tuned against the interaction count per user.

The probabilistic generalisation replaces the dot product with a softmax over the catalogue, and
this is where sampled softmax enters.

```text
FORMULA (catalogue softmax and its sampled estimate)
  p(i | u) = exp(u·v_i) / sum over j in catalogue of exp(u·v_j)
  L = -log p(i_pos | u) = -u·v_pos + log( sum over j of exp(u·v_j) )
  sampled:  L_s = -u·v_pos + log( sum over j in S of exp(u·v_j) )  -  log(|S|)
  where S = {i_pos} union a set of sampled negatives, |S| = 1 + n_neg
WHERE
  the -log(|S|) term corrects the scale for the reduced denominator
  v_j for j in S are shared with the other examples in the batch (in-batch negative)
CONSTRAINT
  sampled softmax is biased for the true likelihood but consistent for ranking
  log(1e5) = 11.513 versus log(101) = 4.615: the sampled objective sees a far smaller spread
NUMERICAL EXAMPLE (scores 0.580, 0.160, 0.390, -0.130 over four items)
  exponentials: 1.7862, 1.1735, 1.4770, 0.8781; sum Z = 5.3146
  p(item 1) = 1.7862 / 5.3146 = 0.3361; NLL = -log(0.3361) = 1.0905
  with one positive and 100 uniform negatives, the denominator is dominated by the samples
  and the NLL falls to roughly log(101) = 4.615 for a random model
INTERPRETATION
  the sampled objective is a different estimator, not a faster exact one; its asymptote is
  the ranking quality against the sampling distribution
```

Two-tower retrieval is the architectural consequence of wanting an exact top-k over a large
catalogue in bounded time. The score is a dot product, so the search is a maximum inner product
search, and the cost moves from scoring to indexing.

```text
FORMULA (two-tower inference cost)
  MAC_full_scan = |catalogue| * d
  MAC_user_tower = sum over layers of (in * out)
  storage_bytes = (|users| + |items|) * d * 4
WHERE
  d = 64, catalogue = 100,000 items (ILLUSTRATIVE), float32 = 4 bytes
CONSTRAINT
  a full scan is exact and is the reference for ANN recall
  the towers cannot use cross features, by construction; anything that needs both sides
  belongs in the ranking stage
NUMERICAL EXAMPLE
  MAC_full_scan = 100,000 * 64 = 6.4e6 multiply-accumulates per request
  storage = (1,000,000 users + 100,000 items) * 64 * 4 bytes = 2.816e8 bytes = 0.282 GB
  a 256 -> 64 user tower costs 256*64 = 16,384 MAC, negligible against the scan
INTERPRETATION
  the scan is affordable in raw arithmetic and expensive in memory bandwidth, which is the
  real reason ANN indices exist
```

### 4.3 Variants

**Bias-only baseline.** Set d = 0 and keep mu, b_u, b_i. This is the model every other model
must beat, and in explicit-rating datasets it captures a large share of the achievable error
reduction. Skipping this baseline is how teams report a 4% RMSE improvement that a two-parameter
model already achieves.

**SVD++ / item-item neighbourhoods.** Add a term for the set of items the user interacted with,
which captures "users who liked these also liked that" without increasing d. The cost is one
extra embedding table and an aggregation over the user's history; the benefit is better
performance for users with short histories, which is where factorisation is weakest.

**Two-tower with in-batch negatives.** During training, every other positive in the batch serves
as a negative for the current user. The effective number of negatives per step is the batch size
minus one, which is far more than the 4 to 10 explicit samples a BPR loop would use. The known
distortion is that popular items appear as negatives more often, so the model learns a
popularity-corrected geometry. A log-Q correction (subtract the log sampling frequency) removes
most of the bias and costs one vector subtraction.

**Sequential model with attention.** Encode the user's last L items and score the next item from
the encoded state. With L = 50 and d = 64, one attention layer costs about L^2 * d = 50*50*64 =
160,000 multiply-accumulates, which is fine for ranking a shortlist and impossible for scanning
a catalogue. This is precisely why sequence models live in the ranking stage, not retrieval.

```text
FORMULA (sequence ranker cost)
  MAC_attention = L^2 * d   per layer
  MAC_projection = L * d * d per layer
WHERE
  L is the sequence length, d the model width
CONSTRAINT
  L^2 growth makes long histories expensive; windowing or summarising is the usual fix
NUMERICAL EXAMPLE (L = 50, d = 64)
  attention 50*50*64 = 160,000; projections 2 * 50*64*64 = 409,600
  total per layer ~5.7e5 MAC; three layers ~1.7e6 MAC per candidate scored
INTERPRETATION
  at 500 candidates per request that is 8.5e8 MAC per request, which is a ranking-stage
  budget, not a retrieval-stage one
```

**GBDT ranker on cross features.** Gradient-boosted trees on dozens to hundreds of engineered
features (user history aggregates, item statistics, cross counts, context) remain competitive
for tabular ranking and are far cheaper to operate than deep models. They cannot generalise to
unseen categorical combinations, which is the reason deep models with embeddings win on large
catalogues; on small catalogues the tree model usually wins on both latency and maintenance.

**Hybrid retrieval.** Union of a factorisation-based ANN, a co-visitation graph, and a lexical
matcher. The architecture question is not which is best but how the budgets divide, since the
ranked list sees only the union. A retriever that contributes 5% of the final winning items
while consuming 30% of the budget is a candidate for removal, and the measurement is marginal
recall per millisecond.

### 4.4 Limits and counterexamples

**Counterexample 1 — factor rotation makes factors uninterpretable.** Two runs with identical
loss can produce factor vectors related by an orthogonal transform. Concretely, if u = (0.8, 0.1)
and v = (0.7, 0.2) with dot product 0.58, then applying the 90-degree rotation Q = [[0,-1],[1,0]]
gives Qu = (-0.1, 0.8) and Qv = (-0.2, 0.7), whose dot product is 0.02 + 0.56 = 0.58. Same
model, different coordinates, and any story about "factor 1 is price sensitivity" survives only
one run. Interpret factors only after aligning to an external variable and measuring correlation.

**Counterexample 2 — a rank-d model cannot represent a rank-(d+1) preference.** With U = 1,000
users, I = 1,000 items and d = 5, the model has at most 5*(1,000 + 1,000) = 10,000 free
parameters against 1e6 cells. If the true preference structure is a strict cycle among six items
(rock-paper-scissors style, which is rank 6 in the pairwise sense), no d = 5 model represents it.
The symptom is a loss floor that additional epochs cannot break; the fix is capacity, not
optimisation.

**Counterexample 3 — sampled softmax changes the learned geometry.** With one positive and four
negatives, the model is trained to separate five items. It will happily place two items that are
both true positives on opposite sides of the boundary, because their relative order never enters
the loss. The symptom is a model with excellent NLL against the sampled distribution and
mediocre full-catalogue recall; the check is to evaluate full-softmax recall and compare it with
the sampled NLL trend.

**Counterexample 4 — in-batch negatives create a popularity gradient.** If item A appears in 20%
of batches as a negative and item B in 2%, the model is pushed 10x harder to make A's score low
for users who did not interact with it. Items with high interaction counts therefore acquire
systematically smaller norm embeddings, and cosine similarity versus dot product then produce
different rankings. Check the correlation between item norm and item popularity; a strong
negative correlation is the signature.

**Counterexample 5 — a sequence model overfits the last click.** A model that conditions on the
most recent item can achieve a large next-item accuracy gain by learning "users who clicked a
phone click phone accessories". That is a real pattern and a fragile one: it degrades instantly
on users whose last click was accidental. The check is performance on sessions where the last
event was a bounce, segmented separately.

**Counterexample 6 — the ridge term hides a degenerate fit.** With lambda = 0.5 and only three
observed items for a user in d = 2, the ALS solution above fits targets of 1 with scores 0.368
and 0.016. The model is not broken; the regularisation dominates. Raising capacity or lowering
lambda changes predictions drastically, so the tuning of lambda must be evaluated per interaction
count bucket, not globally.

**Validity condition for this section.** Every result here assumes the training distribution
matches the serving distribution, the negative sampler is documented, and the evaluation uses
the same candidate budget as serving. Break any of the three and the numbers describe a
different system than the one deployed.

## 5. Worked Numerical Example

The example is a complete factorisation cycle over a 3-user, 4-item matrix with d = 2, on
implicit binary feedback. All inputs are ILLUSTRATIVE; every figure below is recomputed from
the stated vectors.

Users: u1 = (0.8, 0.1), u2 = (0.2, 0.9), u3 = (-0.3, 0.4).
Items: v1 = (0.7, 0.2), v2 = (0.1, 0.8), v3 = (0.5, -0.1), v4 = (-0.2, 0.3).

**Step 1 — score matrix.** Dot products, with no biases:

| | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| u1 | 0.580 | 0.160 | 0.390 | -0.130 |
| u2 | 0.320 | 0.740 | 0.010 | 0.230 |
| u3 | -0.130 | 0.290 | -0.190 | 0.180 |

Recheck u1's row: 0.8*0.7 + 0.1*0.2 = 0.56 + 0.02 = 0.58; 0.8*0.1 + 0.1*0.8 = 0.08 + 0.08 =
0.16; 0.8*0.5 + 0.1*(-0.1) = 0.40 - 0.01 = 0.39; 0.8*(-0.2) + 0.1*0.3 = -0.16 + 0.03 = -0.13.
All four confirm.

**Step 2 — softmax for u1.** Exponentials of 0.580, 0.160, 0.390, -0.130 are 1.7862, 1.1735,
1.4770, 0.8781 (using exp(0.58) = 1.78604, exp(0.16) = 1.17351, exp(0.39) = 1.47698, exp(-0.13)
= 0.87810). Z = 1.78604 + 1.17351 + 1.47698 + 0.87810 = 5.31463.

Probabilities: item 1 = 1.78604/5.31463 = 0.33606; item 2 = 0.22080; item 3 = 0.27791; item 4
= 0.16522. Sum check: 0.33606 + 0.22080 + 0.27791 + 0.16522 = 0.99999, confirming rounding only
in the fifth decimal.

**Step 3 — negative log-likelihood.** With item 1 as the observed positive, NLL = -log(0.33606)
= 1.09046. Compare with the uniform baseline log(4) = 1.38629. The model is better than chance
by 0.29583 nats on this pair — a small gain, consistent with the four-item toy scale.

**Step 4 — BPR on one pair.** Take u1 with positive item 1 and negative item 3. Margin s = 0.580
- 0.390 = 0.190. sigmoid(0.190) = 1/(1 + exp(-0.190)) = 1/(1 + 0.82696) = 0.54736. Loss =
-log(0.54736) = 0.60265. The gradient on u1 is -(1 - 0.54736) * (v1 - v3) = -0.45264 * (0.2, 0.3)
= (-0.09053, -0.13579).

**Step 5 — independent re-check of the gradient.** Compute the loss as log(1 + exp(-s)) directly:
exp(-0.190) = 0.82696, 1 + 0.82696 = 1.82696, log(1.82696) = 0.60265. Identical to step 4, as
required since -log(sigmoid(s)) = log(1 + exp(-s)) is an identity.

The gradient check uses the logistic derivative: dL/ds = -1/(1 + exp(s))... evaluate at s = 0.190
gives -1/(1 + 1.20925) = -0.45264, matching 1 - sigmoid(s) to five decimals. Then by the chain
rule ds/du = (v1 - v3) = (0.2, 0.3), so dL/du = -0.45264 * (0.2, 0.3). Both routes agree.

**Step 6 — one ALS user update.** For a user who interacted with items 1 and 3 (r = 1) and not
item 2 (r = 0), with lambda = 0.5, the closed form from section 4.2 gives u = (0.4285, 0.3397).
Compare with the hand-set u1 = (0.8, 0.1): the learned vector is smaller in the first coordinate
by 0.3715 and larger in the second by 0.2397. This is the ridge shrinking toward zero, and the
angle between the two vectors is large — 0.4285*0.8 + 0.3397*0.1 = 0.3428 + 0.0340 = 0.3768
against norms 0.8062 and 0.5468, so cosine = 0.3768/(0.8062*0.5468) = 0.8546. That is a
31.3-degree rotation from one regularisation setting, which is why hyperparameters must be fixed
before any interpretation of an embedding.

**Step 7 — sensitivity to lambda.** Halve lambda to 0.25 and the matrix becomes [[0.79, 0.16],
[0.16, 1.02]] with determinant 0.8058 - 0.0256 = 0.7802. Then u_1 = (0.50*1.02 - 0.16*0.50)/0.7802
= (0.51 - 0.08)/0.7802 = 0.5511 and u_2 = (0.79*0.50 - 0.50*0.16)/0.7802 = (0.395 - 0.08)/0.7802
= 0.4037. The vector grew by 0.1226 and 0.0640 in the two coordinates for a factor-of-two change
in lambda. Consequence: lambda must be tuned per interaction-count bucket, and any reported
embedding norm is a statement about lambda as much as about preference.

**Step 8 — cost of one epoch.** Suppose the real matrix has 5e7 non-zeros over 1e6 users and
1e5 items, so density is 5e7/(1e6 * 1e5) = 5.0e-4 and the average user has 50 interactions.
With 4 negatives per positive, training pairs per epoch = 5e7 * 5 = 2.5e8. Each pair costs d = 64
multiply-accumulates for the dot product plus the gradient, so roughly 64 MAC: total 2.5e8 * 64 =
1.6e10 MAC per epoch. At 2e9 MAC/s that is 8.0 seconds per epoch; at 5e10 MAC/s, 0.32 seconds.

**Step 9 — parameter count versus data.** The model has d(U + I) = 64*(1,100,000) = 7.04e7
parameters against 5e7 observed interactions, a ratio of 1.41 parameters per observation. That
ratio is the reason regularisation is not optional: the model can nearly memorise the training
matrix, and the only defence is lambda plus holdout evaluation. If d were 256, the ratio would be
5.63 and the memorisation risk would rise accordingly.

**Step 10 — two-tower retrieval on the same catalogue.** Full scan cost = 1e5 * 64 = 6.4e6 MAC
per request, storage 0.282 GB for users plus items. Serving 500 candidates out of 1e5 means the
retriever presents 0.50% of the catalogue, so every recall number must be read against that
budget: a recall@500 of 0.60 does not imply the retriever is bad, and a recall of 0.60 at a
budget of 5,000 would.

## 6. Visual / Geometric Intuition

The score matrix is a low-rank surface, and the geometry of that surface explains nearly every
architectural choice. Draw the 4 items in the 2-dimensional embedding plane from the worked
example: v1 = (0.7, 0.2), v2 = (0.1, 0.8), v3 = (0.5, -0.1), v4 = (-0.2, 0.3). Each user vector
is a direction, and the score is the projection of an item onto that direction.

```text
        v2 (0.1, 0.8)          <- u2 points here (score 0.740)
             *
             |      v4 (-0.2, 0.3)
             |        *
   ----------*--------*--------> factor 1
        v1 (0.7,0.2)  v3 (0.5,-0.1)
             *
        u1 direction -> mostly factor 1, so v1 scores highest for u1
```

The picture makes three things visible. First, a user's ranking is a single direction, so items
that are angularly close to each other always rank together; this is the geometric origin of the
filter bubble. Second, an item near the origin has a small norm and therefore a small score
against every user, which is how a weak item can be invisible without being wrong. Third, the
dot product conflates angle and magnitude: an item with norm 2 at 60 degrees can outscore an item
with norm 1 at 10 degrees. That is why cosine similarity and dot product rank differently, and
why a norm-popularity correlation (counterexample 4) silently reshuffles the list.

The two-tower space adds a hard constraint: the user representation is a function of user features
only, so it cannot rotate to meet a specific item. The consequence is geometric. A model with
cross features can carve a boundary between item A and item B for one particular user; a two-tower
model can only order items by their projection onto a user direction. The lost capacity is exactly
the interaction terms, and it is the price paid for making retrieval an inner-product search.

```text
cross-feature ranker:  score(u,i) = f(u, i)        arbitrary surface
two-tower:             score(u,i) = f_u(u) · f_i(i) a rank-d hyperplane family
```

The softmax geometry explains negative sampling. The loss pushes the positive's cosine up and
every sampled negative's down, so the learned space is shaped by the sampling distribution rather
than by the catalogue. If negatives are drawn uniformly, the model spreads items evenly; if they
are drawn by popularity, the model reserves more angular room around popular items, which improves
head accuracy and starves the tail.

```text
uniform negatives            popularity negatives
  o   o   o   o                 oooo  o
 o  *pos*  o   o               o **pos** o   o
  o   o   o   o                  o   o      (tail items crowd together)
```

The sequence-model geometry is different again: the user is not a point but a trajectory. A
session that moves from laptops to monitors to keyboards traces a path, and next-item prediction
extrapolates along it. This buys precision in-session and loses stability across sessions, and
the failure is visible as a model that recommends a second monitor to someone who just bought
one.

Finally, the cost geometry. Plot cost per item against number of items scored on log axes: the
retrieval stage is a cheap model over a huge set, the ranking stage an expensive model over a
small set, and the product is roughly constant. That constancy is the reason the cascade has the
shape it does; changing either axis without the other either blows the latency budget or
underuses it.

| stage | model cost per item (ILLUSTRATIVE) | items | total |
|---|---|---|---|
| retrieval | 64 MAC | 100,000 | 6.4e6 MAC |
| ranking (GBDT) | 1,800 comparisons | 500 | 9.0e5 comparisons |
| ranking (deep) | 1.3e5 MAC | 500 | 6.6e7 MAC |
| rerank / policy | 1e3 ops | 50 | 5.0e4 ops |

Read the column as a set of ceilings, not a ranking of importance. With a tree ranker, retrieval
costs 6.4e6 and ranking 9.0e5, so retrieval is seven times the arithmetic and the candidate budget
can be widened almost freely. Switch to the deep ranker and the ranking stage becomes 6.6e7, ten
times retrieval, so every extra candidate costs ten times what the tree model would have charged.
That single ratio is why the candidate budget and the ranker architecture must be chosen together:
raising 500 candidates to 5,000 moves the deep ranker from 6.6e7 to 6.6e8 MAC per request and
consumes the entire latency budget in section 9.

## 7. Comparison / Selection Table

Architecture selection is driven by four constraints that must be stated before any model
comparison: catalogue size, interaction count per user, latency budget, and how much of the
catalogue is cold. The table maps the constraint pattern to the architecture that fits it.

| Constraint pattern | Retrieval choice | Ranking choice | Objective | Watch for |
|---|---|---|---|---|
| Catalogue under 10k, dense interactions | item-item co-visitation | GBDT on cross features | pointwise logistic | feature staleness |
| Catalogue 10k-1M, long-tail demand | two-tower + ANN | deep model on embeddings | sampled softmax | popularity amplification |
| Catalogue over 1M, heavy cold start | two-tower with content features | two-stage (deep + GBDT) | sampled softmax + BPR mix | content-feature drift |
| Session-driven, short horizon | session co-visitation | sequence model with attention | next-item softmax | last-click overfit |
| Explicit ratings, small user base | bias + MF, ALS | bias + MF is enough | squared error | scale-boundary predictions |
| New product, no interactions | trending + content similarity | popularity-biased GBDT | pointwise | no personalisation signal at all |

The objective column deserves its own table, because the loss determines what the model will
be good at and the symptom of choosing wrong is measurable.

| Objective | Optimises | Data it needs | Symptom when wrong |
|---|---|---|---|
| squared error on ratings | predicted value | explicit ratings | good RMSE, bad ordering |
| pointwise logistic on clicks | probability of click | impressions + clicks | calibrated, position-biased |
| BPR pairwise | order within a pair | positives + sampled negatives | excellent ranking on sampled set only |
| sampled softmax | probability against sampler | positives + negatives | full-catalogue recall lower than NLL suggests |
| next-item softmax | next event | ordered sessions | in-session precision, session-boundary drop |
| listwise (e.g. softmax over slate) | slate quality | one positive per slate | needs full slate logs, rare in practice |

The bias-baseline table is the cheapest possible check and the most often skipped. Before any
factorisation, compute these three baselines and record them in the design document.

| Baseline | Score for (u,i) | What it establishes | Typical failure it exposes |
|---|---|---|---|
| global mean mu | constant | the floor for RMSE work | a model that does not beat mu adds nothing |
| mu + b_i | item popularity | share of signal that is popularity | personalisation claims smaller than expected |
| mu + b_u + b_i | popularity plus user leniency | the real MF baseline | a 4% gain that was really 4% of nothing |

A concrete measurement from the worked data shows the mechanism. Take twelve items with
consumption shares 0.40, 0.18, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005.
The Herfindahl index is the sum of squared shares = 0.2223, so the effective catalogue size is
1/0.2223 = 4.50 items out of 12. A recommender trained on this data sees the equivalent of four
and a half distinct items, which is why tail performance needs explicit intervention rather
than a better loss function.

```text
FORMULA (effective catalogue size)
  HHI = sum over i of share_i^2
  N_eff = 1 / HHI
WHERE
  share_i is item i's fraction of all interactions, summing to 1
  N_eff is in items and is a lower bound on the diversity the data supports
CONSTRAINT
  HHI is undefined when there are no interactions; with a single item it equals 1
NUMERICAL EXAMPLE (ILLUSTRATIVE shares above)
  HHI = 0.1600 + 0.0324 + 0.0144 + 0.0064 + 0.0036 + 0.0025 + 0.0016 + 0.0009 + 0.0004
        + 0.0001 + 0.000025 + 0.000025 = 0.2223
  N_eff = 4.50 of 12 items, or 37.5% of the nominal catalogue
INTERPRETATION
  inverse propensity weights span 1/0.40 = 2.50 to 1/0.005 = 200, an 80x range, so IPS
  corrections on this data have heavy tails and need clipping
```

The two-tower versus cross-feature decision is the architecture choice most often made by
habit rather than by measurement. The distinguishing test is whether the winners for a user
depend on interactions between user and item attributes that cannot be expressed as a dot
product. If they do — "this user buys large sizes only when the brand is X" — a two-tower model
will lose, and the loss shows up as a gap between ranking-stage AUC and end-to-end business
metric.

| Property | Two-tower | Cross-feature ranker | Consequence for design |
|---|---|---|---|
| scores whole catalogue | yes, via ANN | no | two-tower belongs in retrieval |
| uses user x item interactions | no | yes | ranker captures conditional preferences |
| indexable | yes | no | two-tower has a serving path, GBDT does not |
| feature freshness sensitivity | low | high | ranker needs a feature store |
| retraining cost | moderate | low | trees retrain in minutes |
| cold item handling | content features in item tower | item features as columns | both work, differently |

## 8. Failure Modes & Edge Cases

**FM-1 — offline AUC improves and online engagement does not move.**
Competing causes: (a) the AUC gain comes from items the retriever never sends, so the ranker's
improvement is unreachable; (b) the gain is concentrated in the tail where impressions are
rare. Discriminating check: recompute AUC restricted to the candidate set actually served by
the deployed retriever, and separately report the share of improved items that appear in the
retrieved set. If the restricted AUC gain is near zero while the global gain is large, cause
(a); if the restricted gain holds but impressions per improved item are under 1 per day, cause
(b) and the model needs a distribution-aware evaluation rather than a better architecture.

**FM-2 — recommendations collapse onto one category after two weeks of training.**
Competing causes: (a) the popularity feedback loop, where the model trains on its own
impressions; (b) negative sampling with popularity weighting pushing tail items into a crowded
region of the embedding space. Check: compute the HHI of recommended items weekly. A rise from
0.22 toward 0.6 with a stable catalogue indicates (a); a stable HHI but a shrinking mean
embedding norm for tail items indicates (b). The fixes differ completely: (a) needs exposure
correction or exploration, (b) needs log-Q correction in the sampler.

**FM-3 — embedding factors are "explained" in a design review and the explanation does not
reproduce.**
Competing causes: (a) factor rotation makes coordinates arbitrary between runs; (b) the run
used a different random seed and converged to a different local optimum. Check: refit with the
same seed and compare cosine similarity per user vector; then refit with a different seed and
compute the best orthogonal alignment (Procrustes) before comparing. High similarity with the
same seed and low similarity without alignment indicates (a); low similarity even with the same
seed indicates (b). Only factors that survive alignment to an external variable are worth naming.

**FM-4 — the two-tower model cannot represent a pattern the business knows is real.**
Competing causes: (a) capacity — d is too small for the interaction structure; (b) architecture
— the pattern is a genuine user x item interaction that a dot product cannot express. Check:
train a cross-feature ranker on the same data and compare AUC on the disputed segment. If the
cross-feature model wins only on that segment, cause (b) and the fix is a ranking-stage model;
if it wins everywhere, cause (a) and the fix is a wider embedding.

**FM-5 — recall@500 falls from 0.62 to 0.41 after an index migration.**
Competing causes: (a) ANN recall degraded, so the index returns worse neighbours; (b) the
candidate budget was silently reduced by the new serving path's payload limit. Check: compare
the number of candidates actually scored per request against the configured budget, then measure
ANN recall against a brute-force scan on 200 sampled requests. A candidate count below budget
indicates (b) — a serving bug, not a model change; matching counts with degraded neighbours
indicates (a).

**FM-6 — a new sequence model improves next-item accuracy 18% and reduces session length.**
Competing causes: (a) the model learned short-horizon patterns that produce redundant
recommendations, ending the session early because the user found the item; (b) the model overfits
the last click and recommends near-duplicates, ending the session by frustration. Check: measure
the repeat-purchase and near-duplicate rate in the recommended slate, and the fraction of
sessions ending within two recommendations. High duplicate rate with a satisfaction signal
unchanged indicates (b); a clean slate with higher conversion indicates (a), which is a win and
should not be "fixed".

**FM-7 — predicted click probabilities are 1.67x the observed rate.**
Competing causes: (a) the model was trained on clicks per impression but served where impressions
are counted differently (for example, slots rendered but never in the viewport); (b) negative
sampling changed the intercept, since a sampled softmax estimates a different base rate. Check:
recompute the observed rate from the exact serving log definition and compare the base rate. A
match at the serving definition points to (b); a mismatch points to (a), and the fix is a log
definition, not a model. Calibration matters immediately when the score is multiplied by a
value, because a 1.67x probability error becomes a 67% value error.

**FM-8 — training loss decreases monotonically while validation loss flatlines at epoch 2.**
Competing causes: (a) the parameter count exceeds the data, as in the 1.41-parameters-per-
observation example; (b) the validation split leaks, so the model appears to generalise while it
memorises. Check: raise lambda by 10x and see whether validation loss improves; then rebuild the
split by time rather than by random assignment. Improvement from regularisation indicates (a);
a validation score that falls back to the training-like value after a time split indicates (b).

**FM-9 — a cold item never gets recommended, even though it is a perfect match.**
Competing causes: (a) the item has no interactions, so its factorisation vector is only the
regularised prior, effectively the origin; (b) the policy layer filters it out for stock or
compliance. Check: print the item's embedding norm and compare with the median; then trace one
request through the policy stage. A norm near zero indicates (a), which is the mathematical
signature of cold start and requires content features; a healthy norm indicates (b), a policy
bug.

**Validity condition.** Every failure above is diagnosed by comparing a measurement against a
reference — a brute-force scan, a baseline model, or a re-derived log definition. A diagnosis
without a reference is a hypothesis, and the section exists to force the reference to be named
before the investigation starts.

## 9. Complexity & Cost Analysis

Architecture cost has four components: training MACs, inference MACs, memory, and pipeline
cost. Each is computed separately because they scale with different variables.

| Component | Formula | Worked value (ILLUSTRATIVE shapes) | Scales with |
|---|---|---|---|
| MF training | pairs_per_epoch * d | 2.5e8 * 64 = 1.6e10 MAC/epoch | interactions, negatives, d |
| MF memory | (U+I) * d * 4 bytes | 1.1e6 * 64 * 4 = 0.282 GB | users, items, d |
| ANN scan | catalogue * d | 1e5 * 64 = 6.4e6 MAC/request | catalogue, d |
| deep ranker | candidates * layers * width * d * 2 | 500 * 4 * 256 * 64 * 2 = 6.6e7 | candidates, width, depth |
| GBDT ranker | candidates * trees * depth | 500 * 300 * 6 = 9e5 comparisons | candidates, trees, depth |
| two-tower training | pairs * epochs * 2 * in * out | 5e7 * 10 * 2 * 256*64 = 1.64e13 MAC | pairs, epochs, width |

The GBDT versus deep comparison is the most useful row pair. For 500 candidates the tree model
needs 9e5 comparisons and the deep model 6.6e7 multiply-accumulates — a 73x difference in
arithmetic. But tree comparison cost is dominated by cache misses on feature lookups, not by the
comparison itself, so the wall-clock ratio is smaller than 36. The honest conclusion is that a
tree model is the correct default at this scale, and a deep ranker must earn its cost with a
measurable accuracy gain on a fixed evaluation set.

Training cost for a two-tower model at real scale is often assumed to be prohibitive and usually
is not. With 5e7 positives, 10 epochs, a batch of 4,096, and in-batch negatives, the number of
steps is 5e7 * 10 / 4096 = 122,070. Each step costs 2 * 256 * 64 = 32,768 MAC per sample for the
two towers, so a full step on 4,096 samples costs 1.34e8 MAC, and the whole run is
122,070 * 1.34e8 = 1.64e13 MAC. At an assumed 5e10 MAC/s that is 328 seconds of arithmetic —
under six minutes. The real cost is data reading and embedding lookups, so the bottleneck is the
input pipeline and the sparse gradient updates, not the dense arithmetic.

```text
FORMULA (training wall-clock lower bound)
  t_arith = total_MAC / throughput
  t_io = bytes_read / bandwidth
  t_max = max(t_arith, t_io)   (they overlap in a pipelined loader)
WHERE
  bytes_read = positives * epochs * bytes_per_example
CONSTRAINT
  sparse embedding updates are memory-bound; the arithmetic lower bound is not the real bound
NUMERICAL EXAMPLE (5e7 positives, 10 epochs, 200 bytes/example, 500 MB/s)
  t_arith = 1.64e13 / 5e10 = 328 s
  bytes = 5e7 * 10 * 200 = 1.0e11 bytes = 100 GB; t_io = 100e9 / 500e6 = 200 s
  measured runs land at 2-8x the max of these because of shuffle and gradient synchronisation
INTERPRETATION
  optimise the input pipeline and the optimiser state before optimising the model width
```

Memory is the constraint that decides index type and replica count. User embeddings for 1e6
users at d = 64 in float32 need 1e6 * 64 * 4 = 2.56e8 bytes = 0.256 GB, which fits comfortably
in one process. At d = 256 and 1e7 users the same table needs 1e7 * 256 * 4 = 1.02e10 bytes =
10.2 GB, which forces sharding and changes the serving architecture from "load at start" to
"remote lookup". The dimension decision is therefore a serving-topology decision, and it should
be made with the table size printed next to it.

| Table | 1e6 x 64 | 1e6 x 256 | 1e7 x 256 |
|---|---|---|---|
| bytes float32 | 256 MB | 1.02 GB | 10.2 GB |
| fits one process | yes | marginal with model | no |
| serving pattern | in-process array | in-process with care | sharded lookup |

Request payload is the other invisible cost. Ranking 512 candidates with 128 float32 features
needs 512 * 128 * 4 = 262,144 bytes = 256 KiB per request. At 23.1 requests per second that is
5.78 MiB/s, trivial; at a 3x peak of 69.4 requests per second it is 17.35 MiB/s, and if the feature
store is remote with a 1 ms service time the 512-candidate gather becomes the latency floor. This
is why feature stores batch by entity and why ranker batch sizes are chosen to match the store's
preferred fan-out.

Latency budget, ILLUSTRATIVE:

```text
stage                      ms
user tower forward          2
ANN search (1e5 items)      8
feature gather (512 x 128) 25
deep ranker or GBDT        12
slate policy                6
serialise                  4
                          ----
sum                        57
headroom to 100 ms         43
```

Serving 500 candidates from a 100,000-item catalogue through a 64-dimensional index is a
0.50% selection rate, so ANN recall at the budget is the number that bounds everything. The
index itself is affordable: a flat scan at 6.4e6 MAC per request is under a millisecond on a
modern core, which means a catalogue up to roughly 1e6 items can skip the ANN index entirely
and keep exactness. The crossover is the important engineering fact: below about 1e6 items,
exact inner-product search is usually cheaper than maintaining an index and proving its recall.

## 10. Key Theorems / Results Reference

**T1 — an orthogonal transform of the factors leaves all scores unchanged.** For any orthogonal
Q, `(Qu)·(Qv) = u·v` because Q preserves inner products. With u = (0.8, 0.1) and v = (0.7, 0.2),
score 0.58; applying Q = [[0,-1],[1,0]] gives (-0.1, 0.8) and (-0.2, 0.7), score 0.02 + 0.56 =
0.58. Consequence: factors are identified only up to rotation, so any naming of a factor needs
external validation.

**T2 — a rank-d factorisation has at most d(U + I) free parameters.** Each user and item
contributes d numbers; the score matrix has rank at most d. For U = 1e6, I = 1e5 and d = 64 the
count is 7.04e7. Consequence: the model cannot represent preference structure of rank above d,
and no amount of training fixes that; only capacity does.

**T3 — the squared-loss factorisation with fixed item vectors reduces to ridge regression per
user.** The ALS update `(V_obs^T V_obs + lambda I)^-1 V_obs^T r_obs` is exactly ridge regression
on the observed items, which is why the ridge term both stabilises the inverse (determinant
1.2952 at lambda 0.5 in the worked example) and shrinks the solution. Consequence: the
regularisation strength interacts with the number of observed items, so it must be tuned per
interaction-count bucket.

**T4 — the BPR loss is a smooth upper bound on the zero-one ranking loss.** Since
`log(1 + exp(-s)) >= 0` with equality only at infinite margin, and it is convex in s, it is a
convex surrogate for "positive outranks negative". Consequence: minimising it optimises an
ordering and says nothing about the magnitude of the scores, so BPR scores are not probabilities
and must not be used as such.

**T5 — sampled softmax is a consistent estimator of ranking order but not of probability.**
The sampled denominator omits the excluded items, so the probability scale shifts by roughly the
log ratio of catalogue size to sample size: log(1e5) = 11.513 against log(101) = 4.615, a gap of
6.9 nats. Consequence: compare models by recall, not by NLL, when the sampler changed between
runs.

**T6 — inner-product search cannot be replaced by cosine search without a norm argument.**
Cosine similarity is the inner product of the unit-normalised vectors, so it discards norms. If
item norms correlate with popularity, the two rankings differ. In the bias-share example the
top item holds 40% of interactions, and a model trained with in-batch negatives gives it a
smaller norm; the dot product then depresses it relative to cosine. Consequence: fix the
similarity function in the design document and use the same one in training and serving.

**T7 — the effective catalogue size from shares is 1/sum(share^2).** For the twelve-item example
this gives 4.50. Consequence: a diversity or coverage metric quoted against the nominal catalogue
(12 items, or 100%) overstates the room available; it must be quoted against the effective size
the data supports.

**T8 — a two-tower model's expressiveness is exactly the set of rank-d bilinear score
functions.** Since the score is a bilinear form in features of each side only, no function that
requires a product of a user attribute with an item attribute outside that bilinear form is
representable. Consequence: the architecture choice is a statement about which interactions are
assumed not to matter, and that assumption should be written down and tested on one segment.

**T9 — position bias makes logged clicks a biased sample of relevance, and IPS corrects it
under positivity.** With examination propensities ranging from 0.40 to 0.005 the inverse weights
range from 2.50 to 200, an 80x spread. Consequence: the estimator is unbiased but has variance
proportional to the square of the weight, so effective sample size, not raw click count, is the
relevant quantity.

**T10 — adding a bias term strictly increases the representable function class at constant d.**
The bias-only model is the d = 0 case; with d > 0 the biases remain and the dot product adds
deviation. Consequence: a model without biases wastes capacity re-learning popularity, and the
comparison against the bias baseline is the minimum bar for any claim of personalisation gain.

**T11 — the cascade's total cost is the sum over stages of (items x cost per item).** For the
illustrative split with the tree ranker, retrieval 6.4e6 + ranking 9.0e5 + policy 5.0e4 = 7.35e6
operations per request. Consequence: with a tree ranker retrieval dominates by 7x and the candidate
budget is cheap to widen; with the deep ranker the ordering inverts to 6.6e7 and the budget becomes
the binding constraint. The stage ratio, not the absolute cost, is what a design review must state.

## 11. Interview / Exam Questions

**Q1 (calculation).** Given u = (0.8, 0.1) and v = (0.7, 0.2), compute the factorisation score
and the score after applying the 90-degree rotation Q = [[0,-1],[1,0]] to both vectors.
*Answer:* score = 0.56 + 0.02 = 0.58. Rotated: Qu = (-0.1, 0.8) and Qv = (-0.2, 0.7), score =
0.02 + 0.56 = 0.58.
*Reasoning:* orthogonal transforms preserve inner products, so the score is invariant and the
coordinates are not identified.
*Follow-up:* what does this imply for a claim that "factor 2 is brand loyalty"? The claim must be
supported by correlation with an external brand signal after alignment, not by inspecting the
raw coordinate.
*Red flag:* asserting that factors have intrinsic meaning because the loss converged.

**Q2 (calculation).** A user interacted with items (0.7,0.2) and (-0.2,0.3) with values 1 and 1,
and not with (0.1,0.8). With lambda = 0.5, compute the ALS user vector.
*Answer:* V^T V = [[0.53, 0.08],[0.08, 0.13]] before ridge; adding 0.5 gives [[1.03, 0.08],
[0.08, 0.63]]. V^T r = (0.5, 0.5). Determinant = 0.6489 - 0.0064 = 0.6425. u_1 = (0.5*0.63 -
0.08*0.5)/0.6425 = (0.315 - 0.04)/0.6425 = 0.4280; u_2 = (1.03*0.5 - 0.5*0.08)/0.6425 = (0.515 -
0.04)/0.6425 = 0.7393.
*Reasoning:* only observed items enter V^T V and V^T r; the unobserved item is absent, not zero.
*Follow-up:* what happens with lambda = 0? The system is still invertible here (determinant
0.0625 is small), and the answer swings to u_1 = 0.4000 and u_2 = 3.6000 — the second
coordinate grows 4.87x from 0.7393. That swing is what the ridge term suppresses.
*Red flag:* including the unobserved item with value 0 in the regression.

**Q3 (calculation).** Four scores are 0.580, 0.160, 0.390, -0.130. Compute the softmax
probability of the first and its NLL.
*Answer:* exponentials 1.78604, 1.17351, 1.47698, 0.87810; Z = 5.31463; p = 0.33606; NLL =
1.09046 nats. Uniform over four items would be 1.38629, so the model beats chance by 0.29583.
*Reasoning:* the softmax is shift-invariant, so only differences of scores matter; the raw
offset of -0.130 is irrelevant to the ranking.
*Follow-up:* with one positive and 100 uniform negatives, what does the NLL become for a random
model? Roughly log(101) = 4.615, so NLL values from the two regimes are not comparable.
*Red flag:* comparing sampled NLL with full-catalogue NLL directly.

**Q4 (calculation).** Compute the BPR loss and the gradient on u for u = (0.8,0.1), v_pos =
(0.7,0.2), v_neg = (0.5,-0.1).
*Answer:* margins 0.58 and 0.39, s = 0.19, sigmoid = 0.54736, loss = 0.60265. dL/du =
-(1 - 0.54736) * (0.2, 0.3) = (-0.09053, -0.13579).
*Reasoning:* the coefficient 1 - sigmoid(s) is the model's uncertainty; a well-separated pair
gives a near-zero gradient and stops learning from that pair.
*Follow-up:* at what margin does the loss become negligible? At s = 5, sigmoid = 0.99331, loss =
0.00672, and the gradient coefficient is 0.0067 — three orders of magnitude smaller than at s =
0.19.
*Red flag:* a sign error that moves u toward the negative item.

**Q5 (calculation).** A catalogue has 100,000 items at d = 64 float32, and 1e6 users. Compute
index memory and full-scan cost per request.
*Answer:* items 1e5 * 64 * 4 = 2.56e7 bytes = 25.6 MB; users 1e6 * 64 * 4 = 2.56e8 bytes =
256 MB; total 0.282 GB. Full scan = 1e5 * 64 = 6.4e6 MAC per request.
*Reasoning:* the scan is affordable in arithmetic; the reason ANN exists is memory bandwidth and
the tail of the distance distribution, not raw FLOPs.
*Follow-up:* at what catalogue size does the scan stop being affordable? At 1e7 items and 64
dimensions the scan is 6.4e8 MAC per request; at 69 requests per second that is 4.4e10 MAC/s of
purely bandwidth-bound work, which is where an index becomes mandatory.
*Red flag:* assuming ANN is needed at every scale.

**Q6 (diagnosis).** Two training runs produce embeddings with cosine similarity 0.3 between the
same users' vectors. Is the model unstable?
*Answer:* not necessarily. Check the seed first; then align with an orthogonal Procrustes
transform and recompute. High similarity with the same seed proves determinism; similarity
recovering after alignment proves the model found the same structure in rotated coordinates.
*Reasoning:* factors are identified only up to rotation, so raw coordinate comparison across
runs is not a stability measure.
*Follow-up:* what would real instability look like? Low similarity after alignment, or a large
difference in downstream recall at matched budget.
*Red flag:* concluding instability from raw coordinate difference.

**Q7 (calculation).** With 5e7 interactions, 1e6 users and 1e5 items, compute density, average
interactions per user, parameter count at d = 64, and the parameter-to-observation ratio.
*Answer:* density = 5e7/1e11 = 5.0e-4; average 50 interactions per user; parameters = 64 *
1.1e6 = 7.04e7; ratio = 7.04e7/5e7 = 1.41.
*Reasoning:* a ratio above 1 means the model can nearly memorise the training matrix, so
regularisation and time-based holdout are mandatory.
*Follow-up:* what does d = 256 do? The ratio becomes 5.63 and the memorisation risk rises roughly
fourfold for the same data.
*Red flag:* reporting a holdout score from a random split on data with this ratio.

**Q8 (calculation).** Twelve items have shares 0.40, 0.18, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03,
0.02, 0.01, 0.005, 0.005. Compute the Herfindahl index and the effective catalogue size.
*Answer:* HHI = 0.2223; N_eff = 4.50 items of 12, or 37.5%.
*Reasoning:* squaring the shares weights the head heavily, so a single dominant item collapses
the effective size.
*Follow-up:* what is the inverse propensity weight range? From 1/0.40 = 2.50 to 1/0.005 = 200, a
factor of 80, which is why IPS on this data needs clipping.
*Red flag:* reporting coverage against the nominal 12 items as if all were equally available.

**Q9 (diagnosis).** Offline ranking AUC rises 0.02 but the product metric is flat. What are the
two most likely explanations and how do you tell them apart?
*Answer:* (a) the improvement applies to items the retriever does not send; (b) the improvement
applies to items with almost no impressions. Separate by recomputing AUC on the served candidate
set and by measuring impressions per improved item.
*Reasoning:* a ranker's AUC over the full catalogue describes a distribution the product never
sees, because retrieval has already truncated it.
*Follow-up:* what would you change in the evaluation to prevent a repeat? Add a retrieval-conditional
metric to the offline suite so the two stages cannot be conflated.
*Red flag:* accepting the offline gain as causal evidence for the product.

**Q10 (diagnosis).** After an index migration, recall@500 falls from 0.62 to 0.41 with no model
change. Where do you look first?
*Answer:* count the candidates actually scored per request. If it is below the configured budget,
the new serving path truncated the payload. If it matches, measure ANN recall against a
brute-force scan on 200 requests; degraded neighbours indicate the index, not the model.
*Reasoning:* a recall change with no model change must come from the candidate set or the
evaluation, and those are two distinct artefacts with distinct checks.
*Follow-up:* what recall does a brute-force scan guarantee? Exactly 1.0 at the budget, which is
why it is the reference and why catalogues below about 1e6 items can skip ANN entirely.
*Red flag:* retraining the model to fix a serving-path truncation.

**Q11 (judgment).** A deep ranker costs 73x more arithmetic than a GBDT and improves offline AUC
by 0.008 on 2e5 judged impressions. Ship it?
*Answer:* compute the interval first. With 2e5 paired impressions and AUC standard error on the
order of 0.005 for a difference of that size, 0.008 is not clearly outside noise; require either
a larger set or a pre-registered segment where the gain is concentrated. The cost difference also
has a maintenance implication: 73x arithmetic with a feature store is not free.
*Reasoning:* the decision is effect size against resolution, then cost against benefit, in that
order.
*Follow-up:* what evidence would change the answer? A gain confined to a segment with high
impression volume and a stable sign across three weeks.
*Red flag:* shipping on a point estimate with no interval and no segment analysis.

**Q12 (calculation).** A two-tower training run has 5e7 positives, 10 epochs, batch 4,096, tower
widths 256 -> 64. Compute steps and total MAC.
*Answer:* steps = 5e7 * 10 / 4096 = 122,070. Per sample 2 * 256 * 64 = 32,768 MAC; per step
4,096 * 32,768 = 1.34e8; total 122,070 * 1.34e8 = 1.64e13 MAC.
*Reasoning:* dense arithmetic is not the bottleneck; at an assumed 5e10 MAC/s the arithmetic is
328 seconds, so the input pipeline and sparse optimiser updates dominate wall-clock.
*Follow-up:* how would you cut the cost in half? Halve the epochs and check the validation
curve; if it flatlines at epoch 2, the saving is free.
*Red flag:* assuming the run must be distributed because the MAC total is large.

**Q13 (diagnosis).** Predicted CTR is 1.67x the observed rate. Give two causes and the check.
*Answer:* (a) the impression definition differs between training and serving, for example
viewport-rendered versus served; (b) negative sampling shifted the intercept, since a sampled
softmax estimates a different base rate. Check by recomputing the observed rate with the serving
definition: a match points to (b); a mismatch to (a).
*Reasoning:* calibration is about the intercept and the base rate, and both are defined by the
logging pipeline rather than the model.
*Follow-up:* why does this matter beyond aesthetics? If the score multiplies a value, a 1.67x
probability error becomes a 67% error in expected value and reorders the slate.
*Red flag:* recalibrating on the same mis-defined log.

**Q14 (calculation).** Twelve candidate scores, of which 20 are relevant in the catalogue; you
retrieve 500 of 100,000. What does a random retriever achieve in expected hits, and what does the
popularity baseline achieve at rank 1?
*Answer:* random recall@500 with 20 relevant among 1e5 items yields 20 * 500/1e5 = 0.1 expected
hits, so recall 0.005. A popularity baseline that always returns the top item achieves recall@1
equal to that item's share, 0.40 in the example.
*Reasoning:* the random baseline scales with budget over catalogue, so a retriever must be
compared against it at the same budget or the comparison is meaningless.
*Follow-up:* why is the popularity baseline so strong? Because 40% of interactions concentrate on
one item, so the expected number of relevant hits at rank 1 is far above random.
*Red flag:* quoting recall@500 of 0.6 without stating the 0.5% selection rate.

**Q15 (judgment).** A sequence model gains 18% next-item accuracy and shortens sessions. Is that
a regression?
*Answer:* not necessarily. Shortened sessions can mean the user found what they wanted faster.
Separate the paths: measure near-duplicate rate in the slate and conversion per session. Clean
slates with higher conversion mean the model is working; duplicate-heavy slates with unchanged
satisfaction mean it learned the last click and is echoing it.
*Reasoning:* session length conflates success and frustration, so it is a diagnostic trigger
rather than a metric.
*Follow-up:* which guardrail would you add permanently? Repeat-recommendation rate within a
session, since it separates the two mechanisms cheaply.
*Red flag:* reverting a real improvement because a proxy moved in the unexpected direction.

## 12. Quick Reference Card

| Item | Value or rule |
|---|---|
| MF parameters | d(U + I); at 1e6 users, 1e5 items, d = 64 -> 7.04e7 |
| ALS user update | (V_obs^T V_obs + lambda I)^-1 V_obs^T r_obs, ridge is mandatory |
| BPR loss | log(1 + exp(-s)) with s = u·v_pos - u·v_neg; gradient scales by 1 - sigmoid(s) |
| Sampled softmax gap | log(1e5) = 11.513 vs log(101) = 4.615; compare by recall, not NLL |
| Two-tower scan | catalogue * d = 6.4e6 MAC at 1e5 items, d = 64 |
| Embedding memory | (U+I) * d * 4 bytes; 1e6 x 64 = 256 MB, 1e7 x 256 = 10.2 GB |
| Selection rate | 500 of 100,000 = 0.50%; always quote recall with its budget |
| Effective catalogue | N_eff = 1/sum(share^2) = 4.50 of 12 in the worked example |
| Architecture rule | retriever maximises recall at budget, ranker maximises precision on candidates |
| Never | explain a latent factor without external alignment evidence |
