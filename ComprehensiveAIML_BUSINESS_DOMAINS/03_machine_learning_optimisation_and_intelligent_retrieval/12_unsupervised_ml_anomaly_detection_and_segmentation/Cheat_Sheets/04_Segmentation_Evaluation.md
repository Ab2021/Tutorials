---
collection: "unsupervised-ml-anomaly-detection-and-segmentation"
file_id: "04"
title: "Segmentation Evaluation"
topic: "Segmentation evaluation: geometric, agreement and decision-value criteria"
collection_section: "Part 03 — Machine learning optimisation and intelligent retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
status: "draft"
version: "2.0"
notation_style: "plain-text, ASCII formulas inside fenced blocks"
review_trigger: "Change in the decision the segmentation feeds, the outcome window, the offer/pricing map, or the feature pipeline that defines segment membership."
source_status: "OPEN"
---

# Segmentation Evaluation

> **Evidence:** the identities and metrics in this sheet are `DERIVED` from their standard definitions and every printed number was recomputed by hand in a fenced block. Population sizes, response rates, offer costs and outcome values are `ILLUSTRATIVE` teaching inputs, not observations from any organisation. No deployment, benchmark result or vendor claim is asserted. Where a statement depends on a literature result it is marked `SOURCED` and carries the reference in Section 10. Everything about a real segmentation programme is `OPEN`.

## 1. File Header

**Purpose.** Decide whether a given segmentation is good enough to drive a business action, and be able to defend that decision when someone else's segmentation scores better on a different metric.

**Prerequisites.** You should be comfortable with: the distinction between a distance and a dissimilarity; the definition of variance and of a between/within decomposition; contingency tables and the idea of a chance-corrected index; expected value under a binary outcome; and the habit of naming a denominator. If "the silhouette went up" already feels like a conclusion to you, this sheet is aimed at exactly that reflex.

**Key outcome.** After working through this sheet you can: (a) compute the per-segment expected value of an action and the break-even response rate at which the action stops paying; (b) decompose the value of a segmentation into a targeting component and an action-differentiation component, so you know which half your clustering actually earned; (c) compute and correctly interpret an internal index (silhouette, Calinski–Harabasz, Davies–Bouldin) and state what each one is monotone in; (d) compute partition agreement (Rand, adjusted Rand, F1 on pairs, purity, NMI under three normalisations) and explain why four of those five can disagree on one table; (e) run a stability check that costs a defensible amount of compute; (f) reject the six most common invalid comparisons, which are enumerated in Section 8.

**Not covered.** This sheet does not teach how to fit a clustering algorithm (that is sheet 02), does not cover change-point-driven resegmentation (sheet 03), does not design the human review loop around segment-level decisions (sheet 05), and does not contain a single measured result from a production system. It also does not settle the choice of `k` for you: no metric can, because `k` is a decision variable with a cost, not a parameter with a true value.

**Level.** Core. The arithmetic is elementary; the difficulty is entirely in keeping the evaluation object aligned with the decision object.

**How to read the evidence tags.** `DERIVED` means the result follows from the stated definition and can be re-derived on paper. `ILLUSTRATIVE` means the input numbers were chosen to make a mechanism visible and carry no empirical claim. `SOURCED` means a named published result is being paraphrased, with the reference listed in Section 10. `OPEN` means nobody has checked it and it must not be used as a premise.

**Retrieval cue.** If a stakeholder says "our new segmentation is 12% better", jump to Section 8, pick the failure mode that matches how the 12% was computed, and then return to Section 7 for the metric that would have been admissible.

**Relation to sibling sheets.** Sheet 02 gives the fitting mechanics; this sheet assumes a partition already exists and asks whether it is worth its maintenance cost. Sheet 01's triage algebra reappears here as the per-segment expected-value formula, because a segment-level decision is a capacity-free special case of the triage problem.

## 2. Concept Map / Mental Model

### 2.1 Three families of criteria, three different questions

A segmentation can be scored against geometry, against another partition, or against the outcome it is supposed to move. These are not three measurements of one quantity. They answer three different questions, and a scheme can be first-rate on one and worthless on the others.

```text
FAMILY 1  GEOMETRIC              FAMILY 2  AGREEMENT              FAMILY 3  DECISION
--------------------------       --------------------------       --------------------------
"are the groups compact         "do two partitions say           "does acting on the groups
 and separated in the            the same thing, beyond           beat not acting, or acting
 space I clustered in?"         chance?"                         on one group?"
inputs: X, labels                inputs: two label vectors        inputs: labels, outcome y,
no outcome needed                or labels + reference            action cost c, value G
silhouette, CH, DB, gap          Rand, ARI, NMI, Jaccard, F1      lift, capture, Gini, KS, V(S)
     |                                |                                |
     v                                v                                v
says nothing about whether       says nothing about whether       says nothing about whether
the clusters differ in y         either partition is useful       the geometry is stable
```

The three families are complementary in a specific way. Family 1 is cheap and label-free, so it is the only one available before an outcome window has matured. Family 2 is the only one that can detect instability, because instability is disagreement between two partitions of the same population. Family 3 is the only one that can authorise spend. A defensible evaluation reports one number from each family and states which one is binding.

### 2.2 The evaluation object must equal the decision object

Most segmentation-evaluation failures are type errors: a quantity is computed on one object and used to justify a decision about a different object. The four objects that get confused are listed below, and the confusion is always resolvable by asking "on what rows, over what window, with what action?"

```text
OBJECT A: the fitted partition      labels for the rows that entered the fit
OBJECT B: the deployed assignment   labels for rows scored after the fit, on live features
OBJECT C: the evaluated cohort      rows whose outcome window has closed by the cut-off
OBJECT D: the acted-on population   rows that actually received the segment's action
```

A partition quality metric is a statement about A. A business value metric is a statement about D, filtered through C. If the assignment pipeline changes a feature definition, B drifts away from A while every Family 1 metric stays frozen at its training value, because those metrics were computed on A. Section 8.6 gives the discriminating check.

### 2.3 The chain from features to value

```text
feature pipeline -> distance -> partition -> segment name -> action map -> outcome -> value
      |                |            |             |              |            |
  scaling,           metric,      k, seed,     human           price,       response
  missingness,       weighting    init         interpretation  channel,     window,
  point-in-time                                                timing       maturity
      |                |            |             |              |            |
      +----------------+------------+-------------+--------------+------------+
                     every link can break the value, and none of them
                     is measured by an internal clustering index
```

Reading the chain right to left is the diagnostic order that works. Start from the value you failed to earn, ask which action map produced it, ask which segment names the action map consumed, and only then ask whether the partition that produced those names was reasonable. Reading left to right — "our silhouette is 0.61, therefore the campaign will work" — skips four links that carry more risk than the clustering itself.

### 2.4 The central invariant

```text
value_claim_is_computed_on_the_population_that_received_the_action
AND comparator_is_the_actual_alternative (blanket action or no action)
AND every metric names its k, its feature space and its evaluation window
```

Those three clauses are the invariant this whole sheet is organised around. Cluster-count choice, distance choice and index choice are subordinate: a scheme satisfying the invariant with a mediocre silhouette is usable, and a scheme violating it with an excellent silhouette is not.

### 2.5 Where segmentation value actually comes from

There are exactly two mechanisms by which splitting a population can increase value, and they can be separated arithmetically (Section 4.2). The first is **targeting**: identifying members whose expected value under the standard action is negative, and not acting on them. The second is **differentiation**: applying a different action — a different price, a different channel, a different contact frequency — to members whose response differs enough to justify the different action's different cost.

A third mechanism is often claimed and is not real: **description**. Naming segments " aspirational families" or "digital natives" produces no value by itself. It becomes value only when it changes an action. That is not a rhetorical point; it is why Section 4.2's decomposition has two terms and not three.

### 2.6 The k-sweep is not monotone, and that is the lesson

It is tempting to read the metric-versus-k curve as a hill to climb. In the worked example (Section 5) the decision value at k=1 is 36,835 currency units, at k=3 it is 35,575, at k=4 it is 40,204 and at k=5 it is 41,968 on the same 20,000 members. The dip at k=3 is not noise: it is a coarser scheme whose merged segment mixes members with very different response rates, combined with a more expensive action for the high-value group. Adding a segment boundary can therefore destroy value if the boundary is drawn in the wrong place, and internal indices frequently prefer that same coarser scheme.

**Takeaway:** geometry, agreement and value are three separate questions; the invariant is that a value claim must be computed on the acted-on population against the real alternative.

## 3. Foundational Definitions

Every symbol below is defined once and reused with the same meaning in Sections 4–12. Where an established term has two incompatible operational meanings, both are given.

| Symbol / term | Meaning | Units / shape | Valid range | Common confusion | Operational check |
|---|---|---|---|---|---|
| `N` | number of members in the population being segmented | count, integer | `N >= 1` | counted after excluding ineligible members in one place, before in another | publish the eligibility filter with every count |
| `k` | number of segments in the scheme | count, integer | `1 <= k <= N` | treated as a fitted parameter rather than a decision | state `k` in every metric line |
| `n_s` | members assigned to segment `s` | count, integer | `0 <= n_s <= N`, `Σ n_s = N` | mixing pre- and post-dedupe counts | assert the sizes sum to `N` |
| `X` | the feature matrix the partition was fitted on | matrix `N × p` | numeric, finite after imputation | the live scoring matrix silently differs | hash the feature spec and store it with the model |
| `p` | number of features entering the distance | count | `p >= 1` | features added to the pipeline but not to the distance | log the exact feature list per fit |
| `d(i,j)` | distance between members `i` and `j` under the declared metric | same units as the scaled features | `>= 0`, symmetric | Euclidean on unscaled features labelled "standardised" | recompute one pair by hand |
| `a(i)` | mean distance from member `i` to all other members of its own segment | distance units | `>= 0`, undefined if `n_s = 1` | computed including `i` itself | check whether the diagonal is excluded |
| `b(i)` | minimum over other segments of the mean distance from `i` to that segment | distance units | `>= 0` | averaged over all other segments instead of taking the minimum | state "nearest other segment" explicitly |
| `s(i)` | silhouette of member `i` | dimensionless | `[-1, 1]` | averaged over a population with undefined members silently dropped | report the count of undefined members |
| `S̄` | mean silhouette over the evaluated members | dimensionless | `[-1, 1]` | called "cluster quality" without naming the feature space | name the space and the scaling |
| `W` | within-segment sum of squared distances to segment centroids | squared feature units | `>= 0` | computed with a different metric than the one used to fit | same metric, same features |
| `B` | between-segment sum of squares, weighted by segment size | squared feature units | `>= 0` | unweighted version used interchangeably | check weights equal `n_s` |
| `CH` | Calinski–Harabasz index `[B/(k-1)] / [W/(N-k)]` | dimensionless | `>= 0`, undefined for `k = N` | read as a probability or as an `F` p-value | report `k` and `N` with it |
| `DB` | Davies–Bouldin index, average over segments of the worst ratio of summed spreads to centroid separation | dimensionless | `>= 0`, lower is better | higher-is-better sign error | state the direction in the table header |
| `M_ij` | distance between centroids of segments `i` and `j` | distance units | `>= 0` | recomputed in a different feature space from the spreads | one metric per index |
| `S_i` | mean distance from members of segment `i` to its centroid | distance units | `>= 0` | confused with `a(i)`, which is pairwise within-cluster | state which definition |
| `n_ij` | count of members in reference class `i` and predicted cluster `j` | contingency count | integer `>= 0` | built on different grains for the two partitions | assert row and column sums equal `N` |
| `a_i` | row sum `Σ_j n_ij` | count | integer | row/column transposition | row sums are the reference partition |
| `b_j` | column sum `Σ_i n_ij` | count | integer | same | column sums are the compared partition |
| `C(n,2)` | number of unordered pairs among `n` items, `n(n-1)/2` | count | `0` for `n < 2` | replaced by `n²/2`, which double counts | recompute one entry |
| `ARI` | adjusted Rand index, chance-corrected pair agreement | dimensionless | `<= 1`, can be negative | compared across different `k` as if on one scale | report both `k` values |
| `RI` | raw Rand index, fraction of pairs on which the two partitions agree | dimensionless | `[0, 1]` | treated as if 0 were the no-agreement baseline | compute its chance level |
| `NMI` | normalised mutual information | dimensionless | `[0, 1]` for the max and geometric normalisations | three normalisations reported under one name | name the normaliser |
| `H` | Shannon entropy in bits | bits | `>= 0` | natural log used for one term and log2 for another | one base per table |
| `purity` | fraction of members in the majority reference class of their cluster | dimensionless | `[0, 1]` | rises mechanically with `k` | report `k` beside it |
| `J` | Jaccard index between two partitions, computed on same-cluster pairs | dimensionless | `[0, 1]` | computed on clusters rather than pairs | state the pair definition |
| `Gini` | twice the area between the Lorenz curve of cumulative outcome and the diagonal | dimensionless | `[0, 1 - 1/m]` for `m` equal groups | compared across different base rates | report the base rate |
| `KS` | maximum absolute gap between cumulative outcome and cumulative non-outcome distributions | dimensionless | `[0, 1]` | read as an accuracy | report both cumulative curves' sample sizes |
| `lift@q` | outcome rate in the top `q` fraction divided by the population rate | dimensionless | `>= 0` | denominator taken after exclusions | state `q` and the exclusions |
| `r_s` | response probability for a member of segment `s` under the declared action | probability | `[0, 1]` | measured on already-targeted members only | state the exposure condition |
| `c_s` | cost of applying the action to one member of segment `s` | currency per member | `>= 0` | excludes fulfilment cost of a positive outcome | itemise the cost |
| `G` | gross value of one positive outcome, before action cost | currency per outcome | `> 0` | net value used interchangeably with gross | define once, use consistently |
| `r*_s` | break-even response probability, `c_s / G` | probability | `[0, 1]` if `c_s <= G` | confused with the observed `r_s` | recompute when either input changes |
| `V(S)` | total expected value of scheme `S` after optimal per-segment action choice | currency | can be negative | computed with actions fixed by policy | state whether action choice is optimised |
| `S_0` | the null scheme: one segment, one action, no targeting | scheme | `k = 1` | omitted entirely, so gains look larger | always report the null comparator |
| `ΔV` | `V(S) - V(S_0)` | currency | any sign | reported without the comparator | print both terms |
| `T` | targeting component of `ΔV`, actions held at the null action | currency | any sign | merged into a single "segmentation gain" | compute separately |
| `A` | action-differentiation component of `ΔV` | currency | any sign | same | compute separately |
| `HHI` | Herfindahl–Hirschman concentration of segment sizes, `Σ (n_s/N)²` | dimensionless | `[1/k, 1]` | confused with a diversity index where higher is better | state the direction |
| `N_eff` | effective number of segments, `1 / HHI` | count | `[1, k]` | read as `k` | compare both |

Four of these entries deserve a sentence each, because they are the ones that produce the most expensive mistakes.

**`r_s` is conditional on exposure, not on membership.** If the only members of segment `s` whose outcome you observe are the ones who received the action, then what you have is `P(outcome | member of s, action applied)`, which is the right quantity for an expected-value calculation and the wrong quantity for a claim about what would have happened without the action. Segment-level uplift requires either a holdout or an identification argument; Section 8.5 handles the failure.

**`Gini` and `KS` are ranking statistics, not segmentation statistics.** They measure how well a continuous score orders members by outcome likelihood. A hard partition into `k` segments can be converted into a score by using each segment's fitted response rate, which produces massive ties and therefore a `Gini` far below what a continuous model achieves on the same information. Reporting `Gini` for a discretised score and comparing it with a competitor's continuous score is a units error, not a modelling error.

**`purity` rises with `k` mechanically.** With `k = N` every cluster contains one member and purity is 1 by construction. It is reported for completeness and should never be used to choose `k`; if it is used, it will always select the largest `k` you are willing to run.

**`N_eff` is the honest size of your segmentation.** A five-segment scheme with sizes 45%, 26%, 15.5%, 9% and 4.5% has `HHI = 0.3043` and `N_eff = 3.287`, which is the number to quote when someone asks how many segments you really have. Three segments carry 86.5% of the population; the smallest two are decision devices for small groups, not population splits.

**Takeaway:** the table is the contract for Sections 4–11; if a metric appears in a report without the units, direction and evaluation population from this table, it is not yet a measurement.

## 4. Core Formulas with Full Derivation

Each subsection uses the same anatomy: formula, then where each symbol comes from, then the conditions under which the derivation holds, then a numerical example with the arithmetic shown, then the interpretation. Every subsection closes with an explicit validity condition, because the failure mode of a formula is almost always a violated precondition rather than an arithmetic slip.

### 4.1 Base case

The base case is a single segment and a single action: is it worth applying the action to a member at all?

```text
FORMULA
  EV_s   = n_s * ( r_s * G - c_s )                    total expected value of acting on segment s
  EV_1   = r_s * G - c_s                              per-member expected value
  r*_s   = c_s / G                                    break-even response probability
  treat s  <=>  r_s > r*_s                            the decision rule

WHERE
  n_s   segment size, count, integer >= 0
  r_s   response probability under the action, dimensionless, 0 <= r_s <= 1
  G     gross value of one positive outcome, currency per outcome, G > 0
  c_s   cost of applying the action to one member, currency per member, c_s >= 0
  r*_s  break-even response probability, dimensionless
```

The derivation is one line of expectation. Let `Y` be the indicator that a member responds, with `E[Y] = r_s`. The action costs `c_s` whether or not the member responds, and pays `G` only if the member responds. Then `E[value per member] = G·E[Y] - c_s = r_s·G - c_s`, and summing over independent members multiplies by `n_s`. The break-even value of `r_s` solves `r_s·G - c_s = 0`.

```text
CONSTRAINT
  1. r_s is homogeneous within the segment: every member has the same response probability.
  2. c_s does not depend on whether the member responds.
  3. G is constant across responders and independent of segment.
  4. No capacity limit: acting on every eligible member is feasible.
  5. No interaction between members: one member's response does not change another's.
```

Constraint 1 is the one that does the work, and it is also the only one that is never exactly true. Its practical reading is that `r_s` must be an average over a group small enough that the average is decision-relevant. If a segment contains both 4% and 7.5% responders and the break-even rate is 2.5%, the segment average is adequate for the treat/do-not-treat decision because both sub-groups are far from the boundary. If the segment straddles the boundary, the average is not adequate and the segmentation is too coarse for that decision.

```text
NUMERICAL EXAMPLE  (ILLUSTRATIVE inputs, DERIVED arithmetic)
  Segment A:  n_A = 9,000   r_A = 0.041   c_A = 2.40   G = 95
  gross per member    = 0.041 * 95   = 3.895
  net per member      = 3.895 - 2.40 = 1.495
  total               = 9,000 * 1.495 = 13,455.00

  break-even for A    = 2.40 / 95 = 0.0252632
  margin              = 0.041 / 0.0252632 = 1.6229  (62.3% above break-even)

  Segment C:  n_C = 3,100   r_C = 0.018   c_C = 2.40
  gross per member    = 0.018 * 95   = 1.710
  net per member      = 1.710 - 2.40 = -0.690
  total               = 3,100 * (-0.690) = -2,139.00

INTERPRETATION
  A pays 1.495 per member contacted and 13,455 in total; C destroys 0.690 per member
  contacted and 2,139 in total. The gap between A and C is not the response-rate gap
  (0.023) but the net-value gap (2.185 per member), because the same 2.40 is spent
  on both. Break-even at 2.53% is the number to quote in a design review, not the
  absolute response rate: it is the only number that stays meaningful when G changes.
```

Validity condition for 4.1: the rule is valid only while `1` holds in the weak sense above (no segment straddles `r*`), `c_s` is a real per-contact cost rather than an allocated overhead, and the outcome window over which `r_s` was measured matches the window over which `G` is realised. A campaign measured on 30-day response and valued on 12-month revenue is two different decisions wearing one formula.

### 4.2 General case

The general case has `k` segments, a per-segment action with its own cost, and a null comparator. Two results matter: the total value of the scheme, and its decomposition into targeting and differentiation.

```text
FORMULA
  V(S)  = Σ_s n_s * max( 0, r_s * G - c_s )                        value of scheme S with per-segment action choice

  V(S_0) = N * max( 0, r̄ * G - c_0 )                               null scheme: one segment, pooled rate, one action

  r̄     = ( Σ_s n_s * r_s ) / N                                    pooled response rate

  ΔV    = V(S) - V(S_0)                                            total gain over the null scheme

  T     = Σ_s n_s * max(0, r_s*G - c_0)  -  N * max(0, r̄*G - c_0)  targeting component (actions held at c_0)
  A     = Σ_s n_s * [ max(0, r_s*G - c_s) - max(0, r_s*G - c_0) ]  action-differentiation component
  ΔV    = T + A                                                    by construction; verify numerically

WHERE
  S     the scheme: a partition plus a map from segment to action cost c_s
  c_0   the null action's per-member cost, currency per member
  r̄     pooled rate, dimensionless, 0 <= r̄ <= 1
  T, A  currency; either may be negative
```

The derivation of the decomposition is an algebraic identity, which is why it is worth trusting: write `V(S) = Σ_s n_s f(r_s, c_s)` with `f(r,c) = max(0, rG - c)`, and add and subtract the same intermediate term `Σ_s n_s f(r_s, c_0)`. The first difference is `T` (segmentation with actions frozen at the null action), the second is `A` (action optimisation with the partition frozen). Nothing is double counted.

```text
CONSTRAINT
  1. Each segment's action is chosen independently; no global capacity constraint.
  2. c_s is the true incremental cost of segment s's action, including fulfilment.
  3. r_s is estimated from a population representative of the acted-on population.
  4. The null comparator uses the same G, the same outcome window, and the same eligibility filter.
  5. Segment members are exchangeable with respect to the action (no within-segment targeting).
```

Constraint 4 is where most "segmentation uplift" claims die. If the null comparator uses a different offer price, a different eligibility rule or a different measurement window, `ΔV` mixes the segmentation effect with three other changes.

```text
NUMERICAL EXAMPLE  (ILLUSTRATIVE inputs, DERIVED arithmetic)
  Five segments, N = 20,000, G = 95, null action cost c_0 = 2.40

  seg   n_s      r_s     c_s    r_s*G    EV/member   decision
  A     9,000   0.041   2.40    3.895    +1.495      treat at 2.40
  B     5,200   0.062   1.80    5.890    +4.090      treat at 1.80
  C     3,100   0.018   2.40    1.710    -0.690      do not treat
  D     1,800   0.075   3.10    7.125    +4.025      treat at 3.10
  E       900   0.012   2.40    1.140    -1.260      do not treat

  pooled rate  r̄ = (9,000*0.041 + 5,200*0.062 + 3,100*0.018 + 1,800*0.075 + 900*0.012) / 20,000
                 = (369 + 322.4 + 55.8 + 135 + 10.8) / 20,000
                 = 893.0 / 20,000 = 0.0446500

  null value   V(S_0) = 20,000 * (0.04465*95 - 2.40) = 20,000 * (4.24175 - 2.40)
                      = 20,000 * 1.84175 = 36,835.00

  scheme value V(S)   = 9,000*1.495 + 5,200*4.090 + 1,800*4.025
                      = 13,455.00 + 21,268.00 + 7,245.00 = 41,968.00

  total gain   ΔV     = 41,968.00 - 36,835.00 = 5,133.00   (256.65 per 1,000 members)

  targeting    T      = (13,455 + 5,200*(5.890-2.40) + 1,800*(7.125-2.40)) - 36,835
                      = (13,455 + 18,148 + 8,505) - 36,835
                      = 40,108 - 36,835 = 3,273.00

  action term  A      = 41,968 - 40,108 = 1,860.00
                      check: B gains 0.60 per member * 5,200 = +3,120
                             D loses 0.70 per member * 1,800 = -1,260
                             net = +1,860    (matches)

  identity check      T + A = 3,273 + 1,860 = 5,133 = ΔV   (holds)

INTERPRETATION
  Two thirds of the gain (3,273 of 5,133) comes from not contacting 4,000 members whose
  expected value is negative. One third (1,860) comes from spending less on B and more on D.
  If procurement freezes all offers at 2.40, the defensible claim is 3,273, not 5,133.
  Quoting 5,133 while crediting the clustering obscures that 36% of it is a pricing decision.
```

Validity condition for 4.2: the identity `ΔV = T + A` always holds, but the *interpretation* of `T` and `A` as separable business effects requires that the action map was not chosen by looking at the same `r_s` estimates that enter `T`. If the action for each segment was tuned on the evaluation cohort, `A` is an in-sample maximum and overstates the achievable differentiation; Section 5.6 quantifies how much.

### 4.3 Variants

Four variant families matter in practice: the internal geometric indices, the agreement indices, the ranking indices, and the sampling variants that make any of them affordable. Each is given with its own derivation because the normalisations are where errors enter.

#### 4.3.1 Silhouette and its simplified form

```text
FORMULA
  s(i)      = ( b(i) - a(i) ) / max( a(i), b(i) )              per-member silhouette
  a(i)      = ( 1 / (n_{c(i)} - 1) ) * Σ_{j in c(i), j != i} d(i,j)
  b(i)      = min_{l != c(i)} ( 1 / n_l ) * Σ_{j in l} d(i,j)
  S̄         = ( 1 / |M| ) * Σ_{i in M} s(i)                    mean over defined members M

  simplified silhouette (for large N):
  a'(i)     = d( i, centroid( c(i) ) )
  b'(i)     = min_{l != c(i)} d( i, centroid( l ) )
  s'(i)     = ( b'(i) - a'(i) ) / max( a'(i), b'(i) )

WHERE
  c(i)      segment containing member i
  n_l       size of segment l, integer >= 1
  d         declared distance, same scaling as the fit
  M         members with n_{c(i)} >= 2, since a(i) is undefined for singletons
```

The per-member formula is a normalised margin: how much closer the member is to its own group than to the nearest competing group, scaled so that the answer is comparable across members with different absolute distances. The scaling is what makes `S̄` aggregatable; the raw difference `b - a` is not, because a member in a sparse region has large `a` and `b` and would dominate the average.

The simplified form replaces the mean pairwise distance with the distance to the centroid. That substitution is exact only when all members of a group are equidistant from the centroid, which happens for points on a sphere around the centroid and essentially nowhere else. For an elongated group, `a'` underestimates `a` because the centroid sits in the middle of the elongation while members sit along it; the resulting `s'` is biased upward, and the bias is larger for more elongated groups. The practical consequence: simplified silhouette may be used to compare partitions in the *same* feature space, but its absolute value must not be compared with a literature value computed with the exact form.

```text
CONSTRAINT
  1. d is the metric used to fit the partition, not a different one chosen for evaluation.
  2. All features are scaled comparably; otherwise the largest-variance feature sets d.
  3. n_{c(i)} >= 2 for every member entering the average.
  4. k >= 2, otherwise b(i) does not exist.
  5. Features carrying missingness are imputed before d is computed, and the imputation is fixed.
```

```text
NUMERICAL EXAMPLE  (8 points in 2-D, DERIVED arithmetic, ILLUSTRATIVE coordinates)

  Segment X = {(1,1), (2,1), (1,2), (2,2)}     Segment Y = {(6,5), (7,5), (6,6), (7,6)}

  member (1,1):  within X distances   d to (2,1)=1.0000, (1,2)=1.0000, (2,2)=1.4142
                 a = (1.0000 + 1.0000 + 1.4142)/3 = 3.4142/3 = 1.13807
                 distances to Y      6.4031, 7.2111, 7.0711, 7.8102   sum = 28.4955
                 b = 28.4955/4 = 7.12388
                 s = (7.12388 - 1.13807)/7.12388 = 5.98581/7.12388 = 0.84025

  all eight members, by symmetry of the construction:
     (1,1) 0.84025   (2,1) 0.82166   (1,2) 0.82593   (2,2) 0.80113
     (6,5) 0.80113   (7,5) 0.82593   (6,6) 0.82166   (7,6) 0.84025

  S̄ = (0.84025+0.82166+0.82593+0.80113+0.80113+0.82593+0.82166+0.84025)/8
    = 6.57794/8 = 0.822242   -> 0.8222

INTERPRETATION
  Every member sits closer to its own group than to the other by a factor of roughly 6/1,
  so s is near the upper part of its range. The variation across members (0.801 to 0.840)
  comes entirely from the corner effect: corner points have b larger and a smaller.
  A mean of 0.82 here is not evidence that the partition is decision-useful; it is
  evidence that two well-separated boxes were split into two. Section 4.4 shows the same
  statistic on two concentric rings, where it goes negative for the correct partition.
```

#### 4.3.2 Calinski–Harabasz and Davies–Bouldin

```text
FORMULA
  W      = Σ_s Σ_{i in s} || x_i - μ_s ||²                       within-segment scatter
  B      = Σ_s n_s * || μ_s - μ ||²                               between-segment scatter, size weighted
  CH     = ( B / (k - 1) ) / ( W / (N - k) )                      Calinski-Harabasz

  S_s    = ( 1 / n_s ) * Σ_{i in s} || x_i - μ_s ||              mean radius of segment s
  DB     = ( 1 / k ) * Σ_s max_{l != s} ( S_s + S_l ) / || μ_s - μ_l ||     Davies-Bouldin

WHERE
  x_i    feature vector of member i in the evaluated space, numeric, dimension p
  μ_s    centroid of segment s, dimension p
  μ      population centroid, dimension p
  ||·||  Euclidean norm (CH and DB are defined with squared Euclidean)
```

`CH` is a ratio of a between-degree-of-freedom-normalised between-scatter to a within-degree-of-freedom-normalised within-scatter. Its algebra is identical to the one-way ANOVA `F` statistic for the segment label as a factor, which is the source of the most common misreading: `CH` is not a p-value and cannot be interpreted as evidence against a null hypothesis of no segments, because the segments were chosen to maximise exactly this quantity. The numerator degrees of freedom `k-1` corrects the mechanical rise of `B` with `k`; the denominator degrees of freedom `N-k` corrects the mechanical fall of `W`. Without those corrections, `CH` would increase monotonically in `k` all the way to `k = N`, where `W = 0` and `CH` is infinite.

`DB` is a worst-case average: for each segment it finds the most similar other segment by the ratio of combined radii to centroid distance, then averages that worst ratio over segments. Lower is better, which inverts the direction of the other indices. The `max` makes `DB` sensitive to a single pair of overlapping segments; a scheme with nine excellent segments and one overlapping pair scores badly even though the overlapping pair may be operationally irrelevant.

```text
CONSTRAINT
  1. Euclidean distances in an unscaled feature space make both indices an artefact of units.
  2. Both require k >= 2 and N > k; CH is undefined at k = N.
  3. Both assume roughly spherical, similarly sized segments; elongated or nested shapes are penalised.
  4. DB's max is over all other segments, so a single bad pair dominates.
  5. Both must be computed on the same feature matrix as the fit for the comparison to mean anything.
```

```text
NUMERICAL EXAMPLE  (same 8 points, DERIVED arithmetic)

  centroids:  μ_X = (1.5, 1.5)   μ_Y = (6.5, 5.5)   μ = (4.0, 3.5)

  W:  for X, each of the four points is at squared distance 0.25+0.25 = 0.5 from μ_X
      W_X = 4 * 0.5 = 2.0 ;  likewise W_Y = 2.0 ;  W = 4.0
  B:  ||μ_X - μ||² = (1.5-4.0)² + (1.5-3.5)² = 6.25 + 4.00 = 10.25
      ||μ_Y - μ||² = (6.5-4.0)² + (5.5-3.5)² = 6.25 + 4.00 = 10.25
      B = 4*10.25 + 4*10.25 = 82.0

  CH = (82.0/(2-1)) / (4.0/(8-2)) = 82.0 / 0.666667 = 123.0

  S_X = mean ||x_i - μ_X|| = 0.7071068 (each point at Euclidean distance sqrt(0.5))
  S_Y = 0.7071068
  ||μ_X - μ_Y|| = sqrt(10.25 + 4.00 + ... ) : (1.5-6.5)² = 25, (1.5-5.5)² = 16, norm = sqrt(41) = 6.40312
  DB  = (1/2) * [ (0.7071+0.7071)/6.40312 ] * 2 = 1.41421/6.40312 = 0.22086

INTERPRETATION
  CH = 123 says the between-scatter per degree of freedom is 123 times the within-scatter
  per degree of freedom. On its own that number is meaningless: it has no upper bound and
  no sampling distribution under the selection process that produced the partition.
  DB = 0.2209 has an interpretable reading: the average worst-case radius-sum is about
  22% of the centroid separation, so the segments are compact relative to their spacing.
  Both indices agree here because the constructed geometry is the case they were designed for.
```

#### 4.3.3 Agreement indices

```text
FORMULA
  contingency:   n_ij = |{members in reference class i and compared cluster j}|
                 a_i  = Σ_j n_ij     b_j = Σ_i n_ij     Σ_i a_i = Σ_j b_j = N

  pairs:         N_p     = C(N,2)                        total unordered pairs
                 same_r  = Σ_i C(a_i,2)                   pairs together in the reference
                 same_c  = Σ_j C(b_j,2)                   pairs together in the compared
                 both    = Σ_i Σ_j C(n_ij,2)              pairs together in both
                 RI      = [ both + (N_p - same_r - same_c + both) ] / N_p
                 ARI     = ( both - same_r*same_c/N_p ) / ( (same_r+same_c)/2 - same_r*same_c/N_p )

  F1 on pairs:   P = both/same_c     R = both/same_r     F1 = 2PR/(P+R)
  Jaccard:       J = both / ( same_r + same_c - both )

  purity:        purity = ( 1/N ) * Σ_j max_i n_ij
  NMI:           NMI = MI / norm(H_ref, H_cmp),  norm in { arithmetic mean, geometric mean, max }
                 MI = Σ_i Σ_j (n_ij/N) * log( (n_ij/N) / ((a_i/N)(b_j/N)) )

WHERE
  C(n,2) = n(n-1)/2, and C(n,2) = 0 for n < 2
  all counts are integers on the same grain; N_p is the pair population
```

`RI` counts agreeing pairs and divides by all pairs, so a random pair of partitions that both have many small clusters agrees on nearly every pair (they are mostly in different clusters in both), and `RI` approaches 1 without any real agreement. `ARI` subtracts the pair-agreement expected under the permutation model — sampling pairs without replacement from the two marginals — and divides by the maximum possible excess. That is why `ARI` has an expectation of 0 under independence and can be negative, which `RI` cannot.

`F1` on pairs is `ARI`'s cousin without the chance correction: it is the harmonic mean of the probability that a same-cluster pair in the compared partition is also same-cluster in the reference, and its converse. Because it lacks the chance correction, `F1` rises with `k` in the same way `purity` does, and the two are the pair-level and member-level versions of the same bias.

`purity` counts members whose cluster's majority reference class matches their own class. It ignores the number of clusters entirely, so a partition with `N` singleton clusters attains purity 1. It is a descriptive statistic for a fixed `k`, never a selection criterion.

`NMI` is mutual information divided by a normaliser. The three normalisers in common use — arithmetic mean of the entropies, geometric mean, and maximum — differ in value whenever the two partitions have different entropies, and `MI/max(H)` is bounded by `min(H_ref,H_cmp)/max(...)` rather than by 1. If two analysts report `NMI = 0.40` and `NMI = 0.40` on the same table, and one used `max` and the other the geometric mean, they will in fact have printed 0.3973 and 0.3988 here — close, but the divergence grows as the entropies diverge, and it reaches a factor of 2 when one partition has `k = 1`.

```text
CONSTRAINT
  1. Both partitions are on the same member set, same grain, same eligibility filter.
  2. Neither partition contains duplicate member keys; duplicates inflate same-cluster pairs.
  3. For ordinal labels, weighted kappa or ARI on the ordinal-aware variant is required.
  4. NMI's normaliser and logarithm base must be reported.
  5. RI and purity must never be compared across different k without stating k.
```

```text
NUMERICAL EXAMPLE  (2 x 2 contingency, DERIVED arithmetic)

  reference  i=1 (responders)  n_11 = 40   n_12 = 10     a_1 = 50
  reference  i=2 (non-respond.) n_21 =  5   n_22 = 45     a_2 = 50
                                b_1 = 45    b_2 = 55     N = 100

  pairs:  N_p    = C(100,2) = 4,950
          same_r = C(50,2) + C(50,2) = 1,225 + 1,225 = 2,450
          same_c = C(45,2) + C(55,2) =   990 + 1,485 = 2,475
          both   = C(40,2)+C(10,2)+C(5,2)+C(45,2) = 780 + 45 + 10 + 990 = 1,825
          expected = same_r * same_c / N_p = 2,450 * 2,475 / 4,950 = 1,225.0
          max      = (same_r + same_c)/2 = 2,462.5

  RI   = [1,825 + (4,950 - 2,450 - 2,475 + 1,825)] / 4,950
       = [1,825 + 1,850] / 4,950 = 3,675/4,950 = 0.742424
  ARI  = (1,825 - 1,225) / (2,462.5 - 1,225) = 600/1,237.5 = 0.484848
  P    = 1,825/2,475 = 0.737374    R = 1,825/2,450 = 0.744898
  F1   = 2*0.737374*0.744898 / (0.737374+0.744898) = 1.098513/1.482272 = 0.741117
  purity = (40 + 45)/100 = 0.85

  MI   = 0.4*log2(0.4/(0.5*0.45)) + 0.1*log2(0.1/(0.5*0.55))
       + 0.05*log2(0.05/(0.5*0.45)) + 0.45*log2(0.45/(0.5*0.55))
       = 0.4*0.830075 + 0.1*(-1.456)\ + 0.05*(-2.169925) + 0.45*(0.710)
       = 0.33203 - 0.14564 - 0.10850 + 0.31944 = 0.397313  bits
  H_ref = 1.000000 bits   H_cmp = 0.992774 bits
  NMI(arithmetic) = 0.397313 / 0.996387 = 0.398753
  NMI(geometric)  = 0.397313 / 0.996381 = 0.398756
  NMI(max)        = 0.397313 / 1.000000 = 0.397313

INTERPRETATION
  RI 0.742 looks like a decent agreement until you see that two unrelated 50/50 splits
  average about 0.495 agreement here; ARI removes that floor and reports 0.485. F1 reports
  0.741, almost exactly RI, because F1 has no chance correction either. Purity 0.85 flatters
  a 2-cluster partition by ignoring that 2 clusters can capture at most 2 classes.
  The three NMI values differ only in the fourth decimal here, which is precisely why
  "NMI 0.40" is an ambiguous statement: the ambiguity is small on balanced tables and
  large on unbalanced ones.
```

#### 4.3.4 Ranking indices for a discretised score

```text
FORMULA
  score of segment s  =  r_s                      ties for all members within a segment

  lift@q     = ( outcomes in top q fraction / members in top q fraction ) / ( total outcomes / N )
  capture@q  = outcomes in top q fraction / total outcomes
  Gini       = 2 * Σ_{g=1}^{m} ( cum_share_outcome_before_g + 0.5 * share_outcome_g ) * share_members_g  -  1
             = 1 - Σ_g ( share_members_g ) * ( 2 * cum_share_outcome_before_g + share_outcome_g )   (equivalent)
  KS         = max_g | cum_share_outcome_g - cum_share_nonoutcome_g |

WHERE
  m           number of ordered groups (deciles, or k segments ordered by r_s)
  share_*     group share of members / of outcomes, dimensionless, sums to 1 over g
  q           top fraction, 0 < q <= 1
```

The `Gini` formula is the trapezoid rule applied to the Lorenz curve: for each group, the area of the strip between the diagonal and the curve is `(cum_before + 0.5 * share) - 0.5 * share_members` ... written more carefully, the area under the Lorenz curve up to group `g` is `share_members_g * (cum_before + 0.5*share_outcome_g)`, and `Gini = 1 - 2 * Σ_g` of that. With `m` equal-sized groups and a perfect ordering (the first `1/m` of members contains all outcomes) the `Gini` equals `1 - 1/m`, which is why `Gini` values must never be compared across different bucket counts. Ten deciles cap the statistic at 0.9; twenty buckets cap it at 0.95; a continuous score has no cap below 1.

`KS` compares two cumulative distributions rather than one curve against the diagonal, so it is invariant to the base rate. That makes it the more honest statistic when the outcome prevalence changes between windows, and the less intuitive one, because a `KS` of 0.26 has no direct monetary reading.

```text
CONSTRAINT
  1. Groups are ordered by the score, descending, before the cumulative sums.
  2. Every member is in exactly one group; exclusions are reported, not silently dropped.
  3. Ties must be broken deterministically; with `k` segments, ties are within segment and the
     statistic is a step function with `k` steps, so it cannot be compared with a continuous model.
  4. The outcome window is identical across groups and has matured for all of them.
  5. `Gini` requires the bucket count to be stated; `KS` requires both cumulative curves' sizes.
```

```text
NUMERICAL EXAMPLE  (10 deciles, N = 20,000, R = 900 outcomes, DERIVED arithmetic)

  decile      1     2     3     4     5     6     7     8     9    10
  outcomes  210   160   120    95    80    65    55    45    40    30     sum = 900
  lift     2.333 1.778 1.333 1.056 0.889 0.722 0.611 0.500 0.444 0.333
  cum cap. 0.2333 0.4111 0.5444 0.6500 0.7389 0.8111 0.8722 0.9222 0.9667 1.0000

  STEP 1  Lorenz areas.  strip_g = share_members_g * (cum_share_outcome_before_g + 0.5*share_outcome_g)
          shares of outcomes are .2333 .1778 .1333 .1056 .0889 .0722 .0611 .0500 .0444 .0333
          cumulative before: 0, .2333, .4111, .5444, .6500, .7389, .8111, .8722, .9222, .9667

          strip_1  = 0.1*(0      + .1167) = 0.011667
          strip_2  = 0.1*(.2333  + .0889) = 0.032222
          strip_3  = 0.1*(.4111  + .0667) = 0.047778
          strip_4  = 0.1*(.5444  + .0528) = 0.059722
          strip_5  = 0.1*(.6500  + .0444) = 0.069444
          strip_6  = 0.1*(.7389  + .0361) = 0.077500
          strip_7  = 0.1*(.8111  + .0306) = 0.084167
          strip_8  = 0.1*(.8722  + .0250) = 0.089722
          strip_9  = 0.1*(.9222  + .0222) = 0.094444
          strip_10 = 0.1*(.9667  + .0167) = 0.098333
          area under Lorenz curve = 0.665000

  STEP 2  Gini = 2*area - 1 = 1.330000 - 1 = 0.330000

  STEP 3  boundary checks that must hold before the number is trusted
          uniform outcomes (90 per decile): area = 0.1*[.05+.15+.25+.35+.45+.55+.65+.75+.85+.95]
                                                 = 0.1*5.00 = 0.500  -> Gini = 0.000  (passes)
          perfectly ordered (900 in decile 1): area = 0.1*(0+.5) + 9*0.1*1.0
                                                 = 0.05 + 0.90 = 0.950 -> Gini = 0.900  (= 1 - 1/10, passes)

  STEP 4  KS.  non-outcomes per decile = 2,000 - outcomes
          = 1,790 / 1,840 / 1,880 / 1,905 / 1,920 / 1,935 / 1,945 / 1,955 / 1,960 / 1,970
          denominator N - R = 20,000 - 900 = 19,100

          cum outcome     .2333  .4111  .5444  .6500  .7389  .8111  .8722  .9222  .9667  1.0000
          cum non-outcome .0937  .1901  .2885  .3882  .4887  .5901  .6919  .7942  .8969  1.0000
          gap             .1396  .2211  .2560  .2618  .2501  .2211  .1803  .1280  .0698  0
          KS = max gap = 0.2618  (at decile 4)

  STEP 5  capture@20% = (210+160)/900 = 370/900 = 0.411111 ; lift@20% = 0.411111/0.20 = 2.0556

INTERPRETATION
  Gini 0.330 and KS 0.2618 describe the same ordering at different scalings; neither says
  anything about whether the segments are homogeneous. The maximum attainable Gini with ten
  equal deciles is 0.900, so 0.330 is 36.7% of the attainable range, not 33% of 1 — which is
  the error made by anyone quoting Gini against an absolute 0-to-1 scale. The three boundary
  checks in STEP 3 cost nothing and catch the two transcription errors (missing factor 2,
  inverted cumulative) that this example was originally written with.
```

Validity condition for 4.3: every variant above is a function of the partition, the feature space and the outcome window. Any two of those changing between two reported numbers makes the comparison a comparison of three things at once. The variants exist to answer different questions and are not interchangeable evidence about one underlying "segmentation quality".

### 4.4 Limits and counterexamples

Four limits are worth memorising, each with the case where the formula breaks rather than merely degrades.

**Limit 1: silhouette is a Euclidean-shaped criterion, and it fails on nested structure.** Take ten points on the unit circle and forty points on the circle of radius 4, evenly spaced, both centred at the origin. The correct partition — inner ring versus outer ring — is the one that any business would call correct, because the two rings have different behaviour profiles. Compute the mean silhouette of that partition.

```text
COUNTEREXAMPLE ARITHMETIC  (DERIVED, ILLUSTRATIVE coordinates)
  inner ring: radius 1, 10 points at angles 0, 36, 72, ... degrees
  outer ring: radius 4, 40 points at angles 4.5, 13.5, ... degrees

  RING PARTITION (the semantically correct one)
    inner member: mean distance to the other 9 inner points a = 1.4031
                  the maximum chord 2*cos(18 deg) = 1.9021 is the far point, not the mean;
                  1.4031 is the average over the 9 chords, which is why the naive
                  "use the farthest point" estimate overstates a by 36%
                  mean distance to the 40 outer points       b = 4.0627
                  s = (4.0627 - 1.4031)/4.0627 = 0.6547
    outer member: mean distance to the other 39 outer points a = 5.2209
                  mean distance to the 10 inner points       b = 4.0627
                  s = (4.0627 - 5.2209)/5.2209 = -0.2218
    S̄ = (10*0.6547 + 40*(-0.2218))/50 = (6.547 - 8.872)/50 = -0.0465

  ANGULAR PARTITION (left half vs right half, cutting both rings)
    sizes 25 and 25 (5 inner + 20 outer on each side)
    an inner member near angle 0 has a = 3.0098, b = 4.1161, s = 0.2688
    an outer member near angle 4.5 deg has a = 3.1981, b = 6.6995, s = 0.5226
    S̄ = 0.3177

  WHY the outer ring scores negative: its own mean pairwise distance (5.2209) exceeds the
  distance to the inner ring (4.0627), because a 40-gon of radius 4 has neighbours 0.628
  apart but also points almost 8 units away across the diameter. The mean over the whole
  ring dominates the nearest-other-segment distance even when the nearest points are far.
```

The negative outer-ring silhouette is the whole point, and it is not an artifact of the spacing I chose. Any ring thick enough in circumference relative to its radius has a large mean internal distance; when that internal distance exceeds the distance to the neighbouring ring, every member of the ring is told it belongs outside its own group. Silhouette is measuring "is my group more compact than the next group", and a thin annulus is not compact in the mean-pairwise-distance sense even when it is perfectly separable. The angular partition scores higher because slicing a ring into two arcs halves each arc's internal distance while keeping the two halves at comparable separation — the metric rewards a partition that splits a semantically single group, and penalises the partition that respects the generator of the data.

Two practical consequences follow. First, never use silhouette to reject a segmentation on ring, spiral or manifold-shaped data; use it only where the business hypothesis is "compact blobs of similar members", which is the case it can adjudicate. Second, if a segmentation is fitted with a non-Euclidean metric (cosine similarity on embeddings, for instance), silhouette computed with Euclidean distance is measuring a different geometry from the one that produced the partition, and its sign carries no information about the fit. Compute the index in the metric used to fit, or not at all.

**Limit 2: all internal indices fall when the data has no clusters but the pipeline forces `k`.** If the population is a single Gaussian blob, every `k >= 2` partition cuts through the middle of it. `W` falls with `k` because more centroids always fit better, `CH` is then governed by the degrees-of-freedom correction, and `DB` grows because the radii are large relative to centroid separation. A pipeline that reports only the winner of a `k` sweep on blob data will report a `k` of 2 or 3 with a positive silhouette and no indication that nothing was found. The discriminating check is a null-data simulation: generate data from the fitted single-component model, run the same sweep, and see what the index reports. If the real data's best index is inside the simulated distribution, the segmentation is not distinguishable from noise.

**Limit 3: agreement metrics cannot compare partitions of different populations, and they punish different `k` asymmetrically.** `ARI` between a 5-cluster partition and a 20-cluster partition of the same 1,000 members can be high or low but is not comparable with `ARI` between two 5-cluster partitions. The reason is structural: with 20 clusters, most pairs are in different clusters in that partition, so the `same_c` count is small, and the chance correction shrinks the attainable maximum. A stability study that re-fits with a different `k` and reports `ARI` is measuring a mixture of instability and granularity change.

**Limit 4: decision value is not additive across segments when there is a global constraint.** Formula 4.2 assumes unlimited capacity. If the contact centre can make 5,000 calls, the correct problem is a knapsack: rank segments by `r_s*G - c_s` per unit of capacity and take them in order until capacity binds. Under a binding capacity constraint the sum of segment values is an upper bound, and `ΔV` computed without the constraint overstates the achievable gain by the value of the excluded segments. The boundary case is instructive: with capacity exactly equal to the size of segment A in the worked example, the entire value of segments B and D is unreachable, and the segmentation's value reduces to the targeting term alone.

**Takeaway:** indices are shape detectors, not usefulness detectors; the only evaluation that survives contact with a business decision is one computed on the acted-on population against the real alternative.

