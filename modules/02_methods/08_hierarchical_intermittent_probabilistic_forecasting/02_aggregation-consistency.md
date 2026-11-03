# Aggregation Consistency and the Four Classical Methods

> Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 07 topic 05 (hierarchy and aggregation); Module 08 topic 01 (hierarchy structures)
> Version 2026.5

---

## 1. Why this topic exists

Three numbers for the same week appear in one planning meeting. Finance has a category forecast for
the quarter of 118,000 units. Supply planning has a stock-keeping-unit (SKU) by distribution-centre
(DC) plan summing to 111,200. Commercial has a customer-level number of 121,500. All three were
computed correctly, by competent people, from the same demand history. The meeting spends twenty
minutes on which is right, and the question is malformed: the three numbers answer three different
questions, and no arithmetic reconciles them without deciding which question the business is asking.

Module 07 topic 05 established why the gap exists. The level with the best signal is rarely the level
with the decision; aggregation buys less than the square-root heuristic promises; disaggregation is
not the inverse of aggregation. That topic presented the four classical strategies informally, as
four answers to "where is the information?", and stopped where the estimators begin. This topic takes
the next step, and it does three things the earlier treatment did not.

**It states coherence as a condition on a vector.** "Every parent equals the sum of its children" is
a sentence a committee can agree with and still violate. Written as a linear constraint it becomes a
matrix product, a number per bucket, and an audit that takes seconds. The difference between a slogan
and a constraint is the difference between a meeting that ends in agreement and a system that
enforces one.

**It shows the four methods are one method with four choices.** All of them produce a coherent
forecast by applying a linear map to the base forecasts; they differ only in which map. Once that is
visible, arguments between advocates become arguments about a matrix, and those can be settled with
evidence. It also explains a fact that surprises practitioners: top-down is not wrong because it is
old-fashioned. It is biased because its map has rank one, and better share estimation cannot give a
rank-one map a second direction.

**It separates "which is most accurate?" from "which should we use?".** The accuracy literature is
genuinely mixed, and the mixing is not noise. It is what you expect when the answer depends on
evaluation level, weighting, leaf intermittency, and history length. A team reading for a winner will
find whichever winner it looked for. A team reading for conditions will find a decision procedure.

Two failures make this urgent rather than academic. **Coherent-but-wrong** is the first.
Reconciliation guarantees the numbers add up and guarantees nothing about whether they are right. A
coherent set carries an authority an incoherent one does not, because it survives the arithmetic
check that a room instinctively performs. The most expensive forecast in a company is often the one
that reconciled beautifully and was wrong at the top. **The default setting** is the second. Most
planning systems implement one method, chosen when the system was configured, and expose it as "the
forecast" — usually bottom-up, because it needs no share model, or proration, because it is one line
of arithmetic. Neither was chosen for a reason the current data would support.

This topic sits between the structure (topic 01) and the estimator (topic 03). Topic 01 said what a
hierarchy is; topic 03 does minimum-trace reconciliation in detail. This file is the bridge, and its
job is to let you judge an estimator before you can build one.

> **Note:** Nothing here claims a coherent forecast is better than an incoherent one; the literature
> is not unanimous on that either. It claims coherence is a requirement you can state, test, and
> price, and that how you satisfy it is a modelling decision with measurable consequences.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** the coherence condition for a named hierarchy as `C y = 0` and as `y = S b`, and
   **compute** the coherence gap in units, as a ratio, and by node.
2. **Construct** the summing matrix `S` and the constraint matrix `C` for a tree of up to fifteen
   nodes, and **state** their dimensions.
3. **Explain** why bottom-up is coherent by construction and why top-down is not unbiased, using the
   rank of the linear map rather than the age of the method.
4. **Compute** by hand the bottom-up, top-down, and least-squares reconciled forecasts for a
   three-node hierarchy, and **show** they are three different coherent answers to one base.
5. **Distinguish** average-of-proportions, proportion-of-averages, and forecast proportions, and
   **state** what each assumes about the future mix.
6. **Describe** optimal reconciliation as a weighted least-squares projection onto the coherent
   subspace, and **state** precisely what its optimality claim depends on.
7. **Choose** among the four methods using forecastability by level, the level the decision needs,
   and the organisation's tolerance for a coherent-but-wrong number, and **design** a backtest that
   compares them at every level with common origins.

Outcomes 1, 3, 4, and 5 are exercised with arithmetic in section 12 and answered in section 13.

## 3. The core mental model

Hold two pictures: a plane, and a family of maps onto that plane. Everything else follows.

### 3.1 The coherence condition is a plane

Order every series into one vector `y`. For a total and two children, `Total = North + South` is one
linear equation in three unknowns, so the coherent vectors form a plane through the origin. Every
real set of base forecasts lies off it.

```text
   Panel A — the cross-section worth memorising

        ^
        |                     *  y_hat = (100, 60, 35)    base, incoherent
        |                    /|
        |                   / |
        |                  /  |  residual = (I - S P) y_hat
        |                 /   |  (what reconciliation discards)
        |                /    |
   =====+==============+=====+==============================>  coherent subspace
        |              /      ^        S = { y : Total = North + South }
        |             /       |
        |            /        +---- foot of a weighted projection
        |           /              (WLS, MinT): slides along the plane
        |          *               toward the more reliable nodes
        |         foot of the
        |         OLS projection
        |         (W = I): the perpendicular
        v

   Every point on the horizontal line is a coherent forecast. The base
   forecast is one point off the line. Reconciliation picks a point on
   the line, and WHICH point is the only thing the four methods disagree about.
```

Three readings repay attention. **The plane is not a target**; it contains infinitely many coherent
forecasts, including wildly wrong ones. Coherence constrains, in the way that "the budget must
balance" constrains and says nothing about whether the budget is good. **The residual is
information**: its length is how far the base forecasts were from agreement, and reporting it before
reconciliation distinguishes a rounding artefact from two processes with different assumptions. **The
projection is chosen, not given**: orthogonal projection drops straight down, a weighted one slides
sideways, and a wrong weight lands you at the wrong coherent point with perfect arithmetic.

### 3.2 All four methods are one map

Write `S` for the `n x n_b` summing matrix mapping the bottom-level vector `b` to the full vector
`y = S b`. Every method produces its reconciled forecast as `y_tilde = S P y_hat`, where `P` is
`n_b x n` and satisfies `P S = I`. That identity is what makes the result coherent: the map leaves
coherent forecasts unchanged. The methods differ only in `P`, and therefore in rank.

```text
   Method        What P does                                  Rank    Consequence
   -----------------------------------------------------------------------------------------
   Bottom-up     selects the bottom rows, ignores the rest   n_b     coherent; unbiased if
                                                                     the leaves are unbiased
   Middle-out    selects a middle level, derives both ways    k       coherent; inherits the
                                                                     middle level's bias
   Optimal       uses every level, weighted by W^-1           n_b     coherent; optimal only if
   reconciliation                                                    W is the true covariance
   Top-down      rank one: share vector times the top         1       coherent; biased in
                 forecast                                            general
```

Rank is capacity. A rank-one map cannot correct an error not aligned with its one direction, and no
amount of share estimation changes that. This is the whole bias argument.

### 3.3 Why rank one is the problem with top-down

```text
   Panel B — what a rank-one projection cannot do

   coherent subspace, drawn as the line of total forecasts
   ==================================================================>
        *          *          *          *          *
       T=90       T=95      T=100      T=105      T=110

   A fixed share vector p gives, for each total, exactly one point: y = p * T.
   As T varies the plan traces a single ray. Suppose the true total is 100 and
   the true mix is 50/50 while the shares are 62/38. Top-down produces
   (100, 62, 38): total exactly right, mix off by 12 points. There is no
   direction in which the method can move to fix the mix.
```

Better shares move the ray; they do not give it a second dimension. Forecast proportions improve
the ray by making it respond to known events, which is why they help and why they do not remove the
bias. Only a rank-`n_b` map can represent an arbitrary coherent forecast.

> **Note:** Rank is not accuracy. A full-rank method can be wrong in every direction. Rank is the
> capacity to represent a correction, not the knowledge of which correction to make.

### 3.4 Orientation table

Module 07 topic 05 presented the four methods as four answers to "where is the information?". That
presentation is assumed, not repeated. This table exists so the rest of the file has a reference.

| Method | Forecast at | Derive the rest by | Rank | Encodes the belief that |
|---|---|---|---|---|
| Bottom-up | Every leaf | Summing upward | `n_b` | Leaf dynamics are real; aggregates are consequences |
| Top-down, historical shares | The total | Allocating by past proportions | 1 | The mix repeats and the total is the only forecastable thing |
| Top-down, forecast proportions | The total | Allocating by proportions from base forecasts | 1 | The mix follows known events |
| Middle-out | One intermediate level | Summing up, allocating down | `k` | The middle has volume and specificity |
| Optimal reconciliation | Every level | Weighted projection onto the subspace | `n_b` | Every level informs; the weights carry reliability |

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Coherence | Every parent forecast equals the sum of its children at every level and bucket | Confused with accuracy | Boolean per node per bucket |
| Coherent subspace | The set of vectors satisfying every aggregation constraint; written `span(S)` | Treated as one target forecast | Subspace of dimension `n_b` |
| Incoherent forecast | A base vector violating at least one aggregation constraint | Assumed to be an error | Units per bucket |
| Summing matrix `S` | The `n x n_b` matrix mapping the bottom level to every series | Confused with a history table | Matrix, entries 0 or 1 |
| Constraint matrix `C` | The `(n - n_b) x n` matrix whose rows are the constraints, so `C y = 0` means coherent | Omitted because `S` is "enough" | Matrix, entries 0, 1, -1 |
| Bottom-level vector `b` | The `n_b` finest series that generate every other series | Confused with all stored series | Units per bucket |
| Base forecast | An independently produced forecast at each level, before reconciliation | Called "the forecast", implying final | Units per bucket |
| Reconciled forecast | A coherent forecast produced by a stated map from the base forecasts | Called "adjusted" without saying how | Units per bucket |
| Projection `P` | The `n_b x n` matrix in `y_tilde = S P y_hat`, with `P S = I` | Assumed unique | Matrix |
| Idempotence | `(S P)(S P) = S P`, the property that makes the map a projection | Treated as a synonym for unbiased | Boolean per method |
| Rank of the map | Dimension of the subspace the method can reach | Assumed to be `n_b` for all methods | Integer |
| Orthogonal projection | The projection minimising Euclidean distance, obtained when `W = I` | Assumed for every least-squares method | Method label |
| Residual of reconciliation | `(I - S P) y_hat`, the part of the base the map discards | Not reported, so the adjustment is invisible | Units per bucket |
| Coherence gap | Discrepancy between a parent base forecast and the sum of its children | Reported as a total only, hiding the node | Units, and ratio to parent |
| Bottom-up | Reconciling by summing leaf forecasts | Described as "the natural method" | Method label |
| Top-down | Reconciling by allocating a top forecast down shares | Described as obsolete rather than rank one | Method label |
| Middle-out | Reconciling from an intermediate level in both directions | Used without naming the level | Method label plus level |
| Optimal reconciliation | Weighted least-squares projection using an estimated error covariance | Treated as one method called "MinT" | Estimator plus `W` choice |
| Generalised least squares (GLS) | `(S' W^-1 S)^-1 S' W^-1`, optimal when `W` is the error covariance | Assumed optimal with an estimated `W` | Estimator |
| Ordinary least squares (OLS) | GLS with `W = I`; the orthogonal projection | Treated as a worthless baseline | Estimator |
| Weighted least squares (WLS) | GLS with diagonal `W` | Confused with MinT | Estimator |
| Minimum trace (MinT) | The family minimising the trace of the reconciled error covariance, with variants for estimating `W` | Used as the name of one estimator | Estimator variant |
| Covariance identifiability | Whether a covariance can be recovered uniquely; the reconciled error covariance lacks it, with rank deficiency `n - n_b` | Assumed for any sample size | Boolean plus rank deficiency |
| Shrinkage | Pulling an unstable sample covariance toward a structured target | Treated as a detail rather than a design choice | Estimator plus intensity |
| Historical proportions | Shares computed from past demand | Assumed to forecast the future mix | Ratio, sums to 1 per parent |
| Average of proportions | Mean over periods of each period's share; Gross-Sohl method A | Confused with proportion of averages | Ratio |
| Proportion of averages | Member total divided by parent total; Gross-Sohl method F | Confused with average of proportions | Ratio |
| Forecast proportions | Shares from base forecasts, normalised within each parent and multiplied down | Assumed to need a reconciled forecast | Ratio |
| Unbiasedness condition | `S G S = S` for a combination `G`; the identity top-down fails | Rarely stated when top-down is recommended | Matrix identity |
| Proration | Scaling every member by one factor to close a gap | Presented as reconciliation | Multiplicative factor |
| Additivity | A measure summing across children without conversion | Assumed for ratios, prices, and service levels | Boolean per measure |
| Level weighting | Weight given to each level when one score spans the hierarchy | Assumed equal, which is a strong choice | Vector of weights |
| Reconciliation register | Document naming method, weights, owner, and evidence per stream | Replaced by verbal convention | One row per stream |

Four rows repay expansion.

**Coherence versus accuracy.** Orthogonal properties that the profession routinely conflates. A
coherent set can be worse at every level than the incoherent set it replaced, and this happens
whenever the base forecasts are biased, because reconciliation spreads bias rather than removing it.
The test is not whether the numbers add up; it is whether the reconciled numbers beat the base at the
level where the decision is made.

**Rank versus weight.** Beginners assume the difference between methods is the weights. The
first-order question is rank: a rank-one map cannot represent a mix correction however its one
direction is chosen. This is why better shares do not rescue top-down, and better covariance does not
rescue a method that never saw the leaf data.

**Optimal versus estimated.** GLS is optimal when `W` is the true covariance of the base errors. In
every real application `W` is estimated, and the estimator is at least as uncertain as the forecasts.
The optimality claim describes an oracle, not a procedure.

**Historical versus forecast proportions.** Both say what the future mix will be. One says it will
resemble the past; the other says it will resemble what the level-specific models expect. Neither is
a statement about the world. Both are statements about which information the method may use, and the
choice only matters when the two disagree.
## 5. The mechanics, step by step

Four phases. Phase A states the requirement. Phase B decides what to trust. Phase C applies each
method to the same base forecasts. Phase D judges them. Each step names its input, output, and the
branch where the analysis can stop.

### Phase A — State the requirement

**Step 1. Fix the hierarchy and the ordering of the vector.** Write the node list root first, then
each level, then the leaves. The order is arbitrary but must be stable, because `S` and `C` are
defined relative to it and a silent reordering turns a correct reconciliation into a wrong one.
*Output:* an ordered node list with a level index per node. *Branch:* if the structure is grouped
rather than strictly hierarchical, stop and use the grouped machinery; a crossed structure has more
than one summing matrix and the projection is not unique without an extra constraint.

**Step 2. Build the summing matrix `S`.** One row per node, one column per leaf. A row holds a 1 in
every column whose leaf sits beneath that node, and 0 elsewhere. The root row is all ones; a leaf row
is a unit vector. *Check:* every column sums to the depth of its leaf plus one; every row sums to the
number of leaves beneath its node.

**Step 3. Build the constraint matrix `C`.** One row per aggregation constraint, parent minus the sum
of its children, so that `C y = 0` if and only if the vector is coherent. Verify `C S = 0`, which is
the check that the two matrices describe the same hierarchy. *Branch:* optional in production, where
`S` alone suffices, and mandatory in review, where the constraint form is what a non-specialist can
audit.

**Step 4. Measure the coherence gap on the base forecasts.** Report three numbers together.

| Measure | Formula | What it tells you | Unit |
|---|---|---|---|
| Gap in units | `y_parent - sum(y_children)` per parent | Where the disagreement sits | Units per bucket |
| Gap ratio | Gap divided by the parent base forecast | Whether it is material | Ratio |
| Gap norm | `norm(C y_hat)` over the whole vector | How far the base is from the subspace | Units |

The gap is usually concentrated: in a nine-node hierarchy with a 4% total gap, one region commonly
carries most of it, because that region's planner held a different view of a promotion. Report it by
node or the concentration is invisible. *Branch:* if the gap is under 0.5% and uniformly spread,
proration is defensible and the rest of this procedure is over-engineering. If it is concentrated,
find the cause first, because a concentrated gap is a disagreement between people and reconciliation
will average it away rather than resolve it.

**Step 5. Decide whether coherence is required, at which grain, and for whom.** Coherence is required
when a number is consumed by an accounting, contracting, or capacity process: an order equals the sum
of its lines, a schedule equals the sum of its work orders. It is optional when the number is
consumed by a judgement, provided the bridge is documented. *Branch:* if a stream requires coherence
but nobody can say at which level, the requirement is not yet specified and no method can satisfy it.

### Phase B — Decide what to trust

**Step 6. Measure forecastability at every level, not just the level you plan to use.** Reuse the
diagnostics from Module 07 topic 05, and read the shape across levels rather than the value at any
one.

| Level | Series | Median non-zero observations | Zero share | Reading |
|---|---:|---:|---:|---|
| Total | 1 | 104 | 0.00 | Smooth, seasonal, one series |
| Region | 4 | 104 | 0.00 | Smooth, but four short series |
| Family x region | 12 | 40 | 0.02 | Seasonal signal recoverable |
| SKU x region | 48 | 22 | 0.31 | Intermittent at the tail |
| SKU x DC | 396 | 9 | 0.58 | Most weeks carry no information |

The gradient is the argument: whatever method you choose, it is choosing which row to believe, and
the rows differ by a factor of ten in information per series. *Branch:* if the leaf row has a zero
share above 0.50, no method that trusts the leaves as its primary source will be stable.

**Step 7. Name the candidates honestly, including those your tooling does not offer.** Four, plus the
option of not reconciling. Most organisations compare two and conclude; the comparison costs one
backtest and is the difference between a defensible choice and a default.

**Step 8. Write down the projection each candidate implies.** This step is the one most likely to be
skipped and most likely to change the conclusion.

- **Bottom-up:** `P = [0 | I]`, selecting the leaf rows. Rank `n_b`.
- **Top-down with share vector `p`:** `P = p e_root'`, where `e_root` selects the root. Rank 1.
- **Middle-out at level `k`:** select the level-`k` rows, derive up by summing and down by
  proportions. Rank equal to the number of nodes at that level.
- **Optimal reconciliation:** `P = (S' W^-1 S)^-1 S' W^-1`. Rank `n_b`.

*Check:* `P S = I` fails for top-down unless there is a single leaf. That failure is the bias stated
in algebra, and it is why the unbiasedness condition `S G S = S` divides the methods in the
literature.

**Step 9. Test the belief each projection encodes.**

| Belief | Test | Evidence that falsifies it |
|---|---|---|
| Leaf dynamics are real (bottom-up) | Leaf zero share, average inter-demand interval, leaf error against a naive benchmark | Leaf forecasts fail to beat seasonal naive |
| The mix repeats (top-down, historical) | Share range, trend, year-on-year persistence | Share range above 0.10 on a member averaging 0.15 |
| The mix follows events (top-down, forecast) | Forecast shares against realised shares in held-out periods | Forecast shares no better than historical shares |
| The middle has volume and specificity (middle-out) | Observations and zero share at the chosen middle level | The middle is as sparse as the leaves |
| Every level informs (reconciliation) | Stability of the estimated weights across windows | Weights changing sign or magnitude between folds |

### Phase C — Apply the four methods

**Step 10. Bottom-up.** Forecast every leaf, then set each parent to the sum of its children.
Arithmetic is addition; there is no parameter beyond the leaf models. Coherent by construction, and
unbiased whenever the leaves are, because the expectation of a sum is the sum of expectations.
*Failure:* the leaves are the noisiest data in the hierarchy, and their errors accumulate. Positively
correlated errors make the aggregate nearly as bad as the sum of the individual errors; negatively
correlated errors produce cancellation and can make bottom-up win outright. Which holds is an
empirical question answered by residual correlation, not by intuition.

**Step 11. Top-down with historical proportions.** Forecast the root, compute a share vector, and
allocate down level by level. Two variants matter and they diverge under drift:

| Variant | Formula for member `j` with periods `t` | Weights | Behaviour for a growing member |
|---|---|---|---|
| Average of proportions (method A) | `p_j = (1/T) SUM_t (y_jt / y_t)` | Every period equally | Under-allocates; early low periods count as much as late high ones |
| Proportion of averages (method F) | `p_j = (SUM_t y_jt) / (SUM_t y_t)` | Every unit equally | Closer; high-volume periods dominate |

Coherent, rank one, biased in general, because shares are estimated from data that already contains
the parent's history, so shares and parent forecast are not independent. The bias is small when the
mix is stable and large exactly when the mix moves. *Failure:* new items have a historical share near
zero and are starved at launch; promoted items are under-allocated by the size of the promotion;
discontinued items inflate their share until removed from the base.

**Step 12. Top-down with forecast proportions.** Produce base forecasts at every level, compute each
member's share as its base forecast divided by its parent's base forecast, and multiply the shares
down from the root. The share for a leaf at depth `K` is the product of within-parent ratios along
its path. Coherent; rank one, so still biased in general; responsive to known events because the base
forecasts already encode them. *Failure:* it needs base forecasts at every level, which is half the
cost of full reconciliation, and it inherits whatever the intermediate models believe.
Implementations differ in whether they renormalise shares so they sum to one at each parent; if they
do not, an incoherent base produces shares summing to something other than one and the allocation
will not add up. Check this before trusting a package default.

> **Note:** Forecast proportions are often described as fixing top-down. They improve it; they do not
> remove the rank-one limitation. A method that can move only along one direction cannot correct a
> mix error orthogonal to that direction, however well the direction is chosen.

**Step 13. Middle-out.** Choose a middle level `k`, forecast at `k`, aggregate upward by summing, and
disaggregate downward by proportions. The choice of `k` is the whole method, and it is a judgement
with three inputs: observation counts at each level, the level with a credible owner, and the level at
which events are known. Coherent; rank equal to the number of nodes at `k`; it inherits the middle
level's bias upward and the share method's bias downward. *Failure:* the middle level is frequently
unowned. A company planning centrally at category and locally at SKU may have nobody accountable for
family by region, and a level with no owner degrades silently. *Branch:* if no middle level has both
data and an owner, middle-out is a name for an unowned process and reconciliation is the better frame.

**Step 14. Optimal reconciliation.** Produce base forecasts at every level, choose a weight matrix
`W`, and compute

```text
   y_tilde = S (S' W^-1 S)^-1 S' W^-1 y_hat
```

which is the weighted least-squares projection of `y_hat` onto `span(S)`. With `W = I` it is OLS;
with `W` diagonal it is WLS; with `W` the covariance of the base errors it is the GLS estimator that
minimises the trace of the reconciled error covariance — the sense in which it is called optimal.

State the claim precisely, because the imprecise version causes the trouble. The estimator minimises
expected total error **conditional on `W` being the true error covariance**. Three problems follow.
**The covariance is not identifiable**: the covariance of the reconciled errors has rank deficiency
`n - n_b`, so it cannot be recovered even in principle, and what is estimated instead is the base
error covariance, which carries its own error. **The estimate may not exist**: a sample covariance
of `n` series from `T` observations is singular when `n > T`, which is the normal case in a hierarchy
of several hundred series with two years of monthly data. **The estimate may exist and be useless**:
with `n` close to `T` it is ill-conditioned, the inverse amplifies noise into weights, and the result
can be worse than OLS, which assumes nothing.

The responses are the `W` variants — identity, variance scaling, structural scaling, sample, and
shrinkage toward a structured target. Topic 03 does them properly. What you need here is the
judgement: **"optimal" describes an oracle, and the method you run is an approximation that can lose
to a method that never claimed optimality.**

> **Note:** A useful discipline is to run OLS, WLS with structural scaling, and one shrinkage
> variant, and report all three. If they agree, the choice of `W` did not matter for this data. If
> they disagree, you have measured the estimation uncertainty the word "optimal" conceals.

### Phase D — Judge

**Step 15. Backtest all candidates at every level with common origins.** Hold four things fixed:
origin dates, horizon, evaluation buckets, and the base forecasts. The last is the one most often
violated; comparing bottom-up built on exponential smoothing with a reconciliation built on ARIMA
measures the base models, not the reconciliation.

**Step 16. Choose a metric and a level weighting, and state both.** This is where comparisons become
incomparable. A metric at the leaf and a metric at the root answer different questions, and a score
weighting levels equally is a third. M5 weighted its twelve levels equally, a defensible choice for a
competition and not the weighting any single business would choose.

**Step 17. Compare each method against the base forecast, not only against each other.** "Which of
the four is best" is less useful than "did reconciliation earn its keep". Report the error of the
reconciled forecast against the base at each level. A method that improves the aggregate by 4% and
worsens the leaf by 2% has not obviously improved anything, and this is the only comparison that
cannot be gamed by choosing a flattering level.

**Step 18. Decide on three criteria, and write the decision down.**

| Criterion | Question | What it favours |
|---|---|---|
| Forecastability by level | Which level's series actually carry signal? | Bottom-up if leaves are dense; top-down or middle-out if sparse |
| Decision level | Which level does the commitment require? | The method with the smallest error at that level, not the best overall |
| Tolerance for coherent-but-wrong | Can the organisation absorb a number that adds up and is wrong? | Coherence where contracted or audited; documented incoherence where the number is a judgement |

The third criterion is the one nobody writes down and everybody uses. Coherence distributes a wrong
top-level number into every descendant and makes it harder to challenge, because each descendant now
agrees with its parent. Where a wrong number is expensive and an inconsistent one is cheap,
publishing an incoherent set with a documented bridge is a legitimate engineering choice.

**Step 19. Publish the register.** One row per planning stream: level, bucket, method, `W` choice,
owner, backtest result, and the rule for what happens when a planner overrides a node. The register
ends the argument in section 1, because it makes "which number is right" answerable: the one that
governs your decision, from the stream that owns it, measured where you consume it.

## 6. Worked example

Halloway Surgical distributes sterile procedure packs through two regions and six stock-keeping
units. The figures are constructed for teaching and are internally consistent; the arithmetic is
shown so you can check it. The unit is eaches per month and the horizon is one month.

### 6.1 The hierarchy

```text
                        TOTAL  (1 node)
                       /            \
                  NORTH             SOUTH        (2 nodes)
                 /  |  \           /  |  \
               N1  N2  N3        S1  S2  S3      (6 leaves)

   n = 9 nodes.   n_b = 6 leaves.   constraints = 3.
   Coherent subspace dimension = 6 inside a 9-dimensional space.
```

A strict tree: one parent per node, no crossed dimensions. Replenishment happens weekly at SKU by DC
and is outside this example.

### 6.2 Base forecasts at every level

Five models ran independently, one per level, each with its own features and history. That is how a
real organisation produces base forecasts, and it is why they do not add up.

| Node | Level | Base forecast (eaches) | Sum of children | Gap |
|---|---|---:|---:|---:|
| Total | 0 | 1,000 | 960 | +40 |
| North | 1 | 560 | 570 | -10 |
| South | 1 | 400 | 420 | -20 |
| N1 | 2 | 300 | — | — |
| N2 | 2 | 180 | — | — |
| N3 | 2 | 90 | — | — |
| S1 | 2 | 150 | — | — |
| S2 | 2 | 160 | — | — |
| S3 | 2 | 110 | — | — |

Three gaps in different directions. The gap norm is `sqrt(1600 + 100 + 400) = 45.8` units, and 4.0%
of the base total: material, because at a 30 euro unit cost and a 22% annual holding cost a 40-unit
error on a monthly total is real money, on one product among hundreds.

### 6.3 The historical shares

Twelve months of history give these proportions of historical averages: North 0.58 and South 0.42 of
the total; within North, N1 0.50, N2 0.32, N3 0.18; within South, S1 0.36, S2 0.39, S3 0.25. Module 07
topic 05 showed that the average-of-proportions variant gives North a slightly lower share in a
growing region, for the same reason it always does: the early months get equal weight.

### 6.4 Bottom-up

The leaves are forecast directly at 300, 180, 90, 150, 160, and 110. North becomes 570, South 420, and
the total 990 — ten units below the base total and forty below where top-down will put it. Bottom-up
does not know the base total exists. That is its strength, because it cannot be contaminated by a
top-level model's error, and its weakness, because it discards whatever that model knew.

### 6.5 Top-down with historical shares

Take the total of 1,000 as authoritative. North is `1,000 x 0.58 = 580`; South is `420`. Within North:
`580 x 0.50 = 290.0`, `580 x 0.32 = 185.6`, `580 x 0.18 = 104.4`. Within South: `420 x 0.36 = 151.2`,
`420 x 0.39 = 163.8`, `420 x 0.25 = 105.0`. Both rows sum exactly to their parent, so the set is
coherent. Compare with the base: N1 was 300 and is now 290; S3 was 110 and is now 105. The method has
moved every leaf by the amount the shares imply, which is another way of saying it has discarded
every leaf-specific signal.

### 6.6 Top-down with forecast proportions

Compute shares from the base forecasts, normalising within each parent, then multiply down.

| Node | Share calculation | Share | Forecast at total 1,000 |
|---|---|---:|---:|
| North | `560 / (560 + 400)` | 0.58333 | 583.33 |
| South | `400 / 960` | 0.41667 | 416.67 |
| N1 | `0.58333 x (300/570)` | 0.30702 | 307.02 |
| N2 | `0.58333 x (180/570)` | 0.18421 | 184.21 |
| N3 | `0.58333 x (90/570)` | 0.09211 | 92.11 |
| S1 | `0.41667 x (150/420)` | 0.14881 | 148.81 |
| S2 | `0.41667 x (160/420)` | 0.15873 | 158.73 |
| S3 | `0.41667 x (110/420)` | 0.10913 | 109.13 |

The region shares were renormalised from 0.5600 and 0.4000, which sum to 0.96, to 0.58333 and
0.41667. Without that renormalisation the allocation would not sum to the parent and the result would
be incoherent in a way that looks like a rounding error. This is the trap flagged in step 12. The
within-region shares needed no adjustment here only because the base SKU forecasts happen to sum to
their region base, which is not guaranteed.

### 6.7 Middle-out at the region level

Choose region as the middle: twelve monthly observations a year, a named regional planner, and the
level at which promotional calendars are set. Adopt 560 and 400. The total becomes 960. N1 is
`560 x (300/570) = 294.74`, N2 176.84, N3 88.42, summing to 560.00; S1 is `400 x (150/420) = 142.86`,
S2 152.38, S3 104.76, summing to 400.00. The total is lower than either other method: middle-out
inherits the region-level belief that the total is 960 and propagates it in both directions. If the
region models are good that is a feature; if they missed a national promotion, the error is now in
every node.

### 6.8 Optimal reconciliation on a three-node subtree

Full reconciliation of nine nodes needs matrix inversion, which is topic 03's business. The
three-node closed form is hand-computable and shows the mechanism. Take `Total = North + South` with
base forecasts 1,000, 560, and 400, and `W = diag(400, 900, 1600)`, read as error variances in units
squared. The gap is `d = 40` and `sum(w) = 2,900`. Each node moves by `d x w_i / sum(w)`, with the
parent moving in the opposite direction to close the gap:

| Node | Calculation | Base | Reconciled |
|---|---|---:|---:|
| Total | `1,000 - 40 x 400/2,900` | 1,000 | 994.48 |
| North | `560 + 40 x 900/2,900` | 560 | 572.41 |
| South | `400 + 40 x 1,600/2,900` | 400 | 422.07 |

`572.41 + 422.07 = 994.48`, coherent to rounding. The total, with the smallest variance, moves 5.5
units; North, with three times the variance, moves 12.4; South, with four times, moves 22.1. The
method distributed the gap in proportion to how much it distrusts each number. Change one weight to
`w_T = 10,000` and the total barely moves, pushing almost all of the gap into the regions. The
weights are the model, and the data does not supply them.

### 6.9 Four coherent answers to one incoherent base

| Node | Base | Bottom-up | Top-down (historical) | Middle-out | Reconciled (subtree) |
|---|---:|---:|---:|---:|---:|
| Total | 1,000 | 990 | 1,000 | 960 | 994.48 |
| North | 560 | 570 | 580 | 560 | 572.41 |
| South | 400 | 420 | 420 | 400 | 422.07 |
| N1 | 300 | 300 | 290.0 | 294.74 | not computed |
| N2 | 180 | 180 | 185.6 | 176.84 | not computed |
| N3 | 90 | 90 | 104.4 | 88.42 | not computed |
| S1 | 150 | 150 | 151.2 | 142.86 | not computed |
| S2 | 160 | 160 | 163.8 | 152.38 | not computed |
| S3 | 110 | 110 | 105.0 | 104.76 | not computed |

The spread in the total is 40 units, 4% of the base. The spread in N3 is 16 units, 18% of its base.
The smaller the node, the larger the relative disagreement, which is the arithmetic consequence of
closing a fixed gap across members differing threefold in size.

Now add a promotion on N3 that lifts its demand by 60%. The leaf model, which sees the calendar,
forecasts 144 instead of 90. The historical share does not move, so top-down allocates 104.4 and is
27.5% short. Forecast proportions allocate about 134.6, because the leaf forecast raises N3's
within-North share to `144/(300+180+144) = 0.2308`. Reconciliation and bottom-up fix it directly. The
rank argument of section 3.3, in one number.

### 6.10 What this comparison does and does not tell you

It does not tell you which method is accurate: there is no realised demand here, only base
forecasts. It tells you the size of the decision space — four defensible coherent answers spanning
4% of the total from identical inputs, which is larger than most forecast improvement programmes
deliver. It also shows where the sensitivity sits: 4% at the total, 18% at N3.

## 7. Code, query, or calculation

One hand computation, because the point is the algebra rather than the library.

```text
Three-node hierarchy: Total = A + B.  Base forecasts  y = (100, 60, 35).

S = [1 1; 1 0; 0 1]      rows: Total, A, B;   columns: A, B

BOTTOM-UP       y_bu  = S (60, 35)'  = (95, 60, 35)          total 95
TOP-DOWN        shares p = (0.62, 0.38)
                y_td  = p * 100      = (100, 62, 38)        total 100
OPTIMAL, OLS    beta  = (S'S)^-1 S'y = (1/3)(185, 110) = (61.667, 36.667)
                y_ols = S beta      = (98.333, 61.667, 36.667)   total 98.33
OPTIMAL, WLS    W = diag(400, 900, 1600),  gap d = 100 - 60 - 35 = 5
                each node moves by d * w_i / sum(w),  sum(w) = 2900
                Total  100 - 5*400/2900  =  99.310
                A       60 + 5*900/2900  =  61.552
                B       35 + 5*1600/2900 =  37.759
                check   61.552 + 37.759  =  99.311
```

Expected output: four coherent vectors and four different totals — 95, 100, 98.33, and 99.31. The
comparison of interest is not which is right, since none can be judged without an outcome, but that
they differ by up to 5% of the total on a three-node hierarchy with a 5-unit gap.

Three checks before trusting any implementation. **`P S = I`:** multiply the projection by the
summing matrix and confirm the identity. For top-down it holds only with one leaf; a package
returning a top-down `P` that passes on a multi-leaf hierarchy is not doing top-down. **Coherence of
the output:** compute `C y_tilde` and confirm zero to tolerance; results coherent to six decimals are
fine for planning and not for invoicing. **Round-trip:** feed a coherent vector in and confirm it
comes back unchanged; a method that alters an already-coherent forecast is not a projection.

## 8. Data, standards, and regulatory context

Coherence is not only a statistical property. It is a property of identifiers, unit conversions, and
calendars, and those are governed by standards. A forecast can be arithmetically coherent and still
fail an audit because the arithmetic was done in the wrong units or across a calendar in which the
parents do not contain their children.

### 8.1 Three hierarchies, not one

Every planning system carries at least three hierarchical structures, and reconciliation usually
addresses one of them.

| Structure | What nests | Failure when ignored | Who owns the standard |
|---|---|---|---|
| Product | Item in family in category in department | A new item enters a category mid-year and every history-based share is wrong | GS1 for identifiers; the retailer or manufacturer for the taxonomy |
| Location | Site in region in country in theatre | A site is reassigned between regions and the historical shares double-count it | GS1 for identifiers; the company for the rollup |
| Time | Week in month in quarter in year | Weeks do not sum to months; a weekly plan cannot equal a monthly one | ISO for weeks; retail and fiscal convention for 4-4-5 |

A coherent forecast must be coherent on all three at once. A SKU-by-DC-by-week plan that reconciles
across products but not across weeks is coherent in the sense of section 3 and unusable in the sense
of section 1.

### 8.2 Product identifiers and pack hierarchies

The GS1 system supplies the identifiers that make product hierarchies joinable. A Global Trade Item
Number (GTIN) identifies a trade item at a given packaging level; distinct GTINs are assigned to the
each, the case, and the pallet, so the packaging hierarchy is a genuine hierarchy with its own summing
matrix. The Global Product Classification (GPC) supplies the category taxonomy above it. The Global
Location Number (GLN) does the same for sites, and the Serial Shipping Container Code (SSCC) for
logistic units.

The standard constrains identification, not aggregation. It guarantees that the each, the case, and
the pallet can be told apart; it does not guarantee that the conversion factor between them is an
integer, that it is stable, or that a planner has not created a case pack of seven because a customer
asked. Pack-quantity changes are a common source of silent incoherence: the history records twelve
cases of twelve, demand planning reports 144 eaches, and the case pack has since become ten. The
numbers reconcile in eaches and not in cases, and both are "the forecast".

### 8.3 Units of measure

GS1 maintains code lists for units of measure, and most enterprise systems carry a base unit per
item with conversions to purchasing, stocking, and selling units. Three rules follow for
reconciliation.

**Reconcile in the base unit, report in the consumption unit.** Convert once, at the boundary.
Reconciling in cases and then rounding to pallets produces a vector that is coherent before rounding
and not after.

**Rounding is not neutral.** If the leaf forecasts are integers and the allocation produces
fractions, some rounding rule must close the residual, and the rule redistributes error. Largest-
remainder rounding is the usual choice because it minimises the total absolute change; rounding each
member independently does not.

**Unit changes are structural breaks.** A change of base unit rewrites the whole history in a way
that is arithmetically exact and causally wrong, because the physical item did not change.

### 8.4 Calendars: where coherence fails by construction

The ISO 8601 week calendar defines weeks starting Monday, with 52 or 53 weeks in a year. Weeks do not
nest into Gregorian months: a month contains four or five partial weeks, and December can contain
parts of five different ISO weeks. Any pair of statements "the month equals the sum of its weeks" and
"the week is an ISO week" cannot both hold.

Retail solves this with the 4-4-5 calendar, in which each quarter is thirteen weeks arranged as four,
four, and five, and each year has 52 weeks plus an occasional 53rd. The National Retail Federation
(NRF) publishes this convention, and it is a convention rather than a law: it makes weeks nest into
months exactly, at the cost of months that are not calendar months. Fiscal years in many companies
use the same construction for the same reason.

Three consequences for reconciliation.

| Choice | What is coherent | What is lost | Where it bites |
|---|---|---|---|
| ISO weeks, ISO months | Nothing across the week-month boundary | — | Weekly plan cannot sum to the monthly financial plan |
| 4-4-5 periods | Weeks into periods, periods into years | Alignment with calendar months | Reporting to a parent company on calendar months |
| Calendar months, weekly buckets derived | Months into quarters and years | Weeks into months | Operational scheduling, where weeks are the unit of work |

The practical resolution is to reconcile within one calendar and translate across calendars with a
stated, documented bridge — a day-weighted split of the month into weeks, applied consistently, with
the residual assigned by rule. The bridge is not reconciliation in the sense of section 3; it is a
second projection, and it should be named in the register as one.

### 8.5 Financial reporting requires an explicit reconciliation

IFRS 8, *Operating Segments*, requires an entity to report segment revenue, profit or loss, assets,
and liabilities, and to provide reconciliations of those totals to the amounts recognised in the
financial statements. The reconciliation is a disclosure requirement, not an option. The standard is
issued by the International Accounting Standards Board (IASB) and is binding for entities within its
scope.

Read that requirement as a design precedent. A regulator concluded that a segment total presented
without a reconciliation to the audited whole is not decision-useful, because the reader cannot tell
whether the difference is a definitional artefact or an error. The same logic applies to a demand
plan: a category number presented without a bridge to the SKU plan invites exactly the meeting
described in section 1.

What IFRS 8 does not do is constrain the forecasting method. It requires that the numbers be
reconcilable and disclosed, not that a particular projection produced them. The choice among the four
methods is a management decision; the obligation to explain the bridge is not.

## 9. Industry practice and case evidence

The evidence base is thinner than the literature's confidence suggests. Forecast reconciliation has
a substantial peer-reviewed literature and very little public documentation of what firms actually
run. What follows separates the two.

### 9.1 What the literature establishes

The reconciliation literature has a clear shape. The minimum-trace (MinT) estimator of
Wickramasuriya, Athanasopoulos, and Hyndman (2019) made optimal reconciliation practical by supplying
a family of covariance estimators that work when the sample covariance is singular — which, as
section 5 noted, is the normal case. Their paper, published in the *Journal of the American
Statistical Association*, is the reference point for the estimator that topic 03 develops.

The comparative literature that followed is genuinely mixed, and the mixing has structure. A 2024
review in the *International Journal of Forecasting* by Athanasopoulos, Hyndman, Kourentzes, and
Panagiotelis surveys the field and reports the recurring finding: no method dominates across
hierarchies, and the ranking depends on the data's structure, the evaluation level, and the choice of
weights. That is not a failure of the literature. It is the literature reporting that the decision is
conditional, which is why section 5 spends a phase on deciding what to trust rather than on choosing
a winner.

Three regularities recur often enough to be worth stating as expectations rather than laws.

**Reconciliation usually helps at the aggregate and less often at the leaf.** The aggregate is where
the information from all levels can be combined; the leaf is where the base forecast already had the
most specific signal and reconciliation can only disturb it. This is the pattern behind the
evaluation-level dependence in section 9.2.

**Simple weights are competitive more often than the theory predicts.** Ordinary least squares, which
assumes nothing about the covariance, is a frequent strong baseline, and estimated covariance
matrices frequently lose to it out of sample. The reason is the estimation error of section 5, not a
flaw in the theory.

**Top-down is rarely the best method and sometimes is.** When the shares are stable and the leaves
are intermittent, a stable share vector estimated from many periods can beat a leaf model fitted to
nine non-zero observations. That is the condition in which the rank-one limitation costs less than
the noise it avoids.

### 9.2 The M5 evidence

The M5 forecasting competition, run in 2020 on Walmart store-department data, is the largest public
hierarchical forecasting evaluation and the most cited evidence on level weighting. The data comprise
daily sales for 3,049 products across 10 stores and 3 states, organised into twelve aggregation
levels, with 42,840 series at the most detailed level. The evaluation weighted the twelve levels
equally in the primary metric.

Two lessons transfer. **Equal weighting across levels is a choice with consequences.** It gives the
most detailed level, which has the most series, the same total weight as the aggregate, which has one.
A company weighting by decision value would weight very differently, and would produce a different
ranking. **The winning entries combined multiple methods rather than finding one.** The competition's
top results came from ensembles, which is consistent with the literature's finding that the
reconciliation choice is conditional: if no single projection is best everywhere, averaging
projections is a sensible response.

Do not read M5 as evidence that any particular method wins. Read it as evidence that the evaluation
design determines the answer, and that twelve levels equally weighted is one defensible design among
many.

### 9.3 What firms appear to do

The public record on corporate practice is mostly inference, and should be labelled as such.

| Practice | Evidence status | Reasoning |
|---|---|---|
| Bottom-up as the default in replenishment | Inference from system design and job descriptions | Replenishment needs a number per SKU per location; if the system forecasts there, summation upward is the cheapest coherent answer |
| Proration rather than reconciliation in finance-facing plans | Inference from the prevalence of top-down allocation in enterprise planning tools | Proration is one line of arithmetic and needs no covariance estimate |
| Middle-out where a category team exists | Inference from organisational structure | The level with an owner becomes the level with the forecast, for reasons of accountability rather than statistics |
| Reconciliation confined to the statistical centre of excellence | Inference from the vendor and conference literature | The estimator is technically demanding; the organisational change it implies is larger than the method |

No widely documented public case of a reconciliation failure exists, in the sense of a named company
and an audited cost. Companies do not publish their forecasting errors. Treat that absence as a
limitation of the evidence, not as evidence that the failures are rare — the mechanism in section 10
is common, and it is simply not disclosed.

## 10. Failure modes and red flags

Each row names a failure, the signal visible in a meeting or a dataset, the mechanism, the
consequence, and the mitigation. The red flag column is the one to remember: most of these are
detectable in ten minutes and are detected in almost none.

| # | Failure mode | Red flag in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Silent incoherence | Two reports for the same period differ by 2–5% and nobody can say why | Each system computes independently and no constraint is enforced | Every downstream sum is wrong; trust erodes | Compute `C y` per bucket and publish the gap by node |
| 2 | Proration mistaken for reconciliation | The adjustment is described as "spreading the difference" | One multiplicative factor applied to every member | Distorts members by the same percentage regardless of reliability | Replace with a stated projection, or label proration as such |
| 3 | Top-down starving new items | New SKUs forecast near zero for their first quarter | Historical share is zero or near zero | Stockouts at launch, lost sales, manual override | Forecast-proportion shares, or an explicit launch prior |
| 4 | Top-down missing promotions | A promoted SKU is short by roughly the promotion uplift | Shares estimated from non-promotional history | Chronic under-supply of exactly the items the business is pushing | Use forecasts that carry the promotion calendar as the share source |
| 5 | Rank-one bias | Mix errors persist after "fixing" the shares | The map has one direction | Structural under-performance that no share revision fixes | Move to a rank-`n_b` method |
| 6 | Covariance estimated from too little data | `W` estimated on 18 months for 400 series | Sample covariance singular when `n > T` | Numerical failure or wild weights | Shrinkage or structural scaling; report which |
| 7 | Optimality claimed without the oracle | "Our method is provably optimal" with no mention of `W` | GLS optimality is conditional on `W` being true | False confidence; the method can lose to OLS | State the conditioning set aloud whenever the word appears |
| 8 | Scoring at the fitted level only | Backtest reports one number, for the total | Errors at leaf and root are different quantities | A method that is bad where the decision is made looks good | Backtest and report at every level |
| 9 | Incomparable backtests | Competing methods use different origins, horizons, or base models | The comparison measures everything except the method | The chosen method wins on the base model, not on the reconciliation | Hold base forecasts, origins, horizon, and buckets fixed |
| 10 | Level weighting hidden | A single accuracy score with no statement of weights | Equal weighting is a choice | The ranking flips under a decision-relevant weighting | Publish the weight vector with the score |
| 11 | Weekly plan against monthly plan | Operations plan sums to 4.3 weeks per month | Week-month calendars do not nest | Permanent, unexplained variance that is read as forecast error | Reconcile within a calendar; bridge across with a stated rule |
| 12 | Rounding after reconciliation | Reconciled integers sum to one less than the parent | Independent rounding of fractions | Incoherence reintroduced at the last step | Largest-remainder rounding with the residual assigned once |
| 13 | Overrides outside the register | Planners adjust nodes and nobody records it | Overrides bypass the projection | The measured accuracy describes a forecast nobody used | Log every override with reason and value; measure overridden and not |
| 14 | Pack-size change unnoticed | History and plan agree in eaches, not in cases | Conversion factor changed mid-history | A structural break read as a demand signal | Version the conversion; restate or segment the history |
| 15 | One method, chosen once | Nobody can name who chose it or when | Configured at implementation and never revisited | The method reflects a data regime that may no longer exist | Review the register annually against current forecastability |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The hierarchy is a strict tree | Gives one summing matrix and one coherent subspace | Crossed dimensions (product by channel by region with shared stock) | Check that every node has exactly one parent; if not, you have a grouped structure, not a tree |
| 2 | Aggregation is additive | Makes the constraint linear | Ratios, prices, service levels, averages | Ask whether the parent is the sum or the mean of the children; means need weighted reconciliation |
| 3 | The node set is stable over the forecast horizon | `S` is constant across buckets | New items, discontinued items, site closures, reorganisations | Diff the node list across the horizon; any difference invalidates a single `S` |
| 4 | Base forecasts are the only inputs worth using | Reconciliation adjusts a given base | Base forecasts already incorporate information that the projection discards | Compare the reconciled result with a direct forecast at the target level |
| 5 | Errors are the right thing to weight | `W` encodes reliability | Cost of error is asymmetric (stockout versus holding) | Ask what an error costs in each direction; if asymmetric, weight the loss, not the variance |
| 6 | The error covariance is estimable | GLS needs `W` | `n > T`; non-stationarity; structural breaks | Report `n/T`; if above 0.5, prefer a structured `W` and say so |
| 7 | The projection is the right operation | Reconciliation is a projection | The base forecasts are biased and the bias is level-specific | Test whether a coherent vector is left unchanged, and whether the output beats the base at the decision level |
| 8 | Coherence is worth its cost | Justifies the whole exercise | The reconciled number is coherent and wrong at the decision level | Backtest reconciled against base at the decision level; if it loses, coherence is not paying |
| 9 | Level weighting reflects decision value | Makes a single score meaningful | Equal weighting is used as a convenience | Ask which level each decision consumes, and how many decisions per level |
| 10 | The bucket calendar is consistent | Weeks nest into months | ISO weeks versus Gregorian months | Sum the weeks in a month; if they do not equal the month, a bridge is required |
## 12. Exercises

Nine exercises. Exercises 1–6 are calculation and application, 7–8 are critique, and 9 is design.
Each states a deliverable, acceptance criteria, and an estimated time. Work them in order; exercise
4 depends on the share definitions from exercise 2, and exercise 6 depends on the renormalisation
argument in section 6.6.

### Exercise 1 — Build `S` and `C`

A hierarchy has a total `T`, three regions `R1`, `R2`, `R3`, and five leaves: `L1` and `L2` under
`R1`, `L3` under `R2`, and `L4` and `L5` under `R3`.

**Deliverable:** write the `9 x 5` summing matrix `S` in full, write the `4 x 9` constraint matrix
`C`, state both dimensions, and show the product `C S` is the zero matrix.

**Acceptance criteria:** every entry of `S` is 0 or 1; every column of `S` has exactly three ones;
`C` has one row per parent with children; the product is exactly zero, not approximately zero; the
node ordering used is stated explicitly.

**Estimated time:** 25 minutes.

### Exercise 2 — Three methods on one three-node hierarchy

Base forecasts `(Total, A, B) = (100, 60, 35)`. Historical shares `p_A = 0.62`, `p_B = 0.38`.

**Deliverable:** the bottom-up vector, the top-down vector, and the OLS reconciled vector, each with
its total, and one sentence explaining why the totals differ.

**Acceptance criteria:** bottom-up total is 95; top-down total is 100; OLS total is 98.33 to two
decimals; all three vectors satisfy `Total = A + B`; the explanation names which level each method
treats as authoritative rather than calling one method "more accurate".

**Estimated time:** 30 minutes.

### Exercise 3 — Weighted least squares by hand

Same hierarchy and base forecasts as exercise 2, with `W = diag(400, 900, 1600)`, read as error
variances in units squared.

**Deliverable:** the reconciled vector, the movement of each node from its base, and a check that the
three reconciled values satisfy the constraint to two decimals.

**Acceptance criteria:** the gap `d = 5` is stated with its sign; the movement of each node is
reported in units and equals `d x w_i / sum(w)`; the node with the largest weight moves most; the sum
check is shown, not asserted.

**Estimated time:** 25 minutes.

### Exercise 4 — Average of proportions against proportion of averages

Six months of history for a parent and one member:

| Month | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Parent | 100 | 110 | 120 | 130 | 140 | 150 |
| Member | 50 | 57.2 | 64.8 | 71.5 | 78.4 | 87.0 |

**Deliverable:** compute the member's share by the average-of-proportions method and by the
proportion-of-averages method, state which is higher and why, and compute both forecasts at a parent
forecast of 160.

**Acceptance criteria:** method A gives 0.5417 and method F gives 0.5452, to four decimals; the
explanation identifies growth as the reason the two differ; both forecasts at 160 are reported in
units; the direction of the bias (both low if the share keeps rising) is stated.

**Estimated time:** 30 minutes.

### Exercise 5 — Coherence gap audit

Using the Halloway Surgical base forecasts in section 6.2, with gaps of +40 at the total, -10 at
North, and -20 at South.

**Deliverable:** the gap in units for each parent, the gap ratio for each, the gap norm for the
vector, and the share of the squared gap contributed by each parent.

**Acceptance criteria:** gap norm is 45.8 units to one decimal; the total contributes about 76% of
the squared gap; the write-up identifies which node should be investigated first and why
concentration rather than magnitude drives that choice.

**Estimated time:** 25 minutes.

### Exercise 6 — Forecast proportions and the renormalisation trap

Base forecasts: total 1,000, North 560, South 400, N1 300, N2 180, N3 90, S1 150, S2 160, S3 110.

**Deliverable:** the forecast-proportion shares at every level, the resulting nine-node vector, and a
demonstration of what happens to the total if the region shares are not renormalised.

**Acceptance criteria:** region shares are 0.58333 and 0.41667 after renormalisation; without it the
allocated total is 960 rather than 1,000; every leaf in the final vector sums to its parent; the
explanation connects the trap to the base forecasts being incoherent at the region level.

**Estimated time:** 30 minutes.

### Exercise 7 — Critique a backtest

A vendor reports: "Our reconciliation improves forecast accuracy by 12%. We tested on two years of
data, comparing our reconciled forecasts against bottom-up. Accuracy is measured by mean absolute
percentage error at the total level."

**Deliverable:** a written critique identifying at least four defects and stating what evidence would
be needed to support the claim.

**Acceptance criteria:** identifies the single evaluation level, the absence of a base-forecast
comparison, the risk that the two arm's base forecasts differ, the unstated weighting, and the
absence of common origins or a stated horizon; proposes a corrected design in no more than six
sentences.

**Estimated time:** 20 minutes.

### Exercise 8 — Interrogate the word "optimal"

A slide reads: "Minimum-trace reconciliation is the optimal way to make forecasts coherent."

**Deliverable:** a three-sentence rewrite that preserves the true content and removes the false
implication, plus a list of the three conditions under which the method can under-perform a simpler
alternative.

**Acceptance criteria:** the rewrite names `W` and states that optimality is conditional on it; the
three conditions cover estimation error, non-identifiability, and the case `n > T`; no sentence
claims the method is best in general.

**Estimated time:** 20 minutes.

### Exercise 9 — Choose a method under a stated forecastability gradient

A retailer has 1 total, 4 regions, 12 categories, and 900 SKU-by-store series. The SKU series are 62%
zeros; the category series are seasonal and dense; the regional planners own the promotional
calendar; finance consumes the total; store replenishment consumes the SKU series and cannot carry a
number that does not add up to its parent.

**Deliverable:** a one-page recommendation naming a method, a fallback, the evidence you would
collect in the first four weeks, and the failure you would watch for.

**Acceptance criteria:** the recommendation addresses all three criteria from section 5 step 18; the
answer recognises that the sparse leaves and the coherence requirement at the leaf pull in opposite
directions; the fallback is a different method rather than the same method with different weights;
the four-week evidence plan names a specific diagnostic per level.

**Estimated time:** 40 minutes.

## 13. Self-check answers

Answers to exercises 1–8 with the reasoning; a rubric for 9. Check your arithmetic against these
before reading the reasoning, because the reasoning is easier to follow once the numbers are known.

### Answer 1

Ordering `T, R1, R2, R3, L1, L2, L3, L4, L5`, with leaves as columns:

```text
        L1 L2 L3 L4 L5
   T  [  1  1  1  1  1 ]
   R1 [  1  1  0  0  0 ]
   R2 [  0  0  1  0  0 ]
   R3 [  0  0  0  1  1 ]
   L1 [  1  0  0  0  0 ]
   L2 [  0  1  0  0  0 ]
   L3 [  0  0  1  0  0 ]
   L4 [  0  0  0  1  0 ]
   L5 [  0  0  0  0  1 ]
```

`S` is `9 x 5`. `C` is `4 x 9`, one row per parent, with a 1 in the parent's column and a -1 in each
child's column: `T - R1 - R2 - R3`, `R1 - L1 - L2`, `R2 - L3`, `R3 - L4 - L5`. The product `C S` is
`4 x 5` and every entry is zero, because each row subtracts exactly the columns its parent row sums.

The point of the exercise is the dimension count: nine nodes, five leaves, four constraints, and
`9 - 5 = 4`. That equality is not a coincidence; it is the rank-nullity relation for a tree.

### Answer 2

Bottom-up selects the leaf rows: `(60, 35)` and sums to `(95, 60, 35)`.

Top-down applies the shares to the total: `0.62 x 100 = 62` and `0.38 x 100 = 38`, giving
`(100, 62, 38)`.

OLS solves `beta = (S'S)^-1 S'y`. With `S'S = [[2, 1], [1, 2]]` and `S'y = (160, 135)`, the inverse is
`(1/3)[[2, -1], [-1, 2]]`, so `beta = (1/3)(185, 110) = (61.667, 36.667)` and the vector is
`(98.333, 61.667, 36.667)`.

The totals differ because each method treats a different base forecast as authoritative: bottom-up
treats the leaves, top-down treats the total, OLS treats all three equally and lands between them.
None is "more accurate" without an outcome; they are three different coherent points on the plane of
section 3.1.

### Answer 3

The gap is `d = 100 - 60 - 35 = +5`, meaning the parent is five units above its children. With
`sum(w) = 2,900`, the parent moves down by `5 x 400/2,900 = 0.690` to 99.310, `A` moves up by
`5 x 900/2,900 = 1.552` to 61.552, and `B` moves up by `5 x 1,600/2,900 = 2.759` to 37.759. The check
is `61.552 + 37.759 = 99.311`, coherent to rounding.

`B` moves most because its variance, 1,600, is the largest, so the method trusts it least. That is the
whole content of `W`: it is a statement about which base forecast is more reliable, and the data does
not supply it.

### Answer 4

Method A averages the six ratios: `(0.50 + 0.52 + 0.54 + 0.55 + 0.56 + 0.58)/6 = 3.25/6 = 0.5417`.
Method F divides the sums: member total `408.9`, parent total `750`, ratio `0.5452`.

Method F is higher, because the member's share is rising and the parent is also growing. In method A
each month counts equally, so the early months with low shares pull the average down. In method F a
unit in a high-volume month counts more than a unit in a low-volume month, and the high-volume months
are the recent ones with the higher shares.

At a parent forecast of 160, method A gives `86.67` units and method F gives `87.23`. If the share
continues to rise toward 0.60, both are short by roughly 9% and 10% respectively. Both methods
extrapolate a past average into a future that is drifting; the difference between them is an order of
magnitude smaller than the drift.

### Answer 5

Gaps: total `+40`, North `-10`, South `-20`, in units. Ratios: `40/1,000 = 4.0%`, `10/560 = 1.8%`,
`20/400 = 5.0%`. The gap norm is `sqrt(1,600 + 100 + 400) = sqrt(2,100) = 45.8` units.

Shares of the squared gap: `1,600/2,100 = 76.2%` at the total, `4.8%` at North, `19.0%` at South.
Investigate the total first, because three-quarters of the disagreement sits in a single comparison
between one model and the sum of two others. That comparison is where the two processes differ about
the quarter, most likely a national promotion or a channel shift, and it is the gap that will
propagate to every node once you reconcile.

### Answer 6

Region shares after renormalisation: `560/960 = 0.58333` and `400/960 = 0.41667`. Within North:
`300/570 = 0.52632`, `180/570 = 0.31579`, `90/570 = 0.15789`. Within South: `150/420 = 0.35714`,
`160/420 = 0.38095`, `110/420 = 0.26190`.

Applied to the total of 1,000: North `583.33`, South `416.67`; N1 `307.02`, N2 `184.21`, N3 `92.11`,
S1 `148.81`, S2 `158.73`, S3 `109.13`.

Without renormalisation, North's share would be `0.5600` and South's `0.4000`, summing to `0.9600`.
Applied to 1,000 the allocation would total 960, not 1,000, and the vector would be incoherent in
exactly the way section 6.2 described — a defect that looks like a rounding error and is not. The
renormalisation is necessary precisely because the base forecasts at the region level do not sum to
the base total.

### Answer 7

Four defects, at minimum. **One level only:** total-level accuracy says nothing about whether the SKU
plan improved, and the SKU plan is where replenishment acts. **No base comparison:** reconciled
against bottom-up conflates the reconciliation with the choice of base forecasts; the correct
comparison is reconciled against the base it was computed from. **Different bases:** if the two arms
use different base models, the 12% measures the base models. **No common origins, horizon, or
weighting:** without these the number is not reproducible.

A corrected design holds one set of base forecasts, one set of origins, one horizon, and one bucket
definition; evaluates at every level; reports base against reconciled at each level; and publishes
the level weights. Six sentences suffice to state this.

### Answer 8

Rewrite: "Minimum-trace reconciliation is a weighted least-squares projection onto the coherent
subspace, and it minimises expected total error when the weight matrix `W` equals the true covariance
of the base forecast errors. Because `W` must be estimated, the method is optimal in expectation
conditional on that estimate, and simpler choices such as `W = I` can outperform it out of sample."

The three conditions: estimation error in `W` large relative to the differences between candidate
weights; the reconciled error covariance not identifiable, so no estimator can recover it even with
unlimited data of the wrong kind; and `n > T`, where the sample covariance is singular and the
inverse is undefined or numerically unstable.

### Answer 9 — rubric

A strong recommendation recognises the tension: sparse leaves argue for middle-out or top-down, but
the leaf coherence requirement means whatever is chosen must produce a coherent leaf vector, so
bottom-up built on intermittent leaf models is a poor choice. A defensible answer is middle-out at
category with forecast proportions, reconciled at the leaf level, with a documented fallback to
bottom-up for dense categories. The four-week evidence plan measures zero share and inter-demand
interval at each level, share drift at category, and base-versus-reconciled error at the store level.
The failure to watch is a promotion that the category-level model does not see. Weak answers pick a
method on accuracy grounds alone and never mention the coherence requirement at the leaf.

## 14. Interview questions

Ten questions, each with the content a strong answer contains and what a weak answer reveals. Ask
them in this order; the later ones assume the earlier ones are answered.

### Q1. What does it mean for a set of forecasts to be coherent?

**Strong answer:** every parent equals the sum of its children at every level and every bucket, which
can be written as `C y = 0` or `y = S b`; it is a linear constraint on the vector, it can be measured
per bucket, and it says nothing about accuracy.
**Weak answer:** "the numbers add up", offered as though that were the whole of it, with no
distinction between coherence and correctness.

### Q2. Bottom-up or top-down?

**Strong answer:** the question is malformed without naming the level at which the forecast will be
scored and the level of the decision; bottom-up trusts the leaves, top-down trusts the total and the
mix, and the choice depends on which of those is more forecastable in this hierarchy.
**Weak answer:** a preference stated as a principle, usually "bottom-up because it uses the most
detail" or "top-down because the total is most stable", with no reference to the data.

### Q3. Why is top-down biased if the shares are estimated well?

**Strong answer:** because the map has rank one; a share vector defines a single ray through the
coherent subspace, so the method cannot move in any direction orthogonal to it, and better shares
choose a better ray without adding a dimension.
**Weak answer:** "because historical shares are out of date", which is a fixable defect and not the
structural one.

### Q4. What is optimal reconciliation optimal for?

**Strong answer:** it minimises the trace of the reconciled error covariance conditional on the weight
matrix `W` being the true covariance of the base errors; since `W` is estimated, sometimes not
identifiable, and sometimes singular when the series count exceeds the observation count, the
optimality applies to an oracle rather than to the procedure as run.
**Weak answer:** "it is the mathematically optimal method", with no conditioning set named.

### Q5. How do you set the weights?

**Strong answer:** `W = I` for ordinary least squares, a diagonal matrix from error variances or
structural scaling for weighted least squares, an estimated sample covariance when `n` is small
relative to `T`, and shrinkage toward a structured target when it is not; then compare at least two
choices out of sample and report whether they agree.
**Weak answer:** "the software picks them", or a single choice with no statement of how it was
validated.

### Q6. How would you compare the four methods?

**Strong answer:** hold the base forecasts fixed, use common origins, horizon, and buckets, evaluate
at every level, weight the levels according to decision value, and report each method against the base
forecast at the level where the decision is made.
**Weak answer:** a single accuracy number at one level, usually the aggregate, with no statement of
how it was produced.

### Q7. A planner overrides a SKU forecast. What happens to coherence?

**Strong answer:** the override is a new incoherence unless it is propagated by the same map; it must
be logged with its reason and value, the parent nodes must be restated, and the measured accuracy must
distinguish overridden from untouched series.
**Weak answer:** "the system handles it", or an acknowledgement that overrides happen without any
statement of what they do to the constraint.

### Q8. Weekly plan against monthly plan: why do they never reconcile?

**Strong answer:** ISO weeks do not nest into Gregorian months; a month contains partial weeks, so no
allocation makes both statements true at once, and the resolution is to reconcile within one calendar
and bridge across calendars with a documented day-weighted rule.
**Weak answer:** "forecast error" or "rounding", neither of which is the cause.

### Q9. When would you publish an incoherent forecast deliberately?

**Strong answer:** when the number is consumed by a judgement rather than by a transaction, when the
coherent alternative would distribute a wrong top-level number into every descendant, and when a
documented bridge is cheaper than the distortion reconciliation would impose.
**Weak answer:** never, because coherence is always better; or always, because coherence is
bureaucratic.

### Q10. What evidence would make you change your method?

**Strong answer:** an out-of-sample result at the decision level showing the reconciled forecast losing
to its own base, a forecastability gradient that has shifted, a change in the hierarchy or the
calendar, or an override rate high enough that the published forecast is not the one being used.
**Weak answer:** no evidence is named, which means the method is a preference rather than a decision.

## 15. Key terms

One line each, cross-referenced to the section where the term is developed.

- **Base forecast** — the independently produced forecast at each level before reconciliation; the
  input to every method in this topic (section 3.2).
- **Bottom-up** — reconciling by summing leaf forecasts; coherent by construction, noisy in proportion
  to the leaves' errors (section 5, step 10).
- **Coherence** — the property that every parent equals the sum of its children in every bucket;
  written `C y = 0` and measured per node (section 3.1).
- **Coherence gap** — the parent-minus-children discrepancy in units and as a ratio; report it before
  reconciling (section 5, step 4).
- **Coherent subspace** — the set of vectors satisfying every constraint, of dimension `n_b`, written
  `span(S)` (section 3.1).
- **Constraint matrix `C`** — the `(n - n_b) x n` matrix of aggregation constraints, with entries 0,
  1, and -1 (section 5, step 3).
- **Covariance identifiability** — whether the reconciled error covariance can be recovered; it
  cannot, with rank deficiency `n - n_b` (section 5, step 14).
- **Forecast proportions** — shares computed from base forecasts, normalised within each parent and
  multiplied down (section 6.6).
- **Forecast value added (FVA)** — error reduction relative to a stated benchmark; meaningless without
  the benchmark (section 5, step 17).
- **Generalised least squares (GLS)** — `(S' W^-1 S)^-1 S' W^-1`; optimal when `W` is the true error
  covariance (section 5, step 14).
- **Historical proportions** — shares computed from past demand, in average-of-proportions or
  proportion-of-averages form (section 6.3).
- **Idempotence** — `(S P)(S P) = S P`; the property that makes the map a projection, and not a
  synonym for unbiasedness (section 3.2).
- **Level weighting** — the weight each level receives in a single score spanning the hierarchy; a
  choice, never a neutral default (section 5, step 16).
- **Middle-out** — reconciling from a named intermediate level upward and downward; the level is the
  method (section 6.7).
- **Minimum trace (MinT)** — the family of estimators minimising the trace of the reconciled error
  covariance, with variants for estimating `W` (section 5, step 14).
- **Oblique projection** — a weighted projection, moving along a direction that is not perpendicular
  to the subspace (section 3.1).
- **Optimal reconciliation** — the weighted least-squares projection onto the coherent subspace using
  an estimated error covariance (section 5, step 14).
- **Ordinary least squares (OLS)** — GLS with `W = I`; the perpendicular projection, and a strong
  baseline (section 5, step 14).
- **Proration** — scaling every member by one factor to close a gap; not reconciliation (section 10,
  row 2).
- **Reconciliation register** — the document naming method, weights, owner, evidence, and override
  rule per planning stream (section 5, step 19).
- **Residual of reconciliation** — `(I - S P) y_hat`, the part of the base forecast the map discards
  (section 3.1).
- **Shrinkage** — pulling an unstable sample covariance toward a structured target; a design choice,
  not a detail (section 5, step 14).
- **Summing matrix `S`** — the `n x n_b` 0/1 matrix mapping the leaf vector to every series (section 5,
  step 2).
- **Top-down** — allocating a total forecast down a share structure; coherent, rank one, biased in
  general (section 5, steps 11–12).
- **Unbiasedness condition** — `S G S = S` for a combination matrix `G`; the identity top-down fails
  (section 5, step 8).
- **Weighted least squares (WLS)** — GLS with a diagonal `W`; the practical compromise between OLS
  and a full covariance estimate (section 6.8).

## 16. Sources

Grouped by type. Every entry carries a grade: A for peer-reviewed or audited, B for regulatory or
official standard, C for company case or investor claim, D for vendor marketing, E for press or
anecdote. Entries cite title and venue; page numbers and digital object identifiers are omitted where
they could not be checked against the source.

### Peer-reviewed literature (Grade A)

| Source | What it supports | Grade |
|---|---|---|
| Hyndman, Ahmed, Athanasopoulos, Shang, "Optimal combination forecasts for hierarchical time series", *Computational Statistics and Data Analysis*, 2011 | The projection formulation of reconciliation and the OLS/GLS framing | A |
| Wickramasuriya, Athanasopoulos, Hyndman, "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization", *Journal of the American Statistical Association*, 2019 | The MinT estimator and the covariance variants that make it usable when `n > T` | A |
| Panagiotelis, Athanasopoulos, Gamakumara, Hyndman, "Forecast reconciliation: A geometric view with new insights on bias correction", *International Journal of Forecasting*, 2021 | The distance-reducing property and the geometric interpretation used in section 3 | A |
| Athanasopoulos, Hyndman, Kourentzes, Panagiotelis, "Forecast reconciliation: A review", *International Journal of Forecasting*, 2024 | The survey finding that no method dominates and that ranking depends on data structure, level, and weights | A |
| Makridakis, Spiliotis, Assimakopoulos, M5 competition results, *International Journal of Forecasting*, 2022 | The twelve-level hierarchy, the equal weighting of levels, and the strength of ensembles | A |
| Gross and Sohl, "Disaggregation methods to expedite product line forecasting", *Journal of Forecasting*, 1990 | The average-of-proportions and proportion-of-averages distinction | A |

### Standards and official bodies (Grade B)

| Source | What it supports | Grade |
|---|---|---|
| GS1, *General Specifications* (annual release) | GTIN, GPC, GLN, SSCC identifiers and the pack hierarchy | B |
| ISO 8601-1:2019, *Date and time — Representations for information interchange* | The ISO week calendar and its non-alignment with Gregorian months | B |
| IFRS 8, *Operating Segments*, International Accounting Standards Board | The requirement to reconcile segment totals to the financial statements | B |
| National Retail Federation, 4-4-5 retail calendar convention | The 52/53-week retail calendar that makes weeks nest into periods | B |

### Industry and vendor material (Grade C and D)

| Source | What it supports | Grade |
|---|---|---|
| Enterprise planning vendor documentation on hierarchical forecasting | The claim that reconciliation is available and that one method is default; used here only as evidence of what is offered, not of what works | D |
| Conference and practitioner accounts of planning transformations | The inference in section 9.3 about who owns which level; treated as inference, not fact | C |

No Grade E material is relied on. Named company cases of reconciliation failure are absent from the
public record, and section 9.3 labels the corresponding claims as inference rather than evidence.

## 17. Links

Within this module:

- Module map and topic status: [`_index.md`](_index.md)
- Previous topic, hierarchy structures: [`01_hierarchy-structures.md`](01_hierarchy-structures.md)
- Next topic, minimum-trace reconciliation: `03_minimum-trace-reconciliation.md` (planned)
- Later topics in this module: `04_intermittent-demand.md` (planned), `05_probabilistic-forecasting.md` (planned)

Within the part:

- Part index: [`../_index.md`](../_index.md)
- Prerequisite module topic on hierarchy and aggregation: [`../07_time_series_forecasting/05_hierarchy-aggregation.md`](../07_time_series_forecasting/05_hierarchy-aggregation.md)
- Prerequisite module map: [`../07_time_series_forecasting/_index.md`](../07_time_series_forecasting/_index.md)

Elsewhere in the course:

- Hierarchy design and the choice of planning level: [`../../01_foundations/01A_foundations_units_time_assumptions/07_hierarchy-design.md`](../../01_foundations/01A_foundations_units_time_assumptions/07_hierarchy-design.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)

> **Note:** Topic 03 of this module covers the minimum-trace estimator, the covariance variants, and
> the estimation problem that this file deliberately stops short of. Read it next.
