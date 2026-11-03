# MinT in Detail — covariance shrinkage, constraints, and a reference implementation

> Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 04 topics 02 and 05 (probability and the bootstrap); Module 08 topic 01 (hierarchy structures); Module 08 topic 02 (aggregation consistency)
> Version 2026.5

---

## 1. Why this topic exists

A planning team runs minimum-trace reconciliation on a hierarchy of 4,800 series. The base forecasts
are non-negative. The reconciled vector contains a leaf at -3.2 units. The same week, a second team
runs the same method on the same base forecasts with a covariance estimated from a different rolling
window, and its total differs by 1.4%. Both results are called MinT. Neither team can explain the
difference, because the object that caused it — a 4,800 by 4,800 matrix — is not on any slide.

Three problems sit behind that scene, and this topic exists because they are usually taught as one.

**The covariance does not exist at the size you need it.** A covariance matrix for `n` series has
`n(n + 1) / 2` distinct entries. For 4,800 series that is 11,522,400 parameters. Thirty-six months of
history give 36 observations, so a sample covariance has rank at most 35 and is singular: it has no
inverse, and the formula that defines the method cannot be evaluated. At 400 series the matrix is
invertible and nearly useless, because the inverse amplifies estimation noise into weights. This is
not an edge case. It is the normal operating point of a product-by-location hierarchy.

**The word "optimal" is doing more work than the method can support.** Module 08 topic 02 stated the
projection and warned that optimality is conditional on the weight matrix; this topic makes the
condition concrete. Minimum-trace reconciliation minimises the expected sum of reconciled error
variances given a particular covariance of the base forecast errors. If that covariance is estimated
badly, the method is still optimal with respect to the bad estimate, which is a precise way of saying
it can lose to a method that never claimed optimality.

**The arithmetic meets constraints it was not built for.** Units cannot be negative; a committed
production order cannot be revised by a forecasting routine; a pallet is an integer. A linear
projection respects none of these. It will produce a negative forecast, and it will produce it from
entirely non-negative inputs. The practitioner's question is not "is the projection elegant" but
"what do I do with the negative number, and what does doing it cost".

This topic is the third in a sequence. Module 07 topic 05 introduced hierarchical aggregation and
previewed reconciliation as a way to make levels agree. Module 08 topic 02 set up the projection view:
coherence as a subspace, every method as a linear map, optimal reconciliation as a weighted
projection. That topic deliberately stopped at the estimator. This file supplies the estimator in
enough detail to buy it, run it, defend it, and criticise it.

> **Note:** Minimum trace (MinT) is a family of estimators, not a single method. The family name
> refers to the objective — minimising the trace of the reconciled error covariance — and the members
> differ only in how the error covariance is estimated. Saying "we use MinT" without naming the member
> is like saying "we use a regression" without naming the features.

Sections 3 to 5 explain the weight matrix and its estimation; section 6 works a five-node hierarchy
end to end; section 7 gives a projection you can reproduce without a library; sections 8 to 11 cover
the standards context, the evidence, the failure modes, and the limits.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** the minimum-trace projection `y_tilde = S (S' W^-1 S)^-1 S' W^-1 y_hat` and **explain**
   the role of `W` as a metric on the error space rather than as a model of the business.
2. **Compute** the number of covariance parameters `n(n + 1) / 2` for a named hierarchy, compare it
   with the number of available residual observations `T`, and **state** whether the sample
   covariance is singular, ill-conditioned, or usable.
3. **Distinguish** ordinary least squares (OLS), weighted least squares with variance scaling
   (WLSv), weighted least squares with structural scaling (WLSs), MinT with the sample covariance
   (MinT(Sample)), and MinT with shrinkage (MinT(Shrink)), and **choose** among them with a reason.
4. **State** the Schäfer–Strimmer shrinkage estimator in plain language, **compute** the shrinkage
   intensity for a two-series example, and **explain** why the result is always invertible.
5. **Separate** reconciling means from reconciling distributions, and **explain** why a reconciled
   point forecast does not by itself determine a reconciled interval.
6. **Handle** non-negativity and other constraints by at least three routes, and **state** what each
   costs in coherence, optimality, and computation.
7. **Cache** a projection matrix, **name** the conditions that invalidate the cache, and **design** a
   version key for it.
8. **Critique** a claim that reconciliation is "optimal" by naming the conditioning set, the
   estimation error in `W`, and the assumptions the method inherits from the base forecasts.

Outcomes 2, 3, 4, and 5 are exercised with arithmetic in section 12 and answered in section 13.
Outcomes 6, 7, and 8 are exercised as design and critique tasks.

## 3. The core mental model

Hold one picture: a vector of numbers that do not agree, a matrix that says how much each number is
trusted, and a linear map that produces the nearest agreeing vector under that trust.

```text
   PANEL A — one hierarchy, three objects, one projection

        BASE FORECASTS              WEIGHT MATRIX W                 COHERENT RESULT
        y_hat  (eaches/week)        (eaches squared, 5 x 5)         y_tilde = S P y_hat

        Total   1,180   ┐           T      N      S     N1    N2      Total   1,200.9
        North     720   │        T [ 8,840  5,290  2,610  3,150 2,330 ]  North     687.5
        South     520   ├─── e ──>N [ 5,290  5,360  1,400  2,660 1,960 ]  South     513.4
        N1        430   │        S [ 2,610  1,400  2,120    860   660 ]  N1        434.7
        N2        250   ┘        N1[ 3,150  2,660    860  1,980   890 ]  N2        252.8
                                 N2[ 2,330  1,960    660    890 1,220 ]
           gaps:                      ^                                  gaps: 0
           Total  -60                 |                                  (687.5 + 513.4
           North  +40            "how much do I                          = 1,200.9)
                                  distrust each
                                  number?"

   Read across: the base vector is off the coherent subspace by two gaps. W says which
   numbers are noisy. The projection returns the closest coherent vector in the metric
   W^-1. Which point it returns depends on W; that the point is coherent does not.
```

Four properties are worth memorising, because everything else in this topic follows from them.

**Property 1 — The constraint fixes the direction of the correction; `W` fixes how it is shared.**
In the five-node example the total base is 60 units below its children and the North base is 40
above its own, so every method moves the total up and North down. What changes across OLS, WLS, and
MinT is the size of each movement and how much of it lands on South, which had no gap of its own.
The first question about a surprising reconciled number is therefore "what did the constraints
require"; only the second is "what did the weights choose".

**Property 2 — `W` is a metric, and a metric is a claim about reliability.** The inverse `W^-1` is a
precision matrix. A series with a large variance has a small precision, so the projection is willing
to move it a long way to satisfy the constraint; a series with a small variance barely moves. That is
the whole content of the weights. Nothing in the data determines them; they are an estimate of how
wrong each base forecast has been, and an estimate is a claim.

**Property 3 — The only estimated object is `W`.** `S` is built from the hierarchy definition and is
exact. `P` is a function of `S` and `W`. Applying `P` is arithmetic. This is why the method is fast
and why the estimation problem is concentrated: all of the statistical difficulty lives in one
matrix, and if that matrix is wrong, no amount of careful linear algebra rescues the answer.

**Property 4 — Linearity is the great convenience and the great blind spot.** Because `P` is linear,
it can be computed once and applied to any base vector, cached, applied draw by draw, and composed
with scenario scaling. The same linearity means the map cannot respect an inequality: a linear
function of non-negative inputs can be negative, and a projection onto a subspace does not know that
the subspace is supposed to intersect the positive orthant.

```text
   PANEL B — the pipeline, and where the linearity stops

   base vector --> [ P ] --> coherent vector --> constraint gate --> published plan
                   cached     (linear, cheap)    negative? capacity?
                   on S, W                       integer? immutable?
                                                       |
                                        accept  |  clip  |  QP / NNLS
```

Everything left of the gate is linear, cacheable, and cheap. Everything right of it is iterative,
case-specific, and expensive, and it is usually where projects either quietly cut scope or quietly
break coherence.

Shrinkage is a path between those two extremes: at `lambda = 0` the weight matrix is the sample
covariance, noisy and possibly singular; at `lambda = 1` it is its diagonal, stable and always
invertible; the estimator of section 5 step 9 chooses a point between them.

The mental model to carry into a meeting: the hierarchy supplies a constraint, the error covariance
supplies a metric, the method supplies a projection. When someone disagrees with a reconciled
number, establish which of the three they are disagreeing with, because the remedies differ.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Base forecast | An independently produced forecast at each node, before reconciliation | Called "the forecast", implying it is final | Units per bucket per node |
| Reconciled forecast | A coherent forecast produced by applying a stated linear map to the base forecasts | Called "adjusted" without saying how | Units per bucket per node |
| Coherence | Every parent equals the sum of its children at every level and bucket | Confused with accuracy | Boolean per node per bucket |
| Summing matrix `S` | The `n x m` 0/1 matrix mapping the `m` bottom-level series to all `n` nodes | Confused with a history table or a rollup view | Matrix, entries 0 or 1 |
| Bottom-level vector | The `m` finest series from which every other series is an exact sum | Confused with all stored series | Units per bucket |
| Projection matrix `P` | The `m x n` matrix in `y_tilde = S P y_hat`, satisfying `S P S = S` | Assumed unique to the method | Matrix, real entries, often negative |
| Minimum trace (MinT) | The family of projections minimising the trace of the reconciled error covariance | Used as the name of one estimator | Family name plus a `W` variant |
| Ordinary least squares (OLS) | The projection with `W = I`; the Euclidean, or orthogonal, projection | Treated as a worthless baseline | Estimator variant |
| Weighted least squares (WLS) | The projection with a diagonal `W` | Confused with MinT | Estimator variant |
| Variance scaling (WLSv) | WLS with `W = diag` of estimated error variances | Assumed to be the same as structural scaling | Units squared on the diagonal |
| Structural scaling (WLSs) | WLS with `W = diag(S 1)`, the number of leaves beneath each node | Assumed to require no assumption | Dimensionless weights |
| Generalised least squares (GLS) | The projection with a full `W`; algebraically identical to MinT | Believed to be a different, older method | Estimator |
| MinT(Sample) | MinT with `W` equal to the unrestricted sample covariance of base errors | Used with fewer observations than series | Estimator variant |
| MinT(Shrink) | MinT with `W` shrunk toward its diagonal | Treated as a minor tweak | Estimator variant plus intensity |
| Shrinkage | Pulling an unstable estimate toward a structured target to reduce variance | Confused with regularisation of the forecasts themselves | Estimator operation |
| Shrinkage intensity | The weight `lambda` on the target, between 0 and 1 | Fixed by convention in many implementations | Dimensionless |
| Target matrix | The structured matrix shrunk toward, usually the diagonal of the sample covariance | Chosen without stating it | Matrix |
| Sample covariance | `(1/T) sum_t e_t e_t'` over residual vectors | Assumed to estimate the forecast error covariance | Units squared |
| Positive definite | A symmetric matrix with all eigenvalues strictly positive, hence invertible | Confused with "numerically invertible" | Boolean |
| Singular | Having at least one zero eigenvalue, hence no inverse | Handled with a pseudoinverse without comment | Boolean |
| Condition number | Ratio of largest to smallest singular value; amplification of input error | Reported only when the code fails | Dimensionless ratio |
| Rank deficiency | `n` minus the rank; the number of directions in which a matrix carries no information | Assumed to be zero for a covariance | Integer |
| Identifiability | Whether a quantity can be recovered from data even in principle | Assumed for any well-posed-looking formula | Boolean |
| Singular value decomposition (SVD) | Factorisation into rotations and a diagonal of singular values; used for rank and pseudoinverses | Treated as an implementation detail | Matrix factorisation |
| Precision matrix | The inverse of a covariance; large entries mean high trust | Ignored in favour of the covariance itself | Inverse units squared |
| Trace | The sum of the diagonal entries of a square matrix; here, total reconciled variance | Assumed to be scale-free | Units squared |
| Oracle property | Optimality that holds when an unobservable quantity is known exactly | Quoted as if it were a property of the procedure | Property of an estimator |
| Residual | The difference between an observed value and a fitted value in sample | Treated as a forecast error | Units |
| Forecast error | The difference between a forecast and the outcome it aimed at, out of sample | Conflated with the in-sample residual | Units |
| Horizon-specific covariance | A distinct `W` for each forecast horizon | Replaced by one pooled `W` for convenience | Matrix per horizon |
| Non-negativity constraint | The requirement that every reconciled value be at least zero | Assumed to be satisfied by non-negative inputs | Units per node |
| Non-negative least squares (NNLS) | Constrained least squares with all coefficients non-negative | Used as a synonym for clipping | Optimisation problem |
| Quadratic programme (QP) | Minimisation of a quadratic objective under linear constraints | Treated as an exotic method | Optimisation problem |
| Set-negative-to-zero (sntz) | A heuristic that replaces negative reconciled values with zero | Presented as constraint enforcement | Units per node |
| Immutable forecast | A node whose value is fixed by commitment and may not be changed by reconciliation | Ignored, so the plan and the forecast diverge | Units, flagged per node |
| Constrained projection | Projection onto the intersection of the coherent subspace and a constraint set | Assumed to be linear and cacheable | Non-linear map |
| Push-forward distribution | The distribution induced on the output of a map applied to a random input | Confused with reconciling the marginals | Distribution over `n` nodes |
| Energy score | A strictly proper scoring rule for multivariate distributions | Replaced by the log score without comment | Score, lower is better |
| Projection cache | A stored `P` reused until `S`, `W`, or the node ordering changes | Left unversioned, so stale weights persist | Matrix plus version key |

Four rows repay expansion, because they cause the most damage when confused.

**Trace versus scale.** MinT minimises the sum of the reconciled error variances in squared units. A
one-unit-squared error at the total and a one-unit-squared error at a leaf count equally, so in a
hierarchy with 900 leaves and 30 aggregates the leaves carry about 97% of the objective. That is a
defensible default and it is still a choice: the same level-weighting problem Module 08 topic 02
raised about accuracy scores.

**Sample versus oracle.** The estimator is optimal when `W` is the covariance of the base errors. `W`
is never known; it is estimated from residuals produced by models that are themselves misspecified.
The optimality statement is about an oracle, and it is the reason a simpler weight can beat MinT out
of sample.

**Residual versus forecast error.** Residuals are in sample and usually one step ahead; forecast
errors are out of sample and `h` steps ahead, and their covariance matrices differ by more as `h`
grows. A method calibrated on residuals and applied to a twelve-week horizon is using a metric
estimated in the wrong regime.

**Immutable versus free.** A committed order or a frozen production plan is not a forecast to be
adjusted. It is a constraint. Treating it as free means reconciliation moves a number that cannot
move, and the published plan is not the plan anyone executes.

## 5. The mechanics, step by step

Six phases and twenty steps. Phase A builds the objects. Phase B estimates the weight matrix, which
is where the method is won or lost. Phase C builds and caches the projection. Phase D applies
constraints. Phase E handles distributions. Phase F judges and governs.

The projection itself is stated once and not derived: for a base vector `y_hat` and a summing matrix
`S`, the reconciled vector is

```text
   y_tilde = S P y_hat,     P = (S' W^-1 S)^-1 S' W^-1
```

with `S P S = S`. Module 08 topic 02 derived this as a weighted least-squares projection. What
follows is about the choice of `W`, the numerical life of that formula, and what happens after it.

### Phase A — Build the objects

**Step 1. Fix the node ordering, the horizon, and the unit.** The ordering is arbitrary and must be
stable, because `S`, `W`, and `P` are all defined relative to it. The horizon matters because the
error covariance is horizon-specific: a one-week-ahead error covariance is not a twelve-week-ahead
one. State the unit once and reconcile in the base unit (eaches, kilograms, litres), never in cases,
pallets, or currency, because pack sizes and prices change and the constraint does not.
*Output:* an ordered node list, one horizon index, one base unit. *Branch:* if different nodes use
different units or different calendars, stop and fix the data model first; reconciliation will
happily produce a coherent vector in mixed units.

**Step 2. Build `S` and count.** `S` has one row per node and one column per bottom-level series, with
a 1 wherever a leaf lies beneath the node. Record `n` (nodes), `m` (leaves), and the number of
constraints `n - m`. *Check:* `C S = 0` for the constraint matrix `C`; every column of `S` sums to the
depth of its leaf plus one. *Branch:* if the structure is grouped rather than a tree — product by
channel with shared stock, for example — there is more than one summing matrix and the projection is
not unique without an extra assumption. Use the grouped machinery, not this one.

**Step 3. Count the parameters before you count the data.** The covariance has `n(n + 1) / 2`
distinct entries. Tabulate it against the number of residual vectors `T` you can assemble.

| Hierarchy | Nodes `n` | Leaves `m` | Covariance parameters | `T` at 36 months | `n / T` |
|---|---:|---:|---:|---:|---:|
| Total, 2 regions, 4 SKUs | 7 | 4 | 28 | 36 | 0.19 |
| Total, 4 regions, 12 families, 120 SKUs | 137 | 120 | 9,453 | 36 | 3.8 |
| Total, 5 levels, 500 leaves | 656 | 500 | 215,496 | 36 | 18.2 |
| Total, 5 levels, 4,200 leaves | 4,800 | 4,200 | 11,522,400 | 36 | 133.3 |

The last two rows are singular by construction: the sample covariance has rank at most 35, so at
least `n - 35` eigenvalues are exactly zero and the inverse does not exist. The second row is
invertible and hopeless: `n / T` close to four means the estimate is dominated by noise. Only the
first row is a comfortable estimation problem, and it is not the row most companies are in.

**Step 4. Decide what counts as an error.** The covariance should describe the errors of the
forecasts you are about to reconcile, at the horizon you are about to reconcile, produced by the
models you are actually running. Three substitutions are common and all of them matter:

- In-sample residuals instead of out-of-sample errors. Residuals understate error, sometimes by a
  factor of two, and their correlation structure is flattered by overfitting.
- One-step errors instead of `h`-step errors. Multi-step errors are larger and differently
  correlated across nodes.
- Errors from a pilot model instead of the production model, so the weights describe a model that no
  longer exists.

If you cannot produce out-of-sample errors at the required horizon, say so and use a structured
weight rather than a sample covariance of the wrong object.

### Phase B — Estimate the weight matrix

**Step 5. Assemble residuals on a common origin grid.** Every node must contribute a residual for the
same origins. A node with missing history cannot be dropped without changing `S`, and changing `S`
changes the problem. *Check:* the residual matrix is `T x n`, complete, and in the base unit.

**Step 6. Compute the sample covariance and its diagnostics.** Use a consistent denominator and state
it. Then compute, before looking at any reconciled number:

| Diagnostic | Formula or method | Reading | Action if bad |
|---|---|---|---|
| Rank | SVD of the `T x n` residual matrix | `min(n, T)` at most | If rank `< n`, the inverse does not exist |
| Smallest eigenvalue | SVD or eigendecomposition | Near zero means near-singularity | Shrink or use a structured weight |
| Condition number | Largest over smallest singular value | Above about 100 is fragile here | Shrink; report the number |
| `n / T` | Counts | Above 0.5 is uncomfortable | Prefer a structured weight |
| Correlation spread | Histogram of off-diagonal correlations | Many values above 0.9 imply near-collinearity | Shrink |
| Sign pattern of `W^-1` | Inspect the inverse | Large alternating entries amplify noise | Shrink |

The condition number is the most useful number in this table. A covariance with condition number 50
is usable; at 5,000 the inverse is mostly estimation noise, and the projection will move nodes in
directions that reflect sampling error rather than reliability.

**Step 7. Understand why the matrix is near-singular even when `T` is large.** Aggregate errors are
not independent of leaf errors. If every level were forecast perfectly except for noise at the
leaves, the error vector would be exactly `S` times the leaf error vector and `W` would have rank
`m`, singular by construction. Real hierarchies depart from that because each level has its own
model, which is what makes MinT useful; but the departure is a small signal on a near-collinear
structure, so `W` is badly conditioned even at `n / T` ratios that look comfortable.

**Step 8. Choose a weight variant and write down what it assumes.**

| Variant | `W` | Parameters estimated | Assumes | Fails when |
|---|---|---:|---|---|
| OLS | `I` | 0 | A unit of error is equally serious everywhere | Series differ greatly in scale; small series get distorted |
| WLSs (structural) | `diag(S 1)` | 0 | Leaf variances are equal and independent | Leaf scales differ by more than about 3:1 |
| WLSv (variance) | `diag(sigma_i^2)` | `n` | Errors are uncorrelated across nodes | Cross-level correlation is strong, which it usually is |
| MinT(Sample) | Sample covariance | `n(n + 1) / 2` | `T` is large relative to `n` | `n / T` above about 0.5; almost always in practice |
| MinT(Shrink) | `lambda D + (1 - lambda) W_s` | `n` plus one | Sample correlations are noisy but informative | Correlations are genuinely near zero and the target is right |

Structural scaling deserves a note because it is the most under-used variant. `S 1` has as its `i`-th
entry the number of leaves beneath node `i`; using that as the variance assumes every leaf has the
same error variance and that leaf errors are independent, so an aggregate of `k` leaves has variance
proportional to `k`. It requires no estimation, cannot be singular, and is often within a few
percent of the best estimated weight: for a first implementation, a better default than a sample
covariance.

**Step 9. Shrink, and record the intensity.** Take the sample covariance `W_s` and its diagonal `D`.
Form `W* = lambda * D + (1 - lambda) * W_s` with `lambda` in `[0, 1]`. Every off-diagonal entry is
multiplied by `(1 - lambda)`; every variance is unchanged. The result is positive definite for any
`lambda > 0` whenever `D` has positive entries, because it is a convex combination of a positive
definite matrix and a positive semi-definite one.

The intensity is estimated, not chosen by taste. The Schäfer–Strimmer idea, stated plainly: **your
sample correlations are noisy; estimate how noisy; pull every correlation toward zero by that
fraction, and leave the variances alone.** In formulas, with `r_ij` the sample correlation,

```text
   lambda_hat = sum_{i < j} Var(r_ij)  /  sum_{i < j} r_ij^2
```

and `lambda_hat` is clipped to `[0, 1]`. When the correlations are large relative to their sampling
noise, the denominator dominates and `lambda` is near zero, so almost nothing changes. When the
series are short and the correlations are modest, the numerator dominates and `lambda` approaches
one, so the matrix collapses toward the diagonal — which is WLSv. The estimator is analytic: no
cross-validation, no bootstrap, no tuning. It comes from genomics, where the same problem (`p` much
larger than `n`) was solved with the same trick in 2005.

> **Note:** Shrinkage of the covariance is not shrinkage of the forecasts. It does not pull forecasts
> toward zero or toward the mean. It changes only the metric in which the projection is measured, and
> it changes it in the direction of assuming less cross-node correlation than the sample suggests.

### Phase C — Build and cache the projection

**Step 10. Compute `P` once.** `P` is `m x n`. Form `S' W^-1 S`, which is `m x m`, factorise it
(Cholesky if it is well conditioned, SVD if not), and solve. If `m` is large, use the equivalent form
that inverts an `(n - m) x (n - m)` matrix via a basis for the null space of `S'`; if `n` is very
large, use a sparse factorisation or an iterative solver rather than materialising `W^-1`.

**Step 11. Verify four things before trusting the output.**

| Check | What it proves | Failure means |
|---|---|---|
| `S P S = S` | The map leaves coherent vectors unchanged | The projection is wrong, or `S` and `P` use different orderings |
| `P S = I` | The bottom-level map is a projection | Usually a sign error in `S` |
| Round-trip on a coherent vector | The map is idempotent on its own subspace | The implementation is not a projection |
| Coherence of the output, `C y_tilde = 0` | The arithmetic closed | Rounding tolerance or a constraint violation |

The round-trip check is the cheapest and the most often skipped. Feed in a vector that already adds
up and confirm it comes back unchanged to machine precision. A method that alters a coherent
forecast is not reconciling; it is doing something else and calling it reconciliation.

**Step 12. Cache `P` with a version key.** `P` depends on `S`, `W`, and the node ordering, and on
nothing else. It does not depend on the base forecasts. So compute it once per hierarchy version,
weight vintage, and horizon, and reuse it. The key should contain at least: the hierarchy identifier
and version, the node ordering hash, the weight variant name, the shrinkage intensity, the
estimation window end date, and the horizon. Invalidate on any change to the node set — a new SKU, a
discontinued item, a site closure, a pack-size change.

> **Note:** A cached `P` is a stored model. The commonest silent failure in production reconciliation
> is a `P` built for a 4,798-node hierarchy still being applied to a 4,812-node hierarchy after a
> range review, because the code broadcast the mismatch instead of failing.

### Phase D — Apply, then meet the constraints

**Step 13. Apply per horizon and check.** `y_tilde = S (P y_hat)`. One matrix-vector product. Check
coherence to a stated tolerance; check that the total is inside a plausible band; check the largest
single-node adjustment as a fraction of that node's base.

**Step 14. Test the constraints the projection cannot see.**

| Constraint | Why it arises | Does a linear projection respect it? |
|---|---|---|
| Non-negativity | Units cannot be negative | No |
| Integer values | Pallets, cases, whole units | No |
| Immutable nodes | Committed orders, frozen schedules | No |
| Capacity bounds | A plant cannot produce above its limit | No |
| Another hierarchy | Weeks must also sum to months | No, unless included in `S` |
| Ratio identities | Price times volume equals revenue | No, the constraint is non-linear |

**Step 15. Handle each constraint by an explicit route, and measure the cost.**

- **Clip, or set-negative-to-zero (sntz).** Replace negatives with zero. Cheap, one line, and it
  breaks coherence: the clipped vector no longer sums to its parent. Acceptable when negatives are
  tiny and rare, and only if the resulting incoherence is reported.
- **Clip and redistribute.** Set negatives to zero, then spread the gap across the remaining free
  nodes by the same weights. Restores coherence, changes the projection, and has no optimality claim.
  A pragmatic default in production.
- **Constrained least squares (NNLS or a quadratic programme).** Minimise the same weighted objective
  subject to the coherent subspace and the bounds. Correct, but no longer linear, no longer
  cacheable, and not guaranteed to reproduce the unconstrained answer when constraints are inactive.
  Block principal pivoting and operator-splitting quadratic programming are the usual solvers.
- **Immutables as equality constraints.** Fix the committed nodes and reconcile the remainder. This
  changes `S` effectively, so the cached `P` no longer applies.

> **Note:** Every constraint route trades one property for another. Clipping keeps the code simple
> and loses coherence. A quadratic programme keeps coherence and optimality and loses the cache and
> the closed form. Redistribution keeps coherence and loses the optimality claim. There is no route
> that keeps all four, and a proposal that claims otherwise is worth reading closely.

### Phase E — Means are not distributions

**Step 16. Decide whether a point forecast is what the decision needs.** If the decision consumes a
quantile — safety stock, service level, capacity buffer — a reconciled mean is not enough. You need a
reconciled distribution, and that requires a joint distribution over the nodes, not `n` marginal
distributions.

**Step 17. If you need a distribution, reconcile draws rather than quantiles.** Because the map is
linear, applying it to each draw of a joint sample produces a sample from the reconciled
distribution: simulate base errors under a stated dependence model, add the base mean, reconcile each
draw, and summarise. Reconciling quantiles independently fails, because the projection has negative
coefficients and quantiles do not commute with linear combinations.

### Phase F — Judge, sensitivity-test, and govern

**Step 18. Compare weight variants at the level where the decision is made.** Hold the base forecasts
fixed, use common origins and horizons, and report OLS, WLSs, WLSv, MinT(Sample), and MinT(Shrink)
side by side. The spread across variants measures how much the weight choice matters for this data:
if it is smaller than the decision tolerance, stop arguing about weights; if it is larger, the weight
estimate is the binding constraint on forecast quality.

**Step 19. Sensitivity-test the answer.** Perturb `W` within its estimation uncertainty (bootstrap
the residual panel, or vary `lambda`), recompute, and record how far the reconciled total and the
largest leaf move. A number that moves by 3% under a defensible change of weights should not be
presented to two decimal places, and probably should not be presented without the range.

**Step 20. Write the register.** One row per stream: hierarchy version, node count, weight variant,
`lambda`, estimation window, horizon, constraint handling, owner, last backtest, review date. Without
it, "we use MinT" is unfalsifiable and therefore unmanageable.

## 6. Worked example

Meridian Components distributes a consumable through two regions. The hierarchy has five nodes and
three leaves. The unit is eaches per week and the horizon is one week. The figures are constructed
for teaching and internally consistent; every number below can be reproduced with the arithmetic
shown.

### 6.1 The hierarchy and the base forecasts

```text
                        TOTAL
                       /     \
                  NORTH      SOUTH
                  /    \
                N1      N2

   n = 5 nodes.  m = 3 leaves.  Constraints = 2:
        Total = North + South
        North = N1 + N2
   Node order used everywhere below: T, N, S, N1, N2.
```

| Node | Base forecast (eaches/week) | Constraint check | Gap |
|---|---:|---|---:|
| Total | 1,180 | 720 + 520 = 1,240 | -60 |
| North | 720 | 430 + 250 = 680 | +40 |
| South | 520 | leaf | - |
| N1 | 430 | leaf | - |
| N2 | 250 | leaf | - |

Two gaps in opposite directions: the total is 60 units below the sum of the regions, while North is
40 above the sum of its SKUs. That is ordinary when the total, region, and SKU models are fitted
separately on different feature sets.

### 6.2 The error covariance

Thirty-six months of rolling one-week-ahead errors give these standard deviations and correlations.
The structure is realistic: a common demand factor makes everything positively correlated, and the
aggregates inherit their children's errors, so the correlation between North and N1 is higher than
the correlation between South and N1.

| Node | Error standard deviation (eaches) | Variance (eaches squared) |
|---|---:|---:|
| Total | 94.0 | 8,844 |
| North | 73.2 | 5,357 |
| South | 46.0 | 2,121 |
| N1 | 44.5 | 1,977 |
| N2 | 35.0 | 1,224 |

| | T | N | S | N1 | N2 |
|---|---:|---:|---:|---:|---:|
| T | 1.00 | 0.77 | 0.60 | 0.75 | 0.71 |
| N | 0.77 | 1.00 | 0.42 | 0.82 | 0.76 |
| S | 0.60 | 0.42 | 1.00 | 0.42 | 0.41 |
| N1 | 0.75 | 0.82 | 0.42 | 1.00 | 0.57 |
| N2 | 0.71 | 0.76 | 0.41 | 0.57 | 1.00 |

The covariance rounded to tens is the matrix in panel A of section 3. Its eigenvalues are
approximately 302.7, 659.7, 1,057.9, 1,967.6, and 15,532.1, so the condition number is about 51.
With `n = 5` and `T = 36` the sample covariance is invertible and not pathological. This is a
deliberately comfortable example, and the inverse still has entries that alternate in sign, which is
where the interesting behaviour comes from.

### 6.3 Shrinkage arithmetic

Take `lambda = 0.35`, as an estimated intensity would plausibly be for 36 monthly residuals and five
correlated series; section 5 step 9 gives the estimator, and the value is data-dependent. Every
correlation is multiplied by 0.65 and every variance is unchanged. The eigenvalues become
approximately 801.7, 1,178.2, 1,715.8, 3,355.9, and 12,468.4, and the condition number falls from 51
to about 15.6.

That is the point of shrinkage in one number: the same data and hierarchy, a matrix whose inverse is
roughly three times less sensitive to perturbation. The forecasts did not change; the metric did.

### 6.4 Five reconciliations of the same base

Each variant was computed with its own `W` and the same base vector. Figures are in eaches per week,
to two decimals; sums may differ by 0.01 from rounding.

| Node | Base | OLS | WLSs | WLSv | MinT(Sample) | MinT(Shrink) |
|---|---:|---:|---:|---:|---:|---:|
| Total | 1,180.00 | 1,197.50 | 1,204.00 | 1,203.84 | 1,195.88 | 1,200.91 |
| North | 720.00 | 695.00 | 692.00 | 689.55 | 682.72 | 687.54 |
| South | 520.00 | 502.50 | 512.00 | 514.28 | 513.17 | 513.37 |
| N1 | 430.00 | 437.50 | 436.00 | 435.91 | 431.75 | 434.74 |
| N2 | 250.00 | 257.50 | 256.00 | 253.64 | 250.96 | 252.80 |

Every column is coherent. The totals span 1,195.88 to 1,204.00, a range of 8.1 units, or 0.7% of the
base total. The North forecasts span 682.72 to 695.00, a range of 12.3 units, or 1.7%. The choice of
weights matters less than the size of the incoherence it is correcting, and both are smaller than
the base forecast's own standard error of 94 units.

### 6.5 Reading the differences

| Observation | Explanation |
|---|---|
| Every method moves the total up and North down | The gap structure fixes the signs: the total is below its children, North is above its own |
| South moves down in every variant although South has no gap of its own | The total constraint `Total = North + South` forces a correction onto South once North is moved |
| OLS moves every node by 7.5 to 25 units | The Euclidean projection spreads the correction without reference to scale, so the small series are distorted proportionally more |
| WLSv moves the total most (23.84) | The total has the largest variance (8,844), so it is trusted least and corrected most |
| MinT(Sample) moves North most (-37.28) | North's error is highly correlated with N1 and N2; the inverse exploits that correlation and corrects North aggressively |
| MinT(Shrink) sits between MinT and WLSv | Shrinkage scales the correlations by 0.65, so the aggressive correlation-driven correction is damped |

The comparison that matters is not "which is right", since there is no realised outcome here. It is
that five defensible choices produce five different coherent answers from one base, and that the
differences concentrate at the node with the most contradictory information. That concentration is a
diagnostic: North is where the level-specific models disagree most, and the node to investigate
before trusting any of the five.

### 6.6 When the answer is negative

Take a three-child subtree: Total 100, A 60, B 45, C 1, with `W = diag(300, 800, 900, 1600)` for
Total, A, B, C. The gap is `100 - 106 = -6`, so the children collectively exceed the parent. In the
diagonal case each node moves by `d * w_i / sum(w)` with `sum(w) = 3,600`:

| Node | Calculation | Reconciled |
|---|---|---:|
| Total | `100 + 6 x 300 / 3,600` | 100.500 |
| A | `60 - 6 x 800 / 3,600` | 58.667 |
| B | `45 - 6 x 900 / 3,600` | 43.500 |
| C | `1 - 6 x 1,600 / 3,600` | -1.667 |

The vector sums to 100.500, coherent to rounding, and contains a negative quantity of a physical
item. C had the smallest base and the largest variance, so it absorbed the largest share of a
negative correction and went below zero. Nothing in the method noticed. This is the ordinary
mechanism: negativity arises when a small node carries a large weight and is asked to absorb a
correction larger than its own value, which happens most often when a sibling's base exceeds the
parent total.

Three routes, with their costs. **Clip:** set C to zero, and the vector sums to 102.167 against a
total of 100.500, coherent no longer. **Clip and redistribute:** move the 1.667 back onto A and B in
proportion to their weights, so A falls by `1.667 x 800 / 1,700 = 0.784` to 57.883 and B by 0.882 to
42.618; coherent again, but no longer the minimiser of the original objective. **Constrain
properly:** solve the same weighted least-squares problem with `b >= 0` as a quadratic programme;
correct in principle, iterative in practice, and no longer a fixed linear map.

There is also a fourth response, often the right one: **investigate C**. A series with a base of 1
unit per week and an error variance of 1,600 has a coefficient of variation of 40. Either it is
genuinely tiny and intermittent, in which case the variance estimate is unstable and shrinkage will
fix most of the problem, or one of the two numbers is wrong.

## 7. Code, query, or calculation

One hand computation. The point is to see the projection matrix, including its negative entries, and
to confirm that the reconciled vector is coherent without any library.

```text
Three-node hierarchy: Total = A + B.  Base forecasts y = (100, 60, 35).
Error covariance W (eaches squared), rows and columns ordered T, A, B:

      [ 900  500  300 ]
  W = [ 500  400  150 ]     eigenvalues 73.8, 139.4, 1,311.8; condition 17.8
      [ 300  150  225 ]

Shrink toward the diagonal with lambda = 0.35: off-diagonals x 0.65,
variances unchanged. Condition falls to 6.5.

P = (S' W^-1 S)^-1 S' W^-1        (2 x 3; rows A, B; columns T, A, B)
  row A = [ 0.2537   0.7463  -0.2537 ]
  row B = [ 0.1875  -0.1875   0.8125 ]

b       = P y      = (61.268, 35.938)
y_tilde = S b      = (97.206, 61.268, 35.938)
check   : 61.268 + 35.938 = 97.206
row sums: 0.7463 and 0.8125 (not 1: P maps into the leaf space)

Read row A: reconciled A is 25% of the base total, 75% of its own base,
minus 25% of base B. That negative weight is not an error: if the base
total rises by 1 unit, reconciled A rises by 0.2537 and B by 0.1875.
```

Four checks before trusting any implementation of this calculation.

**`S P S = S`.** For the matrix above, `S P S` reproduces `S` to four decimals: the first column of
`P S` is `(1, 0)` and the second is `(0, 1)`. A projection that fails this is not coherent.

**Round-trip.** Feed `(97.206, 61.268, 35.938)` back through `P` and confirm you get the same vector.
If it moves, the map is not idempotent and something is wrong with the ordering.

**Scale.** Multiply every base forecast by 1,000 and confirm every reconciled value scales by 1,000.
OLS passes this, and fails the opposite test: rescale one series by 1,000 and it distorts that series
against the others, which is why OLS distorts small series.

**Negativity.** With `W` as given, reconciled B is 35.938, above its base, because the gap is
positive. Change the base to `(30, 28, 20)` and check whether any reconciled value crosses zero; the
negative coefficient in each row makes it possible even though every input is positive.

## 8. Data, standards, and regulatory context

No standard prescribes how to reconcile forecasts. There is no ISO committee for weight matrices and
no regulation that says a reconciled number must use a shrinkage covariance. What exists is a set of
standards that constrain the inputs — identifiers, units, calendars, accounting boundaries — and a
set of governance expectations that constrain the process. Both matter, because a method that is
mathematically correct on badly identified data is wrong in a way no amount of linear algebra fixes.

### 8.1 What governs what

| Requirement | Body | What it constrains | Binding? | Grade |
|---|---|---|---|---|
| GTIN, GLN, SSCC identifiers | GS1 | Which node is which, and that the pack hierarchy is identifiable | Voluntary standard, contractually mandatory in retail | B |
| Global Product Classification | GS1 | The category taxonomy above the item | Voluntary | B |
| ISO 8601-1:2019 | ISO | Week and date representation, hence which buckets nest | National adoption varies | B |
| IFRS 8, *Operating Segments* | IASB | Reconciliation of segment totals to the financial statements | Binding for entities applying IFRS | B |
| SR 11-7, model risk guidance | US Federal Reserve | Validation, documentation, and monitoring of models, including forecasting components | Supervisory guidance, not a rule | B |
| National accounts benchmarking practice | Statistical offices | Reconciling quarterly and annual series to annual totals | Methodological convention | A |

### 8.2 Identifiers and the pack hierarchy

The GS1 system supplies the identifiers that make a product hierarchy joinable. A Global Trade Item
Number (GTIN) identifies a trade item at a packaging level, and the Global Location Number (GLN) does
the same for sites. The consequence for reconciliation is narrow and important: **`S` is defined over
identifiers, so a change of identifier is a change of `S`.** A pack-size change from twelve to ten
eaches does not change the physical product and does change the summing matrix if the hierarchy is
maintained in cases, and the cached projection is then wrong, coherently and silently.

### 8.3 Units, calendars, and rounding

Reconcile in the base unit and convert at the boundary. A reconciled vector in cases that is then
rounded to pallets is coherent before rounding and not after. If the output must be integral, the
rounding rule is part of the method: largest-remainder rounding preserves the parent sum, independent
rounding does not. State the rule in the register, because "the numbers do not add up" is usually a
rounding rule nobody wrote down.

ISO weeks do not nest into Gregorian months, and a hierarchy that reconciles across products and
locations but not across time is coherent in one dimension and unusable in another. Module 08 topic
04 treats cross-temporal reconciliation; the point here is that the weight matrix is horizon- and
calendar-specific, and a `W` estimated on weekly errors is the wrong metric for a monthly
reconciliation.

### 8.4 The accounting interface and model governance

IFRS 8 requires disclosure of reconciliations of segment totals to the financial statements. That is
an obligation about a bridge, not a method for producing one. Read it as a design precedent: a total
presented without a bridge invites the reader to assume the difference is an error. Where a
reconciled forecast feeds a regulated process — capital planning, regulatory reporting, pricing in a
supervised industry — the projection is a model component, and SR 11-7 style expectations apply:
documented purpose, documented data, validation against alternatives, ongoing monitoring, a named
owner. The register of section 5 step 20 is not bureaucracy; it is the minimum evidence that the
number was produced deliberately.

### 8.5 What none of this constrains

None of these standards says whether to use OLS, WLS, or MinT; whether to shrink; which target to
shrink toward; how to handle negative values; or whether to reconcile at all. An appeal to a standard
in a weights argument is usually an appeal to authority the standard does not supply.

## 9. Industry practice and case evidence

The evidence base for reconciliation is stronger than for most topics in this program, because the
method originated in academic forecasting and has been tested in public competitions. It is still
thin on what firms actually run, and the gap between the two should be labelled.

### 9.1 The origin, and why it is worth knowing

Optimal combination reconciliation was formalised by Hyndman, Ahmed, Athanasopoulos, and Shang in
2011, building on least-squares reconciliation methods used in official statistics since the 1970s.
Their formulation required the covariance matrix of the reconciliation errors. Wickramasuriya,
Athanasopoulos, and Hyndman showed in a paper circulated from 2017 and published in the *Journal of
the American Statistical Association* in 2019 that this matrix is not identifiable: because the
reconciled vector lies in an `m`-dimensional subspace of `R^n`, its error covariance has rank at most
`m` and cannot be recovered from data, however much you have. Their contribution was to replace it
with the covariance of the base forecast errors, which is estimable, and to name the family minimum
trace (MinT), after the objective `tr(S P W P' S')`.

Two things follow. MinT and generalised least squares (GLS) with the base error covariance give the
same projection matrix; they differ only in which covariance the method is understood to require.
And the name is a statement about an objective, not a claim about a unique method: the variants
differ entirely in how `W` is estimated.

### 9.2 What the simulations and applications report

The MinT paper evaluates six weight estimators in Monte Carlo designs covering correlation,
aggregation smoothing, seasonality, and a large five-level hierarchy, and in an application to
Australian domestic tourism data with 555 series. As reported there, MinT(Shrink) performs best
overall, MinT(Sample) approaches it as the number of observations grows, bottom-up performs poorly
except in very small samples, and variance scaling is the best of the diagonal methods.

The 2024 review by Athanasopoulos, Hyndman, Kourentzes, and Panagiotelis in the *International
Journal of Forecasting* is the best entry point to the subsequent literature. Its finding is
conditional rather than triumphant: no method dominates across hierarchies, and rankings depend on
the data structure, the evaluation level, and the weights. That is why this topic spends a phase on
choosing weights rather than on proving one is best.

### 9.3 The shrinkage debate

Three positions are visible in the literature.

**Shrink fully toward the diagonal.** The MinT(Shrink) result above, and the argument that a
well-conditioned biased estimator beats an unbiased singular one. This is the mainstream position.

**Shrink partially, in blocks.** Pritularga, Svetunkov, and Kourentzes argue that full-information
shrinkage has high variability, and propose estimators that shrink within blocks of related series
rather than toward a single diagonal, on the claim that the block structure captures the correlation
that is real.

**Skip the covariance and use a structured weight.** Structural scaling needs no estimation, cannot
be singular, and is competitive when leaf scales are similar. The MinT paper itself reports variance
scaling winning at longer horizons in some designs, a reminder that best average performance is not
best performance in every regime.

The lesson is not a winner. The choice is empirical, it should be made on out-of-sample evidence at
the decision level, and reporting two variants is cheap insurance against having picked the wrong
one.

### 9.4 Scale

Hyndman, Lee, and Wang (2016) showed how to exploit the structure of `S` so that a 100,000-leaf
hierarchy can be reconciled without materialising a dense projection, using recursion for trees,
Kronecker methods for two-way grouped structures, and sparse solvers in general. A 2026 preprint
reports reconciling on the order of billions of forecasts with sparse constraint matrices and
iterative solvers — a reminder that the linear algebra is not the binding constraint at scale, the
covariance estimate is.

> **Note:** The scale literature is about applying a chosen `W` efficiently, not about estimating a
> better `W`. A fast solver applied to a singular covariance is a fast way to be wrong.

### 9.5 What firms appear to do

The public record on corporate practice is mostly inference. The table labels it as such.

| Practice | Evidence status | Reasoning |
|---|---|---|
| Bottom-up as the replenishment default | Inference from system design | Replenishment needs a number per SKU per location; summation is the cheapest coherent answer |
| Proration in finance-facing plans | Inference from the prevalence of top-down allocation in enterprise tools | One line of arithmetic, no covariance |
| OLS or WLS structural in the statistical centre of excellence | Inference from vendor documentation and conference material | No estimation of `W`, so no failure mode to defend |
| MinT(Shrink) where a forecasting team owns the hierarchy | Inference from the applied literature | The method is documented, the diagnostics are standard, and the team can defend the intensity |
| No reconciliation at all | Inference from the absence of a register in most organisations | The levels are owned by different teams and nobody has been asked to make them agree |

No named public case of a reconciliation failure exists, in the sense of a company and an audited
cost. Companies do not publish their forecasting errors. Treat the absence as a limitation of the
evidence, not as evidence that the failures are rare.

### 9.6 What practitioners argue about

Five arguments recur, and knowing them is more useful than knowing the method.

1. **Whether to reconcile at all.** The coherent number is not necessarily the accurate one, and a
   wrong top-level number propagated to every descendant is harder to challenge than an incoherent
   set with a documented bridge.
2. **Which residuals.** In-sample residuals are available and optimistic; rolling out-of-sample
   errors are honest and expensive. The choice changes the weights more than the choice of variant.
3. **One `W` per horizon or one pooled `W`.** Horizon-specific is correct and multiplies the
   estimation problem by the number of horizons.
4. **Sample, shrink, or structure.** Section 9.3.
5. **What to do with negatives.** Clip, redistribute, or solve a quadratic programme — usually
   decided by whoever owns the downstream system, not by the forecaster.

## 10. Failure modes and red flags

| # | Failure mode | Red flag in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Sample covariance from too little data | `n / T` above 1 and "MinT" on the slide | Sample covariance has rank at most `T - 1` | Singular matrix, pseudoinverse, or a crash | Report `n`, `T`, and rank with every run; shrink or use a structured weight |
| 2 | Pseudoinverse used silently | The code runs and nobody knows which directions were dropped | A singular matrix is inverted on a subspace | Weights that assume zero variance where there is none | Fail loudly on rank deficiency; require an explicit decision |
| 3 | Weights estimated on in-sample residuals | Backtest accuracy far better than production accuracy | Residuals are optimistic and overfitted | The metric describes a model that does not exist | Estimate on rolling-origin errors at the target horizon |
| 4 | One `W` for all horizons | The twelve-week forecast moves like the one-week forecast | Error covariance grows and changes shape with horizon | Intervals and corrections too small at long horizons | Estimate per horizon, or state the pooling assumption |
| 5 | Stale cached projection | The hierarchy has 4,812 nodes; `P` was built for 4,798 | Cache key does not include the node set | Silent misalignment or broadcast error | Version key including a hierarchy hash; fail on dimension mismatch |
| 6 | Negative reconciled values | A leaf below zero in a plan review | Linear projection ignores inequality constraints | Manual overrides, lost trust, incoherent patches | Detect, then clip with disclosure, redistribute, or solve a quadratic programme |
| 7 | "Optimal" with no conditioning set | "MinT is the optimal reconciliation method" | Optimality is conditional on `W` being true | False confidence; a simpler method may win out of sample | State the conditioning set whenever the word appears |
| 8 | Weight choice invisible | Two analysts, same base, different totals | Different `W` variants or windows | Unreproducible numbers and unproductive argument | Register the variant, intensity, window, and horizon |
| 9 | Residuals and errors conflated | The covariance is computed from fitted values | In-sample versus out-of-sample | Understated uncertainty everywhere | Align the error object with the forecast object |
| 10 | Bias treated as variance | Base forecasts are biased; reconciliation makes it worse | The projection minimises variance, not bias | Every node inherits and sometimes amplifies the bias | Test bias at the total before reconciling; correct the base first |
| 11 | Quantiles reconciled independently | Reconciled 5th percentile above the reconciled median | Quantiles do not commute with linear maps | Crossing intervals and wrong safety stock | Reconcile draws from a joint sample, then summarise |
| 12 | Immutable nodes moved | A committed order changes after reconciliation | Commitments treated as free forecasts | Plan and system disagree; customer impact | Flag immutables and treat them as equality constraints |
| 13 | Coherence achieved by rounding | Parent and children agree to the unit but not to the decimal | Independent rounding after projection | Small permanent gaps read as forecast error | Largest-remainder rounding, applied once, documented |
| 14 | Weight matrix estimated on a different hierarchy | A range review added and removed SKUs last quarter | `W` rows correspond to nodes that no longer exist | Misaligned weights, arbitrary corrections | Rebuild `W` and `P` on every structural change |

Two rows deserve a note. **Row 5** costs the most and is discussed the least: the mathematics is
right, the code is right, and the matrix describes last quarter's hierarchy, so the output is
coherent but allocated to the wrong nodes — worse than an incoherent vector, because it passes every
arithmetic check. **Row 10** is the conceptual failure: if the base forecasts are biased, the bias is
not in the covariance in any way the projection can remove.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The hierarchy is a tree with additive constraints | Gives one summing matrix and one coherent subspace | Crossed dimensions; ratios; prices | Check one parent per node; ask whether the parent is a sum or a mean |
| 2 | The base error covariance is stable over the horizon | `W` estimated in the past is applied in the future | Promotions, regime change, model retraining, new items | Compare `W` across rolling windows; test for structural breaks |
| 3 | Base errors are the right object to weight | The projection's metric is reliability | Errors are asymmetric in cost: stockout versus holding | Ask what an error costs in each direction; if asymmetric, weight the loss, not the variance |
| 4 | `W` is estimable at the required size | The formula needs an inverse | `n` close to or above `T`; near-collinear aggregates | Report eigenvalues and condition number; require a threshold |
| 5 | Residuals represent forecast errors | The covariance is computed from residuals | In-sample fitting; one-step versus `h`-step | Compare residual and out-of-sample error variances at the target horizon |
| 6 | Shrinkage toward the diagonal is appropriate | It supplies the missing structure | True correlations are strong and stable, and the target destroys them | Compare `lambda` estimates across windows; test block targets |
| 7 | The trace objective reflects decision value | It defines "optimal" | Leaves and aggregates differ in decision importance | Ask how many decisions consume each level; consider a weighted trace |
| 8 | The projection is the right operation | Reconciliation is a projection | Base forecasts are biased, and the bias is level-specific | Test whether reconciliation beats the base at the decision level |
| 9 | Constraints can be handled after projection | The projection is linear | Non-negativity, integer, immutable, and capacity constraints bind often | Count violations per run; if frequent, embed the constraints |
| 10 | A reconciled mean is enough | Most planning systems consume a point | The decision is a quantile: safety stock, capacity buffer | Ask which quantile the downstream rule uses, and where it comes from |
| 11 | The node set is fixed over the horizon | `S` and `W` are constant | New items, discontinuations, reorganisations | Diff the node list across the horizon; version the hierarchy |

Three limits deserve to be stated in their own words.

**MinT inherits whatever the base forecasts got wrong.** A projection cannot add information. If the
total is biased upward by 4%, the reconciled total is biased upward by approximately 4%, and every
node now carries a share of that bias. The guarantee is about variance under unbiasedness, and it is
silent when the base is biased.

**The optimality is conditional and the condition is an estimate.** The trace is minimised given
`W`. You do not have `W`; you have `W_hat`, estimated from residuals whose own covariance is unknown.
The gap between the two is the gap between the oracle and the procedure, and it is large exactly when
the hierarchy is large, which is exactly when reconciliation is most needed.

**The objective is scale-dependent.** The trace weights every node equally in squared units,
regardless of how many decisions consume it, so a hierarchy with 500 leaves and 5 aggregates is in
effect optimising for the leaves. That may be right; it should be a stated choice rather than an
unexamined consequence of the formula.

## 12. Exercises

Nine exercises: 1 to 4 are calculation, 5 and 6 application, 7 and 8 design, 9 critique.

**Exercise 1 — Count the parameters.** Three hierarchies: (a) 1 total, 3 regions, 9 categories, 180
SKUs; (b) 1 total, 4 regions, 20 families, 400 SKUs; (c) 1 total, 6 regions, 60 families, 2,400 SKUs.
Assume 36 monthly residual vectors.

**Deliverable:** for each, `n`, `m`, the number of covariance parameters, `n / T`, and a one-line
verdict on whether the sample covariance is singular, ill-conditioned, or usable.

**Acceptance criteria:** all three node counts correct; parameters computed as `n(n + 1)/2`; the
verdict names the rank bound `min(n, T)`; at least one hierarchy is identified as singular by
construction. **Estimated time:** 25 minutes.

**Exercise 2 — Sequential versus simultaneous projection.** Using the base forecasts and variances
from section 6, apply the WLSv weights to the Total
constraint first and then to the North constraint, treating the two constraints one at a time.
Compare the result with the simultaneous WLSv column of section 6.4.

**Deliverable:** the reconciled Total, North, South, N1, and N2 after each step, the coherence check
after the second step, and the simultaneous values for comparison.

**Acceptance criteria:** the step-1 total is 1,212.50; the step-2 North is 687.59; the total constraint
is shown to be broken by 12.70 after step 2; the simultaneous column is quoted correctly; one
sentence explains why the sequential result is not a valid reconciliation. **Estimated time:** 35
minutes.

**Exercise 3 — Shrinkage by hand.** A two-series residual panel has `T = 36`, sample correlation `r = 0.62`, and standard deviations 40
and 25. Assume the large-sample variance of a correlation estimate is approximately
`(1 - r^2)^2 / T` for each off-diagonal pair.

**Deliverable:** the estimated shrinkage intensity `lambda`, the shrunk correlation, and the shrunk
covariance matrix. State whether the result is positive definite and why.

**Acceptance criteria:** `lambda` computed from the formula and clipped to `[0, 1]`; the shrunk
correlation equals `(1 - lambda) * 0.62`; the determinant is positive; the explanation names the
diagonal target and the unchanged variances. **Estimated time:** 25 minutes.

**Exercise 4 — Read a projection matrix.** The shrunk three-node projection in section 7 is

```text
  row A = [ 0.2537   0.7463  -0.2537 ]
  row B = [ 0.1875  -0.1875   0.8125 ]
```

**Deliverable:** the reconciled vector for base `(100, 60, 35)`, the effect on each reconciled leaf
of a one-unit increase in the base total, and an explanation of why a negative coefficient is not an
error.

**Acceptance criteria:** the vector is `(97.206, 61.268, 35.938)`; the effects are +0.2537 on A and
+0.1875 on B; the explanation distinguishes a projection weight from a share and notes that the rows
do not sum to one. **Estimated time:** 20 minutes.

**Exercise 5 — The negative leaf.** A four-node hierarchy has Total = A + B + C with base
`(100, 60, 45, 1)` and diagonal weights `(300, 800, 900, 1600)`.

**Deliverable:** the reconciled vector, the identification of the negative node, the clipped vector
and its coherence gap, and the redistributed vector that restores coherence.

**Acceptance criteria:** C reconciles to -1.667; the clip leaves a gap of 1.667; the redistribution
gives A = 57.883 and B = 42.618; all arithmetic shown; one sentence on which route you would choose
and why. **Estimated time:** 30 minutes.

**Exercise 6 — Means, quantiles, and a vendor claim.** A vendor states: "Our reconciliation produces
coherent 80% prediction intervals at every level by reconciling the 10th and 90th percentiles of the
base forecasts."

**Deliverable:** a written critique of at least five sentences, identifying what is missing and
proposing a correct alternative.

**Acceptance criteria:** identifies the need for a joint distribution; explains that quantiles do not
commute with a linear map with negative weights; notes that reconciled percentile vectors can cross;
proposes reconciling draws from a stated dependence model and summarising afterwards; mentions that
the resulting intervals are narrower than the naive combination because the projection reduces
variance. **Estimated time:** 30 minutes.

**Exercise 7 — Design the cache.** A production system reconciles 3,000 nodes weekly at four
horizons with MinT(Shrink), and the hierarchy changes on average twice a quarter.

**Deliverable:** a cache key specification, an invalidation policy, and a list of the checks that run
on every cache hit.

**Acceptance criteria:** the key contains hierarchy version, node-order hash, weight variant,
intensity, window end date, and horizon; invalidation covers node additions, removals, pack-size
changes, and weight-window rollover; at least three runtime checks; a stated fallback if the key does
not match. **Estimated time:** 40 minutes.

**Exercise 8 — Design the weight comparison.** You must choose between OLS, WLSs, WLSv, and
MinT(Shrink) for a hierarchy with 137 nodes and 36 months of history, evaluated where the SKU-level
plan is consumed.

**Deliverable:** a one-page test design naming what is held fixed, the origins, the horizon, the
metric, the level weights, and the decision rule.

**Acceptance criteria:** base forecasts held fixed across variants; common origins; evaluation at the
consumption level and at the total; the metric stated with units; a rule for what happens if two
variants are within 1% of each other; a statement of the estimation uncertainty in `W`. **Estimated time:** 40 minutes.

**Exercise 9 — Rewrite the slide.** A slide reads: "MinT is the optimal way to reconcile hierarchical
forecasts. It minimises total error across all levels and removes the need for judgement in
weighting."

**Deliverable:** a three-sentence rewrite that preserves what is true, removes what is false, and
names the conditions under which a simpler weight can win.

**Acceptance criteria:** the rewrite names `W` and the conditioning; it distinguishes minimising
variance from minimising error; it states that the trace objective is unweighted across nodes; no
sentence claims the method is best in general; the rewrite fits in three sentences. **Estimated time:** 20 minutes.

## 13. Self-check answers

Answers to all nine exercises. Check your arithmetic against these before reading the reasoning.

**Answer 1.** (a) `n = 1 + 3 + 9 + 180 = 193`, `m = 180`, parameters `193 x 194 / 2 = 18,721`,
`n / T = 5.36`. (b) `n = 425`, `m = 400`, `425 x 426 / 2 = 90,525`, `n / T = 11.8`. (c) `n = 2,467`,
`m = 2,400`, `2,467 x 2,468 / 2 = 3,044,278`, `n / T = 68.5`.

All three are singular, because `T = 36` bounds the rank of the sample covariance at 35 regardless of
`n`. The useful distinction is how far the ratio sits above one and therefore how much shrinkage has
to do. At realistic sizes the unrestricted sample covariance is not an estimator that needs
improving. It is an estimator that does not exist.

**Answer 2.** Step 1, the Total constraint. The gap is `1,180 - 720 - 520 = -60`; weights
`(8,840, 5,360, 2,120)` sum to 16,320. Total `+60 x 8,840 / 16,320 = +32.50` to 1,212.50; North
`-60 x 5,360 / 16,320 = -19.71` to 700.29; South `-60 x 2,120 / 16,320 = -7.79` to 512.21. Check:
`700.29 + 512.21 = 1,212.50`.

Step 2, the North constraint. North is now 700.29 and its children are 430 and 250, so the gap is
`+20.29`; weights `(5,360, 1,980, 1,220)` sum to 8,560. North `-20.29 x 5,360 / 8,560 = -12.70` to
687.59; N1 `+4.69` to 434.69; N2 `+2.89` to 252.89. Check: `434.69 + 252.89 = 687.58`.

Now the check that matters: `687.59 + 512.21 = 1,199.80`, not 1,212.50. The second projection moved
North by 12.70 and broke the constraint the first had satisfied. Each sequential step repairs one
constraint and damages the other, so the vector is never coherent.

The simultaneous projection solves both at once: Total 1,203.84, North 689.55, South 514.28, N1
435.91, N2 253.64, the WLSv column of section 6.4. Projecting onto the intersection of two constraint
surfaces is not the same as projecting onto each in turn. MinT is defined by the simultaneous
projection; a hand calculation that applies constraints one at a time is computing a different and
generally inferior estimator.

**Answer 3.** With `r = 0.62` and `T = 36`, the approximate sampling variance of the correlation is
`(1 - 0.62^2)^2 / 36 = (1 - 0.3844)^2 / 36 = 0.6156^2 / 36 = 0.3789 / 36 = 0.01053`. There is one
off-diagonal pair, so `lambda = 0.01053 / 0.62^2 = 0.01053 / 0.3844 = 0.027`, clipped to `[0, 1]`.

The shrunk correlation is `(1 - 0.027) x 0.62 = 0.603`, giving covariance `[[1,600.0, 603.0],
[603.0, 625.0]]` with variances 40 squared and 25 squared unchanged. The determinant is
`1,600 x 625 - 603^2 = 1,000,000 - 363,609 = 636,391`, positive, and the leading entry is positive,
so the matrix is positive definite.

The intensity is small because 36 observations and a correlation of 0.62 are enough information to be
fairly confident the correlation is not zero. With `T = 8` the same correlation gives `lambda =
0.123`, and with `T = 4` about 0.25: shrink more when you know less. That is also why the estimator
beats a fixed `lambda = 0.5`, which encodes no information about the data at all.

**Answer 4.** Row A gives reconciled A `= 0.2537 x 100 + 0.7463 x 60 - 0.2537 x 35 = 61.27`; row B
gives `0.1875 x 100 - 0.1875 x 60 + 0.8125 x 35 = 35.94`. The vector is `(97.206, 61.268, 35.938)`.

A one-unit rise in the base total raises reconciled A by 0.2537 and B by 0.1875, so the reconciled
total rises by 0.4412, not by 1. That is not a paradox: the total is a dependent node whose
reconciled value is the sum of the reconciled leaves, and a rise in the base total is mostly treated
as a rise in the gap rather than as a rise in demand.

The negative coefficient means the projection is not a set of shares. A share is non-negative and
sums to one down a column; a projection weight sums across a row to less than one here (0.7463 for A,
0.8125 for B) and can be negative, because the reconciled value of a node may fall when another
node's base rises. That is how reconciliation moves a node against its own gap.

**Answer 5.** Gap `d = 100 - 60 - 45 - 1 = -6`; `sum(w) = 3,600`. Total
`100 + 6 x 300 / 3,600 = 100.500`; A `60 - 6 x 800 / 3,600 = 58.667`; B `45 - 6 x 900 / 3,600 =
43.500`; C `1 - 6 x 1,600 / 3,600 = -1.667`. Check: `58.667 + 43.500 - 1.667 = 100.500`.

Clipping C to zero gives `102.167`, a coherence gap of 1.667 units, or 1.66% of the total.
Redistributing that gap across A and B in proportion to their weights: A falls by
`1.667 x 800 / 1,700 = 0.784` to 57.883, B by `1.667 x 900 / 1,700 = 0.882` to 42.618, and the vector
sums to 100.501, coherent to rounding.

A defensible answer chooses redistribution when negatives are rare and small, a quadratic programme
when they are frequent or the constrained solution is materially different, and investigation when
the offending series has a coefficient of variation above about 1. The last option is the one most
often skipped and the one most likely to find a data problem.

**Answer 6.** The claim is wrong in four ways.

First, reconciling percentiles is not reconciling a distribution. A quantile of a linear combination
is not the linear combination of the quantiles unless the map is monotone and the dependence is
degenerate; with negative weights, the reconciled 10th percentile can sit above the median.

Second, the joint dependence is missing. Two nodes with the same marginals and different copulas
produce the same base percentiles and different reconciled percentiles.

Third, the result need not be a valid set of quantiles: independently reconciled percentile vectors
can cross, producing an interval that is empty or inverted at some level.

Fourth, the intervals will be too wide. Reconciliation reduces variance because the reconciled total
combines several noisy estimates of one quantity; an interval built from base percentiles does not
inherit that reduction, so it overstates uncertainty even when it is otherwise coherent.

The correct alternative is to simulate: draw a joint sample under a stated dependence model, add the
base mean, apply the projection draw by draw, and summarise. Score the result with an energy or
variogram score rather than the log score, which is improper for comparing incoherent with coherent
forecasts.

**Answer 7.** A workable key is a tuple: hierarchy identifier; hierarchy version or node-list hash; weight variant
name; shrinkage intensity to four decimals; estimation window end date; forecast horizon index; and
node-ordering hash. It should be stored with the matrix, not alongside it in a log.

Invalidation: any change to the node set, including additions, removals, and pack-size changes; any
change to the weight variant or intensity; rollover of the estimation window; retraining of any base
model that contributes residuals; a change of horizon grid; and a change in constraint handling that
alters the projection.

Checks on every cache hit: the stored dimension matches the current `n` and `m`; the node-order hash
matches; the key's hierarchy version matches the live hierarchy; a stored checksum of `S P S - S` is
below tolerance; and the projection's condition number is below the agreed threshold.

Fallback if the key does not match: rebuild synchronously and fail the run if the rebuild fails,
rather than applying a stale matrix. The worst outcome is a silent mismatch; a failed job is
recoverable.

**Answer 8.** Hold fixed: the base forecasts, the origins, the horizon, the evaluation buckets, and the node set.
Vary: only the weight matrix.

Origins: rolling, at least 24, spaced to cover a full seasonal cycle. Horizon: the horizon at which
the plan is consumed. Metric: mean absolute error in eaches at the SKU level and at the total, with
the level weights stated. If a single score is required, weight by the number of decisions at each
level rather than by the number of series, and say so.

Decision rule: choose the variant with the lowest error at the consumption level, subject to the
total not deteriorating by more than 1%. If two variants are within 1% at both levels, choose the
simpler one, and record that the choice was made on parsimony rather than accuracy.

Estimation uncertainty: report the spread of results across bootstrap replications of the residual
panel. If that spread is larger than the difference between the top two variants, the comparison does
not resolve the choice, and the honest conclusion is that the weights do not matter at this sample
size.

**Answer 9.** Rewrite: "Minimum-trace reconciliation is a weighted least-squares projection onto the coherent
subspace, and it minimises the sum of reconciled error variances across all nodes when the weight
matrix equals the true covariance of the base forecast errors. Because that covariance must be
estimated, the method is optimal only with respect to the estimate, and it can lose out of sample to
ordinary or structural least squares when the estimate is noisy. The objective weights every node
equally in squared units, which is a choice about whose errors matter rather than a property of the
business."

Three sentences, and each one removes a specific false implication: "optimal" without a conditioning
set, "total error" without distinguishing variance from bias, and "removes judgement" when the
judgement has simply moved into the covariance estimate.

## 14. Interview questions

Eleven questions, each with the content a strong answer contains and what a weak answer reveals. Ask
them in this order; the later ones assume the earlier ones are answered.

### Q1. What is MinT?

**Strong answer:** a family of projections onto the coherent subspace, minimising the trace of the
reconciled error covariance; the members differ only in how `W` is estimated.
**Weak answer:** "the optimal reconciliation method", naming neither the objective nor the variants.

### Q2. Why does the formula need a covariance at all?

**Strong answer:** the projection must decide how far to move each node, and the only principled
basis is how wrong each node has been; `W` supplies that metric, so high-variance nodes move more.
**Weak answer:** "because it is in the formula", or a claim that the covariance measures correlation
between the forecasts rather than between their errors.

### Q3. Your hierarchy has 4,000 series and 36 months of data. What do you do?

**Strong answer:** the sample covariance is singular by construction, with rank at most 35, so it
cannot be inverted; shrink toward a diagonal target, use structural or variance scaling, or split
into sub-hierarchies, and report `n`, `T`, and the condition number so the choice is visible.
**Weak answer:** "use a pseudoinverse", which silently assumes zero variance in the discarded
directions.

### Q4. Explain shrinkage to a planner.

**Strong answer:** the correlations are noisy, so we estimate how noisy and pull them toward zero by
that fraction, leaving the variances alone; informative data change almost nothing, uninformative
data fall back to a diagonal matrix, and the result is always invertible.
**Weak answer:** "it regularises the covariance", which tells the planner nothing about their
numbers.

### Q5. Why can a non-negative forecast reconcile to a negative value?

**Strong answer:** the projection is linear and has no inequality constraint; a small node with a
large weight can be asked to absorb a correction larger than its own value, especially when a
sibling's base exceeds the parent total.
**Weak answer:** "that should not happen", or a claim that the software prevents it.

### Q6. What is the difference between reconciling a mean and reconciling a distribution?

**Strong answer:** the mean needs only `W`; a distribution needs a joint distribution over the nodes,
which the marginals do not determine, so reconcile draws from a stated dependence model rather than
quantiles, which do not commute with a linear map.
**Weak answer:** "you reconcile each quantile separately", which is the error the question tests for.

### Q7. Is MinT optimal?

**Strong answer:** it minimises the expected trace conditional on `W` being the true covariance of
the base errors; since `W` is estimated, and the problem is singular when series outnumber
observations, the optimality applies to an oracle rather than to the procedure as run.
**Weak answer:** "yes", or "no", without a conditioning set; both are uninformative.

### Q8. How would you choose between OLS, WLS, and MinT(Shrink)?

**Strong answer:** backtest all three with the base forecasts held fixed, common origins and horizon,
evaluation at the level where the decision is made, and the spread across variants reported; choose
the simplest that is not materially worse.
**Weak answer:** "MinT is the most advanced", a preference ordered by sophistication rather than
evidence.

### Q9. What does MinT inherit from the base forecasts?

**Strong answer:** everything. A projection cannot add information; a biased total becomes a biased
set of every node, transformed by the same map, so bias must be corrected before reconciliation.
**Weak answer:** "nothing, reconciliation corrects the base forecasts", which confuses coherence with
accuracy.

### Q10. How would you productionise the projection?

**Strong answer:** compute `P` once per hierarchy version, weight vintage, and horizon; cache it under
a version key including a node-order hash that fails on dimension mismatch; run the constraint gate
after applying it; and log the variant, intensity, and window with every run.
**Weak answer:** "recompute it every time", which is safe but wasteful, or "the library handles it",
which is neither an answer nor a check.

### Q11. When would you not reconcile at all?

**Strong answer:** when the number feeds a judgement rather than a transaction, when reconciliation
would distribute a wrong top-level number into every descendant, and when a documented bridge is
cheaper than the distortion; the decision should be recorded, not defaulted.
**Weak answer:** "never", because coherence is always better, a claim the evidence does not support.

## 15. Key terms

- **Base forecast** — the independent forecast at each node before reconciliation; the input to every variant (section 3).
- **Bias propagation** — the redistribution of base forecast bias by the projection; reconciliation transforms bias, not removes it (section 11).
- **Coherence** — every parent equals the sum of its children in every bucket (section 3).
- **Coherent subspace** — the `m`-dimensional set of vectors satisfying every aggregation constraint (section 3).
- **Condition number** — largest over smallest singular value; the amplification of estimation error by the inverse (section 5, step 6).
- **Constrained projection** — projection onto the coherent subspace intersected with a constraint set; non-linear and not cacheable (section 5, step 15).
- **Energy score** — a strictly proper scoring rule for multivariate distributions, recommended for comparing reconciled distributions (section 13).
- **Forecast error** — the difference between a forecast and its outcome, out of sample; not an in-sample residual (section 4).
- **Generalised least squares (GLS)** — the projection with a full weight matrix; algebraically the same as MinT (section 9.1).
- **Horizon-specific covariance** — a distinct `W` per forecast horizon, since error covariance grows and changes shape with lead time (section 5, step 4).
- **Identifiability** — whether a quantity can be recovered from data in principle; the reconciled error covariance cannot (section 9.1).
- **Immutable forecast** — a node fixed by commitment, which reconciliation must treat as a constraint (section 5, step 15).
- **Minimum trace (MinT)** — the family minimising `tr(S P W P' S')` subject to `S P S = S` (section 5).
- **MinT(Sample)** — MinT with the unrestricted sample covariance; singular when `n` exceeds `T` (section 5, step 8).
- **MinT(Shrink)** — MinT with the sample covariance shrunk toward its diagonal (section 5, step 9).
- **Ordinary least squares (OLS)** — the projection with `W = I`; the orthogonal projection, and a strong baseline (section 5, step 8).
- **Positive definite** — all eigenvalues strictly positive, hence invertible; what shrinkage guarantees (section 5, step 9).
- **Precision matrix** — the inverse of a covariance; large entries mean high trust (section 3).
- **Projection cache** — a stored `P` reused until `S`, `W`, or the ordering changes (section 5, step 12).
- **Push-forward distribution** — the distribution induced by applying a map to a random vector; the basis of draw-wise reconciliation (section 5, step 17).
- **Quadratic programme (QP)** — minimisation of a quadratic objective under linear constraints; the correct route for non-negativity when it matters (section 5, step 15).
- **Residual** — an in-sample fitted difference; a proxy for, not a measurement of, forecast error (section 4).
- **Sample covariance** — `(1/T) sum e_t e_t'`; rank at most `T - 1` (section 5, step 6).
- **Schäfer–Strimmer estimator** — an analytic shrinkage intensity that pulls correlations toward zero in proportion to their estimated noise (section 5, step 9).
- **Set-negative-to-zero (sntz)** — a heuristic that replaces negatives with zero and breaks coherence (section 5, step 15).
- **Shrinkage intensity** — the weight `lambda` on the diagonal target, in `[0, 1]` (section 5, step 9).
- **Structural scaling (WLSs)** — `W = diag(S 1)`, the leaf count beneath each node; no estimation (section 5, step 8).
- **Summing matrix `S`** — the `n x m` 0/1 matrix mapping leaves to all nodes (section 5, step 2).
- **Trace** — the sum of diagonal entries; here the total reconciled variance, in squared units (section 4).
- **Variance scaling (WLSv)** — WLS with diagonal entries equal to estimated error variances (section 5, step 8).

## 16. Sources

Grouped by type. Every entry carries a grade: A for peer-reviewed or audited, B for regulatory or
official standard, C for working paper, software documentation, or company case, D for vendor
marketing, E for press or anecdote. Entries cite title and venue; volume and page numbers are given
only where they were checked against the source.

### Peer-reviewed literature (Grade A)

| Source | What it supports | Grade |
|---|---|---|
| Wickramasuriya, Athanasopoulos, Hyndman, "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization", *Journal of the American Statistical Association*, 114(526), 804–819, 2019. DOI: 10.1080/01621459.2018.1448825 | The MinT family, the non-identifiability of the reconciled error covariance, the `W` variants, and the tourism application | A |
| Hyndman, Ahmed, Athanasopoulos, Shang, "Optimal combination forecasts for hierarchical time series", *Computational Statistics and Data Analysis*, 2011 | The projection formulation and the OLS/GLS framing that preceded MinT | A |
| Panagiotelis, Athanasopoulos, Gamakumara, Hyndman, "Forecast reconciliation: A geometric view with new insights on bias correction", *International Journal of Forecasting*, 2021 | The geometric interpretation and the effect of bias on the projection | A |
| Panagiotelis, Gamakumara, Athanasopoulos, Hyndman, "Probabilistic forecast reconciliation: Properties, evaluation and score optimisation", *European Journal of Operational Research*, 306(2), 693–706, 2023. DOI: 10.1016/j.ejor.2022.07.040 | Reconciliation of distributions as a push-forward, draw-wise reconciliation, and the impropriety of the log score for this comparison | A |
| Athanasopoulos, Hyndman, Kourentzes, Panagiotelis, "Forecast reconciliation: A review", *International Journal of Forecasting*, 40(2), 430–456, 2024. DOI: 10.1016/j.ijforecast.2023.10.010 | The conditional finding that no method dominates, and the map of the field | A |
| Schäfer, Strimmer, "A shrinkage approach to large-scale covariance matrix estimation and implications for functional genomics", *Statistical Applications in Genetics and Molecular Biology*, 4(1), Article 32, 2005. DOI: 10.2202/1544-6115.1175 | The analytic shrinkage intensity and the diagonal target | A |
| Wickramasuriya, Turlach, Hyndman, "Optimal non-negative forecast reconciliation", *Statistics and Computing*, 30, 1167–1182, 2020 | Constrained reconciliation with non-negativity | A |
| Hyndman, Lee, Wang, "Fast computation of reconciled forecasts for hierarchical and grouped time series", *Computational Statistics and Data Analysis*, 97, 16–32, 2016 | Recursive, Kronecker, and sparse methods for large hierarchies | A |
| Girolimetto, Athanasopoulos, Di Fonzo, Hyndman, "Cross-temporal probabilistic forecast reconciliation: Methodological and practical issues", *International Journal of Forecasting*, 40, 1134–1151, 2024. DOI: 10.1016/j.ijforecast.2023.10.003 | Reconciliation of sample paths and the treatment of negatives in probabilistic reconciliation | A |
| Makridakis, Spiliotis, Assimakopoulos, M5 competition results, *International Journal of Forecasting*, 2022 | The twelve-level hierarchy and the equal weighting of levels in a large public evaluation | A |

### Standards and official bodies (Grade B)

| Source | What it supports | Grade |
|---|---|---|
| GS1, *General Specifications* (current release) | GTIN, GLN, SSCC identifiers and the packaging hierarchy that defines `S` | B |
| ISO 8601-1:2019, *Date and time — Representations for information interchange* | Week and date representation, and the non-nesting of weeks into months | B |
| IFRS 8, *Operating Segments*, International Accounting Standards Board | The disclosure requirement to reconcile segment totals to the financial statements | B |
| Federal Reserve Board, *Guidance on Model Risk Management* (SR 11-7), 2011 | Validation, documentation, and monitoring expectations for models used in supervised processes | B |

### Working papers, software, and industry material (Grade C and D)

| Source | What it supports | Grade |
|---|---|---|
| Pritularga, Svetunkov, Kourentzes, "Stochastic coherency in forecast reconciliation", working paper, 2021 | The argument that full-information shrinkage is high-variance, and the block-structured alternatives | C |
| Steinmeister, Pauly, "Iterative trace minimization for the reconciliation of very short hierarchical time series", arXiv:2409.18550, 2024 | Behaviour of MinT variants when `T` is small, and a semiconductor data case study | C |
| "Billions-scale forecast reconciliation", arXiv:2602.05030, 2026 | Sparse and iterative computation at very large scale; reported as a preprint, not independently verified | C |
| Girolimetto, "Non-negative forecast reconciliation: optimal methods and operational solutions", *Forecasting*, 7(4), 64, 2025 | Comparative evaluation of clipping heuristics and constrained solvers | A |
| R packages `hts`, `FoReco`, `fable`, `ProbReco`, `bayesRecon`; Python packages `hierarchicalforecast`, `pyhts`, `Darts` | Implementation availability and default conventions; evidence of what is offered, not of what performs | C |
| Enterprise planning vendor documentation on hierarchical forecasting | The claim that reconciliation is available and that one method is a default | D |

No Grade E material is relied on. Named company cases of reconciliation failure are absent from the
public record, and section 9.5 labels the corresponding claims as inference rather than evidence.
The MinT paper's ranking is a result about the designs it studied; the 2024 review's conditional
conclusion is a better guide to a new hierarchy than any single study's ordering.

## 17. Links

Within this module:

- Module map and topic status: [`_index.md`](_index.md)
- Previous topic, aggregation consistency and the four classical methods: [`02_aggregation-consistency.md`](02_aggregation-consistency.md)
- Foundations of the hierarchy itself: [`01_hierarchy-structures.md`](01_hierarchy-structures.md)
- Next topic, temporal and cross-temporal hierarchies: `04_temporal-cross-temporal.md` (planned)

Within the part:

- Part index: [`../_index.md`](../_index.md)
- Prerequisite module on hierarchy and aggregation: [`../07_time_series_forecasting/05_hierarchy-aggregation.md`](../07_time_series_forecasting/05_hierarchy-aggregation.md)
- Prerequisite module map: [`../07_time_series_forecasting/_index.md`](../07_time_series_forecasting/_index.md)

Elsewhere in the course:

- Probability, sampling, and the bootstrap: [`../../01_foundations/04_statistics_causal_inference/02_probability-bootstrap.md`](../../01_foundations/04_statistics_causal_inference/02_probability-bootstrap.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)

> **Note:** Topic 04 extends the projection across time buckets as well as entities, which multiplies
> the covariance size by the number of temporal aggregation levels and makes shrinkage more necessary,
> not less. Read it next if you reconcile weekly but report monthly.
