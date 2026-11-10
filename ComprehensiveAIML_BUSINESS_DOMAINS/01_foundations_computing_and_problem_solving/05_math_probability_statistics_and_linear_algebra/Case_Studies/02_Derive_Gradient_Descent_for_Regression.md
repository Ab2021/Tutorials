---
case_id: "CASE-05-002"
case_title: "Derive Gradient Descent for Regression"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Foundations and computing"
primary_pattern: "optimization-contract"
secondary_patterns: ["numerical-conditioning", "correctness-contract"]
evidence_level: "illustrative"
source_basis: "constructed 240-row teaching design; every derivation recomputed by direct linear algebra"
as_of_date: "2026-11-10"
owner_role: "modeling lead"
source_index: "05_math_probability_statistics_and_linear_algebra/Case_Studies/INDEX.md"
review_trigger: "Any change to the design matrix, the loss function, the solver, or the stopping rule"
---

# Derive Gradient Descent for Regression

**Evidence declaration:** This is an ILLUSTRATIVE teaching case. The 240-row rent design, the 200-row collinearity design and the five-point worked example are constructed inputs; no company, market, or vendor supplied them, and no production run is reported. What *is* verified is the algebra: every number printed below was recomputed from the stated inputs with double-precision linear algebra, and each derivation carries its own independent check. The optimizer comparison is therefore a verified statement about arithmetic, not evidence about any real pricing system. **Implementation stage:** proposed. **Editorial status:** draft; independent review is OPEN.

The case does one thing. It derives the gradient-descent update for ordinary least squares from the objective, proves the step-size bound from the Hessian, measures how badly the bound degrades on an unscaled design, and turns that measurement into a decision rule about when to use descent at all. The recurring failure being taught is not "the code crashed". It is that a run reported a falling loss, printed a coefficient vector, and was still 912 units away from the optimum after a million iterations - a number that looks like a converged model in a slide deck.

## 1. The One-Line Framing

Given a regression design and a loss, decide **whether to train by gradient descent, and with which step size and stopping rule**, from the measured spectrum of the Hessian rather than from a default parameter, so that the modeling lead can approve a reproducible training contract instead of accepting an unexplained library call.

The decision has three possible answers, and the case insists on picking one: closed form when `n*p^2` is affordable, descent in a preconditioned basis when the loss is not quadratic or the data do not fit in memory, and refuse when neither the spectrum nor the per-sample curvature has been measured.

## 2. Context & Background

A pricing team maintains a rent model that is retrained nightly from a lease ledger. The ledger records, for each signed lease, the floor area in square feet, the building age in years at signing, and the achieved monthly rent in US dollars. The model is a linear regression with an intercept and two slopes, so `p = 3` parameters and the design matrix has three columns.

The nightly job calls a library fitter with default settings. That call is the baseline of this case and it is deliberately kept as the comparator: it is one line of code, it is deterministic, and nobody on the team can say what step size it used, what tolerance it stopped at, or how far from the optimum it landed. The baseline is not wrong. It is unaudited, and the audit is what the case constructs.

Two facts about the design dominate everything that follows, and both are measured rather than assumed. First, the intercept column is a constant of magnitude one while the area column has values near 1200, so the columns differ in scale by three orders of magnitude. Second, the two real features are nearly uncorrelated with each other (sample correlation -0.00947), which means the bad conditioning does *not* come from feature collinearity. It comes from the intercept sitting in the same unscaled basis as a large-magnitude feature.

The team's retraining cadence is nightly, the decision window is one day, and the owner of the model is the modeling lead. Reviewer capacity is not the binding constraint; the binding constraint is that nobody can currently state a convergence criterion in units the business understands.

The workload scale is small - 240 rows, 3 columns - and that smallness is itself part of the lesson. At this size the closed-form solve costs 4,320 floating-point operations, so any argument for descent has to be about something other than arithmetic volume.

**Takeaway:** the context fixes the comparator (an unaudited library call), the size regime (tiny, in-memory), and the missing artefact (a stated convergence criterion). Without those three, "which optimizer" is a matter of taste.

## 3. The Business Question

| Field | Contract |
|---|---|
| Question | Should the nightly rent model train by gradient descent, and with which step size and stopping rule? |
| Owner | modeling lead, named in the model registry, with authority to approve or reject the training contract |
| Trigger | A retrain job must be signed off before it can write coefficients into the pricing service |
| Options | closed-form solve; descent in a standardized basis; descent in the raw basis; retain the unaudited library default |
| Objective | Minimise the probability that a shipped coefficient vector is further from the optimum than the stated tolerance |
| Hard constraint | No step size ships unless `L = lambda_max(Hessian)` has been measured on the exact design used for training |
| Soft constraint | Nightly retrain must finish inside the batch window, which is generous at this size and is not the binding limit |
| Scope | One design, one loss, one fitted model, one decision cycle; no claim extends past that |
| Out of scope | Feature selection, label definition, price elasticity, and the business value of the model itself |

The proxy metric is dangerous here in a specific way. "Loss went down every iteration" is the metric a training log naturally reports, and it is satisfied by a run that has moved 4 percent of the way to the optimum. The business objective is a coefficient vector close enough to the optimum that the pricing service's output does not depend on which solver produced it. Those two are not the same, and §6 quantifies the gap.

A second framing question matters for the owner: is the loss even the right thing to converge? For a squared-error rent model, the fitted coefficients feed a price, and a 4-percent error in the age slope is a 1.6 USD/month difference at age 40. The tolerance decision is therefore expressible in business units, not only in gradient norms, and §9 states it that way.

## 4. Assumptions & Constraints

- **A-01 - Objective is mean squared error.** The training objective is `J(w) = (1/2n) * ||Xw - y||^2`. Verification: read the loss definition from the training configuration, not from the library default. If violated: every bound below is void, because the Hessian is no longer `X'X/n`.
- **A-02 - Design is fixed during the run.** `X` and `y` do not change between iterations. Verification: hash the training table and compare it before and after the job. If violated: the constant-Hessian analysis fails and only stochastic bounds apply.
- **A-03 - Features are available at decision time.** Area and age are recorded at lease signing, before the rent is agreed. Verification: compare each column's event timestamp to the label timestamp. If violated: the fit is contaminated and the optimizer question is moot.
- **A-04 - Intercept is present and unscaled.** The design includes a column of ones. Verification: check the column means; the first column must be exactly 1.0 everywhere. If violated: the centring shortcuts in §6 do not apply as written.
- **A-05 - Arithmetic is double precision.** The reference solve uses float64. Verification: print `X.dtype` in the run manifest. If violated: the conditioning budget in D-13 must be recomputed against the smaller machine epsilon.
- **A-06 - The comparator is the closed-form solve.** "Better" means closer to the exact normal-equation solution, not closer to a previous run of the same approximate method. Verification: the closed-form vector is stored as a fixed artefact. If violated: two approximations are being compared and neither is a reference.
- **A-07 - Labels are final at signing.** No rent is renegotiated inside the fitting window. Verification: query for amendments dated after the lease row. If violated: the target is a moving quantity and residuals are not iid.
- **A-08 - Duplicate leases are removed.** One row per executed lease. Verification: count distinct lease identifiers against row count. If violated: the effective sample size is smaller than `n` and the standard errors in D-14 are too small.
- **A-09 - Out-of-range inputs are excluded, not clipped.** Area is restricted to 420-2400 sq ft and age to 0-40 years. Verification: re-run the range filter and count exclusions. If violated: a single 40,000 sq ft row changes `lambda_max` materially.
- **A-10 - The five-point example is hand-checkable.** The small dataset in D-02 is chosen so every intermediate is an exact rational. Verification: recompute by hand. If violated: the worked example stops being an oracle.
- **A-11 - Convergence is declared on a stated criterion.** The run reports the relative gradient norm and the loss gap to the closed form, not the iteration count. Verification: grep the run manifest for both fields. If violated: the run is unauditable and the baseline wins by default.
- **A-12 - Randomness is seeded.** Any stochastic run records its seed and its row order. Verification: replay with the recorded seed and compare the first ten iterates. If violated: the reported noise floor is not reproducible.
- **A-13 - Nightly budget is not binding at this size.** The batch window is hours, not seconds. Verification: compare measured wall time to the window. If violated: the flop comparison in §9 becomes a hard constraint rather than a note.
- **A-14 - No claim about production.** Nothing in this case is a measured outcome of a deployed system. Verification: the evidence level in the front matter reads `illustrative`. If violated: the manuscript is mislabelled and should be rejected.

**Takeaway:** the assumptions are not decoration. A-01, A-02 and A-05 each switch the analysis to a different mathematical regime, and A-14 is the line that keeps the case honest.

## 5. Data, Tools & Method

### 5a Data Specification

The unit of analysis is one executed lease, identified by `lease_id`. The grain is one row per lease, one column per decision-time attribute, and one label column.

| Element | Specification |
|---|---|
| Rows | 240 executed leases (ILLUSTRATIVE, constructed) |
| Keys | `lease_id`, unique and non-null; no natural key is reused across buildings |
| Feature 1 | floor area in square feet, integer, restricted to 420-2400 |
| Feature 2 | building age in years at signing, integer, restricted to 0-40 |
| Label | achieved monthly rent in US dollars, continuous |
| Cut-off | the lease signing timestamp; every feature must be recorded at or before it |
| Not available at decision time | renewal outcome, arrears history after signing, and any post-signing amendment |
| Exclusions | area outside range, age outside range, missing label, duplicate `lease_id` |
| Population statement | leases signed in one portfolio during one year; no external validity claimed |

Column statistics of the constructed design, used throughout §6: area has mean 1202.14 sq ft and standard deviation 262.11; age has mean 19.75 years and standard deviation 12.03; rent has mean 1120.86 USD and its total sum of squares about the mean is 69,063,743.

The label was generated as `900 + 0.85*area - 40*age + noise` with noise standard deviation 90 USD, so the "true" coefficients are known and are *not* the fitted ones. That distinction matters: the fitted vector is `(948.92, 0.804708, -40.2666)`, and a case that reported the true generator as the answer would be teaching the wrong lesson about what a solver produces.

### 5b Tools & Stack

The reference implementation is a small numeric stack: one dense matrix library, one power-iteration routine for the top eigenvalue, one pivoted QR solve for the reference coefficients, and one scripted gradient-descent loop that logs the loss, the gradient norm, and the distance to the reference vector at fixed iteration marks.

| Component | Role | Why this one |
|---|---|---|
| Dense linear algebra (LAPACK-class) | spectrum, QR reference solve | gives `lambda_min`/`lambda_max` directly instead of inferring them from convergence behaviour |
| Power iteration | independent estimate of `L` | only needs matrix-vector products, so it is the same routine that would be used when `X` is streamed |
| Fixed-step descent loop | the thing being audited | explicit `eta` and explicit stop rule, with no library heuristic in the path |
| Mini-batch loop with seed | per-sample curvature evidence | needed for D-09, where full-batch stability is not sufficient |
| Ledger writer | records design hash, `L`, `eta`, stop reason | makes the run reproducible after the fact |

### 5c Method Summary

The method is a five-step chain, and each step produces an artefact the next step consumes.

1. Standardize the two real features using statistics computed on the training rows only, and keep the intercept column unscaled; record the centring means and scales in the model artefact.
2. Compute the Gram matrix `G = X'X/n` on the exact training design and take its eigenvalues; report `lambda_min`, `lambda_max` and `kappa = lambda_max/lambda_min`.
3. Set the step size from the measured spectrum (`eta = 2/(lambda_max + lambda_min)`), not from a default, and compute the implied contraction factor and iteration budget.
4. If descent is retained, run it with a stopping rule expressed as a relative gradient norm, and log the loss gap against the closed-form solution at every decade of iterations.
5. Fit the closed-form reference on the same design, compare coefficients, and reject the descent run if the relative difference exceeds the tolerance or the loss gap exceeds the stated band.

The baseline, retained throughout, is the unaudited library call with default settings. The alternatives are closed form only, and descent in the raw (unstandardized) basis, which is included precisely so that the case can show what it costs rather than assert that it is bad.

### 5d Provenance, Quality, Exclusions, Leakage

The constructed design is generated from a recorded seed, so the exact numbers in §6 are reproducible by anyone who repeats the stated generator. That is the only provenance claim the case makes.

Quality checks applied before fitting, each with a rejection path rather than a warning:

- Any row whose area or age falls outside the stated range is dropped, not clipped, because clipping would silently move the extreme-leverage row that drives `lambda_max`.
- Any row with a missing label is dropped; imputing the target would shrink the residual variance and make every standard error in D-14 too small.
- Any duplicate `lease_id` is dropped after keeping the earliest record, because the ledger can contain a re-issued contract.
- Any column whose event timestamp is after the label timestamp is excluded from the design, which is the leakage control that matters most for a signed-lease dataset.

Leakage is checked by a specific test rather than by inspection: for each feature, the maximum event timestamp within the training window is compared to the maximum label timestamp. A feature whose maximum event time exceeds the label's is, by construction, not available when the label is set. On this constructed design the test passes trivially, and the case states that explicitly instead of claiming a real audit.

Exclusions are counted and reported: the number of dropped rows travels with the model artefact, because a run that quietly drops 30 percent of the ledger is a different model even if every diagnostic looks the same.

### 5e Reproducibility & Method-Selection Ledger

| Artefact | Purpose | Accessibility | Limitation |
|---|---|---|---|
| DES-240 | the 240-row constructed design and label | reproducible from a stated seed | synthetic; no market content |
| DES-005 | the five-point hand-checkable design | printed in D-02 in full | too small for any inferential claim |
| SPECTRUM | eigenvalues of `X'X/n` on the exact training design | printed in D-05 and D-06 | valid only for this design hash |
| L-POWER | top eigenvalue from 30 power iterations | reproducible in 43,200 flops | needs a spectral gap or many iterations |
| REF-QR | closed-form coefficient vector via pivoted QR | printed in D-02 and D-13 | reference only for the stated loss |
| TRACE-ETA | loss and distance-to-optimum at fixed iteration marks | printed in D-05 | deterministic, full batch, float64 |
| SGD-GRID | measured excess loss over `(eta, batch)` | printed in D-09 | one seed; not a variance estimate |
| CV-RIDGE | 40-split cross-validated RMSE over `lambda` | printed in D-12 | splits are random, not time-ordered |

## 6. Analysis Walkthrough

Every derivation below uses the same anatomy: formula, then where each symbol lives, then the constraint under which the formula is valid, then a numerical example with intermediates, then the interpretation that changes a decision. Arithmetic in this section was computed directly and is quoted to the digits shown; the inputs are ILLUSTRATIVE and the algebra is exact.

### D-01 From the objective to the update, and why the Hessian is the whole story

```text
FORMULA [D-01]:  J(w) = (1/2n) ||Xw - y||^2
                 grad J(w) = (1/n) X'(Xw - y) = G w - c,  G = X'X/n,  c = X'y/n
                 Hess J(w) = G   (constant in w)
WHERE:
  X  : n x p design matrix, real, n = 240 rows, p = 3 columns
  y  : n x 1 label vector, monthly rent in USD
  w  : p x 1 parameter vector, units mixed (USD, USD/sq ft, USD/year)
  G  : p x p Gram matrix scaled by 1/n, symmetric positive semidefinite
  c  : p x 1 scaled cross-moment vector
  J  : scalar, units USD^2 (mean squared error halved)
CONSTRAINT: squared-error loss with a fixed design. For any other loss the
            Hessian depends on w and none of the step-size algebra carries over.
NUMERICAL EXAMPLE (constructive check at w0 = (0.5, 0.3, -2.0)):
  analytic gradient     = (-799.225116, -996597.8147, -10236.32727)
  central difference, h = 1e-6, per coordinate:
                        = (-799.225149, -996597.8147, -10236.32727)
  relative difference   = (4.1e-08, 1.3e-11, 3.4e-09)
INTERPRETATION: the update w <- w - eta*(G w - c) is exact to the
                gradient-check tolerance, so any failure downstream is a
                step-size or conditioning failure, not a calculus error.
```

The gradient check is worth doing even when the derivation is textbook, because the two most common real defects - a transposed design matrix and a missing `1/n` - both produce a gradient that is wrong by a smooth scale factor. A finite-difference check catches the transpose immediately (the coordinate pattern differs) and catches the missing `1/n` only if the check compares magnitudes rather than directions. The check above compares magnitudes per coordinate, which is why it would have caught both.

Because the Hessian is constant, the entire convergence behaviour is determined by the spectrum of `G`. There is no local minimum to be trapped in, no saddle point to escape, and no reason for a loss curve to be non-monotone once `eta` is below the stability limit. That is a strong statement and it is what makes least squares the right place to learn step-size discipline: any wobble in the loss curve is attributable to `eta` or to minibatching, never to geometry.

### D-02 The closed-form reference, computed exactly on five points

```text
FORMULA [D-02]:  w* = G^{-1} c  = (X'X)^{-1} X'y
WHERE:
  G^{-1} exists iff X has full column rank p; here p = 3
  w* : the unique minimiser of J, in the same mixed units as w
CONSTRAINT: full column rank; with an intercept, rank deficiency is exactly
            the case where a feature column is an affine combination of others.
NUMERICAL EXAMPLE (five points, all intermediates exact):
  x = (1, 2, 3, 4, 5)   y = (2, 4, 5, 4, 5)
  sum x = 15, sum y = 20, sum xy = 66, sum x^2 = 55, n = 5
  xbar = 3, ybar = 4
  Sxx = sum (x - xbar)^2 = 55 - 15^2/5 = 10
  Sxy = sum (x - xbar)(y - ybar) = 66 - 15*20/5 = 6
  Syy = 25 - 400/5 = 6
  slope  w1 = Sxy/Sxx = 6/10 = 0.6
  intercept w0 = ybar - w1*xbar = 4 - 0.6*3 = 2.2
  fitted  = (2.8, 3.4, 4.0, 4.6, 5.2)
  residual= (-0.8, 0.6, 1.0, -0.6, -0.2),  and the residuals sum to 0.0
  SSE = 0.64 + 0.36 + 1.00 + 0.36 + 0.04 = 2.4
  R^2 = 1 - 2.4/6 = 0.6,  MSE = 0.48,  RMSE = 0.6928
INTERPRETATION: five points and two parameters give an exactly checkable
                oracle. Every descent run on this design must approach
                (2.2, 0.6) and no other vector.
```

The five-point design is kept in the case because it is the only artefact a reviewer can verify with paper and pencil. Its Gram matrix, in the basis `[1, x]`, is `[[1, 3], [3, 11]]` and its scaled cross-moment is `(4, 13.2)`. Solving the 2x2 system by hand gives `w = (2.2, 0.6)`, matching the centred formulas, which is the independent arithmetic re-check the contract asks for: two different routes to the same vector.

For contrast, the exact solution on the 240-row design is `w* = (948.9198 USD, 0.804708 USD per sq ft, -40.26664 USD per year)`. Its in-sample fit is `SSE = 1,620,053`, `MSE = 6,750.22`, `RMSE = 82.16 USD`, `R^2 = 0.97654`. The residual vector satisfies `X'r = 0` to a maximum relative error of 2.7e-15, which is the normal-equation identity that any candidate solution must obey; checking it costs one matrix-vector product and is the cheapest possible audit of a returned coefficient vector.

### D-03 The step-size bound, and the contraction factor that predicts the budget

```text
FORMULA [D-03]:  error recursion per eigendirection i:
                   e_{k+1,i} = (1 - eta*lambda_i) e_{k,i}
                 stability requires |1 - eta*lambda_i| < 1 for all i, hence
                   0 < eta < 2/lambda_max = 2/L
                 optimal fixed step for the worst direction:
                   eta* = 2/(L + mu),   rho = (L - mu)/(L + mu)
                 iterations to shrink the error by a factor eps:
                   k(eps) = ln(eps) / ln(rho)
WHERE:
  lambda_i : eigenvalues of G, sorted; L = lambda_max, mu = lambda_min > 0
  eta      : step size, units inverse to G, i.e. (flops-free) dimensionless
             in the scaled basis because G carries the 1/n
  rho      : per-iteration worst-case contraction factor, 0 < rho < 1
  k(eps)   : iteration count, dimensionless
CONSTRAINT: quadratic objective, exact arithmetic, full batch. With minibatch
            noise the recursion holds in expectation only, and the noise floor
            in D-10 becomes the binding limit long before k(eps) is reached.
NUMERICAL EXAMPLE (five-point design, basis [1, x]):
  G = [[1, 3], [3, 11]]  ->  eigenvalues 11.83095189 and 0.16904811
  kappa = 11.83095189 / 0.16904811 = 69.9857
  eta*  = 2 / (11.83095189 + 0.16904811) = 2/12 = 0.1666667
  rho   = 10.83095189 / 12 = 0.9718253
  k(1e-6) = ln(1e-6)/ln(0.9718253) = 483.4 iterations
  predicted error at step k = ||w*|| * rho^k, with ||w*|| = 2.28035
INTERPRETATION: the budget is a closed-form function of the spectrum. No
                experiment is needed to know that 483 iterations will do and
                that 40 will not.
```

The prediction is testable, and the test is the point. Running the loop from `w = (0, 0)` at `eta = 1/6` produces distances to the optimum of 2.2161 at step 1, 1.7135 at step 10, 0.13086 at step 100, 0.546276 at step 50 and 2.3073e-06 at step 483. The closed form `2.28035 * 0.9718253^k` gives 2.2162, 1.7135, 0.130865, 0.546276 and 2.30731e-06 for the same five marks. Agreement to five significant figures means the whole trajectory is understood, not merely its endpoint.

The practical value is that the budget can be computed *before* the run from two numbers, `L` and `mu`, both of which come from one eigen-decomposition of a `p x p` matrix. For `p = 3` that decomposition is essentially free, and it converts "how many epochs should we allow?" from an experiment into a calculation.

### D-04 Where the bound breaks: the marginal step and the sign flip

```text
FORMULA [D-04]:  at eta = 2/L exactly, the multiplier on the top
                 eigendirection is 1 - (2/L)*L = -1, so
                   e_{k+1} = -e_k  (magnitude preserved, sign alternating)
                 for eta = 2/L + delta, the multiplier is -1 - delta*L, whose
                 magnitude exceeds 1 for every delta > 0.
WHERE:
  multiplier_i = 1 - eta*lambda_i, dimensionless, must lie in (-1, 1)
  delta : excess step size, same units as eta
CONSTRAINT: exact arithmetic and a quadratic. In floating point the marginal
            case is worse than the analysis suggests, because rounding injects
            energy into the direction that never decays.
NUMERICAL EXAMPLE A (two-parameter quadratic with G = diag(1, 4)):
  L = 4, 2/L = 0.5
  eta = 0.5     : multipliers (0.5, -1.0). Fast component alternates
                  between 1.0 and 0.0 while the optimum is 0.5; after 200
                  steps w = (1.0, 0.0) - converged in one coordinate, stuck
                  in a 2-cycle in the other.
  eta = 0.5001  : multipliers (0.4999, -1.0004). After 200 steps the second
                  coordinate is -6.2e-05 and growing geometrically.
  eta = 0.51    : second coordinate reaches -1.27e+03 after 200 steps.
  eta = 0.60    : second coordinate reaches -8.4e+28 after 200 steps.
NUMERICAL EXAMPLE B (five-point design): 2/L = 0.16904811.
  eta = 0.16900 : max multiplier 0.99943, still contracting
  eta = 0.16905 : max multiplier 1.0000224, expanding slowly; after 400
                  steps w = (1.886, -0.533) against the optimum (2.2, 0.6)
  eta = 0.17000 : after 400 steps w = (-25.238, -98.460)
INTERPRETATION: the boundary is not a soft shoulder. Crossing it by 0.003
                percent produces growth, and the growth rate is exactly
                delta*L - 1 per iteration, so one can compute the failure
                time instead of waiting for a NaN.
```

This is the counterexample the section is required to carry, and it is stronger than the usual "too large a step diverges" statement because it names the mechanism. At the boundary the fast direction does not diverge, it *oscillates*, so a loss curve can look stable for a long time while one coordinate alternates around its optimum. A run that logs only the loss may therefore look healthy at `eta = 2/L` even though that coordinate will never converge, because the loss contribution from that coordinate is constant rather than growing.

The diagnosis is cheap: after each iteration, project the parameter change onto the top eigenvector and check whether the sign alternates. An alternating sign with constant magnitude is the signature of `eta` at or above `2/L`, and it is distinguishable from a genuinely diverging run, where the magnitude grows. Both are distinguishable from the healthy case, where the projected change keeps the same sign and shrinks by roughly a constant factor.

Note also the asymmetry in Example A: the slow coordinate converged in a single step while the fast one never did. That is the opposite of the intuition most people carry into debugging ("the slow direction is the problem"), and it happens because `eta` near `2/L` is aggressive for fast directions and correct for slow ones. The intuition only holds for `eta` at the *stable* optimum `2/(L+mu)`, which is where D-03 lives.

### D-05 The real design: measuring the spectrum and pricing the run

```text
FORMULA [D-05]:  G = X'X/n,  spectrum of G decides everything:
                   L = lambda_max(G),  mu = lambda_min(G),  kappa = L/mu
                   eta* = 2/(L+mu),  rho = (kappa-1)/(kappa+1)
                   k(eps) ~ (kappa/2) * ln(1/eps)
WHERE:
  X : 240 x 3 design, columns [1, area, age], unscaled
  G : 3 x 3, units mixed because column 1 is dimensionless and column 2
      is sq ft, so the eigenvalues are not comparable across designs
  kappa : condition number of G, dimensionless but scale-dependent
CONSTRAINT: kappa is meaningful only for a stated scaling of the columns.
            Recomputing kappa after rescaling is not a bug fix; it is a
            different matrix and therefore a different analysis.
NUMERICAL EXAMPLE (constructed 240-row design):
  eigenvalues of X'X/n = (0.04019518, 163.29572417, 1514216.25991398)
  L = 1,514,216.26      mu = 0.04019518
  kappa = 37,671,583.3  sqrt(kappa) = 6,137.72
  eta*  = 1.32081526e-06     2/L = 1.32081530e-06
  rho   = 0.9999999469
  k(1e-6) = 260,226,079 iterations
  flops per iteration = 2*n*p = 1,440
  total = 260,226,079 * 1,440 = 3.75e+11 flops (375 GFLOP)
INTERPRETATION: at this conditioning, an honest fixed-step descent needs
                about 2.6e8 passes over 240 rows. The arithmetic is
                affordable; the claim "it converged" is what is at risk.
```

The measured trace at a step size a reviewer might plausibly pick by hand, `eta = 1e-6`, is the substance of the finding. Starting from `w = 0` where `J(0) = 772,045.56` and the optimum is `J* = 3,375.11`:

| Iterations | Loss `J` | In-sample RMSE (USD) | Distance to optimum | Largest coefficient |
|---|---|---|---|---|
| 1 | 293,220.34 | 765.79 | 949.7735 | 1.404 |
| 10 | 120,818.13 | 491.57 | 949.7711 | 0.927 |
| 100 | 117,939.34 | 485.67 | 949.7492 | 0.936 |
| 1,000 | 93,370.68 | 432.14 | 949.5564 | 1.010 |
| 10,000 | 25,267.75 | 224.80 | 948.7739 | 1.367 |
| 100,000 | 21,335.05 | 206.57 | 945.3234 | 1.472 |
| 1,000,000 | 20,081.52 | 200.41 | 911.7369 | 1.448 |

After a million iterations the loss is 5.95 times the optimum, the excess over the optimum is 4.95 times the optimum's own value, and the parameter vector is still 911.74 units away from a vector whose norm is 949.77. In business terms the implied rent error is 200.41 USD against a converged 82.16 USD.

The reason is visible in the numbers rather than in the curve. The loss falls by 65 percent in the first ten iterations and then barely moves, because the components along the two larger eigenvalues are killed almost immediately while the component along `lambda_min = 0.040195` decays at the rate `1 - 1e-6*0.040195 = 0.9999999598` per step. Its half-life is `ln(2)/4.0195e-08 = 1.72e7` steps and its e-fold time is 2.49e7 steps.

An independent consistency check closes the loop. Over 1e6 steps the slow component should retain `exp(-1e6 * 4.0195e-08) = exp(-0.040195) = 0.9606` of its initial size, and the measured distance ratio is `911.7369/949.7735 = 0.9599`. The small residual difference is the contribution of the fast components, which are already at zero. The prediction therefore explains the trace, and a reviewer can reproduce it from two eigenvalues.

### D-06 The conditioning ladder: raw, centred, standardized

```text
FORMULA [D-06]:  for a design with an intercept, the smallest eigenvalue
                 satisfies mu = 1 - R^2, where R^2 is the coefficient of
                 determination from regressing the constant column on the
                 other columns (no intercept in that auxiliary regression).
                 Therefore kappa -> infinity exactly when the feature columns
                 can reproduce the intercept.
WHERE:
  R^2 : fraction of the unit column's sum of squares explained by the
        features, dimensionless, in [0, 1)
  mu  : smallest eigenvalue of X'X/n, dimensionless
CONSTRAINT: requires the intercept column to be present and unscaled, and
            requires the features to be non-degenerate. With kappa infinite,
            the closed form does not exist and descent cannot converge.
NUMERICAL EXAMPLE (240-row design, three scalings):
  raw:          eig = (0.04019518, 163.29572, 1514216.25991)
                kappa = 3.7672e+07
  centred only: eig = (1.0, 144.67239, 68699.10961)
                kappa = 6.8699e+04
  standardized: eig = (0.99052590, 1.0, 1.00947410)
                kappa = 1.01913
  auxiliary check of the identity: regressing 1 on (area, age) gives
    R^2 = 0.9598035  ->  1 - R^2 = 0.0401965
    lambda_min(G)    = 0.0401952   (agreement to 3e-06 absolute)
INTERPRETATION: the bad conditioning is the intercept's, not the features'.
                Centring removes 99.8% of the condition number; scaling
                removes almost all of the rest.
```

This is the single most useful diagnostic in the case, because it tells the engineer *which* transformation to apply. Centring alone takes `kappa` from 3.77e7 to 6.87e4, a factor of 548. Standardization finishes the job. A team that only rescales without centring, or only centres without rescaling, gets an intermediate improvement and usually concludes that "rescaling did not help much" - when in fact it helped enormously and incompletely.

The identity also predicts a failure mode directly. If a feature can reproduce the constant - for example a `building_age` column that is 0 for every row in the fitting window because the portfolio contains only new buildings - then `R^2 = 1`, `mu = 0`, and no step size converges. The formula does not merely warn about this; it gives the diagnostic: compute `1 - R^2` of the features against a constant column, compare to `lambda_min`, and refuse to train when the difference falls below the arithmetic-noise scale.

The auxiliary regression has a second reading that is worth carrying into an interview. `R^2 = 0.9598` means the two features, taken together, can predict a constant almost perfectly *in the least-squares sense* - not because they contain a constant, but because their magnitude range is so large relative to their variation that a tiny weighted combination nearly cancels. The fitted combination is `(0.000706, 0.005634)` on standardized features, with squared norm 3.22e-05. A small coefficient vector with a large effect is exactly the signature of a near-null direction, and it is why the slow eigendirection has almost no weight on the intercept: the eigenvector is `(0.99998, -0.00071, -0.00564)`.

### D-07 Standardized descent: what the same loop does in the right basis

```text
FORMULA [D-07]:  z = (x - m)/s applied per feature, s > 0 measured on the
                 training rows only; the intercept column stays at 1.
                 Then G_z = Z'Z/n has diagonal entries equal to 1 in the
                 feature block, and kappa(G_z) = (1 + |r|)/(1 - |r|) for
                 two features with sample correlation r.
WHERE:
  m : p-vector of training means, feature block only
  s : p-vector of training standard deviations, feature block only
  r : sample Pearson correlation between the two features
CONSTRAINT: m and s must come from the training rows. Using full-data
            statistics leaks the evaluation window into the design scaling
            and makes the validation scores optimistic.
NUMERICAL EXAMPLE (240-row design, both features standardized):
  m = (1202.1417, 19.7542),  s = (262.1051, 12.0285)
  r = -0.0094741  ->  (1 + |r|)/(1 - |r|) = 1.0094741/0.9905259 = 1.01913
  eig(G_z) = (0.99052590, 1.0, 1.00947410),  kappa = 1.01913
  eta* = 2/(1.0094741 + 0.9905259) = 1.0,  rho = 0.0094741
  w*_z = (1120.8593, 210.9182, -484.3482)
  back-transform: 210.9182/262.1051 = 0.804708 (USD per sq ft)
                 -484.3482/12.0285 = -40.26664 (USD per year)
                 intercept = 948.9198 (USD)
INTERPRETATION: with eta = 1.0 the same objective converges in four
                iterations instead of 2.6e8, and the back-transformed
                coefficients agree with the closed form to six decimals.
```

The trace is short enough to print in full, which is why it belongs in the case rather than in an appendix. From `w = 0`, at `eta = 1.0`:

| Iterations | Loss `J` | Loss gap `J - J*` | Distance to optimum |
|---|---|---|---|
| 0 | 769,650.83 | 766,275.72 | 1223.32 |
| 1 | 3,387.7222 | 12.6117 | 5.005 |
| 2 | 3,375.1116 | 0.001132 | 0.047418 |
| 3 | 3,375.1104536 | 5.6e-08 | 4.49e-04 |
| 4 | 3,375.1104535 | 4.5e-11 | 4.26e-06 |
| 5 | 3,375.1104535 | 0.0 | 4.03e-08 |
| 8 | 3,375.1104535 | 0.0 | 2.8e-14 |

Two details in that table carry lessons that a summary would lose. First, the loss at iteration 1 is already within 0.37 percent of the optimum, while the parameter vector is still 5.005 units away - the loss is much flatter than the coefficients near the optimum, which is precisely why "the loss stopped moving" is an unsafe stopping rule. Second, the distance to the optimum falls by roughly a factor of 95 per iteration while `rho = 0.0095` predicts a factor of 105; the discrepancy is the contribution of the two non-minimal eigendirections, which contract faster than `rho` because `rho` is defined by the worst direction.

Back-transformation is where a standardized pipeline most often breaks. The slopes rescale by `1/s`, the intercept absorbs both mean shifts, and a sign error in either step produces a model that fits the training data perfectly and prices wrongly in production. The check is to predict on a couple of raw rows with both the standardized and the raw coefficient vector and compare in USD; the case does this and finds agreement to 1e-9 USD, which is the tolerance a back-transform should be held to.

### D-08 Momentum: the square-root of the condition number, measured

```text
FORMULA [D-08]:  heavy-ball recursion
                   v_{k+1} = beta*v_k + grad J(w_k)
                   w_{k+1} = w_k - eta*v_{k+1}
                 with the optimal choices
                   beta = ((sqrt(L) - sqrt(mu))/(sqrt(L) + sqrt(mu)))^2
                   eta  = 4/(sqrt(L) + sqrt(mu))^2
                 the worst-case contraction becomes
                   rho = (sqrt(kappa) - 1)/(sqrt(kappa) + 1)
WHERE:
  beta : momentum coefficient, dimensionless, in [0, 1)
  v    : p-vector of accumulated updates, same units as grad J
  eta  : step size, same units as before but a different value
  rho  : contraction factor per iteration, applied to the error norm
CONSTRAINT: fixed L and mu, exact quadratic, no noise. With minibatch noise
            the velocity accumulator amplifies variance, so beta must fall as
            the batch size falls.
NUMERICAL EXAMPLE (240-row design, unscaled basis):
  sqrt(kappa) = 6137.72
  rho_gd  = (kappa-1)/(kappa+1) = 0.9999999469
  rho_mom = (sqrt(kappa)-1)/(sqrt(kappa)+1) = 0.9996742
  steps to 1e-6 : GD 260,226,079    momentum 42,397.9
  ratio = 6137.72 = sqrt(kappa) exactly
  five-point design: rho_mom = 0.786456, steps to 1e-6 = 57.5 vs 483.4
INTERPRETATION: the step-count ratio is sqrt(kappa), not kappa, so momentum
                buys a factor of 6,138 here and cannot buy more.
```

The five-point trace confirms the rate and also shows the early transient that the asymptotic formula hides. Distances to the optimum at steps 1, 2, 5, 10, 20, 40 and 60 are 1.932, 2.020, 1.752, 1.409, 0.934, 5.807e-03 and 7.095e-05. The first two steps actually move *away* from the optimum, because the velocity accumulator starts at zero while the gradient points along the fast direction. A team monitoring only "is it improving?" would abort at step 2, and that is a real operational failure mode rather than a hypothetical one.

Even with momentum, `sqrt(3.77e7) = 6,138` means 42,398 iterations - affordable but still unnecessary, because the standardized basis reaches the same answer in four steps. Momentum is the right tool when the basis cannot be changed; it is not a substitute for fixing the basis.

### D-09 Stochastic descent needs a different bound

```text
FORMULA [D-09]:  a single-sample update uses grad_i = z_i (z_i'w - y_i),
                 whose Hessian is z_i z_i' with spectrum {0, ||z_i||^2}.
                 Stability of the per-step iteration therefore requires
                   eta < 2 / max_i ||z_i||^2
                 which is tighter than 2/L whenever some row's leverage
                 exceeds the average curvature, i.e. whenever
                   max_i ||z_i||^2 > lambda_max(X'X/n)
WHERE:
  z_i : i-th row of the design, p-vector, units mixed
  ||z_i||^2 : squared Euclidean row norm, dimensionless after standardization
  L : lambda_max of the full-batch Gram, dimensionless
CONSTRAINT: applies to batch size 1. For batch size B the bound relaxes toward
            2/L only if the batch row norms are balanced; a batch containing
            the extreme row still carries that row's curvature.
NUMERICAL EXAMPLE (240-row standardized design):
  max_i ||z_i||^2 = 13.0560 at row 155 (area 2092 sq ft, age 11 years)
  min_i ||z_i||^2 = 1.00632,  mean = 3.0
  L = 1.00947
  ratio max||z_i||^2 / L = 12.933
  single-sample bound: eta < 2/13.0560 = 0.15317
MEASURED EXCESS LOSS over the optimum (late-average J - J*, 60 epochs):
  eta = 0.05 : B=1 266.17   B=8 13.58    B=32 1.78
  eta = 0.10 : B=1 646.90   B=8 43.56    B=32 5.85
  eta = 0.20 : B=1 1812.56  B=8 124.46   B=32 20.40
  eta = 0.50 : B=1 diverged (NaN)  B=8 495.02  B=32 102.89
INTERPRETATION: 0.153 is the empirical cliff. eta = 0.10 is stable, eta = 0.20
                is already past the bound (0.20*13.056 = 2.61 > 2), and
                0.50 produces a NaN within the epoch budget.
```

The gap between `2/L = 1.98` and `2/max||z_i||^2 = 0.153` is a factor of 13, and it is invisible to anyone who reasons only from the full-batch spectrum. This is the concrete form of the sentence "SGD is not just noisy GD": the stability requirement is set by the worst row, not by the average curvature, and a design with one large-magnitude row is a design with a much smaller safe step size than its Gram matrix suggests.

Row 155 is worth naming because it is the same row that shows up in the leverage analysis of D-14 with `h = 0.0542`. One row governs two different diagnostics, which is a useful reminder that outliers are not a nuisance to be averaged away; they set the constants.

The batch-size column is equally instructive. At fixed `eta = 0.10`, moving from `B = 1` to `B = 32` cuts the excess loss from 646.90 to 5.85, a factor of 111 for a 32-fold reduction in gradient noise. The reduction is better than `1/B` because the noise also has a bias component near the optimum; either way, the actionable statement is that batch size is a variance knob and step size is a bias-plus-stability knob, and their effects are not interchangeable.

### D-10 The stochastic noise floor is a formula, not a mystery

```text
FORMULA [D-10]:  scalar quadratic  f(w) = (w - w*)^2 / 2 with a noisy
                 gradient  g_k = (w_k - w*) + eps_k,  eps ~ N(0, sigma^2):
                   w_{k+1} = (1 - eta) w_k + eta w* - eta eps_k
                 stationary variance (exact, since |1 - eta| < 1):
                   Var(w_inf) = eta*sigma^2 / (2 - eta)
                 with a batch of size B, sigma^2 is replaced by sigma^2/B.
WHERE:
  w    : scalar parameter
  eta  : step size, dimensionless for unit curvature
  sigma: per-sample gradient noise standard deviation
  Var  : stationary variance of the iterate, units of w squared
CONSTRAINT: unit curvature and independent noise. For curvature h, substitute
            eta -> eta*h and divide the numerator by h; the shape is the same.
NUMERICAL EXAMPLE (sigma = 2, 400,000 steps, last 200,000 averaged):
  eta = 0.02 : measured variance 0.0405, formula 0.0404
  eta = 0.10 : measured variance 0.2096, formula 0.2105
  eta = 0.20 : measured variance 0.4412, formula 0.4444
  eta = 0.50 : measured variance 1.3312, formula 1.3333
  batch scaling at eta = 0.10: B=1 0.2105, B=4 0.0526, B=16 0.0132
INTERPRETATION: the floor is linear in eta and inverse linear in B, so
                halving the step size halves the noise amplitude's effect
                quadratically and costs convergence time linearly.
```

Three digits of agreement between measurement and formula is the evidence. What the formula buys is a decision procedure: given a tolerated parameter jitter, the step size is determined. If the pricing service tolerates 0.05 USD jitter in the intercept with `sigma = 2`, then `eta*4/(2-eta) = 0.0025` gives `eta ~= 0.00125`, which is 800 times smaller than the stability limit - and therefore convergence in the deterministic sense is irrelevant, because the run will never get close enough to matter. At that point the correct move is to increase the batch or switch to the closed form, not to tune `eta`.

The same formula explains why people believe learning-rate decay is magic. Decay reduces `eta` over time, which reduces the floor, but it also freezes the remaining deterministic error at whatever value it had reached. The honest statement is a trade: decay converts an asymptotic variance floor into a fixed residual bias, and the choice between them is a product decision about whether jitter or bias is more expensive downstream.

### D-11 Regularization changes the objective, and the intercept proves it

```text
FORMULA [D-11]:  penalized objective
                   J_lam(w) = J(w) + (lam/2) * sum_{j in P} w_j^2
                 If the intercept is in the penalized set P, the first-order
                 condition at the optimum in a standardized basis gives
                   mean(fitted) = ybar / (1 + lam)
                 If the intercept is excluded, mean(fitted) = ybar exactly.
WHERE:
  lam   : penalty strength, in units of the Gram matrix, not of the features
  P     : index set of penalized coordinates
  ybar  : mean of the training label
CONSTRAINT: the standardized feature block has unit diagonal in G, so a
            penalty lam competes directly with a curvature of 1. Applying the
            same lam to an unscaled design makes the penalty negligible.
NUMERICAL EXAMPLE (240-row standardized design):
  intercept penalized:
    lam = 0.0 : mean fitted 1120.8593, ybar 1120.8593
    lam = 0.1 : mean fitted 1018.9630, ybar/1.1 = 1018.9630
    lam = 1.0 : mean fitted  560.4296, ybar/2   = 560.4296
  intercept excluded:
    lam = 10  : mean fitted 1120.8593 (unchanged), slopes shrink 210.92 -> 19.55
    lam = 100 : mean fitted 1120.8593 (unchanged), slopes shrink to 2.13
INTERPRETATION: penalizing the intercept biases the model by a factor
                1/(1+lam), which is a level shift the business will notice
                long before it notices the slope shrinkage.
```

The `lam = 10` row is the one that catches teams in practice. On a standardized design the Gram eigenvalue is about 1, so `lam = 10` means the penalty is ten times the data curvature and the slopes collapse from 210.92 to 19.55. Train RMSE moves from 82.16 to 488.57 - a 6-fold degradation that looks like a bug in the gradient code and is actually a badly scaled penalty. Penalty strength has to be chosen in the units of the matrix it is added to.

The intercept row is a cleaner demonstration of the same principle. A regularization path that shrinks toward zero in a standardized basis is shrinking toward a rent of zero dollars, which is not a plausible prior for monthly rent. The `1/(1+lam)` factor makes this quantitative: `lam = 1` implies a model whose average prediction is half the observed average rent. Any report showing "ridge improved validation RMSE" alongside an intercept penalized by 1 should be rejected on inspection, without needing the validation numbers.

### D-12 Early stopping, and when it is not a substitute for a penalty

```text
FORMULA [D-12]:  after k steps from w = 0 in the eigenbasis of G,
                   w_k = sum_i (1 - (1 - eta*lambda_i)^k) * (u_i'w*) u_i
                 so each eigendirection is shrunk by a factor depending on k.
                 Early stopping acts like a penalty only if some directions
                 are still far from converged at the chosen k.
WHERE:
  u_i : i-th eigenvector of G, unit norm
  k   : number of steps, dimensionless
  shrinkage_i = (1 - (1 - eta*lambda_i)^k), in [0, 1)
CONSTRAINT: the equivalence is spectral. If kappa ~ 1, all directions converge
            within a few steps and early stopping shrinks nothing.
NUMERICAL EXAMPLE A (standardized design, eta = 1.0, kappa = 1.019):
  k = 3 : w = (1120.8593, 210.9186, -484.3484), which is w* to 4 decimals
  ridge with lam = 0.33 : slopes (159.4385, -364.5383)
  ridge with lam = 1.00 : slopes (106.6039, -242.6683)
  ridge with lam = 3.00 : slopes  (53.5891, -121.4597)
NUMERICAL EXAMPLE B (raw design, eta = 1e-6, kappa = 3.77e7):
  k = 1e6 : w = (37.198, 1.44821, -35.1289) against w* = (948.92, 0.8047, -40.27)
  slopes are partially shrunk, intercept is 96% short of its target
INTERPRETATION: in the standardized basis early stopping cannot regularize
                because there is nothing left to stop; in the raw basis it
                appears to regularize while actually truncating one direction.
```

Example B is the trap. A run stopped at a million iterations has slopes `(1.448, -35.13)` against a target of `(0.805, -40.27)`, which looks like mild shrinkage and is often rationalized as "the model is regularized". The intercept is 37.2 against 948.9, so the fit is nonsense in absolute terms; the shrinkage is not a tuned prior but the accident of stopping while one eigendirection is 96 percent unconverged.

The discriminating check is a two-step procedure. First, compute the per-direction shrinkage `1 - (1 - eta*lambda_i)^k` for each eigenvalue; the pattern should look like a smooth function of `lambda_i` if early stopping is the mechanism. Second, fit the same design with an explicit ridge penalty over a grid and compare the coefficient paths; if the early-stopped path is not on the ridge path at any `lam`, the two are different mechanisms and the "equivalent regularization" claim is false.

### D-13 Precision: the normal equations square the condition number

```text
FORMULA [D-13]:  forming G = X'X squares the condition number,
                   kappa(G) = kappa(X)^2
                 and the number of correct digits in the solve is roughly
                   digits_lost ~ log10(kappa(G)) = 2*log10(kappa(X))
                 with a usable solve while kappa(G) < 1/eps_machine.
WHERE:
  kappa(X) : ratio of extreme singular values of the design, dimensionless
  eps      : machine epsilon, 2.2e-16 for float64, 1.2e-07 for float32
  digits   : decimal digits, approximate
CONSTRAINT: the bound is worst case. Exceeding it does not guarantee a wrong
            answer; it removes the guarantee of a right one.
NUMERICAL EXAMPLE A (240-row design, float64):
  kappa(G) = 3.7672e7,  kappa(G)^2 implied 1.42e15
  1/eps_64 = 4.50e15  ->  kappa^2 still below the limit, marginally
  normal equations, pivoted QR and an SVD-free least-squares call all return
  w = (948.9198, 0.80470828, -40.266636) to the printed digits
  float32 normal equations differ from float64 by
    (1.08e-05, 1.09e-05, 3.9e-07) relative
NUMERICAL EXAMPLE B (200-row near-collinear design, x2 = x1 + 1e-4*noise):
  eig(X'X/n) = (4.64e-09, 0.990286, 1.868116),  kappa = 4.026e+08
  fitted = (1.00006, 3.45865, -2.45849) against a true (1, 2, -1)
  SSE(fitted) = 0.020033, SSE(true) = 0.020041, difference 8.79e-06
INTERPRETATION: kappa^2 = 1.62e17 for Example B is 36 times past 1/eps_64,
                yet all three solvers agreed. The coefficients moved by 1.46
                while the loss moved by 0.04 percent - conditioning destroys
                parameters, not fit.
```

Example B is the most important number in this derivation, because it separates two things people conflate. A condition number of 4e8 does not mean "the answer is garbage"; it means "the answer is not unique in the metric you care about, and the objective cannot tell the difference". The fitted slopes `3.45865` and `-2.45849` are 1.46 away from the truth, yet the loss they achieve is 0.04 percent *lower* than the loss at the truth. The model with the wrong coefficients is genuinely a better fit to the sampled data, and no amount of solver care will change that - the information is not in the data.

The practical consequence for a regression model is that coefficient tables from a collinear design must be read with the objective's flatness in mind, and that any decision resting on the *sign* of a collinear coefficient is unsafe. The sign here is right, but the magnitude is inflated by 73 percent, which is exactly the kind of error that survives a validation RMSE check and fails in a business conversation about elasticity.

### D-14 What the fit is worth: leverage and leave-one-out

```text
FORMULA [D-14]:  leverage  h_i = x_i' (X'X)^{-1} x_i,  sum_i h_i = p
                 leave-one-out residual  r_i^{(loo)} = r_i / (1 - h_i)
                 PRESS  = sum_i (r_i / (1 - h_i))^2
                 coefficient covariance  Cov(w) = s^2 (X'X)^{-1},
                   s^2 = SSE / (n - p)
WHERE:
  h_i : leverage of row i, in [0, 1], dimensionless
  r_i : ordinary residual of row i, USD
  s^2 : residual variance estimate, USD^2
  p   : number of parameters, 3; n = 240
CONSTRAINT: requires full column rank and a correct design matrix. Leverage is
            about the design only; it says nothing about whether the label is
            wrong.
NUMERICAL EXAMPLE (240-row design):
  mean h = 3/240 = 0.0125, min 0.004193, max 0.054209 (row 155)
  rows above the 2p/n = 0.025 threshold: 18 of 240
  in-sample RMSE 82.1597, PRESS RMSE 83.2458 (1.3% higher)
  largest leave-one-out residual 216.84 USD at row 128
  s^2 = 6835.667, s = 82.678
  standard errors (26.619, 0.020362, 0.44370)
  t-statistics (35.65, 39.52, -90.75)
INTERPRETATION: no single row dominates the fit, so the coefficients are not
                an artefact of one lease; the age slope is 90 standard errors
                from zero while its business magnitude is 40 USD per year.
```

The leverage numbers answer a different question from the conditioning numbers, and the case keeps them separate on purpose. Conditioning asks whether the loss surface determines the parameters; leverage asks whether one row determines the surface. A design can be well conditioned and still have one row with `h = 0.9`, in which case the fit is a fit to that row.

Here the maximum leverage is 0.054, which is 4.3 times the mean and only 2.2 times the conventional `2p/n` threshold. Eighteen rows exceed that threshold, which is normal for a design with a continuous feature of wide range. The PRESS RMSE exceeding the in-sample RMSE by 1.3 percent says the same thing quantitatively: deleting any one row changes the fit by about a percent, not by a factor.

The t-statistic for the age slope is -90.75, and this is where a case like this must resist its own momentum. A t of 90 does not mean the coefficient is right; it means the coefficient is precisely estimated *given the model*. If the true relationship is nonlinear in age - a 1910 building and a 1990 building do not differ by 40 USD per year in the same way - the standard error is measuring precision around a misspecified centre. The optimizer question is settled by these numbers; the model-specification question is not.

## 7. The Finding

The finding is a decision rule with a measured basis, stated in the order the evidence supports it.

First, for this design the closed form is not merely cheaper, it is the only defensible reference: 4,320 flops for a pivoted QR solve against 3.75e11 flops for an honest fixed-step descent, a ratio of 8.7e7.

Second, fixed-step descent in the raw basis at a hand-chosen step size is not a training method; it is a partial solve with a plausible-looking log. At `eta = 1e-6` the loss falls 65 percent in the first ten iterations and then stalls, ending a million iterations later at 5.95 times the optimal loss with a coefficient vector 96 percent of its initial distance from the optimum.

Third, the cause is the intercept's scale, not the features' collinearity. The features correlate at -0.00947; the smallest Gram eigenvalue equals `1 - R^2 = 0.0402` of an auxiliary regression of the constant column on the features. Standardizing both features moves `kappa` from 3.77e7 to 1.019 and the iteration count from 2.6e8 to 4.

Fourth, descent earns its place under conditions the case can state precisely: when the loss is not quadratic, when the design does not fit in memory and only matrix-vector products are available, or when the data arrive as a stream. Under those conditions the contract is: measure `L` by power iteration (43,200 flops for 30 passes here), set `eta = 2/(L+mu)`, express the stop rule as a relative gradient norm, and for any minibatch run replace the stability bound with `2/max_i ||z_i||^2`.

The decision, therefore, is to approve a training contract that forbids an unmeasured step size and requires the spectrum in the model artefact - and to use the closed form for this particular model, with descent retained only for the streaming variant that the unit will examine next.

The magnitude of the arithmetic is illustrative; the identities are not. `mu = 1 - R^2`, `rho = (kappa-1)/(kappa+1)`, `k(eps) = ln(eps)/ln(rho)`, the `sqrt(kappa)` momentum speedup and `eta*sigma^2/(2-eta)` are exact statements about least squares and were each checked numerically here to at least three significant figures.

## 8. System/Solution Design

The design is a training pipeline whose every stage emits an artefact that an auditor can read without re-running anything.

```text
[training rows]
      |
      v
[split: train / validation by signing date]
      |                                   \
      v                                    v
[standardize: m, s from TRAIN only]   [validation rows, untouched]
      |                                    |
      v                                    |
[G = Z'Z/n, eigendecomposition]            |
      |  -> L, mu, kappa, eta*, rho         |
      |  -> k(eps) budget                   |
      v                                    |
{plan gate: kappa < 1e6 ? } ---- no ---> [refuse: precondition or use closed form]
      | yes                                |
      v                                    |
[reference solve: pivoted QR] --> w_ref    |
      |                                    |
      v                                    |
{descent only if non-quadratic or streaming}
      |                                    |
      v                                    v
[fit + stop on ||grad||/||grad_0||]   [score on validation]
      |                                    |
      v                                    v
[ledger: design hash, spectrum, eta, stop reason, seed, w, w_ref, delta]
```

State ownership is explicit. The ledger owns the design hash and the spectrum; the fitter owns nothing durable; the serving layer owns the back-transformation constants (`m`, `s`) because a wrong back-transform is a serving failure, not a training failure. Retries re-read the design hash before recomputing anything, so a nightly job that fires twice produces one artefact.

The plan gate is the load-bearing control. It refuses to run descent when `kappa` is large enough that the iteration budget exceeds the batch window, and it names the two acceptable alternatives instead of silently substituting one. A gate that only warns is a gate that will be ignored at 2 a.m.

Two failure paths are wired in deliberately. If `lambda_min` falls below `1e-8`, the gate refuses outright rather than returning a large `eta`, because the fitted vector in that regime is not determined by the data (D-13, Example B). If the measured `L` from power iteration disagrees with `lambda_max` from the eigendecomposition by more than 1 percent, the run halts: that disagreement indicates either a stale design hash or a convergence failure in the power iteration, and both invalidate the step size.

Instrumentation is chosen so that the three most common incidents are visible from a single log line: `kappa`, the iteration count actually used, and `||w - w_ref||/||w_ref||` at stop. A run that stops early shows a large third number; a run that stops because of a step-size problem shows a wildly oscillating loss in the first ten entries; a run with a scaling problem shows `kappa` in the millions.

Rollback is the closed form. Because the reference vector is computed on the same design and stored in the ledger, switching the serving model back to `w_ref` is a pointer change with no retraining, and the validation score for it is already in the ledger.

## 9. Evaluation & Validation

The evaluation question is not "is the model accurate" but "is the training procedure correct", so the metrics are about the solver.

| Element | Definition |
|---|---|
| Numerator (parameter agreement) | `max_j |w_j - w_ref_j| / |w_ref_j|` over the three coordinates, dimensionless |
| Denominator (parameter agreement) | the same expression evaluated for the baseline library call, so the ratio is relative improvement |
| Numerator (loss gap) | `J(w) - J(w_ref)`, USD^2 |
| Denominator (loss gap) | `J(0) - J(w_ref)`, USD^2, which is the total achievable improvement |
| Population | the 240 constructed leases in the training design; no external population |
| Window | one training run per configuration; iteration marks at 1, 10, 100, 1e3, 1e4, 1e5, 1e6 |
| Comparator | pivoted QR closed-form solution on the identical design hash |
| Guardrail 1 | in-sample RMSE must not exceed the closed-form RMSE by more than 1 percent, because RMSE can look acceptable while coefficients are wrong |
| Guardrail 2 | the loss must be non-increasing for full-batch runs; any increase invalidates the step size |
| Guardrail 3 | validation RMSE must be computed with `m` and `s` from train rows only |
| Uncertainty | coefficient standard errors from D-14; the 40-split validation standard deviation for any `lambda` comparison |
| Executed status | guards 1-3, the closed form, and the full trace to 1e6 steps: executed on the constructed design. Any production claim: not-executed |

Three acceptance bands are proposed, each with the arithmetic that produced it.

The parameter band is 1e-6 relative, which is the point at which back-transformed rent predictions agree to 1e-9 USD on the constructed rows. A looser band would let a standardized run stop at iteration 3, where the distance to the optimum is 4.49e-04 in standardized units and the implied rent error is about 0.03 USD - tolerable, but the band exists to be a contract, not a preference.

The loss-gap band is 1e-3 relative to the total achievable improvement, i.e. `(J - J*)/(J(0) - J*) < 1e-3`. Under this band the `eta = 1e-6` raw run fails at every mark recorded, since it reaches only 97.8 percent of the achievable improvement after a million steps: `(20,081.52 - 3,375.11)/(772,045.56 - 3,375.11) = 2.18 percent` remaining.

The monotonicity guardrail is the one that catches step-size errors, and it is cheap. Any increase in a full-batch loss means `eta >= 2/L`, so the diagnostic is deterministic: the run is invalid, and no amount of additional iterations will fix it.

Uncertainty is reported where it changes a decision and omitted where it does not. The `lambda` comparison in D-11/D-12 uses 40 random splits, which gives a paired difference of `+4.79 +/- 0.662` standard error for `lam = 10` against `lam = 0` at the same `eta` - a real degradation, not noise. The iteration-count comparison needs no interval because it is computed from eigenvalues, and the only uncertainty is in the eigenvalue estimate, which matched to seven digits between two methods.

Nothing in this table is a deployed result. The status column says executed for the constructed design and not-executed for anything a business would rely on, and that distinction is the difference between an audit and a claim.

## 10. Action Taken

No production action was taken. The baseline library call is still in place, unchanged, because the case has no authority to modify a serving system and because replacing a working fitter on the strength of a synthetic design would be exactly the kind of unsourced action the contract forbids.

Three preparatory actions are proposed, each with an owner and a stopping condition.

The first is to instrument the nightly job so that it records `kappa`, `lambda_max`, the iteration count, and the stop reason for the design actually used. This is a read-only change to logging and needs no model change. The stopping condition is that ten consecutive nightly runs produce a readable spectrum; if any run cannot produce one, the design hash mechanism is broken and that becomes the finding.

The second is to add the closed-form reference solve to the same job and log `||w - w_ref||/||w_ref||`. This converts an invisible failure into a number. The stopping condition is that the comparison is stored for every run, including runs where the library returns successfully.

The third is a one-week offline replay on the same 240-row design with three configurations - raw descent at a measured step, standardized descent at `eta = 1`, and the closed form - scored on the three acceptance bands of §9. The replay is bounded: it runs on one design, and its output is a decision about the training contract, not a model improvement claim.

Rollback for all three is trivial because none of them changes a coefficient that is served. The instrumentation can be disabled by configuration, and the reference solve is an additional artefact rather than a replacement.

What was explicitly rejected as an action: retuning the library's default step size by trial until the nightly loss looked smooth. That action would produce a better-looking log with no evidence that the parameters are closer to the optimum, and it would destroy the audit trail the case is trying to build.

## 11. Outcome & Results

**Result status: not measured.** No deployed system, no company, and no market is claimed. The results section is therefore a statement about arithmetic, and it is separated into what was checked and what remains open.

Checked, with the method used:

| Claim | Value | Check |
|---|---|---|
| Gradient of the objective is correct | relative error <= 4.1e-08 per coordinate | central finite difference at h = 1e-6 |
| Closed form on five points | `(2.2, 0.6)`, SSE 2.4 | rational arithmetic on `Sxx = 10`, `Sxy = 6` |
| Normal-equation identity on 240 rows | `max |X'r| ` relative 2.7e-15 | direct matrix-vector product at `w_ref` |
| Spectrum of the raw design | `(0.040195, 163.296, 1514216.26)` | symmetric eigendecomposition of `X'X/n` |
| Smallest eigenvalue identity | `1 - R^2 = 0.0401965` against `0.0401952` | auxiliary least-squares of a constant on the features |
| Contraction prediction on five points | predicted 2.30731e-06 at k = 483, measured 2.30731e-06 | closed-form `||w*|| rho^k` |
| Iteration budget on 240 rows | 2.602e8 steps to 1e-6 | `ln(1e-6)/ln(rho)` with measured `L`, `mu` |
| Loss after 1e6 raw steps | 20,081.52 against optimum 3,375.11 | direct iteration with `eta = 1e-6` |
| Slow-component decay over 1e6 steps | predicted 0.9606, measured 0.9599 | `exp(-eta*mu*k)` against distance ratio |
| Standardized convergence | 4 steps to 4.3e-06 distance | direct iteration with `eta = 1.0` |
| Momentum speedup | 6,137.72 = `sqrt(kappa)` | ratio of computed step counts |
| Single-sample stability cliff | 0.1532 predicted, between 0.10 and 0.20 observed | 60-epoch mini-batch grid, one seed |
| SGD noise floor | formula within 1 percent of measured variance | 400,000-step scalar simulation |
| Intercept penalty bias | mean prediction = `ybar/(1+lam)` exactly | three `lam` values, standardized design |
| Precision limit | float32 drift 1.1e-05 relative | duplicate solve in float32 |
| Influence | PRESS RMSE 83.246 against in-sample 82.160 | leverage from `X(X'X)^{-1}X'` |

Open, and stated as open rather than assumed favourable:

- Whether the real lease ledger is as well conditioned as the constructed design. The constructed design was built with independent features, and real ledgers often contain building-level attributes that correlate strongly with area.
- Whether the production fitter already standardizes internally. If it does, the case's central complaint does not apply to it and the audit should start from its spectrum instead.
- Whether the business tolerance is 0.03 USD or 3 USD per month in rent error, which determines the parameter band and therefore whether four iterations or four hundred are needed.
- Whether the residual standard deviation in production is near the constructed 82 USD, which sets the stochastic floor in D-10 and therefore whether minibatching is viable at all.
- The wall-clock cost of the closed form on the production ledger size, which was not measured here because the production size is unknown to this case.

## 12. Failure Modes & What Went Wrong

Each entry names a mechanism, the competing cause that produces the same symptom, the check that separates them, the immediate recovery, and the permanent repair. All of them occurred or were reproduced during the construction of this case.

- **Step size at or above `2/L`.** Symptom: the loss falls, then flattens at a value far above the optimum, or a loss increase appears after an apparently healthy start. Competing cause: a genuinely hard optimisation landscape. Discriminating check: project the update onto the top eigenvector; an alternating sign with constant magnitude means `eta >= 2/L`, whereas a shrinking same-sign projection means the run is merely slow. Recovery: halve `eta` and confirm the loss becomes non-increasing. Repair: compute `2/L` from a measured spectrum and store it in the run manifest.

- **Intercept scale dominating the spectrum.** Symptom: the intercept crawls toward its target while slopes leap to roughly the right magnitude; `kappa` in the millions on a design with two nearly uncorrelated features. Competing cause: a bad learning-rate schedule. Discriminating check: compare `lambda_min` to `1 - R^2` of the features against a constant column - equality identifies the intercept as the culprit. Recovery: centre and scale the feature block, refit. Repair: make standardization part of the pipeline rather than a notebook step.

- **Intercept inside the penalty.** Symptom: validation RMSE improves marginally while the average prediction is far from the average label. Competing cause: distribution shift between train and validation. Discriminating check: compare `mean(fitted)` to `ybar/(1+lam)`; exact agreement identifies the penalty as the cause. Recovery: exclude the intercept from the penalized set and refit. Repair: assert in code that the intercept is unpenalized, and log the mean prediction next to the mean label.

- **Early stopping mistaken for regularization.** Symptom: coefficients are shrunk relative to the closed form, and a reviewer describes the run as "gently regularized". Competing cause: a genuinely tuned ridge penalty. Discriminating check: compute per-direction shrinkage from `1 - (1 - eta*lambda_i)^k`; if the pattern tracks `k` rather than a penalty grid, the mechanism is truncation. Recovery: run to convergence and add an explicit penalty if shrinkage is wanted. Repair: forbid "stop early for regularization" as a rationale in the training contract.

- **Single-sample step size inherited from the full-batch spectrum.** Symptom: a mini-batch run with `eta = 2/L` produces NaN or an exploding loss while the same `eta` is stable in full batch. Competing cause: a data-loading bug producing duplicated rows. Discriminating check: compute `max_i ||z_i||^2` and compare `eta * max_i ||z_i||^2` to 2; a value above 2 confirms the step-size mechanism. Recovery: reduce `eta` to `2/max_i ||z_i||^2`. Repair: report the per-sample bound alongside the full-batch bound for every stochastic run.

- **Penalty strength scaled to the wrong matrix.** Symptom: slope shrinkage by a factor of ten at `lam = 10` on a standardized design, with train RMSE rising 6-fold. Competing cause: a gradient bug that shrinks all coordinates. Discriminating check: compute the ratio of `lam` to the diagonal of the Gram; a ratio of 10 means the penalty dwarfs the data. Recovery: set `lam` on a grid scaled to the Gram's trace. Repair: parameterize the penalty as a fraction of `trace(G)/p`.

- **Near-collinear features blamed on the solver.** Symptom: coefficients move by 1.46 when the design is perturbed slightly, while the loss moves by 0.04 percent. Competing cause: numerical instability in the normal equations. Discriminating check: solve with pivoted QR and with an SVD; if all three agree, the solution is stable and the *problem* is ill-posed. Recovery: do not re-run the solver; report coefficient uncertainty and remove or combine the collinear columns. Repair: add a variance-inflation or eigenvalue check to the design review.

- **Float32 drift accepted as negligible.** Symptom: coefficients differ in the fourth significant figure between two environments that run the same code. Competing cause: a different library or BLAS build. Discriminating check: recompute the Gram in float64 on the same design and compare; a systematic `1e-05` relative difference points to dtype, not to library version. Recovery: pin to float64 for the Gram and the solve. Repair: record the dtype in the manifest and fail the run if it is not float64.

- **Standardization fitted on the full dataset.** Symptom: validation scores are slightly but consistently better than a re-run with train-only scaling, and the gap grows when `n` is small. Competing cause: a lucky split. Discriminating check: refit the scaler on train only and compare the validation score on the identical split; a drop confirms leakage. Recovery: refit the pipeline end to end. Repair: make the scaler a fitted object owned by the training pipeline, not a preprocessing step applied to the table.

- **Loss used as the stopping rule.** Symptom: the run stops when the loss stops improving, at a point where the coefficient distance is still 96 percent of its initial value. Competing cause: the model has genuinely converged. Discriminating check: print the loss gap relative to the closed form alongside the loss; a 5.95-fold gap with a flat curve is unambiguous. Recovery: continue until the relative gradient norm meets the band. Repair: make the relative gradient norm the stop rule and the loss a logged diagnostic.

## 13. Transferable Lesson

The transferable content is a five-part contract, and it is about the object being optimised rather than about the optimiser's name.

The first part is a measured curvature. Any claim that a fixed-step method works is a claim about `L`, and `L` is available in two numbers - a power iteration and an eigenvalue - for the cost of a handful of passes over the data.

The second part is a basis chosen for the geometry, not for convenience. The same design had `kappa = 3.77e7` raw and `1.02` standardized, with identical data and identical loss. Scaling is not a cosmetic preprocessing step; it is a change in the effective problem, and it changed the iteration budget by seven orders of magnitude.

The third part is a stop rule in units of the quantity that matters. The loss is flat near the optimum in the directions that matter least, so a flat curve is compatible with a wildly wrong parameter vector. Relative gradient norm and distance to a reference are the honest measures.

The fourth part is a reference. Without the closed form, the case has no comparator, and the 20,081-versus-3,375 gap is invisible. Any approximate procedure should be audited against an exact solution of the same objective on the same data, whenever one exists.

The fifth part is a boundary statement. Every formula here has one: `eta < 2/L`, `eta < 2/max||z_i||^2`, `kappa` finite, intercept excluded from the penalty, full column rank. A method taught without its boundary is a method that will be applied outside it.

None of the numbers transfer. A different design has a different spectrum, and the fact that 2.6e8 iterations were needed here says nothing about a 2,000-row ledger with different features. What transfers is the procedure: measure, choose the basis, set the step from the measurement, stop on a stated criterion, and compare against an exact reference.

## 14. Connections & Cross-References

Within this unit, the case is the numerical counterpart of the calibration case: calibration asks whether a score means what it claims, and this case asks whether a parameter vector is where it claims to be. Both reduce to comparing a produced artefact against an independent reference on the same inputs.

| Neighbour | Relation | What differs |
|---|---|---|
| `01_Calibrate_a_Risk_Model.md` | same unit, decision-audit pattern | that case audits a probability, this one audits a parameter vector |
| `Cheat_Sheets/01_Matrix_and_Vector_Identities.md` | supplies the identities used in D-01 and D-06 | the sheet states them; this case measures them |
| `Cheat_Sheets/99_25_Common_Issues.md` | catalog of the defects D-04 and D-09 instantiate | the catalog names causes; this case shows the arithmetic |
| Unit 04 streaming cases | the regime where descent is the right answer | there the design does not fit in memory, so the closed form is unavailable |
| Any unit using ridge or L2 regularization | D-11 and D-12 supply the intercept and early-stopping traps | those units tune `lam`; this one shows what `lam` is measured against |

Two external anchors are worth naming because they are checkable. The heavy-ball momentum constants are Nesterov's; the `mu = 1 - R^2` identity is a standard result in regression diagnostics. Neither is claimed as a novel finding, and both were re-derived numerically here rather than quoted.

### Interview pack

**Q1. Derive the gradient of `J(w) = (1/2n)||Xw - y||^2` and say which term is easy to forget.**
**Answer:** `grad J = (1/n)X'(Xw - y)`. Expanding gives `(1/2n)(w'X'Xw - 2w'X'y + y'y)`, whose derivative is `(1/n)(X'Xw - X'y)`.
**Reasoning:** the `1/n` survives differentiation because it multiplies the whole expression; dropping it scales every step by `n` and makes a stable step size look unstable.
**Follow-up:** at `w = 0` on this design the gradient is `(-1120.86, -1,403,917, -16,292)`, so which coordinate dominates, and what does that say about a shared step size?
**Red flag:** writing `X'(Xw - y)` without the `1/n` and then adjusting `eta` by eye until it works.
**Scoring:** full credit for the derivation plus a statement that the Hessian is `X'X/n` and is constant.

**Q2. The five-point design has `Sxx = 10` and `Sxy = 6`. Compute the fit and the RMSE.**
**Answer:** slope `0.6`, intercept `2.2`, fitted `(2.8, 3.4, 4.0, 4.6, 5.2)`, residuals `(-0.8, 0.6, 1.0, -0.6, -0.2)`, `SSE = 2.4`, `MSE = 0.48`, `RMSE = 0.6928`.
**Reasoning:** `Sxx = 55 - 225/5 = 10` and `Sxy = 66 - 300/5 = 6`, so `w1 = 0.6`; then `w0 = 4 - 0.6*3 = 2.2`. Residuals sum to zero as a check.
**Follow-up:** what is `R^2` here, and why is it exactly `0.6` rather than approximately?
**Red flag:** reporting a fit without checking that the residuals sum to zero.
**Scoring:** full credit requires the two centred sums to be shown, not just the final slope.

**Q3. `kappa = 3.77e7` on a design whose features correlate at `-0.0095`. Explain the contradiction.**
**Answer:** there is no contradiction; the condition number is not a feature-correlation measure. With an intercept, `lambda_min = 1 - R^2` where `R^2` is from regressing the constant column on the features, which is 0.9598 here.
**Reasoning:** the features span a direction that is almost exactly the constant, because a small combination of a large-magnitude column can reproduce it. That near-null direction is the slow eigendirection, `(0.99998, -0.00071, -0.00564)`.
**Follow-up:** if you centred but did not scale, what `kappa` would you get, and what does that tell you about which transformation is doing the work?
**Red flag:** concluding that the features are collinear because the condition number is large.
**Scoring:** full credit for the identity and for naming centring as the transformation that removes most of the problem.

**Q4. Estimate the iteration count needed for `eta = 1e-6` on this design, and reconcile it with the measured trace.**
**Answer:** the dominant decay is `1 - eta*lambda_min = 1 - 4.0195e-08`, so an e-fold takes 2.49e7 steps and 1e6 steps retain `exp(-0.0402) = 0.9606` of the slow error. Measured distance ratio at 1e6 steps is `911.74/949.77 = 0.9599`, which matches.
**Reasoning:** the error decomposes per eigendirection, and the fast two are gone within a hundred steps, so the distance is dominated by the slow one.
**Follow-up:** what loss gap does that predict, given `J* = 3,375` and the measured `J = 20,082`?
**Red flag:** quoting `k(eps) = ln(eps)/ln(rho)` with `rho` computed at `eta = 1e-6` instead of at the optimal step.
**Scoring:** full credit for the per-direction argument and the numerical reconciliation.

**Q5. Why does `eta = 2/L` fail even though it is the textbook boundary?**
**Answer:** at that step the top eigendirection has multiplier `-1`, so its error alternates sign with constant magnitude instead of decaying. Any excess `delta` makes the multiplier `-(1 + delta*L)`, which grows.
**Reasoning:** the stability condition is `|1 - eta*lambda_i| < 1`; at equality the magnitude is one and the sign flips.
**Follow-up:** on the two-parameter quadratic with eigenvalues 1 and 4, `eta = 0.5` leaves `w = (1.0, 0.0)` against an optimum of `(1.0, 0.5)`. Describe the iteration.
**Red flag:** saying "`2/L` is safe because it is the maximum stable value".
**Scoring:** full credit for the sign-alternation mechanism and for the observation that a loss curve can look flat during it.

**Q6. You must choose between closed form, standardized descent, and raw descent for a 240x3 design. Which and why?**
**Answer:** closed form or standardized descent; raw descent only if neither is available. The closed form costs 4,320 flops; standardized descent reaches the answer in four iterations of 1,440 flops; raw descent needs 2.6e8 iterations.
**Reasoning:** with `p = 3` the decomposition is free, so the budget is known before the run.
**Follow-up:** at what `n` does the comparison change, given that the closed form costs `2np^2` and each descent iteration costs `2np`?
**Red flag:** choosing descent because "it scales better" without computing the iteration count.
**Scoring:** full credit for quoting both budgets and the condition under which descent wins.

**Q7. Where does the `sqrt(kappa)` momentum speedup come from?**
**Answer:** with `beta = ((sqrt(L)-sqrt(mu))/(sqrt(L)+sqrt(mu)))^2` and `eta = 4/(sqrt(L)+sqrt(mu))^2`, the worst-case contraction is `rho = (sqrt(kappa)-1)/(sqrt(kappa)+1)`, so `ln(eps)/ln(rho)` shrinks by a factor of about `sqrt(kappa)`.
**Reasoning:** the velocity accumulator turns the first-order recursion into a second-order one whose roots are symmetric in `sqrt(lambda)`, not `lambda`.
**Follow-up:** here `sqrt(kappa) = 6,138` and the measured step ratio is 6,137.72. What is the cost of this speedup in the presence of gradient noise?
**Red flag:** claiming a `kappa`-fold speedup, or claiming momentum removes the need for a good basis.
**Scoring:** full credit for the square root and for noting the noise amplification.

**Q8. A mini-batch run with `eta = 2/L` diverges while the full-batch run is stable. Diagnose.**
**Answer:** the single-sample Hessian is `z_i z_i'` with top eigenvalue `||z_i||^2`. Here `max||z_i||^2 = 13.056` while `L = 1.0095`, so the safe step is `0.1532`, not `1.98`.
**Reasoning:** stability is governed by the worst curvature actually evaluated, and one row can exceed the average by a factor of thirteen.
**Follow-up:** the empirical cliff sits between `eta = 0.10` and `0.20`; which predicted value does that bracket, and why is it not exactly the theoretical boundary?
**Red flag:** blaming the batch size for the divergence when the step size is the cause.
**Scoring:** full credit for `max||z_i||^2`, the number 0.153, and the row-155 observation.

**Q9. Write the stationary variance of scalar SGD and use it to pick a step size.**
**Answer:** for unit curvature and gradient noise variance `sigma^2`, `Var(w_inf) = eta*sigma^2/(2-eta)`; with batch `B` replace `sigma^2` by `sigma^2/B`. For `sigma = 2` and a tolerated standard deviation of 0.05, `eta*(4)/(2-eta) = 0.0025` gives `eta ~ 0.00125`.
**Reasoning:** solving `w_{k+1} = (1-eta)w_k + eta w* - eta eps_k` at stationarity gives the recursion `v = (1-eta)^2 v + eta^2 sigma^2`.
**Follow-up:** halving `eta` halves the variance but doubles the convergence time; when is that trade bad?
**Red flag:** reporting a converged loss while the parameter is jittering by more than the tolerance.
**Scoring:** full credit for the recursion-derived formula and one worked `eta` choice.

**Q10. A ridge fit has a validation RMSE slightly better than OLS but predicts a mean rent half the observed mean. What happened?**
**Answer:** the intercept was inside the penalty, so the optimum satisfies `mean(fitted) = ybar/(1+lam)`. At `lam = 1` on the standardized design that is `1120.86/2 = 560.43`, which matches the observed mean prediction exactly.
**Reasoning:** the first-order condition for the intercept in a penalized objective includes `lam*w0`, so the residual mean is `-lam*w0` rather than zero.
**Follow-up:** does excluding the intercept from the penalty fix the slope shrinkage too, and what does `lam = 10` do to the slopes when the Gram has unit diagonal?
**Red flag:** accepting the improvement without checking the mean prediction against the mean label.
**Scoring:** full credit for the `1/(1+lam)` identity and for the unit-diagonal observation.

**Q11. Is stopping at `k = 1000` a form of regularization?**
**Answer:** only in the raw basis, and there it is truncation rather than regularization. In the standardized basis, `k = 3` already matches the closed form, so there is nothing left to shrink.
**Reasoning:** per-direction shrinkage is `1 - (1-eta*lambda_i)^k`; with `kappa = 1.019` all three factors reach one within four steps.
**Follow-up:** in the raw basis at `k = 1e6`, the slopes are `(1.448, -35.13)` against `(0.805, -40.27)` and the intercept is `37.2` against `948.9`. Is that shrinkage or truncation?
**Red flag:** describing early stopping as "L2 regularization" without checking that the directions are actually converged.
**Scoring:** full credit for the spectral shrinkage formula and the distinction between truncation and a penalty.

**Q12. Two collinear columns give coefficients 1.46 away from the truth while the loss is 0.04 percent lower. Is the solver broken?**
**Answer:** no. With `x2 = x1 + 1e-4*noise`, the Gram has eigenvalue `4.64e-09` against `1.87` at the top, `kappa = 4.03e8`. The fitted vector achieves `SSE = 0.020033` against `0.020041` at the true coefficients, so the fit is genuinely better on the sample.
**Reasoning:** a near-null direction means the loss is flat along a line, and the sample optimum can lie far along that line from the population optimum.
**Follow-up:** `kappa^2 = 1.62e17` exceeds `1/eps_64 = 4.5e15`; why did pivoted QR still agree with the normal equations?
**Red flag:** "fixing" the instability by switching solvers when the problem is ill-posedness.
**Scoring:** full credit for separating conditioning from ill-posedness and for quoting both SSE values.

**Q13. What does `mu = 1 - R^2` predict about a feature that is constant in the training window?**
**Answer:** the auxiliary regression of the constant column on the features is then perfect, `R^2 = 1`, `mu = 0`, and no step size converges because the objective has a flat direction. The closed form does not exist either.
**Reasoning:** a feature column that is itself constant is a duplicate of the intercept, so the design is rank deficient by one.
**Follow-up:** how would you detect this before fitting, and what would you do with the column?
**Red flag:** adding a small ridge penalty and calling it a fix without saying that the objective changed.
**Scoring:** full credit for the rank-deficiency mechanism and for naming the pre-fit check.

**Q14. Give a stopping rule that would have caught the `1e6`-step run.**
**Answer:** stop on relative gradient norm `||grad J(w)||/||grad J(0)|| <= 1e-5`, and log the distance to a closed-form reference. At 1e6 steps the ratio is `36.65/1,404,012 = 2.61e-05` and the distance to the reference is 911.74 units, so both diagnostics flag the run.
**Reasoning:** the gradient norm measures distance from optimality directly, whereas the loss measures a quantity that is quadratic and therefore flat near the optimum.
**Follow-up:** why is `2.61e-05` already below a `1e-5`-style band on the gradient yet still 5.95 times the optimal loss?
**Red flag:** using "loss changed by less than `1e-6`" as the criterion on a loss whose scale is 772,045.
**Scoring:** full credit for a scale-free criterion plus the reference comparison.

**Q15. Your standardized pipeline predicts well in validation but prices wrongly in production. Where do you look first?**
**Answer:** the back-transformation. Slopes rescale by `1/s`, the intercept absorbs `m` and `s` together, and a sign or ordering error there leaves validation untouched because validation also runs on standardized inputs.
**Reasoning:** standardization is inside the model, so any train/serve mismatch is invisible to a metric computed in standardized space.
**Follow-up:** here `210.9182/262.1051 = 0.804708` and the intercept reconstructs to `948.9198`. What single test would catch a swapped mean vector?
**Red flag:** re-checking the optimizer when the symptom is a level shift in predictions.
**Scoring:** full credit for the back-transform mechanism and for a test in raw units.

**Q16. Rank the three scalings by iteration budget, and state what each transformation contributes.**
**Answer:** raw `2.6e8`, centred `k(1e-6) = ln(1e-6)/ln(0.99997089) = 4.74e5` at the optimal step, standardized about 4 iterations at `eta = 1`. Centring removes the intercept's magnitude, and scaling equalizes the feature curvatures.
**Reasoning:** raw `kappa = 3.77e7`, centred `6.87e4`, standardized `1.019`; the budget scales as `kappa/2 * ln(1/eps)`.
**Follow-up:** why does centring alone leave a factor of 68,699 in the spectrum when the features are nearly uncorrelated?
**Red flag:** attributing the improvement to a single transformation without measuring the intermediate spectrum.
**Scoring:** full credit for all three spectra and for stating which transformation fixes which defect.

## 15. Quick Reference Card

| Field | Decision card |
|---|---|
| Unit | one fitted coefficient vector on one design hash |
| Baseline | unaudited library call with default step size |
| Contract | measure `L` and `mu`, choose the basis, set `eta` from the spectrum, stop on relative gradient norm, compare to closed form |
| Update | `w <- w - eta*(G w - c)` with `G = X'X/n`, `c = X'y/n` |
| Stability | `0 < eta < 2/L`; strict, because at `eta = 2/L` the top direction alternates forever |
| Budget | `k(eps) = ln(eps)/ln(rho)`, `rho = (kappa-1)/(kappa+1)`; here 2.6e8 iterations |
| Diagnosis | `lambda_min = 1 - R^2` against a constant column identifies an intercept-scale problem |
| Stochastic bound | `eta < 2/max_i ||z_i||^2`; here 0.153 instead of 1.98 |
| Noise floor | `Var = eta*sigma^2/(2-eta)` for unit curvature, `sigma^2` divided by batch size |
| Momentum | step-count ratio `sqrt(kappa)`; here 6,138x, not `kappa` |
| Penalty | intercept outside the penalty, or the mean prediction becomes `ybar/(1+lam)` |
| Precision | `kappa(X'X) = kappa(X)^2`; here `1.42e15` against a `4.5e15` limit |
| Escalate | any run that stops without reporting `kappa`, the step size, and the distance to a reference |
| Evidence | constructed 240-row design, verified arithmetic, no production measurement |
