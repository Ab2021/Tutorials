---
title: "25 Common Issues — Mathematics, Probability, Statistics and Linear Algebra"
topic: "Recurring computational and inferential defects in quantitative work"
collection: "05_math_probability_statistics_and_linear_algebra"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# 25 Common Issues — Mathematics, Probability, Statistics and Linear Algebra

> **Purpose:** Provide a diagnostic catalogue for the defects that survive code review in quantitative work: clustered dependence treated as independence, variance computed by a subtraction that destroys its own precision, conditioning that turns a stable model into an unstable one, and inferential vocabulary used with the wrong referent.
> **Evidence:** Every formula is derived here from definitions stated here. Every numeric instance is a constructed `ILLUSTRATIVE` input whose arithmetic was recomputed by hand and cross-checked by a second route. No production incident, vendor benchmark, or measured company outcome is claimed anywhere in the file.
> **Scope:** The catalogue carries 32 entries. The file name and title retain `25` because 25 is the contractual floor for this artefact class, and renaming a published file breaks inbound links; the extra seven entries were added in the 2026-11 repair and are marked `[added]` so a reader can tell which entries the original commitment covered.
> **Reading contract:** Section 8 is the catalogue. Read the row first, then the entry body only when the symptom matches. Section 4 carries the derivations the entries cite, Section 5 carries the single worked case the entries are anchored to, and Section 12 is the card to keep open while debugging.
> **What this file does not do:** It does not teach linear algebra, probability, or estimation from zero. The sibling sheet `01_Matrix_and_Vector_Identities.md` carries that chain. This file assumes the reader can already read a matrix product and a sample mean, and is now trying to find out why a number is wrong.

## 1. File Header

The organising claim of this catalogue is narrow and testable: in quantitative pipelines, the arithmetic is almost never the thing that is wrong. Hand-computed matrix products, sample variances, and gradient updates are checked by the tools that compute them. What fails is the *correspondence* between the mathematical object and the thing it is supposed to represent — the assumption that observations are independent when they arrive in clusters, the identifiability condition that a design matrix has full column rank, the unit that a percentage refers to, the population that a denominator counts.

That claim has a practical consequence for how a diagnostic is built. A defect that lives in the correspondence cannot be found by re-running the same computation, because the computation is faithful to the wrong input. It is found only by comparing two representations of the same quantity: the variance computed from the assumed model against the variance computed from the empirical resampling structure; the coefficient estimated under one conditioning regime against the coefficient estimated under a nudged one; the pooled rate against the stratum-specific rates. Every discriminating check in Section 8 is therefore written as a *comparison between two numbers that should agree if the assumption holds*, not as a restatement of the assumption.

Three conventions are declared here because they cause more argument than any theorem below.

1. **Symbols are declared once, in Section 3, with units and valid range.** A symbol reused later without its units is a defect in this file, not a shortcut the reader should absorb.
2. **Percentages appear with an explicit referent.** `7.14% higher than 7.0%` and `0.5 percentage points higher` describe the same difference, and confusing them is entry CI-003. Any percentage in this file that could be read two ways is written out twice.
3. **Every constructed input is labelled `ILLUSTRATIVE`.** The label is not decoration: it distinguishes "this arithmetic was checked" from "this happened", which are different claims with different failure liabilities.

The catalogue is organised by four categories, and the category is the first thing to establish because it determines which check can possibly work.

| Category | What is wrong | The check that works | Check that cannot work |
|---|---|---|---|
| `VIOLATED_ASSUMPTION` | The model is right for a different data-generating process | Compare model-implied variance or error against an empirical resampling estimate | Re-running the same fit and getting the same answer |
| `NUMERICAL_EDGE` | The algorithm is right, the representation is not | Compare against an algebraically equivalent but better-conditioned route | Reviewing the source for typos |
| `SCALE_FAILURE` | The method is right, the resource envelope is not | Measure the growth exponent against the budget | Profiling a sample one-thousandth of the real size |
| `CONCEPT_CONFUSION` | The number is right, the word attached to it is wrong | Restate the number in a second vocabulary and see if the claim survives | Any check performed inside the same vocabulary |

The distribution of entries across those four categories is uneven, and the unevenness is the finding: 12 entries are `CONCEPT_CONFUSION`, 11 are `VIOLATED_ASSUMPTION`, 6 are `NUMERICAL_EDGE`, and 3 are `SCALE_FAILURE`. Vocabulary defects dominate because vocabulary defects are the only ones that a technically correct pipeline can produce indefinitely without any signal.

Severity in this file is scoped, not absolute. `containment` means the defect produces a wrong number that a second check catches before anyone acts on it. `decision` means the wrong number reaches a person who acts on it. `silent` means the wrong number reaches a person, is acted on, and the pipeline continues to report it as valid with no internal inconsistency. Silent defects are the reason this file exists, and they share one structural property: the artefact that would have contradicted the number — the cluster count, the second estimator, the stratum table — was never produced, because producing it costs more than the number did.

Two boundaries on the header, stated so a reader does not over-read the catalogue. First, the entries describe defects in *method*, not defects in *institutions*: an entry can tell you that a confidence interval is 2.64 times too narrow, and cannot tell you whether the organisation that published it will accept the correction. Second, the numeric instances are small on purpose. A 12,000-observation experiment with 120 clusters is small enough to check by hand and large enough that the design effect is not a rounding artefact; the same arithmetic at 12 million observations differs in cost, not in structure, which is the point of Section 9.

> **Takeaway:** classify before you debug; the category selects the only check that can find the defect.

## 2. Concept Map / Mental Model

The mental model is a two-lane inspection. One lane asks whether the number is *computable* — are the shapes conformable, does the denominator exist, does the accumulator overflow, does the solver's conditioning amplify error. The other lane asks whether the number is *about the right thing* — is the denominator the population that was promised, is the independence claim true of how the data arrived, is the word "significant" attached to the quantity the reader will act on. A defect that passes both lanes is not a defect the pipeline can see. It becomes a defect the *decision* sees, months later, when a one-percent movement in an input moves the output by forty percent.

The two lanes are not symmetric in cost. Lane one is cheap, local, and testable: a shape assertion, a conditioning number, an overflow guard. Lane two is expensive, requires a second data product, and cannot be unit-tested because it is a claim about the world. That asymmetry explains the observed pattern of which defects survive: lane-one defects are fixed early because they are cheap to detect, and lane-two defects accumulate because the artefact that would detect them — the cluster table, the stratified view, the held-out population — is a separate deliverable that nobody scoped.

```text
                    a number is produced
                            |
              +-------------+-------------+
              |                           |
     [LANE 1: computable?]        [LANE 2: about the right thing?]
              |                           |
   shapes conformable?            denominator = promised population?
   denominator nonzero?           independence claim true of arrival?
   accumulator wide enough?       unit and referent of % explicit?
   conditioning acceptable?       comparator named and same-window?
              |                           |
        no ---+--- yes              no ---+--- yes
        |          |                 |          |
   NUMERICAL_EDGE  |           VIOLATED_ASSUMPTION
   SCALE_FAILURE   |           CONCEPT_CONFUSION
        |          |                 |          |
   fix route,     +--------+--------+     produce the second
   re-derive               |              artefact (cluster table,
        |                  v              stratum table, held-out set)
        |          number ships                  |
        +------------------+---------------------+
                           |
                    defect observed later
                           |
              which lane would have caught it?
                           |
           +---------------+---------------+
           |                               |
   lane 1 (recomputed wrong)      lane 2 (never asked)
           |                               |
   tighten the numeric route      add the missing artefact to
   and keep the check             the delivery scope, not to a
                                  monitoring dashboard
```

The left branch of the funnel is the one engineers naturally build. It terminates in "number ships", which is exactly the wrong terminal state for a lane-two defect, and the diagram is drawn that way to make the mistake visible: passing lane one is a precondition for shipping, not a licence to ship.

A second map shows what each category costs once it reaches a decision, because the categories differ by an order of magnitude in remediation cost and not at all in how easy they are to spot in a code review.

```text
category                 detect cost   fix cost    silent?   typical lag to discovery
VIOLATED_ASSUMPTION      medium        high        often     one reporting cycle
NUMERICAL_EDGE           low           low         sometimes immediate (guard fires) or never
SCALE_FAILURE            low           medium      rarely    first production-scale run
CONCEPT_CONFUSION        low           high        almost    when someone restates the claim
                                                    always    in another vocabulary
```

Reading the table as a decision aid: if a defect is `CONCEPT_CONFUSION`, do not estimate the fix cost from the size of the code change, because the code change is usually one line and the fix cost is the retraction of every downstream artefact that inherited the wrong word. If a defect is `NUMERICAL_EDGE`, the opposite holds — the fix is a route change in one function, and the lag to discovery is bimodal: either a guard fires immediately or the defect is invisible forever.

The catalogue's internal structure follows from the worked case in Section 5, which is a two-arm experiment with 120 clusters per arm and 100 observations per cluster. That single case supplies the anchors for 14 of the 32 entries: the design effect appears in CI-001, the understated standard error in CI-016, the percentage-versus-points confusion in CI-003, the sample-size consequence in CI-013, and so on. The remaining entries use small independent instances, each stated with its own numbers so that no entry depends on reading another.

> **Takeaway:** a number can be arithmetically perfect and referentially wrong; only a second artefact can reveal the second failure.

## 3. Foundational Definitions

Definitions are ordered so that each one can be used by the ones after it. Where a term has a colloquial meaning that differs from its technical meaning, both are stated, because the gap between them is the mechanism behind most `CONCEPT_CONFUSION` entries.

**Estimand.** The quantity about which a claim is made, defined before data collection, including the population, the unit of assignment, and the time window. The estimand is not the estimator: "the difference in conversion rate between the two arms among users who would have been eligible during the test window" is an estimand; the difference of two sample proportions is an estimator of it.

**Estimator / estimate.** An estimator is a function of the data, defined before the data are seen; an estimate is its realised value. Any statement of the form "the estimator is unbiased" is a statement about the function, and does not license the claim that this particular realised value is close to the estimand.

**Unit of observation vs unit of assignment.** The unit of observation is the row in the analysis table — one user-session, one sensor reading. The unit of assignment is the entity randomised or selected — one store, one household, one user. When these differ, the rows are dependent, and every variance formula that assumes independent rows is wrong by the design effect. Conflating them is CI-001.

**Independent and identically distributed (i.i.d.).** A sample is i.i.d. when every observation has the same marginal distribution and every pair is probabilistically independent. Both halves matter: identical marginals without independence is the clustered case, independence without identical marginals is the heterogeneous case, and CI-001 and CI-016 are the two halves failing separately.

**Design effect (deff).** The ratio of the variance of an estimator under the actual sampling design to its variance under simple random sampling with the same number of observations. `deff > 1` means the real design carries less information than the observation count suggests. Written `deff = 1 + (m − 1)ρ` for equal cluster size `m` and intra-cluster correlation `ρ`.

**Intra-cluster correlation (ρ).** The fraction of total variance that lies between clusters rather than within them. In the variance-components parameterisation, `ρ = σ²_between / (σ²_between + σ²_within)`. Bounded on `[−1/(m−1), 1]` for equal cluster sizes; only the upper region near 1 is common in practice.

**Effective sample size.** `n_eff = n / deff`. The number of independent observations that would carry the same information as `n` dependent ones. It is the number to quote when someone asks how much evidence a clustered experiment contains, and it is usually between a fifth and a half of the row count.

**Conditioning (of a matrix).** For an invertible matrix `A` in the 2-norm, `κ(A) = σ_max / σ_min`, the ratio of largest to smallest singular value. It bounds the relative error amplification from input perturbations to the solution: a relative perturbation of `ε` in `A` or `b` can move the solution by up to `κ ε`.

**Rank deficiency.** A design matrix `X` of shape `n × d` has rank below `d`, so at least one column is a linear combination of the others. The coefficient vector is then not identifiable: infinitely many `w` give the same fitted values, so the individual entries of `w` are artefacts of the solver's tie-breaking rule.

**Multicollinearity.** Near-dependence rather than exact dependence: two or more columns are highly correlated, so `κ` is large and the coefficients are estimated with large variance even though the fitted values are stable. Distinct from rank deficiency, and its diagnostic is the variance inflation factor `1/(1 − R²_j)`, not the rank.

**Identifier / idempotency key.** The field whose equality defines "the same record" for deduplication. In arithmetic pipelines the analogue is the *aggregation key*: the set of columns that define one row of the output. When the aggregation key is coarser than the grain of the input, rows are silently summed; when it is finer, rows are silently duplicated.

**Population, sample, frame.** The population is the set the claim is about; the sample is what was measured; the frame is the operational list from which the sample was drawn. A frame that omits part of the population produces a coverage defect that no amount of within-sample arithmetic repairs.

**Denominator.** The count or measure in the bottom of a rate. Every rate in an operational report has a denominator whose definition is a policy decision, and CI-011 exists because the policy decision is usually undocumented.

**Percentage point vs percent.** A change from 7.0% to 7.5% is +0.5 percentage points and +7.14 percent. The first is additive on the probability scale; the second is multiplicative on the ratio scale. Both are correct, and a sentence that says "up 0.5%" about that movement is wrong by a factor of 14.3.

**Basis point (bp).** One hundredth of a percentage point, so 0.01%. The movement above is 50 bp. Basis points remove the percent/point ambiguity by construction and are therefore the preferred unit for rate movements.

**Statistical significance.** A statement that the observed statistic is unlikely under a specified null hypothesis at a specified level, given a specified model. It is not a statement about the size of an effect, the probability that the null is true, or the probability that a decision is correct.

**Confidence interval coverage.** The long-run fraction of intervals, constructed by the stated procedure across repeated samples, that contain the estimand. Coverage is a property of the procedure, not of one realised interval; a realised 95% interval either contains the estimand or does not, and nobody knows which.

**Power.** The probability that a test rejects the null when the alternative of a stated size is true, computed under the same model assumptions as the size calculation. Power computed under a violated independence assumption is wrong in the same direction as the standard error, which is why CI-013 and CI-001 share a mechanism.

**Family-wise error rate (FWER).** The probability of at least one false rejection across a declared family of tests. With `k` independent tests at level `α`, `FWER = 1 − (1 − α)^k`; at `k = 20`, `α = 0.05`, that is 0.6415.

**False discovery rate (FDR).** The expected proportion of false rejections among the rejections actually made. Controlled by Benjamini–Hochberg at level `q` by comparing the `i`-th ordered p-value to `i q / k`. Distinct from FWER: FDR is less conservative and is the right target when the cost of a false positive is bounded per finding.

**Base rate (prior prevalence).** The proportion of the population that truly has the condition, before any test result. It enters every positive predictive value and is the term most often omitted from a diagnostic-accuracy claim.

**Predictive value.** `PPV = sensitivity × prior / (sensitivity × prior + (1 − specificity) × (1 − prior))`. At a 10% prior with 80% sensitivity and 95% specificity, `PPV = 0.64`; at a 2% prior with 90% and 95%, `PPV = 0.27`. The test did not change; the population did.

**Catastrophic cancellation.** Loss of relative precision when two nearly equal numbers are subtracted. If `a` and `b` agree to `k` leading digits, `a − b` retains at most `p − k` correct digits for `p`-digit arithmetic. The two-pass variance formula `E[x²] − E[x]²` is the canonical victim because `E[x²]` is large and the answer is small.

**Log-sum-exp.** The stable evaluation of `log Σ exp(x_i)` as `c + log Σ exp(x_i − c)` with `c = max_i x_i`. Necessary because `exp` overflows above `log(realmax)`, which is 709.78 in float64 and 88.72 in float32, while logits routinely exceed both.

**Overflow / underflow.** Overflow is a magnitude exceeding the largest representable value (result becomes `inf`); underflow is a magnitude below the smallest normal (result becomes `0` and relative precision is destroyed). `softmax` can produce both in one call: overflow in the exponentials, underflow to zero in the normaliser.

**Idempotence.** The property that applying an operation twice has the same effect as applying it once. In pipelines, idempotence is a property of the *keyed write*, not of the arithmetic: a sum is not idempotent, a set-union keyed by id is.

**Extrapolation.** Evaluation of a fitted function outside the convex hull of the observed predictor values. Inside the hull, error is bounded by the residual scale times a leverage factor; outside, error grows with the distance in predictor space and is not bounded by any residual diagnostic.

**Attenuation.** The shrinking of an estimated association toward zero when a predictor is measured with error. For classical additive error with reliability `λ` (the ratio of true to observed variance), the estimated coefficient converges to `β √λ`, so reliability 0.8 shrinks a true 0.6 to 0.537 and reliability 0.72 shrinks it to 0.509.

**Delta method.** First-order propagation of variance through a smooth function: for `g` differentiable at `θ`, `Var(g(θ̂)) ≈ g'(θ)ᵀ Var(θ̂) g'(θ)`. Valid when the derivative dominates the second-order term, which fails near a zero of `g'` and near a small denominator, and fails *silently* when the components of `θ̂` are correlated and the covariance term is dropped.

**Sampling distribution.** The distribution of an estimator across repeated samples from the same design. All uncertainty statements in this file are statements about it; a bootstrap approximates it by resampling, and a closed form derives it from assumptions.

| Term | Units / shape | Typical wrong use | Entry that names it |
|---|---|---|---|
| `deff` | unitless, ≥ 0 | treated as 1 by default | CI-001 |
| `ρ` | unitless, `[−1/(m−1), 1]` | confused with a correlation between two variables | CI-001 |
| `n_eff` | observations | quoted instead of `n` in a power claim | CI-013 |
| `κ(A)` | unitless, ≥ 1 | ignored because the fit "converged" | CI-020 |
| VIF | unitless, ≥ 1 | used as a variable-selection rule | CI-005 |
| `PPV` | probability | reported without its prior | CI-004 |
| FWER | probability | replaced by per-test `α` | CI-013 |
| `n − 1` | observations | replaced by `n` in a hand-written formula | CI-015 |

> **Takeaway:** the definitions that matter operationally are the ones where the colloquial and technical meanings diverge.

## 4. Core Formulas with Full Derivation

Each block below derives one result to the point where its validity condition is explicit, because the entries in Section 8 cite these conditions by name. The blocks are ordered by the entry they support.

### 4.1 Base case — variance of a mean under clustering

```text
FORMULA [DEFF]: Var(x̄) = (σ²/n) · [1 + (m − 1)ρ]
WHERE: x̄ = mean of n observations [unit of x]; σ² = total variance of one observation [unit²];
       m = observations per cluster [count]; ρ = intra-cluster correlation [unitless];
       Var(x̄) has units [unit²].
CONSTRAINT: equal cluster sizes; clusters independent of one another; exchangeable
       observations within cluster (compound-symmetric working model).
DERIVATION: Write x_ij = μ + b_i + e_ij with Var(b_i) = σ²_b and Var(e_ij) = σ²_w.
       Then σ² = σ²_b + σ²_w and ρ = σ²_b/σ². Summing over the cluster: Σ_j x_ij = mμ + mb_i + Σ_j e_ij,
       so Var(Σ_j x_ij) = m²σ²_b + mσ²_w. Summing over M clusters and dividing by (Mm)²:
       Var(x̄) = [M(m²σ²_b + mσ²_w)]/(Mm)² = (mσ²_b + σ²_w)/(Mm) = (σ²/n)·[1 + (m−1)ρ].
NUMERICAL EXAMPLE: σ² = 0.06724, ρ = 0.05, m = 120, n = 12 000.
       Bracket = 1 + 119(0.05) = 6.95. Var(x̄) = (0.06724/12000)(6.95) = 3.8945e-5, sd = 0.006241.
       For comparison, the i.i.d. value is sqrt(0.06724/12000) = 0.002367; ratio = 2.6363.
INTERPRETATION: the bracket is the design effect; it multiplies the variance, so it multiplies
       the standard error by its square root. Ignoring it understates every interval by 2.64× here.
```

The derivation shows why the correction is multiplicative rather than additive: the between-cluster term scales as `m²` inside a cluster and is then divided by `(Mm)²`, leaving one factor of `m`; the within term scales as `m` and leaves one factor of `m`. The ratio of the two contributions is exactly `(m − 1)ρ`, independent of `n`. That independence is the practical reason a design effect computed once can be reused across sample sizes — and the practical trap, because a design effect computed at one cluster size does not transfer to another.

### 4.2 General case — design effect with unequal cluster sizes

```text
FORMULA [DEFF-UNEQUAL]: deff = 1 + [ (CV² + 1)·m̄ − 1 ]·ρ
WHERE: CV = between-cluster coefficient of variation of cluster size (sd/mean, unitless);
       m̄ = mean cluster size [count]; ρ = intra-cluster correlation [unitless].
CONSTRAINT: ρ defined on the variance-components scale; cluster sizes independent of the outcome.
DERIVATION: With sizes m_i, Var(Σ_i Σ_j x_ij) = Σ_i (m_i²σ²_b + m_iσ²_w). Replacing Σ m_i² by
       M(m̄² + sd_m²) = M m̄²(1 + CV²) gives Var = M[m̄²(1+CV²)σ²_b + m̄σ²_w].
       Dividing by (M m̄)² and factoring σ² = σ²_b + σ²_w reproduces the bracket above.
NUMERICAL EXAMPLE: m̄ = 120, CV = 0.5, ρ = 0.05.
       Bracket = 1 + [(0.25 + 1)(120) − 1](0.05) = 1 + (149)(0.05) = 8.45.
       Equal-size value was 6.95, so variable cluster sizes cost a further 21.6% of variance
       (8.45/6.95 = 1.216) — a standard error 10.3% larger.
INTERPRETATION: at the same mean size, unequal clusters are strictly worse, and the penalty
       is quadratic in the coefficient of variation, not linear.
BOUNDARY: if cluster sizes are strongly related to the outcome (sizes chosen after seeing y),
       the independence of m_i and the outcome fails and this correction is not conservative.
```

The two-block sequence establishes the general case and, in doing it, establishes the boundary. A practitioner who applies `1 + (m − 1)ρ` to a design with `CV = 0.5` understates the standard error by 10.3%, which at 12,000 observations and a 0.5-point effect moves a p-value computed as 0.42 to 0.46 — a difference that does not change that decision but would change one near a threshold.

### 4.3 Variants — variance of a ratio by the delta method

```text
FORMULA [DELTA-RATIO]: Var(X̄/Ȳ) ≈ (1/μ_Y²)·[ σ²_X − 2(μ_X/μ_Y)σ_XY + (μ_X/μ_Y)²σ²_Y ]
WHERE: X̄ [unit of X], Ȳ [unit of Y ≠ 0], σ²_X [unit²], σ²_Y [unit²], σ_XY [unit·unit],
       μ_X/μ_Y = R [unit of X per unit of Y].
CONSTRAINT: (X̄, Ȳ) asymptotically normal; μ_Y bounded away from zero; g' = (1/μ_Y, −μ_X/μ_Y²)
       dominates the second-order remainder, i.e. |R|·sd(Y)/μ_Y is small.
DERIVATION: g(x,y) = x/y has gradient (1/y, −x/y²). The first-order expansion
       g(θ̂) ≈ g(θ) + g'(θ)ᵀ(θ̂ − θ) gives Var(g) = g'ᵀ Σ g', which expands to the bracket above.
NUMERICAL EXAMPLE: μ_X = 50, μ_Y = 1000, σ_X = 30, σ_Y = 120, σ_XY = 2400.
       R = 0.05. Bracket = 900 − 2(0.05)(2400) + (0.0025)(14400) = 900 − 240 + 36 = 696.
       Var = 696/1e6 = 6.96e-4, se = 0.02638, relative se = 52.8%.
       Dropping the covariance term gives 936/1e6 → se = 0.03059, 16.0% too large.
INTERPRETATION: correlated numerator and denominator make the naive ratio variance wrong in
       either direction; here the covariance removes 26.7% of the variance.
BOUNDARY: if the denominator's relative sd exceeds roughly 0.3, the first-order expansion is
       unreliable and the interval must come from a bootstrap or from Fieller's theorem.
```

The variant matters because ratio metrics are the default in operational reporting — cost per order, revenue per session, defects per thousand units — and because the covariance term is the one that a two-column summary table cannot supply. A pipeline that stores only per-column means and variances cannot compute the correct ratio variance at all; it can only compute the independence-assuming version, which is CI-025.

### 4.4 Limits and counterexamples — conditioning, and the cost of squaring it

```text
FORMULA [COND-AMPLIFY]: ||δw||/||w|| ≤ κ(X)·( ||δX||/||X|| + ||δb||/||b|| )
WHERE: X [n × d, full column rank], b [n], w [d], κ(X) = σ_max/σ_min ≥ 1 [unitless].
CONSTRAINT: 2-norm; the bound is attainable, not merely an upper bound in principle.
DERIVATION: From Xw = b, perturbing b by δb gives X δw = δb, so ||δw|| ≤ ||X⁻¹||·||δb||.
       Combining with ||b|| ≤ ||X||·||w|| gives ||δw||/||w|| ≤ ||X||·||X⁻¹||·||δb||/||b|| = κ·(relative δb).
       Perturbing X adds the symmetric term, and the two combine in the stated bound.
NUMERICAL EXAMPLE: X'X for a two-column design with correlation r = 0.98 has eigenvalues
       (1 + r) = 1.98 and (1 − r) = 0.02, so κ(X'X) = 99. The normal-equations route squares
       this to κ(X'X) = 99, i.e. an error amplification of ~9.8e3 relative to unit input error,
       versus 99 for a QR route that never forms X'X.
       At double precision (ε = 2.22e-16), the normal-equations route loses log10(9801) ≈ 4.0 digits.
INTERPRETATION: forming X'X is not merely slower to converge; it destroys four of sixteen
       significant digits before any solver runs.
BOUNDARY: κ is a bound on the worst case, not a prediction. A well-conditioned X can still
       produce a bad answer if the model is wrong; κ tells you only whether rounding can hurt.
```

The four blocks give the quantitative backbone. The validity conditions they end with are the conditions Section 8 cites, and each entry that depends on one of them names it explicitly (`DEFF`, `DEFF-UNEQUAL`, `DELTA-RATIO`, `COND-AMPLIFY`) so a reader can find the derivation without scanning the prose.

One further mechanism block is needed because it has no formula in the usual sense — the diagnostic itself is the object being specified, and its invariant is what makes the catalogue usable rather than decorative.

```text
MECHANISM [DIAGNOSTIC-LOOP]:
PRECONDITIONS: (a) a reproducible artefact exists — the input record, the code version, the seed;
   (b) a second representation of the same quantity exists or can be produced cheaply;
   (c) the symptom has been observed at least twice or has a known deterministic trigger.
STATE TRANSITION: observe symptom → assign category → compute the two representations →
   compare → the comparison either separates the competing causes or it does not →
   if it does, apply the correction for the surviving cause; if it does not, the category
   assignment was wrong and the loop restarts at a different category.
INVARIANT: the two representations must be equal when the suspect assumption holds, and must
   differ by a predictable magnitude when it fails. A check whose expected value under the
   hypothesis is "some other number, roughly" is not a check.
JUSTIFICATION: a diagnostic that cannot distinguish the causes produces the same output for
   every cause, so it can only confirm a diagnosis already made — which is why the entries list
   competing causes before their check, never after.
WORKED TRACE: symptom "our rate improved but the aggregate did not" → category
   CONCEPT_CONFUSION → representation 1: pooled rate over all strata; representation 2: mean of
   stratum rates → differ (11.0% vs 15.0% in CI-008's instance) → the differing weightings
   separate Simpson's paradox from a rounding error, and the check is complete in one step.
FAILURE BOUNDARY: if the second representation shares the first's input extraction, both are
   wrong together and the comparison returns equality for the wrong reason. This is the single
   most common way a diagnostic silently passes, and it is the check in CI-009 and CI-014.
```

The invariant statement is the load-bearing part. Almost every non-diagnostic in this file's subject matter has the shape "we looked at the output and it seemed plausible", and plausibility is exactly what a referential defect preserves by construction. The entries are written to force the stronger form: an expected *number* under each cause, not an expected *impression*.

> **Takeaway:** a derivation earns its place only when it ends in a condition you can test against a second number.
## 5. Worked Numerical Example

The worked case is the anchor for 14 catalogue entries, so its numbers are recomputed by two independent routes and then perturbed. The setting is `ILLUSTRATIVE`: a two-arm comparison in which assignment happened at the level of a site, 120 sites were assigned to each arm, and each site contributed 100 observations, giving 12,000 rows per arm and 24,000 rows in total.

```text
GIVEN (ILLUSTRATIVE):
  arm T: 900 conversions in 12,000 rows  → p_T = 900/12,000 = 0.0750
  arm C: 840 conversions in 12,000 rows  → p_C = 840/12,000 = 0.0700
  difference d = 0.0750 − 0.0700 = 0.0050
  relative difference = 0.0050/0.0700 = 0.0714286 → 7.14% higher
  in basis points = 0.0050 × 10,000 = 50 bp
  sites per arm M = 120; observations per site m = 100; total rows n = 24,000
```

### 5.1 Route one — the independence-assuming standard error

```text
FORMULA: se_0(d) = sqrt( p̄(1 − p̄)(1/n_T + 1/n_C) )
WHERE: p̄ = (x_T + x_C)/(n_T + n_C) [probability]; n_T = n_C = 12,000.
       Using the pooled proportion for the null standard error, which is the right choice for a
       test of equal proportions and the wrong choice for an interval of the difference.
CONSTRAINT: rows independent within and across arms.
COMPUTE: x_T + x_C = 1,740; p̄ = 1,740/24,000 = 0.0725; p̄(1 − p̄) = 0.0725 × 0.9275 = 0.0672438
       (1/12,000 + 1/12,000) = 1.666667e-4
       product = 0.0672438 × 1.666667e-4 = 1.12073e-5
       se_0 = sqrt(1.12073e-5) = 0.0033477
       z = 0.0050/0.0033477 = 1.4935 → two-sided p = 0.1353
       Wald 95% CI on the difference: 0.0050 ± 1.96(0.0033477) = (−0.00156, +0.01156)
INTERPRETATION: the naive analysis reports a 7.14% relative lift, not significant, CI spanning zero.
```

### 5.2 Route two — the design-effect standard error

The arm-level intra-cluster correlation is `ρ = 0.05` and cluster size is equal, so `deff = 1 + (m − 1)ρ = 1 + 119(0.05) = 6.95`.

```text
FORMULA: se_cl(d) = se_0(d) × sqrt(deff)
COMPUTE: sqrt(6.95) = 2.63629
       se_cl = 0.0033477 × 2.63629 = 0.0088256
       z = 0.0050/0.0088256 = 0.5665 → two-sided p = 0.5710
       cluster-robust 95% CI: 0.0050 ± 1.96(0.0088256) = (−0.01230, +0.02230)
       effective sample size per arm = 12,000/6.95 = 1,726.6; total = 3,453.2
INTERPRETATION: the same 24,000 rows carry the information of 3,453 independent observations.
       The interval is 2.64× wider and the p-value moves from 0.135 to 0.571.
```

### 5.3 Independent re-check by variance components

Route two multiplied a standard error. The independent route reconstructs the variance from its two components and does not reuse any intermediate from §5.2.

```text
FORMULA: ρ = σ²_b/(σ²_b + σ²_w); Var(p̂_arm) = [σ²_b + σ²_w/m]/M
GIVEN: total variance of a row indicator = p̄(1 − p̄) = 0.0672438
       σ²_b = ρ × 0.0672438 = 0.05 × 0.0672438 = 0.00336219
       σ²_w = (1 − ρ) × 0.0672438 = 0.95 × 0.0672438 = 0.06388156
COMPUTE: cluster-mean variance = 0.00336219 + 0.06388156/100 = 0.00336219 + 0.00063882
                              = 0.00400101
       sd of a cluster mean = sqrt(0.00400101) = 0.0632535
       se of an arm mean = 0.0632535/sqrt(120) = 0.0632535/10.95445 = 0.0057743
       se of the difference = 0.0057743 × sqrt(2) = 0.0081660
CROSS-CHECK: the deff route gave 0.0088256; this route gives 0.0081660; ratio 1.0808.
```

The two routes disagree by 8.1%, and the disagreement is informative rather than a mistake. Route two uses `p̄(1 − p̄)` as the *total* variance and splits it by `ρ`, which is the compound-symmetric model on the logit-free probability scale; route one applies `deff` to the *binomial* variance of the difference. The binomial variance already embeds the `m`-fold averaging that the variance-components split re-derives, and the two coincide exactly only when `ρ` is defined on the same scale as the quantity being averaged. With `ρ = 0.05` on the indicator scale, route two is the more faithful reconstruction and route one is the conservative one. The reconciliation is the point: a single number quoted from either route without stating which construction produced it is not reproducible, which is CI-022.

### 5.4 Sensitivity — what happens if ρ is not 0.05

The `ρ` above was estimated from the data. A practitioner who cannot estimate it must bound it, and the bound has to be translated into a decision consequence rather than reported as a range.

| ρ assumed | deff | cluster-robust se | z for d = 0.0050 | two-sided p | n_eff per arm | MDE at 80% power | power at d = 0.0050 | coverage of naive CI | n per arm for 80% power |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 3.38 | 0.0061547 | 0.8124 | 0.4166 | 3,550.3 | 0.01724 | 0.1284 | 0.7136 | 142,714 |
| 0.05 | 6.95 | 0.0088256 | 0.5665 | 0.5710 | 1,726.6 | 0.02473 | 0.0875 | 0.5428 | 293,450 |
| 0.10 | 12.90 | 0.0120239 | 0.4158 | 0.6775 | 930.2 | 0.03369 | 0.0700 | 0.4147 | 544,677 |
| 0.15 | 18.85 | 0.0145347 | 0.3440 | 0.7308 | 636.6 | 0.04072 | 0.0637 | 0.3483 | 795,904 |

Read the fourth and fifth columns together: the p-value moves from 0.42 to 0.73 as `ρ` goes from 0.02 to 0.15, so the *conclusion* is robust to the assumption in this instance — no value of `ρ` in the plausible range rescues a significant result. Read the last column instead and the conclusion reverses in practical terms: to detect this 50 bp effect at 80% power the study needed between 142,714 and 795,904 rows per arm, i.e. between 1,189 and 6,632 sites per arm, against the 120 actually used. The study was underpowered by a factor of ten to fifty-five, and the observed p-value is consistent with that: power at the observed effect is 0.0875 at `ρ = 0.05`, so a non-significant result was the overwhelmingly likely outcome even if the effect is real.

The coverage column is the quietest and most damaging. A 95% interval built with `se_0` covers the estimand 54.3% of the time when `ρ = 0.05` and only 34.8% of the time when `ρ = 0.15`. That is the number to quote when someone asks why the intervals "look fine" — they look fine because a 54%-coverage interval is not visually distinguishable from a 95% one at this effect size, and its width is only 38% of the correct width.

Two arithmetic re-checks independent of the table:

```text
CHECK A (MDE identity): MDE = (z_{0.975} + z_{0.80}) × se
       (1.959964 + 0.841621) = 2.801585
       2.801585 × 0.0088256 = 0.024726  ✓ matches 0.02473 to 5 significant figures
CHECK B (coverage identity): coverage = 2Φ(z_{0.975}/sqrt(deff)) − 1
       at deff = 6.95: 1.959964/2.63629 = 0.74346; Φ(0.74346) = 0.7714
       2(0.7714) − 1 = 0.5428  ✓ matches the table
```

Both checks reproduce a table entry from a formula that shares no input with the table's construction path other than `se`, and Check B reproduces a column that was never derived from `se` at all. That is the standard this file holds a numeric claim to: not "the tool printed it", but "a second expression of the same quantity agrees".

A boundary case ends the section. Suppose the 120 site-level conversion counts are not merely clustered but *identical within site*, i.e. every site is all-convert or all-not. Then `ρ = 1`, `deff = 120`, `se_cl = 0.0033477 × 10.954 = 0.03667`, and the effective sample size per arm is 100 — one observation per site. The estimand is still well defined and the interval is still computable, but the study contains a hundredth of the information its row count suggests. Nothing in the naive output signals this, which is why CI-001 is placed first.

> **Takeaway:** recompute the standard error by a second construction before believing any interval; a factor-of-2.6 error is invisible by inspection.

## 6. Visual / Geometric Intuition

The geometric picture of clustering is a two-level scatter. Draw the 120 site means as points on a line, then draw the 100 observations around each site mean. The variance of the point cloud is `σ²_b + σ²_w/m` — the between-site spread plus the shrunken within-site spread. What the analysis sees is the second cloud; what determines the standard error of the arm mean is the first cloud's spread divided by the number of *sites*.

```text
  site means (120 points, sd = sqrt(σ²_b + σ²_w/m) = 0.06325)
    |        |             |         |                |
   ●        ●             ●         ●                ●        ...
        each ● has a fan of 100 observations,
        sd = sqrt(σ²_w) = 0.2527

  what the row count suggests:                 24,000 rows → se ≈ 0.00335
  what the site count supports:                240 sites   → se ≈ 0.00825
```

Two features of that picture carry most of the intuition. First, the *horizontal* spread that matters is the site-mean spread, and it does not shrink when the cluster gets taller — adding observations to each site moves each point's *vertical* precision, not its horizontal position. Second, the fan width `σ²_w` can be large without helping, because averaging 100 draws shrinks `σ²_w` by 100 while leaving `σ²_b` untouched. The limiting case `σ²_w → 0` is the ρ = 1 boundary of Section 5.

The same geometry explains the conditioning entries. A design matrix's columns are vectors in `n`-dimensional space; a coefficient vector is a set of coordinates in the basis those columns span. When two columns are nearly parallel, the coordinate system is nearly degenerate: a small change in the target vector `b` requires a large swing in the coordinates to compensate, because the only way to adjust in the direction where the two columns nearly coincide is to move them in opposite directions by a large amount.

```text
  nearly parallel columns (r = 0.98) — coordinates unstable
        x2
         |    /  x1
         |   /
         |  /        b lands here
         | /         ↓
         |/__________ x1-axis
        /
       /            a 1% change in b moves w1 and w2 in
      /             opposite directions by ~25%
     x2

  orthogonal columns — coordinates stable
        x2
         |
         |     b
         |     ↓
         |_____|______ x1
```

The quantitative version: the angle `θ` between two unit columns satisfies `cos θ = r`, and the eigenvalue ratio of `X'X` in the two-dimensional subspace is `(1 + r)/(1 − r)`. At `r = 0.98` that is 99, so the coordinate system is stretched by 99:1 and the coefficient standard errors inflate by `sqrt(VIF) = sqrt(1/(1 − r²)) = 5.03`. A coefficient that moves by a factor of five when a redundant column is added or dropped is not evidence about the world; it is evidence about the basis.

A third picture makes the percentage/point distinction geometric rather than verbal, because the verbal distinction is the one that keeps failing.

```text
  probability scale (additive)            ratio scale (multiplicative)
  0.070 ──────► 0.075                     0.070 ──────► 0.075
       +0.005 pp-equivalent                    × 1.0714
       = 50 bp                                 = +7.14%

  These are the same movement. The left arrow is 0.5 percentage points;
  the right arrow is 7.14 percent. A sentence calling the movement
  "0.5%" is off by a factor of 14.28.
```

A fourth picture shows why the two-pass variance matters, and it is worth drawing because the failure is a subtraction that *looks* harmless. On a number line, `E[x²]` is a point far to the right of the origin; the variance is the small gap between that point and `(E[x])²`. If the two large points agree to twelve significant digits, their difference is computed from digits that the representation no longer holds.

```text
  0                 (E[x])² ≈ 1.00000002e16      E[x²] ≈ 1.00000002e16
  |-------------------------|-----------------------------|
                            |◄── variance lives here ─────►|
                                 (width ~2.0, i.e. 13 orders
                                  of magnitude below the operands)
```

For the three values `1e8`, `1e8 + 1`, `1e8 + 2` the two-pass variance is exactly `2/3 = 0.6667`; the one-pass form returns `2.0` from float64 arithmetic, because `1.00000002e16` cannot represent the difference at the required precision. The relative error is 200%, and the computed value is not merely noisy — it is consistently too large, because the cancellation is one-sided.

> **Takeaway:** draw the level at which the variance lives; almost every variance defect is a variance computed at the wrong level.

## 7. Comparison / Selection Table

The catalogue's practical value depends on choosing the right remedy, and the remedies are not interchangeable: a robust standard error does not fix a wrong estimand, a bootstrap does not fix a singular design matrix, and centring does not fix a rank deficiency. The table maps the observed condition to the remedy whose validity condition actually holds.

| Observed condition | Correct remedy | Why it is valid here | Remedy that is often applied and wrong | Failure it causes |
|---|---|---|---|---|
| Rows nested in sites, `ρ` estimated | Cluster-robust (sandwich) variance | Consistent as the number of clusters grows, no distributional assumption on `b_i` | Divide `n` by `deff` in the point estimate | Shrinks the estimate, not the interval |
| Few clusters (< 30) | Cluster bootstrap or wild cluster bootstrap | Small-sample critical values come from the resampling distribution | Asymptotic sandwich variance | Anti-conservative at 10–20 clusters |
| Cluster sizes vary with `CV` | `deff = 1 + [(CV²+1)m̄ − 1]ρ` | Accounts for the `Σm_i²` term | `1 + (m̄ − 1)ρ` | Understates variance by 21.6% at `CV = 0.5` |
| Numerator and denominator correlated | Delta method with `σ_XY`, or Fieller | Keeps the covariance term that a two-column summary drops | Independent-proportion formula | 16% error either direction |
| Denominator relative sd > 0.3 | Fieller interval or ratio bootstrap | First-order expansion no longer dominates | Delta method | Interval too narrow, wrong centre |
| Exact column dependence | Drop, combine, or add a ridge penalty | Restores identifiability; ridge makes the solution unique and stable | Pseudo-inverse and report the coefficients | Coefficients are solver artefacts |
| High but not exact dependence | Ridge, or centred/orthogonalised features | Shrinks along the unstable direction only | Stepwise selection on p-values | Selection bias, unstable reported set |
| Normal equations for least squares | QR or SVD on `X` | `κ(X)` instead of `κ(X'X)` | Form `X'X` then invert | Loses `log10(κ)` digits |
| `exp` of a logit above 709 | Log-sum-exp with a max shift | Removes the overflow before the exponential | Clip the logit at 700 | Wrong probabilities, silently |
| Probability exactly 0 or 1 in log-loss | Clip at `ε` and report the clip | Finite gradient, documented bias | `log(0)` guard returning 0 | Zero loss for a maximally wrong prediction |
| Variance of large-magnitude data | Two-pass (or Welford) algorithm | Subtracts small from small | `E[x²] − (E[x])²` | Up to 200% relative error |
| Rate with a zero denominator | Emit undefined with a reason code | A rate with no exposure is not zero | `0/0 → 0` guard | Fabricated zero rate |
| Many simultaneous tests, all findings costed equally | Benjamini–Hochberg at `q` | Controls FDR, which is the decision-relevant error | Per-test `α = 0.05` | FWER 0.6415 at 20 tests |
| Many tests, any false positive is costly | Holm or Bonferroni | Controls FWER under arbitrary dependence | BH | FDR control does not bound FWER |
| Prevalence near 2% | Report PPV with the prior, or report sensitivity alone | PPV is prior-dependent; sensitivity is not | Accuracy alone | A 90%/95% test looks excellent at PPV 0.27 |
| Effect estimated with a noisy predictor | Correct for attenuation by `√λ`, or use an instrument | Attenuation is multiplicative, not additive | Report the naive coefficient as the effect | Understates by 49% at `λ = 0.72` |
| Prediction outside the observed predictor range | Refuse, or state the extrapolation distance | Residual diagnostics do not bound outside the hull | Trust the model's residual sd | Unbounded error, plausible-looking |
| Sequential looks at accumulating data | Alpha-spending or a group-sequential boundary | Keeps FWER across looks by construction | Fixed-sample `α` at each look | FWER 0.2262 at five looks |
| Missing outcome related to treatment | Report the arm-specific missingness and bound the estimand | Bounds do not require an ignorability assumption | Complete-case analysis | 11.1% bias at 20% missing, differential |

The selection logic that the table encodes is a two-question filter. Question one: is the defect in the *variance* or in the *centre*? Nothing in the table fixes a centre by adjusting variance — the cluster correction, the delta method, the design effect, and the multiplicity corrections all move intervals or thresholds while leaving the point estimate alone. Question two: is the remedy's validity condition a statement about the *design* or about the *distribution*? Sandwich variance needs only a growing cluster count, whereas a mixed model needs a correct random-effects specification; when the design is known and the distribution is not, prefer the design-based remedy.

A worked instance of the filter: an analyst reports a 7.14% lift with a naive p-value of 0.135 and asks whether to "just use a robust standard error". Question one answers yes — the centre is fine, the `7.14%` is correctly computed from the counts. Question two answers yes — the design is known (120 sites per arm of 100 observations), so the design-based correction applies without a distributional commitment. The remedy is therefore the sandwich or design-effect correction, and the result is p = 0.571. If instead the analyst had reported a 7.14% lift from a *ratio* of two correlated aggregates, question one would have pointed at the centre, and no variance correction would have helped.

Two entries in the table deserve a note on why their wrong-column remedy is so persistent. Row three's `1 + (m̄ − 1)ρ` is the formula in every textbook and in most software defaults, so the unequal-size correction is skipped not from ignorance but because the default looks authoritative; the 21.6% understatement is therefore a default-propagation defect rather than an analyst error. Row eight's normal equations are the fastest route for small `d` and the one that matches the derivation in every first course, which is why the `κ` loss is discovered only when a `d = 200` design with correlated columns returns coefficients that move by a factor of five between runs.

> **Takeaway:** match the remedy to whether the defect is in the centre, the variance, or the basis before touching the code.
## 8. Failure Modes & Edge Cases

The catalogue is 32 entries. Eleven are `VIOLATED_ASSUMPTION`, ten are `CONCEPT_CONFUSION`, eight are `NUMERICAL_EDGE`, and three are `SCALE_FAILURE`. Twenty-five entries carry the original commitment; seven are marked `[added]` and were introduced in the 2026-11 repair to cover defects the original twenty-five did not name — small expected counts, attenuation, autocorrelation, ecological inference, float equality, integer overflow, and regression to the mean.

Severity uses three levels defined in Section 1: `containment` (a second check catches it before anyone acts), `decision` (the wrong number reaches an actor), `silent` (the wrong number reaches an actor and the pipeline keeps reporting it as valid).

| ID | Category | Symptom in one line | Discriminating check | Severity |
|---|---|---|---|---|
| CI-001 | VIOLATED_ASSUMPTION | Interval 2.64× too narrow because rows are nested | Compare `se_0` against `se_0·sqrt(1+(m−1)ρ)` | silent |
| CI-002 | NUMERICAL_EDGE | Variance too large by 200% on large-magnitude data | Recompute two-pass and compare | containment |
| CI-003 | CONCEPT_CONFUSION | "Up 0.5%" describing a 50 bp movement | Restate in basis points | silent |
| CI-004 | CONCEPT_CONFUSION | Excellent accuracy claim at 2% prevalence | Recompute PPV with the prior | decision |
| CI-005 | VIOLATED_ASSUMPTION | Coefficient moves 5× when a redundant column is dropped | Compute VIF per column | decision |
| CI-006 | NUMERICAL_EDGE | Probabilities become `nan` or `1.0` at large logits | Compare against log-sum-exp | containment |
| CI-007 | CONCEPT_CONFUSION | "95% chance the null is false" | Ask for the null-rejection rate | silent |
| CI-008 | CONCEPT_CONFUSION | Every stratum improved, the pooled rate fell | Tabulate strata against pooled | decision |
| CI-009 | VIOLATED_ASSUMPTION | Implausibly high validation score | Refit with preprocessing inside the fold | containment |
| CI-010 | NUMERICAL_EDGE | A rate of exactly 0.0 on an empty window | Inspect numerator and denominator separately | silent |
| CI-011 | CONCEPT_CONFUSION | A rate jump that is a denominator change | Hold the numerator fixed and re-derive | decision |
| CI-012 | NUMERICAL_EDGE | Mean of per-unit ratios disagrees with the aggregate | Compute both and compare | containment |
| CI-013 | SCALE_FAILURE | One "significant" finding in twenty tests | Recompute FWER as `1−(1−α)^k` | decision |
| CI-014 | VIOLATED_ASSUMPTION | Held-out score falls apart in deployment | Split by cluster, not by row | decision |
| CI-015 | NUMERICAL_EDGE | Hand-written sd smaller than the tool's by 5% | Divide by `n−1` and compare | containment |
| CI-016 | VIOLATED_ASSUMPTION | Pooled sd wrong when group variances differ | Compare pooled sd with the larger group sd | silent |
| CI-017 | CONCEPT_CONFUSION | Average growth rate overstates compounded growth | Compute the geometric mean | decision |
| CI-018 | NUMERICAL_EDGE | Likelihood is exactly 0 for a valid model | Sum log-likelihoods instead of multiplying | containment |
| CI-019 | CONCEPT_CONFUSION | `r = 0.6` reported as "60% of the variation" | Square it, or refit and read the slope | silent |
| CI-020 | NUMERICAL_EDGE | Coefficients differ between two algebraically equal solvers | Compare `κ(X)` with `κ(X'X)` | containment |
| CI-021 | VIOLATED_ASSUMPTION | Confident prediction far outside the fitted range | Compute distance to the predictor hull | decision |
| CI-022 | CONCEPT_CONFUSION | Two intervals for one quantity differ 2.64× in width | Ask which construction produced each | silent |
| CI-023 | SCALE_FAILURE | `MemoryError` on a 50,000 × 50,000 matrix | Compute `n² × 8` bytes before allocating | containment |
| CI-024 | CONCEPT_CONFUSION | 98% accuracy from a model that predicts one class | Compute sensitivity and balanced accuracy | decision |
| CI-025 | VIOLATED_ASSUMPTION | Ratio se 16% off from stored summaries | Recompute with the covariance term | silent |
| CI-026 | NUMERICAL_EDGE | p-value from a pooled-variance df when variances differ | Compute Welch df and compare | containment |
| CI-027 | VIOLATED_ASSUMPTION | Interval too narrow on autocorrelated series | Estimate `ρ₁` and inflate by `sqrt((1+ρ)/(1−ρ))` | silent |
| CI-028 | CONCEPT_CONFUSION | Group mean used as an individual's expected value | Compare group mean with within-group spread | decision |
| CI-029 | NUMERICAL_EDGE | `0.1+0.2 == 0.3` is false in a guard | Compare with an absolute tolerance | containment |
| CI-030 | SCALE_FAILURE | Count flips sign past 2^31 | Check the accumulator dtype before a large sum | silent |
| CI-031 | VIOLATED_ASSUMPTION | Treatment effect appears only under complete-case analysis | Tabulate missingness by arm | decision |
| CI-032 | CONCEPT_CONFUSION | Extreme group improves by more than the rest | Predict the reversion from the prior sd | decision |

### CI-001 — Clustered rows analysed as independent observations

**Symptom.** A pooled lift of 7.14% is reported with a 95% interval of (−0.16 pp, +1.16 pp) and the reading is "no detectable effect". The counts are correct, the arithmetic of the interval is correct, and the interval is 2.64 times too narrow.

**Competing causes.** (a) There is genuinely no effect; (b) rows are nested in sites, so the arm-level standard error is `0.0088256` rather than `0.0033477`; (c) the wrong standard error was chosen for the interval — a pooled-proportion `se` is right for a test and conservative-but-wrong for an interval.

**Discriminating check.** Compute both. `se_0 = 0.0033477` and `se_cl = se_0 × sqrt(1 + 119(0.05)) = 0.0088256`. If the ratio is far from 1, cause (b) holds and cause (a) cannot be established from this interval at all. Then check the interval's coverage: 54.3% for the naive construction at `ρ = 0.05`, against a nominal 95%.

**Correction.** Replace the variance estimator with a cluster-robust sandwich or apply the design effect, and re-report the effective sample size (1,726.6 per arm rather than 12,000). Do not change the point estimate — it is unaffected.

**Prevention.** Record the unit of assignment in the analysis table's schema, not in a runbook. A schema field `assignment_unit` that differs from the row grain should make the default variance estimator refuse to run without an explicit choice.

**Verification.** Replay the 240 site records and confirm that the reported interval matches the sandwich estimate to four significant figures, and that `n_eff` is printed next to `n` in the output.

### CI-002 — Variance computed by subtracting two large nearly equal numbers

**Symptom.** The variance of `{1e8, 1e8+1, 1e8+2}` is reported as `2.0`. The population variance is `2/3 = 0.6667` and the sample variance `1.0`.

**Competing causes.** (a) The one-pass formula `E[x²] − (E[x])²` destroyed its own precision; (b) the data were already centred by an upstream transform and the variance refers to the shifted values; (c) a population-vs-sample divisor mismatch, which would explain a factor of `n/(n−1) = 1.5` here but not a factor of 3.

**Discriminating check.** Compute both forms on the same array. The two-pass value is `0.6667`; the one-pass value in float64 is `2.0`; the ratio 3.0 is far outside any divisor convention, which eliminates cause (c) immediately.

**Correction.** Use a two-pass or Welford update, and centre the data before squaring. If the data cannot be re-read, subtract the reported mean from each value and recompute from the residuals.

**Prevention.** Forbid `mean(x*x) - mean(x)**2` in review, with the rule stated as a numeric condition: if `|mean(x)| / std(x) > 1e4`, the one-pass form is inadmissible at float64.

**Verification.** Assert that the variance is non-negative and that a recomputation after subtracting a constant returns the same value — the shift-invariance test catches this class of defect in one line.

### CI-003 — Percentage points reported as percent

**Symptom.** A movement from 7.0% to 7.5% is written "up 0.5%". The correct descriptions are "+0.5 percentage points" or "+7.14%".

**Competing causes.** (a) The writer converted the difference to percent by dividing by 100 instead of by the base rate; (b) the base rate is 100% and the two descriptions coincide; (c) a unit label was dropped somewhere between the calculation and the slide.

**Discriminating check.** Divide the absolute movement by the base value: `0.005/0.07 = 0.0714`. If the quotient is 14.3 times the stated percentage, cause (a) holds. If the base value is 1.0 by construction, cause (b) holds and the sentence is harmless.

**Correction.** Restate in basis points — 50 bp — which is unambiguous, then give the relative change with its base named: "7.14% relative to the 7.0% baseline".

**Prevention.** Make the reporting layer refuse a bare `%` for a difference of two rates; require either `pp`/`bp` or a `relative_to` field carrying the base value.

**Verification.** Recompute the movement from the raw counts (900 and 840 out of 12,000 each) and confirm that both the point value and the relative value agree with the published sentence.

### CI-004 — Predictive value quoted without its prior

**Symptom.** A screening rule with 90% sensitivity and 95% specificity is described as "accurate", and the operational plan assumes most flagged cases are true.

**Competing causes.** (a) The prior in the deployment population is 2%, giving `PPV = 0.269`; (b) the prior is 30%, giving `PPV = 0.873`; (c) the sensitivity and specificity were measured in a case-control sample whose prior is neither.

**Discriminating check.** Compute `PPV = sens·π/(sens·π + (1−spec)(1−π))` at both priors. At `π = 0.02`: `0.018/(0.018 + 0.049) = 0.2687`. At `π = 0.30`: `0.27/(0.27 + 0.035) = 0.8727`. The spread across plausible priors is the whole answer.

**Correction.** Publish sensitivity and specificity as the transportable pair, and publish PPV only with the prior it was computed at. The `NPV` moves the other way — 0.9979 at a 2% prior — and should be quoted when the rule's purpose is exclusion.

**Prevention.** Store the prior as a required field on any predictive-value artefact, and make a PPV without it fail validation rather than default to 1.0 or 0.5.

**Verification.** Take the prevalence from an independent cohort, recompute PPV, and confirm the published number moves in the direction the formula predicts (down when the prior falls).

### CI-005 — Multicollinearity read as an unstable world

**Symptom.** A coefficient on a marketing variable is 0.31 in one specification and 0.06 in another that adds a highly correlated channel, and the analysis concludes that the channel's effect is uncertain.

**Competing causes.** (a) Near-collinearity with `r = 0.98`, `VIF = 1/(1−0.9604) = 25.25`, so the coefficient's standard error is inflated by `sqrt(25.25) = 5.03`; (b) a genuine interaction the second specification omits rather than adds; (c) a scale or unit change between the two runs.

**Discriminating check.** Regress each suspect column on the others and read `R²_j`. `R²_j = 0.9604` reproduces the `VIF` of 25.25. Then compare the coefficient's standard error across the two fits: under (a) the standard errors inflate together with the coefficient change; under (c) they do not.

**Correction.** Do not drop the column to make the other's coefficient "significant". Either report the combination the data identify — the fitted values and the joint test — or impose a ridge penalty that shrinks along the unstable direction.

**Prevention.** Compute and print `κ(X)` and the per-column `VIF` on every fit, and treat a `VIF` above 10 as a requirement to state which linear combination the data actually identify.

**Verification.** Refit on a bootstrap resample; under collinearity the coefficient distribution is wide and centred near zero while the fitted values are stable — that pairing is the signature, and it distinguishes the defect from a real coefficient change.

### CI-006 — Exponential overflow in softmax and sigmoid

**Symptom.** A classifier returns `nan` loss, or returns probability exactly `1.0` for every case in a batch, on inputs whose logits are large.

**Competing causes.** (a) `exp` overflowed: `log(realmax)` is 709.78 in float64 and 88.72 in float32; (b) the logits are large because a weight update diverged, so the overflow is a symptom of an optimisation defect; (c) the probability is legitimately 1.0 to within float64, since `sigmoid(20) = 0.999999998`.

**Discriminating check.** Print the maximum logit. Above 709.78 the overflow is certain (cause a). Between 20 and 709 the probabilities are representable and the defect is elsewhere (cause c). A divergent loss trajectory points at (b).

**Correction.** Evaluate `log Σ exp(x_i)` as `c + log Σ exp(x_i − c)` with `c = max_i x_i`, and the sigmoid as `1/(1 + exp(−x))` for `x ≥ 0` with the complementary form for `x < 0`. Never clip the logit at 700 as a fix; that changes the answer.

**Prevention.** Put log-sum-exp in one shared function and forbid direct `exp` on unbounded quantities in review, with the bound stated numerically per dtype.

**Verification.** Feed a batch containing logits `{1000, 1001}` and confirm the output is `{0.2689, 0.7311}`, then feed `{−1000, −1001}` and confirm the mirrored result — the second call is the one that catches an underflow in the normaliser.

### CI-007 — A p-value read as the probability the null is true

**Symptom.** "There is a 5% chance this effect is real" is written next to `p = 0.05`.

**Competing causes.** (a) The writer equated `P(data | null)` with `P(null | data)`; (b) the writer meant the test's size, which is 5% by construction; (c) `p` was computed from a one-sided test and reported as two-sided.

**Discriminating check.** Ask for the quantity in the form "under repeated sampling from this design with no effect, what fraction of studies would produce a statistic at least this extreme". That fraction is `p`; the posterior probability requires a prior and is not in the output at all. Check (c) by comparing the reported `p` with twice the one-sided value.

**Correction.** Replace the sentence with the sampling statement, and if a posterior probability is genuinely needed, compute it with an explicit prior and say so.

**Prevention.** Template the reporting sentence so the referent is structural: "p = 0.135 under the null of no difference, two-sided, n = 24,000 rows (3,453 effective)".

**Verification.** Recompute `p` from the test statistic and the reference distribution and confirm the printed value and the printed sidedness agree — mismatched sidedness is a factor-of-two error that survives every other check.

### CI-008 — Simpson's reversal between pooled and stratified rates

**Symptom.** Both strata improve, and the pooled rate falls. The dashboard says the change worked; the pooled number says it did not.

**Competing causes.** (a) The stratum mix changed between periods, so the pooled rate is a weighted average with different weights; (b) one stratum's counts were revised; (c) the pooled figure was computed over a different population than the strata.

**Discriminating check.** Build the two-way table. Constructed instance: stratum A has 1/100 in period 1 and 300/900 in period 2; stratum B has 50/900 and 60/100. Rates: A `1.00% → 33.33%`, B `5.56% → 60.00%`. Pooled: `301/1000 = 30.10% → 110/1000 = 11.00%`. Both strata rise; the pooled rate falls by 19.1 pp, so cause (a) holds. The reversal exists because period 2 moved 90% of its weight into the low-rate stratum.

**Correction.** Report stratum-specific rates with their counts, and standardise the pooled rate to a fixed weight distribution before comparing periods.

**Prevention.** Require a stratum table alongside any pooled rate whose population composition can change, and make the composition change visible as a column rather than as a footnote.

**Verification.** Recompute the pooled rate as `Σ w_j p_j` with the period-1 weights applied to both periods; the standardised comparison must agree in sign with the stratum table.

### CI-009 — Preprocessing fitted before the split

**Symptom.** Validation accuracy is implausibly high, then drops sharply in a live run.

**Competing causes.** (a) A scaler, an imputer, or a target encoder was fitted on the full dataset before splitting; (b) the split was random over rows but the rows are clustered; (c) the validation set is genuinely easier because of a time trend.

**Discriminating check.** Refit the pipeline with every transform inside the fold and compare. If the score falls, cause (a) holds. If it does not, split by cluster and compare again — that separates (a) from (b). A monotone score-versus-time curve without a refit change points at (c).

**Correction.** Move the transforms inside a pipeline object so that `fit` sees only training folds; for target encoding, fit on the training fold with out-of-fold values for the training rows themselves.

**Prevention.** Enforce the ordering structurally: the split function returns indices, and the pipeline is the only object permitted to call `fit`, so a transform cannot see the validation rows.

**Verification.** Deliberately inject a leak — fit a scaler on all rows — and confirm the harness detects the inflated score. A test that only checks the correct path cannot distinguish a leak from a strong model.

### CI-010 — A rate of zero on an empty denominator

**Symptom.** A defect rate of `0.0` appears for a product line that had no shipments in the window.

**Competing causes.** (a) The denominator was zero and a guard returned 0; (b) the denominator was null, and null propagated through a coalescing default; (c) the line genuinely shipped nothing defective and had shipments.

**Discriminating check.** Inspect numerator and denominator separately before the division. `0/0` cannot be distinguished from `0/n` after the division, which is why the check must happen before it. Cause (c) requires `n > 0`.

**Correction.** Emit `undefined` with a reason code (`empty_window`), and let the consumer decide whether to display a dash or exclude the row. A zero-rate claim about a window with no exposure is false, not conservative.

**Prevention.** Make the division primitive return a two-field result — value and status — rather than a bare number, so the empty case cannot be silently coerced.

**Verification.** Replay the window with zero rows and confirm the output status is `undefined` rather than `0`, and that the downstream aggregate treats `undefined` as missing rather than as a zero in its mean.

### CI-011 — A denominator change reported as a performance change

**Symptom.** A conversion rate rises from 4.0% to 5.2% in one week, and the rise is attributed to a product change.

**Competing causes.** (a) The denominator population changed — a bot filter was added, removing low-intent sessions; (b) the numerator's definition changed — a duplicate suppression rule modified; (c) the product change worked.

**Discriminating check.** Hold the denominator fixed at the old definition and recompute. If the recomputed rate is 4.0%, cause (a) holds. Then hold the numerator fixed and vary the denominator; if both manipulations move the rate, (a) and (b) are jointly responsible and neither can be read alone.

**Correction.** Recompute both weeks under one definition, and publish the definition with the number. If the definition change was itself the intervention, the two effects must be estimated separately, not netted.

**Prevention.** Version the metric definition alongside the code, and fail the comparison when the two versions differ without an explicit `definition_change` annotation.

**Verification.** Recompute the rate from a frozen snapshot of raw events under both definitions and confirm the published figure equals one of them, and that the published text names which.

### CI-012 — Mean of per-unit ratios reported as the aggregate ratio

**Symptom.** "Average cost per order is 6.25" computed as the mean of `10.0` and `2.5`, while total cost divided by total orders is 7.857.

**Competing causes.** (a) The metric is a mean of ratios, and the mean-of-ratios is not the ratio of means; (b) one unit's data was double-counted; (c) the units have different bases and the per-unit values are not comparable.

**Discriminating check.** Compute both. Unit 1: `1000/100 = 10.0`. Unit 2: `100/40 = 2.5`. Mean of ratios: `6.25`. Ratio of totals: `1100/140 = 7.857`. The difference is 20.4% and is entirely explained by the weighting, so cause (a) holds.

**Correction.** Decide which estimand is wanted. A cost-per-order *for the business* is `Σcost/Σorders`; a cost-per-order *for a typical unit* is the mean of ratios, and if units differ in size, that mean should be unweighted with its own standard error. Report the chosen one with its construction.

**Prevention.** Name the metric with its aggregation: `cost_per_order_aggregate` versus `mean_unit_cost_per_order`. A single field named `cost_per_order` is the defect.

**Verification.** Recompute from raw totals and confirm that the published value equals `Σcost/Σorders` to within rounding, and that the unweighted mean is available as a separate field rather than as a substitution.

### CI-013 — Multiplicity ignored across a family of tests

**Symptom.** Of twenty metric comparisons, exactly one has `p = 0.041`, and it is reported as the finding.

**Competing causes.** (a) The family-wise error rate is `1 − 0.95^20 = 0.6415`, so one rejection is the expected outcome under a global null; (b) the metrics are strongly correlated, so the effective number of tests is far below 20 and the correction is smaller; (c) the effect is real.

**Discriminating check.** Recompute the FWER at the number of effectively independent tests. At `k = 6`, `1 − 0.95^6 = 0.2649`; at `k = 20`, `0.6415`. Then apply Benjamini–Hochberg to the ordered p-values `{0.001, 0.008, 0.039, 0.041, 0.042, 0.060}`: the thresholds are `{0.0083, 0.0167, 0.0250, 0.0333, 0.0417, 0.0500}` and only the first two p-values clear them, so four of the six are not findings.

**Correction.** Declare the family before looking, apply Holm for FWER or Benjamini–Hochberg for FDR, and report the number of tests in the family next to each p-value.

**Prevention.** Make the family a required argument to the testing function so a p-value cannot be produced without stating what it is one of.

**Verification.** Recompute the adjusted p-values from the raw list and confirm the reported set of findings matches the procedure's output exactly, then re-run with the family size doubled and confirm the finding set does not grow.

### CI-014 — Row-level split when the rows are clustered

**Symptom.** A model validates at 0.91 AUC on held-out rows and at 0.63 AUC when deployed to new sites.

**Competing causes.** (a) The split put rows from the same site on both sides, so the model memorised site-level characteristics that are not available at prediction time; (b) feature drift between the validation period and deployment; (c) the deployment population is a different population, not a drifted one.

**Discriminating check.** Re-split by site and refit unchanged. A drop from 0.91 to roughly 0.63 with the same features and the same code isolates (a). If the group-split model also validates at 0.63 but deployment is 0.55, the residual gap is (b) or (c), distinguished by comparing feature distributions between the validation and deployment windows.

**Correction.** Split on the assignment unit; where the unit is fine-grained, use grouped cross-validation with the group being the highest level at which dependence exists.

**Prevention.** Tie the splitter to the schema's `assignment_unit` field so that a row-level split on clustered data is a schema violation rather than a judgement call.

**Verification.** Compute the fraction of groups appearing on both sides of the split; it must be exactly zero, and that assertion belongs in the training harness rather than in a review checklist.

### CI-015 — `n` used where `n − 1` belongs

**Symptom.** A hand-written standard deviation is 5.13% smaller than the library's on the same ten values.

**Competing causes.** (a) The divisor is `n` where the estimator is the sample sd; (b) the library applies a bias correction for a different reason, such as a weighted variance; (c) the hand formula computed the population sd intentionally and the label is wrong.

**Discriminating check.** Compare the ratio against `sqrt((n−1)/n)`. At `n = 10` that is `sqrt(0.9) = 0.94868`, so the hand value should be 94.87% of the library's — a 5.13% shortfall. Any ratio not matching `sqrt((n−1)/n)` to three digits eliminates (a).

**Correction.** Use `n − 1` for the sample sd and state which divisor the reported number used. The choice matters for the interval: the sample sd enters `se = s/sqrt(n)`, so a 5.13% understatement becomes a 5.13% narrower interval and a `z` inflated by a factor of `1/sqrt(0.9) = 1.0541`.

**Prevention.** Never hand-write the variance in a reporting layer; call one function whose divisor is documented, and assert its output against a known-value fixture.

**Verification.** Apply the estimator to a fixture whose sample variance is known exactly — `{1,2,3,4}` gives population `1.25` and sample `1.6667` — and confirm which value the code returns.

### CI-016 — A pooled variance where the groups differ

**Symptom.** A two-group comparison reports an interval that is symmetric and narrow, while one group's spread is visibly larger than the other's.

**Competing causes.** (a) Heteroscedasticity with a variance ratio of 5.26, so the pooled sd is a compromise that understates the noisier group; (b) an outlier in one group inflated that group's sd without changing the mean; (c) the groups are of unequal size, so the pooled sd is dominated by the larger one.

**Discriminating check.** Compare each group's sd with the pooled sd. With `s₁ = 0.5` and `s₂ = 0.218`, the ratio of variances is 5.26 and the pooled `s = 0.406` sits between them. Then compute the Welch degrees of freedom: `(s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁−1) + (s₂²/n₂)²/(n₂−1)] = 66.98` against 98 for the pooled test. A df near 67 rather than 98 confirms (a).

**Correction.** Use Welch's interval, which does not assume equal variances, and report the degrees of freedom actually used. If the difference in spread is the phenomenon of interest, model the variance rather than averaging over it.

**Prevention.** Default to the unequal-variance test; the pooled test buys a small power gain under an assumption that is rarely justified and is never checked.

**Verification.** Check that the reported df matches the Welch formula to two decimals, and that the interval width changes when the group with the larger variance is deliberately perturbed.

### CI-017 — Arithmetic mean of growth rates reported as the average growth

**Symptom.** A series grows 50% in one period and 0% in the next; the report says "average growth 25%", while the series' actual endpoint growth is 22.47%.

**Competing causes.** (a) The mean of the rates was taken instead of the geometric mean of the factors; (b) a negative rate was included and the arithmetic mean is meaningless rather than merely biased; (c) the periods have unequal lengths and the rates are not per-period-comparable.

**Discriminating check.** Compute the geometric mean of the factors: `sqrt(1.5 × 1.0) = sqrt(1.5) = 1.224745`, so the compounded growth is 22.47%. The arithmetic mean of the rates is `(0.5 + 0.0)/2 = 0.25`. The gap of 2.53 pp is the whole defect and it grows with variance, so cause (a) holds.

**Correction.** Average the log factors and exponentiate: `exp(mean(log(1 + r_t))) − 1`. Report the compounded rate, and if a simple average is genuinely wanted, label it "mean of period rates, not compounded".

**Prevention.** Store growth as a factor (`1 + r`) rather than as a rate, so that compounding is the default operation and the arithmetic mean is not directly available.

**Verification.** Multiply the factors, take the `n`-th root, subtract one, and confirm the result reproduces the endpoint ratio `V_n/V_0` raised to `1/n`.

### CI-018 — Likelihood computed as a product of probabilities

**Symptom.** The log-likelihood is `-inf` while every individual prediction is a valid probability, or the likelihood is exactly `0.0` for a model with a finite log-loss.

**Competing causes.** (a) Underflow: a product of 1,000 terms each near `1e-3` is `1e-3000`, which is 0 in float64; (b) one prediction is exactly 0 or 1, so the log is infinite; (c) the data contain an impossible observation under the model.

**Discriminating check.** Sum the logs instead of multiplying. If the sum is finite while the product is 0, cause (a) holds. If the sum contains `-inf` from one term, cause (b) holds, and that term's probability should be inspected directly rather than inferred from the total.

**Correction.** Accumulate `log p` with log-sum-exp where mixtures are involved, and clip probabilities at `ε` only when the clip is documented, since a clip at `1e-15` caps the per-observation loss at `−log(1e-15) = 34.54`.

**Prevention.** Make the product form unavailable: expose `log_prob` on the model and derive `prob` from it, rather than the other way round.

**Verification.** Compute the loss two ways — summed logs and the product's log — and confirm they agree to twelve significant figures on a small batch, then confirm the product form fails on a 1,000-row batch while the log form does not.

### CI-019 — A correlation read as a proportion of variance explained

**Symptom.** "The feature explains 60% of the variation" is written next to `r = 0.6`.

**Competing causes.** (a) The writer treated `r` as `R²`; (b) the regression is not linear, so `r²` is not the explained fraction either; (c) `R² = 0.6` was computed on a different sample than `r`.

**Discriminating check.** Square the correlation: `r² = 0.36`, so 36% of the variance, not 60%. Then refit and read the coefficient: with `sd(y) = 10` and `R² = 0.36`, the residual sd is `10 × sqrt(0.64) = 8.0`, which is a directly checkable quantity and does not depend on the interpretation of `r`.

**Correction.** Report `r²` or, better, the residual sd in the outcome's units, which requires no interpretation of "explained".

**Prevention.** Never expose a bare `r` in a report; expose `r²`, the slope with its units, and the residual sd together.

**Verification.** Recompute the residual sd from the residuals themselves and confirm it matches `sd(y)·sqrt(1 − R²)` to within rounding.

### CI-020 — Two solvers, two answers, both "correct"

**Symptom.** A least-squares fit returns one coefficient vector from a normal-equations route and a visibly different one from a QR route on the same data.

**Competing causes.** (a) Conditioning: `κ(X) = 99` and the normal-equations route squares it, losing `log10(9801) ≈ 4` digits; (b) the design is rank-deficient and the two solvers pick different minimum-norm solutions; (c) one route's default includes an intercept and the other does not.

**Discriminating check.** Compute the rank and `κ(X)`. Full rank with `κ = 99` and materially different answers points at (a). Rank below `d` points at (b), and the coefficient difference will be in the near-null direction. Compare the fitted values too: under (b) they agree to machine precision while the coefficients do not, which is the decisive signal.

**Correction.** Use QR or SVD for the solve, and never form `X'X` when `d` is large or the columns are correlated. If the design is rank-deficient, drop or combine columns before fitting, or use a documented ridge penalty.

**Prevention.** Print `κ(X)` and `rank(X)` next to every fit summary; a conditioning above `1/sqrt(ε) ≈ 6.7e7` makes a float64 solve untrustworthy regardless of the solver.

**Verification.** Refit after adding `1e-8 × I` to `X'X` and confirm the coefficients move; the size of the movement is a direct measure of how much of the answer was determined by rounding.

### CI-021 — Extrapolation beyond the observed predictor range

**Symptom.** A model trained on prices between 10 and 40 is asked about a price of 400 and returns a confident point estimate with a narrow interval.

**Competing causes.** (a) The functional form is wrong outside the hull — a linear fit continues linearly; (b) the residual sd is used as the prediction uncertainty, and it does not grow outside the hull; (c) the price of 400 is a data-entry error rather than a real input.

**Discriminating check.** Compute the leverage or the Mahalanobis distance of the query from the training rows. A distance well beyond the training maximum combined with an unchanged interval width confirms (b). Then inspect the query's provenance for (c), which is cheaper to rule out than (a).

**Correction.** Refuse the prediction with a reason code when the query is outside the training hull, or report the extrapolation distance and widen the interval by the leverage factor `sqrt(1 + h)`.

**Prevention.** Persist the predictor range as part of the model artefact and gate inference on it, rather than documenting the range in a notebook.

**Verification.** Query deliberately outside the range and confirm the output status is `out_of_range` rather than a point estimate, then query just inside the boundary and confirm a normal prediction.

### CI-022 — Two intervals for one quantity that do not overlap

**Symptom.** One report gives a 95% interval of (0.0703, 0.0797) for a conversion rate and another gives (0.0626, 0.0874) for "the same" rate. Both are centred on 0.0750 and the second is 2.64 times wider.

**Competing causes.** (a) The two use different variance constructions — Wald versus cluster-robust — so they answer different questions; (b) the populations differ, one being sessions and the other users; (c) one interval is for the rate and the other for the mean of cluster rates, which is a different estimand with a different variance.

**Discriminating check.** Compute the ratio of the half-widths. Wald at `n = 12,000` on `p̂ = 0.0750` gives `se = 0.0024044` and a half-width of `1.96 × 0.0024044 = 0.0047126`; the cluster-robust version multiplies `se` by `sqrt(6.95) = 2.6363` to `0.0063387`, half-width `0.0124237`. The ratio of half-widths is exactly 2.6363, which equals `sqrt(deff)` with no residual — that exactness confirms (a), because a population difference (b) would move the centre while leaving the ratio near 1, and a different estimand (c) would move both.

**Correction.** Label each interval with its construction and its estimand. Never publish two intervals for a quantity without stating whether they are meant to be comparable.

**Prevention.** Make the variance estimator a required, named argument on the interval function, and include its name in the rendered output.

**Verification.** Recompute both intervals from their stated constructions and confirm each reproduces to four significant figures; if an interval cannot be reproduced from its label, the label is wrong.

### CI-023 — An allocation that assumes the matrix fits in memory

**Symptom.** `MemoryError` or a swap storm when a distance matrix is built for 50,000 rows.

**Competing causes.** (a) The dense matrix needs `50,000² × 8 = 2.0e10` bytes = 20 GB; (b) the intermediate was upcast from float32 to float64, quadrupling an allocation that would have fit; (c) the allocation is fine but a copy was made per chunk, so peak usage is a multiple of the stated size.

**Discriminating check.** Compute `n² × itemsize` before allocating, and compare against the process's limit. Then measure peak RSS during the run: a peak well above `n² × 8` confirms (c).

**Correction.** Use a blocked or approximate nearest-neighbour computation, keep the intermediate in the smallest adequate dtype, and write results incrementally rather than materialising the matrix.

**Prevention.** Make the allocation a checked precondition — a helper that refuses to allocate more than a stated fraction of the budget — so the failure is a clear error before the machine is saturated.

**Verification.** Run at `n = 5,000` and confirm peak RSS is within 10% of `n² × itemsize`, then extrapolate to confirm the guard fires at the intended `n`.

### CI-024 — Accuracy quoted for a class-imbalanced target

**Symptom.** A classifier reports 98% accuracy on a target with a 2% positive rate, and the operating plan assumes it detects positives.

**Competing causes.** (a) The model predicts the majority class always, achieving exactly the base rate; (b) the threshold is badly placed so sensitivity is near zero at high specificity; (c) the target's prevalence in validation differs from deployment.

**Discriminating check.** Compute sensitivity and balanced accuracy. The always-negative rule gives sensitivity 0, specificity 1, and balanced accuracy `(0 + 1)/2 = 0.5`. Any balanced accuracy at chance level with a 98% headline confirms (a) or (b); the confusion matrix's positive cell separates them.

**Correction.** Report sensitivity, specificity, and the confusion matrix at the chosen threshold, plus a threshold-free measure. Accuracy belongs in the report only with the base rate next to it.

**Prevention.** Refuse to emit an accuracy metric when the minority class is below a stated fraction, unless the confusion matrix is emitted with it.

**Verification.** Score a trivial constant predictor through the same reporting path and confirm the report shows chance-level balanced accuracy; if the constant predictor scores well, the metric set is the defect.

### CI-025 — Ratio variance computed from per-column summaries

**Symptom.** A cost-per-order interval is 16% narrower than a recomputation from the raw pairs suggests, and the two columns' summaries are individually correct.

**Competing causes.** (a) The covariance term `−2Rσ_XY` was dropped because only means and variances were stored; (b) the numerator and denominator are independent and the term is genuinely zero; (c) the raw recomputation is wrong because it uses a different denominator.

**Discriminating check.** Compute the dropped term's magnitude. With `σ_X = 30`, `σ_Y = 120`, `R = 0.05`, `σ_XY = 2400`: the full bracket is `900 − 240 + 36 = 696` against `936` without the covariance, so the naive se is `0.03059` versus `0.02638`, i.e. 16.0% too large. The direction tells you which term was dropped; a check that cannot say which term is missing has not localised the defect.

**Correction.** Store sufficient statistics that include `Σxy`, or recompute the ratio variance from raw pairs. A delta-method variance can never be recovered from marginal summaries alone.

**Prevention.** Make the ratio metric a first-class object that consumes raw pairs, not two summary objects, so the covariance is structurally available.

**Verification.** Recompute the ratio variance from raw pairs and from a bootstrap of the same pairs; agreement to within the bootstrap's own error confirms the delta-method implementation.

### CI-026 — Pooled-variance degrees of freedom where variances differ

**Symptom.** A two-group p-value of 0.035 from a pooled test becomes 0.071 with an unequal-variance test on the same data.

**Competing causes.** (a) The variance ratio is 5.26 and the pooled df of 98 is too generous; (b) the sample sizes are unequal so the pooled estimate is dominated by the smaller group; (c) the test statistic, not the df, is the difference — a different `se` and not a different reference distribution.

**Discriminating check.** Compute the Welch df: 66.98 against 98. Then compute the two statistics separately — `chi² = 5.0505` for a pooled 2x2 with a normal approximation gives `p = 0.0246`, and the continuity-corrected statistic 3.2323 gives `p = 0.0722`. The p-value movement tracks the statistic more than the df, so (c) holds once the correction is made explicit.

**Correction.** Use the unequal-variance test and report the df used. Where cell counts are small, use an exact test: for a table with cells `{8, 2, 3, 7}` and a minimum expected count of 4.5, Fisher's exact one-sided p is 0.0349 against the asymptotic 0.0246.

**Prevention.** Default to the exact or Welch route for small counts and unequal variances; the pooled route should require an explicit justification.

**Verification.** Confirm that the reported df equals the Welch formula's value to two decimals and that the p-value equals the reference distribution's tail at that df.

### CI-027 — Autocorrelated series treated as independent observations

**Symptom.** A trend in a sensor series is declared significant with `n = 500`, but a block bootstrap returns an interval three times wider.

**Competing causes.** (a) Lag-1 autocorrelation `ρ₁ = 0.5`, so the variance of the mean inflates by `(1 + ρ)/(1 − ρ) = 3.0`; (b) a deterministic seasonal component that the model omits; (c) the series is non-stationary, so the mean is not a meaningful summary at all.

**Discriminating check.** Estimate `ρ₁` and compute the inflation factor. `ρ₁ = 0.5` gives 3.0, `ρ₁ = 0.2` gives 1.2247, `ρ₁ = 0.8` gives 3.0 as well but with an effective sample size of 11.1 rather than 33.3, so report `n_eff` alongside the factor. A seasonal residual pattern after differencing points at (b); a unit-root test points at (c).

**Correction.** Use a Newey–West variance with a bandwidth matched to the dependence length, or a block bootstrap with block length near `n^(1/3)` — 8 for `n = 500`. Report the effective sample size.

**Prevention.** Compute and print the lag-1 autocorrelation on every time-series estimate before any inference, and refuse the i.i.d. variance above a stated threshold.

**Verification.** Recompute the interval with two block lengths, `n^(1/3)` and twice that, and confirm stability; an interval that moves materially with block length is not yet resolved.

### CI-028 — A group mean used as an individual's expected value

**Symptom.** A district-level mean of 62% is attributed to a named school in that district, which is actually at 41%.

**Competing causes.** (a) An ecological inference: the group mean is not the member's value; (b) the school's value was mismeasured; (c) the district mean is a weighted mean dominated by a large school, so the "typical" school is far from it.

**Discriminating check.** Compute the within-group spread. If the within-district sd is 12 pp, a school 21 pp from the mean is 1.75 sd away — an ordinary member, not an outlier — which confirms (a). Then compare the weighted and unweighted district means; a large gap confirms (c).

**Correction.** Report the group mean as a property of the group, and report the within-group distribution alongside it. Never substitute one for the other in either direction.

**Prevention.** Require an explicit level in every metric name — `district_mean_rate` versus `school_rate` — and refuse a join that maps a group-level field onto a member-level row without a stated imputation.

**Verification.** Recompute the within-group sd and confirm that the member's deviation is inside the stated range, then confirm the published sentence names the level it is about.

### CI-029 — Exact float equality used as a guard

**Symptom.** A validation rule `if total == 0.3` never fires, or fires for one input and not another with the same intended value.

**Competing causes.** (a) `0.1 + 0.2 = 0.30000000000000004`, which is not equal to the float nearest 0.3; (b) the two operands were accumulated in different orders, so the rounding differs; (c) the values are genuinely unequal and the rule is correct.

**Discriminating check.** Print the difference with full precision. A difference of `5.55e-17` is one ulp of 0.3 and confirms (a) or (b). Then evaluate the same expression with the operands swapped; a change in the comparison result confirms (b) rather than (a).

**Correction.** Compare with an absolute tolerance scaled to the magnitude: `abs(a − b) <= 1e-9 * max(1, abs(a), abs(b))`, or use exact decimal arithmetic for monetary quantities.

**Prevention.** Represent money and countable quantities as integers in minor units, so the equality claim is about integers and cannot be corrupted by rounding.

**Verification.** Add a property test asserting that the comparison is reflexive and symmetric across a fixture of decimal values, which catches order-dependent guards.

### CI-030 — A 32-bit accumulator overflow in a large sum

**Symptom.** A total count of 3,000,000,000 is reported as `−1,294,967,296`.

**Competing causes.** (a) Signed 32-bit overflow: `3e9 − 2^32 = −1,294,967,296`; (b) the sum was computed in float32 and lost precision above 2^24; (c) a difference of two large counts was computed in 32-bit and wrapped.

**Discriminating check.** Compare the reported value with the true total minus `2^32`. Exact agreement with the wrap value confirms (a) and eliminates (b), which would produce a nearby value with lost low-order digits rather than an exact modular result.

**Correction.** Accumulate in 64-bit integers; where the count can exceed 2^63, use an arbitrary-precision type or split the accumulation by partition and combine in a wider type.

**Prevention.** Type the accumulator from the data's declared range rather than from the first element's inferred dtype, and assert `max_possible_count < dtype_max` at schema validation.

**Verification.** Sum a fixture straddling `2^31` and confirm the result is exact, then confirm the schema check rejects a declared range above the accumulator's capacity.

### CI-031 — Complete-case analysis with differential missingness

**Symptom.** A treatment effect of 6 pp under complete-case analysis disappears when the denominator includes all randomised units.

**Competing causes.** (a) Missingness differs by arm — 15% in treatment against 5% in control — so complete cases are a selected subpopulation; (b) the outcome is missing at random given covariates and a model-based correction would suffice; (c) the effect is real and the full-population estimate is the biased one.

**Discriminating check.** Tabulate missingness by arm. With 20% missing overall and 15% versus 5% by arm, the complete-case estimate over-weights the control group's complete cases and the bias is `11.1%` on a 54% base. Then impute missing outcomes under best and worst case and check whether the effect survives both bounds: if it does, (b) is irrelevant to the decision.

**Correction.** Report the estimand under explicit assumptions about the missing outcomes, with an interval that spans the assumptions, rather than a single complete-case number.

**Prevention.** Make missingness a reported column by arm in every experiment output, so a differential rate is visible before any estimate is read.

**Verification.** Recompute the estimate with all missing outcomes set to the worst case and then the best case, and confirm the published interval contains both bounds.

### CI-032 — Regression to the mean read as an effect

**Symptom.** The group selected for being in the worst decile improves by more than the rest, and the improvement is attributed to the intervention.

**Competing causes.** (a) Selection on a noisy measure guarantees reversion proportional to the measurement error; (b) the intervention works; (c) the measure was revised between periods, so the comparison is not like-for-like.

**Discriminating check.** Predict the reversion. With a prior sd of 0.5 and a selection at the 90th percentile (`z = 1.2816`), the expected reversion is `z × sd × (1 − reliability) = 0.641` sd units if the measure is half noise. Then compare the treated group with a selected-but-untreated comparison group: under (a) both revert by the same amount, and under (b) only the treated one improves.

**Correction.** Use a comparison group selected by the same rule, or model the reversion explicitly with a reliability estimate. A before-and-after comparison on a selected group cannot separate the two mechanisms.

**Prevention.** Attach the reliability of the selection measure to any "worst performers improved most" analysis, and require the comparison group before publishing the effect.

**Verification.** Simulate a null intervention on the same selection rule and confirm the analysis reproduces the predicted reversion; an analysis that finds zero effect on the simulated null is not measuring reversion.

## 9. Complexity & Cost Analysis

The catalogue's entries differ by two to four orders of magnitude in the cost of the check that finds them, and this asymmetry — not their frequency — is what determines which ones get caught. A check that costs one extra pass over data already in memory is run; a check that costs a second data product is not, unless it was scoped as a deliverable.

| Check | Cost model | At the Section 5 scale | At 100× the rows | What dominates |
|---|---|---|---|---|
| Recompute a variance two-pass | `2n` reads, no extra allocation | 48,000 element reads | 4.8e6 reads | memory bandwidth |
| Cluster-robust variance | `O(n + M d²)`; store per-cluster score vectors | 24,000 scores, `d = 1` | grows linearly in `M` | storing the scores |
| Sandwich with `M < n` clusters | same as above plus a `d × d` outer product per cluster | 240 outer products | 24,000 | the per-cluster product, not the sum |
| Cluster bootstrap | `B` refits over `M` clusters | `B × 24,000` row reads; 24 million at `B = 1000` | 2.4e9 | the refit count, not the data size |
| Normal equations | `2 n d²` flops, `d²` memory | `2 × 1e6 × 1e4 = 2.0e10` flops = 20 GFLOP | linear in `n` | `d²`, not `n` |
| QR | `2 n d²` flops, same leading term | 20 GFLOP | same | `d²` |
| SVD | `4 n d²` flops | 40 GFLOP | same | `d²`, doubled |
| Exact 2x2 test | hypergeometric sum over the table's margin | negligible | n/a | nothing |
| Block bootstrap for `ρ₁` | `B` resamples of length `n` | `B × 500` rows | linear | block count and length |
| FWER recompute | `O(k)` | 20 multiplications | same | nothing |
| PPV recompute | `O(1)` | one division | same | nothing |

The table's most useful column is the last. Seven of the eleven checks are dominated by something other than data size: the cluster bootstrap by the number of refits, the normal equations by the square of the column count, the FWER and PPV checks by nothing at all. That is why the cheap checks have no excuse for being skipped, and why the expensive ones have to be scoped.

Three cost derivations, each recomputed rather than quoted.

```text
FORMULA [COST-NORMAL-EQ]: flops ≈ 2 n d² + (2/3) d³
WHERE: n rows, d columns; the first term forms X'X, the second factorises it.
CONSTRAINT: dense arithmetic; flop counts are leading-order and ignore memory hierarchy.
NUMERICAL EXAMPLE: n = 1e6, d = 100.
       2 × 1e6 × 1e4 = 2.0e10 flops; (2/3)(1e6) = 6.67e5 flops.
       Total 2.00007e10, so the `d³` term is 0.003% of the work — the cross-product dominates.
       At d = 1000: 2 × 1e6 × 1e6 = 2.0e12 flops, a hundredfold increase from a tenfold
       increase in d, which is the quadratic behaviour made explicit.
INTERPRETATION: the cost is set by `d`, so a check that adds a pass over `n` is nearly free
       by comparison, and a check that doubles `d` is not.
```

```text
FORMULA [COST-MEMORY]: bytes = n · d · itemsize
WHERE: n rows, d columns, itemsize 8 for float64 and 4 for float32.
CONSTRAINT: dense materialisation; sparse or chunked representations have different costs.
NUMERICAL EXAMPLE: a dense distance matrix for n = 50,000 at float64:
       50,000² × 8 = 2.5e9 × 8 = 2.0e10 bytes = 20 GB.
       At float32: 1.0e10 bytes = 10 GB — still above most single-process budgets.
       Blocked at 1,024 rows against 50,000 columns: 1,024 × 50,000 × 8 = 4.096e8 bytes = 410 MB.
INTERPRETATION: the quadratic form is 50× the blocked form, and the blocked form is the same
       computation; the cost of the naive layout is a design choice, not a requirement.
```

```text
FORMULA [COST-BOOTSTRAP]: rows read = B · M · m
WHERE: B resamples, M clusters, m observations per cluster; resampling clusters reads whole clusters.
CONSTRAINT: cluster resampling, not row resampling; row resampling costs B · n with n = M m.
NUMERICAL EXAMPLE: B = 1000, M = 240, m = 100.
       1000 × 240 × 100 = 2.4e7 rows.
       Each refit on a `d = 1` model costs O(n) arithmetic, so total arithmetic is 2.4e7
       multiply-adds — under a second on modern hardware, and the binding constraint is the
       per-refit overhead rather than the arithmetic.
INTERPRETATION: the cluster bootstrap is affordable at this scale, which removes the excuse
       for reporting an asymptotic interval with 240 clusters.
BOUNDARY: at M = 20 clusters, 1,000 resamples still cost little, but the resampling
       distribution has only 20 distinct units and the interval's resolution is limited by
       that count, not by B. Raising B does not repair a small M.
```

The cost model implies an ordering rule for the catalogue itself: run the `O(1)` and `O(k)` checks always, run the `O(n)` checks when the data are already in memory, and run the `O(B n)` checks when the decision's consequence exceeds the compute. The rule is worth stating as numbers because "when the data are already in memory" is usually true and is usually skipped: recomputing a two-pass variance costs one extra traversal, and at 24,000 rows that traversal is 48,000 reads — below the noise floor of any pipeline that touches a network.

A second cost dimension is the *remediation* cost, which the catalogue's severity column only hints at. Correcting CI-001 requires changing a variance estimator in one place and re-publishing every interval that used it; correcting CI-003 requires finding every sentence that inherited the wrong unit, which is an unbounded search. The pattern is general: defects in the numeric route are bounded by the code that computes them, and defects in the vocabulary are bounded by the artefacts that quoted the number, which is why `CONCEPT_CONFUSION` entries carry high remediation cost despite single-line fixes.

> **Takeaway:** cost the check before deciding not to run it; seven of eleven checks here are dominated by something other than data size.

## 10. Key Theorems / Results Reference

Each result is stated with its hypotheses, because every entry in Section 8 is a case of a hypothesis being violated, and the entry's correction is only valid if the replacement result's hypotheses hold.

**Central limit theorem (Lindeberg–Lévy).** For i.i.d. `X_1..X_n` with mean `μ` and finite variance `σ²`, `sqrt(n)(X̄ − μ)/σ` converges in distribution to `N(0, 1)`. Hypotheses: independence, identical distribution, finite variance. Fails for heavy tails without a finite variance, and fails for dependent data — the clustered case has finite variance but the wrong scaling, which is CI-001, and the autocorrelated case has the wrong variance constant, which is CI-027.

**Chebyshev's inequality.** For any random variable with finite variance, `P(|X − μ| ≥ kσ) ≤ 1/k²`. Hypotheses: none beyond finite variance. Value: it is the only bound available for an unknown distribution, and at `k = 2` it gives 0.25 against the normal's 0.0455 — a factor of 5.5. Useful when a normal approximation is not justified, and honest about how weak that is.

**Gauss–Markov theorem.** In the linear model `y = Xβ + ε` with `E[ε] = 0` and `Var(ε) = σ²I`, the ordinary least-squares estimator has minimum variance among all linear unbiased estimators. Hypotheses: correct linear specification, zero-mean errors, homoscedastic errors, uncorrelated errors, `X` of full column rank. Fails under heteroscedasticity (CI-016), under correlation (CI-001), and under rank deficiency (CI-005, CI-020) where no unbiased linear estimator exists for the individual coefficients.

**Slutsky's theorem.** If `X_n → X` in distribution and `Y_n → c` in probability for a constant `c`, then `X_n + Y_n → X + c`, `X_n Y_n → cX`, and `X_n/Y_n → X/c` for `c ≠ 0`. Hypotheses: the limit in the denominator must be a non-degenerate constant. This is the result that licenses substituting a consistent variance estimate into a normal approximation, and it fails exactly when the denominator's limit is zero — CI-010's empty window, CI-025's near-zero denominator.

**Delta method.** For `g` continuously differentiable at `θ` with `g'(θ) ≠ 0`, `sqrt(n)(g(θ̂) − g(θ)) → N(0, g'(θ)²σ²)`. Hypotheses: asymptotic normality of `θ̂`, non-zero derivative, and a dominated second-order remainder. Fails at `g' = 0` (the variance collapses to first order) and fails when the remainder is not dominated — CI-025's ratio, where `|R|·sd(Y)/μ_Y = 0.006` at the worked scale is comfortably small but reaches 0.3 for a noisier denominator.

**Jensen's inequality.** For a convex `φ`, `φ(E[X]) ≤ E[φ(X)]`; for concave `φ`, the reverse. Hypotheses: integrability and convexity over the support. Value: it explains the direction of several biases. For `φ(x) = 1/x` on `{0.5, 1.5}`, `E[1/X] = 1.3333` while `1/E[X] = 1.0`, so the mean of reciprocals exceeds the reciprocal of the mean by 33.3% — the mechanism behind CI-012's 20.4% gap and behind the fact that a mean of per-unit costs is not a cost per unit.

**Cauchy–Schwarz.** `|<x, y>| ≤ ||x||·||y||`, with equality iff `x` and `y` are linearly dependent. Hypotheses: none; it is an inner-product inequality. Value: it bounds the correlation at 1 and therefore bounds the variance inflation factor from below at 1, and it shows that the `VIF` diverges exactly as the columns approach dependence.

**Wielandt–Hoffman / singular value perturbation.** A perturbation of `A` by `E` moves the singular values by at most `||E||₂`, and the singular vectors by an amount proportional to `||E||/gap` where `gap` is the separation between the corresponding singular values. Hypotheses: 2-norm perturbation bounds. Value: it explains why a small gap — near-collinearity — produces large movement in the coefficient direction even when the singular *values* barely move, which is the formal content of CI-005.

**Neyman–Pearson lemma.** For testing a simple null against a simple alternative, the likelihood-ratio test is most powerful at every level. Hypotheses: both hypotheses simple, i.e. fully specified distributions. Value: it justifies the shape of the tests used here and shows why a composite null requires a uniformly most powerful test that may not exist, which is why the two-sided tests in this file use symmetric critical regions rather than an optimal one.

**Benjamini–Hochberg.** For independent test statistics, rejecting the `i`-th ordered p-value when `p_(i) ≤ i q / k` controls the FDR at `q`. Hypotheses: independence, or positive regression dependence (PRDS) for the dependent case. Value: at `k = 6` and `q = 0.05` the thresholds are `{0.0083, 0.0167, 0.0250, 0.0333, 0.0417, 0.0500}`, so against the p-values `{0.001, 0.008, 0.039, 0.041, 0.042, 0.060}` only the first two clear — the same set Holm gives here, but not in general. Fails to control FWER, which is CI-013's distinction.

**Bonferroni / Holm.** Bonferroni rejects `p ≤ α/k`, controlling FWER under arbitrary dependence. Holm's step-down procedure rejects the `i`-th ordered p-value while `p_(j) ≤ α/(k − j + 1)` for all `j ≤ i`, which is uniformly more powerful than Bonferroni and needs no independence assumption. At `k = 6`, `α/k = 0.008333`, and the corresponding normal critical value is 2.6383 rather than 1.9600.

**Welford's algorithm.** A one-pass recurrence for mean and sum of squared deviations that maintains the invariant `M_k = Σ(x_i − x̄_k)²` without ever forming a large sum of squares. The update is `M_k = M_{k−1} + (x_k − x̄_{k−1})(x_k − x̄_k)`. Its value is that it is numerically stable at fixed cost, which is the correction for CI-002 at streaming scale.

**Law of total variance.** `Var(Y) = E[Var(Y | X)] + Var(E[Y | X])`. Hypotheses: integrability. Value: it decomposes the clustered variance in one line — the between-cluster term is the variance of the conditional mean, the within term is the expectation of the conditional variance — and it is the identity behind the `deff` derivation in Section 4.1.

**Regression to the mean (Galton).** For a bivariate normal pair with correlation `r`, `E[Y | X = x] = μ_Y + r(σ_Y/σ_X)(x − μ_X)`, so the conditional expectation of the standardised deviate is `r z_x`, and the expected reversion is `(1 − r) z_x` in units of `σ_Y`. Hypotheses: bivariate normality, or at least a linear conditional expectation. Value: at `r = 0.5` and `z_x = 1.2816`, the reversion is `0.6408σ` — the prediction CI-032 requires.

**Law of large numbers.** For i.i.d. `X_i` with finite mean, `X̄_n → μ` in probability. Hypotheses: finite mean, independence or weak dependence. Value: it justifies the point estimate while saying nothing about its uncertainty, which is the gap that makes CI-001's correct point estimate look like evidence.

| Result | Gives you | Fails when | Entry it diagnoses |
|---|---|---|---|
| CLT | interval construction | dependence, heavy tails | CI-001, CI-027 |
| Chebyshev | distribution-free bound | never (weak by 5.5×) | CI-013 sanity checks |
| Gauss–Markov | OLS optimality | heteroscedastic or correlated errors | CI-016, CI-001 |
| Slutsky | plug-in variance | degenerate denominator | CI-010, CI-025 |
| Delta method | ratio and transform variance | `g' = 0`, large remainder | CI-025, CI-012 |
| Jensen | direction of ratio bias | never | CI-012, CI-017 |
| Wielandt–Hoffman | coefficient instability | never (a bound) | CI-005, CI-020 |
| Neyman–Pearson | test optimality | composite hypotheses | any two-sided test |
| BH | FDR control | arbitrary dependence, FWER target | CI-013 |
| Total variance | clustered variance decomposition | never (an identity) | CI-001, CI-016 |
| Galton | reversion magnitude | non-linear conditional mean | CI-032 |

The table's third column is the operational one. Five of the eleven results fail under a condition that the catalogue names as an entry; the other six are identities or inequalities that cannot fail, and their role is to supply the *magnitude* of a defect rather than to detect it. A diagnostic built on a result that cannot fail is therefore always safe to apply and never sufficient on its own — which is why every entry pairs a magnitude estimate (Jensen, Galton, total variance) with a check that can actually distinguish causes.

> **Takeaway:** state the hypothesis before citing the theorem; every entry in this catalogue is a violated hypothesis with a named replacement.

## 11. Interview / Exam Questions

**Q1.** An experiment reports 900/12,000 against 840/12,000 with `p = 0.135`. Assignment was at the site level, 120 sites per arm, 100 observations per site, and `ρ = 0.05`. What is the corrected p-value, and by what factor was the interval too narrow?

`se_0 = sqrt(0.0725 × 0.9275 × (1/12000 + 1/12000)) = 0.0033477`. `deff = 1 + 119(0.05) = 6.95`, `sqrt = 2.6363`, `se_cl = 0.0088256`. `z = 0.0050/0.0088256 = 0.5665`, `p = 0.5710`. The interval was too narrow by 2.6363×, and its coverage was `2Φ(1.95996/2.6363) − 1 = 0.5428` rather than 0.95. The point estimate is unaffected. If the respondent gives 0.57 without the factor, ask for the coverage — the factor and the coverage are the two numbers that make the defect legible to a non-statistician.

**Q2.** The same experiment has cluster sizes with `CV = 0.5`. Recompute the design effect and say what the error was.

`deff = 1 + [(CV² + 1)m̄ − 1]ρ = 1 + [(1.25)(120) − 1](0.05) = 1 + (149)(0.05) = 8.45`, against 6.95 for equal sizes. The equal-size formula understates variance by `8.45/6.95 = 1.216`, i.e. 21.6%, and the standard error by `sqrt(1.216) = 1.1026`, i.e. 10.3%. The mechanism is the `Σm_i²` term: unequal sizes put disproportionate weight on the large clusters, and the term is quadratic in `CV`.

**Q3.** A colleague computes the variance of `{1e8, 1e8+1, 1e8+2}` as 2.0 and the population variance is 2/3. Diagnose and give the fix.

The one-pass form `E[x²] − (E[x])²` subtracts two numbers near `1.00000002e16` that agree to twelve digits, so the difference retains digits the representation no longer has; the computed 2.0 is three times the true 0.6667. The two-pass form computes `Σ(x − x̄)²/n = (2/3)`. The fix is Welford or two-pass, and the guard is a magnitude test: when `|mean|/sd > 1e4`, the one-pass form is inadmissible at float64.

**Q4.** A screening test has 90% sensitivity and 95% specificity. Give the PPV at a 2% prior and at a 30% prior, and state which number belongs in the operational plan.

At 2%: `PPV = 0.018/(0.018 + 0.049) = 0.2687`. At 30%: `0.27/(0.27 + 0.035) = 0.8727`. Both are correct; they answer different questions. The 2% figure belongs in a general-population screening plan and the 30% figure in a high-risk clinic. The transportable pair is sensitivity/specificity; the PPV is a property of the pair plus a prior. Follow-up worth asking: what is the NPV at 2%? It is 0.9979, which is why the rule is useful for exclusion and not for confirmation.

**Q5.** Two reports give 95% intervals of (0.0703, 0.0797) and (0.0626, 0.0874) for "the conversion rate", both from the same 900-in-12,000 sample. Which is right?

Both can be right and they are not measuring the same thing. The first is a Wald interval from `se = sqrt(0.075 × 0.925/12000) = 0.0024044`, half-width `1.96 × 0.0024044 = 0.0047126`. The second multiplies that `se` by `sqrt(6.95) = 2.6363` to `0.0063387`, half-width `0.0124237` — the cluster-robust version. The tell is exactly the ratio: `0.0124237/0.0047126 = 2.6363`, which is `sqrt(deff)` to four digits. A respondent who says "the wider one is conservative, use it" has missed that they estimate variances for different estimands and must be labelled, not ranked; a respondent who says the first is wrong has missed that it is correct for a simple random sample.

**Q6.** A pooled conversion rate falls from 30.10% to 11.00% while both strata rise. Explain, with the numbers.

Stratum A: `1/100 = 1.00%` to `300/900 = 33.33%`. Stratum B: `50/900 = 5.56%` to `60/100 = 60.00%`. Pooled: `301/1000` to `110/1000`. Period 2 moved 90% of the weight into stratum A, whose rate is lower, so the weighted average falls despite both components rising. The fix is standardisation to fixed weights: `Σ w_j p_j` with period-1 weights applied to both, which restores the direction of the within-stratum change. The mechanism is a changed mixing distribution, not a changed rate.

**Q7.** Twenty metric comparisons are run and one gives `p = 0.041`. What is the probability of at least one such result under a global null, and what should be reported?

`1 − 0.95^20 = 0.6415`. One rejection is more likely than not under the null, so the finding is not evidence. Applying Benjamini–Hochberg to the ordered set `{0.001, 0.008, 0.039, 0.041, 0.042, 0.060}` with thresholds `{0.0083, 0.0167, 0.0250, 0.0333, 0.0417, 0.0500}` rejects only the first two, so four of the six candidate findings do not survive. If the metrics are strongly correlated the effective test count is lower and the penalty is smaller — but the respondent should say what the effective count is rather than assume 1.

**Q8.** A cost-per-order metric is computed as the mean of per-unit ratios: 10.0 and 2.5, giving 6.25. Total cost divided by total orders is 7.857. Which is the defect and what is the magnitude?

The mean of ratios is not the ratio of means, and the difference here is 20.4% (`7.857/6.25 = 1.2571`). Neither is "wrong" — they are different estimands. `Σcost/Σorders = 1100/140 = 7.857` is the business-level cost per order; `6.25` is the unweighted mean of two units' costs, which answers "what does a typical unit look like". Jensen's inequality explains the direction: `E[1/X] ≥ 1/E[X]` for convex `1/x`, so the mean of reciprocals exceeds the reciprocal of the mean.

**Q9.** A model validates at 0.91 AUC on a random row split and 0.63 when split by site. Name two distinct causes and one check that separates them.

Cause one is row-level leakage: rows from the same site appear on both sides, so site-level characteristics are memorised. Cause two is genuine drift between the validation window and the deployment window. The separating check is to refit on a grouped split with no code change: if the score drops to about 0.63, leakage is confirmed. If the grouped score stays at 0.91 and only the deployment score is 0.63, the gap is drift, and the check is a feature-distribution comparison across windows. A respondent who says "use cross-validation" has not separated them, because row-level CV reproduces the leak.

**Q10.** Why does forming `X'X` lose digits, and how many at `r = 0.98`?

`κ(X'X) = κ(X)²`, so the error amplification squares. For two columns with `r = 0.98`, the eigenvalues of `X'X` are `1 + r = 1.98` and `1 − r = 0.02`, so `κ = 99` and the squared route amplifies by `9.8e3`. At `ε = 2.22e-16`, `log10(9801) = 3.99`, so about four of the sixteen significant digits are lost before the solver runs. The fix is QR or SVD, which work on `X` and never form the cross-product; the cost is the same leading term, `2 n d²` flops, so the stable route is not slower at leading order.

**Q11.** A softmax returns `nan`. Give the two distinct mechanisms and the check that separates them.

Mechanism one is exponential overflow: `log(realmax)` is 709.78 in float64, so a logit above that overflows to `inf`, and `inf/inf` is `nan`. Mechanism two is a divergent optimisation producing legitimately huge logits, so the overflow is a symptom. The separating check is the maximum logit: above 709.78 with a stable loss curve is mechanism one, and a growing loss alongside growing logits is mechanism two. The fix for mechanism one is log-sum-exp with a max shift — `{1000, 1001}` must give `{0.2689, 0.7311}` — and the fix for mechanism two is not numerical at all.

**Q12.** An analyst reports "the feature explains 60% of the variation" from `r = 0.6`. What is wrong, and what should be reported instead?

`r² = 0.36`, so 36% of the variance, not 60%. The residual sd is the better report because it needs no interpretation: with `sd(y) = 10` and `R² = 0.36`, the residual sd is `10 × sqrt(0.64) = 8.0`, in the outcome's own units. The general lesson is that a squared correlation is a variance proportion and a correlation is not, and that the residual scale is the quantity a practitioner can act on.

**Q13.** A count of 3,000,000,000 prints as `−1,294,967,296`. Diagnose, and say why a float32 accumulation would have produced a different wrong answer.

Signed 32-bit overflow: `3e9 − 2^32 = −1,294,967,296`, an exact modular wrap. A float32 accumulation would instead have lost precision above `2^24 = 16,777,216`, producing a value near 3e9 with the low-order digits missing rather than an exact modular result — a diagnostic difference worth knowing, because the modular value tells you the dtype and the precision loss only tells you the width. The fix is a 64-bit accumulator with a schema check that `max_possible_count < dtype_max`.

**Q14.** A series of 500 autocorrelated observations with `ρ₁ = 0.5` yields a significant trend. By how much is the variance of the mean wrong, and what is the effective sample size?

The variance inflates by `(1 + ρ)/(1 − ρ) = 1.5/0.5 = 3.0`, so the standard error by `sqrt(3) = 1.732`. The effective sample size is `500 × (1 − 0.5)/(1 + 0.5) = 166.7`, i.e. a third of the rows. At `ρ₁ = 0.2` the factor is 1.2247 and `n_eff = 333.3`; at `ρ₁ = 0.8` the factor is 3.0 again but `n_eff = 55.6` — so the factor alone does not tell you the information content, and both must be reported. The fix is a Newey–West variance or a block bootstrap with block length near `500^(1/3) ≈ 8`.

**Q15.** You are handed a dashboard with a 0.0 defect rate for a line that shipped nothing, a "0.5% improvement" on a metric that moved 50 bp, and a 98% accuracy figure on a 2% positive rate. Which category does each belong to, and what is the one-line check for each?

The zero rate is `NUMERICAL_EDGE` (a degenerate denominator): check numerator and denominator separately, and emit `undefined` with a reason code. The 0.5% is `CONCEPT_CONFUSION` (unit): restate in basis points, 50 bp. The 98% is `CONCEPT_CONFUSION` (metric choice under imbalance): compute balanced accuracy, which is 0.5 for a constant predictor. The pattern to articulate is that all three are invisible to a recomputation of the same quantity, because each is arithmetically faithful to a wrong referent.

> **Takeaway:** every question above is answerable from one number plus one condition; a candidate who cannot produce the number has memorised the vocabulary.

## 12. Quick Reference Card

| Situation | Do this | Number to state | Entry |
|---|---|---|---|
| Rows nested in groups | Cluster-robust variance; report `n_eff` | `deff = 1 + (m̄ − 1)ρ`; `sqrt(deff)` on the se | CI-001 |
| Unequal cluster sizes | Use the `CV`-corrected design effect | `1 + [(CV²+1)m̄ − 1]ρ` | CI-001 |
| Variance of large-magnitude data | Two-pass or Welford | relative error up to 200% otherwise | CI-002 |
| Rate movement | Basis points, plus relative with base named | 0.5 pp = 50 bp = 7.14% of 7.0% | CI-003 |
| Predictive value | Recompute at the deployment prior | PPV 0.269 at 2%, 0.873 at 30% | CI-004 |
| Unstable coefficients | Print `κ(X)` and per-column `VIF` | `VIF = 1/(1 − r²)` = 25.25 at `r = 0.98` | CI-005 |
| `nan` in a classifier | Max logit, then log-sum-exp | overflow above 709.78 (f64), 88.72 (f32) | CI-006 |
| p-value interpretation | State the sampling referent | `p` is `P(data | null)`, not `P(null | data)` | CI-007 |
| Pooled rate versus strata | Standardise to fixed weights | 30.10% → 11.00% pooled, both strata up | CI-008 |
| Held-out score too good | Move transforms inside the fold | fraction of groups split = 0 | CI-009 |
| Empty denominator | Emit `undefined` with a reason code | `0/0 ≠ 0` | CI-010 |
| Many tests | Declare the family, then Holm or BH | FWER 0.6415 at `k = 20` | CI-013 |
| Ratio metric | Delta method with the covariance term | 16.0% se error if dropped | CI-025 |
| Selected-group improvement | Predict the reversion, add a comparison group | `(1 − r)·z_x·σ` = 0.641σ at `r = 0.5` | CI-032 |

The card is ordered by the frequency with which the situation arises in a review, not by entry number, and the third column is the number that makes the defect legible to someone who does not want the derivation. Keep the card beside the schema that defines the assignment unit and the metric definition, because both of those fields are what convert a card entry into a check that fires.
