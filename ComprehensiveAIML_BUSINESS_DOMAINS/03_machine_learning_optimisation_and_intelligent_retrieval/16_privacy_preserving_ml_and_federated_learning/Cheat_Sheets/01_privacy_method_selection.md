---
title: "Privacy method selection and differential privacy"
topic: "Choosing between differential privacy, local privacy, anonymisation, and cryptographic computation for a stated release decision"
collection: "Part 03 - Machine Learning, Optimisation and Intelligent Retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Privacy method selection and differential privacy

> **Evidence boundary.** Every count, budget, and interval in this sheet is a CONSTRUCTED teaching value
> labelled ILLUSTRATIVE; the arithmetic is DERIVED and can be re-run by hand from the formulas in
> section 4. Theorems are named with their authors in section 10 and their bounds are derived here
> rather than quoted. No deployment, benchmark, vendor performance figure, or organisational result is
> asserted anywhere in this file, and no source is claimed to have been consulted live.

## 1. File Header

**Purpose of this sheet.** Give a reviewer a repeatable procedure for answering one question: given a
stated release decision over a stated population, which privacy mechanism is defensible, what does it
cost in utility, and what exactly does its guarantee exclude. The sheet is written to be used during a
review meeting, not read once for orientation.

**The decision this sheet settles.** The choice of *release level* for a dataset or a statistic. The
five realistic levels are: raw row-level data under contract, de-identified rows under a k-anonymity-like
rule, differentially private aggregates, differentially private trained models or synthetic rows, and
cryptographic computation that never reveals the inputs at all. A sixth level, "federated training
without a formal mechanism", is discussed in section 7 because teams routinely mistake it for a
privacy method.

**Who this is for.** Four readers with different needs. A data owner needs sections 3 and 7 to phrase
the offer. A privacy engineer needs sections 4, 8, and 9 to calibrate and to defend the accountant.
An analyst needs section 5 to know whether the release can answer the question at all. An incident
responder needs section 8 to decide whether a claimed breach is a mechanism failure or a governance
failure. Nobody should skip section 4.4.

**Prerequisites.** You must be able to state a population and a denominator, distinguish a probability
per record from an expected count over a table, and read a cumulative distribution function. You must
also accept one uncomfortable premise: a privacy guarantee is a statement about an adversary's beliefs,
not a statement about whether a column looks identifying.

**Units used throughout.** Counts are patients or customers, whichever the population declares.
Epsilon is dimensionless and, for sequential composition, additive. Delta is a probability per release.
Noise scale *b* carries the unit of the queried statistic. Byte figures are decimal and are used only
in section 9.

**Notation table.**

| Symbol | Meaning | Unit / type | Valid range |
|---|---|---|---|
| `eps` | privacy parameter of an (eps, delta) guarantee | dimensionless | `eps > 0`, typically 0.01 to 10 |
| `del` | failure probability in the guarantee | probability per release | `0 <= del < 1`, typically `1e-5` to `1e-9` |
| `Df` | global L1 sensitivity of the query | unit of the answer | `Df >= 0` |
| `b` | Laplace scale parameter | unit of the answer | `b = Df / eps` |
| `sig` | Gaussian standard deviation per coordinate | unit of the answer | `sig >= Df sqrt(2 ln(1.25/del)) / eps` |
| `rho` | zCDP parameter | dimensionless | `rho = Df^2 / (2 sig^2)` |
| `n` | number of records in the release population | count | `n >= 1` |
| `k` | equivalence-class size in a k-anonymity rule | count | `k >= 2` |

**Outcome if the sheet is applied correctly.** You can name the mechanism, the adjacency relation, the
sensitivity derivation, the accountant, the total budget over a stated window, the utility loss with an
interval, and the single condition under which the whole thing stops being valid. You can also say
which of those seven items your predecessor left out.

**What a weak selection looks like.** Four recurring shapes. "We used differential privacy" with no
epsilon. "Epsilon is 0.1" with no unit of protection, so nobody knows whether the record or the
household is protected. "We added noise" where the noise was calibrated to the range rather than to the
sensitivity. "We anonymised it" where anonymisation means dropping the name column.

**Exclusions.** This sheet does not decide lawfulness of a purpose, does not cover access control, key
management, or network security, and does not cover the federated topology itself, which is sheet 02.
It also does not cover data minimisation as a field-by-field exercise, which is sheet 03, even though
minimisation usually beats noise on utility per unit of risk.

**How to use it under time pressure.** Read section 4.4 first, then the table in section 7, then the
failure catalogue in section 8. If the release survives those three, calibrate with section 5 and
record the budget with section 9. Section 11 is for interviews and for testing whether a colleague has
internalised the difference between a guarantee and a technique.

**One naming caution.** "Differential privacy" names a definition, not a product feature. Two systems
both described as differentially private can differ by three orders of magnitude in the noise they
inject at the same epsilon if they disagree about adjacency, about composition, or about whether the
noise is calibrated per query or per release.

## 2. Concept Map / Mental Model

**The pipeline.** Selection is a sequence of six narrowing decisions, and an error in any one of them
invalidates everything downstream. The order matters because later stages consume the output of
earlier ones.

```text
stated purpose
   |
   v
[1] adjacency unit: what counts as one person's data?  record / household / device / client
   |
   v
[2] query family: counts, means, histograms, gradients, or free-form SQL
   |
   v
[3] mechanism: Laplace / Gaussian / exponential / randomised response / MPC / none
   |
   v
[4] accountant: basic, RDP, zCDP, or an exact privacy-loss distribution
   |
   v
[5] release gate: total budget over a declared window, owner, and stop condition
   |
   v
[6] monitoring: what signal tells you the guarantee has been silently voided
```

**Reading the pipeline.** Stage 1 sets the sensitivity. Stage 2 sets whether sensitivity is even
bounded. Stage 3 buys the guarantee. Stage 4 decides how fast the budget is consumed. Stage 5 decides
who may spend it. Stage 6 is the only stage that catches drift, and it is the stage most often missing.

**The mental model that survives review.** Think of the mechanism as a filter that makes neighbouring
datasets produce *indistinguishable* outputs. The unit of indistinguishability is fixed by stage 1. If
you cannot state which two datasets are neighbours, you have not chosen a mechanism; you have chosen a
library.

**Two different questions, routinely conflated.** "Does the release reveal this individual?" is a
question about worst-case inference and is answered by the guarantee. "Would a reasonable person be
surprised?" is a question about context and is answered by governance. A release can pass the first
and fail the second, as when a differentially private count confirms a rumour that was already widely
believed. It can also fail the first and pass the second, as when a small cell is suppressed only when
it is small.

**The adversary ladder.** Selection should name the rung it defends against, because noise that stops
a curious analyst does nothing against an adversary who can difference two releases. Rung one: an
analyst who sees one release. Rung two: an analyst who sees a time series of releases. Rung three: an
analyst who can also query an auxiliary dataset. Rung four: an analyst who participates in the data
and can run membership inference. Rung five: a participant who can corrupt the protocol itself.

**Where the model breaks.** The pipeline assumes the query family is fixed before the data is seen.
The moment an analyst chooses queries adaptively by looking at prior noisy answers, the accountant must
cover the *adaptive* composition, which is valid for differential privacy but not for naive
sensitivity arguments that assume a fixed query set. This is the single most common conceptual error
in review documents.

**Where federated training sits in the model.** Federated training changes *where* computation happens,
so it changes rung five of the adversary ladder and the data-movement story. It does not by itself
bound the information in the model update, which is why federated systems still need stage 3. Sheet 02
covers the architectural consequences.

**What "selection" is not.** It is not picking the library with the best documentation, it is not
choosing the largest epsilon that passes an internal threshold, and it is not a one-time decision.
Every material change in population, query family, or refresh cadence reopens stage 1 through stage 3.

**The one-sentence version.** Choose the adjacency unit, prove the sensitivity, buy the guarantee at a
declared epsilon, account for every release in a window, and write down the condition that would
invalidate the proof.

## 3. Foundational Definitions

**Differential privacy.** A randomised mechanism `M` is (eps, del)-differentially private if for every
pair of neighbouring datasets `D`, `D'` and every measurable set `S` of outputs,
`P[M(D) in S] <= exp(eps) * P[M(D') in S] + del`. The definition constrains the *ratio* of output
probabilities, which is why it survives post-processing and why it composes.

**Pure versus approximate.** `del = 0` gives pure epsilon-DP, a hard cap on the privacy loss. Any
`del > 0` admits a set of outputs, of total probability at most `del`, on which the ratio can be
arbitrary. That escape hatch is what makes the Gaussian mechanism possible and what makes a delta
without a stated interpretation meaningless.

**Neighbouring datasets.** Two datasets are neighbours if they differ in the data of exactly one unit,
where the unit is declared by policy: one record, one person, one household, or one client. The whole
guarantee inherits this declaration. A guarantee over records says nothing about a person who
contributes forty rows unless you re-derive with group privacy.

**Global sensitivity.** For a query `f` and a metric `||.||`, the global sensitivity is
`Df = max over neighbours ||f(D) - f(D')||`. It is a worst case over the entire data universe, not over
the observed data, which is why it can be much larger than intuition suggests for medians and quantiles.

**Local sensitivity.** The same quantity restricted to the observed dataset. It is smaller and it is
*data dependent*, so it cannot be plugged directly into a Laplace scale without a smoothing argument.
Confusing local with global sensitivity is the defect behind most "our noise is too small" incidents.

**Laplace mechanism.** `M(D) = f(D) + Lap(b)` per coordinate with `b = Df / eps` satisfies pure
epsilon-DP. The noise is calibrated to sensitivity alone; it does not depend on `n`, which is why the
mechanism costs the same at any table size and why relative error shrinks as the table grows.

**Gaussian mechanism.** `M(D) = f(D) + N(0, sig^2)` per coordinate with
`sig >= Df * sqrt(2 ln(1.25/del)) / eps` satisfies (eps, del)-DP for `eps` in the regime below one.
Its privacy loss is unbounded but sub-Gaussian, so composition behaves like a square root rather than a
sum when analysed through zCDP or RDP.

**Epsilon-delta accounting terms.** Basic composition adds epsilons. Advanced composition and RDP give
a square-root saving in the number of queries. zCDP tracks a single number `rho` and converts to
(eps, del) at the end. All four are upper bounds; the exact privacy-loss distribution is tighter still
and is what production accountants compute numerically.

**Parallel composition.** Queries over disjoint subsets of the data cost the maximum epsilon, not the
sum. It is exact, it needs no theorem beyond the definition, and it is the cheapest available lever:
partitioning a table into strata is often worth more than any mechanism tweak.

**Post-processing immunity.** Any function of a differentially private output, computed without further
access to the data, is as private as the output. This is why clamping a noisy count at zero is free and
why re-running the mechanism after inspecting the answer is not.

**Group privacy.** If `M` is eps-DP for one record, then for a group of `g` records it is `g*eps`-DP.
The bound is tight and it grows linearly, so a household-level guarantee costs the household size in
epsilon. Ignoring this is how record-level guarantees get misrepresented as person-level ones.

**Randomised response.** A respondent answers truthfully with probability `p` and with the opposite
answer otherwise. The mechanism is eps-DP with `eps = ln(p/(1-p))`, so `p = 0.75` gives `eps = 1.0986`.
It requires no trusted curator and it is the canonical local mechanism.

**Local versus central trust.** Central DP assumes a curator that sees raw data and publishes only
noisy answers. Local DP assumes no trusted party at all and pays for that with noise that grows with
the population size rather than staying constant. Shuffle-model protocols sit between the two and rely
on a server that sees only randomly permuted messages.

**K-anonymity.** Every record is indistinguishable from at least `k-1` others on the released
quasi-identifier set. It is a syntactic property of the release, not a bound on an adversary's
posterior, so it does not compose and it does not survive auxiliary information.

**L-diversity and t-closeness.** Strengthenings that add a constraint on the sensitive attribute within
each equivalence class. They repair specific attacks on k-anonymity and inherit its structural
weakness: the guarantee is stated over the released table, not over the adversary's beliefs.

**Membership advantage.** For a prior of one half, an (eps, del)-DP mechanism bounds an adversary's
advantage at distinguishing "in" from "out" by roughly `tanh(eps/2)/2`. At `eps = 0.5` that is
`0.1225`, so twelve percentage points of advantage remain. This conversion is the fastest way to make
an epsilon meaningful to a non-specialist.

**Sensitivity of a proportion.** With `n` records and a predicate, the proportion has sensitivity
`1/n`. Counts have sensitivity `1`. The two are the same statement divided by `n`, and mixing them up
produces noise that is either a thousand times too large or a thousand times too small.

**Reconstruction attack.** Given enough noisy linear answers, an adversary can solve for individual
records. The defence is not a better mechanism but a budget: the attack needs a number of answers
comparable to the number of individuals, at a per-answer epsilon that makes each answer informative.

**Epsilon laundering.** Re-labelling a release with an epsilon smaller than the one actually spent,
typically by resetting the budget at each dashboard refresh or each new analyst. It is a governance
failure that looks like a numerical one, and section 8 gives the discriminating check.

**Terms that are confused in review documents.**

| Term | Common misuse | What it actually bounds |
|---|---|---|
| anonymisation | used as a synonym for de-identification | nothing formal; a legal category in some regimes |
| epsilon | read as a percentage | a likelihood ratio exponent, not a risk percentage |
| delta | read as "small, so fine" | the probability mass where the ratio bound fails |
| sensitivity | read as model sensitivity | the worst-case change in a query answer |
| composition | assumed to be additive only | additive for pure DP, sub-additive under RDP and zCDP |
| local DP | assumed stronger, therefore better | stronger trust model, far higher noise at scale |
| secure aggregation | assumed to give DP | hides individual messages from the server, not the aggregate |
| k-anonymity | assumed to be a privacy guarantee | a group-size property of one released table |

**The definition that decides most arguments.** Write down the neighbouring relation before writing
down the mechanism. If two reviewers disagree about the guarantee, they almost always disagree about
what one unit of data is, and no amount of arithmetic resolves that.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**The count query with Laplace noise.**

```text
FORMULA      M(D) = f(D) + Lap(b),      b = Df / eps,      Df = 1 for a count

WHERE        f(D) = #{i in D : predicate(i) is true}
             Lap(b) has density (1/2b) exp(-|x|/b) and variance 2b^2
             adjacency is add-or-remove of exactly one record

CONSTRAINT   the predicate is fixed before the data is inspected, and the release
             population is the unit declared at stage 1

EXAMPLE      eps = 0.5, so b = 1 / 0.5 = 2.0 patients
             sd = sqrt(2) * b = 1.41421 * 2 = 2.82843 patients
             E|noise| = b = 2.0 patients
             P(|noise| > t) = exp(-t/b) = 0.05  =>  t = b * ln(20) = 5.99147

INTERPRETATION  a 95 percent interval around any released count is plus or minus six
             patients, independent of the table size
```

**Why the ratio bound holds.** For two neighbouring datasets the true counts differ by one, so the
shifted Laplace densities differ by a factor of at most `exp(Df/b) = exp(eps)` at every point. Taking
the worst point gives the epsilon. The density is the whole proof; there is no asymptotic argument and
no assumption about the data distribution.

**Why the noise ignores `n`.** Sensitivity is a property of the query, not of the table. A count over
ten million rows and a count over ten rows both change by one when a record is added, so both need the
same absolute noise. The consequence, derived in section 5, is that the *relative* error falls like
`1/n`, which is why differential privacy is cheap for large tables and expensive for small cells.

**The tail formula.** The two-sided Laplace tail is exactly `exp(-t/b)`: each side contributes the
half-tail `(1/2)exp(-t/b)`, and the two halves sum to the whole. Setting `exp(-t/b) = alpha` gives
`t = b ln(1/alpha)`, so at `alpha = 0.05` the halfwidth is `b ln 20 = 2.995732 b`. The one-sided
statement at the same confidence is a different quantity, `t = b ln 10 = 2.302585 b`, because the
half-density must reach `alpha/2` rather than `alpha`. Quote the convention next to the number:
one-sided and two-sided constants differ by thirty percent and get mixed more often than any other
constant in this topic.

**Validity condition.** The base case is valid when the query is a count of a fixed predicate, the
adjacency unit is one record, the mechanism is invoked once, and the released value is used as a point
estimate with a stated interval. It is invalid the moment the same record can be counted twice through
two different predicates that the analyst will combine.

### 4.2 General case

**Arbitrary query with L1 sensitivity, then composition.**

```text
FORMULA      for k adaptive queries, pure DP:   eps_total = sum over i of eps_i
             for k queries under zCDP:           rho_total = sum over i of rho_i
             conversion:                          eps = rho + 2 * sqrt(rho * ln(1/del))

WHERE        rho_i = Df_i^2 / (2 sig_i^2) for a Gaussian mechanism on query i
             the conversion is valid for all rho > 0 and all del in (0,1)
             adaptivity is permitted: query i+1 may depend on answers 1..i

CONSTRAINT   rho accounting requires the Gaussian mechanism; the Laplace mechanism
             has no exact rho without an extra conversion step

EXAMPLE      k = 20, target eps_total = 1.0, del = 1e-6, ln(1/del) = 13.81551
             solve rho + 2 sqrt(13.81551 rho) = 1.0
             let u = sqrt(rho):  u^2 + 7.43384 u - 1 = 0  =>  u = 0.13217
             rho_total = 0.017469, so rho_i = 0.00087345 per query
             sig_i = Df / sqrt(2 rho_i) = 1 / sqrt(0.00174690) = 23.925 per query

INTERPRETATION  spreading one unit of epsilon over twenty Gaussian releases leaves each
             release with a standard deviation of about 23.9 units of the answer
```

**Why the conversion has that shape.** For a `rho`-zCDP mechanism the privacy loss random variable is
sub-Gaussian with variance proxy `2 rho` rather than bounded, so the probability that the loss exceeds
`eps` is controlled by a Gaussian tail instead of being zero. Inverting that tail gives
`eps = rho + 2 sqrt(rho ln(1/del))`. The first term is the mean drift and the second is the tail
allowance; at small `rho` the square-root term dominates, which is the entire reason RDP-style
accounting beats basic composition.

**Verifying the root.** Substitute the answer back into the original equation:
`0.017469 + 2 sqrt(0.017469 * 13.81551) = 0.017469 + 2 sqrt(0.241349) = 0.017469 + 0.982553 = 1.000022`,
which matches the target to rounding. Do this substitution on paper before trusting any accountant
output; a factor-of-two error in the coefficient of `u` produces a `rho` that is too small by a factor
of four and looks entirely plausible in a report.

**The coefficient is the trap.** The equation is `u^2 + 2 sqrt(L) u - eps = 0` where `L = ln(1/del)`.
Dropping the 2 turns the coefficient into `sqrt(L)`, moves the positive root from 0.13217 to 0.25196,
and inflates `rho` from 0.017469 to 0.063484. That larger `rho` looks safe and is the opposite: it
converts to an epsilon well above one, so the release is under-noised by a factor of
`sqrt(0.063484/0.017469) = 1.906`. Write the equation, then the root, then the substitution, on adjacent
lines, and never carry a root forward without substituting it back.

**Validity condition.** The general case is valid when each query's sensitivity is bounded and known,
the accountant covers adaptivity, and the total is compared against a budget declared over a stated
window. It is invalid when the query family is chosen after seeing the answers *and* the sensitivity
argument assumed a fixed family, which is the case for most ad hoc analytics.

### 4.3 Variants

**Parallel composition.** Partition the population into disjoint strata, query each stratum once at
`eps`, and the joint release is `eps`-DP overall, not `k*eps`. Derivation: an add-or-remove of one
record changes the output of exactly one stratum, so the probability ratio across the joint release is
the ratio of that one stratum. This is exact and needs no theorem. It is the single cheapest
engineering lever in the topic: the same epsilon buys twenty stratum-level answers instead of one
table-level answer whenever the strata genuinely partition the population.

**Randomised response in the local model.** Each respondent reports the truth with probability `p` and
the opposite with `1-p`. The likelihood ratio between "true yes" and "true no" for either observed
answer is `p/(1-p)`, so `eps = ln(p/(1-p))`. At `p = 0.75` this is `ln 3 = 1.0986`. The estimator is
`theta_hat = (x_bar - (1-p)) / (2p - 1)`, unbiased but with variance inflated by
`1/(2p-1)^2 = 4` at `p = 0.75`, before any population correction.

**The high-privacy regime.** As `eps` falls below roughly `0.1`, both the Laplace and the Gaussian
mechanism produce intervals wider than typical subgroup counts, so the honest options are to aggregate
over a longer window, to coarsen the predicate, or to refuse the release. Reporting a wide interval and
calling it a result is the failure mode this sheet exists to prevent.

**Amplification by subsampling.** If each record is included with probability `q` before the mechanism
runs, the effective epsilon is smaller than the full-dataset epsilon, and the saving is roughly
quadratic in `q` for small `q`. The saving is real but it is an accounting artefact: it requires the
sampling to be genuinely random and independent of the data, which is a property of the ingest
pipeline, not of the analytics.

**The exponential mechanism.** For a discrete answer set with a utility function `u` of sensitivity
`Du`, sampling answer `r` with probability proportional to `exp(eps * u(D, r) / (2 Du))` is
`eps`-DP. It is the right tool when the answer is a choice rather than a number, for example
publishing which of several thresholds is most informative.

**Clipping as a mechanism component.** Any bounded-sensitivity claim requires bounding the contribution
of one unit, which means clipping. Clipping is part of the mechanism, so the analyst must not choose
the clip from the same data and then treat the choice as free. Choosing it from a disjoint calibration
sample is the standard escape.

### 4.4 Limits and counterexamples

**Limit one: the median has unbounded global sensitivity.** For a dataset of `n` values in `[0,1]`,
moving one value from `0` to `1` can move the median by up to one half of the range when `n` is even
and the middle two order statistics straddle the move. Calibrating a Laplace scale from "one record out
of `n`" gives noise that is far too small, and the release is not private despite the code looking
correct. Use the exponential mechanism over a discretised answer set, or a smooth-sensitivity
construction, or a fixed-bin histogram.

**Limit two: k-anonymity gives no ratio bound.** An equivalence class of size five in which all five
members share the same diagnosis leaves the posterior at one for each member when the adversary knows
the class. The k constraint is satisfied and the disclosure is total. This is not a corner case; it is
the normal behaviour of a released table in which the sensitive attribute is correlated with the
quasi-identifiers, which is exactly the situation where release is tempting.

**Limit three: composition makes a good single-query epsilon useless.** At `eps = 0.5` a count has a
95 percent interval of plus or minus six. Twenty such releases cost ten units of epsilon by basic
composition. To hold a one-unit annual budget the per-release epsilon must fall to `0.05`, which
multiplies the interval by ten. This is arithmetic, not pessimism, and it is the constraint that kills
most dashboard proposals.

**Limit four: the guarantee is about a declared population.** A mechanism calibrated for records does
not protect a person who appears in the table forty times. Under group privacy the guarantee degrades
by a factor of forty, so a nominal `eps = 0.5` becomes twenty. The discriminating question is always
"how many rows can one unit contribute", and it must be answered from the key structure, not from the
documentation.

**Limit five: post-processing is only free if it touches nothing else.** Clamping, rounding, and
thresholding a published noisy count are free. Rounding *and then re-querying* the underlying table to
repair the rounding is not, because the second query consumes budget. Teams that treat rounding as free
while also re-deriving suppressed cells have silently doubled their spend.

**Limit six: neighbouring datasets must be closed under your preprocessing.** Normalising a feature by
a data-dependent maximum means the transformation depends on the data, so the effective sensitivity is
that of the composed map, which may be unbounded. Fix the normalisation constant from public
information or from a disjoint sample, and record that choice.

## 5. Worked Numerical Example

**The setting (ILLUSTRATIVE).** A health registry holds `n = 100,000` patients. An analyst wants the
count of patients with a named condition. The true count is 2,000, a prevalence of two percent. The
publication channel is a quarterly report, so four releases per year. The privacy target is stated as
"one unit of epsilon per patient per year".

**Step one: sensitivity.** The predicate is fixed and the adjacency unit is one patient, so a
neighbouring dataset changes the count by at most one. `Df = 1` patient. Note the unit: the sensitivity
carries patients, not a probability.

**Step two: calibration at the quarterly cadence.** With four releases per year and a one-unit annual
budget, each release gets `eps = 0.25` by basic composition.

```text
FORMULA      b = Df / eps = 1 / 0.25 = 4.0 patients
             halfwidth(two-sided 95 percent) = b * ln(20) = 4.0 * 2.995732 = 11.98293

WHERE        Df = 1 patient, eps = 0.25 per release, four releases per year
             ln(20) = 2.995732 comes from solving exp(-t/b) = 0.05

CONSTRAINT   the four releases must cover disjoint query families, or the budget
             must be re-split; four releases of the same count cost the same as one

NUMERICAL EXAMPLE   at a true count of 2,000 the relative halfwidth is
             11.98293 / 2,000 = 0.0059915, that is 0.6 percent of the count

INTERPRETATION  the quarterly figure is precise to about a dozen patients, which is
             adequate for a trend line and useless for a single clinic
```

**Step three: cross-check by a different route.** Express the same release as a proportion rather than
a count. The proportion has sensitivity `1/n = 1e-5`, so `b_p = 1e-5/0.25 = 4e-5`, and the halfwidth
is `4e-5 * 2.995732 = 1.19829e-4`. Multiply by `n = 100,000` to return to patients:
`1.19829e-4 * 100,000 = 11.9829`. The two routes agree to five significant figures, which confirms
that the `/n` in the proportion route and the count route are the same statement.

**Step four: what the annual budget forbids.** Suppose the report is upgraded to an hourly dashboard,
8,760 releases per year, with the same one-unit annual budget. Each release gets
`eps = 1/8,760 = 1.1416e-4`, so `b = 8,760` patients and the 95 percent halfwidth is
`8,760 * 2.995732 = 26,242.6` patients on a count of 2,000. The release is pure noise, and its interval
spans an order of magnitude more than the quantity it reports.

```text
FORMULA      b = Df / (budget / releases) = Df * releases / budget
             halfwidth = b * ln(20) = Df * releases * ln(20) / budget

WHERE        Df = 1, releases = 8,760, budget = 1.0, ln(20) = 2.995732

CONSTRAINT   valid only under basic composition; an RDP or zCDP accountant would
             give a smaller total, roughly by the square-root saving, but not
             enough to rescue an hourly cadence at this budget

NUMERICAL EXAMPLE   1 * 8,760 * 2.995732 / 1.0 = 26,242.6 patients
             the same formula at 365 releases gives 1,093.4 patients
             at 52 releases it gives 155.8 patients
             at 4 releases it gives 11.98 patients

INTERPRETATION  refresh frequency is the most expensive design choice in the whole
             system, and it is usually chosen by product, not by the privacy owner
```

**Step five: the parallel-composition alternative.** Instead of publishing one national count four
times, publish 20 regional counts once each, with regions that partition the 100,000 patients. Each
region holds 5,000 patients, sensitivity is still one patient, and the joint release is `0.25`-DP in
total because each patient appears in exactly one region.

```text
FORMULA      eps_total = max over strata of eps_stratum     (parallel composition)

WHERE        the strata partition the population; every record is in exactly one
             each stratum query is a count with Df = 1

CONSTRAINT   disjointness is over records, not over geography: a patient treated in
             two regions is in two strata and the bound fails

NUMERICAL EXAMPLE   twenty regions of 5,000 patients, true regional counts near 100
             b = 4.0 patients, halfwidth = 11.98 patients
             relative halfwidth at a count of 100 is 11.98 percent
             compare the national single release at 0.25: 0.6 percent

INTERPRETATION  parallel composition buys twenty answers for one epsilon, but each
             answer is twenty times coarser in relative terms because the cells are
             smaller; the trade is cell count against cell size, not privacy against
             accuracy
```

**Step six: the small-cell boundary.** The release is informative when the 95 percent halfwidth is a
small fraction of the count. Setting the relative halfwidth to ten percent and solving for the count:

```text
FORMULA      c_min = b * ln(20) / 0.10

WHERE        b = 4.0 patients at eps = 0.25, ln(20) = 2.995732

CONSTRAINT   the relative-error criterion assumes the count is the quantity of interest;
             for a rate the criterion applies to the numerator, not the denominator

NUMERICAL EXAMPLE   c_min = 4.0 * 2.995732 / 0.10 = 119.83, so a cell needs about
             120 patients before a ten percent claim is defensible
             at eps = 0.5 the same calculation gives 59.91 patients
             at eps = 0.1 it gives 299.57 patients

INTERPRETATION  the publishable cell-size floor is a direct function of epsilon, which
             is why small-area reporting and strong privacy are in genuine conflict
```

**Independent arithmetic re-check.** Take `eps = 0.25`, `b = 4`, and evaluate the tail at the reported
halfwidth: `exp(-11.98293/4) = exp(-2.995732) = 0.049999`. That is the two-sided five percent mass
split as `0.025` per side, confirming the constant. Then check the variance route: `Var = 2b^2 = 32`,
so `sd = 5.65685`, and `11.98293 / 5.65685 = 2.1183` standard deviations, consistent with a Laplace
95 percent point being about 2.12 sigma rather than the Gaussian 1.96. A sheet that reports 1.96 sigma
for Laplace intervals is off by eight percent.

**Sensitivity note.** Four levers, each with its own multiplier on the halfwidth. Doubling epsilon halves
the halfwidth. Doubling the population leaves the halfwidth unchanged but halves the relative error.
Switching from record-level to household-level adjacency with a mean household size of 3 multiplies the
halfwidth by three. Dropping delta from `1e-6` to `1e-9` in a Gaussian calibration multiplies sigma by
`sqrt(20.94641 / 14.03865) = 1.2215`, a twenty-two percent increase that surprises reviewers who expect
delta to be free.

**The comparison that settles most debates.** At the same `eps = 0.5` and the same count, the Laplace
mechanism gives a 95 percent halfwidth of 5.99 patients while the classical Gaussian calibration at
`del = 1e-6` gives `sig = sqrt(28.07730)/0.5 = 10.5976` and a halfwidth of
`1.959964 * 10.5976 = 20.77` patients. Laplace wins by a factor of 3.5 on a single release. Run twenty
releases under a one-unit total budget and the ordering flips: Laplace gives 59.91 patients per release
while zCDP-calibrated Gaussian gives 46.89, a twenty-two percent advantage for the Gaussian. Neither
mechanism is "better"; the number of releases decides.

**What this example deliberately excludes.** No claim is made that any registry publishes at these
numbers, that four releases per year is a real cadence, or that a one-unit annual budget is a legal
requirement. The arithmetic is the deliverable; the parameters are placeholders chosen to make the
trade-offs visible at a scale a reader can check.

## 6. Visual / Geometric Intuition

**The noise as an interval, not a number.** A release at `eps = 0.25` on a count of 2,000 is best
imagined as the interval `[1,988, 2,012]` with a point at its centre. Every downstream consumer should
see the interval. The moment the centre is quoted alone, the mechanism's entire value is discarded and
the release becomes a false precision claim.

**Tail shape decides the mechanism.** Laplace density is `(1/2b) exp(-|x|/b)`, a sharp peak with
exponential shoulders. Gaussian density is a rounded peak with `exp(-x^2)` shoulders. For the same
epsilon, Laplace is far narrower near zero and far wider in the extreme tail relative to a
variance-matched Gaussian. That is why Laplace needs no delta: its privacy loss is capped at
`Df/b = eps` with probability one.

**The privacy loss as a bounded random variable.** For the Laplace mechanism the log-ratio between the
densities of neighbouring datasets is `( |y - f(D)| - |y - f(D')| ) / b`, which lies in
`[-Df/b, +Df/b]` by the triangle inequality. At `Df = 1` and `b = 4` the loss lives in `[-0.25, 0.25]`
and never leaves. The Gaussian mechanism's loss is unbounded, with a sub-Gaussian tail, and delta is
precisely the mass beyond the cap.

**Geometry of the sensitivity ball.** The L1 sensitivity is the radius of a cross-polytope that must
contain every possible change in the answer. A count has radius one in one dimension. A histogram over
twenty categories has L1 radius two, because moving one record between two buckets changes two
coordinates, which is why histogram releases often use `b = 2/eps` and why analysts are surprised that
the noise is twice the count noise.

**Where the budget goes.** Picture the annual budget as a fixed area and each release as a rectangle of
width one and height epsilon. Twenty releases at `0.05` fill the same area as one release at `1.0`, but
the interval width is proportional to `1/eps`, so splitting the area across many releases costs
hyperbolically in accuracy. This single picture explains why "just add more noise per query" does not
scale as a design.

**The signal-to-noise picture.** For a cell of true count `c`, the ratio of the count to the 95 percent
halfwidth is `c * eps / (Df ln 20)`. At `eps = 0.25` and `c = 120` the ratio is ten, which is the visual
meaning of the ten percent relative-error floor derived in section 5: a halfwidth of one tenth of the
count is a ratio of ten, not of one. A ratio below one means the interval is wider than the count and the
cell supports no claim at all; a ratio between one and ten supports an order-of-magnitude statement; only
above ten does a percentage-level claim become defensible.

**Local versus central, drawn as two funnels.** Central DP adds noise of fixed width regardless of `n`,
so the relative interval shrinks like `1/n`. Local DP adds noise per respondent, so the estimator's
variance is `O(1/n)` in the *probability* but the count-level interval is
`O(sqrt(n))` rather than `O(1)`. The two funnels cross at small `n` and diverge thereafter, which is
why local DP is usable for millions of respondents and useless for a hundred thousand.

**The equivalence-class picture for k-anonymity.** Each released row sits in a box of at least `k`
rows sharing its quasi-identifier values. The box fixes the adversary's belief about *which* row is the
target but says nothing about the sensitive values inside the box. When the box is homogeneous the
belief about the sensitive value collapses to a point, which is the geometric statement of the
homogeneity attack.

## 7. Comparison / Selection Table

**The selection matrix.** Utility figures are computed for a count of 2,000 over 100,000 records at
`eps = 0.5` (ILLUSTRATIVE), using the formulas in section 4, so that the comparison is arithmetic
rather than opinion.

| Method | Protects | Formal guarantee | 95 percent interval on the count | Composes how | Precondition | Wrong when |
|---|---|---|---|---|---|---|
| Laplace, central | record in output | pure eps-DP | plus or minus 6 patients | additive in eps | trusted curator with raw access | more than a handful of releases |
| Gaussian with zCDP | record in output | (eps, del)-DP | plus or minus 20.8 patients | additive in rho, sub-additive in eps | same as Laplace, plus a delta interpretation | delta would need to be zero |
| Exponential mechanism | record in output | pure eps-DP | answer-level, no interval | additive in eps | discrete answer set and a bounded utility | the answer is a continuous statistic |
| Randomised response | respondent from the collector | pure eps-DP, local | plus or minus 1,230 patients | additive in eps | no trusted party; large n | prevalence is below a few percent |
| Shuffle model | respondent from the shuffler | eps-DP plus a shuffle assumption | between local and central | additive with amplification | a server that cannot see sender identity | the server can correlate senders |
| DP-SGD training | training example in the weights | (eps, del)-DP | not a count; per-example gradient norm | additive in rho over steps | gradient clipping per example | the model is fine-tuned on private data later |
| PATE | training example via teacher votes | (eps, del)-DP with data-dependent eps | not a count; vote margins | additive in eps over queries | an ensemble of teachers trained on disjoint shards | teachers are correlated by shared data |
| DP synthetic rows | row in the generator | (eps, del)-DP on the generator | model-dependent, no closed form | additive in rho | a generator you can train privately | downstream users assume row-level truth |
| k-anonymity, k = 5 | nothing formal | none | exact value, zero noise | not composable | quasi-identifier set is genuinely identifying | sensitive attribute correlates with quasi-identifiers |
| Homomorphic encryption | computation inputs from the server | cryptographic, not statistical | exact value | no budget consumption | the function is expressible as circuits | the aggregate itself is sensitive |
| Secure multiparty computation | inputs from every other party | cryptographic, not statistical | exact value | no budget consumption | honest-majority or a corruption bound | output leakage is the real risk |
| Trusted execution environment | inputs from the host | hardware attestation, not statistical | exact value | no budget consumption | a genuinely trusted enclave and supply chain | side channels are in scope |
| Federated averaging alone | raw data from the server, maybe | none | exact value, but the update leaks | not applicable | clients hold data locally | the threat model includes the server |

**Reading the table by column.** The fourth column is where most selection errors happen, because it
mixes two different kinds of answer: a statistical interval for the DP rows and "exact value" for the
cryptographic rows. Exactness is not privacy when the exact answer is about one person. A query for a
single patient's status returns an exact answer under MPC and discloses that patient completely.

**Choosing by threat model, in one pass.** If the adversary sees the released artefact and may hold
auxiliary data, you need a statistical guarantee over a declared population: rows one, two, four, or
five. If the adversary is the infrastructure operator and the output is an aggregate over many people,
you need cryptography: rows ten to twelve. If both, you need both, because they defend different
things and neither substitutes for the other.

**The federated trap, stated plainly.** Federated averaging hides raw rows from the server and reveals
a gradient or weight update that is a function of those rows. Without clipping plus noise, repeated
access to that function reconstructs training examples, and the number of rounds is the number of
queries. Row thirteen in the table is "none" for exactly this reason, and sheet 02 covers the
architecture that fixes it.

**Selection rule one: fix the population before the mechanism.** Household-level protection with a mean
household size of three costs three times the epsilon of record-level protection. Deciding this after
calibration is a rewrite, not a tweak.

**Selection rule two: count the releases, not the queries.** A dashboard with six tiles that refresh
daily is 2,190 releases per year, not six queries. Budget arithmetic operates on releases.

**Selection rule three: prefer partition to noise.** Parallel composition over disjoint strata costs
the maximum epsilon and buys many answers. When the product asks for regional breakdowns, partitioning
the population is usually cheaper than raising epsilon.

**Selection rule four: refuse rather than mislead.** When the derived interval is wider than the
quantity, the honest deliverable is the interval plus a statement that the cell is not publishable.
Shipping a point estimate with no interval is the failure mode that turns a privacy system into a
liability.

**Selection rule five: write the invalidating condition.** Every choice in the table has one condition
that voids it: a new join key, a new refresh cadence, a new analyst with query access, or a change in
what one unit of data means. Name it at selection time so monitoring has something to watch.

## 8. Failure Modes & Edge Cases

**How to use this catalogue.** Each entry names a symptom, the competing causes that produce that
symptom, the check that separates them, and the correction. The distinguishing feature of a useful
entry is that a competent engineer could be wrong about the cause, which is why "monitor the model" is
not an entry.

**F1: intervals far wider than the released number.**

| Field | Content |
|---|---|
| Symptom | a cell of 40 patients is released as 41 with a plus or minus 60 interval |
| Competing causes | epsilon split across too many releases; wrong sensitivity unit; population declared one record but the query counts events |
| Discriminating check | recompute `b = Df/eps` from the ledger; if `b` matches the design, the split is the cause; if `b` is ten times the design, the sensitivity or the epsilon unit is wrong |
| Correction | cut releases or coarsen the cell; never raise epsilon without re-approval |
| Prevention | publish the cell-size floor next to the epsilon in the design document |
| Verification | the floor computed from `b ln 20 / 0.10` matches the smallest published cell |

**F2: the mechanism is right and the release is still identifying.**

| Field | Content |
|---|---|
| Symptom | a nominally 0.5-DP release lets an insider name a patient with high confidence |
| Competing causes | auxiliary data outside the formal model; group privacy violated by duplicate rows; the query was about one person, not a population |
| Discriminating check | count rows per person in the key structure; if the maximum exceeds one, group privacy is the cause; if it is one, the auxiliary data is |
| Correction | re-derive at the person level, or suppress the singleton queries, or raise epsilon deliberately with a documented rationale |
| Prevention | adjacency unit recorded in the ledger, not in a comment |
| Verification | the maximum row count per declared unit appears in the release review |

**F3: epsilon resets silently.**

| Field | Content |
|---|---|
| Symptom | the annual report claims one unit, the dashboard shows forty releases at 0.1 |
| Competing causes | per-refresh budget reset; two teams running separate accountants; a re-issued report treated as a correction rather than a new release |
| Discriminating check | sum the ledger entries for the calendar window; if the sum exceeds the declared budget, the reset is the cause |
| Correction | one ledger per population, not per team; corrections consume budget |
| Prevention | the ledger is the only path to the mechanism; no direct API access |
| Verification | a reconciliation job fails the build when the window sum exceeds the declared budget |

**F4: the noise is calibrated to the range, not to the sensitivity.**

| Field | Content |
|---|---|
| Symptom | a mean over `[0,1]` with `n = 100` gets noise sized for a count |
| Competing causes | sensitivity copied from a count example; proportion and count units mixed; the query was a sum over an unbounded column |
| Discriminating check | recompute the sensitivity by moving one record to each extreme of its domain; compare with the code constant |
| Correction | `Df = 1/n` for a bounded mean, and for an unbounded sum, bound the column first |
| Prevention | every mechanism call takes a sensitivity argument with a unit, and the code refuses a dimensionless literal |
| Verification | a unit test that shifts one record and asserts the observed answer change is at most `Df` |

**F5: the clamp was chosen from the data.**

| Field | Content |
|---|---|
| Symptom | the code clips at the 99th percentile of the observed distribution and claims bounded sensitivity |
| Competing causes | the percentile is a data-dependent statistic; the clip is genuinely fixed but the *normaliser* is not; two-stage training where the second stage sees private rows |
| Discriminating check | ask whether the constant would change if one record changed; if yes, it is data dependent |
| Correction | fix the clip from public domain knowledge or a disjoint calibration sample |
| Prevention | constants enter through a configuration file with a provenance note |
| Verification | reproducibility run with one record perturbed produces an identical clip value |

**F6: post-processing that is not post-processing.**

| Field | Content |
|---|---|
| Symptom | suppression of small cells is described as free, yet the suppressed cells are re-derived |
| Competing causes | suppression thresholds are applied after seeing the noisy value and then a second query fills the gap; rounding is applied and then corrected by a fresh query |
| Discriminating check | count mechanism invocations, not published cells; if invocations exceed cells, the second query is the cause |
| Correction | either publish the suppressed cell as a range derived from the same noisy value, or accept the gap |
| Prevention | the release path has no second data access after the first mechanism call |
| Verification | the invocation counter equals the number of published cells in an audited run |

**F7: correlation inside the population.**

| Field | Content |
|---|---|
| Symptom | a release that is provably 0.5-DP for records behaves like 5-DP for families |
| Competing causes | duplicate person identifiers; shared devices; a household key collapsed into a person key |
| Discriminating check | group by the intended unit and take the maximum row count; multiply by epsilon |
| Correction | re-calibrate at the group level or reduce the per-unit contribution |
| Prevention | the population definition is an input to the mechanism, not documentation |
| Verification | a data-quality assertion fails when the maximum group size exceeds one |

**F8: delta treated as decoration.**

| Field | Content |
|---|---|
| Symptom | a release advertises `del = 1e-5` on 50,000 people with no interpretation |
| Competing causes | delta copied from a paper; delta chosen to make the sigma pass a utility test; delta never converted to an expected count |
| Discriminating check | multiply delta by the population; if the product is a person or more, the choice needs an explicit justification |
| Correction | state the interpretation: `del * n = 0.5` means half a person is expected to fall outside the guarantee |
| Prevention | the release template requires the product, not just the parameter |
| Verification | the reviewer sees `del`, `n`, and the product on one line |

**F9: the accountant and the mechanism disagree.**

| Field | Content |
|---|---|
| Symptom | the ledger says `eps = 2`, a reviewer recomputes 9 |
| Competing causes | the accountant assumed amplification the sampler does not provide; the number of steps in the ledger is the epochs count not the step count; the accountant is for a different adjacency |
| Discriminating check | recompute with the same accountant and the same adjacency from the raw step count; if it matches 9, the ledger is wrong; if it matches 2, the sampler claim is wrong |
| Correction | the accountant is code, versioned, with a pinned configuration |
| Prevention | the ledger entry stores the accountant version and its configuration hash |
| Verification | a nightly job recomputes the ledger from the stored telemetry |

**F10: local privacy applied to a small population.**

| Field | Content |
|---|---|
| Symptom | a local-DP prevalence estimate on 100,000 respondents has an interval of plus or minus 1.2 percentage points |
| Competing causes | local noise is irreducible; the estimator ignores the flip probability; the population is smaller than the design assumed |
| Discriminating check | compute `Var = p(1-p)/(n (2p-1)^2)` from the flip probability; if it matches the observed interval, the mechanism is behaving as designed |
| Correction | move to central or shuffle trust, or increase `n`, or accept a coarser estimate |
| Prevention | the design document states the minimum `n` for each supported epsilon |
| Verification | the observed interval matches the analytic variance within sampling error |

**F11: the release is a k-anonymous table.**

| Field | Content |
|---|---|
| Symptom | a table with `k = 5` is described as anonymised and shipped |
| Competing causes | the sensitive attribute is homogeneous inside classes; the quasi-identifier set is not the adversary's set; the table was joined with a public register |
| Discriminating check | compute the maximum class-level frequency of the sensitive attribute; if it is one, homogeneity is the cause |
| Correction | do not ship; publish DP aggregates over the analysis, not the rows |
| Prevention | the release checklist asks for the maximum class homogeneity, not just the minimum class size |
| Verification | the homogeneity value appears in the release record |

**F12: noise reuse.**

| Field | Content |
|---|---|
| Symptom | two consecutive releases are suspiciously similar and can be differenced to a precise estimate |
| Competing causes | a fixed random seed; a cached noisy answer; the same perturbation drawn once and applied to multiple queries |
| Discriminating check | re-run with a different seed; if the outputs move, the seed is the cause; if they do not, the cache is |
| Correction | fresh randomness per invocation, and no caching of noisy answers across releases |
| Prevention | randomness sourced from a cryptographic generator with no fixed seed path |
| Verification | two invocations on identical inputs produce different outputs in a test |

**Edge case: a query family with unbounded sensitivity.** Free-form SQL over an unbounded numeric column
has no finite sensitivity, so no epsilon exists. The fix is to bound the column, which is a
data-engineering task with a privacy consequence, and the release cannot proceed until the bound is
justified from outside the data.

**Edge case: the empty cell.** A true count of zero released with `b = 4` is negative thirty percent of
the time. Clamping at zero is free post-processing but it biases the estimate upward, and the bias must
be reported or corrected by a published estimator rather than hidden in the UI.

## 9. Complexity & Cost Analysis

**Three cost currencies.** Every selection spends privacy budget, engineering effort, and analyst
attention. Budget is the only one that is explicitly accounted, and the other two are what actually
determine whether the system survives a year.

**Budget cost as a table.** For a count of 2,000 over 100,000 records with `Df = 1`, the interval
halfwidth is `2.995732/eps` patients, so the entire cost curve is one division.

| Epsilon per release | Laplace halfwidth in patients | Relative to the count | Releases affordable in a one-unit budget |
|---|---|---|---|
| 0.05 | 59.91 | 3.0 percent | 20 |
| 0.10 | 29.96 | 1.5 percent | 10 |
| 0.25 | 11.98 | 0.6 percent | 4 |
| 0.50 | 5.99 | 0.3 percent | 2 |
| 1.00 | 3.00 | 0.15 percent | 1 |
| 2.00 | 1.50 | 0.075 percent | 0.5 |

**Reading the last column honestly.** "0.5 releases" is not a meaningful quantity; it means the design
cannot have two releases at that epsilon and still hold the annual budget. Fractional rows exist to make
the impossibility visible rather than to be rounded.

**Refresh cadence dominates.** With a one-unit annual budget, the per-release epsilon is
`1/releases`, so the halfwidth is `2.995732 * releases` patients. At four releases it is 11.98; at 52
it is 155.81; at 365 it is 1,093.4; at 8,760 it is 26,242.6. The cost is linear in refresh frequency and
the utility loss is linear in the same variable, which is why cadence deserves an explicit owner.

**Compute cost.** A Laplace draw costs one uniform sample and a logarithm, so a 10,000-cell histogram
costs 10,000 draws, which is microseconds. A Gaussian draw is a Box-Muller pair, comparable. The
mechanism is never the bottleneck; the accountant can be, because exact privacy-loss computation for a
subsampled mechanism is a numerical integration over an alpha grid and takes seconds per configuration
rather than per release.

**Accounting cost grows with the number of distinct configurations.** Ten query templates, four
cadences, and three populations is 120 accountant evaluations, each of which must be versioned. Budget
systems cost more in review time than in compute, and the review time is proportional to the number of
*configurations*, not the number of releases.

**Utility cost of the wrong trust model.** The local-DP comparison at `eps = 0.5` and a two percent
prevalence: the flip probability is `1/(1+e^0.5) = 0.377541`, so the response probability is
`p = 0.622459` and an honest prevalence of `0.02` is observed at
`0.622459 * 0.02 + 0.377541 * 0.98 = 0.382439`. The estimator variance is
`q(1-q)/(n(2p-1)^2)` with `q = 0.382439`, giving `0.236179/(0.060025 * 100,000) = 3.93730e-5`, a standard
deviation of `0.0062748` on the proportion, which is 627 patients on a count of 2,000. The central
mechanism at the same epsilon has a standard deviation of `2.82843` patients. Choosing local trust costs
a factor of `627.48/2.82843 = 221.85` in standard deviation at this population size.

**Engineering cost of each option, in descending order.** Cryptographic computation carries protocol
design, key management, and a security review. DP training carries clipping infrastructure,
per-example gradient machinery, and accountant integration. Central DP carries a curator, a ledger, and
a review board. k-anonymity carries the least engineering and the most legal exposure. The ordering is
stable across organisations; the absolute numbers are not.

**Cost of over-protection.** At `eps = 0.01` the halfwidth on the count is 299.57 patients, so every
subgroup below 3,000 is unpublishable. The observed organisational response is to stop publishing and
let analysts query the raw table directly, which is worse than either extreme. Selection must include
the fallback, not just the mechanism.

**Cost of under-protection.** A reconstruction attack needs roughly one informative answer per
individual. Making a singleton query informative requires `b` well below one patient, so `eps` well
above one per query, so total epsilon on the order of the population size: `1e5` for a 100,000-person
table against a budget of one. The gap between attack cost and budget is five orders of magnitude,
which is the quantitative version of "a budget is a real defence".

**Cost of the delta loophole.** `del = 1e-5` over 50,000 people is an expected half person outside the
guarantee. `del = 1e-6` over the same population is an expected 0.05 people. The sigma factor is
`sqrt(2 ln(1.25/del))`, which moves from `4.84481` at `1e-5` to `5.29880` at `1e-6`: a 9.37 percent
increase in noise for a tenfold improvement in delta. Going on to `1e-9` costs `6.47247`, a further 22.1
percent. Deltas near `1e-5` are cheap to improve; the last three orders of magnitude are what make the
noise visibly worse, and delta of exactly zero costs the guarantee entirely.

**Cost model summary, in one sentence per line.** The budget is spent by releases, not queries. The
interval scales as `1/eps`. The relative interval scales as `releases * ln(20) / (budget * c)`. The
engineering cost is dominated by the accountant and the review. The trust model choice has a
population-dependent utility multiplier that can exceed two orders of magnitude. The cheapest available
saving is partitioning, because parallel composition is exact.

## 10. Key Theorems / Results Reference

**Laplace mechanism (Dwork, McSherry, Nissim, Smith, 2006).** For `f` with L1 sensitivity `Df`, adding
`Lap(Df/eps)` per coordinate is `eps`-DP. Validity: `eps > 0`, sensitivity is global, the mechanism is
invoked once. This is the theorem every other result in the sheet is calibrated against.

**Gaussian mechanism (Dwork and Roth, 2014, classical form).** For `eps` in `(0,1)` and
`sig >= Df sqrt(2 ln(1.25/del))/eps`, adding `N(0, sig^2)` per coordinate is `(eps, del)`-DP with L2
sensitivity `Df`. Validity: the epsilon regime matters; outside it the constant is not the advertised
one and the guarantee must be recomputed through zCDP or RDP.

**Basic composition (Dwork, Rothblum, Vadhan, 2010, and the textbook form).** The `k`-fold adaptive
composition of `(eps_i, del_i)`-DP mechanisms is `(sum eps_i, sum del_i)`-DP. Validity: adaptivity is
allowed. Tightness: worst case is attained, which is why basic composition is pessimistic for
Gaussian mechanisms and exact for pure-DP worst cases.

**Advanced composition (Dwork, Rothblum, Vadhan, 2010).** For `k` mechanisms each `(eps, del)`-DP, the
composition is `(sqrt(2k ln(1/del')) eps + k eps (e^eps - 1), k del + del')`-DP for any `del' > 0`.
Validity: `eps` small. This is the result that introduces the square-root saving and the reason RDP and
zCDP are preferred as bookkeeping devices.

**Parallel composition.** Mechanisms over disjoint subsets cost the maximum epsilon. Validity: the
subsets partition the data with respect to the declared adjacency unit. It needs no citation beyond the
definition, and it is the only composition rule that is both exact and free.

**Post-processing (Dwork et al., 2006).** If `M` is `(eps, del)`-DP then `g(M(D))` is `(eps, del)`-DP for
any `g` that does not access `D`. Validity: no further data access, including through a cache or a
second model trained on the same rows.

**Group privacy.** `eps`-DP for one record implies `g eps`-DP for groups of `g` records. Validity: pure
DP, no delta. The bound is tight, and the linear growth is the reason population definition is the first
decision in the pipeline.

**zCDP (Bun and Steinke, 2016).** A mechanism with `rho = Df^2/(2 sig^2)` for a Gaussian release is
`rho`-zCDP; `rho` composes additively; `rho`-zCDP implies `(rho + 2 sqrt(rho ln(1/del)), del)`-DP.
Validity: exact for the Gaussian mechanism, and the conversion holds for all `rho`. This is the
accounting route used in section 4.2 because it needs one number and one square root.

**RDP (Mironov, 2017).** The `alpha`-Renyi divergence between neighbouring output distributions,
`eps_RDP(alpha)`, composes additively over `alpha` and converts to `(eps, del)`-DP by minimising
`eps_RDP(alpha) + ln(1/del)/(alpha-1)` over `alpha`. Validity: integer and fractional `alpha` both
allowed. RDP is tighter than zCDP for subsampled mechanisms and looser for very small epsilon.

**Exponential mechanism (McSherry and Talwar, 2007).** Sampling with probability proportional to
`exp(eps u(D,r)/(2 Du))` is `eps`-DP. Validity: a finite answer set and a bounded utility. This is the
correct tool for threshold selection and for reporting which of several statistics is most extreme.

**Amplification by subsampling (Kasiviswanathan et al., 2011; Li, Miklau, Hay, McGregor, Rastogi, 2012;
Balle, Barthe, Gaboardi, 2018 for tight constants).** Sampling records or clients with rate `q` before a
private mechanism reduces the effective epsilon, with savings of order `q^2`. Validity: sampling is
random and data independent, and the tight constants need the subsampled-RDP computation rather than a
formula.

**k-anonymity and the homogeneity attack (Sweeney, 2002).** A released table satisfying a `k`-group-size
constraint can still disclose a sensitive attribute when classes are homogeneous. Validity: the result is
a counterexample, not a bound, which is why no epsilon can be derived from a `k`.

**Reconstruction (Dinur and Nissim, 2003).** With sufficiently many sufficiently accurate linear answers,
individual records can be recovered regardless of the mechanism's intent. The relevant defence is a
budget that keeps each answer's accuracy below the informational threshold, which is what an epsilon
does.

**PATE (Papernot et al., 2017).** Aggregating teacher votes with noisy argmax over disjoint shards gives
`(eps, del)`-DP with an epsilon that depends on the vote margin rather than on the number of teachers.
Validity: teachers trained on disjoint shards, and the student may not see the private data.

**What the theorem list is for.** Every entry names a validity condition, and the conditions are where
implementations fail. A design document that cites the Laplace mechanism without the global-sensitivity
condition, or zCDP without the Gaussian condition, has cited a result that does not apply to it.

## 11. Interview / Exam Questions

**Q1. A count of patients with a condition is required for four consecutive quarters. State the
sensitivity and the composition rule you would use.**

Answer: sensitivity is one patient under add-or-remove adjacency, and the four releases compose
additively in epsilon because they are sequential over the same population and the same query family.
Reasoning: the query asks the same thing four times, so an adversary can difference the answers; parallel
composition would only apply if the four quarters covered disjoint patient sets, which they do not.
Follow-up: what changes if the quarters are disjoint cohorts of newly diagnosed patients? Then the four
queries are over disjoint populations and the total cost is `max`, not `sum`. Red flag: answering "four
releases cost four epsilon" without checking disjointness, or answering "max" without checking.
Scoring: one point for sensitivity, one for additive composition, two for the disjointness discussion.

**Q2. Compute the 95 percent interval for a count released with Laplace at `eps = 0.5`.**

Answer: `b = 1/0.5 = 2`, and the two-sided 95 percent halfwidth is `2 ln 20 = 5.991`, so plus or minus
six patients. Reasoning: the tail is `exp(-t/b)`, so `t = b ln(2/0.05)`. Follow-up: what is the
halfwidth at `eps = 0.1`? Thirty patients, because the halfwidth is inversely proportional to epsilon
with a constant of `2.9957`. Red flag: using `1.96 * sd` with `sd = 2b^2`, which gives 11.3 and is wrong
by a factor of nearly two. Scoring: two points for the constant, one for the direction of the epsilon
dependency, one for rejecting the Gaussian shortcut.

**Q3. A dashboard cell with a true value of 12 is displayed as 3. Is the system broken?**

Answer: not necessarily. At `eps = 0.25` the scale is 4 and the noise has a standard deviation of 5.66,
so a draw of `-9` is 1.6 sigma and unremarkable. The release is behaving as designed and the design is
wrong for the cell size. Reasoning: the publishable floor at that epsilon is about 120, so a cell of 12
must not be published at all. Follow-up: what is the cheapest fix? Suppress the cell or coarsen the
predicate; raising epsilon is a governance change, not a fix. Red flag: proposing to "add a floor" to
the displayed value, which is post-processing that biases every cell upward.
Scoring: one point for the arithmetic, two for naming the floor, one for the bias critique.

**Q4. You need 200 aggregate answers under a total budget of one epsilon. Which accountant do you
choose and what does it cost per answer?**

Answer: a Gaussian mechanism accounted through zCDP or RDP, because basic composition would give 0.005
per answer and a halfwidth 600 times the single-release value. Under zCDP the total rho solves
`rho + 2 sqrt(rho ln(1/del)) = 1`, giving `rho = 0.01747` and `rho_i = 8.73e-5` per answer, so sigma per
answer is `1/sqrt(2 rho_i) = 75.7` for unit sensitivity. Reasoning: the square-root saving is the entire
point of the advanced accountants, and 200 queries is well inside the regime where it helps.
Follow-up: at what query count does the Gaussian's larger per-query sigma stop being worth it? Compare
`46.89` at twenty queries against `59.91` for Laplace and note that the gap widens with count.
Red flag: quoting a per-query epsilon of 0.005 and calling it infeasible without trying an accountant.
Scoring: two points for the accountant choice, two for the rho arithmetic, one for the crossover
awareness.

**Q5. A record-level guarantee is 0.4. The population includes households with up to six members. What
is the household-level guarantee?**

Answer: `6 * 0.4 = 2.4` by group privacy, and the bound is tight. Reasoning: group privacy charges the
group size linearly in pure DP because the adversary may see up to six simultaneous changes.
Follow-up: how would you instead buy a 0.4 household guarantee? Calibrate with sensitivity six, which
multiplies the noise by six, or restructure the schema so each household contributes one row.
Red flag: claiming the household guarantee is still 0.4 because "each record is protected".
Scoring: one point for the multiplication, one for tightness, two for the recalibration options.

**Q6. The ledger says the annual total is 1.0, but the release log shows 40 releases at 0.1 each.
Diagnose.**

Answer: the releases sum to 4.0, so the ledger is not accounting for the same set of releases: either it
counts templates rather than invocations, or it resets per refresh. Reasoning: 40 times 0.1 is 4.0 under
basic composition, and no accountant reduces a sum of pure-DP releases below the sum.
Follow-up: what check distinguishes a template-counting ledger from a reset? Compare the invocation
counter in the release path with the ledger row count; if invocations exceed rows, the ledger is
aggregating.
Red flag: proposing a different accountant to reconcile the numbers. Accountants do not shrink pure-DP
sums. Scoring: two points for the arithmetic, two for the discriminating check.

**Q7. At `eps = 0.2`, what membership advantage does an adversary have against a balanced prior?**

Answer: `tanh(0.1)/2 = 0.09967/2 = 0.0498`, about five percentage points. Reasoning: the likelihood ratio
is capped at `exp(0.2) = 1.2214`, so the posterior moves from 0.5 to `1.2214/2.2214 = 0.5498`.
Follow-up: what epsilon gives a one percent advantage? `2 atanh(0.02) = 0.0400`. Red flag: reporting
`eps` as a percentage of records. Scoring: two points for the conversion, one for the inverse problem,
one for units.

**Q8. A survey of 100,000 respondents reports a two percent prevalence under local DP at
`eps = 0.5`. How precise is it, and would central DP be better?**

Answer: the flip probability is `1/(1+e^0.5) = 0.377541`, so `p = 0.622459` and the observed rate at a
true prevalence of 0.02 is `0.382439`. The estimator variance is
`q(1-q)/(n(2p-1)^2) = 0.236179/(0.060025 * 100,000) = 3.93730e-5`, a standard deviation of 0.0062748,
which is 627.5 patients on a count of 2,000. Central DP at the same epsilon has a standard deviation of
2.82843 patients, `627.48/2.82843 = 221.85` times smaller. Reasoning: local noise is paid per respondent,
so it scales with `sqrt(n)` in count terms while central noise is constant. Follow-up: when is local
still correct? When no party may be trusted with raw responses, in which case the 222 factor is the
price of the trust model, not a defect. Red flag: recommending local DP "because it is stronger" without
the precision consequence. Scoring: two points for the variance, two for the comparison, one for the
trust argument.

**Q9. A released table with `k = 10` was re-identified. Which property failed?**

Answer: homogeneity, not the group size. If every member of a class shares the sensitive value, `k = 10`
protects nothing about that value. Reasoning: k-anonymity constrains the adversary's uncertainty about
which row is the target, not about the target's attributes. Follow-up: what check would have caught it?
Maximum class-level frequency of the sensitive attribute, computed before release. Red flag: proposing a
larger `k`, which does not repair homogeneity. Scoring: two points for homogeneity, one for the check,
one for rejecting larger `k`.

**Q10. State the sensitivity of the mean of 500 values in `[0,1]` and the resulting Laplace scale at
`eps = 0.5`.**

Answer: `1/500 = 0.002`, so `b = 0.004`, and the 95 percent halfwidth is `0.01198` on a mean of, say,
0.4. Reasoning: replacing one value moves the mean by at most `(1-0)/500`. Follow-up: what happens if
the column is unbounded? No finite sensitivity exists, so the column must be bounded before any
mechanism applies. Red flag: using sensitivity one because the code adds Laplace with scale `1/eps`.
Scoring: one point for `1/n`, one for the scale, two for the unbounded case.

**Q11. Two consecutive releases of the same count are nearly identical. What happened and why is it
serious?**

Answer: the noise was reused, either through a fixed seed or a cache. The pair can be differenced or
averaged to reduce variance by `sqrt(2)` per repeat, and over many repeats the estimate converges to the
true value, so the effective epsilon grows without bound while the ledger records two releases.
Reasoning: differential privacy guarantees hold per independent draw; correlated draws are not the
mechanism the accountant analysed. Follow-up: what test detects it? Invoke twice on identical inputs and
assert the outputs differ. Red flag: attributing similarity to "small noise". Scoring: two points for
the mechanism, two for the consequence, one for the test.

**Q12. A release covers 50,000 people at `del = 1e-5`. Is that acceptable?**

Answer: only with an interpretation, because `del * n = 0.5` means about half a person is expected to
fall outside the guarantee, and the failure is not bounded by epsilon when it occurs. Reasoning: delta is
a probability per release over the output set, so multiplying by the population converts it to an
expected count of affected individuals under a uniform reading. Follow-up: what does `del = 1e-7` cost?
Sigma grows by `sqrt(ln(1.25e7)/ln(1.25e5)) = sqrt(16.34/11.74) = 1.180`, an eighteen percent noise
increase. Red flag: "delta is small, so it does not matter". Scoring: two points for the product, one for
the cost of reducing it, one for the failure semantics.

**Q13. Why is clamping a published noisy count free while choosing the clamp from the data is not?**

Answer: clamping is a function of the published output only, so post-processing immunity applies.
Choosing the clamp requires reading the data, so it is a data-dependent step that must be accounted for
or fixed from external knowledge. Reasoning: the guarantee is about functions that do not access `D`;
a clamp constant derived from `D` is a statistic of `D`. Follow-up: what is the standard escape? Fix the
constant from a disjoint calibration sample or from domain knowledge, and record the provenance.
Red flag: "clamping is just formatting". Scoring: two points for post-processing, two for the provenance
requirement.

**Q14. Give the condition under which your chosen release stops being valid.**

Answer: any change to the population definition, the refresh cadence, the query family, or the set of
people with query access invalidates the calibration, because each changes sensitivity, budget, or the
adversary. Reasoning: the guarantee is a function of all four, and none is enforced by the mechanism
itself. Follow-up: which of the four is hardest to detect in production? Query access, because it leaves
no trace in the release log. Red flag: naming a model-drift metric, which is a utility signal and not a
privacy signal. Scoring: two points for the four conditions, two for the detection argument.

## 12. Quick Reference Card

| Item | Value or rule | Why it matters |
|---|---|---|
| Laplace scale | `b = Df / eps` | the only calibration that needs no delta |
| 95 percent halfwidth, Laplace | `b * ln 20 = 2.9957 b` | the number to put next to every release |
| Gaussian sigma | `Df sqrt(2 ln(1.25/del)) / eps` | valid for `eps` below one; above it, use zCDP |
| zCDP conversion | `eps = rho + 2 sqrt(rho ln(1/del))` | one number, one square root, composes additively |
| Sequential cost | `sum eps_i` | releases, not queries, consume the budget |
| Parallel cost | `max eps_i` | exact, free, and the cheapest lever available |
| Group privacy | `g * eps` | population definition is the first decision |
| Membership advantage | `tanh(eps/2)/2` at a balanced prior | makes epsilon legible to non-specialists |
| Publishable cell floor | `c >= b ln 20 / 0.10` | at `eps = 0.25`, about 120 records |
| Invalidating condition | population, cadence, query family, or access change | the only thing monitoring can usefully watch |
