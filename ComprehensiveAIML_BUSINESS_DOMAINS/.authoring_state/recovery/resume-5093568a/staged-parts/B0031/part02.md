## 5. Data, Tools & Method

### 5a. Data Specification

Grain, keys, and cut-off have to be fixed before any number is quoted, because most of the ways this
exercise can go wrong are grain errors that look like modelling wins.

| Element | Specification |
|---|---|
| Grain | one row per eligible session per instrument; the exercise has one instrument |
| Primary key | (instrument_id, session_date) with session_date on the exchange calendar |
| Session rows | 2,200 constructed sessions; first 700 discarded as burn-in; 1,499 scored origins |
| Calendar | a synthetic five-day trading week; no holidays, halts, or half sessions |
| Return field | log close-to-close, decimal, computed from consecutive eligible closes |
| Price field | adjusted close in currency units, vendor revision window up to 5 sessions |
| Cut-off timestamp | 07:30 local on the issue morning, before the session being forecast opens |
| Event time vs processing time | session close is event time; vendor receipt at 06:40 next day is processing time |
| Output units | variance in squared decimal returns; volatility in decimal; risk in currency units |
| Comparison baseline | mean of squared returns over the last 20 eligible sessions, zero mean assumed |
| Notional | constant 250 million currency units, never updated within the evaluation |

The single most important row in that table is the cut-off. Everything the candidate is allowed to
see must have a processing timestamp strictly before 07:30 on the issue morning. The recursion
consumes returns up to and including the previous session, all of which closed before that morning,
so the lag structure is safe by construction. What is not safe by construction is the *value* of
those returns: a close published yesterday evening can still be replaced by the vendor three or four
sessions later, so a backtest rebuilt from today's file can see a revised number that no morning
pack ever saw.

What is deliberately not available at decision time: the current session's close, the current
session's realised range, option-implied volatilities, order-book depth, news sentiment for the
session in progress, and the vendor's later restatements. The exclusion of implied volatility is a
scope decision rather than a data limitation, and it matters because an implied measure would
probably dominate a purely historical recursion. Adding it would change the case from "can we
improve the existing historical estimator" to "should we buy an options feed", which is a different
purchase decision with a different owner.

The population is one instrument with a stationary unconditional variance of 1.0e-4 in squared
decimal returns, that is a daily volatility of 1.0% and an annualised figure of 15.8745% under a
252-session convention. The realised sample standard deviation over the scored window is 0.011011,
which annualises to 17.48%, higher than the population value because the window contains a
volatility regime that the burn-in does not. That gap between the population parameter and the
sample statistic is itself a teaching point: it is the reason a single realised squared return is a
bad target, and the reason the evaluation in Section 9 reports against two different proxies.

### 5b. Tools & Stack

The stack is deliberately minimal, because a method that needs more machinery than this cannot be
audited by a risk committee in an afternoon.

| Layer | Choice | Reason |
|---|---|---|
| Series generation | NumPy default generator, seed 20261110 | Reproducible constructed world |
| Candidate estimator | recursive variance update, one decay parameter | Auditable in a single equation |
| Baseline estimator | trailing 20-session mean of squares | Already in production |
| Diagnostics | quasi-likelihood, Mincer-Zarnowitz slope, Kupiec, Christoffersen | Standard, published, recomputable |
| Distribution functions | SciPy norm, chi-square, binomial | Avoid hand-rolled tail quantiles |
| Storage | immutable daily snapshot plus revision history | Enables as-of reconstruction |
| Orchestration | one scheduled job, hard 20-minute budget | Fits the 07:30 deadline |

Version pinning matters more here than in a typical analytics project, because the two headline
losses in this case differ by a factor of fifty depending on which proxy they are computed against.
A reviewer who cannot reproduce the proxy choice cannot reproduce the ranking, and the ranking is
the only claim the case makes.

### 5c. Method Summary

The candidate is the recursion in which today's variance estimate is a weighted average of
yesterday's estimate and yesterday's squared return. Its single parameter is the decay, and the
method's behaviour is fully described by three derived quantities: the half-life, the effective
sample size, and the weight placed on an observation of a given age. Section 6 derives all three and
shows how the half-life converts a decay into a statement a risk manager can argue about.

The baseline is the unweighted mean of the last twenty squared returns. It has no parameter to fit
but it does have a structural choice, the window length, and Section 9 shows that this choice is
worth more than the choice between the two families. That is an uncomfortable result for a proposal
framed as "replace the baseline with something better", and it is why the case's recommendation is
about a selection protocol rather than about a specific estimator.

The evaluation protocol is chronological and fixed before any candidate is computed. The first 899
of the 1,499 scored origins are a development fold used to choose the decay; the last 600 are a test
fold used once. Six hundred sessions is about 2.38 years of trading, which is long enough to compare
average losses and far too short to certify a one percent tail, as the power calculation in Section
9 demonstrates. The protocol therefore distinguishes an average-loss claim, which the data can
support, from a tail claim, which it cannot.

### 5d. Data Provenance, Quality, Exclusions, Leakage

Provenance for the constructed world is a script, not a vendor, but the discipline is identical. The
generating recursion is a variance equation with a persistent component and an innovation, plus a
deterministic variance multiplier of 1.5 applied to sessions 1500 through 1559 to create a regime
that the estimators must react to. Because the scale is generated, the case has access to the true
conditional variance, a luxury no real desk has. That access is used in exactly one place, as one of
two evaluation proxies, and it is labelled as an unobservable oracle rather than a realistic input.
Any analysis that optimised against it would be cheating in the most direct possible way, and
Section 12 names that failure with its discriminating check.

Quality rules applied before scoring: a session missing its close is dropped and the return spanning
the gap is treated as a multi-session return, not as a one-session return; a session with a
zero or unchanged close is flagged as stale and excluded from the return sample; a session whose
return exceeds a sanity band of 20% in absolute value is quarantined for review rather than
silently winsorised. In the constructed series no such session occurs, which is a known limitation:
real data would exercise all three rules and the case would then need to report how many sessions
each rule removed.

Exclusions are recorded, not applied quietly. The burn-in of 700 sessions exists because the
recursion starts from an arbitrary initial variance and needs time for the influence of that
starting value to decay. With a decay of 0.94 the weight still attached to the initial condition
after sixty sessions is 0.94 raised to 60, which is 0.0244; after 700 sessions it is vanishing. The
burn-in is therefore generous by two orders of magnitude, and the scored window can be treated as
free of initialisation artefacts.

Leakage controls are the part of the specification most likely to be relaxed under schedule
pressure, so they are written as four checkable rules rather than as intentions. First, every input
carries the timestamp at which it became available and the estimator may only read inputs whose
timestamp is before the cut-off. Second, the decay is chosen on the development fold only, and the
frozen value is written into the manifest before the test fold is scored. Third, no evaluation
statistic may be recomputed with a different proxy after seeing which proxy favours which method.
Fourth, the vendor restatement window is honoured by evaluating the candidate twice, once on the
as-published vintage and once on the fully restated vintage, and reporting both.

The restatement check is worth spelling out with arithmetic because it is the difference between a
defensible and an indefensible backtest. Suppose the vendor replaces a return five sessions after
the fact, from a published value of 0.5% to a restated value of 2.0%. The squared values are 0.000025
and 0.0004. At the moment that observation is the newest one, its contribution to the candidate
variance is 0.06 times 0.0004, which is 0.000024, against 0.0000015 for the published value. A
backtest using restated data would therefore have seen a volatility estimate roughly 0.5 percentage
points higher on that morning than the desk actually saw, and would credit the method with a
reaction it did not have.

### 5e. Reproducibility & Method-Selection Ledger

| Field | Value |
|---|---|
| Series generator | variance recursion, omega 4.0e-6, alpha 0.06, beta 0.90 |
| Persistence | alpha plus beta equals 0.96; shock half-life 16.98 sessions |
| Unconditional variance | omega divided by one minus persistence, equals 1.0e-4 |
| Regime multiplier | variance times 1.5 for sessions 1500 to 1559 inclusive |
| Random seed | 20261110, NumPy default generator |
| Sessions generated | 2,200; burn-in 700; scored origins 1,499 |
| Development fold | scored origins 1 to 899 |
| Test fold | scored origins 900 to 1,499, that is 600 sessions |
| Candidate | decay 0.94, initialised to the mean of the first 60 squared returns |
| Baseline | trailing 20-session mean of squared returns |
| Oracle | true conditional variance from the generator, unobservable in practice |
| Observable proxy | next-session squared return |
| Loss | Gaussian quasi-likelihood, y over f minus log of y over f minus one |
| Tail test | Kupiec unconditional coverage, Christoffersen independence, 5% size |
| Notional | 250 million currency units, constant |
| Limit | 10 million currency units, one-day 99% |

Two ledger entries need justification rather than restatement. The first is the initialisation of
the recursion: starting from the mean of the first sixty squared returns rather than from a single
squared return avoids an initial value that is itself an extreme draw, and it is documented so a
reviewer can substitute a different rule and see how much of the result depends on it. The second is
the choice of a Gaussian loss rather than an absolute-error loss on volatility. Quasi-likelihood is
used because it is robust to the exact innovation distribution and because it penalises
understatement of variance more heavily than overstatement, which matches the direction of the
desk's concern. Section 6 shows the asymmetry numerically: a fourfold understatement of variance
costs 1.6137 in loss while the same fourfold overstatement costs 0.6363.

The ledger also records what was *not* done, because absence of an experiment is a fact about the
evidence. No implied-volatility model was fitted, no intraday range estimator was computed, no
fat-tailed innovation was used in the pack, and no parameter was re-optimised after the test fold
was scored. Each omission narrows the claim the case can support, and Section 13 lists the three
that a reviewer is most likely to challenge.

## 6. Analysis Walkthrough

### Step 1 — Rebuild the recursion from first principles

The candidate is defined by one equation, and every subsequent number in this section follows from
it. Writing the estimate at the end of session t as f_t and the squared return of session t as y_t,
the recursion weights the previous estimate by the decay and the newest observation by its
complement.

```text
FORMULA [D1]: f_t = lambda * f_{t-1} + (1 - lambda) * y_{t-1}
WHERE: f_t is conditional variance in squared decimal returns, dimensionless units;
       y_{t-1} is the squared log return of the previous session, same units;
       lambda is the decay in (0,1), a pure number;
       the estimate f_t is formed before session t opens, so y_t is not an input.
CONSTRAINT: f_0 must be supplied from outside the recursion; lambda must be frozen
            before scoring; y must be the same return definition throughout.
NUMERICAL EXAMPLE: f_{t-1} = 0.000016, y_{t-1} = 0.0004, lambda = 0.94.
            lambda * f_{t-1} = 0.94 * 0.000016 = 0.00001504.
            (1 - lambda) * y_{t-1} = 0.06 * 0.0004 = 0.000024.
            f_t = 0.00001504 + 0.000024 = 0.00003904.
INTERPRETATION: a 2% return, squared, is 25 times the prior variance, so it lifts the
            estimate by a factor of 2.44 despite the decay.
```

The units check is immediate and worth doing every time: both terms on the right are a number
multiplied by a variance, so the output is a variance. Taking square roots gives the daily
volatility, and in the example that moves from 0.4000% to 0.6248%. The percentage change in
volatility is 56.2%, much smaller than the 144% change in variance, which is the general pattern:
volatility reacts to shocks with roughly half the relative amplitude that variance does.

Expanding the recursion by substitution shows why one parameter controls everything. After k
substitutions the estimate is a weighted sum of the initial value and all past squared returns with
geometric weights, and the weight on the observation aged k sessions is (1 - lambda) times lambda to
the k. Summing that geometric series over all ages gives exactly one, so the estimator is a genuine
weighted average and is not biased by construction when the series is stationary.

```text
FORMULA [D2]: f_t = lambda^n * f_{t-n} + (1 - lambda) * sum_{k=0}^{n-1} lambda^k * y_{t-1-k}
WHERE: n is the number of substitutions, a positive integer;
       k indexes age in sessions from the newest observation;
       the weights (1 - lambda) * lambda^k sum to 1 - lambda^n.
CONSTRAINT: the sum converges only for lambda strictly inside (0,1); at lambda = 1
            the estimator has no finite memory and never updates.
NUMERICAL EXAMPLE: lambda = 0.94, n = 5.
            1 - 0.94^5 = 1 - 0.7339 = 0.2661, so five observations carry 26.61%
            of the total weight; the remaining 73.39% sits on older data and the seed.
INTERPRETATION: five sessions is not a meaningful sample; the effective sample size,
            derived in Step 2, is what a risk manager should quote.
```

The boundary case in D2 is the informative one. As lambda tends to one the estimator stops moving
and freezes on its initial value; as lambda tends to zero it becomes the last squared return, whose
sampling error is enormous. Both limits are useless, and the useful region is in between, which is
why the choice of decay cannot be made by appeal to principle and has to be made by out-of-sample
loss.

### Step 2 — Derive the half-life and effective sample size

Two derived quantities turn the decay into language the desk already uses. The half-life is the age
at which an observation's weight has fallen to half its original value, and the effective sample
size is the reciprocal of the sum of squared normalised weights, which is how many equally weighted
observations the estimator is actually equivalent to.

```text
FORMULA [D3]: h = ln(0.5) / ln(lambda)
WHERE: h is the half-life in sessions, a real number greater than zero;
       lambda is the decay in (0,1); ln is natural logarithm.
CONSTRAINT: the identity requires lambda > 0, and h is undefined at lambda = 1.
NUMERICAL EXAMPLE: lambda = 0.94.
            ln(0.5) = -0.693147; ln(0.94) = -0.061875.
            h = -0.693147 / -0.061875 = 11.2023 sessions.
INTERPRETATION: a shock's contribution is halved after about eleven sessions, and
            the check 0.94 raised to 11.2023 equals 0.5 confirms it.
```

```text
FORMULA [D4]: ESS = (1 + lambda) / (1 - lambda)
WHERE: ESS is the effective sample size in sessions, dimensionless but expressed
       in session units for interpretability; lambda in (0,1).
CONSTRAINT: this is the exact result for the infinite geometric weighting; a
            truncated sum with finite history gives a slightly smaller value.
NUMERICAL EXAMPLE: lambda = 0.94 gives 1.94 / 0.06 = 32.33 sessions.
            A 20-session equal-weight window has ESS = 20 exactly.
INTERPRETATION: the candidate behaves like a 32-session average with a recency
            tilt, not like a 20-session average, so the comparison is not
            "same memory, different weights".
```

That last interpretation is the first place the case earns its keep. A reviewer who hears "we
replaced a 20-day average with an exponentially weighted average" will reasonably ask whether the
improvement is just a longer memory. The ESS calculation answers that the memories differ by 62%,
and the grid in Step 4 then shows that the decay, not the memory length, is doing the work.

The boundary here has a practical consequence. With a decay of 0.99 the effective sample size is 199
sessions, which is nearly a trading year; such an estimator responds so slowly that it will not move
materially during a two-month regime, and Section 6's later steps show it losing to a simple
30-session window on the same data. With a decay of 0.85 the effective sample size is 12.33, and the
estimator's own sampling noise becomes comparable to the signal it is trying to track.

### Step 3 — Show what the baseline does at its window edge

The baseline is the mean of the last twenty squared returns, and its defect is not that it is
simple but that its weighting is a step function. An observation carries full weight on the session
after it occurs and zero weight on the twenty-first, with no gradual transition.

```text
FORMULA [D5]: b_t = (1/20) * sum_{j=1}^{20} y_{t-j}
WHERE: b_t is the baseline variance estimate formed before session t;
       y_{t-j} is the squared return of the session j sessions before t;
       the divisor 20 is the window length and is a free choice, not a constant.
CONSTRAINT: the window must contain exactly twenty eligible sessions; a gap or a
            dropped session changes the divisor or the span and must be documented.
NUMERICAL EXAMPLE: nineteen sessions with squared return 0.000016 and one newest
            session with squared return 0.0004.
            b_t = (19 * 0.000016 + 0.0004) / 20 = (0.000304 + 0.0004) / 20
                = 0.000704 / 20 = 0.0000352.
            The candidate on the same history is 0.00003904 from D1.
INTERPRETATION: the candidate sits 10.91% above the baseline in variance terms
            immediately after the shock, because the baseline dilutes the shock
            across twenty sessions while the candidate weights it at 6%.
```

The edge effect is the counterexample that breaks the formula's usefulness. Take the same history
but move the shock back by one session, so that it becomes the twenty-first observation. The
baseline now reads 0.000016, a fall of 54.5% in one session, while nothing about the market has
changed and the candidate's estimate has only decayed by a factor of 0.94. A risk number that drops
by half because of an index boundary is a governance problem: the desk can watch its own limit
utilisation improve without any change in exposure or market conditions.

The mechanism is worth stating precisely because it generalises. Any fixed-window estimator has a
discontinuity of size (y_shock - y_replacement) / w at the moment a shock leaves the window, where w
is the window length. For a shock of 0.0004 against a replacement of 0.000016 in a 20-session
window, that discontinuity is 0.0000192, larger than the whole baseline estimate. Longer windows
reduce the jump but delay the response, and Section 9 shows the trade-off empirically: a 120-session
window has the smallest mean bias in this sample but the worst average loss of any window tested,
because it is still carrying a shock from six months ago.
